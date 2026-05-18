"""Download and persist historical OHLCV data for research and live trading.

Adapted from crypto-momentum-strategy download_ohlcv.py for multi-strategy use.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import SETTINGS, get_data_symbols
from data.fetch_ohlc import update_symbol_ohlcv_incremental


EXPANDED_UNIVERSE_20: tuple[str, ...] = (
    "AAVE/USD",
    "ADA/USD",
    "APT/USD",
    "ARB/USD",
    "ATOM/USD",
    "AVAX/USD",
    "BCH/USD",
    "BTC/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "INJ/USD",
    "LINK/USD",
    "LTC/USD",
    "NEAR/USD",
    "OP/USD",
    "POL/USD",
    "SOL/USD",
    "UNI/USD",
    "XRP/USD",
)


def symbol_to_local_filename(symbol: str, timeframe: str, data_dir: Path = Path("data/local")) -> Path:
    """Convert a market symbol to its canonical local OHLCV CSV path."""
    slug = symbol.strip().lower().replace("/", "-").replace(" ", "-")
    return Path(data_dir) / f"{slug}_{timeframe}.csv"


def fetch_ohlcv_paginated(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    max_batches: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Download OHLCV data via forward pagination using ccxt-style fetch_ohlcv."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if since_ms < 0:
        raise ValueError("since_ms must be non-negative")

    rows: list[list[float | int]] = []
    current_since_ms = int(since_ms)
    batches_completed = 0

    for _ in range(max_batches):
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=current_since_ms,
            limit=limit,
        )
        batches_completed += 1

        if not batch:
            break

        rows.extend(batch)
        last_timestamp = int(batch[-1][0])

        if last_timestamp <= current_since_ms:
            break

        current_since_ms = last_timestamp + 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        out = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        out.attrs["batches_completed"] = batches_completed
        return out

    out = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    out.attrs["batches_completed"] = batches_completed
    return out


def _to_utc_series(series: pd.Series) -> pd.Series:
    """Normalize a timestamp-like series to UTC timestamps."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _load_existing_ohlcv(path: Path) -> pd.DataFrame:
    """Load existing symbol OHLCV if present."""
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    existing = pd.read_csv(path)
    if existing.empty:
        return existing
    existing = existing.copy()
    existing["timestamp"] = _to_utc_series(existing["timestamp"])
    return existing


def merge_ohlcv_frames(existing: pd.DataFrame, downloaded: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and downloaded OHLCV by timestamp, keeping latest values."""
    frames = []
    for frame in (existing, downloaded):
        if frame is None or frame.empty:
            continue
        cur = frame.copy()
        cur["timestamp"] = _to_utc_series(cur["timestamp"])
        frames.append(cur)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.dropna(subset=["timestamp"])
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


DownloaderFn = Callable[[Any, str, str, int, int, int, float], pd.DataFrame]


def backfill_symbol_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since: str,
    limit: int,
    max_batches: int,
    sleep_seconds: float,
    data_dir: Path = Path("data/local"),
    downloader: DownloaderFn = fetch_ohlcv_paginated,
) -> dict[str, Any]:
    """Backfill one symbol from a requested since date and persist merged local CSV."""
    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    output_path = symbol_to_local_filename(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_since_ts = pd.to_datetime(since, utc=True)
    requested_since_ms = int(requested_since_ts.timestamp() * 1000)

    existing = _load_existing_ohlcv(output_path)
    existing_rows = int(len(existing))

    downloaded = downloader(
        exchange,
        symbol,
        timeframe,
        requested_since_ms,
        limit,
        max_batches,
        sleep_seconds,
    )
    downloaded_rows = int(len(downloaded))
    batches_completed = int(downloaded.attrs.get("batches_completed", 0))

    first_returned: str | None = None
    latest_returned: str | None = None
    if not downloaded.empty:
        downloaded = downloaded.copy()
        downloaded["timestamp"] = _to_utc_series(downloaded["timestamp"])
        first_ts = downloaded["timestamp"].min()
        latest_ts = downloaded["timestamp"].max()
        first_returned = first_ts.isoformat() if pd.notna(first_ts) else None
        latest_returned = latest_ts.isoformat() if pd.notna(latest_ts) else None

        if pd.notna(first_ts) and first_ts > (requested_since_ts + pd.Timedelta(days=30)):
            logger.warning(
                "Older history may not be available from this source or current fetch method."
            )

    final_df = merge_ohlcv_frames(existing=existing, downloaded=downloaded)
    final_df.to_csv(output_path, index=False)
    final_rows = int(len(final_df))

    logger.info(
        "backfill_result symbol=%s requested_since=%s existing_rows=%d downloaded_rows=%d final_rows=%d first_returned=%s latest_returned=%s batches_completed=%d",
        symbol,
        requested_since_ts.isoformat(),
        existing_rows,
        downloaded_rows,
        final_rows,
        first_returned,
        latest_returned,
        batches_completed,
    )

    return {
        "symbol": symbol,
        "requested_since": requested_since_ts.isoformat(),
        "existing_rows": existing_rows,
        "downloaded_rows": downloaded_rows,
        "final_rows": final_rows,
        "first_returned": first_returned,
        "latest_returned": latest_returned,
        "batches_completed": batches_completed,
        "path": output_path,
    }


def backfill_symbols(
    symbols: tuple[str, ...],
    timeframe: str,
    since: str,
    limit: int,
    max_batches: int,
    sleep_seconds: float,
    data_dir: Path | None = None,
) -> None:
    """Backfill local OHLCV files for a symbol list using Kraken pagination."""
    try:
        import ccxt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("ccxt is required for downloader mode. Install with: pip install ccxt") from exc

    logger = logging.getLogger(__name__)
    target_dir = data_dir or SETTINGS.data_dir
    exchange = ccxt.kraken({"enableRateLimit": True})

    logger.info(
        "Starting OHLCV backfill: exchange=%s timeframe=%s since=%s symbols=%d limit=%d max_batches=%d",
        "kraken",
        timeframe,
        since,
        len(symbols),
        limit,
        max_batches,
    )

    for symbol in symbols:
        backfill_symbol_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
            max_batches=max_batches,
            sleep_seconds=sleep_seconds,
            data_dir=target_dir,
        )


def configure_logging() -> None:
    """Configure readable logging for downloader runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def download_all_symbols(
    symbols: tuple[str, ...] | None = None,
    timeframe: str | None = None,
    data_dir: Path | None = None,
) -> None:
    """Incrementally update local OHLCV CSVs for the requested symbol universe."""
    try:
        import ccxt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("ccxt is required for downloader mode. Install with: pip install ccxt") from exc

    logger = logging.getLogger(__name__)
    symbols_to_refresh = tuple(symbols) if symbols is not None else get_data_symbols()
    timeframe_to_use = timeframe or SETTINGS.timeframe
    data_dir = data_dir or SETTINGS.data_dir
    exchange = ccxt.kraken({"enableRateLimit": True})

    logger.info(
        "Starting OHLCV incremental update: exchange=%s timeframe=%s symbols=%d",
        "kraken",
        timeframe_to_use,
        len(symbols_to_refresh),
    )
    logger.info("Data refresh universe: %s", ", ".join(symbols_to_refresh))

    for symbol in symbols_to_refresh:
        result = update_symbol_ohlcv_incremental(
            symbol=symbol,
            timeframe=timeframe_to_use,
            data_dir=data_dir,
            exchange=exchange,
            limit=720,
        )
        logger.info(
            "Incremental update complete for %s | fetched=%d dropped=%d total=%d",
            result["symbol"],
            result["fetched_rows"],
            result["dropped_rows"],
            result["final_rows"],
        )


def main() -> None:
    """Entry point for downloading and persisting OHLCV history."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Update/backfill local OHLCV files from Kraken")
    parser.add_argument("--expanded-universe", action="store_true", help="Use the expanded 20-coin universe")
    parser.add_argument("--symbols", nargs="+", default=None, help="Optional explicit symbol list")
    parser.add_argument("--timeframe", default=SETTINGS.timeframe, help="OHLCV timeframe (e.g., 4h)")
    parser.add_argument("--since", default=None, help="Backfill start date, e.g., 2020-01-01")
    parser.add_argument("--limit", type=int, default=720, help="Candles per fetch_ohlcv request")
    parser.add_argument("--max-batches", type=int, default=1000, help="Max pagination requests per symbol")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between paginated requests")
    parser.add_argument("--data-dir", default=str(SETTINGS.data_dir), help="Local OHLCV directory")
    args = parser.parse_args()

    run_backfill_mode = args.expanded_universe or args.since is not None or bool(args.symbols)

    if not run_backfill_mode:
        download_all_symbols(
            symbols=None,
            timeframe=args.timeframe,
            data_dir=Path(args.data_dir),
        )
        return

    if args.expanded_universe:
        symbols = EXPANDED_UNIVERSE_20
    elif args.symbols:
        symbols = tuple(args.symbols)
    else:
        symbols = get_data_symbols()

    since = args.since or "2020-01-01"
    backfill_symbols(
        symbols=symbols,
        timeframe=args.timeframe,
        since=since,
        limit=args.limit,
        max_batches=args.max_batches,
        sleep_seconds=args.sleep_seconds,
        data_dir=Path(args.data_dir),
    )


if __name__ == "__main__":
    main()
