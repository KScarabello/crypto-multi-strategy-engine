"""Research-only downloader for Binance public monthly spot kline ZIP files."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Iterator
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

LOGGER = logging.getLogger(__name__)

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

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


class MissingRemoteFileError(Exception):
    """Raised when a remote monthly ZIP file is not available (HTTP 404)."""


def symbol_to_binance_pair(symbol: str) -> str:
    """Convert a normalized symbol (e.g. BTC/USD) to Binance spot pair (e.g. BTCUSDT)."""
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" not in normalized:
        raise ValueError(f"Unsupported symbol format: {symbol}")

    base, quote = normalized.split("/", maxsplit=1)
    if quote == "USD":
        quote = "USDT"

    if not base or not quote:
        raise ValueError(f"Unsupported symbol format: {symbol}")
    return f"{base}{quote}"


def symbol_to_local_filename(symbol: str, timeframe: str) -> str:
    """Convert normalized symbol and timeframe to local OHLCV CSV filename."""
    slug = symbol.strip().lower().replace("/", "-").replace(" ", "-")
    return f"{slug}_{timeframe}.csv"


def build_monthly_kline_url(binance_pair: str, timeframe: str, month: str) -> str:
    """Build Binance public monthly spot kline ZIP URL."""
    return (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"{binance_pair}/{timeframe}/{binance_pair}-{timeframe}-{month}.zip"
    )


def parse_binance_timestamp(series: pd.Series) -> pd.Series:
    """Parse Binance open_time values as UTC with ms/us defensiveness."""
    numeric = pd.to_numeric(series, errors="coerce")
    max_abs = numeric.abs().max(skipna=True)
    unit = "us" if pd.notna(max_abs) and float(max_abs) > 1e15 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """Normalize timestamp input to UTC for integer epoch and datetime-like strings."""
    if pd.api.types.is_numeric_dtype(series):
        return parse_binance_timestamp(series)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_ratio > 0.9:
        return parse_binance_timestamp(numeric)

    return pd.to_datetime(series, utc=True, errors="coerce")


def read_binance_kline_zip(zip_path: Path) -> pd.DataFrame:
    """Read and normalize one Binance monthly kline ZIP file."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in zip {zip_path}, found {len(members)}")

        with zf.open(members[0]) as handle:
            df = pd.read_csv(handle, header=None, names=BINANCE_KLINE_COLUMNS)

    normalized = pd.DataFrame(
        {
            "timestamp": parse_binance_timestamp(df["open_time"]),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )

    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    return normalized


def download_file(url: str, output_path: Path) -> bool:
    """Download a remote file to disk; return False when file is missing (404)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urlrequest.urlretrieve(url, output_path)
        return True
    except urlerror.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def merge_ohlcv_frames(existing: pd.DataFrame, downloaded: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and downloaded OHLCV, keeping latest duplicate timestamp."""
    frames = [df for df in (existing, downloaded) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["timestamp"] = _normalize_timestamp_series(merged["timestamp"])

    for col in ("open", "high", "low", "close", "volume"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged[["timestamp", "open", "high", "low", "close", "volume"]]


def iterate_months(start_month: str, end_month: str) -> Iterator[str]:
    """Yield YYYY-MM months from start to end inclusive."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must be greater than or equal to start_month")

    current = start
    while current <= end:
        yield str(current)
        current += 1


def backfill_symbol_from_binance_public(
    symbol: str,
    timeframe: str,
    start_month: str,
    end_month: str,
    data_dir: Path,
    cache_dir: Path,
) -> dict[str, object]:
    """Backfill one symbol from Binance public monthly ZIP files into local CSV."""
    binance_pair = symbol_to_binance_pair(symbol)
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    monthly_frames: list[pd.DataFrame] = []
    months_found = 0
    months_missing = 0

    for month in iterate_months(start_month=start_month, end_month=end_month):
        url = build_monthly_kline_url(binance_pair=binance_pair, timeframe=timeframe, month=month)
        zip_name = f"{binance_pair}-{timeframe}-{month}.zip"
        zip_path = cache_dir / zip_name

        downloaded = zip_path.exists() or download_file(url, zip_path)
        if not downloaded:
            months_missing += 1
            continue

        frame = read_binance_kline_zip(zip_path)
        if not frame.empty:
            monthly_frames.append(frame)
        months_found += 1

    new_data = (
        pd.concat(monthly_frames, axis=0, ignore_index=True)
        if monthly_frames
        else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    )

    out_path = data_dir / symbol_to_local_filename(symbol=symbol, timeframe=timeframe)
    if out_path.exists():
        existing = pd.read_csv(out_path)
    else:
        existing = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    merged = merge_ohlcv_frames(existing=existing, downloaded=new_data)
    merged.to_csv(out_path, index=False)

    first_timestamp = merged["timestamp"].min().isoformat() if not merged.empty else None
    latest_timestamp = merged["timestamp"].max().isoformat() if not merged.empty else None

    LOGGER.info(
        "binance_public_backfill symbol=%s binance_pair=%s months_found=%d months_missing=%d rows_written=%d first_timestamp=%s latest_timestamp=%s",
        symbol,
        binance_pair,
        months_found,
        months_missing,
        len(merged),
        first_timestamp,
        latest_timestamp,
    )

    return {
        "symbol": symbol,
        "binance_pair": binance_pair,
        "months_found": months_found,
        "months_missing": months_missing,
        "rows_written": int(len(merged)),
        "first_timestamp": first_timestamp,
        "latest_timestamp": latest_timestamp,
        "output_path": out_path,
    }


def configure_logging() -> None:
    """Configure readable logging for research data backfill runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Binance public historical backfill."""
    parser = argparse.ArgumentParser(description="Backfill OHLCV from Binance public monthly spot klines")
    parser.add_argument("--symbols", nargs="+", default=None, help="Normalized symbols (e.g. BTC/USD ETH/USD)")
    parser.add_argument("--expanded-universe", action="store_true", help="Use the expanded 20-coin universe")
    parser.add_argument("--timeframe", required=True, help="Binance kline timeframe (e.g. 4h)")
    parser.add_argument("--start-month", required=True, help="Inclusive start month YYYY-MM")
    parser.add_argument("--end-month", required=True, help="Inclusive end month YYYY-MM")
    parser.add_argument("--data-dir", default="data/local", help="Local OHLCV output directory")
    parser.add_argument("--cache-dir", default="data/cache/binance", help="Download cache directory")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for Binance public monthly spot kline backfill."""
    configure_logging()
    args = parse_args()

    if args.expanded_universe:
        symbols = EXPANDED_UNIVERSE_20
    elif args.symbols:
        symbols = tuple(args.symbols)
    else:
        raise ValueError("Either --symbols or --expanded-universe must be provided")

    for symbol in symbols:
        backfill_symbol_from_binance_public(
            symbol=symbol,
            timeframe=args.timeframe,
            start_month=args.start_month,
            end_month=args.end_month,
            data_dir=Path(args.data_dir),
            cache_dir=Path(args.cache_dir),
        )


if __name__ == "__main__":
    main()
