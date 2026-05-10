"""Load and validate OHLCV history for research backtests.

Reusable OHLCV utilities transferred from crypto-momentum-strategy and adapted
for multi-strategy use. Supports:
- Loading from local CSV/parquet files
- Downloading from Kraken via ccxt
- Downloading from CryptoCompare
- Incremental CSV updates
- Symbol normalization
- Data validation and cleaning
"""

from __future__ import annotations

import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

NORMALIZED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]

DownloadFunction = Callable[[str, str], pd.DataFrame]

PROVIDER_TO_EXCHANGE = {
    "kraken": "kraken",
    "ccxt_kraken": "kraken",
    "binance": "binance",
    "ccxt_binance": "binance",
}

SUPPORTED_PROVIDERS = tuple(sorted((*PROVIDER_TO_EXCHANGE.keys(), "cryptocompare")))


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert common symbol formats to ccxt unified symbol format."""
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" in normalized:
        return normalized
    raise ValueError(f"Unsupported symbol format: {symbol}")


def _to_ccxt_timeframe(timeframe: str) -> str:
    """Map local timeframe aliases to ccxt timeframe strings."""
    mapping = {
        "1d": "1d",
        "d": "1d",
        "daily": "1d",
        "4h": "4h",
    }
    key = timeframe.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use 4h or 1d.")
    return mapping[key]


def _timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert supported timeframe strings to milliseconds."""
    mapping = {
        "1d": 24 * 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "daily": 24 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }
    key = timeframe.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use 4h or 1d.")
    return mapping[key]


def _to_utc_timestamp(since: pd.Timestamp | str | int | None) -> pd.Timestamp | None:
    """Normalize optional since input to a UTC pandas Timestamp."""
    if since is None:
        return None
    if isinstance(since, int):
        return pd.to_datetime(since, unit="ms", utc=True)
    return pd.to_datetime(since, utc=True)


def _to_quote_symbol(symbol: str) -> tuple[str, str]:
    """Convert symbols like BTC/USD into provider fsym/tsym parts."""
    normalized = _to_ccxt_symbol(symbol)
    base, quote = normalized.split("/", maxsplit=1)
    return base, quote


def _fetch_json(url: str, params: dict[str, str | int]) -> dict:
    """Fetch a JSON payload from a public HTTP endpoint."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return json.loads(response.text)


def cryptocompare_downloader(
    symbol: str,
    timeframe: str,
    since: pd.Timestamp | str | int | None = None,
    max_batches: int | None = 200,
    max_rows: int | None = None,
    limit_per_request: int = 2000,
    request_pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download deeper historical OHLCV from CryptoCompare with backward pagination."""
    if limit_per_request <= 0:
        raise ValueError("limit_per_request must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    base, quote = _to_quote_symbol(symbol)
    tf = timeframe.strip().lower()
    if tf == "4h":
        endpoint = "https://min-api.cryptocompare.com/data/v2/histohour"
        aggregate = 4
    elif tf in {"1d", "d", "daily"}:
        endpoint = "https://min-api.cryptocompare.com/data/v2/histoday"
        aggregate = 1
    else:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use 4h or 1d.")

    capped_limit = min(int(limit_per_request), 2000)
    since_ts = _to_utc_timestamp(since)
    since_seconds = int(since_ts.timestamp()) if since_ts is not None else None

    rows_by_ts_seconds: dict[int, dict[str, float | int]] = {}
    to_ts: int | None = None
    batches = 0

    LOGGER.info("Downloading full history from cryptocompare for %s (%s)", symbol, timeframe)
    while True:
        if max_batches is not None and batches >= max_batches:
            break
        if max_rows is not None and len(rows_by_ts_seconds) >= max_rows:
            break

        params: dict[str, str | int] = {
            "fsym": base,
            "tsym": quote,
            "limit": capped_limit,
            "aggregate": aggregate,
        }
        if to_ts is not None:
            params["toTs"] = to_ts

        payload = _fetch_json(endpoint, params)
        if payload.get("Response") != "Success":
            message = payload.get("Message", "Unknown provider error")
            raise ValueError(f"cryptocompare error for {symbol}: {message}")

        data_rows = payload.get("Data", {}).get("Data", [])
        batches += 1
        if not data_rows:
            break

        previous_oldest = min(rows_by_ts_seconds) if rows_by_ts_seconds else None
        for row in data_rows:
            ts_seconds = int(row["time"])
            if since_seconds is not None and ts_seconds < since_seconds:
                continue
            rows_by_ts_seconds[ts_seconds] = {
                "timestamp": ts_seconds * 1000,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volumefrom"),
            }

        current_oldest = min(rows_by_ts_seconds) if rows_by_ts_seconds else None
        if batches % 10 == 0 and rows_by_ts_seconds:
            oldest = pd.to_datetime(min(rows_by_ts_seconds) * 1000, unit="ms", utc=True)
            newest = pd.to_datetime(max(rows_by_ts_seconds) * 1000, unit="ms", utc=True)
            LOGGER.info(
                "cryptocompare progress for %s: batches=%d rows=%d range=%s -> %s",
                symbol,
                batches,
                len(rows_by_ts_seconds),
                oldest,
                newest,
            )
        if previous_oldest is not None and current_oldest is not None and current_oldest >= previous_oldest:
            break
        if current_oldest is None:
            break

        if since_seconds is not None and current_oldest <= since_seconds:
            break

        next_to_ts = current_oldest - 1
        if to_ts is not None and next_to_ts >= to_ts:
            break
        to_ts = next_to_ts

        if request_pause_seconds > 0:
            time.sleep(request_pause_seconds)

    if not rows_by_ts_seconds:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    ordered = [rows_by_ts_seconds[k] for k in sorted(rows_by_ts_seconds)]
    if max_rows is not None:
        ordered = ordered[-max_rows:]

    df = pd.DataFrame(ordered, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    LOGGER.info("Downloaded %d OHLCV rows for %s from cryptocompare", len(df), symbol)
    return _validate_and_clean(df, symbol_hint=symbol)


def _fetch_ohlcv_paginated(
    symbol: str,
    timeframe: str,
    exchange_name: str = "kraken",
    since: pd.Timestamp | str | int | None = None,
    max_batches: int | None = 200,
    max_rows: int | None = None,
    limit_per_request: int = 720,
    request_pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Fetch OHLCV history with safe pagination and loop guards."""
    try:
        import ccxt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("ccxt is required for downloader mode. Install with: pip install ccxt") from exc

    if limit_per_request <= 0:
        raise ValueError("limit_per_request must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    exchange_class = getattr(ccxt, exchange_name, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported ccxt exchange: {exchange_name}")

    exchange = exchange_class({"enableRateLimit": True})
    ccxt_symbol = _to_ccxt_symbol(symbol)
    ccxt_timeframe = _to_ccxt_timeframe(timeframe)
    timeframe_ms = _timeframe_to_milliseconds(timeframe)
    since_ts = _to_utc_timestamp(since)
    since_ms = int(since_ts.timestamp() * 1000) if since_ts is not None else None

    rows_by_timestamp: dict[int, list[float]] = {}
    batch_count = 0

    if since_ms is None:
        LOGGER.info("Downloading full history from %s for %s (%s)", exchange_name, ccxt_symbol, ccxt_timeframe)
        latest_batch = exchange.fetch_ohlcv(
            symbol=ccxt_symbol,
            timeframe=ccxt_timeframe,
            limit=limit_per_request,
        )
        if not latest_batch:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)

        for row in latest_batch:
            rows_by_timestamp[int(row[0])] = row
        batch_count += 1

        earliest_ms = min(rows_by_timestamp)
        while True:
            if max_batches is not None and batch_count >= max_batches:
                break
            if max_rows is not None and len(rows_by_timestamp) >= max_rows:
                break

            window_ms = timeframe_ms * limit_per_request
            next_since_ms = max(0, earliest_ms - window_ms)
            if next_since_ms >= earliest_ms:
                break

            batch = exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=ccxt_timeframe,
                since=next_since_ms,
                limit=limit_per_request,
            )
            batch_count += 1
            if not batch:
                break

            previous_earliest = earliest_ms
            for row in batch:
                rows_by_timestamp[int(row[0])] = row
            earliest_ms = min(rows_by_timestamp)
            if earliest_ms >= previous_earliest:
                break

            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)
    else:
        LOGGER.info(
            "Downloading history from %s for %s (%s) since %s",
            exchange_name,
            ccxt_symbol,
            ccxt_timeframe,
            pd.to_datetime(since_ms, unit="ms", utc=True),
        )
        cursor_ms = since_ms
        while True:
            if max_batches is not None and batch_count >= max_batches:
                break
            if max_rows is not None and len(rows_by_timestamp) >= max_rows:
                break

            batch = exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=ccxt_timeframe,
                since=cursor_ms,
                limit=limit_per_request,
            )
            batch_count += 1
            if not batch:
                break

            newest_batch_ms = max(int(row[0]) for row in batch)
            for row in batch:
                rows_by_timestamp[int(row[0])] = row

            next_cursor_ms = newest_batch_ms + timeframe_ms
            if next_cursor_ms <= cursor_ms:
                break
            cursor_ms = next_cursor_ms

            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

    if not rows_by_timestamp:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    ordered_rows = [rows_by_timestamp[key] for key in sorted(rows_by_timestamp)]
    if max_rows is not None:
        ordered_rows = ordered_rows[-max_rows:]

    df = pd.DataFrame(ordered_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    LOGGER.info("Downloaded %d OHLCV rows for %s", len(df), symbol)
    return df


def ccxt_downloader(
    symbol: str,
    timeframe: str,
    exchange_name: str = "kraken",
    since: pd.Timestamp | str | int | None = None,
    max_batches: int | None = 200,
    max_rows: int | None = None,
    limit_per_request: int = 720,
    request_pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download OHLCV data via ccxt using safe pagination."""
    raw = _fetch_ohlcv_paginated(
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        since=since,
        max_batches=max_batches,
        max_rows=max_rows,
        limit_per_request=limit_per_request,
        request_pause_seconds=request_pause_seconds,
    )
    return _validate_and_clean(raw, symbol_hint=symbol)


def provider_downloader(
    symbol: str,
    timeframe: str,
    provider: str = "binance",
    exchange_name: str | None = None,
    since: pd.Timestamp | str | int | None = None,
    max_batches: int | None = 200,
    max_rows: int | None = None,
    limit_per_request: int = 720,
    request_pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download OHLCV from a configured historical provider."""
    key = provider.strip().lower()
    if key not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported historical_data_provider '{provider}'. Supported: {supported}")

    if key == "cryptocompare":
        return cryptocompare_downloader(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            max_batches=max_batches,
            max_rows=max_rows,
            limit_per_request=limit_per_request,
            request_pause_seconds=request_pause_seconds,
        )

    resolved_exchange = exchange_name or PROVIDER_TO_EXCHANGE[key]
    return ccxt_downloader(
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=resolved_exchange,
        since=since,
        max_batches=max_batches,
        max_rows=max_rows,
        limit_per_request=limit_per_request,
        request_pause_seconds=request_pause_seconds,
    )


def build_historical_downloader(
    provider: str,
    exchange_name: str | None = None,
    since: pd.Timestamp | str | int | None = None,
    max_batches: int | None = 200,
    max_rows: int | None = None,
    limit_per_request: int = 720,
    request_pause_seconds: float = 0.15,
) -> DownloadFunction:
    """Build a two-argument downloader hook compatible with load_ohlcv_history."""
    return partial(
        provider_downloader,
        provider=provider,
        exchange_name=exchange_name,
        since=since,
        max_batches=max_batches,
        max_rows=max_rows,
        limit_per_request=limit_per_request,
        request_pause_seconds=request_pause_seconds,
    )


def _sanitize_symbol(symbol: str) -> str:
    """Convert symbol to a filesystem-safe lowercase slug."""
    return symbol.strip().lower().replace("/", "-").replace(" ", "-")


def local_symbol_file_path(symbol: str, timeframe: str, data_dir: Path = Path("data/local")) -> Path:
    """Return canonical local per-symbol OHLCV file path."""
    slug = _sanitize_symbol(symbol)
    return Path(data_dir) / f"{slug}_{timeframe}.csv"


def load_local_symbol_ohlcv(
    symbol: str,
    timeframe: str,
    data_dir: Path = Path("data/local"),
) -> pd.DataFrame:
    """Load one local OHLCV file if present, else return an empty normalized frame."""
    output_path = local_symbol_file_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if not output_path.exists():
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return _validate_and_clean(_read_local_file(output_path), symbol_hint=symbol)


def _timestamp_to_kraken_since_ms(timestamp: pd.Timestamp | None) -> int | None:
    """Convert latest local timestamp to Kraken since milliseconds."""
    if timestamp is None or pd.isna(timestamp):
        return None
    ts = pd.Timestamp(timestamp)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts_utc.timestamp() * 1000)


def fetch_incremental_kraken_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    last_timestamp: pd.Timestamp | None,
    limit: int = 720,
) -> pd.DataFrame:
    """Fetch incremental OHLCV rows from Kraken using the local file watermark."""
    ccxt_symbol = _to_ccxt_symbol(symbol)
    ccxt_timeframe = _to_ccxt_timeframe(timeframe)
    since_ms = _timestamp_to_kraken_since_ms(last_timestamp)

    kwargs: dict[str, Any] = {
        "symbol": ccxt_symbol,
        "timeframe": ccxt_timeframe,
        "limit": int(limit),
    }
    if since_ms is not None:
        kwargs["since"] = since_ms

    rows = exchange.fetch_ohlcv(**kwargs)
    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    fetched = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    fetched["timestamp"] = pd.to_datetime(fetched["timestamp"], unit="ms", utc=True)
    fetched["symbol"] = symbol
    return fetched[NORMALIZED_COLUMNS]


def update_symbol_ohlcv_incremental(
    symbol: str,
    timeframe: str,
    data_dir: Path = Path("data/local"),
    exchange: Any | None = None,
    limit: int = 720,
) -> dict[str, Any]:
    """Incrementally update one local OHLCV CSV using Kraken candles only."""
    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    output_path = local_symbol_file_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_local_symbol_ohlcv(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    last_timestamp = existing["timestamp"].max() if not existing.empty else None

    logger.info(
        "Updating %s from last timestamp %s",
        symbol,
        last_timestamp if last_timestamp is not None else "none",
    )

    kraken = exchange
    if kraken is None:
        try:
            import ccxt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("ccxt is required for downloader mode. Install with: pip install ccxt") from exc
        kraken = ccxt.kraken({"enableRateLimit": True})

    fetched_raw = fetch_incremental_kraken_ohlcv(
        exchange=kraken,
        symbol=symbol,
        timeframe=timeframe,
        last_timestamp=last_timestamp,
        limit=limit,
    )
    fetched_rows = int(len(fetched_raw))
    logger.info("Fetched %d new rows", fetched_rows)

    invalid_fetched_rows = int((pd.to_numeric(fetched_raw.get("close"), errors="coerce") <= 0).sum()) if not fetched_raw.empty else 0

    frames = [frame for frame in (existing, fetched_raw) if not frame.empty]
    if frames:
        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged = _validate_and_clean(merged, symbol_hint=symbol)
    else:
        merged = pd.DataFrame(columns=NORMALIZED_COLUMNS)

    logger.info("Dropped %d invalid rows", invalid_fetched_rows)

    merged.to_csv(output_path, index=False)
    logger.info("Saved total rows: %d", len(merged))

    return {
        "symbol": symbol,
        "path": output_path,
        "last_timestamp": last_timestamp,
        "fetched_rows": fetched_rows,
        "dropped_rows": invalid_fetched_rows,
        "final_rows": int(len(merged)),
    }


def merge_and_save_symbol_ohlcv(
    symbol: str,
    timeframe: str,
    new_data: pd.DataFrame,
    data_dir: Path = Path("data/local"),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Merge downloaded data with existing local file and save per-symbol CSV."""
    output_path = local_symbol_file_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_new = _validate_and_clean(new_data, symbol_hint=symbol)
    if overwrite or not output_path.exists():
        merged = clean_new
    else:
        existing = _validate_and_clean(_read_local_file(output_path), symbol_hint=symbol)
        merged = pd.concat([existing, clean_new], axis=0, ignore_index=True)
        merged = _validate_and_clean(merged, symbol_hint=symbol)

    merged.to_csv(output_path, index=False)
    return merged


def _candidate_paths(symbol: str, timeframe: str, data_dir: Path) -> list[Path]:
    """Return candidate local paths for a symbol/timeframe pair."""
    slug = _sanitize_symbol(symbol)
    candidates = [
        data_dir / f"{slug}_{timeframe}.csv",
        data_dir / f"{slug}_{timeframe}.parquet",
        data_dir / f"{slug}.csv",
        data_dir / f"{slug}.parquet",
    ]
    return candidates


def _read_local_file(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet OHLCV file from disk."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for OHLCV data: {path.suffix}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common raw column names to the project schema."""
    rename_map = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "pair": "symbol",
        "ticker": "symbol",
    }
    lower_map = {str(col): str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=lower_map)
    return df.rename(columns=rename_map)


def _validate_and_clean(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    """Validate and clean an OHLCV DataFrame to normalized output columns."""
    df = _normalize_columns(df).copy()

    if "symbol" not in df.columns and symbol_hint is not None:
        df["symbol"] = symbol_hint

    missing = [col for col in NORMALIZED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[NORMALIZED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df["symbol"] != ""]

    # Drop non-positive close prices: data providers (e.g. CryptoCompare) use 0
    # as a sentinel for periods before an asset existed. Keeping them causes
    # pct_change to produce inf when the first real price appears.
    n_bad_close = int((df["close"] <= 0).sum())
    if n_bad_close > 0:
        LOGGER.warning(
            "Dropping %d rows with non-positive close prices for %s",
            n_bad_close,
            symbol_hint or "unknown",
        )
        df = df[df["close"] > 0]

    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _download_symbol_ohlcv(symbol: str, timeframe: str, downloader: DownloadFunction) -> pd.DataFrame:
    """Download OHLCV for one symbol using an injected downloader callback."""
    LOGGER.info("Downloading OHLCV for %s (%s)", symbol, timeframe)
    raw = downloader(symbol, timeframe)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("downloader must return a pandas DataFrame")
    return _validate_and_clean(raw, symbol_hint=symbol)


def load_ohlcv_history(
    symbols: Sequence[str],
    timeframe: str,
    data_dir: Path = Path("data/local"),
    file_map: Mapping[str, Path] | None = None,
    downloader: DownloadFunction | None = None,
) -> pd.DataFrame:
    """Load OHLCV history for multiple symbols and return one combined DataFrame."""
    if not symbols:
        raise ValueError("symbols must not be empty")

    frames: list[pd.DataFrame] = []
    data_dir = Path(data_dir)

    for symbol in symbols:
        symbol = symbol.strip()
        if not symbol:
            continue

        local_path: Path | None = None
        if file_map and symbol in file_map:
            candidate = Path(file_map[symbol])
            if candidate.exists():
                local_path = candidate
        else:
            for candidate in _candidate_paths(symbol=symbol, timeframe=timeframe, data_dir=data_dir):
                if candidate.exists():
                    local_path = candidate
                    break

        if local_path is not None:
            LOGGER.info("Loading OHLCV for %s from %s", symbol, local_path)
            frame = _validate_and_clean(_read_local_file(local_path), symbol_hint=symbol)
            frames.append(frame)
            continue

        if downloader is not None:
            frames.append(_download_symbol_ohlcv(symbol, timeframe, downloader))
            continue

        expected = ", ".join(str(p) for p in _candidate_paths(symbol, timeframe, data_dir))
        raise FileNotFoundError(
            f"No local OHLCV file found for symbol '{symbol}'. Expected one of: {expected}"
        )

    if not frames:
        raise ValueError("No OHLCV rows loaded for the requested symbols")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    LOGGER.info(
        "Loaded %d OHLCV rows across %d symbols",
        len(combined),
        combined["symbol"].nunique(),
    )
    return combined


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV data from a single CSV/parquet file and return normalized output."""
    if not path.exists():
        raise FileNotFoundError(f"OHLC file not found: {path}")
    df = _validate_and_clean(_read_local_file(path), symbol_hint=None)
    LOGGER.info("Loaded %d rows across %d symbols from %s", len(df), df["symbol"].nunique(), path)
    return df


def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format OHLC data into a close-price matrix."""
    close = (
        df.pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .astype(float)
    )
    close = close.dropna(how="all")
    LOGGER.info("Close matrix shape: %s", close.shape)
    return close


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    universe = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "XRP/USD"]
    try:
        sample = load_ohlcv_history(symbols=universe, timeframe="4h", data_dir=Path("data/local"))
        print(sample.head(10).to_string(index=False))
        print(f"Rows: {len(sample)} | Symbols: {sample['symbol'].nunique()}")
    except Exception as exc:
        LOGGER.error("Manual validation failed: %s", exc)
