"""Download and persist historical OHLCV data for research and live trading.

Adapted from crypto-momentum-strategy download_ohlcv.py for multi-strategy use.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import SETTINGS, get_data_symbols
from data.fetch_ohlc import update_symbol_ohlcv_incremental


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
    download_all_symbols()


if __name__ == "__main__":
    main()
