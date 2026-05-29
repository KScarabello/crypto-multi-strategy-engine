"""Run BTC-only time-series momentum parameter sweep."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.btc_time_series_sweep import (
    DEFAULT_MEDIUM_LOOKBACKS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REBALANCE_EVERY,
    DEFAULT_SHORT_LOOKBACKS,
    run_btc_time_series_sweep,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for sweep execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for BTC momentum parameter sweep."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run BTC-only time-series momentum parameter sweep",
    )
    parser.add_argument("--short-lookbacks", type=int, nargs="+", default=list(DEFAULT_SHORT_LOOKBACKS))
    parser.add_argument("--medium-lookbacks", type=int, nargs="+", default=list(DEFAULT_MEDIUM_LOOKBACKS))
    parser.add_argument("--rebalance-every", type=int, nargs="+", default=list(DEFAULT_REBALANCE_EVERY))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    result = run_btc_time_series_sweep(
        short_lookbacks=tuple(args.short_lookbacks),
        medium_lookbacks=tuple(args.medium_lookbacks),
        rebalance_every=tuple(args.rebalance_every),
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    display_columns = [
        "short_lookback",
        "medium_lookback",
        "rebalance_every",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
        "vs_buy_hold_total_return",
        "vs_buy_hold_cagr",
        "vs_buy_hold_sharpe",
        "vs_buy_hold_max_drawdown",
    ]

    print("\nBTC Time-Series Momentum Sweep (sorted by Sharpe)")
    print(result[display_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved sweep results to %s", args.output_path)


if __name__ == "__main__":
    main()
