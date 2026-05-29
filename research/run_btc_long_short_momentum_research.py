"""Run exploratory BTC long/short momentum research.

This CLI is research-only and not connected to live trading execution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.btc_long_short_momentum_research import (
    BtcLongShortResearchConfig,
    DEFAULT_BY_YEAR_OUTPUT_PATH,
    DEFAULT_SHORT_TRADES_OUTPUT_PATH,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    run_btc_long_short_momentum_research,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run exploratory BTC long/short momentum research")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--short-lookback", type=int, default=60)
    parser.add_argument("--medium-lookback", type=int, default=240)
    parser.add_argument("--rebalance-bars", type=int, default=12)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--summary-output-path", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--by-year-output-path", type=Path, default=DEFAULT_BY_YEAR_OUTPUT_PATH)
    parser.add_argument("--short-trades-output-path", type=Path, default=DEFAULT_SHORT_TRADES_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    config = BtcLongShortResearchConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        short_lookback_bars=args.short_lookback,
        medium_lookback_bars=args.medium_lookback,
        rebalance_every_bars=args.rebalance_bars,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    summary, by_year, short_trades = run_btc_long_short_momentum_research(
        config=config,
        save_csv=not args.no_save_csv,
        summary_output_path=args.summary_output_path,
        by_year_output_path=args.by_year_output_path,
        short_trades_output_path=args.short_trades_output_path,
    )

    print("\nBTC Long/Short Momentum Summary (Research-Only)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBTC Long/Short Momentum By Year (Research-Only)")
    print(by_year.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nShort Trade Diagnostics (first 30 rows)")
    print(short_trades.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        print(
            "\nSaved CSV outputs:\n"
            f"- summary: {args.summary_output_path}\n"
            f"- by-year: {args.by_year_output_path}\n"
            f"- short trades: {args.short_trades_output_path}"
        )
        LOGGER.info("Saved long/short summary to %s", args.summary_output_path)
        LOGGER.info("Saved long/short by-year report to %s", args.by_year_output_path)
        LOGGER.info("Saved long/short short-trade report to %s", args.short_trades_output_path)


if __name__ == "__main__":
    main()
