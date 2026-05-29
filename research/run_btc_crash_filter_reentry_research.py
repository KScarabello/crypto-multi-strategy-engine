"""Run exploratory BTC crash-filter re-entry overlay research.

This CLI is research-only and not connected to live trading execution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.btc_crash_filter_reentry_research import (
    BtcCrashFilterReentryResearchConfig,
    DEFAULT_BY_YEAR_OUTPUT_PATH,
    DEFAULT_EVENTS_OUTPUT_PATH,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    run_btc_crash_filter_reentry_research,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run exploratory BTC crash-filter re-entry research")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--base-short-lookback", type=int, default=60)
    parser.add_argument("--base-medium-lookback", type=int, default=240)
    parser.add_argument("--rebalance-bars", type=int, default=12)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--summary-output-path", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--by-year-output-path", type=Path, default=DEFAULT_BY_YEAR_OUTPUT_PATH)
    parser.add_argument("--events-output-path", type=Path, default=DEFAULT_EVENTS_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    config = BtcCrashFilterReentryResearchConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        base_short_lookback=args.base_short_lookback,
        base_medium_lookback=args.base_medium_lookback,
        rebalance_every_bars=args.rebalance_bars,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    summary, by_year, events = run_btc_crash_filter_reentry_research(
        config=config,
        save_csv=not args.no_save_csv,
        summary_output_path=args.summary_output_path,
        by_year_output_path=args.by_year_output_path,
        events_output_path=args.events_output_path,
    )

    display_cols = [
        "candidate_name",
        "base_filter_name",
        "reentry_type",
        "total_return",
        "sharpe",
        "max_drawdown",
        "vs_btc_long_cash_total_return",
        "vs_btc_long_cash_sharpe",
        "vs_original_crash_filter_total_return",
    ]

    print("\nBTC Crash-Filter Re-entry Summary (Research-Only)")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBTC Crash-Filter Re-entry By Year (Research-Only)")
    print(by_year.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBTC Crash-Filter Re-entry Events (first 30 rows)")
    print(events.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        print(
            "\nSaved CSV outputs:\n"
            f"- summary: {args.summary_output_path}\n"
            f"- by-year: {args.by_year_output_path}\n"
            f"- events: {args.events_output_path}"
        )
        LOGGER.info("Saved crash-filter re-entry summary to %s", args.summary_output_path)
        LOGGER.info("Saved crash-filter re-entry by-year report to %s", args.by_year_output_path)
        LOGGER.info("Saved crash-filter re-entry event report to %s", args.events_output_path)


if __name__ == "__main__":
    main()
