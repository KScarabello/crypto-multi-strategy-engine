"""Run BTC-only buy-and-hold vs time-series momentum experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import SETTINGS
from research.btc_time_series import (
    BtcTimeSeriesConfig,
    format_btc_experiment_summary,
    run_btc_time_series_experiment,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH = Path("research/results/btc_time_series_experiment_metrics.csv")


def configure_logging() -> None:
    """Configure readable logging for research execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for BTC-only time-series momentum research."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run BTC buy-and-hold vs BTC time-series momentum experiment",
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--short-lookback", type=int, default=42)
    parser.add_argument("--medium-lookback", type=int, default=180)
    parser.add_argument("--rebalance-bars", type=int, default=6)
    parser.add_argument("--short-weight", type=float, default=0.5)
    parser.add_argument("--medium-weight", type=float, default=0.5)
    parser.add_argument("--initial-capital", type=float, default=SETTINGS.initial_capital)
    parser.add_argument("--transaction-cost-bps", type=float, default=SETTINGS.transaction_cost_bps)
    parser.add_argument("--slippage-bps", type=float, default=SETTINGS.slippage_bps)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    config = BtcTimeSeriesConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        short_lookback_bars=args.short_lookback,
        medium_lookback_bars=args.medium_lookback,
        short_weight=args.short_weight,
        medium_weight=args.medium_weight,
        rebalance_every_bars=args.rebalance_bars,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    report = run_btc_time_series_experiment(config=config)
    summary = format_btc_experiment_summary(report)

    print("\nBTC Time-Series Experiment")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_path, index=False)
        LOGGER.info("Saved BTC experiment metrics to %s", args.output_path)


if __name__ == "__main__":
    main()
