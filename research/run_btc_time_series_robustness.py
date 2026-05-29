"""Run BTC time-series momentum yearly robustness report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.btc_time_series_robustness import DEFAULT_OUTPUT_PATH, run_btc_time_series_robustness


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for robustness execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for BTC yearly robustness report."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run BTC-only time-series robustness report by calendar year",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    report = run_btc_time_series_robustness(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    columns = [
        "year",
        "strategy",
        "total_return",
        "annual_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
    ]

    print("\nBTC Time-Series Robustness by Year")
    print(report[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved robustness results to %s", args.output_path)


if __name__ == "__main__":
    main()
