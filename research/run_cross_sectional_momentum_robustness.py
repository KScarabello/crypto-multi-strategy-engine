"""Run cross-sectional momentum yearly robustness report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.cross_sectional_momentum_robustness import (
    DEFAULT_OUTPUT_PATH,
    run_cross_sectional_momentum_robustness,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for robustness execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for CS yearly robustness report."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run cross-sectional momentum robustness report by calendar year",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    report = run_cross_sectional_momentum_robustness(
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
        "turnover",
        "percent_time_invested",
        "worst_drawdown_start",
        "worst_drawdown_end",
    ]

    print("\nCross-Sectional Momentum Robustness by Year")
    print(report[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved robustness results to %s", args.output_path)


if __name__ == "__main__":
    main()
