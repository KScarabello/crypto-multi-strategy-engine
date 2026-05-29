"""Run the research-only sleeve blend gating experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_gating_experiment import (
    FULL_OUTPUT_PATH,
    YEARLY_OUTPUT_PATH,
    run_sleeve_blend_gating_experiment,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for sleeve blend gating execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for sleeve blend gating experiment."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run research-only gating experiment for BTC plus CS sleeve blends",
    )
    parser.add_argument("--full-output-path", type=Path, default=FULL_OUTPUT_PATH)
    parser.add_argument("--yearly-output-path", type=Path, default=YEARLY_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    full_report, _ = run_sleeve_blend_gating_experiment(
        save_csv=not args.no_save_csv,
        full_output_path=args.full_output_path,
        yearly_output_path=args.yearly_output_path,
    )

    columns = [
        "gate_rule",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "worst_month_return",
        "percent_time_cs_enabled",
        "sleeve_return_correlation",
    ]

    print("\nSleeve Blend Gating Full-Period Results")
    print(full_report[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved gating full-period results to %s", args.full_output_path)
        LOGGER.info("Saved gating yearly results to %s", args.yearly_output_path)


if __name__ == "__main__":
    main()