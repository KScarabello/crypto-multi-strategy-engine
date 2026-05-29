"""CLI runner for research-only walk-forward portfolio validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.portfolio_walk_forward_validation import (
    DEFAULT_FIXED_COMPARISON_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RANKINGS_OUTPUT_PATH,
    run_portfolio_walk_forward_validation,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run walk-forward validation for BTC/CS portfolio candidates")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ranking-output-path", type=Path, default=DEFAULT_RANKINGS_OUTPUT_PATH)
    parser.add_argument("--fixed-comparison-output-path", type=Path, default=DEFAULT_FIXED_COMPARISON_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    steps, aggregates, _, _, fixed_aggregates = run_portfolio_walk_forward_validation(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
        ranking_output_path=args.ranking_output_path,
        fixed_comparison_output_path=args.fixed_comparison_output_path,
    )

    print("\nPortfolio Walk-Forward Validation Steps")
    print(steps.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPortfolio Walk-Forward Aggregate Metrics")
    print(aggregates.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nFixed-Candidate Aggregate Comparison Vs BTC_ONLY")
    print(fixed_aggregates.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        print(
            "\nSaved CSV outputs:\n"
            f"- steps: {args.output_path}\n"
            f"- candidate rankings: {args.ranking_output_path}\n"
            f"- fixed-candidate comparison: {args.fixed_comparison_output_path}"
        )
        LOGGER.info("Saved walk-forward validation steps to %s", args.output_path)
        LOGGER.info("Saved walk-forward candidate rankings to %s", args.ranking_output_path)
        LOGGER.info("Saved fixed-candidate comparison rows to %s", args.fixed_comparison_output_path)


if __name__ == "__main__":
    main()
