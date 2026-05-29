"""Run yearly robustness report for selected sleeve blends."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_yearly_robustness import DEFAULT_OUTPUT_PATH, run_sleeve_blend_yearly_robustness


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for yearly sleeve blend robustness execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for yearly sleeve blend robustness report."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run yearly robustness report for selected BTC plus CS sleeve blends",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    report = run_sleeve_blend_yearly_robustness(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    columns = [
        "year",
        "btc_weight",
        "cs_weight",
        "cs_candidate",
        "total_return",
        "annual_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_month_return",
        "growth_of_1",
    ]

    print("\nSleeve Blend Yearly Robustness")
    print(report[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved yearly sleeve blend robustness results to %s", args.output_path)


if __name__ == "__main__":
    main()