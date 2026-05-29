"""Run the focused sleeve blend robustness sweep."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_robustness import DEFAULT_OUTPUT_PATH, run_sleeve_blend_robustness


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for sleeve blend robustness execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for the focused sleeve blend robustness sweep."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run focused robustness sweep for BTC TS plus CS sleeve blends",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    report = run_sleeve_blend_robustness(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    columns = [
        "blend_name",
        "cs_strategy",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "sleeve_return_correlation",
        "delta_total_return",
        "delta_cagr",
        "delta_sharpe",
        "delta_max_drawdown",
    ]

    print("\nSleeve Blend Robustness Results")
    print(report[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved sleeve blend robustness results to %s", args.output_path)


if __name__ == "__main__":
    main()