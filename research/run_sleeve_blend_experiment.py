"""Run the research-only BTC plus CS sleeve blend experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_experiment import DEFAULT_OUTPUT_PATH, run_sleeve_blend_experiment


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for sleeve blend execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for the sleeve blend research experiment."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run BTC time-series plus CS momentum sleeve blend experiment",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    report = run_sleeve_blend_experiment(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    full_period = report.loc[report["year"] == "FULL"].sort_values("sharpe", ascending=False)
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
    ]

    print("\nSleeve Blend Experiment Full-Period Results")
    print(full_period[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved sleeve blend results to %s", args.output_path)


if __name__ == "__main__":
    main()