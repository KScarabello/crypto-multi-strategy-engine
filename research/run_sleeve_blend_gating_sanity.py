"""Run overfit/stress sanity panel for sleeve blend gating experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_gating_experiment import FULL_OUTPUT_PATH, YEARLY_OUTPUT_PATH
from research.sleeve_blend_gating_sanity import (
    DEFAULT_OUTPUT_PATH,
    run_sleeve_blend_gating_sanity,
    summarize_gating_sanity,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for sleeve blend gating sanity execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for gating sanity panel."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run overfit/stress sanity panel for BTC plus CS blend gating experiment",
    )
    parser.add_argument("--full-input-path", type=Path, default=FULL_OUTPUT_PATH)
    parser.add_argument("--yearly-input-path", type=Path, default=YEARLY_OUTPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    panel = run_sleeve_blend_gating_sanity(
        full_input_path=args.full_input_path,
        yearly_input_path=args.yearly_input_path,
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )
    ranking = summarize_gating_sanity(panel)

    display_cols = [
        "gate_name",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "minimum_yearly_return",
        "count_negative_years",
        "2022_return",
        "delta_sharpe_vs_always_on_same_weight",
        "delta_max_drawdown_vs_always_on_same_weight",
        "delta_2022_return_vs_always_on_same_weight",
        "delta_sharpe_vs_btc_only",
        "delta_max_drawdown_vs_btc_only",
    ]
    print("\nSleeve Blend Gating Sanity Panel")
    print(panel[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    def _print_section(name: str, frame) -> None:
        print(f"\n{name}")
        if frame.empty:
            print("(none)")
            return
        print(frame[["gate_name", "btc_weight", "cs_weight", "sharpe", "max_drawdown", "2022_return", "count_negative_years"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    _print_section("Best Sharpe", ranking["best_sharpe"])
    _print_section("Best Max Drawdown Improvement Vs Always-On", ranking["best_drawdown_improvement"])
    _print_section("Best 2022 Improvement Vs Always-On", ranking["best_2022_improvement"])
    _print_section("Suspicious Candidates", ranking["suspicious"])
    _print_section("Conservative Candidates", ranking["conservative"])

    if not args.no_save_csv:
        LOGGER.info("Saved gating sanity panel to %s", args.output_path)


if __name__ == "__main__":
    main()
