"""Run corrected sleeve blend gating panel with lagged gates and overlay costs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_gating_corrected import (
    DEFAULT_OUTPUT_PATH,
    run_sleeve_blend_gating_corrected,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _print_or_none(frame, columns: list[str]) -> None:
    if frame.empty:
        print("(none)")
    else:
        print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run corrected sleeve blend gating panel",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    panel, summary = run_sleeve_blend_gating_corrected(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    columns = [
        "gate_name",
        "ma_length",
        "btc_weight",
        "cs_weight",
        "gate_lag_bars",
        "overlay_cost_bps",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "worst_month_return",
        "2022_return",
        "2022_max_drawdown",
        "count_negative_years",
        "percent_time_cs_enabled",
        "overlay_turnover",
        "overlay_cost_drag",
        "delta_sharpe_vs_btc_only",
        "delta_max_drawdown_vs_btc_only",
        "delta_sharpe_vs_always_on_same_weight_and_cost",
        "delta_max_drawdown_vs_always_on_same_weight_and_cost",
    ]
    print("\nSleeve Blend Gating Corrected")
    print(panel[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBest Corrected Gate By Sharpe")
    _print_or_none(summary.best_by_sharpe, ["gate_name", "overlay_cost_bps", "btc_weight", "cs_weight", "sharpe", "max_drawdown"])

    print("\nBest Corrected Gate By Max Drawdown")
    _print_or_none(summary.best_by_max_drawdown, ["gate_name", "overlay_cost_bps", "btc_weight", "cs_weight", "max_drawdown", "sharpe"])

    print("\nBest Corrected Gate By 2022 Return")
    _print_or_none(summary.best_by_2022_return, ["gate_name", "overlay_cost_bps", "btc_weight", "cs_weight", "2022_return", "sharpe"])

    print("\nBest Conservative Candidate")
    _print_or_none(summary.best_conservative, ["gate_name", "overlay_cost_bps", "btc_weight", "cs_weight", "delta_sharpe_vs_always_on_same_weight_and_cost", "delta_max_drawdown_vs_always_on_same_weight_and_cost"])

    print("\nBest Aggressive Candidate")
    _print_or_none(summary.best_aggressive, ["gate_name", "overlay_cost_bps", "btc_weight", "cs_weight", "sharpe", "max_drawdown"])

    if not args.no_save_csv:
        LOGGER.info("Saved corrected gating panel to %s", args.output_path)


if __name__ == "__main__":
    main()
