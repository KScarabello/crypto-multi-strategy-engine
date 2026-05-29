"""Run gate parameter stability sweep for sleeve blend gating experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research.sleeve_blend_gate_stability import DEFAULT_OUTPUT_PATH, run_sleeve_blend_gate_stability


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable logging for gate stability execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entrypoint for sleeve blend gate stability sweep."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run MA-length gate stability sweep for BTC plus CS blend gating",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()

    panel, summary = run_sleeve_blend_gate_stability(
        save_csv=not args.no_save_csv,
        output_path=args.output_path,
    )

    columns = [
        "gate_name",
        "ma_length",
        "btc_weight",
        "cs_weight",
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
        "delta_sharpe_vs_always_on_same_weight",
        "delta_max_drawdown_vs_always_on_same_weight",
        "delta_2022_return_vs_always_on_same_weight",
        "delta_sharpe_vs_btc_only",
        "delta_max_drawdown_vs_btc_only",
    ]
    print("\nSleeve Blend Gate Stability")
    print(panel[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBest MA Length By Sharpe")
    if summary["best_ma_by_sharpe"].empty:
        print("(none)")
    else:
        print(summary["best_ma_by_sharpe"][["gate_name", "ma_length", "btc_weight", "cs_weight", "sharpe"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBest MA Length By 2022 Improvement")
    if summary["best_ma_by_2022"].empty:
        print("(none)")
    else:
        print(summary["best_ma_by_2022"][["gate_name", "ma_length", "btc_weight", "cs_weight", "delta_2022_return_vs_always_on_same_weight"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nStable MA Neighborhood")
    print(summary["stable_neighborhood"])

    print("\nMost Conservative Combo")
    if summary["most_conservative"].empty:
        print("(none)")
    else:
        print(summary["most_conservative"][["gate_name", "ma_length", "btc_weight", "cs_weight", "delta_2022_return_vs_always_on_same_weight", "delta_sharpe_vs_always_on_same_weight"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nMost Aggressive Combo")
    print(summary["most_aggressive"][["gate_name", "ma_length", "btc_weight", "cs_weight", "sharpe", "max_drawdown"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_csv:
        LOGGER.info("Saved gate stability sweep to %s", args.output_path)


if __name__ == "__main__":
    main()
