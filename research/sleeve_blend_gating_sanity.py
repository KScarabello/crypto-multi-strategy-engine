"""Overfit/stress sanity panel for sleeve blend gating experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.sleeve_blend_gating_experiment import (
    FULL_OUTPUT_PATH,
    YEARLY_OUTPUT_PATH,
    run_sleeve_blend_gating_experiment,
)


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_gating_sanity.csv")


def _load_or_build_reports(
    full_input_path: Path = FULL_OUTPUT_PATH,
    yearly_input_path: Path = YEARLY_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gating reports from disk, generating them when missing."""
    if full_input_path.exists() and yearly_input_path.exists():
        return pd.read_csv(full_input_path), pd.read_csv(yearly_input_path)
    return run_sleeve_blend_gating_experiment(
        save_csv=True,
        full_output_path=full_input_path,
        yearly_output_path=yearly_input_path,
    )


def _metrics_for_year(group: pd.DataFrame, year: str, metric: str) -> float:
    """Return one yearly metric for a grouped gate/allocation row."""
    hit = group.loc[group["year"].astype(str) == str(year), metric]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0])


def build_gating_sanity_panel(
    full_report: pd.DataFrame,
    yearly_report: pd.DataFrame,
) -> pd.DataFrame:
    """Build the requested overfit/stress sanity panel from gating reports."""
    full = full_report.copy()
    yearly = yearly_report.copy()
    full["btc_weight"] = full["btc_weight"].astype(float)
    full["cs_weight"] = full["cs_weight"].astype(float)
    yearly["btc_weight"] = yearly["btc_weight"].astype(float)
    yearly["cs_weight"] = yearly["cs_weight"].astype(float)

    records: list[dict[str, float | int | str | bool]] = []
    grouped = yearly.groupby(["gate_rule", "btc_weight", "cs_weight"], sort=False)
    for (gate_rule, btc_weight, cs_weight), group in grouped:
        full_match = full.loc[
            (full["gate_rule"] == gate_rule)
            & ((full["btc_weight"] - float(btc_weight)).abs() < 1e-12)
            & ((full["cs_weight"] - float(cs_weight)).abs() < 1e-12)
        ]
        if full_match.empty:
            continue
        row = full_match.iloc[0]
        annual = group["annual_return"].astype(float)

        records.append(
            {
                "gate_name": str(gate_rule),
                "btc_weight": float(btc_weight),
                "cs_weight": float(cs_weight),
                "total_return": float(row["total_return"]),
                "cagr": float(row["cagr"]),
                "sharpe": float(row["sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "worst_year_return": float(row["worst_year_return"]),
                "minimum_yearly_return": float(annual.min()) if not annual.empty else float("nan"),
                "count_negative_years": int((annual < 0.0).sum()),
                "count_positive_years": int((annual > 0.0).sum()),
                "2022_return": _metrics_for_year(group, year="2022", metric="annual_return"),
                "2022_max_drawdown": _metrics_for_year(group, year="2022", metric="max_drawdown"),
                "2024_return": _metrics_for_year(group, year="2024", metric="annual_return"),
                "2025_return": _metrics_for_year(group, year="2025", metric="annual_return"),
                "2026_return": _metrics_for_year(group, year="2026", metric="annual_return"),
                "worst_month_return": float(row["worst_month_return"]),
                "percent_time_cs_enabled": float(row["percent_time_cs_enabled"]),
            }
        )

    panel = pd.DataFrame(records)
    if panel.empty:
        raise ValueError("No sanity rows were produced from gating reports")

    always_on = panel.loc[panel["gate_name"] == "ALWAYS_ON", ["btc_weight", "cs_weight", "sharpe", "max_drawdown", "2022_return", "2022_max_drawdown"]].copy()
    always_on = always_on.rename(
        columns={
            "sharpe": "always_on_sharpe",
            "max_drawdown": "always_on_max_drawdown",
            "2022_return": "always_on_2022_return",
            "2022_max_drawdown": "always_on_2022_max_drawdown",
        }
    )
    panel = panel.merge(always_on, on=["btc_weight", "cs_weight"], how="left")

    btc_only = panel.loc[
        (panel["gate_name"] == "ALWAYS_ON")
        & ((panel["btc_weight"] - 1.0).abs() < 1e-12)
        & (panel["cs_weight"].abs() < 1e-12)
    ]
    if btc_only.empty:
        raise ValueError("Missing ALWAYS_ON 100/0 baseline row for BTC-only deltas")
    btc_only_row = btc_only.iloc[0]

    panel["delta_sharpe_vs_always_on_same_weight"] = panel["sharpe"] - panel["always_on_sharpe"]
    panel["delta_max_drawdown_vs_always_on_same_weight"] = panel["max_drawdown"] - panel["always_on_max_drawdown"]
    panel["delta_2022_return_vs_always_on_same_weight"] = panel["2022_return"] - panel["always_on_2022_return"]
    panel["delta_2022_drawdown_vs_always_on_same_weight"] = panel["2022_max_drawdown"] - panel["always_on_2022_max_drawdown"]
    panel["delta_sharpe_vs_btc_only"] = panel["sharpe"] - float(btc_only_row["sharpe"])
    panel["delta_max_drawdown_vs_btc_only"] = panel["max_drawdown"] - float(btc_only_row["max_drawdown"])

    panel["suspicious_candidate"] = (
        (panel["sharpe"] >= panel["sharpe"].quantile(0.8))
        & ((panel["count_negative_years"] >= 3) | (panel["2022_return"] < -0.15))
    )
    panel["conservative_candidate"] = (
        (panel["delta_sharpe_vs_always_on_same_weight"] > 0.0)
        & (panel["delta_2022_return_vs_always_on_same_weight"] >= -0.02)
        & (panel["delta_max_drawdown_vs_always_on_same_weight"] >= -0.03)
    )

    drop_cols = [
        "always_on_sharpe",
        "always_on_max_drawdown",
        "always_on_2022_return",
        "always_on_2022_max_drawdown",
    ]
    panel = panel.drop(columns=drop_cols)
    panel = panel.sort_values(["sharpe", "cagr", "total_return"], ascending=[False, False, False]).reset_index(drop=True)
    return panel


def summarize_gating_sanity(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return compact ranking/flag sections for the sanity panel."""
    best_sharpe = panel.nlargest(1, "sharpe")
    best_drawdown_improvement = panel.nlargest(1, "delta_max_drawdown_vs_always_on_same_weight")
    best_2022_improvement = panel.nlargest(1, "delta_2022_return_vs_always_on_same_weight")
    suspicious = panel.loc[panel["suspicious_candidate"]].copy()
    conservative = panel.loc[panel["conservative_candidate"]].copy()
    return {
        "best_sharpe": best_sharpe,
        "best_drawdown_improvement": best_drawdown_improvement,
        "best_2022_improvement": best_2022_improvement,
        "suspicious": suspicious,
        "conservative": conservative,
    }


def run_sleeve_blend_gating_sanity(
    full_input_path: Path = FULL_OUTPUT_PATH,
    yearly_input_path: Path = YEARLY_OUTPUT_PATH,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run sanity panel build from existing gating experiment outputs."""
    full_report, yearly_report = _load_or_build_reports(
        full_input_path=full_input_path,
        yearly_input_path=yearly_input_path,
    )
    panel = build_gating_sanity_panel(full_report=full_report, yearly_report=yearly_report)
    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
    return panel
