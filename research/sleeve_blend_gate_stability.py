"""Gate parameter stability sweep for BTC plus CS sleeve blends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import SETTINGS
from research.sleeve_blend_gating_experiment import (
    GATING_CS_CANDIDATE,
    GateSpec,
    run_sleeve_blend_gating_experiment,
)


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_gate_stability.csv")
STABILITY_ALLOCATIONS: tuple[tuple[float, float], ...] = (
    (0.9, 0.1),
    (0.85, 0.15),
    (0.8, 0.2),
    (0.75, 0.25),
)
STABILITY_MA_LENGTHS: tuple[int, ...] = (80, 100, 120, 140, 160, 180, 200, 240)


def build_gate_stability_specs() -> tuple[GateSpec, ...]:
    """Return the requested non-MA and MA gate definitions for stability sweep."""
    base = (
        GateSpec(name="ALWAYS_ON"),
        GateSpec(name="BTC_TS_INVESTED"),
        GateSpec(name="BTC_PRICE_MOMENTUM_240"),
    )
    ma_specs = tuple(
        GateSpec(name=f"BTC_ABOVE_MA_{window}", ma_window=int(window))
        for window in STABILITY_MA_LENGTHS
    )
    return base + ma_specs


def _validate_allocations(allocations: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Validate requested allocation tuples."""
    if not allocations:
        raise ValueError("allocations must not be empty")
    validated: list[tuple[float, float]] = []
    for btc_weight, cs_weight in allocations:
        btc = float(btc_weight)
        cs = float(cs_weight)
        if btc < 0.0 or cs < 0.0:
            raise ValueError("weights must be non-negative")
        if abs((btc + cs) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")
        validated.append((btc, cs))
    return tuple(validated)


def _year_metric(group: pd.DataFrame, year: str, metric: str) -> float:
    """Extract one yearly metric from grouped yearly report rows."""
    hit = group.loc[group["year"].astype(str) == str(year), metric]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0])


def _gate_name_to_ma_length(gate_name: str) -> float:
    """Parse MA length from gate name when applicable."""
    if gate_name.startswith("BTC_ABOVE_MA_"):
        try:
            return float(int(gate_name.rsplit("_", 1)[-1]))
        except ValueError:
            return float("nan")
    return float("nan")


def build_gate_stability_panel(
    full_report: pd.DataFrame,
    yearly_report: pd.DataFrame,
    keep_allocations: Sequence[tuple[float, float]] = STABILITY_ALLOCATIONS,
) -> pd.DataFrame:
    """Build stability table with yearly stress fields and baseline deltas."""
    desired_allocations = _validate_allocations(keep_allocations)
    full = full_report.copy()
    yearly = yearly_report.copy()
    full["btc_weight"] = full["btc_weight"].astype(float)
    full["cs_weight"] = full["cs_weight"].astype(float)
    yearly["btc_weight"] = yearly["btc_weight"].astype(float)
    yearly["cs_weight"] = yearly["cs_weight"].astype(float)

    records: list[dict[str, float | int | str]] = []
    grouped = yearly.groupby(["gate_rule", "btc_weight", "cs_weight"], sort=False)
    for (gate_rule, btc_weight, cs_weight), group in grouped:
        alloc = (float(btc_weight), float(cs_weight))
        if alloc not in set(desired_allocations):
            continue

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
                "ma_length": _gate_name_to_ma_length(str(gate_rule)),
                "btc_weight": float(btc_weight),
                "cs_weight": float(cs_weight),
                "total_return": float(row["total_return"]),
                "cagr": float(row["cagr"]),
                "annualized_volatility": float(row["annualized_volatility"]),
                "sharpe": float(row["sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "worst_year_return": float(row["worst_year_return"]),
                "worst_month_return": float(row["worst_month_return"]),
                "2022_return": _year_metric(group, year="2022", metric="annual_return"),
                "2022_max_drawdown": _year_metric(group, year="2022", metric="max_drawdown"),
                "count_negative_years": int((annual < 0.0).sum()),
                "percent_time_cs_enabled": float(row["percent_time_cs_enabled"]),
            }
        )

    panel = pd.DataFrame(records)
    if panel.empty:
        raise ValueError("No stability rows produced")

    always_on = panel.loc[panel["gate_name"] == "ALWAYS_ON", ["btc_weight", "cs_weight", "sharpe", "max_drawdown", "2022_return"]]
    always_on = always_on.rename(
        columns={
            "sharpe": "always_on_sharpe",
            "max_drawdown": "always_on_max_drawdown",
            "2022_return": "always_on_2022_return",
        }
    )
    panel = panel.merge(always_on, on=["btc_weight", "cs_weight"], how="left")

    btc_only = full.loc[
        (full["gate_rule"] == "ALWAYS_ON")
        & ((full["btc_weight"] - 1.0).abs() < 1e-12)
        & (full["cs_weight"].abs() < 1e-12)
    ]
    if btc_only.empty:
        raise ValueError("Missing ALWAYS_ON 100/0 row for BTC-only deltas")
    btc_only_row = btc_only.iloc[0]

    panel["delta_sharpe_vs_always_on_same_weight"] = panel["sharpe"] - panel["always_on_sharpe"]
    panel["delta_max_drawdown_vs_always_on_same_weight"] = panel["max_drawdown"] - panel["always_on_max_drawdown"]
    panel["delta_2022_return_vs_always_on_same_weight"] = panel["2022_return"] - panel["always_on_2022_return"]
    panel["delta_sharpe_vs_btc_only"] = panel["sharpe"] - float(btc_only_row["sharpe"])
    panel["delta_max_drawdown_vs_btc_only"] = panel["max_drawdown"] - float(btc_only_row["max_drawdown"])

    panel = panel.drop(columns=["always_on_sharpe", "always_on_max_drawdown", "always_on_2022_return"])
    panel = panel.sort_values(["sharpe", "cagr", "total_return"], ascending=[False, False, False]).reset_index(drop=True)
    return panel


def summarize_gate_stability(panel: pd.DataFrame) -> dict[str, Any]:
    """Compute concise stability summary sections for script output."""
    ma_rows = panel.loc[panel["gate_name"].str.startswith("BTC_ABOVE_MA_")].copy()
    best_ma_by_sharpe = ma_rows.nlargest(1, "sharpe") if not ma_rows.empty else ma_rows
    best_ma_by_2022 = ma_rows.nlargest(1, "delta_2022_return_vs_always_on_same_weight") if not ma_rows.empty else ma_rows

    stable_neighborhood = "insufficient_ma_rows"
    if len(ma_rows) >= 3:
        top = ma_rows.nlargest(1, "sharpe").iloc[0]
        best_len = int(top["ma_length"])
        neighbor_mask = ma_rows["ma_length"].isin([best_len - 20, best_len, best_len + 20])
        neighbors = ma_rows.loc[neighbor_mask]
        threshold = float(top["sharpe"]) * 0.97
        if len(neighbors) >= 2 and (neighbors["sharpe"] >= threshold).sum() >= 2:
            stable_neighborhood = f"stable_around_{best_len}"
        else:
            stable_neighborhood = f"isolated_or_narrow_{best_len}"

    conservative = panel.loc[
        (panel["delta_sharpe_vs_always_on_same_weight"] > 0.0)
        & (panel["delta_2022_return_vs_always_on_same_weight"] >= 0.0)
        & (panel["delta_max_drawdown_vs_always_on_same_weight"] >= -0.03)
    ].copy()
    most_conservative = conservative.nlargest(1, "delta_2022_return_vs_always_on_same_weight") if not conservative.empty else conservative
    most_aggressive = panel.nlargest(1, "sharpe")

    return {
        "best_ma_by_sharpe": best_ma_by_sharpe,
        "best_ma_by_2022": best_ma_by_2022,
        "stable_neighborhood": stable_neighborhood,
        "most_conservative": most_conservative,
        "most_aggressive": most_aggressive,
    }


def run_sleeve_blend_gate_stability(
    allocations: Sequence[tuple[float, float]] = STABILITY_ALLOCATIONS,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run gate stability sweep and return panel plus summary sections."""
    requested_allocations = _validate_allocations(allocations)
    run_allocations = ((1.0, 0.0),) + requested_allocations
    full_report, yearly_report = run_sleeve_blend_gating_experiment(
        allocations=run_allocations,
        gate_specs=build_gate_stability_specs(),
        close_prices=close_prices,
        data_dir=data_dir,
        save_csv=False,
    )
    panel = build_gate_stability_panel(
        full_report=full_report,
        yearly_report=yearly_report,
        keep_allocations=requested_allocations,
    )
    summary = summarize_gate_stability(panel)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
    return panel, summary
