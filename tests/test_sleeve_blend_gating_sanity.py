"""Tests for sleeve blend gating sanity panel."""

from __future__ import annotations

import pandas as pd
import pytest

from research.sleeve_blend_gating_sanity import build_gating_sanity_panel


def _mini_full_report() -> pd.DataFrame:
    rows = [
        {
            "year": "FULL",
            "gate_rule": "ALWAYS_ON",
            "btc_weight": 1.0,
            "cs_weight": 0.0,
            "total_return": 10.0,
            "cagr": 0.5,
            "sharpe": 1.0,
            "max_drawdown": -0.40,
            "worst_year_return": -0.20,
            "worst_month_return": -0.10,
            "percent_time_cs_enabled": 0.0,
        },
        {
            "year": "FULL",
            "gate_rule": "ALWAYS_ON",
            "btc_weight": 0.9,
            "cs_weight": 0.1,
            "total_return": 12.0,
            "cagr": 0.55,
            "sharpe": 1.10,
            "max_drawdown": -0.45,
            "worst_year_return": -0.25,
            "worst_month_return": -0.12,
            "percent_time_cs_enabled": 1.0,
        },
        {
            "year": "FULL",
            "gate_rule": "BTC_TS_INVESTED",
            "btc_weight": 0.9,
            "cs_weight": 0.1,
            "total_return": 13.0,
            "cagr": 0.57,
            "sharpe": 1.20,
            "max_drawdown": -0.42,
            "worst_year_return": -0.18,
            "worst_month_return": -0.11,
            "percent_time_cs_enabled": 0.6,
        },
    ]
    return pd.DataFrame(rows)


def _mini_yearly_report() -> pd.DataFrame:
    rows = [
        {"year": "2022", "gate_rule": "ALWAYS_ON", "btc_weight": 1.0, "cs_weight": 0.0, "annual_return": -0.20, "max_drawdown": -0.30},
        {"year": "2024", "gate_rule": "ALWAYS_ON", "btc_weight": 1.0, "cs_weight": 0.0, "annual_return": 0.10, "max_drawdown": -0.20},
        {"year": "2025", "gate_rule": "ALWAYS_ON", "btc_weight": 1.0, "cs_weight": 0.0, "annual_return": -0.05, "max_drawdown": -0.25},
        {"year": "2026", "gate_rule": "ALWAYS_ON", "btc_weight": 1.0, "cs_weight": 0.0, "annual_return": 0.01, "max_drawdown": -0.10},
        {"year": "2022", "gate_rule": "ALWAYS_ON", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": -0.30, "max_drawdown": -0.40},
        {"year": "2024", "gate_rule": "ALWAYS_ON", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": 0.12, "max_drawdown": -0.22},
        {"year": "2025", "gate_rule": "ALWAYS_ON", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": -0.08, "max_drawdown": -0.28},
        {"year": "2026", "gate_rule": "ALWAYS_ON", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": -0.02, "max_drawdown": -0.12},
        {"year": "2022", "gate_rule": "BTC_TS_INVESTED", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": -0.10, "max_drawdown": -0.25},
        {"year": "2024", "gate_rule": "BTC_TS_INVESTED", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": 0.15, "max_drawdown": -0.20},
        {"year": "2025", "gate_rule": "BTC_TS_INVESTED", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": -0.03, "max_drawdown": -0.24},
        {"year": "2026", "gate_rule": "BTC_TS_INVESTED", "btc_weight": 0.9, "cs_weight": 0.1, "annual_return": 0.02, "max_drawdown": -0.08},
    ]
    return pd.DataFrame(rows)


def test_sanity_output_required_columns() -> None:
    panel = build_gating_sanity_panel(_mini_full_report(), _mini_yearly_report())
    required = {
        "gate_name",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "minimum_yearly_return",
        "count_negative_years",
        "count_positive_years",
        "2022_return",
        "2022_max_drawdown",
        "2024_return",
        "2025_return",
        "2026_return",
        "worst_month_return",
        "percent_time_cs_enabled",
        "delta_sharpe_vs_always_on_same_weight",
        "delta_max_drawdown_vs_always_on_same_weight",
        "delta_2022_return_vs_always_on_same_weight",
        "delta_2022_drawdown_vs_always_on_same_weight",
        "delta_sharpe_vs_btc_only",
        "delta_max_drawdown_vs_btc_only",
    }
    assert required.issubset(set(panel.columns))


def test_same_weight_deltas_use_always_on_reference() -> None:
    panel = build_gating_sanity_panel(_mini_full_report(), _mini_yearly_report())
    row = panel.loc[
        (panel["gate_name"] == "BTC_TS_INVESTED")
        & ((panel["btc_weight"] - 0.9).abs() < 1e-12)
        & ((panel["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]

    assert float(row["delta_sharpe_vs_always_on_same_weight"]) == pytest.approx(0.10, abs=1e-12)
    assert float(row["delta_max_drawdown_vs_always_on_same_weight"]) == pytest.approx(0.03, abs=1e-12)
    assert float(row["delta_2022_return_vs_always_on_same_weight"]) == pytest.approx(0.20, abs=1e-12)
    assert float(row["delta_2022_drawdown_vs_always_on_same_weight"]) == pytest.approx(0.15, abs=1e-12)


def test_btc_only_deltas_use_100_0_baseline() -> None:
    panel = build_gating_sanity_panel(_mini_full_report(), _mini_yearly_report())
    row = panel.loc[
        (panel["gate_name"] == "BTC_TS_INVESTED")
        & ((panel["btc_weight"] - 0.9).abs() < 1e-12)
        & ((panel["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]

    assert float(row["delta_sharpe_vs_btc_only"]) == pytest.approx(0.20, abs=1e-12)
    assert float(row["delta_max_drawdown_vs_btc_only"]) == pytest.approx(-0.02, abs=1e-12)


def test_year_2022_fields_present_when_available() -> None:
    panel = build_gating_sanity_panel(_mini_full_report(), _mini_yearly_report())
    assert panel["2022_return"].notna().all()
    assert panel["2022_max_drawdown"].notna().all()
