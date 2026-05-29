"""Tests for corrected sleeve blend gating panel."""

from __future__ import annotations

import pandas as pd
import pytest

from research.sleeve_blend_gating_corrected import (
    CORRECTED_CS_CANDIDATE,
    CORRECTED_GATE_SPECS,
    apply_gate_lag,
    build_corrected_gated_stream,
    run_sleeve_blend_gating_corrected,
)
from research.sleeve_blend_experiment import run_sleeve_backtests
from research.sleeve_blend_gating_experiment import build_cs_gate_series


def _toy_universe_close_cross_year(periods: int = 3200) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=periods, freq="4h", tz="UTC")
    data: dict[str, list[float]] = {}
    symbols = [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "ADA/USD",
        "DOGE/USD",
        "LINK/USD",
        "AVAX/USD",
        "LTC/USD",
        "ATOM/USD",
    ]
    split = periods // 2
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 8.0
        up = [base + (0.12 + idx * 0.01) * step for step in range(split)]
        down = [up[-1] - (0.09 + idx * 0.008) * step for step in range(periods - split)]
        data[symbol] = up + down
    return pd.DataFrame(data, index=index, dtype=float)


@pytest.fixture(scope="module")
def corrected_panel() -> pd.DataFrame:
    close = _toy_universe_close_cross_year()
    panel, _ = run_sleeve_blend_gating_corrected(
        allocations=((0.9, 0.1), (0.8, 0.2), (0.75, 0.25)),
        overlay_cost_bps_values=(0.0, 15.0, 30.0),
        close_prices=close,
        save_csv=False,
    )
    return panel


def test_gate_lag_is_applied_correctly() -> None:
    gate = pd.Series([1.0, 0.0, 1.0, 0.0], index=pd.RangeIndex(4), dtype=float)
    lagged = apply_gate_lag(gate, lag_bars=1)
    expected = pd.Series([1.0, 1.0, 0.0, 1.0], index=pd.RangeIndex(4), dtype=float)
    assert lagged.equals(expected)


def test_overlay_cost_reduces_returns_when_turnover_positive() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    aligned = pd.DataFrame(
        {
            "btc_sleeve_return": [0.0, 0.01, -0.01, 0.02, -0.01, 0.01],
            "cs_sleeve_return": [0.0, 0.02, -0.02, 0.03, -0.02, 0.02],
        },
        index=idx,
    )
    gate = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], index=idx, dtype=float)

    no_cost = build_corrected_gated_stream(
        aligned_returns=aligned,
        raw_gate_series=gate,
        btc_weight=0.8,
        cs_weight=0.2,
        gate_lag_bars=1,
        overlay_cost_bps=0.0,
    )
    with_cost = build_corrected_gated_stream(
        aligned_returns=aligned,
        raw_gate_series=gate,
        btc_weight=0.8,
        cs_weight=0.2,
        gate_lag_bars=1,
        overlay_cost_bps=30.0,
    )

    assert float(with_cost["overlay_turnover"].sum()) > 0.0
    assert float(with_cost["overlay_cost"].sum()) > 0.0
    assert float(with_cost["blended_return"].sum()) < float(no_cost["blended_return"].sum())


def test_always_on_rows_unaffected_by_gate_lag_and_overlay_turnover_zero(corrected_panel: pd.DataFrame) -> None:
    rows = corrected_panel.loc[
        (corrected_panel["gate_name"] == "ALWAYS_ON")
        & (corrected_panel["cs_weight"] > 0.0)
    ]
    assert not rows.empty
    assert (rows["overlay_turnover"].abs() < 1e-12).all()


def test_btc_only_baseline_unchanged_across_overlay_cost_scenarios(corrected_panel: pd.DataFrame) -> None:
    baseline = corrected_panel.loc[
        (corrected_panel["gate_name"] == "ALWAYS_ON")
        & ((corrected_panel["btc_weight"] - 1.0).abs() < 1e-12)
        & (corrected_panel["cs_weight"].abs() < 1e-12)
    ].copy()
    assert not baseline.empty
    assert baseline["sharpe"].nunique() == 1
    assert baseline["total_return"].nunique() == 1
    assert (baseline["overlay_turnover"].abs() < 1e-12).all()


def test_same_weight_cost_deltas_reference_correct_always_on_row(corrected_panel: pd.DataFrame) -> None:
    row = corrected_panel.loc[
        (corrected_panel["gate_name"] == "BTC_ABOVE_MA_100")
        & ((corrected_panel["btc_weight"] - 0.8).abs() < 1e-12)
        & ((corrected_panel["cs_weight"] - 0.2).abs() < 1e-12)
        & ((corrected_panel["overlay_cost_bps"] - 15.0).abs() < 1e-12)
    ].iloc[0]

    baseline = corrected_panel.loc[
        (corrected_panel["gate_name"] == "ALWAYS_ON")
        & ((corrected_panel["btc_weight"] - 0.8).abs() < 1e-12)
        & ((corrected_panel["cs_weight"] - 0.2).abs() < 1e-12)
        & ((corrected_panel["overlay_cost_bps"] - 15.0).abs() < 1e-12)
    ].iloc[0]

    assert float(row["delta_sharpe_vs_always_on_same_weight_and_cost"]) == pytest.approx(
        float(row["sharpe"]) - float(baseline["sharpe"]),
        rel=1e-12,
        abs=1e-12,
    )
    assert float(row["delta_max_drawdown_vs_always_on_same_weight_and_cost"]) == pytest.approx(
        float(row["max_drawdown"]) - float(baseline["max_drawdown"]),
        rel=1e-12,
        abs=1e-12,
    )


def test_required_columns_present(corrected_panel: pd.DataFrame) -> None:
    required = {
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
    }
    assert required.issubset(set(corrected_panel.columns))


def test_ma_gate_list_is_covered_in_corrected_panel(corrected_panel: pd.DataFrame) -> None:
    observed = set(corrected_panel.loc[corrected_panel["gate_name"].str.startswith("BTC_ABOVE_MA_"), "gate_name"].unique())
    expected = {spec.name for spec in CORRECTED_GATE_SPECS if spec.name.startswith("BTC_ABOVE_MA_")}
    assert expected.issubset(observed)


def test_corrected_panel_uses_locked_cs_candidate() -> None:
    close = _toy_universe_close_cross_year()
    _, _, _, cs_results = run_sleeve_backtests(cs_candidates=(CORRECTED_CS_CANDIDATE,), close_prices=close)
    assert CORRECTED_CS_CANDIDATE.name in cs_results


def test_gate_lag_changes_nontrivial_ma_gate_series() -> None:
    close = _toy_universe_close_cross_year()
    close_frame, _, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(CORRECTED_CS_CANDIDATE,),
        close_prices=close,
    )
    aligned = pd.concat(
        [
            btc_result.portfolio["strategy_return"].rename("btc_sleeve_return"),
            cs_results[CORRECTED_CS_CANDIDATE.name].portfolio["strategy_return"].rename("cs_sleeve_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    raw_gate = build_cs_gate_series(
        gate_spec=next(spec for spec in CORRECTED_GATE_SPECS if spec.name == "BTC_ABOVE_MA_120"),
        aligned_index=aligned.index,
        btc_close=close_frame["BTC/USD"].reindex(aligned.index),
        btc_invested=btc_result.holdings_history["BTC/USD"].reindex(aligned.index).fillna(0.0),
    )
    lagged_gate = apply_gate_lag(raw_gate, lag_bars=1)
    assert (lagged_gate != raw_gate).any()
