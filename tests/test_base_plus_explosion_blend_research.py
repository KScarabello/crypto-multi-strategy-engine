from __future__ import annotations

import pandas as pd
import pytest

from research.base_plus_explosion_blend_research import (
    align_return_streams,
    blend_returns_cash_overlay,
    blend_returns_cs_only,
    blend_returns_proportional,
    build_benchmark_comparison_row,
    compute_marginal_contribution,
    equity_curve_from_returns,
    max_drawdown_from_equity,
)


def _idx() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")


def test_base_explosion_alignment_by_timestamp() -> None:
    i = _idx()
    base = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, 0.05], index=i)
    exp = pd.Series([0.1, 0.2, 0.3, 0.4], index=i[2:])
    b, e = align_return_streams(base, exp)
    assert list(b.index) == list(i[2:])
    assert list(e.index) == list(i[2:])


def test_blend_allocation_math_proportional() -> None:
    i = _idx()
    base = pd.Series([0.01] * len(i), index=i)
    exp = pd.Series([0.03] * len(i), index=i)
    out = blend_returns_proportional(base, exp, explosion_weight=0.10)
    assert (out - 0.012).abs().max() < 1e-12


def test_proportional_base_reduction_identity_at_zero_explosion() -> None:
    i = _idx()
    base = pd.Series([0.01, -0.01, 0.0, 0.02, -0.02, 0.01], index=i)
    exp = pd.Series([0.03] * len(i), index=i)
    out = blend_returns_proportional(base, exp, explosion_weight=0.0)
    assert (out - base).abs().max() < 1e-12


def test_cs_only_reduction_formula() -> None:
    i = _idx()
    btc = pd.Series([0.01] * len(i), index=i)
    cs = pd.Series([0.02] * len(i), index=i)
    ov = pd.Series([0.001] * len(i), index=i)
    exp = pd.Series([0.03] * len(i), index=i)
    # alloc 0.10 => cs scale = (0.25-0.10)/0.25 = 0.6
    out = blend_returns_cs_only(btc, cs, ov, exp, explosion_weight=0.10)
    expected = 0.01 + 0.6 * 0.02 - 0.001 + 0.10 * 0.03
    assert (out - expected).abs().max() < 1e-12


def test_cash_overlay_never_exceeds_cash_and_no_leverage() -> None:
    i = _idx()
    base = pd.Series([0.01] * len(i), index=i)
    exp = pd.Series([0.05] * len(i), index=i)
    cash = pd.Series([0.02, 0.1, 0.0, 0.5, 0.03, 0.2], index=i)
    blended, eff = blend_returns_cash_overlay(base, exp, cash, explosion_weight=0.10)
    assert (eff <= 0.10 + 1e-12).all()
    assert (eff <= cash + 1e-12).all()
    assert len(blended) == len(base)


def test_no_lookahead_alignment_intersection_only() -> None:
    base_idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    exp_idx = pd.date_range("2024-01-01 08:00:00", periods=4, freq="4h", tz="UTC")
    base = pd.Series([0.0, 0.01, 0.0, 0.01], index=base_idx)
    exp = pd.Series([0.0, 0.02, 0.0, 0.02], index=exp_idx)
    b, e = align_return_streams(base, exp)
    assert b.index.min() == pd.Timestamp("2024-01-01 08:00:00+00:00")
    assert b.index.max() == pd.Timestamp("2024-01-01 12:00:00+00:00")
    assert len(b) == 2
    assert len(e) == 2


def test_equity_curve_calculation() -> None:
    i = _idx()
    r = pd.Series([0.1, -0.1, 0.0, 0.1, 0.0, 0.0], index=i)
    eq = equity_curve_from_returns(r)
    assert eq.iloc[0] == pytest.approx(1.1)
    assert eq.iloc[1] == pytest.approx(0.99)


def test_drawdown_calculation() -> None:
    i = _idx()
    eq = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0, 1.3], index=i)
    mdd = max_drawdown_from_equity(eq)
    # trough 0.9 from prior peak 1.2 => -25%
    assert mdd == pytest.approx(-0.25, abs=1e-12)


def test_marginal_contribution_calculation() -> None:
    i = _idx()
    blend = pd.Series([0.02] * len(i), index=i)
    exp = pd.Series([0.01] * len(i), index=i)
    out = compute_marginal_contribution(blend, exp, explosion_weight=0.10)
    assert out["explosion_contribution_mean"] == pytest.approx(0.001, abs=1e-12)
    assert out["explosion_contribution_sum"] == pytest.approx(0.006, abs=1e-12)


def test_benchmark_comparison_row_contains_deltas() -> None:
    blend_metrics = {"total_return": 0.5, "cagr": 0.1, "sharpe": 1.2, "max_drawdown": -0.3}
    bench_metrics = {"total_return": 0.4, "cagr": 0.08, "sharpe": 1.0, "max_drawdown": -0.35}
    row = build_benchmark_comparison_row("blendA", "BASE_ONLY", blend_metrics, bench_metrics)
    assert row["delta_total_return"] == pytest.approx(0.1, abs=1e-12)
    assert row["delta_cagr"] == pytest.approx(0.02, abs=1e-12)
    assert row["delta_sharpe"] == pytest.approx(0.2, abs=1e-12)
    assert row["delta_max_drawdown"] == pytest.approx(0.05, abs=1e-12)
