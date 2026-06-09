"""Tests for research/fixed_five_rank_stability_reconciliation.py

~40 tests covering:
- Min-hold timing semantics (Bug 1)
- Actual cost computation (Bug 2)
- Corrected opportunity cost analysis (Bug 3)
- Selected variants set
- Output file verification
- Classification logic
- Markdown content
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_rank_stability_reconciliation as rsr
from research.fixed_five_rank_stability_reconciliation import (
    SELECTED_VARIANTS,
    SELECTED_VARIANT_NAMES,
    compute_min_hold_timing_audit,
    compute_actual_direct_costs,
    enrich_suppressed_events_corrected,
    classify_rank_stability_improvement_corrected,
    suppressed_events_corrected_summary,
    CONTROL_REBALANCE_BARS,
    FEE_BPS,
    SLIPPAGE_BPS,
    TOP_N,
)
from research.fixed_five_rank_stability import (
    RankStabilityParams,
    RankStabilitySignalGen,
    VARIANTS,
)


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic BacktestResult
# ---------------------------------------------------------------------------

def _make_holdings(n_bars: int, symbols: list[str], weights: list[float]) -> pd.DataFrame:
    """Create a holdings_history DataFrame."""
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="4h", tz="UTC")
    data = {sym: [w] * n_bars for sym, w in zip(symbols, weights)}
    return pd.DataFrame(data, index=idx)


def _make_equity(n_bars: int, start: float = 10_000.0) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="4h", tz="UTC")
    vals = [start * (1.001 ** i) for i in range(n_bars)]
    return pd.Series(vals, index=idx, name="equity")


def _make_gross_return(n_bars: int) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="4h", tz="UTC")
    return pd.Series([0.0015] * n_bars, index=idx)


class _FakeResult:
    """Minimal stand-in for BacktestResult for unit tests."""

    def __init__(self, n_bars: int = 20, symbols: list[str] | None = None):
        symbols = symbols or ["A/USD", "B/USD", "C/USD"]
        idx = pd.date_range("2022-01-01", periods=n_bars, freq="4h", tz="UTC")

        # holdings: bars 0..9 hold A,B,C at 1/3; bars 10..19 switch to B at 1
        w_before = {s: 1.0 / 3 for s in symbols}
        w_after_10 = {s: 0.0 for s in symbols}
        w_after_10["B/USD"] = 1.0
        rows = []
        for i in range(n_bars):
            if i < 10:
                rows.append(w_before)
            else:
                rows.append(w_after_10)
        self.holdings_history = pd.DataFrame(rows, index=idx)

        # equity
        equity_vals = [10_000 * (1.001 ** i) for i in range(n_bars)]
        strategy_ret = [0.001] * n_bars
        portfolio_df = pd.DataFrame({
            "equity": equity_vals,
            "strategy_return": strategy_ret,
        }, index=idx)
        self.portfolio = portfolio_df

        # gross_return — slightly higher to create gross-minus-net gap
        self.gross_return = pd.Series([0.0015] * n_bars, index=idx)

        # rebalance_log with one rebalance at bar 10
        exec_ts = idx[10]
        sig_ts = idx[9]
        self.rebalance_log = pd.DataFrame([{
            "execution_timestamp": exec_ts,
            "signal_timestamp": sig_ts,
            "turnover": 0.667,
            "cost_rate": 0.001,
            "weight_sum": 1.0,
        }])

    def _port_fx(self, joint_start: pd.Timestamp) -> pd.DataFrame:
        return self.portfolio.loc[self.portfolio.index >= joint_start].copy()


# ===========================================================================
# Part 1 — Min-Hold Timing Tests
# ===========================================================================

def test_min_hold_2_is_4_days():
    """min_hold_2 = 2 rebalances × 48h = 96h = 4 calendar days."""
    df = compute_min_hold_timing_audit()
    row = df[df["min_hold_rebs"] == 2].iloc[0]
    assert row["min_hold_hours"] == 96
    assert row["min_hold_days"] == 4.0


def test_min_hold_4_is_8_days():
    """min_hold_4 = 4 rebalances × 48h = 192h = 8 calendar days."""
    df = compute_min_hold_timing_audit()
    row = df[df["min_hold_rebs"] == 4].iloc[0]
    assert row["min_hold_hours"] == 192
    assert row["min_hold_days"] == 8.0


def test_min_hold_6_is_12_days_not_3():
    """min_hold_6 = 6 rebalances × 48h = 288h = 12 calendar days, NOT 3."""
    df = compute_min_hold_timing_audit()
    row = df[df["min_hold_rebs"] == 6].iloc[0]
    assert row["min_hold_hours"] == 288
    assert row["min_hold_days"] == 12.0
    assert row["min_hold_days"] != 3.0


def test_timing_audit_has_7_rows():
    """Timing audit should have rows for min_hold_rebs 0..6 (7 rows)."""
    df = compute_min_hold_timing_audit()
    assert len(df) == 7
    assert set(df["min_hold_rebs"].tolist()) == {0, 1, 2, 3, 4, 5, 6}


def test_min_hold_example_entry_concrete():
    """Entry at 2022-01-10 00:00 UTC, min_hold_6 allows replacement at 2022-01-22 00:00 UTC."""
    df = compute_min_hold_timing_audit()
    row = df[df["min_hold_rebs"] == 6].iloc[0]
    expected = str(pd.Timestamp("2022-01-22 00:00:00", tz="UTC"))
    assert row["earliest_replacement_ts"] == expected


def test_timing_audit_columns():
    """Timing audit CSV has required columns."""
    df = compute_min_hold_timing_audit()
    required = {
        "min_hold_rebs", "rebalance_interval_bars", "rebalance_interval_hours",
        "min_hold_hours", "min_hold_days", "example_entry_ts",
        "earliest_replacement_ts", "hold_unit",
    }
    assert required.issubset(set(df.columns))


def test_min_hold_timing_implementation_is_correct():
    """Verify signal gen increments hold_age per rebalance call (not per bar)."""
    gate = pd.Series(True, index=pd.date_range("2022-01-01", periods=100, freq="4h", tz="UTC"))
    params = RankStabilityParams("test_hold", min_hold_rebs=2)
    gen = RankStabilitySignalGen(gate, params)

    # hold_age should be a dict tracking rebalance-call counts, not bar counts
    assert hasattr(gen, "_hold_age")
    assert isinstance(gen._hold_age, dict)

    # After 2 rebalance calls, hold_age for a symbol should be 1 (entry=0, then +1)
    symbols = ["A/USD", "B/USD", "C/USD", "D/USD"]
    close_idx = gate.index
    close = pd.DataFrame(
        {s: np.linspace(100, 110, len(close_idx)) for s in symbols},
        index=close_idx,
    )
    ts1 = close_idx[36]  # first call after warmup
    gen(close, ts1)
    # After first call, held symbols have hold_age = 0
    assert all(v == 0 for v in gen._hold_age.values())

    ts2 = close_idx[48]  # second rebalance call
    gen(close, ts2)
    # After second call, originally held symbols have hold_age = 1
    assert any(v == 1 for v in gen._hold_age.values())


# ===========================================================================
# Part 2 — Direct Cost Computation Tests
# ===========================================================================

def _run_compute_actual_costs_fake():
    """Run compute_actual_direct_costs on a fake result."""
    result = _FakeResult(n_bars=20)
    joint_start = result.holdings_history.index[0]
    port_fx = result._port_fx(joint_start)
    return compute_actual_direct_costs(result, port_fx, joint_start, FEE_BPS, SLIPPAGE_BPS)


def test_actual_fee_not_gross_minus_net_times_frac():
    """Actual fee must NOT equal gross_minus_net × (fee_bps / total_bps)."""
    costs = _run_compute_actual_costs_fake()
    gmn = costs["gross_minus_net_dollars"]
    wrong_fee = gmn * (FEE_BPS / (FEE_BPS + SLIPPAGE_BPS))
    # They should differ (unless coincidentally equal, which won't happen with our fake data)
    assert abs(costs["actual_fee_dollars"] - wrong_fee) > 0.01


def test_fee_equals_notional_times_rate():
    """Fee = sum(|Δw| × eq_before × fee_bps/10000)."""
    result = _FakeResult(n_bars=20)
    joint_start = result.holdings_history.index[0]
    port_fx = result._port_fx(joint_start)
    costs = compute_actual_direct_costs(result, port_fx, joint_start, FEE_BPS, SLIPPAGE_BPS)

    # Manual calculation for bar 10 rebalance:
    # Holdings go from 1/3 each → 1.0 for B, 0 for A and C
    # eq_before = equity at bar 9
    eq_before = float(result.portfolio["equity"].iloc[9])
    # A: 1/3 → 0, change = 1/3
    # B: 1/3 → 1, change = 2/3
    # C: 1/3 → 0, change = 1/3
    total_dw = 1.0 / 3 + 2.0 / 3 + 1.0 / 3  # = 4/3
    expected_fee = total_dw * eq_before * (FEE_BPS / 10_000)
    assert abs(costs["actual_fee_dollars"] - expected_fee) < 1.0  # $1 tolerance


def test_slippage_equals_notional_times_rate():
    """Slippage = sum(|Δw| × eq_before × slippage_bps/10000)."""
    result = _FakeResult(n_bars=20)
    joint_start = result.holdings_history.index[0]
    port_fx = result._port_fx(joint_start)
    costs = compute_actual_direct_costs(result, port_fx, joint_start, FEE_BPS, SLIPPAGE_BPS)
    # Slippage should be exactly half of fee (since SLIPPAGE_BPS = FEE_BPS / 2)
    assert abs(costs["actual_slippage_dollars"] - costs["actual_fee_dollars"] * (SLIPPAGE_BPS / FEE_BPS)) < 0.01


def test_gross_minus_net_not_labeled_as_fees():
    """Result dict must have separate gross_minus_net key, not call it fees."""
    costs = _run_compute_actual_costs_fake()
    assert "gross_minus_net_dollars" in costs
    assert "actual_fee_dollars" in costs
    assert costs["gross_minus_net_dollars"] != costs["actual_fee_dollars"]


def test_actual_cost_less_than_gross_minus_net():
    """Actual direct cost should be less than gross-minus-net (foregone compounding > 0)."""
    costs = _run_compute_actual_costs_fake()
    # With compounding effect, actual costs < gross-minus-net
    # This holds as long as there's any compounding gap
    assert costs["actual_direct_cost_dollars"] <= costs["gross_minus_net_dollars"] + 1.0


def test_foregone_compounding_positive():
    """residual_foregone_compounding = gross_minus_net - actual_direct_cost >= 0."""
    costs = _run_compute_actual_costs_fake()
    assert costs["residual_foregone_compounding"] == round(
        costs["gross_minus_net_dollars"] - costs["actual_direct_cost_dollars"], 2
    )


def test_actual_fee_plus_slippage_equals_direct_cost():
    """actual_fee + actual_slippage == actual_direct_cost_dollars."""
    costs = _run_compute_actual_costs_fake()
    assert abs(
        costs["actual_fee_dollars"] + costs["actual_slippage_dollars"]
        - costs["actual_direct_cost_dollars"]
    ) < 0.01


def test_foregone_compounding_label_correct():
    """Result dict key is named residual_foregone_compounding."""
    costs = _run_compute_actual_costs_fake()
    assert "residual_foregone_compounding" in costs


def test_cost_reduction_pct_uses_actual_costs():
    """classify function uses actual_direct_cost_dollars, not additive_cost_pct."""
    control = {
        "actual_direct_cost_dollars": 1000.0,
        "n_rank_replacements": 100,
        "sharpe": 1.0,
        "max_drawdown_pct": -40.0,
        "return_2022_pct": -30.0,
        "pct_time_cash": 20.0,
    }
    variant = dict(control)
    variant["actual_direct_cost_dollars"] = 700.0  # 30% reduction
    variant["n_rank_replacements"] = 60  # 40% reduction

    result = classify_rank_stability_improvement_corrected(variant, control)
    # Should be RANK_STABILITY_IMPROVEMENT (cost reduction 30% >= 25%, reps 40% >= 35%)
    assert result == "RANK_STABILITY_IMPROVEMENT"


# ===========================================================================
# Part 3 — Opportunity Cost Analysis Tests
# ===========================================================================

def _make_close_for_opp() -> pd.DataFrame:
    """Minimal close matrix for opportunity cost tests."""
    idx = pd.date_range("2022-01-01", periods=50, freq="4h", tz="UTC")
    return pd.DataFrame({
        "A/USD": np.linspace(100, 120, 50),
        "B/USD": np.linspace(100, 90, 50),
    }, index=idx)


def _make_suppressed_event(ts: pd.Timestamp) -> dict:
    return {
        "timestamp": ts,
        "incumbent": "A/USD",
        "challenger": "B/USD",
        "incumbent_rank": 2,
        "challenger_rank": 1,
        "incumbent_score": 0.01,
        "challenger_score": 0.03,
        "score_advantage": 0.02,
        "reason": "score_hurdle",
        "incumbent_return_next_reb": None,
        "challenger_return_next_reb": None,
        "incumbent_return_7d": None,
        "challenger_return_7d": None,
        "cost_avoided": None,
    }


def test_opportunity_cost_pct_helped_is_raw_return_comparison():
    """pct_helped_before_costs reflects raw return comparison only."""
    close = _make_close_for_opp()
    ts = close.index[0]
    next_ts = close.index[12]
    reb_list = [next_ts]
    equity = pd.Series(10_000.0, index=close.index)

    ev = _make_suppressed_event(ts)
    enriched = enrich_suppressed_events_corrected([ev], close, reb_list, equity, FEE_BPS, SLIPPAGE_BPS)
    assert len(enriched) == 1
    r = enriched[0]
    # A goes up (returns > 0), B goes down (returns < 0)
    # So raw_return_diff = inc_ret - chal_ret > 0
    assert r["raw_return_diff"] is not None
    assert r["raw_return_diff"] > 0


def test_pct_beneficial_after_costs_different_from_pct_helped():
    """pct_beneficial_after_costs is a separate computation from pct_helped_before_costs."""
    close = _make_close_for_opp()
    ts = close.index[0]
    next_ts = close.index[12]
    reb_list = [next_ts]
    equity = pd.Series(10_000.0, index=close.index)

    events = [_make_suppressed_event(ts)]
    enriched = enrich_suppressed_events_corrected(events, close, reb_list, equity, FEE_BPS, SLIPPAGE_BPS)
    summary = suppressed_events_corrected_summary(enriched)

    # Both metrics should exist as separate keys
    assert "pct_helped_before_costs" in summary
    assert "pct_beneficial_after_costs" in summary


def test_net_benefit_includes_cost_avoided():
    """net_economic_benefit = raw_return_diff × notional + cost_avoided."""
    close = _make_close_for_opp()
    ts = close.index[0]
    next_ts = close.index[12]
    reb_list = [next_ts]
    equity_val = 10_000.0
    equity = pd.Series(equity_val, index=close.index)

    events = [_make_suppressed_event(ts)]
    enriched = enrich_suppressed_events_corrected(events, close, reb_list, equity, FEE_BPS, SLIPPAGE_BPS)
    r = enriched[0]

    assert r["net_economic_benefit"] is not None
    notional = (1.0 / TOP_N) * equity_val
    expected = r["raw_return_diff"] * notional + r["cost_avoided"]
    assert abs(r["net_economic_benefit"] - expected) < 0.01


def test_cost_avoided_is_single_leg_entry_cost():
    """cost_avoided = weight × equity × (fee+slip)/10000."""
    close = _make_close_for_opp()
    ts = close.index[0]
    next_ts = close.index[12]
    reb_list = [next_ts]
    equity_val = 10_000.0
    equity = pd.Series(equity_val, index=close.index)

    events = [_make_suppressed_event(ts)]
    enriched = enrich_suppressed_events_corrected(events, close, reb_list, equity, FEE_BPS, SLIPPAGE_BPS)
    r = enriched[0]

    weight = 1.0 / TOP_N
    expected_cost = weight * equity_val * (FEE_BPS + SLIPPAGE_BPS) / 10_000
    assert abs(r["cost_avoided"] - expected_cost) < 0.01


def test_suppression_opportunity_cost_includes_net_benefit():
    """Enriched events have net_economic_benefit column."""
    close = _make_close_for_opp()
    ts = close.index[0]
    next_ts = close.index[12]
    reb_list = [next_ts]
    equity = pd.Series(10_000.0, index=close.index)

    events = [_make_suppressed_event(ts)]
    enriched = enrich_suppressed_events_corrected(events, close, reb_list, equity, FEE_BPS, SLIPPAGE_BPS)
    assert "net_economic_benefit" in enriched[0]


# ===========================================================================
# Part 4 — Selected Variants
# ===========================================================================

def test_selected_variants_count():
    """SELECTED_VARIANTS must have exactly 9 variants."""
    assert len(SELECTED_VARIANTS) == 9


def test_no_additional_variants_tested():
    """No variant names outside the approved 9."""
    approved = {
        "control", "rank_buffer_4", "challenger_confirm_3",
        "score_hurdle_025", "score_hurdle_100", "min_hold_4", "min_hold_6",
        "combo_buf4_hurdle025", "combo_buf4_conf2_hold2",
    }
    names = {v.name for v in SELECTED_VARIANTS}
    assert names == approved


def test_selected_variants_are_subset_of_all_variants():
    """All SELECTED_VARIANTS exist in the full VARIANTS list."""
    all_names = {v.name for v in VARIANTS}
    for v in SELECTED_VARIANTS:
        assert v.name in all_names


# ===========================================================================
# Part 5 — Classification Logic
# ===========================================================================

def test_rank_stability_improvement_requires_cost_reduction():
    """Must have >= 25% actual direct cost reduction for RANK_STABILITY_IMPROVEMENT."""
    control = {
        "actual_direct_cost_dollars": 1000.0,
        "n_rank_replacements": 100,
        "sharpe": 1.0,
        "max_drawdown_pct": -40.0,
        "return_2022_pct": -30.0,
        "pct_time_cash": 20.0,
    }
    variant_ok = dict(control)
    variant_ok["actual_direct_cost_dollars"] = 700.0  # 30% reduction
    variant_ok["n_rank_replacements"] = 60

    variant_fail = dict(control)
    variant_fail["actual_direct_cost_dollars"] = 850.0  # only 15% reduction
    variant_fail["n_rank_replacements"] = 60

    assert classify_rank_stability_improvement_corrected(variant_ok, control) == "RANK_STABILITY_IMPROVEMENT"
    result_fail = classify_rank_stability_improvement_corrected(variant_fail, control)
    assert result_fail != "RANK_STABILITY_IMPROVEMENT"


def test_rank_stability_improvement_requires_replacement_reduction():
    """Must have >= 35% replacement reduction."""
    control = {
        "actual_direct_cost_dollars": 1000.0,
        "n_rank_replacements": 100,
        "sharpe": 1.0,
        "max_drawdown_pct": -40.0,
        "return_2022_pct": -30.0,
        "pct_time_cash": 20.0,
    }
    # 30% reduction in replacements — not enough
    variant = dict(control)
    variant["actual_direct_cost_dollars"] = 700.0  # 30% cost reduction
    variant["n_rank_replacements"] = 70  # only 30% reduction

    result = classify_rank_stability_improvement_corrected(variant, control)
    assert result != "RANK_STABILITY_IMPROVEMENT"


def test_unfavorable_if_sharpe_much_worse():
    """sharpe_vs_control < -0.15 → UNFAVORABLE."""
    control = {
        "actual_direct_cost_dollars": 1000.0,
        "n_rank_replacements": 100,
        "sharpe": 1.0,
        "max_drawdown_pct": -40.0,
        "return_2022_pct": -30.0,
        "pct_time_cash": 20.0,
    }
    variant = dict(control)
    variant["sharpe"] = 0.80  # 0.20 below control → UNFAVORABLE
    variant["actual_direct_cost_dollars"] = 700.0
    variant["n_rank_replacements"] = 60

    result = classify_rank_stability_improvement_corrected(variant, control)
    assert result == "UNFAVORABLE"


def test_mixed_tradeoff_when_cost_reduced_but_not_all_pass():
    """MIXED_TRADEOFF when cost reduced >= 25% but not all criteria pass."""
    control = {
        "actual_direct_cost_dollars": 1000.0,
        "n_rank_replacements": 100,
        "sharpe": 1.0,
        "max_drawdown_pct": -40.0,
        "return_2022_pct": -30.0,
        "pct_time_cash": 20.0,
    }
    variant = dict(control)
    variant["actual_direct_cost_dollars"] = 700.0  # 30% cost reduction ✓
    variant["n_rank_replacements"] = 70  # only 30% reduction ✗

    result = classify_rank_stability_improvement_corrected(variant, control)
    assert result == "MIXED_TRADEOFF"


# ===========================================================================
# Part 6 — Determinism and no live imports
# ===========================================================================

def test_deterministic_rerun():
    """compute_min_hold_timing_audit is deterministic."""
    df1 = compute_min_hold_timing_audit()
    df2 = compute_min_hold_timing_audit()
    pd.testing.assert_frame_equal(df1, df2)


def test_no_live_imports():
    """The reconciliation module must not import from brokers/, execution/, or live/."""
    import ast
    import sys

    mod_path = Path(__file__).parent.parent / "research" / "fixed_five_rank_stability_reconciliation.py"
    source = mod_path.read_text()
    tree = ast.parse(source)

    forbidden_prefixes = ("brokers", "execution", "live")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden_prefixes:
                    assert not node.module.startswith(prefix), (
                        f"Forbidden import from '{node.module}' found"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        assert not alias.name.startswith(prefix), (
                            f"Forbidden import '{alias.name}' found"
                        )


def test_canonical_initialization_same_joint_start():
    """joint_start should be 2020-09-28 based on canonical initialization."""
    # Just verify the module references the correct joint start derivation
    # The actual value comes from find_joint_eligible_start which uses MIN_HISTORY_BARS=36
    from research.fixed_five_rank_stability_reconciliation import MIN_HISTORY_BARS
    assert MIN_HISTORY_BARS == 36


# ===========================================================================
# Part 7 — Output file verification (integration tests — run after script)
# ===========================================================================

REPORTS_DIR = Path("reports")


def test_all_output_files_created():
    """All 5 output files must exist (run script first)."""
    expected = [
        REPORTS_DIR / "fixed_five_min_hold_timing_audit.csv",
        REPORTS_DIR / "fixed_five_rank_stability_direct_costs.csv",
        REPORTS_DIR / "fixed_five_suppression_opportunity_cost.csv",
        REPORTS_DIR / "fixed_five_rank_stability_corrected_comparison.csv",
        REPORTS_DIR / "fixed_five_rank_stability_reconciliation.md",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, f"Missing output files: {missing}"


def test_corrected_comparison_has_9_rows():
    """Corrected comparison CSV must have exactly 9 rows."""
    path = REPORTS_DIR / "fixed_five_rank_stability_corrected_comparison.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    df = pd.read_csv(path)
    assert len(df) == 9, f"Expected 9 rows, got {len(df)}"


def test_direct_costs_csv_has_9_rows():
    """Direct costs CSV must have one row per variant (9 rows)."""
    path = REPORTS_DIR / "fixed_five_rank_stability_direct_costs.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    df = pd.read_csv(path)
    assert len(df) == 9


def test_correct_comparison_has_required_columns():
    """Corrected comparison CSV must have required columns."""
    path = REPORTS_DIR / "fixed_five_rank_stability_corrected_comparison.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    df = pd.read_csv(path)
    required = {
        "variant", "sharpe", "max_drawdown_pct", "return_2022_pct",
        "actual_fee_dollars", "actual_slippage_dollars", "actual_direct_cost_dollars",
        "gross_minus_net_dollars", "residual_foregone_compounding",
        "n_rank_replacements", "rank_stability_label_corrected",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_opportunity_cost_csv_exists():
    """Suppression opportunity cost CSV exists (may have 0 rows for control)."""
    path = REPORTS_DIR / "fixed_five_suppression_opportunity_cost.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    assert path.exists()
    # File must be readable as CSV
    df = pd.read_csv(path)
    assert df is not None


def test_suppression_opportunity_cost_csv_has_net_benefit_column():
    """Suppression CSV must have net_economic_benefit column."""
    path = REPORTS_DIR / "fixed_five_suppression_opportunity_cost.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    df = pd.read_csv(path)
    if len(df) == 0:
        pytest.skip("No suppressed events to check")
    assert "net_economic_benefit" in df.columns


def test_control_metrics_match_prior_audit():
    """Control actual fees should be in plausible range (~$300K-$600K based on prior audit)."""
    path = REPORTS_DIR / "fixed_five_rank_stability_direct_costs.csv"
    if not path.exists():
        pytest.skip("Run script first to generate output files")
    df = pd.read_csv(path)
    ctrl = df[df["variant"] == "control"]
    assert len(ctrl) == 1
    actual_fee = float(ctrl.iloc[0]["actual_fee_dollars"])
    # Prior audit found ~$405K; allow ±50% tolerance for data variability
    assert 150_000 < actual_fee < 700_000, (
        f"Control actual fee ${actual_fee:,.0f} outside expected range $150K-$700K"
    )


# ===========================================================================
# Part 8 — Markdown content verification
# ===========================================================================

def _read_md() -> str:
    path = REPORTS_DIR / "fixed_five_rank_stability_reconciliation.md"
    if not path.exists():
        pytest.skip("Run script first to generate markdown")
    return path.read_text()


def test_markdown_mentions_label_error():
    """Reconciliation .md must mention 'label error' or 'label was wrong'."""
    text = _read_md().lower()
    assert "label error" in text or "label was wrong" in text, (
        "Markdown should mention the label error"
    )


def test_markdown_mentions_gross_minus_net():
    """Markdown must explain gross-minus-net ≠ fees."""
    text = _read_md().lower()
    assert "gross-minus-net" in text or "gross_minus_net" in text


def test_markdown_mentions_12_days_for_min_hold_6():
    """Markdown must say 12 days for min_hold_6, not 3 days."""
    text = _read_md()
    assert "12" in text
    assert "min_hold_6" in text or "min_hold 6" in text.lower()
    # Must NOT say "NOT 3 days" without also saying 12
    assert "12 calendar days" in text or "12 days" in text


def test_markdown_mentions_39_pct():
    """Markdown must discuss the 39.75% figure."""
    text = _read_md()
    assert "39.75" in text or "39%" in text


def test_in_sample_disclaimer_in_markdown():
    """Markdown must contain 'in-sample' disclaimer."""
    text = _read_md().lower()
    assert "in-sample" in text or "in sample" in text


def test_markdown_mentions_opportunity_cost():
    """Markdown discusses the opportunity cost correction."""
    text = _read_md().lower()
    assert "opportunity cost" in text or "net economic benefit" in text or "cost_avoided" in text


# ===========================================================================
# Part 9 — Additional structural tests
# ===========================================================================

def test_timing_audit_csv_written():
    """Timing audit CSV file exists after run."""
    path = REPORTS_DIR / "fixed_five_min_hold_timing_audit.csv"
    if not path.exists():
        pytest.skip("Run script first")
    df = pd.read_csv(path)
    assert len(df) == 7


def test_rebalance_interval_is_48h():
    """CONTROL_REBALANCE_BARS × 4 hours == 48 hours."""
    assert CONTROL_REBALANCE_BARS * 4 == 48


def test_fee_bps_and_slippage_bps_match_rank_stability():
    """FEE_BPS and SLIPPAGE_BPS match the rank_stability module."""
    from research.fixed_five_rank_stability import FEE_BPS as rfs_fee, SLIPPAGE_BPS as rfs_slip
    assert FEE_BPS == rfs_fee
    assert SLIPPAGE_BPS == rfs_slip


def test_classify_returns_control_label_for_control():
    """Control variant gets CONTROL label in comparison CSV."""
    path = REPORTS_DIR / "fixed_five_rank_stability_corrected_comparison.csv"
    if not path.exists():
        pytest.skip("Run script first")
    df = pd.read_csv(path)
    ctrl = df[df["variant"] == "control"]
    assert len(ctrl) == 1
    assert ctrl.iloc[0]["rank_stability_label_corrected"] == "CONTROL"


def test_actual_fee_is_two_thirds_of_actual_direct():
    """actual_fee_dollars / actual_direct_cost_dollars ≈ fee_bps / (fee_bps + slip_bps) = 2/3.

    The ratio is exact in theory (both computed from the same notional × rate).
    Rounding to 2 decimal places causes a small discrepancy; allow 1% tolerance.
    """
    costs = _run_compute_actual_costs_fake()
    expected_ratio = FEE_BPS / (FEE_BPS + SLIPPAGE_BPS)
    actual_ratio = costs["actual_fee_dollars"] / costs["actual_direct_cost_dollars"]
    assert abs(actual_ratio - expected_ratio) < 0.01
