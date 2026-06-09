"""Tests for research/memecoin_catcher/exit_policy_experiment.py."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.exit_policy_experiment import (
    POLICY_NAMES,
    apply_all_policies,
    apply_entry_filters,
    build_event_level_output,
    build_policy_summary,
    require_completed_4h,
    summarise_policy,
    _fixed_4h,
    _fixed_24h,
    _take_profit_else_4h,
    _stop_loss_else_4h,
    _tp_sl_else_4h,
    _trailing_stop_else_4h,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs) -> pd.Series:
    defaults = {
        "symbol": "TEST/USD",
        "snapshot_ts_utc": "2026-01-01T00:00:00Z",
        "entry_price": 1.0,
        "future_ret_4h_pct": 3.0,
        "future_ret_24h_pct": 5.0,
        "max_favorable_4h_pct": 8.0,
        "max_adverse_4h_pct": -2.0,
        "max_favorable_24h_pct": 10.0,
        "max_adverse_24h_pct": -4.0,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "danger_terminal_spike": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _df(*rows: dict) -> pd.DataFrame:
    base = {
        "symbol": "TEST/USD",
        "snapshot_ts_utc": "2026-01-01T00:00:00Z",
        "entry_price": 1.0,
        "future_ret_4h_pct": 3.0,
        "future_ret_24h_pct": 5.0,
        "max_favorable_4h_pct": 8.0,
        "max_adverse_4h_pct": -2.0,
        "max_favorable_24h_pct": 10.0,
        "max_adverse_24h_pct": -4.0,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "danger_terminal_spike": False,
    }
    records = []
    for override in rows:
        r = dict(base)
        r.update(override)
        records.append(r)
    return pd.DataFrame(records if records else [base])


# ---------------------------------------------------------------------------
# _fixed_4h
# ---------------------------------------------------------------------------


def test_fixed_4h_returns_4h_close():
    r = _row(future_ret_4h_pct=7.5)
    ret, trigger = _fixed_4h(r)
    assert ret == pytest.approx(7.5)
    assert trigger == "4h_close"


def test_fixed_4h_negative():
    r = _row(future_ret_4h_pct=-4.0)
    ret, trigger = _fixed_4h(r)
    assert ret == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# _fixed_24h
# ---------------------------------------------------------------------------


def test_fixed_24h_returns_24h_close():
    r = _row(future_ret_24h_pct=12.0)
    ret, trigger = _fixed_24h(r)
    assert ret == pytest.approx(12.0)
    assert trigger == "24h_close"


def test_fixed_24h_missing_col():
    r = pd.Series({"future_ret_4h_pct": 3.0})
    ret, trigger = _fixed_24h(r)
    assert math.isnan(ret)


# ---------------------------------------------------------------------------
# _take_profit_else_4h
# ---------------------------------------------------------------------------


def test_tp_hits_returns_tp():
    r = _row(max_favorable_4h_pct=6.0, future_ret_4h_pct=2.0)
    ret, trigger = _take_profit_else_4h(r, 5.0)
    assert ret == pytest.approx(5.0)
    assert "take_profit" in trigger


def test_tp_not_hit_returns_4h_close():
    r = _row(max_favorable_4h_pct=3.0, future_ret_4h_pct=2.0)
    ret, trigger = _take_profit_else_4h(r, 5.0)
    assert ret == pytest.approx(2.0)
    assert trigger == "4h_close"


def test_tp_exactly_at_threshold():
    r = _row(max_favorable_4h_pct=5.0, future_ret_4h_pct=1.0)
    ret, trigger = _take_profit_else_4h(r, 5.0)
    assert ret == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _stop_loss_else_4h
# ---------------------------------------------------------------------------


def test_sl_hits_returns_sl():
    r = _row(max_adverse_4h_pct=-4.0, future_ret_4h_pct=-1.0)
    ret, trigger = _stop_loss_else_4h(r, -3.0)
    assert ret == pytest.approx(-3.0)
    assert "stop_loss" in trigger


def test_sl_not_hit_returns_4h_close():
    r = _row(max_adverse_4h_pct=-2.0, future_ret_4h_pct=1.5)
    ret, trigger = _stop_loss_else_4h(r, -3.0)
    assert ret == pytest.approx(1.5)
    assert trigger == "4h_close"


def test_sl_exactly_at_threshold():
    r = _row(max_adverse_4h_pct=-3.0, future_ret_4h_pct=-1.0)
    ret, trigger = _stop_loss_else_4h(r, -3.0)
    assert ret == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# _tp_sl_else_4h  (combined, conservative ordering)
# ---------------------------------------------------------------------------


def test_combined_only_tp_hit():
    r = _row(max_favorable_4h_pct=12.0, max_adverse_4h_pct=-1.0, future_ret_4h_pct=5.0)
    ret, trigger = _tp_sl_else_4h(r, 10.0, -5.0)
    assert ret == pytest.approx(10.0)
    assert "take_profit" in trigger


def test_combined_only_sl_hit():
    r = _row(max_favorable_4h_pct=2.0, max_adverse_4h_pct=-6.0, future_ret_4h_pct=-4.0)
    ret, trigger = _tp_sl_else_4h(r, 10.0, -5.0)
    assert ret == pytest.approx(-5.0)
    assert "stop_loss" in trigger


def test_combined_both_hit_conservative_sl_first():
    """Both TP and SL touched inside window → conservative: book stop-loss."""
    r = _row(max_favorable_4h_pct=12.0, max_adverse_4h_pct=-6.0, future_ret_4h_pct=3.0)
    ret, trigger = _tp_sl_else_4h(r, 10.0, -5.0)
    assert ret == pytest.approx(-5.0)
    assert "conservative" in trigger


def test_combined_neither_hit_returns_4h_close():
    r = _row(max_favorable_4h_pct=4.0, max_adverse_4h_pct=-2.0, future_ret_4h_pct=3.0)
    ret, trigger = _tp_sl_else_4h(r, 10.0, -5.0)
    assert ret == pytest.approx(3.0)
    assert trigger == "4h_close"


# ---------------------------------------------------------------------------
# _trailing_stop_else_4h
# ---------------------------------------------------------------------------


def test_trailing_activation_never_fired():
    r = _row(max_favorable_4h_pct=3.0, future_ret_4h_pct=2.0)
    ret, trigger = _trailing_stop_else_4h(r, 5.0)
    assert ret == pytest.approx(2.0)
    assert trigger == "4h_close"


def test_trailing_fired_close_below_trail():
    # MFE=15%, activation=5%, trail_level=10%. Close=2% < 10% → trail fires at 10%
    r = _row(max_favorable_4h_pct=15.0, future_ret_4h_pct=2.0)
    ret, trigger = _trailing_stop_else_4h(r, 5.0)
    assert ret == pytest.approx(10.0)
    assert trigger == "trailing_stop_approx"


def test_trailing_close_above_trail():
    # MFE=10%, activation=5%, trail_level=5%. Close=7% > 5% → exit at close
    r = _row(max_favorable_4h_pct=10.0, future_ret_4h_pct=7.0)
    ret, trigger = _trailing_stop_else_4h(r, 5.0)
    assert ret == pytest.approx(7.0)
    assert "4h_close" in trigger


def test_trailing_exactly_at_activation():
    # MFE exactly equals activation threshold → trail activates
    r = _row(max_favorable_4h_pct=5.0, future_ret_4h_pct=1.0)
    ret, trigger = _trailing_stop_else_4h(r, 5.0)
    # trail_level = 5 - 5 = 0.0; close=1.0 > 0.0 → exit at close
    assert ret == pytest.approx(1.0)


def test_trailing_up10_not_activated():
    r = _row(max_favorable_4h_pct=8.0, future_ret_4h_pct=5.0)
    ret, trigger = _trailing_stop_else_4h(r, 10.0)
    assert ret == pytest.approx(5.0)
    assert trigger == "4h_close"


def test_trailing_up10_activated_and_trail_fires():
    # MFE=20%, trail_level=10%. Close=3% < 10% → exit at 10%
    r = _row(max_favorable_4h_pct=20.0, future_ret_4h_pct=3.0)
    ret, trigger = _trailing_stop_else_4h(r, 10.0)
    assert ret == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# apply_entry_filters
# ---------------------------------------------------------------------------


def test_entry_filter_keeps_clean_continuation():
    df = _df(
        {"is_clean_continuation": True, "is_wide_spread": False},
        {"is_clean_continuation": False, "is_wide_spread": False},  # excluded
    )
    result = apply_entry_filters(df)
    assert len(result) == 1
    assert result.iloc[0]["is_clean_continuation"] == True


def test_entry_filter_excludes_wide_spread():
    df = _df(
        {"is_clean_continuation": True, "is_wide_spread": False},
        {"is_clean_continuation": True, "is_wide_spread": True},  # excluded
    )
    result = apply_entry_filters(df)
    assert len(result) == 1
    assert not result.iloc[0]["is_wide_spread"]


def test_entry_filter_excludes_danger_terminal_spike():
    df = _df(
        {"is_clean_continuation": True, "danger_terminal_spike": False},
        {"is_clean_continuation": True, "danger_terminal_spike": True},  # excluded
    )
    result = apply_entry_filters(df)
    assert len(result) == 1
    assert not result.iloc[0]["danger_terminal_spike"]


def test_entry_filter_missing_column_skipped(caplog):
    """Missing optional columns should be skipped, not cause an error."""
    df = _df({"is_clean_continuation": True})
    # long_explosion and is_terminal_spike columns are absent — should not crash
    df = df.drop(columns=["is_wide_spread", "danger_terminal_spike"], errors="ignore")
    result = apply_entry_filters(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# require_completed_4h
# ---------------------------------------------------------------------------


def test_require_completed_4h_drops_nan_rows():
    df = _df(
        {"future_ret_4h_pct": 3.0, "max_favorable_4h_pct": 5.0, "max_adverse_4h_pct": -2.0},
        {"future_ret_4h_pct": float("nan"), "max_favorable_4h_pct": float("nan"), "max_adverse_4h_pct": float("nan")},
    )
    result = require_completed_4h(df)
    assert len(result) == 1


def test_require_completed_4h_keeps_all_when_complete():
    df = _df(
        {"future_ret_4h_pct": 2.0, "max_favorable_4h_pct": 4.0, "max_adverse_4h_pct": -1.0},
        {"future_ret_4h_pct": 5.0, "max_favorable_4h_pct": 7.0, "max_adverse_4h_pct": -0.5},
    )
    result = require_completed_4h(df)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# apply_all_policies
# ---------------------------------------------------------------------------


def test_apply_all_policies_adds_all_columns():
    df = _df()
    result = apply_all_policies(df)
    for p in POLICY_NAMES:
        assert f"ret_{p}" in result.columns
        assert f"trigger_{p}" in result.columns


def test_apply_all_policies_correct_fixed_4h():
    df = _df({"future_ret_4h_pct": 7.0})
    result = apply_all_policies(df)
    assert result.iloc[0]["ret_fixed_4h_exit"] == pytest.approx(7.0)
    assert result.iloc[0]["trigger_fixed_4h_exit"] == "4h_close"


def test_apply_all_policies_does_not_modify_input():
    df = _df()
    cols_before = set(df.columns)
    apply_all_policies(df)
    assert set(df.columns) == cols_before


# ---------------------------------------------------------------------------
# summarise_policy
# ---------------------------------------------------------------------------


def test_summarise_policy_basic_stats():
    s = pd.Series([10.0, 2.0, -5.0, 1.0])
    result = summarise_policy("test_policy", s)
    assert result["n_events"] == 4
    assert result["avg_return_pct"] == pytest.approx(s.mean(), rel=1e-4)
    assert result["median_return_pct"] == pytest.approx(s.median(), rel=1e-4)
    assert result["best_trade_pct"] == pytest.approx(10.0)
    assert result["worst_trade_pct"] == pytest.approx(-5.0)


def test_summarise_policy_win_rate():
    s = pd.Series([1.0, 2.0, -1.0, -2.0])
    result = summarise_policy("p", s)
    assert result["win_rate"] == pytest.approx(0.5)


def test_summarise_policy_success_failure_rates():
    s = pd.Series([6.0, 5.0, 1.0, -4.0])
    result = summarise_policy("p", s)
    assert result["success_rate_ge_5pct"] == pytest.approx(0.5)  # 2/4
    assert result["failure_rate_le_neg3pct"] == pytest.approx(0.25)  # 1/4


def test_summarise_policy_excl_best():
    s = pd.Series([100.0, 2.0, 3.0, 1.0])
    result = summarise_policy("p", s)
    excl = pd.Series([2.0, 3.0, 1.0])
    assert result["avg_return_excl_best_pct"] == pytest.approx(excl.mean(), rel=1e-4)
    assert result["median_return_excl_best_pct"] == pytest.approx(excl.median(), rel=1e-4)


def test_summarise_policy_empty_returns_nan():
    s = pd.Series([], dtype=float)
    result = summarise_policy("p", s)
    assert result["n_events"] == 0
    assert math.isnan(result["avg_return_pct"])


def test_summarise_policy_with_mae_mfe():
    s = pd.Series([5.0, 3.0])
    mae = pd.Series([-2.0, -1.0])
    mfe = pd.Series([7.0, 5.0])
    result = summarise_policy("p", s, mae, mfe)
    assert result["avg_mae_pct"] == pytest.approx(-1.5)
    assert result["avg_mfe_pct"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# build_policy_summary
# ---------------------------------------------------------------------------


def test_build_policy_summary_has_all_policies():
    df = _df()
    df_with = apply_all_policies(df)
    summary = build_policy_summary(df_with)
    policies_in_summary = set(summary["policy"].tolist())
    for p in POLICY_NAMES:
        assert p in policies_in_summary


def test_build_policy_summary_columns():
    df = _df()
    df_with = apply_all_policies(df)
    summary = build_policy_summary(df_with)
    expected = [
        "policy", "n_events", "avg_return_pct", "median_return_pct",
        "win_rate", "success_rate_ge_5pct", "failure_rate_le_neg3pct",
        "worst_trade_pct", "best_trade_pct",
        "avg_mae_pct", "avg_mfe_pct",
        "avg_return_excl_best_pct", "median_return_excl_best_pct",
    ]
    for col in expected:
        assert col in summary.columns, f"Missing column: {col}"


def test_build_policy_summary_fixed_24h_uses_24h_mae_mfe():
    """fixed_24h_exit row should use 24h MAE/MFE, not 4h."""
    df = _df(
        {
            "max_adverse_4h_pct": -2.0,
            "max_adverse_24h_pct": -8.0,
            "max_favorable_4h_pct": 5.0,
            "max_favorable_24h_pct": 15.0,
        }
    )
    df_with = apply_all_policies(df)
    summary = build_policy_summary(df_with)
    row_24h = summary[summary["policy"] == "fixed_24h_exit"].iloc[0]
    row_4h = summary[summary["policy"] == "fixed_4h_exit"].iloc[0]
    # 24h policy should show 24h MAE (-8) not 4h MAE (-2)
    assert row_24h["avg_mae_pct"] == pytest.approx(-8.0)
    assert row_4h["avg_mae_pct"] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# build_event_level_output
# ---------------------------------------------------------------------------


def test_event_level_output_has_identity_and_flag_cols():
    df = _df()
    df_with = apply_all_policies(df)
    out = build_event_level_output(df_with)
    for col in ["symbol", "snapshot_ts_utc"]:
        assert col in out.columns
    for p in POLICY_NAMES:
        assert f"ret_{p}" in out.columns
        assert f"trigger_{p}" in out.columns


def test_event_level_output_row_count_preserved():
    df = _df({"future_ret_4h_pct": 1.0}, {"future_ret_4h_pct": 2.0})
    df_with = apply_all_policies(df)
    out = build_event_level_output(df_with)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------


def test_no_broker_or_live_imports():
    import ast
    src = Path("research/memecoin_catcher/exit_policy_experiment.py").read_text()
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    forbidden = ["brokers", "live_trading", "place_order", "ccxt", "execution"]
    for token in forbidden:
        matches = [n for n in names if token in n]
        assert not matches, f"Forbidden import found: {token!r} in {matches}"
