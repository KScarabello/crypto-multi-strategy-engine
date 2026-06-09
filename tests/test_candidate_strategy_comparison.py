"""Tests for research/memecoin_catcher/candidate_strategy_comparison.py."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.candidate_strategy_comparison import (
    STRATEGY_NAMES,
    SUBSET_NAMES,
    OUTLIER_SYMBOL,
    _is_clean_continuation_entry,
    _compute_exit_return,
    _strategy_mask,
    _subsets,
    build_strategy_summary,
    build_event_log,
    compute_stats,
    load_completed,
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
        "max_favorable_4h_pct": 5.0,
        "max_adverse_4h_pct": -2.0,
        "max_favorable_24h_pct": 7.0,
        "max_adverse_24h_pct": -3.0,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "danger_terminal_spike": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_df(*overrides: dict) -> pd.DataFrame:
    base = {
        "symbol": "TEST/USD",
        "snapshot_ts_utc": "2026-01-01T00:00:00Z",
        "entry_price": 1.0,
        "future_ret_4h_pct": 3.0,
        "future_ret_24h_pct": 5.0,
        "max_favorable_4h_pct": 5.0,
        "max_adverse_4h_pct": -2.0,
        "max_favorable_24h_pct": 7.0,
        "max_adverse_24h_pct": -3.0,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "danger_terminal_spike": False,
    }
    records = []
    for override in (overrides if overrides else [{}]):
        r = dict(base)
        r.update(override)
        records.append(r)
    return pd.DataFrame(records)


def _make_full_outcomes(tmp_path: Path, extra_rows: list[dict] | None = None) -> Path:
    """Write a minimal outcomes CSV for load_completed tests."""
    rows = [
        {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "wsname": "ALLO/USD",
            "pair_id": "ALLOUSD",
            "snapshot_ts_utc": "2026-01-01T00:00:00Z",
            "last_price": "1.0",
            "spread_pct": "0.3",
            "ret_15m_pct": "1.0",
            "ret_1h_pct": "2.0",
            "ret_4h_pct": "3.0",
            "ret_24h_pct": "10.0",
            "volume_ratio_1h": "2.0",
            "volume_ratio_4h": "3.0",
            "primary_ohlc_score": "10.0",
            "today_return_pct": "5.0",
            "future_ret_4h_pct": "44.0",
            "future_ret_24h_pct": "35.0",
            "max_favorable_4h_pct": "80.0",
            "max_adverse_4h_pct": "-3.0",
            "max_favorable_24h_pct": "80.0",
            "max_adverse_24h_pct": "-3.0",
            "outcome_4h": "SUCCESS",
            "outcome_24h": "SUCCESS",
            "outcome_15m": "SUCCESS",
            "outcome_1h": "SUCCESS",
        },
        {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "wsname": "OTHER/USD",
            "pair_id": "OTHERUSD",
            "snapshot_ts_utc": "2026-01-02T00:00:00Z",
            "last_price": "1.0",
            "spread_pct": "0.5",
            "ret_15m_pct": "1.0",
            "ret_1h_pct": "1.5",
            "ret_4h_pct": "2.0",
            "ret_24h_pct": "5.0",
            "volume_ratio_1h": "2.0",
            "volume_ratio_4h": "3.0",
            "primary_ohlc_score": "8.0",
            "today_return_pct": "3.0",
            "future_ret_4h_pct": "2.0",
            "future_ret_24h_pct": "-3.0",
            "max_favorable_4h_pct": "4.0",
            "max_adverse_4h_pct": "-1.0",
            "max_favorable_24h_pct": "4.0",
            "max_adverse_24h_pct": "-4.0",
            "outcome_4h": "FLAT",
            "outcome_24h": "FAILURE",
            "outcome_15m": "FLAT",
            "outcome_1h": "FLAT",
        },
        {
            "ohlc_signal_type": "SHORT_EXPLOSION",  # excluded by load_long_explosion
            "wsname": "SKIP/USD",
            "pair_id": "SKIPUSD",
            "snapshot_ts_utc": "2026-01-03T00:00:00Z",
            "last_price": "1.0",
            "spread_pct": "0.5",
            "ret_15m_pct": "1.0",
            "ret_1h_pct": "1.0",
            "ret_4h_pct": "1.0",
            "ret_24h_pct": "1.0",
            "volume_ratio_1h": "2.0",
            "volume_ratio_4h": "3.0",
            "primary_ohlc_score": "5.0",
            "today_return_pct": "2.0",
            "future_ret_4h_pct": "1.0",
            "future_ret_24h_pct": "1.0",
            "max_favorable_4h_pct": "2.0",
            "max_adverse_4h_pct": "-0.5",
            "max_favorable_24h_pct": "2.0",
            "max_adverse_24h_pct": "-0.5",
            "outcome_4h": "FLAT",
            "outcome_24h": "FLAT",
            "outcome_15m": "FLAT",
            "outcome_1h": "FLAT",
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    p = tmp_path / "outcomes.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _is_clean_continuation_entry
# ---------------------------------------------------------------------------


def test_cc_entry_true_when_all_flags_good():
    df = _make_df()
    mask = _is_clean_continuation_entry(df)
    assert mask.all()


def test_cc_entry_false_when_not_clean():
    df = _make_df({"is_clean_continuation": False})
    mask = _is_clean_continuation_entry(df)
    assert not mask.any()


def test_cc_entry_excludes_wide_spread():
    df = _make_df({"is_wide_spread": True})
    mask = _is_clean_continuation_entry(df)
    assert not mask.any()


def test_cc_entry_excludes_danger_terminal_spike():
    df = _make_df({"danger_terminal_spike": True})
    mask = _is_clean_continuation_entry(df)
    assert not mask.any()


def test_cc_entry_missing_optional_col_skipped():
    df = _make_df()
    df = df.drop(columns=["is_wide_spread", "danger_terminal_spike"])
    mask = _is_clean_continuation_entry(df)
    assert mask.all()


# ---------------------------------------------------------------------------
# _compute_exit_return
# ---------------------------------------------------------------------------


def test_baseline_fixed_4h():
    r = _row(future_ret_4h_pct=5.0)
    ret, trigger = _compute_exit_return(r, "baseline_long_explosion_fixed_4h")
    assert ret == pytest.approx(5.0)
    assert trigger == "4h_close"


def test_clean_continuation_fixed_4h():
    r = _row(future_ret_4h_pct=3.5)
    ret, trigger = _compute_exit_return(r, "clean_continuation_fixed_4h")
    assert ret == pytest.approx(3.5)


def test_tp10_triggers():
    r = _row(max_favorable_4h_pct=12.0, future_ret_4h_pct=5.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_else_4h")
    assert ret == pytest.approx(10.0)
    assert "take_profit" in trigger


def test_tp10_sl5_sl_only():
    r = _row(max_favorable_4h_pct=3.0, max_adverse_4h_pct=-6.0, future_ret_4h_pct=-4.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_sl5_else_4h")
    assert ret == pytest.approx(-5.0)
    assert "stop_loss" in trigger


def test_tp10_sl7_sl_only():
    r = _row(max_favorable_4h_pct=3.0, max_adverse_4h_pct=-9.0, future_ret_4h_pct=-8.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_sl7_else_4h")
    assert ret == pytest.approx(-7.0)


def test_tp10_sl10_sl_only():
    r = _row(max_favorable_4h_pct=3.0, max_adverse_4h_pct=-12.0, future_ret_4h_pct=-11.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_sl10_else_4h")
    assert ret == pytest.approx(-10.0)


def test_tp10_sl5_both_hit_conservative():
    r = _row(max_favorable_4h_pct=12.0, max_adverse_4h_pct=-6.0, future_ret_4h_pct=3.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_sl5_else_4h")
    assert ret == pytest.approx(-5.0)
    assert "conservative" in trigger


def test_tp10_sl7_neither_hit():
    r = _row(max_favorable_4h_pct=5.0, max_adverse_4h_pct=-4.0, future_ret_4h_pct=2.0)
    ret, trigger = _compute_exit_return(r, "clean_continuation_tp10_sl7_else_4h")
    assert ret == pytest.approx(2.0)
    assert trigger == "4h_close"


def test_unknown_strategy_raises():
    r = _row()
    with pytest.raises(ValueError, match="Unknown strategy"):
        _compute_exit_return(r, "nonexistent_strategy")


# ---------------------------------------------------------------------------
# _strategy_mask
# ---------------------------------------------------------------------------


def test_baseline_mask_keeps_all():
    df = _make_df({}, {"is_clean_continuation": False})
    mask = _strategy_mask(df, "baseline_long_explosion_fixed_4h")
    assert mask.all()


def test_cc_strategy_mask_filters():
    df = _make_df({"is_clean_continuation": True}, {"is_clean_continuation": False})
    mask = _strategy_mask(df, "clean_continuation_fixed_4h")
    assert mask.sum() == 1


# ---------------------------------------------------------------------------
# _subsets
# ---------------------------------------------------------------------------


def test_subsets_returns_three():
    df = _make_df({"symbol": "ALLO/USD", "future_ret_4h_pct": 50.0}, {"symbol": "X/USD", "future_ret_4h_pct": 2.0})
    df["_ret"] = df["future_ret_4h_pct"]
    subs = _subsets(df, "_ret")
    names = [s[0] for s in subs]
    assert "all" in names
    assert "excl_ALLO" in names
    assert "excl_best_trade" in names


def test_excl_allo_removes_outlier():
    df = _make_df({"symbol": "ALLO/USD", "future_ret_4h_pct": 50.0}, {"symbol": "X/USD", "future_ret_4h_pct": 2.0})
    df["_ret"] = df["future_ret_4h_pct"]
    subs = dict(_subsets(df, "_ret"))
    assert len(subs["all"]) == 2
    assert len(subs["excl_ALLO"]) == 1
    assert subs["excl_ALLO"].iloc[0]["symbol"] == "X/USD"


def test_excl_best_trade_removes_max():
    df = _make_df({"future_ret_4h_pct": 50.0}, {"future_ret_4h_pct": 2.0}, {"future_ret_4h_pct": 3.0})
    df["_ret"] = df["future_ret_4h_pct"]
    subs = dict(_subsets(df, "_ret"))
    assert len(subs["excl_best_trade"]) == 2
    assert 50.0 not in subs["excl_best_trade"]["_ret"].values


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_basic():
    s = pd.Series([10.0, 2.0, -4.0, 1.0])
    mae = pd.Series([-3.0, -1.0, -4.0, -0.5])
    mfe = pd.Series([12.0, 3.0, 1.0, 2.0])
    result = compute_stats(s, mae, mfe, "test", "all")
    assert result["n_events"] == 4
    assert result["avg_return_pct"] == pytest.approx(s.mean(), rel=1e-4)
    assert result["win_rate"] == pytest.approx(0.75)
    assert result["success_rate_ge_5pct"] == pytest.approx(0.25)
    assert result["failure_rate_le_neg3pct"] == pytest.approx(0.25)
    assert result["failure_rate_le_neg5pct"] == pytest.approx(0.0)
    assert result["best_trade_pct"] == pytest.approx(10.0)
    assert result["worst_trade_pct"] == pytest.approx(-4.0)
    assert result["avg_mae_4h_pct"] == pytest.approx(mae.mean(), rel=1e-4)
    assert result["avg_mfe_4h_pct"] == pytest.approx(mfe.mean(), rel=1e-4)


def test_compute_stats_excl_best():
    s = pd.Series([100.0, 2.0, 3.0, 1.0])
    result = compute_stats(s, pd.Series(dtype=float), pd.Series(dtype=float), "p", "all")
    excl = pd.Series([2.0, 3.0, 1.0])
    assert result["avg_return_excl_best_pct"] == pytest.approx(excl.mean(), rel=1e-4)
    assert result["median_return_excl_best_pct"] == pytest.approx(excl.median(), rel=1e-4)


def test_compute_stats_empty():
    s = pd.Series([], dtype=float)
    result = compute_stats(s, pd.Series(dtype=float), pd.Series(dtype=float), "p", "all")
    assert result["n_events"] == 0
    assert math.isnan(result["avg_return_pct"])


def test_compute_stats_strategy_and_subset_preserved():
    s = pd.Series([1.0])
    result = compute_stats(s, pd.Series(dtype=float), pd.Series(dtype=float), "my_strat", "excl_ALLO")
    assert result["strategy"] == "my_strat"
    assert result["subset"] == "excl_ALLO"


# ---------------------------------------------------------------------------
# build_strategy_summary
# ---------------------------------------------------------------------------


def test_build_strategy_summary_shape():
    df = _make_df()
    summary = build_strategy_summary(df)
    expected_rows = len(STRATEGY_NAMES) * len(SUBSET_NAMES)
    assert len(summary) == expected_rows


def test_build_strategy_summary_all_strategies_present():
    df = _make_df()
    summary = build_strategy_summary(df)
    assert set(summary["strategy"].unique()) == set(STRATEGY_NAMES)


def test_build_strategy_summary_all_subsets_present():
    df = _make_df()
    summary = build_strategy_summary(df)
    assert set(summary["subset"].unique()) == set(SUBSET_NAMES)


def test_build_strategy_summary_baseline_larger_than_cc():
    """Baseline should have >= events vs clean_continuation (it includes all LONG_EXPLOSION)."""
    df = _make_df(
        {"is_clean_continuation": True},
        {"is_clean_continuation": False},
        {"is_clean_continuation": True},
    )
    summary = build_strategy_summary(df)
    baseline_n = summary[
        (summary["strategy"] == "baseline_long_explosion_fixed_4h") & (summary["subset"] == "all")
    ]["n_events"].iloc[0]
    cc_n = summary[
        (summary["strategy"] == "clean_continuation_fixed_4h") & (summary["subset"] == "all")
    ]["n_events"].iloc[0]
    assert baseline_n >= cc_n


def test_build_strategy_summary_tp10_best_capped_at_10():
    """Take-profit at 10% should never exceed 10% as best trade."""
    df = _make_df({"max_favorable_4h_pct": 50.0, "future_ret_4h_pct": 30.0})
    summary = build_strategy_summary(df)
    tp10_row = summary[
        (summary["strategy"] == "clean_continuation_tp10_else_4h") & (summary["subset"] == "all")
    ].iloc[0]
    assert tp10_row["best_trade_pct"] == pytest.approx(10.0)


def test_build_strategy_summary_sl5_worst_bounded():
    """Stop-loss at -5% should never produce a worse trade than -5%."""
    df = _make_df({"max_adverse_4h_pct": -20.0, "future_ret_4h_pct": -15.0})
    summary = build_strategy_summary(df)
    sl5_row = summary[
        (summary["strategy"] == "clean_continuation_tp10_sl5_else_4h") & (summary["subset"] == "all")
    ].iloc[0]
    # SL is -5, TP is 10, MFE is 5 (default) so TP not hit, SL is hit → exit at -5
    assert sl5_row["worst_trade_pct"] >= -5.01  # ≥ -5% (with float tolerance)


# ---------------------------------------------------------------------------
# build_event_log
# ---------------------------------------------------------------------------


def test_build_event_log_row_count():
    df = _make_df({}, {})
    event_log = build_event_log(df)
    assert len(event_log) == 2


def test_build_event_log_has_in_strategy_columns():
    df = _make_df()
    event_log = build_event_log(df)
    for s in STRATEGY_NAMES:
        assert f"in_{s}" in event_log.columns


def test_build_event_log_has_ret_and_trigger_columns():
    df = _make_df()
    event_log = build_event_log(df)
    for s in STRATEGY_NAMES:
        assert f"ret_{s}" in event_log.columns
        assert f"trigger_{s}" in event_log.columns


def test_build_event_log_non_cc_event_has_nan_for_cc_strategies():
    """A row that fails the CC filter should have NaN ret for CC strategies."""
    df = _make_df({"is_clean_continuation": False})
    event_log = build_event_log(df)
    for s in STRATEGY_NAMES:
        if s != "baseline_long_explosion_fixed_4h":
            assert math.isnan(event_log.iloc[0][f"ret_{s}"])


def test_build_event_log_baseline_always_has_ret():
    """Baseline should always have a ret value (it includes all rows)."""
    df = _make_df({"is_clean_continuation": False})
    event_log = build_event_log(df)
    assert not math.isnan(event_log.iloc[0]["ret_baseline_long_explosion_fixed_4h"])


# ---------------------------------------------------------------------------
# load_completed
# ---------------------------------------------------------------------------


def test_load_completed_filters_non_long_explosion(tmp_path):
    p = _make_full_outcomes(tmp_path)
    df = load_completed(p)
    # SHORT_EXPLOSION row should be excluded
    assert "SKIP/USD" not in df["symbol"].values


def test_load_completed_drops_nan_4h_rows(tmp_path):
    extra = [
        {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "wsname": "NODATA/USD",
            "pair_id": "NODATAUSD",
            "snapshot_ts_utc": "2026-01-04T00:00:00Z",
            "last_price": "1.0",
            "spread_pct": "0.3",
            "ret_15m_pct": "1.0",
            "ret_1h_pct": "2.0",
            "ret_4h_pct": "3.0",
            "ret_24h_pct": "5.0",
            "volume_ratio_1h": "2.0",
            "volume_ratio_4h": "3.0",
            "primary_ohlc_score": "7.0",
            "today_return_pct": "2.0",
            "future_ret_4h_pct": "",   # missing
            "future_ret_24h_pct": "",
            "max_favorable_4h_pct": "",
            "max_adverse_4h_pct": "",
            "max_favorable_24h_pct": "",
            "max_adverse_24h_pct": "",
            "outcome_4h": "",
            "outcome_24h": "",
            "outcome_15m": "",
            "outcome_1h": "",
        }
    ]
    p = _make_full_outcomes(tmp_path, extra)
    df = load_completed(p)
    assert "NODATA/USD" not in df["symbol"].values


def test_load_completed_adds_symbol_column(tmp_path):
    p = _make_full_outcomes(tmp_path)
    df = load_completed(p)
    assert "symbol" in df.columns
    assert "ALLO/USD" in df["symbol"].values


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------


def test_no_broker_or_live_imports():
    import ast
    src = Path("research/memecoin_catcher/candidate_strategy_comparison.py").read_text()
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
