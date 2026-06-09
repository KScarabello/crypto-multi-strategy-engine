"""Tests for research/memecoin_catcher/analyze_signal_traits.py."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.analyze_signal_traits import (
    add_diagnostic_flags,
    apply_rule,
    build_flag_summary,
    build_rule_experiment_summary,
    load_long_explosion,
    write_rule_summary,
    write_trait_summary,
    TRAIT_SUMMARY_PATH,
    RULE_SUMMARY_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(**kwargs) -> dict:
    defaults = {
        "ohlc_signal_type": "LONG_EXPLOSION",
        "primary_ohlc_score": "10.0",
        "spread_pct": "0.5",
        "today_return_pct": "5.0",
        "ret_15m_pct": "1.0",
        "ret_1h_pct": "2.0",
        "ret_4h_pct": "3.0",
        "ret_24h_pct": "10.0",
        "volume_ratio_1h": "2.0",
        "volume_ratio_4h": "3.0",
        "future_ret_4h_pct": "5.0",
        "future_ret_24h_pct": "8.0",
        "max_favorable_4h_pct": "7.0",
        "max_adverse_4h_pct": "-2.0",
        "max_favorable_24h_pct": "12.0",
        "max_adverse_24h_pct": "-3.0",
        "outcome_15m": "SUCCESS",
        "outcome_1h": "SUCCESS",
        "outcome_4h": "SUCCESS",
        "outcome_24h": "SUCCESS",
    }
    defaults.update(kwargs)
    return defaults


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a CSV-like dataframe from rows of string values."""
    import io, csv, tempfile
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return pd.read_csv(buf, dtype=str)


def _load_from_rows(rows: list[dict], tmp_path: Path) -> pd.DataFrame:
    """Write rows to a temp CSV and load via load_long_explosion."""
    p = tmp_path / "outcomes.csv"
    df = _make_df(rows)
    df.to_csv(p, index=False)
    from research.memecoin_catcher.analyze_signal_traits import load_long_explosion
    return load_long_explosion(p)


# ---------------------------------------------------------------------------
# Diagnostic flag tests
# ---------------------------------------------------------------------------


def test_is_clean_continuation_true(tmp_path):
    rows = [_make_row(ret_15m_pct="1.0", ret_1h_pct="2.0", ret_4h_pct="3.0")]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["is_clean_continuation"] == True


def test_is_clean_continuation_false_when_1h_negative(tmp_path):
    rows = [_make_row(ret_15m_pct="1.0", ret_1h_pct="-1.0", ret_4h_pct="3.0")]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["is_clean_continuation"] == False


def test_danger_terminal_spike_true(tmp_path):
    rows = [_make_row(spread_pct="1.5", ret_1h_pct="-2.0", ret_4h_pct="5.0")]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["danger_terminal_spike"] == True


def test_danger_terminal_spike_false_tight_spread(tmp_path):
    rows = [_make_row(spread_pct="0.3", ret_1h_pct="-2.0", ret_4h_pct="5.0")]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["danger_terminal_spike"] == False


def test_is_overextended_24h(tmp_path):
    rows = [
        _make_row(ret_24h_pct="30.0"),
        _make_row(ret_24h_pct="10.0"),
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["is_overextended_24h"] == True
    assert df.iloc[1]["is_overextended_24h"] == False


def test_is_volume_climax(tmp_path):
    rows = [
        _make_row(volume_ratio_4h="15.0"),
        _make_row(volume_ratio_4h="5.0"),
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["is_volume_climax"] == True
    assert df.iloc[1]["is_volume_climax"] == False


def test_is_pullback_rebound_candidate(tmp_path):
    rows = [
        _make_row(ret_1h_pct="-1.0", ret_4h_pct="3.0", spread_pct="0.4"),
        _make_row(ret_1h_pct="-1.0", ret_4h_pct="3.0", spread_pct="0.8"),  # spread too wide
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    assert df.iloc[0]["is_pullback_rebound_candidate"] == True
    assert df.iloc[1]["is_pullback_rebound_candidate"] == False


# ---------------------------------------------------------------------------
# Rule experiment tests
# ---------------------------------------------------------------------------


def _flagged_df(tmp_path: Path, extra_rows: list[dict] | None = None) -> pd.DataFrame:
    rows = [
        _make_row(),  # clean row
        _make_row(spread_pct="1.5", ret_1h_pct="-2.0", ret_4h_pct="5.0"),  # terminal spike + wide spread
        _make_row(ret_24h_pct="30.0"),  # overextended
        _make_row(volume_ratio_4h="15.0", ret_1h_pct="-1.0"),  # volume climax rollover
    ]
    if extra_rows:
        rows.extend(extra_rows)
    df = _load_from_rows(rows, tmp_path)
    return add_diagnostic_flags(df)


def test_baseline_keeps_all_rows(tmp_path):
    df = _flagged_df(tmp_path)
    mask = apply_rule(df, "baseline_all_long_explosion")
    assert mask.all()


def test_avoid_wide_spread_removes_wide_rows(tmp_path):
    df = _flagged_df(tmp_path)
    mask = apply_rule(df, "avoid_wide_spread")
    kept = df[mask]
    assert all(kept["spread_pct"] <= 1.0)
    removed = df[~mask]
    assert all(removed["spread_pct"] > 1.0)


def test_avoid_terminal_spike_removes_expected_rows(tmp_path):
    df = _flagged_df(tmp_path)
    mask = apply_rule(df, "avoid_terminal_spike")
    kept = df[mask]
    assert not kept["danger_terminal_spike"].any()


def test_require_clean_continuation_keeps_expected_rows(tmp_path):
    rows = [
        _make_row(ret_15m_pct="1.0", ret_1h_pct="2.0", ret_4h_pct="3.0"),   # clean
        _make_row(ret_15m_pct="1.0", ret_1h_pct="-1.0", ret_4h_pct="3.0"),  # not clean
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    mask = apply_rule(df, "require_clean_continuation")
    assert mask.sum() == 1
    assert df[mask].iloc[0]["ret_1h_pct"] > 0


def test_rule_counts_kept_removed(tmp_path):
    df = _flagged_df(tmp_path)
    total = len(df)
    mask = apply_rule(df, "avoid_overextended_24h")
    n_kept = mask.sum()
    n_removed = total - n_kept
    assert n_kept + n_removed == total


def test_rule_experiment_summary_columns(tmp_path):
    df = _flagged_df(tmp_path)
    summary = build_rule_experiment_summary(df)
    expected_cols = [
        "rule", "rows_kept", "rows_removed",
        "avg_future_ret_4h_pct", "median_future_ret_4h_pct",
        "success_rate_4h", "failure_rate_4h", "avg_max_adverse_4h_pct",
        "avg_future_ret_24h_pct", "success_rate_24h", "failure_rate_24h",
    ]
    for col in expected_cols:
        assert col in summary.columns, f"Missing column: {col}"
    assert len(summary) == 8  # one row per rule


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_future_ret_ignored_in_rule_summary(tmp_path):
    """Rows with NaN future_ret_4h_pct must not inflate counts."""
    rows = [
        _make_row(future_ret_4h_pct="5.0", outcome_4h="SUCCESS"),
        _make_row(future_ret_4h_pct="", outcome_4h=""),   # NaN
        _make_row(future_ret_4h_pct="", outcome_4h=""),   # NaN
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    summary = build_rule_experiment_summary(df)
    baseline = summary[summary["rule"] == "baseline_all_long_explosion"].iloc[0]
    # Only 1 row has completed future_ret_4h_pct
    assert baseline["rows_kept"] == 3  # all 3 rows kept by baseline
    assert baseline["success_rate_4h"] == 1.0  # 1 success out of 1 completed


def test_nan_24h_does_not_crash(tmp_path):
    """All 24h future returns NaN: should not crash, 24h metrics should be NaN."""
    rows = [
        _make_row(future_ret_24h_pct="", outcome_24h=""),
        _make_row(future_ret_24h_pct="", outcome_24h=""),
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    summary = build_rule_experiment_summary(df)
    baseline = summary[summary["rule"] == "baseline_all_long_explosion"].iloc[0]
    assert math.isnan(baseline["avg_future_ret_24h_pct"])
    assert math.isnan(baseline["success_rate_24h"])


def test_flag_summary_nan_ignored(tmp_path):
    """build_flag_summary with NaN future_ret must not treat NaN as 0."""
    rows = [
        _make_row(spread_pct="1.5", future_ret_4h_pct="10.0", outcome_4h="SUCCESS"),
        _make_row(spread_pct="1.5", future_ret_4h_pct="", outcome_4h=""),  # NaN
    ]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    flag_df = build_flag_summary(df)
    wide = flag_df[flag_df["flag"] == "is_wide_spread"].iloc[0]
    assert wide["n_flagged"] == 2
    assert abs(wide["avg_future_ret_4h_pct"] - 10.0) < 1e-6  # NaN row excluded


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------


def test_feature_grouping_ignores_nan_outcomes(tmp_path):
    """Rows with NaN outcome_4h should not appear in feature summary groups."""
    rows = [
        _make_row(outcome_4h="SUCCESS", future_ret_4h_pct="5.0"),
        _make_row(outcome_4h="", future_ret_4h_pct="3.0"),
    ]
    df = _load_from_rows(rows, tmp_path)
    # Only 1 row should be grouped (the SUCCESS one)
    rows_with_outcome = df[df["outcome_4h"].notna() & (df["outcome_4h"] != "")]
    assert len(rows_with_outcome) == 1


# ---------------------------------------------------------------------------
# CSV output columns
# ---------------------------------------------------------------------------


def test_trait_summary_csv_columns(tmp_path):
    rows = [_make_row()]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    flag_df = build_flag_summary(df)
    out_path = tmp_path / "trait_summary.csv"
    write_trait_summary(flag_df, out_path)
    result = pd.read_csv(out_path)
    expected_cols = [
        "flag", "n_flagged",
        "avg_future_ret_4h_pct", "median_future_ret_4h_pct",
        "success_4h", "flat_4h", "failure_4h",
        "avg_future_ret_24h_pct", "success_24h", "flat_24h", "failure_24h",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_rule_summary_csv_columns(tmp_path):
    rows = [_make_row()]
    df = _load_from_rows(rows, tmp_path)
    df = add_diagnostic_flags(df)
    rule_df = build_rule_experiment_summary(df)
    out_path = tmp_path / "rule_summary.csv"
    write_rule_summary(rule_df, out_path)
    result = pd.read_csv(out_path)
    expected_cols = [
        "rule", "rows_kept", "rows_removed",
        "avg_future_ret_4h_pct", "median_future_ret_4h_pct",
        "success_rate_4h", "failure_rate_4h", "avg_max_adverse_4h_pct",
        "avg_future_ret_24h_pct", "success_rate_24h", "failure_rate_24h",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
