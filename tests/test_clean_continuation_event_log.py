"""Tests for research/memecoin_catcher/clean_continuation_event_log.py."""

from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.clean_continuation_event_log import (
    build_event_log,
    build_holding_period_summary,
    filter_clean_continuation,
    AVAILABLE_HORIZONS,
    MISSING_HORIZONS,
    DIAGNOSTIC_FLAG_COLS,
    EVENT_LOG_PATH,
)
from research.memecoin_catcher.analyze_signal_traits import (
    add_diagnostic_flags,
    load_long_explosion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**kwargs) -> dict:
    defaults = {
        "ohlc_signal_type": "LONG_EXPLOSION",
        "wsname": "TEST/USD",
        "pair_id": "TESTUSD",
        "snapshot_ts_utc": "2026-05-29T04:00:00Z",
        "last_price": "1.00",
        "primary_ohlc_score": "10.0",
        "spread_pct": "0.3",
        "today_return_pct": "5.0",
        "ret_15m_pct": "1.0",
        "ret_1h_pct": "2.0",
        "ret_4h_pct": "3.0",
        "ret_24h_pct": "10.0",
        "volume_ratio_1h": "2.0",
        "volume_ratio_4h": "3.0",
        "future_ret_15m_pct": "0.5",
        "future_ret_1h_pct": "2.0",
        "future_ret_4h_pct": "5.0",
        "future_ret_24h_pct": "8.0",
        "max_favorable_4h_pct": "7.0",
        "max_adverse_4h_pct": "-2.0",
        "max_favorable_24h_pct": "12.0",
        "max_adverse_24h_pct": "-3.0",
        "outcome_15m": "FLAT",
        "outcome_1h": "SUCCESS",
        "outcome_4h": "SUCCESS",
        "outcome_24h": "SUCCESS",
    }
    defaults.update(kwargs)
    return defaults


def _rows_to_loaded_df(rows: list[dict], tmp_path: Path) -> pd.DataFrame:
    p = tmp_path / "outcomes.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_long_explosion(p)


# ---------------------------------------------------------------------------
# filter_clean_continuation
# ---------------------------------------------------------------------------


def test_filter_keeps_only_clean_continuation(tmp_path):
    rows = [
        _make_row(ret_15m_pct="1.0", ret_1h_pct="2.0", ret_4h_pct="3.0"),   # clean
        _make_row(ret_15m_pct="1.0", ret_1h_pct="-1.0", ret_4h_pct="3.0"),  # not clean (negative 1h)
        _make_row(ret_15m_pct="-1.0", ret_1h_pct="2.0", ret_4h_pct="3.0"),  # not clean (negative 15m)
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    filtered = filter_clean_continuation(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["ret_1h_pct"] > 0
    assert filtered.iloc[0]["ret_15m_pct"] > 0


def test_filter_excludes_wide_spread(tmp_path):
    rows = [
        _make_row(spread_pct="0.3"),   # tight — kept
        _make_row(spread_pct="1.5"),   # wide — excluded even if clean continuation
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    filtered = filter_clean_continuation(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["spread_pct"] <= 1.0


def test_filter_excludes_terminal_spike(tmp_path):
    rows = [
        _make_row(spread_pct="0.3", ret_1h_pct="2.0", ret_4h_pct="3.0"),          # clean, no spike
        _make_row(spread_pct="1.5", ret_1h_pct="-2.0", ret_4h_pct="5.0",           # terminal spike
                  ret_15m_pct="1.0"),
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    filtered = filter_clean_continuation(df)
    # Second row is clean_continuation=False (ret_1h negative) AND terminal spike, so excluded regardless
    for _, row in filtered.iterrows():
        assert not row.get("danger_terminal_spike", False)


def test_filter_empty_df_returns_empty(tmp_path):
    rows = [
        _make_row(ret_15m_pct="-1.0", ret_1h_pct="-2.0", ret_4h_pct="-3.0"),  # nothing clean
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    filtered = filter_clean_continuation(df)
    assert len(filtered) == 0


# ---------------------------------------------------------------------------
# build_event_log
# ---------------------------------------------------------------------------


def test_event_log_has_required_columns(tmp_path):
    rows = [_make_row()]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)

    # Available horizon return columns
    for label, _, _ in AVAILABLE_HORIZONS:
        assert f"future_ret_{label}_pct" in event_log.columns

    # Missing horizon placeholder columns
    for label in MISSING_HORIZONS:
        assert f"future_ret_{label}_pct" in event_log.columns

    # Diagnostic flags
    for flag in DIAGNOSTIC_FLAG_COLS:
        assert flag in event_log.columns

    # Core identity columns
    for col in ["symbol", "snapshot_ts_utc", "entry_price"]:
        assert col in event_log.columns


def test_missing_horizon_columns_are_nan(tmp_path):
    rows = [_make_row()]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)

    for label in MISSING_HORIZONS:
        col = f"future_ret_{label}_pct"
        assert event_log[col].isna().all(), f"{col} should be all-NaN (not yet collected)"


def test_available_horizons_populated(tmp_path):
    rows = [_make_row(future_ret_4h_pct="5.5")]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    assert abs(event_log.iloc[0]["future_ret_4h_pct"] - 5.5) < 1e-6


def test_mae_mfe_columns_present(tmp_path):
    rows = [_make_row()]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    for col in ["max_favorable_4h_pct", "max_adverse_4h_pct",
                "max_favorable_24h_pct", "max_adverse_24h_pct"]:
        assert col in event_log.columns


# ---------------------------------------------------------------------------
# build_holding_period_summary
# ---------------------------------------------------------------------------


def test_holding_period_summary_has_all_horizons(tmp_path):
    rows = [_make_row()]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    summary = build_holding_period_summary(event_log)

    horizons_in_summary = set(summary["horizon"].tolist())
    for label, _, _ in AVAILABLE_HORIZONS:
        assert label in horizons_in_summary
    for label in MISSING_HORIZONS:
        assert label in horizons_in_summary


def test_holding_period_missing_horizons_not_available(tmp_path):
    rows = [_make_row()]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    summary = build_holding_period_summary(event_log)

    for label in MISSING_HORIZONS:
        row = summary[summary["horizon"] == label].iloc[0]
        assert row["data_available"] == False
        assert row["n_events"] == 0


def test_holding_period_nan_returns_excluded_from_stats(tmp_path):
    rows = [
        _make_row(future_ret_4h_pct="10.0", outcome_4h="SUCCESS"),
        _make_row(future_ret_4h_pct="",     outcome_4h=""),    # NaN — excluded
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    summary = build_holding_period_summary(event_log)

    row_4h = summary[summary["horizon"] == "4h"].iloc[0]
    assert row_4h["n_events"] == 1  # only 1 non-NaN row
    assert abs(row_4h["avg_return_pct"] - 10.0) < 1e-4


def test_holding_period_all_nan_returns_zero_events(tmp_path):
    rows = [
        _make_row(future_ret_4h_pct="", outcome_4h=""),
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    summary = build_holding_period_summary(event_log)

    row_4h = summary[summary["horizon"] == "4h"].iloc[0]
    assert row_4h["n_events"] == 0
    assert math.isnan(row_4h["avg_return_pct"])


def test_success_rate_computed_correctly(tmp_path):
    rows = [
        _make_row(future_ret_4h_pct="5.0", outcome_4h="SUCCESS"),
        _make_row(future_ret_4h_pct="1.0", outcome_4h="FLAT"),
        _make_row(future_ret_4h_pct="-3.0", outcome_4h="FAILURE"),
        _make_row(future_ret_4h_pct="2.0", outcome_4h="SUCCESS"),
    ]
    df = _rows_to_loaded_df(rows, tmp_path)
    flagged = add_diagnostic_flags(df)
    event_log = build_event_log(flagged)
    summary = build_holding_period_summary(event_log)

    row_4h = summary[summary["horizon"] == "4h"].iloc[0]
    assert row_4h["n_events"] == 4
    assert abs(row_4h["success_rate"] - 0.5) < 1e-4   # 2/4
    assert abs(row_4h["failure_rate"] - 0.25) < 1e-4  # 1/4


def test_event_log_output_path_uses_reports_dir():
    assert str(EVENT_LOG_PATH).startswith("reports/")


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------


def test_no_broker_imports():
    import ast
    src = Path("research/memecoin_catcher/clean_continuation_event_log.py").read_text()
    tree = ast.parse(src)
    import_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_names.append(node.module)
    forbidden = ["brokers", "live_trading", "place_order", "ccxt"]
    for token in forbidden:
        matches = [n for n in import_names if token in n]
        assert not matches, f"Forbidden import: {token!r} in {matches}"
