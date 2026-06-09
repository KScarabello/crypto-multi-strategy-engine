"""Tests for research/memecoin_catcher/save_signal_snapshot.py.

Unit tests only; no live network calls, no external file dependencies.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.save_signal_snapshot import (
    SNAPSHOT_COLUMNS,
    append_to_history,
    filter_active_signals,
    format_snapshot_filename,
    format_snapshot_ts,
    run_signal_snapshot,
    stamp_snapshot,
    write_snapshot_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXED_DT = datetime(2026, 5, 28, 21, 30, 0, tzinfo=timezone.utc)
_FIXED_TS = "2026-05-28T21:30:00Z"
_FIXED_TS_STR = "20260528_213000"


def _make_signal_df(
    n_long: int = 2,
    n_dump: int = 1,
    n_blank: int = 1,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(n_long):
        rows.append(
            {
                "pair_id": f"LONG{i}USD",
                "wsname": f"LONG{i}/USD",
                "base": f"LONG{i}",
                "quote": "USD",
                "scanner_label": "HOT_MOVER",
                "ohlc_signal_type": "LONG_EXPLOSION",
                "last_price": 1.0 + i,
                "spread_pct": 0.2,
                "today_return_pct": 10.0 + i,
                "ret_15m_pct": 0.5,
                "ret_1h_pct": 3.0 + i,
                "ret_4h_pct": 8.0,
                "ret_24h_pct": 20.0,
                "volume_ratio_1h": 3.0,
                "volume_ratio_4h": 2.5,
                "breakout_24h": False,
                "long_explosion_score": 12.0 + i,
                "dump_score": None,
                "reversal_watch_score": None,
                "primary_ohlc_score": 12.0 + i,
            }
        )
    for i in range(n_dump):
        rows.append(
            {
                "pair_id": f"DUMP{i}USD",
                "wsname": f"DUMP{i}/USD",
                "base": f"DUMP{i}",
                "quote": "USD",
                "scanner_label": "DUMPING",
                "ohlc_signal_type": "DUMPING",
                "last_price": 0.5,
                "spread_pct": 0.5,
                "today_return_pct": -10.0,
                "ret_15m_pct": -0.2,
                "ret_1h_pct": -5.0,
                "ret_4h_pct": -8.0,
                "ret_24h_pct": -20.0,
                "volume_ratio_1h": 8.0,
                "volume_ratio_4h": 3.0,
                "breakout_24h": False,
                "long_explosion_score": None,
                "dump_score": 18.0,
                "reversal_watch_score": None,
                "primary_ohlc_score": 18.0,
            }
        )
    for i in range(n_blank):
        rows.append(
            {
                "pair_id": f"BLANK{i}USD",
                "wsname": f"BLANK{i}/USD",
                "base": f"BLANK{i}",
                "quote": "USD",
                "scanner_label": "WATCH",
                "ohlc_signal_type": "",
                "last_price": 2.0,
                "spread_pct": 0.3,
                "today_return_pct": 1.0,
                "ret_15m_pct": 0.1,
                "ret_1h_pct": 0.2,
                "ret_4h_pct": 0.3,
                "ret_24h_pct": -0.5,
                "volume_ratio_1h": 0.9,
                "volume_ratio_4h": 1.0,
                "breakout_24h": False,
                "long_explosion_score": None,
                "dump_score": None,
                "reversal_watch_score": None,
                "primary_ohlc_score": None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# filter_active_signals
# ---------------------------------------------------------------------------


def test_filter_active_signals_removes_blank_rows() -> None:
    df = _make_signal_df(n_long=1, n_dump=1, n_blank=2)
    out = filter_active_signals(df)
    assert len(out) == 2
    assert all(out["ohlc_signal_type"] != "")


def test_filter_active_signals_keeps_all_three_signal_types() -> None:
    rows = _make_signal_df(n_long=1, n_dump=1, n_blank=0)
    rows = pd.concat(
        [rows, pd.DataFrame([{**rows.iloc[0].to_dict(), "ohlc_signal_type": "REVERSAL_WATCH", "pair_id": "REVUSD"}])]
    )
    out = filter_active_signals(rows)
    assert "LONG_EXPLOSION" in out["ohlc_signal_type"].values
    assert "DUMPING" in out["ohlc_signal_type"].values
    assert "REVERSAL_WATCH" in out["ohlc_signal_type"].values


# ---------------------------------------------------------------------------
# stamp_snapshot
# ---------------------------------------------------------------------------


def test_stamp_snapshot_adds_timestamp_column() -> None:
    df = filter_active_signals(_make_signal_df())
    out = stamp_snapshot(df, snapshot_ts=_FIXED_TS)
    assert "snapshot_ts_utc" in out.columns
    assert (out["snapshot_ts_utc"] == _FIXED_TS).all()


def test_stamp_snapshot_contains_all_required_columns() -> None:
    df = filter_active_signals(_make_signal_df())
    out = stamp_snapshot(df, snapshot_ts=_FIXED_TS)
    for col in SNAPSHOT_COLUMNS:
        assert col in out.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# write_snapshot_file — creates timestamped file
# ---------------------------------------------------------------------------


def test_write_snapshot_file_creates_file(tmp_path: Path) -> None:
    df = stamp_snapshot(filter_active_signals(_make_signal_df()), snapshot_ts=_FIXED_TS)
    path = write_snapshot_file(df, snapshot_dir=tmp_path, snapshot_ts_str=_FIXED_TS_STR)
    assert path.exists()
    assert path.name == f"memecoin_signals_{_FIXED_TS_STR}.csv"


def test_write_snapshot_file_contains_correct_rows(tmp_path: Path) -> None:
    df = stamp_snapshot(filter_active_signals(_make_signal_df(n_long=2, n_dump=1, n_blank=0)), snapshot_ts=_FIXED_TS)
    path = write_snapshot_file(df, snapshot_dir=tmp_path, snapshot_ts_str=_FIXED_TS_STR)
    loaded = pd.read_csv(path)
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# append_to_history — creates and appends
# ---------------------------------------------------------------------------


def test_append_to_history_creates_when_missing(tmp_path: Path) -> None:
    df = stamp_snapshot(filter_active_signals(_make_signal_df()), snapshot_ts=_FIXED_TS)
    history_path = tmp_path / "history.csv"
    action = append_to_history(df, history_path, snapshot_ts=_FIXED_TS)
    assert action == "created"
    assert history_path.exists()


def test_history_file_created_has_correct_rows(tmp_path: Path) -> None:
    df = stamp_snapshot(filter_active_signals(_make_signal_df(n_long=1, n_dump=1, n_blank=0)), snapshot_ts=_FIXED_TS)
    history_path = tmp_path / "history.csv"
    append_to_history(df, history_path, snapshot_ts=_FIXED_TS)
    loaded = pd.read_csv(history_path)
    assert len(loaded) == 2


def test_append_to_history_appends_to_existing(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"

    ts1 = "2026-05-28T21:30:00Z"
    df1 = stamp_snapshot(filter_active_signals(_make_signal_df(n_long=1, n_dump=0, n_blank=0)), snapshot_ts=ts1)
    append_to_history(df1, history_path, snapshot_ts=ts1)

    ts2 = "2026-05-28T21:45:00Z"
    df2 = stamp_snapshot(filter_active_signals(_make_signal_df(n_long=0, n_dump=1, n_blank=0)), snapshot_ts=ts2)
    action = append_to_history(df2, history_path, snapshot_ts=ts2)

    assert action == "appended"
    loaded = pd.read_csv(history_path)
    assert len(loaded) == 2
    assert set(loaded["snapshot_ts_utc"].unique()) == {ts1, ts2}


def test_append_to_history_skips_duplicate_snapshot_ts(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    df = stamp_snapshot(filter_active_signals(_make_signal_df(n_long=1, n_dump=0, n_blank=0)), snapshot_ts=_FIXED_TS)

    append_to_history(df, history_path, snapshot_ts=_FIXED_TS)
    action = append_to_history(df, history_path, snapshot_ts=_FIXED_TS)

    assert action == "duplicate_skipped"
    loaded = pd.read_csv(history_path)
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# run_signal_snapshot integration
# ---------------------------------------------------------------------------


def test_run_signal_snapshot_creates_expected_files(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    _make_signal_df(n_long=2, n_dump=1, n_blank=1).to_csv(input_path, index=False)
    snapshot_dir = tmp_path / "snapshots"
    history_path = tmp_path / "history.csv"

    _, snap_path, action = run_signal_snapshot(
        input_path=input_path,
        snapshot_dir=snapshot_dir,
        history_path=history_path,
        now=_FIXED_DT,
    )

    assert snap_path.exists()
    assert snap_path.name == f"memecoin_signals_{_FIXED_TS_STR}.csv"
    assert history_path.exists()
    assert action == "created"


def test_run_signal_snapshot_filters_blank_signals(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    _make_signal_df(n_long=1, n_dump=0, n_blank=3).to_csv(input_path, index=False)
    snapshot_dir = tmp_path / "snapshots"
    history_path = tmp_path / "history.csv"

    df, _, _ = run_signal_snapshot(
        input_path=input_path,
        snapshot_dir=snapshot_dir,
        history_path=history_path,
        now=_FIXED_DT,
    )
    assert all(df["ohlc_signal_type"] != "")
    assert len(df) == 1


def test_run_snapshot_history_rows_have_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    _make_signal_df(n_long=1, n_dump=1, n_blank=0).to_csv(input_path, index=False)
    snapshot_dir = tmp_path / "snapshots"
    history_path = tmp_path / "history.csv"

    run_signal_snapshot(
        input_path=input_path,
        snapshot_dir=snapshot_dir,
        history_path=history_path,
        now=_FIXED_DT,
    )
    loaded = pd.read_csv(history_path)
    for col in SNAPSHOT_COLUMNS:
        assert col in loaded.columns, f"History missing column: {col}"


# ---------------------------------------------------------------------------
# Module safety check
# ---------------------------------------------------------------------------


def test_module_does_not_import_live_or_broker_modules() -> None:
    module_path = Path("research/memecoin_catcher/save_signal_snapshot.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert all(not m.startswith("live") for m in imported)
    assert all(not m.startswith("brokers") for m in imported)
    assert all("config" not in m for m in imported)
