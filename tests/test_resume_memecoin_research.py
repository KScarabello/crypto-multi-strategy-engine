"""Tests for research/memecoin_catcher/resume_memecoin_research.py."""

from __future__ import annotations

import csv
import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

import research.memecoin_catcher.resume_memecoin_research as rr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "data" / "memecoin_signal_history.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with p.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        p.write_text("snapshot_ts_utc\n")
    return p


def _make_outcomes_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "data" / "memecoin_signal_outcomes.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with p.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        p.write_text("snapshot_ts_utc,outcome_15m,outcome_1h,outcome_4h,outcome_24h\n")
    return p


# ---------------------------------------------------------------------------
# No live trading / private imports
# ---------------------------------------------------------------------------


def test_no_broker_imports():
    """Module must not import broker, credentials, or trading modules.

    Uses import-pattern matching so that docstring mentions (e.g.
    "No private credentials.") do not trigger a false positive.
    """
    import re
    src = Path("research/memecoin_catcher/resume_memecoin_research.py").read_text()
    # Match actual import/from-import lines containing forbidden tokens
    import_lines = [
        line for line in src.splitlines()
        if re.match(r"\s*(import|from)\s+", line)
    ]
    import_block = "\n".join(import_lines)
    # Also flag non-import call patterns that would execute live code
    call_patterns = ["place_order", "ccxt"]
    forbidden_imports = ["brokers", "credentials", "live_trading"]
    for token in forbidden_imports:
        assert token not in import_block, (
            f"Forbidden import/reference found in import statements: {token!r}"
        )
    for token in call_patterns:
        assert token not in src, (
            f"Forbidden call-site reference found: {token!r}"
        )


# ---------------------------------------------------------------------------
# read_data_state
# ---------------------------------------------------------------------------


def test_read_data_state_missing_files(tmp_path):
    state = rr.read_data_state(
        history_path=tmp_path / "nope_history.csv",
        outcomes_path=tmp_path / "nope_outcomes.csv",
    )
    assert state["history_rows"] == 0
    assert state["outcomes_rows"] == 0
    assert state["newest_signal_ts"] is None
    assert state["oldest_signal_ts"] is None
    assert state["complete_15m"] == 0


def test_read_data_state_with_files(tmp_path):
    hist_rows = [
        {"snapshot_ts_utc": "2026-05-28T10:00:00Z", "pair_id": "XXYYZZ"},
        {"snapshot_ts_utc": "2026-05-29T12:00:00Z", "pair_id": "AABCC"},
    ]
    out_rows = [
        {
            "snapshot_ts_utc": "2026-05-28T10:00:00Z",
            "pair_id": "XXYYZZ",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "outcome_15m": "SUCCESS",
            "outcome_1h": "FLAT",
            "outcome_4h": "",
            "outcome_24h": "",
        }
    ]
    h = _make_history_csv(tmp_path, hist_rows)
    o = _make_outcomes_csv(tmp_path, out_rows)
    state = rr.read_data_state(history_path=h, outcomes_path=o)
    assert state["history_rows"] == 2
    assert state["outcomes_rows"] == 1
    assert state["complete_15m"] == 1
    assert state["complete_1h"] == 1
    assert state["complete_4h"] == 0
    assert state["complete_24h"] == 0
    assert state["newest_signal_ts"] is not None
    assert state["oldest_signal_ts"] is not None


# ---------------------------------------------------------------------------
# check_downtime_warning
# ---------------------------------------------------------------------------


def test_downtime_warning_triggered(capsys):
    now = datetime(2026, 5, 31, 18, 0, 0, tzinfo=timezone.utc)
    old_ts = pd.Timestamp("2026-05-29T12:00:00Z", tz="UTC")
    state = {"newest_signal_ts": old_ts}
    rr.check_downtime_warning(state, now=now)
    captured = capsys.readouterr()
    assert "Scanner has not captured signals since" in captured.out
    assert "not recoverable" in captured.out


def test_downtime_warning_not_triggered(capsys):
    now = datetime(2026, 5, 31, 18, 0, 0, tzinfo=timezone.utc)
    recent_ts = pd.Timestamp("2026-05-31T10:00:00Z", tz="UTC")
    state = {"newest_signal_ts": recent_ts}
    rr.check_downtime_warning(state, now=now)
    captured = capsys.readouterr()
    assert "Scanner has not captured signals since" not in captured.out


def test_downtime_warning_no_signal(capsys):
    now = datetime(2026, 5, 31, 18, 0, 0, tzinfo=timezone.utc)
    state = {"newest_signal_ts": None}
    rr.check_downtime_warning(state, now=now)
    captured = capsys.readouterr()
    assert "Scanner has not captured signals since" not in captured.out


# ---------------------------------------------------------------------------
# run_resume step ordering
# ---------------------------------------------------------------------------


def _mock_step_fns():
    """Return a dict of patch targets → MagicMock for all inner step functions."""
    targets = [
        "_step_evaluate_outcomes",
        "_step_summarize_outcomes",
        "_step_fetch_universe",
        "_step_rank_candidates",
        "_step_enrich_ohlc",
        "_step_save_snapshot",
    ]
    return {t: MagicMock() for t in targets}


def test_steps_run_in_correct_order(tmp_path):
    """All steps run in A→B→C1→C2→C3→C4→D→E order."""
    order = []

    def _make_fn(label):
        def fn(*args, **kwargs):
            order.append(label)
        return fn

    patches = {
        "research.memecoin_catcher.resume_memecoin_research._step_evaluate_outcomes": _make_fn("eval"),
        "research.memecoin_catcher.resume_memecoin_research._step_summarize_outcomes": _make_fn("summarize"),
        "research.memecoin_catcher.resume_memecoin_research._step_fetch_universe": _make_fn("fetch"),
        "research.memecoin_catcher.resume_memecoin_research._step_rank_candidates": _make_fn("rank"),
        "research.memecoin_catcher.resume_memecoin_research._step_enrich_ohlc": _make_fn("enrich"),
        "research.memecoin_catcher.resume_memecoin_research._step_save_snapshot": _make_fn("snapshot"),
    }

    with patch.multiple("research.memecoin_catcher.resume_memecoin_research", **{
        k.split(".")[-1]: v for k, v in patches.items()
    }):
        rr.run_resume(
            history_path=tmp_path / "h.csv",
            outcomes_path=tmp_path / "o.csv",
        )

    assert order == ["eval", "summarize", "fetch", "rank", "enrich", "snapshot", "eval", "summarize"]


def test_skip_fresh_snapshot_skips_scanner_steps(tmp_path):
    order = []

    def _make_fn(label):
        def fn(*args, **kwargs):
            order.append(label)
        return fn

    with patch.multiple(
        "research.memecoin_catcher.resume_memecoin_research",
        _step_evaluate_outcomes=_make_fn("eval"),
        _step_summarize_outcomes=_make_fn("summarize"),
        _step_fetch_universe=_make_fn("fetch"),
        _step_rank_candidates=_make_fn("rank"),
        _step_enrich_ohlc=_make_fn("enrich"),
        _step_save_snapshot=_make_fn("snapshot"),
    ):
        rr.run_resume(
            skip_fresh_snapshot=True,
            history_path=tmp_path / "h.csv",
            outcomes_path=tmp_path / "o.csv",
        )

    assert "fetch" not in order
    assert "rank" not in order
    assert "enrich" not in order
    assert "snapshot" not in order
    assert "eval" in order
    assert "summarize" in order


def test_skip_outcome_eval_skips_evaluator_steps(tmp_path):
    order = []

    def _make_fn(label):
        def fn(*args, **kwargs):
            order.append(label)
        return fn

    with patch.multiple(
        "research.memecoin_catcher.resume_memecoin_research",
        _step_evaluate_outcomes=_make_fn("eval"),
        _step_summarize_outcomes=_make_fn("summarize"),
        _step_fetch_universe=_make_fn("fetch"),
        _step_rank_candidates=_make_fn("rank"),
        _step_enrich_ohlc=_make_fn("enrich"),
        _step_save_snapshot=_make_fn("snapshot"),
    ):
        rr.run_resume(
            skip_outcome_eval=True,
            history_path=tmp_path / "h.csv",
            outcomes_path=tmp_path / "o.csv",
        )

    assert "eval" not in order
    assert "summarize" not in order
    assert "fetch" in order
    assert "rank" in order
    assert "enrich" in order
    assert "snapshot" in order


def test_dry_run_does_not_call_mutating_steps(tmp_path):
    mocks = {}
    step_names = [
        "_step_evaluate_outcomes",
        "_step_summarize_outcomes",
        "_step_fetch_universe",
        "_step_rank_candidates",
        "_step_enrich_ohlc",
        "_step_save_snapshot",
    ]
    for name in step_names:
        mocks[name] = MagicMock()

    with patch.multiple("research.memecoin_catcher.resume_memecoin_research", **mocks):
        results = rr.run_resume(
            dry_run=True,
            history_path=tmp_path / "h.csv",
            outcomes_path=tmp_path / "o.csv",
        )

    for name in step_names:
        mocks[name].assert_not_called()

    for r in results:
        assert r.skipped is True


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


def test_step_result_success_str():
    r = rr.StepResult(name="foo", success=True, duration_s=1.23)
    assert "✓ OK" in str(r)
    assert "foo" in str(r)


def test_step_result_failure_str():
    r = rr.StepResult(name="bar", success=False, duration_s=0.5, error="boom")
    assert "✗ FAILED" in str(r)
    assert "boom" in str(r)


def test_step_result_skipped_str():
    r = rr.StepResult(name="baz", success=True, duration_s=0.0, skipped=True)
    assert "SKIPPED" in str(r)


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = rr._parse_args([])
    assert args.dry_run is False
    assert args.skip_fresh_snapshot is False
    assert args.skip_outcome_eval is False
    assert args.continue_on_error is False


def test_parse_args_flags():
    args = rr._parse_args([
        "--dry-run",
        "--skip-fresh-snapshot",
        "--skip-outcome-eval",
        "--continue-on-error",
    ])
    assert args.dry_run is True
    assert args.skip_fresh_snapshot is True
    assert args.skip_outcome_eval is True
    assert args.continue_on_error is True
