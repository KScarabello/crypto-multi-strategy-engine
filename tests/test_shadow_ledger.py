"""Tests for append-only ledger."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from research.shadow.specs import SPEC_HASHES
from research.shadow.portfolio import init_portfolio
from research.shadow.signal_runner import BarDecision
from research.shadow.ledger import (
    append_decision,
    load_ledger,
    get_last_decision_ts,
    is_bar_already_processed,
)


def _make_state_dir() -> Path:
    return Path(tempfile.mkdtemp())


def _make_decision(ts: str = "2026-06-09T04:00:00+00:00", name: str = "control") -> BarDecision:
    return BarDecision(
        candidate_name=name,
        spec_hash=SPEC_HASHES[name],
        decision_ts=ts,
        execution_ts="2026-06-09T08:00:00+00:00",
        data_cutoff_ts=ts,
        btc_regime_state=True,
        btc_price=65000.0,
        btc_ma_value=62000.0,
        should_rebalance=False,
        current_holdings={},
        target_holdings={"BTC/USD": 0.33},
        momentum_scores={"BTC/USD": 0.05},
        momentum_ranks={"BTC/USD": 1},
        hypothetical_trades=[],
        bar_count_at_decision=1,
        rebalance_bar_index=0,
        is_prospective=True,
    )


def test_append_creates_file():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    decision = _make_decision()
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    ledger_path = state_dir / "ledger" / "control.jsonl"
    assert ledger_path.exists()


def test_append_adds_exactly_one_record():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    decision = _make_decision()
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    records = load_ledger("control", state_dir)
    assert len(records) == 1


def test_duplicate_bar_is_idempotent():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    decision = _make_decision()
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    records = load_ledger("control", state_dir)
    assert len(records) == 1


def test_load_ledger_returns_all_records():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    for i, ts in enumerate(["2026-06-09T04:00:00+00:00", "2026-06-09T08:00:00+00:00", "2026-06-09T12:00:00+00:00"]):
        d = _make_decision(ts=ts)
        append_decision("control", d, portfolio, 10_000.0 + i * 100, 0.01, state_dir)
    records = load_ledger("control", state_dir)
    assert len(records) == 3


def test_get_last_decision_ts_returns_latest():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    for ts in ["2026-06-09T04:00:00+00:00", "2026-06-09T08:00:00+00:00"]:
        d = _make_decision(ts=ts)
        append_decision("control", d, portfolio, 10_000.0, 0.0, state_dir)
    last_ts = get_last_decision_ts("control", state_dir)
    assert last_ts == pd.Timestamp("2026-06-09T08:00:00+00:00")


def test_ledger_records_are_valid_json():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    decision = _make_decision()
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    path = state_dir / "ledger" / "control.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert isinstance(record, dict)


def test_ledger_file_is_append_only():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    d1 = _make_decision(ts="2026-06-09T04:00:00+00:00")
    append_decision("control", d1, portfolio, 10_000.0, 0.0, state_dir)
    path = state_dir / "ledger" / "control.jsonl"
    size_after_first = path.stat().st_size

    d2 = _make_decision(ts="2026-06-09T08:00:00+00:00")
    append_decision("control", d2, portfolio, 10_100.0, 0.01, state_dir)
    size_after_second = path.stat().st_size

    assert size_after_second > size_after_first


def test_is_bar_already_processed_true_after_append():
    state_dir = _make_state_dir()
    portfolio = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    ts = "2026-06-09T04:00:00+00:00"
    decision = _make_decision(ts=ts)
    append_decision("control", decision, portfolio, 10_000.0, 0.0, state_dir)
    assert is_bar_already_processed("control", ts, state_dir) is True


def test_is_bar_already_processed_false_before_append():
    state_dir = _make_state_dir()
    ts = "2026-06-09T04:00:00+00:00"
    assert is_bar_already_processed("control", ts, state_dir) is False
