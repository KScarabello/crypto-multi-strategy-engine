"""Tests for shadow portfolio state management."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research.shadow.portfolio import (
    ShadowPortfolio,
    init_portfolio,
    load_portfolio,
    save_portfolio,
    compute_state_hash,
)
from research.shadow.specs import SPEC_HASHES


def _make_state_dir():
    tmp = tempfile.mkdtemp()
    return Path(tmp)


def test_init_portfolio_starts_at_initial_capital():
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00", 10_000.0)
    assert p.equity == 10_000.0


def test_init_portfolio_has_zero_holdings():
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    assert p.holdings == {}
    assert p.cash_weight == 1.0


def test_save_and_load_portfolio_roundtrip():
    state_dir = _make_state_dir()
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    p.equity = 10_500.0
    p.holdings = {"BTC/USD": 0.5, "ETH/USD": 0.5}
    save_portfolio(p, state_dir)
    loaded = load_portfolio("control", state_dir)
    assert loaded is not None
    assert loaded.equity == 10_500.0
    assert loaded.holdings == {"BTC/USD": 0.5, "ETH/USD": 0.5}


def test_state_hash_changes_after_equity_update():
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    h1 = compute_state_hash(p)
    p.equity = 11_000.0
    h2 = compute_state_hash(p)
    assert h1 != h2


def test_state_hash_is_deterministic():
    p = init_portfolio("min_hold_6", SPEC_HASHES["min_hold_6"], "2026-06-09T00:00:00+00:00")
    h1 = compute_state_hash(p)
    h2 = compute_state_hash(p)
    assert h1 == h2


def test_portfolio_prev_hash_set_on_save():
    state_dir = _make_state_dir()
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    new_hash = save_portfolio(p, state_dir)
    assert p.prev_state_hash == new_hash
    assert len(new_hash) == 16


def test_equity_history_accumulates():
    state_dir = _make_state_dir()
    p = init_portfolio("control", SPEC_HASHES["control"], "2026-06-09T00:00:00+00:00")
    p.equity_history.append({"ts": "2026-06-09T00:00:00+00:00", "equity": 10_000.0})
    p.equity_history.append({"ts": "2026-06-09T04:00:00+00:00", "equity": 10_100.0})
    save_portfolio(p, state_dir)
    loaded = load_portfolio("control", state_dir)
    assert len(loaded.equity_history) == 2


def test_missing_portfolio_file_returns_none():
    state_dir = _make_state_dir()
    result = load_portfolio("nonexistent_candidate", state_dir)
    assert result is None
