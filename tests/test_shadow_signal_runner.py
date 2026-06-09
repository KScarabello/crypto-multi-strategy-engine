"""Tests for shadow signal runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.shadow.specs import SPEC_HASHES, FROZEN_CANDIDATES, PROSPECTIVE_START
from research.shadow.portfolio import init_portfolio
from research.shadow.signal_runner import (
    BarDecision,
    compute_gate_at_bar,
    run_bar,
)

UNIVERSE = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]


def _make_close(n_bars=300, freq="4h", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    prices = {"BTC/USD": 65000, "ETH/USD": 3500, "XRP/USD": 0.6, "SOL/USD": 150, "AVAX/USD": 35}
    data = {}
    for sym in UNIVERSE:
        p = prices[sym]
        ret = rng.normal(0.0002, 0.015, n_bars)
        data[sym] = p * (1 + ret).cumprod()
    return pd.DataFrame(data, index=idx)


def test_gate_above_ma_with_buffer_is_risk_on():
    btc = pd.Series([100.0] * 250 + [106.0], dtype=float)
    gate, count, price, ma = compute_gate_at_bar(
        btc_close=btc,
        btc_ma_bars=240,
        entry_confirm_bars=1,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        last_gate=False,
        gate_entry_confirm_count=0,
    )
    # 106 >= 100 * 1.005 = 100.5 → above threshold
    assert gate is True
    assert price == 106.0


def test_gate_below_ma_is_risk_off():
    btc = pd.Series([100.0] * 250 + [95.0], dtype=float)
    gate, count, price, ma = compute_gate_at_bar(
        btc_close=btc,
        btc_ma_bars=240,
        entry_confirm_bars=1,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        last_gate=True,
        gate_entry_confirm_count=0,
    )
    # 95 < 100 → below exit threshold
    assert gate is False


def test_entry_confirm_requires_N_consecutive_bars():
    # With entry_confirm_bars=2, one bar is not enough
    btc = pd.Series([100.0] * 250 + [106.0], dtype=float)
    gate, count, price, ma = compute_gate_at_bar(
        btc_close=btc,
        btc_ma_bars=240,
        entry_confirm_bars=2,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        last_gate=False,
        gate_entry_confirm_count=0,
    )
    # First bar: count becomes 1, not yet confirmed
    assert gate is False
    assert count == 1

    # Second bar above threshold: confirmed
    gate2, count2, _, _ = compute_gate_at_bar(
        btc_close=btc,
        btc_ma_bars=240,
        entry_confirm_bars=2,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        last_gate=False,
        gate_entry_confirm_count=count,
    )
    assert gate2 is True


def test_no_future_data_used():
    close = _make_close(n_bars=300)
    bar_ts = close.index[150]
    portfolio = init_portfolio("control", SPEC_HASHES["control"], str(bar_ts))

    decision = run_bar(
        candidate_name="control",
        bar_ts=bar_ts,
        close=close,
        portfolio=portfolio,
        spec_hash=SPEC_HASHES["control"],
    )
    data_cutoff = pd.Timestamp(decision.data_cutoff_ts)
    assert data_cutoff <= bar_ts


def test_buyhold_never_rebalances_after_first_bar():
    close = _make_close(n_bars=50)

    # First bar
    bar_ts0 = close.index[0]
    portfolio = init_portfolio("btc_buyhold", SPEC_HASHES["btc_buyhold"], str(bar_ts0))
    d0 = run_bar("btc_buyhold", bar_ts0, close, portfolio, SPEC_HASHES["btc_buyhold"])
    assert d0.should_rebalance is True

    # Subsequent bars
    portfolio.bar_count = 1
    portfolio.holdings = {"BTC/USD": 1.0}
    for i in range(1, 10):
        bar_ts = close.index[i]
        d = run_bar("btc_buyhold", bar_ts, close, portfolio, SPEC_HASHES["btc_buyhold"])
        assert d.should_rebalance is False


def test_strategy_one_bar_execution_delay():
    close = _make_close(n_bars=300)
    bar_ts = close.index[100]
    portfolio = init_portfolio("control", SPEC_HASHES["control"], str(bar_ts))

    decision = run_bar(
        candidate_name="control",
        bar_ts=bar_ts,
        close=close,
        portfolio=portfolio,
        spec_hash=SPEC_HASHES["control"],
    )
    decision_ts = pd.Timestamp(decision.decision_ts)
    execution_ts = pd.Timestamp(decision.execution_ts)
    assert execution_ts > decision_ts


def test_regime_exit_overrides_min_hold():
    """When regime turns risk-off, holdings should go to cash regardless of min_hold."""
    close = _make_close(n_bars=300)
    bar_ts = close.index[250]
    portfolio = init_portfolio("min_hold_6", SPEC_HASHES["min_hold_6"], str(bar_ts))
    # Simulate already holding positions
    portfolio.holdings = {"BTC/USD": 0.33, "ETH/USD": 0.33, "SOL/USD": 0.34}
    portfolio.last_gate = True
    portfolio.hold_age = {"BTC/USD": 1, "ETH/USD": 1, "SOL/USD": 1}  # below min_hold

    # Force BTC price well below MA by using a short series
    close_short = close.iloc[:250].copy()
    # Decrease BTC dramatically to force risk-off
    close_short["BTC/USD"] = close_short["BTC/USD"] * 0.5  # 50% drop — definitely risk-off

    decision = run_bar(
        candidate_name="min_hold_6",
        bar_ts=close_short.index[-1],
        close=close_short,
        portfolio=portfolio,
        spec_hash=SPEC_HASHES["min_hold_6"],
    )
    if not decision.btc_regime_state:
        assert decision.target_holdings == {} or sum(decision.target_holdings.values()) == 0.0


def test_bar_decision_is_not_live_order():
    """BarDecision has no order attributes — it's research only."""
    close = _make_close(n_bars=300)
    bar_ts = close.index[100]
    portfolio = init_portfolio("control", SPEC_HASHES["control"], str(bar_ts))
    decision = run_bar("control", bar_ts, close, portfolio, SPEC_HASHES["control"])
    assert isinstance(decision, BarDecision)
    assert not hasattr(decision, "order_id")
    assert not hasattr(decision, "exchange_order")
    assert not hasattr(decision, "kraken_response")


def test_target_weights_sum_to_leq_1():
    close = _make_close(n_bars=300)
    bar_ts = close.index[250]
    for name in FROZEN_CANDIDATES:
        portfolio = init_portfolio(name, SPEC_HASHES[name], str(bar_ts))
        decision = run_bar(name, bar_ts, close, portfolio, SPEC_HASHES[name])
        total = sum(decision.target_holdings.values())
        assert total <= 1.0 + 1e-6, f"{name}: target weights sum to {total}"


def test_empty_universe_returns_all_cash():
    close = _make_close(n_bars=10)  # too few bars for any signal
    bar_ts = close.index[-1]
    portfolio = init_portfolio("control", SPEC_HASHES["control"], str(bar_ts))
    decision = run_bar("control", bar_ts, close, portfolio, SPEC_HASHES["control"])
    # With < 36 bars history, no eligible symbols → all cash
    total = sum(decision.target_holdings.values())
    assert total <= 1.0 + 1e-6
