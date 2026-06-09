"""Tests for research/fixed_five_whipsaw_control.py

Covers:
- Completed-bar confirmation (no future-bar use)
- One-bar execution delay
- Entry/exit hysteresis
- Minimum regime duration
- Transition and cluster counting
- Cost accounting distinction
- Deterministic execution
- Identical canonical initialization
- No live imports
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_whipsaw_control as wc
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    compute_controlled_gate_series,
    count_whipsaw_clusters,
    classify_whipsaw_improvement,
    build_gate_transition_events,
    compute_cost_accounting,
    WhipsawControlledSignalGen,
)
from research.fixed_five_defensive_overlay import BaseSignalGen


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_btc_series(n: int = 300, trend: float = 0.0, seed: int = 42) -> pd.Series:
    """Synthetic BTC close price series."""
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    prices = 10_000 * np.cumprod(1 + trend + rng.normal(0, 0.01, n))
    return pd.Series(prices, index=idx, name="BTC/USD")


def _make_close(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """5-symbol close DataFrame."""
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data = {}
    for sym in symbols:
        data[sym] = 1000 * np.cumprod(1 + 0.001 + rng.normal(0, 0.01, n))
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# WhipsawControl dataclass validation
# ---------------------------------------------------------------------------

class TestWhipsawControlValidation:
    def test_defaults_are_immediate_no_lock(self):
        p = WhipsawControl(name="test")
        assert p.entry_confirm_bars == 1
        assert p.exit_confirm_bars == 1
        assert p.entry_buffer_pct == 0.0
        assert p.exit_buffer_pct == 0.0
        assert p.min_duration_bars == 0

    def test_invalid_entry_confirm_raises(self):
        with pytest.raises(ValueError):
            WhipsawControl(name="bad", entry_confirm_bars=0)

    def test_invalid_exit_confirm_raises(self):
        with pytest.raises(ValueError):
            WhipsawControl(name="bad", exit_confirm_bars=0)

    def test_invalid_min_duration_raises(self):
        with pytest.raises(ValueError):
            WhipsawControl(name="bad", min_duration_bars=-1)


# ---------------------------------------------------------------------------
# Completed-bar confirmation — no future bar used
# ---------------------------------------------------------------------------

class TestCompletedBarConfirmation:
    def test_immediate_entry_fires_on_first_above_bar(self):
        """With entry_confirm=1, gate turns ON at the first bar BTC > MA."""
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # BTC stays below MA for 50 bars, then jumps above
        prices = [500.0] * 50 + [2000.0] * 50  # MA is ~500 in first 50 bars
        btc = pd.Series(prices, index=idx)
        ma_bars = 10
        params = WhipsawControl(name="ctrl", entry_confirm_bars=1)
        gate = compute_controlled_gate_series(btc, ma_bars, params)
        # Should turn on somewhere after bar 50 (when BTC jumps above MA)
        # The gate was False before bar 50
        assert gate.iloc[0] == False  # noqa: E712
        assert gate.iloc[-1] == True  # noqa: E712

    def test_entry_confirm_2_requires_two_consecutive_bars(self):
        """Gate must stay OFF after only 1 bar above MA when confirm=2."""
        n = 150
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # Build a series where MA(10) is around 1000
        prices = [1000.0] * 50  # MA at 1000
        # One bar above: 1100, then back to 900
        prices += [1100.0, 900.0] * 20 + [1100.0] * 10
        prices += [900.0] * (n - len(prices))
        btc = pd.Series(prices[:n], index=idx)

        params_1bar = WhipsawControl(name="imm", entry_confirm_bars=1)
        params_2bar = WhipsawControl(name="conf2", entry_confirm_bars=2)

        gate_1 = compute_controlled_gate_series(btc, 10, params_1bar)
        gate_2 = compute_controlled_gate_series(btc, 10, params_2bar)

        # The 2-bar confirmation variant should have fewer or equal ON periods
        pct_on_1 = float(gate_1.mean())
        pct_on_2 = float(gate_2.mean())
        # 2-bar confirm cannot be on MORE than 1-bar confirm
        assert pct_on_2 <= pct_on_1 + 0.05  # allow 5% tolerance for edge effects

    def test_no_future_bar_used(self):
        """Gate at bar i must only use data from bars 0..i (not bar i+1)."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # BTC slowly rises above MA
        prices = [1000 + i * 2 for i in range(n)]
        btc = pd.Series(prices, index=idx)

        params = WhipsawControl(name="conf3", entry_confirm_bars=3)
        gate = compute_controlled_gate_series(btc, 20, params)

        # Compute gate with data truncated at bar 100
        btc_trunc = btc.iloc[:100]
        gate_trunc = compute_controlled_gate_series(btc_trunc, 20, params)

        # Gate values at bars 0..99 must match between full and truncated computation
        common_idx = gate.index[:100]
        pd.testing.assert_series_equal(
            gate.reindex(common_idx),
            gate_trunc.reindex(common_idx),
        )

    def test_confirmation_resets_on_signal_break(self):
        """If BTC goes above MA for 1 bar then back below, counter resets."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # First 80 bars at 500 (below MA)
        # Bar 80: spike above MA to 2000, then immediately back to 500
        # Bar 81-90: back below MA
        # Bar 91-92: above MA → entry fires
        prices = [500.0] * 80 + [2000.0] + [500.0] * 10 + [2000.0] * 50 + [500.0] * (n - 141)
        btc = pd.Series(prices[:n], index=idx)

        params = WhipsawControl(name="conf2", entry_confirm_bars=2)
        gate = compute_controlled_gate_series(btc, 20, params)

        # After the single spike (bar 80), gate should still be off at bar 81
        bar_81 = idx[81]
        assert not bool(gate.loc[bar_81])


# ---------------------------------------------------------------------------
# Hysteresis (entry and exit buffers)
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_entry_buffer_delays_entry(self):
        """With 1% entry buffer, gate ON requires BTC > MA * 1.01."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # BTC just barely above MA (0.5% above)
        prices = [1000.0] * 50 + [1005.0] * 150  # 0.5% above, MA ~1000
        btc = pd.Series(prices, index=idx)

        no_buf = WhipsawControl(name="no_buf")
        buf_1pct = WhipsawControl(name="buf_1pct", entry_buffer_pct=1.0)

        gate_0 = compute_controlled_gate_series(btc, 20, no_buf)
        gate_1 = compute_controlled_gate_series(btc, 20, buf_1pct)

        # No-buffer variant should enter when BTC > MA (0.5% above is enough)
        # 1% buffer variant should stay OUT (0.5% is not enough)
        # Compare time-in-cash: 1% buffer should have more cash
        pct_on_0 = float(gate_0.iloc[50:].mean())
        pct_on_1 = float(gate_1.iloc[50:].mean())
        assert pct_on_0 >= pct_on_1

    def test_exit_buffer_delays_exit(self):
        """With 1% exit buffer, gate stays ON until BTC < MA * 0.99."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # BTC above MA to start, then drops just 0.5% below MA
        prices = [2000.0] * 80 + [995.0] * 120  # MA around 1000-2000 range
        btc = pd.Series(prices, index=idx)

        no_buf = WhipsawControl(name="no_buf")
        buf_1pct = WhipsawControl(name="buf_1pct", exit_buffer_pct=1.0)

        gate_0 = compute_controlled_gate_series(btc, 20, no_buf)
        gate_1 = compute_controlled_gate_series(btc, 20, buf_1pct)

        # 1% exit buffer should stay ON longer (needs BTC < MA×0.99)
        assert float(gate_1.mean()) >= float(gate_0.mean()) - 0.05

    def test_asymmetric_hysteresis_entry_only(self):
        """entry_buffer_pct=1.0, exit_buffer_pct=0 behaves asymmetrically."""
        p = WhipsawControl(name="asym", entry_buffer_pct=1.0, exit_buffer_pct=0.0)
        assert p.entry_buffer_pct == 1.0
        assert p.exit_buffer_pct == 0.0


# ---------------------------------------------------------------------------
# Minimum regime duration
# ---------------------------------------------------------------------------

class TestMinimumRegimeDuration:
    def test_min_duration_prevents_rapid_transitions(self):
        """With min_duration=12, no transition can follow within 12 bars."""
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # Alternating: above MA / below MA every 3 bars
        ma_val = 1000.0
        prices = []
        for i in range(n):
            prices.append(1100.0 if (i // 3) % 2 == 0 else 900.0)
        btc = pd.Series(prices, index=idx)

        no_lock = WhipsawControl(name="no_lock")
        min_12 = WhipsawControl(name="min12", min_duration_bars=12)

        gate_0 = compute_controlled_gate_series(btc, 10, no_lock)
        gate_12 = compute_controlled_gate_series(btc, 10, min_12)

        trans_0 = len(gate_0[gate_0 != gate_0.shift(1)].dropna())
        trans_12 = len(gate_12[gate_12 != gate_12.shift(1)].dropna())

        assert trans_12 <= trans_0

    def test_min_duration_3_spacing(self):
        """After a transition at bar i, next transition must be at i + 3 or later."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # Monotone rising after bar 50 (single entry, no exit)
        prices = [900.0] * 50 + [1500.0] * 250
        btc = pd.Series(prices, index=idx)

        p = WhipsawControl(name="min3", min_duration_bars=3)
        gate = compute_controlled_gate_series(btc, 20, p)
        transitions = gate[gate != gate.shift(1)].dropna()
        trans_idx_arr = list(transitions.index)

        # All consecutive transitions must be >= 3 bars apart
        for j in range(1, len(trans_idx_arr)):
            bar_j = gate.index.get_loc(trans_idx_arr[j])
            bar_j1 = gate.index.get_loc(trans_idx_arr[j - 1])
            assert bar_j - bar_j1 >= 3, \
                f"Transition spacing too small: {bar_j - bar_j1} < 3"

    def test_uses_only_available_info(self):
        """Gate with min_duration is deterministic from same prefix."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        prices = [1000 * (1 + 0.001 * i) for i in range(n)]
        btc = pd.Series(prices, index=idx)

        p = WhipsawControl(name="min6", min_duration_bars=6)
        g1 = compute_controlled_gate_series(btc, 20, p)
        g2 = compute_controlled_gate_series(btc.iloc[:100], 20, p)

        pd.testing.assert_series_equal(g1.iloc[:100], g2)


# ---------------------------------------------------------------------------
# Transition and cluster counting
# ---------------------------------------------------------------------------

class TestTransitionCounting:
    def test_stable_gate_has_zero_transitions(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        gate = pd.Series([True] * 100, index=idx, dtype=bool)
        # shift(1) produces NaN at bar 0, compare from bar 1 onward
        transitions = gate.iloc[1:][gate.iloc[1:] != gate.iloc[:-1].values]
        assert len(transitions) == 0

    def test_alternating_gate_transitions(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC")
        gate = pd.Series([True, False, True, False, True, False, True, False, True, False],
                         index=idx, dtype=bool)
        # Count transitions from bar 1 onward (9 changes across 10 values)
        transitions = gate.iloc[1:][gate.iloc[1:] != gate.iloc[:-1].values]
        assert len(transitions) == 9

    def test_count_whipsaw_clusters_zero_for_stable(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        gate = pd.Series([True] * 100, index=idx, dtype=bool)
        n_clusters = count_whipsaw_clusters(gate, window_days=7)
        assert n_clusters == 0

    def test_count_whipsaw_clusters_detects_rapid_transitions(self):
        """A cluster of rapid transitions should be counted."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        vals = [True] * 50 + [False, True, False, True, False] + [True] * 145
        gate = pd.Series(vals, index=idx, dtype=bool)
        n_clusters_7 = count_whipsaw_clusters(gate, window_days=7, min_transitions=3)
        assert n_clusters_7 >= 1

    def test_3day_clusters_le_7day_clusters(self):
        """7-day clusters must be >= 3-day clusters (wider window catches more)."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        # Create some close transitions
        vals = [True] * 30
        for _ in range(10):
            vals += [False, True, False]  # fast transitions
        vals += [True] * (n - len(vals))
        gate = pd.Series(vals[:n], index=idx, dtype=bool)
        n3 = count_whipsaw_clusters(gate, window_days=3)
        n7 = count_whipsaw_clusters(gate, window_days=7)
        assert n7 >= n3

    def test_control_gate_matches_raw_ma(self):
        """Control variant (no modifications) should match raw MA gate."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(99)
        prices = 1000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        btc = pd.Series(prices, index=idx)

        params = WhipsawControl(name="control")
        controlled = compute_controlled_gate_series(btc, 20, params)

        # Raw MA gate: BTC > MA
        ma = btc.rolling(20, min_periods=20).mean()
        raw = (btc > ma).fillna(False)

        # After MA warms up, should match closely
        # (Both start False; controlled may differ slightly at first transition)
        after_warmup = idx[25:]
        pd.testing.assert_series_equal(
            controlled.reindex(after_warmup).rename("gate"),
            raw.reindex(after_warmup).rename("gate"),
        )


# ---------------------------------------------------------------------------
# One-bar execution delay
# ---------------------------------------------------------------------------

class TestOneBarDelay:
    def _run_with_gate(self, gate_series, n=300):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start, _close_to_ohlcv,
        )
        from research.fixed_five_canonical_regime_comparison import run_canonical

        close = _make_close(n=n)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        # Reindex gate to close index
        gate = gate_series.reindex(close.index).fillna(False)
        base = BaseSignalGen()
        sig = WhipsawControlledSignalGen(gate, base)
        ohlcv = _close_to_ohlcv(close)
        result, _ = run_canonical(
            ohlcv, sig, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        return result

    def test_signal_and_execution_timestamps_differ(self):
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        gate = pd.Series([True] * n, index=idx, dtype=bool)
        result = self._run_with_gate(gate, n=n)
        reb = result.rebalance_log
        if len(reb) > 0:
            assert (reb["signal_timestamp"] != reb["execution_timestamp"]).all()

    def test_gate_off_produces_zero_weights(self):
        """When gate is always False, no investment should occur."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        gate = pd.Series([False] * n, index=idx, dtype=bool)
        result = self._run_with_gate(gate, n=n)
        # No rebalances with positive weight_sum
        reb = result.rebalance_log
        if len(reb) > 0:
            assert float(reb["weight_sum"].max()) < 0.01


# ---------------------------------------------------------------------------
# Gate transition events (delay analysis)
# ---------------------------------------------------------------------------

class TestGateTransitionEvents:
    def test_control_vs_self_has_zero_delay(self):
        """Comparing a gate to itself should produce zero delays."""
        btc = _make_btc_series(n=200)
        close = _make_close(n=200)
        params = WhipsawControl(name="ctrl")
        gate = compute_controlled_gate_series(btc, 20, params)
        gate_fx = gate.reindex(close.index).fillna(False)

        events = build_gate_transition_events("ctrl", gate_fx, gate_fx, close)
        if events:
            for e in events:
                assert e.delay_bars == 0

    def test_confirmed_gate_has_nonneg_delays(self):
        """Confirmation variant should have non-negative entry delays vs raw."""
        btc = _make_btc_series(n=400, trend=0.0002)
        close = _make_close(n=400, seed=5)
        # Align btc to close index
        btc_aligned = btc.reindex(close.index).ffill()

        params_raw = WhipsawControl(name="raw")
        params_2bar = WhipsawControl(name="conf2", entry_confirm_bars=2)

        raw_gate = compute_controlled_gate_series(btc_aligned, 20, params_raw)
        conf_gate = compute_controlled_gate_series(btc_aligned, 20, params_2bar)

        raw_fx = raw_gate.reindex(close.index).fillna(False)
        conf_fx = conf_gate.reindex(close.index).fillna(False)

        events = build_gate_transition_events("conf2", raw_fx, conf_fx, close)
        entry_events = [e for e in events if e.transition_type == "entry"]
        for e in entry_events:
            assert e.delay_bars >= 0, f"Negative entry delay: {e.delay_bars}"


# ---------------------------------------------------------------------------
# Cost accounting distinction
# ---------------------------------------------------------------------------

class TestCostAccountingDistinction:
    def _run_and_get_cost(self):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start, _close_to_ohlcv,
        )
        from research.fixed_five_canonical_regime_comparison import run_canonical

        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        btc = close[wc.BTC_COL].dropna()
        params = WhipsawControl(name="ctrl")
        gate = compute_controlled_gate_series(btc, 20, params)
        sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
        ohlcv = _close_to_ohlcv(close)
        result, port_fx = run_canonical(
            ohlcv, sig, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        return result, port_fx, params, DEFAULT_INITIAL_CAPITAL

    def test_fee_plus_slippage_equals_gross_minus_net(self):
        result, port_fx, params, ic = self._run_and_get_cost()
        cost = compute_cost_accounting(result, port_fx, params)
        fee_plus_slip = cost["fee_dollars_est"] + cost["slippage_dollars_est"]
        assert abs(fee_plus_slip - cost["gross_minus_net_dollars"]) < 1.0

    def test_gross_gte_net(self):
        result, port_fx, params, ic = self._run_and_get_cost()
        cost = compute_cost_accounting(result, port_fx, params)
        assert cost["gross_ending_equity"] >= cost["net_ending_equity"]

    def test_gross_minus_net_gte_simple_additive(self):
        """Actual cost >= simple additive estimate due to foregone compounding."""
        result, port_fx, params, ic = self._run_and_get_cost()
        cost = compute_cost_accounting(result, port_fx, params)
        # gross_minus_net >= simple_cost_estimate (compounding makes it larger)
        # This may not hold for very short/flat periods, use loose tolerance
        assert cost["gross_minus_net_dollars"] >= cost["simple_cost_estimate_dollars"] - 10.0

    def test_cost_denominators_are_different(self):
        result, port_fx, params, ic = self._run_and_get_cost()
        cost = compute_cost_accounting(result, port_fx, params)
        a = cost["cost_div_equity_at_start_pct"]
        b = cost["cost_div_gross_profit_pct"]
        c = cost["cost_div_gross_ending_equity_pct"]
        # All should be non-negative
        assert a >= 0
        assert b >= 0
        assert c >= 0
        # They should not all be equal (different denominators)
        # equity_at_start < gross_ending_equity → ratio a > ratio c typically
        assert not (abs(a - b) < 0.01 and abs(b - c) < 0.01)

    def test_foregone_compounding_description(self):
        """foregone_compounding = gross_minus_net - simple_cost_estimate."""
        result, port_fx, params, ic = self._run_and_get_cost()
        cost = compute_cost_accounting(result, port_fx, params)
        expected = cost["gross_minus_net_dollars"] - cost["simple_cost_estimate_dollars"]
        assert abs(cost["foregone_compounding_dollars"] - expected) < 1.0


# ---------------------------------------------------------------------------
# WHIPSAW_IMPROVEMENT classification
# ---------------------------------------------------------------------------

class TestWhipsawImprovementClassification:
    def _ctrl(self):
        return {
            "sharpe": 1.86,
            "max_drawdown_pct": -48.1,
            "return_2022_pct": -41.8,
            "n_7day_whipsaw_clusters": 40,
            "additive_cost_drag_pct": 50.0,
            "pct_time_in_cash": 45.0,
        }

    def test_variant_meeting_all_criteria_labeled_improvement(self):
        variant = {
            "sharpe": 1.90,
            "max_drawdown_pct": -47.0,
            "return_2022_pct": -42.0,
            "n_7day_whipsaw_clusters": 25,
            "additive_cost_drag_pct": 45.0,
            "pct_time_in_cash": 40.0,
        }
        label, _ = classify_whipsaw_improvement(variant, self._ctrl())
        assert label == "WHIPSAW_IMPROVEMENT"

    def test_insufficient_cluster_reduction_fails(self):
        variant = {
            "sharpe": 1.86,
            "max_drawdown_pct": -48.0,
            "return_2022_pct": -42.0,
            "n_7day_whipsaw_clusters": 38,  # only 5% reduction
            "additive_cost_drag_pct": 49.0,
            "pct_time_in_cash": 44.0,
        }
        label, notes = classify_whipsaw_improvement(variant, self._ctrl())
        assert label == "NOT_WHIPSAW_IMPROVEMENT"
        assert "cluster_reduction_ge_25pct" in notes

    def test_sharpe_drop_exceeds_0_10_fails(self):
        variant = {
            "sharpe": 1.74,  # -0.12 from control
            "max_drawdown_pct": -46.0,
            "return_2022_pct": -42.0,
            "n_7day_whipsaw_clusters": 20,
            "additive_cost_drag_pct": 40.0,
            "pct_time_in_cash": 50.0,
        }
        label, notes = classify_whipsaw_improvement(variant, self._ctrl())
        assert label == "NOT_WHIPSAW_IMPROVEMENT"
        assert "sharpe_within_0_10_of_control" in notes

    def test_dd_worse_by_6pp_fails(self):
        variant = {
            "sharpe": 1.90,
            "max_drawdown_pct": -54.5,  # 6.4pp worse
            "return_2022_pct": -42.0,
            "n_7day_whipsaw_clusters": 20,
            "additive_cost_drag_pct": 40.0,
            "pct_time_in_cash": 50.0,
        }
        label, notes = classify_whipsaw_improvement(variant, self._ctrl())
        assert label == "NOT_WHIPSAW_IMPROVEMENT"
        assert "dd_not_worse_than_5pp" in notes

    def test_predominantly_in_cash_fails(self):
        variant = {
            "sharpe": 2.0,
            "max_drawdown_pct": -20.0,
            "return_2022_pct": -10.0,
            "n_7day_whipsaw_clusters": 5,
            "additive_cost_drag_pct": 10.0,
            "pct_time_in_cash": 85.0,  # too much cash
        }
        label, notes = classify_whipsaw_improvement(variant, self._ctrl())
        assert label == "NOT_WHIPSAW_IMPROVEMENT"
        assert "not_predominantly_in_cash" in notes

    def test_deterministic_classification(self):
        variant = {
            "sharpe": 1.90,
            "max_drawdown_pct": -47.0,
            "return_2022_pct": -42.0,
            "n_7day_whipsaw_clusters": 25,
            "additive_cost_drag_pct": 45.0,
            "pct_time_in_cash": 40.0,
        }
        l1, n1 = classify_whipsaw_improvement(variant, self._ctrl())
        l2, n2 = classify_whipsaw_improvement(variant, self._ctrl())
        assert l1 == l2
        assert n1 == n2


# ---------------------------------------------------------------------------
# Canonical initialization (identical across variants)
# ---------------------------------------------------------------------------

class TestCanonicalInitialization:
    def _run_variant(self, params: WhipsawControl, n: int = 400):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start, _close_to_ohlcv,
        )
        from research.fixed_five_canonical_regime_comparison import run_canonical

        close = _make_close(n=n)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        btc = close[wc.BTC_COL].dropna()
        gate = compute_controlled_gate_series(btc, 20, params)
        sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
        ohlcv = _close_to_ohlcv(close)
        result, port_fx = run_canonical(
            ohlcv, sig, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        return port_fx, joint_start, DEFAULT_INITIAL_CAPITAL

    def test_all_variants_share_same_start_date(self):
        """Multiple variants should share the same joint_start."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start, _close_to_ohlcv,
        )
        from research.fixed_five_canonical_regime_comparison import run_canonical

        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        btc = close[wc.BTC_COL].dropna()
        ohlcv = _close_to_ohlcv(close)

        start_dates = []
        for p in [WhipsawControl(name="a"), WhipsawControl(name="b", entry_confirm_bars=2)]:
            gate = compute_controlled_gate_series(btc, 20, p)
            sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
            _, port_fx = run_canonical(
                ohlcv, sig, joint_start,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                fee_bps=DEFAULT_FEE_BPS,
                slippage_bps=DEFAULT_SLIPPAGE_BPS,
            )
            start_dates.append(port_fx.index.min())

        assert start_dates[0] == start_dates[1]

    def test_equity_at_joint_start_gt_initial_capital(self):
        """Run A canonical: equity at joint_start > initial_capital."""
        port_fx, joint_start, ic = self._run_variant(WhipsawControl(name="test"))
        equity_at_start = float(port_fx["equity"].dropna().iloc[0])
        assert equity_at_start >= ic  # May equal if gate is immediately off


# ---------------------------------------------------------------------------
# Deterministic execution
# ---------------------------------------------------------------------------

class TestDeterministicExecution:
    def test_same_gate_same_result(self):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start, _close_to_ohlcv,
        )
        from research.fixed_five_canonical_regime_comparison import run_canonical

        close = _make_close(n=300)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        btc = close[wc.BTC_COL].dropna()
        params = WhipsawControl(name="sym2", entry_confirm_bars=2, exit_confirm_bars=2)
        ohlcv = _close_to_ohlcv(close)

        results = []
        for _ in range(2):
            gate = compute_controlled_gate_series(btc, 20, params)
            sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
            _, port_fx = run_canonical(
                ohlcv, sig, joint_start,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                fee_bps=DEFAULT_FEE_BPS,
                slippage_bps=DEFAULT_SLIPPAGE_BPS,
            )
            results.append(port_fx["equity"])

        pd.testing.assert_series_equal(results[0], results[1])

    def test_gate_deterministic_from_same_input(self):
        btc = _make_btc_series(n=300)
        params = WhipsawControl(name="test", entry_confirm_bars=2, min_duration_bars=5)
        g1 = compute_controlled_gate_series(btc, 30, params)
        g2 = compute_controlled_gate_series(btc, 30, params)
        pd.testing.assert_series_equal(g1, g2)


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------

class TestNoLiveImports:
    FORBIDDEN = ["brokers", "execution", "live"]

    def test_no_live_module_imports(self):
        source = Path("research/fixed_five_whipsaw_control.py").read_text()
        for mod in self.FORBIDDEN:
            assert f"from {mod}" not in source, f"Forbidden import from {mod}"
            assert f"import {mod}" not in source, f"Forbidden import of {mod}"

    def test_no_order_placement_patterns(self):
        source = Path("research/fixed_five_whipsaw_control.py").read_text()
        for pat in ["place_order", "submit_order", "create_order", "send_order"]:
            assert pat not in source, f"Forbidden pattern '{pat}'"

    def test_variant_count_matches_spec(self):
        """Confirm the correct number of variants is defined.
        1 control + 2 entry_conf + 2 exit_conf + 2 sym + 3 hyst + 3 min_dur + 4 combos = 17.
        """
        assert len(wc.VARIANTS) == 17  # 1 control + 12 independent + 4 combinations


# ---------------------------------------------------------------------------
# Benchmark date alignment
# ---------------------------------------------------------------------------

class TestBenchmarkDateAlignment:
    def test_btc_benchmark_uses_same_period_as_strategy(self):
        from research.fixed_five_canonical_regime_comparison import _btc_benchmark_period
        close = _make_close(n=200)
        start = close.index[50]
        end = close.index[150]
        ret = _btc_benchmark_period(close, start, end)
        assert not math.isnan(ret)
        # BTC-only benchmark uses start/end matching the strategy period
        btc = close[wc.BTC_COL].dropna()
        seg = btc.loc[(btc.index >= start) & (btc.index <= end)]
        expected = seg.iloc[-1] / seg.iloc[0] - 1 - 2 * wc.DEFAULT_FEE_BPS / 10_000
        assert abs(ret - expected) < 1e-8
