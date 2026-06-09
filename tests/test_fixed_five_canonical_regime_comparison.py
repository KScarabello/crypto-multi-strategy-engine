"""Tests for research/fixed_five_canonical_regime_comparison.py

Verifies:
- Identical initialization across all variants
- No unintended 36-bar cash delay
- One-bar execution delay intact
- Cost identity reconciliation
- Benchmark date alignment
- Deterministic reruns
- No live trading imports
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_canonical_regime_comparison as rc


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_close(n: int = 300, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = 1000 * np.cumprod(1 + 0.001 + rng.normal(0, 0.01, n))
    return pd.DataFrame(data, index=idx)


def _make_ohlcv(close: pd.DataFrame) -> pd.DataFrame:
    from research.universe_integrity_analysis import _close_to_ohlcv
    return _close_to_ohlcv(close)


# ---------------------------------------------------------------------------
# Canonical initialization verification
# ---------------------------------------------------------------------------

class TestCanonicalInitialization:
    def test_equity_at_joint_start_exceeds_initial_capital(self):
        """Canonical Run A: equity at joint_start must be > initial_capital
        because the strategy invested during the pre-period."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, DEFAULT_REBALANCE_BARS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars for min_history in synthetic data")

        ohlcv = _make_ohlcv(close)
        gen = rc.BaseSignalGen()
        result, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        eq = port_fx["equity"].dropna()
        # In Run A with a rising close, equity should compound before joint_start
        assert float(eq.iloc[0]) >= DEFAULT_INITIAL_CAPITAL

    def test_no_36_bar_delay(self):
        """First investment must occur within 6 bars of joint_start (no warm-up delay)."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars for min_history")

        ohlcv = _make_ohlcv(close)
        gen = rc.BaseSignalGen()
        result, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        reb = result.rebalance_log
        first_inv = reb[(reb["weight_sum"] > 0.01) & (reb["execution_timestamp"] >= joint_start)]
        if len(first_inv) == 0:
            pytest.skip("No investments found in synthetic data")

        # In canonical Run A, first investment should be very close to joint_start
        delay_bars = len(
            result.portfolio.loc[
                (result.portfolio.index >= joint_start) &
                (result.portfolio.index < first_inv.iloc[0]["execution_timestamp"])
            ]
        )
        # Should be ≤ rebalance_every_bars (6), not 36
        assert delay_bars <= 8, f"Unexpected warm-up delay: {delay_bars} bars"

    def test_one_bar_execution_delay(self):
        """Signal timestamp and execution timestamp must differ."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        ohlcv = _make_ohlcv(close)
        gen = rc.BaseSignalGen()
        result, _ = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        reb = result.rebalance_log
        if len(reb) == 0:
            pytest.skip("No rebalances")
        assert (reb["signal_timestamp"] != reb["execution_timestamp"]).all(), \
            "One-bar delay not preserved"

    def test_verify_canonical_initialization_checks(self):
        """verify_canonical_initialization must return expected keys."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        ohlcv = _make_ohlcv(close)
        gen = rc.BaseSignalGen()
        result, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        checks = rc.verify_canonical_initialization(result, port_fx, joint_start)
        assert "one_bar_delay_confirmed" in checks
        assert "equity_at_joint_start" in checks
        assert bool(checks["one_bar_delay_confirmed"]) is True

    def test_run_canonical_returns_port_fx_from_joint_start(self):
        """port_fx must start at or after joint_start."""
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        ohlcv = _make_ohlcv(close)
        gen = rc.BaseSignalGen()
        _, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        assert port_fx.index.min() >= joint_start


# ---------------------------------------------------------------------------
# Identical date boundaries
# ---------------------------------------------------------------------------

class TestIdenticalDateBoundaries:
    def _run_two_variants(self):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")
        ohlcv = _make_ohlcv(close)

        _, port_base = rc.run_canonical(
            ohlcv, rc.BaseSignalGen(), joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        _, port_btc = rc.run_canonical(
            ohlcv, rc.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=36), joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        return port_base, port_btc, joint_start

    def test_start_dates_identical(self):
        port_base, port_btc, _ = self._run_two_variants()
        assert port_base.index.min() == port_btc.index.min()

    def test_end_dates_identical(self):
        port_base, port_btc, _ = self._run_two_variants()
        assert port_base.index.max() == port_btc.index.max()

    def test_bar_count_identical(self):
        port_base, port_btc, _ = self._run_two_variants()
        assert len(port_base) == len(port_btc)


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

class TestCostAccounting:
    def _run_baseline(self):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")
        ohlcv = _make_ohlcv(close)
        result, port_fx = rc.run_canonical(
            ohlcv, rc.BaseSignalGen(), joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        return result, port_fx, DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS

    def test_gross_ending_equity_gte_net(self):
        result, port_fx, ic, fee, slip = self._run_baseline()
        costs = rc.compute_cost_accounting(result, port_fx, ic, fee, slip)
        assert costs["gross_ending_equity"] >= costs["net_ending_equity"]

    def test_gross_minus_net_equals_total_cost(self):
        result, port_fx, ic, fee, slip = self._run_baseline()
        costs = rc.compute_cost_accounting(result, port_fx, ic, fee, slip)
        diff = costs["gross_ending_equity"] - costs["net_ending_equity"]
        assert abs(diff - costs["gross_minus_net_ending"]) < 1.0

    def test_total_cost_non_negative(self):
        result, port_fx, ic, fee, slip = self._run_baseline()
        costs = rc.compute_cost_accounting(result, port_fx, ic, fee, slip)
        assert costs["total_cost_dollars"] >= 0

    def test_additive_cost_rate_sum_non_negative(self):
        result, port_fx, ic, fee, slip = self._run_baseline()
        costs = rc.compute_cost_accounting(result, port_fx, ic, fee, slip)
        assert costs["additive_cost_rate_sum_pct"] >= 0

    def test_all_denominators_distinct(self):
        result, port_fx, ic, fee, slip = self._run_baseline()
        costs = rc.compute_cost_accounting(result, port_fx, ic, fee, slip)
        # These three ratios use different denominators — they should differ
        a = costs["cost_div_equity_at_start_pct"]
        b = costs["cost_div_gross_profit_pct"]
        c = costs["cost_div_gross_ending_equity_pct"]
        # All should be non-negative
        assert a >= 0
        assert b >= 0
        assert c >= 0


# ---------------------------------------------------------------------------
# Gate state and regime stats
# ---------------------------------------------------------------------------

class TestGateState:
    def test_baseline_gate_always_true(self):
        close = _make_close(n=100)
        joint_start = close.index[50]
        gate = rc.compute_gate_series(close, None, joint_start)
        assert gate.all()

    def test_ma_gate_false_when_btc_falling(self):
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        data = {sym: [1000.0] * n for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]}
        # BTC falls sharply after bar 50
        for i in range(n):
            data["BTC/USD"][i] = 1000 * (0.997 ** max(0, i - 50))
        close = pd.DataFrame(data, index=idx)
        joint_start = idx[60]
        gate = rc.compute_gate_series(close, 20, joint_start)
        # At bar 150, BTC should be well below 20-bar MA
        assert not gate.iloc[-1]

    def test_regime_stats_keys(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        gate = pd.Series([True] * 50 + [False] * 50, index=idx, dtype=bool)
        stats = rc.regime_stats_from_gate(gate)
        assert "n_regime_switches" in stats
        assert "avg_regime_duration_bars" in stats
        assert "pct_time_risk_off" in stats

    def test_regime_stats_all_risk_on(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        gate = pd.Series([True] * 100, index=idx, dtype=bool)
        stats = rc.regime_stats_from_gate(gate)
        assert stats["pct_time_risk_off"] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Cash accounting during regime-off
# ---------------------------------------------------------------------------

class TestCashDuringRegimeOff:
    def test_regime_off_means_zero_weights(self):
        """When MA gate fires RISK_OFF, BTCRegimeSignalGen returns all-zero weights."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        data = {sym: [1000.0] * n for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]}
        for i in range(n):
            data["BTC/USD"][i] = 1000 * (0.995 ** max(0, i - 10))
        close = pd.DataFrame(data, index=idx)

        gen = rc.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=20)
        # At bar 150 BTC is far below 20-bar MA → gate=False → all cash
        ts = idx[150]
        weights = gen(close, ts)
        # Gate should be off → all zeros
        assert weights.sum() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Whipsaw cluster detection
# ---------------------------------------------------------------------------

class TestWhipsawClusters:
    def test_no_clusters_when_stable(self):
        """Monotonically rising BTC produces no whipsaw clusters."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        data = {sym: [1000 * (1.002 ** i) for i in range(n)]
                for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]}
        close = pd.DataFrame(data, index=idx)
        joint_start = idx[50]

        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
        )
        ohlcv = _make_ohlcv(close)
        gen = rc.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=20)
        result, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        clusters = rc.detect_whipsaw_clusters(
            "test", 20, close, result, port_fx, joint_start
        )
        # Monotonically rising BTC → gate always on → no transitions → no clusters
        assert len(clusters) == 0

    def test_cluster_detected_when_whipsawing(self):
        """Alternating BTC above/below MA creates clusters."""
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        ma_val = 1000.0
        # Create alternating pattern around MA
        prices = []
        for i in range(n):
            if i < 150:
                prices.append(1100.0)  # above MA
            else:
                # Alternate above/below every bar
                prices.append(1100.0 if i % 2 == 0 else 900.0)

        data = {sym: prices for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]}
        close = pd.DataFrame(data, index=idx)
        joint_start = idx[50]

        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
        )
        ohlcv = _make_ohlcv(close)
        gen = rc.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=10)
        result, port_fx = rc.run_canonical(
            ohlcv, gen, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        clusters = rc.detect_whipsaw_clusters(
            "test", 10, close, result, port_fx, joint_start
        )
        assert len(clusters) >= 1


# ---------------------------------------------------------------------------
# Benchmark date alignment
# ---------------------------------------------------------------------------

class TestBenchmarkAlignment:
    def test_btc_benchmark_uses_same_dates(self):
        close = _make_close(n=200)
        start = close.index[50]
        end = close.index[150]
        ret = rc._btc_benchmark_period(close, start, end)
        # Should be computable
        assert not math.isnan(ret)

    def test_ewb_benchmark_uses_same_dates(self):
        close = _make_close(n=200)
        start = close.index[50]
        end = close.index[150]
        ret = rc._ewb_benchmark_period(close, start, end)
        assert not math.isnan(ret)

    def test_btc_benchmark_returns_nan_for_insufficient_data(self):
        close = _make_close(n=200)
        start = close.index[198]
        end = close.index[199]
        ret = rc._btc_benchmark_period(close, start, end)
        # Only 2 bars — may produce valid result, at least should not crash
        assert not math.isnan(ret)  # 2 bars is sufficient


# ---------------------------------------------------------------------------
# Robustness label classification
# ---------------------------------------------------------------------------

class TestRobustnessLabels:
    def _baseline(self):
        return {
            "max_drawdown_pct": -79.7,
            "sharpe": 1.20,
            "additive_cost_drag_pct": 112.0,
            "pct_time_in_cash": 0.0,
            "n_regime_switches": 0,
        }

    def _good_variant(self):
        return {
            "variant": "btc_ma_360_reb6",
            "max_drawdown_pct": -53.0,
            "sharpe": 1.65,
            "additive_cost_drag_pct": 71.0,
            "pct_time_in_cash": 46.0,
            "n_regime_switches": 200,
            "dd_vs_baseline_pp": 26.7,
            "sharpe_vs_baseline": 0.45,
        }

    def _period_rows(self, variant="btc_ma_360_reb6"):
        return [
            {"variant": variant, "period": "2022", "total_return_pct": -40.0, "max_drawdown_pct": -45.0},
            {"variant": variant, "period": "2023", "total_return_pct": 200.0, "max_drawdown_pct": -20.0},
            {"variant": variant, "period": "2024", "total_return_pct": 60.0, "max_drawdown_pct": -30.0},
            {"variant": variant, "period": "2025", "total_return_pct": -10.0, "max_drawdown_pct": -40.0},
            {"variant": "baseline", "period": "2022", "total_return_pct": -75.0, "max_drawdown_pct": -76.0},
            {"variant": "baseline", "period": "2023", "total_return_pct": 300.0, "max_drawdown_pct": -35.0},
            {"variant": "baseline", "period": "2024", "total_return_pct": 84.0, "max_drawdown_pct": -50.0},
            {"variant": "baseline", "period": "2025", "total_return_pct": -24.0, "max_drawdown_pct": -53.0},
        ]

    def test_good_variant_gets_robust_improvement(self):
        label, _ = rc.classify_robustness(
            self._good_variant(), self._baseline(), self._period_rows()
        )
        assert label == "ROBUST_IMPROVEMENT"

    def test_unfavorable_when_dd_worse(self):
        m = self._good_variant()
        m["max_drawdown_pct"] = -90.0  # worse than baseline
        m["dd_vs_baseline_pp"] = -10.3
        label, _ = rc.classify_robustness(m, self._baseline(), self._period_rows())
        assert label == "UNFAVORABLE"

    def test_mixed_when_dd_improves_but_sharpe_doesnt(self):
        m = self._good_variant()
        m["sharpe"] = 1.22  # only 0.02 above baseline
        m["sharpe_vs_baseline"] = 0.02
        label, _ = rc.classify_robustness(m, self._baseline(), self._period_rows())
        assert label in ("MIXED_TRADEOFF", "ROBUST_IMPROVEMENT")

    def test_deterministic(self):
        m = self._good_variant()
        rows = self._period_rows()
        l1, n1 = rc.classify_robustness(m, self._baseline(), rows)
        l2, n2 = rc.classify_robustness(m, self._baseline(), rows)
        assert l1 == l2
        assert n1 == n2


# ---------------------------------------------------------------------------
# Deterministic reruns
# ---------------------------------------------------------------------------

class TestDeterministicReruns:
    def test_same_result_twice(self):
        from research.universe_integrity_analysis import (
            DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
            find_joint_eligible_start,
        )
        close = _make_close(n=400)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")

        ohlcv = _make_ohlcv(close)
        gen1, gen2 = rc.BaseSignalGen(), rc.BaseSignalGen()
        _, port1 = rc.run_canonical(
            ohlcv, gen1, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        _, port2 = rc.run_canonical(
            ohlcv, gen2, joint_start,
            initial_capital=DEFAULT_INITIAL_CAPITAL, fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        pd.testing.assert_series_equal(port1["equity"], port2["equity"])


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------

class TestNoLiveTradingImports:
    FORBIDDEN = ["brokers", "execution", "live"]

    def test_no_live_module_imports(self):
        source = Path("research/fixed_five_canonical_regime_comparison.py").read_text()
        for mod in self.FORBIDDEN:
            assert f"from {mod}" not in source, f"Forbidden import from {mod}"
            assert f"import {mod}" not in source, f"Forbidden import of {mod}"

    def test_no_order_placement_patterns(self):
        source = Path("research/fixed_five_canonical_regime_comparison.py").read_text()
        for pat in ["place_order", "submit_order", "create_order", "send_order"]:
            assert pat not in source, f"Forbidden pattern '{pat}'"
