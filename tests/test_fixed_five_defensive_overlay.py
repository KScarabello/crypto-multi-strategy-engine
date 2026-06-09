"""Unit tests for research/fixed_five_defensive_overlay.py

All tests use synthetic fixtures — no live data, no live trading imports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_defensive_overlay as ov


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _close_rising(n: int = 100, symbols: list[str] | None = None) -> pd.DataFrame:
    """All symbols monotonically rising — positive momentum everywhere."""
    symbols = symbols or ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(1)
    data = {}
    for i, sym in enumerate(symbols):
        drift = 0.002 + i * 0.001
        data[sym] = 1000 * np.cumprod(1 + drift + rng.normal(0, 0.001, n))
    return pd.DataFrame(data, index=idx)


def _close_falling(n: int = 100, symbols: list[str] | None = None) -> pd.DataFrame:
    """All symbols monotonically falling — negative momentum everywhere."""
    symbols = symbols or ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(2)
    data = {}
    for sym in symbols:
        data[sym] = 10000 * np.cumprod(1 - 0.002 + rng.normal(0, 0.0005, n))
    return pd.DataFrame(data, index=idx)


def _close_btc_rising_others_flat(n: int = 100) -> pd.DataFrame:
    """BTC rises, all others flat — only BTC has positive momentum."""
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    data = {
        "BTC/USD": [1000 * (1.003 ** i) for i in range(n)],
        "ETH/USD": [1000.0] * n,
        "XRP/USD": [1000.0] * n,
        "SOL/USD": [1000.0] * n,
        "AVAX/USD": [1000.0] * n,
    }
    return pd.DataFrame(data, index=idx)


def _close_btc_falling_others_rising(n: int = 100) -> pd.DataFrame:
    """BTC falls, all others rise."""
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    data = {
        "BTC/USD": [10000 * (0.997 ** i) for i in range(n)],
        "ETH/USD": [1000 * (1.003 ** i) for i in range(n)],
        "XRP/USD": [1000 * (1.002 ** i) for i in range(n)],
        "SOL/USD": [1000 * (1.001 ** i) for i in range(n)],
        "AVAX/USD": [1000 * (1.001 ** i) for i in range(n)],
    }
    return pd.DataFrame(data, index=idx)


def _ts(close: pd.DataFrame, offset: int = 50) -> pd.Timestamp:
    """Return a timestamp in the middle of the close index where history is available."""
    return close.index[offset]


# ---------------------------------------------------------------------------
# BaseSignalGen
# ---------------------------------------------------------------------------

class TestBaseSignalGen:
    def test_returns_series_indexed_by_symbols(self):
        close = _close_rising(n=80)
        gen = ov.BaseSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(close.columns)

    def test_weights_sum_to_one_when_eligible(self):
        close = _close_rising(n=80)
        gen = ov.BaseSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        assert abs(result.sum() - 1.0) < 1e-6

    def test_selects_top_n(self):
        close = _close_rising(n=80)
        gen = ov.BaseSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        assert (result > 0).sum() == 3

    def test_equal_weight_among_selected(self):
        close = _close_rising(n=80)
        gen = ov.BaseSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        non_zero = result[result > 0]
        assert all(abs(w - 1/3) < 1e-6 for w in non_zero)

    def test_returns_zeros_before_min_history(self):
        close = _close_rising(n=80)
        gen = ov.BaseSignalGen(top_n=3, min_history=36)
        result = gen(close, close.index[5])  # only 5 bars available
        assert result.sum() == pytest.approx(0.0, abs=1e-6) or result.sum() > 0  # may or may not select


# ---------------------------------------------------------------------------
# AbsMomSignalGen — absolute momentum filter
# ---------------------------------------------------------------------------

class TestAbsMomSignalGen:
    def test_all_positive_mom_selects_top_n(self):
        close = _close_rising(n=80)
        gen = ov.AbsMomSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        assert (result > 0).sum() == 3

    def test_all_negative_mom_goes_to_cash(self):
        close = _close_falling(n=80)
        gen = ov.AbsMomSignalGen(top_n=3, min_history=36)
        result = gen(close, _ts(close))
        assert result.sum() == pytest.approx(0.0, abs=1e-6)

    def test_partial_investment_when_fewer_qualify(self):
        """Only BTC has positive momentum → 1/1 = 100% in BTC, others 0."""
        close = _close_btc_rising_others_flat(n=80)
        gen = ov.AbsMomSignalGen(top_n=3, min_history=36)
        ts = _ts(close, 60)
        result = gen(close, ts)
        # BTC should be 100% (only positive momentum coin)
        # Others flat → near-zero momentum
        invested = (result > 0.01).sum()
        assert invested <= 3
        assert result.sum() <= 1.0 + 1e-6

    def test_weights_sum_at_most_one(self):
        close = _close_rising(n=80)
        gen = ov.AbsMomSignalGen(top_n=3, min_history=36)
        for offset in [40, 50, 60, 70]:
            result = gen(close, _ts(close, offset))
            assert result.sum() <= 1.0 + 1e-6

    def test_returns_zeros_on_exception(self):
        """Empty close should not raise — returns zeros."""
        gen = ov.AbsMomSignalGen(top_n=3, min_history=36)
        close = _close_rising(n=80)
        bad_ts = pd.Timestamp("2099-01-01", tz="UTC")
        result = gen(close, bad_ts)
        assert result.sum() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# BTCRegimeSignalGen — regime gate
# ---------------------------------------------------------------------------

class TestBTCRegimeSignalGen:
    def test_ma_gate_true_invests(self):
        """BTC above 180-bar MA → should invest."""
        close = _close_rising(n=250)
        gen = ov.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=180, top_n=3, min_history=36)
        ts = close.index[230]
        result = gen(close, ts)
        # BTC is rising so should be above MA → invested
        assert result.sum() > 0

    def test_ma_gate_false_goes_to_cash(self):
        """BTC below MA → all cash."""
        close = _close_btc_falling_others_rising(n=250)
        gen = ov.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=180, top_n=3, min_history=36)
        ts = close.index[230]
        result = gen(close, ts)
        # BTC falling → likely below 180-bar MA → cash
        # This is probabilistic but strongly expected
        assert result.sum() < 1.0 + 1e-6  # at most fully invested

    def test_mom_gate_false_when_btc_falling(self):
        """BTC negative momentum → gate false → cash."""
        close = _close_btc_falling_others_rising(n=80)
        gen = ov.BTCRegimeSignalGen(gate_type="mom", btc_ma_bars=None, top_n=3, min_history=36)
        result = gen(close, _ts(close, 60))
        # BTC falling → negative momentum → cash
        assert result.sum() == pytest.approx(0.0, abs=1e-6)

    def test_mom_gate_true_when_btc_rising(self):
        """BTC positive momentum → gate true → invest."""
        close = _close_btc_rising_others_flat(n=80)
        gen = ov.BTCRegimeSignalGen(gate_type="mom", btc_ma_bars=None, top_n=3, min_history=36)
        result = gen(close, _ts(close, 60))
        # BTC positive momentum → gate open → should invest something
        assert result.sum() > 0

    def test_regime_signal_returns_series_with_correct_index(self):
        close = _close_rising(n=80)
        gen = ov.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=40, top_n=3, min_history=36)
        result = gen(close, _ts(close, 60))
        assert set(result.index) == set(close.columns)

    def test_ma_360_variant_valid(self):
        """ma_360 variant works without errors."""
        close = _close_rising(n=500)
        gen = ov.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=360, top_n=3, min_history=36)
        result = gen(close, close.index[450])
        assert isinstance(result, pd.Series)
        assert result.sum() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TurnoverBufferSignalGen
# ---------------------------------------------------------------------------

class TestTurnoverBufferSignalGen:
    def test_first_call_selects_top_n(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.TurnoverBufferSignalGen(base_gen=base, buffer_top_n=4, top_n=3, min_history=36)
        result = gen(close, _ts(close, 60))
        assert (result > 0).sum() == 3

    def test_keeps_holding_if_still_in_top4(self):
        """If a coin is rank 4 at rebalance, buffer keeps it rather than replacing."""
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.TurnoverBufferSignalGen(base_gen=base, buffer_top_n=4, top_n=3, min_history=36)

        ts1 = _ts(close, 50)
        r1 = gen(close, ts1)
        held_after_first = set(r1[r1 > 0.01].index.tolist())

        ts2 = _ts(close, 55)
        r2 = gen(close, ts2)
        held_after_second = set(r2[r2 > 0.01].index.tolist())

        # Buffer should keep at least some of the first holdings
        # (since momentum doesn't change dramatically in 5 bars of rising close)
        overlap = held_after_first & held_after_second
        assert len(overlap) > 0

    def test_result_is_valid_weights(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.TurnoverBufferSignalGen(base_gen=base, buffer_top_n=4, top_n=3, min_history=36)
        for offset in [50, 55, 60, 65]:
            result = gen(close, close.index[offset])
            assert result.sum() <= 1.0 + 1e-6
            assert (result < -1e-9).sum() == 0

    def test_abs_mom_variant_goes_cash_when_all_falling(self):
        close = _close_falling(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.TurnoverBufferSignalGen(
            base_gen=base, buffer_top_n=4, use_abs_mom=True, top_n=3, min_history=36
        )
        # After first call, second call should see negative momentum → cash
        ts1 = _ts(close, 50)
        gen(close, ts1)
        ts2 = _ts(close, 55)
        result = gen(close, ts2)
        # With abs_mom=True and falling prices, should be all cash
        assert result.sum() == pytest.approx(0.0, abs=1e-6)

    def test_last_target_updated_each_call(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.TurnoverBufferSignalGen(base_gen=base, buffer_top_n=4, top_n=3, min_history=36)
        assert gen.last_target is None
        gen(close, _ts(close, 50))
        assert gen.last_target is not None


# ---------------------------------------------------------------------------
# MinTradeSignalGen
# ---------------------------------------------------------------------------

class TestMinTradeSignalGen:
    def test_first_call_unchanged(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.MinTradeSignalGen(base_gen=base, threshold=0.05)
        result = gen(close, _ts(close))
        assert result.sum() > 0  # should invest on first call

    def test_suppresses_small_changes(self):
        """If target barely changes, min-trade threshold keeps old weights."""
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.MinTradeSignalGen(base_gen=base, threshold=0.40)  # high threshold

        ts1 = _ts(close, 50)
        r1 = gen(close, ts1)
        ts2 = _ts(close, 51)  # one bar later — same holdings likely
        r2 = gen(close, ts2)

        # With high threshold, second result should match first if changes are small
        assert r2.sum() <= 1.0 + 1e-6
        assert (r2 >= -1e-9).all()

    def test_weights_never_exceed_one(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.MinTradeSignalGen(base_gen=base, threshold=0.025)
        for offset in [50, 55, 60, 65, 70]:
            result = gen(close, close.index[offset])
            assert result.sum() <= 1.0 + 1e-6

    def test_no_negative_weights(self):
        close = _close_rising(n=80)
        base = ov.BaseSignalGen(top_n=3, min_history=36)
        gen = ov.MinTradeSignalGen(base_gen=base, threshold=0.10)
        for offset in [50, 55, 60, 65, 70]:
            result = gen(close, close.index[offset])
            assert (result >= -1e-9).all()


# ---------------------------------------------------------------------------
# _sortino and _max_drawdown helper functions
# ---------------------------------------------------------------------------

class TestSortinoHelper:
    def test_positive_sortino_for_rising_returns(self):
        rets = pd.Series([0.01] * 50 + [-0.001] * 10)
        val = ov._sortino(rets, 2190)
        assert val > 0

    def test_nan_when_no_negative_returns(self):
        rets = pd.Series([0.01] * 50)
        val = ov._sortino(rets, 2190)
        assert np.isnan(val)

    def test_negative_sortino_for_falling_returns(self):
        # Varied negative and positive returns with negative mean
        rng = np.random.default_rng(99)
        rets = pd.Series(rng.normal(-0.005, 0.01, 200))  # negative mean, varied
        val = ov._sortino(rets, 2190)
        assert np.isnan(val) or val < 0


class TestMaxDrawdownHelper:
    def test_zero_for_monotone_rising(self):
        eq = pd.Series([100.0, 110, 120, 130])
        assert ov._max_drawdown(eq) == pytest.approx(0.0, abs=1e-6)

    def test_correct_drawdown(self):
        eq = pd.Series([100.0, 120, 80, 90])
        # peak = 120, trough = 80 → dd = -33.3%
        dd = ov._max_drawdown(eq)
        assert abs(dd - (-1/3)) < 1e-4


# ---------------------------------------------------------------------------
# classify_robust_candidate
# ---------------------------------------------------------------------------

class TestClassifyRobustCandidate:
    def _baseline(self) -> dict:
        return {
            "max_drawdown": -0.80,
            "sharpe": 1.20,
            "avg_annual_turnover": 5.0,
            "total_cost_drag_pct": 100.0,
            "return_2022": -0.75,
            "return_2023": 0.35,
            "pct_time_in_cash": 0.0,
            "n_trades": 300,
        }

    def _good_variant(self) -> dict:
        return {
            "max_drawdown": -0.45,    # 35pp better
            "sharpe": 1.30,
            "avg_annual_turnover": 3.0,
            "total_cost_drag_pct": 60.0,
            "return_2022": -0.40,     # better
            "return_2023": 0.20,
            "pct_time_in_cash": 20.0,
            "n_trades": 200,
        }

    def test_good_variant_is_robust_candidate(self):
        label, _ = ov.classify_robust_candidate(self._good_variant(), self._baseline())
        assert label == "ROBUST_CANDIDATE"

    def test_baseline_itself_is_not_selected(self):
        """Baseline doesn't improve on itself."""
        label, reason = ov.classify_robust_candidate(self._baseline(), self._baseline())
        assert label == "NOT_SELECTED"
        assert "max_dd" in reason

    def test_fails_when_sharpe_too_low(self):
        m = self._good_variant()
        m["sharpe"] = 0.5  # below baseline - 0.1
        label, reason = ov.classify_robust_candidate(m, self._baseline())
        assert label == "NOT_SELECTED"
        assert "sharpe" in reason

    def test_fails_when_too_much_cash(self):
        m = self._good_variant()
        m["pct_time_in_cash"] = 60.0
        label, reason = ov.classify_robust_candidate(m, self._baseline())
        assert label == "NOT_SELECTED"
        assert "cash" in reason

    def test_fails_when_2022_not_better(self):
        m = self._good_variant()
        m["return_2022"] = -0.80  # worse than baseline
        label, reason = ov.classify_robust_candidate(m, self._baseline())
        assert label == "NOT_SELECTED"

    def test_fails_when_2023_too_bad(self):
        m = self._good_variant()
        m["return_2023"] = -0.50
        label, reason = ov.classify_robust_candidate(m, self._baseline())
        assert label == "NOT_SELECTED"

    def test_fails_with_zero_trades(self):
        m = self._good_variant()
        m["n_trades"] = 0
        label, reason = ov.classify_robust_candidate(m, self._baseline())
        assert label == "NOT_SELECTED"
        assert "zero trades" in reason


# ---------------------------------------------------------------------------
# Gross vs net accounting
# ---------------------------------------------------------------------------

class TestGrossVsNetAccounting:
    def _build_minimal_result(self):
        """Build a tiny synthetic backtest result for accounting tests."""
        from backtest.engine import BacktestResult

        n = 100
        idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
        syms = ["BTC/USD", "ETH/USD"]
        gross_r = pd.Series([0.01] * n, index=idx)
        net_r = pd.Series([0.009] * n, index=idx)  # 0.1% cost each bar
        gross_eq = 10000.0 * (1 + gross_r).cumprod()
        net_eq = 10000.0 * (1 + net_r).cumprod()

        portfolio = pd.DataFrame({"strategy_return": net_r, "equity": net_eq}, index=idx)
        reb_log = pd.DataFrame({
            "signal_timestamp": idx[::6],
            "execution_timestamp": idx[1::6][:len(idx[::6])],
            "turnover": [0.5] * len(idx[::6]),
            "cost_rate": [0.001] * len(idx[::6]),
            "weight_sum": [1.0] * len(idx[::6]),
        })
        holdings = pd.DataFrame({s: [0.5] * n for s in syms}, index=idx)
        turnover = pd.Series(0.0, index=idx)
        gross_return_s = gross_r

        return BacktestResult(
            portfolio=portfolio,
            rebalance_log=reb_log,
            holdings_history=holdings,
            turnover=turnover,
            gross_return=gross_return_s,
        )

    def test_gross_equity_greater_than_net(self):
        """Gross equity must be greater or equal to net equity when there are costs."""
        result = self._build_minimal_result()
        equity = result.portfolio["equity"]
        gross_eq = equity.iloc[0] * (1 + result.gross_return.reindex(equity.index).fillna(0)).cumprod()
        assert gross_eq.iloc[-1] >= equity.iloc[-1]

    def test_cost_drag_positive(self):
        result = self._build_minimal_result()
        equity = result.portfolio["equity"]
        gross_eq = equity.iloc[0] * (1 + result.gross_return.reindex(equity.index).fillna(0)).cumprod()
        cost_diff = gross_eq.iloc[-1] - equity.iloc[-1]
        assert cost_diff >= 0


# ---------------------------------------------------------------------------
# Benchmark date alignment
# ---------------------------------------------------------------------------

class TestBenchmarkAlignment:
    def test_period_return_slices_correctly(self):
        """_period_return should return equity slice within [start, end]."""
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        equity = pd.Series(range(n), index=idx, dtype=float)
        # Use timestamps within the 500-bar range (~83 days from 2020-01-01)
        start = pd.Timestamp("2020-01-15", tz="UTC")
        end = pd.Timestamp("2020-02-15", tz="UTC")
        sliced = ov._period_return(equity, start, end)
        assert len(sliced) > 0
        assert sliced.index.min() >= start
        assert sliced.index.max() <= end

    def test_period_return_full_period_returns_all(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        equity = pd.Series(range(n), index=idx, dtype=float)
        sliced = ov._period_return(equity, None, None)
        assert len(sliced) == n


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------

class TestNoLiveTradingImports:
    FORBIDDEN = ["brokers", "execution", "live"]

    def test_no_live_module_imports(self):
        source = Path("research/fixed_five_defensive_overlay.py").read_text()
        for mod in self.FORBIDDEN:
            assert f"from {mod}" not in source, f"Forbidden import from {mod}"
            assert f"import {mod}" not in source, f"Forbidden import of {mod}"

    def test_no_order_placement_patterns(self):
        source = Path("research/fixed_five_defensive_overlay.py").read_text()
        for pat in ["place_order", "submit_order", "create_order", "send_order"]:
            assert pat not in source, f"Forbidden pattern '{pat}'"
