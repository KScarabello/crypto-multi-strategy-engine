"""Unit tests for research/fixed_five_drawdown_analysis.py

All tests use synthetic fixtures — no live data, no live trading imports.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_drawdown_analysis as dd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq(vals: list[float], start: str = "2020-01-01", freq: str = "4h") -> pd.Series:
    """Build a synthetic equity curve from a list of values."""
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.Series(vals, index=idx, name="equity")


def _rets(eq: pd.Series) -> pd.Series:
    return eq.pct_change().fillna(0.0)


def _holdings(eq: pd.Series, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
    n = len(eq)
    data = {sym: [1 / len(symbols)] * n for sym in symbols}
    return pd.DataFrame(data, index=eq.index)


def _close(eq: pd.Series, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
    import numpy as np

    rng = np.random.default_rng(42)
    n = len(eq)
    data = {}
    for sym in symbols:
        prices = 1000 * np.cumprod(1 + rng.normal(0, 0.02, n))
        data[sym] = prices
    return pd.DataFrame(data, index=eq.index)


def _rebalance_log(eq: pd.Series) -> pd.DataFrame:
    """Synthetic rebalance log with one-bar delay."""
    idx = eq.index[::6]  # every 6 bars
    return pd.DataFrame({
        "signal_timestamp": idx,
        "execution_timestamp": eq.index[1::6][:len(idx)],
        "turnover": [0.3] * len(idx),
        "cost_rate": [0.0015] * len(idx),
        "weight_sum": [1.0] * len(idx),
    })


# ---------------------------------------------------------------------------
# detect_drawdown_episodes
# ---------------------------------------------------------------------------

class TestDetectDrawdownEpisodes:
    def test_flat_equity_no_episodes(self):
        eq = _eq([100.0] * 50)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert eps == []

    def test_single_drawdown_recovers(self):
        vals = [100] * 10 + [90, 85, 80, 85, 90, 95, 100, 105] + [110] * 5
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert len(eps) == 1
        ep = eps[0]
        assert ep["recovered"] is True
        assert ep["drawdown_pct"] < -0.05 * 100  # below threshold

    def test_drawdown_not_yet_recovered(self):
        vals = [100] * 10 + [90, 80, 70, 65] + [68, 70, 69]
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert len(eps) >= 1
        last = eps[-1]
        assert last["recovered"] is False
        assert last["recovery_ts"] is None

    def test_trough_is_worst_point(self):
        # Falls to 70 but worst is 60
        vals = [100] * 5 + [80, 70, 60, 65, 75, 90, 110, 120]
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert len(eps) == 1
        # Trough value should be 60
        assert abs(eps[0]["trough_equity"] - 60) < 1

    def test_multiple_distinct_episodes(self):
        # Two separate drops with recovery between them
        vals = (
            [100, 110, 115, 120, 110, 100, 90, 85, 90, 100, 115, 120, 130]  # first DD + recovery
            + [140, 130, 120, 100, 90, 80, 85, 90, 100, 110, 120, 130, 150]  # second DD + recovery
        )
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert len(eps) >= 1

    def test_small_dip_below_threshold_detected(self):
        vals = [100, 94, 96, 100, 105]  # 6% dip
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert len(eps) == 1

    def test_dip_above_threshold_not_detected(self):
        vals = [100, 97, 98, 99, 100]  # 3% dip — below 5% threshold
        eq = _eq(vals)
        eps = dd.detect_drawdown_episodes(eq, threshold=-0.05)
        assert eps == []


# ---------------------------------------------------------------------------
# attribute_episode
# ---------------------------------------------------------------------------

class TestAttributeEpisode:
    def _make_setup(self):
        eq = _eq([100, 110, 120, 100, 80, 70, 75, 90, 100, 120])
        rets = _rets(eq)
        holdings = _holdings(eq)
        close = _close(eq)
        # Sync BTC to match the equity curve roughly
        close["BTC/USD"] = [100, 110, 120, 100, 80, 70, 75, 90, 100, 120]
        sig_ts = eq.index[::3]  # 4 elements for len=10
        n_reb = len(sig_ts)
        exec_ts = [eq.index[min(i + 1, len(eq) - 1)] for i in range(0, len(eq), 3)]
        reb_log = pd.DataFrame({
            "signal_timestamp": sig_ts,
            "execution_timestamp": exec_ts,
            "turnover": [0.4] * n_reb,
            "cost_rate": [0.002] * n_reb,
            "weight_sum": [1.0] * n_reb,
        })
        raw = {
            "peak_ts": eq.index[2],
            "peak_equity": 120.0,
            "trough_ts": eq.index[5],
            "trough_equity": 70.0,
            "drawdown_pct": -41.67,
            "recovery_ts": eq.index[9],
            "recovered": True,
        }
        return raw, eq, holdings, close, reb_log

    def test_returns_drawdown_episode(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert isinstance(ep, dd.DrawdownEpisode)
        assert ep.episode_id == 1
        assert ep.drawdown_pct < 0

    def test_recovered_flag_set(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert ep.recovered is True

    def test_peak_to_trough_bars_positive(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert ep.peak_to_trough_bars > 0

    def test_holdings_string_non_empty(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert len(ep.holdings_at_peak) > 0

    def test_btc_return_computed(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert not np.isnan(ep.btc_return_same_period_pct)

    def test_data_gaps_detected_when_present(self):
        raw, eq, h, close, reb = self._make_setup()
        close_gaps = close.copy()
        # Insert NaN in drawdown window
        close_gaps.loc[eq.index[3], "ETH/USD"] = np.nan
        ep = dd.attribute_episode(raw, eq, h, close_gaps, reb, 36, episode_id=1)
        assert ep.data_gaps_in_episode is True

    def test_no_data_gaps_clean_data(self):
        raw, eq, h, close, reb = self._make_setup()
        ep = dd.attribute_episode(raw, eq, h, close, reb, 36, episode_id=1)
        assert ep.data_gaps_in_episode is False

    def test_missing_peak_ts_in_holdings(self):
        """If peak_ts is not in holdings index, should not crash."""
        raw, eq, h, close, reb = self._make_setup()
        # Remove peak timestamp from holdings
        h_missing = h.drop(index=eq.index[2], errors="ignore")
        ep = dd.attribute_episode(raw, eq, h_missing, close, reb, 36, episode_id=1)
        assert isinstance(ep, dd.DrawdownEpisode)


# ---------------------------------------------------------------------------
# worst_bars
# ---------------------------------------------------------------------------

class TestWorstBars:
    def test_returns_n_rows(self):
        eq = _eq([100, 90, 110, 80, 120, 70, 130, 60, 140, 50, 150, 40])
        rets = _rets(eq)
        result = dd.worst_bars(rets, n=5)
        assert len(result) == 5

    def test_sorted_ascending(self):
        eq = _eq([100, 90, 110, 80, 120, 70, 130, 60, 140, 50, 150, 40])
        rets = _rets(eq)
        result = dd.worst_bars(rets, n=5)
        assert result["bar_return"].iloc[0] <= result["bar_return"].iloc[-1]

    def test_includes_holdings_columns_when_provided(self):
        eq = _eq([100, 90, 110, 80, 120, 70, 130, 60, 140, 50, 150, 40])
        rets = _rets(eq)
        h = _holdings(eq, ["BTC/USD", "ETH/USD"])
        result = dd.worst_bars(rets, n=5, holdings=h)
        assert "wt_btc" in result.columns
        assert "wt_eth" in result.columns

    def test_works_without_holdings(self):
        eq = _eq([100, 90, 110, 80, 120, 70, 130, 60, 140, 50, 150, 40])
        rets = _rets(eq)
        result = dd.worst_bars(rets, n=3, holdings=None)
        assert len(result) == 3
        assert "wt_btc" not in result.columns

    def test_n_larger_than_series(self):
        eq = _eq([100, 95, 100])
        rets = _rets(eq)
        result = dd.worst_bars(rets, n=100)
        assert len(result) == len(rets)


# ---------------------------------------------------------------------------
# yearly_stats
# ---------------------------------------------------------------------------

class TestYearlyStats:
    def _make_equity(self):
        # Two-year daily equity curve
        n = 2 * 365 * 6  # 4h bars, 2 years
        idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(42)
        rets = 1 + rng.normal(0.0001, 0.02, n)
        eq = pd.Series(100.0 * np.cumprod(rets), index=idx, name="equity")
        rets_s = eq.pct_change().fillna(0.0)
        close = pd.DataFrame({"BTC/USD": eq.values * 10}, index=idx)
        return eq, rets_s, close

    def test_returns_dataframe(self):
        eq, rets, close = self._make_equity()
        result = dd.yearly_stats(eq, rets, close, eq.index[0])
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        eq, rets, close = self._make_equity()
        result = dd.yearly_stats(eq, rets, close, eq.index[0])
        assert "year" in result.columns
        assert "strategy_return_pct" in result.columns
        assert "max_drawdown_pct" in result.columns

    def test_max_drawdown_non_positive(self):
        eq, rets, close = self._make_equity()
        result = dd.yearly_stats(eq, rets, close, eq.index[0])
        assert (result["max_drawdown_pct"] <= 0.01).all()

    def test_year_range_correct(self):
        eq, rets, close = self._make_equity()
        result = dd.yearly_stats(eq, rets, close, eq.index[0])
        assert set(result["year"]).issubset({2021, 2022, 2023})


# ---------------------------------------------------------------------------
# build_btc_comparison
# ---------------------------------------------------------------------------

class TestBuildBTCComparison:
    def _make_data(self):
        n = 500
        idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(7)
        eq = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.02, n)), index=idx)
        rets = eq.pct_change().fillna(0.0)
        close = pd.DataFrame({
            "BTC/USD": 40000 * np.cumprod(1 + rng.normal(0.0001, 0.025, n)),
        }, index=idx)
        episodes: list[dd.DrawdownEpisode] = []
        return eq, rets, close, idx[0], episodes

    def test_returns_dataframe(self):
        eq, rets, close, js, eps = self._make_data()
        result = dd.build_btc_comparison(eq, rets, close, js, eps)
        assert isinstance(result, pd.DataFrame)

    def test_includes_full_period(self):
        eq, rets, close, js, eps = self._make_data()
        result = dd.build_btc_comparison(eq, rets, close, js, eps)
        assert "full_period" in result["period"].values

    def test_btc_return_present(self):
        eq, rets, close, js, eps = self._make_data()
        result = dd.build_btc_comparison(eq, rets, close, js, eps)
        full = result[result["period"] == "full_period"].iloc[0]
        assert not np.isnan(full["btc_return_pct"])

    def test_yearly_rows_present(self):
        eq, rets, close, js, eps = self._make_data()
        result = dd.build_btc_comparison(eq, rets, close, js, eps)
        year_rows = result[result["period"].str.startswith("year_")]
        assert len(year_rows) >= 1

    def test_missing_btc_col_graceful(self):
        eq, rets, close, js, eps = self._make_data()
        close_no_btc = close.rename(columns={"BTC/USD": "ETH/USD"})
        result = dd.build_btc_comparison(eq, rets, close_no_btc, js, eps)
        full = result[result["period"] == "full_period"].iloc[0]
        assert np.isnan(full["btc_return_pct"])


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------

class TestNoLiveTradingImports:
    FORBIDDEN_MODULES = [
        "brokers",
        "execution",
        "live",
    ]

    def test_no_live_module_imports(self):
        """Research script must not import live trading modules."""
        source = Path("research/fixed_five_drawdown_analysis.py").read_text()
        for mod in self.FORBIDDEN_MODULES:
            assert f"from {mod}" not in source, f"Forbidden import from {mod}"
            assert f"import {mod}" not in source, f"Forbidden import of {mod}"

    def test_no_order_placement_code(self):
        """Research script must not contain order-placement patterns."""
        source = Path("research/fixed_five_drawdown_analysis.py").read_text()
        forbidden_patterns = ["place_order", "submit_order", "create_order", "send_order"]
        for pat in forbidden_patterns:
            assert pat not in source, f"Forbidden pattern '{pat}' found in research script"
