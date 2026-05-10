"""Tests for backtest metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
    turnover_summary_stats,
)


def test_total_return() -> None:
    """Test total return calculation."""
    equity = pd.Series([10_000.0, 11_000.0, 12_100.0], dtype=float)
    ret = total_return(equity)
    assert ret == pytest.approx(0.21)


def test_total_return_no_change() -> None:
    """Test total return when equity is flat."""
    equity = pd.Series([10_000.0, 10_000.0], dtype=float)
    ret = total_return(equity)
    assert ret == pytest.approx(0.0)


def test_total_return_with_nans() -> None:
    """Test total return with NaN values."""
    equity = pd.Series([10_000.0, float("nan"), 12_000.0], dtype=float)
    ret = total_return(equity)
    assert ret == pytest.approx(0.2)


def test_cagr() -> None:
    """Test CAGR calculation."""
    equity = pd.Series([10_000.0, 12_000.0, 14_400.0], dtype=float)
    cagr_val = cagr(equity, bars_per_year=1)
    assert cagr_val == pytest.approx(0.2, abs=0.01)


def test_cagr_single_bar() -> None:
    """Test CAGR with only two bars."""
    equity = pd.Series([10_000.0, 11_000.0], dtype=float)
    cagr_val = cagr(equity, bars_per_year=1)
    assert cagr_val == pytest.approx(0.1)


def test_max_drawdown() -> None:
    """Test max drawdown calculation."""
    equity = pd.Series([10_000.0, 15_000.0, 8_000.0, 12_000.0], dtype=float)
    dd = max_drawdown(equity)
    assert dd == pytest.approx(-0.4666, abs=0.001)


def test_max_drawdown_no_drawdown() -> None:
    """Test max drawdown when equity only increases."""
    equity = pd.Series([10_000.0, 11_000.0, 12_000.0], dtype=float)
    dd = max_drawdown(equity)
    assert dd == pytest.approx(0.0)


def test_annualized_volatility() -> None:
    """Test annualized volatility calculation."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005], dtype=float)
    vol = annualized_volatility(returns, bars_per_year=252)
    # Daily returns std * sqrt(252)
    expected_daily_std = returns.std(ddof=0)
    expected_vol = expected_daily_std * (252 ** 0.5)
    assert vol == pytest.approx(expected_vol)


def test_sharpe_ratio() -> None:
    """Test Sharpe ratio calculation."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005], dtype=float)
    sharpe = sharpe_ratio(returns, bars_per_year=252)
    expected = (returns.mean() / returns.std(ddof=0)) * (252 ** 0.5)
    assert sharpe == pytest.approx(expected)


def test_sharpe_ratio_zero_returns() -> None:
    """Test Sharpe ratio with zero returns."""
    returns = pd.Series([0.0] * 10, dtype=float)
    sharpe = sharpe_ratio(returns, bars_per_year=252)
    assert sharpe == 0.0


def test_turnover_summary_stats() -> None:
    """Test turnover statistics calculation."""
    turnover = pd.Series([0.2, 0.3, 0.1, 0.15], dtype=float)
    stats = turnover_summary_stats(turnover)
    assert stats["avg_turnover"] == pytest.approx(0.1875)
    assert stats["median_turnover"] == pytest.approx(0.175)
    assert stats["max_turnover"] == pytest.approx(0.3)
    assert stats["total_turnover"] == pytest.approx(0.75)


def test_turnover_summary_stats_empty() -> None:
    """Test turnover statistics with empty series."""
    turnover = pd.Series([], dtype=float)
    stats = turnover_summary_stats(turnover)
    assert stats["avg_turnover"] == 0.0
    assert stats["median_turnover"] == 0.0
    assert stats["max_turnover"] == 0.0
    assert stats["total_turnover"] == 0.0
