"""Tests for BTC time-series momentum parameter sweep."""

from __future__ import annotations

import pandas as pd
import pytest

from research.btc_time_series_sweep import (
    build_btc_sweep_parameter_grid,
    run_btc_time_series_sweep,
)


def _toy_close(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
    prices = [100.0 + i * 0.2 for i in range(periods)]
    return pd.DataFrame({"BTC/USD": prices}, index=index, dtype=float)


def test_sweep_parameter_grid_requires_short_less_than_medium() -> None:
    with pytest.raises(ValueError, match="short_lookback < medium_lookback"):
        build_btc_sweep_parameter_grid(
            short_lookbacks=(180, 240),
            medium_lookbacks=(120,),
            rebalance_every=(6,),
        )


def test_sweep_parameter_grid_expected_default_count() -> None:
    combos = build_btc_sweep_parameter_grid(
        short_lookbacks=(24, 42, 60, 90),
        medium_lookbacks=(120, 180, 240, 360),
        rebalance_every=(3, 6, 12),
    )
    assert len(combos) == 48


def test_sweep_output_includes_required_columns() -> None:
    close = _toy_close(periods=220)
    result = run_btc_time_series_sweep(
        short_lookbacks=(24, 42),
        medium_lookbacks=(120,),
        rebalance_every=(3,),
        close_prices=close,
        save_csv=False,
    )

    required = {
        "short_lookback",
        "medium_lookback",
        "rebalance_every",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
        "vs_buy_hold_total_return",
        "vs_buy_hold_cagr",
        "vs_buy_hold_sharpe",
        "vs_buy_hold_max_drawdown",
    }
    assert required.issubset(set(result.columns))
    assert len(result) == 2
