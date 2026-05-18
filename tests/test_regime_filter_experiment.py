"""Focused tests for the regime-filter experiment runner."""

from __future__ import annotations

import pandas as pd

from research.run_regime_filter_experiment import (
    CandidateStrategy,
    _btc_ma_filter,
    _btc_momentum_filter,
    _equal_weight_momentum_filter,
    apply_regime_filter_to_weights,
    run_regime_filter_experiment,
)


def test_btc_momentum_filter_risk_on_risk_off_behavior() -> None:
    """BTC momentum filter should switch based on trailing BTC return sign."""
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USD": [100.0, 110.0, 121.0, 115.0, 110.0, 105.0],
            "ETH/USD": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        },
        index=index,
    )

    signal = _btc_momentum_filter(close, lookback_bars=2)

    assert bool(signal.loc[index[2]]) is True
    assert bool(signal.loc[index[4]]) is False


def test_btc_ma_filter_risk_on_risk_off_behavior() -> None:
    """BTC MA filter should be risk-on above MA and risk-off below MA."""
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USD": [100.0, 105.0, 110.0, 90.0, 85.0, 80.0],
            "ETH/USD": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        },
        index=index,
    )

    signal = _btc_ma_filter(close, ma_bars=3)

    assert bool(signal.loc[index[2]]) is True
    assert bool(signal.loc[index[5]]) is False


def test_equal_weight_regime_filter_does_not_use_future_data() -> None:
    """Equal-weight momentum regime should be unaffected by future-only shocks."""
    index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    close_base = pd.DataFrame(
        {
            "BTC/USD": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "ETH/USD": [50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5],
        },
        index=index,
    )
    close_changed = close_base.copy()
    close_changed.loc[index[-1], "BTC/USD"] = 10_000.0

    eligibility = pd.DataFrame(True, index=index, columns=close_base.columns)

    signal_base = _equal_weight_momentum_filter(close_base, eligibility_mask=eligibility, lookback_bars=2)
    signal_changed = _equal_weight_momentum_filter(close_changed, eligibility_mask=eligibility, lookback_bars=2)

    pd.testing.assert_series_equal(signal_base.iloc[:-1], signal_changed.iloc[:-1])


def test_risk_off_produces_zero_target_weights() -> None:
    """Risk-off gating should force all-cash weights."""
    weights = pd.Series({"BTC/USD": 0.6, "ETH/USD": 0.4}, dtype=float)
    gated = apply_regime_filter_to_weights(weights, risk_on=False)

    assert gated.sum() == 0.0
    assert (gated == 0.0).all()


def test_toy_run_completes_without_network_or_real_data_files() -> None:
    """Toy in-memory regime experiment should run without local files or network access."""
    index = pd.date_range("2024-01-01", periods=500, freq="4h", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USD": [100.0 + i * 0.2 for i in range(500)],
            "ETH/USD": [50.0 + i * 0.1 for i in range(500)],
            "SOL/USD": [20.0 + i * 0.15 for i in range(500)],
        },
        index=index,
        dtype=float,
    )

    report = run_regime_filter_experiment(
        symbols=tuple(close.columns),
        close_prices=close,
        save_csv=False,
        candidates=(
            CandidateStrategy("ts_momentum", rebalance_bars=6, lookback_config_name="short_42_only"),
        ),
        cost_configs=((10, 5),),
        regime_filters=("none", "btc_momentum_180"),
    )

    assert len(report) == 2
    assert set(report["regime_filter_name"]) == {"none", "btc_momentum_180"}
    assert set(report["strategy_name"]) == {"ts_momentum"}
