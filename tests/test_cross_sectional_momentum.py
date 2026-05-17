"""Tests for cross-sectional momentum strategy.

These tests focus on pure strategy behavior and intentionally avoid broker,
network, and live orchestration dependencies.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
    compute_momentum_score,
    rank_symbols_for_date,
)


def _sample_close_matrix() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "BTC/USD": [100, 102, 104, 106, 108, 111, 114, 118],
            "ETH/USD": [100, 100, 100, 100, 100, 100, 100, 100],
            "SOL/USD": [50, 50.5, 51, 51.7, 52.5, 53.4, 54.4, 55.5],
            "XRP/USD": [20, 19.9, 19.8, 19.7, 19.5, 19.3, 19.1, 18.9],
        },
        index=index,
        dtype=float,
    )


def test_momentum_scores_rank_assets_correctly_on_synthetic_data() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    score = compute_momentum_score(
        close=close,
        short_lookback_bars=1,
        medium_lookback_bars=3,
        short_weight=0.5,
        medium_weight=0.5,
    )
    ranked = rank_symbols_for_date(score, rebalance_timestamp=ts, top_n=4)

    assert ranked[0] == "BTC/USD"
    assert ranked[1] == "SOL/USD"
    assert ranked[-1] == "XRP/USD"


def test_strategy_selects_top_n_and_equal_weights() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 3,
            "min_history_bars": 3,
        },
    )

    assert set(weights.keys()) == {"BTC/USD", "SOL/USD"}
    assert weights["BTC/USD"] == pytest.approx(0.5)
    assert weights["SOL/USD"] == pytest.approx(0.5)


def test_assets_with_insufficient_history_are_excluded() -> None:
    close = _sample_close_matrix()
    close["NEW/USD"] = [float("nan")] * 6 + [10.0, 20.0]
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 2,
            "min_history_bars": 4,
        },
    )

    assert "NEW/USD" not in weights
    assert set(weights.keys()) == {"BTC/USD", "SOL/USD"}


def test_invalid_missing_zero_or_negative_prices_are_handled_safely() -> None:
    close = _sample_close_matrix()
    close["BAD/USD"] = [10.0, 9.0, 0.0, -1.0, 5.0, 5.0, 5.0, 0.0]
    close.loc[close.index[-1], "ETH/USD"] = float("nan")
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 3,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 2,
            "min_history_bars": 3,
        },
    )

    assert "BAD/USD" not in weights
    assert "ETH/USD" not in weights
    assert all(weight >= 0.0 for weight in weights.values())
    assert sum(weights.values()) <= 1.0 + 1e-9


def test_regime_filter_risk_off_returns_cash_only() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USD": [100, 100, 100, 100, 100, 100, 100, 90],
            "ETH/USD": [50, 51, 52, 53, 54, 55, 56, 57],
            "SOL/USD": [10, 10.2, 10.4, 10.6, 10.9, 11.2, 11.4, 11.6],
        },
        index=index,
        dtype=float,
    )
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 2,
            "use_regime_filter": True,
            "btc_symbol": "BTC/USD",
            "regime_lookback_bars": 3,
            "min_history_bars": 3,
        },
    )

    assert weights == {}


def test_regime_filter_risk_on_allows_selection() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 3,
            "use_regime_filter": True,
            "btc_symbol": "BTC/USD",
            "regime_lookback_bars": 3,
            "min_history_bars": 3,
        },
    )

    assert set(weights.keys()) == {"BTC/USD", "SOL/USD"}


def test_weight_output_uses_repo_cash_convention_by_default() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 3,
            "min_history_bars": 3,
            "max_gross_exposure": 0.75,
        },
    )

    assert "USD" not in weights
    assert sum(weights.values()) == pytest.approx(0.75)


def test_optional_explicit_cash_symbol_output() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 2,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 3,
            "min_history_bars": 3,
            "max_gross_exposure": 0.75,
            "include_cash_symbol": True,
            "cash_symbol": "USD",
        },
    )

    assert weights["USD"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_strategy_runs_without_broker_credentials_or_live_setup() -> None:
    close = _sample_close_matrix()
    ts = close.index[-1]

    strategy = CrossSectionalMomentumStrategy()
    weights = strategy.generate_target_weights(
        close_prices=close,
        timestamp=ts,
        config={
            "top_n": 1,
            "short_lookback_bars": 1,
            "medium_lookback_bars": 2,
            "min_history_bars": 2,
        },
    )

    assert weights == {"BTC/USD": pytest.approx(1.0)}


def test_parity_style_weighted_score_formula_matches_reference_math() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    close = pd.DataFrame(
        {
            "BTC/USD": [100.0, 110.0, 121.0, 133.1, 146.41],
            "ETH/USD": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )

    score = compute_momentum_score(
        close=close,
        short_lookback_bars=1,
        medium_lookback_bars=2,
        short_weight=0.5,
        medium_weight=0.5,
    )

    # Legacy formula parity: 0.5 * 1-bar return + 0.5 * 2-bar return
    ts = index[-1]
    btc_short = (146.41 / 133.1) - 1.0
    btc_medium = (146.41 / 121.0) - 1.0
    expected_btc = 0.5 * btc_short + 0.5 * btc_medium

    assert score.loc[ts, "BTC/USD"] == pytest.approx(expected_btc)
    assert score.loc[ts, "ETH/USD"] == pytest.approx(0.0)
