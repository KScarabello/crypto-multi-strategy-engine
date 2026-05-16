"""Tests for modular momentum/reversal strategy variants."""

from __future__ import annotations

import pandas as pd
import pytest

from research.strategy_variants import (
    momentum_with_entry_filter_and_exit_signal_weights,
    momentum_with_entry_filter_weights,
    momentum_with_exit_signal_weights,
    short_term_reversal_weights,
    time_series_momentum_weights,
)


def _toy_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")


def test_momentum_only_selects_positive_momentum_assets() -> None:
    idx = _toy_index()
    momentum = pd.DataFrame(
        {
            "BTC/USD": [-0.1, 0.2, 0.3, -0.05],
            "ETH/USD": [0.1, -0.1, 0.2, -0.2],
        },
        index=idx,
    )

    weights = time_series_momentum_weights(momentum)
    assert weights.loc[idx[1], "BTC/USD"] == pytest.approx(1.0)
    assert weights.loc[idx[1], "ETH/USD"] == pytest.approx(0.0)
    assert weights.loc[idx[2], "BTC/USD"] == pytest.approx(0.5)
    assert weights.loc[idx[2], "ETH/USD"] == pytest.approx(0.5)
    assert weights.loc[idx[3]].sum() == pytest.approx(0.0)


def test_reversal_only_selects_pullback_assets() -> None:
    idx = _toy_index()
    over = pd.DataFrame(
        {
            "BTC/USD": [0.05, -0.02, 0.01, -0.10],
            "ETH/USD": [0.03, -0.03, -0.01, 0.20],
        },
        index=idx,
    )

    weights = short_term_reversal_weights(over, pullback_threshold=0.0)
    assert weights.loc[idx[0]].sum() == pytest.approx(0.0)
    assert weights.loc[idx[1], "BTC/USD"] == pytest.approx(0.5)
    assert weights.loc[idx[1], "ETH/USD"] == pytest.approx(0.5)
    assert weights.loc[idx[3], "BTC/USD"] == pytest.approx(1.0)


def test_entry_filter_excludes_overextended_assets() -> None:
    idx = _toy_index()
    momentum = pd.DataFrame(
        {
            "BTC/USD": [0.2, 0.2, 0.2, 0.2],
            "ETH/USD": [0.2, 0.2, 0.2, 0.2],
        },
        index=idx,
    )
    over = pd.DataFrame(
        {
            "BTC/USD": [0.10, 0.16, 0.05, 0.20],
            "ETH/USD": [0.05, 0.04, 0.30, 0.02],
        },
        index=idx,
    )

    weights = momentum_with_entry_filter_weights(
        momentum,
        over,
        entry_overextension_threshold=0.15,
    )
    assert weights.loc[idx[1], "BTC/USD"] == pytest.approx(0.0)
    assert weights.loc[idx[1], "ETH/USD"] == pytest.approx(1.0)
    assert weights.loc[idx[2], "BTC/USD"] == pytest.approx(1.0)
    assert weights.loc[idx[2], "ETH/USD"] == pytest.approx(0.0)


def test_exit_signal_exits_existing_holding_when_overextended() -> None:
    idx = _toy_index()
    momentum = pd.DataFrame(
        {
            "BTC/USD": [0.2, 0.2, 0.2, 0.2],
            "ETH/USD": [-0.1, -0.1, -0.1, -0.1],
        },
        index=idx,
    )
    over = pd.DataFrame(
        {
            "BTC/USD": [0.05, 0.10, 0.35, 0.10],
            "ETH/USD": [0.00, 0.00, 0.00, 0.00],
        },
        index=idx,
    )

    weights = momentum_with_exit_signal_weights(
        momentum,
        over,
        exit_overextension_threshold=0.30,
    )

    assert weights.loc[idx[1], "BTC/USD"] == pytest.approx(1.0)
    # Exits on spike bar and does not re-enter same bar.
    assert weights.loc[idx[2], "BTC/USD"] == pytest.approx(0.0)
    # Re-entry is allowed on later bars based on momentum-only entry logic.
    assert weights.loc[idx[3], "BTC/USD"] == pytest.approx(1.0)


def test_combined_entry_exit_differs_for_new_entries_vs_existing_positions() -> None:
    idx = _toy_index()
    momentum = pd.DataFrame(
        {
            "BTC/USD": [0.2, 0.2, 0.2, 0.2],
            "ETH/USD": [-0.2, -0.2, 0.2, 0.2],
        },
        index=idx,
    )
    over = pd.DataFrame(
        {
            "BTC/USD": [0.05, 0.20, 0.20, 0.20],
            "ETH/USD": [0.05, 0.20, 0.20, 0.20],
        },
        index=idx,
    )

    weights = momentum_with_entry_filter_and_exit_signal_weights(
        momentum,
        over,
        entry_overextension_threshold=0.15,
        exit_overextension_threshold=0.30,
    )

    # BTC enters on bar 0. At bars 1-3 it remains held despite over=0.20 (>entry)
    # because it is below exit threshold.
    assert weights.loc[idx[1], "BTC/USD"] == pytest.approx(1.0)
    # ETH becomes momentum-positive later but should not enter due to entry filter.
    assert weights.loc[idx[2], "ETH/USD"] == pytest.approx(0.0)
    assert weights.loc[idx[3], "ETH/USD"] == pytest.approx(0.0)


def test_weights_sum_to_one_or_zero() -> None:
    idx = _toy_index()
    momentum = pd.DataFrame(
        {
            "BTC/USD": [0.1, 0.2, -0.1, -0.2],
            "ETH/USD": [0.1, -0.2, -0.1, -0.2],
        },
        index=idx,
    )
    over = pd.DataFrame(
        {
            "BTC/USD": [0.01, 0.02, 0.03, 0.04],
            "ETH/USD": [0.01, 0.02, 0.03, 0.04],
        },
        index=idx,
    )

    variants = [
        time_series_momentum_weights(momentum),
        short_term_reversal_weights(over, pullback_threshold=0.0),
        momentum_with_entry_filter_weights(momentum, over),
        momentum_with_exit_signal_weights(momentum, over),
        momentum_with_entry_filter_and_exit_signal_weights(momentum, over),
    ]

    for weights in variants:
        sums = weights.sum(axis=1)
        assert (((sums - 0.0).abs() < 1e-12) | ((sums - 1.0).abs() < 1e-12)).all()


def test_no_lookahead_bias_weights_before_future_change_are_unchanged() -> None:
    idx = _toy_index()
    momentum_base = pd.DataFrame(
        {
            "BTC/USD": [0.2, 0.2, 0.2, 0.2],
            "ETH/USD": [-0.2, -0.2, -0.2, -0.2],
        },
        index=idx,
    )

    weights_base = time_series_momentum_weights(momentum_base)

    momentum_changed = momentum_base.copy()
    momentum_changed.loc[idx[-1], "ETH/USD"] = 10.0
    weights_changed = time_series_momentum_weights(momentum_changed)

    pd.testing.assert_series_equal(weights_base.loc[idx[0]], weights_changed.loc[idx[0]])
    pd.testing.assert_series_equal(weights_base.loc[idx[1]], weights_changed.loc[idx[1]])
    pd.testing.assert_series_equal(weights_base.loc[idx[2]], weights_changed.loc[idx[2]])
