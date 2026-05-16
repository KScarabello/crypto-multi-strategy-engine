"""Tests for research signal calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from research.signals import (
    calculate_momentum_signal,
    calculate_overextension_signal,
    calculate_return_signal,
)


def test_calculate_return_signal_matches_expected_toy_values() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    prices = pd.DataFrame(
        {
            "BTC/USD": [100.0, 105.0, 110.0, 121.0, 133.1],
            "ETH/USD": [50.0, 50.0, 55.0, 60.5, 66.55],
        },
        index=idx,
    )

    signal = calculate_return_signal(prices, lookback_bars=2)
    assert signal.loc[idx[2], "BTC/USD"] == pytest.approx(0.10)
    assert signal.loc[idx[4], "BTC/USD"] == pytest.approx(0.21)
    assert signal.loc[idx[4], "ETH/USD"] == pytest.approx(0.21)


def test_momentum_signal_positive_when_price_trends_up() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    prices = pd.DataFrame({"BTC/USD": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]}, index=idx)

    momentum = calculate_momentum_signal(prices, lookback_bars=3)
    assert momentum.loc[idx[-1], "BTC/USD"] > 0.0


def test_overextension_signal_is_high_after_sharp_spike() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    prices = pd.DataFrame(
        {
            "BTC/USD": [100.0, 100.0, 100.0, 100.0, 140.0, 150.0],
        },
        index=idx,
    )

    over = calculate_overextension_signal(prices, lookback_bars=2)
    # Last value compares 150 against 100 two bars ago -> +50%
    assert over.loc[idx[-1], "BTC/USD"] == pytest.approx(0.50)
