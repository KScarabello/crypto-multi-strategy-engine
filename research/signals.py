"""Signal utilities for time-series momentum and short-term overextension research."""

from __future__ import annotations

import pandas as pd


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate close-price matrix shape and type for research signals."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty:
        raise ValueError("prices matrix is empty")
    if prices.columns.empty:
        raise ValueError("prices must include at least one symbol column")
    return prices.astype(float)


def calculate_return_signal(prices: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    """Return trailing percentage return signal over a fixed lookback window.

    Research meaning:
    - Positive value: price at t is above price at t-lookback (trend strength)
    - Negative value: price at t is below price at t-lookback (weakness/pullback)
    """
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")

    close = _validate_prices(prices)
    return close.pct_change(periods=lookback_bars)


def calculate_momentum_signal(prices: pd.DataFrame, lookback_bars: int = 336) -> pd.DataFrame:
    """Compute medium-term momentum signal from trailing returns.

    Default lookback 336 bars corresponds to 8 weeks on 4-hour bars.
    """
    return calculate_return_signal(prices=prices, lookback_bars=lookback_bars)


def calculate_overextension_signal(prices: pd.DataFrame, lookback_bars: int = 42) -> pd.DataFrame:
    """Compute short-term overextension signal from trailing returns.

    Default lookback 42 bars corresponds to 1 week on 4-hour bars.
    Higher positive values indicate stronger short-term run-ups.
    """
    return calculate_return_signal(prices=prices, lookback_bars=lookback_bars)
