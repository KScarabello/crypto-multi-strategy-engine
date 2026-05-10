"""Tests for Strategy base interface."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import Strategy, StrategyState


class SimpleTestStrategy(Strategy):
    """Simple test implementation of Strategy."""

    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state=None,
        config=None,
    ) -> dict[str, float]:
        """Return equal weights for all symbols."""
        n_symbols = len(close_prices.columns)
        if n_symbols == 0:
            return {}
        weight = 1.0 / n_symbols
        return {symbol: weight for symbol in close_prices.columns}


def test_strategy_name_default() -> None:
    """Test that strategy uses class name as default."""
    strategy = SimpleTestStrategy()
    assert strategy.name == "SimpleTestStrategy"


def test_strategy_name_custom() -> None:
    """Test that custom strategy name is used."""
    strategy = SimpleTestStrategy(name="my_test_strategy")
    assert strategy.name == "my_test_strategy"


def test_strategy_generate_weights() -> None:
    """Test weight generation."""
    strategy = SimpleTestStrategy()
    close = pd.DataFrame(
        {"BTC/USD": [1.0, 2.0], "ETH/USD": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    )
    weights = strategy.generate_target_weights(
        close,
        close.index[0],
    )

    assert len(weights) == 2
    assert weights["BTC/USD"] == pytest.approx(0.5)
    assert weights["ETH/USD"] == pytest.approx(0.5)


def test_validate_weights_valid() -> None:
    """Test weight validation with valid weights."""
    strategy = SimpleTestStrategy()
    weights = {"BTC/USD": 0.3, "ETH/USD": 0.3, "XRP/USD": 0.4}
    # Should not raise
    strategy.validate_weights(weights)


def test_validate_weights_exceeds_100() -> None:
    """Test weight validation with weights exceeding 100%."""
    strategy = SimpleTestStrategy()
    weights = {"BTC/USD": 0.6, "ETH/USD": 0.6}
    with pytest.raises(ValueError, match="exceeds 1.0"):
        strategy.validate_weights(weights)


def test_validate_weights_negative() -> None:
    """Test weight validation with negative weight."""
    strategy = SimpleTestStrategy()
    weights = {"BTC/USD": -0.1, "ETH/USD": 1.1}
    with pytest.raises(ValueError, match="negative weight"):
        strategy.validate_weights(weights)


def test_validate_weights_not_dict() -> None:
    """Test weight validation with non-dict input."""
    strategy = SimpleTestStrategy()
    with pytest.raises(TypeError, match="must be dict"):
        strategy.validate_weights([0.5, 0.5])  # type: ignore


def test_strategy_state() -> None:
    """Test StrategyState dataclass."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    state = StrategyState(timestamp=ts, data_fresh=True, data_available=True)
    
    assert state.timestamp == ts
    assert state.data_fresh is True
    assert state.data_available is True
