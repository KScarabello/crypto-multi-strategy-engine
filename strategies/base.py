"""Base strategy interface for multi-strategy framework.

All strategies should inherit from Strategy and implement generate_target_weights().
This provides a common interface for signal generation across different strategy types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyState:
    """Immutable state snapshot for a strategy execution."""

    timestamp: pd.Timestamp
    data_fresh: bool
    data_available: bool


class Strategy(ABC):
    """Base class for all trading strategies.
    
    Strategies generate target weights that specify the desired portfolio
    allocation across assets. Weights must sum to <= 1.0, with the remainder
    allocated to cash.
    
    Example target weight output:
    {
        "BTC/USD": 0.30,
        "ETH/USD": 0.20,
        "SOL/USD": 0.25,
        "USD": 0.25  # implicit cash
    }
    """

    def __init__(self, name: str | None = None):
        """Initialize strategy.
        
        Args:
            name: Optional strategy identifier (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state: StrategyState | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Generate target portfolio weights for the current timestamp.
        
        Args:
            close_prices: DataFrame with index=timestamp, columns=symbol, values=close price
            timestamp: Current bar timestamp
            state: Optional strategy state (data freshness, availability, etc)
            config: Optional strategy configuration overrides
        
        Returns:
            Dictionary mapping symbol -> target weight (sum <= 1.0, remainder is cash)
        
        Raises:
            ValueError: If weights are invalid
        """
        pass

    def validate_weights(self, weights: dict[str, float]) -> None:
        """Validate that generated weights are well-formed.
        
        Args:
            weights: Dictionary of symbol -> weight
            
        Raises:
            ValueError: If weights are invalid
        """
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be dict, got {type(weights)}")
        
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"weights sum to {total:.6f}, exceeds 1.0")
        
        if total < 0:
            raise ValueError(f"weights sum to {total:.6f}, negative")
        
        for symbol, weight in weights.items():
            if weight < -1e-12:
                raise ValueError(f"negative weight for {symbol}: {weight}")


class MomentumStrategy(Strategy):
    """Placeholder for momentum strategy implementation.
    
    This is a base class showing the pattern for implementing
    strategy-specific logic. Actual momentum calculation would be
    delegated to strategy modules.
    """

    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state: StrategyState | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Generate target weights using momentum scoring.
        
        Configuration keys (from config dict):
        - top_n: Number of top assets to hold
        - short_lookback_bars: Short window for momentum
        - medium_lookback_bars: Medium window for momentum
        - short_weight: Weight for short momentum
        - medium_weight: Weight for medium momentum
        - use_regime_filter: Optional BTC-based regime filter
        """
        raise NotImplementedError(
            "MomentumStrategy is a placeholder. Implement generate_target_weights() "
            "or use a concrete subclass."
        )


class TimeSeriesReversalStrategy(Strategy):
    """Placeholder for time-series reversal strategy.
    
    Reversal strategies bet against recent momentum.
    """

    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state: StrategyState | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Generate target weights using reversal scoring.
        
        Reversal favors assets that have performed poorly recently.
        """
        raise NotImplementedError(
            "TimeSeriesReversalStrategy is a placeholder. Implement generate_target_weights() "
            "or use a concrete subclass."
        )


class BlendedStrategy(Strategy):
    """Placeholder for blended/ensemble strategy.
    
    Blended strategies combine multiple sub-strategies into a single
    portfolio allocation.
    """

    def __init__(self, name: str | None = None, sub_strategies: list[Strategy] | None = None):
        """Initialize blended strategy.
        
        Args:
            name: Strategy identifier
            sub_strategies: List of strategies to blend
        """
        super().__init__(name)
        self.sub_strategies = sub_strategies or []

    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state: StrategyState | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Generate target weights by blending sub-strategies.
        
        Configuration keys:
        - blend_weights: List of weights for each sub-strategy
        """
        raise NotImplementedError(
            "BlendedStrategy is a placeholder. Implement generate_target_weights() "
            "or use a concrete subclass."
        )
