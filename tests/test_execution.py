"""Tests for order planning and execution utilities."""

from __future__ import annotations

import pytest

from execution.orders import (
    Order,
    compute_target_exposure,
    plan_trades,
)


def test_compute_target_exposure() -> None:
    """Test computation of target exposure from equity and weight."""
    equity = 100_000.0
    weight = 0.25
    exposure = compute_target_exposure(equity, weight)
    assert exposure == pytest.approx(25_000.0)


def test_compute_target_exposure_invalid_equity() -> None:
    """Test compute_target_exposure with invalid equity."""
    with pytest.raises(ValueError, match="equity must be positive"):
        compute_target_exposure(-100_000.0, 0.25)


def test_compute_target_exposure_negative_weight() -> None:
    """Test compute_target_exposure with negative weight."""
    with pytest.raises(ValueError, match="target_weight must be non-negative"):
        compute_target_exposure(100_000.0, -0.1)


def test_plan_trades_simple_rebalance() -> None:
    """Test simple trade planning."""
    equity = 100_000.0
    current_positions = {
        "BTC/USD": 20_000.0,
        "ETH/USD": 15_000.0,
        "XRP/USD": 10_000.0,
    }
    target_weights = {
        "BTC/USD": 0.25,
        "ETH/USD": 0.25,
        "XRP/USD": 0.25,
    }

    orders = plan_trades(
        equity=equity,
        current_positions=current_positions,
        target_weights=target_weights,
        min_trade_notional=0.0,
    )

    # All positions are below target and should be buys.
    assert len(orders) == 3

    by_symbol = {order.symbol: order for order in orders}
    assert set(by_symbol) == {"BTC/USD", "ETH/USD", "XRP/USD"}

    assert by_symbol["BTC/USD"].action == "BUY"
    assert by_symbol["BTC/USD"].delta_notional == pytest.approx(5_000.0)

    assert by_symbol["ETH/USD"].action == "BUY"
    assert by_symbol["ETH/USD"].delta_notional == pytest.approx(10_000.0)

    assert by_symbol["XRP/USD"].action == "BUY"
    assert by_symbol["XRP/USD"].delta_notional == pytest.approx(15_000.0)


def test_plan_trades_with_min_notional_filter() -> None:
    """Test trade planning with minimum notional filter."""
    equity = 100_000.0
    current_positions = {
        "BTC/USD": 25_000.0,
        "ETH/USD": 24_900.0,
        "XRP/USD": 24_900.0,
    }
    target_weights = {
        "BTC/USD": 0.25,
        "ETH/USD": 0.25,
        "XRP/USD": 0.25,
    }

    # With high min_trade_notional, no trades should be returned
    orders = plan_trades(
        equity=equity,
        current_positions=current_positions,
        target_weights=target_weights,
        min_trade_notional=500.0,
    )

    assert len(orders) == 0


def test_plan_trades_all_cash() -> None:
    """Test moving to all cash."""
    equity = 100_000.0
    current_positions = {
        "BTC/USD": 50_000.0,
        "ETH/USD": 50_000.0,
    }
    target_weights = {}  # All cash

    orders = plan_trades(
        equity=equity,
        current_positions=current_positions,
        target_weights=target_weights,
        min_trade_notional=0.0,
    )

    # Should have sell orders
    assert len(orders) == 2
    for order in orders:
        assert order.action == "SELL"


def test_plan_trades_empty_start() -> None:
    """Test moving from empty positions."""
    equity = 100_000.0
    current_positions = {}
    target_weights = {
        "BTC/USD": 0.5,
        "ETH/USD": 0.5,
    }

    orders = plan_trades(
        equity=equity,
        current_positions=current_positions,
        target_weights=target_weights,
        min_trade_notional=0.0,
    )

    assert len(orders) == 2
    for order in orders:
        assert order.action == "BUY"


def test_plan_trades_invalid_equity() -> None:
    """Test plan_trades with invalid equity."""
    with pytest.raises(ValueError, match="equity must be positive"):
        plan_trades(
            equity=-100_000.0,
            current_positions={},
            target_weights={},
        )


def test_plan_trades_invalid_weights_exceeds_100() -> None:
    """Test plan_trades with weights that exceed 100%."""
    with pytest.raises(ValueError, match="target_weights sum must be in"):
        plan_trades(
            equity=100_000.0,
            current_positions={},
            target_weights={"BTC/USD": 0.6, "ETH/USD": 0.6},
        )


def test_order_dataclass() -> None:
    """Test Order dataclass creation."""
    order = Order(
        symbol="BTC/USD",
        action="BUY",
        current_exposure=0.0,
        target_exposure=25_000.0,
        delta_exposure=25_000.0,
        delta_notional=25_000.0,
    )

    assert order.symbol == "BTC/USD"
    assert order.action == "BUY"
    assert order.delta_notional == 25_000.0
