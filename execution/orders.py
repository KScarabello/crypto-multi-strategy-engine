"""Order planning and execution utilities for rebalancing.

Transferred from crypto-momentum-strategy and adapted for multi-strategy use.

Provides:
- Trade planning (compute trades from current to target weights)
- Order representations
- Sizing calculations
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    """Trade instruction for one symbol."""

    symbol: str
    action: str  # "BUY" or "SELL"
    current_exposure: float
    target_exposure: float
    delta_exposure: float
    delta_notional: float


def compute_target_exposure(equity: float, target_weight: float) -> float:
    """Compute target dollar exposure from account equity and target weight."""
    if equity <= 0:
        raise ValueError("equity must be positive")
    if target_weight < 0:
        raise ValueError("target_weight must be non-negative")
    return equity * target_weight


def plan_trades(
    equity: float,
    current_positions: dict[str, float],
    target_weights: dict[str, float],
    min_trade_notional: float = 1.0,
) -> list[Order]:
    """Plan trades to move from current positions to target weights.

    Args:
        equity: Current account equity in dollars.
        current_positions: Mapping of symbol -> current dollar exposure.
        target_weights: Mapping of symbol -> target portfolio weight.
        min_trade_notional: Minimum trade size to include in plan (in dollars).

    Returns:
        List of Order objects sorted by trade magnitude (largest first).
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    if min_trade_notional < 0:
        raise ValueError("min_trade_notional must be non-negative")

    # Validate target weights (allow sum < 1.0 for implicit cash)
    total_target = sum(target_weights.values())
    if total_target < 0 or total_target > 1.0:
        raise ValueError(f"target_weights sum must be in [0, 1]: {total_target}")

    all_symbols = set(current_positions.keys()) | set(target_weights.keys())
    orders = []

    for symbol in sorted(all_symbols):
        current_exp = current_positions.get(symbol, 0.0)
        target_wt = target_weights.get(symbol, 0.0)
        target_exp = compute_target_exposure(equity=equity, target_weight=target_wt)

        delta_exp = target_exp - current_exp
        delta_notional = abs(delta_exp)

        # Skip tiny changes below threshold
        if delta_notional < min_trade_notional:
            continue

        action = "BUY" if delta_exp > 0 else "SELL"
        orders.append(
            Order(
                symbol=symbol,
                action=action,
                current_exposure=current_exp,
                target_exposure=target_exp,
                delta_exposure=delta_exp,
                delta_notional=delta_notional,
            )
        )

    # Sort by magnitude (largest changes first)
    orders.sort(key=lambda o: o.delta_notional, reverse=True)
    return orders


def print_trade_plan(orders: list[Order], equity: float, title: str = "TRADE PLAN") -> None:
    """Print a formatted trade plan with position summaries.
    
    Args:
        orders: List of Order objects to display.
        equity: Account equity in USD (for percentage calculations).
        title: Title for the printed plan.
    """
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)
    print(f"Account Equity: ${equity:,.2f}\n")

    if not orders:
        print("No trades needed (all positions already at target).\n")
    else:
        print(f"{len(orders)} trade(s) to execute:\n")
        total_buy_notional = 0.0
        total_sell_notional = 0.0

        for i, order in enumerate(orders, 1):
            sign = "+" if order.delta_exposure > 0 else ""
            pct_current = (order.current_exposure / equity * 100.0) if equity > 0 else 0.0
            pct_target = (order.target_exposure / equity * 100.0) if equity > 0 else 0.0

            print(f"{i}. {order.symbol}")
            print(f"   Current:  ${order.current_exposure:>13,.2f}  ({pct_current:>6.2f}%)")
            print(f"   Target:   ${order.target_exposure:>13,.2f}  ({pct_target:>6.2f}%)")
            print(f"   {order.action:6s}:    ${order.delta_notional:>13,.2f}  ({sign}{order.delta_notional:,.2f})")
            print()

            if order.action == "BUY":
                total_buy_notional += order.delta_notional
            else:
                total_sell_notional += order.delta_notional

        print(f"Total Buy Notional:   ${total_buy_notional:>11,.2f}")
        print(f"Total Sell Notional:  ${total_sell_notional:>11,.2f}")
        print(f"Net Turnover:         ${abs(total_buy_notional - total_sell_notional):>11,.2f}")

    print("=" * 85 + "\n")
