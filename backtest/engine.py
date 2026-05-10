"""Generic backtesting engine for multi-strategy framework.

Provides execution-aware simulation with:
- Portfolio rebalancing support
- Transaction cost and slippage modeling
- One-bar-delayed execution
- Position constraints
- Rebalance scheduling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


@dataclass(frozen=True)
class BacktestResult:
    """Container for backtest outputs."""

    portfolio: pd.DataFrame  # equity curve with returns
    rebalance_log: pd.DataFrame  # detailed rebalance records
    holdings_history: pd.DataFrame  # timestamp x symbol weight matrix
    turnover: pd.Series  # per-rebalance turnover
    gross_return: pd.Series  # pre-cost returns


def _validate_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize long-format OHLCV input."""
    if ohlcv.empty:
        raise ValueError("ohlcv input is empty")

    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")

    frame = ohlcv[REQUIRED_OHLCV_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "symbol", "close"])
    frame = frame.sort_values(["symbol", "timestamp"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    )
    # Guard: non-positive close prices cause pct_change to generate inf returns.
    bad_close = frame["close"] <= 0
    if bad_close.any():
        n = int(bad_close.sum())
        LOGGER.warning("Dropping %d rows with non-positive close prices from engine input", n)
        frame = frame[~bad_close]
    return frame.reset_index(drop=True)


def _close_matrix(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format OHLCV to a timestamp x symbol close matrix."""
    close = ohlcv.pivot(index="timestamp", columns="symbol", values="close").sort_index().astype(float)
    close = close.dropna(how="all")
    if close.empty:
        raise ValueError("close matrix is empty after pivot")
    return close


def _validate_weights(weights: pd.Series, context: str) -> None:
    """Validate long-only normalized portfolio weights."""
    if weights.isna().any():
        raise ValueError(f"NaN weight detected in {context}")
    if (weights < -1e-12).any():
        raise ValueError(f"Negative weight detected in {context}")

    total_weight = float(weights.sum())
    if total_weight > 1.0 + 1e-9:
        raise ValueError(
            f"Weight sum exceeds 1.0 in {context}: {total_weight:.6f}"
        )


def _is_rebalance_bar_utc_hour(timestamp: pd.Timestamp, rebalance_hour_utc: int) -> bool:
    """Return True when the bar timestamp aligns with the configured UTC hour."""
    if not 0 <= int(rebalance_hour_utc) <= 23:
        raise ValueError(f"rebalance_hour_utc must be in [0, 23], got {rebalance_hour_utc}")

    ts = pd.Timestamp(timestamp)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts_utc.hour) == int(rebalance_hour_utc)


def run_backtest(
    ohlcv: pd.DataFrame,
    signal_generator: callable,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 0.0,
    rebalance_every_bars: int = 1,
    rebalance_hour_utc: int | None = None,
    max_position_weight: float | None = None,
    max_gross_exposure: float | None = None,
    max_turnover_per_rebalance: float | None = None,
) -> BacktestResult:
    """Run a generic backtest using a signal generator function.
    
    Args:
        ohlcv: Long-format OHLCV DataFrame with columns: timestamp, open, high, low, close, volume, symbol
        signal_generator: Callable that takes (close: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series
                         and returns target weights indexed by symbol
        initial_capital: Starting account value in USD
        transaction_cost_bps: Transaction costs in basis points (spread, fees)
        slippage_bps: Slippage cost in basis points
        rebalance_every_bars: Rebalance every N bars (ignored if rebalance_hour_utc is set)
        rebalance_hour_utc: If set, only rebalance at this UTC hour
        max_position_weight: Maximum weight per asset (clipping constraint)
        max_gross_exposure: Maximum total gross exposure (scaling constraint)
        max_turnover_per_rebalance: Maximum allowed turnover per rebalance
    
    Returns:
        BacktestResult with portfolio, rebalance_log, holdings_history, turnover, gross_return
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("cost and slippage bps must be non-negative")
    if rebalance_every_bars <= 0:
        raise ValueError("rebalance_every_bars must be positive")

    clean = _validate_ohlcv(ohlcv)
    close = _close_matrix(clean)
    volume = clean.pivot(index="timestamp", columns="symbol", values="volume").sort_index().astype(float)
    volume = volume.reindex(close.index)
    
    returns = close.pct_change().fillna(0.0)

    # Safety net: non-positive close prices can still produce inf after pivot
    n_inf = int(np.isinf(returns.values).sum())
    if n_inf > 0:
        LOGGER.warning(
            "Replacing %d inf/-inf return values with 0.0 (check input prices)", n_inf
        )
        returns = returns.replace([np.inf, -np.inf], 0.0)

    index = close.index
    symbols = close.columns

    holdings_history = pd.DataFrame(0.0, index=index, columns=symbols)
    turnover = pd.Series(0.0, index=index, dtype=float)
    gross_return = pd.Series(0.0, index=index, dtype=float)
    strategy_return = pd.Series(0.0, index=index, dtype=float)
    equity = pd.Series(float("nan"), index=index, dtype=float)

    rebalance_records: list[dict[str, object]] = []

    # Start at bar 1 (need at least one return observation)
    start_bar = 1
    if start_bar >= len(index) - 1:
        raise ValueError("Not enough bars for backtesting")

    equity.iloc[start_bar] = initial_capital
    current_weights = pd.Series(0.0, index=symbols, dtype=float)
    pending_target: pd.Series | None = None
    pending_signal_timestamp: pd.Timestamp | None = None

    total_bps = transaction_cost_bps + slippage_bps
    bar_count = 0

    for i in range(start_bar, len(index) - 1):
        ts = index[i]
        cost_rate = 0.0

        # Check if we should rebalance
        should_rebalance = False
        if rebalance_hour_utc is not None:
            should_rebalance = _is_rebalance_bar_utc_hour(ts, rebalance_hour_utc)
        else:
            bar_count += 1
            should_rebalance = (bar_count % rebalance_every_bars) == 0

        # Execute prior signal with one-bar delay
        if pending_target is not None:
            exec_turnover = float((pending_target - current_weights).abs().sum())
            cost_rate = (exec_turnover * total_bps) / 10_000.0
            turnover.iloc[i] = exec_turnover
            current_weights = pending_target
            _validate_weights(current_weights, context=f"execution at {ts}")

            rebalance_records.append(
                {
                    "signal_timestamp": pending_signal_timestamp,
                    "execution_timestamp": ts,
                    "turnover": exec_turnover,
                    "cost_rate": cost_rate,
                    "weight_sum": float(current_weights.sum()),
                }
            )

            pending_target = None
            pending_signal_timestamp = None

        if should_rebalance:
            try:
                target_weights = signal_generator(close, ts)
                if not isinstance(target_weights, pd.Series):
                    raise TypeError(f"signal_generator must return pd.Series, got {type(target_weights)}")
                
                # Ensure all symbols are represented
                target_weights = target_weights.reindex(symbols, fill_value=0.0)
                
                # Apply position weight cap if specified
                if max_position_weight is not None:
                    target_weights = target_weights.clip(lower=0.0, upper=max_position_weight)
                
                # Apply gross exposure cap if specified
                if max_gross_exposure is not None:
                    total_weight = float(target_weights.sum())
                    if total_weight > max_gross_exposure and total_weight > 0:
                        target_weights = target_weights * (max_gross_exposure / total_weight)
                
                # Apply turnover cap if specified
                if max_turnover_per_rebalance is not None:
                    raw_turnover = float((target_weights - current_weights).abs().sum())
                    if raw_turnover > max_turnover_per_rebalance and raw_turnover > 0:
                        scale = max_turnover_per_rebalance / raw_turnover
                        target_weights = current_weights + (target_weights - current_weights) * scale
                        target_weights = target_weights.clip(lower=0.0)
                        if target_weights.sum() > 1.0:
                            target_weights = target_weights / float(target_weights.sum())
                
                _validate_weights(target_weights, context=f"signal at {ts}")
                pending_target = target_weights
                pending_signal_timestamp = ts

            except Exception as exc:
                LOGGER.error("Signal generation failed at %s: %s", ts, exc)
                # Keep using current weights on error
                pending_target = None

        next_bar_return = float((current_weights * returns.iloc[i + 1]).sum())
        gross_return.iloc[i + 1] = next_bar_return
        net_return = next_bar_return - cost_rate
        strategy_return.iloc[i + 1] = net_return

        current_equity = float(equity.iloc[i])
        new_equity = current_equity * (1.0 + net_return)
        if not np.isfinite(new_equity):
            LOGGER.error(
                "Equity became non-finite at bar %d (ts=%s): "
                "net_return=%.6f, current_equity=%.4f — clamping to previous value",
                i + 1,
                index[i + 1],
                net_return,
                current_equity,
            )
            new_equity = current_equity
        equity.iloc[i + 1] = new_equity
        holdings_history.iloc[i] = current_weights

    holdings_history.iloc[-1] = current_weights

    portfolio = pd.DataFrame(
        {
            "strategy_return": strategy_return,
            "equity": equity,
        },
        index=index,
    ).dropna(subset=["equity"])

    turnover = turnover.loc[portfolio.index]
    holdings_history = holdings_history.loc[portfolio.index]
    rebalance_log = pd.DataFrame(rebalance_records)

    LOGGER.info(
        "Backtest finished: bars=%d, rebalances=%d, final_equity=%.2f",
        len(portfolio),
        len(rebalance_log),
        float(portfolio["equity"].iloc[-1]),
    )

    return BacktestResult(
        portfolio=portfolio,
        rebalance_log=rebalance_log,
        holdings_history=holdings_history,
        turnover=turnover,
        gross_return=gross_return.loc[portfolio.index],
    )
