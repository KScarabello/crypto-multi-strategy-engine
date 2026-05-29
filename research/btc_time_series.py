"""BTC-only time-series momentum research helpers.

This module is intentionally scoped to research/backtesting. It does not
interact with live execution code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.run_expanded_universe_experiment import bars_per_year_for_timeframe, close_prices_to_ohlcv
from research.signals import calculate_momentum_signal


@dataclass(frozen=True)
class BtcTimeSeriesConfig:
    """Config for BTC-only buy-and-hold vs time-series momentum experiment."""

    symbol: str = "BTC/USD"
    timeframe: str = "4h"
    short_lookback_bars: int = 42
    medium_lookback_bars: int = 180
    short_weight: float = 0.5
    medium_weight: float = 0.5
    rebalance_every_bars: int = 6
    initial_capital: float = SETTINGS.initial_capital
    transaction_cost_bps: float = SETTINGS.transaction_cost_bps
    slippage_bps: float = SETTINGS.slippage_bps


def _validate_close_prices(close_prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and normalize close matrix for BTC-only research."""
    if not isinstance(close_prices, pd.DataFrame):
        raise TypeError("close_prices must be a pandas DataFrame")
    if close_prices.empty:
        raise ValueError("close_prices is empty")
    if symbol not in close_prices.columns:
        raise ValueError(f"close_prices must include {symbol}")

    clean = close_prices[[symbol]].copy()
    clean.index = pd.to_datetime(clean.index, utc=True)
    clean = clean.sort_index().astype(float)
    clean = clean.dropna(how="all")
    if clean.empty:
        raise ValueError(f"No valid close prices found for {symbol}")
    return clean


def calculate_btc_combined_momentum(
    close_prices: pd.DataFrame,
    symbol: str = "BTC/USD",
    short_lookback_bars: int = 42,
    medium_lookback_bars: int = 180,
    short_weight: float = 0.5,
    medium_weight: float = 0.5,
) -> pd.Series:
    """Compute weighted short/medium BTC momentum score."""
    if short_lookback_bars <= 0 or medium_lookback_bars <= 0:
        raise ValueError("lookback bars must be positive")

    total_weight = float(short_weight + medium_weight)
    if total_weight <= 0:
        raise ValueError("short_weight + medium_weight must be positive")

    close = _validate_close_prices(close_prices, symbol=symbol)
    short_signal = calculate_momentum_signal(close, lookback_bars=short_lookback_bars)[symbol]
    medium_signal = calculate_momentum_signal(close, lookback_bars=medium_lookback_bars)[symbol]

    short_scale = float(short_weight) / total_weight
    medium_scale = float(medium_weight) / total_weight
    return short_scale * short_signal + medium_scale * medium_signal


def build_btc_ts_momentum_signal_generator(
    close_prices: pd.DataFrame,
    config: BtcTimeSeriesConfig,
) -> Callable[[pd.DataFrame, pd.Timestamp], pd.Series]:
    """Build a signal generator for BTC-only time-series momentum.

    Rule:
    - combined score > 0: hold BTC (weight 1)
    - combined score <= 0: hold cash (weight 0)
    """
    close = _validate_close_prices(close_prices, symbol=config.symbol)
    combined = calculate_btc_combined_momentum(
        close_prices=close,
        symbol=config.symbol,
        short_lookback_bars=config.short_lookback_bars,
        medium_lookback_bars=config.medium_lookback_bars,
        short_weight=config.short_weight,
        medium_weight=config.medium_weight,
    )
    invested = (combined > 0.0).fillna(False).astype(float)

    def signal_generator(close_matrix: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        weight = float(invested.get(timestamp, 0.0))
        row = pd.Series(0.0, index=close_matrix.columns, dtype=float)
        if config.symbol in row.index:
            row.loc[config.symbol] = weight
        return row

    return signal_generator


def build_btc_buy_and_hold_signal_generator(
    symbol: str,
) -> Callable[[pd.DataFrame, pd.Timestamp], pd.Series]:
    """Build a simple BTC buy-and-hold signal generator."""

    def signal_generator(close_matrix: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        row = pd.Series(0.0, index=close_matrix.columns, dtype=float)
        if symbol in row.index:
            row.loc[symbol] = 1.0
        return row

    return signal_generator


def _load_btc_close_prices(
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Load BTC close prices using existing local-first data conventions."""
    ohlcv = load_ohlcv_history(symbols=(symbol,), timeframe=timeframe, data_dir=data_dir)
    close = pivot_close(ohlcv)
    return _validate_close_prices(close, symbol=symbol)


def run_btc_time_series_backtests(
    config: BtcTimeSeriesConfig | None = None,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
) -> tuple[pd.DataFrame, int, dict[str, BacktestResult]]:
    """Run BTC TS and buy-and-hold backtests and return raw execution objects."""
    cfg = config or BtcTimeSeriesConfig()
    close = (
        _validate_close_prices(close_prices, symbol=cfg.symbol)
        if close_prices is not None
        else _load_btc_close_prices(symbol=cfg.symbol, timeframe=cfg.timeframe, data_dir=data_dir)
    )
    ohlcv = close_prices_to_ohlcv(close)
    bars_per_year = bars_per_year_for_timeframe(cfg.timeframe)

    ts_result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=build_btc_ts_momentum_signal_generator(close_prices=close, config=cfg),
        initial_capital=float(cfg.initial_capital),
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        slippage_bps=float(cfg.slippage_bps),
        rebalance_every_bars=int(cfg.rebalance_every_bars),
    )
    buy_hold_result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=build_btc_buy_and_hold_signal_generator(symbol=cfg.symbol),
        initial_capital=float(cfg.initial_capital),
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        slippage_bps=float(cfg.slippage_bps),
        rebalance_every_bars=1,
    )

    return close, bars_per_year, {
        "btc_buy_and_hold": buy_hold_result,
        "btc_time_series_momentum": ts_result,
    }


def _metrics_row(
    strategy_name: str,
    execution: BacktestResult,
    symbol: str,
    bars_per_year: int,
) -> dict[str, float | int | str]:
    """Build one strategy row with required experiment metrics."""
    portfolio = execution.portfolio
    metrics = summary_metrics(
        equity=portfolio["equity"],
        returns=portfolio["strategy_return"],
        turnover=execution.turnover.reindex(portfolio.index).fillna(0.0),
        bars_per_year=bars_per_year,
    )

    invested = execution.holdings_history[symbol].reindex(portfolio.index).fillna(0.0)
    turnover_aligned = execution.turnover.reindex(portfolio.index).fillna(0.0)
    # Count only bars where holdings actually changed; signal evaluations with
    # zero turnover are excluded for cleaner reporting.
    num_trades = int((turnover_aligned > 1e-12).sum())
    percent_time_invested = float((invested > 0.0).mean())

    return {
        "strategy": strategy_name,
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "num_trades": num_trades,
        "percent_time_invested": percent_time_invested,
        "num_rebalances": num_trades,
        "start_date": str(portfolio.index.min()),
        "end_date": str(portfolio.index.max()),
    }


def run_btc_time_series_experiment(
    config: BtcTimeSeriesConfig | None = None,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
) -> pd.DataFrame:
    """Run BTC buy-and-hold vs BTC time-series momentum comparison."""
    cfg = config or BtcTimeSeriesConfig()
    _, bars_per_year, results = run_btc_time_series_backtests(
        config=cfg,
        close_prices=close_prices,
        data_dir=data_dir,
    )

    rows = [
        _metrics_row(
            strategy_name="btc_buy_and_hold",
            execution=results["btc_buy_and_hold"],
            symbol=cfg.symbol,
            bars_per_year=bars_per_year,
        ),
        _metrics_row(
            strategy_name="btc_time_series_momentum",
            execution=results["btc_time_series_momentum"],
            symbol=cfg.symbol,
            bars_per_year=bars_per_year,
        ),
    ]
    report = pd.DataFrame(rows)

    buy_hold = report.loc[report["strategy"] == "btc_buy_and_hold"].iloc[0]
    ts = report["strategy"] == "btc_time_series_momentum"
    report["vs_buy_hold_total_return"] = 0.0
    report["vs_buy_hold_cagr"] = 0.0
    report["vs_buy_hold_sharpe"] = 0.0
    report["vs_buy_hold_max_drawdown"] = 0.0
    report.loc[ts, "vs_buy_hold_total_return"] = report.loc[ts, "total_return"] - float(buy_hold["total_return"])
    report.loc[ts, "vs_buy_hold_cagr"] = report.loc[ts, "cagr"] - float(buy_hold["cagr"])
    report.loc[ts, "vs_buy_hold_sharpe"] = report.loc[ts, "sharpe"] - float(buy_hold["sharpe"])
    report.loc[ts, "vs_buy_hold_max_drawdown"] = report.loc[ts, "max_drawdown"] - float(buy_hold["max_drawdown"])

    return report


def format_btc_experiment_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Return a display-friendly sorted view of experiment output."""
    required = {
        "strategy",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
        "vs_buy_hold_total_return",
    }
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"report missing required columns: {sorted(missing)}")

    ordered = [
        "strategy",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
        "num_rebalances",
        "vs_buy_hold_total_return",
        "vs_buy_hold_cagr",
        "vs_buy_hold_sharpe",
        "vs_buy_hold_max_drawdown",
        "start_date",
        "end_date",
    ]
    return report[ordered].sort_values("strategy").reset_index(drop=True)
