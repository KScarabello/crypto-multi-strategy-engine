"""Exploratory BTC long/short/cash momentum research module.

This file is research-only and is not wired to live trading execution paths.
Short exposure here is a synthetic backtest assumption with no liquidation modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_backtests
from research.run_expanded_universe_experiment import bars_per_year_for_timeframe


DEFAULT_SUMMARY_OUTPUT_PATH = Path("btc_long_short_momentum_summary.csv")
DEFAULT_BY_YEAR_OUTPUT_PATH = Path("btc_long_short_momentum_by_year.csv")
DEFAULT_SHORT_TRADES_OUTPUT_PATH = Path("btc_long_short_momentum_short_trades.csv")


@dataclass(frozen=True)
class LongShortThresholdSpec:
    name: str
    short_threshold: float
    medium_threshold: float


DEFAULT_LONG_SHORT_THRESHOLD_SPECS: tuple[LongShortThresholdSpec, ...] = (
    LongShortThresholdSpec(name="ls_symmetric_0_0", short_threshold=0.0, medium_threshold=0.0),
    LongShortThresholdSpec(name="ls_conservative_3_5", short_threshold=-0.03, medium_threshold=-0.05),
    LongShortThresholdSpec(name="ls_conservative_5_10", short_threshold=-0.05, medium_threshold=-0.10),
    LongShortThresholdSpec(name="ls_conservative_8_15", short_threshold=-0.08, medium_threshold=-0.15),
)


@dataclass(frozen=True)
class BtcLongShortResearchConfig:
    symbol: str = "BTC/USD"
    timeframe: str = "4h"
    short_lookback_bars: int = 60
    medium_lookback_bars: int = 240
    rebalance_every_bars: int = 12
    transaction_cost_bps: float = SETTINGS.transaction_cost_bps
    slippage_bps: float = SETTINGS.slippage_bps
    thresholds: Sequence[LongShortThresholdSpec] = DEFAULT_LONG_SHORT_THRESHOLD_SPECS


@dataclass(frozen=True)
class LongShortBacktestArtifacts:
    candidate_name: str
    signals: pd.DataFrame
    annual_report: pd.DataFrame
    short_trades: pd.DataFrame


def _validate_close_prices(close_prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
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


def _load_btc_close_prices(symbol: str, timeframe: str, data_dir: Path) -> pd.DataFrame:
    ohlcv = load_ohlcv_history(symbols=(symbol,), timeframe=timeframe, data_dir=data_dir)
    close = pivot_close(ohlcv)
    return _validate_close_prices(close, symbol=symbol)


def _build_target_positions(
    close: pd.DataFrame,
    short_lookback_bars: int,
    medium_lookback_bars: int,
    rebalance_every_bars: int,
    short_threshold: float,
    medium_threshold: float,
) -> pd.Series:
    if short_lookback_bars <= 0 or medium_lookback_bars <= 0:
        raise ValueError("lookback bars must be positive")
    if rebalance_every_bars <= 0:
        raise ValueError("rebalance_every_bars must be positive")

    px = close.iloc[:, 0].astype(float)
    short_momentum = px.pct_change(int(short_lookback_bars))
    medium_momentum = px.pct_change(int(medium_lookback_bars))

    target = pd.Series(0.0, index=close.index, dtype=float)
    current_target = 0.0

    for i, ts in enumerate(close.index):
        if i == 0:
            target.iloc[i] = 0.0
            continue

        if (i % int(rebalance_every_bars)) != 0:
            target.iloc[i] = current_target
            continue

        sm = float(short_momentum.loc[ts]) if pd.notna(short_momentum.loc[ts]) else float("nan")
        mm = float(medium_momentum.loc[ts]) if pd.notna(medium_momentum.loc[ts]) else float("nan")

        if pd.isna(sm) or pd.isna(mm):
            current_target = 0.0
        elif sm > 0.0 and mm > 0.0:
            current_target = 1.0
        elif sm < float(short_threshold) and mm < float(medium_threshold):
            current_target = -1.0
        else:
            current_target = 0.0

        target.iloc[i] = current_target

    return target


def _extract_short_trades(candidate_name: str, signals: pd.DataFrame) -> pd.DataFrame:
    position = signals["position"].astype(float)
    returns = signals["strategy_return"].astype(float)
    btc_returns = signals["btc_return"].astype(float)

    rows: list[dict[str, Any]] = []
    in_trade = False
    entry_idx = -1

    for i, is_short in enumerate((position < 0.0).tolist()):
        if is_short and not in_trade:
            in_trade = True
            entry_idx = i
            continue

        if in_trade and not is_short:
            exit_idx = i - 1
            trade_returns = returns.iloc[entry_idx : exit_idx + 1]
            trade_btc = btc_returns.iloc[entry_idx : exit_idx + 1]
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "short_entry_time": str(signals.index[entry_idx]),
                    "short_exit_time": str(signals.index[exit_idx]),
                    "short_trade_return": float((1.0 + trade_returns).prod() - 1.0),
                    "btc_return_during_short": float((1.0 + trade_btc).prod() - 1.0),
                    "holding_period_bars": int(exit_idx - entry_idx + 1),
                    "exit_reason": "position_not_short",
                }
            )
            in_trade = False
            entry_idx = -1

    if in_trade and entry_idx >= 0:
        exit_idx = len(signals.index) - 1
        trade_returns = returns.iloc[entry_idx : exit_idx + 1]
        trade_btc = btc_returns.iloc[entry_idx : exit_idx + 1]
        rows.append(
            {
                "candidate_name": candidate_name,
                "short_entry_time": str(signals.index[entry_idx]),
                "short_exit_time": str(signals.index[exit_idx]),
                "short_trade_return": float((1.0 + trade_returns).prod() - 1.0),
                "btc_return_during_short": float((1.0 + trade_btc).prod() - 1.0),
                "holding_period_bars": int(exit_idx - entry_idx + 1),
                "exit_reason": "end_of_sample",
            }
        )

    return pd.DataFrame(rows)


def run_single_long_short_candidate(
    close_prices: pd.DataFrame,
    bars_per_year: int,
    candidate_name: str,
    short_lookback_bars: int,
    medium_lookback_bars: int,
    rebalance_every_bars: int,
    short_threshold: float,
    medium_threshold: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> LongShortBacktestArtifacts:
    close = _validate_close_prices(close_prices, symbol=close_prices.columns[0])
    btc_returns = close.iloc[:, 0].pct_change().fillna(0.0).astype(float)

    target_position = _build_target_positions(
        close=close,
        short_lookback_bars=int(short_lookback_bars),
        medium_lookback_bars=int(medium_lookback_bars),
        rebalance_every_bars=int(rebalance_every_bars),
        short_threshold=float(short_threshold),
        medium_threshold=float(medium_threshold),
    )
    position = target_position.shift(1).fillna(0.0).astype(float)

    position_change = position.diff().abs().fillna(position.abs()).astype(float)
    per_turnover_cost = float(transaction_cost_bps + slippage_bps) / 10_000.0
    transaction_cost = position_change * per_turnover_cost

    gross_return = position * btc_returns
    strategy_return = gross_return - transaction_cost

    signals = pd.DataFrame(
        {
            "btc_return": btc_returns,
            "target_position": target_position,
            "position": position,
            "position_change": position_change,
            "transaction_cost": transaction_cost,
            "gross_return": gross_return,
            "strategy_return": strategy_return,
            "long_return_component": (position > 0.0).astype(float) * gross_return,
            "short_return_component": (position < 0.0).astype(float) * gross_return,
        },
        index=close.index,
    )

    equity = (1.0 + signals["strategy_return"]).cumprod()
    metrics = summary_metrics(
        equity=equity,
        returns=signals["strategy_return"],
        bars_per_year=bars_per_year,
    )

    annual_rows: list[dict[str, Any]] = []
    for year in sorted(pd.Index(signals.index.year).unique().tolist()):
        mask = signals.index.year == int(year)
        year_slice = signals.loc[mask]
        if len(year_slice) < 2:
            continue

        year_equity = (1.0 + year_slice["strategy_return"]).cumprod()
        year_metrics = summary_metrics(
            equity=year_equity,
            returns=year_slice["strategy_return"],
            bars_per_year=bars_per_year,
        )

        annual_rows.append(
            {
                "year": int(year),
                "candidate_name": candidate_name,
                "strategy_return": float(year_metrics["total_return"]),
                "sharpe": float(year_metrics["sharpe"]),
                "max_drawdown": float(year_metrics["max_drawdown"]),
                "percent_time_long": float((year_slice["position"] > 0.0).mean()),
                "percent_time_short": float((year_slice["position"] < 0.0).mean()),
                "percent_time_cash": float((year_slice["position"] == 0.0).mean()),
                "short_side_pnl_contribution": float(year_slice["short_return_component"].sum()),
                "long_side_pnl_contribution": float(year_slice["long_return_component"].sum()),
            }
        )

    short_trades = _extract_short_trades(candidate_name=candidate_name, signals=signals)

    return LongShortBacktestArtifacts(
        candidate_name=candidate_name,
        signals=signals,
        annual_report=pd.DataFrame(annual_rows),
        short_trades=short_trades,
    )


def _trade_episode_returns(signals: pd.DataFrame, side: float) -> list[float]:
    position = signals["position"].astype(float)
    returns = signals["strategy_return"].astype(float)

    out: list[float] = []
    start_idx: int | None = None

    for i, matches in enumerate((position == side).tolist()):
        if matches and start_idx is None:
            start_idx = i
            continue

        if not matches and start_idx is not None:
            segment = returns.iloc[start_idx:i]
            out.append(float((1.0 + segment).prod() - 1.0))
            start_idx = None

    if start_idx is not None:
        segment = returns.iloc[start_idx:]
        out.append(float((1.0 + segment).prod() - 1.0))

    return out


def run_btc_long_short_momentum_research(
    config: BtcLongShortResearchConfig | None = None,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    by_year_output_path: Path = DEFAULT_BY_YEAR_OUTPUT_PATH,
    short_trades_output_path: Path = DEFAULT_SHORT_TRADES_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or BtcLongShortResearchConfig()
    close = (
        _validate_close_prices(close_prices, symbol=cfg.symbol)
        if close_prices is not None
        else _load_btc_close_prices(symbol=cfg.symbol, timeframe=cfg.timeframe, data_dir=data_dir)
    )
    bars_per_year = bars_per_year_for_timeframe(cfg.timeframe)

    btc_returns = close[cfg.symbol].pct_change().fillna(0.0).astype(float)

    long_cash_cfg = BtcTimeSeriesConfig(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        short_lookback_bars=int(cfg.short_lookback_bars),
        medium_lookback_bars=int(cfg.medium_lookback_bars),
        rebalance_every_bars=int(cfg.rebalance_every_bars),
        short_weight=0.5,
        medium_weight=0.5,
        transaction_cost_bps=float(cfg.transaction_cost_bps),
        slippage_bps=float(cfg.slippage_bps),
    )
    _, _, benchmark_map = run_btc_time_series_backtests(
        config=long_cash_cfg,
        close_prices=close,
        data_dir=data_dir,
    )
    long_cash_returns = benchmark_map["btc_time_series_momentum"].portfolio["strategy_return"].reindex(close.index).fillna(0.0)

    summary_rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []
    short_trade_tables: list[pd.DataFrame] = []

    for spec in tuple(cfg.thresholds):
        artifacts = run_single_long_short_candidate(
            close_prices=close,
            bars_per_year=bars_per_year,
            candidate_name=spec.name,
            short_lookback_bars=int(cfg.short_lookback_bars),
            medium_lookback_bars=int(cfg.medium_lookback_bars),
            rebalance_every_bars=int(cfg.rebalance_every_bars),
            short_threshold=float(spec.short_threshold),
            medium_threshold=float(spec.medium_threshold),
            transaction_cost_bps=float(cfg.transaction_cost_bps),
            slippage_bps=float(cfg.slippage_bps),
        )

        signals = artifacts.signals
        equity = (1.0 + signals["strategy_return"]).cumprod()
        metrics = summary_metrics(
            equity=equity,
            returns=signals["strategy_return"],
            bars_per_year=bars_per_year,
        )

        long_trades = _trade_episode_returns(signals, side=1.0)
        short_trades = _trade_episode_returns(signals, side=-1.0)

        summary_rows.append(
            {
                "candidate_name": spec.name,
                "short_lookback": int(cfg.short_lookback_bars),
                "medium_lookback": int(cfg.medium_lookback_bars),
                "rebalance_every": int(cfg.rebalance_every_bars),
                "short_threshold": float(spec.short_threshold),
                "medium_threshold": float(spec.medium_threshold),
                "total_return": float(metrics["total_return"]),
                "cagr": float(metrics["cagr"]),
                "annualized_return": float(metrics["cagr"]),
                "annualized_volatility": float(metrics["annualized_volatility"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "worst_month_return": float((1.0 + signals["strategy_return"]).resample("ME").prod().sub(1.0).min()),
                "number_of_trades": int((signals["position_change"] > 1e-12).sum()),
                "number_of_long_entries": int(((signals["position"] == 1.0) & (signals["position"].shift(1).fillna(0.0) != 1.0)).sum()),
                "number_of_short_entries": int(((signals["position"] == -1.0) & (signals["position"].shift(1).fillna(0.0) != -1.0)).sum()),
                "number_of_cash_periods": int((signals["position"] == 0.0).sum()),
                "percent_time_long": float((signals["position"] > 0.0).mean()),
                "percent_time_short": float((signals["position"] < 0.0).mean()),
                "percent_time_cash": float((signals["position"] == 0.0).mean()),
                "average_long_trade_return": float(pd.Series(long_trades, dtype=float).mean()) if long_trades else 0.0,
                "average_short_trade_return": float(pd.Series(short_trades, dtype=float).mean()) if short_trades else 0.0,
                "short_trade_win_rate": float((pd.Series(short_trades, dtype=float) > 0.0).mean()) if short_trades else 0.0,
                "worst_short_trade_return": float(min(short_trades)) if short_trades else 0.0,
                "best_short_trade_return": float(max(short_trades)) if short_trades else 0.0,
                "total_transaction_costs": float(signals["transaction_cost"].sum()),
                "vs_btc_buy_hold_total_return": float(metrics["total_return"] - ((1.0 + btc_returns).prod() - 1.0)),
                "vs_btc_long_cash_total_return": float(metrics["total_return"] - ((1.0 + long_cash_returns).prod() - 1.0)),
                "vs_cash_total_return": float(metrics["total_return"]),
            }
        )

        per_year = artifacts.annual_report.copy()
        if not per_year.empty:
            year_to_btc = (
                (1.0 + btc_returns)
                .groupby(btc_returns.index.year)
                .prod()
                .sub(1.0)
                .rename("btc_buy_hold_return")
                .rename_axis("year")
            )
            year_to_long_cash = (
                (1.0 + long_cash_returns)
                .groupby(long_cash_returns.index.year)
                .prod()
                .sub(1.0)
                .rename("btc_long_cash_return")
                .rename_axis("year")
            )

            per_year = per_year.merge(year_to_btc.to_frame().reset_index(), on="year", how="left")
            per_year = per_year.merge(year_to_long_cash.to_frame().reset_index(), on="year", how="left")
            per_year["excess_return_vs_btc"] = per_year["strategy_return"] - per_year["btc_buy_hold_return"]
            per_year["excess_return_vs_long_cash"] = per_year["strategy_return"] - per_year["btc_long_cash_return"]
            by_year_rows.extend(per_year.to_dict(orient="records"))

        if not artifacts.short_trades.empty:
            short_trade_tables.append(artifacts.short_trades)

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    by_year = pd.DataFrame(by_year_rows).sort_values(["year", "candidate_name"]).reset_index(drop=True)
    short_trades = (
        pd.concat(short_trade_tables, axis=0, ignore_index=True)
        if short_trade_tables
        else pd.DataFrame(
            columns=[
                "candidate_name",
                "short_entry_time",
                "short_exit_time",
                "short_trade_return",
                "btc_return_during_short",
                "holding_period_bars",
                "exit_reason",
            ]
        )
    )

    if save_csv:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        by_year_output_path.parent.mkdir(parents=True, exist_ok=True)
        short_trades_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output_path, index=False)
        by_year.to_csv(by_year_output_path, index=False)
        short_trades.to_csv(short_trades_output_path, index=False)

    return summary, by_year, short_trades
