"""BTC-only time-series robustness report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import summary_metrics
from config import SETTINGS
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_backtests


DEFAULT_OUTPUT_PATH = Path("reports/btc_time_series_robustness_by_year.csv")


@dataclass(frozen=True)
class RobustnessStrategySpec:
    """One strategy specification included in robustness analysis."""

    name: str
    config: BtcTimeSeriesConfig


DEFAULT_ROBUSTNESS_STRATEGIES: tuple[RobustnessStrategySpec, ...] = (
    RobustnessStrategySpec(
        name="btc_ts_candidate_60_240_12",
        config=BtcTimeSeriesConfig(short_lookback_bars=60, medium_lookback_bars=240, rebalance_every_bars=12),
    ),
    RobustnessStrategySpec(
        name="btc_ts_baseline_42_180_6",
        config=BtcTimeSeriesConfig(short_lookback_bars=42, medium_lookback_bars=180, rebalance_every_bars=6),
    ),
)


def _slice_metrics_row(
    strategy: str,
    year: str | int,
    execution: BacktestResult,
    symbol: str,
    bars_per_year: int,
    mask: pd.Series,
) -> dict[str, float | int | str] | None:
    """Compute one metrics row from a boolean time mask."""
    portfolio = execution.portfolio.loc[mask]
    if len(portfolio) < 2:
        return None

    turnover = execution.turnover.reindex(portfolio.index).fillna(0.0)
    holdings = execution.holdings_history.reindex(portfolio.index).fillna(0.0)

    metrics = summary_metrics(
        equity=portfolio["equity"],
        returns=portfolio["strategy_return"],
        turnover=turnover,
        bars_per_year=bars_per_year,
    )

    num_trades = int((turnover > 1e-12).sum())
    invested = holdings[symbol] if symbol in holdings.columns else pd.Series(0.0, index=portfolio.index)

    return {
        "year": str(year),
        "strategy": strategy,
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        # For yearly rows this is literal in-year return; for FULL it equals full-sample return.
        "annual_return": float(metrics.get("total_return", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "num_trades": num_trades,
        "percent_time_invested": float((invested > 0.0).mean()),
    }


def run_btc_time_series_robustness(
    strategies: Sequence[RobustnessStrategySpec] = DEFAULT_ROBUSTNESS_STRATEGIES,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run yearly and full-period robustness report for BTC strategies."""
    if not strategies:
        raise ValueError("strategies must not be empty")

    close_cache = close_prices
    rows: list[dict[str, float | int | str]] = []

    # Run buy-and-hold once using the first strategy config cost assumptions.
    base_cfg = strategies[0].config
    close, bars_per_year, base_results = run_btc_time_series_backtests(
        config=base_cfg,
        close_prices=close_cache,
        data_dir=data_dir,
    )
    close_cache = close
    buy_hold = base_results["btc_buy_and_hold"]
    symbol = base_cfg.symbol

    years = sorted(close.index.year.unique().tolist())

    for year in years:
        mask = pd.Series(close.index.year == int(year), index=close.index)
        row = _slice_metrics_row(
            strategy="btc_buy_and_hold",
            year=int(year),
            execution=buy_hold,
            symbol=symbol,
            bars_per_year=bars_per_year,
            mask=mask,
        )
        if row is not None:
            rows.append(row)

    full_mask = pd.Series(True, index=close.index)
    full_row = _slice_metrics_row(
        strategy="btc_buy_and_hold",
        year="FULL",
        execution=buy_hold,
        symbol=symbol,
        bars_per_year=bars_per_year,
        mask=full_mask,
    )
    if full_row is not None:
        rows.append(full_row)

    for spec in strategies:
        _, _, result_map = run_btc_time_series_backtests(
            config=spec.config,
            close_prices=close_cache,
            data_dir=data_dir,
        )
        execution = result_map["btc_time_series_momentum"]

        for year in years:
            mask = pd.Series(close.index.year == int(year), index=close.index)
            row = _slice_metrics_row(
                strategy=spec.name,
                year=int(year),
                execution=execution,
                symbol=spec.config.symbol,
                bars_per_year=bars_per_year,
                mask=mask,
            )
            if row is not None:
                rows.append(row)

        full_row = _slice_metrics_row(
            strategy=spec.name,
            year="FULL",
            execution=execution,
            symbol=spec.config.symbol,
            bars_per_year=bars_per_year,
            mask=full_mask,
        )
        if full_row is not None:
            rows.append(full_row)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No robustness metrics produced")

    report = report.sort_values(["year", "strategy"], key=lambda s: s.astype(str)).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    return report
