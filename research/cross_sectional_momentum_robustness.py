"""Cross-sectional momentum robustness report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import summary_metrics
from config import SETTINGS
from research.run_expanded_universe_experiment import (
    EXPANDED_UNIVERSE_20,
    LOOKBACK_CONFIGS,
    _rebalance_signal_generator,
    bars_per_year_for_timeframe,
    build_universal_eligibility_mask,
    close_prices_to_ohlcv,
    load_universe_close_prices,
)


DEFAULT_OUTPUT_PATH = Path("reports/cross_sectional_momentum_robustness_by_year.csv")


@dataclass(frozen=True)
class CrossSectionalRobustnessSpec:
    """One CS candidate configuration for robustness analysis."""

    name: str
    lookback_config_name: str
    top_n: int
    rebalance_bars: int
    fee_bps: int = 10
    slippage_bps: int = 5


DEFAULT_CS_ROBUSTNESS_STRATEGIES: tuple[CrossSectionalRobustnessSpec, ...] = (
    CrossSectionalRobustnessSpec(
        name="cs_42_180_top3_reb6",
        lookback_config_name="short_42_medium_180",
        top_n=3,
        rebalance_bars=6,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_180_top5_reb12",
        lookback_config_name="medium_180_only",
        top_n=5,
        rebalance_bars=12,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_42_180_top3_reb12",
        lookback_config_name="short_42_medium_180",
        top_n=3,
        rebalance_bars=12,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_180_top8_reb42",
        lookback_config_name="medium_180_only",
        top_n=8,
        rebalance_bars=42,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_42_180_top5_reb42",
        lookback_config_name="short_42_medium_180",
        top_n=5,
        rebalance_bars=42,
    ),
)


def _worst_drawdown_period(equity: pd.Series) -> tuple[str | None, str | None]:
    """Return timestamps of peak and trough for the worst drawdown in a slice."""
    clean = equity.dropna().astype(float)
    if len(clean) < 2:
        return None, None

    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    trough_ts = drawdown.idxmin()
    if pd.isna(trough_ts):
        return None, None

    peak_equity = running_max.loc[trough_ts]
    peak_candidates = clean.loc[:trough_ts]
    peak_ts = peak_candidates[peak_candidates == peak_equity].index[0]
    return str(peak_ts), str(trough_ts)


def _slice_metrics_row(
    year: str | int,
    strategy: str,
    execution: BacktestResult,
    bars_per_year: int,
    mask: pd.Series,
) -> dict[str, float | int | str] | None:
    """Compute one metrics row for a year/full-period slice."""
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
    worst_start, worst_end = _worst_drawdown_period(portfolio["equity"])

    return {
        "year": str(year),
        "strategy": strategy,
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "num_trades": int((turnover > 1e-12).sum()),
        "turnover": float(turnover.sum()),
        "percent_time_invested": float(holdings.gt(0.0).any(axis=1).mean()),
        "worst_drawdown_start": worst_start,
        "worst_drawdown_end": worst_end,
    }


def _equal_weight_benchmark_signal_generator(
    close_prices: pd.DataFrame,
    min_required_lookback: int,
) -> callable:
    """Build a simple equal-weight benchmark that respects universal eligibility."""
    eligibility = build_universal_eligibility_mask(close_prices, min_required_lookback=min_required_lookback)

    def signal_generator(close: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        if timestamp not in close.index:
            raise KeyError(f"timestamp not found in close matrix: {timestamp}")
        eligible_row = eligibility.loc[timestamp]
        selected = eligible_row[eligible_row].index.tolist()
        row = pd.Series(0.0, index=close.columns, dtype=float)
        if selected:
            row.loc[selected] = 1.0 / len(selected)
        return row

    return signal_generator


def _load_close_prices(
    symbols: Sequence[str],
    timeframe: str,
    data_dir: Path,
    close_prices: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load or validate close-price matrix for robustness analysis."""
    if close_prices is not None:
        close = close_prices.sort_index().copy()
        close.index = pd.to_datetime(close.index, utc=True)
        return close.astype(float)
    return load_universe_close_prices(symbols=symbols, timeframe=timeframe, data_dir=data_dir)


def run_cross_sectional_momentum_robustness(
    strategies: Sequence[CrossSectionalRobustnessSpec] = DEFAULT_CS_ROBUSTNESS_STRATEGIES,
    symbols: Sequence[str] = EXPANDED_UNIVERSE_20,
    timeframe: str = "4h",
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run yearly and full-period robustness report for CS momentum candidates."""
    if not strategies:
        raise ValueError("strategies must not be empty")

    close = _load_close_prices(symbols=symbols, timeframe=timeframe, data_dir=data_dir, close_prices=close_prices)
    ohlcv = close_prices_to_ohlcv(close)
    bars_per_year = bars_per_year_for_timeframe(timeframe)
    years = sorted(close.index.year.unique().tolist())
    full_mask = pd.Series(True, index=close.index)
    rows: list[dict[str, float | int | str]] = []

    max_required_lookback = max(LOOKBACK_CONFIGS[spec.lookback_config_name].min_required_lookback for spec in strategies)
    benchmark_result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=_equal_weight_benchmark_signal_generator(close, min_required_lookback=max_required_lookback),
        initial_capital=SETTINGS.initial_capital,
        transaction_cost_bps=float(strategies[0].fee_bps),
        slippage_bps=float(strategies[0].slippage_bps),
        rebalance_every_bars=1,
    )

    for year in years:
        year_mask = pd.Series(close.index.year == int(year), index=close.index)
        row = _slice_metrics_row(
            year=int(year),
            strategy="equal_weight_dynamic_benchmark",
            execution=benchmark_result,
            bars_per_year=bars_per_year,
            mask=year_mask,
        )
        if row is not None:
            rows.append(row)
    full_row = _slice_metrics_row(
        year="FULL",
        strategy="equal_weight_dynamic_benchmark",
        execution=benchmark_result,
        bars_per_year=bars_per_year,
        mask=full_mask,
    )
    if full_row is not None:
        rows.append(full_row)

    for spec in strategies:
        lookback_config = LOOKBACK_CONFIGS[spec.lookback_config_name]
        eligibility_mask = build_universal_eligibility_mask(close, min_required_lookback=lookback_config.min_required_lookback)
        signal_generator = _rebalance_signal_generator(
            strategy_name="cs_momentum",
            lookback_config=lookback_config,
            close_prices=close,
            eligibility_mask=eligibility_mask,
            top_n=spec.top_n,
        )
        execution = run_backtest(
            ohlcv=ohlcv,
            signal_generator=signal_generator,
            initial_capital=SETTINGS.initial_capital,
            transaction_cost_bps=float(spec.fee_bps),
            slippage_bps=float(spec.slippage_bps),
            rebalance_every_bars=int(spec.rebalance_bars),
        )

        for year in years:
            year_mask = pd.Series(close.index.year == int(year), index=close.index)
            row = _slice_metrics_row(
                year=int(year),
                strategy=spec.name,
                execution=execution,
                bars_per_year=bars_per_year,
                mask=year_mask,
            )
            if row is not None:
                rows.append(row)

        full_row = _slice_metrics_row(
            year="FULL",
            strategy=spec.name,
            execution=execution,
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
