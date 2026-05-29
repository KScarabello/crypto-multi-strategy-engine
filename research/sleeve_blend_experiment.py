"""Research-only portfolio blend experiment for BTC TS plus CS sleeves."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import summary_metrics
from config import SETTINGS
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_backtests
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.run_expanded_universe_experiment import (
    EXPANDED_UNIVERSE_20,
    LOOKBACK_CONFIGS,
    _rebalance_signal_generator,
    bars_per_year_for_timeframe,
    build_universal_eligibility_mask,
    close_prices_to_ohlcv,
    load_universe_close_prices,
)


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_experiment_metrics.csv")
LOCKED_BTC_SLEEVE = BtcTimeSeriesConfig(
    short_lookback_bars=60,
    medium_lookback_bars=240,
    rebalance_every_bars=12,
)
DEFAULT_BLEND_ALLOCATIONS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.9, 0.1),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
)
DEFAULT_BLEND_CS_CANDIDATES: tuple[CrossSectionalRobustnessSpec, ...] = (
    CrossSectionalRobustnessSpec(
        name="cs_42_180_top5_reb42",
        lookback_config_name="short_42_medium_180",
        top_n=5,
        rebalance_bars=42,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_180_top8_reb42",
        lookback_config_name="medium_180_only",
        top_n=8,
        rebalance_bars=42,
    ),
    CrossSectionalRobustnessSpec(
        name="cs_180_top5_reb12",
        lookback_config_name="medium_180_only",
        top_n=5,
        rebalance_bars=12,
    ),
)


def _validate_close_prices(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the close-price matrix for the blend experiment."""
    if not isinstance(close_prices, pd.DataFrame):
        raise TypeError("close_prices must be a pandas DataFrame")
    if close_prices.empty:
        raise ValueError("close_prices is empty")
    if "BTC/USD" not in close_prices.columns:
        raise ValueError("close_prices must include BTC/USD")

    clean = close_prices.sort_index().copy()
    clean.index = pd.to_datetime(clean.index, utc=True)
    clean = clean.astype(float)
    clean = clean.dropna(how="all")
    if clean.empty:
        raise ValueError("close_prices did not contain any valid observations")
    return clean


def build_blend_allocation_grid(
    allocations: Sequence[tuple[float, float]] = DEFAULT_BLEND_ALLOCATIONS,
) -> tuple[tuple[float, float], ...]:
    """Validate and normalize the requested BTC/CS blend allocation grid."""
    if not allocations:
        raise ValueError("allocations must not be empty")

    validated: list[tuple[float, float]] = []
    for btc_weight, cs_weight in allocations:
        btc = float(btc_weight)
        cs = float(cs_weight)
        if btc < 0.0 or cs < 0.0:
            raise ValueError("blend weights must be non-negative")
        if not math.isclose(btc + cs, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("blend weights must sum to 1.0")
        validated.append((btc, cs))
    return tuple(validated)


def align_sleeve_return_streams(
    btc_returns: pd.Series,
    cs_returns: pd.Series,
) -> pd.DataFrame:
    """Align BTC and CS sleeve return streams on their common timestamps."""
    if not isinstance(btc_returns, pd.Series) or not isinstance(cs_returns, pd.Series):
        raise TypeError("btc_returns and cs_returns must be pandas Series")

    aligned = pd.concat(
        [
            btc_returns.rename("btc_sleeve_return"),
            cs_returns.rename("cs_sleeve_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping return observations between BTC and CS sleeves")
    aligned.index = pd.to_datetime(aligned.index, utc=True)
    return aligned.sort_index()


def _run_cs_candidate_backtest(
    close_prices: pd.DataFrame,
    candidate: CrossSectionalRobustnessSpec,
) -> BacktestResult:
    """Run one CS candidate sleeve backtest on the provided universe close matrix."""
    lookback_config = LOOKBACK_CONFIGS[candidate.lookback_config_name]
    eligibility_mask = build_universal_eligibility_mask(
        close_prices,
        min_required_lookback=lookback_config.min_required_lookback,
    )
    signal_generator = _rebalance_signal_generator(
        strategy_name="cs_momentum",
        lookback_config=lookback_config,
        close_prices=close_prices,
        eligibility_mask=eligibility_mask,
        top_n=candidate.top_n,
    )
    return run_backtest(
        ohlcv=close_prices_to_ohlcv(close_prices),
        signal_generator=signal_generator,
        initial_capital=SETTINGS.initial_capital,
        transaction_cost_bps=float(candidate.fee_bps),
        slippage_bps=float(candidate.slippage_bps),
        rebalance_every_bars=int(candidate.rebalance_bars),
    )


def run_sleeve_backtests(
    cs_candidates: Sequence[CrossSectionalRobustnessSpec] = DEFAULT_BLEND_CS_CANDIDATES,
    symbols: Sequence[str] = EXPANDED_UNIVERSE_20,
    timeframe: str = "4h",
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
) -> tuple[pd.DataFrame, int, BacktestResult, dict[str, BacktestResult]]:
    """Run the locked BTC sleeve and each CS sleeve candidate once."""
    if not cs_candidates:
        raise ValueError("cs_candidates must not be empty")

    close = (
        _validate_close_prices(close_prices)
        if close_prices is not None
        else _validate_close_prices(load_universe_close_prices(symbols=symbols, timeframe=timeframe, data_dir=data_dir))
    )
    btc_close = close[[LOCKED_BTC_SLEEVE.symbol]].copy()
    _, bars_per_year, btc_results = run_btc_time_series_backtests(
        config=LOCKED_BTC_SLEEVE,
        close_prices=btc_close,
        data_dir=data_dir,
    )

    cs_results = {
        candidate.name: _run_cs_candidate_backtest(close_prices=close, candidate=candidate)
        for candidate in cs_candidates
    }
    return close, bars_per_year, btc_results["btc_time_series_momentum"], cs_results


def _blend_row(
    year: str,
    candidate_name: str,
    btc_weight: float,
    cs_weight: float,
    aligned_returns: pd.DataFrame,
    bars_per_year: int,
    worst_year_return: float,
) -> dict[str, float | str]:
    """Build one metrics row for a blended sleeve return stream."""
    blend_returns = (
        aligned_returns["btc_sleeve_return"] * btc_weight
        + aligned_returns["cs_sleeve_return"] * cs_weight
    )
    equity = SETTINGS.initial_capital * (1.0 + blend_returns).cumprod()
    metrics = summary_metrics(
        equity=equity,
        returns=blend_returns,
        bars_per_year=bars_per_year,
    )
    correlation = float(aligned_returns["btc_sleeve_return"].corr(aligned_returns["cs_sleeve_return"]))
    if pd.isna(correlation):
        correlation = 0.0

    btc_pct = int(round(btc_weight * 100))
    cs_pct = int(round(cs_weight * 100))
    return {
        "year": year,
        "btc_strategy": "btc_ts_60_240_12",
        "cs_strategy": candidate_name,
        "blend_name": f"btc_ts_60_240_12__btc_{btc_pct:02d}_cs_{cs_pct:02d}__{candidate_name}",
        "btc_weight": float(btc_weight),
        "cs_weight": float(cs_weight),
        "total_return": float(metrics.get("total_return", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_year_return": float(worst_year_return),
        "sleeve_return_correlation": correlation,
        "start_date": str(equity.index.min()),
        "end_date": str(equity.index.max()),
    }


def run_sleeve_blend_experiment(
    cs_candidates: Sequence[CrossSectionalRobustnessSpec] = DEFAULT_BLEND_CS_CANDIDATES,
    allocations: Sequence[tuple[float, float]] = DEFAULT_BLEND_ALLOCATIONS,
    symbols: Sequence[str] = EXPANDED_UNIVERSE_20,
    timeframe: str = "4h",
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run yearly and full-period portfolio blends for BTC plus CS sleeves."""
    allocation_grid = build_blend_allocation_grid(allocations)
    _, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=cs_candidates,
        symbols=symbols,
        timeframe=timeframe,
        close_prices=close_prices,
        data_dir=data_dir,
    )

    btc_returns = btc_result.portfolio["strategy_return"].copy()
    rows: list[dict[str, float | str]] = []

    for candidate in cs_candidates:
        candidate_result = cs_results[candidate.name]
        aligned = align_sleeve_return_streams(
            btc_returns=btc_returns,
            cs_returns=candidate_result.portfolio["strategy_return"],
        )
        year_masks = {
            str(year): pd.Series(aligned.index.year == int(year), index=aligned.index)
            for year in sorted(aligned.index.year.unique().tolist())
        }

        for btc_weight, cs_weight in allocation_grid:
            yearly_rows: list[dict[str, float | str]] = []
            annual_returns: list[float] = []

            for year, mask in year_masks.items():
                yearly_aligned = aligned.loc[mask]
                if len(yearly_aligned) < 2:
                    continue
                year_row = _blend_row(
                    year=year,
                    candidate_name=candidate.name,
                    btc_weight=btc_weight,
                    cs_weight=cs_weight,
                    aligned_returns=yearly_aligned,
                    bars_per_year=bars_per_year,
                    worst_year_return=0.0,
                )
                year_row["worst_year_return"] = float(year_row["annual_return"])
                annual_returns.append(float(year_row["annual_return"]))
                yearly_rows.append(year_row)

            if len(aligned) < 2:
                continue

            worst_year_return = min(annual_returns) if annual_returns else 0.0
            full_row = _blend_row(
                year="FULL",
                candidate_name=candidate.name,
                btc_weight=btc_weight,
                cs_weight=cs_weight,
                aligned_returns=aligned,
                bars_per_year=bars_per_year,
                worst_year_return=worst_year_return,
            )
            rows.extend(yearly_rows)
            rows.append(full_row)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No blend metrics produced")

    report = report.sort_values(
        ["cs_strategy", "btc_weight", "cs_weight", "year"],
        ascending=[True, False, True, True],
        key=lambda s: s.astype(str) if s.name == "year" else s,
    ).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report