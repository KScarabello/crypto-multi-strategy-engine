"""Year-by-year robustness report for selected BTC plus CS sleeve blends."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import SETTINGS
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.run_expanded_universe_experiment import EXPANDED_UNIVERSE_20
from research.sleeve_blend_experiment import align_sleeve_return_streams, run_sleeve_backtests
from backtest.metrics import summary_metrics


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_yearly_robustness.csv")
YEARLY_BLEND_CS_CANDIDATE = CrossSectionalRobustnessSpec(
    name="cs_180_top5_reb12",
    lookback_config_name="medium_180_only",
    top_n=5,
    rebalance_bars=12,
)
YEARLY_BLEND_ALLOCATIONS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.9, 0.1),
    (0.85, 0.15),
    (0.8, 0.2),
    (0.75, 0.25),
)


def _validate_allocations(allocations: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Validate the requested yearly robustness blend set."""
    validated: list[tuple[float, float]] = []
    if not allocations:
        raise ValueError("allocations must not be empty")
    for btc_weight, cs_weight in allocations:
        btc = float(btc_weight)
        cs = float(cs_weight)
        if btc < 0.0 or cs < 0.0:
            raise ValueError("weights must be non-negative")
        if abs((btc + cs) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")
        validated.append((btc, cs))
    return tuple(validated)


def build_yearly_blend_allocations() -> tuple[tuple[float, float], ...]:
    """Return the requested top focused blend set for yearly comparison."""
    return _validate_allocations(YEARLY_BLEND_ALLOCATIONS)


def _worst_month_return(returns: pd.Series) -> float:
    """Return the worst compounded monthly return inside the slice."""
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    monthly = (1.0 + clean).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float(monthly.min())


def _slice_blend_metrics_row(
    year: str | int,
    btc_weight: float,
    cs_weight: float,
    cs_candidate: str,
    aligned_returns: pd.DataFrame,
    bars_per_year: int,
) -> dict[str, float | str] | None:
    """Compute metrics for one yearly or full-period slice from aligned full-run returns."""
    if len(aligned_returns) < 2:
        return None

    blend_returns = (
        aligned_returns["btc_sleeve_return"] * float(btc_weight)
        + aligned_returns["cs_sleeve_return"] * float(cs_weight)
    )
    equity = SETTINGS.initial_capital * (1.0 + blend_returns).cumprod()
    metrics = summary_metrics(
        equity=equity,
        returns=blend_returns,
        bars_per_year=bars_per_year,
    )

    return {
        "year": str(year),
        "btc_strategy": "btc_ts_60_240_12",
        "btc_weight": float(btc_weight),
        "cs_weight": float(cs_weight),
        "cs_candidate": cs_candidate,
        "blend_name": f"btc_ts_60_240_12__btc_{int(round(btc_weight * 100)):02d}_cs_{int(round(cs_weight * 100)):02d}__{cs_candidate}",
        "total_return": float(metrics.get("total_return", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_month_return": _worst_month_return(blend_returns),
        "ending_equity": float(equity.iloc[-1]),
        "growth_of_1": float(equity.iloc[-1] / float(SETTINGS.initial_capital)),
        "start_date": str(equity.index.min()),
        "end_date": str(equity.index.max()),
    }


def run_sleeve_blend_yearly_robustness(
    cs_candidate: CrossSectionalRobustnessSpec = YEARLY_BLEND_CS_CANDIDATE,
    allocations: Sequence[tuple[float, float]] = YEARLY_BLEND_ALLOCATIONS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run yearly and full-period robustness rows for the requested top blends."""
    allocation_grid = _validate_allocations(allocations)
    close, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(cs_candidate,),
        symbols=EXPANDED_UNIVERSE_20,
        timeframe="4h",
        close_prices=close_prices,
        data_dir=data_dir,
    )
    aligned = align_sleeve_return_streams(
        btc_returns=btc_result.portfolio["strategy_return"],
        cs_returns=cs_results[cs_candidate.name].portfolio["strategy_return"],
    )
    years = sorted(aligned.index.year.unique().tolist())
    rows: list[dict[str, float | str]] = []

    for btc_weight, cs_weight in allocation_grid:
        for year in years:
            mask = pd.Series(aligned.index.year == int(year), index=aligned.index)
            yearly_aligned = aligned.loc[mask]
            row = _slice_blend_metrics_row(
                year=int(year),
                btc_weight=btc_weight,
                cs_weight=cs_weight,
                cs_candidate=cs_candidate.name,
                aligned_returns=yearly_aligned,
                bars_per_year=bars_per_year,
            )
            if row is not None:
                rows.append(row)

        full_row = _slice_blend_metrics_row(
            year="FULL",
            btc_weight=btc_weight,
            cs_weight=cs_weight,
            cs_candidate=cs_candidate.name,
            aligned_returns=aligned,
            bars_per_year=bars_per_year,
        )
        if full_row is not None:
            rows.append(full_row)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No yearly robustness metrics produced")

    year_order = {str(year): idx for idx, year in enumerate(years)}
    year_order["FULL"] = len(years)
    report["_year_order"] = report["year"].astype(str).map(year_order)
    report = report.sort_values(["_year_order", "btc_weight"], ascending=[True, False]).drop(columns=["_year_order"])
    report = report.reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report