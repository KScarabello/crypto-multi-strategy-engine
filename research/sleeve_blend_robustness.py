"""Focused robustness sweep for BTC plus CS sleeve blends."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import SETTINGS
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.run_expanded_universe_experiment import EXPANDED_UNIVERSE_20
from research.sleeve_blend_experiment import (
    build_blend_allocation_grid,
    run_sleeve_backtests,
    run_sleeve_blend_experiment,
)


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_robustness_metrics.csv")
FOCUSED_BLEND_ALLOCATIONS: tuple[tuple[float, float], ...] = tuple(
    (round(btc_weight / 100.0, 2), round(1.0 - btc_weight / 100.0, 2))
    for btc_weight in range(100, 69, -5)
)
FOCUSED_BLEND_CS_CANDIDATES: tuple[CrossSectionalRobustnessSpec, ...] = (
    CrossSectionalRobustnessSpec(
        name="cs_180_top5_reb12",
        lookback_config_name="medium_180_only",
        top_n=5,
        rebalance_bars=12,
    ),
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
)


def build_focused_blend_allocation_grid() -> tuple[tuple[float, float], ...]:
    """Return the requested focused BTC/CS allocation sweep."""
    return build_blend_allocation_grid(FOCUSED_BLEND_ALLOCATIONS)


def _btc_baseline_metrics(
    close_prices: pd.DataFrame | None,
    data_dir: Path,
) -> dict[str, float]:
    """Run the locked BTC sleeve once and return full-period baseline metrics."""
    _, bars_per_year, btc_result, _ = run_sleeve_backtests(
        cs_candidates=(FOCUSED_BLEND_CS_CANDIDATES[0],),
        symbols=EXPANDED_UNIVERSE_20,
        timeframe="4h",
        close_prices=close_prices,
        data_dir=data_dir,
    )
    portfolio = btc_result.portfolio
    return {
        "total_return": float(portfolio["equity"].iloc[-1] / portfolio["equity"].iloc[0] - 1.0),
        "cagr": float(
            ((portfolio["equity"].iloc[-1] / portfolio["equity"].iloc[0]) ** (1.0 / ((len(portfolio) - 1) / bars_per_year))) - 1.0
        ),
        "sharpe": float(portfolio["strategy_return"].mean() / portfolio["strategy_return"].std(ddof=0) * (bars_per_year ** 0.5))
        if float(portfolio["strategy_return"].std(ddof=0)) > 0.0
        else 0.0,
        "max_drawdown": float((portfolio["equity"] / portfolio["equity"].cummax() - 1.0).min()),
    }


def run_sleeve_blend_robustness(
    cs_candidates: Sequence[CrossSectionalRobustnessSpec] = FOCUSED_BLEND_CS_CANDIDATES,
    allocations: Sequence[tuple[float, float]] = FOCUSED_BLEND_ALLOCATIONS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the focused full-period blend sweep and add delta columns vs BTC TS alone."""
    baseline = _btc_baseline_metrics(close_prices=close_prices, data_dir=data_dir)
    report = run_sleeve_blend_experiment(
        cs_candidates=cs_candidates,
        allocations=allocations,
        close_prices=close_prices,
        data_dir=data_dir,
        save_csv=False,
    )
    full_period = report.loc[report["year"] == "FULL"].copy()
    if full_period.empty:
        raise ValueError("No full-period blend metrics produced")

    full_period["delta_total_return"] = full_period["total_return"] - baseline["total_return"]
    full_period["delta_cagr"] = full_period["cagr"] - baseline["cagr"]
    full_period["delta_sharpe"] = full_period["sharpe"] - baseline["sharpe"]
    full_period["delta_max_drawdown"] = full_period["max_drawdown"] - baseline["max_drawdown"]

    full_period = full_period.sort_values(
        ["sharpe", "cagr", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_period.to_csv(output_path, index=False)
    return full_period