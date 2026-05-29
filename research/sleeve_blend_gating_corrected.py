"""Research-only corrected gating panel with explicit gate lag and overlay costs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.metrics import max_drawdown, summary_metrics
from config import SETTINGS
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.sleeve_blend_experiment import run_sleeve_backtests
from research.sleeve_blend_gating_experiment import GateSpec, build_cs_gate_series


DEFAULT_OUTPUT_PATH = Path("sleeve_blend_gating_corrected.csv")
CORRECTED_ALLOCATIONS: tuple[tuple[float, float], ...] = (
    (0.9, 0.1),
    (0.85, 0.15),
    (0.8, 0.2),
    (0.75, 0.25),
)
CORRECTED_OVERLAY_COST_BPS: tuple[float, ...] = (0.0, 15.0, 30.0, 75.0)
CORRECTED_GATE_LAG_BARS = 1
CORRECTED_CS_CANDIDATE = CrossSectionalRobustnessSpec(
    name="cs_180_top5_reb12",
    lookback_config_name="medium_180_only",
    top_n=5,
    rebalance_bars=12,
)
CORRECTED_GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(name="ALWAYS_ON"),
    GateSpec(name="BTC_TS_INVESTED"),
    GateSpec(name="BTC_PRICE_MOMENTUM_240"),
    GateSpec(name="BTC_ABOVE_MA_80", ma_window=80),
    GateSpec(name="BTC_ABOVE_MA_100", ma_window=100),
    GateSpec(name="BTC_ABOVE_MA_120", ma_window=120),
    GateSpec(name="BTC_ABOVE_MA_140", ma_window=140),
    GateSpec(name="BTC_ABOVE_MA_160", ma_window=160),
    GateSpec(name="BTC_ABOVE_MA_180", ma_window=180),
    GateSpec(name="BTC_ABOVE_MA_200", ma_window=200),
    GateSpec(name="BTC_ABOVE_MA_240", ma_window=240),
)


@dataclass(frozen=True)
class CorrectedSummary:
    """Small summary blocks used by the CLI reporter."""

    best_by_sharpe: pd.DataFrame
    best_by_max_drawdown: pd.DataFrame
    best_by_2022_return: pd.DataFrame
    best_conservative: pd.DataFrame
    best_aggressive: pd.DataFrame


def _validate_allocations(allocations: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not allocations:
        raise ValueError("allocations must not be empty")
    validated: list[tuple[float, float]] = []
    for btc_weight, cs_weight in allocations:
        btc = float(btc_weight)
        cs = float(cs_weight)
        if btc < 0.0 or cs < 0.0:
            raise ValueError("weights must be non-negative")
        if abs((btc + cs) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")
        validated.append((btc, cs))
    return tuple(validated)


def _validate_overlay_costs(overlay_cost_bps_values: Sequence[float]) -> tuple[float, ...]:
    if not overlay_cost_bps_values:
        raise ValueError("overlay_cost_bps_values must not be empty")
    values = tuple(float(x) for x in overlay_cost_bps_values)
    if any(x < 0.0 for x in values):
        raise ValueError("overlay cost bps must be non-negative")
    return values


def _gate_name_to_ma_length(gate_name: str) -> float:
    if gate_name.startswith("BTC_ABOVE_MA_"):
        try:
            return float(int(gate_name.rsplit("_", 1)[-1]))
        except ValueError:
            return float("nan")
    return float("nan")


def apply_gate_lag(gate_series: pd.Series, lag_bars: int) -> pd.Series:
    """Apply explicit decision lag so gate at t only affects t+lag onward.

    The first lagged bars are filled with the first observed decision, which keeps
    ALWAYS_ON unchanged while still enforcing delayed propagation for changing gates.
    """
    if lag_bars < 0:
        raise ValueError("lag_bars must be non-negative")
    base = gate_series.astype(float)
    if lag_bars == 0 or base.empty:
        return base
    first = float(base.iloc[0])
    lagged = base.shift(int(lag_bars))
    return lagged.fillna(first).astype(float)


def estimate_overlay_turnover(
    effective_btc_weight: pd.Series,
    effective_cs_weight: pd.Series,
) -> pd.Series:
    """Estimate overlay turnover from changes in effective BTC/CS/cash sleeve weights."""
    btc = effective_btc_weight.astype(float)
    cs = effective_cs_weight.astype(float)
    cash = 1.0 - btc - cs

    turnover = (
        btc.diff().abs().fillna(0.0)
        + cs.diff().abs().fillna(0.0)
        + cash.diff().abs().fillna(0.0)
    )
    return turnover.astype(float)


def build_corrected_gated_stream(
    aligned_returns: pd.DataFrame,
    raw_gate_series: pd.Series,
    btc_weight: float,
    cs_weight: float,
    gate_lag_bars: int,
    overlay_cost_bps: float,
) -> pd.DataFrame:
    """Build corrected return stream with lagged gate and overlay cost drag."""
    stream = aligned_returns.copy()
    stream["raw_cs_gate"] = raw_gate_series.reindex(stream.index).fillna(0.0).astype(float)
    stream["cs_gate"] = apply_gate_lag(stream["raw_cs_gate"], lag_bars=gate_lag_bars)

    stream["effective_btc_weight"] = float(btc_weight)
    stream["effective_cs_weight"] = float(cs_weight) * stream["cs_gate"]
    stream["effective_cash_weight"] = 1.0 - stream["effective_btc_weight"] - stream["effective_cs_weight"]

    stream["effective_btc_return"] = stream["btc_sleeve_return"] * stream["effective_btc_weight"]
    stream["effective_cs_return"] = stream["cs_sleeve_return"] * stream["effective_cs_weight"]
    stream["blended_return_pre_overlay_cost"] = stream["effective_btc_return"] + stream["effective_cs_return"]

    stream["overlay_turnover"] = estimate_overlay_turnover(
        effective_btc_weight=stream["effective_btc_weight"],
        effective_cs_weight=stream["effective_cs_weight"],
    )
    stream["overlay_cost"] = stream["overlay_turnover"] * (float(overlay_cost_bps) / 10_000.0)
    stream["blended_return"] = stream["blended_return_pre_overlay_cost"] - stream["overlay_cost"]
    return stream


def _worst_month_return(returns: pd.Series) -> float:
    monthly = (1.0 + returns.dropna().astype(float)).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float(monthly.min())


def _annual_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).groupby(clean.index.year).prod() - 1.0


def _year_max_drawdown(returns: pd.Series, year: int) -> float:
    year_returns = returns.loc[returns.index.year == int(year)].dropna().astype(float)
    if year_returns.empty:
        return float("nan")
    equity = pd.Series(1.0, index=year_returns.index, dtype=float) * (1.0 + year_returns).cumprod()
    return float(max_drawdown(equity))


def _build_row(
    gate_name: str,
    btc_weight: float,
    cs_weight: float,
    gate_lag_bars: int,
    overlay_cost_bps: float,
    bars_per_year: int,
    stream: pd.DataFrame,
) -> dict[str, float | int | str]:
    returns = stream["blended_return"].astype(float)
    equity = SETTINGS.initial_capital * (1.0 + returns).cumprod()
    metrics = summary_metrics(equity=equity, returns=returns, bars_per_year=bars_per_year)

    annual_returns = _annual_returns(returns)
    year_2022_return = float(annual_returns.loc[2022]) if 2022 in annual_returns.index else float("nan")

    return {
        "gate_name": gate_name,
        "ma_length": _gate_name_to_ma_length(gate_name),
        "btc_weight": float(btc_weight),
        "cs_weight": float(cs_weight),
        "gate_lag_bars": int(gate_lag_bars),
        "overlay_cost_bps": float(overlay_cost_bps),
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_year_return": float(annual_returns.min()) if not annual_returns.empty else 0.0,
        "worst_month_return": _worst_month_return(returns),
        "2022_return": year_2022_return,
        "2022_max_drawdown": _year_max_drawdown(returns, year=2022),
        "count_negative_years": int((annual_returns < 0.0).sum()),
        "percent_time_cs_enabled": float(stream["cs_gate"].mean()) if float(cs_weight) > 0.0 else 0.0,
        "overlay_turnover": float(stream["overlay_turnover"].sum()),
        "overlay_cost_drag": float(stream["overlay_cost"].sum()),
    }


def build_corrected_summary(panel: pd.DataFrame) -> CorrectedSummary:
    """Build compact summary blocks from corrected panel."""
    candidates = panel.loc[(panel["cs_weight"] > 0.0) & (panel["gate_name"] != "ALWAYS_ON")].copy()

    best_by_sharpe = candidates.nlargest(1, "sharpe") if not candidates.empty else candidates
    best_by_max_drawdown = candidates.nlargest(1, "max_drawdown") if not candidates.empty else candidates
    best_by_2022_return = candidates.nlargest(1, "2022_return") if not candidates.empty else candidates

    conservative = candidates.loc[
        (candidates["delta_sharpe_vs_always_on_same_weight_and_cost"] > 0.0)
        & (candidates["delta_max_drawdown_vs_always_on_same_weight_and_cost"] >= 0.0)
    ].copy()
    best_conservative = conservative.sort_values(["max_drawdown", "sharpe"], ascending=[False, False]).head(1)
    best_aggressive = candidates.nlargest(1, "sharpe") if not candidates.empty else candidates

    return CorrectedSummary(
        best_by_sharpe=best_by_sharpe,
        best_by_max_drawdown=best_by_max_drawdown,
        best_by_2022_return=best_by_2022_return,
        best_conservative=best_conservative,
        best_aggressive=best_aggressive,
    )


def run_sleeve_blend_gating_corrected(
    allocations: Sequence[tuple[float, float]] = CORRECTED_ALLOCATIONS,
    gate_specs: Sequence[GateSpec] = CORRECTED_GATE_SPECS,
    overlay_cost_bps_values: Sequence[float] = CORRECTED_OVERLAY_COST_BPS,
    gate_lag_bars: int = CORRECTED_GATE_LAG_BARS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, CorrectedSummary]:
    """Run corrected gating panel with lagged gates and optional overlay cost scenarios."""
    if gate_lag_bars < 0:
        raise ValueError("gate_lag_bars must be non-negative")
    allocation_grid = _validate_allocations(allocations)
    overlay_bps_grid = _validate_overlay_costs(overlay_cost_bps_values)

    # Include BTC-only baseline rows for delta comparisons.
    run_allocations = ((1.0, 0.0),) + allocation_grid

    close, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(CORRECTED_CS_CANDIDATE,),
        close_prices=close_prices,
        data_dir=data_dir,
    )

    aligned = pd.concat(
        [
            btc_result.portfolio["strategy_return"].rename("btc_sleeve_return"),
            cs_results[CORRECTED_CS_CANDIDATE.name].portfolio["strategy_return"].rename("cs_sleeve_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    aligned.index = pd.to_datetime(aligned.index, utc=True)
    aligned = aligned.sort_index()

    btc_close = close["BTC/USD"].reindex(aligned.index).astype(float)
    btc_invested = btc_result.holdings_history["BTC/USD"].reindex(aligned.index).fillna(0.0)

    rows: list[dict[str, float | int | str]] = []
    for overlay_cost_bps in overlay_bps_grid:
        for btc_weight, cs_weight in run_allocations:
            applicable_gates = (GateSpec(name="ALWAYS_ON"),) if cs_weight <= 0.0 else tuple(gate_specs)
            for gate_spec in applicable_gates:
                raw_gate = build_cs_gate_series(
                    gate_spec=gate_spec,
                    aligned_index=aligned.index,
                    btc_close=btc_close,
                    btc_invested=btc_invested,
                )
                stream = build_corrected_gated_stream(
                    aligned_returns=aligned,
                    raw_gate_series=raw_gate,
                    btc_weight=float(btc_weight),
                    cs_weight=float(cs_weight),
                    gate_lag_bars=int(gate_lag_bars),
                    overlay_cost_bps=float(overlay_cost_bps),
                )
                rows.append(
                    _build_row(
                        gate_name=gate_spec.name,
                        btc_weight=float(btc_weight),
                        cs_weight=float(cs_weight),
                        gate_lag_bars=int(gate_lag_bars),
                        overlay_cost_bps=float(overlay_cost_bps),
                        bars_per_year=bars_per_year,
                        stream=stream,
                    )
                )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No corrected gating rows produced")

    # Delta vs ALWAYS_ON for same weight and same overlay cost scenario.
    always_on = panel.loc[
        panel["gate_name"] == "ALWAYS_ON",
        ["overlay_cost_bps", "btc_weight", "cs_weight", "sharpe", "max_drawdown"],
    ].rename(
        columns={
            "sharpe": "always_on_sharpe",
            "max_drawdown": "always_on_max_drawdown",
        }
    )
    panel = panel.merge(always_on, on=["overlay_cost_bps", "btc_weight", "cs_weight"], how="left")

    # Delta vs BTC-only baseline inside each overlay cost scenario.
    btc_only = panel.loc[
        (panel["gate_name"] == "ALWAYS_ON")
        & ((panel["btc_weight"] - 1.0).abs() < 1e-12)
        & (panel["cs_weight"].abs() < 1e-12),
        ["overlay_cost_bps", "sharpe", "max_drawdown"],
    ].rename(
        columns={
            "sharpe": "btc_only_sharpe",
            "max_drawdown": "btc_only_max_drawdown",
        }
    )
    panel = panel.merge(btc_only, on=["overlay_cost_bps"], how="left")

    panel["delta_sharpe_vs_btc_only"] = panel["sharpe"] - panel["btc_only_sharpe"]
    panel["delta_max_drawdown_vs_btc_only"] = panel["max_drawdown"] - panel["btc_only_max_drawdown"]
    panel["delta_sharpe_vs_always_on_same_weight_and_cost"] = panel["sharpe"] - panel["always_on_sharpe"]
    panel["delta_max_drawdown_vs_always_on_same_weight_and_cost"] = panel["max_drawdown"] - panel["always_on_max_drawdown"]

    panel = panel.drop(
        columns=[
            "always_on_sharpe",
            "always_on_max_drawdown",
            "btc_only_sharpe",
            "btc_only_max_drawdown",
        ]
    )
    panel = panel.sort_values(["sharpe", "cagr", "total_return"], ascending=[False, False, False]).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)

    summary = build_corrected_summary(panel)
    return panel, summary
