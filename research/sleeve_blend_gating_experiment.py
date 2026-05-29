"""Research-only gating experiment for BTC plus CS sleeve blends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.run_expanded_universe_experiment import EXPANDED_UNIVERSE_20
from research.signals import calculate_momentum_signal
from research.sleeve_blend_experiment import LOCKED_BTC_SLEEVE, align_sleeve_return_streams, run_sleeve_backtests


FULL_OUTPUT_PATH = Path("sleeve_blend_gating_metrics.csv")
YEARLY_OUTPUT_PATH = Path("sleeve_blend_gating_yearly.csv")
GATING_CS_CANDIDATE = CrossSectionalRobustnessSpec(
    name="cs_180_top5_reb12",
    lookback_config_name="medium_180_only",
    top_n=5,
    rebalance_bars=12,
)
GATING_BASE_ALLOCATIONS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.9, 0.1),
    (0.85, 0.15),
    (0.8, 0.2),
    (0.75, 0.25),
)


@dataclass(frozen=True)
class GateSpec:
    """One regime gate definition for CS exposure."""

    name: str
    ma_window: int | None = None


DEFAULT_GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(name="ALWAYS_ON"),
    GateSpec(name="BTC_TS_INVESTED"),
    GateSpec(name="BTC_PRICE_MOMENTUM_240"),
    GateSpec(name="BTC_ABOVE_MA_120", ma_window=120),
    GateSpec(name="BTC_ABOVE_MA_240", ma_window=240),
)


def _validate_allocations(allocations: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Validate the requested base blend allocation set."""
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


def build_gating_base_allocations() -> tuple[tuple[float, float], ...]:
    """Return the requested base allocations for the gating experiment."""
    return _validate_allocations(GATING_BASE_ALLOCATIONS)


def _worst_month_return(returns: pd.Series) -> float:
    """Return the worst compounded monthly return inside the slice."""
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    monthly = (1.0 + clean).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float(monthly.min())


def build_cs_gate_series(
    gate_spec: GateSpec,
    aligned_index: pd.Index,
    btc_close: pd.Series,
    btc_invested: pd.Series,
) -> pd.Series:
    """Build a boolean/float gate series aligned to the sleeve return timestamps."""
    gate_name = gate_spec.name
    if gate_name == "ALWAYS_ON":
        return pd.Series(1.0, index=aligned_index, dtype=float)

    if gate_name == "BTC_TS_INVESTED":
        gate = btc_invested.reindex(aligned_index).fillna(0.0) > 0.0
        return gate.astype(float)

    if gate_name == "BTC_PRICE_MOMENTUM_240":
        momentum = calculate_momentum_signal(
            btc_close.to_frame(name=LOCKED_BTC_SLEEVE.symbol),
            lookback_bars=int(LOCKED_BTC_SLEEVE.medium_lookback_bars),
        )[LOCKED_BTC_SLEEVE.symbol]
        gate = momentum.reindex(aligned_index).fillna(0.0) > 0.0
        return gate.astype(float)

    if gate_name.startswith("BTC_ABOVE_MA_"):
        if gate_spec.ma_window is None:
            raise ValueError(f"Missing ma_window for gate {gate_name}")
        moving_average = btc_close.rolling(window=int(gate_spec.ma_window), min_periods=int(gate_spec.ma_window)).mean()
        gate = (btc_close > moving_average).reindex(aligned_index).fillna(False)
        return gate.astype(float)

    raise ValueError(f"Unsupported gate spec: {gate_name}")


def build_gated_blend_stream(
    aligned_returns: pd.DataFrame,
    gate_series: pd.Series,
    btc_weight: float,
    cs_weight: float,
) -> pd.DataFrame:
    """Build the effective BTC/CS contribution stream under a gating rule."""
    stream = aligned_returns.copy()
    stream["cs_gate"] = gate_series.reindex(stream.index).fillna(0.0).astype(float)
    stream["effective_btc_weight"] = float(btc_weight)
    stream["effective_cs_weight"] = float(cs_weight) * stream["cs_gate"]
    stream["effective_btc_return"] = stream["btc_sleeve_return"] * float(btc_weight)
    stream["effective_cs_return"] = stream["cs_sleeve_return"] * stream["effective_cs_weight"]
    stream["blended_return"] = stream["effective_btc_return"] + stream["effective_cs_return"]
    return stream


def _full_row(
    gate_name: str,
    btc_weight: float,
    cs_weight: float,
    cs_candidate: str,
    stream: pd.DataFrame,
    bars_per_year: int,
    worst_year_return: float,
) -> dict[str, float | str]:
    """Compute one full-period metrics row from a gated blend stream."""
    blend_returns = stream["blended_return"]
    equity = SETTINGS.initial_capital * (1.0 + blend_returns).cumprod()
    metrics = summary_metrics(equity=equity, returns=blend_returns, bars_per_year=bars_per_year)
    correlation = float(stream["btc_sleeve_return"].corr(stream["effective_cs_return"]))
    if pd.isna(correlation):
        correlation = 0.0

    return {
        "year": "FULL",
        "gate_rule": gate_name,
        "btc_strategy": "btc_ts_60_240_12",
        "btc_weight": float(btc_weight),
        "cs_weight": float(cs_weight),
        "cs_candidate": cs_candidate,
        "blend_name": f"btc_ts_60_240_12__btc_{int(round(btc_weight * 100)):02d}_cs_{int(round(cs_weight * 100)):02d}__{cs_candidate}__{gate_name.lower()}",
        "total_return": float(metrics.get("total_return", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_year_return": float(worst_year_return),
        "worst_month_return": _worst_month_return(blend_returns),
        "percent_time_cs_enabled": float(stream["cs_gate"].mean()) if float(cs_weight) > 0.0 else 0.0,
        "sleeve_return_correlation": correlation,
    }


def _year_row(
    year: int,
    gate_name: str,
    btc_weight: float,
    cs_weight: float,
    cs_candidate: str,
    stream: pd.DataFrame,
    bars_per_year: int,
) -> dict[str, float | str] | None:
    """Compute one yearly metrics row from the full gated stream."""
    if len(stream) < 2:
        return None
    blend_returns = stream["blended_return"]
    equity = SETTINGS.initial_capital * (1.0 + blend_returns).cumprod()
    metrics = summary_metrics(equity=equity, returns=blend_returns, bars_per_year=bars_per_year)
    correlation = float(stream["btc_sleeve_return"].corr(stream["effective_cs_return"]))
    if pd.isna(correlation):
        correlation = 0.0

    return {
        "year": str(year),
        "gate_rule": gate_name,
        "btc_strategy": "btc_ts_60_240_12",
        "btc_weight": float(btc_weight),
        "cs_weight": float(cs_weight),
        "cs_candidate": cs_candidate,
        "blend_name": f"btc_ts_60_240_12__btc_{int(round(btc_weight * 100)):02d}_cs_{int(round(cs_weight * 100)):02d}__{cs_candidate}__{gate_name.lower()}",
        "total_return": float(metrics.get("total_return", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_year_return": float(metrics.get("total_return", 0.0)),
        "worst_month_return": _worst_month_return(blend_returns),
        "percent_time_cs_enabled": float(stream["cs_gate"].mean()) if float(cs_weight) > 0.0 else 0.0,
        "sleeve_return_correlation": correlation,
    }


def run_sleeve_blend_gating_experiment(
    cs_candidate: CrossSectionalRobustnessSpec = GATING_CS_CANDIDATE,
    allocations: Sequence[tuple[float, float]] = GATING_BASE_ALLOCATIONS,
    gate_specs: Sequence[GateSpec] = DEFAULT_GATE_SPECS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    full_output_path: Path = FULL_OUTPUT_PATH,
    yearly_output_path: Path = YEARLY_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the gating experiment and return full-period plus yearly reports."""
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
    btc_close = close[LOCKED_BTC_SLEEVE.symbol].reindex(aligned.index).astype(float)
    btc_invested = btc_result.holdings_history[LOCKED_BTC_SLEEVE.symbol].reindex(aligned.index).fillna(0.0)
    years = sorted(aligned.index.year.unique().tolist())
    full_rows: list[dict[str, float | str]] = []
    yearly_rows: list[dict[str, float | str]] = []

    for btc_weight, cs_weight in allocation_grid:
        applicable_gates = (GateSpec(name="ALWAYS_ON"),) if float(cs_weight) == 0.0 else tuple(gate_specs)
        for gate_spec in applicable_gates:
            gate_series = build_cs_gate_series(
                gate_spec=gate_spec,
                aligned_index=aligned.index,
                btc_close=btc_close,
                btc_invested=btc_invested,
            )
            stream = build_gated_blend_stream(
                aligned_returns=aligned,
                gate_series=gate_series,
                btc_weight=btc_weight,
                cs_weight=cs_weight,
            )

            annual_returns: list[float] = []
            for year in years:
                yearly_stream = stream.loc[stream.index.year == int(year)]
                row = _year_row(
                    year=int(year),
                    gate_name=gate_spec.name,
                    btc_weight=btc_weight,
                    cs_weight=cs_weight,
                    cs_candidate=cs_candidate.name,
                    stream=yearly_stream,
                    bars_per_year=bars_per_year,
                )
                if row is not None:
                    annual_returns.append(float(row["annual_return"]))
                    yearly_rows.append(row)

            full_rows.append(
                _full_row(
                    gate_name=gate_spec.name,
                    btc_weight=btc_weight,
                    cs_weight=cs_weight,
                    cs_candidate=cs_candidate.name,
                    stream=stream,
                    bars_per_year=bars_per_year,
                    worst_year_return=min(annual_returns) if annual_returns else 0.0,
                )
            )

    full_report = pd.DataFrame(full_rows)
    yearly_report = pd.DataFrame(yearly_rows)
    if full_report.empty or yearly_report.empty:
        raise ValueError("No gating metrics produced")

    full_report = full_report.sort_values(["sharpe", "cagr", "total_return"], ascending=[False, False, False]).reset_index(drop=True)
    year_order = {str(year): idx for idx, year in enumerate(years)}
    yearly_report["_year_order"] = yearly_report["year"].astype(str).map(year_order)
    yearly_report = yearly_report.sort_values(["_year_order", "gate_rule", "btc_weight"], ascending=[True, True, False]).drop(columns=["_year_order"]).reset_index(drop=True)

    if save_csv:
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        yearly_output_path.parent.mkdir(parents=True, exist_ok=True)
        full_report.to_csv(full_output_path, index=False)
        yearly_report.to_csv(yearly_output_path, index=False)
    return full_report, yearly_report