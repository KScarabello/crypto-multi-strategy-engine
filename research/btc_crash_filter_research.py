"""Exploratory BTC long/cash crash-filter overlay research module.

This module is research-only and is not wired to live execution paths.
No shorting or leverage is used in this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.btc_long_short_momentum_research import DEFAULT_LONG_SHORT_THRESHOLD_SPECS
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_backtests
from research.run_expanded_universe_experiment import bars_per_year_for_timeframe


DEFAULT_SUMMARY_OUTPUT_PATH = Path("btc_crash_filter_summary.csv")
DEFAULT_BY_YEAR_OUTPUT_PATH = Path("btc_crash_filter_by_year.csv")
DEFAULT_EVENTS_OUTPUT_PATH = Path("btc_crash_filter_events.csv")


@dataclass(frozen=True)
class DrawdownFilterSpec:
    rolling_high_window: int
    drawdown_threshold: float


@dataclass(frozen=True)
class VolatilityFilterSpec:
    vol_window: int
    vol_multiplier: float


@dataclass(frozen=True)
class SevereReturnFilterSpec:
    recent_return_window: int
    severe_return_threshold: float
    cooldown_bars: int


@dataclass(frozen=True)
class CrashFilterCandidateSpec:
    name: str
    drawdown_filter: DrawdownFilterSpec | None = None
    volatility_filter: VolatilityFilterSpec | None = None
    severe_return_filter: SevereReturnFilterSpec | None = None


DEFAULT_CRASH_FILTER_CANDIDATES: tuple[CrashFilterCandidateSpec, ...] = (
    CrashFilterCandidateSpec(name="baseline_long_cash"),
    CrashFilterCandidateSpec(
        name="drawdown_180_15",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=180, drawdown_threshold=-0.15),
    ),
    CrashFilterCandidateSpec(
        name="drawdown_360_20",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=360, drawdown_threshold=-0.20),
    ),
    CrashFilterCandidateSpec(
        name="vol_60_1p5",
        volatility_filter=VolatilityFilterSpec(vol_window=60, vol_multiplier=1.5),
    ),
    CrashFilterCandidateSpec(
        name="vol_120_2p0",
        volatility_filter=VolatilityFilterSpec(vol_window=120, vol_multiplier=2.0),
    ),
    CrashFilterCandidateSpec(
        name="severe_12_8_cd12",
        severe_return_filter=SevereReturnFilterSpec(recent_return_window=12, severe_return_threshold=-0.08, cooldown_bars=12),
    ),
    CrashFilterCandidateSpec(
        name="severe_24_10_cd24",
        severe_return_filter=SevereReturnFilterSpec(recent_return_window=24, severe_return_threshold=-0.10, cooldown_bars=24),
    ),
    CrashFilterCandidateSpec(
        name="combo_drawdown_vol",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=360, drawdown_threshold=-0.20),
        volatility_filter=VolatilityFilterSpec(vol_window=60, vol_multiplier=1.5),
    ),
    CrashFilterCandidateSpec(
        name="combo_drawdown_severe",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=360, drawdown_threshold=-0.20),
        severe_return_filter=SevereReturnFilterSpec(recent_return_window=24, severe_return_threshold=-0.10, cooldown_bars=24),
    ),
    CrashFilterCandidateSpec(
        name="combo_vol_severe",
        volatility_filter=VolatilityFilterSpec(vol_window=60, vol_multiplier=1.5),
        severe_return_filter=SevereReturnFilterSpec(recent_return_window=24, severe_return_threshold=-0.10, cooldown_bars=24),
    ),
    CrashFilterCandidateSpec(
        name="combo_all",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=360, drawdown_threshold=-0.20),
        volatility_filter=VolatilityFilterSpec(vol_window=60, vol_multiplier=1.5),
        severe_return_filter=SevereReturnFilterSpec(recent_return_window=24, severe_return_threshold=-0.10, cooldown_bars=24),
    ),
)


@dataclass(frozen=True)
class BtcCrashFilterResearchConfig:
    symbol: str = "BTC/USD"
    timeframe: str = "4h"
    base_short_lookback: int = 60
    base_medium_lookback: int = 240
    rebalance_every_bars: int = 12
    transaction_cost_bps: float = SETTINGS.transaction_cost_bps
    slippage_bps: float = SETTINGS.slippage_bps
    candidates: Sequence[CrashFilterCandidateSpec] = DEFAULT_CRASH_FILTER_CANDIDATES


@dataclass(frozen=True)
class CrashFilterCandidateArtifacts:
    candidate_name: str
    signals: pd.DataFrame
    annual_report: pd.DataFrame
    events: pd.DataFrame


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


def _build_base_long_cash_target_positions(
    close: pd.DataFrame,
    short_lookback_bars: int,
    medium_lookback_bars: int,
    rebalance_every_bars: int,
) -> pd.Series:
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
        else:
            current_target = 0.0

        target.iloc[i] = current_target

    return target


def _drawdown_force_cash(close: pd.DataFrame, spec: DrawdownFilterSpec | None) -> pd.Series:
    if spec is None:
        return pd.Series(False, index=close.index)

    px = close.iloc[:, 0].astype(float)
    rolling_high = px.rolling(window=int(spec.rolling_high_window), min_periods=1).max()
    drawdown = px / rolling_high - 1.0
    return (drawdown < float(spec.drawdown_threshold)).fillna(False)


def _volatility_force_cash(
    close: pd.DataFrame,
    short_lookback_bars: int,
    spec: VolatilityFilterSpec | None,
) -> pd.Series:
    if spec is None:
        return pd.Series(False, index=close.index)

    px = close.iloc[:, 0].astype(float)
    btc_returns = px.pct_change().fillna(0.0)
    realized_vol = btc_returns.rolling(window=int(spec.vol_window), min_periods=int(spec.vol_window)).std(ddof=0)
    rolling_median_vol = realized_vol.rolling(window=int(spec.vol_window), min_periods=1).median()
    short_momentum = px.pct_change(int(short_lookback_bars))

    return ((realized_vol > float(spec.vol_multiplier) * rolling_median_vol) & (short_momentum < 0.0)).fillna(False)


def _severe_return_force_cash(close: pd.DataFrame, spec: SevereReturnFilterSpec | None) -> pd.Series:
    if spec is None:
        return pd.Series(False, index=close.index)

    px = close.iloc[:, 0].astype(float)
    recent_return = px.pct_change(int(spec.recent_return_window)).fillna(0.0)
    trigger = recent_return < float(spec.severe_return_threshold)

    cooldown = pd.Series(False, index=close.index)
    remaining = 0
    for i, ts in enumerate(close.index):
        if bool(trigger.iloc[i]):
            remaining = int(spec.cooldown_bars)
        if remaining > 0:
            cooldown.loc[ts] = True
            remaining -= 1

    return cooldown


def _force_cash_trigger_strings(
    drawdown_force: pd.Series,
    volatility_force: pd.Series,
    severe_force: pd.Series,
) -> pd.Series:
    labels = []
    for ts in drawdown_force.index:
        triggers: list[str] = []
        if bool(drawdown_force.loc[ts]):
            triggers.append("drawdown")
        if bool(volatility_force.loc[ts]):
            triggers.append("volatility")
        if bool(severe_force.loc[ts]):
            triggers.append("severe_return")
        labels.append("|".join(triggers))
    return pd.Series(labels, index=drawdown_force.index, dtype=str)


def _extract_forced_cash_events(
    candidate_name: str,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    forced = signals["forced_cash_flag"].astype(bool)
    rows: list[dict[str, Any]] = []

    start_idx: int | None = None
    current_trigger = ""
    for i, is_forced in enumerate(forced.tolist()):
        trigger_label = str(signals["trigger_type"].iloc[i])

        if is_forced and start_idx is None:
            start_idx = i
            current_trigger = trigger_label
            continue

        if is_forced and start_idx is not None:
            if trigger_label and trigger_label not in current_trigger.split("|"):
                merged = sorted(set([x for x in (current_trigger + "|" + trigger_label).split("|") if x]))
                current_trigger = "|".join(merged)
            continue

        if (not is_forced) and start_idx is not None:
            end_idx = i - 1
            event_slice = signals.iloc[start_idx : end_idx + 1]
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "event_start_time": str(signals.index[start_idx]),
                    "event_end_time": str(signals.index[end_idx]),
                    "trigger_type": current_trigger,
                    "btc_return_during_event": float((1.0 + event_slice["btc_return"]).prod() - 1.0),
                    "baseline_long_cash_return_during_event": float((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0),
                    "strategy_return_during_event": float((1.0 + event_slice["strategy_return"]).prod() - 1.0),
                    "avoided_loss_or_missed_gain_vs_baseline": float(
                        ((1.0 + event_slice["strategy_return"]).prod() - 1.0)
                        - ((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0)
                    ),
                    "holding_period_bars": int(end_idx - start_idx + 1),
                }
            )
            start_idx = None
            current_trigger = ""

    if start_idx is not None:
        end_idx = len(signals.index) - 1
        event_slice = signals.iloc[start_idx : end_idx + 1]
        rows.append(
            {
                "candidate_name": candidate_name,
                "event_start_time": str(signals.index[start_idx]),
                "event_end_time": str(signals.index[end_idx]),
                "trigger_type": current_trigger,
                "btc_return_during_event": float((1.0 + event_slice["btc_return"]).prod() - 1.0),
                "baseline_long_cash_return_during_event": float((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0),
                "strategy_return_during_event": float((1.0 + event_slice["strategy_return"]).prod() - 1.0),
                "avoided_loss_or_missed_gain_vs_baseline": float(
                    ((1.0 + event_slice["strategy_return"]).prod() - 1.0)
                    - ((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0)
                ),
                "holding_period_bars": int(end_idx - start_idx + 1),
            }
        )

    return pd.DataFrame(rows)


def _candidate_filter_type(candidate: CrashFilterCandidateSpec) -> str:
    names: list[str] = []
    if candidate.drawdown_filter is not None:
        names.append("drawdown")
    if candidate.volatility_filter is not None:
        names.append("volatility")
    if candidate.severe_return_filter is not None:
        names.append("severe_return")
    return "baseline" if not names else "+".join(names)


def _candidate_filter_parameters(candidate: CrashFilterCandidateSpec) -> str:
    parts: list[str] = []
    if candidate.drawdown_filter is not None:
        parts.append(
            f"drawdown(window={candidate.drawdown_filter.rolling_high_window},threshold={candidate.drawdown_filter.drawdown_threshold})"
        )
    if candidate.volatility_filter is not None:
        parts.append(
            f"vol(window={candidate.volatility_filter.vol_window},mult={candidate.volatility_filter.vol_multiplier})"
        )
    if candidate.severe_return_filter is not None:
        parts.append(
            f"severe(window={candidate.severe_return_filter.recent_return_window},thr={candidate.severe_return_filter.severe_return_threshold},cooldown={candidate.severe_return_filter.cooldown_bars})"
        )
    return "none" if not parts else ";".join(parts)


def run_single_crash_filter_candidate(
    close_prices: pd.DataFrame,
    bars_per_year: int,
    candidate: CrashFilterCandidateSpec,
    base_short_lookback: int,
    base_medium_lookback: int,
    rebalance_every_bars: int,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> CrashFilterCandidateArtifacts:
    close = _validate_close_prices(close_prices, symbol=close_prices.columns[0])
    btc_returns = close.iloc[:, 0].pct_change().fillna(0.0).astype(float)

    base_target = _build_base_long_cash_target_positions(
        close=close,
        short_lookback_bars=int(base_short_lookback),
        medium_lookback_bars=int(base_medium_lookback),
        rebalance_every_bars=int(rebalance_every_bars),
    )

    drawdown_force = _drawdown_force_cash(close, candidate.drawdown_filter)
    volatility_force = _volatility_force_cash(close, int(base_short_lookback), candidate.volatility_filter)
    severe_force = _severe_return_force_cash(close, candidate.severe_return_filter)

    force_cash_flag = (drawdown_force | volatility_force | severe_force).astype(bool)
    trigger_type = _force_cash_trigger_strings(drawdown_force, volatility_force, severe_force)

    target_position = base_target.copy()
    target_position.loc[force_cash_flag & (target_position > 0.0)] = 0.0

    position = target_position.shift(1).fillna(0.0).astype(float)
    baseline_position = base_target.shift(1).fillna(0.0).astype(float)

    position_change = position.diff().abs().fillna(position.abs()).astype(float)
    per_turnover_cost = float(transaction_cost_bps + slippage_bps) / 10_000.0
    transaction_cost = position_change * per_turnover_cost

    gross_return = position * btc_returns
    strategy_return = gross_return - transaction_cost
    baseline_long_cash_return = baseline_position * btc_returns

    signals = pd.DataFrame(
        {
            "btc_return": btc_returns,
            "base_target": base_target,
            "target_position": target_position,
            "position": position,
            "baseline_position": baseline_position,
            "position_change": position_change,
            "transaction_cost": transaction_cost,
            "gross_return": gross_return,
            "strategy_return": strategy_return,
            "baseline_long_cash_return": baseline_long_cash_return,
            "force_drawdown": drawdown_force.astype(bool),
            "force_volatility": volatility_force.astype(bool),
            "force_severe_return": severe_force.astype(bool),
            "forced_cash_flag": force_cash_flag.astype(bool),
            "trigger_type": trigger_type,
        },
        index=close.index,
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
        forced_now = year_slice["forced_cash_flag"].astype(bool)
        forced_prev = year_slice["forced_cash_flag"].shift(1).fillna(False).astype(bool)
        number_of_events = int((forced_now & (~forced_prev)).sum())

        annual_rows.append(
            {
                "year": int(year),
                "candidate_name": candidate.name,
                "strategy_return": float(year_metrics["total_return"]),
                "sharpe": float(year_metrics["sharpe"]),
                "max_drawdown": float(year_metrics["max_drawdown"]),
                "worst_month_return": float((1.0 + year_slice["strategy_return"]).resample("ME").prod().sub(1.0).min()),
                "percent_time_long": float((year_slice["position"] > 0.0).mean()),
                "percent_time_cash": float((year_slice["position"] == 0.0).mean()),
                "percent_time_forced_cash": float(year_slice["forced_cash_flag"].mean()),
                "number_of_forced_cash_events": number_of_events,
            }
        )

    events = _extract_forced_cash_events(candidate_name=candidate.name, signals=signals)

    return CrashFilterCandidateArtifacts(
        candidate_name=candidate.name,
        signals=signals,
        annual_report=pd.DataFrame(annual_rows),
        events=events,
    )


def run_btc_crash_filter_research(
    config: BtcCrashFilterResearchConfig | None = None,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    by_year_output_path: Path = DEFAULT_BY_YEAR_OUTPUT_PATH,
    events_output_path: Path = DEFAULT_EVENTS_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or BtcCrashFilterResearchConfig()
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
        short_lookback_bars=int(cfg.base_short_lookback),
        medium_lookback_bars=int(cfg.base_medium_lookback),
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

    buy_hold_returns = benchmark_map["btc_buy_and_hold"].portfolio["strategy_return"].reindex(close.index).fillna(0.0)
    long_cash_returns = benchmark_map["btc_time_series_momentum"].portfolio["strategy_return"].reindex(close.index).fillna(0.0)

    best_long_short_name = DEFAULT_LONG_SHORT_THRESHOLD_SPECS[-1].name

    summary_rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []

    buy_hold_equity = (1.0 + buy_hold_returns).cumprod()
    long_cash_equity = (1.0 + long_cash_returns).cumprod()
    buy_hold_metrics = summary_metrics(equity=buy_hold_equity, returns=buy_hold_returns, bars_per_year=bars_per_year)
    long_cash_metrics = summary_metrics(equity=long_cash_equity, returns=long_cash_returns, bars_per_year=bars_per_year)

    for candidate in tuple(cfg.candidates):
        artifacts = run_single_crash_filter_candidate(
            close_prices=close,
            bars_per_year=bars_per_year,
            candidate=candidate,
            base_short_lookback=int(cfg.base_short_lookback),
            base_medium_lookback=int(cfg.base_medium_lookback),
            rebalance_every_bars=int(cfg.rebalance_every_bars),
            transaction_cost_bps=float(cfg.transaction_cost_bps),
            slippage_bps=float(cfg.slippage_bps),
        )

        signals = artifacts.signals
        equity = (1.0 + signals["strategy_return"]).cumprod()
        metrics = summary_metrics(equity=equity, returns=signals["strategy_return"], bars_per_year=bars_per_year)

        summary_rows.append(
            {
                "candidate_name": candidate.name,
                "base_short_lookback": int(cfg.base_short_lookback),
                "base_medium_lookback": int(cfg.base_medium_lookback),
                "rebalance_every": int(cfg.rebalance_every_bars),
                "filter_type": _candidate_filter_type(candidate),
                "filter_parameters": _candidate_filter_parameters(candidate),
                "total_return": float(metrics["total_return"]),
                "cagr": float(metrics["cagr"]),
                "annualized_return": float(metrics["cagr"]),
                "annualized_volatility": float(metrics["annualized_volatility"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "worst_month_return": float((1.0 + signals["strategy_return"]).resample("ME").prod().sub(1.0).min()),
                "number_of_trades": int((signals["position_change"] > 1e-12).sum()),
                "percent_time_long": float((signals["position"] > 0.0).mean()),
                "percent_time_cash": float((signals["position"] == 0.0).mean()),
                "forced_cash_periods": int(signals["forced_cash_flag"].sum()),
                "percent_time_forced_cash": float(signals["forced_cash_flag"].mean()),
                "total_transaction_costs": float(signals["transaction_cost"].sum()),
                "vs_btc_buy_hold_total_return": float(metrics["total_return"] - buy_hold_metrics["total_return"]),
                "vs_btc_buy_hold_sharpe": float(metrics["sharpe"] - buy_hold_metrics["sharpe"]),
                "vs_btc_buy_hold_max_drawdown": float(metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"]),
                "vs_btc_long_cash_total_return": float(metrics["total_return"] - long_cash_metrics["total_return"]),
                "vs_btc_long_cash_sharpe": float(metrics["sharpe"] - long_cash_metrics["sharpe"]),
                "vs_btc_long_cash_max_drawdown": float(metrics["max_drawdown"] - long_cash_metrics["max_drawdown"]),
                "long_short_benchmark_included": False,
                "long_short_benchmark_name": best_long_short_name,
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

        if not artifacts.events.empty:
            event_tables.append(artifacts.events)

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    by_year = pd.DataFrame(by_year_rows).sort_values(["year", "candidate_name"]).reset_index(drop=True)
    events = (
        pd.concat(event_tables, axis=0, ignore_index=True)
        if event_tables
        else pd.DataFrame(
            columns=[
                "candidate_name",
                "event_start_time",
                "event_end_time",
                "trigger_type",
                "btc_return_during_event",
                "baseline_long_cash_return_during_event",
                "strategy_return_during_event",
                "avoided_loss_or_missed_gain_vs_baseline",
                "holding_period_bars",
            ]
        )
    )

    if save_csv:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        by_year_output_path.parent.mkdir(parents=True, exist_ok=True)
        events_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output_path, index=False)
        by_year.to_csv(by_year_output_path, index=False)
        events.to_csv(events_output_path, index=False)

    return summary, by_year, events
