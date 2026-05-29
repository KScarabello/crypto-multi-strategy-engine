"""Exploratory BTC crash-filter re-entry overlay research module.

This module is research-only and is not wired to live execution paths.
Positions are long/cash only and no leverage/shorting is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.btc_crash_filter_research import (
    CrashFilterCandidateSpec,
    DEFAULT_CRASH_FILTER_CANDIDATES,
    run_btc_crash_filter_research,
)
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_backtests
from research.run_expanded_universe_experiment import bars_per_year_for_timeframe


DEFAULT_SUMMARY_OUTPUT_PATH = Path("btc_crash_filter_reentry_summary.csv")
DEFAULT_BY_YEAR_OUTPUT_PATH = Path("btc_crash_filter_reentry_by_year.csv")
DEFAULT_EVENTS_OUTPUT_PATH = Path("btc_crash_filter_reentry_events.csv")


@dataclass(frozen=True)
class ReentryRuleSpec:
    reentry_type: str
    short_momentum_recovery: bool = False
    rebound_window: int | None = None
    rebound_threshold: float | None = None
    ma_window: int | None = None
    max_forced_cash_bars: int | None = None


@dataclass(frozen=True)
class CrashFilterReentryCandidateSpec:
    name: str
    base_filter_name: str
    reentry: ReentryRuleSpec


BASE_FILTERS_FOR_REENTRY: tuple[str, ...] = (
    "baseline_long_cash",
    "combo_drawdown_severe",
    "combo_all",
    "vol_120_2p0",
    "severe_12_8_cd12",
)


DEFAULT_REENTRY_CANDIDATES: tuple[CrashFilterReentryCandidateSpec, ...] = (
    CrashFilterReentryCandidateSpec(
        name="baseline_long_cash_none",
        base_filter_name="baseline_long_cash",
        reentry=ReentryRuleSpec(reentry_type="none"),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_none",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="none"),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_none",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="none"),
    ),
    CrashFilterReentryCandidateSpec(
        name="vol_120_2p0_none",
        base_filter_name="vol_120_2p0",
        reentry=ReentryRuleSpec(reentry_type="none"),
    ),
    CrashFilterReentryCandidateSpec(
        name="severe_12_8_cd12_none",
        base_filter_name="severe_12_8_cd12",
        reentry=ReentryRuleSpec(reentry_type="none"),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_reentry_short_mom",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="short_momentum_recovery", short_momentum_recovery=True),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_reentry_rebound_60_8",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="rebound_from_low", rebound_window=60, rebound_threshold=0.08),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_reentry_ma_60",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="ma_reclaim", ma_window=60),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_reentry_max_36",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="max_forced_cash_duration", max_forced_cash_bars=36),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_drawdown_severe_reentry_combined",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(
            reentry_type="combined",
            short_momentum_recovery=True,
            rebound_window=60,
            rebound_threshold=0.10,
            ma_window=60,
            max_forced_cash_bars=36,
        ),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_reentry_short_mom",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="short_momentum_recovery", short_momentum_recovery=True),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_reentry_rebound_60_8",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="rebound_from_low", rebound_window=60, rebound_threshold=0.08),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_reentry_ma_60",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="ma_reclaim", ma_window=60),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_reentry_max_36",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="max_forced_cash_duration", max_forced_cash_bars=36),
    ),
    CrashFilterReentryCandidateSpec(
        name="combo_all_reentry_combined",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(
            reentry_type="combined",
            short_momentum_recovery=True,
            rebound_window=60,
            rebound_threshold=0.10,
            ma_window=60,
            max_forced_cash_bars=36,
        ),
    ),
)


@dataclass(frozen=True)
class BtcCrashFilterReentryResearchConfig:
    symbol: str = "BTC/USD"
    timeframe: str = "4h"
    base_short_lookback: int = 60
    base_medium_lookback: int = 240
    rebalance_every_bars: int = 12
    transaction_cost_bps: float = SETTINGS.transaction_cost_bps
    slippage_bps: float = SETTINGS.slippage_bps
    candidates: Sequence[CrashFilterReentryCandidateSpec] = DEFAULT_REENTRY_CANDIDATES


@dataclass(frozen=True)
class CrashFilterReentryCandidateArtifacts:
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


def _base_filter_map() -> dict[str, CrashFilterCandidateSpec]:
    return {spec.name: spec for spec in DEFAULT_CRASH_FILTER_CANDIDATES}


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


def _drawdown_force_cash(close: pd.DataFrame, window: int, threshold: float) -> pd.Series:
    px = close.iloc[:, 0].astype(float)
    rolling_high = px.rolling(window=int(window), min_periods=1).max()
    drawdown = px / rolling_high - 1.0
    return (drawdown < float(threshold)).fillna(False)


def _volatility_force_cash(close: pd.DataFrame, short_lookback_bars: int, vol_window: int, vol_multiplier: float) -> pd.Series:
    px = close.iloc[:, 0].astype(float)
    btc_returns = px.pct_change().fillna(0.0)
    realized_vol = btc_returns.rolling(window=int(vol_window), min_periods=int(vol_window)).std(ddof=0)
    rolling_median_vol = realized_vol.rolling(window=int(vol_window), min_periods=1).median()
    short_momentum = px.pct_change(int(short_lookback_bars))
    return ((realized_vol > float(vol_multiplier) * rolling_median_vol) & (short_momentum < 0.0)).fillna(False)


def _severe_return_force_cash(close: pd.DataFrame, recent_return_window: int, threshold: float, cooldown_bars: int) -> pd.Series:
    px = close.iloc[:, 0].astype(float)
    recent_return = px.pct_change(int(recent_return_window)).fillna(0.0)
    trigger = recent_return < float(threshold)

    cooldown = pd.Series(False, index=close.index)
    remaining = 0
    for i, ts in enumerate(close.index):
        if bool(trigger.iloc[i]):
            remaining = int(cooldown_bars)
        if remaining > 0:
            cooldown.loc[ts] = True
            remaining -= 1
    return cooldown


def _build_raw_force_cash(
    close: pd.DataFrame,
    base_filter: CrashFilterCandidateSpec,
    base_short_lookback: int,
) -> tuple[pd.Series, pd.Series]:
    drawdown_force = pd.Series(False, index=close.index)
    volatility_force = pd.Series(False, index=close.index)
    severe_force = pd.Series(False, index=close.index)

    if base_filter.drawdown_filter is not None:
        drawdown_force = _drawdown_force_cash(
            close=close,
            window=base_filter.drawdown_filter.rolling_high_window,
            threshold=base_filter.drawdown_filter.drawdown_threshold,
        )

    if base_filter.volatility_filter is not None:
        volatility_force = _volatility_force_cash(
            close=close,
            short_lookback_bars=int(base_short_lookback),
            vol_window=base_filter.volatility_filter.vol_window,
            vol_multiplier=base_filter.volatility_filter.vol_multiplier,
        )

    if base_filter.severe_return_filter is not None:
        severe_force = _severe_return_force_cash(
            close=close,
            recent_return_window=base_filter.severe_return_filter.recent_return_window,
            threshold=base_filter.severe_return_filter.severe_return_threshold,
            cooldown_bars=base_filter.severe_return_filter.cooldown_bars,
        )

    raw_forced = (drawdown_force | volatility_force | severe_force).astype(bool)

    trigger_labels = []
    for ts in close.index:
        labels: list[str] = []
        if bool(drawdown_force.loc[ts]):
            labels.append("drawdown")
        if bool(volatility_force.loc[ts]):
            labels.append("volatility")
        if bool(severe_force.loc[ts]):
            labels.append("severe_return")
        trigger_labels.append("|".join(labels))
    trigger_type = pd.Series(trigger_labels, index=close.index, dtype=str)

    return raw_forced, trigger_type


def _reentry_short_momentum_recovery(close: pd.DataFrame, short_lookback_bars: int) -> pd.Series:
    short_momentum = close.iloc[:, 0].astype(float).pct_change(int(short_lookback_bars))
    return (short_momentum > 0.0).fillna(False)


def _reentry_rebound_from_low(close: pd.DataFrame, rebound_window: int, rebound_threshold: float) -> pd.Series:
    px = close.iloc[:, 0].astype(float)
    recent_low = px.rolling(window=int(rebound_window), min_periods=1).min()
    rebound = px / recent_low - 1.0
    return (rebound >= float(rebound_threshold)).fillna(False)


def _reentry_ma_reclaim(close: pd.DataFrame, ma_window: int) -> pd.Series:
    px = close.iloc[:, 0].astype(float)
    ma = px.rolling(window=int(ma_window), min_periods=int(ma_window)).mean()
    return (px > ma).fillna(False)


def _reentry_max_forced_duration(eligible_forced: pd.Series, max_forced_cash_bars: int) -> pd.Series:
    out = pd.Series(False, index=eligible_forced.index)
    streak = 0
    for ts in eligible_forced.index:
        if bool(eligible_forced.loc[ts]):
            streak += 1
        else:
            streak = 0
        out.loc[ts] = streak >= int(max_forced_cash_bars)
    return out


def _build_reentry_condition(
    close: pd.DataFrame,
    short_lookback_bars: int,
    eligible_forced: pd.Series,
    spec: ReentryRuleSpec,
) -> tuple[pd.Series, pd.Series]:
    condition = pd.Series(False, index=close.index)
    reasons: list[str] = []

    short_recovery = _reentry_short_momentum_recovery(close, short_lookback_bars) if spec.short_momentum_recovery else pd.Series(False, index=close.index)
    rebound = (
        _reentry_rebound_from_low(close, spec.rebound_window, spec.rebound_threshold)
        if spec.rebound_window is not None and spec.rebound_threshold is not None
        else pd.Series(False, index=close.index)
    )
    ma_reclaim = (
        _reentry_ma_reclaim(close, spec.ma_window)
        if spec.ma_window is not None
        else pd.Series(False, index=close.index)
    )
    max_duration = (
        _reentry_max_forced_duration(eligible_forced=eligible_forced, max_forced_cash_bars=spec.max_forced_cash_bars)
        if spec.max_forced_cash_bars is not None
        else pd.Series(False, index=close.index)
    )

    condition = (short_recovery | rebound | ma_reclaim | max_duration).fillna(False)

    for ts in close.index:
        labels: list[str] = []
        if bool(short_recovery.loc[ts]):
            labels.append("short_momentum_recovery")
        if bool(rebound.loc[ts]):
            labels.append("rebound_from_low")
        if bool(ma_reclaim.loc[ts]):
            labels.append("ma_reclaim")
        if bool(max_duration.loc[ts]):
            labels.append("max_forced_cash_duration")
        reasons.append("|".join(labels))

    return condition, pd.Series(reasons, index=close.index, dtype=str)


def _extract_forced_cash_events_with_reentry(candidate_name: str, signals: pd.DataFrame) -> pd.DataFrame:
    forced = signals["raw_forced_cash_flag"].astype(bool)
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
            reentry_labels = sorted(set([x for x in event_slice["reentry_trigger_type"].astype(str).tolist() if x]))
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "event_start_time": str(signals.index[start_idx]),
                    "event_end_time": str(signals.index[end_idx]),
                    "trigger_type": current_trigger,
                    "reentry_trigger_type": "|".join(reentry_labels),
                    "btc_return_during_event": float((1.0 + event_slice["btc_return"]).prod() - 1.0),
                    "baseline_long_cash_return_during_event": float((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0),
                    "strategy_return_during_event": float((1.0 + event_slice["strategy_return"]).prod() - 1.0),
                    "avoided_loss_or_missed_gain_vs_baseline": float(
                        ((1.0 + event_slice["strategy_return"]).prod() - 1.0)
                        - ((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0)
                    ),
                    "holding_period_bars": int(end_idx - start_idx + 1),
                    "was_reentered_early": bool(event_slice["forced_cash_overridden_by_reentry"].any()),
                }
            )
            start_idx = None
            current_trigger = ""

    if start_idx is not None:
        end_idx = len(signals.index) - 1
        event_slice = signals.iloc[start_idx : end_idx + 1]
        reentry_labels = sorted(set([x for x in event_slice["reentry_trigger_type"].astype(str).tolist() if x]))
        rows.append(
            {
                "candidate_name": candidate_name,
                "event_start_time": str(signals.index[start_idx]),
                "event_end_time": str(signals.index[end_idx]),
                "trigger_type": current_trigger,
                "reentry_trigger_type": "|".join(reentry_labels),
                "btc_return_during_event": float((1.0 + event_slice["btc_return"]).prod() - 1.0),
                "baseline_long_cash_return_during_event": float((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0),
                "strategy_return_during_event": float((1.0 + event_slice["strategy_return"]).prod() - 1.0),
                "avoided_loss_or_missed_gain_vs_baseline": float(
                    ((1.0 + event_slice["strategy_return"]).prod() - 1.0)
                    - ((1.0 + event_slice["baseline_long_cash_return"]).prod() - 1.0)
                ),
                "holding_period_bars": int(end_idx - start_idx + 1),
                "was_reentered_early": bool(event_slice["forced_cash_overridden_by_reentry"].any()),
            }
        )

    return pd.DataFrame(rows)


def _reentry_parameters_string(spec: ReentryRuleSpec) -> str:
    fields: list[str] = []
    if spec.short_momentum_recovery:
        fields.append("short_momentum_recovery=true")
    if spec.rebound_window is not None and spec.rebound_threshold is not None:
        fields.append(f"rebound_window={spec.rebound_window}")
        fields.append(f"rebound_threshold={spec.rebound_threshold}")
    if spec.ma_window is not None:
        fields.append(f"ma_window={spec.ma_window}")
    if spec.max_forced_cash_bars is not None:
        fields.append(f"max_forced_cash_bars={spec.max_forced_cash_bars}")
    return "none" if not fields else ";".join(fields)


def run_single_crash_filter_reentry_candidate(
    close_prices: pd.DataFrame,
    bars_per_year: int,
    base_filter: CrashFilterCandidateSpec,
    candidate: CrashFilterReentryCandidateSpec,
    base_short_lookback: int,
    base_medium_lookback: int,
    rebalance_every_bars: int,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> CrashFilterReentryCandidateArtifacts:
    close = _validate_close_prices(close_prices, symbol=close_prices.columns[0])
    btc_returns = close.iloc[:, 0].pct_change().fillna(0.0).astype(float)

    base_target = _build_base_long_cash_target_positions(
        close=close,
        short_lookback_bars=int(base_short_lookback),
        medium_lookback_bars=int(base_medium_lookback),
        rebalance_every_bars=int(rebalance_every_bars),
    )

    raw_forced, trigger_type = _build_raw_force_cash(
        close=close,
        base_filter=base_filter,
        base_short_lookback=int(base_short_lookback),
    )

    eligible_forced = (raw_forced & (base_target > 0.0)).astype(bool)
    reentry_condition, reentry_reason = _build_reentry_condition(
        close=close,
        short_lookback_bars=int(base_short_lookback),
        eligible_forced=eligible_forced,
        spec=candidate.reentry,
    )

    forced_cash_overridden = (eligible_forced & reentry_condition).astype(bool)

    target_position = base_target.copy()
    target_position.loc[eligible_forced] = 0.0
    target_position.loc[forced_cash_overridden] = 1.0

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
            "raw_forced_cash_flag": raw_forced.astype(bool),
            "eligible_forced_cash": eligible_forced.astype(bool),
            "forced_cash_overridden_by_reentry": forced_cash_overridden.astype(bool),
            "trigger_type": trigger_type,
            "reentry_trigger_type": reentry_reason.where(forced_cash_overridden, "").astype(str),
        },
        index=close.index,
    )

    annual_rows: list[dict[str, Any]] = []
    for year in sorted(pd.Index(signals.index.year).unique().tolist()):
        year_slice = signals.loc[signals.index.year == int(year)]
        if len(year_slice) < 2:
            continue

        year_equity = (1.0 + year_slice["strategy_return"]).cumprod()
        year_metrics = summary_metrics(
            equity=year_equity,
            returns=year_slice["strategy_return"],
            bars_per_year=bars_per_year,
        )

        forced_now = year_slice["raw_forced_cash_flag"].astype(bool)
        forced_prev = year_slice["raw_forced_cash_flag"].shift(1).fillna(False).astype(bool)
        number_of_forced_events = int((forced_now & (~forced_prev)).sum())

        override_now = year_slice["forced_cash_overridden_by_reentry"].astype(bool)
        override_prev = year_slice["forced_cash_overridden_by_reentry"].shift(1).fillna(False).astype(bool)
        reentry_events = int((override_now & (~override_prev)).sum())

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
                "percent_time_forced_cash": float(year_slice["raw_forced_cash_flag"].mean()),
                "reentry_events": reentry_events,
                "number_of_forced_cash_events": number_of_forced_events,
            }
        )

    events = _extract_forced_cash_events_with_reentry(candidate_name=candidate.name, signals=signals)

    return CrashFilterReentryCandidateArtifacts(
        candidate_name=candidate.name,
        signals=signals,
        annual_report=pd.DataFrame(annual_rows),
        events=events,
    )


def run_btc_crash_filter_reentry_research(
    config: BtcCrashFilterReentryResearchConfig | None = None,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    by_year_output_path: Path = DEFAULT_BY_YEAR_OUTPUT_PATH,
    events_output_path: Path = DEFAULT_EVENTS_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or BtcCrashFilterReentryResearchConfig()
    close = (
        _validate_close_prices(close_prices, symbol=cfg.symbol)
        if close_prices is not None
        else _load_btc_close_prices(symbol=cfg.symbol, timeframe=cfg.timeframe, data_dir=data_dir)
    )
    bars_per_year = bars_per_year_for_timeframe(cfg.timeframe)

    base_filter_map = _base_filter_map()
    missing_filters = sorted({c.base_filter_name for c in cfg.candidates} - set(base_filter_map.keys()))
    if missing_filters:
        raise ValueError(f"Unknown base_filter_name in candidates: {missing_filters}")

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

    buy_hold_metrics = summary_metrics(
        equity=(1.0 + buy_hold_returns).cumprod(),
        returns=buy_hold_returns,
        bars_per_year=bars_per_year,
    )
    long_cash_metrics = summary_metrics(
        equity=(1.0 + long_cash_returns).cumprod(),
        returns=long_cash_returns,
        bars_per_year=bars_per_year,
    )

    original_summary, _, _ = run_btc_crash_filter_research(
        close_prices=close,
        data_dir=data_dir,
        save_csv=False,
    )
    original_map = original_summary.set_index("candidate_name")

    summary_rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []

    for candidate in tuple(cfg.candidates):
        artifacts = run_single_crash_filter_reentry_candidate(
            close_prices=close,
            bars_per_year=bars_per_year,
            base_filter=base_filter_map[candidate.base_filter_name],
            candidate=candidate,
            base_short_lookback=int(cfg.base_short_lookback),
            base_medium_lookback=int(cfg.base_medium_lookback),
            rebalance_every_bars=int(cfg.rebalance_every_bars),
            transaction_cost_bps=float(cfg.transaction_cost_bps),
            slippage_bps=float(cfg.slippage_bps),
        )

        signals = artifacts.signals
        metrics = summary_metrics(
            equity=(1.0 + signals["strategy_return"]).cumprod(),
            returns=signals["strategy_return"],
            bars_per_year=bars_per_year,
        )

        override_now = signals["forced_cash_overridden_by_reentry"].astype(bool)
        override_prev = signals["forced_cash_overridden_by_reentry"].shift(1).fillna(False).astype(bool)
        reentry_events = int((override_now & (~override_prev)).sum())

        forced_cash_periods = int(signals["raw_forced_cash_flag"].sum())
        overridden_periods = int(signals["forced_cash_overridden_by_reentry"].sum())
        percent_overridden = float(overridden_periods / forced_cash_periods) if forced_cash_periods > 0 else 0.0

        base_filter_total = float(original_map.loc[candidate.base_filter_name, "total_return"]) if candidate.base_filter_name in original_map.index else 0.0
        base_filter_sharpe = float(original_map.loc[candidate.base_filter_name, "sharpe"]) if candidate.base_filter_name in original_map.index else 0.0
        base_filter_dd = float(original_map.loc[candidate.base_filter_name, "max_drawdown"]) if candidate.base_filter_name in original_map.index else 0.0

        summary_rows.append(
            {
                "candidate_name": candidate.name,
                "base_filter_name": candidate.base_filter_name,
                "reentry_type": candidate.reentry.reentry_type,
                "reentry_parameters": _reentry_parameters_string(candidate.reentry),
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
                "forced_cash_periods": forced_cash_periods,
                "percent_time_forced_cash": float(signals["raw_forced_cash_flag"].mean()),
                "reentry_events": reentry_events,
                "percent_forced_cash_overridden_by_reentry": percent_overridden,
                "total_transaction_costs": float(signals["transaction_cost"].sum()),
                "vs_btc_buy_hold_total_return": float(metrics["total_return"] - buy_hold_metrics["total_return"]),
                "vs_btc_buy_hold_sharpe": float(metrics["sharpe"] - buy_hold_metrics["sharpe"]),
                "vs_btc_buy_hold_max_drawdown": float(metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"]),
                "vs_btc_long_cash_total_return": float(metrics["total_return"] - long_cash_metrics["total_return"]),
                "vs_btc_long_cash_sharpe": float(metrics["sharpe"] - long_cash_metrics["sharpe"]),
                "vs_btc_long_cash_max_drawdown": float(metrics["max_drawdown"] - long_cash_metrics["max_drawdown"]),
                "vs_original_crash_filter_total_return": float(metrics["total_return"] - base_filter_total),
                "vs_original_crash_filter_sharpe": float(metrics["sharpe"] - base_filter_sharpe),
                "vs_original_crash_filter_max_drawdown": float(metrics["max_drawdown"] - base_filter_dd),
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

    summary = pd.DataFrame(summary_rows).sort_values(["sharpe", "total_return"], ascending=[False, False]).reset_index(drop=True)
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
                "reentry_trigger_type",
                "btc_return_during_event",
                "baseline_long_cash_return_during_event",
                "strategy_return_during_event",
                "avoided_loss_or_missed_gain_vs_baseline",
                "holding_period_bars",
                "was_reentered_early",
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
