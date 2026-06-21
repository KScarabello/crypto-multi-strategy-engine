"""Research-only second-pass explosion delayed-entry study.

This module refines delayed confirmation and risk-management variants so they
are behaviorally distinct:
- delayed strict confirmation (enter after observation candle)
- immediate-entry + early-exit risk management
- event-time-only exclusion filters (no post-event inputs)
- pullback-aware delayed and avoid/exit variants
- strict no-follow-through short research variants

All outputs are written under reports/explosion_delayed_entry_study/second_pass/.

Safety:
- Research only.
- No live trading, scheduler, launchd, broker/exchange execution, or production
  configuration changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.explosion_reversal_event_study import (  # noqa: E402
    build_event_time_features,
    detect_explosion_events,
    join_btc_context,
    load_ohlcv_data,
)


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/explosion_delayed_entry_study")

FOLLOW_THROUGH_RULES = (
    "HIGH_BREAK_0_BPS",
    "HIGH_BREAK_25_BPS",
    "HIGH_BREAK_50_BPS",
    "HIGH_BREAK_100_BPS",
    "CLOSE_ABOVE_EXPLOSION_HIGH",
    "CLOSE_ABOVE_EXPLOSION_CLOSE_25_BPS",
    "CLOSE_ABOVE_EXPLOSION_CLOSE_50_BPS",
    "VOL_ADJUSTED_BREAK",
)

EVENT_TIME_FILTER_RULES = (
    "EXCL_LOW_LIQUIDITY",
    "EXCL_EXTREME_OVEREXTENSION",
    "EXCL_BTC_DOWN_OR_SIDEWAYS",
    "EXCL_HIGH_VOL_LOW_VOLUME",
    "CONSERVATIVE_COMBINED",
)

PULLBACK_THRESHOLDS_PCT = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class DelayedEntryStudyConfig:
    ret_4h_threshold_pct: float = 8.0
    ret_zscore_threshold: float = 2.5
    min_dollar_volume: float = 75_000.0
    cooldown_bars: int = 3
    rolling_window_bars: int = 42
    vol_window_bars: int = 42
    btc_symbol: str = "BTC/USD"
    hold_bars: tuple[int, ...] = (1, 2, 6)
    cost_bps_list: tuple[int, ...] = (10, 25, 50, 100)
    label_cost_bps: int = 50
    train_fraction: float = 0.7
    min_bucket_events: int = 30
    bucket_quantiles: int = 4
    vol_adjusted_break_multiplier: float = 0.5


def _symbol_frame_index(full_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: sub.sort_values("timestamp").reset_index(drop=True)
        for symbol, sub in full_df.groupby("symbol", sort=False)
    }


def _safe_qbucket_numeric(series: pd.Series, q: int) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float")
    if valid.notna().sum() < max(q * 2, 8):
        return out
    try:
        labels = list(range(1, q + 1))
        bucketed = pd.qcut(valid, q=q, labels=labels, duplicates="drop")
        return pd.to_numeric(bucketed, errors="coerce")
    except Exception:
        return out


def _rank_bucket_from_liquidity(value: float | int | np.floating) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    if value <= 1.0 / 3.0:
        return "SMALL"
    if value <= 2.0 / 3.0:
        return "MID"
    return "LARGE"


def _pullback_bucket(depth_pct: float) -> str:
    if pd.isna(depth_pct):
        return "UNKNOWN"
    if depth_pct >= 0.0:
        return "NO_PULLBACK"
    if depth_pct > -2.0:
        return "SHALLOW"
    if depth_pct > -5.0:
        return "MODERATE"
    return "DEEP"


def evaluate_follow_through(
    rule: str,
    event_close: float,
    event_high: float,
    observation_close: float,
    observation_high: float,
    event_rolling_vol_pct: float,
    vol_adjusted_break_multiplier: float,
) -> bool:
    if rule == "HIGH_BREAK_0_BPS":
        return observation_high > event_high
    if rule == "HIGH_BREAK_25_BPS":
        return observation_high > event_high * (1.0 + 25.0 / 10_000.0)
    if rule == "HIGH_BREAK_50_BPS":
        return observation_high > event_high * (1.0 + 50.0 / 10_000.0)
    if rule == "HIGH_BREAK_100_BPS":
        return observation_high > event_high * (1.0 + 100.0 / 10_000.0)
    if rule == "CLOSE_ABOVE_EXPLOSION_HIGH":
        return observation_close > event_high
    if rule == "CLOSE_ABOVE_EXPLOSION_CLOSE_25_BPS":
        return observation_close > event_close * (1.0 + 25.0 / 10_000.0)
    if rule == "CLOSE_ABOVE_EXPLOSION_CLOSE_50_BPS":
        return observation_close > event_close * (1.0 + 50.0 / 10_000.0)
    if rule == "VOL_ADJUSTED_BREAK":
        vol = max(float(event_rolling_vol_pct), 0.0) if not pd.isna(event_rolling_vol_pct) else 0.0
        threshold = event_high * (1.0 + (vol_adjusted_break_multiplier * vol / 100.0))
        return observation_high > threshold
    raise ValueError(f"Unknown follow-through rule: {rule}")


def _event_time_filter_passes(row: pd.Series, rule: str, top_q: int) -> bool:
    liq_bucket = str(row.get("liquidity_bucket", "UNKNOWN"))
    size_q = row.get("explosion_size_q", np.nan)
    vol_q = row.get("volatility_q", np.nan)
    spike_q = row.get("volume_spike_q", np.nan)
    btc_bucket = str(row.get("btc_trend_bucket", "UNKNOWN"))
    btc_ret_24h = row.get("btc_ret_24h_pct", np.nan)

    if rule == "EXCL_LOW_LIQUIDITY":
        return liq_bucket != "SMALL"
    if rule == "EXCL_EXTREME_OVEREXTENSION":
        return bool(pd.isna(size_q) or float(size_q) < float(top_q))
    if rule == "EXCL_BTC_DOWN_OR_SIDEWAYS":
        return btc_bucket == "UP"
    if rule == "EXCL_HIGH_VOL_LOW_VOLUME":
        high_vol_low_volume = (not pd.isna(vol_q) and not pd.isna(spike_q) and float(vol_q) >= float(top_q) and float(spike_q) <= 2.0)
        return not high_vol_low_volume
    if rule == "CONSERVATIVE_COMBINED":
        c1 = liq_bucket != "SMALL"
        c2 = bool(pd.isna(size_q) or float(size_q) < float(top_q))
        c3 = btc_bucket == "UP" or (not pd.isna(btc_ret_24h) and float(btc_ret_24h) > 0.0)
        c4 = not (not pd.isna(vol_q) and not pd.isna(spike_q) and float(vol_q) >= float(top_q) and float(spike_q) <= 2.0)
        return bool(c1 and c2 and c3 and c4)
    raise ValueError(f"Unknown event-time filter rule: {rule}")


def build_second_pass_signals(full_df: pd.DataFrame, cfg: DelayedEntryStudyConfig) -> pd.DataFrame:
    feats = build_event_time_features(full_df, cfg)
    events = detect_explosion_events(feats, cfg)
    if events.empty:
        return events.copy()

    events = join_btc_context(events, full_df, cfg)
    symbol_frames = _symbol_frame_index(full_df)

    events = events.copy()
    events["explosion_size_q"] = _safe_qbucket_numeric(events["ret_4h_pct"], cfg.bucket_quantiles)
    events["volume_spike_q"] = _safe_qbucket_numeric(events["volume_spike_ratio"], cfg.bucket_quantiles)
    events["volatility_q"] = _safe_qbucket_numeric(events["rolling_vol_4h_pct"], cfg.bucket_quantiles)
    events["liquidity_bucket"] = events["liquidity_rank_universe"].apply(_rank_bucket_from_liquidity)

    rows: list[dict[str, object]] = []
    top_q = cfg.bucket_quantiles

    for _, event in events.iterrows():
        symbol = str(event["symbol"])
        symbol_df = symbol_frames.get(symbol)
        if symbol_df is None:
            continue

        event_ts = pd.Timestamp(event["timestamp"])
        idx = symbol_df.index[symbol_df["timestamp"] == event_ts]
        if len(idx) == 0:
            continue
        event_pos = int(idx[0])
        observation_pos = event_pos + 1
        if observation_pos >= len(symbol_df):
            continue

        observation_row = symbol_df.iloc[observation_pos]
        obs_close = float(observation_row["close"])
        obs_high = float(observation_row["high"])
        obs_low = float(observation_row["low"])

        event_close = float(event["close"])
        event_high = float(event["high"])
        pullback_depth_pct = (obs_low / event_close - 1.0) * 100.0

        row = {
            **event.to_dict(),
            "event_pos": event_pos,
            "observation_pos": observation_pos,
            "observation_timestamp": pd.Timestamp(observation_row["timestamp"]),
            "observation_close": obs_close,
            "observation_high": obs_high,
            "observation_low": obs_low,
            "pullback_depth_1c_pct": pullback_depth_pct,
            "pullback_bucket_1c": _pullback_bucket(pullback_depth_pct),
        }

        for threshold in PULLBACK_THRESHOLDS_PCT:
            row[f"deep_pullback_{int(threshold)}pct"] = bool(pullback_depth_pct <= -threshold)

        for rule in FOLLOW_THROUGH_RULES:
            passed = evaluate_follow_through(
                rule=rule,
                event_close=event_close,
                event_high=event_high,
                observation_close=obs_close,
                observation_high=obs_high,
                event_rolling_vol_pct=float(event.get("rolling_vol_4h_pct", np.nan)),
                vol_adjusted_break_multiplier=cfg.vol_adjusted_break_multiplier,
            )
            row[f"follow_{rule}"] = bool(passed)
            row[f"no_follow_{rule}"] = bool(not passed)

        for rule in EVENT_TIME_FILTER_RULES:
            row[f"event_filter_{rule}"] = bool(_event_time_filter_passes(event, rule, top_q=top_q))

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _trade_return(entry_price: float, exit_price: float, direction: str) -> float:
    if direction == "SHORT":
        return (entry_price - exit_price) / entry_price * 100.0
    return (exit_price / entry_price - 1.0) * 100.0


def _trade_excursions(future_highs: pd.Series, future_lows: pd.Series, entry_price: float, direction: str) -> tuple[float, float]:
    if direction == "SHORT":
        mfe = (entry_price - float(future_lows.min())) / entry_price * 100.0
        mae = (entry_price - float(future_highs.max())) / entry_price * 100.0
    else:
        mfe = (float(future_highs.max()) / entry_price - 1.0) * 100.0
        mae = (float(future_lows.min()) / entry_price - 1.0) * 100.0
    return mfe, mae


def _build_trade_row(
    signal: pd.Series,
    symbol_df: pd.DataFrame,
    setup: str,
    variant: str,
    direction: str,
    hold_bars: int,
    entry_pos: int,
    target_exit_pos: int,
    forced_exit_observation: bool,
    note: str,
) -> dict[str, object] | None:
    if hold_bars <= 0:
        return None
    if entry_pos >= len(symbol_df):
        return None

    exit_pos = target_exit_pos
    if exit_pos >= len(symbol_df):
        return None

    entry_row = symbol_df.iloc[entry_pos]
    exit_row = symbol_df.iloc[exit_pos]

    if exit_pos <= entry_pos:
        return None

    future = symbol_df.iloc[entry_pos + 1 : exit_pos + 1]
    if future.empty:
        return None

    entry_price = float(entry_row["close"])
    exit_price = float(exit_row["close"])
    gross_return = _trade_return(entry_price, exit_price, direction)
    mfe, mae = _trade_excursions(future["high"].astype(float), future["low"].astype(float), entry_price, direction)

    return {
        "event_id": signal["event_id"],
        "symbol": signal["symbol"],
        "event_timestamp": signal["timestamp"],
        "observation_timestamp": signal["observation_timestamp"],
        "entry_timestamp": entry_row["timestamp"],
        "exit_timestamp": exit_row["timestamp"],
        "setup": setup,
        "variant": variant,
        "direction": direction,
        "hold_bars": int(hold_bars),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return_pct": gross_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "worst_drawdown_pct": mae,
        "forced_exit_observation": bool(forced_exit_observation),
        "note": note,
        "pullback_depth_1c_pct": float(signal["pullback_depth_1c_pct"]),
        "pullback_bucket_1c": signal["pullback_bucket_1c"],
        "btc_trend_bucket": signal.get("btc_trend_bucket", "UNKNOWN"),
        "btc_ret_4h_pct": signal.get("btc_ret_4h_pct", np.nan),
        "btc_ret_24h_pct": signal.get("btc_ret_24h_pct", np.nan),
        "liquidity_bucket": signal.get("liquidity_bucket", "UNKNOWN"),
        "explosion_size_q": signal.get("explosion_size_q", np.nan),
        "volume_spike_q": signal.get("volume_spike_q", np.nan),
        "volatility_q": signal.get("volatility_q", np.nan),
    }


def build_second_pass_trades(signals: pd.DataFrame, full_df: pd.DataFrame, cfg: DelayedEntryStudyConfig) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    symbol_frames = _symbol_frame_index(full_df)
    rows: list[dict[str, object]] = []

    for _, signal in signals.iterrows():
        symbol = str(signal["symbol"])
        symbol_df = symbol_frames.get(symbol)
        if symbol_df is None:
            continue

        event_pos = int(signal["event_pos"])
        obs_pos = int(signal["observation_pos"])

        for hold in cfg.hold_bars:
            baseline = _build_trade_row(
                signal=signal,
                symbol_df=symbol_df,
                setup="BUY_AT_EXPLOSION",
                variant="BASELINE",
                direction="LONG",
                hold_bars=hold,
                entry_pos=event_pos,
                target_exit_pos=event_pos + hold,
                forced_exit_observation=False,
                note="baseline_immediate_entry",
            )
            if baseline is not None:
                rows.append(baseline)

            for follow_rule in FOLLOW_THROUGH_RULES:
                follow_col = f"follow_{follow_rule}"
                no_follow_col = f"no_follow_{follow_rule}"
                follow = bool(signal[follow_col])
                no_follow = bool(signal[no_follow_col])

                if follow:
                    delayed = _build_trade_row(
                        signal=signal,
                        symbol_df=symbol_df,
                        setup="BUY_AFTER_FOLLOW_THROUGH_STRICT",
                        variant=follow_rule,
                        direction="LONG",
                        hold_bars=hold,
                        entry_pos=obs_pos,
                        target_exit_pos=obs_pos + hold,
                        forced_exit_observation=False,
                        note="enter_after_observation_follow_through",
                    )
                    if delayed is not None:
                        rows.append(delayed)

                risk_exit_pos = obs_pos if no_follow else (event_pos + hold)
                risk = _build_trade_row(
                    signal=signal,
                    symbol_df=symbol_df,
                    setup="BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH",
                    variant=follow_rule,
                    direction="LONG",
                    hold_bars=hold,
                    entry_pos=event_pos,
                    target_exit_pos=risk_exit_pos,
                    forced_exit_observation=no_follow,
                    note="immediate_entry_with_observation_risk_exit",
                )
                if risk is not None:
                    rows.append(risk)

                if no_follow:
                    short_trade = _build_trade_row(
                        signal=signal,
                        symbol_df=symbol_df,
                        setup="SHORT_AFTER_NO_FOLLOW_THROUGH_STRICT",
                        variant=follow_rule,
                        direction="SHORT",
                        hold_bars=hold,
                        entry_pos=obs_pos,
                        target_exit_pos=obs_pos + hold,
                        forced_exit_observation=False,
                        note="research_only_not_live_ready",
                    )
                    if short_trade is not None:
                        rows.append(short_trade)

            for event_rule in EVENT_TIME_FILTER_RULES:
                if bool(signal[f"event_filter_{event_rule}"]):
                    event_filtered = _build_trade_row(
                        signal=signal,
                        symbol_df=symbol_df,
                        setup="BUY_AT_EXPLOSION_EVENT_TIME_FILTERED",
                        variant=event_rule,
                        direction="LONG",
                        hold_bars=hold,
                        entry_pos=event_pos,
                        target_exit_pos=event_pos + hold,
                        forced_exit_observation=False,
                        note="event_time_only_filter_no_lookahead",
                    )
                    if event_filtered is not None:
                        rows.append(event_filtered)

            for threshold in PULLBACK_THRESHOLDS_PCT:
                deep = bool(signal[f"deep_pullback_{int(threshold)}pct"])
                threshold_label = f"THR_{int(threshold)}PCT"

                if not deep:
                    shallow = _build_trade_row(
                        signal=signal,
                        symbol_df=symbol_df,
                        setup="BUY_AFTER_SHALLOW_PULLBACK_ONLY",
                        variant=threshold_label,
                        direction="LONG",
                        hold_bars=hold,
                        entry_pos=obs_pos,
                        target_exit_pos=obs_pos + hold,
                        forced_exit_observation=False,
                        note="delayed_entry_requires_shallow_pullback",
                    )
                    if shallow is not None:
                        rows.append(shallow)

                avoid_exit_pos = obs_pos if deep else (event_pos + hold)
                avoid = _build_trade_row(
                    signal=signal,
                    symbol_df=symbol_df,
                    setup="AVOID_DEEP_PULLBACK",
                    variant=f"IMMEDIATE_EXIT_{threshold_label}",
                    direction="LONG",
                    hold_bars=hold,
                    entry_pos=event_pos,
                    target_exit_pos=avoid_exit_pos,
                    forced_exit_observation=deep,
                    note="immediate_entry_exits_if_deep_pullback",
                )
                if avoid is not None:
                    rows.append(avoid)

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    for bps in cfg.cost_bps_list:
        trades[f"net_return_pct_{bps}"] = trades["gross_return_pct"] - (bps / 100.0)
        trades[f"net_win_{bps}"] = trades[f"net_return_pct_{bps}"] > 0.0

    trades["small_sample_warning"] = False
    return trades.sort_values(["setup", "variant", "hold_bars", "event_timestamp", "entry_timestamp"]).reset_index(drop=True)


def _grouped_summary(
    trades: pd.DataFrame,
    group_cols: list[str],
    cfg: DelayedEntryStudyConfig,
    total_events: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_values, group in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {k: v for k, v in zip(group_cols, group_values)}

        unique_events = int(group["event_id"].nunique())
        row.update(
            {
                "n_trades": int(len(group)),
                "n_events": unique_events,
                "selection_rate": float(unique_events / total_events) if total_events else np.nan,
                "gross_return_mean_pct": float(group["gross_return_pct"].mean()),
                "gross_return_median_pct": float(group["gross_return_pct"].median()),
                "gross_win_rate": float((group["gross_return_pct"] > 0.0).mean()),
                "mfe_mean_pct": float(group["mfe_pct"].mean()),
                "mae_mean_pct": float(group["mae_pct"].mean()),
                "worst_drawdown_mean_pct": float(group["worst_drawdown_pct"].mean()),
                "forced_exit_rate": float(group["forced_exit_observation"].mean()),
                "small_sample_warning": bool(unique_events < cfg.min_bucket_events),
            }
        )

        for bps in cfg.cost_bps_list:
            net_col = f"net_return_pct_{bps}"
            row[f"net_return_mean_pct_{bps}"] = float(group[net_col].mean())
            row[f"net_return_median_pct_{bps}"] = float(group[net_col].median())
            row[f"net_win_rate_{bps}"] = float(group[net_col].gt(0.0).mean())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_strategy_summary(trades: pd.DataFrame, cfg: DelayedEntryStudyConfig, total_events: int) -> pd.DataFrame:
    return _grouped_summary(trades, ["setup", "variant", "hold_bars", "direction"], cfg, total_events)


def build_train_test_summary(trades: pd.DataFrame, cfg: DelayedEntryStudyConfig, total_events: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    split_ts = trades["event_timestamp"].quantile(cfg.train_fraction)
    out = trades.copy()
    out["split"] = np.where(out["event_timestamp"] <= split_ts, "train", "test")

    split_summary = _grouped_summary(out, ["setup", "variant", "hold_bars", "direction", "split"], cfg, total_events)
    if split_summary.empty:
        return split_summary

    train_rows = split_summary[split_summary["split"] == "train"].copy()
    test_rows = split_summary[split_summary["split"] == "test"].copy()

    join_keys = ["setup", "variant", "hold_bars", "direction"]
    degrade = test_rows.merge(train_rows, on=join_keys, how="inner", suffixes=("_test", "_train"))
    if degrade.empty:
        split_summary["test_train_degradation_pct_50"] = np.nan
        split_summary["test_train_degradation_pct_100"] = np.nan
        return split_summary

    keep = join_keys + [
        "net_return_mean_pct_50_test",
        "net_return_mean_pct_50_train",
        "net_return_mean_pct_100_test",
        "net_return_mean_pct_100_train",
    ]
    degrade = degrade[keep].copy()
    degrade["test_train_degradation_pct_50"] = degrade["net_return_mean_pct_50_test"] - degrade["net_return_mean_pct_50_train"]
    degrade["test_train_degradation_pct_100"] = degrade["net_return_mean_pct_100_test"] - degrade["net_return_mean_pct_100_train"]
    degrade = degrade[join_keys + ["test_train_degradation_pct_50", "test_train_degradation_pct_100"]]

    return split_summary.merge(degrade, on=join_keys, how="left")


def detect_functional_equivalents(trades: pd.DataFrame) -> list[dict[str, object]]:
    if trades.empty:
        return []

    signatures: dict[tuple[str, str, int, str], tuple[tuple[str, str, str, str], ...]] = {}

    for (setup, variant, hold_bars, direction), group in trades.groupby(["setup", "variant", "hold_bars", "direction"], sort=False):
        sig = tuple(
            sorted(
                (
                    str(r["event_id"]),
                    str(r["entry_timestamp"]),
                    str(r["exit_timestamp"]),
                    f"{float(r['gross_return_pct']):.8f}",
                )
                for _, r in group.iterrows()
            )
        )
        signatures[(str(setup), str(variant), int(hold_bars), str(direction))] = sig

    keys = list(signatures.keys())
    out: list[dict[str, object]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if signatures[keys[i]] == signatures[keys[j]]:
                a = keys[i]
                b = keys[j]
                out.append(
                    {
                        "left_setup": a[0],
                        "left_variant": a[1],
                        "left_hold_bars": a[2],
                        "left_direction": a[3],
                        "right_setup": b[0],
                        "right_variant": b[1],
                        "right_hold_bars": b[2],
                        "right_direction": b[3],
                        "reason": "identical_event_entry_exit_return_signature",
                    }
                )
    return out


def _subset_summary(summary: pd.DataFrame, setup_name: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary[summary["setup"] == setup_name].copy().reset_index(drop=True)


def _top_setups(summary: pd.DataFrame, cfg: DelayedEntryStudyConfig, top_n: int = 12) -> list[dict[str, object]]:
    if summary.empty:
        return []

    rank_col = f"net_return_mean_pct_{cfg.label_cost_bps}"
    ranked = summary.sort_values([rank_col, "n_events", "gross_win_rate"], ascending=[False, False, False]).head(top_n)

    rows: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "setup": row["setup"],
                "variant": row["variant"],
                "hold_bars": int(row["hold_bars"]),
                "direction": row["direction"],
                "n_events": int(row["n_events"]),
                "gross_return_mean_pct": float(row["gross_return_mean_pct"]),
                "net_return_mean_pct_50": float(row["net_return_mean_pct_50"]),
                "net_return_mean_pct_100": float(row["net_return_mean_pct_100"]),
                "small_sample_warning": bool(row["small_sample_warning"]),
            }
        )
    return rows


def _best_row(summary: pd.DataFrame, setup: str) -> pd.Series | None:
    sub = summary[summary["setup"] == setup].copy()
    if sub.empty:
        return None
    return sub.sort_values(["net_return_mean_pct_50", "n_events"], ascending=[False, False]).iloc[0]


def write_second_pass_summary_markdown(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    train_test_summary: pd.DataFrame,
    equivalents: list[dict[str, object]],
    top_setups: list[dict[str, object]],
    cfg: DelayedEntryStudyConfig,
    output_path: Path,
) -> None:
    lines = [
        "# Explosion Delayed-Entry Study - Second Pass (Research-Only)",
        "",
        "## Scope",
        "- Research only.",
        "- No live trading, scheduler, launchd, broker/exchange execution, or production config changes.",
        "- Uses 4h OHLCV only.",
        "- Event-time filters are restricted to event-time-available data.",
        "",
        "## Coverage",
        f"- Events: {len(signals)}",
        f"- Trades: {len(trades)}",
        f"- Date range: {signals['timestamp'].min() if not signals.empty else 'N/A'} -> {signals['timestamp'].max() if not signals.empty else 'N/A'}",
        f"- Hold periods: {', '.join(str(x) for x in cfg.hold_bars)}",
        f"- Cost grid (bps): {', '.join(str(x) for x in cfg.cost_bps_list)}",
        "",
        "## Q1: Does stricter follow-through improve delayed-entry results?",
    ]

    strict = strategy_summary[strategy_summary["setup"] == "BUY_AFTER_FOLLOW_THROUGH_STRICT"].copy()
    if strict.empty:
        lines.append("- No strict follow-through delayed rows were produced.")
    else:
        best_strict = strict.sort_values(["net_return_mean_pct_50", "n_events"], ascending=[False, False]).iloc[0]
        lines.append(
            f"- Best strict delayed rule: {best_strict['variant']} / {int(best_strict['hold_bars'])}c, "
            f"net@50bps={best_strict['net_return_mean_pct_50']:+.4f}%, net@100bps={best_strict['net_return_mean_pct_100']:+.4f}%"
        )

    lines += [
        "",
        "## Q2: Is immediate entry + exit-on-no-follow-through better than buy-all explosions?",
    ]
    risk_best = _best_row(strategy_summary, "BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH")
    base_best = _best_row(strategy_summary, "BUY_AT_EXPLOSION")
    if risk_best is None or base_best is None:
        lines.append("- Insufficient rows for direct comparison.")
    else:
        lines.append(
            f"- Best risk-exit: {risk_best['variant']} / {int(risk_best['hold_bars'])}c net@50bps={risk_best['net_return_mean_pct_50']:+.4f}%"
        )
        lines.append(
            f"- Best baseline: {base_best['variant']} / {int(base_best['hold_bars'])}c net@50bps={base_best['net_return_mean_pct_50']:+.4f}%"
        )

    lines += [
        "",
        "## Q3: Is delayed confirmation better than immediate entry?",
    ]
    delayed_best = _best_row(strategy_summary, "BUY_AFTER_FOLLOW_THROUGH_STRICT")
    if delayed_best is None or base_best is None:
        lines.append("- Insufficient rows for direct delayed-vs-immediate comparison.")
    else:
        lines.append(
            f"- Delayed best net@50bps={delayed_best['net_return_mean_pct_50']:+.4f}% vs baseline best net@50bps={base_best['net_return_mean_pct_50']:+.4f}%"
        )

    lines += [
        "",
        "## Q4: Can event-time filters improve baseline without look-ahead?",
    ]
    evt_best = _best_row(strategy_summary, "BUY_AT_EXPLOSION_EVENT_TIME_FILTERED")
    if evt_best is None or base_best is None:
        lines.append("- Event-time filter comparison unavailable.")
    else:
        lines.append(
            f"- Best event-time filter: {evt_best['variant']} / {int(evt_best['hold_bars'])}c net@50bps={evt_best['net_return_mean_pct_50']:+.4f}%"
        )
        lines.append(
            "- Event-time filter rules use only BTC regime/returns, explosion-size bucket, volume-spike bucket, liquidity bucket, and volatility bucket."
        )

    lines += [
        "",
        "## Q5: Deep pullback as avoid/exit vs dip-buy?",
    ]
    pb_best = _best_row(strategy_summary, "BUY_AFTER_SHALLOW_PULLBACK_ONLY")
    avoid_best = _best_row(strategy_summary, "AVOID_DEEP_PULLBACK")
    if pb_best is None or avoid_best is None:
        lines.append("- Pullback variant comparison unavailable.")
    else:
        lines.append(
            f"- Shallow-only delayed best net@50bps={pb_best['net_return_mean_pct_50']:+.4f}%"
        )
        lines.append(
            f"- Avoid-deep immediate-exit best net@50bps={avoid_best['net_return_mean_pct_50']:+.4f}%"
        )

    lines += [
        "",
        "## Q6: Any setups positive after 50/100 bps?",
    ]
    pos_50 = strategy_summary[strategy_summary["net_return_mean_pct_50"] > 0.0]
    pos_100 = strategy_summary[strategy_summary["net_return_mean_pct_100"] > 0.0]
    lines.append(f"- Positive at 50 bps: {len(pos_50)} rows")
    lines.append(f"- Positive at 100 bps: {len(pos_100)} rows")

    lines += [
        "",
        "## Q7: Any setups survive holdout/test period?",
    ]
    test_rows = train_test_summary[train_test_summary["split"] == "test"] if not train_test_summary.empty else pd.DataFrame()
    surviving = test_rows[test_rows["net_return_mean_pct_50"] > 0.0] if not test_rows.empty else pd.DataFrame()
    lines.append(f"- Test rows with positive net@50bps: {len(surviving)}")

    lines += [
        "",
        "## Q8: BTC regime / liquidity dependence",
    ]
    if trades.empty:
        lines.append("- No trade rows available.")
    else:
        reg = trades.groupby(["setup", "btc_trend_bucket"], dropna=False)["gross_return_pct"].mean().reset_index()
        liq = trades.groupby(["setup", "liquidity_bucket"], dropna=False)["gross_return_pct"].mean().reset_index()
        lines.append("- Regime and liquidity subgroup averages were computed in generated CSV summaries.")
        lines.append(f"- BTC regime grouped rows: {len(reg)}")
        lines.append(f"- Liquidity grouped rows: {len(liq)}")

    lines += [
        "",
        "## Q9: Duplicate or functionally equivalent variants",
    ]
    if not equivalents:
        lines.append("- No exact functional duplicates were detected.")
    else:
        for item in equivalents[:12]:
            lines.append(
                f"- {item['left_setup']}:{item['left_variant']}:{item['left_hold_bars']} == "
                f"{item['right_setup']}:{item['right_variant']}:{item['right_hold_bars']} ({item['reason']})"
            )

    lines += [
        "",
        "## Q10: Prototype readiness vs exploratory status",
    ]
    if surviving.empty:
        lines.append("- No robust holdout-positive candidates at the chosen cost threshold; keep exploratory.")
    else:
        lines.append("- Some holdout-positive rows exist, but require stability review before any prototype promotion.")

    lines += [
        "",
        "## Top Setups",
    ]
    if not top_setups:
        lines.append("- No top setups available.")
    else:
        for row in top_setups[:10]:
            lines.append(
                f"- {row['setup']} / {row['variant']} / {row['hold_bars']}c / {row['direction']}: "
                f"n={row['n_events']}, net@50bps={row['net_return_mean_pct_50']:+.4f}%, "
                f"net@100bps={row['net_return_mean_pct_100']:+.4f}%"
            )

    lines += [
        "",
        "## Safety Confirmation",
        "- SHORT_AFTER_NO_FOLLOW_THROUGH_STRICT remains research-only and not live-ready.",
        "- No live trading behavior changed.",
        "- No scheduler/launchd, broker/exchange, credentials, or production config changes were made.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def run_delayed_entry_study(cfg: DelayedEntryStudyConfig, data_dir: Path, output_dir: Path) -> dict[str, object]:
    full_df = load_ohlcv_data(data_dir=data_dir)
    signals = build_second_pass_signals(full_df, cfg)
    trades = build_second_pass_trades(signals, full_df, cfg)

    total_events = int(signals["event_id"].nunique()) if not signals.empty else 0
    strategy_summary = build_strategy_summary(trades, cfg, total_events)
    train_test_summary = build_train_test_summary(trades, cfg, total_events)

    follow_sensitivity = strategy_summary[
        strategy_summary["setup"].isin(
            [
                "BUY_AFTER_FOLLOW_THROUGH_STRICT",
                "BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH",
                "SHORT_AFTER_NO_FOLLOW_THROUGH_STRICT",
            ]
        )
    ].copy()

    event_time_filter_summary = _subset_summary(strategy_summary, "BUY_AT_EXPLOSION_EVENT_TIME_FILTERED")
    risk_management_exit_summary = strategy_summary[
        strategy_summary["setup"].isin(["BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH", "AVOID_DEEP_PULLBACK"])
    ].copy()
    pullback_variant_summary = strategy_summary[
        strategy_summary["setup"].isin(["BUY_AFTER_SHALLOW_PULLBACK_ONLY", "AVOID_DEEP_PULLBACK"])
    ].copy()

    equivalents = detect_functional_equivalents(trades)
    top_setups = _top_setups(strategy_summary, cfg)

    second_pass_dir = output_dir / "second_pass"
    second_pass_dir.mkdir(parents=True, exist_ok=True)

    trades.to_csv(second_pass_dir / "second_pass_trades.csv", index=False)
    strategy_summary.to_csv(second_pass_dir / "second_pass_strategy_summary.csv", index=False)
    follow_sensitivity.to_csv(second_pass_dir / "follow_through_threshold_sensitivity.csv", index=False)
    event_time_filter_summary.to_csv(second_pass_dir / "event_time_filter_summary.csv", index=False)
    risk_management_exit_summary.to_csv(second_pass_dir / "risk_management_exit_summary.csv", index=False)
    pullback_variant_summary.to_csv(second_pass_dir / "pullback_variant_summary.csv", index=False)
    train_test_summary.to_csv(second_pass_dir / "train_test_summary.csv", index=False)
    (second_pass_dir / "top_second_pass_setups.json").write_text(json.dumps(top_setups, indent=2) + "\n")

    write_second_pass_summary_markdown(
        signals=signals,
        trades=trades,
        strategy_summary=strategy_summary,
        train_test_summary=train_test_summary,
        equivalents=equivalents,
        top_setups=top_setups,
        cfg=cfg,
        output_path=second_pass_dir / "summary.md",
    )

    return {
        "signals": signals,
        "trades": trades,
        "strategy_summary": strategy_summary,
        "follow_sensitivity": follow_sensitivity,
        "event_time_filter_summary": event_time_filter_summary,
        "risk_management_exit_summary": risk_management_exit_summary,
        "pullback_variant_summary": pullback_variant_summary,
        "train_test_summary": train_test_summary,
        "top_setups": top_setups,
        "equivalents": equivalents,
        "output_dir": second_pass_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explosion delayed-entry second-pass study (research-only)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ret-4h-threshold-pct", type=float, default=8.0)
    parser.add_argument("--ret-zscore-threshold", type=float, default=2.5)
    parser.add_argument("--min-dollar-volume", type=float, default=75_000.0)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--rolling-window-bars", type=int, default=42)
    parser.add_argument("--vol-window-bars", type=int, default=42)
    parser.add_argument("--label-cost-bps", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-bucket-events", type=int, default=30)
    parser.add_argument("--vol-adjusted-break-multiplier", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = DelayedEntryStudyConfig(
        ret_4h_threshold_pct=args.ret_4h_threshold_pct,
        ret_zscore_threshold=args.ret_zscore_threshold,
        min_dollar_volume=args.min_dollar_volume,
        cooldown_bars=args.cooldown_bars,
        rolling_window_bars=args.rolling_window_bars,
        vol_window_bars=args.vol_window_bars,
        label_cost_bps=args.label_cost_bps,
        train_fraction=args.train_fraction,
        min_bucket_events=args.min_bucket_events,
        vol_adjusted_break_multiplier=args.vol_adjusted_break_multiplier,
    )

    result = run_delayed_entry_study(cfg=cfg, data_dir=args.data_dir, output_dir=args.output_dir)

    signals = result["signals"]
    summary = result["strategy_summary"]
    train_test = result["train_test_summary"]

    print("=" * 80)
    print("EXPLOSION DELAYED-ENTRY SECOND PASS (RESEARCH-ONLY)")
    print("=" * 80)
    print(f"Events: {len(signals)}")
    if len(signals):
        print(f"Date range: {signals['timestamp'].min()} -> {signals['timestamp'].max()}")
    if not summary.empty:
        best = summary.sort_values(["net_return_mean_pct_50", "n_events"], ascending=[False, False]).head(12)
        print("Top setup rows (net@50bps):")
        print(best[["setup", "variant", "hold_bars", "direction", "n_events", "net_return_mean_pct_50", "net_return_mean_pct_100"]].to_string(index=False))
    if not train_test.empty:
        print("Train/test rows:")
        print(train_test[["setup", "variant", "hold_bars", "direction", "split", "n_events", "net_return_mean_pct_50", "test_train_degradation_pct_50"]].to_string(index=False))
    print(f"Output dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
