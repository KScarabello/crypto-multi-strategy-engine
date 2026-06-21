"""Research-only delayed-entry study after explosion events.

This module builds on the first-stage explosion event study and asks a narrower
question: if we wait one 4h candle after an explosion, do the resulting
follow-through, pullback, or no-follow-through conditions create tradable
continuation / reversal / exclusion signals?

Outputs are written under reports/explosion_delayed_entry_study/:
- delayed_entry_trades.csv
- strategy_summary.csv
- strategy_by_btc_regime.csv
- strategy_by_liquidity_bucket.csv
- pullback_bucket_summary.csv
- train_test_summary.csv
- top_delayed_entry_setups.json
- summary.md

Safety:
- Research only.
- No live trading, no order execution, no scheduler integration.
- No modifications to production configs or production strategy logic.
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

from research.explosion_reversal_event_study import (
    build_event_time_features,
    detect_explosion_events,
    join_btc_context,
    load_ohlcv_data,
)


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/explosion_delayed_entry_study")


@dataclass(frozen=True)
class DelayedEntryStudyConfig:
    ret_4h_threshold_pct: float = 8.0
    ret_zscore_threshold: float = 2.5
    min_dollar_volume: float = 75_000.0
    cooldown_bars: int = 3
    rolling_window_bars: int = 42
    vol_window_bars: int = 42
    btc_symbol: str = "BTC/USD"
    pullback_trigger_pct: float = 2.0
    hold_bars: tuple[int, ...] = (1, 2, 6)
    cost_bps_list: tuple[int, ...] = (0, 10, 25, 50, 100)
    label_cost_bps: int = 50
    train_fraction: float = 0.7
    min_bucket_events: int = 20
    bucket_quantiles: int = 4


def _symbol_frame_index(full_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: sub.sort_values("timestamp").reset_index(drop=True)
        for symbol, sub in full_df.groupby("symbol", sort=False)
    }


def _safe_rank_bucket(value: float | int | np.floating, split_points: tuple[float, float]) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    low, high = split_points
    if value <= low:
        return "SMALL"
    if value <= high:
        return "MID"
    return "LARGE"


def _pullback_bucket(depth_pct: float | int | np.floating) -> str:
    if pd.isna(depth_pct):
        return "UNKNOWN"
    if depth_pct >= 0.0:
        return "NO_PULLBACK"
    if depth_pct > -2.0:
        return "SHALLOW"
    if depth_pct > -5.0:
        return "MODERATE"
    return "DEEP"


def _trade_return(entry_price: float, exit_price: float, direction: str) -> float:
    if direction == "SHORT":
        return (entry_price - exit_price) / entry_price * 100.0
    return (exit_price / entry_price - 1.0) * 100.0


def _trade_excursions(
    future_highs: pd.Series,
    future_lows: pd.Series,
    entry_price: float,
    direction: str,
) -> tuple[float, float]:
    if direction == "SHORT":
        mfe = (entry_price - float(future_lows.min())) / entry_price * 100.0
        mae = (entry_price - float(future_highs.max())) / entry_price * 100.0
    else:
        mfe = (float(future_highs.max()) / entry_price - 1.0) * 100.0
        mae = (float(future_lows.min()) / entry_price - 1.0) * 100.0
    return mfe, mae


def _lookup_row(frame: pd.DataFrame, symbol: str, timestamp: pd.Timestamp) -> pd.Series | None:
    sub = frame[(frame["symbol"] == symbol) & (frame["timestamp"] == timestamp)]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_delayed_entry_signals(full_df: pd.DataFrame, cfg: DelayedEntryStudyConfig) -> pd.DataFrame:
    """Detect explosions and classify the first post-event 4h candle.

    All decision columns are based only on information available by the close of
    the first candle after the explosion.
    """
    feats = build_event_time_features(full_df, cfg)
    events = detect_explosion_events(feats, cfg)

    if events.empty:
        return events.copy()

    symbol_frames = _symbol_frame_index(full_df)

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        symbol = str(event["symbol"])
        event_ts = pd.Timestamp(event["timestamp"])
        symbol_df = symbol_frames.get(symbol)
        if symbol_df is None:
            continue

        match = symbol_df.index[symbol_df["timestamp"] == event_ts]
        if len(match) == 0:
            continue
        event_pos = int(match[0])
        decision_pos = event_pos + 1
        if decision_pos >= len(symbol_df):
            continue

        decision_row = symbol_df.iloc[decision_pos]
        entry_ts = pd.Timestamp(decision_row["timestamp"])

        event_close = float(event["close"])
        event_high = float(event["high"])
        decision_close = float(decision_row["close"])
        decision_high = float(decision_row["high"])
        decision_low = float(decision_row["low"])

        pullback_depth_pct = (decision_low / event_close - 1.0) * 100.0
        follow_through = decision_close > event_high
        no_follow_through = not follow_through
        deep_pullback = pullback_depth_pct <= -cfg.pullback_trigger_pct
        no_deep_pullback = not deep_pullback

        entry_feats = _lookup_row(feats, symbol, entry_ts)
        liquidity_rank = float(entry_feats["liquidity_rank_universe"]) if entry_feats is not None else np.nan
        entry_dollar_volume = float(entry_feats["dollar_volume"]) if entry_feats is not None else np.nan

        rows.append(
            {
                **event.to_dict(),
                "decision_timestamp": entry_ts,
                "entry_timestamp": entry_ts,
                "wait_bars": 1,
                "signal_delay_bars": 1,
                "entry_close": decision_close,
                "entry_high": decision_high,
                "entry_low": decision_low,
                "follow_through_1c": bool(follow_through),
                "no_follow_through_1c": bool(no_follow_through),
                "deep_pullback_1c": bool(deep_pullback),
                "no_deep_pullback_1c": bool(no_deep_pullback),
                "pullback_depth_1c_pct": pullback_depth_pct,
                "pullback_bucket_1c": _pullback_bucket(pullback_depth_pct),
                "entry_liquidity_rank_universe": liquidity_rank,
                "entry_dollar_volume": entry_dollar_volume,
                "entry_liquidity_bucket": _safe_rank_bucket(liquidity_rank, (1 / 3, 2 / 3)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    btc_context = join_btc_context(out[["entry_timestamp"]].rename(columns={"entry_timestamp": "timestamp"}), full_df, cfg)
    btc_context = btc_context.rename(columns={"timestamp": "entry_timestamp"})

    btc_cols = [
        "entry_timestamp",
        "btc_ret_1h_pct",
        "btc_ret_4h_pct",
        "btc_ret_24h_pct",
        "btc_rolling_vol_pct",
        "btc_above_ma",
        "btc_trend_bucket",
        "btc_drawdown_pct",
    ]
    out = out.merge(btc_context[btc_cols], on="entry_timestamp", how="left")
    return out


def _trade_row(
    signal_row: pd.Series,
    symbol_df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
    direction: str,
    setup: str,
) -> dict[str, object] | None:
    if hold_bars <= 0 or entry_pos + hold_bars >= len(symbol_df):
        return None

    entry_row = symbol_df.iloc[entry_pos]
    exit_row = symbol_df.iloc[entry_pos + hold_bars]
    future = symbol_df.iloc[entry_pos + 1 : entry_pos + hold_bars + 1]
    if future.empty:
        return None

    entry_price = float(entry_row["close"])
    exit_price = float(exit_row["close"])
    gross_return = _trade_return(entry_price, exit_price, direction)
    mfe, mae = _trade_excursions(future["high"].astype(float), future["low"].astype(float), entry_price, direction)

    return {
        "event_id": signal_row["event_id"],
        "symbol": signal_row["symbol"],
        "event_timestamp": signal_row["timestamp"],
        "decision_timestamp": signal_row["decision_timestamp"],
        "entry_timestamp": entry_row["timestamp"],
        "setup": setup,
        "direction": direction,
        "hold_bars": int(hold_bars),
        "event_close": float(signal_row["close"]),
        "event_high": float(signal_row["high"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return_pct": gross_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "follow_through_1c": bool(signal_row["follow_through_1c"]),
        "no_follow_through_1c": bool(signal_row["no_follow_through_1c"]),
        "deep_pullback_1c": bool(signal_row["deep_pullback_1c"]),
        "no_deep_pullback_1c": bool(signal_row["no_deep_pullback_1c"]),
        "pullback_depth_1c_pct": float(signal_row["pullback_depth_1c_pct"]),
        "pullback_bucket_1c": signal_row["pullback_bucket_1c"],
        "btc_trend_bucket": signal_row.get("btc_trend_bucket", "UNKNOWN"),
        "btc_ret_4h_pct": signal_row.get("btc_ret_4h_pct", np.nan),
        "btc_ret_24h_pct": signal_row.get("btc_ret_24h_pct", np.nan),
        "btc_rolling_vol_pct": signal_row.get("btc_rolling_vol_pct", np.nan),
        "btc_above_ma": signal_row.get("btc_above_ma", np.nan),
        "entry_liquidity_rank_universe": float(signal_row.get("entry_liquidity_rank_universe", np.nan)),
        "entry_liquidity_bucket": signal_row.get("entry_liquidity_bucket", "UNKNOWN"),
        "entry_dollar_volume": float(signal_row.get("entry_dollar_volume", np.nan)),
        "signal_delay_bars": int(signal_row["wait_bars"]),
    }


def build_delayed_entry_trades(signals: pd.DataFrame, full_df: pd.DataFrame, cfg: DelayedEntryStudyConfig) -> pd.DataFrame:
    """Build long-form trade rows for each setup and requested hold period."""
    if signals.empty:
        return signals.copy()

    symbol_frames = _symbol_frame_index(full_df)
    rows: list[dict[str, object]] = []

    for _, signal in signals.iterrows():
        symbol = str(signal["symbol"])
        symbol_df = symbol_frames.get(symbol)
        if symbol_df is None:
            continue

        match = symbol_df.index[symbol_df["timestamp"] == signal["timestamp"]]
        if len(match) == 0:
            continue
        event_pos = int(match[0])
        entry_pos = event_pos + 1
        if entry_pos >= len(symbol_df):
            continue

        setup_specs = [
            ("BUY_AT_EXPLOSION", event_pos, "LONG", True),
            ("BUY_AFTER_FOLLOW_THROUGH", entry_pos, "LONG", bool(signal["follow_through_1c"])),
            ("BUY_IF_NO_DEEP_PULLBACK", entry_pos, "LONG", bool(signal["no_deep_pullback_1c"])),
            ("SHORT_AFTER_NO_FOLLOW_THROUGH", entry_pos, "SHORT", bool(signal["no_follow_through_1c"])),
            ("AVOID_NO_FOLLOW_THROUGH", entry_pos, "LONG", bool(signal["follow_through_1c"])),
        ]

        for setup, setup_entry_pos, direction, include in setup_specs:
            if not include:
                continue
            for hold_bars in cfg.hold_bars:
                trade_row = _trade_row(
                    signal_row=signal,
                    symbol_df=symbol_df,
                    entry_pos=setup_entry_pos,
                    hold_bars=hold_bars,
                    direction=direction,
                    setup=setup,
                )
                if trade_row is not None:
                    rows.append(trade_row)

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    trades = trades.sort_values(["setup", "hold_bars", "event_timestamp", "entry_timestamp"], ascending=[True, True, True, True]).reset_index(drop=True)

    for bps in cfg.cost_bps_list:
        trades[f"net_return_pct_{bps}"] = trades["gross_return_pct"] - (bps / 100.0)
        trades[f"win_{bps}"] = trades[f"net_return_pct_{bps}"] > 0.0

    return trades


def _grouped_trade_summary(
    trades: pd.DataFrame,
    group_cols: list[str],
    cfg: DelayedEntryStudyConfig,
    total_event_count: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_values, group in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row.update(
            {
                "n_trades": int(len(group)),
                "n_events": int(group["event_id"].nunique()),
                "selection_rate": float(group["event_id"].nunique() / total_event_count) if total_event_count else np.nan,
                "start_ts": str(group["entry_timestamp"].min()),
                "end_ts": str(group["entry_timestamp"].max()),
                "gross_return_mean_pct": float(group["gross_return_pct"].mean()),
                "gross_return_median_pct": float(group["gross_return_pct"].median()),
                "gross_win_rate": float((group["gross_return_pct"] > 0).mean()),
                "mfe_mean_pct": float(group["mfe_pct"].mean()),
                "mae_mean_pct": float(group["mae_pct"].mean()),
            }
        )
        for bps in cfg.cost_bps_list:
            net_col = f"net_return_pct_{bps}"
            row[f"net_return_mean_pct_{bps}"] = float(group[net_col].mean())
            row[f"net_return_median_pct_{bps}"] = float(group[net_col].median())
            row[f"net_win_rate_{bps}"] = float(group[net_col].gt(0).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_strategy_summary(trades: pd.DataFrame, cfg: DelayedEntryStudyConfig, total_event_count: int) -> pd.DataFrame:
    return _grouped_trade_summary(trades, ["setup", "hold_bars"], cfg, total_event_count)


def build_strategy_by_btc_regime(trades: pd.DataFrame, cfg: DelayedEntryStudyConfig, total_event_count: int) -> pd.DataFrame:
    return _grouped_trade_summary(trades, ["setup", "hold_bars", "btc_trend_bucket"], cfg, total_event_count)


def build_strategy_by_liquidity_bucket(
    trades: pd.DataFrame,
    cfg: DelayedEntryStudyConfig,
    total_event_count: int,
) -> pd.DataFrame:
    return _grouped_trade_summary(trades, ["setup", "hold_bars", "entry_liquidity_bucket"], cfg, total_event_count)


def build_pullback_bucket_summary(signals: pd.DataFrame, cfg: DelayedEntryStudyConfig) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for bucket, group in signals.groupby("pullback_bucket_1c", dropna=False, sort=False):
        rows.append(
            {
                "pullback_bucket_1c": str(bucket),
                "n_events": int(len(group)),
                "follow_through_rate": float(group["follow_through_1c"].mean()),
                "no_follow_through_rate": float(group["no_follow_through_1c"].mean()),
                "deep_pullback_rate": float(group["deep_pullback_1c"].mean()),
                "avg_pullback_depth_1c_pct": float(group["pullback_depth_1c_pct"].mean()),
                "small_sample_warning": bool(len(group) < cfg.min_bucket_events),
            }
        )

    return pd.DataFrame(rows).sort_values(["n_events", "pullback_bucket_1c"], ascending=[False, True]).reset_index(drop=True)


def build_train_test_summary(trades: pd.DataFrame, cfg: DelayedEntryStudyConfig, total_event_count: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    split_ts = trades["event_timestamp"].quantile(cfg.train_fraction)
    out = trades.copy()
    out["split"] = np.where(out["event_timestamp"] <= split_ts, "train", "test")

    rows: list[dict[str, object]] = []
    for (setup, hold_bars, split), group in out.groupby(["setup", "hold_bars", "split"], sort=False):
        row = {
            "setup": setup,
            "hold_bars": int(hold_bars),
            "split": split,
            "n_trades": int(len(group)),
            "n_events": int(group["event_id"].nunique()),
            "selection_rate": float(group["event_id"].nunique() / total_event_count) if total_event_count else np.nan,
            "start_ts": str(group["event_timestamp"].min()),
            "end_ts": str(group["event_timestamp"].max()),
            "gross_return_mean_pct": float(group["gross_return_pct"].mean()),
            "gross_return_median_pct": float(group["gross_return_pct"].median()),
            "gross_win_rate": float((group["gross_return_pct"] > 0).mean()),
            "mfe_mean_pct": float(group["mfe_pct"].mean()),
            "mae_mean_pct": float(group["mae_pct"].mean()),
        }
        for bps in cfg.cost_bps_list:
            net_col = f"net_return_pct_{bps}"
            row[f"net_return_mean_pct_{bps}"] = float(group[net_col].mean())
            row[f"net_return_median_pct_{bps}"] = float(group[net_col].median())
            row[f"net_win_rate_{bps}"] = float(group[net_col].gt(0).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["setup", "hold_bars", "split"]).reset_index(drop=True)


def extract_top_setups(strategy_summary: pd.DataFrame, cfg: DelayedEntryStudyConfig, top_n: int = 10) -> list[dict[str, object]]:
    if strategy_summary.empty:
        return []

    rank_col = f"net_return_mean_pct_{cfg.label_cost_bps}"
    ranked = strategy_summary.sort_values([rank_col, "gross_win_rate", "n_trades"], ascending=[False, False, False]).head(top_n)

    rows: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "setup": row["setup"],
                "hold_bars": int(row["hold_bars"]),
                "n_trades": int(row["n_trades"]),
                "selection_rate": float(row["selection_rate"]),
                "gross_return_mean_pct": float(row["gross_return_mean_pct"]),
                f"net_return_mean_pct_{cfg.label_cost_bps}": float(row[rank_col]),
                f"net_win_rate_{cfg.label_cost_bps}": float(row[f"net_win_rate_{cfg.label_cost_bps}"]),
                "gross_win_rate": float(row["gross_win_rate"]),
            }
        )
    return rows


def write_summary_markdown(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    btc_summary: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    pullback_summary: pd.DataFrame,
    train_test: pd.DataFrame,
    top_setups: list[dict[str, object]],
    cfg: DelayedEntryStudyConfig,
    output_path: Path,
) -> None:
    lines = [
        "# Explosion Delayed-Entry Study (Research-Only)",
        "",
        "## Scope",
        "- Research only.",
        "- No live trading, order execution, scheduler, launchd, broker, or production config changes.",
        "- Decisions are made using the first 4h candle after an explosion only.",
        "- All trade rows are delayed-entry diagnostics; they are not production signals.",
        "",
        "## Setups Tested",
        "- BUY_AT_EXPLOSION baseline.",
        "- BUY_AFTER_FOLLOW_THROUGH.",
        "- BUY_IF_NO_DEEP_PULLBACK.",
        "- SHORT_AFTER_NO_FOLLOW_THROUGH, research-only.",
        "- AVOID_NO_FOLLOW_THROUGH as a long-only exclusion filter.",
        "- Pullback continuation / reversal buckets.",
        "",
        "## Cost / Hold Grid",
        f"- Hold periods: {', '.join(str(x) for x in cfg.hold_bars)} candles.",
        f"- Cost assumptions (bps): {', '.join(str(x) for x in cfg.cost_bps_list)}.",
        f"- Canonical ranking cost: {cfg.label_cost_bps} bps.",
        "",
        "## Event Coverage",
        f"- Explosion events: {len(signals)}",
        f"- Tradable delayed rows: {len(trades)}",
        f"- Date range: {signals['timestamp'].min() if not signals.empty else 'N/A'} -> {signals['timestamp'].max() if not signals.empty else 'N/A'}",
        "",
        "## Signal Definitions",
        f"- Follow-through: first post-explosion close > explosion high.",
        f"- Deep pullback: first post-explosion low <= {cfg.pullback_trigger_pct:.2f}% below the explosion close.",
        "- Pullback buckets are based on the first post-explosion low relative to the explosion close.",
        "",
        "## Strategy Summary",
    ]

    if strategy_summary.empty:
        lines.append("- No strategy rows were produced.")
    else:
        for _, row in strategy_summary.head(12).iterrows():
            lines.append(
                f"- {row['setup']} / {int(row['hold_bars'])}c: n={int(row['n_trades'])}, "
                f"gross={row['gross_return_mean_pct']:+.4f}%, net@{cfg.label_cost_bps}bps={row[f'net_return_mean_pct_{cfg.label_cost_bps}']:+.4f}%, "
                f"win@{cfg.label_cost_bps}bps={row[f'net_win_rate_{cfg.label_cost_bps}']:.3f}"
            )

    lines += [
        "",
        "## BTC Regime Buckets",
    ]
    if btc_summary.empty:
        lines.append("- No BTC regime summary available.")
    else:
        for _, row in btc_summary.head(12).iterrows():
            lines.append(
                f"- {row['setup']} / {int(row['hold_bars'])}c / {row['btc_trend_bucket']}: "
                f"n={int(row['n_trades'])}, net@{cfg.label_cost_bps}bps={row[f'net_return_mean_pct_{cfg.label_cost_bps}']:+.4f}%"
            )

    lines += [
        "",
        "## Liquidity Buckets",
    ]
    if liquidity_summary.empty:
        lines.append("- No liquidity summary available.")
    else:
        for _, row in liquidity_summary.head(12).iterrows():
            lines.append(
                f"- {row['setup']} / {int(row['hold_bars'])}c / {row['entry_liquidity_bucket']}: "
                f"n={int(row['n_trades'])}, net@{cfg.label_cost_bps}bps={row[f'net_return_mean_pct_{cfg.label_cost_bps}']:+.4f}%"
            )

    lines += [
        "",
        "## Pullback Buckets",
    ]
    if pullback_summary.empty:
        lines.append("- No pullback summary available.")
    else:
        for _, row in pullback_summary.iterrows():
            lines.append(
                f"- {row['pullback_bucket_1c']}: n={int(row['n_events'])}, follow-through={row['follow_through_rate']:.3f}, "
                f"reversal={row['no_follow_through_rate']:.3f}, avg_pullback={row['avg_pullback_depth_1c_pct']:+.4f}%"
            )

    lines += [
        "",
        "## Train / Test Robustness",
    ]
    if train_test.empty:
        lines.append("- Train/test summary unavailable.")
    else:
        for _, row in train_test.iterrows():
            lines.append(
                f"- {row['setup']} / {int(row['hold_bars'])}c / {row['split']}: n={int(row['n_trades'])}, "
                f"gross={row['gross_return_mean_pct']:+.4f}%, net@{cfg.label_cost_bps}bps={row[f'net_return_mean_pct_{cfg.label_cost_bps}']:+.4f}%"
            )

    lines += [
        "",
        "## Top Delayed-Entry Setups",
    ]
    if not top_setups:
        lines.append("- No ranked setups met the minimum data requirements.")
    else:
        for row in top_setups[:8]:
            lines.append(
                f"- {row['setup']} / {row['hold_bars']}c: n={row['n_trades']}, "
                f"gross={row['gross_return_mean_pct']:+.4f}%, net@{cfg.label_cost_bps}bps={row[f'net_return_mean_pct_{cfg.label_cost_bps}']:+.4f}%, "
                f"win@{cfg.label_cost_bps}bps={row[f'net_win_rate_{cfg.label_cost_bps}']:.3f}"
            )

    lines += [
        "",
        "## Safety Confirmation",
        "- No live trading behavior changed.",
        "- No broker/exchange execution logic changed.",
        "- No scheduled jobs, launchd files, or production config changed.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def run_delayed_entry_study(
    cfg: DelayedEntryStudyConfig,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    full_df = load_ohlcv_data(data_dir=data_dir)
    signals = build_delayed_entry_signals(full_df, cfg)
    trades = build_delayed_entry_trades(signals, full_df, cfg)

    total_event_count = int(signals["event_id"].nunique()) if not signals.empty else 0
    strategy_summary = build_strategy_summary(trades, cfg, total_event_count)
    btc_summary = build_strategy_by_btc_regime(trades, cfg, total_event_count)
    liquidity_summary = build_strategy_by_liquidity_bucket(trades, cfg, total_event_count)
    pullback_summary = build_pullback_bucket_summary(signals, cfg)
    train_test = build_train_test_summary(trades, cfg, total_event_count)
    top_setups = extract_top_setups(strategy_summary, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_dir / "delayed_entry_trades.csv", index=False)
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    btc_summary.to_csv(output_dir / "strategy_by_btc_regime.csv", index=False)
    liquidity_summary.to_csv(output_dir / "strategy_by_liquidity_bucket.csv", index=False)
    pullback_summary.to_csv(output_dir / "pullback_bucket_summary.csv", index=False)
    train_test.to_csv(output_dir / "train_test_summary.csv", index=False)
    (output_dir / "top_delayed_entry_setups.json").write_text(json.dumps(top_setups, indent=2) + "\n")

    write_summary_markdown(
        signals=signals,
        trades=trades,
        strategy_summary=strategy_summary,
        btc_summary=btc_summary,
        liquidity_summary=liquidity_summary,
        pullback_summary=pullback_summary,
        train_test=train_test,
        top_setups=top_setups,
        cfg=cfg,
        output_path=output_dir / "summary.md",
    )

    return {
        "signals": signals,
        "trades": trades,
        "strategy_summary": strategy_summary,
        "btc_summary": btc_summary,
        "liquidity_summary": liquidity_summary,
        "pullback_summary": pullback_summary,
        "train_test": train_test,
        "top_setups": top_setups,
        "output_dir": output_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explosion delayed-entry study (research-only)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ret-4h-threshold-pct", type=float, default=8.0)
    parser.add_argument("--ret-zscore-threshold", type=float, default=2.5)
    parser.add_argument("--min-dollar-volume", type=float, default=75_000.0)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--rolling-window-bars", type=int, default=42)
    parser.add_argument("--vol-window-bars", type=int, default=42)
    parser.add_argument("--pullback-trigger-pct", type=float, default=2.0)
    parser.add_argument("--label-cost-bps", type=int, default=50)
    parser.add_argument("--min-bucket-events", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.7)
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
        pullback_trigger_pct=args.pullback_trigger_pct,
        label_cost_bps=args.label_cost_bps,
        min_bucket_events=args.min_bucket_events,
        train_fraction=args.train_fraction,
    )

    result = run_delayed_entry_study(cfg=cfg, data_dir=args.data_dir, output_dir=args.output_dir)
    signals = result["signals"]
    strategy_summary = result["strategy_summary"]
    train_test = result["train_test"]

    print("=" * 80)
    print("EXPLOSION DELAYED-ENTRY STUDY (RESEARCH-ONLY)")
    print("=" * 80)
    print(f"Events: {len(signals)}")
    if len(signals):
        print(f"Date range: {signals['timestamp'].min()} -> {signals['timestamp'].max()}")
        print(f"Follow-through rate: {signals['follow_through_1c'].mean():.3f}")
        print(f"No-follow-through rate: {signals['no_follow_through_1c'].mean():.3f}")
    if not strategy_summary.empty:
        rank_col = f"net_return_mean_pct_{cfg.label_cost_bps}"
        best = strategy_summary.sort_values([rank_col, "gross_win_rate"], ascending=[False, False]).head(8)
        print("Top setup rows:")
        print(best[["setup", "hold_bars", "n_trades", "gross_return_mean_pct", rank_col]].to_string(index=False))
    if not train_test.empty:
        print("Train/test summary:")
        print(train_test[["setup", "hold_bars", "split", "n_trades", "gross_return_mean_pct", f"net_return_mean_pct_{cfg.label_cost_bps}"]].to_string(index=False))
    print(f"Output dir: {result['output_dir']}")


if __name__ == "__main__":
    main()