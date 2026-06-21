"""Research-only explosion reversal/continuation event study.

This module scans local 4h OHLCV symbol files, detects configurable
"explosion" events, computes event-time features (known at detection time),
then computes post-event outcomes and labels (diagnostic only).

Outputs are written under reports/explosion_reversal_event_study/:
- explosion_events.csv
- bucket_summary.csv
- train_test_summary.csv
- top_reversal_traits.json
- top_continuation_traits.json
- summary.md

Safety:
- Research only.
- No live trading, no order execution, no scheduler integration.
- No modifications to production configs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/explosion_reversal_event_study")


@dataclass(frozen=True)
class EventStudyConfig:
    ret_4h_threshold_pct: float = 8.0
    ret_zscore_threshold: float = 2.5
    min_dollar_volume: float = 75_000.0
    cooldown_bars: int = 3
    rolling_window_bars: int = 42
    vol_window_bars: int = 42
    btc_symbol: str = "BTC/USD"
    cost_bps_list: tuple[int, ...] = (0, 50, 100, 150, 200)
    label_cost_bps: int = 50
    continuation_return_threshold_pct: float = 1.0
    continuation_mfe_threshold_pct: float = 1.5
    reversal_return_threshold_pct: float = -1.0
    reversal_mae_threshold_pct: float = -3.0
    pullback_trigger_pct: float = 2.0
    train_fraction: float = 0.7
    min_bucket_events: int = 20
    bucket_quantiles: int = 4


def _symbol_from_file(path: Path) -> str:
    stem = path.stem  # e.g. btc-usd_4h
    base = stem.replace("_4h", "")
    left, right = base.split("-", maxsplit=1)
    return f"{left.upper()}/{right.upper()}"


def _rolling_last_rank_pct(series: pd.Series, window: int) -> pd.Series:
    def _rank_last(values: Iterable[float]) -> float:
        s = pd.Series(values)
        return float(s.rank(pct=True).iloc[-1])

    min_periods = max(5, window // 3)
    return series.rolling(window=window, min_periods=min_periods).apply(_rank_last, raw=False)


def load_ohlcv_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load all *-usd_4h.csv OHLCV files into one DataFrame.

    Required columns per file: timestamp, open, high, low, close, volume.
    """
    files = sorted(data_dir.glob("*-usd_4h.csv"))
    rows: list[pd.DataFrame] = []
    required = {"timestamp", "open", "high", "low", "close", "volume"}

    for path in files:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            continue
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
        df["symbol"] = _symbol_from_file(path)
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid OHLCV files found under data directory")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def build_event_time_features(df: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    """Build features that are known at event detection time (no look-ahead)."""
    out = df.copy()

    g = out.groupby("symbol", group_keys=False)
    out["ret_4h_pct"] = g["close"].pct_change() * 100.0
    out["ret_24h_pct"] = g["close"].pct_change(6) * 100.0

    # Not available from 4h candles at detection-time granularity.
    out["ret_15m_pct"] = np.nan
    out["ret_1h_pct"] = np.nan

    out["range_pct"] = (out["high"] - out["low"]) / out["close"] * 100.0
    out["rolling_vol_4h_pct"] = g["ret_4h_pct"].transform(
        lambda s: s.rolling(cfg.vol_window_bars, min_periods=max(5, cfg.vol_window_bars // 3)).std()
    )
    out["ret_zscore"] = out["ret_4h_pct"] / out["rolling_vol_4h_pct"].replace(0.0, np.nan)

    vol_med = g["volume"].transform(
        lambda s: s.rolling(cfg.rolling_window_bars, min_periods=max(5, cfg.rolling_window_bars // 3)).median()
    )
    out["volume_spike_ratio"] = out["volume"] / vol_med.replace(0.0, np.nan)

    out["dollar_volume"] = out["close"] * out["volume"]
    dv_med = g["dollar_volume"].transform(
        lambda s: s.rolling(cfg.rolling_window_bars, min_periods=max(5, cfg.rolling_window_bars // 3)).median()
    )
    out["dollar_volume_spike_ratio"] = out["dollar_volume"] / dv_med.replace(0.0, np.nan)

    out["move_pct_rank_symbol"] = g["ret_4h_pct"].transform(
        lambda s: _rolling_last_rank_pct(s, cfg.rolling_window_bars)
    )
    out["volume_pct_rank_symbol"] = g["volume"].transform(
        lambda s: _rolling_last_rank_pct(s, cfg.rolling_window_bars)
    )

    range_med = g["range_pct"].transform(
        lambda s: s.rolling(cfg.rolling_window_bars, min_periods=max(5, cfg.rolling_window_bars // 3)).median()
    )
    out["volatility_expansion"] = out["range_pct"] / range_med.replace(0.0, np.nan)

    # Universe-relative ranks at each timestamp.
    out["move_pct_rank_universe"] = out.groupby("timestamp")["ret_4h_pct"].rank(pct=True)
    out["liquidity_rank_universe"] = out.groupby("timestamp")["dollar_volume"].rank(pct=True)

    return out


def apply_cooldown(events: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """Keep first event per symbol within cooldown window."""
    if events.empty:
        return events.copy()

    kept: list[pd.Series] = []
    cooldown = pd.Timedelta(hours=4 * cooldown_bars)

    for _, sub in events.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        last_ts: pd.Timestamp | None = None
        for _, row in sub.iterrows():
            ts = row["timestamp"]
            if last_ts is None or (ts - last_ts) >= cooldown:
                kept.append(row)
                last_ts = ts

    if not kept:
        return events.iloc[0:0].copy()
    return pd.DataFrame(kept).reset_index(drop=True)


def detect_explosion_events(df: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    """Detect explosion events using configurable return / z-score rules."""
    cond = (
        (df["ret_4h_pct"] >= cfg.ret_4h_threshold_pct)
        | (df["ret_zscore"] >= cfg.ret_zscore_threshold)
    )

    if "dollar_volume" in df.columns and cfg.min_dollar_volume > 0:
        cond = cond & (df["dollar_volume"] >= cfg.min_dollar_volume)

    candidates = df[cond].copy()
    candidates = apply_cooldown(candidates, cooldown_bars=cfg.cooldown_bars)
    candidates["event_id"] = (
        candidates["symbol"].astype(str)
        + "_"
        + candidates["timestamp"].dt.strftime("%Y%m%d%H%M%S")
    )
    return candidates.reset_index(drop=True)


def _forward_bar_metrics(symbol_df: pd.DataFrame, pos: int, bars: int) -> dict[str, float]:
    """Compute forward return and excursion metrics for a fixed horizon."""
    entry_close = float(symbol_df.iloc[pos]["close"])
    entry_high = float(symbol_df.iloc[pos]["high"])

    if pos + bars >= len(symbol_df) or bars <= 0:
        return {
            "fwd_ret_pct": np.nan,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "new_high": np.nan,
            "time_to_high_bars": np.nan,
            "worst_drawdown_pct": np.nan,
            "first_pullback_depth_pct": np.nan,
            "time_to_first_pullback_bars": np.nan,
        }

    future = symbol_df.iloc[pos + 1 : pos + bars + 1]
    if future.empty:
        return {
            "fwd_ret_pct": np.nan,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "new_high": np.nan,
            "time_to_high_bars": np.nan,
            "worst_drawdown_pct": np.nan,
            "first_pullback_depth_pct": np.nan,
            "time_to_first_pullback_bars": np.nan,
        }

    end_close = float(symbol_df.iloc[pos + bars]["close"])
    fwd_ret = (end_close / entry_close - 1.0) * 100.0

    highs = future["high"].astype(float)
    lows = future["low"].astype(float)

    mfe = (float(highs.max()) / entry_close - 1.0) * 100.0
    mae = (float(lows.min()) / entry_close - 1.0) * 100.0

    new_high = float(highs.max() > entry_high)
    time_to_high = float(highs.idxmax() - future.index.min() + 1)

    # Pullback is measured as first time low drops >= 2% below entry.
    drawdowns = (lows / entry_close - 1.0) * 100.0
    pullback_mask = drawdowns <= -2.0
    if pullback_mask.any():
        first_idx = int(np.argmax(pullback_mask.values))
        first_depth = float(drawdowns.iloc[: first_idx + 1].min())
        time_to_pullback = float(first_idx + 1)
    else:
        first_depth = np.nan
        time_to_pullback = np.nan

    return {
        "fwd_ret_pct": fwd_ret,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "new_high": new_high,
        "time_to_high_bars": time_to_high,
        "worst_drawdown_pct": mae,
        "first_pullback_depth_pct": first_depth,
        "time_to_first_pullback_bars": time_to_pullback,
    }


def compute_forward_outcomes(events: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Compute post-event outcomes from symbol-level future candles."""
    if events.empty:
        return events.copy()

    by_symbol = {
        sym: sub.sort_values("timestamp").reset_index(drop=True)
        for sym, sub in full_df.groupby("symbol", sort=False)
    }

    rows: list[dict] = []
    for _, e in events.iterrows():
        sym = e["symbol"]
        ts = e["timestamp"]
        sdf = by_symbol.get(sym)
        if sdf is None:
            continue

        idx_match = sdf.index[sdf["timestamp"] == ts]
        if len(idx_match) == 0:
            continue
        pos = int(idx_match[0])

        out = dict(e)

        # 4h and 24h are available on 4h candles; 15m/1h unavailable.
        out["future_ret_15m_pct"] = np.nan
        out["future_ret_1h_pct"] = np.nan

        m4 = _forward_bar_metrics(sdf, pos=pos, bars=1)
        m24 = _forward_bar_metrics(sdf, pos=pos, bars=6)

        out["future_ret_4h_pct"] = m4["fwd_ret_pct"]
        out["future_ret_24h_pct"] = m24["fwd_ret_pct"]

        out["max_favorable_4h_pct"] = m4["mfe_pct"]
        out["max_adverse_4h_pct"] = m4["mae_pct"]
        out["max_favorable_24h_pct"] = m24["mfe_pct"]
        out["max_adverse_24h_pct"] = m24["mae_pct"]

        out["new_high_within_4h"] = m4["new_high"]
        out["new_high_within_24h"] = m24["new_high"]
        out["time_to_post_event_high_4h_bars"] = m4["time_to_high_bars"]
        out["time_to_post_event_high_24h_bars"] = m24["time_to_high_bars"]

        out["worst_drawdown_4h_pct"] = m4["worst_drawdown_pct"]
        out["worst_drawdown_24h_pct"] = m24["worst_drawdown_pct"]

        out["first_pullback_depth_4h_pct"] = m4["first_pullback_depth_pct"]
        out["first_pullback_depth_24h_pct"] = m24["first_pullback_depth_pct"]
        out["time_to_first_pullback_4h_bars"] = m4["time_to_first_pullback_bars"]
        out["time_to_first_pullback_24h_bars"] = m24["time_to_first_pullback_bars"]

        rows.append(out)

    if not rows:
        return events.iloc[0:0].copy()
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _btc_trend_bucket(ret_24h: float) -> str:
    if pd.isna(ret_24h):
        return "UNKNOWN"
    if ret_24h >= 1.0:
        return "UP"
    if ret_24h <= -1.0:
        return "DOWN"
    return "SIDEWAYS"


def join_btc_context(events: pd.DataFrame, full_df: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    """Join BTC regime context aligned by event timestamp."""
    if events.empty:
        return events.copy()

    btc = full_df[full_df["symbol"] == cfg.btc_symbol].copy()
    if btc.empty:
        out = events.copy()
        out["btc_ret_1h_pct"] = np.nan
        out["btc_ret_4h_pct"] = np.nan
        out["btc_ret_24h_pct"] = np.nan
        out["btc_rolling_vol_pct"] = np.nan
        out["btc_above_ma"] = np.nan
        out["btc_trend_bucket"] = "UNKNOWN"
        out["btc_drawdown_pct"] = np.nan
        return out

    btc = btc.sort_values("timestamp").copy()
    btc["btc_ret_1h_pct"] = np.nan  # unavailable from 4h bars
    btc["btc_ret_4h_pct"] = btc["close"].pct_change() * 100.0
    btc["btc_ret_24h_pct"] = btc["close"].pct_change(6) * 100.0
    btc["btc_rolling_vol_pct"] = btc["btc_ret_4h_pct"].rolling(42, min_periods=10).std()
    btc["btc_ma"] = btc["close"].rolling(18, min_periods=10).mean()
    btc["btc_above_ma"] = (btc["close"] > btc["btc_ma"]).astype(float)
    btc["btc_drawdown_pct"] = (btc["close"] / btc["close"].rolling(42, min_periods=10).max() - 1.0) * 100.0
    btc["btc_trend_bucket"] = btc["btc_ret_24h_pct"].apply(_btc_trend_bucket)

    keep_cols = [
        "timestamp",
        "btc_ret_1h_pct",
        "btc_ret_4h_pct",
        "btc_ret_24h_pct",
        "btc_rolling_vol_pct",
        "btc_above_ma",
        "btc_trend_bucket",
        "btc_drawdown_pct",
    ]
    return events.merge(btc[keep_cols], on="timestamp", how="left")


def _qbucket(series: pd.Series, q: int) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() < max(8, q * 2):
        return pd.Series(["INSUFFICIENT"] * len(series), index=series.index)
    try:
        b = pd.qcut(valid, q=q, duplicates="drop")
        return b.astype(str)
    except Exception:
        return pd.Series(["INSUFFICIENT"] * len(series), index=series.index)


def label_outcomes(events: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = events.copy()

    for bps in cfg.cost_bps_list:
        out[f"net_future_ret_4h_bps_{bps}"] = out["future_ret_4h_pct"] - (bps / 100.0)
        out[f"net_future_ret_24h_bps_{bps}"] = out["future_ret_24h_pct"] - (bps / 100.0)

    net_col = f"net_future_ret_4h_bps_{cfg.label_cost_bps}"
    cont = (out[net_col] >= cfg.continuation_return_threshold_pct) | (
        out["max_favorable_4h_pct"] >= cfg.continuation_mfe_threshold_pct
    )
    rev = (out[net_col] <= cfg.reversal_return_threshold_pct) | (
        out["max_adverse_4h_pct"] <= cfg.reversal_mae_threshold_pct
    )

    out["outcome_label"] = np.where(cont, "CONTINUATION", np.where(rev, "REVERSAL", "CHOP"))

    depth = out["first_pullback_depth_24h_pct"]
    out["pullback_depth_bucket"] = np.where(
        depth.isna(),
        "NO_PULLBACK_OBS",
        np.where(depth > -2.0, "SHALLOW", np.where(depth > -5.0, "MODERATE", "DEEP")),
    )

    out["explosion_size_bucket"] = _qbucket(out["ret_4h_pct"], cfg.bucket_quantiles)
    out["volume_spike_bucket"] = _qbucket(out["volume_spike_ratio"], cfg.bucket_quantiles)
    out["liquidity_bucket"] = pd.cut(
        out["liquidity_rank_universe"],
        bins=[-0.01, 1 / 3, 2 / 3, 1.01],
        labels=["SMALL", "MID", "LARGE"],
    ).astype(str)
    out["volatility_bucket"] = _qbucket(out["rolling_vol_4h_pct"], cfg.bucket_quantiles)

    out["price_bucket"] = pd.cut(
        out["close"], bins=[-np.inf, 0.5, 2.0, 10.0, 100.0, np.inf], labels=["MICRO", "LOW", "MID", "HIGH", "VERY_HIGH"]
    ).astype(str)

    out["small_mid_large_proxy"] = out["liquidity_bucket"]

    return out


def build_bucket_summary(events: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    bucket_traits = {
        "explosion_size": "explosion_size_bucket",
        "volume_spike": "volume_spike_bucket",
        "liquidity": "liquidity_bucket",
        "btc_regime": "btc_trend_bucket",
        "follow_through": "new_high_within_4h",
        "pullback_depth": "pullback_depth_bucket",
        "volatility": "volatility_bucket",
        "price": "price_bucket",
        "size_proxy": "small_mid_large_proxy",
    }

    rows: list[dict] = []
    net_col = f"net_future_ret_4h_bps_{cfg.label_cost_bps}"

    for trait_name, col in bucket_traits.items():
        if col not in events.columns:
            continue
        sub = events[[col, "outcome_label", "future_ret_4h_pct", net_col]].copy()
        sub[col] = sub[col].astype(str)

        grouped = sub.groupby(col, dropna=False)
        for bucket, g in grouped:
            n = int(len(g))
            rows.append(
                {
                    "trait": trait_name,
                    "bucket": str(bucket),
                    "n_events": n,
                    "continuation_rate": float((g["outcome_label"] == "CONTINUATION").mean()),
                    "reversal_rate": float((g["outcome_label"] == "REVERSAL").mean()),
                    "chop_rate": float((g["outcome_label"] == "CHOP").mean()),
                    "avg_future_ret_4h_pct": float(g["future_ret_4h_pct"].mean()),
                    "avg_net_future_ret_4h_pct": float(g[net_col].mean()),
                    "small_sample_warning": bool(n < cfg.min_bucket_events),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["trait", "n_events"], ascending=[True, False]).reset_index(drop=True)


def build_train_test_summary(events: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    split_ts = events["timestamp"].quantile(cfg.train_fraction)
    net_col = f"net_future_ret_4h_bps_{cfg.label_cost_bps}"

    out = events.copy()
    out["split"] = np.where(out["timestamp"] <= split_ts, "train", "test")

    rows: list[dict] = []
    for split, g in out.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "n_events": int(len(g)),
                "start_ts": str(g["timestamp"].min()),
                "end_ts": str(g["timestamp"].max()),
                "continuation_rate": float((g["outcome_label"] == "CONTINUATION").mean()),
                "reversal_rate": float((g["outcome_label"] == "REVERSAL").mean()),
                "chop_rate": float((g["outcome_label"] == "CHOP").mean()),
                "avg_future_ret_4h_pct": float(g["future_ret_4h_pct"].mean()),
                "median_future_ret_4h_pct": float(g["future_ret_4h_pct"].median()),
                "avg_net_future_ret_4h_pct": float(g[net_col].mean()),
                "median_net_future_ret_4h_pct": float(g[net_col].median()),
            }
        )

    return pd.DataFrame(rows)


def extract_top_traits(bucket_summary: pd.DataFrame, label: str, min_events: int, top_n: int = 10) -> list[dict]:
    if bucket_summary.empty:
        return []

    eligible = bucket_summary[bucket_summary["n_events"] >= min_events].copy()
    if eligible.empty:
        return []

    if label == "REVERSAL":
        ranked = eligible.sort_values(["reversal_rate", "n_events"], ascending=[False, False])
        rate_col = "reversal_rate"
    else:
        ranked = eligible.sort_values(["continuation_rate", "n_events"], ascending=[False, False])
        rate_col = "continuation_rate"

    out: list[dict] = []
    for _, row in ranked.head(top_n).iterrows():
        out.append(
            {
                "trait": row["trait"],
                "bucket": row["bucket"],
                "n_events": int(row["n_events"]),
                "label_rate": float(row[rate_col]),
                "avg_future_ret_4h_pct": float(row["avg_future_ret_4h_pct"]),
                "avg_net_future_ret_4h_pct": float(row["avg_net_future_ret_4h_pct"]),
                "small_sample_warning": bool(row["small_sample_warning"]),
            }
        )
    return out


def write_summary_markdown(
    events: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    train_test: pd.DataFrame,
    top_reversal: list[dict],
    top_continuation: list[dict],
    cfg: EventStudyConfig,
    output_path: Path,
) -> None:
    net_col = f"net_future_ret_4h_bps_{cfg.label_cost_bps}"

    unavailable: list[str] = []
    for col in ["future_ret_15m_pct", "future_ret_1h_pct", "btc_ret_1h_pct"]:
        if col in events.columns and events[col].isna().all():
            unavailable.append(col)

    overall = {
        "n_events": int(len(events)),
        "date_min": str(events["timestamp"].min()) if not events.empty else "N/A",
        "date_max": str(events["timestamp"].max()) if not events.empty else "N/A",
        "avg_fwd_4h": float(events["future_ret_4h_pct"].mean()) if not events.empty else float("nan"),
        "median_fwd_4h": float(events["future_ret_4h_pct"].median()) if not events.empty else float("nan"),
        "avg_net_4h": float(events[net_col].mean()) if not events.empty else float("nan"),
    }

    cont_rate = float((events["outcome_label"] == "CONTINUATION").mean()) if not events.empty else float("nan")
    rev_rate = float((events["outcome_label"] == "REVERSAL").mean()) if not events.empty else float("nan")
    chop_rate = float((events["outcome_label"] == "CHOP").mean()) if not events.empty else float("nan")

    lines = [
        "# Explosion Reversal Event Study (Research-Only)",
        "",
        "## Scope",
        "- Research only.",
        "- No live trading or scheduling changes.",
        "- Event-time features use only information available at detection time.",
        "- Follow-through and pullback traits are explicitly post-event diagnostics.",
        "",
        "## Explosion Definition (Configurable)",
        f"- ret_4h_pct >= {cfg.ret_4h_threshold_pct:.2f} OR ret_zscore >= {cfg.ret_zscore_threshold:.2f}",
        f"- min dollar volume >= {cfg.min_dollar_volume:.0f}",
        f"- cooldown: {cfg.cooldown_bars} bars (4h bars)",
        "",
        "## Data Availability Notes",
        "- OHLCV resolution in local files is 4h.",
        "- 15m/1h features and outcomes are unavailable from current OHLCV resolution and are reported as NaN.",
    ]

    if unavailable:
        lines.append(f"- Unavailable from current data: {', '.join(unavailable)}")

    lines += [
        "",
        "## Overall Event Counts",
        f"- Event count: {overall['n_events']}",
        f"- Date range: {overall['date_min']} -> {overall['date_max']}",
        f"- Avg forward 4h return: {overall['avg_fwd_4h']:+.4f}%",
        f"- Median forward 4h return: {overall['median_fwd_4h']:+.4f}%",
        f"- Avg net forward 4h return ({cfg.label_cost_bps} bps): {overall['avg_net_4h']:+.4f}%",
        f"- Continuation rate: {cont_rate:.3f}",
        f"- Reversal rate: {rev_rate:.3f}",
        f"- Chop rate: {chop_rate:.3f}",
        "",
        "## Robustness Summary",
    ]

    if train_test.empty:
        lines.append("- Train/test split unavailable (insufficient events).")
    else:
        for _, row in train_test.iterrows():
            lines.append(
                f"- {row['split']}: n={int(row['n_events'])}, continuation={row['continuation_rate']:.3f}, "
                f"reversal={row['reversal_rate']:.3f}, avg_net_4h={row['avg_net_future_ret_4h_pct']:+.4f}%"
            )

    lines += [
        "",
        "## Top Reversal-Linked Trait Buckets",
    ]
    if not top_reversal:
        lines.append("- No buckets met minimum sample threshold.")
    else:
        for row in top_reversal[:8]:
            lines.append(
                f"- {row['trait']}={row['bucket']}: n={row['n_events']}, reversal_rate={row['label_rate']:.3f}, "
                f"avg_net_4h={row['avg_net_future_ret_4h_pct']:+.4f}%"
            )

    lines += [
        "",
        "## Top Continuation-Linked Trait Buckets",
    ]
    if not top_continuation:
        lines.append("- No buckets met minimum sample threshold.")
    else:
        for row in top_continuation[:8]:
            lines.append(
                f"- {row['trait']}={row['bucket']}: n={row['n_events']}, continuation_rate={row['label_rate']:.3f}, "
                f"avg_net_4h={row['avg_net_future_ret_4h_pct']:+.4f}%"
            )

    lines += [
        "",
        "## Preliminary Interpretation",
        "- Explosion outcomes depend strongly on context and bucket definitions rather than being uniformly one-sided.",
        "- Buckets with small sample warnings should be treated as unstable.",
        "- Gross-vs-net comparison can materially reduce apparent edge.",
        "",
        "## What To Test Next (Research-Only)",
        "- Add delayed-entry variants (e.g., post-pullback) with strict no-look-ahead event-time features.",
        "- Stress-test top buckets across longer holdout windows and symbol-universe subsets.",
        "- Recheck robustness as new genuine 4h events accumulate.",
        "",
        "## Safety Confirmation",
        "- No live execution behavior changed.",
        "- No scheduler/launchd changes.",
        "- No production strategy logic changed.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def run_event_study(cfg: EventStudyConfig, data_dir: Path, output_dir: Path) -> dict[str, object]:
    full_df = load_ohlcv_data(data_dir=data_dir)
    feats = build_event_time_features(full_df, cfg)
    events = detect_explosion_events(feats, cfg)
    events = compute_forward_outcomes(events, full_df)
    events = join_btc_context(events, full_df, cfg)
    events = label_outcomes(events, cfg)

    bucket_summary = build_bucket_summary(events, cfg)
    train_test = build_train_test_summary(events, cfg)

    top_reversal = extract_top_traits(bucket_summary, label="REVERSAL", min_events=cfg.min_bucket_events)
    top_cont = extract_top_traits(bucket_summary, label="CONTINUATION", min_events=cfg.min_bucket_events)

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "explosion_events.csv", index=False)
    bucket_summary.to_csv(output_dir / "bucket_summary.csv", index=False)
    train_test.to_csv(output_dir / "train_test_summary.csv", index=False)

    (output_dir / "top_reversal_traits.json").write_text(json.dumps(top_reversal, indent=2) + "\n")
    (output_dir / "top_continuation_traits.json").write_text(json.dumps(top_cont, indent=2) + "\n")

    write_summary_markdown(
        events=events,
        bucket_summary=bucket_summary,
        train_test=train_test,
        top_reversal=top_reversal,
        top_continuation=top_cont,
        cfg=cfg,
        output_path=output_dir / "summary.md",
    )

    return {
        "events": events,
        "bucket_summary": bucket_summary,
        "train_test": train_test,
        "top_reversal": top_reversal,
        "top_continuation": top_cont,
        "output_dir": output_dir,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explosion reversal/continuation event study (research-only)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ret-4h-threshold-pct", type=float, default=8.0)
    p.add_argument("--ret-zscore-threshold", type=float, default=2.5)
    p.add_argument("--min-dollar-volume", type=float, default=75_000.0)
    p.add_argument("--cooldown-bars", type=int, default=3)
    p.add_argument("--rolling-window-bars", type=int, default=42)
    p.add_argument("--vol-window-bars", type=int, default=42)
    p.add_argument("--label-cost-bps", type=int, default=50)
    p.add_argument("--min-bucket-events", type=int, default=20)
    p.add_argument("--train-fraction", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = EventStudyConfig(
        ret_4h_threshold_pct=args.ret_4h_threshold_pct,
        ret_zscore_threshold=args.ret_zscore_threshold,
        min_dollar_volume=args.min_dollar_volume,
        cooldown_bars=args.cooldown_bars,
        rolling_window_bars=args.rolling_window_bars,
        vol_window_bars=args.vol_window_bars,
        label_cost_bps=args.label_cost_bps,
        min_bucket_events=args.min_bucket_events,
        train_fraction=args.train_fraction,
    )

    result = run_event_study(cfg=cfg, data_dir=args.data_dir, output_dir=args.output_dir)
    events = result["events"]
    train_test = result["train_test"]

    print("=" * 80)
    print("EXPLOSION REVERSAL EVENT STUDY (RESEARCH-ONLY)")
    print("=" * 80)
    print(f"Events: {len(events)}")
    if len(events):
        print(f"Date range: {events['timestamp'].min()} -> {events['timestamp'].max()}")
        print(f"Continuation rate: {(events['outcome_label']=='CONTINUATION').mean():.3f}")
        print(f"Reversal rate: {(events['outcome_label']=='REVERSAL').mean():.3f}")
        net_col = f"net_future_ret_4h_bps_{cfg.label_cost_bps}"
        print(f"Avg net 4h return ({cfg.label_cost_bps} bps): {events[net_col].mean():+.4f}%")

    if not train_test.empty:
        print("Train/test summary:")
        print(train_test.to_string(index=False))

    print(f"Output dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
