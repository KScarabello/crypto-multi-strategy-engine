"""Research-only locked-candidate robustness validation for explosion delayed entry.

Frozen candidate:
- setup: BUY_AFTER_FOLLOW_THROUGH_STRICT
- follow-through rule: HIGH_BREAK_100_BPS
- observation delay: one 4h candle
- entry: after strict follow-through confirmation
- hold: 6 candles

This pass is validation/falsification only. No parameter optimization is
performed and no live-trading behavior is modified.
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

from research.explosion_delayed_entry_study import (  # noqa: E402
    DelayedEntryStudyConfig,
    build_second_pass_signals,
)
from research.explosion_reversal_event_study import load_ohlcv_data  # noqa: E402


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/explosion_delayed_entry_study/locked_candidate_validation")

COST_BPS = (10, 25, 50, 100, 150, 200)


@dataclass(frozen=True)
class LockedCandidateConfig:
    follow_rule: str = "HIGH_BREAK_100_BPS"
    hold_bars: int = 6
    train_fraction: float = 0.7
    recent_holdout_fraction: float = 0.2
    cooldown_bars_symbol: int = 6
    placebo_iterations: int = 200
    bootstrap_iterations: int = 500
    random_seed: int = 7


def _symbol_frames(full_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: sub.sort_values("timestamp").reset_index(drop=True)
        for symbol, sub in full_df.groupby("symbol", sort=False)
    }


def _candidate_signals(signals: pd.DataFrame, cfg: LockedCandidateConfig) -> pd.DataFrame:
    col = f"follow_{cfg.follow_rule}"
    out = signals[signals[col].astype(bool)].copy()
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _trade_excursions(future_highs: pd.Series, future_lows: pd.Series, entry_price: float) -> tuple[float, float]:
    mfe = (float(future_highs.max()) / entry_price - 1.0) * 100.0
    mae = (float(future_lows.min()) / entry_price - 1.0) * 100.0
    return mfe, mae


def _add_cost_columns(df: pd.DataFrame, costs: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for bps in costs:
        out[f"net_return_pct_{bps}"] = out["gross_return_pct"] - (bps / 100.0)
        out[f"win_rate_flag_{bps}"] = out[f"net_return_pct_{bps}"] > 0.0
    return out


def _build_candidate_trades(
    signals: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: LockedCandidateConfig,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    by_symbol = _symbol_frames(full_df)
    rows: list[dict[str, object]] = []

    for _, sig in signals.iterrows():
        symbol = str(sig["symbol"])
        sdf = by_symbol.get(symbol)
        if sdf is None:
            continue

        event_ts = pd.Timestamp(sig["timestamp"])
        idx = sdf.index[sdf["timestamp"] == event_ts]
        if len(idx) == 0:
            continue

        event_pos = int(idx[0])
        obs_pos = int(sig["observation_pos"])
        next_open_pos = obs_pos + 1
        close_exit_pos = obs_pos + cfg.hold_bars
        open_exit_pos = next_open_pos + cfg.hold_bars

        if close_exit_pos >= len(sdf):
            continue

        event_close = float(sig["close"])
        obs_close = float(sdf.iloc[obs_pos]["close"])
        obs_ts = pd.Timestamp(sdf.iloc[obs_pos]["timestamp"])
        obs_next_open = float(sdf.iloc[next_open_pos]["open"]) if next_open_pos < len(sdf) else np.nan

        # Assumption A: entry at observation close.
        future_close = sdf.iloc[obs_pos + 1 : close_exit_pos + 1]
        entry_close = obs_close
        exit_close = float(sdf.iloc[close_exit_pos]["close"])
        gross_close = (exit_close / entry_close - 1.0) * 100.0
        mfe_close, mae_close = _trade_excursions(future_close["high"].astype(float), future_close["low"].astype(float), entry_close)

        rows.append(
            {
                "assumption": "ENTRY_OBSERVATION_CLOSE",
                "slippage_bps": 0,
                "event_id": sig["event_id"],
                "symbol": symbol,
                "event_timestamp": event_ts,
                "observation_timestamp": obs_ts,
                "entry_timestamp": obs_ts,
                "exit_timestamp": pd.Timestamp(sdf.iloc[close_exit_pos]["timestamp"]),
                "entry_price": entry_close,
                "exit_price": exit_close,
                "gross_return_pct": gross_close,
                "mfe_pct": mfe_close,
                "mae_pct": mae_close,
                "btc_trend_bucket": sig.get("btc_trend_bucket", "UNKNOWN"),
                "btc_ret_4h_pct": sig.get("btc_ret_4h_pct", np.nan),
                "btc_ret_24h_pct": sig.get("btc_ret_24h_pct", np.nan),
                "btc_rolling_vol_pct": sig.get("btc_rolling_vol_pct", np.nan),
                "liquidity_bucket": sig.get("liquidity_bucket", "UNKNOWN"),
            }
        )

        # Assumption B: entry at next candle open.
        if next_open_pos < len(sdf) and open_exit_pos < len(sdf):
            future_open = sdf.iloc[next_open_pos + 1 : open_exit_pos + 1]
            entry_open = float(sdf.iloc[next_open_pos]["open"])
            exit_open = float(sdf.iloc[open_exit_pos]["close"])
            gross_open = (exit_open / entry_open - 1.0) * 100.0
            mfe_open, mae_open = _trade_excursions(future_open["high"].astype(float), future_open["low"].astype(float), entry_open)

            rows.append(
                {
                    "assumption": "ENTRY_NEXT_OPEN",
                    "slippage_bps": 0,
                    "event_id": sig["event_id"],
                    "symbol": symbol,
                    "event_timestamp": event_ts,
                    "observation_timestamp": obs_ts,
                    "entry_timestamp": pd.Timestamp(sdf.iloc[next_open_pos]["timestamp"]),
                    "exit_timestamp": pd.Timestamp(sdf.iloc[open_exit_pos]["timestamp"]),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "gross_return_pct": gross_open,
                    "mfe_pct": mfe_open,
                    "mae_pct": mae_open,
                    "btc_trend_bucket": sig.get("btc_trend_bucket", "UNKNOWN"),
                    "btc_ret_4h_pct": sig.get("btc_ret_4h_pct", np.nan),
                    "btc_ret_24h_pct": sig.get("btc_ret_24h_pct", np.nan),
                    "btc_rolling_vol_pct": sig.get("btc_rolling_vol_pct", np.nan),
                    "liquidity_bucket": sig.get("liquidity_bucket", "UNKNOWN"),
                }
            )

            for slip in (10, 25, 50):
                rows.append(
                    {
                        "assumption": "ENTRY_NEXT_OPEN_WITH_SLIPPAGE",
                        "slippage_bps": slip,
                        "event_id": sig["event_id"],
                        "symbol": symbol,
                        "event_timestamp": event_ts,
                        "observation_timestamp": obs_ts,
                        "entry_timestamp": pd.Timestamp(sdf.iloc[next_open_pos]["timestamp"]),
                        "exit_timestamp": pd.Timestamp(sdf.iloc[open_exit_pos]["timestamp"]),
                        "entry_price": entry_open,
                        "exit_price": exit_open,
                        "gross_return_pct": gross_open - (slip / 100.0),
                        "mfe_pct": mfe_open,
                        "mae_pct": mae_open,
                        "btc_trend_bucket": sig.get("btc_trend_bucket", "UNKNOWN"),
                        "btc_ret_4h_pct": sig.get("btc_ret_4h_pct", np.nan),
                        "btc_ret_24h_pct": sig.get("btc_ret_24h_pct", np.nan),
                        "btc_rolling_vol_pct": sig.get("btc_rolling_vol_pct", np.nan),
                        "liquidity_bucket": sig.get("liquidity_bucket", "UNKNOWN"),
                    }
                )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    return _add_cost_columns(trades.sort_values(["assumption", "slippage_bps", "entry_timestamp", "symbol"]).reset_index(drop=True), COST_BPS)


def _summary_from_trades(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for vals, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = {k: v for k, v in zip(group_cols, vals)}
        row.update(
            {
                "n_events": int(group["event_id"].nunique()),
                "n_trades": int(len(group)),
                "gross_return_mean_pct": float(group["gross_return_pct"].mean()),
                "gross_return_median_pct": float(group["gross_return_pct"].median()),
                "gross_win_rate": float((group["gross_return_pct"] > 0).mean()),
                "mfe_mean_pct": float(group["mfe_pct"].mean()),
                "mae_mean_pct": float(group["mae_pct"].mean()),
            }
        )
        for bps in COST_BPS:
            net = group[f"net_return_pct_{bps}"]
            row[f"net_return_mean_pct_{bps}"] = float(net.mean())
            row[f"net_return_median_pct_{bps}"] = float(net.median())
            row[f"net_win_rate_{bps}"] = float((net > 0).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def build_execution_assumption_summary(trades: pd.DataFrame) -> pd.DataFrame:
    out = _summary_from_trades(trades, ["assumption", "slippage_bps"])
    if out.empty:
        return out
    out["break_even_cost_bps"] = out["gross_return_mean_pct"] * 100.0
    return out.sort_values(["assumption", "slippage_bps"]).reset_index(drop=True)


def build_yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()
    base["year"] = pd.to_datetime(base["event_timestamp"], utc=True).dt.year
    return _summary_from_trades(base, ["year"]).sort_values("year").reset_index(drop=True)


def build_train_test_holdout_summary(trades: pd.DataFrame, cfg: LockedCandidateConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    ts = pd.to_datetime(base["event_timestamp"], utc=True)
    split_train = ts.quantile(cfg.train_fraction)
    split_holdout = ts.quantile(1.0 - cfg.recent_holdout_fraction)

    base["partition"] = np.where(
        ts <= split_train,
        "train",
        np.where(ts >= split_holdout, "recent_holdout", "test"),
    )

    rows = []
    for part, g in base.groupby("partition", sort=False):
        s = _summary_from_trades(g, ["partition"])
        if not s.empty:
            rows.append(s)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Simple expanding walk-forward by year.
    years = sorted(base["event_timestamp"].dt.year.unique())
    wf_rows: list[dict[str, object]] = []
    for i in range(2, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_g = base[base["event_timestamp"].dt.year.isin(train_years)]
        test_g = base[base["event_timestamp"].dt.year == test_year]
        if train_g.empty or test_g.empty:
            continue
        train_mean = float(train_g["net_return_pct_50"].mean())
        test_mean = float(test_g["net_return_pct_50"].mean())
        wf_rows.append(
            {
                "partition": "walk_forward",
                "walk_train_years": f"{min(train_years)}-{max(train_years)}",
                "walk_test_year": int(test_year),
                "n_events": int(test_g["event_id"].nunique()),
                "net_return_mean_pct_50": test_mean,
                "test_train_degradation_pct_50": test_mean - train_mean,
            }
        )

    if wf_rows:
        out = pd.concat([out, pd.DataFrame(wf_rows)], ignore_index=True)

    return out.reset_index(drop=True)


def build_symbol_concentration_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    sym = base.groupby("symbol", dropna=False).agg(
        n_events=("event_id", "nunique"),
        n_trades=("event_id", "size"),
        gross_contribution=("gross_return_pct", "sum"),
        net50_contribution=("net_return_pct_50", "sum"),
    ).reset_index().sort_values("net50_contribution", ascending=False)

    top_symbols = sym["symbol"].tolist()
    rows = []
    for k in (0, 1, 3, 5):
        excluded = set(top_symbols[:k])
        g = base[~base["symbol"].isin(excluded)]
        if g.empty:
            continue
        rows.append(
            {
                "excluded_top_k": k,
                "remaining_events": int(g["event_id"].nunique()),
                "remaining_symbols": int(g["symbol"].nunique()),
                "net_return_mean_pct_50": float(g["net_return_pct_50"].mean()),
                "net_return_mean_pct_100": float(g["net_return_pct_100"].mean()),
            }
        )

    exclude_df = pd.DataFrame(rows)
    if exclude_df.empty:
        return sym

    return pd.concat([sym, pd.DataFrame([{}]), exclude_df], ignore_index=True)


def _cap_positions_per_timestamp(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if df.empty:
        return df
    picked = []
    for _, g in df.sort_values(["entry_timestamp", "liquidity_bucket", "symbol"]).groupby("entry_timestamp", sort=False):
        picked.append(g.head(cap))
    return pd.concat(picked, ignore_index=True) if picked else df.iloc[0:0]


def _apply_symbol_cooldown(df: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    if df.empty:
        return df
    cooldown = pd.Timedelta(hours=4 * cooldown_bars)
    kept = []
    for _, g in df.sort_values(["symbol", "entry_timestamp"]).groupby("symbol", sort=False):
        last: pd.Timestamp | None = None
        for _, row in g.iterrows():
            ts = pd.Timestamp(row["entry_timestamp"])
            if last is None or (ts - last) >= cooldown:
                kept.append(row)
                last = ts
    return pd.DataFrame(kept).reset_index(drop=True) if kept else df.iloc[0:0]


def build_overlap_concurrency_summary(trades: pd.DataFrame, cfg: LockedCandidateConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    counts = base.groupby("entry_timestamp").size().rename("n_at_ts")
    multi_ts = int((counts > 1).sum())
    avg_simul = float(counts.mean())

    c1 = _cap_positions_per_timestamp(base, cap=1)
    c3 = _cap_positions_per_timestamp(base, cap=3)
    cool = _apply_symbol_cooldown(base, cooldown_bars=cfg.cooldown_bars_symbol)

    rows = [
        {
            "scenario": "all_signals",
            "timestamps_with_multi": multi_ts,
            "avg_simultaneous_per_ts": avg_simul,
            "n_events": int(base["event_id"].nunique()),
            "net_return_mean_pct_50": float(base["net_return_pct_50"].mean()),
            "net_return_mean_pct_100": float(base["net_return_pct_100"].mean()),
        },
        {
            "scenario": "max_1_position_per_timestamp",
            "timestamps_with_multi": int((_cap_positions_per_timestamp(base, cap=1).groupby("entry_timestamp").size() > 1).sum()),
            "avg_simultaneous_per_ts": float(c1.groupby("entry_timestamp").size().mean()) if not c1.empty else np.nan,
            "n_events": int(c1["event_id"].nunique()),
            "net_return_mean_pct_50": float(c1["net_return_pct_50"].mean()) if not c1.empty else np.nan,
            "net_return_mean_pct_100": float(c1["net_return_pct_100"].mean()) if not c1.empty else np.nan,
        },
        {
            "scenario": "max_3_positions_per_timestamp",
            "timestamps_with_multi": int((c3.groupby("entry_timestamp").size() > 1).sum()) if not c3.empty else 0,
            "avg_simultaneous_per_ts": float(c3.groupby("entry_timestamp").size().mean()) if not c3.empty else np.nan,
            "n_events": int(c3["event_id"].nunique()),
            "net_return_mean_pct_50": float(c3["net_return_pct_50"].mean()) if not c3.empty else np.nan,
            "net_return_mean_pct_100": float(c3["net_return_pct_100"].mean()) if not c3.empty else np.nan,
        },
        {
            "scenario": "symbol_cooldown",
            "timestamps_with_multi": int((cool.groupby("entry_timestamp").size() > 1).sum()) if not cool.empty else 0,
            "avg_simultaneous_per_ts": float(cool.groupby("entry_timestamp").size().mean()) if not cool.empty else np.nan,
            "n_events": int(cool["event_id"].nunique()),
            "net_return_mean_pct_50": float(cool["net_return_pct_50"].mean()) if not cool.empty else np.nan,
            "net_return_mean_pct_100": float(cool["net_return_pct_100"].mean()) if not cool.empty else np.nan,
        },
    ]
    return pd.DataFrame(rows)


def build_btc_regime_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    vol_med = float(base["btc_rolling_vol_pct"].median(skipna=True))
    base["btc_vol_bucket"] = np.where(base["btc_rolling_vol_pct"] >= vol_med, "BTC_HIGH_VOL", "BTC_LOW_VOL")

    regime = _summary_from_trades(base, ["btc_trend_bucket"])
    vol = _summary_from_trades(base, ["btc_vol_bucket"])
    return pd.concat([regime, vol], ignore_index=True)


def build_liquidity_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    rows = []
    rows.append(_summary_from_trades(base, ["liquidity_bucket"]))

    for thr, label in ((0.50, "MIN_LIQ_RANK_50"), (2.0 / 3.0, "MIN_LIQ_RANK_67")):
        if "liquidity_bucket" not in base.columns:
            continue
        filt = base[base["liquidity_bucket"].isin(["MID", "LARGE"]) if thr == 0.50 else base["liquidity_bucket"].eq("LARGE")]
        if filt.empty:
            continue
        s = _summary_from_trades(filt, ["liquidity_bucket"])
        s["diagnostic_filter"] = label
        rows.append(s)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_cost_sensitivity_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    row = {
        "n_events": int(base["event_id"].nunique()),
        "gross_return_mean_pct": float(base["gross_return_pct"].mean()),
        "break_even_cost_bps": float(base["gross_return_pct"].mean() * 100.0),
    }
    for bps in COST_BPS:
        net = base[f"net_return_pct_{bps}"]
        row[f"net_return_mean_pct_{bps}"] = float(net.mean())
        row[f"net_win_rate_{bps}"] = float((net > 0).mean())
    return pd.DataFrame([row])


def build_bootstrap_summary(trades: pd.DataFrame, cfg: LockedCandidateConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.random_seed)
    vals_gross = base["gross_return_pct"].to_numpy()
    vals_median = base["gross_return_pct"].to_numpy()
    vals_net50 = base["net_return_pct_50"].to_numpy()
    vals_net100 = base["net_return_pct_100"].to_numpy()

    means = []
    medians = []
    wins = []
    nets50 = []
    nets100 = []

    n = len(base)
    for _ in range(cfg.bootstrap_iterations):
        idx = rng.integers(0, n, size=n)
        g = vals_gross[idx]
        means.append(float(np.mean(g)))
        medians.append(float(np.median(vals_median[idx])))
        wins.append(float(np.mean(g > 0.0)))
        nets50.append(float(np.mean(vals_net50[idx])))
        nets100.append(float(np.mean(vals_net100[idx])))

    def _ci(arr: list[float]) -> tuple[float, float, float]:
        return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    m_mean, m_lo, m_hi = _ci(means)
    md_mean, md_lo, md_hi = _ci(medians)
    w_mean, w_lo, w_hi = _ci(wins)
    n50_mean, n50_lo, n50_hi = _ci(nets50)
    n100_mean, n100_lo, n100_hi = _ci(nets100)

    return pd.DataFrame(
        [
            {
                "metric": "mean_return_pct",
                "mean": m_mean,
                "ci_2_5": m_lo,
                "ci_97_5": m_hi,
            },
            {
                "metric": "median_return_pct",
                "mean": md_mean,
                "ci_2_5": md_lo,
                "ci_97_5": md_hi,
            },
            {
                "metric": "win_rate",
                "mean": w_mean,
                "ci_2_5": w_lo,
                "ci_97_5": w_hi,
            },
            {
                "metric": "net_return_pct_50",
                "mean": n50_mean,
                "ci_2_5": n50_lo,
                "ci_97_5": n50_hi,
            },
            {
                "metric": "net_return_pct_100",
                "mean": n100_mean,
                "ci_2_5": n100_lo,
                "ci_97_5": n100_hi,
            },
        ]
    )


def build_placebo_summary(
    candidate_trades: pd.DataFrame,
    full_df: pd.DataFrame,
    candidate_signals: pd.DataFrame,
    cfg: LockedCandidateConfig,
) -> pd.DataFrame:
    if candidate_trades.empty or candidate_signals.empty:
        return pd.DataFrame()

    base = candidate_trades[candidate_trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if base.empty:
        return pd.DataFrame()

    feats_cfg = DelayedEntryStudyConfig()
    feats = build_second_pass_signals(full_df, feats_cfg)
    # Exclude actual candidate events from placebo pool.
    used = set(zip(candidate_signals["symbol"].astype(str), candidate_signals["timestamp"].astype(str)))

    by_symbol = _symbol_frames(full_df)
    pool_by_symbol: dict[str, list[int]] = {}
    for sym, sdf in by_symbol.items():
        idxs = []
        for i in range(len(sdf) - (cfg.hold_bars + 1)):
            ts = str(pd.Timestamp(sdf.iloc[i]["timestamp"]))
            if (sym, ts) in used:
                continue
            idxs.append(i)
        pool_by_symbol[sym] = idxs

    rng = np.random.default_rng(cfg.random_seed)
    placebo_means_50 = []
    placebo_means_100 = []

    counts = candidate_signals.groupby("symbol").size().to_dict()

    for _ in range(cfg.placebo_iterations):
        sampled = []
        for sym, n in counts.items():
            pool = pool_by_symbol.get(sym, [])
            if not pool:
                continue
            choose = rng.choice(pool, size=min(len(pool), int(n)), replace=False)
            sampled.extend((sym, int(i)) for i in choose)

        rows = []
        for sym, i in sampled:
            sdf = by_symbol[sym]
            entry_pos = i + 1
            exit_pos = entry_pos + cfg.hold_bars
            if exit_pos >= len(sdf):
                continue
            entry = float(sdf.iloc[entry_pos]["open"])
            exitp = float(sdf.iloc[exit_pos]["close"])
            gross = (exitp / entry - 1.0) * 100.0
            rows.append(gross)

        if not rows:
            continue
        arr = np.asarray(rows, dtype=float)
        placebo_means_50.append(float(np.mean(arr - 0.5)))
        placebo_means_100.append(float(np.mean(arr - 1.0)))

    if not placebo_means_50:
        return pd.DataFrame()

    cand50 = float(base["net_return_pct_50"].mean())
    cand100 = float(base["net_return_pct_100"].mean())

    return pd.DataFrame(
        [
            {
                "metric": "net_return_pct_50",
                "candidate_mean": cand50,
                "placebo_mean": float(np.mean(placebo_means_50)),
                "candidate_minus_placebo": cand50 - float(np.mean(placebo_means_50)),
                "placebo_p_ge_candidate": float(np.mean(np.asarray(placebo_means_50) >= cand50)),
            },
            {
                "metric": "net_return_pct_100",
                "candidate_mean": cand100,
                "placebo_mean": float(np.mean(placebo_means_100)),
                "candidate_minus_placebo": cand100 - float(np.mean(placebo_means_100)),
                "placebo_p_ge_candidate": float(np.mean(np.asarray(placebo_means_100) >= cand100)),
            },
        ]
    )


def build_benchmark_comparison(
    candidate_trades: pd.DataFrame,
    signals_all: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: LockedCandidateConfig,
) -> pd.DataFrame:
    if candidate_trades.empty:
        return pd.DataFrame()

    candidate = candidate_trades[candidate_trades["assumption"] == "ENTRY_NEXT_OPEN"].copy()
    if candidate.empty:
        return pd.DataFrame()

    by_symbol = _symbol_frames(full_df)
    btc = by_symbol.get("BTC/USD")

    baseline_rows = []
    btc_rows = []

    for _, sig in signals_all.iterrows():
        sym = str(sig["symbol"])
        sdf = by_symbol.get(sym)
        if sdf is None:
            continue

        ts = pd.Timestamp(sig["timestamp"])
        idx = sdf.index[sdf["timestamp"] == ts]
        if len(idx) == 0:
            continue
        pos = int(idx[0])
        exit_pos = pos + cfg.hold_bars
        if exit_pos >= len(sdf):
            continue

        entry = float(sdf.iloc[pos]["close"])
        exitp = float(sdf.iloc[exit_pos]["close"])
        baseline_rows.append((exitp / entry - 1.0) * 100.0)

        if btc is not None:
            ent_ts = pd.Timestamp(sdf.iloc[pos + 1]["timestamp"]) if pos + 1 < len(sdf) else pd.NaT
            if pd.isna(ent_ts):
                continue
            btc_i = btc.index[btc["timestamp"] == ent_ts]
            if len(btc_i) == 0:
                continue
            bi = int(btc_i[0])
            bx = bi + cfg.hold_bars
            if bx >= len(btc):
                continue
            bentry = float(btc.iloc[bi]["open"])
            bexit = float(btc.iloc[bx]["close"])
            btc_rows.append((bexit / bentry - 1.0) * 100.0)

    rows = []
    cand_mean_50 = float(candidate["net_return_pct_50"].mean())
    cand_mean_100 = float(candidate["net_return_pct_100"].mean())

    rows.append(
        {
            "benchmark": "LOCKED_CANDIDATE",
            "n_events": int(candidate["event_id"].nunique()),
            "net_return_mean_pct_50": cand_mean_50,
            "net_return_mean_pct_100": cand_mean_100,
        }
    )

    if baseline_rows:
        arr = np.asarray(baseline_rows, dtype=float)
        rows.append(
            {
                "benchmark": "BUY_AT_EXPLOSION_BASELINE",
                "n_events": int(len(arr)),
                "net_return_mean_pct_50": float(np.mean(arr - 0.5)),
                "net_return_mean_pct_100": float(np.mean(arr - 1.0)),
            }
        )

    if btc_rows:
        barr = np.asarray(btc_rows, dtype=float)
        rows.append(
            {
                "benchmark": "BTC_MATCHED_WINDOW",
                "n_events": int(len(barr)),
                "net_return_mean_pct_50": float(np.mean(barr - 0.5)),
                "net_return_mean_pct_100": float(np.mean(barr - 1.0)),
            }
        )

    return pd.DataFrame(rows)


def write_summary_markdown(
    execution_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    train_test_holdout: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    btc_regime_summary: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    def _fmt(v: float) -> str:
        return f"{v:+.4f}%"

    lines = [
        "# Locked Candidate Validation (Research-Only)",
        "",
        "Frozen candidate:",
        "- BUY_AFTER_FOLLOW_THROUGH_STRICT",
        "- HIGH_BREAK_100_BPS",
        "- one-candle observation delay",
        "- hold 6 candles",
        "",
        "## 1) Does the locked candidate survive entry-at-next-open execution?",
    ]

    next_open = execution_summary[(execution_summary["assumption"] == "ENTRY_NEXT_OPEN") & (execution_summary["slippage_bps"] == 0)]
    obs_close = execution_summary[(execution_summary["assumption"] == "ENTRY_OBSERVATION_CLOSE") & (execution_summary["slippage_bps"] == 0)]
    if not next_open.empty and not obs_close.empty:
        lines.append(
            f"- Observation-close net@50: {_fmt(float(obs_close.iloc[0]['net_return_mean_pct_50']))}; "
            f"next-open net@50: {_fmt(float(next_open.iloc[0]['net_return_mean_pct_50']))}."
        )
    else:
        lines.append("- Insufficient execution summary rows.")

    lines += [
        "",
        "## 2) Does it remain positive after 100/150/200 bps?",
    ]
    if not cost_summary.empty:
        r = cost_summary.iloc[0]
        lines.append(
            f"- net@100: {_fmt(float(r['net_return_mean_pct_100']))}, "
            f"net@150: {_fmt(float(r['net_return_mean_pct_150']))}, "
            f"net@200: {_fmt(float(r['net_return_mean_pct_200']))}."
        )
    else:
        lines.append("- Cost summary unavailable.")

    lines += [
        "",
        "## 3) Is it stable year by year?",
        f"- Year rows: {len(yearly_summary)}",
        "",
        "## 4) Is it dominated by one year, one coin, or one regime?",
        f"- Symbol rows: {len(symbol_summary)}",
        f"- BTC regime rows: {len(btc_regime_summary)}",
        "",
        "## 5) Does it survive excluding top contributors?",
    ]
    excl = symbol_summary[symbol_summary.get("excluded_top_k", pd.Series(dtype=float)).notna()] if not symbol_summary.empty else pd.DataFrame()
    if not excl.empty:
        for _, row in excl.iterrows():
            lines.append(
                f"- Excluding top {int(row['excluded_top_k'])}: net@50={_fmt(float(row['net_return_mean_pct_50']))}, "
                f"net@100={_fmt(float(row['net_return_mean_pct_100']))}."
            )
    else:
        lines.append("- Exclusion diagnostics unavailable.")

    lines += [
        "",
        "## 6) Does it work only during favorable BTC regimes?",
        f"- See btc_regime_summary.csv (rows={len(btc_regime_summary)}).",
        "",
        "## 7) Does it beat random/placebo entries?",
    ]
    if not placebo_summary.empty:
        for _, row in placebo_summary.iterrows():
            lines.append(
                f"- {row['metric']}: candidate-placebo={_fmt(float(row['candidate_minus_placebo']))}, "
                f"p(placebo>=candidate)={float(row['placebo_p_ge_candidate']):.3f}."
            )
    else:
        lines.append("- Placebo check unavailable.")

    lines += [
        "",
        "## 8) Does it beat BUY_AT_EXPLOSION after realistic costs?",
    ]
    if not benchmark_summary.empty:
        cand = benchmark_summary[benchmark_summary["benchmark"] == "LOCKED_CANDIDATE"]
        base = benchmark_summary[benchmark_summary["benchmark"] == "BUY_AT_EXPLOSION_BASELINE"]
        if not cand.empty and not base.empty:
            lines.append(
                f"- Candidate net@50={_fmt(float(cand.iloc[0]['net_return_mean_pct_50']))} vs baseline net@50={_fmt(float(base.iloc[0]['net_return_mean_pct_50']))}."
            )
        else:
            lines.append("- Benchmark comparison rows incomplete.")
    else:
        lines.append("- Benchmark comparison unavailable.")

    lines += [
        "",
        "## 9) Prototype-ready or still exploratory?",
    ]
    if not cost_summary.empty:
        r = cost_summary.iloc[0]
        robust = float(r["net_return_mean_pct_100"]) > 0 and float(r["net_return_mean_pct_150"]) > 0
        lines.append(
            "- Preliminary verdict: "
            + ("candidate is stronger but still needs further out-of-sample monitoring." if robust else "still exploratory; robustness under heavier costs is limited.")
        )

    lines += [
        "",
        "## Safety Confirmation",
        "- Research-only validation pass.",
        "- No live trading behavior changed.",
        "- No scheduler/launchd, broker/exchange, credentials, allocation, or production config changes.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def run_locked_candidate_validation(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    study_cfg: DelayedEntryStudyConfig | None = None,
    candidate_cfg: LockedCandidateConfig | None = None,
) -> dict[str, pd.DataFrame]:
    study_cfg = study_cfg or DelayedEntryStudyConfig()
    candidate_cfg = candidate_cfg or LockedCandidateConfig()

    full_df = load_ohlcv_data(data_dir)
    signals_all = build_second_pass_signals(full_df, study_cfg)
    locked_signals = _candidate_signals(signals_all, candidate_cfg)
    locked_trades = _build_candidate_trades(locked_signals, full_df, candidate_cfg)

    execution_summary = build_execution_assumption_summary(locked_trades)
    yearly_summary = build_yearly_summary(locked_trades)
    train_test_holdout = build_train_test_holdout_summary(locked_trades, candidate_cfg)
    symbol_concentration = build_symbol_concentration_summary(locked_trades)
    overlap_concurrency = build_overlap_concurrency_summary(locked_trades, candidate_cfg)
    btc_regime = build_btc_regime_summary(locked_trades)
    liquidity = build_liquidity_summary(locked_trades)
    cost_sensitivity = build_cost_sensitivity_summary(locked_trades)
    bootstrap = build_bootstrap_summary(locked_trades, candidate_cfg)
    placebo = build_placebo_summary(locked_trades, full_df, locked_signals, candidate_cfg)
    benchmark = build_benchmark_comparison(locked_trades, signals_all, full_df, candidate_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    locked_trades.to_csv(output_dir / "locked_candidate_trades.csv", index=False)
    execution_summary.to_csv(output_dir / "execution_assumption_summary.csv", index=False)
    yearly_summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    train_test_holdout.to_csv(output_dir / "train_test_holdout_summary.csv", index=False)
    symbol_concentration.to_csv(output_dir / "symbol_concentration_summary.csv", index=False)
    overlap_concurrency.to_csv(output_dir / "overlap_concurrency_summary.csv", index=False)
    btc_regime.to_csv(output_dir / "btc_regime_summary.csv", index=False)
    liquidity.to_csv(output_dir / "liquidity_summary.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity_summary.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    if not placebo.empty:
        placebo.to_csv(output_dir / "placebo_summary.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_comparison.csv", index=False)

    write_summary_markdown(
        execution_summary=execution_summary,
        yearly_summary=yearly_summary,
        train_test_holdout=train_test_holdout,
        symbol_summary=symbol_concentration,
        overlap_summary=overlap_concurrency,
        btc_regime_summary=btc_regime,
        liquidity_summary=liquidity,
        cost_summary=cost_sensitivity,
        placebo_summary=placebo,
        benchmark_summary=benchmark,
        output_path=output_dir / "summary.md",
    )

    return {
        "locked_candidate_trades": locked_trades,
        "execution_assumption_summary": execution_summary,
        "yearly_summary": yearly_summary,
        "train_test_holdout_summary": train_test_holdout,
        "symbol_concentration_summary": symbol_concentration,
        "overlap_concurrency_summary": overlap_concurrency,
        "btc_regime_summary": btc_regime,
        "liquidity_summary": liquidity,
        "cost_sensitivity_summary": cost_sensitivity,
        "bootstrap_summary": bootstrap,
        "placebo_summary": placebo,
        "benchmark_comparison": benchmark,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locked candidate robustness validation (research-only)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bootstrap-iterations", type=int, default=500)
    p.add_argument("--placebo-iterations", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    candidate_cfg = LockedCandidateConfig(
        bootstrap_iterations=args.bootstrap_iterations,
        placebo_iterations=args.placebo_iterations,
    )

    out = run_locked_candidate_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        candidate_cfg=candidate_cfg,
    )

    trades = out["locked_candidate_trades"]
    print("=" * 80)
    print("LOCKED CANDIDATE VALIDATION (RESEARCH-ONLY)")
    print("=" * 80)
    print(f"Trades: {len(trades)}")
    if not trades.empty:
        print(f"Date range: {trades['event_timestamp'].min()} -> {trades['event_timestamp'].max()}")
    cost = out["cost_sensitivity_summary"]
    if not cost.empty:
        row = cost.iloc[0]
        print(
            "Cost sensitivity mean net returns: "
            + ", ".join(
                f"{bps}bps={row[f'net_return_mean_pct_{bps}']:+.4f}%" for bps in COST_BPS
            )
        )
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
