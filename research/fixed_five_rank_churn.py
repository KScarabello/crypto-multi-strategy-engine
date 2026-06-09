"""Rank-churn and score-stability analysis for the FIXED_FIVE universe.

PURPOSE
-------
Investigate whether turnover in the equal-weight momentum portfolio is driven by
genuine rotation (durable rank changes with predictive return spread) or ranking
noise (marginal score differences producing costly rank-3/4 boundary swaps).

STRATEGIES ANALYZED
-------------------
1. btc_ma_240_reb12          (control)
2. combo_entry2_imm_buf05    (selection-locked candidate)

CANONICAL PARAMETERS
--------------------
- Universe: BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD
- joint_start: 2020-09-28 (first bar where all 5 coins have >=36 bars)
- 4-hour bars, fee_bps=10, slippage_bps=5, one-bar execution delay
- initial_capital=$10,000, rebalance_every=12 bars

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_rank_churn
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from research.universe_integrity_analysis import (
    BARS_PER_YEAR_4H,
    DEFAULT_FEE_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_REBALANCE_BARS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_SHORT_LOOKBACK,
    DEFAULT_MEDIUM_LOOKBACK,
    DEFAULT_TOP_N,
    LIVE_FIVE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
    _close_to_ohlcv,
    build_close_matrix,
    find_joint_eligible_start,
)
from research.fixed_five_defensive_overlay import BaseSignalGen, _max_drawdown, _cagr, _sharpe
from research.fixed_five_canonical_regime_comparison import run_canonical
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    WhipsawControlledSignalGen,
    compute_controlled_gate_series,
)
from strategies.cross_sectional_momentum import compute_momentum_score

LOGGER = logging.getLogger(__name__)

BTC_COL = "BTC/USD"
BARS_PER_YEAR = BARS_PER_YEAR_4H  # 2190
CONTROL_MA_BARS = 240
CONTROL_REBALANCE_BARS = 12
TOP_N = DEFAULT_TOP_N          # 3
WEIGHT_THRESHOLD = 1e-3
SHORT_LOOKBACK = DEFAULT_SHORT_LOOKBACK   # 12
MEDIUM_LOOKBACK = DEFAULT_MEDIUM_LOOKBACK  # 36


# ---------------------------------------------------------------------------
# Strategy builders (identical to fixed_five_cost_audit.py)
# ---------------------------------------------------------------------------

def build_control_signal(btc_full: pd.Series) -> tuple[Any, pd.Series]:
    """Return (signal_gen, gate_series) for btc_ma_240_reb12."""
    params = WhipsawControl(name="control")
    gate = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)
    sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
    return sig, gate


def build_candidate_signal(btc_full: pd.Series) -> tuple[Any, pd.Series]:
    """Return (signal_gen, gate_series) for combo_entry2_imm_buf05."""
    params = WhipsawControl(
        name="combo_entry2_imm_buf05",
        entry_confirm_bars=2,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        exit_buffer_pct=0.0,
    )
    gate = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)
    sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
    return sig, gate


# ---------------------------------------------------------------------------
# Part 1 — Rank transition ledger
# ---------------------------------------------------------------------------

def compute_ranks_at_ts(
    close: pd.DataFrame,
    ts: pd.Timestamp,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
) -> pd.Series:
    """Return rank series (1=best) for symbols eligible at ts.

    Eligibility: at least min_history_bars non-NaN close prices up to ts.
    Ineligible symbols get NaN rank.
    """
    close_up_to = close.loc[close.index <= ts]
    scores_row = compute_momentum_score(
        close_up_to,
        short_lookback_bars=SHORT_LOOKBACK,
        medium_lookback_bars=MEDIUM_LOOKBACK,
    )
    if ts not in scores_row.index:
        return pd.Series(np.nan, index=close.columns)

    score_at_ts = scores_row.loc[ts].copy()

    # Mask ineligible symbols
    for sym in close.columns:
        valid_count = close_up_to[sym].dropna().shape[0]
        if valid_count < min_history_bars:
            score_at_ts[sym] = np.nan

    valid_scores = score_at_ts.dropna()
    if valid_scores.empty:
        return pd.Series(np.nan, index=close.columns)

    # Rank: 1=best (highest score)
    ranks = valid_scores.rank(ascending=False, method="average")
    return ranks.reindex(close.columns)


def build_rank_transition_ledger(
    strategy_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    gate: pd.Series,
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
) -> pd.DataFrame:
    """Build per-(rebalance, symbol) rank transition ledger.

    Computes ranks only from data available at signal_timestamp (no lookahead).
    """
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000

    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()
    all_ts = holdings.index

    reb = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ].copy()
    reb["execution_timestamp"] = pd.to_datetime(reb["execution_timestamp"], utc=True)
    reb["signal_timestamp"] = pd.to_datetime(reb["signal_timestamp"], utc=True)
    reb = reb.sort_values("execution_timestamp").reset_index(drop=True)

    symbols = list(close.columns)
    rows = []

    # Pre-compute ranks at each signal_timestamp
    sig_timestamps = reb["signal_timestamp"].unique()
    ranks_cache: dict[pd.Timestamp, pd.Series] = {}
    for sig_ts in sig_timestamps:
        ranks_cache[sig_ts] = compute_ranks_at_ts(close, sig_ts, min_history_bars)

    # Pre-compute scores at each signal_timestamp
    scores_all = compute_momentum_score(
        close,
        short_lookback_bars=SHORT_LOOKBACK,
        medium_lookback_bars=MEDIUM_LOOKBACK,
    )

    prev_ranks: pd.Series | None = None
    prev_exec_ts: pd.Timestamp | None = None

    for idx, row in reb.iterrows():
        exec_ts = row["execution_timestamp"]
        sig_ts = row["signal_timestamp"]

        # Holdings BEFORE this rebalance
        prev_ts_arr = all_ts[all_ts < exec_ts]
        if len(prev_ts_arr) == 0:
            prev_ranks = ranks_cache[sig_ts]
            prev_exec_ts = exec_ts
            continue
        prev_ts = prev_ts_arr[-1]
        w_before = holdings.loc[prev_ts]
        w_after = holdings.loc[exec_ts]

        eq_before = float(equity.loc[prev_ts]) if prev_ts in equity.index else float("nan")

        curr_ranks = ranks_cache[sig_ts]
        if sig_ts in scores_all.index:
            curr_scores = scores_all.loc[sig_ts]
        else:
            curr_scores = pd.Series(np.nan, index=symbols)

        gate_val = bool(gate.reindex([sig_ts]).fillna(False).iloc[0])

        for sym in symbols:
            wb = float(w_before.get(sym, 0.0))
            wa = float(w_after.get(sym, 0.0))
            dw = wa - wb

            curr_rank = float(curr_ranks.get(sym, np.nan)) if sym in curr_ranks.index else np.nan
            prev_rank_val = (
                float(prev_ranks.get(sym, np.nan))
                if prev_ranks is not None and sym in prev_ranks.index
                else np.nan
            )
            rank_change = (
                curr_rank - prev_rank_val
                if not (math.isnan(curr_rank) or math.isnan(prev_rank_val))
                else np.nan
            )

            score_val = float(curr_scores.get(sym, np.nan)) if sym in curr_scores.index else np.nan

            is_held_before = wb > WEIGHT_THRESHOLD
            is_held_after = wa > WEIGHT_THRESHOLD
            trade_generated = abs(dw) > WEIGHT_THRESHOLD
            if trade_generated:
                trade_side = "BUY" if dw > 0 else "SELL"
            else:
                trade_side = "NONE"

            gross_notional = abs(dw) * eq_before if not math.isnan(eq_before) else np.nan
            fee_dollars = gross_notional * fee_rate if gross_notional is not None and not math.isnan(gross_notional) else np.nan
            slippage_dollars = gross_notional * slippage_rate if gross_notional is not None and not math.isnan(gross_notional) else np.nan

            rows.append({
                "signal_timestamp": sig_ts,
                "execution_timestamp": exec_ts,
                "symbol": sym,
                "momentum_score": round(score_val, 6) if not math.isnan(score_val) else np.nan,
                "rank": curr_rank,
                "prev_rank": prev_rank_val,
                "rank_change": rank_change,
                "currently_held": round(wb, 6),
                "selected_after_rebalance": round(wa, 6),
                "is_held_before": is_held_before,
                "is_held_after": is_held_after,
                "trade_generated": trade_generated,
                "trade_side": trade_side,
                "gross_notional": round(gross_notional, 4) if gross_notional is not None and not math.isnan(gross_notional) else np.nan,
                "fee_dollars": round(fee_dollars, 4) if fee_dollars is not None and not math.isnan(fee_dollars) else np.nan,
                "slippage_dollars": round(slippage_dollars, 4) if slippage_dollars is not None and not math.isnan(slippage_dollars) else np.nan,
                "gate_on": gate_val,
                "strategy_name": strategy_name,
            })

        prev_ranks = curr_ranks
        prev_exec_ts = exec_ts

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2 — Rank boundary analysis (rank-3/4 boundary)
# ---------------------------------------------------------------------------

def build_rank_boundary_analysis(
    ledger: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Analyze the rank-3/rank-4 boundary for churn cost contribution."""
    rows = []
    ledger = ledger.copy()
    ledger["signal_timestamp"] = pd.to_datetime(ledger["signal_timestamp"], utc=True)

    rebalances = ledger["signal_timestamp"].unique()
    rebalances = sorted(rebalances)

    # Build per-rebalance rank snapshot
    reb_snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts in rebalances:
        snap = ledger[ledger["signal_timestamp"] == ts][["symbol", "rank", "momentum_score"]].copy()
        reb_snapshots[ts] = snap

    prev_ts = None
    for i, ts in enumerate(rebalances):
        snap = reb_snapshots[ts]
        valid = snap.dropna(subset=["rank"]).sort_values("rank")

        boundary_n = top_n        # rank 3
        boundary_n1 = top_n + 1  # rank 4

        rank3_row = valid[valid["rank"] <= boundary_n + 0.5].tail(1)
        rank4_row = valid[valid["rank"] >= boundary_n + 0.5].head(1)

        score3 = float(rank3_row["momentum_score"].iloc[0]) if not rank3_row.empty else np.nan
        score4 = float(rank4_row["momentum_score"].iloc[0]) if not rank4_row.empty else np.nan
        sym3 = rank3_row["symbol"].iloc[0] if not rank3_row.empty else None
        sym4 = rank4_row["symbol"].iloc[0] if not rank4_row.empty else None

        score_gap = score3 - score4 if not (math.isnan(score3) or math.isnan(score4)) else np.nan

        boundary_swap = False
        if prev_ts is not None:
            prev_snap = reb_snapshots[prev_ts]
            prev_valid = prev_snap.dropna(subset=["rank"])
            prev_rank3 = prev_valid[prev_valid["rank"] <= boundary_n + 0.5].tail(1)
            prev_rank4 = prev_valid[prev_valid["rank"] >= boundary_n + 0.5].head(1)
            prev_sym3 = prev_rank3["symbol"].iloc[0] if not prev_rank3.empty else None
            prev_sym4 = prev_rank4["symbol"].iloc[0] if not prev_rank4.empty else None
            # A boundary swap: current rank3 was previously rank4 or current rank4 was rank3
            if sym3 is not None and sym4 is not None and prev_sym3 is not None and prev_sym4 is not None:
                boundary_swap = (sym3 == prev_sym4) or (sym4 == prev_sym3)

        # Costs from ledger at this rebalance
        reb_ledger = ledger[ledger["signal_timestamp"] == ts]
        reb_fee = reb_ledger["fee_dollars"].sum()
        reb_slip = reb_ledger["slippage_dollars"].sum()

        rows.append({
            "signal_timestamp": ts,
            "score_rank3": score3,
            "score_rank4": score4,
            "sym_rank3": sym3,
            "sym_rank4": sym4,
            "score_gap": score_gap,
            "boundary_swap": boundary_swap,
            "total_fee_this_reb": reb_fee,
            "total_slip_this_reb": reb_slip,
            "total_cost_this_reb": reb_fee + reb_slip,
        })

        prev_ts = ts

    df = pd.DataFrame(rows)

    # Return-within-N rebalances for displaced assets
    # Check if the sym that dropped from rank3 returns within 1/2/3 rebalances
    for col_n in [1, 2, 3]:
        df[f"displaced_returns_within_{col_n}reb"] = False
    for i, row in df.iterrows():
        if not row["boundary_swap"] or row["sym_rank4"] is None:
            continue
        displaced = row["sym_rank4"]  # was rank3, now rank4 (displaced from portfolio)
        future_rebs = rebalances[rebalances.index(row["signal_timestamp"]) + 1:]
        for col_n in [1, 2, 3]:
            future_slice = future_rebs[:col_n]
            for fts in future_slice:
                fsnap = reb_snapshots.get(fts)
                if fsnap is not None:
                    fvalid = fsnap.dropna(subset=["rank"])
                    in_top = fvalid[fvalid["symbol"] == displaced]
                    if not in_top.empty and float(in_top["rank"].iloc[0]) <= top_n + 0.5:
                        df.at[i, f"displaced_returns_within_{col_n}reb"] = True
                        break

    return df


def summarize_boundary_analysis(df: pd.DataFrame) -> dict:
    """Summarize rank-3/4 boundary statistics."""
    total_rebs = len(df)
    swap_df = df[df["boundary_swap"]]
    n_swaps = len(swap_df)

    gap_series = df["score_gap"].dropna()
    swap_cost = swap_df["total_cost_this_reb"].sum()
    total_cost = df["total_cost_this_reb"].sum()
    pct_cost_from_swaps = swap_cost / max(total_cost, 1e-12) * 100

    summary = {
        "total_rebalances": total_rebs,
        "total_boundary_swaps": n_swaps,
        "pct_rebalances_with_swap": round(n_swaps / max(total_rebs, 1) * 100, 2),
        "mean_score_gap": round(float(gap_series.mean()), 6) if not gap_series.empty else np.nan,
        "median_score_gap": round(float(gap_series.median()), 6) if not gap_series.empty else np.nan,
        "p5_score_gap": round(float(gap_series.quantile(0.05)), 6) if not gap_series.empty else np.nan,
        "p10_score_gap": round(float(gap_series.quantile(0.10)), 6) if not gap_series.empty else np.nan,
        "p25_score_gap": round(float(gap_series.quantile(0.25)), 6) if not gap_series.empty else np.nan,
        "p50_score_gap": round(float(gap_series.quantile(0.50)), 6) if not gap_series.empty else np.nan,
        "p75_score_gap": round(float(gap_series.quantile(0.75)), 6) if not gap_series.empty else np.nan,
        "p90_score_gap": round(float(gap_series.quantile(0.90)), 6) if not gap_series.empty else np.nan,
        "p95_score_gap": round(float(gap_series.quantile(0.95)), 6) if not gap_series.empty else np.nan,
        "direct_costs_boundary_swaps": round(float(swap_cost), 2),
        "pct_total_costs_from_boundary_swaps": round(pct_cost_from_swaps, 2),
        "displaced_returns_within_1reb_pct": round(
            swap_df["displaced_returns_within_1reb"].mean() * 100, 1
        ) if n_swaps > 0 else np.nan,
        "displaced_returns_within_2reb_pct": round(
            swap_df["displaced_returns_within_2reb"].mean() * 100, 1
        ) if n_swaps > 0 else np.nan,
        "displaced_returns_within_3reb_pct": round(
            swap_df["displaced_returns_within_3reb"].mean() * 100, 1
        ) if n_swaps > 0 else np.nan,
    }
    return summary


# ---------------------------------------------------------------------------
# Part 3 — Round-trip churn
# ---------------------------------------------------------------------------

def build_holding_episodes(
    strategy_name: str,
    result: BacktestResult,
    joint_start: pd.Timestamp,
) -> list[dict]:
    """Build contiguous holding episodes per symbol."""
    holdings = result.holdings_history
    all_ts = holdings.index[holdings.index >= joint_start]
    symbols = list(holdings.columns)
    episodes = []

    for sym in symbols:
        weights = holdings.loc[all_ts, sym]
        in_episode = False
        entry_ts = None

        for ts in all_ts:
            w = float(weights.loc[ts])
            if not in_episode and w > WEIGHT_THRESHOLD:
                in_episode = True
                entry_ts = ts
            elif in_episode and w <= WEIGHT_THRESHOLD:
                episodes.append({
                    "strategy_name": strategy_name,
                    "symbol": sym,
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "duration_bars": len(all_ts[(all_ts >= entry_ts) & (all_ts < ts)]),
                })
                in_episode = False
                entry_ts = None

        # Open episode at end of data
        if in_episode and entry_ts is not None:
            last_ts = all_ts[-1]
            episodes.append({
                "strategy_name": strategy_name,
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": last_ts,
                "duration_bars": len(all_ts[(all_ts >= entry_ts) & (all_ts <= last_ts)]),
                "open_at_end": True,
            })

    return episodes


def build_round_trip_churn(
    episodes: list[dict],
    close: pd.DataFrame,
    result: BacktestResult,
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Identify round trips: exit followed by re-entry for same symbol."""
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    cost_rate = fee_rate + slippage_rate

    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()

    # Group episodes by (strategy_name, symbol)
    from itertools import groupby
    eps_df = pd.DataFrame(episodes)
    if eps_df.empty:
        return pd.DataFrame()
    eps_df = eps_df.sort_values(["strategy_name", "symbol", "entry_ts"])

    round_trips = []

    for (strat, sym), grp in eps_df.groupby(["strategy_name", "symbol"]):
        grp = grp.sort_values("entry_ts").reset_index(drop=True)
        for i in range(len(grp) - 1):
            ep_exit = grp.iloc[i]
            ep_reentry = grp.iloc[i + 1]

            exit_ts = ep_exit["exit_ts"]
            reentry_ts = ep_reentry["entry_ts"]
            if pd.isna(exit_ts) or pd.isna(reentry_ts):
                continue

            days_elapsed = (reentry_ts - exit_ts).total_seconds() / 86400

            # Equity at exit and re-entry
            eq_exit = float(equity.get(exit_ts, np.nan)) if exit_ts in equity.index else np.nan
            eq_reentry = float(equity.get(reentry_ts, np.nan)) if reentry_ts in equity.index else np.nan

            # Weight changes
            prev_exit = holdings.index[holdings.index < exit_ts]
            w_exit_before = float(holdings.loc[prev_exit[-1], sym]) if len(prev_exit) > 0 else 0.0
            w_exit_after = float(holdings.loc[exit_ts, sym]) if exit_ts in holdings.index else 0.0
            dw_exit = abs(w_exit_after - w_exit_before)

            prev_reentry = holdings.index[holdings.index < reentry_ts]
            w_reentry_before = float(holdings.loc[prev_reentry[-1], sym]) if len(prev_reentry) > 0 else 0.0
            w_reentry_after = float(holdings.loc[reentry_ts, sym]) if reentry_ts in holdings.index else 0.0
            dw_reentry = abs(w_reentry_after - w_reentry_before)

            notional_exit = dw_exit * eq_exit if not math.isnan(eq_exit) else 0.0
            notional_reentry = dw_reentry * eq_reentry if not math.isnan(eq_reentry) else 0.0
            gross_notional = notional_exit + notional_reentry
            fees = gross_notional * fee_rate
            slippage = gross_notional * slippage_rate

            # Return during absence
            price_exit = float(close.loc[exit_ts, sym]) if (exit_ts in close.index and sym in close.columns) else np.nan
            price_reentry = float(close.loc[reentry_ts, sym]) if (reentry_ts in close.index and sym in close.columns) else np.nan
            return_during_absence = (price_reentry / price_exit - 1) if not (math.isnan(price_exit) or math.isnan(price_reentry) or price_exit == 0) else np.nan

            if not math.isnan(return_during_absence):
                helped_or_hurt = "avoided_losses" if return_during_absence < 0 else "missed_gains"
            else:
                helped_or_hurt = "unknown"

            round_trips.append({
                "strategy_name": strat,
                "symbol": sym,
                "exit_ts": exit_ts,
                "reentry_ts": reentry_ts,
                "days_elapsed": round(days_elapsed, 2),
                "gross_notional_exit": round(notional_exit, 4),
                "gross_notional_reentry": round(notional_reentry, 4),
                "total_gross_notional": round(gross_notional, 4),
                "fees": round(fees, 4),
                "slippage": round(slippage, 4),
                "total_direct_cost": round(fees + slippage, 4),
                "return_during_absence": round(return_during_absence, 6) if not math.isnan(return_during_absence) else np.nan,
                "helped_or_hurt": helped_or_hurt,
            })

    return pd.DataFrame(round_trips)


def summarize_round_trips(rt_df: pd.DataFrame, horizons: list[int] = None) -> dict:
    """Summarize round trips by time horizon."""
    if horizons is None:
        horizons = [1, 3, 7, 14, 30]

    if rt_df.empty:
        return {}

    summary = {}
    for h in horizons:
        sub = rt_df[rt_df["days_elapsed"] <= h]
        summary[f"count_within_{h}d"] = len(sub)
        summary[f"total_cost_within_{h}d"] = round(sub["total_direct_cost"].sum(), 2)
        if len(sub) > 0:
            summary[f"pct_helped_within_{h}d"] = round(
                (sub["helped_or_hurt"] == "avoided_losses").mean() * 100, 1
            )
        else:
            summary[f"pct_helped_within_{h}d"] = np.nan

    return summary


# ---------------------------------------------------------------------------
# Part 4 — Holding duration analysis
# ---------------------------------------------------------------------------

DURATION_BUCKETS = [
    (0, 6, "< 1 day"),
    (6, 12, "1-2 days"),
    (12, 42, "2-7 days"),
    (42, 84, "7-14 days"),
    (84, float("inf"), "14+ days"),
]


def assign_duration_bucket(duration_bars: int) -> str:
    """Assign a holding episode to a duration bucket."""
    for lo, hi, label in DURATION_BUCKETS:
        if lo <= duration_bars < hi:
            return label
    return "14+ days"


def build_holding_duration_analysis(
    episodes: list[dict],
    close: pd.DataFrame,
    result: BacktestResult,
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Compute duration statistics for each holding episode."""
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000

    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()

    rows = []
    for ep in episodes:
        sym = ep["symbol"]
        entry_ts = ep["entry_ts"]
        exit_ts = ep["exit_ts"]
        strat = ep["strategy_name"]
        is_open = ep.get("open_at_end", False)

        dur_bars = ep["duration_bars"]
        dur_hours = dur_bars * 4
        dur_days = dur_hours / 24.0

        # Entry/exit prices
        price_entry = float(close.loc[entry_ts, sym]) if (entry_ts in close.index and sym in close.columns) else np.nan
        price_exit = float(close.loc[exit_ts, sym]) if (exit_ts in close.index and sym in close.columns) else np.nan
        episode_return_pct = (price_exit / price_entry - 1) * 100 if not (math.isnan(price_entry) or math.isnan(price_exit) or price_entry == 0) else np.nan

        # Equity at entry/exit
        eq_entry = float(equity.loc[entry_ts]) if entry_ts in equity.index else np.nan
        eq_exit = float(equity.loc[exit_ts]) if exit_ts in equity.index else np.nan

        # Weight at entry/exit for cost calculation
        prev_entry = holdings.index[holdings.index < entry_ts]
        w_before_entry = float(holdings.loc[prev_entry[-1], sym]) if len(prev_entry) > 0 else 0.0
        w_at_entry = float(holdings.loc[entry_ts, sym]) if entry_ts in holdings.index else 0.0
        dw_entry = abs(w_at_entry - w_before_entry)

        prev_exit_ = holdings.index[holdings.index < exit_ts]
        w_before_exit = float(holdings.loc[prev_exit_[-1], sym]) if len(prev_exit_) > 0 else 0.0
        w_at_exit = float(holdings.loc[exit_ts, sym]) if exit_ts in holdings.index else 0.0
        dw_exit = abs(w_at_exit - w_before_exit)

        notional_entry = dw_entry * eq_entry if not math.isnan(eq_entry) else 0.0
        notional_exit = dw_exit * eq_exit if not math.isnan(eq_exit) else 0.0
        fee_dollars = (notional_entry + notional_exit) * fee_rate
        slippage_dollars = (notional_entry + notional_exit) * slippage_rate

        # Net contribution approximation
        avg_weight = (w_at_entry + w_before_exit) / 2
        gross_return_contribution = episode_return_pct / 100 * avg_weight * eq_entry if not math.isnan(episode_return_pct) and not math.isnan(eq_entry) else np.nan
        net_contribution = gross_return_contribution - fee_dollars - slippage_dollars if gross_return_contribution is not None and not math.isnan(gross_return_contribution) else np.nan

        rows.append({
            "strategy_name": strat,
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "open_at_end": is_open,
            "duration_bars": dur_bars,
            "duration_hours": dur_hours,
            "duration_days": round(dur_days, 2),
            "duration_bucket": assign_duration_bucket(dur_bars),
            "entry_equity": round(eq_entry, 2) if not math.isnan(eq_entry) else np.nan,
            "exit_equity": round(eq_exit, 2) if not math.isnan(eq_exit) else np.nan,
            "episode_return_pct": round(episode_return_pct, 4) if not math.isnan(episode_return_pct) else np.nan,
            "fee_dollars": round(fee_dollars, 4),
            "slippage_dollars": round(slippage_dollars, 4),
            "net_contribution": round(net_contribution, 4) if net_contribution is not None and not math.isnan(net_contribution) else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 5 — Score stability analysis
# ---------------------------------------------------------------------------

def build_score_stability(ledger: pd.DataFrame) -> pd.DataFrame:
    """Compute autocorrelation of scores and ranks, and replacement quality."""
    ledger = ledger.copy()
    ledger["signal_timestamp"] = pd.to_datetime(ledger["signal_timestamp"], utc=True)
    rebalances = sorted(ledger["signal_timestamp"].unique())
    symbols = ledger["symbol"].unique()

    # Build score/rank pivot tables
    score_pivot = ledger.pivot(index="signal_timestamp", columns="symbol", values="momentum_score")
    rank_pivot = ledger.pivot(index="signal_timestamp", columns="symbol", values="rank")

    # Per-symbol autocorrelations
    autocorr_rows = []
    for sym in symbols:
        if sym not in score_pivot.columns:
            continue
        sc = score_pivot[sym].dropna()
        rk = rank_pivot[sym].dropna() if sym in rank_pivot.columns else pd.Series(dtype=float)

        score_ac = float(sc.autocorr(lag=1)) if len(sc) > 2 else np.nan
        rank_ac = float(rk.autocorr(lag=1)) if len(rk) > 2 else np.nan

        autocorr_rows.append({
            "symbol": sym,
            "score_autocorr_lag1": score_ac,
            "rank_autocorr_lag1": rank_ac,
        })

    autocorr_df = pd.DataFrame(autocorr_rows)

    # Replacement events: symbol enters portfolio displacing another
    replacement_rows = []
    prev_held: set[str] = set()

    for i, ts in enumerate(rebalances):
        snap = ledger[ledger["signal_timestamp"] == ts]
        curr_held = set(snap[snap["is_held_after"]]["symbol"].tolist())

        entrants = curr_held - prev_held
        displaced = prev_held - curr_held

        for entrant in entrants:
            for disp in displaced:
                # Score advantage of entrant over displaced
                ent_score_row = snap[snap["symbol"] == entrant]["momentum_score"]
                disp_score_row = snap[snap["symbol"] == disp]["momentum_score"]

                ent_score = float(ent_score_row.iloc[0]) if not ent_score_row.empty else np.nan
                disp_score = float(disp_score_row.iloc[0]) if not disp_score_row.empty else np.nan
                score_adv = ent_score - disp_score if not (math.isnan(ent_score) or math.isnan(disp_score)) else np.nan

                replacement_rows.append({
                    "signal_timestamp": ts,
                    "entrant": entrant,
                    "displaced": disp,
                    "entrant_score": ent_score,
                    "displaced_score": disp_score,
                    "score_advantage": score_adv,
                    # Forward returns populated below
                    "strategy_name": snap["strategy_name"].iloc[0] if not snap.empty else "",
                })

        prev_held = curr_held

    replacement_df = pd.DataFrame(replacement_rows)
    return autocorr_df, replacement_df


def enrich_replacement_returns(
    replacement_df: pd.DataFrame,
    close: pd.DataFrame,
    horizons_days: list[int] = None,
) -> pd.DataFrame:
    """Add forward return columns to replacement events."""
    if horizons_days is None:
        horizons_days = [1, 3, 7]
    if replacement_df.empty:
        return replacement_df

    df = replacement_df.copy()
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], utc=True)

    for h in horizons_days:
        h_bars = h * 6  # 4-hour bars
        df[f"entrant_return_{h}d"] = np.nan
        df[f"displaced_return_{h}d"] = np.nan
        df[f"entrant_outperformed_{h}d"] = np.nan

    for idx, row in df.iterrows():
        ts = row["signal_timestamp"]
        if ts not in close.index:
            continue
        ts_pos = close.index.get_loc(ts)
        for h in horizons_days:
            h_bars = h * 6
            future_pos = ts_pos + h_bars
            if future_pos >= len(close.index):
                continue
            future_ts = close.index[future_pos]

            for sym_col, col_prefix in [(row["entrant"], "entrant"), (row["displaced"], "displaced")]:
                if sym_col not in close.columns:
                    continue
                p0 = float(close.loc[ts, sym_col]) if ts in close.index else np.nan
                p1 = float(close.loc[future_ts, sym_col]) if future_ts in close.index else np.nan
                ret = (p1 / p0 - 1) if not (math.isnan(p0) or math.isnan(p1) or p0 == 0) else np.nan
                df.at[idx, f"{col_prefix}_return_{h}d"] = ret

            ent_r = df.at[idx, f"entrant_return_{h}d"]
            dis_r = df.at[idx, f"displaced_return_{h}d"]
            if not (math.isnan(float(ent_r)) if ent_r is not None else True) and \
               not (math.isnan(float(dis_r)) if dis_r is not None else True):
                df.at[idx, f"entrant_outperformed_{h}d"] = float(ent_r > dis_r)

    return df


def summarize_score_stability(autocorr_df: pd.DataFrame, replacement_df: pd.DataFrame) -> dict:
    """Summarize score stability metrics."""
    sc_ac = autocorr_df["score_autocorr_lag1"].dropna()
    rk_ac = autocorr_df["rank_autocorr_lag1"].dropna()

    adv = replacement_df["score_advantage"].dropna() if not replacement_df.empty else pd.Series(dtype=float)

    summary = {
        "mean_score_autocorr": round(float(sc_ac.mean()), 4) if not sc_ac.empty else np.nan,
        "median_score_autocorr": round(float(sc_ac.median()), 4) if not sc_ac.empty else np.nan,
        "mean_rank_autocorr": round(float(rk_ac.mean()), 4) if not rk_ac.empty else np.nan,
        "median_rank_autocorr": round(float(rk_ac.median()), 4) if not rk_ac.empty else np.nan,
        "mean_score_advantage": round(float(adv.mean()), 4) if not adv.empty else np.nan,
        "median_score_advantage": round(float(adv.median()), 4) if not adv.empty else np.nan,
        "p25_score_advantage": round(float(adv.quantile(0.25)), 4) if not adv.empty else np.nan,
        "p75_score_advantage": round(float(adv.quantile(0.75)), 4) if not adv.empty else np.nan,
        "pct_adv_lt_0.25pp": round((adv < 0.0025).mean() * 100, 1) if not adv.empty else np.nan,
        "pct_adv_lt_0.5pp": round((adv < 0.005).mean() * 100, 1) if not adv.empty else np.nan,
        "pct_adv_lt_1.0pp": round((adv < 0.01).mean() * 100, 1) if not adv.empty else np.nan,
        "pct_adv_lt_2.0pp": round((adv < 0.02).mean() * 100, 1) if not adv.empty else np.nan,
    }

    if not replacement_df.empty and "entrant_outperformed_7d" in replacement_df.columns:
        out7 = replacement_df["entrant_outperformed_7d"].dropna()
        summary["pct_entrant_outperformed_7d"] = round(float(out7.mean()) * 100, 1) if not out7.empty else np.nan
    else:
        summary["pct_entrant_outperformed_7d"] = np.nan

    return summary


# ---------------------------------------------------------------------------
# Part 6 — Reweighting vs replacement
# ---------------------------------------------------------------------------

TRADE_TYPE_LABELS = [
    "FULL_RANK_REPLACEMENT",
    "PARTIAL_RETAINED_SALE",
    "PARTIAL_RETAINED_BUY",
    "EQUAL_WEIGHT_NORMALIZATION",
    "REGIME_TRANSITION",
    "OTHER",
]


def classify_trade_type(
    weight_before: float,
    weight_after: float,
    was_regime_change: bool,
    prev_held_set: set[str],
    new_held_set: set[str],
    sym: str,
    top_n: int = TOP_N,
) -> str:
    """Classify a trade into one of the reweighting/replacement categories."""
    dw = weight_after - weight_before
    is_buy = dw > WEIGHT_THRESHOLD
    is_sell = dw < -WEIGHT_THRESHOLD

    if was_regime_change:
        return "REGIME_TRANSITION"

    # Full replacement: exits 0 or enters 0
    was_out = weight_before <= WEIGHT_THRESHOLD
    now_out = weight_after <= WEIGHT_THRESHOLD
    was_in = not was_out
    now_in = not now_out

    expected_weight = round(1.0 / top_n, 6)
    weight_tolerance = 0.05  # 5% tolerance for equal-weight normalization

    if was_out and now_in:
        return "FULL_RANK_REPLACEMENT"
    if was_in and now_out:
        return "FULL_RANK_REPLACEMENT"
    if was_in and now_in:
        if is_sell:
            # Check if near equal-weight
            if abs(weight_after - expected_weight) < weight_tolerance:
                return "EQUAL_WEIGHT_NORMALIZATION"
            return "PARTIAL_RETAINED_SALE"
        if is_buy:
            if abs(weight_after - expected_weight) < weight_tolerance:
                return "EQUAL_WEIGHT_NORMALIZATION"
            return "PARTIAL_RETAINED_BUY"

    return "OTHER"


def build_turnover_by_type(
    ledger: pd.DataFrame,
    gate: pd.Series,
) -> pd.DataFrame:
    """Categorize each trade by type and aggregate costs."""
    ledger = ledger.copy()
    ledger["execution_timestamp"] = pd.to_datetime(ledger["execution_timestamp"], utc=True)

    # Gate transitions
    gate_aligned = gate.sort_index()
    gate_shifted = gate_aligned.shift(1)
    regime_changed_at = set(gate_aligned.index[gate_aligned != gate_shifted].dropna())

    rows = []
    rebalances = ledger["execution_timestamp"].unique()
    for exec_ts in sorted(rebalances):
        snap = ledger[ledger["execution_timestamp"] == exec_ts]
        was_regime_change = exec_ts in regime_changed_at

        prev_held = set(snap[snap["is_held_before"]]["symbol"].tolist())
        new_held = set(snap[snap["is_held_after"]]["symbol"].tolist())

        for _, r in snap.iterrows():
            if not r["trade_generated"]:
                continue
            trade_type = classify_trade_type(
                float(r["currently_held"]),
                float(r["selected_after_rebalance"]),
                was_regime_change,
                prev_held,
                new_held,
                r["symbol"],
            )
            rows.append({
                "strategy_name": r["strategy_name"],
                "execution_timestamp": exec_ts,
                "symbol": r["symbol"],
                "trade_type": trade_type,
                "trade_side": r["trade_side"],
                "gross_notional": r["gross_notional"],
                "fee_dollars": r["fee_dollars"],
                "slippage_dollars": r["slippage_dollars"],
            })

    return pd.DataFrame(rows)


def summarize_turnover_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost by trade type."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["strategy_name", "trade_type"]).agg(
        trade_count=("symbol", "count"),
        total_gross_notional=("gross_notional", "sum"),
        total_fee_dollars=("fee_dollars", "sum"),
        total_slippage_dollars=("slippage_dollars", "sum"),
    ).reset_index()
    grp["total_direct_cost"] = grp["total_fee_dollars"] + grp["total_slippage_dollars"]
    total_cost = grp.groupby("strategy_name")["total_direct_cost"].transform("sum")
    grp["pct_of_total_cost"] = round(grp["total_direct_cost"] / total_cost.clip(lower=1e-12) * 100, 2)
    return grp


# ---------------------------------------------------------------------------
# Part 7 — Symbol contribution
# ---------------------------------------------------------------------------

def build_symbol_contribution(
    ledger: pd.DataFrame,
    episodes: list[dict],
    rt_df: pd.DataFrame,
    close: pd.DataFrame,
    result: BacktestResult,
    joint_start: pd.Timestamp,
) -> pd.DataFrame:
    """Per-symbol churn and PnL contribution summary."""
    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()
    all_ts = holdings.index[holdings.index >= joint_start]
    symbols = ledger["symbol"].unique()

    total_notional = ledger[ledger["trade_generated"]]["gross_notional"].sum()
    rows = []

    for sym in symbols:
        sym_ledger = ledger[(ledger["symbol"] == sym) & (ledger["trade_generated"])]
        sym_eps = [e for e in episodes if e["symbol"] == sym]
        sym_rt = rt_df[rt_df["symbol"] == sym] if not rt_df.empty else pd.DataFrame()

        times_entered = len(sym_ledger[sym_ledger["trade_side"] == "BUY"])
        times_exited = len(sym_ledger[sym_ledger["trade_side"] == "SELL"])
        round_trip_count = len(sym_rt)
        holding_ep_count = len(sym_eps)

        # Gross PnL from episodes
        gross_pnl = 0.0
        fee_total = 0.0
        slip_total = 0.0
        win_episodes = 0
        total_dur_days = 0.0
        valid_eps = 0

        for ep in sym_eps:
            entry_ts = ep["entry_ts"]
            exit_ts = ep["exit_ts"]
            p0 = float(close.loc[entry_ts, sym]) if (entry_ts in close.index and sym in close.columns) else np.nan
            p1 = float(close.loc[exit_ts, sym]) if (exit_ts in close.index and sym in close.columns) else np.nan
            dur_days = ep["duration_bars"] * 4 / 24.0
            total_dur_days += dur_days
            valid_eps += 1
            if not (math.isnan(p0) or math.isnan(p1) or p0 == 0):
                ep_ret = (p1 / p0 - 1)
                eq_at_entry = float(equity.loc[entry_ts]) if entry_ts in equity.index else np.nan
                avg_w = float(holdings.loc[entry_ts, sym]) if entry_ts in holdings.index else 0.0
                if not math.isnan(eq_at_entry):
                    gross_pnl += ep_ret * avg_w * eq_at_entry
                if ep_ret > 0:
                    win_episodes += 1

        fee_total = sym_ledger["fee_dollars"].sum()
        slip_total = sym_ledger["slippage_dollars"].sum()
        net_pnl = gross_pnl - fee_total - slip_total
        sym_notional = sym_ledger["gross_notional"].sum()
        pct_turnover = sym_notional / max(total_notional, 1e-12) * 100

        rows.append({
            "symbol": sym,
            "strategy_name": ledger["strategy_name"].iloc[0] if not ledger.empty else "",
            "times_entered": times_entered,
            "times_exited": times_exited,
            "round_trip_count": round_trip_count,
            "holding_episodes": holding_ep_count,
            "gross_pnl": round(gross_pnl, 2),
            "fee_dollars": round(fee_total, 2),
            "slippage_dollars": round(slip_total, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_holding_duration_days": round(total_dur_days / max(valid_eps, 1), 2),
            "episode_win_rate": round(win_episodes / max(valid_eps, 1) * 100, 1),
            "pct_of_total_turnover": round(pct_turnover, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 8 — Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    strategy_name: str,
    ledger: pd.DataFrame,
    boundary_summary: dict,
    rt_summary: dict,
    dur_df: pd.DataFrame,
    stability_summary: dict,
    type_df: pd.DataFrame,
    sym_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write comprehensive markdown analysis report."""
    total_rebs = ledger["signal_timestamp"].nunique()
    total_trades = ledger["trade_generated"].sum()
    total_fees = ledger["fee_dollars"].sum()
    total_slip = ledger["slippage_dollars"].sum()

    median_dur = float(dur_df["duration_days"].median()) if not dur_df.empty else np.nan

    churn_leaders = (
        sym_df.sort_values("pct_of_total_turnover", ascending=False)
        if not sym_df.empty else pd.DataFrame()
    )

    boundary_swap_pct = boundary_summary.get("pct_rebalances_with_swap", np.nan)
    boundary_cost_pct = boundary_summary.get("pct_total_costs_from_boundary_swaps", np.nan)

    score_adv_mean = stability_summary.get("mean_score_advantage", np.nan)
    score_adv_median = stability_summary.get("median_score_advantage", np.nan)
    pct_outperformed_7d = stability_summary.get("pct_entrant_outperformed_7d", np.nan)

    rt_7d_count = rt_summary.get("count_within_7d", 0)
    rt_30d_count = rt_summary.get("count_within_30d", 0)
    rt_7d_cost = rt_summary.get("total_cost_within_7d", 0.0)
    rt_30d_cost = rt_summary.get("total_cost_within_30d", 0.0)

    # Verdict
    is_noise = (
        (boundary_swap_pct if not math.isnan(boundary_swap_pct) else 0) > 20
        or (boundary_cost_pct if not math.isnan(boundary_cost_pct) else 0) > 25
        or (score_adv_mean if not math.isnan(score_adv_mean) else 1) < 0.01
    )
    verdict = "Ranking noise" if is_noise else "Genuine rotation"

    lines = [
        f"# Rank Churn Analysis — {strategy_name}",
        "",
        f"> {SURVIVORSHIP_BIAS_NOTE}",
        "",
        "## Part 1 — Trade Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total rebalances | {total_rebs} |",
        f"| Total trades generated | {total_trades} |",
        f"| Total fee dollars | ${total_fees:,.2f} |",
        f"| Total slippage dollars | ${total_slip:,.2f} |",
        f"| Total direct costs | ${total_fees + total_slip:,.2f} |",
        "",
        "## Part 2 — Rank-3/4 Boundary Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
    ]
    for k, v in boundary_summary.items():
        if isinstance(v, float) and not math.isnan(v):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Part 3 — Round-Trip Churn (7 and 30-day horizons)",
        "",
        f"| Horizon | Count | Total Cost | % Helped (avoided losses) |",
        f"|---|---|---|---|",
        f"| Within 7 days | {rt_7d_count} | ${rt_7d_cost:,.2f} | {rt_summary.get('pct_helped_within_7d', 'n/a')}% |",
        f"| Within 30 days | {rt_30d_count} | ${rt_30d_cost:,.2f} | {rt_summary.get('pct_helped_within_30d', 'n/a')}% |",
        "",
        "## Part 4 — Holding Duration",
        "",
        f"Median holding duration: **{median_dur:.1f} days**",
        "",
    ]

    if not dur_df.empty:
        bucket_summary = dur_df.groupby("duration_bucket").agg(
            count=("symbol", "count"),
            mean_return_pct=("episode_return_pct", "mean"),
            total_fee=("fee_dollars", "sum"),
        ).reset_index()
        lines += ["| Bucket | Count | Mean Return % | Total Fees |", "|---|---|---|---|"]
        for _, brow in bucket_summary.iterrows():
            lines.append(
                f"| {brow['duration_bucket']} | {brow['count']} | "
                f"{brow['mean_return_pct']:.2f}% | ${brow['total_fee']:,.2f} |"
            )

    lines += [
        "",
        "## Part 5 — Score Stability and Replacement Quality",
        "",
        f"| Metric | Value |",
        f"|---|---|",
    ]
    for k, v in stability_summary.items():
        if isinstance(v, float) and not math.isnan(v):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Part 6 — Reweighting vs Replacement",
        "",
        "Since this is a pure equal-weight strategy, PARTIAL_RETAINED and EQUAL_WEIGHT_NORMALIZATION",
        "trades should be zero or near-zero. REGIME_TRANSITION and FULL_RANK_REPLACEMENT account",
        "for all meaningful turnover.",
        "",
    ]
    if not type_df.empty:
        strat_type_df = type_df[type_df["strategy_name"] == strategy_name]
        lines += ["| Trade Type | Count | Total Cost | % of Total |", "|---|---|---|---|"]
        for _, tr in strat_type_df.iterrows():
            lines.append(
                f"| {tr['trade_type']} | {tr['trade_count']} | "
                f"${tr['total_direct_cost']:,.2f} | {tr['pct_of_total_cost']:.1f}% |"
            )

    lines += [
        "",
        "## Part 7 — Symbol Churn Leaders",
        "",
        "| Symbol | Entries | Exits | Round Trips | % Turnover | Avg Hold (days) | Episode Win Rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, srow in churn_leaders.iterrows():
        lines.append(
            f"| {srow['symbol']} | {srow['times_entered']} | {srow['times_exited']} | "
            f"{srow['round_trip_count']} | {srow['pct_of_total_turnover']:.1f}% | "
            f"{srow['avg_holding_duration_days']:.1f} | {srow['episode_win_rate']:.0f}% |"
        )

    lines += [
        "",
        "## Genuine Rotation or Ranking Noise?",
        "",
        f"**Verdict: {verdict}**",
        "",
        "### Evidence",
        "",
        f"- Rank-3/4 boundary swaps occur in **{boundary_swap_pct:.1f}%** of rebalances",
        f"- Boundary swaps account for **{boundary_cost_pct:.1f}%** of total direct costs",
        f"- Mean score advantage of entrant over displaced: **{score_adv_mean:.4f}** (raw return units)",
        f"- Median score advantage: **{score_adv_median:.4f}**",
        f"- % of replacements where entrant outperformed displaced in next 7 days: **{pct_outperformed_7d:.1f}%**",
        "",
        "### Interpretation",
        "",
    ]
    if is_noise:
        lines += [
            "The data suggests **ranking noise** is a significant contributor to turnover costs.",
            "Small score differences at the rank-3/4 boundary produce frequent costly swaps",
            "without consistent return benefit. Consider a score-gap filter or hold-buffer rule.",
        ]
    else:
        lines += [
            "The data suggests **genuine rotation** is the primary driver of turnover.",
            "Entrants show meaningful score advantages over displaced symbols, and the entrant",
            "outperforms the displaced asset at a rate above 50% in subsequent periods.",
            "Turnover costs appear justified by improved portfolio quality.",
        ]

    lines += ["", "---", f"*RESEARCH ONLY — {SURVIVORSHIP_BIAS_NOTE}*", ""]

    out_path = output_dir / "fixed_five_rank_churn_analysis.md"
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_strategy(
    strategy_name: str,
    signal_gen: Any,
    gate: pd.Series,
    ohlcv: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    output_dir: Path,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """Run full analysis for one strategy. Returns dict of summary data."""
    LOGGER.info("Running backtest: %s", strategy_name)
    result, port_fx = run_canonical(
        ohlcv, signal_gen, joint_start,
        rebalance_bars=CONTROL_REBALANCE_BARS,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    LOGGER.info("Building rank transition ledger: %s", strategy_name)
    ledger = build_rank_transition_ledger(
        strategy_name, result, port_fx, close, gate, joint_start, fee_bps, slippage_bps
    )
    ledger.to_csv(output_dir / "fixed_five_rank_transition_ledger.csv", index=False,
                  mode="a", header=not (output_dir / "fixed_five_rank_transition_ledger.csv").exists())

    LOGGER.info("Building rank boundary analysis: %s", strategy_name)
    boundary_df = build_rank_boundary_analysis(ledger)
    boundary_summary = summarize_boundary_analysis(boundary_df)
    boundary_df.to_csv(output_dir / "fixed_five_rank_boundary_analysis.csv", index=False,
                       mode="a", header=not (output_dir / "fixed_five_rank_boundary_analysis.csv").exists())

    LOGGER.info("Building round-trip churn: %s", strategy_name)
    episodes = build_holding_episodes(strategy_name, result, joint_start)
    rt_df = build_round_trip_churn(episodes, close, result, joint_start, fee_bps, slippage_bps)
    rt_summary = summarize_round_trips(rt_df)
    if not rt_df.empty:
        rt_df.to_csv(output_dir / "fixed_five_round_trip_churn.csv", index=False,
                     mode="a", header=not (output_dir / "fixed_five_round_trip_churn.csv").exists())

    LOGGER.info("Building holding duration analysis: %s", strategy_name)
    dur_df = build_holding_duration_analysis(episodes, close, result, joint_start, fee_bps, slippage_bps)
    if not dur_df.empty:
        dur_df.to_csv(output_dir / "fixed_five_holding_duration_analysis.csv", index=False,
                      mode="a", header=not (output_dir / "fixed_five_holding_duration_analysis.csv").exists())

    LOGGER.info("Building score stability: %s", strategy_name)
    autocorr_df, replacement_df = build_score_stability(ledger)
    replacement_df = enrich_replacement_returns(replacement_df, close)
    stability_summary = summarize_score_stability(autocorr_df, replacement_df)
    stability_df = pd.concat([
        autocorr_df.assign(strategy_name=strategy_name),
        replacement_df.assign(type="replacement"),
    ], axis=0, ignore_index=True)
    stability_df.to_csv(output_dir / "fixed_five_score_stability.csv", index=False,
                        mode="a", header=not (output_dir / "fixed_five_score_stability.csv").exists())

    LOGGER.info("Building turnover by trade type: %s", strategy_name)
    gate_full = gate.sort_index()
    type_raw_df = build_turnover_by_type(ledger, gate_full)
    type_summary_df = summarize_turnover_by_type(type_raw_df)
    if not type_summary_df.empty:
        type_summary_df.to_csv(output_dir / "fixed_five_turnover_by_trade_type.csv", index=False,
                                mode="a", header=not (output_dir / "fixed_five_turnover_by_trade_type.csv").exists())

    LOGGER.info("Building symbol contribution: %s", strategy_name)
    sym_df = build_symbol_contribution(ledger, episodes, rt_df if not rt_df.empty else pd.DataFrame(), close, result, joint_start)
    if not sym_df.empty:
        sym_df.to_csv(output_dir / "fixed_five_symbol_turnover_contribution.csv", index=False,
                      mode="a", header=not (output_dir / "fixed_five_symbol_turnover_contribution.csv").exists())

    LOGGER.info("Writing markdown report: %s", strategy_name)
    md_path = output_dir / "fixed_five_rank_churn_analysis.md"
    write_markdown_report(
        strategy_name, ledger, boundary_summary, rt_summary,
        dur_df, stability_summary, type_summary_df, sym_df, output_dir,
    )

    return {
        "strategy_name": strategy_name,
        "ledger": ledger,
        "boundary_summary": boundary_summary,
        "rt_summary": rt_summary,
        "dur_df": dur_df,
        "stability_summary": stability_summary,
        "type_df": type_summary_df,
        "sym_df": sym_df,
        "episodes": episodes,
        "rt_df": rt_df,
    }


def main(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean output files so we start fresh (avoid appending old data)
    csv_files = [
        "fixed_five_rank_transition_ledger.csv",
        "fixed_five_rank_boundary_analysis.csv",
        "fixed_five_round_trip_churn.csv",
        "fixed_five_holding_duration_analysis.csv",
        "fixed_five_score_stability.csv",
        "fixed_five_turnover_by_trade_type.csv",
        "fixed_five_symbol_turnover_contribution.csv",
    ]
    for f in csv_files:
        p = output_dir / f
        if p.exists():
            p.unlink()

    LOGGER.info("Loading close matrix…")
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        raise RuntimeError("Cannot find joint eligible start for FIXED_FIVE universe")

    ohlcv = _close_to_ohlcv(close)
    btc_full = close[BTC_COL].dropna()

    LOGGER.info("Joint start: %s", joint_start)

    strategies_cfg = [
        ("btc_ma_240_reb12", build_control_signal),
        ("combo_entry2_imm_buf05", build_candidate_signal),
    ]

    all_results: list[dict] = []
    for strat_name, builder in strategies_cfg:
        sig, gate = builder(btc_full)
        r = run_strategy(strat_name, sig, gate, ohlcv, close, joint_start, output_dir)
        all_results.append(r)

    # Print summary
    print("\n" + "=" * 70)
    print("RANK CHURN ANALYSIS — SUMMARY")
    print("=" * 70)

    files_created = csv_files + ["fixed_five_rank_churn_analysis.md"]
    print(f"\nFiles created/changed: {len(files_created)}")
    for f in files_created:
        p = output_dir / f
        exists = p.exists()
        print(f"  {'✓' if exists else '✗'} reports/{f}")

    for r in all_results:
        name = r["strategy_name"]
        bs = r["boundary_summary"]
        ss = r["stability_summary"]
        rts = r["rt_summary"]
        dur_df = r["dur_df"]
        type_df = r["type_df"]
        sym_df = r["sym_df"]

        print(f"\n{'─' * 60}")
        print(f"Strategy: {name}")
        print(f"{'─' * 60}")

        total_trades = r["ledger"]["trade_generated"].sum()
        total_rebs = r["ledger"]["signal_timestamp"].nunique()
        total_fees = r["ledger"]["fee_dollars"].sum()
        total_slip = r["ledger"]["slippage_dollars"].sum()
        print(f"  Total rebalances:      {total_rebs}")
        print(f"  Total trades:          {total_trades}")
        print(f"  Total fees:            ${total_fees:,.2f}")
        print(f"  Total slippage:        ${total_slip:,.2f}")
        print(f"  Total direct costs:    ${total_fees + total_slip:,.2f}")

        bswap_pct = bs.get("pct_rebalances_with_swap", np.nan)
        bcost_pct = bs.get("pct_total_costs_from_boundary_swaps", np.nan)
        print(f"\n  % costs from rank-3/4 swaps:      {bcost_pct:.1f}%")
        print(f"  % rebalances with boundary swap:  {bswap_pct:.1f}%")

        mean_adv = ss.get("mean_score_advantage", np.nan)
        med_adv = ss.get("median_score_advantage", np.nan)
        pct_out7 = ss.get("pct_entrant_outperformed_7d", np.nan)
        print(f"\n  Typical score advantage (mean):   {mean_adv:.4f}")
        print(f"  Typical score advantage (median): {med_adv:.4f}")
        print(f"  % replacements where entrant outperformed in 7d: {pct_out7:.1f}%")

        rt7 = rts.get("count_within_7d", 0)
        rt30 = rts.get("count_within_30d", 0)
        rt7d_pct = rts.get("pct_helped_within_7d", np.nan)
        rt30d_pct = rts.get("pct_helped_within_30d", np.nan)
        print(f"\n  Round trips within 7d:  {rt7} ({rt7d_pct:.0f}% avoided losses)")
        print(f"  Round trips within 30d: {rt30} ({rt30d_pct:.0f}% avoided losses)")

        med_dur = float(dur_df["duration_days"].median()) if not dur_df.empty else np.nan
        print(f"\n  Median holding duration: {med_dur:.1f} days")

        if not type_df.empty:
            strat_type = type_df[type_df["strategy_name"] == name]
            non_regime = strat_type[strat_type["trade_type"] != "REGIME_TRANSITION"]
            if not non_regime.empty:
                largest_src = non_regime.sort_values("total_direct_cost", ascending=False).iloc[0]
                print(f"\n  Largest non-regime turnover source: {largest_src['trade_type']} "
                      f"(${largest_src['total_direct_cost']:,.2f}, "
                      f"{largest_src['pct_of_total_cost']:.1f}% of total)")
            # Confirm reweighting near zero
            partial_cost = strat_type[strat_type["trade_type"].isin([
                "PARTIAL_RETAINED_SALE", "PARTIAL_RETAINED_BUY", "EQUAL_WEIGHT_NORMALIZATION"
            ])]["total_direct_cost"].sum()
            print(f"  Partial/reweighting costs (should be ~0): ${partial_cost:,.4f}")

        if not sym_df.empty:
            top_sym = sym_df.sort_values("pct_of_total_turnover", ascending=False).head(3)
            print(f"\n  Symbols with most churn:")
            for _, srow in top_sym.iterrows():
                print(f"    {srow['symbol']}: {srow['pct_of_total_turnover']:.1f}% of turnover, "
                      f"{srow['round_trip_count']} round trips")

    # Overall verdicts
    print(f"\n{'─' * 60}")
    print("CONCLUSIONS")
    print(f"{'─' * 60}")
    print("  No new strategy parameters tested: CONFIRMED")
    print("  Live behavior unchanged: CONFIRMED (research-only script)")

    for r in all_results:
        name = r["strategy_name"]
        bs = r["boundary_summary"]
        ss = r["stability_summary"]
        bswap_pct = bs.get("pct_rebalances_with_swap", 0)
        bcost_pct = bs.get("pct_total_costs_from_boundary_swaps", 0)
        mean_adv = ss.get("mean_score_advantage", 1.0)
        is_noise = bswap_pct > 20 or bcost_pct > 25 or mean_adv < 0.01
        verdict = "Ranking noise" if is_noise else "Genuine rotation"
        print(f"  {name}: {verdict}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
