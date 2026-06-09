"""Candidate filter grid experiment for memecoin backfilled signal events.

Evaluates a matrix of entry filters × exit policies over
data/memecoin_backfilled_signal_events.csv and ranks them against
a clear validation criterion.

Exit policies:
  fixed_4h      — return = future_ret_4h_pct
  tp10_else_4h  — +10% if max_favorable_4h_pct >= 10, else future_ret_4h_pct

Validation criteria (ranked priority):
  1. event_count >= 30
  2. positive median return
  3. positive avg excluding single best trade
  4. failure_rate_3 <= baseline failure_rate_3
  5. severe_failure_rate_5 <= baseline severe_failure_rate_5

Research only. No trading. No orders. No live code touched.

Usage:
    .venv/bin/python -m research.memecoin_catcher.test_backfill_candidate_filter_grid
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("data/memecoin_backfilled_signal_events.csv")
GRID_SUMMARY_PATH = Path("reports/memecoin_backfill_filter_grid_summary.csv")
GRID_EVENT_LOG_PATH = Path("reports/memecoin_backfill_filter_grid_event_log.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_FAILURE_HOURS: frozenset[int] = frozenset({4, 11, 12, 13, 22, 23})
SAFE_HOURS: list[int] = [17, 18, 19]
WIDER_SAFE_HOURS: list[int] = [16, 17, 18, 19, 20]
MIN_N: int = 30
TOP_PCT: float = 0.01   # top 1 % exclusion threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_events(path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load backfill event log, coerce types, and add derived columns."""
    df = pd.read_csv(path)
    for col in [
        "future_ret_4h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "max_favorable_24h_pct", "max_adverse_24h_pct",
        "volume_ratio_1h", "volume_ratio_4h",
        "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
        "primary_ohlc_score", "long_explosion_score",
        "spread_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive hour_of_day_utc from snapshot timestamp
    if "snapshot_ts_utc" in df.columns and "hour_of_day_utc" not in df.columns:
        ts = pd.to_datetime(df["snapshot_ts_utc"], utc=True, errors="coerce")
        df["hour_of_day_utc"] = ts.dt.hour

    return df


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------


def exit_fixed_4h(row: pd.Series) -> float:
    """Return the raw 4h forward return."""
    return float(row["future_ret_4h_pct"])


def exit_tp10_else_4h(row: pd.Series) -> float:
    """Return +10 if MFE >= 10, else raw 4h return."""
    mfe = row.get("max_favorable_4h_pct", float("nan"))
    ret = row["future_ret_4h_pct"]
    if pd.notna(mfe) and mfe >= 10.0:
        return 10.0
    return float(ret)


EXIT_POLICIES: dict[str, Callable[[pd.Series], float]] = {
    "fixed_4h": exit_fixed_4h,
    "tp10_else_4h": exit_tp10_else_4h,
}


def apply_exit_policy(df: pd.DataFrame, policy_fn: Callable) -> pd.Series:
    """Vectorised application of an exit policy; returns a float Series."""
    if policy_fn is exit_fixed_4h:
        return pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")
    if policy_fn is exit_tp10_else_4h:
        mfe_raw = df.get("max_favorable_4h_pct", None)
        ret = pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")
        if mfe_raw is None or (isinstance(mfe_raw, pd.Series) and mfe_raw.empty):
            return ret
        mfe = pd.to_numeric(mfe_raw, errors="coerce")
        tp_triggered = pd.notna(mfe) & (mfe >= 10.0)
        return ret.where(~tp_triggered, 10.0)
    # Fallback: row-by-row
    return df.apply(policy_fn, axis=1).astype(float)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _safe(v: float, d: int = 4) -> float:
    return round(v, d) if not (isinstance(v, float) and math.isnan(v)) else float("nan")


def compute_stats(
    returns: pd.Series,
    mfe: pd.Series | None = None,
    mae: pd.Series | None = None,
) -> dict:
    """Compute all required metrics for a return Series."""
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return {
            "n": 0,
            "avg": float("nan"), "median": float("nan"),
            "win_rate": float("nan"),
            "big_win_rate": float("nan"),
            "fail_rate_3": float("nan"),
            "fail_rate_5": float("nan"),
            "worst": float("nan"), "best": float("nan"),
            "avg_excl_best": float("nan"),
            "median_excl_best": float("nan"),
            "avg_excl_top1pct": float("nan"),
            "median_excl_top1pct": float("nan"),
            "avg_mfe": float("nan"), "median_mfe": float("nan"),
            "avg_mae": float("nan"), "median_mae": float("nan"),
        }

    excl_best = s.drop(index=s.idxmax())

    # Top 1 % exclusion: drop rows at or above the 99th-percentile threshold
    cutoff_99 = s.quantile(1.0 - TOP_PCT)
    excl_top1 = s[s < cutoff_99]

    row: dict = {
        "n": n,
        "avg": _safe(s.mean()),
        "median": _safe(s.median()),
        "win_rate": _safe((s > 0).mean()),
        "big_win_rate": _safe((s >= 5.0).mean()),
        "fail_rate_3": _safe((s <= -3.0).mean()),
        "fail_rate_5": _safe((s <= -5.0).mean()),
        "worst": _safe(s.min()),
        "best": _safe(s.max()),
        "avg_excl_best": _safe(excl_best.mean()) if len(excl_best) else float("nan"),
        "median_excl_best": _safe(excl_best.median()) if len(excl_best) else float("nan"),
        "avg_excl_top1pct": _safe(excl_top1.mean()) if len(excl_top1) else float("nan"),
        "median_excl_top1pct": _safe(excl_top1.median()) if len(excl_top1) else float("nan"),
    }

    # MFE / MAE (always from raw 4h excursions, not from policy-adjusted returns)
    if mfe is not None:
        mfe_s = pd.to_numeric(mfe, errors="coerce").reindex(s.index).dropna()
        row["avg_mfe"] = _safe(mfe_s.mean()) if len(mfe_s) else float("nan")
        row["median_mfe"] = _safe(mfe_s.median()) if len(mfe_s) else float("nan")
    else:
        row["avg_mfe"] = float("nan")
        row["median_mfe"] = float("nan")

    if mae is not None:
        mae_s = pd.to_numeric(mae, errors="coerce").reindex(s.index).dropna()
        row["avg_mae"] = _safe(mae_s.mean()) if len(mae_s) else float("nan")
        row["median_mae"] = _safe(mae_s.median()) if len(mae_s) else float("nan")
    else:
        row["avg_mae"] = float("nan")
        row["median_mae"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------------

FilterFn = Callable[[pd.DataFrame], pd.Series]  # returns bool mask


def _make_filters(df: pd.DataFrame) -> list[tuple[str, FilterFn]]:
    """
    Build the full list of (filter_name, mask_fn) pairs.

    Each mask_fn accepts a DataFrame and returns a boolean Series.
    Filters are defined here; the DataFrame is needed only to know which
    columns exist (for graceful degradation).
    """
    available = set(df.columns)
    has_hour = "hour_of_day_utc" in available
    has_vr1h = "volume_ratio_1h" in available
    has_vc = "is_volume_climax" in available
    has_r24h = "ret_24h_pct" in available
    has_cc = "is_clean_continuation" in available

    filters: list[tuple[str, FilterFn]] = []

    def add(name: str, fn: FilterFn) -> None:
        filters.append((name, fn))

    # -- 1. Baselines --------------------------------------------------------
    add("baseline_all", lambda df: pd.Series([True] * len(df), index=df.index))

    # -- 2. Volume-ratio filters ---------------------------------------------
    if has_vr1h:
        add("vr1h_ge_3",        lambda df: df["volume_ratio_1h"] >= 3)
        add("vr1h_ge_5",        lambda df: df["volume_ratio_1h"] >= 5)
        add("vr1h_ge_7",        lambda df: df["volume_ratio_1h"] >= 7)
        add("vr1h_ge_10",       lambda df: df["volume_ratio_1h"] >= 10)
        add("vr1h_3_to_15",     lambda df: (df["volume_ratio_1h"] >= 3) & (df["volume_ratio_1h"] <= 15))
        add("vr1h_5_to_20",     lambda df: (df["volume_ratio_1h"] >= 5) & (df["volume_ratio_1h"] <= 20))
    else:
        logger.warning("volume_ratio_1h absent — volume-ratio filters skipped.")

    # -- 3. Volume-climax filters --------------------------------------------
    if has_vc:
        add("volume_climax",
            lambda df: df["is_volume_climax"].astype(bool))
        if has_vr1h:
            add("volume_climax_AND_vr1h_ge_3",
                lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 3))
            add("volume_climax_AND_vr1h_ge_5",
                lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 5))
            add("volume_climax_AND_vr1h_5_to_20",
                lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 5) & (df["volume_ratio_1h"] <= 20))
    else:
        logger.warning("is_volume_climax absent — volume-climax filters skipped.")

    # -- 4. Hour-of-day filters ----------------------------------------------
    if has_hour:
        safe = frozenset(SAFE_HOURS)
        wider = frozenset(WIDER_SAFE_HOURS)
        high_fail = HIGH_FAILURE_HOURS

        add("hour_safe_17_18_19",
            lambda df: df["hour_of_day_utc"].isin(safe))
        add("hour_wider_16_to_20",
            lambda df: df["hour_of_day_utc"].isin(wider))
        add("excl_high_failure_hours",
            lambda df: ~df["hour_of_day_utc"].isin(high_fail))
    else:
        logger.warning("hour_of_day_utc absent — hour filters skipped.")

    # -- 5. Overextension filters --------------------------------------------
    if has_r24h:
        add("ret24h_lt_15",     lambda df: df["ret_24h_pct"] < 15)
        add("ret24h_lt_25",     lambda df: df["ret_24h_pct"] < 25)
        add("ret24h_lt_40",     lambda df: df["ret_24h_pct"] < 40)
        add("ret24h_0_to_25",   lambda df: (df["ret_24h_pct"] >= 0) & (df["ret_24h_pct"] < 25))
        add("ret24h_5_to_25",   lambda df: (df["ret_24h_pct"] >= 5) & (df["ret_24h_pct"] < 25))
        add("ret24h_0_to_40",   lambda df: (df["ret_24h_pct"] >= 0) & (df["ret_24h_pct"] < 40))
    else:
        logger.warning("ret_24h_pct absent — overextension filters skipped.")

    # -- 6. Combined candidates ----------------------------------------------
    if has_vr1h and has_hour:
        add("vr1h_ge_5_AND_hour_17_18_19",
            lambda df: (df["volume_ratio_1h"] >= 5) & df["hour_of_day_utc"].isin(frozenset(SAFE_HOURS)))
    if has_vc and has_hour:
        add("volume_climax_AND_hour_17_18_19",
            lambda df: df["is_volume_climax"].astype(bool) & df["hour_of_day_utc"].isin(frozenset(SAFE_HOURS)))
    if has_vr1h and has_vc:
        add("vr1h_ge_5_AND_volume_climax",
            lambda df: (df["volume_ratio_1h"] >= 5) & df["is_volume_climax"].astype(bool))
    if has_vr1h and has_r24h:
        add("vr1h_ge_5_AND_ret24h_lt_25",
            lambda df: (df["volume_ratio_1h"] >= 5) & (df["ret_24h_pct"] < 25))
    if has_vr1h and has_hour and has_r24h:
        add("vr1h_ge_5_AND_hour_17_18_19_AND_ret24h_lt_25",
            lambda df: (df["volume_ratio_1h"] >= 5) & df["hour_of_day_utc"].isin(frozenset(SAFE_HOURS)) & (df["ret_24h_pct"] < 25))
    if has_vc and has_vr1h and has_r24h:
        add("volume_climax_AND_vr1h_ge_5_AND_ret24h_lt_25",
            lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 5) & (df["ret_24h_pct"] < 25))
    if has_vc and has_vr1h and has_hour:
        add("volume_climax_AND_vr1h_ge_5_AND_hour_17_18_19",
            lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 5) & df["hour_of_day_utc"].isin(frozenset(SAFE_HOURS)))
    if has_vc and has_vr1h and has_hour and has_r24h:
        add("volume_climax_AND_vr1h_ge_5_AND_hour_17_18_19_AND_ret24h_lt_25",
            lambda df: df["is_volume_climax"].astype(bool) & (df["volume_ratio_1h"] >= 5) & df["hour_of_day_utc"].isin(frozenset(SAFE_HOURS)) & (df["ret_24h_pct"] < 25))

    # -- 7. Clean-continuation diagnostics -----------------------------------
    if has_cc:
        add("cc_true",          lambda df: df["is_clean_continuation"].astype(bool))
        add("cc_false",         lambda df: ~df["is_clean_continuation"].astype(bool))
        if has_vc:
            add("volume_climax_AND_cc_true",
                lambda df: df["is_volume_climax"].astype(bool) & df["is_clean_continuation"].astype(bool))
            add("volume_climax_AND_cc_false",
                lambda df: df["is_volume_climax"].astype(bool) & ~df["is_clean_continuation"].astype(bool))
    else:
        logger.warning("is_clean_continuation absent — CC diagnostic filters skipped.")

    return filters


# ---------------------------------------------------------------------------
# Validation criteria
# ---------------------------------------------------------------------------


def passes_validation(
    stats: dict,
    baseline_fail3: float,
    baseline_fail5: float,
) -> tuple[bool, list[str]]:
    """
    Return (passes, list_of_failed_criteria) given the metric dict.

    Criteria (all must pass):
      1. n >= MIN_N
      2. median > 0
      3. avg_excl_best > 0
      4. fail_rate_3 <= baseline_fail3
      5. fail_rate_5 <= baseline_fail5
    """
    failed: list[str] = []
    if stats["n"] < MIN_N:
        failed.append(f"n={stats['n']} < {MIN_N}")
    if not (isinstance(stats["median"], float) and not math.isnan(stats["median"]) and stats["median"] > 0):
        failed.append(f"median={stats['median']:.4f} <= 0")
    if not (isinstance(stats["avg_excl_best"], float) and not math.isnan(stats["avg_excl_best"]) and stats["avg_excl_best"] > 0):
        failed.append(f"avg_excl_best={stats['avg_excl_best']:.4f} <= 0")
    if not (isinstance(stats["fail_rate_3"], float) and not math.isnan(stats["fail_rate_3"]) and stats["fail_rate_3"] <= baseline_fail3):
        failed.append(f"fail_rate_3={stats['fail_rate_3']:.4f} > baseline {baseline_fail3:.4f}")
    if not (isinstance(stats["fail_rate_5"], float) and not math.isnan(stats["fail_rate_5"]) and stats["fail_rate_5"] <= baseline_fail5):
        failed.append(f"fail_rate_5={stats['fail_rate_5']:.4f} > baseline {baseline_fail5:.4f}")
    return len(failed) == 0, failed


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------


def run_filter_grid(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all filters × exit policies. Returns (summary_df, event_log_df).

    event_log_df has one row per input event and one column per
    (filter_name × policy_name) pair with the realised return, plus
    a column indicating whether the row was included in each filter.
    """
    completed = df.dropna(subset=["future_ret_4h_pct"]).copy()
    if len(completed) == 0:
        logger.warning("No completed events — returning empty results.")
        return pd.DataFrame(), pd.DataFrame()

    filters = _make_filters(completed)

    mfe_col = completed.get("max_favorable_4h_pct") if "max_favorable_4h_pct" in completed.columns else None
    mae_col = completed.get("max_adverse_4h_pct") if "max_adverse_4h_pct" in completed.columns else None

    # Pre-compute exit-policy returns for the full completed set (vectorised)
    policy_returns: dict[str, pd.Series] = {}
    for pname, pfn in EXIT_POLICIES.items():
        policy_returns[pname] = apply_exit_policy(completed, pfn)

    # Baseline stats (filter = all, policy = fixed_4h)
    baseline_stats = compute_stats(
        policy_returns["fixed_4h"],
        mfe=mfe_col,
        mae=mae_col,
    )
    baseline_fail3 = float(baseline_stats["fail_rate_3"])
    baseline_fail5 = float(baseline_stats["fail_rate_5"])

    summary_rows: list[dict] = []
    event_cols: dict[str, pd.Series] = {}

    for filter_name, mask_fn in filters:
        try:
            mask = mask_fn(completed)
        except Exception as exc:
            logger.warning("Filter %r raised %s — skipped.", filter_name, exc)
            continue

        mask = mask.fillna(False).astype(bool)
        sub = completed[mask]

        for pname, pfn in EXIT_POLICIES.items():
            sub_returns = policy_returns[pname][mask]
            sub_mfe = mfe_col[mask] if mfe_col is not None else None
            sub_mae = mae_col[mask] if mae_col is not None else None

            stats = compute_stats(sub_returns, mfe=sub_mfe, mae=sub_mae)
            passes, failed_criteria = passes_validation(stats, baseline_fail3, baseline_fail5)

            row = {
                "filter_name": filter_name,
                "exit_policy": pname,
                "passes_validation": passes,
                "failed_criteria": "; ".join(failed_criteria) if failed_criteria else "",
                **stats,
            }
            summary_rows.append(row)

            # Store realised returns for event log
            col_key = f"{filter_name}__{pname}"
            event_cols[col_key] = sub_returns.reindex(completed.index)

        # Store membership flag (one column per filter, not per policy)
        event_cols[f"in_filter__{filter_name}"] = mask.reindex(completed.index).astype(bool)

    summary_df = pd.DataFrame(summary_rows)

    # Build event log
    event_log_df = completed[[
        c for c in [
            "symbol", "pair_id", "snapshot_ts_utc",
            "is_clean_continuation", "is_volume_climax",
            "is_overextended_24h", "is_rolling_over",
            "is_wide_spread", "danger_terminal_spike",
            "volume_ratio_1h", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
            "hour_of_day_utc",
            "future_ret_4h_pct", "future_ret_24h_pct",
            "max_favorable_4h_pct", "max_adverse_4h_pct",
        ] if c in completed.columns
    ]].copy()

    for col_key, series in event_cols.items():
        event_log_df[col_key] = series.values

    return summary_df, event_log_df


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _top_n(
    df: pd.DataFrame,
    sort_col: str,
    min_n: int = MIN_N,
    n: int = 5,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return top-n rows filtered by min event count, sorted by sort_col."""
    valid = df[(df["n"] >= min_n) & df[sort_col].notna()]
    return valid.sort_values(sort_col, ascending=ascending).head(n)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _fmt(v, d: int = 2) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  —  "
    return f"{v:+.{d}f}%"


def _fmtr(v, d: int = 3) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  —  "
    return f"{v:.{d}f}"


def print_results(
    summary: pd.DataFrame,
    baseline_stats: dict,
) -> None:
    print()
    print("=" * 80)
    print("  MEMECOIN BACKFILL — CANDIDATE FILTER GRID RESULTS")
    print("=" * 80)
    print("  *** Research only. Not a validated trading strategy. ***")
    print()

    baseline_row = summary[
        (summary["filter_name"] == "baseline_all") &
        (summary["exit_policy"] == "fixed_4h")
    ]
    if not baseline_row.empty:
        r = baseline_row.iloc[0]
        print(f"  Baseline (all events, fixed_4h):  n={int(r['n'])}  avg={r['avg']:+.2f}%  "
              f"med={r['median']:+.2f}%  fail_rate_3={r['fail_rate_3']:.3f}  "
              f"fail_rate_5={r['fail_rate_5']:.3f}")
    print()

    # Best by median
    for policy in EXIT_POLICIES:
        sub = summary[summary["exit_policy"] == policy]
        print(f"  ─── [{policy}] Top 5 by MEDIAN return (n >= {MIN_N}) ──────────────────────")
        top = _top_n(sub, "median", n=5)
        if top.empty:
            print("    None with n >= 30.")
        else:
            print(f"  {'filter':<55} {'n':>5} {'avg':>7} {'med':>7} {'wr':>6} {'fail3':>6} {'fail5':>6} {'avg-x':>8}")
            print("  " + "─" * 105)
            for _, r in top.iterrows():
                print(
                    f"  {r['filter_name']:<55} {int(r['n']):>5} "
                    f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} "
                    f"{_fmtr(r['win_rate']):>6} {_fmtr(r['fail_rate_3']):>6} "
                    f"{_fmtr(r['fail_rate_5']):>6} {_fmt(r['avg_excl_best']):>8}"
                )
        print()

    # Best by avg_excl_best
    for policy in EXIT_POLICIES:
        sub = summary[summary["exit_policy"] == policy]
        print(f"  ─── [{policy}] Top 5 by AVG EXCL BEST TRADE (n >= {MIN_N}) ─────────────────")
        top = _top_n(sub, "avg_excl_best", n=5)
        if top.empty:
            print("    None with n >= 30.")
        else:
            print(f"  {'filter':<55} {'n':>5} {'avg-x':>8} {'med':>7} {'fail3':>6} {'fail5':>6}")
            print("  " + "─" * 90)
            for _, r in top.iterrows():
                print(
                    f"  {r['filter_name']:<55} {int(r['n']):>5} "
                    f"{_fmt(r['avg_excl_best']):>8} {_fmt(r['median']):>7} "
                    f"{_fmtr(r['fail_rate_3']):>6} {_fmtr(r['fail_rate_5']):>6}"
                )
        print()

    # Best by lowest fail_rate_3
    for policy in EXIT_POLICIES:
        sub = summary[summary["exit_policy"] == policy]
        print(f"  ─── [{policy}] Top 5 by LOWEST FAILURE RATE ≤ -3% (n >= {MIN_N}) ──────────")
        top = _top_n(sub, "fail_rate_3", n=5, ascending=True)
        if top.empty:
            print("    None with n >= 30.")
        else:
            print(f"  {'filter':<55} {'n':>5} {'fail3':>6} {'fail5':>6} {'med':>7} {'avg-x':>8}")
            print("  " + "─" * 90)
            for _, r in top.iterrows():
                print(
                    f"  {r['filter_name']:<55} {int(r['n']):>5} "
                    f"{_fmtr(r['fail_rate_3']):>6} {_fmtr(r['fail_rate_5']):>6} "
                    f"{_fmt(r['median']):>7} {_fmt(r['avg_excl_best']):>8}"
                )
        print()

    # Validation section
    print("  ─── Validation: filters that pass ALL criteria ──────────────────────────")
    passing = summary[summary["passes_validation"] == True]
    if passing.empty:
        print("  NO candidate filter passes all 5 validation criteria.")
    else:
        print(f"  {'filter':<55} {'policy':<16} {'n':>5} {'med':>7} {'avg-x':>8} {'fail3':>6}")
        print("  " + "─" * 100)
        for _, r in passing.iterrows():
            print(
                f"  {r['filter_name']:<55} {r['exit_policy']:<16} {int(r['n']):>5} "
                f"{_fmt(r['median']):>7} {_fmt(r['avg_excl_best']):>8} "
                f"{_fmtr(r['fail_rate_3']):>6}"
            )
    print()

    # Interpretation
    print("  ─── Interpretation ─────────────────────────────────────────────────────")

    # CC diagnostic
    cc_true = summary[(summary["filter_name"] == "cc_true") & (summary["exit_policy"] == "fixed_4h")]
    cc_false = summary[(summary["filter_name"] == "cc_false") & (summary["exit_policy"] == "fixed_4h")]
    baseline = summary[(summary["filter_name"] == "baseline_all") & (summary["exit_policy"] == "fixed_4h")]
    if not cc_true.empty and not baseline.empty:
        cc_med = float(cc_true.iloc[0]["median"])
        base_med = float(baseline.iloc[0]["median"])
        cc_fail = float(cc_true.iloc[0]["fail_rate_3"])
        base_fail = float(baseline.iloc[0]["fail_rate_3"])
        print(f"\n  is_clean_continuation:")
        print(f"    CC=True  → median={cc_med:+.2f}%  fail_rate_3={cc_fail:.3f}")
        if not cc_false.empty:
            cF_med = float(cc_false.iloc[0]["median"])
            cF_fail = float(cc_false.iloc[0]["fail_rate_3"])
            print(f"    CC=False → median={cF_med:+.2f}%  fail_rate_3={cF_fail:.3f}")
        print(f"    Baseline → median={base_med:+.2f}%  fail_rate_3={base_fail:.3f}")
        if cc_med > base_med + 0.1:
            print("    VERDICT: CC improves median → keep as entry filter.")
        elif cc_fail < base_fail - 0.01:
            print("    VERDICT: CC reduces failures → keep as protective filter.")
        else:
            print("    VERDICT: CC provides no measurable edge → DIAGNOSTIC ONLY or DROP.")

    # Volume climax
    vc = summary[(summary["filter_name"] == "volume_climax") & (summary["exit_policy"] == "fixed_4h")]
    if not vc.empty and not baseline.empty:
        vc_med = float(vc.iloc[0]["median"])
        vc_n = int(vc.iloc[0]["n"])
        base_med = float(baseline.iloc[0]["median"])
        print(f"\n  is_volume_climax:")
        print(f"    volume_climax=True → n={vc_n}  median={vc_med:+.2f}%")
        print(f"    Baseline           → median={base_med:+.2f}%")
        if vc_med > base_med + 0.1:
            print("    VERDICT: Volume climax improves median → PROMOTE TO ENTRY FILTER.")
        else:
            print("    VERDICT: Volume climax does not clearly improve median → test in combos only.")

    # Hour filtering
    hour_row = summary[(summary["filter_name"] == "excl_high_failure_hours") & (summary["exit_policy"] == "fixed_4h")]
    hour_safe = summary[(summary["filter_name"] == "hour_safe_17_18_19") & (summary["exit_policy"] == "fixed_4h")]
    if not hour_row.empty and not baseline.empty:
        h_fail3 = float(hour_row.iloc[0]["fail_rate_3"])
        h_med = float(hour_row.iloc[0]["median"])
        h_n = int(hour_row.iloc[0]["n"])
        b_fail3 = float(baseline.iloc[0]["fail_rate_3"])
        b_med = float(baseline.iloc[0]["median"])
        print(f"\n  Hour-of-day filtering:")
        print(f"    excl_high_failure_hours → n={h_n}  median={h_med:+.2f}%  fail_rate_3={h_fail3:.3f}")
        if not hour_safe.empty:
            hs_fail3 = float(hour_safe.iloc[0]["fail_rate_3"])
            hs_med = float(hour_safe.iloc[0]["median"])
            hs_n = int(hour_safe.iloc[0]["n"])
            print(f"    hour_safe_17_18_19      → n={hs_n}  median={hs_med:+.2f}%  fail_rate_3={hs_fail3:.3f}")
        print(f"    Baseline                → median={b_med:+.2f}%  fail_rate_3={b_fail3:.3f}")
        if h_fail3 < b_fail3 - 0.02:
            print("    VERDICT: Hour filtering MATERIALLY reduces failure rate.")
        elif h_med > b_med + 0.1:
            print("    VERDICT: Hour filtering improves median return.")
        else:
            print("    VERDICT: Hour filtering has limited impact on overall risk/return.")

    # Recommended rules
    print("\n  ─── Recommended Candidate Rules for Next Stage ─────────────────────────")
    best_by_metric: dict[str, pd.Series | None] = {
        "best_median": None,
        "best_avg_excl_best": None,
        "best_fail_rate": None,
    }
    sub_tp10 = summary[(summary["exit_policy"] == "tp10_else_4h") & (summary["n"] >= MIN_N)]
    if not sub_tp10.empty:
        bm = sub_tp10.loc[sub_tp10["median"].idxmax()]
        best_by_metric["best_median"] = bm
        bax = sub_tp10.dropna(subset=["avg_excl_best"]).loc[sub_tp10.dropna(subset=["avg_excl_best"])["avg_excl_best"].idxmax()]
        best_by_metric["best_avg_excl_best"] = bax
        bf = sub_tp10.loc[sub_tp10["fail_rate_3"].idxmin()]
        best_by_metric["best_fail_rate"] = bf

    seen: set[str] = set()
    counter = 1
    for label, row in best_by_metric.items():
        if row is None:
            continue
        fname = str(row["filter_name"])
        if fname in seen:
            continue
        seen.add(fname)
        print(f"  Rule {counter} [{label}]: {fname}")
        print(f"    exit=tp10_else_4h  n={int(row['n'])}  "
              f"median={row['median']:+.2f}%  avg-x={row['avg_excl_best']:+.2f}%  "
              f"fail3={row['fail_rate_3']:.3f}")
        counter += 1
        if counter > 3:
            break

    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    summary_path: Path = GRID_SUMMARY_PATH,
    event_log_path: Path = GRID_EVENT_LOG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 80)
    print("  Memecoin Backfill Candidate Filter Grid")
    print("=" * 80)

    df = load_events(input_path)
    print(f"\n  Loaded {len(df)} events from {input_path}")
    n_complete = df["future_ret_4h_pct"].notna().sum()
    print(f"  Completed 4h events: {n_complete}")

    summary, event_log = run_filter_grid(df)

    if not summary.empty:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\n  -> Summary written to {summary_path}  ({len(summary)} rows)")

    if not event_log.empty:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log.to_csv(event_log_path, index=False)
        print(f"  -> Event log written to {event_log_path}  ({len(event_log)} rows)")

    # Compute baseline stats for interpretation header
    baseline_stats = compute_stats(
        pd.to_numeric(df["future_ret_4h_pct"], errors="coerce").dropna()
    )
    print_results(summary, baseline_stats)

    return summary, event_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir
    main(
        input_path=args.input,
        summary_path=(out / GRID_SUMMARY_PATH.name) if out else GRID_SUMMARY_PATH,
        event_log_path=(out / GRID_EVENT_LOG_PATH.name) if out else GRID_EVENT_LOG_PATH,
    )
