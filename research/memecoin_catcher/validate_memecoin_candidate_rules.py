"""Robustness validation for top-3 memecoin candidate entry rules.

BACKFILL-ONLY VALIDATION:
- This module evaluates simulated/backfilled events only.
- It does not evaluate current genuine prospective evidence.
- It is not used for Gate A readiness.

Input:  data/memecoin_backfilled_signal_events.csv
Outputs:
  reports/memecoin_candidate_rule_validation_summary.csv
  reports/memecoin_candidate_rule_time_splits.csv
  reports/memecoin_candidate_rule_symbol_robustness.csv
  reports/memecoin_candidate_rule_fee_stress.csv
  reports/memecoin_candidate_rule_liquidity_diagnostics.csv

Research only. No trading. No orders. No live code touched.

Usage:
    .venv/bin/python -m research.memecoin_catcher.validate_memecoin_candidate_rules
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("data/memecoin_backfilled_signal_events.csv")
SUMMARY_PATH        = Path("reports/memecoin_candidate_rule_validation_summary.csv")
TIME_SPLITS_PATH    = Path("reports/memecoin_candidate_rule_time_splits.csv")
SYMBOL_ROBUST_PATH  = Path("reports/memecoin_candidate_rule_symbol_robustness.csv")
FEE_STRESS_PATH     = Path("reports/memecoin_candidate_rule_fee_stress.csv")
LIQUIDITY_PATH      = Path("reports/memecoin_candidate_rule_liquidity_diagnostics.csv")

SOURCE_DATASET_LABEL = "data/memecoin_backfilled_signal_events.csv"
SAMPLE_TYPE_LABEL = "SIMULATED_BACKFILL_ONLY"
SCOPE_NOTE = "BACKFILL-ONLY VALIDATION — not current genuine prospective evidence"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFE_HOURS: frozenset[int] = frozenset({17, 18, 19})
FEE_BPS: list[int] = [0, 25, 50, 100, 150, 200]
MIN_EVENTS_FOR_BUCKET: int = 5   # suppress buckets below this
TOP_PCT: float = 0.01

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "future_ret_4h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "max_favorable_24h_pct", "max_adverse_24h_pct",
        "volume_ratio_1h", "volume_ratio_4h",
        "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
        "primary_ohlc_score", "long_explosion_score",
        "spread_pct", "quote_volume_est", "entry_price",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "snapshot_ts_utc" in df.columns:
        df["_ts"] = pd.to_datetime(df["snapshot_ts_utc"], utc=True, errors="coerce")
        if "hour_of_day_utc" not in df.columns:
            df["hour_of_day_utc"] = df["_ts"].dt.hour
        df["date_utc"] = df["_ts"].dt.date
        df["day_index"] = (
            (df["_ts"] - df["_ts"].min()).dt.total_seconds() / 86400
        ).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Exit policy
# ---------------------------------------------------------------------------


def apply_tp10_else_4h(df: pd.DataFrame) -> pd.Series:
    """Vectorised tp10_else_4h exit. Returns realized return Series."""
    ret = pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")
    if "max_favorable_4h_pct" not in df.columns:
        return ret
    mfe = pd.to_numeric(df["max_favorable_4h_pct"], errors="coerce")
    return ret.where(mfe.isna() | (mfe < 10.0), 10.0)


def apply_fixed_4h(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")


# ---------------------------------------------------------------------------
# Candidate rule filters
# ---------------------------------------------------------------------------


def rule1_mask(df: pd.DataFrame) -> pd.Series:
    """Locked Rule 1: LONG_EXPLOSION AND VC=True AND CC=False."""
    mask = pd.Series([True] * len(df), index=df.index)
    if "ohlc_signal_type" in df.columns:
        mask &= df["ohlc_signal_type"].astype(str) == "LONG_EXPLOSION"
    else:
        logger.warning("ohlc_signal_type absent — Rule 1 will be empty.")
        return pd.Series([False] * len(df), index=df.index)
    if "is_volume_climax" in df.columns:
        mask &= df["is_volume_climax"].fillna(False).astype(bool)
    else:
        logger.warning("is_volume_climax absent — Rule 1 will be empty.")
        return pd.Series([False] * len(df), index=df.index)
    if "is_clean_continuation" in df.columns:
        mask &= ~df["is_clean_continuation"].fillna(False).astype(bool)
    else:
        logger.warning("is_clean_continuation absent — Rule 1 CC=False condition skipped.")
    return mask


def rule2_mask(df: pd.DataFrame) -> pd.Series:
    """volume_climax_safe_hours: VC=True AND hour in {17,18,19}."""
    mask = pd.Series([True] * len(df), index=df.index)
    if "is_volume_climax" in df.columns:
        mask &= df["is_volume_climax"].fillna(False).astype(bool)
    else:
        logger.warning("is_volume_climax absent — Rule 2 will be empty.")
        return pd.Series([False] * len(df), index=df.index)
    if "hour_of_day_utc" in df.columns:
        mask &= df["hour_of_day_utc"].isin(SAFE_HOURS)
    else:
        logger.warning("hour_of_day_utc absent — Rule 2 hour condition skipped.")
    return mask


def rule3_mask(df: pd.DataFrame) -> pd.Series:
    """volume_ratio_safe_hours_not_extreme: VR1h>=5 AND hour in {17,18,19} AND ret24h<25."""
    mask = pd.Series([True] * len(df), index=df.index)
    if "volume_ratio_1h" in df.columns:
        mask &= df["volume_ratio_1h"].fillna(0) >= 5.0
    else:
        logger.warning("volume_ratio_1h absent — Rule 3 VR condition skipped.")
    if "hour_of_day_utc" in df.columns:
        mask &= df["hour_of_day_utc"].isin(SAFE_HOURS)
    else:
        logger.warning("hour_of_day_utc absent — Rule 3 hour condition skipped.")
    if "ret_24h_pct" in df.columns:
        mask &= df["ret_24h_pct"].fillna(999) < 25.0
    else:
        logger.warning("ret_24h_pct absent — Rule 3 overextension condition skipped.")
    return mask


CANDIDATE_RULES: dict[str, tuple[Callable, Callable]] = {
    "rule1_volume_climax_cc_false_tp10":            (rule1_mask, apply_tp10_else_4h),
    "rule2_volume_climax_safe_hours_tp10":          (rule2_mask, apply_tp10_else_4h),
    "rule3_vr5_safe_hours_ret24lt25_tp10":          (rule3_mask, apply_tp10_else_4h),
}

BASELINE_RULES: dict[str, tuple[Callable, Callable]] = {
    "baseline_all_fixed_4h":            (lambda df: pd.Series([True] * len(df), index=df.index), apply_fixed_4h),
    "baseline_all_tp10":                (lambda df: pd.Series([True] * len(df), index=df.index), apply_tp10_else_4h),
    "baseline_volume_climax_tp10":      (lambda df: df["is_volume_climax"].fillna(False).astype(bool) if "is_volume_climax" in df.columns else pd.Series([False] * len(df), index=df.index), apply_tp10_else_4h),
    "baseline_safe_hours_tp10":         (lambda df: df["hour_of_day_utc"].isin(SAFE_HOURS) if "hour_of_day_utc" in df.columns else pd.Series([False] * len(df), index=df.index), apply_tp10_else_4h),
}

ALL_RULES: dict[str, tuple[Callable, Callable]] = {**CANDIDATE_RULES, **BASELINE_RULES}

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _s(v: float, d: int = 4) -> float:
    return round(float(v), d) if not math.isnan(float(v)) else float("nan")


def compute_stats(returns: pd.Series, label: str = "") -> dict:
    s = returns.dropna().astype(float)
    n = len(s)
    if n == 0:
        nan = float("nan")
        return dict(label=label, n=0, avg=nan, median=nan, win_rate=nan,
                    big_win_rate=nan, fail_rate_3=nan, fail_rate_5=nan,
                    worst=nan, best=nan,
                    avg_excl_best=nan, median_excl_best=nan,
                    avg_excl_top1pct=nan, median_excl_top1pct=nan)

    excl_best = s.drop(index=s.idxmax())
    cutoff = float(s.quantile(1.0 - TOP_PCT))
    excl_top1 = s[s < cutoff]

    return dict(
        label=label, n=n,
        avg=_s(s.mean()), median=_s(s.median()),
        win_rate=_s((s > 0).mean()),
        big_win_rate=_s((s >= 5.0).mean()),
        fail_rate_3=_s((s <= -3.0).mean()),
        fail_rate_5=_s((s <= -5.0).mean()),
        worst=_s(s.min()), best=_s(s.max()),
        avg_excl_best=_s(excl_best.mean()) if len(excl_best) else float("nan"),
        median_excl_best=_s(excl_best.median()) if len(excl_best) else float("nan"),
        avg_excl_top1pct=_s(excl_top1.mean()) if len(excl_top1) else float("nan"),
        median_excl_top1pct=_s(excl_top1.median()) if len(excl_top1) else float("nan"),
    )


# ---------------------------------------------------------------------------
# 1. Full-sample validation summary
# ---------------------------------------------------------------------------


def build_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.dropna(subset=["future_ret_4h_pct"])
    rows: list[dict] = []
    for rule_name, (mask_fn, exit_fn) in ALL_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask]
        rets = exit_fn(sub)
        stats = compute_stats(rets, label=rule_name)
        stats["rule"] = rule_name
        stats["is_candidate"] = rule_name in CANDIDATE_RULES
        rows.append(stats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Time-split validation
# ---------------------------------------------------------------------------


def build_time_splits(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.dropna(subset=["future_ret_4h_pct"]).copy()
    if "day_index" not in completed.columns:
        logger.warning("day_index absent — time splits unavailable.")
        return pd.DataFrame()

    max_day = int(completed["day_index"].max())
    rows: list[dict] = []

    def _add(rule_name: str, rets: pd.Series, split_type: str, split_label: str) -> None:
        s = rets.dropna()
        if len(s) < MIN_EVENTS_FOR_BUCKET:
            return
        rows.append(dict(
            rule=rule_name,
            split_type=split_type,
            split_label=split_label,
            n=len(s),
            avg=_s(s.mean()),
            median=_s(s.median()),
            fail_rate_3=_s((s <= -3.0).mean()),
            fail_rate_5=_s((s <= -5.0).mean()),
            win_rate=_s((s > 0).mean()),
        ))

    for rule_name, (mask_fn, exit_fn) in CANDIDATE_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask].copy()
        rets_all = exit_fn(sub)

        # First half / second half
        mid = max_day // 2
        first_mask = sub["day_index"] <= mid
        second_mask = sub["day_index"] > mid
        _add(rule_name, rets_all[first_mask], "half", f"first_half (days 0-{mid})")
        _add(rule_name, rets_all[second_mask], "half", f"second_half (days {mid+1}-{max_day})")

        # Daily buckets
        for d in range(max_day + 1):
            day_mask = sub["day_index"] == d
            date_label = sub.loc[day_mask, "date_utc"].iloc[0] if day_mask.any() else f"day_{d}"
            _add(rule_name, rets_all[day_mask], "daily", str(date_label))

        # Rolling 2-day buckets
        for d in range(0, max_day, 2):
            roll_mask = sub["day_index"].isin([d, d + 1])
            date_label = f"days_{d}_{d+1}"
            _add(rule_name, rets_all[roll_mask], "rolling_2day", date_label)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Symbol robustness
# ---------------------------------------------------------------------------


def build_symbol_robustness(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.dropna(subset=["future_ret_4h_pct"])
    sym_col = "symbol" if "symbol" in completed.columns else "pair_id"
    rows: list[dict] = []

    for rule_name, (mask_fn, exit_fn) in CANDIDATE_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask].copy()
        rets_all = exit_fn(sub).dropna()
        n_total = len(rets_all)
        if n_total == 0:
            continue

        # Per-symbol contribution
        sub = sub.loc[rets_all.index]
        sub = sub.copy()
        sub["_ret"] = rets_all

        sym_stats = (
            sub.groupby(sym_col)["_ret"]
            .agg(n="count", avg="mean", total="sum")
            .reset_index()
            .rename(columns={sym_col: "symbol"})
        )
        sym_stats["pct_of_events"] = sym_stats["n"] / n_total
        sym_stats["contribution"] = sym_stats["total"] / n_total  # avg impact on overall avg
        sym_stats = sym_stats.sort_values("contribution", ascending=False).reset_index(drop=True)

        for _, r in sym_stats.iterrows():
            rows.append(dict(
                rule=rule_name,
                symbol=r["symbol"],
                n=int(r["n"]),
                sym_avg=_s(float(r["avg"])),
                sym_total=_s(float(r["total"])),
                pct_of_events=_s(float(r["pct_of_events"])),
                contribution_to_avg=_s(float(r["contribution"])),
            ))

        # Leave-one-symbol-out
        for sym in sym_stats["symbol"].tolist():
            excl_mask = sub[sym_col] != sym
            excl_rets = sub.loc[excl_mask, "_ret"].dropna()
            s = excl_rets
            if len(s) == 0:
                continue
            rows.append(dict(
                rule=rule_name,
                symbol=f"EXCL_{sym}",
                n=len(s),
                sym_avg=_s(s.mean()),
                sym_total=_s(s.sum()),
                pct_of_events=_s(len(s) / n_total),
                contribution_to_avg=float("nan"),
            ))

        # Exclude top 3 contributors
        top3 = sym_stats.head(3)["symbol"].tolist()
        excl3_mask = ~sub[sym_col].isin(top3)
        excl3_rets = sub.loc[excl3_mask, "_ret"].dropna()
        if len(excl3_rets):
            rows.append(dict(
                rule=rule_name,
                symbol="EXCL_TOP3",
                n=len(excl3_rets),
                sym_avg=_s(float(excl3_rets.mean())),
                sym_total=_s(float(excl3_rets.sum())),
                pct_of_events=_s(len(excl3_rets) / n_total),
                contribution_to_avg=float("nan"),
            ))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Hour robustness (per-hour breakdown for each rule)
# ---------------------------------------------------------------------------


def build_hour_robustness(df: pd.DataFrame) -> list[dict]:
    """Returns rows suitable for embedding in the summary report."""
    completed = df.dropna(subset=["future_ret_4h_pct"])
    if "hour_of_day_utc" not in completed.columns:
        return []
    rows = []
    for rule_name, (mask_fn, exit_fn) in CANDIDATE_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask].copy()
        rets_all = exit_fn(sub)
        for h in range(24):
            h_mask = sub["hour_of_day_utc"] == h
            s = rets_all[h_mask].dropna()
            if len(s) == 0:
                continue
            rows.append(dict(
                rule=rule_name,
                hour_utc=h,
                n=len(s),
                avg=_s(s.mean()),
                median=_s(s.median()),
                fail_rate_3=_s((s <= -3.0).mean()),
                win_rate=_s((s > 0).mean()),
            ))
    return rows


# ---------------------------------------------------------------------------
# 5. Fee / slippage stress test
# ---------------------------------------------------------------------------


def apply_fee(returns: pd.Series, bps: int) -> pd.Series:
    """Subtract a round-trip fee in percent (bps / 100 / 100 * 100 = bps/100 pct)."""
    fee_pct = bps / 100.0
    return returns - fee_pct


def build_fee_stress(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.dropna(subset=["future_ret_4h_pct"])
    rows: list[dict] = []
    for rule_name, (mask_fn, exit_fn) in CANDIDATE_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask]
        gross_rets = exit_fn(sub).dropna()
        for bps in FEE_BPS:
            net = apply_fee(gross_rets, bps)
            s = net.dropna()
            if len(s) == 0:
                continue
            rows.append(dict(
                rule=rule_name,
                fee_bps=bps,
                n=len(s),
                avg_net=_s(s.mean()),
                median_net=_s(s.median()),
                win_rate=_s((s > 0).mean()),
                fail_rate_3=_s((s <= -3.0).mean()),
            ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. Liquidity diagnostics
# ---------------------------------------------------------------------------


def build_liquidity_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.dropna(subset=["future_ret_4h_pct"])
    rows: list[dict] = []

    for rule_name, (mask_fn, _) in CANDIDATE_RULES.items():
        mask = mask_fn(completed).fillna(False).astype(bool)
        sub = completed[mask]
        n = len(sub)
        row: dict = dict(rule=rule_name, n=n)

        for col, label in [("spread_pct", "spread"), ("quote_volume_est", "qvol")]:
            s = pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            if len(s) == 0:
                row[f"{label}_available"] = False
                row[f"{label}_median"] = float("nan")
                row[f"{label}_worst"] = float("nan")
                row[f"{label}_pct_gt_1pct"] = float("nan") if col == "spread_pct" else float("nan")
            else:
                row[f"{label}_available"] = True
                if col == "spread_pct":
                    row[f"{label}_median"] = _s(s.median())
                    row[f"{label}_worst"] = _s(s.max())
                    row[f"{label}_pct_gt_1pct"] = _s((s > 1.0).mean())
                else:
                    row[f"{label}_median"] = _s(s.median())
                    row[f"{label}_min"] = _s(s.min())
                    row[f"{label}_pct_lt_10k"] = _s((s < 10_000).mean())
                    row[f"{label}_pct_lt_25k"] = _s((s < 25_000).mean())
                    row[f"{label}_pct_lt_50k"] = _s((s < 50_000).mean())

        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _fmt(v, d: int = 2) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  —  "
    return f"{float(v):+.{d}f}%"


def _fmtn(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return " — "
    return f"{float(v):.3f}"


def print_results(
    summary: pd.DataFrame,
    time_splits: pd.DataFrame,
    sym_robust: pd.DataFrame,
    fee_stress: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("  MEMECOIN CANDIDATE RULE VALIDATION — RESULTS")
    print("=" * 80)
    print("  BACKFILL-ONLY VALIDATION — not current genuine prospective evidence.")
    print("  Source dataset: data/memecoin_backfilled_signal_events.csv")
    print("  Not used for Gate A readiness.")
    print("  All n/backfill_n counts below are backfill-only sample counts.")
    print("  *** Research only. Not a validated trading strategy. ***")

    # Full-sample summary
    print("\n  ─── Full-Sample Metrics ─────────────────────────────────────────────")
    print(f"  {'rule':<47} {'backfill_n':>10} {'avg':>7} {'med':>7} {'wr':>6} {'bwr':>6} {'f3':>6} {'f5':>6} {'avg-x':>8} {'avg-t1':>8}")
    print("  " + "─" * 108)
    for _, r in summary.iterrows():
        marker = "★" if r.get("is_candidate") else " "
        print(
            f"  {marker}{r['rule']:<46} {int(r['n']):>10} "
            f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} "
            f"{_fmtn(r['win_rate']):>6} {_fmtn(r['big_win_rate']):>6} "
            f"{_fmtn(r['fail_rate_3']):>6} {_fmtn(r['fail_rate_5']):>6} "
            f"{_fmt(r['avg_excl_best']):>8} {_fmt(r['avg_excl_top1pct']):>8}"
        )

    # Time-split flags
    if not time_splits.empty:
        print("\n  ─── Time-Split Flags ────────────────────────────────────────────────")
        halves = time_splits[time_splits["split_type"] == "half"]
        if not halves.empty:
            print(f"  {'rule':<47} {'split':<28} {'backfill_n':>10} {'avg':>7} {'med':>7} {'f3':>6}")
            print("  " + "─" * 100)
            for _, r in halves.iterrows():
                print(
                    f"  {r['rule']:<47} {str(r['split_label']):<28} {int(r['n']):>10} "
                    f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} {_fmtn(r['fail_rate_3']):>6}"
                )

        # Flag rules that only work in one half
        print()
        for rule in CANDIDATE_RULES:
            rule_halves = halves[halves["rule"] == rule]
            if len(rule_halves) == 2:
                m1 = float(rule_halves.iloc[0]["median"])
                m2 = float(rule_halves.iloc[1]["median"])
                if m1 > 0 and m2 <= 0:
                    print(f"  ⚠  {rule}: positive median only in FIRST half.")
                elif m2 > 0 and m1 <= 0:
                    print(f"  ⚠  {rule}: positive median only in SECOND half.")
                elif m1 > 0 and m2 > 0:
                    print(f"  ✓  {rule}: positive median in BOTH halves.")
                else:
                    print(f"  ✗  {rule}: negative median in both halves.")

    # Symbol robustness
    if not sym_robust.empty:
        print("\n  ─── Symbol Robustness ───────────────────────────────────────────────")
        for rule in CANDIDATE_RULES:
            sub = sym_robust[sym_robust["rule"] == rule]
            actual_syms = sub[~sub["symbol"].str.startswith("EXCL_")]
            excl_rows = sub[sub["symbol"].str.startswith("EXCL_")]
            print(f"\n  {rule}:")
            print(f"    Top contributing symbols:")
            top_syms = actual_syms.sort_values("contribution_to_avg", ascending=False).head(5)
            for _, r in top_syms.iterrows():
                print(f"      {str(r['symbol']):<20} backfill_n={int(r['n']):>3}  avg={r['sym_avg']:+.2f}%  contrib={r['contribution_to_avg']:+.3f}%")

            # leave-one-out
            full_row = summary[summary["rule"] == rule]
            if not full_row.empty:
                full_med = float(full_row.iloc[0]["median"])
                full_avg = float(full_row.iloc[0]["avg"])
                print(f"    Leave-one-symbol-out (median impact):")
                loso = excl_rows[excl_rows["symbol"].str.startswith("EXCL_") & ~excl_rows["symbol"].str.startswith("EXCL_TOP")]
                for _, r in loso.sort_values("sym_avg").head(5).iterrows():
                    sym_name = str(r["symbol"]).replace("EXCL_", "")
                    delta = float(r["sym_avg"]) - full_avg
                    print(f"      excl {sym_name:<18} backfill_n={int(r['n']):>3}  avg={r['sym_avg']:+.2f}%  Δ={delta:+.2f}%")

            top3_row = excl_rows[excl_rows["symbol"] == "EXCL_TOP3"]
            if not top3_row.empty:
                print(f"    Excl top-3 symbols: backfill_n={int(top3_row.iloc[0]['n'])}  avg={top3_row.iloc[0]['sym_avg']:+.2f}%")

    # Fee stress
    if not fee_stress.empty:
        print("\n  ─── Fee/Slippage Stress Test ────────────────────────────────────────")
        print(f"  {'rule':<47} {'bps':>5} {'backfill_n':>10} {'avg_net':>9} {'med_net':>9} {'wr':>6} {'f3':>6}")
        print("  " + "─" * 90)
        for _, r in fee_stress.iterrows():
            print(
                f"  {r['rule']:<47} {int(r['fee_bps']):>5} {int(r['n']):>10} "
                f"{_fmt(r['avg_net']):>9} {_fmt(r['median_net']):>9} "
                f"{_fmtn(r['win_rate']):>6} {_fmtn(r['fail_rate_3']):>6}"
            )

    # Final interpretation
    print("\n  ─── Interpretation ─────────────────────────────────────────────────────")

    cand_rows = summary[summary["is_candidate"] == True]

    # Best median
    if not cand_rows.empty:
        best_med = cand_rows.loc[cand_rows["median"].idxmax()]
        print(f"\n  Best median return:  {best_med['rule']}  ({best_med['median']:+.2f}%)")

        best_ax = cand_rows.dropna(subset=["avg_excl_top1pct"])
        if not best_ax.empty:
            bax = best_ax.loc[best_ax["avg_excl_top1pct"].idxmax()]
            print(f"  Best avg excl top-1%: {bax['rule']}  ({bax['avg_excl_top1pct']:+.2f}%)")

        best_f3 = cand_rows.loc[cand_rows["fail_rate_3"].idxmin()]
        print(f"  Lowest fail_rate_3:  {best_f3['rule']}  ({best_f3['fail_rate_3']:.3f})")

    # Fee threshold for 100bps
    if not fee_stress.empty:
        fee100 = fee_stress[fee_stress["fee_bps"] == 100]
        pos_at_100 = fee100[fee100["avg_net"] > 0]
        if pos_at_100.empty:
            print("\n  No candidate rule has positive average net return at 100 bps round-trip.")
        else:
            print(f"\n  Positive average at 100 bps: " + ", ".join(pos_at_100["rule"].tolist()))

    # Symbol dependence
    if not sym_robust.empty:
        print("\n  Symbol concentration check:")
        for rule in CANDIDATE_RULES:
            sub = sym_robust[(sym_robust["rule"] == rule) & ~sym_robust["symbol"].str.startswith("EXCL_")]
            if sub.empty:
                continue
            top1 = sub.iloc[0]
            top1_pct = float(top1["pct_of_events"])
            top1_sym = top1["symbol"]
            full_avg = float(summary[summary["rule"] == rule]["avg"].iloc[0]) if not summary[summary["rule"] == rule].empty else float("nan")
            excl1 = sym_robust[(sym_robust["rule"] == rule) & (sym_robust["symbol"] == f"EXCL_{top1_sym}")]
            excl1_avg = float(excl1.iloc[0]["sym_avg"]) if not excl1.empty else float("nan")
            if top1_pct > 0.25:
                print(f"  ⚠  {rule}: top symbol {top1_sym} is {top1_pct:.0%} of events.")
            elif not math.isnan(full_avg) and not math.isnan(excl1_avg) and abs(excl1_avg - full_avg) > 1.0:
                print(f"  ⚠  {rule}: excluding {top1_sym} changes avg by {excl1_avg - full_avg:+.2f}%.")
            else:
                print(f"  ✓  {rule}: no single-symbol concentration (top={top1_sym} {top1_pct:.0%}).")

    # Hour dependence for safe-hour rules
    hour_rows = build_hour_robustness(load_events())  # cheap second call for display
    if hour_rows:
        hr_df = pd.DataFrame(hour_rows)
        print("\n  Hour concentration check (rules 2 & 3):")
        for rule in ["rule2_volume_climax_safe_hours_tp10", "rule3_vr5_safe_hours_ret24lt25_tp10"]:
            sub = hr_df[hr_df["rule"] == rule]
            safe = sub[sub["hour_utc"].isin(SAFE_HOURS)]
            if safe.empty:
                continue
            pos_hours = safe[safe["median"] > 0]["hour_utc"].tolist()
            neg_hours = safe[safe["median"] <= 0]["hour_utc"].tolist()
            if len(pos_hours) == 1:
                print(f"  ⚠  {rule}: only hour {pos_hours[0]} UTC has positive median.")
            elif len(pos_hours) == len(safe):
                print(f"  ✓  {rule}: all safe hours positive median {sorted(pos_hours)}.")
            else:
                print(f"  ~  {rule}: mixed — positive hours={pos_hours}, negative={neg_hours}.")

    # CC verdict
    print()
    cc_true_row = summary[summary["rule"] == "baseline_all_fixed_4h"]
    r1_row = summary[summary["rule"] == "rule1_volume_climax_cc_false_tp10"]
    if not r1_row.empty:
        print(f"  CC verdict: Rule 1 (CC=False filter) has median={float(r1_row.iloc[0]['median']):+.2f}%  "
              f"fail_rate_3={float(r1_row.iloc[0]['fail_rate_3']):.3f}.")
        print("  → is_clean_continuation should be REMOVED from entry logic; CC=False may be helpful.")

    # Overall recommendation
    print()
    any_positive_med = not cand_rows[cand_rows["median"] > 0].empty if not cand_rows.empty else False
    any_positive_ax = not cand_rows[(cand_rows.get("avg_excl_best", pd.Series()) > 0).fillna(False)].empty if not cand_rows.empty else False
    if any_positive_med or any_positive_ax:
        print("  EDGE VERDICT: Sufficient signal to continue researching.")
        print("  Suggested next step: re-fetch 14–30 days of OHLC data for wider validation.")
    else:
        print("  EDGE VERDICT: No candidate shows positive median or avg-excl-best at this sample size.")
        print("  Consider extending the backfill window before drawing strong conclusions.")

    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    time_splits_path: Path = TIME_SPLITS_PATH,
    symbol_robust_path: Path = SYMBOL_ROBUST_PATH,
    fee_stress_path: Path = FEE_STRESS_PATH,
    liquidity_path: Path = LIQUIDITY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 80)
    print("  Memecoin Candidate Rule Robustness Validation")
    print("=" * 80)
    print("  BACKFILL-ONLY VALIDATION — not current genuine prospective evidence")
    print("  Not used for Gate A readiness")

    df = load_events(input_path)
    n_complete = df["future_ret_4h_pct"].notna().sum()
    print(f"\n  Loaded {len(df)} events from {input_path}")
    print(f"  Completed 4h outcomes: {n_complete}")

    def _attach_backfill_provenance(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "n" in data.columns and "backfill_n" not in data.columns:
            data["backfill_n"] = data["n"]
        data["source_dataset_path"] = str(input_path)
        data["sample_type"] = SAMPLE_TYPE_LABEL
        data["scope_label"] = SCOPE_NOTE
        data["used_for_gate_a_readiness"] = False
        data["is_current_genuine_evidence"] = False
        return data

    summary       = _attach_backfill_provenance(build_validation_summary(df))
    time_splits   = _attach_backfill_provenance(build_time_splits(df))
    sym_robust    = _attach_backfill_provenance(build_symbol_robustness(df))
    fee_stress    = _attach_backfill_provenance(build_fee_stress(df))
    liquidity     = _attach_backfill_provenance(build_liquidity_diagnostics(df))

    for path, data, label in [
        (summary_path,       summary,     "Validation summary"),
        (time_splits_path,   time_splits, "Time splits"),
        (symbol_robust_path, sym_robust,  "Symbol robustness"),
        (fee_stress_path,    fee_stress,  "Fee stress"),
        (liquidity_path,     liquidity,   "Liquidity diagnostics"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"  -> {label} written to {path}  ({len(data)} rows)")

    print_results(summary, time_splits, sym_robust, fee_stress)

    return summary, time_splits, sym_robust, fee_stress, liquidity


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir
    main(
        input_path=args.input,
        summary_path=(out / SUMMARY_PATH.name) if out else SUMMARY_PATH,
        time_splits_path=(out / TIME_SPLITS_PATH.name) if out else TIME_SPLITS_PATH,
        symbol_robust_path=(out / SYMBOL_ROBUST_PATH.name) if out else SYMBOL_ROBUST_PATH,
        fee_stress_path=(out / FEE_STRESS_PATH.name) if out else FEE_STRESS_PATH,
        liquidity_path=(out / LIQUIDITY_PATH.name) if out else LIQUIDITY_PATH,
    )
