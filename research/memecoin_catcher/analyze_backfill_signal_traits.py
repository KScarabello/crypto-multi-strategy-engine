"""Trait discovery analysis on the memecoin backfill event log.

Reads data/memecoin_backfilled_signal_events.csv and produces:
  - reports/memecoin_backfill_trait_group_summary.csv
  - reports/memecoin_backfill_trait_threshold_sweep.csv
  - reports/memecoin_backfill_symbol_summary.csv
  - reports/memecoin_backfill_hour_of_day_summary.csv

Lookahead-safety rule (enforced by OUTCOME_FIELDS constant):
  future_ret_*_pct, max_favorable_*_pct, and max_adverse_*_pct are
  outcome/evaluation fields. They are summarised per group but are never
  used as entry filter definitions in the threshold sweep.

Research only. No trading. No orders. No live code touched.

Usage:
    .venv/bin/python -m research.memecoin_catcher.analyze_backfill_signal_traits
    .venv/bin/python -m research.memecoin_catcher.analyze_backfill_signal_traits \\
        --input data/memecoin_backfilled_signal_events.csv \\
        --min-group-size 30
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("data/memecoin_backfilled_signal_events.csv")
GROUP_SUMMARY_PATH = Path("reports/memecoin_backfill_trait_group_summary.csv")
SWEEP_SUMMARY_PATH = Path("reports/memecoin_backfill_trait_threshold_sweep.csv")
SYMBOL_SUMMARY_PATH = Path("reports/memecoin_backfill_symbol_summary.csv")
HOUR_SUMMARY_PATH = Path("reports/memecoin_backfill_hour_of_day_summary.csv")

# ---------------------------------------------------------------------------
# Fields that must NOT be used as entry filter conditions (lookahead risk).
# ---------------------------------------------------------------------------

OUTCOME_FIELDS: frozenset[str] = frozenset({
    "future_ret_15m_pct",
    "future_ret_1h_pct",
    "future_ret_2h_pct",
    "future_ret_4h_pct",
    "future_ret_8h_pct",
    "future_ret_12h_pct",
    "future_ret_24h_pct",
    "max_favorable_4h_pct",
    "max_adverse_4h_pct",
    "max_favorable_24h_pct",
    "max_adverse_24h_pct",
    "outcome_15m",
    "outcome_1h",
    "outcome_4h",
    "outcome_24h",
})

# ---------------------------------------------------------------------------
# Feature columns to summarise per group
# ---------------------------------------------------------------------------

NUMERIC_TRAIT_COLS: list[str] = [
    "ret_15m_pct",
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
    "spread_pct",
    "primary_ohlc_score",
    "long_explosion_score",
    "quote_volume_est",       # may be absent from backfill
    "max_favorable_4h_pct",   # outcome/eval only — included in group summary, not sweep filters
    "max_adverse_4h_pct",
]

BOOL_TRAIT_COLS: list[str] = [
    "is_clean_continuation",
    "is_volume_climax",
    "is_overextended_24h",
    "is_rolling_over",
    "is_wide_spread",
    "danger_terminal_spike",
    "breakout_24h",
]

# Numeric traits eligible for threshold sweep (no outcome fields).
SWEEP_TRAIT_COLS: list[str] = [
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
    "spread_pct",
    "primary_ohlc_score",
    "long_explosion_score",
    "quote_volume_est",
]

# Minimum events required for a threshold-sweep row to appear in the output.
DEFAULT_MIN_GROUP_SIZE: int = 30


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the backfill event log and coerce numeric columns."""
    df = pd.read_csv(path)
    numeric_cols = (
        NUMERIC_TRAIT_COLS
        + ["future_ret_4h_pct", "future_ret_24h_pct",
           "future_ret_1h_pct", "future_ret_15m_pct",
           "max_favorable_4h_pct", "max_adverse_4h_pct"]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "snapshot_ts_utc" in df.columns:
        df["_ts"] = pd.to_datetime(df["snapshot_ts_utc"], utc=True, errors="coerce")
        df["hour_of_day_utc"] = df["_ts"].dt.hour
        df = df.drop(columns=["_ts"])
    return df


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

GROUP_DEFINITIONS: list[tuple[str, object]] = [
    ("all_events",         None),
    ("big_winners_4h",     lambda df: df["future_ret_4h_pct"] >= 5.0),
    ("winners_4h",         lambda df: df["future_ret_4h_pct"] > 0.0),
    ("losers_4h",          lambda df: df["future_ret_4h_pct"] <= 0.0),
    ("failures_4h",        lambda df: df["future_ret_4h_pct"] <= -3.0),
    ("severe_failures_4h", lambda df: df["future_ret_4h_pct"] <= -5.0),
]

GROUP_NAMES: list[str] = [name for name, _ in GROUP_DEFINITIONS]


def assign_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {group_name: filtered_df} for all defined groups."""
    completed = df.dropna(subset=["future_ret_4h_pct"])
    result: dict[str, pd.DataFrame] = {}
    for name, condition in GROUP_DEFINITIONS:
        if condition is None:
            result[name] = completed.copy()
        else:
            mask = condition(completed)
            result[name] = completed[mask].copy()
    return result


# ---------------------------------------------------------------------------
# Return statistics helper
# ---------------------------------------------------------------------------


def _safe(v: float, d: int = 4) -> float:
    return round(v, d) if not math.isnan(v) else float("nan")


def _ret_stats(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {
            "n": 0, "avg": float("nan"), "median": float("nan"),
            "win_rate": float("nan"), "big_win_rate": float("nan"),
            "fail_rate_3": float("nan"), "fail_rate_5": float("nan"),
            "worst": float("nan"), "best": float("nan"),
            "avg_excl_best": float("nan"), "median_excl_best": float("nan"),
        }
    excl = s.drop(index=s.idxmax())
    return {
        "n": n,
        "avg": _safe(s.mean()),
        "median": _safe(s.median()),
        "win_rate": _safe((s > 0).mean()),
        "big_win_rate": _safe((s >= 5.0).mean()),
        "fail_rate_3": _safe((s <= -3.0).mean()),
        "fail_rate_5": _safe((s <= -5.0).mean()),
        "worst": _safe(s.min()),
        "best": _safe(s.max()),
        "avg_excl_best": _safe(excl.mean()) if len(excl) else float("nan"),
        "median_excl_best": _safe(excl.median()) if len(excl) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Group summary
# ---------------------------------------------------------------------------


def build_group_summary(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per group: numeric trait stats + bool shares + return stats."""
    records: list[dict] = []

    for group_name, df in groups.items():
        row: dict = {"group": group_name}

        # Return stats
        stats = _ret_stats(df.get("future_ret_4h_pct", pd.Series(dtype=float)))
        row.update({f"ret4h_{k}": v for k, v in stats.items()})

        # Numeric traits (avg + median)
        for col in NUMERIC_TRAIT_COLS:
            if col not in df.columns:
                row[f"{col}_avg"] = float("nan")
                row[f"{col}_median"] = float("nan")
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"{col}_avg"] = _safe(s.mean()) if len(s) else float("nan")
            row[f"{col}_median"] = _safe(s.median()) if len(s) else float("nan")

        # Boolean trait shares
        for col in BOOL_TRAIT_COLS:
            if col not in df.columns:
                row[f"share_{col}"] = float("nan")
                continue
            valid = df[col].dropna()
            if len(valid) == 0:
                row[f"share_{col}"] = float("nan")
            else:
                row[f"share_{col}"] = _safe(valid.astype(bool).mean())

        # Signal type distribution
        if "ohlc_signal_type" in df.columns:
            counts = df["ohlc_signal_type"].value_counts(normalize=True)
            for sig in ("LONG_EXPLOSION", "REVERSAL_WATCH", "DUMPING"):
                row[f"share_signal_{sig}"] = _safe(float(counts.get(sig, 0.0)))

        # Scanner label
        if "scanner_label" in df.columns:
            label_counts = df["scanner_label"].value_counts(normalize=True)
            for lbl in ("HOT_MOVER", "WATCH"):
                row[f"share_label_{lbl}"] = _safe(float(label_counts.get(lbl, 0.0)))

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Boolean share table (long-form, easier to read)
# ---------------------------------------------------------------------------


def build_bool_share_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-form table: group × trait → share (True proportion)."""
    records: list[dict] = []
    for group_name, df in groups.items():
        for col in BOOL_TRAIT_COLS:
            if col not in df.columns:
                share = float("nan")
            else:
                valid = df[col].dropna()
                share = _safe(valid.astype(bool).mean()) if len(valid) else float("nan")
            records.append({"group": group_name, "trait": col, "share": share, "n": len(df)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Threshold sweep (lookahead-safe)
# ---------------------------------------------------------------------------


def _sweep_thresholds_for_col(
    df: pd.DataFrame,
    col: str,
    min_group_size: int,
) -> list[dict]:
    """
    Test a set of one-sided and bounded filters on *col* and return stats rows.

    Only uses candle-time features (col is guaranteed not in OUTCOME_FIELDS
    before this function is called).
    """
    assert col not in OUTCOME_FIELDS, f"Lookahead bias: {col!r} is an outcome field."

    s = pd.to_numeric(df[col], errors="coerce")
    non_null = s.dropna()
    if len(non_null) < min_group_size:
        logger.info("Skipping sweep for %r: only %d non-null values.", col, len(non_null))
        return []

    rows: list[dict] = []

    def _add(mask: pd.Series, description: str) -> None:
        sub = df[mask & df["future_ret_4h_pct"].notna()]
        if len(sub) < min_group_size:
            return
        stats = _ret_stats(sub["future_ret_4h_pct"])
        rows.append({
            "trait": col,
            "filter": description,
            **stats,
        })

    # Compute quantile thresholds from non-null values
    q25, q50, q75 = (
        float(non_null.quantile(0.25)),
        float(non_null.quantile(0.50)),
        float(non_null.quantile(0.75)),
    )

    # One-sided: >= quantile thresholds
    for q_label, q_val in [("p25", q25), ("p50", q50), ("p75", q75)]:
        _add(s >= q_val, f"{col} >= {q_val:.3f} (>{q_label})")
        _add(s <= q_val, f"{col} <= {q_val:.3f} (<={q_label})")

    # Bounded: between p25 and p75 ("middle band")
    _add((s >= q25) & (s <= q75), f"{col} in [{q25:.3f}, {q75:.3f}] (p25-p75)")

    # Domain-specific useful thresholds
    if col == "ret_1h_pct":
        for thr in [0.0, 1.0, 2.0, 5.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}%")
        for thr in [0.0, -1.0, -2.0]:
            _add(s < thr, f"{col} < {thr:.1f}%")

    elif col == "ret_4h_pct":
        for thr in [0.0, 2.0, 5.0, 10.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}%")
        for thr in [0.0, -2.0, -5.0]:
            _add(s < thr, f"{col} < {thr:.1f}%")

    elif col == "ret_24h_pct":
        for thr in [0.0, 2.0, 5.0, 10.0, 20.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}%")
        for thr in [25.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}% (overextended)")
            _add(s < thr, f"{col} < {thr:.1f}% (not overextended)")

    elif col == "volume_ratio_1h":
        for thr in [1.0, 2.0, 3.0, 5.0, 10.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}x")
            _add(s <= thr, f"{col} <= {thr:.1f}x")
        _add((s >= 2.0) & (s <= 10.0), f"{col} in [2x, 10x]")

    elif col == "volume_ratio_4h":
        for thr in [1.0, 2.0, 5.0, 10.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}x")
            _add(s <= thr, f"{col} <= {thr:.1f}x")

    elif col == "spread_pct":
        for thr in [0.25, 0.5, 1.0, 2.0]:
            _add(s <= thr, f"{col} <= {thr:.2f}%")

    elif col == "primary_ohlc_score":
        for thr in [5.0, 10.0, 15.0, 20.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}")

    elif col == "long_explosion_score":
        for thr in [5.0, 10.0, 15.0, 20.0]:
            _add(s >= thr, f"{col} >= {thr:.1f}")

    elif col == "quote_volume_est":
        for thr in [10_000, 25_000, 100_000]:
            _add(s >= thr, f"{col} >= {thr:,.0f}")

    return rows


def build_threshold_sweep(
    df: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Sweep all eligible (non-outcome) numeric traits.

    Asserts at runtime that no outcome field is used as an entry filter.
    """
    # Verify OUTCOME_FIELDS is not in sweep list (belt-and-suspenders)
    for col in SWEEP_TRAIT_COLS:
        assert col not in OUTCOME_FIELDS, (
            f"SWEEP_TRAIT_COLS contains outcome field {col!r} — lookahead bias!"
        )

    all_rows: list[dict] = []
    for col in SWEEP_TRAIT_COLS:
        if col not in df.columns:
            logger.info("Sweep: column %r absent — skipped.", col)
            continue
        rows = _sweep_thresholds_for_col(df, col, min_group_size)
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Symbol summary
# ---------------------------------------------------------------------------


def build_symbol_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol return statistics."""
    completed = df.dropna(subset=["future_ret_4h_pct"])
    sym_col = "symbol" if "symbol" in completed.columns else "pair_id"
    records: list[dict] = []

    for sym, grp in completed.groupby(sym_col):
        stats = _ret_stats(grp["future_ret_4h_pct"])
        cc_share = (
            grp["is_clean_continuation"].astype(bool).mean()
            if "is_clean_continuation" in grp.columns else float("nan")
        )
        records.append({"symbol": sym, "cc_share": _safe(cc_share), **stats})

    df_out = pd.DataFrame(records)
    if len(df_out) and "n" in df_out.columns:
        df_out = df_out.sort_values("n", ascending=False).reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Hour-of-day summary
# ---------------------------------------------------------------------------


def build_hour_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour-of-day return statistics."""
    completed = df.dropna(subset=["future_ret_4h_pct"]).copy()
    if "hour_of_day_utc" not in completed.columns:
        if "snapshot_ts_utc" not in completed.columns:
            return pd.DataFrame()
        ts = pd.to_datetime(completed["snapshot_ts_utc"], utc=True, errors="coerce")
        completed["hour_of_day_utc"] = ts.dt.hour
    if completed["hour_of_day_utc"].isna().all():
        return pd.DataFrame()

    records: list[dict] = []
    for hour in range(24):
        grp = completed[completed["hour_of_day_utc"] == hour]
        stats = _ret_stats(grp["future_ret_4h_pct"])
        records.append({"hour_utc": hour, **stats})

    return pd.DataFrame(records)


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


def _fmtn(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  —  "
    return f"{int(v):>5}"


def _print_group_summary(groups: dict[str, pd.DataFrame]) -> None:
    print("\n  ─── Group Trait Summaries ───────────────────────────────────────────")
    hdr = (
        f"  {'group':<22} {'n':>5} {'avg4h':>7} {'med4h':>7} {'wr':>6} "
        f"{'bwr≥5':>6} {'fr≤-3':>6} {'fr≤-5':>6} "
        f"{'cc':>6} {'vcl':>6} {'ext':>6} {'vol_r1h avg':>12} {'ret_1h avg':>11} {'ret_24h avg':>12}"
    )
    print(hdr)
    print("  " + "─" * 125)

    for group_name, df in groups.items():
        stats = _ret_stats(df.get("future_ret_4h_pct", pd.Series(dtype=float)))
        cc = df["is_clean_continuation"].astype(bool).mean() if "is_clean_continuation" in df.columns else float("nan")
        vcl = df["is_volume_climax"].astype(bool).mean() if "is_volume_climax" in df.columns else float("nan")
        ext = df["is_overextended_24h"].astype(bool).mean() if "is_overextended_24h" in df.columns else float("nan")
        vr1h = pd.to_numeric(df.get("volume_ratio_1h", pd.Series(dtype=float)), errors="coerce").mean()
        r1h = pd.to_numeric(df.get("ret_1h_pct", pd.Series(dtype=float)), errors="coerce").mean()
        r24h = pd.to_numeric(df.get("ret_24h_pct", pd.Series(dtype=float)), errors="coerce").mean()

        print(
            f"  {group_name:<22} {stats['n']:>5} "
            f"{_fmt(stats['avg']):>7} {_fmt(stats['median']):>7} "
            f"{_fmtr(stats['win_rate']):>6} {_fmtr(stats['big_win_rate']):>6} "
            f"{_fmtr(stats['fail_rate_3']):>6} {_fmtr(stats['fail_rate_5']):>6} "
            f"{_fmtr(cc):>6} {_fmtr(vcl):>6} {_fmtr(ext):>6} "
            f"{vr1h:>12.2f} {_fmt(r1h):>11} {_fmt(r24h):>12}"
        )


def _print_sweep_highlights(sweep: pd.DataFrame, min_n: int) -> None:
    if sweep.empty:
        return

    print("\n  ─── Threshold Sweep Highlights (positive median, n >= {}) ───────────".format(min_n))
    pos_med = sweep[sweep["median"] > 0].sort_values("median", ascending=False)
    if pos_med.empty:
        print("  No filters produce positive median return.")
    else:
        print(f"  {'filter':<60} {'n':>5} {'avg':>7} {'med':>7} {'wr':>6} {'fr≤-3':>6} {'avg-x':>7}")
        print("  " + "─" * 105)
        for _, r in pos_med.head(20).iterrows():
            print(
                f"  {r['filter']:<60} {int(r['n']):>5} "
                f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} "
                f"{_fmtr(r['win_rate']):>6} {_fmtr(r['fail_rate_3']):>6} "
                f"{_fmt(r['avg_excl_best']):>7}"
            )

    print(f"\n  ─── Positive avg-excluding-best, n >= {min_n} ──────────────────────────────────")
    pos_ax = sweep[sweep["avg_excl_best"] > 0].sort_values("avg_excl_best", ascending=False)
    if pos_ax.empty:
        print("  No filters produce positive avg-excl-best return.")
    else:
        print(f"  {'filter':<60} {'n':>5} {'avg-x':>7} {'med':>7}")
        print("  " + "─" * 85)
        for _, r in pos_ax.head(20).iterrows():
            print(f"  {r['filter']:<60} {int(r['n']):>5} {_fmt(r['avg_excl_best']):>7} {_fmt(r['median']):>7}")


def _print_symbol_highlights(sym_df: pd.DataFrame, min_n: int = 10) -> None:
    if sym_df.empty:
        return
    print(f"\n  ─── Symbol Summary (n >= {min_n}, sorted by events) ──────────────────────────")
    top = sym_df[sym_df["n"] >= min_n].head(20)
    print(f"  {'symbol':<20} {'n':>5} {'avg':>7} {'med':>7} {'wr':>6} {'fr≤-5':>6} {'cc_share':>9}")
    print("  " + "─" * 65)
    for _, r in top.iterrows():
        print(
            f"  {str(r['symbol']):<20} {int(r['n']):>5} "
            f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} "
            f"{_fmtr(r['win_rate']):>6} {_fmtr(r['fail_rate_5']):>6} "
            f"{_fmtr(r['cc_share']):>9}"
        )


def _print_hour_highlights(hour_df: pd.DataFrame) -> None:
    if hour_df.empty:
        return
    print("\n  ─── Hour-of-Day Summary (UTC) ───────────────────────────────────────")
    print(f"  {'hour':>5} {'n':>5} {'avg':>7} {'med':>7} {'wr':>6} {'fr≤-3':>6}")
    print("  " + "─" * 45)
    for _, r in hour_df.iterrows():
        if r["n"] == 0:
            continue
        print(
            f"  {int(r['hour_utc']):>5} {int(r['n']):>5} "
            f"{_fmt(r['avg']):>7} {_fmt(r['median']):>7} "
            f"{_fmtr(r['win_rate']):>6} {_fmtr(r['fail_rate_3']):>6}"
        )


def print_interpretation(
    groups: dict[str, pd.DataFrame],
    sweep: pd.DataFrame,
    sym_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    min_n: int,
) -> None:
    """Print console results and key findings."""
    print()
    print("=" * 70)
    print("  BACKFILL SIGNAL TRAIT ANALYSIS — RESULTS")
    print("=" * 70)
    print()
    print("  *** Research only. Not a validated trading strategy. ***")
    print("  *** Spread data unavailable from OHLC backfill (all NaN). ***")

    _print_group_summary(groups)
    _print_sweep_highlights(sweep, min_n)
    _print_symbol_highlights(sym_df)
    _print_hour_highlights(hour_df)

    print()
    print("  ─── Key Findings ────────────────────────────────────────────────────")

    all_df = groups.get("all_events", pd.DataFrame())
    win_df = groups.get("big_winners_4h", pd.DataFrame())
    fail_df = groups.get("failures_4h", pd.DataFrame())
    severe_df = groups.get("severe_failures_4h", pd.DataFrame())

    def _share(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return float("nan")
        valid = df[col].dropna()
        return float(valid.astype(bool).mean()) if len(valid) else float("nan")

    def _mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return float("nan")
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    print()
    # Big winner traits
    cc_win = _share(win_df, "is_clean_continuation")
    cc_all = _share(all_df, "is_clean_continuation")
    vc_win = _share(win_df, "is_volume_climax")
    ext_win = _share(win_df, "is_overextended_24h")
    vr1h_win = _mean(win_df, "volume_ratio_1h")
    r24h_win = _mean(win_df, "ret_24h_pct")
    print(f"  1. Big winners (4h >= +5%, n={len(win_df)}):")
    print(f"     is_clean_continuation: {cc_win:.3f}  (all: {cc_all:.3f})")
    print(f"     is_volume_climax:      {_share(win_df, 'is_volume_climax'):.3f}  (all: {_share(all_df, 'is_volume_climax'):.3f})")
    print(f"     is_overextended_24h:   {ext_win:.3f}  (all: {_share(all_df, 'is_overextended_24h'):.3f})")
    print(f"     avg volume_ratio_1h:   {vr1h_win:.2f}  (all: {_mean(all_df, 'volume_ratio_1h'):.2f})")
    print(f"     avg ret_24h_pct:       {r24h_win:.2f}%  (all: {_mean(all_df, 'ret_24h_pct'):.2f}%)")

    # Failure traits
    cc_fail = _share(fail_df, "is_clean_continuation")
    vr1h_fail = _mean(fail_df, "volume_ratio_1h")
    r24h_fail = _mean(fail_df, "ret_24h_pct")
    print(f"\n  2. 4h failures (<= -3%, n={len(fail_df)}):")
    print(f"     is_clean_continuation: {cc_fail:.3f}  (all: {cc_all:.3f})")
    print(f"     is_volume_climax:      {_share(fail_df, 'is_volume_climax'):.3f}  (all: {_share(all_df, 'is_volume_climax'):.3f})")
    print(f"     avg volume_ratio_1h:   {vr1h_fail:.2f}  (all: {_mean(all_df, 'volume_ratio_1h'):.2f})")
    print(f"     avg ret_24h_pct:       {r24h_fail:.2f}%  (all: {_mean(all_df, 'ret_24h_pct'):.2f}%)")

    # CC discrimination
    cc_win_avg = _mean(win_df[win_df.get("is_clean_continuation", pd.Series(False)) == True] if "is_clean_continuation" in win_df else pd.DataFrame(), "future_ret_4h_pct") if len(win_df) else float("nan")
    if not math.isnan(cc_all) and not math.isnan(cc_win):
        lift = cc_win - cc_all
        direction = "higher" if lift > 0 else "lower"
        print(f"\n  3. is_clean_continuation: share in big_winners={cc_win:.3f} vs all={cc_all:.3f} → {direction} ({lift:+.3f})")
        print(f"     → {'CC correlates positively with big wins.' if lift > 0.02 else 'CC does not strongly separate winners from losers.'}")

    # Volume climax
    vc_all = _share(all_df, "is_volume_climax")
    vc_fail = _share(fail_df, "is_volume_climax")
    if not math.isnan(vc_all) and not math.isnan(vc_win):
        print(f"\n  4. Volume climax: win_share={vc_win:.3f}, fail_share={vc_fail:.3f}, all={vc_all:.3f}")
        if vc_fail > vc_all + 0.02:
            print(f"     → Volume climax appears DANGEROUS (more common in failures).")
        elif vc_win > vc_all + 0.02:
            print(f"     → Volume climax appears HELPFUL (more common in winners).")
        else:
            print(f"     → Volume climax shows no strong directional signal.")

    # Overextended 24h
    ext_all = _share(all_df, "is_overextended_24h")
    ext_fail = _share(fail_df, "is_overextended_24h")
    if not math.isnan(ext_all) and not math.isnan(ext_win):
        print(f"\n  5. Overextended 24h: win_share={ext_win:.3f}, fail_share={ext_fail:.3f}, all={ext_all:.3f}")
        if ext_win > ext_all + 0.02:
            print(f"     → Overextended 24h correlates with MORE big winners.")
        elif ext_fail > ext_all + 0.02:
            print(f"     → Overextended 24h correlates with MORE failures.")
        else:
            print(f"     → Overextended 24h shows no strong directional signal.")

    # Wide spreads
    ws_all = _share(all_df, "is_wide_spread")
    if not math.isnan(ws_all) and ws_all > 0:
        ws_fail = _share(fail_df, "is_wide_spread")
        print(f"\n  6. Wide spread: all={ws_all:.3f}, fail={ws_fail:.3f}")
    else:
        print("\n  6. Wide spread: all NaN (spread data unavailable from OHLC backfill).")

    # Symbol concentration
    if not sym_df.empty:
        top_sym = sym_df[sym_df["n"] >= 10].nlargest(3, "n")
        print("\n  7. Symbol concentration (top 3 by event count):")
        for _, r in top_sym.iterrows():
            print(f"     {r['symbol']}: n={int(r['n'])}, avg={r['avg']:.2f}%, med={r['median']:.2f}%")

    # Hour of day
    if not hour_df.empty:
        best_hours = hour_df[hour_df["n"] >= 30].nlargest(3, "median")
        if not best_hours.empty:
            print("\n  8. Best hours UTC (n >= 30, by median):")
            for _, r in best_hours.iterrows():
                print(f"     Hour {int(r['hour_utc']):02d}:  n={int(r['n'])}, avg={r['avg']:.2f}%, med={r['median']:.2f}%")

    # Simple filters summary
    print(f"\n  9. Simple filters with positive median return (n >= {min_n}):")
    if not sweep.empty:
        pos_med = sweep[(sweep["median"] > 0) & (sweep["n"] >= min_n)]
        if pos_med.empty:
            print(f"     None found.")
        else:
            for _, r in pos_med.sort_values("median", ascending=False).head(5).iterrows():
                print(f"     {r['filter']}: n={int(r['n'])}, med={r['median']:.2f}%, avg={r['avg']:.2f}%")

    print(f"\n  10. Simple filters with positive avg-excluding-best (n >= {min_n}):")
    if not sweep.empty:
        pos_ax = sweep[(sweep["avg_excl_best"] > 0) & (sweep["n"] >= min_n)]
        if pos_ax.empty:
            print(f"      None found.")
        else:
            for _, r in pos_ax.sort_values("avg_excl_best", ascending=False).head(5).iterrows():
                print(f"      {r['filter']}: n={int(r['n'])}, avg-x={r['avg_excl_best']:.2f}%, med={r['median']:.2f}%")

    # Candidate filter suggestions
    print("\n  ─── Candidate Entry Filter Suggestions ─────────────────────────────")
    print("  Based on this analysis, suggested filters for the next experiment:")

    suggestions: list[str] = []

    # Check if high volume ratio helps
    if not sweep.empty:
        vr_rows = sweep[sweep["trait"] == "volume_ratio_1h"].sort_values("median", ascending=False)
        if not vr_rows.empty:
            best_vr = vr_rows.iloc[0]
            if best_vr["median"] > -sweep[sweep["trait"] == "volume_ratio_1h"]["median"].median():
                suggestions.append(f"A. {best_vr['filter']}  → med={best_vr['median']:.2f}%, n={int(best_vr['n'])}")

    # Check if moderate ret_24h (not overextended) helps
    if not sweep.empty:
        r24_rows = sweep[sweep["trait"] == "ret_24h_pct"].sort_values("median", ascending=False)
        if not r24_rows.empty:
            best_r24 = r24_rows.iloc[0]
            suggestions.append(f"B. {best_r24['filter']}  → med={best_r24['median']:.2f}%, n={int(best_r24['n'])}")

    # CC lift
    if not math.isnan(cc_win) and not math.isnan(cc_all):
        if cc_win > cc_all:
            suggestions.append("C. is_clean_continuation == True  (higher share in big winners)")
        else:
            suggestions.append("C. is_clean_continuation == True  (no uplift in backfill — consider dropping)")

    # Overextended avoidance
    if not math.isnan(ext_win) and not math.isnan(ext_fail):
        if ext_fail > ext_all:
            suggestions.append("D. exclude is_overextended_24h == True  (higher share in failures)")
        else:
            suggestions.append("D. keep is_overextended_24h filter — already in CC definition")

    # Volume climax avoidance
    if not math.isnan(vc_fail) and not math.isnan(vc_all):
        if vc_fail > vc_all + 0.02:
            suggestions.append("E. exclude is_volume_climax == True  (higher share in failures)")
        else:
            suggestions.append("E. volume_climax shows no clear edge — test with and without")

    if not suggestions:
        suggestions = [
            "A. is_clean_continuation == True (baseline candidate filter)",
            "B. volume_ratio_1h > 2x (moderate volume surge)",
            "C. ret_24h_pct < 25% (not overextended)",
            "D. ret_1h_pct > 0% (positive 1h momentum)",
            "E. exclude is_volume_climax == True (parabolic volume spike)",
        ]

    for sug in suggestions[:5]:
        print(f"  {sug}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    group_summary_path: Path = GROUP_SUMMARY_PATH,
    sweep_path: Path = SWEEP_SUMMARY_PATH,
    symbol_path: Path = SYMBOL_SUMMARY_PATH,
    hour_path: Path = HOUR_SUMMARY_PATH,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the trait analysis. Returns (group_summary, sweep, sym_df, hour_df)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 70)
    print("  Memecoin Backfill Signal Trait Analysis")
    print("=" * 70)

    df = load_events(input_path)
    print(f"\n  Loaded {len(df)} events from {input_path}")
    n_complete = df["future_ret_4h_pct"].notna().sum()
    print(f"  Events with completed 4h outcomes: {n_complete}")

    groups = assign_groups(df)
    group_summary = build_group_summary(groups)
    sweep = build_threshold_sweep(df, min_group_size=min_group_size)
    sym_df = build_symbol_summary(df)
    hour_df = build_hour_summary(df)

    for path, data, label in [
        (group_summary_path, group_summary, "Group summary"),
        (sweep_path, sweep, "Threshold sweep"),
        (symbol_path, sym_df, "Symbol summary"),
        (hour_path, hour_df, "Hour-of-day summary"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"  -> {label} written to {path}  ({len(data)} rows)")

    print_interpretation(groups, sweep, sym_df, hour_df, min_group_size)

    return group_summary, sweep, sym_df, hour_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    main(
        input_path=args.input,
        group_summary_path=(out_dir / GROUP_SUMMARY_PATH.name) if out_dir else GROUP_SUMMARY_PATH,
        sweep_path=(out_dir / SWEEP_SUMMARY_PATH.name) if out_dir else SWEEP_SUMMARY_PATH,
        symbol_path=(out_dir / SYMBOL_SUMMARY_PATH.name) if out_dir else SYMBOL_SUMMARY_PATH,
        hour_path=(out_dir / HOUR_SUMMARY_PATH.name) if out_dir else HOUR_SUMMARY_PATH,
        min_group_size=args.min_group_size,
    )
