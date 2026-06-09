"""Candidate strategy comparison for memecoin clean-continuation signals.

Evaluates six candidate strategies over the LONG_EXPLOSION universe and
produces a cross-strategy summary with outlier sensitivity analysis.

Strategies:
  1. baseline_long_explosion_fixed_4h     — all LONG_EXPLOSION, fixed 4h exit
  2. clean_continuation_fixed_4h         — CC-filtered, fixed 4h exit
  3. clean_continuation_tp10_else_4h     — CC-filtered, TP +10% else 4h
  4. clean_continuation_tp10_sl5_else_4h — CC-filtered, TP +10% / SL -5%
  5. clean_continuation_tp10_sl7_else_4h — CC-filtered, TP +10% / SL -7%
  6. clean_continuation_tp10_sl10_else_4h— CC-filtered, TP +10% / SL -10%

Summary rows: one per (strategy × subset), where subset is one of:
  all | excl_ALLO | excl_best_trade

Research only. No trading. No orders. No live code touched.

Usage:
    .venv/bin/python -m research.memecoin_catcher.candidate_strategy_comparison
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from research.memecoin_catcher.analyze_signal_traits import (
    add_diagnostic_flags,
    load_long_explosion,
)
from research.memecoin_catcher.exit_policy_experiment import (
    _take_profit_else_4h,
    _tp_sl_else_4h,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTCOMES_PATH = Path("data/memecoin_signal_outcomes.csv")
SUMMARY_OUT_PATH = Path("reports/memecoin_candidate_strategy_summary.csv")
EVENT_LOG_OUT_PATH = Path("reports/memecoin_candidate_strategy_event_log.csv")

STRATEGY_NAMES: list[str] = [
    "baseline_long_explosion_fixed_4h",
    "clean_continuation_fixed_4h",
    "clean_continuation_tp10_else_4h",
    "clean_continuation_tp10_sl5_else_4h",
    "clean_continuation_tp10_sl7_else_4h",
    "clean_continuation_tp10_sl10_else_4h",
]

SUBSET_NAMES: list[str] = ["all", "excl_ALLO", "excl_best_trade"]

_REQUIRED_4H_COLS = ["future_ret_4h_pct", "max_favorable_4h_pct", "max_adverse_4h_pct"]

OUTLIER_SYMBOL = "ALLO/USD"


# ---------------------------------------------------------------------------
# Entry universe helpers
# ---------------------------------------------------------------------------


def _symbol(df: pd.DataFrame) -> pd.Series:
    """Return canonical symbol column (wsname preferred, then pair_id)."""
    if "wsname" in df.columns:
        return df["wsname"]
    return df.get("pair_id", pd.Series([""] * len(df), index=df.index))


def load_completed(path: Path = DEFAULT_OUTCOMES_PATH) -> pd.DataFrame:
    """
    Load all LONG_EXPLOSION rows, add diagnostic flags, add symbol column,
    and retain only rows with completed 4h outcome data (ret + MFE + MAE).
    """
    df = load_long_explosion(path)
    df = add_diagnostic_flags(df)
    df["symbol"] = _symbol(df)

    # Rename snapshot timestamp column if needed
    if "snapshot_ts_utc" not in df.columns and "ts_utc" in df.columns:
        df = df.rename(columns={"ts_utc": "snapshot_ts_utc"})

    for col in _REQUIRED_4H_COLS:
        if col not in df.columns:
            logger.warning("Column %r missing — strategies needing it will have 0 events.", col)

    before = len(df)
    df = df.dropna(subset=_REQUIRED_4H_COLS).reset_index(drop=True)
    logger.info(
        "Loaded %d LONG_EXPLOSION rows, %d dropped (incomplete 4h), %d usable.",
        before, before - len(df), len(df),
    )
    return df


def _is_clean_continuation_entry(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask for the clean-continuation entry universe:
      - is_clean_continuation == True
      - is_wide_spread == False          (skipped gracefully if absent)
      - danger_terminal_spike == False   (skipped gracefully if absent)
      - is_terminal_spike == False       (skipped gracefully if absent)
    """
    mask = df["is_clean_continuation"].fillna(False).astype(bool)

    for col in ["is_wide_spread", "danger_terminal_spike", "is_terminal_spike"]:
        if col not in df.columns:
            logger.info("Column %r absent — exclusion filter skipped.", col)
            continue
        mask = mask & (~df[col].fillna(False).astype(bool))

    return mask


# ---------------------------------------------------------------------------
# Exit-return computation per row
# ---------------------------------------------------------------------------


def _compute_exit_return(row: pd.Series, strategy_name: str) -> tuple[float, str]:
    """Return (realized_return_pct, trigger_label) for a given strategy."""
    if strategy_name == "baseline_long_explosion_fixed_4h":
        return float(row["future_ret_4h_pct"]), "4h_close"

    if strategy_name == "clean_continuation_fixed_4h":
        return float(row["future_ret_4h_pct"]), "4h_close"

    if strategy_name == "clean_continuation_tp10_else_4h":
        return _take_profit_else_4h(row, 10.0)

    if strategy_name == "clean_continuation_tp10_sl5_else_4h":
        return _tp_sl_else_4h(row, 10.0, -5.0)

    if strategy_name == "clean_continuation_tp10_sl7_else_4h":
        return _tp_sl_else_4h(row, 10.0, -7.0)

    if strategy_name == "clean_continuation_tp10_sl10_else_4h":
        return _tp_sl_else_4h(row, 10.0, -10.0)

    raise ValueError(f"Unknown strategy: {strategy_name!r}")


# ---------------------------------------------------------------------------
# Per-strategy event selection
# ---------------------------------------------------------------------------


def _strategy_mask(df: pd.DataFrame, strategy_name: str) -> pd.Series:
    """Return boolean mask for rows that qualify as entries for the strategy."""
    if strategy_name == "baseline_long_explosion_fixed_4h":
        return pd.Series([True] * len(df), index=df.index)
    return _is_clean_continuation_entry(df)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _safe_round(v: float, d: int = 4) -> float:
    return round(v, d) if not math.isnan(v) else float("nan")


def compute_stats(
    ret_series: pd.Series,
    mae_series: pd.Series,
    mfe_series: pd.Series,
    strategy_name: str,
    subset_name: str,
) -> dict:
    """Compute all required statistics for a (strategy, subset) pair."""
    s = ret_series.dropna()
    n = len(s)

    def _nan() -> float:
        return float("nan")

    base: dict = {
        "strategy": strategy_name,
        "subset": subset_name,
        "n_events": n,
        "avg_return_pct": _nan(),
        "median_return_pct": _nan(),
        "win_rate": _nan(),
        "success_rate_ge_5pct": _nan(),
        "failure_rate_le_neg3pct": _nan(),
        "failure_rate_le_neg5pct": _nan(),
        "worst_trade_pct": _nan(),
        "best_trade_pct": _nan(),
        "avg_return_excl_best_pct": _nan(),
        "median_return_excl_best_pct": _nan(),
        "avg_mae_4h_pct": _nan(),
        "median_mae_4h_pct": _nan(),
        "avg_mfe_4h_pct": _nan(),
        "median_mfe_4h_pct": _nan(),
    }

    if n == 0:
        return base

    base["avg_return_pct"] = _safe_round(s.mean())
    base["median_return_pct"] = _safe_round(s.median())
    base["win_rate"] = _safe_round((s > 0).mean())
    base["success_rate_ge_5pct"] = _safe_round((s >= 5.0).mean())
    base["failure_rate_le_neg3pct"] = _safe_round((s <= -3.0).mean())
    base["failure_rate_le_neg5pct"] = _safe_round((s <= -5.0).mean())
    base["worst_trade_pct"] = _safe_round(s.min())
    base["best_trade_pct"] = _safe_round(s.max())

    excl = s.drop(index=s.idxmax())
    base["avg_return_excl_best_pct"] = _safe_round(excl.mean() if len(excl) else _nan())
    base["median_return_excl_best_pct"] = _safe_round(excl.median() if len(excl) else _nan())

    mae = mae_series.dropna()
    mfe = mfe_series.dropna()
    base["avg_mae_4h_pct"] = _safe_round(mae.mean() if len(mae) else _nan())
    base["median_mae_4h_pct"] = _safe_round(mae.median() if len(mae) else _nan())
    base["avg_mfe_4h_pct"] = _safe_round(mfe.mean() if len(mfe) else _nan())
    base["median_mfe_4h_pct"] = _safe_round(mfe.median() if len(mfe) else _nan())

    return base


def _subsets(df_strat: pd.DataFrame, ret_col: str) -> list[tuple[str, pd.DataFrame]]:
    """Return (subset_name, subset_df) for all three subset views."""
    all_df = df_strat.dropna(subset=[ret_col])
    excl_allo = all_df[all_df["symbol"] != OUTLIER_SYMBOL]

    if len(all_df) > 0:
        best_idx = all_df[ret_col].idxmax()
        excl_best = all_df.drop(index=best_idx)
    else:
        excl_best = all_df.copy()

    return [("all", all_df), ("excl_ALLO", excl_allo), ("excl_best_trade", excl_best)]


# ---------------------------------------------------------------------------
# Main builders
# ---------------------------------------------------------------------------


def build_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (strategy × subset). Includes all required statistics."""
    records: list[dict] = []

    for strategy_name in STRATEGY_NAMES:
        mask = _strategy_mask(df, strategy_name)
        df_strat = df[mask].copy().reset_index(drop=True)

        # Compute exit returns for this strategy
        rets, triggers = [], []
        for _, row in df_strat.iterrows():
            ret, trigger = _compute_exit_return(row, strategy_name)
            rets.append(ret)
            triggers.append(trigger)
        df_strat["_ret"] = rets
        df_strat["_trigger"] = triggers

        for subset_name, sub in _subsets(df_strat, "_ret"):
            records.append(
                compute_stats(
                    sub["_ret"],
                    sub.get("max_adverse_4h_pct", pd.Series(dtype=float)),
                    sub.get("max_favorable_4h_pct", pd.Series(dtype=float)),
                    strategy_name,
                    subset_name,
                )
            )

    return pd.DataFrame(records)


def build_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per LONG_EXPLOSION completed event.

    Includes:
      - identity columns (symbol, snapshot_ts_utc, entry_price)
      - diagnostic flags
      - raw 4h return + MAE/MFE
      - in_strategy_<name> boolean for each strategy
      - ret_<name> and trigger_<name> for each strategy (NaN if not in universe)
    """
    flag_cols = [
        "is_clean_continuation", "is_wide_spread", "is_overextended_24h",
        "is_volume_climax", "is_rolling_over", "danger_terminal_spike",
        "is_pullback_rebound_candidate",
    ]
    raw_cols = [
        "future_ret_4h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "max_favorable_24h_pct", "max_adverse_24h_pct",
    ]
    identity_cols = ["symbol", "snapshot_ts_utc", "entry_price"]

    keep = (
        [c for c in identity_cols if c in df.columns]
        + [c for c in flag_cols if c in df.columns]
        + [c for c in raw_cols if c in df.columns]
    )
    out = df[keep].copy().reset_index(drop=True)

    for strategy_name in STRATEGY_NAMES:
        mask = _strategy_mask(df, strategy_name)
        out[f"in_{strategy_name}"] = mask.values

        rets, triggers = [], []
        for i, row in df.iterrows():
            if mask.iloc[df.index.get_loc(i)]:
                ret, trigger = _compute_exit_return(row, strategy_name)
            else:
                ret, trigger = float("nan"), ""
            rets.append(ret)
            triggers.append(trigger)
        out[f"ret_{strategy_name}"] = rets
        out[f"trigger_{strategy_name}"] = triggers

    return out


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _fmt(v, d: int = 2) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return f"{v:.{d}f}%"


def _fmtr(v, d: int = 3) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return f"{v:.{d}f}"


def _print_subset_table(summary: pd.DataFrame, subset: str, title: str) -> None:
    sub = summary[summary["subset"] == subset]
    if sub.empty:
        return
    print(f"\n  ─── {title} ────────────────────────────────────────────────────────")
    hdr = (
        f"  {'strategy':<44} {'n':>4} {'avg':>7} {'med':>7} "
        f"{'wr':>6} {'sr≥5':>6} {'fr≤-3':>6} {'fr≤-5':>6} "
        f"{'best':>7} {'worst':>7} {'avg-x':>7} {'med-x':>7}"
    )
    print(hdr)
    print("  " + "─" * 122)
    for _, row in sub.iterrows():
        print(
            f"  {row['strategy']:<44} {int(row['n_events']):>4} "
            f"{_fmt(row['avg_return_pct']):>7} {_fmt(row['median_return_pct']):>7} "
            f"{_fmtr(row['win_rate']):>6} {_fmtr(row['success_rate_ge_5pct']):>6} "
            f"{_fmtr(row['failure_rate_le_neg3pct']):>6} {_fmtr(row['failure_rate_le_neg5pct']):>6} "
            f"{_fmt(row['best_trade_pct']):>7} {_fmt(row['worst_trade_pct']):>7} "
            f"{_fmt(row['avg_return_excl_best_pct']):>7} "
            f"{_fmt(row['median_return_excl_best_pct']):>7}"
        )
    print()
    print("  avg-x / med-x = average / median excluding the single best trade")


def print_interpretation(summary: pd.DataFrame) -> None:
    """Print the full results table and key interpretive findings."""
    print()
    print("=" * 70)
    print("  CANDIDATE STRATEGY COMPARISON — RESULTS")
    print("=" * 70)
    print()
    print("  *** Research only. Not a validated trading strategy. ***")

    _print_subset_table(summary, "all", "All Events")
    _print_subset_table(summary, "excl_ALLO", "Excluding ALLO/USD Outlier")
    _print_subset_table(summary, "excl_best_trade", "Excluding Single Best Trade")

    print()
    print("  ─── Interpretation ──────────────────────────────────────────────────")

    def _get(strategy: str, subset: str, col: str) -> float:
        row = summary[(summary["strategy"] == strategy) & (summary["subset"] == subset)]
        if row.empty:
            return float("nan")
        v = row.iloc[0][col]
        return float(v) if pd.notna(v) else float("nan")

    # 1. Does clean_continuation beat baseline?
    b_avg = _get("baseline_long_explosion_fixed_4h", "all", "avg_return_pct")
    b_med = _get("baseline_long_explosion_fixed_4h", "all", "median_return_pct")
    cc_avg = _get("clean_continuation_fixed_4h", "all", "avg_return_pct")
    cc_med = _get("clean_continuation_fixed_4h", "all", "median_return_pct")
    cc_beats = cc_avg > b_avg and cc_med > b_med
    print(
        f"\n  1. clean_continuation_fixed_4h vs baseline_long_explosion_fixed_4h:"
    )
    print(
        f"     baseline: avg={_fmt(b_avg)}  med={_fmt(b_med)}"
    )
    print(
        f"     clean_cc: avg={_fmt(cc_avg)}  med={_fmt(cc_med)}"
    )
    print(
        f"     → {'YES, clean_continuation beats baseline on both avg and median.' if cc_beats else 'Mixed — check table above.'}"
    )

    # 2. tp10 vs fixed_4h excluding best trade
    cc_x = _get("clean_continuation_fixed_4h", "excl_best_trade", "avg_return_excl_best_pct")
    tp10_x = _get("clean_continuation_tp10_else_4h", "excl_best_trade", "avg_return_excl_best_pct")
    print(
        f"\n  2. tp10_else_4h vs fixed_4h (excl best trade):"
    )
    print(
        f"     clean_cc avg-excl-best={_fmt(cc_x)}  tp10 avg-excl-best={_fmt(tp10_x)}"
    )
    if not math.isnan(tp10_x) and not math.isnan(cc_x):
        print(
            f"     → {'tp10 BEATS fixed_4h after excluding the best trade.' if tp10_x > cc_x else 'tp10 does NOT beat fixed_4h after excluding the best trade.'}"
        )

    # 3. Wider stops vs no stop
    print(f"\n  3. Wider stop-loss tiers (vs no stop — clean_continuation_fixed_4h):")
    cc_fr3 = _get("clean_continuation_fixed_4h", "all", "failure_rate_le_neg3pct")
    for sl, sname in [(-5, "tp10_sl5"), (-7, "tp10_sl7"), (-10, "tp10_sl10")]:
        strat = f"clean_continuation_{sname}_else_4h"
        avg_ = _get(strat, "all", "avg_return_pct")
        fr3_ = _get(strat, "all", "failure_rate_le_neg3pct")
        fr5_ = _get(strat, "all", "failure_rate_le_neg5pct")
        avg_x_ = _get(strat, "all", "avg_return_excl_best_pct")
        print(
            f"     SL{sl}%: avg={_fmt(avg_)}  fr≤-3%={_fmtr(fr3_)}  fr≤-5%={_fmtr(fr5_)}  avg-excl-best={_fmt(avg_x_)}"
        )
    cc_fr5 = _get("clean_continuation_fixed_4h", "all", "failure_rate_le_neg5pct")
    cc_avg_x = _get("clean_continuation_fixed_4h", "all", "avg_return_excl_best_pct")
    print(
        f"     no_stop: avg={_fmt(cc_avg)}  fr≤-3%={_fmtr(cc_fr3)}  fr≤-5%={_fmtr(cc_fr5)}  avg-excl-best={_fmt(cc_avg_x)}"
    )

    # 4. Best median
    all_sub = summary[summary["subset"] == "all"].dropna(subset=["median_return_pct"])
    if not all_sub.empty:
        best_med_row = all_sub.loc[all_sub["median_return_pct"].idxmax()]
        print(
            f"\n  4. Best median return: {best_med_row['strategy']}  ({best_med_row['median_return_pct']:.2f}%)"
        )

    # 5. Best avg excl best
    excl_sub = summary[summary["subset"] == "excl_best_trade"].dropna(subset=["avg_return_excl_best_pct"])
    if not excl_sub.empty:
        best_x_row = excl_sub.loc[excl_sub["avg_return_excl_best_pct"].idxmax()]
        print(
            f"  5. Best avg excl best trade: {best_x_row['strategy']}  ({best_x_row['avg_return_excl_best_pct']:.2f}%)"
        )

    # 6. Lowest serious failure (fr≤-5%)
    all_sub2 = summary[summary["subset"] == "all"].dropna(subset=["failure_rate_le_neg5pct"])
    if not all_sub2.empty:
        low_fail_row = all_sub2.loc[all_sub2["failure_rate_le_neg5pct"].idxmin()]
        print(
            f"  6. Lowest serious failure rate (≤ -5%): {low_fail_row['strategy']}  "
            f"({low_fail_row['failure_rate_le_neg5pct']:.3f})"
        )

    # 7. 24h holding
    print(
        "\n  7. 24h holding: No — prior analysis showed fixed_24h_exit has avg=-0.41%,"
        "\n     median=-1.51%, and failure rate 35% vs 5% for fixed_4h. Data gives no"
        "\n     reason to reconsider 24h holding here."
    )

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    summary_out_path: Path = SUMMARY_OUT_PATH,
    event_log_out_path: Path = EVENT_LOG_OUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the candidate strategy comparison. Returns (summary_df, event_log_df)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 70)
    print("  Memecoin Candidate Strategy Comparison")
    print("=" * 70)

    df = load_completed(outcomes_path)
    print(f"\n  {len(df)} LONG_EXPLOSION events with completed 4h data")

    summary = build_strategy_summary(df)
    event_log = build_event_log(df)

    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out_path, index=False)
    print(f"  -> Summary written to {summary_out_path}  ({len(summary)} rows)")

    event_log_out_path.parent.mkdir(parents=True, exist_ok=True)
    event_log.to_csv(event_log_out_path, index=False)
    print(f"  -> Event log written to {event_log_out_path}  ({len(event_log)} events)")

    print_interpretation(summary)

    return summary, event_log


if __name__ == "__main__":
    main()
