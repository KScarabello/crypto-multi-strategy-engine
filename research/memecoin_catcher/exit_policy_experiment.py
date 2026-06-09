"""Exit-policy experiment for clean-continuation memecoin signals.

Tests 10 exit rules on the existing clean-continuation event log and produces:
  - reports/memecoin_exit_policy_summary.csv   (one row per policy)
  - reports/memecoin_exit_policy_event_log.csv (one row per event x all policy returns)

Research only. No trading. No orders. No live code touched.

Trailing-stop approximation (documented in _trailing_stop_else_4h):
  True intraperiod path data is unavailable. The approximation uses
  max_favorable_4h_pct and a trail distance equal to the activation threshold.
  Trail stop level = max_favorable_4h_pct - activation_pct.
  If the 4h close is at or below the trail level, we assume the trail fired
  and book the trail level as the exit. Otherwise we book the 4h close.
  This is conservative — it assumes the worst-case realised from peak.

Usage:
    .venv/bin/python -m research.memecoin_catcher.exit_policy_experiment
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EVENT_LOG_PATH = Path("reports/memecoin_clean_continuation_event_log.csv")
SUMMARY_OUT_PATH = Path("reports/memecoin_exit_policy_summary.csv")
EVENT_LOG_OUT_PATH = Path("reports/memecoin_exit_policy_event_log.csv")

POLICY_NAMES: list[str] = [
    "fixed_4h_exit",
    "fixed_24h_exit",
    "take_profit_5_else_4h",
    "take_profit_10_else_4h",
    "stop_loss_3_else_4h",
    "stop_loss_5_else_4h",
    "take_profit_10_stop_loss_5_else_4h",
    "take_profit_5_stop_loss_3_else_4h",
    "trailing_stop_after_up_5",
    "trailing_stop_after_up_10",
]

_DIAGNOSTIC_FLAG_COLS = [
    "is_clean_continuation",
    "is_wide_spread",
    "is_overextended_24h",
    "is_volume_climax",
    "is_rolling_over",
    "is_terminal_spike",       # may be absent; handled gracefully
    "danger_terminal_spike",
    "is_pullback_rebound_candidate",
]

_NUMERIC_COLS = [
    "future_ret_4h_pct",
    "future_ret_24h_pct",
    "max_favorable_4h_pct",
    "max_adverse_4h_pct",
    "max_favorable_24h_pct",
    "max_adverse_24h_pct",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_event_log(path: Path = EVENT_LOG_PATH) -> pd.DataFrame:
    """Load the clean-continuation event log and coerce numeric columns."""
    df = pd.read_csv(path)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Entry filtering
# ---------------------------------------------------------------------------


def apply_entry_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply entry-universe filters. Missing columns are logged and skipped.

    Kept when:
      - long_explosion == True          (skipped if column absent)
      - is_clean_continuation == True
      Excluded when:
      - is_wide_spread == True          (skipped if column absent)
      - is_terminal_spike == True       (skipped if column absent)
      - danger_terminal_spike == True   (skipped if column absent)
    """
    mask = pd.Series([True] * len(df), index=df.index)

    # {column: True means "keep when True", False means "exclude when True"}
    filter_spec: dict[str, bool] = {
        "long_explosion": True,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "is_terminal_spike": False,
        "danger_terminal_spike": False,
    }

    for col, keep_when_true in filter_spec.items():
        if col not in df.columns:
            logger.info("Column %r not in event log — filter skipped.", col)
            continue
        bool_col = df[col].fillna(False).astype(bool)
        if keep_when_true:
            mask = mask & bool_col
        else:
            mask = mask & (~bool_col)

    n_in, n_out = len(df), int(mask.sum())
    logger.info("Entry filter: %d/%d rows kept.", n_out, n_in)
    return df[mask].copy().reset_index(drop=True)


def require_completed_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 4h outcome data (ret, MFE, MAE) is missing."""
    required = ["future_ret_4h_pct", "max_favorable_4h_pct", "max_adverse_4h_pct"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with incomplete 4h outcome data.", dropped)
    return df


# ---------------------------------------------------------------------------
# Exit policy functions
# Each returns (realized_return: float, trigger_label: str)
# ---------------------------------------------------------------------------


def _fixed_4h(row: pd.Series) -> tuple[float, str]:
    return float(row["future_ret_4h_pct"]), "4h_close"


def _fixed_24h(row: pd.Series) -> tuple[float, str]:
    val = row.get("future_ret_24h_pct", float("nan"))
    return float(val) if pd.notna(val) else float("nan"), "24h_close"


def _take_profit_else_4h(row: pd.Series, tp_pct: float) -> tuple[float, str]:
    if float(row["max_favorable_4h_pct"]) >= tp_pct:
        return tp_pct, f"take_profit_{tp_pct:.0f}pct"
    return float(row["future_ret_4h_pct"]), "4h_close"


def _stop_loss_else_4h(row: pd.Series, sl_pct: float) -> tuple[float, str]:
    """sl_pct is negative, e.g. -3.0."""
    if float(row["max_adverse_4h_pct"]) <= sl_pct:
        return sl_pct, f"stop_loss_{abs(sl_pct):.0f}pct"
    return float(row["future_ret_4h_pct"]), "4h_close"


def _tp_sl_else_4h(
    row: pd.Series, tp_pct: float, sl_pct: float
) -> tuple[float, str]:
    """
    Combined take-profit / stop-loss with conservative ordering.

    If both levels are touched inside the 4h window, the stop-loss is assumed
    to have fired first (conservative). sl_pct is negative, e.g. -5.0.
    """
    mfe = float(row["max_favorable_4h_pct"])
    mae = float(row["max_adverse_4h_pct"])
    ret4h = float(row["future_ret_4h_pct"])

    tp_hit = mfe >= tp_pct
    sl_hit = mae <= sl_pct

    if tp_hit and sl_hit:
        return sl_pct, "stop_loss_first_conservative"
    if tp_hit:
        return tp_pct, f"take_profit_{tp_pct:.0f}pct"
    if sl_hit:
        return sl_pct, f"stop_loss_{abs(sl_pct):.0f}pct"
    return ret4h, "4h_close"


def _trailing_stop_else_4h(
    row: pd.Series, activation_pct: float
) -> tuple[float, str]:
    """
    Trailing stop that activates once price reaches +activation_pct.

    Approximation (no intraperiod path data available):
      trail_distance = activation_pct  (absolute %, matches activation level)
      trail_level    = max_favorable_4h_pct - activation_pct

      Activation never fired (max_favorable < activation_pct):
        → exit at future_ret_4h_pct

      Activation fired (max_favorable >= activation_pct):
        → trail_level = max_favorable - activation_pct
        → if 4h close <= trail_level: trail fired, book trail_level
        → if 4h close >  trail_level: close is above trail, book 4h close

    This is conservative: it treats any drawdown from peak that crosses the
    trail level as a triggered exit, using the trail level as the exit price.
    True intraperiod path data would be needed for exact modeling.
    """
    mfe = float(row["max_favorable_4h_pct"])
    ret4h = float(row["future_ret_4h_pct"])

    if mfe < activation_pct:
        return ret4h, "4h_close"

    trail_level = mfe - activation_pct
    if ret4h <= trail_level:
        return trail_level, "trailing_stop_approx"
    return ret4h, "4h_close_above_trail"


# Map policy name → function
POLICY_FUNCTIONS: dict[str, object] = {
    "fixed_4h_exit":                        lambda r: _fixed_4h(r),
    "fixed_24h_exit":                       lambda r: _fixed_24h(r),
    "take_profit_5_else_4h":                lambda r: _take_profit_else_4h(r, 5.0),
    "take_profit_10_else_4h":               lambda r: _take_profit_else_4h(r, 10.0),
    "stop_loss_3_else_4h":                  lambda r: _stop_loss_else_4h(r, -3.0),
    "stop_loss_5_else_4h":                  lambda r: _stop_loss_else_4h(r, -5.0),
    "take_profit_10_stop_loss_5_else_4h":   lambda r: _tp_sl_else_4h(r, 10.0, -5.0),
    "take_profit_5_stop_loss_3_else_4h":    lambda r: _tp_sl_else_4h(r, 5.0, -3.0),
    "trailing_stop_after_up_5":             lambda r: _trailing_stop_else_4h(r, 5.0),
    "trailing_stop_after_up_10":            lambda r: _trailing_stop_else_4h(r, 10.0),
}


# ---------------------------------------------------------------------------
# Apply all policies
# ---------------------------------------------------------------------------


def apply_all_policies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ret_{policy} and trigger_{policy} columns for every exit policy.
    Returns a copy; does not modify the input.
    """
    out = df.copy()
    for policy_name in POLICY_NAMES:
        fn = POLICY_FUNCTIONS[policy_name]
        rets: list[float] = []
        triggers: list[str] = []
        for _, row in out.iterrows():
            try:
                ret, trigger = fn(row)
            except Exception as exc:
                logger.warning(
                    "Policy %r failed on %r: %s",
                    policy_name,
                    row.get("symbol", "?"),
                    exc,
                )
                ret, trigger = float("nan"), "error"
            rets.append(ret)
            triggers.append(trigger)
        out[f"ret_{policy_name}"] = rets
        out[f"trigger_{policy_name}"] = triggers
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _safe_round(v: float, d: int = 4) -> float:
    return round(v, d) if not math.isnan(v) else float("nan")


def summarise_policy(
    policy_name: str,
    ret_series: pd.Series,
    mae_series: pd.Series | None = None,
    mfe_series: pd.Series | None = None,
) -> dict:
    """Compute per-policy summary statistics."""
    s = ret_series.dropna()
    n = len(s)

    def _nan_dict() -> dict:
        return {k: float("nan") for k in [
            "avg_return_pct", "median_return_pct", "win_rate",
            "success_rate_ge_5pct", "failure_rate_le_neg3pct",
            "worst_trade_pct", "best_trade_pct",
            "avg_mae_pct", "avg_mfe_pct",
            "avg_return_excl_best_pct", "median_return_excl_best_pct",
        ]}

    if n == 0:
        return {"policy": policy_name, "n_events": 0, **_nan_dict()}

    avg = s.mean()
    median = s.median()
    win_rate = (s > 0).mean()
    success_rate = (s >= 5.0).mean()
    failure_rate = (s <= -3.0).mean()
    worst = s.min()
    best = s.max()

    # Exclude the single best trade (by index of first maximum)
    best_idx = s.idxmax()
    excl_best = s.drop(index=best_idx)
    avg_excl = excl_best.mean() if len(excl_best) > 0 else float("nan")
    med_excl = excl_best.median() if len(excl_best) > 0 else float("nan")

    avg_mae = mae_series.dropna().mean() if mae_series is not None else float("nan")
    avg_mfe = mfe_series.dropna().mean() if mfe_series is not None else float("nan")

    return {
        "policy": policy_name,
        "n_events": n,
        "avg_return_pct": _safe_round(avg),
        "median_return_pct": _safe_round(median),
        "win_rate": _safe_round(win_rate),
        "success_rate_ge_5pct": _safe_round(success_rate),
        "failure_rate_le_neg3pct": _safe_round(failure_rate),
        "worst_trade_pct": _safe_round(worst),
        "best_trade_pct": _safe_round(best),
        "avg_mae_pct": _safe_round(avg_mae),
        "avg_mfe_pct": _safe_round(avg_mfe),
        "avg_return_excl_best_pct": _safe_round(avg_excl),
        "median_return_excl_best_pct": _safe_round(med_excl),
    }


def build_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-policy summary dataframe."""
    mae_4h = df.get("max_adverse_4h_pct") if "max_adverse_4h_pct" in df.columns else None
    mfe_4h = df.get("max_favorable_4h_pct") if "max_favorable_4h_pct" in df.columns else None
    mae_24h = df.get("max_adverse_24h_pct") if "max_adverse_24h_pct" in df.columns else None
    mfe_24h = df.get("max_favorable_24h_pct") if "max_favorable_24h_pct" in df.columns else None

    records = []
    for policy_name in POLICY_NAMES:
        col = f"ret_{policy_name}"
        if col not in df.columns:
            logger.warning("Policy column %r not found — skipping.", col)
            continue
        # Use 24h MAE/MFE for the 24h policy; 4h for everything else
        if policy_name == "fixed_24h_exit":
            mae, mfe = mae_24h, mfe_24h
        else:
            mae, mfe = mae_4h, mfe_4h
        records.append(summarise_policy(policy_name, df[col], mae, mfe))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Event-level output
# ---------------------------------------------------------------------------


def build_event_level_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-event output dataframe containing identity columns,
    diagnostic flags, raw return data, and per-policy realized returns.
    """
    identity = ["symbol", "snapshot_ts_utc", "entry_price"]
    flags = [c for c in _DIAGNOSTIC_FLAG_COLS if c in df.columns]
    raw = [c for c in _NUMERIC_COLS if c in df.columns]
    policy_rets = [f"ret_{p}" for p in POLICY_NAMES if f"ret_{p}" in df.columns]
    policy_triggers = [f"trigger_{p}" for p in POLICY_NAMES if f"trigger_{p}" in df.columns]

    keep = (
        [c for c in identity if c in df.columns]
        + flags
        + raw
        + policy_rets
        + policy_triggers
    )
    return df[keep].copy()


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


def print_interpretation(summary: pd.DataFrame) -> None:
    """Print the full summary table and key findings."""
    print()
    print("=" * 70)
    print("  EXIT POLICY EXPERIMENT — RESULTS")
    print("=" * 70)
    print()
    print("  *** Research only. Not a validated trading strategy. ***")
    print()

    # Full table
    print("  ─── Policy Summary ──────────────────────────────────────────────────")
    hdr = (
        f"  {'policy':<42} {'n':>4} {'avg':>7} {'med':>7} "
        f"{'wr':>6} {'sr≥5':>6} {'fr≤-3':>6} "
        f"{'best':>7} {'worst':>7} {'avg-x':>7} {'med-x':>7}"
    )
    print(hdr)
    print("  " + "─" * 106)
    for _, row in summary.iterrows():
        print(
            f"  {row['policy']:<42} {int(row['n_events']):>4} "
            f"{_fmt(row['avg_return_pct']):>7} {_fmt(row['median_return_pct']):>7} "
            f"{_fmtr(row['win_rate']):>6} {_fmtr(row['success_rate_ge_5pct']):>6} "
            f"{_fmtr(row['failure_rate_le_neg3pct']):>6} "
            f"{_fmt(row['best_trade_pct']):>7} {_fmt(row['worst_trade_pct']):>7} "
            f"{_fmt(row['avg_return_excl_best_pct']):>7} "
            f"{_fmt(row['median_return_excl_best_pct']):>7}"
        )

    print()
    print("  avg-x / med-x = average / median excluding the single best trade")

    valid = summary.dropna(
        subset=["median_return_pct", "avg_return_excl_best_pct", "failure_rate_le_neg3pct"]
    )
    if valid.empty:
        print("\n  Not enough data for interpretation.")
        return

    print()
    print("  ─── Key Findings ────────────────────────────────────────────────────")

    best_med = valid.loc[valid["median_return_pct"].idxmax()]
    print(f"  Best median return:               {best_med['policy']}  "
          f"({best_med['median_return_pct']:.2f}%)")

    best_avg_excl = valid.loc[valid["avg_return_excl_best_pct"].idxmax()]
    print(f"  Best avg excl best trade:         {best_avg_excl['policy']}  "
          f"({best_avg_excl['avg_return_excl_best_pct']:.2f}%)")

    low_fail = valid.loc[valid["failure_rate_le_neg3pct"].idxmin()]
    print(f"  Lowest failure rate (≤ -3%):      {low_fail['policy']}  "
          f"({low_fail['failure_rate_le_neg3pct']:.3f})")

    baseline_rows = summary[summary["policy"] == "fixed_4h_exit"]
    if not baseline_rows.empty:
        b = baseline_rows.iloc[0]
        print()
        print(f"  Baseline fixed_4h:  avg={b['avg_return_pct']:.2f}%  "
              f"med={b['median_return_pct']:.2f}%  "
              f"avg-excl-best={b['avg_return_excl_best_pct']:.2f}%  "
              f"fail-rate={b['failure_rate_le_neg3pct']:.3f}")
        print()

        beaten_avg = valid[
            (valid["policy"] != "fixed_4h_exit") &
            (valid["avg_return_excl_best_pct"] > b["avg_return_excl_best_pct"])
        ]["policy"].tolist()
        beaten_med = valid[
            (valid["policy"] != "fixed_4h_exit") &
            (valid["median_return_pct"] > b["median_return_pct"])
        ]["policy"].tolist()
        lower_fail = valid[
            (valid["policy"] != "fixed_4h_exit") &
            (valid["failure_rate_le_neg3pct"] < b["failure_rate_le_neg3pct"])
        ]["policy"].tolist()

        if beaten_avg:
            print(f"  Policies beating fixed_4h avg (excl best):  {', '.join(beaten_avg)}")
        else:
            print("  No policy beats fixed_4h on avg (excl best).")

        if beaten_med:
            print(f"  Policies beating fixed_4h on median:        {', '.join(beaten_med)}")
        else:
            print("  No policy beats fixed_4h on median.")

        if lower_fail:
            print(f"  Policies with lower failure rate than fixed_4h: {', '.join(lower_fail)}")
        else:
            print("  No policy has a lower failure rate than fixed_4h.")

    row_24h_rows = summary[summary["policy"] == "fixed_24h_exit"]
    if not row_24h_rows.empty and not baseline_rows.empty:
        r24 = row_24h_rows.iloc[0]
        b = baseline_rows.iloc[0]
        print()
        avg_worse = r24["avg_return_pct"] < b["avg_return_pct"]
        med_worse = r24["median_return_pct"] < b["median_return_pct"]
        if avg_worse and med_worse:
            print(
                f"  fixed_24h_exit still looks bad:  "
                f"avg={r24['avg_return_pct']:.2f}% (vs {b['avg_return_pct']:.2f}%)  "
                f"med={r24['median_return_pct']:.2f}% (vs {b['median_return_pct']:.2f}%)"
            )
        else:
            print("  fixed_24h_exit does not clearly underperform fixed_4h in this dataset.")

    print()
    print("  NOTE: Trailing stop results are approximations (no intraperiod path")
    print("  data). Trail distance = activation threshold. See module docstring.")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    event_log_path: Path = EVENT_LOG_PATH,
    summary_out_path: Path = SUMMARY_OUT_PATH,
    event_log_out_path: Path = EVENT_LOG_OUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the exit-policy experiment. Returns (summary_df, event_level_df)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 70)
    print("  Memecoin Exit-Policy Experiment")
    print("=" * 70)

    raw = load_event_log(event_log_path)
    print(f"\n  Loaded {len(raw)} rows from {event_log_path}")

    filtered = apply_entry_filters(raw)
    print(f"  After entry filters: {len(filtered)} rows")

    completed = require_completed_4h(filtered)
    print(f"  After requiring completed 4h data: {len(completed)} rows")

    if len(completed) == 0:
        print("\n  No completed events — nothing to analyse.")
        return pd.DataFrame(), pd.DataFrame()

    with_policies = apply_all_policies(completed)
    summary = build_policy_summary(with_policies)
    event_level = build_event_level_output(with_policies)

    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out_path, index=False)
    print(f"\n  -> Policy summary written to {summary_out_path}  ({len(summary)} policies)")

    event_log_out_path.parent.mkdir(parents=True, exist_ok=True)
    event_level.to_csv(event_log_out_path, index=False)
    print(f"  -> Event-level log written to {event_log_out_path}  ({len(event_level)} events)")

    print_interpretation(summary)

    return summary, event_level


if __name__ == "__main__":
    main()
