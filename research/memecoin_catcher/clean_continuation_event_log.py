"""Event-level trade log for clean-continuation LONG_EXPLOSION signals.

Filters to LONG_EXPLOSION rows where is_clean_continuation == True,
excludes danger cases (is_wide_spread, danger_terminal_spike), and
produces a per-event CSV with all available return horizons.

Available horizons in the dataset: 15m, 1h, 4h, 24h.
Horizons NOT yet collected (blank in output): 2h, 8h, 12h.

Research only. No trading. No orders. No cron. No private credentials.

Usage:
    .venv/bin/python -m research.memecoin_catcher.clean_continuation_event_log
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.memecoin_catcher.analyze_signal_traits import (
    add_diagnostic_flags,
    load_long_explosion,
)

DEFAULT_OUTCOMES_PATH = Path("data/memecoin_signal_outcomes.csv")
EVENT_LOG_PATH = Path("reports/memecoin_clean_continuation_event_log.csv")

# Horizons present in the dataset
AVAILABLE_HORIZONS: list[tuple[str, str, str]] = [
    ("15m",  "future_ret_15m_pct", "outcome_15m"),
    ("1h",   "future_ret_1h_pct",  "outcome_1h"),
    ("4h",   "future_ret_4h_pct",  "outcome_4h"),
    ("24h",  "future_ret_24h_pct", "outcome_24h"),
]

# Horizons requested but not yet collected
MISSING_HORIZONS = ["2h", "8h", "12h"]

DIAGNOSTIC_FLAG_COLS = [
    "is_rolling_over",
    "is_wide_spread",
    "is_overextended_24h",
    "is_volume_climax",
    "is_clean_continuation",
    "is_pullback_rebound_candidate",
    "danger_terminal_spike",
]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_clean_continuation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean-continuation filter and exclude danger cases."""
    df = add_diagnostic_flags(df)

    mask = df["is_clean_continuation"] == True  # noqa: E712

    # Exclude wide spread
    if "is_wide_spread" in df.columns:
        mask = mask & (~df["is_wide_spread"])

    # Exclude terminal spike
    if "danger_terminal_spike" in df.columns:
        mask = mask & (~df["danger_terminal_spike"])

    return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Event log builder
# ---------------------------------------------------------------------------


def build_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-event output dataframe."""
    out = pd.DataFrame()

    # Identity columns
    out["symbol"] = df["wsname"] if "wsname" in df.columns else df.get("pair_id", pd.Series([""] * len(df)))
    out["snapshot_ts_utc"] = df.get("snapshot_ts_utc", pd.Series([pd.NaT] * len(df)))
    out["entry_price"] = pd.to_numeric(df.get("last_price", pd.Series([float("nan")] * len(df))), errors="coerce")

    # Signal-time features
    for col in ["primary_ohlc_score", "spread_pct", "ret_15m_pct", "ret_1h_pct",
                "ret_4h_pct", "ret_24h_pct", "volume_ratio_1h", "volume_ratio_4h"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = float("nan")

    # Available future return horizons
    for label, ret_col, out_col in AVAILABLE_HORIZONS:
        out[f"future_ret_{label}_pct"] = (
            pd.to_numeric(df[ret_col], errors="coerce") if ret_col in df.columns
            else float("nan")
        )
        out[f"outcome_{label}"] = df[out_col] if out_col in df.columns else pd.NA

    # Missing horizons — blank placeholder columns
    for label in MISSING_HORIZONS:
        out[f"future_ret_{label}_pct"] = float("nan")
        out[f"outcome_{label}"] = pd.NA

    # MAE / MFE
    for col in ["max_favorable_4h_pct", "max_adverse_4h_pct",
                "max_favorable_24h_pct", "max_adverse_24h_pct"]:
        out[col] = (
            pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else float("nan")
        )

    # Diagnostic flags
    for flag in DIAGNOSTIC_FLAG_COLS:
        if flag in df.columns:
            out[flag] = df[flag]
        else:
            out[flag] = pd.NA

    return out


# ---------------------------------------------------------------------------
# Holding-period summary
# ---------------------------------------------------------------------------


def _outcome_counts(series: pd.Series) -> tuple[int, int, int]:
    s = int((series == "SUCCESS").sum())
    fl = int((series == "FLAT").sum())
    f = int((series == "FAILURE").sum())
    return s, fl, f


def _rate(num: int, denom: int) -> float:
    return round(num / denom, 4) if denom > 0 else float("nan")


def build_holding_period_summary(event_log: pd.DataFrame) -> pd.DataFrame:
    """Summarise each available holding period across all clean-continuation events."""
    records = []

    for label, _ret_col, _out_col in AVAILABLE_HORIZONS:
        ret_col = f"future_ret_{label}_pct"
        out_col = f"outcome_{label}"

        sub = event_log.dropna(subset=[ret_col])
        n = len(sub)
        if n == 0:
            records.append({
                "horizon": label,
                "n_events": 0,
                "avg_return_pct": float("nan"),
                "median_return_pct": float("nan"),
                "best_return_pct": float("nan"),
                "worst_return_pct": float("nan"),
                "success_rate": float("nan"),
                "failure_rate": float("nan"),
                "data_available": False,
            })
            continue

        avg = sub[ret_col].mean()
        median = sub[ret_col].median()
        best = sub[ret_col].max()
        worst = sub[ret_col].min()

        outcome_series = sub[out_col].dropna() if out_col in sub.columns else pd.Series([], dtype=str)
        s, fl, f = _outcome_counts(outcome_series)
        n_with_outcome = s + fl + f

        records.append({
            "horizon": label,
            "n_events": n,
            "avg_return_pct": round(avg, 4),
            "median_return_pct": round(median, 4),
            "best_return_pct": round(best, 4),
            "worst_return_pct": round(worst, 4),
            "success_rate": _rate(s, n_with_outcome),
            "failure_rate": _rate(f, n_with_outcome),
            "data_available": True,
        })

    for label in MISSING_HORIZONS:
        records.append({
            "horizon": label,
            "n_events": 0,
            "avg_return_pct": float("nan"),
            "median_return_pct": float("nan"),
            "best_return_pct": float("nan"),
            "worst_return_pct": float("nan"),
            "success_rate": float("nan"),
            "failure_rate": float("nan"),
            "data_available": False,
        })

    order = ["15m", "1h", "2h", "4h", "8h", "12h", "24h"]
    df = pd.DataFrame(records)
    df["_order"] = df["horizon"].map({h: i for i, h in enumerate(order)})
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_event_log_summary(event_log: pd.DataFrame, holding_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  Clean-Continuation Event Log")
    print("=" * 70)
    print()
    print("  *** Research only. Do not treat as validated signals. ***")
    print()
    print(f"  Events after filtering: {len(event_log)}")
    print(f"  (Filter: is_clean_continuation=True, is_wide_spread=False, danger_terminal_spike=False)")
    print()
    print(f"  Note: 2h / 8h / 12h horizons are NOT yet collected in this dataset.")
    print(f"  Those columns are blank placeholders for future data.")
    print()

    print("  ─── Holding Period Summary ─────────────────────────────────────────")
    print(
        f"\n  {'horizon':<8}  {'n':>5}  {'avg':>8}  {'median':>8}  "
        f"{'best':>8}  {'worst':>8}  {'sr':>7}  {'fr':>7}  {'avail':>6}"
    )
    print("  " + "─" * 72)

    for _, row in holding_summary.iterrows():
        def _f(v, d=2):
            return f"{v:.{d}f}%" if not (isinstance(v, float) and pd.isna(v)) else "—"

        avail = "yes" if row["data_available"] else "no"
        print(
            f"  {row['horizon']:<8}  {int(row['n_events']):>5}  "
            f"{_f(row['avg_return_pct']):>8}  {_f(row['median_return_pct']):>8}  "
            f"{_f(row['best_return_pct']):>8}  {_f(row['worst_return_pct']):>8}  "
            f"{_f(row['success_rate'], 3):>7}  {_f(row['failure_rate'], 3):>7}  "
            f"{avail:>6}"
        )

    print()
    print("  ─── Top 10 Events by 4h Return ─────────────────────────────────────")
    col = "future_ret_4h_pct"
    top10 = event_log.dropna(subset=[col]).nlargest(10, col)
    if top10.empty:
        print("  Not enough data yet.")
    else:
        for _, row in top10.iterrows():
            ts = str(row.get("snapshot_ts_utc", ""))[:19]
            sym = str(row.get("symbol", ""))[:20]
            r4 = row[col]
            r24 = row.get("future_ret_24h_pct", float("nan"))
            r24_str = f"{r24:.2f}%" if not pd.isna(r24) else "—"
            print(f"    {sym:<22}  {ts}  4h={r4:>7.2f}%  24h={r24_str:>8}")

    print()
    print("  ─── Bottom 10 Events by 4h Return ──────────────────────────────────")
    bot10 = event_log.dropna(subset=[col]).nsmallest(10, col)
    if bot10.empty:
        print("  Not enough data yet.")
    else:
        for _, row in bot10.iterrows():
            ts = str(row.get("snapshot_ts_utc", ""))[:19]
            sym = str(row.get("symbol", ""))[:20]
            r4 = row[col]
            r24 = row.get("future_ret_24h_pct", float("nan"))
            r24_str = f"{r24:.2f}%" if not pd.isna(r24) else "—"
            print(f"    {sym:<22}  {ts}  4h={r4:>7.2f}%  24h={r24_str:>8}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    event_log_path: Path = EVENT_LOG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the clean-continuation event log. Returns (event_log, holding_summary)."""
    raw = load_long_explosion(outcomes_path)
    filtered = filter_clean_continuation(raw)
    event_log = build_event_log(filtered)
    holding_summary = build_holding_period_summary(event_log)

    print_event_log_summary(event_log, holding_summary)

    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    event_log.to_csv(event_log_path, index=False)
    print(f"\n  -> Event log written to {event_log_path}  ({len(event_log)} rows)")

    return event_log, holding_summary


if __name__ == "__main__":
    main()
