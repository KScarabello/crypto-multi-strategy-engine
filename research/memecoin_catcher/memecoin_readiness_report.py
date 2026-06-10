"""Genuine-events-only readiness and evidence-gate status report.

Reads data/memecoin_signal_history.csv (genuine prospective events only)
and data/memecoin_signal_outcomes.csv, then produces:

  reports/memecoin_readiness_status.csv   — machine-readable gate status
  reports/memecoin_readiness_status.md    — human-readable status report

IMPORTANT: Simulated backfill events (data/memecoin_backfilled_signal_events.csv)
are NEVER included in this report.  Evidence gates must be met with genuine
prospective events only.

This module is research-only.
- No trading.
- No orders.
- No modifications to five-coin shadow state.
- No modifications to locked evidence thresholds.

Usage:
    python -m research.memecoin_catcher.memecoin_readiness_report
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path("data/memecoin_signal_history.csv")
DEFAULT_OUTCOMES_PATH = Path("data/memecoin_signal_outcomes.csv")
DEFAULT_BACKFILL_PATH = Path("data/memecoin_backfilled_signal_events.csv")
DEFAULT_STATUS_CSV_PATH = Path("reports/memecoin_readiness_status.csv")
DEFAULT_STATUS_MD_PATH = Path("reports/memecoin_readiness_status.md")

# Scanner version for this report
SCANNER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Gate A thresholds (locked 2026-06-09 — must not be modified)
# ---------------------------------------------------------------------------

GATE_A: dict[str, object] = {
    "min_collection_days": 30,
    "min_total_events": 500,
    "min_rule1_events": 150,
    "min_symbols_rule1": 25,
    "max_single_symbol_pct": 0.20,
    "max_single_day_pct": 0.20,
    "rule1_positive_median": True,
    "rule1_positive_mean_50bps": True,
    "rule1_both_halves_positive_median": True,
    "rule1_excl_top3_symbols_positive_mean": True,
}

# Rule 1 definition
RULE1_SIGNAL_TYPE = "LONG_EXPLOSION"
RULE1_VOLUME_CLIMAX = True
RULE1_CLEAN_CONTINUATION = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_genuine_history(path: Path = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Load genuine signal history.  Returns empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    if "snapshot_ts_utc" in df.columns:
        df["_ts"] = pd.to_datetime(df["snapshot_ts_utc"], utc=True, errors="coerce")
        df["_date"] = df["_ts"].dt.date
    return df


def load_genuine_outcomes(path: Path = DEFAULT_OUTCOMES_PATH) -> pd.DataFrame:
    """Load genuine outcome table.  Returns empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    for col in (
        "future_ret_4h_pct", "future_ret_1h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "spread_pct",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_simulated_backfill(path: Path = DEFAULT_BACKFILL_PATH) -> pd.DataFrame:
    """Load simulated backfill events (for counting only — never for gates)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


# ---------------------------------------------------------------------------
# Rule 1 extraction
# ---------------------------------------------------------------------------


def extract_rule1_events(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return genuine outcome rows matching the complete Rule 1 predicate.

    Rule 1:
      ohlc_signal_type == LONG_EXPLOSION
      AND is_volume_climax == True        (volume_ratio_4h > 10)
      AND is_clean_continuation == False  (not all of ret_15m/1h/4h > 0)

    If ``is_volume_climax`` or ``is_clean_continuation`` are absent, they are
    computed from the stored detection-time raw columns rather than silently
    falling back to LONG_EXPLOSION-only.  If the raw columns are also absent
    an empty DataFrame is returned and a warning is logged.

    Returns only rows with a matured (non-NaN) 4h outcome.
    """
    if outcomes.empty:
        return pd.DataFrame()

    df = outcomes.copy()

    # Ensure numeric types for raw columns used in predicate computation
    for c in ["volume_ratio_4h", "ret_15m_pct", "ret_1h_pct", "ret_4h_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build / verify is_volume_climax
    if "is_volume_climax" not in df.columns:
        if "volume_ratio_4h" in df.columns:
            LOGGER.warning(
                "is_volume_climax absent from outcomes; computing from volume_ratio_4h > 10"
            )
            df["is_volume_climax"] = df["volume_ratio_4h"] > 10
        else:
            LOGGER.error(
                "Cannot evaluate Rule 1: is_volume_climax absent and volume_ratio_4h absent"
            )
            return pd.DataFrame()

    # Build / verify is_clean_continuation
    if "is_clean_continuation" not in df.columns:
        needed = ["ret_15m_pct", "ret_1h_pct", "ret_4h_pct"]
        if all(c in df.columns for c in needed):
            LOGGER.warning(
                "is_clean_continuation absent from outcomes; computing from ret_15m/1h/4h > 0"
            )
            df["is_clean_continuation"] = (
                (df["ret_15m_pct"] > 0) & (df["ret_1h_pct"] > 0) & (df["ret_4h_pct"] > 0)
            )
        else:
            LOGGER.error(
                "Cannot evaluate Rule 1: is_clean_continuation absent and ret columns absent"
            )
            return pd.DataFrame()

    # Apply complete Rule 1 predicate
    vc_mask = df["is_volume_climax"].astype(str).str.lower().isin({"true", "1"})
    cc_mask = df["is_clean_continuation"].astype(str).str.lower().isin({"false", "0"})
    mask = (df["ohlc_signal_type"] == RULE1_SIGNAL_TYPE) & vc_mask & cc_mask

    sub = df[mask].copy()

    if "future_ret_4h_pct" not in sub.columns:
        return pd.DataFrame()
    sub["future_ret_4h_pct"] = pd.to_numeric(sub["future_ret_4h_pct"], errors="coerce")
    complete = sub.dropna(subset=["future_ret_4h_pct"])
    return complete.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gate A evaluation
# ---------------------------------------------------------------------------


def _gate_a_pass_count(
    history: pd.DataFrame,
    outcomes: pd.DataFrame,
    rule1: pd.DataFrame,
) -> tuple[dict[str, bool], int]:
    """Evaluate each Gate A criterion; return (result_dict, pass_count)."""

    def _get_dates(df: pd.DataFrame) -> set:
        if df.empty or "_date" not in df.columns:
            return set()
        return set(df["_date"].dropna())

    genuine_dates = _get_dates(history)
    n_days = len(genuine_dates)

    # Symbols for the full genuine set
    wsname_col = "wsname" if "wsname" in history.columns else None
    n_symbols_total = history[wsname_col].nunique() if wsname_col and not history.empty else 0

    # Rule 1 stats
    n_rule1 = len(rule1)
    r1_symbols = set()
    r1_top_sym_pct = 1.0
    r1_median = float("nan")
    r1_mean_50bps = float("nan")
    r1_both_halves_pos = False
    r1_excl_top3_pos = False

    if n_rule1 > 0 and "future_ret_4h_pct" in rule1.columns:
        sym_col = "wsname" if "wsname" in rule1.columns else "pair_id" if "pair_id" in rule1.columns else None
        if sym_col:
            r1_symbols = set(rule1[sym_col].dropna())
            sym_counts = rule1.groupby(sym_col).size()
            r1_top_sym_pct = float(sym_counts.max() / n_rule1)

        ret = rule1["future_ret_4h_pct"]
        r1_median = float(ret.median())
        r1_mean_50bps = float(ret.mean()) - 0.50  # net of 50 bps

        # Time-split: first half vs second half by day_index
        if "_ts" not in rule1.columns and "snapshot_ts_utc" in rule1.columns:
            rule1 = rule1.copy()
            rule1["_ts"] = pd.to_datetime(rule1["snapshot_ts_utc"], utc=True, errors="coerce")
        if "_ts" in rule1.columns:
            rule1 = rule1.copy()
            rule1["_day_idx"] = (
                (rule1["_ts"] - rule1["_ts"].min()).dt.total_seconds() / 86400
            ).fillna(0).astype(int)
            mid = rule1["_day_idx"].median()
            first = rule1[rule1["_day_idx"] <= mid]["future_ret_4h_pct"]
            second = rule1[rule1["_day_idx"] > mid]["future_ret_4h_pct"]
            r1_both_halves_pos = (
                len(first) >= 5 and len(second) >= 5
                and float(first.median()) > 0
                and float(second.median()) > 0
            )

        # Excl top 3 symbols
        if sym_col and len(r1_symbols) > 3:
            top3 = sym_counts.nlargest(3).index
            excl = rule1[~rule1[sym_col].isin(top3)]["future_ret_4h_pct"]
            r1_excl_top3_pos = len(excl) >= 5 and float(excl.mean()) > 0

    # Day concentration (full genuine history)
    n_total_genuine = len(history)
    top_day_pct = 0.0
    if n_total_genuine > 0 and "_date" in history.columns:
        day_counts = history.groupby("_date").size()
        top_day_pct = float(day_counts.max() / n_total_genuine)

    results = {
        "collection_days_ge_30": n_days >= 30,
        "total_events_ge_500": n_total_genuine >= 500,
        "rule1_events_ge_150": n_rule1 >= 150,
        "rule1_symbols_ge_25": len(r1_symbols) >= 25,
        "rule1_top_sym_le_20pct": r1_top_sym_pct <= 0.20,
        "top_day_le_20pct": top_day_pct <= 0.20,
        "rule1_positive_median": pd.notna(r1_median) and r1_median > 0,
        "rule1_positive_mean_50bps": pd.notna(r1_mean_50bps) and r1_mean_50bps > 0,
        "rule1_both_halves_positive_median": r1_both_halves_pos,
        "rule1_excl_top3_symbols_positive_mean": r1_excl_top3_pos,
    }
    pass_count = sum(1 for v in results.values() if v)
    return results, pass_count


# ---------------------------------------------------------------------------
# Outcome completion counts
# ---------------------------------------------------------------------------


def _count_outcomes(outcomes: pd.DataFrame) -> dict[str, int]:
    """Count complete and incomplete outcomes at each horizon."""
    result: dict[str, int] = {}
    total = len(outcomes)
    for h in ("15m", "1h", "4h", "24h"):
        col = f"outcome_{h}"
        if col in outcomes.columns:
            complete = int((outcomes[col].notna() & (outcomes[col] != "")).sum())
        else:
            complete = 0
        result[f"complete_{h}"] = complete
        result[f"incomplete_{h}"] = total - complete
    return result


def _spread_availability(history: pd.DataFrame, rule1_all: pd.DataFrame, rule1_matured: pd.DataFrame) -> dict[str, object]:
    """Report spread availability in genuine history and Rule 1 events.

    ``spread_pct`` is the Kraken bid-ask spread percentage captured from the
    live ticker at detection time.  It is present for every genuine event
    (original schema field).  ``bid``, ``ask``, and ``spread_abs`` require the
    candidates pipeline to pass those fields through — they are currently absent
    from the pipeline so all values are NaN.

    This function reports ``spread_pct`` as the detection-time spread indicator.
    No spread values are back-filled from later snapshots.
    """
    result: dict[str, object] = {
        "history_spread_pct_available": 0,
        "rule1_all_spread_pct_available": 0,
        "rule1_all_spread_pct_total": 0,
        "rule1_matured_spread_pct_available": 0,
        "rule1_matured_spread_pct_total": 0,
        "rule1_matured_spread_availability_pct": 0.0,
        "spread_field_used": "spread_pct (Kraken ticker detection-time)",
        "bid_ask_in_pipeline": False,
    }
    if "spread_pct" in history.columns:
        result["history_spread_pct_available"] = int(history["spread_pct"].notna().sum())
    if not rule1_all.empty and "spread_pct" in rule1_all.columns:
        result["rule1_all_spread_pct_available"] = int(rule1_all["spread_pct"].notna().sum())
        result["rule1_all_spread_pct_total"] = len(rule1_all)
    if not rule1_matured.empty and "spread_pct" in rule1_matured.columns:
        n_avail = int(rule1_matured["spread_pct"].notna().sum())
        result["rule1_matured_spread_pct_available"] = n_avail
        result["rule1_matured_spread_pct_total"] = len(rule1_matured)
        result["rule1_matured_spread_availability_pct"] = (
            n_avail / len(rule1_matured) * 100
        ) if len(rule1_matured) > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Readiness label
# ---------------------------------------------------------------------------


def compute_readiness_label(gate_a_pass: int, gate_a_total: int) -> str:
    """Return the current data readiness label."""
    if gate_a_pass >= gate_a_total:
        return "VALIDATION_READY"
    return "EXPLORATORY_ONLY"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_status_dict(
    history: pd.DataFrame,
    outcomes: pd.DataFrame,
    simulated: pd.DataFrame,
    rule1: pd.DataFrame,
) -> dict:
    """Build a flat status dictionary for CSV and Markdown output.

    ``rule1`` contains only matured-4h Rule 1 events (from extract_rule1_events).
    We additionally compute the total Rule 1 count (all horizons) from outcomes
    to distinguish "events we can evaluate" from "all captured Rule 1 events".
    """
    gate_results, gate_pass_count = _gate_a_pass_count(history, outcomes, rule1)
    gate_total = len(gate_results)
    outcome_counts = _count_outcomes(outcomes)

    total_genuine = len(history)
    total_simulated = len(simulated)
    # Matured-4h Rule 1 events (used by Gate A)
    n_rule1_matured_4h = len(rule1)

    # Total Rule 1 events (complete predicate, any horizon including not-yet-matured)
    n_rule1_all = 0
    rule1_all = pd.DataFrame()
    if not outcomes.empty:
        outcomes_copy = outcomes.copy()
        for c in ["volume_ratio_4h", "ret_15m_pct", "ret_1h_pct", "ret_4h_pct"]:
            if c in outcomes_copy.columns:
                outcomes_copy[c] = pd.to_numeric(outcomes_copy[c], errors="coerce")
        if "is_volume_climax" in outcomes_copy.columns and "is_clean_continuation" in outcomes_copy.columns:
            vc = outcomes_copy["is_volume_climax"].astype(str).str.lower().isin({"true", "1"})
            cc = outcomes_copy["is_clean_continuation"].astype(str).str.lower().isin({"false", "0"})
            le = outcomes_copy["ohlc_signal_type"] == RULE1_SIGNAL_TYPE
            rule1_all = outcomes_copy[le & vc & cc].copy()
            n_rule1_all = len(rule1_all)
        elif "volume_ratio_4h" in outcomes_copy.columns:
            vc = outcomes_copy["volume_ratio_4h"] > 10
            needed = ["ret_15m_pct", "ret_1h_pct", "ret_4h_pct"]
            if all(c in outcomes_copy.columns for c in needed):
                cc_false = ~(
                    (outcomes_copy["ret_15m_pct"] > 0)
                    & (outcomes_copy["ret_1h_pct"] > 0)
                    & (outcomes_copy["ret_4h_pct"] > 0)
                )
                le = outcomes_copy["ohlc_signal_type"] == RULE1_SIGNAL_TYPE
                rule1_all = outcomes_copy[le & vc & cc_false].copy()
                n_rule1_all = len(rule1_all)

    spread_info = _spread_availability(history, rule1_all, rule1)

    n_symbols = history["wsname"].nunique() if "wsname" in history.columns and not history.empty else 0
    n_collection_days = history["_date"].nunique() if "_date" in history.columns and not history.empty else 0

    newest_ts = ""
    oldest_ts = ""
    rule1_date_range = ""
    if not history.empty and "_ts" in history.columns:
        ts_valid = history["_ts"].dropna()
        if not ts_valid.empty:
            newest_ts = ts_valid.max().strftime("%Y-%m-%dT%H:%M:%SZ")
            oldest_ts = ts_valid.min().strftime("%Y-%m-%dT%H:%M:%SZ")
    if not rule1.empty and "snapshot_ts_utc" in rule1.columns:
        rule1_date_range = f"{rule1['snapshot_ts_utc'].min()} -> {rule1['snapshot_ts_utc'].max()}"

    dup_count = 0
    if not history.empty and "event_id" in history.columns:
        dup_count = int(history["event_id"].dropna().duplicated().sum())

    readiness = compute_readiness_label(gate_pass_count, gate_total)

    status: dict = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "genuine_signal_rows": total_genuine,
        "simulated_backfill_rows": total_simulated,
        "newest_genuine_ts": newest_ts,
        "oldest_genuine_ts": oldest_ts,
        "n_collection_days": n_collection_days,
        "collection_day_definition": "distinct UTC calendar dates across all genuine events",
        "n_distinct_symbols": n_symbols,
        "duplicate_event_count": dup_count,
        # Rule 1 — total events vs matured-4h events (different counts)
        "rule1_total_event_count": n_rule1_all,
        "rule1_matured_4h_count": n_rule1_matured_4h,
        "rule1_date_range": rule1_date_range,
        # Backward-compat alias used by Gate A criterion label in Markdown
        "rule1_complete_event_count": n_rule1_matured_4h,
        **outcome_counts,
        **spread_info,
        "gate_a_pass_count": gate_pass_count,
        "gate_a_total": gate_total,
        "readiness_label": readiness,
        "simulated_excluded_from_gates": True,
        **{f"gate_a_{k}": v for k, v in gate_results.items()},
    }
    return status


def write_status_csv(status: dict, path: Path = DEFAULT_STATUS_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([status]).to_csv(path, index=False)
    LOGGER.info("Readiness status CSV written: %s", path)


def write_status_markdown(status: dict, path: Path = DEFAULT_STATUS_MD_PATH) -> None:
    """Write a human-readable markdown readiness report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Memecoin Catcher — Readiness Status Report",
        f"Generated: {status['generated_at']}",
        "",
        "## Data Provenance",
        "- Genuine prospective events and simulated backfill events are **separate files**.",
        "- Simulated events are **never** counted toward evidence gates.",
        "",
        "## Event Counts",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Genuine signal rows | {status['genuine_signal_rows']} |",
        f"| Simulated backfill rows (EXCLUDED from gates) | {status['simulated_backfill_rows']} |",
        f"| Newest genuine detection timestamp | {status['newest_genuine_ts']} |",
        f"| Oldest genuine detection timestamp | {status['oldest_genuine_ts']} |",
        f"| Distinct genuine collection days | {status['n_collection_days']} |",
        f"| Distinct symbols (genuine) | {status['n_distinct_symbols']} |",
        f"| Duplicate event count | {status['duplicate_event_count']} |",
        "",
        "## Outcome Completeness (Genuine Events)",
        "| Horizon | Complete | Incomplete |",
        "|---|---|---|",
    ]
    for h in ("15m", "1h", "4h", "24h"):
        c = status.get(f"complete_{h}", "N/A")
        i = status.get(f"incomplete_{h}", "N/A")
        md.append(f"| {h} | {c} | {i} |")

    md += [
        "",
        "## Rule 1 Status",
        "Definition: `LONG_EXPLOSION + is_volume_climax=True + is_clean_continuation=False`",
        "",
        "> **Note:** `is_volume_climax` = `volume_ratio_4h > 10`; "
        "`is_clean_continuation` = `ret_15m>0 AND ret_1h>0 AND ret_4h>0` — derived from detection-time columns.",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Rule 1 total events (any horizon) | {status.get('rule1_total_event_count', 0)} |",
        f"| Rule 1 matured-4h events (used by Gate A) | {status.get('rule1_matured_4h_count', 0)} |",
        f"| Rule 1 date range | {status['rule1_date_range']} |",
        f"| spread_pct available (detection-time, matured-4h events) | "
        f"{status.get('rule1_matured_spread_pct_available', 0)}/{status.get('rule1_matured_spread_pct_total', 0)} "
        f"({status.get('rule1_matured_spread_availability_pct', 0.0):.1f}%) |",
        f"| bid/ask/spread_abs in candidates pipeline | {status.get('bid_ask_in_pipeline', False)} |",
        f"| Spread field used | {status.get('spread_field_used', 'spread_pct')} |",
        "",
        "## Gate A Status (→ VALIDATION_READY)",
        f"**Pass count: {status['gate_a_pass_count']} / {status['gate_a_total']}**",
        "*(Gate A uses Rule 1 matured-4h events; simulated rows are excluded)*",
        "",
        "| Criterion | Threshold | Pass |",
        "|---|---|---|",
        f"| Collection days | ≥ 30 | {'✓' if status.get('gate_a_collection_days_ge_30') else '❌'} ({status['n_collection_days']}) |",
        f"| Total genuine events | ≥ 500 | {'✓' if status.get('gate_a_total_events_ge_500') else '❌'} ({status['genuine_signal_rows']}) |",
        f"| Rule 1 events | ≥ 150 | {'✓' if status.get('gate_a_rule1_events_ge_150') else '❌'} ({status['rule1_complete_event_count']}) |",
        f"| Rule 1 symbols | ≥ 25 | {'✓' if status.get('gate_a_rule1_symbols_ge_25') else '❌'} |",
        f"| Top symbol ≤ 20% | ≤ 20% | {'✓' if status.get('gate_a_rule1_top_sym_le_20pct') else '❌'} |",
        f"| Top day ≤ 20% | ≤ 20% | {'✓' if status.get('gate_a_top_day_le_20pct') else '❌'} |",
        f"| Rule 1 positive median | > 0 | {'✓' if status.get('gate_a_rule1_positive_median') else '❌'} |",
        f"| Rule 1 positive mean @ 50 bps | > 0 | {'✓' if status.get('gate_a_rule1_positive_mean_50bps') else '❌'} |",
        f"| Both time-split medians positive | Yes | {'✓' if status.get('gate_a_rule1_both_halves_positive_median') else '❌'} |",
        f"| Excl top-3 symbols: positive mean | > 0 | {'✓' if status.get('gate_a_rule1_excl_top3_symbols_positive_mean') else '❌'} |",
        "",
        "## Current Readiness Label",
        f"> **{status['readiness_label']}**",
        "",
        "## Safety Confirmation",
        "- Simulated backfill excluded from gates: ✓",
        "- No live execution capability: ✓",
        "- Five-coin shadow state not modified: ✓",
    ]
    path.write_text("\n".join(md) + "\n")
    LOGGER.info("Readiness status Markdown written: %s", path)


def print_readiness_summary(status: dict) -> None:
    """Print a concise console readiness summary."""
    print("\n" + "=" * 60)
    print("  MEMECOIN READINESS STATUS")
    print("=" * 60)
    print(f"  Generated              : {status['generated_at']}")
    print(f"  Genuine events         : {status['genuine_signal_rows']}")
    print(f"  Simulated (EXCL)       : {status['simulated_backfill_rows']}")
    print(f"  Newest genuine ts      : {status['newest_genuine_ts']}")
    print(f"  Collection days        : {status['n_collection_days']} (distinct UTC calendar dates)")
    print(f"  Symbols (genuine)      : {status['n_distinct_symbols']}")
    print(f"  Rule 1 total events    : {status['rule1_total_event_count']} (LE+vc=T+cc=F, any horizon)")
    print(f"  Rule 1 matured-4h      : {status['rule1_matured_4h_count']} (used by Gate A)")
    spread_avail = status.get('rule1_matured_spread_pct_available', 0)
    spread_total = status.get('rule1_matured_spread_pct_total', 0)
    print(f"  Rule 1 spread (spread_pct, detection-time): {spread_avail}/{spread_total}")
    print(f"  bid/ask in pipeline    : {status.get('bid_ask_in_pipeline', False)}")
    print(f"  Duplicates             : {status['duplicate_event_count']}")
    print(f"  Complete 4h outcomes   : {status.get('complete_4h', 0)}")
    print(f"  Incomplete 4h          : {status.get('incomplete_4h', 0)}")
    print(f"  Gate A: {status['gate_a_pass_count']}/{status['gate_a_total']} pass")
    print(f"  Readiness label        : {status['readiness_label']}")
    print(f"  Simulated excl gates   : {status['simulated_excluded_from_gates']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_readiness_report(
    history_path: Path = DEFAULT_HISTORY_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    backfill_path: Path = DEFAULT_BACKFILL_PATH,
    status_csv_path: Path = DEFAULT_STATUS_CSV_PATH,
    status_md_path: Path = DEFAULT_STATUS_MD_PATH,
) -> dict:
    """Build and write the readiness report. Returns status dict."""
    history = load_genuine_history(history_path)
    outcomes = load_genuine_outcomes(outcomes_path)
    simulated = load_simulated_backfill(backfill_path)
    rule1 = extract_rule1_events(outcomes)

    status = build_status_dict(history, outcomes, simulated, rule1)
    write_status_csv(status, status_csv_path)
    write_status_markdown(status, status_md_path)
    print_readiness_summary(status)
    return status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_readiness_report()


if __name__ == "__main__":
    main()
