"""Forward paper-trading monitor for memecoin candidate Rule 1.

Rule 1 (volume_climax_cc_false_tp10):
  Entry:
    - ohlc_signal_type == LONG_EXPLOSION
    - is_volume_climax == True
    - is_clean_continuation == False
  Exit evaluation:
    - fixed_4h       : return = future_ret_4h_pct
    - tp10_else_4h   : return = +10 if MFE_4h >= 10, else future_ret_4h_pct
    - fixed_24h      : return = future_ret_24h_pct (diagnostic only)

On each run this script:
  1. Scans the current memecoin universe for Rule 1 signals.
  2. Appends new signals to data/memecoin_forward_candidate_rule_signals.csv
     (deduplicates on symbol + signal_timestamp + candidate_rule_name).
  3. Fetches fresh OHLC for any logged signal whose outcomes are incomplete
     and fills them in (append-only outcome upsert).
  4. Prints a summary of completed outcomes if any exist.

Outputs:
  data/memecoin_forward_candidate_rule_signals.csv   — signal log
  data/memecoin_forward_candidate_rule_outcomes.csv  — outcome log
  reports/memecoin_forward_candidate_rule_summary.csv — summary metrics

Research only. No trading. No orders. No live code touched.

Usage:
  .venv/bin/python -m research.memecoin_catcher.monitor_candidate_rule_forward
  .venv/bin/python -m research.memecoin_catcher.monitor_candidate_rule_forward --evaluate-only
  .venv/bin/python -m research.memecoin_catcher.monitor_candidate_rule_forward --scan-only
  .venv/bin/python -m research.memecoin_catcher.monitor_candidate_rule_forward --summary-only
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.memecoin_catcher.backfill_recent_memecoin_signals import (
    MIN_LOOKBACK_CANDLES,
    OHLC_INTERVAL_MINUTES,
    OHLC_INTERVAL_SECONDS,
    _compute_diagnostic_flags,
    compute_features_at_index,
    compute_outcomes_at_index,
    fetch_symbol_ohlc,
    load_candidate_symbols,
)
from research.memecoin_catcher.evaluate_signal_outcomes import (
    compute_excursions,
    compute_forward_returns,
    select_future_window,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SIGNALS_PATH     = Path("data/memecoin_forward_candidate_rule_signals.csv")
OUTCOMES_PATH    = Path("data/memecoin_forward_candidate_rule_outcomes.csv")
SUMMARY_PATH     = Path("reports/memecoin_forward_candidate_rule_summary.csv")
CANDIDATES_PATH  = Path("data/memecoin_candidates_latest.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_RULE_NAME = "volume_climax_cc_false_tp10"
SIGNAL_TYPE_FILTER  = "LONG_EXPLOSION"

# Unique key for deduplication in the signal log
SIGNAL_KEY_COLS = ["symbol", "signal_timestamp", "candidate_rule_name"]

# Unique key for the outcome log
OUTCOME_KEY_COLS = ["symbol", "signal_timestamp", "candidate_rule_name"]

# Candles needed for each outcome horizon (at 15-min interval)
HORIZON_CANDLES: dict[str, int] = {"15m": 1, "1h": 4, "4h": 16, "24h": 96}

SIGNAL_COLUMNS: list[str] = [
    "run_timestamp",
    "signal_timestamp",
    "symbol",
    "pair_id",
    "signal_type",
    "primary_ohlc_score",
    "r1h_pct",
    "r4h_pct",
    "r24h_pct",
    "volume_ratio_1h",
    "spread_pct",
    "quote_volume_est",
    "is_volume_climax",
    "is_clean_continuation",
    "is_wide_spread",
    "is_terminal_spike",
    "is_overextended_24h",
    "is_rolling_over",
    "candidate_rule_name",
    "entry_price",
]

OUTCOME_COLUMNS: list[str] = [
    "symbol",
    "pair_id",
    "signal_timestamp",
    "candidate_rule_name",
    "entry_price",
    "future_ret_15m_pct",
    "future_ret_1h_pct",
    "future_ret_4h_pct",
    "future_ret_24h_pct",
    "max_favorable_4h_pct",
    "max_adverse_4h_pct",
    "max_favorable_24h_pct",
    "max_adverse_24h_pct",
    "fixed_4h_return",
    "tp10_else_4h_return",
    "fixed_24h_return",
    "outcome_complete_4h",
    "outcome_complete_24h",
    "last_evaluated_utc",
]

# ---------------------------------------------------------------------------
# Rule 1 selector
# ---------------------------------------------------------------------------


def is_rule1_signal(features: dict[str, Any]) -> bool:
    """Return True if the features row qualifies as a Rule 1 signal.

    Rule 1 (volume_climax_cc_false_tp10):
      - ohlc_signal_type == LONG_EXPLOSION
      - is_volume_climax == True
      - is_clean_continuation == False
    """
    if features.get("ohlc_signal_type") != SIGNAL_TYPE_FILTER:
        return False
    if not features.get("is_volume_climax", False):
        return False
    if features.get("is_clean_continuation", False):
        return False
    return True


# ---------------------------------------------------------------------------
# Exit policy calculations
# ---------------------------------------------------------------------------


def compute_tp10_else_4h(
    future_ret_4h_pct: float,
    max_favorable_4h_pct: float,
) -> float:
    """Return +10 if MFE >= 10, else raw 4h forward return."""
    if math.isnan(max_favorable_4h_pct) or math.isnan(future_ret_4h_pct):
        return future_ret_4h_pct  # NaN stays NaN
    if max_favorable_4h_pct >= 10.0:
        return 10.0
    return future_ret_4h_pct


# ---------------------------------------------------------------------------
# Outcome completeness helpers
# ---------------------------------------------------------------------------


def _is_complete_4h(row: pd.Series) -> bool:
    v = row.get("future_ret_4h_pct", float("nan"))
    return not (isinstance(v, float) and math.isnan(v))


def _is_complete_24h(row: pd.Series) -> bool:
    v = row.get("future_ret_24h_pct", float("nan"))
    return not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# Signal log helpers
# ---------------------------------------------------------------------------


def load_signal_log(path: Path = SIGNALS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    return df


def load_outcome_log(path: Path = OUTCOMES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for col in [
        "future_ret_15m_pct", "future_ret_1h_pct",
        "future_ret_4h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "max_favorable_24h_pct", "max_adverse_24h_pct",
        "fixed_4h_return", "tp10_else_4h_return", "fixed_24h_return",
        "entry_price",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _signal_key(symbol: str, signal_ts: str, rule: str) -> tuple[str, str, str]:
    return (str(symbol), str(signal_ts), str(rule))


def _existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty or not all(c in df.columns for c in SIGNAL_KEY_COLS):
        return set()
    return {
        _signal_key(r["symbol"], r["signal_timestamp"], r["candidate_rule_name"])
        for _, r in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Step 1: Scan current universe for Rule 1 signals
# ---------------------------------------------------------------------------


def scan_for_signals(
    symbols: list[tuple[str, str]],
    min_quote_volume: float | None = None,
    max_spread: float | None = None,
    fetcher=None,
) -> list[dict[str, Any]]:
    """
    Fetch the latest OHLC for each symbol and check whether the current
    state satisfies Rule 1.

    Returns a list of signal dicts for all matching symbols.
    """
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signals: list[dict[str, Any]] = []

    for pair_id, wsname in symbols:
        ohlc_df = fetch_symbol_ohlc(pair_id, fetcher=fetcher) if fetcher else fetch_symbol_ohlc(pair_id)
        if ohlc_df.empty or len(ohlc_df) < MIN_LOOKBACK_CANDLES:
            logger.debug("Skip %s: only %d candles.", wsname, len(ohlc_df))
            continue

        # Use the last available complete candle as the "now" snapshot
        idx = len(ohlc_df) - 1
        features = compute_features_at_index(ohlc_df, idx)
        if features is None:
            logger.debug("Skip %s: features returned None.", wsname)
            continue

        if not is_rule1_signal(features):
            continue

        sig: dict[str, Any] = {
            "run_timestamp":      run_ts,
            "signal_timestamp":   features.get("snapshot_ts_utc", ""),
            "symbol":             wsname,
            "pair_id":            pair_id,
            "signal_type":        features.get("ohlc_signal_type", ""),
            "primary_ohlc_score": features.get("primary_ohlc_score", float("nan")),
            "r1h_pct":            features.get("ret_1h_pct", float("nan")),
            "r4h_pct":            features.get("ret_4h_pct", float("nan")),
            "r24h_pct":           features.get("ret_24h_pct", float("nan")),
            "volume_ratio_1h":    features.get("volume_ratio_1h", float("nan")),
            "spread_pct":         features.get("spread_pct", float("nan")),
            "quote_volume_est":   features.get("quote_volume_est", float("nan")),
            "is_volume_climax":   features.get("is_volume_climax", False),
            "is_clean_continuation": features.get("is_clean_continuation", False),
            "is_wide_spread":     features.get("is_wide_spread", False),
            "is_terminal_spike":  features.get("danger_terminal_spike", False),
            "is_overextended_24h": features.get("is_overextended_24h", False),
            "is_rolling_over":    features.get("is_rolling_over", False),
            "candidate_rule_name": CANDIDATE_RULE_NAME,
            "entry_price":        features.get("entry_price", float("nan")),
        }
        signals.append(sig)
        logger.info("Rule 1 signal: %s  score=%.1f  VR=%.1fx  ts=%s",
                    wsname,
                    float(sig["primary_ohlc_score"]),
                    float(sig["volume_ratio_1h"]),
                    sig["signal_timestamp"])

    return signals


def append_new_signals(
    new_signals: list[dict[str, Any]],
    signal_log: pd.DataFrame,
    path: Path = SIGNALS_PATH,
) -> tuple[pd.DataFrame, int]:
    """
    Deduplicate and append new signals to the log.

    Returns (updated_df, n_appended).
    """
    existing = _existing_keys(signal_log)
    to_add: list[dict[str, Any]] = []
    for sig in new_signals:
        key = _signal_key(sig["symbol"], sig["signal_timestamp"], sig["candidate_rule_name"])
        if key not in existing:
            to_add.append(sig)
            existing.add(key)

    if not to_add:
        return signal_log, 0

    new_df = pd.DataFrame(to_add)
    for col in SIGNAL_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = float("nan")

    updated = pd.concat([signal_log, new_df[SIGNAL_COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return updated, len(to_add)


# ---------------------------------------------------------------------------
# Step 2: Evaluate / update outcomes
# ---------------------------------------------------------------------------


def evaluate_outcomes(
    signal_log: pd.DataFrame,
    outcome_log: pd.DataFrame,
    fetcher=None,
) -> tuple[pd.DataFrame, int]:
    """
    For each signal in *signal_log* whose 4h outcomes are incomplete,
    fetch fresh OHLC and attempt to fill in forward returns + exit returns.

    Returns (updated_outcome_log, n_updated).
    """
    if signal_log.empty:
        return outcome_log, 0

    # Build lookup of existing outcomes keyed by (symbol, signal_ts, rule)
    existing_outcomes: dict[tuple, dict[str, Any]] = {}
    for _, row in outcome_log.iterrows():
        key = _signal_key(
            str(row.get("symbol", "")),
            str(row.get("signal_timestamp", "")),
            str(row.get("candidate_rule_name", "")),
        )
        existing_outcomes[key] = row.to_dict()

    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated_count = 0
    rows_to_write: list[dict[str, Any]] = []

    # Re-emit all existing outcomes (we'll overwrite incomplete ones)
    rows_to_write.extend(existing_outcomes.values())

    for _, sig in signal_log.iterrows():
        sym       = str(sig.get("symbol", ""))
        pair_id   = str(sig.get("pair_id", sym.replace("/", "").upper()))
        sig_ts    = str(sig.get("signal_timestamp", ""))
        rule      = str(sig.get("candidate_rule_name", CANDIDATE_RULE_NAME))
        entry_price = _as_float(sig.get("entry_price"))
        key = _signal_key(sym, sig_ts, rule)

        existing = existing_outcomes.get(key)

        # Skip if 24h is already complete
        if existing and existing.get("outcome_complete_24h") in (True, "True", "true", 1, "1"):
            continue

        # Parse signal timestamp
        try:
            sig_ts_unix = pd.Timestamp(sig_ts, tz="UTC").timestamp()
        except Exception:
            logger.warning("Cannot parse signal_timestamp=%r for %s — skip.", sig_ts, sym)
            continue

        if entry_price is None or math.isnan(entry_price):
            logger.warning("No entry_price for %s @ %s — skip outcome.", sym, sig_ts)
            continue

        # Fetch OHLC from shortly before the signal to avoid missing edges
        since_unix = int(sig_ts_unix) - OHLC_INTERVAL_SECONDS
        try:
            ohlc_df = (fetcher(pair_id, OHLC_INTERVAL_MINUTES, since_unix)
                       if fetcher else fetch_symbol_ohlc(pair_id))
        except Exception as exc:
            logger.warning("OHLC fetch failed for %s: %s", sym, exc)
            continue

        if ohlc_df.empty:
            continue

        future_df = select_future_window(ohlc_df, sig_ts_unix)

        fwd_returns = compute_forward_returns(future_df, entry_price)
        excursions  = compute_excursions(future_df, entry_price, SIGNAL_TYPE_FILTER)

        ret_4h  = fwd_returns.get("future_ret_4h_pct",  float("nan"))
        ret_24h = fwd_returns.get("future_ret_24h_pct", float("nan"))
        mfe_4h  = excursions.get("max_favorable_4h_pct", float("nan"))
        fixed_4h  = ret_4h
        tp10_4h   = compute_tp10_else_4h(ret_4h, mfe_4h)
        fixed_24h = ret_24h
        complete_4h  = not math.isnan(ret_4h)
        complete_24h = not math.isnan(ret_24h)

        new_outcome: dict[str, Any] = {
            "symbol":             sym,
            "pair_id":            pair_id,
            "signal_timestamp":   sig_ts,
            "candidate_rule_name": rule,
            "entry_price":        entry_price,
            **fwd_returns,
            **excursions,
            "fixed_4h_return":    fixed_4h,
            "tp10_else_4h_return": tp10_4h,
            "fixed_24h_return":   fixed_24h,
            "outcome_complete_4h":  complete_4h,
            "outcome_complete_24h": complete_24h,
            "last_evaluated_utc": now_utc,
        }

        # Replace existing row if present
        rows_to_write = [r for r in rows_to_write if
                         _signal_key(str(r.get("symbol", "")),
                                     str(r.get("signal_timestamp", "")),
                                     str(r.get("candidate_rule_name", ""))) != key]
        rows_to_write.append(new_outcome)
        updated_count += 1

    updated_df = pd.DataFrame(rows_to_write)
    for col in OUTCOME_COLUMNS:
        if col not in updated_df.columns:
            updated_df[col] = float("nan")
    updated_df = updated_df[OUTCOME_COLUMNS]

    return updated_df, updated_count


def save_outcomes(outcome_df: pd.DataFrame, path: Path = OUTCOMES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Step 3: Summary metrics
# ---------------------------------------------------------------------------


def _s(v: float, d: int = 4) -> float:
    return round(float(v), d) if not (isinstance(v, float) and math.isnan(float(v))) else float("nan")


def build_summary(outcome_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-exit-policy summary metrics from the outcome log."""
    if outcome_df.empty:
        return pd.DataFrame()

    policies: dict[str, str] = {
        "fixed_4h_return":     "fixed_4h",
        "tp10_else_4h_return": "tp10_else_4h",
        "fixed_24h_return":    "fixed_24h",
    }
    rows: list[dict] = []

    for col, policy_name in policies.items():
        if col not in outcome_df.columns:
            continue
        rets = pd.to_numeric(outcome_df[col], errors="coerce").dropna()
        n_total = len(outcome_df)
        n_completed = len(rets)
        if n_completed == 0:
            continue

        excl_best = rets.drop(index=rets.idxmax()) if len(rets) > 1 else rets

        rows.append(dict(
            rule=CANDIDATE_RULE_NAME,
            exit_policy=policy_name,
            event_count=n_total,
            completed_count=n_completed,
            avg=_s(rets.mean()),
            median=_s(rets.median()),
            win_rate=_s((rets > 0).mean()),
            big_win_rate=_s((rets >= 5.0).mean()),
            fail_rate_3=_s((rets <= -3.0).mean()),
            fail_rate_5=_s((rets <= -5.0).mean()),
            worst=_s(rets.min()),
            best=_s(rets.max()),
            avg_excl_best=_s(excl_best.mean()) if len(excl_best) else float("nan"),
            median_excl_best=_s(excl_best.median()) if len(excl_best) else float("nan"),
        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _fmt(v, d: int = 2) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  —  "
    return f"{float(v):+.{d}f}%"


def print_run_summary(
    n_found: int,
    n_appended: int,
    n_outcomes_updated: int,
    outcome_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("  Memecoin Forward Monitor — Rule 1 (volume_climax_cc_false_tp10)")
    print("=" * 70)
    print("  *** Research only. No trading. No orders. ***")
    print()
    print(f"  Current Rule 1 signals found:   {n_found}")
    print(f"  New signals appended:           {n_appended}")
    print(f"  Outcomes updated this run:      {n_outcomes_updated}")

    if not outcome_df.empty:
        def _boolcnt(col: str) -> int:
            return int((outcome_df[col].isin([True, "True", "true", 1, "1"])).sum()) if col in outcome_df.columns else 0

        n_4h  = _boolcnt("outcome_complete_4h")
        n_24h = _boolcnt("outcome_complete_24h")
        print(f"  Completed 4h outcomes:          {n_4h}")
        print(f"  Completed 24h outcomes:         {n_24h}")

    if not summary_df.empty:
        print()
        print("  ─── Outcome Summary ──────────────────────────────────────────────")
        print(f"  {'exit_policy':<20} {'n_ev':>6} {'n_cmp':>6} {'avg':>7} {'med':>7} {'wr':>6} {'fail3':>6} {'avg-x':>8}")
        print("  " + "─" * 72)
        for _, r in summary_df.iterrows():
            n_ev  = int(r["event_count"])
            n_cmp = int(r["completed_count"])
            avg   = r["avg"]
            med   = r["median"]
            wr    = r["win_rate"]
            f3    = r["fail_rate_3"]
            ax    = r["avg_excl_best"]
            print(
                f"  {r['exit_policy']:<20} {n_ev:>6} {n_cmp:>6} "
                f"{_fmt(avg):>7} {_fmt(med):>7} "
                f"{wr:.3f} {f3:.3f} {_fmt(ax):>8}"
            )

    print()
    print("=" * 70)


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    evaluate_only: bool = False,
    scan_only: bool = False,
    summary_only: bool = False,
    min_quote_volume: float | None = None,
    max_spread: float | None = None,
    output_dir: Path | None = None,
    symbols_override: list[str] | None = None,
    limit_symbols: int | None = None,
    signals_path: Path = SIGNALS_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    summary_path: Path = SUMMARY_PATH,
    fetcher=None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if output_dir:
        signals_path  = output_dir / signals_path.name
        outcomes_path = output_dir / outcomes_path.name
        summary_path  = output_dir / summary_path.name

    signal_log  = load_signal_log(signals_path)
    outcome_log = load_outcome_log(outcomes_path)

    n_found     = 0
    n_appended  = 0
    n_outcomes_updated = 0

    # --- summary-only mode --------------------------------------------------
    if summary_only:
        summary = build_summary(outcome_log)
        if not summary.empty:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        print_run_summary(0, 0, 0, outcome_log, summary)
        return

    # --- scan phase ---------------------------------------------------------
    if not evaluate_only:
        symbols = load_candidate_symbols(
            CANDIDATES_PATH,
            symbols_override=symbols_override,
            limit=limit_symbols,
        )
        if not symbols:
            logger.warning("No symbols loaded — cannot scan.")
        else:
            new_signals = scan_for_signals(symbols, min_quote_volume, max_spread, fetcher=fetcher)
            n_found = len(new_signals)
            signal_log, n_appended = append_new_signals(new_signals, signal_log, signals_path)
            logger.info("Found %d Rule 1 signals; %d new appended.", n_found, n_appended)

    # --- outcome evaluation phase -------------------------------------------
    if not scan_only:
        outcome_log, n_outcomes_updated = evaluate_outcomes(signal_log, outcome_log, fetcher=fetcher)
        if n_outcomes_updated > 0:
            save_outcomes(outcome_log, outcomes_path)
            logger.info("Updated %d outcome rows.", n_outcomes_updated)

    # --- summary ------------------------------------------------------------
    summary = build_summary(outcome_log)
    if not summary.empty:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

    print_run_summary(n_found, n_appended, n_outcomes_updated, outcome_log, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward paper-trading monitor for Rule 1 memecoin signals."
    )
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip scanning; only evaluate outcomes for existing signals.")
    parser.add_argument("--scan-only", action="store_true",
                        help="Scan and append signals; skip outcome evaluation.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary from existing outcome file; no API calls.")
    parser.add_argument("--min-quote-volume", type=float, default=None)
    parser.add_argument("--max-spread", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of pair symbols to scan.")
    parser.add_argument("--limit-symbols", type=int, default=None)
    args = parser.parse_args()

    main(
        evaluate_only=args.evaluate_only,
        scan_only=args.scan_only,
        summary_only=args.summary_only,
        min_quote_volume=args.min_quote_volume,
        max_spread=args.max_spread,
        output_dir=args.output_dir,
        symbols_override=args.symbols.split(",") if args.symbols else None,
        limit_symbols=args.limit_symbols,
    )
