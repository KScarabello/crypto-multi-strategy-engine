"""Historical backfill: simulate missed memecoin scanner snapshots from Kraken OHLC.

Purpose:
  Use recent Kraken public OHLC data (15-min candles) to replay what the
  memecoin scanner would have seen at historical timestamps. Generate more
  LONG_EXPLOSION / clean-continuation events for out-of-sample testing.

Constraints and limitations:
  - Kraken OHLC API returns at most 720 candles per request.
  - At 15-min interval, 720 candles = ~7.5 days of history.
  - --days defaults to 7; do not exceed 7 without verifying history depth.
  - Spread / bid-ask data is NOT available from OHLC history.
    is_wide_spread is set to False (unknown) and logged.
  - Scanner label (HOT_MOVER / WATCH) is approximated from ret_24h_pct.
    The live scanner uses quote_volume_est which is not in OHLC data.
    See _approx_scanner_label() for details.
  - Avoid lookahead: features at timestamp T use only candles whose close
    time <= T. Forward outcomes use candles with open time > T.
  - This is NOT a full historical backtest. Treat results as exploratory.

Research only. No trading. No orders. No live code touched.

Outputs:
  data/memecoin_backfilled_signal_events.csv
  reports/memecoin_backfill_candidate_strategy_summary.csv
  reports/memecoin_backfill_candidate_strategy_event_log.csv

Usage:
  .venv/bin/python -m research.memecoin_catcher.backfill_recent_memecoin_signals
  .venv/bin/python -m research.memecoin_catcher.backfill_recent_memecoin_signals \\
      --days 5 --snapshot-interval 4h --limit-symbols 20
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from research.memecoin_catcher.enrich_candidates_with_ohlc import (
    compute_breakout_features,
    compute_label_aware_scores,
    compute_return_features,
    compute_volume_features,
    parse_ohlc_response,
)
from research.memecoin_catcher.evaluate_signal_outcomes import (
    compute_excursions,
    compute_forward_returns,
    fetch_ohlc as _fetch_ohlc_raw,
)
from research.memecoin_catcher.candidate_strategy_comparison import (
    OUTLIER_SYMBOL,
    STRATEGY_NAMES,
    _compute_exit_return,
    _strategy_mask,
    compute_stats,
    _subsets,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES_PATH = Path("data/memecoin_candidates_latest.csv")
SIGNAL_EVENTS_PATH = Path("data/memecoin_backfilled_signal_events.csv")
SUMMARY_OUT_PATH = Path("reports/memecoin_backfill_candidate_strategy_summary.csv")
EVENT_LOG_OUT_PATH = Path("reports/memecoin_backfill_candidate_strategy_event_log.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLC_INTERVAL_MINUTES: int = 15
OHLC_INTERVAL_SECONDS: int = OHLC_INTERVAL_MINUTES * 60

# Kraken returns at most 720 candles per request (public API limit).
KRAKEN_MAX_CANDLES: int = 720

# Minimum lookback candles needed to compute all features (24h = 96 candles,
# plus 1 for the "current" close = 97).
MIN_LOOKBACK_CANDLES: int = 97

# Minimum future candles needed for full 24h outcome evaluation.
MIN_FUTURE_CANDLES_24H: int = 96

# Supported snapshot intervals and their candle step sizes.
VALID_SNAPSHOT_INTERVALS: dict[str, int] = {
    "1h": 4,
    "2h": 8,
    "4h": 16,
}

SIGNAL_EVENTS_COLUMNS: list[str] = [
    "pair_id", "symbol", "snapshot_ts_utc", "entry_price",
    "spread_pct",
    "ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
    "volume_ratio_1h", "volume_ratio_4h",
    "breakout_24h",
    "primary_ohlc_score", "ohlc_signal_type", "scanner_label",
    "is_clean_continuation", "is_wide_spread",
    "is_overextended_24h", "is_volume_climax", "is_rolling_over",
    "danger_terminal_spike",
    "future_ret_15m_pct", "future_ret_1h_pct",
    "future_ret_4h_pct", "future_ret_24h_pct",
    "max_favorable_4h_pct", "max_adverse_4h_pct",
    "max_favorable_24h_pct", "max_adverse_24h_pct",
    "data_notes",
]


# ---------------------------------------------------------------------------
# Scanner label approximation
# ---------------------------------------------------------------------------


def _approx_scanner_label(ret_24h_pct: float) -> str:
    """
    Approximate scanner label from 24h return (proxy for today_return_pct).

    The live scanner uses today_return_pct + quote_volume_est thresholds.
    In backfill mode, quote_volume_est is unavailable from OHLC; we use the
    24h return as a proxy and skip the volume threshold.

    HOT_MOVER : ret_24h_pct >= 5%
    WATCH     : 2% <= ret_24h_pct < 5%
    (blank)   : otherwise — no candidate label, no LONG_EXPLOSION possible
    """
    if math.isnan(ret_24h_pct):
        return ""
    if ret_24h_pct >= 5.0:
        return "HOT_MOVER"
    if ret_24h_pct >= 2.0:
        return "WATCH"
    return ""


# ---------------------------------------------------------------------------
# Diagnostic flags (inlined to avoid DataFrame overhead in tight inner loop)
# ---------------------------------------------------------------------------


def _compute_diagnostic_flags(features: dict[str, Any]) -> dict[str, Any]:
    """Compute the same diagnostic flags as add_diagnostic_flags() row-wise."""
    def _gt(col: str, threshold: float) -> bool:
        v = features.get(col, float("nan"))
        return bool(not math.isnan(v) and v > threshold)

    def _lt(col: str, threshold: float) -> bool:
        v = features.get(col, float("nan"))
        return bool(not math.isnan(v) and v < threshold)

    return {
        "is_clean_continuation": (
            _gt("ret_15m_pct", 0) and _gt("ret_1h_pct", 0) and _gt("ret_4h_pct", 0)
        ),
        # Spread unavailable from OHLC history — set to False (unknown, not wide)
        "is_wide_spread": False,
        "is_overextended_24h": _gt("ret_24h_pct", 25),
        "is_volume_climax": _gt("volume_ratio_4h", 10),
        "is_rolling_over": _lt("ret_1h_pct", 0) and _gt("ret_4h_pct", 0),
        # Terminal spike requires live spread data; set to False (unknown)
        "danger_terminal_spike": False,
    }


# ---------------------------------------------------------------------------
# OHLC fetching
# ---------------------------------------------------------------------------


def fetch_symbol_ohlc(
    pair_id: str,
    interval: int = OHLC_INTERVAL_MINUTES,
    since: int | None = None,
    fetcher: Callable[..., dict[str, Any]] = _fetch_ohlc_raw,
) -> pd.DataFrame:
    """Fetch and parse Kraken OHLC for one symbol. Returns empty DataFrame on error."""
    try:
        payload = fetcher(pair_id, interval, since)
        return parse_ohlc_response(payload, pair_id=pair_id)
    except Exception as exc:
        logger.warning("OHLC fetch failed for %s: %s", pair_id, exc)
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Feature computation (lookahead-free)
# ---------------------------------------------------------------------------


def compute_features_at_index(
    ohlc_df: pd.DataFrame,
    idx: int,
    interval_seconds: int = OHLC_INTERVAL_SECONDS,
) -> dict[str, Any] | None:
    """
    Compute signal features at snapshot represented by candle index *idx*.

    Lookahead guarantee:
      Only ohlc_df.iloc[0 : idx+1] is used (candles that closed at or before T).
      Snapshot timestamp T = ohlc_df.iloc[idx]["time"] + interval_seconds.

    Returns None when:
      - fewer than MIN_LOOKBACK_CANDLES candles are available (idx < 96)
      - required return/volume features are NaN after computation
      - scanner label cannot be approximated (ret_24h not a mover)
    """
    if idx < MIN_LOOKBACK_CANDLES - 1:
        return None

    available = ohlc_df.iloc[: idx + 1].copy()

    ret_features = compute_return_features(available)
    vol_features = compute_volume_features(available)
    breakout_features = compute_breakout_features(available)

    features: dict[str, Any] = {**ret_features, **vol_features, **breakout_features}

    # Check required fields are non-NaN
    required = ("ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
                "volume_ratio_1h", "volume_ratio_4h")
    features["ohlc_ready"] = all(
        not (isinstance(features.get(k), float) and math.isnan(features[k]))
        for k in required
    )
    if not features["ohlc_ready"]:
        return None

    scanner_label = _approx_scanner_label(float(features["ret_24h_pct"]))
    if not scanner_label:
        return None  # not a mover at this snapshot

    score_row = compute_label_aware_scores(features, scanner_label=scanner_label)
    features.update(score_row)
    features["scanner_label"] = scanner_label

    # Spread unavailable from OHLC history
    features["spread_pct"] = float("nan")

    # Entry price = close of the last complete candle at snapshot time
    features["entry_price"] = float(ohlc_df.iloc[idx]["close"])

    snap_unix = float(ohlc_df.iloc[idx]["time"]) + interval_seconds
    features["snapshot_ts_unix"] = snap_unix
    features["snapshot_ts_utc"] = (
        pd.Timestamp(snap_unix, unit="s", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    flags = _compute_diagnostic_flags(features)
    features.update(flags)

    return features


# ---------------------------------------------------------------------------
# Forward outcome computation
# ---------------------------------------------------------------------------


def compute_outcomes_at_index(
    ohlc_df: pd.DataFrame,
    idx: int,
    signal_price: float,
    ohlc_signal_type: str = "LONG_EXPLOSION",
) -> dict[str, Any]:
    """
    Compute forward returns and excursions using candles strictly after *idx*.

    Outcomes that require more candles than are available are left as NaN.
    """
    future_df = ohlc_df.iloc[idx + 1 :].reset_index(drop=True)
    fwd = compute_forward_returns(future_df, signal_price)
    exc = compute_excursions(future_df, signal_price, ohlc_signal_type)
    return {**fwd, **exc}


# ---------------------------------------------------------------------------
# Symbol universe loading
# ---------------------------------------------------------------------------


def load_candidate_symbols(
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    symbols_override: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """
    Return a list of (pair_id, wsname) tuples for the backfill universe.

    Priority:
      1. symbols_override: list of pair_ids or wsnames (pair_id used directly)
      2. candidates_path: existing candidates CSV
    """
    if symbols_override:
        pairs = [(s.replace("/", "").upper(), s) for s in symbols_override]
        if limit:
            pairs = pairs[:limit]
        return pairs

    if not candidates_path.exists():
        logger.warning("Candidates file not found at %s — no symbols loaded.", candidates_path)
        return []

    df = pd.read_csv(candidates_path, dtype=str)
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        pid = str(row.get("pair_id", "")).strip()
        wsn = str(row.get("wsname", "")).strip()
        if pid and wsn and pid != "nan" and wsn != "nan":
            pairs.append((pid, wsn))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for pid, wsn in pairs:
        if pid not in seen:
            seen.add(pid)
            unique.append((pid, wsn))

    if limit:
        unique = unique[:limit]
    logger.info("Loaded %d symbols from %s", len(unique), candidates_path)
    return unique


# ---------------------------------------------------------------------------
# Per-symbol simulation
# ---------------------------------------------------------------------------


def simulate_snapshots_for_symbol(
    pair_id: str,
    wsname: str,
    ohlc_df: pd.DataFrame,
    snapshot_step: int,
    interval_seconds: int = OHLC_INTERVAL_SECONDS,
    signal_type_filter: str = "LONG_EXPLOSION",
) -> list[dict[str, Any]]:
    """
    Walk through *ohlc_df* at *snapshot_step* candle intervals.

    At each valid snapshot index:
      - Compute features (lookahead-free)
      - Skip if not a candidate or not the target signal type
      - Compute forward outcomes
      - Append to event list

    Returns a list of event dicts.
    """
    events: list[dict[str, Any]] = []

    if len(ohlc_df) < MIN_LOOKBACK_CANDLES + 1:
        logger.info(
            "%s: only %d candles available (need >= %d) — skipped.",
            wsname, len(ohlc_df), MIN_LOOKBACK_CANDLES + 1,
        )
        return events

    indices = range(MIN_LOOKBACK_CANDLES - 1, len(ohlc_df) - 1, snapshot_step)

    for idx in indices:
        features = compute_features_at_index(ohlc_df, idx, interval_seconds)
        if features is None:
            continue

        if features.get("ohlc_signal_type") != signal_type_filter:
            continue

        signal_price = features["entry_price"]
        if signal_price <= 0:
            continue

        outcomes = compute_outcomes_at_index(ohlc_df, idx, signal_price, signal_type_filter)

        notes: list[str] = []
        notes.append("spread_unavailable_from_ohlc")
        notes.append("scanner_label_approx_from_ret_24h")
        if len(ohlc_df) - idx - 1 < MIN_FUTURE_CANDLES_24H:
            notes.append("partial_24h_outcomes")

        event: dict[str, Any] = {
            "pair_id": pair_id,
            "symbol": wsname,
            "data_notes": "|".join(notes),
            **features,
            **outcomes,
        }
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Full backfill run
# ---------------------------------------------------------------------------


def run_backfill(
    days: int = 7,
    snapshot_interval: str = "1h",
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    symbols_override: list[str] | None = None,
    limit_symbols: int | None = None,
    fetcher: Callable[..., dict[str, Any]] = _fetch_ohlc_raw,
) -> pd.DataFrame:
    """
    Fetch OHLC for all symbols and simulate scanner snapshots.

    Returns a DataFrame of LONG_EXPLOSION signal events with forward outcomes.

    NOTE: min_quote_volume and max_spread filters cannot be applied to OHLC
    history (bid/ask and quote volume are unavailable). These are logged.
    """
    if snapshot_interval not in VALID_SNAPSHOT_INTERVALS:
        raise ValueError(
            f"snapshot_interval must be one of {list(VALID_SNAPSHOT_INTERVALS)}, "
            f"got {snapshot_interval!r}"
        )
    snapshot_step = VALID_SNAPSHOT_INTERVALS[snapshot_interval]

    if days > 7:
        logger.warning(
            "--days=%d exceeds Kraken's ~7.5-day OHLC window at 15-min interval; "
            "some symbols may have less history than requested.",
            days,
        )

    since_ts = int(time.time()) - days * 86400
    symbol_list = load_candidate_symbols(candidates_path, symbols_override, limit_symbols)

    if not symbol_list:
        logger.warning("No symbols to backfill.")
        return pd.DataFrame()

    logger.info(
        "Backfilling %d symbols | days=%d | snapshot_interval=%s | since=%s",
        len(symbol_list), days, snapshot_interval,
        pd.Timestamp(since_ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    all_events: list[dict[str, Any]] = []

    for i, (pair_id, wsname) in enumerate(symbol_list):
        logger.info("[%d/%d] Fetching OHLC for %s (%s) …", i + 1, len(symbol_list), wsname, pair_id)
        ohlc_df = fetch_symbol_ohlc(pair_id, since=since_ts, fetcher=fetcher)

        if ohlc_df.empty:
            logger.info("  %s: empty OHLC — skipped.", wsname)
            continue

        events = simulate_snapshots_for_symbol(
            pair_id, wsname, ohlc_df, snapshot_step,
        )
        logger.info("  %s: %d candles → %d LONG_EXPLOSION events.", wsname, len(ohlc_df), len(events))
        all_events.extend(events)

    if not all_events:
        logger.warning("No LONG_EXPLOSION events found across all symbols.")
        return pd.DataFrame()

    df = pd.DataFrame(all_events)

    # Ensure all required columns exist (fill missing with NaN)
    for col in SIGNAL_EVENTS_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    logger.info("Backfill complete: %d total signal events.", len(df))
    return df


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def apply_exit_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ret_<strategy> and trigger_<strategy> columns for each candidate strategy.

    Only adds columns for rows that pass the strategy's entry filter.
    Rows outside the filter universe get NaN ret / empty trigger.
    """
    out = df.copy()

    for strategy_name in STRATEGY_NAMES:
        mask = _strategy_mask(out, strategy_name)
        rets: list[float] = []
        triggers: list[str] = []

        for i, row in out.iterrows():
            if mask.iloc[out.index.get_loc(i)]:
                try:
                    ret, trigger = _compute_exit_return(row, strategy_name)
                except Exception as exc:
                    logger.warning("Strategy %r failed for %s: %s", strategy_name, row.get("symbol"), exc)
                    ret, trigger = float("nan"), "error"
            else:
                ret, trigger = float("nan"), ""
            rets.append(ret)
            triggers.append(trigger)

        out[f"ret_{strategy_name}"] = rets
        out[f"trigger_{strategy_name}"] = triggers

    return out


def build_backfill_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-(strategy × subset) summary for backfilled events."""
    mae_4h = df.get("max_adverse_4h_pct") if "max_adverse_4h_pct" in df.columns else None
    mfe_4h = df.get("max_favorable_4h_pct") if "max_favorable_4h_pct" in df.columns else None
    mae_24h = df.get("max_adverse_24h_pct") if "max_adverse_24h_pct" in df.columns else None
    mfe_24h = df.get("max_favorable_24h_pct") if "max_favorable_24h_pct" in df.columns else None

    records: list[dict[str, Any]] = []

    for strategy_name in STRATEGY_NAMES:
        ret_col = f"ret_{strategy_name}"
        if ret_col not in df.columns:
            continue

        mae = mae_24h if strategy_name == "fixed_24h_exit" else mae_4h
        mfe = mfe_24h if strategy_name == "fixed_24h_exit" else mfe_4h

        for subset_name, sub in _subsets(df, ret_col):
            row = compute_stats(
                sub[ret_col],
                sub[mae.name] if mae is not None else pd.Series(dtype=float),
                sub[mfe.name] if mfe is not None else pd.Series(dtype=float),
                strategy_name,
                subset_name,
            )
            # Additional ALLO exclusion metrics baked into 'excl_ALLO' subset
            records.append(row)

    return pd.DataFrame(records)


def build_backfill_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event output with identity, flags, raw outcomes, and per-strategy returns.
    """
    identity = ["pair_id", "symbol", "snapshot_ts_utc", "entry_price"]
    flags = [
        "is_clean_continuation", "is_wide_spread", "is_overextended_24h",
        "is_volume_climax", "is_rolling_over", "danger_terminal_spike",
    ]
    raw = [
        "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
        "volume_ratio_1h", "volume_ratio_4h",
        "future_ret_4h_pct", "future_ret_24h_pct",
        "max_favorable_4h_pct", "max_adverse_4h_pct",
        "max_favorable_24h_pct", "max_adverse_24h_pct",
        "primary_ohlc_score", "ohlc_signal_type", "scanner_label",
        "data_notes",
    ]
    strategy_cols = (
        [f"ret_{s}" for s in STRATEGY_NAMES if f"ret_{s}" in df.columns]
        + [f"trigger_{s}" for s in STRATEGY_NAMES if f"trigger_{s}" in df.columns]
    )
    keep = (
        [c for c in identity if c in df.columns]
        + [c for c in flags if c in df.columns]
        + [c for c in raw if c in df.columns]
        + strategy_cols
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


def _print_subset_table(summary: pd.DataFrame, subset: str, title: str) -> None:
    sub = summary[summary["subset"] == subset]
    if sub.empty:
        return
    print(f"\n  ─── {title} ──────────────────────────────────────────────────────────")
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
            f"{_fmtr(row['failure_rate_le_neg3pct']):>6} "
            f"{_fmtr(row['failure_rate_le_neg5pct']):>6} "
            f"{_fmt(row['best_trade_pct']):>7} {_fmt(row['worst_trade_pct']):>7} "
            f"{_fmt(row['avg_return_excl_best_pct']):>7} "
            f"{_fmt(row['median_return_excl_best_pct']):>7}"
        )
    print()
    print("  avg-x / med-x = average / median excluding the single best trade")


def print_backfill_interpretation(
    summary: pd.DataFrame,
    n_total_events: int,
    n_symbols: int,
    days: int,
    snapshot_interval: str,
) -> None:
    """Print full results and key research findings."""
    print()
    print("=" * 70)
    print("  BACKFILL CANDIDATE STRATEGY COMPARISON — RESULTS")
    print("=" * 70)
    print()
    print(f"  Symbols scanned   : {n_symbols}")
    print(f"  History window    : {days} days")
    print(f"  Snapshot interval : {snapshot_interval}")
    print(f"  Total events      : {n_total_events}")
    print()
    print("  *** Research only. Not a validated trading strategy. ***")
    print("  *** Spread data unavailable — is_wide_spread filter not applied. ***")
    print("  *** Scanner label approximated from ret_24h_pct (no volume gate). ***")

    _print_subset_table(summary, "all", "All Events")
    _print_subset_table(summary, "excl_ALLO", f"Excluding {OUTLIER_SYMBOL} Outlier")
    _print_subset_table(summary, "excl_best_trade", "Excluding Single Best Trade")

    def _get(strategy: str, subset: str, col: str) -> float:
        row = summary[(summary["strategy"] == strategy) & (summary["subset"] == subset)]
        if row.empty:
            return float("nan")
        v = row.iloc[0][col]
        return float(v) if pd.notna(v) else float("nan")

    print()
    print("  ─── Interpretation ──────────────────────────────────────────────────")

    b_avg = _get("baseline_long_explosion_fixed_4h", "all", "avg_return_pct")
    b_med = _get("baseline_long_explosion_fixed_4h", "all", "median_return_pct")
    cc_avg = _get("clean_continuation_fixed_4h", "all", "avg_return_pct")
    cc_med = _get("clean_continuation_fixed_4h", "all", "median_return_pct")
    print(f"\n  1. clean_continuation_fixed_4h vs baseline (backfill sample):")
    print(f"     baseline: avg={_fmt(b_avg)}  med={_fmt(b_med)}")
    print(f"     clean_cc: avg={_fmt(cc_avg)}  med={_fmt(cc_med)}")
    beats = (not math.isnan(cc_avg) and not math.isnan(b_avg) and cc_avg > b_avg
             and not math.isnan(cc_med) and not math.isnan(b_med) and cc_med > b_med)
    print(f"     → {'YES, clean_continuation beats baseline on both avg and median.' if beats else 'Mixed or inconclusive — see table.'}")

    cc_x = _get("clean_continuation_fixed_4h", "excl_best_trade", "avg_return_excl_best_pct")
    tp10_x = _get("clean_continuation_tp10_else_4h", "excl_best_trade", "avg_return_excl_best_pct")
    print(f"\n  2. tp10_else_4h vs fixed_4h (excl best trade):")
    print(f"     fixed_4h avg-excl-best={_fmt(cc_x)}  tp10 avg-excl-best={_fmt(tp10_x)}")
    if not math.isnan(tp10_x) and not math.isnan(cc_x):
        print(f"     → {'tp10 BEATS fixed_4h after excluding the best trade.' if tp10_x > cc_x else 'tp10 does NOT beat fixed_4h after excluding best trade.'}")

    b_n = _get("baseline_long_explosion_fixed_4h", "all", "n_events")
    b_n_x = _get("baseline_long_explosion_fixed_4h", "excl_best_trade", "n_events")
    b_avg_x = _get("baseline_long_explosion_fixed_4h", "excl_best_trade", "avg_return_excl_best_pct")
    print(f"\n  3. Outlier dependency:")
    if not math.isnan(b_avg) and not math.isnan(b_avg_x) and abs(b_avg - b_avg_x) > 1.0:
        print(f"     Baseline avg drops from {_fmt(b_avg)} → {_fmt(b_avg_x)} excl best trade.")
        print(f"     Edge remains dependent on one or few large outliers.")
    else:
        print(f"     Baseline avg={_fmt(b_avg)}, excl-best avg={_fmt(b_avg_x)} — relatively stable.")

    r24_avg = _get("fixed_24h_exit", "all", "avg_return_pct")
    r24_med = _get("fixed_24h_exit", "all", "median_return_pct")
    print(f"\n  4. 24h holding:")
    if not math.isnan(r24_avg) and not math.isnan(b_avg):
        worse = r24_avg < b_avg
        print(
            f"     fixed_24h avg={_fmt(r24_avg)}  med={_fmt(r24_med)}"
            f" vs baseline avg={_fmt(b_avg)} — "
            f"{'still worse, 24h holding not recommended.' if worse else 'not clearly worse this sample.'}"
        )

    all_sub = summary[summary["subset"] == "all"].dropna(subset=["median_return_pct"])
    excl_sub = summary[summary["subset"] == "excl_best_trade"].dropna(subset=["avg_return_excl_best_pct"])
    if not all_sub.empty:
        bm = all_sub.loc[all_sub["median_return_pct"].idxmax()]
        print(f"\n  5. Best median: {bm['strategy']}  ({bm['median_return_pct']:.2f}%)")
    if not excl_sub.empty:
        bx = excl_sub.loc[excl_sub["avg_return_excl_best_pct"].idxmax()]
        print(f"  6. Best avg excl best: {bx['strategy']}  ({bx['avg_return_excl_best_pct']:.2f}%)")

    all_sub2 = summary[summary["subset"] == "all"].dropna(subset=["failure_rate_le_neg5pct"])
    if not all_sub2.empty:
        lf = all_sub2.loc[all_sub2["failure_rate_le_neg5pct"].idxmin()]
        print(f"  7. Lowest serious failure (≤ -5%): {lf['strategy']}  ({lf['failure_rate_le_neg5pct']:.3f})")

    print()
    print("  ─── Data Quality Notes ──────────────────────────────────────────────")
    print("  - Spread/bid-ask data not available from OHLC; is_wide_spread=False")
    print("  - Scanner label (HOT_MOVER/WATCH) approximated from ret_24h ≥ 2%/5%")
    print("  - Terminal spike detection disabled (requires live spread data)")
    print(f"  - Kraken OHLC window: {KRAKEN_MAX_CANDLES} × {OHLC_INTERVAL_MINUTES}min = ~{KRAKEN_MAX_CANDLES * OHLC_INTERVAL_MINUTES // 60 / 24:.1f} days")
    print("  - Events near the end of the window may have partial 24h outcomes")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill memecoin scanner snapshots from OHLC.")
    parser.add_argument("--days", type=int, default=7,
                        help="History window in days (max ~7 at 15-min interval).")
    parser.add_argument("--snapshot-interval", default="1h",
                        choices=list(VALID_SNAPSHOT_INTERVALS),
                        help="Interval between simulated snapshots (default: 1h).")
    parser.add_argument("--min-quote-volume", type=float, default=None,
                        help="Minimum quote volume filter (not applicable to OHLC backfill; logged only).")
    parser.add_argument("--max-spread", type=float, default=None,
                        help="Maximum spread filter (not applicable to OHLC backfill; logged only).")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of pair_id/wsnames to scan.")
    parser.add_argument("--limit-symbols", type=int, default=None,
                        help="Limit number of symbols (useful for quick testing).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for report CSVs.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backfill experiment. Returns (summary_df, event_log_df)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = _parse_args(argv)

    if args.min_quote_volume is not None:
        logger.warning(
            "--min-quote-volume is not applicable to OHLC backfill "
            "(quote_volume_est unavailable from OHLC history)."
        )
    if args.max_spread is not None:
        logger.warning(
            "--max-spread is not applicable to OHLC backfill "
            "(spread/bid-ask unavailable from OHLC history)."
        )

    symbols_override = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )

    signal_events_path = SIGNAL_EVENTS_PATH
    summary_out_path = SUMMARY_OUT_PATH
    event_log_out_path = EVENT_LOG_OUT_PATH

    if args.output_dir:
        output_dir = Path(args.output_dir)
        signal_events_path = output_dir / SIGNAL_EVENTS_PATH.name
        summary_out_path = output_dir / SUMMARY_OUT_PATH.name
        event_log_out_path = output_dir / EVENT_LOG_OUT_PATH.name

    print("\n" + "=" * 70)
    print("  Memecoin Backfill — Historical Signal Simulation")
    print("=" * 70)

    df = run_backfill(
        days=args.days,
        snapshot_interval=args.snapshot_interval,
        symbols_override=symbols_override,
        limit_symbols=args.limit_symbols,
    )

    if df.empty:
        print("\n  No events found — nothing to analyse.")
        return pd.DataFrame(), pd.DataFrame()

    n_symbols = df["symbol"].nunique() if "symbol" in df.columns else 0

    # Save raw signal events
    signal_events_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(signal_events_path, index=False)
    print(f"\n  -> Signal events written to {signal_events_path}  ({len(df)} rows)")

    # Apply exit strategies and build outputs
    with_strategies = apply_exit_strategies(df)
    summary = build_backfill_summary(with_strategies)
    event_log = build_backfill_event_log(with_strategies)

    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out_path, index=False)
    print(f"  -> Summary written to {summary_out_path}  ({len(summary)} rows)")

    event_log_out_path.parent.mkdir(parents=True, exist_ok=True)
    event_log.to_csv(event_log_out_path, index=False)
    print(f"  -> Event log written to {event_log_out_path}  ({len(event_log)} events)")

    print_backfill_interpretation(
        summary,
        n_total_events=len(df),
        n_symbols=n_symbols,
        days=args.days,
        snapshot_interval=args.snapshot_interval,
    )

    return summary, event_log


if __name__ == "__main__":
    main()
