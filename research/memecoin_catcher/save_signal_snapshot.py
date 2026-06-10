"""Timestamped signal snapshot logger for the memecoin OHLC enrichment pipeline.

Reads the enriched candidate file, filters to active signal rows, stamps them
with a UTC snapshot timestamp, saves an individual snapshot CSV, and appends to
a persistent history file for later outcome analysis.

This module is research logging only.
- No order placement.
- No private credentials.
- No cron scheduling.

Usage:
    python -m research.memecoin_catcher.save_signal_snapshot
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

SCANNER_VERSION = "1.0.0"

DEFAULT_INPUT_PATH = Path("data/memecoin_candidates_ohlc_latest.csv")
DEFAULT_SNAPSHOT_DIR = Path("data/memecoin_signal_snapshots")
DEFAULT_HISTORY_PATH = Path("data/memecoin_signal_history.csv")
UNIVERSE_META_PATH = Path("data/universe_snapshots/latest_universe_meta.json")

ACTIVE_SIGNAL_TYPES = frozenset({"LONG_EXPLOSION", "REVERSAL_WATCH", "DUMPING"})

# Spread filter threshold — must match rank_memecoin_candidates.MAX_SPREAD_PCT
SPREAD_FILTER_MAX_PCT: float = 2.0
# Minimum quote volume for liquidity flag — must match rank_memecoin_candidates.MIN_QUOTE_VOLUME_EST
LIQUIDITY_MIN_QVOL: float = 10_000.0

SNAPSHOT_COLUMNS: list[str] = [
    # --- identity ---
    "event_id",
    "snapshot_ts_utc",
    "pair_id",
    "wsname",
    "base",
    "quote",
    "scanner_label",
    "ohlc_signal_type",
    # --- Rule 1 predicates (derived from detection-time features) ---
    "is_volume_climax",
    "is_clean_continuation",
    # --- price / spread ---
    "last_price",
    "bid",
    "ask",
    "spread_abs",
    "spread_pct",
    "spread_passes_filter",
    # --- liquidity ---
    "quote_volume_est",
    "liquidity_flag",
    # --- market features ---
    "today_return_pct",
    "ret_15m_pct",
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
    "breakout_24h",
    "long_explosion_score",
    "dump_score",
    "reversal_watch_score",
    "primary_ohlc_score",
    # --- provenance ---
    "event_source",
    "collection_mode",
    "is_simulated",
    "detection_timestamp",
    "data_cutoff_timestamp",
    "collection_timestamp",
    "scanner_version",
    "universe_snapshot_id",
    "source_snapshot_path",
]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(tz=timezone.utc)


def format_snapshot_ts(dt: datetime) -> str:
    """Format a UTC datetime as ISO-8601 string (no microseconds)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_snapshot_filename(dt: datetime) -> str:
    """Format a UTC datetime as the snapshot filename timestamp component."""
    return dt.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def compute_event_id(
    pair_id: str,
    ohlc_signal_type: str,
    scanner_version: str,
    data_cutoff_timestamp: str,
) -> str:
    """Return a 16-character hex event ID stable across repeated runs.

    Based on SHA-256 of four key fields.  Uses data_cutoff_timestamp (the
    15-minute OHLC boundary floor) so that two runs within the same 15-minute
    window produce identical IDs for the same pair/signal/version — enabling
    correct idempotent deduplication.  detection_timestamp (wall clock) is
    intentionally excluded because it changes on every run.
    """
    raw = "|".join([
        str(pair_id),
        str(ohlc_signal_type),
        str(scanner_version),
        str(data_cutoff_timestamp),
    ]).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_universe_meta(meta_path: Path = UNIVERSE_META_PATH) -> dict:
    """Load the latest universe metadata sidecar, or return safe defaults."""
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return {
        "universe_snapshot_id": "univ_unknown",
        "data_cutoff_timestamp": "",
        "collection_timestamp": "",
        "snapshot_file": "",
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_enriched_candidates(input_path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the enriched candidates CSV; coerce numeric/bool columns."""
    df = pd.read_csv(input_path, dtype=str)
    for col in (
        "last_price", "spread_pct", "today_return_pct",
        "ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
        "volume_ratio_1h", "volume_ratio_4h",
        "long_explosion_score", "dump_score", "reversal_watch_score",
        "primary_ohlc_score",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "breakout_24h" in df.columns:
        df["breakout_24h"] = df["breakout_24h"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    if "ohlc_signal_type" not in df.columns:
        df["ohlc_signal_type"] = ""
    return df


# ---------------------------------------------------------------------------
# Filtering and stamping
# ---------------------------------------------------------------------------


def filter_active_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a recognised ohlc_signal_type."""
    mask = df["ohlc_signal_type"].isin(ACTIVE_SIGNAL_TYPES)
    return df[mask].copy()


def _compute_spread_fields(row: pd.Series) -> dict:
    """Derive spread_abs, spread_passes_filter, liquidity_flag from row data."""
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    spread_pct = pd.to_numeric(row.get("spread_pct"), errors="coerce")
    qvol = pd.to_numeric(row.get("quote_volume_est"), errors="coerce")

    import math
    spread_abs = float(ask - bid) if (pd.notna(ask) and pd.notna(bid)) else float("nan")
    passes = bool(pd.notna(spread_pct) and spread_pct <= SPREAD_FILTER_MAX_PCT)
    liquid = bool(pd.notna(qvol) and not math.isnan(float(qvol)) and float(qvol) >= LIQUIDITY_MIN_QVOL)
    return {
        "spread_abs": spread_abs,
        "spread_passes_filter": passes,
        "liquidity_flag": liquid,
    }


def stamp_snapshot(
    df: pd.DataFrame,
    snapshot_ts: str,
    universe_meta: dict | None = None,
    collection_ts: str | None = None,
) -> pd.DataFrame:
    """Add snapshot_ts_utc, provenance, spread fields, Rule-1 predicates, and event_id;
    then reorder to SNAPSHOT_COLUMNS schema."""
    meta = universe_meta or {}
    universe_snapshot_id = meta.get("universe_snapshot_id", "univ_unknown")
    data_cutoff_ts = meta.get("data_cutoff_timestamp", "")
    source_snapshot_path = meta.get("snapshot_file", "")
    coll_ts = collection_ts or snapshot_ts

    out = df.copy()
    out["snapshot_ts_utc"] = snapshot_ts

    # Spread-derived fields
    spread_rows = out.apply(_compute_spread_fields, axis=1)
    for col in ("spread_abs", "spread_passes_filter", "liquidity_flag"):
        out[col] = [r[col] for r in spread_rows]

    # Rule 1 predicate columns — derived from detection-time OHLC features.
    # These are NOT new strategy parameters: the formulas are locked in
    # analyze_signal_traits.py and backfill_recent_memecoin_signals.py.
    # is_volume_climax  = volume_ratio_4h > 10
    # is_clean_continuation = ret_15m_pct > 0 AND ret_1h_pct > 0 AND ret_4h_pct > 0
    if "is_volume_climax" not in out.columns:
        if "volume_ratio_4h" in out.columns:
            vc = pd.to_numeric(out["volume_ratio_4h"], errors="coerce")
            out["is_volume_climax"] = vc > 10
        else:
            out["is_volume_climax"] = None
    if "is_clean_continuation" not in out.columns:
        needed = ["ret_15m_pct", "ret_1h_pct", "ret_4h_pct"]
        if all(c in out.columns for c in needed):
            r15 = pd.to_numeric(out["ret_15m_pct"], errors="coerce")
            r1h = pd.to_numeric(out["ret_1h_pct"], errors="coerce")
            r4h = pd.to_numeric(out["ret_4h_pct"], errors="coerce")
            out["is_clean_continuation"] = (r15 > 0) & (r1h > 0) & (r4h > 0)
        else:
            out["is_clean_continuation"] = None

    # Provenance fields
    out["event_source"] = "GENUINE_PROSPECTIVE"
    out["collection_mode"] = "LIVE"
    out["is_simulated"] = False
    out["detection_timestamp"] = snapshot_ts
    out["data_cutoff_timestamp"] = data_cutoff_ts
    out["collection_timestamp"] = coll_ts
    out["scanner_version"] = SCANNER_VERSION
    out["universe_snapshot_id"] = universe_snapshot_id
    out["source_snapshot_path"] = source_snapshot_path

    # Stable event ID — computed last, after all provenance fields are set.
    # Uses data_cutoff_timestamp (15-min floor) not detection_timestamp (wall clock)
    # so two runs within the same OHLC window produce the same ID for the same event.
    out["event_id"] = out.apply(
        lambda r: compute_event_id(
            pair_id=str(r.get("pair_id", "")),
            ohlc_signal_type=str(r.get("ohlc_signal_type", "")),
            scanner_version=r["scanner_version"],
            data_cutoff_timestamp=r["data_cutoff_timestamp"],
        ),
        axis=1,
    )

    missing = [c for c in SNAPSHOT_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = None
    return out[SNAPSHOT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_snapshot_file(
    df: pd.DataFrame,
    snapshot_dir: Path,
    snapshot_ts_str: str,
) -> Path:
    """Write the snapshot CSV and return the file path."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"memecoin_signals_{snapshot_ts_str}.csv"
    path = snapshot_dir / filename
    df.to_csv(path, index=False)
    LOGGER.info("Snapshot written: %s (%d rows)", path, len(df))
    return path


def append_to_history(
    df: pd.DataFrame,
    history_path: Path,
    snapshot_ts: str,
) -> str:
    """Append snapshot rows to the history CSV.

    Creates the file (with header) if it does not exist.
    Skips the append if snapshot_ts_utc is already present in the file.
    If event_id is available, also skips individual rows whose event_id already
    exists and logs them as SKIPPED.

    Uses pandas concat to handle schema evolution: if the new rows have more
    columns than the existing file, the file is rewritten with the unified
    column set (old rows get NaN for new columns).

    Returns "created", "appended", or "duplicate_skipped".
    """
    if not history_path.exists():
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(history_path, index=False)
        LOGGER.info("History file created: %s", history_path)
        return "created"

    existing_full = pd.read_csv(history_path, dtype=str)

    if "snapshot_ts_utc" in existing_full.columns:
        # Whole-timestamp-level dedup (fast path: exact same snapshot_ts)
        if snapshot_ts in existing_full["snapshot_ts_utc"].values:
            LOGGER.warning(
                "snapshot_ts_utc %s already in history; skipping append.", snapshot_ts
            )
            return "duplicate_skipped"

        # Event-ID-level dedup (catches same event re-collected at a different ts)
        if "event_id" in df.columns and "event_id" in existing_full.columns:
            existing_ids = set(existing_full["event_id"].dropna().astype(str))
            dup_mask = df["event_id"].astype(str).isin(existing_ids)
            n_dup = int(dup_mask.sum())
            if n_dup:
                for _, dup_row in df[dup_mask].iterrows():
                    LOGGER.warning(
                        "SKIPPED duplicate event_id=%s pair=%s ts=%s",
                        dup_row.get("event_id"),
                        dup_row.get("wsname"),
                        dup_row.get("snapshot_ts_utc"),
                    )
                    print(
                        f"  SKIPPED duplicate event_id={dup_row.get('event_id')} "
                        f"pair={dup_row.get('wsname')} ts={dup_row.get('snapshot_ts_utc')}"
                    )
                df = df[~dup_mask]
                if df.empty:
                    return "duplicate_skipped"

    # Use pandas concat for schema-safe append: handles column evolution so
    # old rows keep NaN for new columns rather than producing a malformed CSV.
    combined = pd.concat([existing_full, df], ignore_index=True, sort=False)
    combined.to_csv(history_path, index=False)
    LOGGER.info("Appended %d rows to history: %s", len(df), history_path)
    return "appended"


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_snapshot_summary(
    total_loaded: int,
    df: pd.DataFrame,
    snapshot_path: Path,
    history_action: str,
) -> None:
    """Print required console summary."""
    print("\nMemecoin signal snapshot")
    print(f"  rows loaded from enriched file   : {total_loaded}")
    print(f"  signal rows saved                : {len(df)}")
    print(f"  snapshot file                    : {snapshot_path}")
    print(f"  history file                     : {history_action}")

    ranked = df[pd.notna(df["primary_ohlc_score"])].sort_values(
        "primary_ohlc_score", ascending=False
    )
    n = min(10, len(ranked))
    print(f"\n  Top {n} saved rows by primary_ohlc_score")
    for _, row in ranked.head(n).iterrows():
        wsname = row.get("wsname", "?")
        score = row.get("primary_ohlc_score", float("nan"))
        signal = row.get("ohlc_signal_type", "")
        ret_1h = row.get("ret_1h_pct", float("nan"))
        vol_1h = row.get("volume_ratio_1h", float("nan"))
        ts = row.get("snapshot_ts_utc", "")
        try:
            score_str = f"{float(score):>8.3f}"
        except (TypeError, ValueError):
            score_str = f"{'?':>8}"
        print(
            f"    {wsname:<18} score={score_str} r1h={ret_1h:>7.2f}% "
            f"vr1h={vol_1h:>6.2f} [{signal}] ts={ts}"
        )
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_signal_snapshot(
    input_path: Path = DEFAULT_INPUT_PATH,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    history_path: Path = DEFAULT_HISTORY_PATH,
    now: datetime | None = None,
    universe_meta_path: Path = UNIVERSE_META_PATH,
) -> tuple[pd.DataFrame, Path, str]:
    """Full pipeline: load → filter → stamp → write snapshot + history.

    Returns (signal_df, snapshot_path, history_action).
    """
    dt = now if now is not None else utcnow()
    snapshot_ts = format_snapshot_ts(dt)
    snapshot_ts_str = format_snapshot_filename(dt)
    collection_ts = snapshot_ts

    universe_meta = load_universe_meta(universe_meta_path)

    raw = load_enriched_candidates(input_path)
    total_loaded = len(raw)

    signals = filter_active_signals(raw)
    stamped = stamp_snapshot(
        signals,
        snapshot_ts=snapshot_ts,
        universe_meta=universe_meta,
        collection_ts=collection_ts,
    )

    snapshot_path = write_snapshot_file(stamped, snapshot_dir, snapshot_ts_str)
    history_action = append_to_history(stamped, history_path, snapshot_ts)

    print_snapshot_summary(total_loaded, stamped, snapshot_path, history_action)
    return stamped, snapshot_path, history_action


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_signal_snapshot()


if __name__ == "__main__":
    main()
