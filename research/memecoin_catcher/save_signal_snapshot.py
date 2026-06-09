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

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/memecoin_candidates_ohlc_latest.csv")
DEFAULT_SNAPSHOT_DIR = Path("data/memecoin_signal_snapshots")
DEFAULT_HISTORY_PATH = Path("data/memecoin_signal_history.csv")

ACTIVE_SIGNAL_TYPES = frozenset({"LONG_EXPLOSION", "REVERSAL_WATCH", "DUMPING"})

SNAPSHOT_COLUMNS: list[str] = [
    "snapshot_ts_utc",
    "pair_id",
    "wsname",
    "base",
    "quote",
    "scanner_label",
    "ohlc_signal_type",
    "last_price",
    "spread_pct",
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


def stamp_snapshot(df: pd.DataFrame, snapshot_ts: str) -> pd.DataFrame:
    """Add snapshot_ts_utc column and reorder to SNAPSHOT_COLUMNS schema."""
    out = df.copy()
    out["snapshot_ts_utc"] = snapshot_ts
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
    Returns "created" or "appended".
    """
    if not history_path.exists():
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(history_path, index=False)
        LOGGER.info("History file created: %s", history_path)
        return "created"

    existing = pd.read_csv(history_path, dtype=str, nrows=0)
    if "snapshot_ts_utc" in existing.columns:
        existing_full = pd.read_csv(
            history_path, dtype={"snapshot_ts_utc": str}, usecols=["snapshot_ts_utc"]
        )
        if snapshot_ts in existing_full["snapshot_ts_utc"].values:
            LOGGER.warning(
                "snapshot_ts_utc %s already in history; skipping append.", snapshot_ts
            )
            return "duplicate_skipped"

    df.to_csv(history_path, mode="a", header=False, index=False)
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
) -> tuple[pd.DataFrame, Path, str]:
    """Full pipeline: load → filter → stamp → write snapshot + history.

    Returns (signal_df, snapshot_path, history_action).
    """
    dt = now if now is not None else utcnow()
    snapshot_ts = format_snapshot_ts(dt)
    snapshot_ts_str = format_snapshot_filename(dt)

    raw = load_enriched_candidates(input_path)
    total_loaded = len(raw)

    signals = filter_active_signals(raw)
    stamped = stamp_snapshot(signals, snapshot_ts=snapshot_ts)

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
