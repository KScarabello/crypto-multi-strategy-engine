"""Signal outcome evaluator for the memecoin scanner research pipeline.

Reads data/memecoin_signal_history.csv, fetches later Kraken OHLC data for each
signal, computes forward returns, max favorable/adverse excursion, and outcome
labels. Upserts results into data/memecoin_signal_outcomes.csv so that blank
outcome fields are filled in on later runs as more candle data becomes available.

Upsert semantics:
  - Unique key: (snapshot_ts_utc, pair_id, ohlc_signal_type)
  - New keys   → appended as new rows.
  - Existing keys → refreshable outcome fields updated only when the existing
    value is blank/NaN and the freshly computed value is not blank.
  - evaluated_ts_utc is always updated on every evaluation pass.

This module is research/evaluation only.
- No orders.
- No private credentials.
- No cron scheduling.

Usage:
    python -m research.memecoin_catcher.evaluate_signal_outcomes
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from research.memecoin_catcher.enrich_candidates_with_ohlc import parse_ohlc_response

LOGGER = logging.getLogger(__name__)

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

DEFAULT_INPUT_PATH = Path("data/memecoin_signal_history.csv")
DEFAULT_OUTPUT_PATH = Path("data/memecoin_signal_outcomes.csv")
DEFAULT_INTERVAL_MINUTES = 15

# Number of 15-minute candles in each measurement horizon
HORIZON_CANDLES: dict[str, int] = {
    "15m": 1,
    "1h": 4,
    "4h": 16,
    "24h": 96,
}

SIGNAL_KEY_COLUMNS: list[str] = ["snapshot_ts_utc", "pair_id", "ohlc_signal_type"]

# Columns that may be filled in on later evaluation passes as candles become available.
# evaluated_ts_utc is always refreshed; the rest are only written when blank → non-blank.
REFRESHABLE_OUTCOME_COLUMNS: list[str] = [
    "future_ret_15m_pct",
    "future_ret_1h_pct",
    "future_ret_4h_pct",
    "future_ret_24h_pct",
    "max_favorable_4h_pct",
    "max_adverse_4h_pct",
    "max_favorable_24h_pct",
    "max_adverse_24h_pct",
    "outcome_15m",
    "outcome_1h",
    "outcome_4h",
    "outcome_24h",
    "evaluated_ts_utc",
]


# ---------------------------------------------------------------------------
# Public OHLC fetcher (adds `since` support over the enrichment module's version)
# ---------------------------------------------------------------------------


def fetch_ohlc(
    pair_id: str,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    since: int | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch raw Kraken OHLC payload using public REST API only.

    *since* is an optional Unix timestamp; Kraken returns candles from that
    point onwards (up to 720 candles per request).
    """
    http = session or requests.Session()
    params: dict[str, Any] = {"pair": pair_id, "interval": int(interval)}
    if since is not None:
        params["since"] = int(since)
    response = http.get(KRAKEN_OHLC_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken OHLC API error for {pair_id}: {errors}")
    return payload


# ---------------------------------------------------------------------------
# Future-candle selection
# ---------------------------------------------------------------------------


def select_future_window(
    ohlc_df: pd.DataFrame,
    signal_ts_unix: float,
) -> pd.DataFrame:
    """Return all candles whose open time is strictly after *signal_ts_unix*.

    Candles are already sorted by time; we reset the index for positional
    lookups.
    """
    future = ohlc_df[ohlc_df["time"] > float(signal_ts_unix)].copy()
    return future.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------


def compute_forward_returns(
    future_df: pd.DataFrame,
    signal_price: float,
) -> dict[str, float]:
    """Compute percentage return at each horizon relative to *signal_price*.

    Uses the close price of the N-th future candle where N is the candle count
    for that horizon (e.g. 4 candles = 1 h at 15-min interval).  Leaves a
    field blank (NaN) when fewer than N candles are available.
    """
    results: dict[str, float] = {}
    if signal_price <= 0:
        return {f"future_ret_{h}_pct": float("nan") for h in HORIZON_CANDLES}

    for horizon, n in HORIZON_CANDLES.items():
        col = f"future_ret_{horizon}_pct"
        if len(future_df) >= n:
            close_price = float(future_df.iloc[n - 1]["close"])
            results[col] = (close_price / signal_price - 1.0) * 100.0
        else:
            results[col] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Excursion computation
# ---------------------------------------------------------------------------


def compute_excursions(
    future_df: pd.DataFrame,
    signal_price: float,
    ohlc_signal_type: str,
) -> dict[str, float]:
    """Compute max favorable/adverse excursion for 4h and 24h windows.

    For LONG_EXPLOSION and REVERSAL_WATCH:
      favorable = max high relative to signal price (positive = upside)
      adverse   = min low relative to signal price  (negative = downside)

    For DUMPING:
      favorable = signal_price / min_low - 1  (further downside is good)
      adverse   = max_high / signal_price - 1 (bounce against dump is bad)

    Returns NaN for a window if fewer than the required candles are available.
    """
    out: dict[str, float] = {
        "max_favorable_4h_pct": float("nan"),
        "max_adverse_4h_pct": float("nan"),
        "max_favorable_24h_pct": float("nan"),
        "max_adverse_24h_pct": float("nan"),
    }
    if signal_price <= 0:
        return out

    for window_label, n_candles in (("4h", 16), ("24h", 96)):
        window = future_df.head(n_candles)
        if len(window) < n_candles:
            continue
        max_high = float(window["high"].max())
        min_low = float(window["low"].min())

        if ohlc_signal_type in ("LONG_EXPLOSION", "REVERSAL_WATCH"):
            out[f"max_favorable_{window_label}_pct"] = (max_high / signal_price - 1.0) * 100.0
            out[f"max_adverse_{window_label}_pct"] = (min_low / signal_price - 1.0) * 100.0
        elif ohlc_signal_type == "DUMPING":
            if min_low > 0:
                out[f"max_favorable_{window_label}_pct"] = (signal_price / min_low - 1.0) * 100.0
            out[f"max_adverse_{window_label}_pct"] = (max_high / signal_price - 1.0) * 100.0

    return out


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def classify_outcomes(
    forward_returns: dict[str, float],
    ohlc_signal_type: str,
) -> dict[str, str]:
    """Assign SUCCESS / FAILURE / FLAT labels at each horizon.

    Thresholds:
      LONG_EXPLOSION / REVERSAL_WATCH (long-side):
        SUCCESS  : return >= +3 %
        FAILURE  : return <= -3 % (REVERSAL_WATCH: <= -5 %)
        FLAT     : otherwise

      DUMPING (short/bear side):
        SUCCESS  : return <= -3 %
        FAILURE  : return >= +3 %
        FLAT     : otherwise
    """
    outcome: dict[str, str] = {
        "outcome_15m": "",
        "outcome_1h": "",
        "outcome_4h": "",
        "outcome_24h": "",
    }
    horizon_map = {
        "outcome_15m": "future_ret_15m_pct",
        "outcome_1h": "future_ret_1h_pct",
        "outcome_4h": "future_ret_4h_pct",
        "outcome_24h": "future_ret_24h_pct",
    }

    for out_col, ret_col in horizon_map.items():
        ret = forward_returns.get(ret_col, float("nan"))
        if ret is None or (isinstance(ret, float) and np.isnan(ret)):
            continue
        ret_val = float(ret)

        if ohlc_signal_type == "LONG_EXPLOSION":
            if ret_val >= 3.0:
                outcome[out_col] = "SUCCESS"
            elif ret_val <= -3.0:
                outcome[out_col] = "FAILURE"
            else:
                outcome[out_col] = "FLAT"

        elif ohlc_signal_type == "REVERSAL_WATCH":
            if ret_val >= 3.0:
                outcome[out_col] = "SUCCESS"
            elif ret_val <= -5.0:
                outcome[out_col] = "FAILURE"
            else:
                outcome[out_col] = "FLAT"

        elif ohlc_signal_type == "DUMPING":
            if ret_val <= -3.0:
                outcome[out_col] = "SUCCESS"
            elif ret_val >= 3.0:
                outcome[out_col] = "FAILURE"
            else:
                outcome[out_col] = "FLAT"

    return outcome


# ---------------------------------------------------------------------------
# Per-row evaluation
# ---------------------------------------------------------------------------


def _blank_outcomes() -> dict[str, Any]:
    """Return all refreshable outcome columns blank (NaN for numeric, '' for labels)."""
    out: dict[str, Any] = {col: float("nan") for col in REFRESHABLE_OUTCOME_COLUMNS}
    out.update({"outcome_15m": "", "outcome_1h": "", "outcome_4h": "", "outcome_24h": "", "evaluated_ts_utc": ""})
    return out


def evaluate_signal_row(
    row: pd.Series,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    fetcher: Callable[[str, int, int | None], dict[str, Any]] = fetch_ohlc,
) -> dict[str, Any]:
    """Fetch forward OHLC and compute all outcome fields for one signal row."""
    pair_id = str(row.get("pair_id", "")).strip()
    ohlc_signal_type = str(row.get("ohlc_signal_type", ""))
    snapshot_ts = str(row.get("snapshot_ts_utc", ""))
    outcomes = _blank_outcomes()

    if not pair_id:
        return outcomes

    try:
        signal_price = float(row.get("last_price") or 0.0)
    except (TypeError, ValueError):
        return outcomes
    if signal_price <= 0:
        return outcomes

    try:
        signal_dt = datetime.strptime(snapshot_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        signal_ts_unix = int(signal_dt.timestamp())
    except (ValueError, TypeError):
        LOGGER.warning("Cannot parse snapshot_ts_utc '%s' for pair %s", snapshot_ts, pair_id)
        return outcomes

    payload = fetcher(pair_id, int(interval), signal_ts_unix)
    ohlc_df = parse_ohlc_response(payload, pair_id=pair_id)
    future_df = select_future_window(ohlc_df, float(signal_ts_unix))

    forward_rets = compute_forward_returns(future_df, signal_price=signal_price)
    excursions = compute_excursions(
        future_df, signal_price=signal_price, ohlc_signal_type=ohlc_signal_type
    )
    outcome_labels = classify_outcomes(forward_rets, ohlc_signal_type=ohlc_signal_type)

    outcomes.update(forward_rets)
    outcomes.update(excursions)
    outcomes.update(outcome_labels)
    return outcomes


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_signal_history(input_path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load signal history CSV; coerce numeric columns."""
    df = pd.read_csv(input_path, dtype=str)
    for col in (
        "last_price",
        "ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
        "volume_ratio_1h", "volume_ratio_4h",
        "long_explosion_score", "dump_score",
        "reversal_watch_score", "primary_ohlc_score",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_existing_outcomes(output_path: Path) -> pd.DataFrame:
    """Return existing outcomes DataFrame; empty frame if file is absent."""
    if not output_path.exists():
        return pd.DataFrame(columns=SIGNAL_KEY_COLUMNS)
    # dtype=object keeps columns as plain Python object arrays so numeric
    # values can be written back to them during the upsert pass.
    return pd.read_csv(output_path, dtype=object)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def _is_blank(val: Any) -> bool:
    """Return True if val represents a missing / empty value."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ("", "nan", "NaN"):
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def upsert_outcome_rows(
    existing_df: pd.DataFrame,
    evaluated_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, int]:
    """Merge freshly evaluated outcomes into the existing outcomes table.

    Rules:
    - Key: (snapshot_ts_utc, pair_id, ohlc_signal_type)
    - New key   → row appended.
    - Existing key → refreshable columns updated only when existing value is
      blank and new value is not blank.  ``evaluated_ts_utc`` is always
      refreshed regardless.

    Returns (result_df, n_added, n_updated, n_skipped).
    """
    if existing_df.empty or not all(c in existing_df.columns for c in SIGNAL_KEY_COLUMNS):
        return evaluated_df.copy().reset_index(drop=True), len(evaluated_df), 0, 0

    existing = existing_df.copy().reset_index(drop=True)

    # Build key → row-index lookup
    key_to_idx: dict[tuple[str, ...], int] = {}
    for idx, row in existing.iterrows():
        key = tuple(str(row[c]) for c in SIGNAL_KEY_COLUMNS)
        key_to_idx[key] = int(idx)

    new_rows: list[dict[str, Any]] = []
    n_added = 0
    n_updated = 0
    n_skipped = 0

    for _, erow in evaluated_df.iterrows():
        key = tuple(str(erow[c]) for c in SIGNAL_KEY_COLUMNS)

        if key not in key_to_idx:
            new_rows.append(erow.to_dict())
            n_added += 1
            continue

        idx = key_to_idx[key]
        changed = False

        for col in REFRESHABLE_OUTCOME_COLUMNS:
            if col not in erow.index:
                continue

            if col == "evaluated_ts_utc":
                # Always refresh the evaluation timestamp
                new_val = erow[col]
                if not _is_blank(new_val):
                    existing.at[idx, col] = new_val
                    changed = True
                continue

            new_val = erow[col]
            existing_val = existing.at[idx, col] if col in existing.columns else None
            # Only fill blank → non-blank; never overwrite a real value with blank
            if not _is_blank(new_val) and _is_blank(existing_val):
                existing.at[idx, col] = new_val
                changed = True

        if changed:
            n_updated += 1
        else:
            n_skipped += 1

    if new_rows:
        result = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        result = existing

    return result, n_added, n_updated, n_skipped


def write_all_outcomes(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Write the full outcomes DataFrame, overwriting any prior file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d outcome rows to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_outcome_summary(
    signals_loaded: int,
    n_added: int,
    n_updated: int,
    n_skipped: int,
    outcomes_df: pd.DataFrame,
) -> None:
    """Print evaluation run statistics and per-label outcome breakdown."""
    def _complete(col: str) -> int:
        if col not in outcomes_df.columns:
            return 0
        return int((outcomes_df[col].notna() & (outcomes_df[col] != "")).sum())

    print("\nMemecoin signal outcome evaluator")
    print(f"  signals loaded                           : {signals_loaded}")
    print(f"  new outcome rows added                   : {n_added}")
    print(f"  existing outcome rows updated            : {n_updated}")
    print(f"  existing rows skipped (no new fields)    : {n_skipped}")
    print(f"  complete 15m outcome rows                : {_complete('outcome_15m')}")
    print(f"  complete 1h  outcome rows                : {_complete('outcome_1h')}")
    print(f"  complete 4h  outcome rows                : {_complete('outcome_4h')}")
    print(f"  complete 24h outcome rows                : {_complete('outcome_24h')}")

    if outcomes_df.empty or (n_added == 0 and n_updated == 0):
        print("  (no changes this run)")
        print()
        return

    for horizon, outcome_col in (
        ("15m", "outcome_15m"),
        ("1h", "outcome_1h"),
        ("4h", "outcome_4h"),
        ("24h", "outcome_24h"),
    ):
        if outcome_col not in outcomes_df.columns:
            continue
        sub = outcomes_df[
            outcomes_df[outcome_col].notna() & (outcomes_df[outcome_col] != "")
        ]
        if sub.empty:
            continue
        print(f"\n  Outcome summary — {horizon}")
        print(f"  {'signal_type':<18} {'outcome':<10} count")
        for (sig, out_label), grp in sub.groupby(["ohlc_signal_type", outcome_col]):
            print(f"  {sig:<18} {out_label:<10} {len(grp)}")

    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_outcome_evaluation(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    fetcher: Callable[[str, int, int | None], dict[str, Any]] = fetch_ohlc,
) -> pd.DataFrame:
    """Evaluate all signals, upsert results, write full outcomes file."""
    signals = load_signal_history(input_path)
    signals_loaded = len(signals)

    existing = load_existing_outcomes(output_path)
    now_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results: list[dict[str, Any]] = []
    for _, row in signals.iterrows():
        try:
            outcome_fields = evaluate_signal_row(
                row=row, interval=interval, fetcher=fetcher
            )
        except Exception as exc:  # pragma: no cover — defensive wrapper
            LOGGER.warning("Outcome eval failed for %s: %s", row.get("pair_id"), exc)
            outcome_fields = _blank_outcomes()

        outcome_fields["evaluated_ts_utc"] = now_ts
        result_row = row.to_dict()
        result_row.update(outcome_fields)
        results.append(result_row)

    evaluated_df = pd.DataFrame(results) if results else pd.DataFrame()

    result_df, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated_df)
    write_all_outcomes(result_df, output_path)

    print_outcome_summary(signals_loaded, n_added, n_updated, n_skipped, result_df)
    return result_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_outcome_evaluation()


if __name__ == "__main__":
    main()
