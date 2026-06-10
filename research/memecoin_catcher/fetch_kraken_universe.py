"""Kraken universe scanner — scanner/research infrastructure only.

Fetches all Kraken spot pairs with USD-like quote currencies,
enriches with current ticker data, and writes a snapshot CSV.

This module is NOT a trading script.
- Does not place orders.
- Does not require API keys.
- Does not import broker credentials or live config.
- Safe to run locally at any time.

Usage:
    python -m research.memecoin_catcher.fetch_kraken_universe
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
ASSET_PAIRS_URL = f"{KRAKEN_BASE_URL}/AssetPairs"
TICKER_URL = f"{KRAKEN_BASE_URL}/Ticker"

DEFAULT_SNAPSHOT_PATH = Path("data/kraken_universe_snapshot.csv")
DEFAULT_UNIVERSE_SNAPSHOT_DIR = Path("data/universe_snapshots")
UNIVERSE_META_PATH = Path("data/universe_snapshots/latest_universe_meta.json")

# Quote currencies considered "USD-like"
USD_LIKE_QUOTES: frozenset[str] = frozenset({"USD", "USDT", "USDC"})

# Kraken "status" values that mean the pair is tradeable
TRADEABLE_STATUSES: frozenset[str] = frozenset({"online", ""})

# Max pairs per single Ticker call to stay well under URL length limits
_TICKER_BATCH_SIZE = 100

OUTPUT_COLUMNS = [
    "pair_id",
    "altname",
    "wsname",
    "base",
    "quote",
    "status",
    "last_price",
    "bid",
    "ask",
    "spread_pct",
    "open_price",
    "today_return_pct",
    "volume_24h",
    "quote_volume_est",
    "eligible",
    "ineligible_reason",
]


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def fetch_asset_pairs(session: requests.Session | None = None) -> dict[str, Any]:
    """Return the raw 'result' dict from Kraken AssetPairs endpoint.

    Raises requests.HTTPError or ValueError on failure.
    """
    http = session or requests.Session()
    response = http.get(ASSET_PAIRS_URL, timeout=15)
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken AssetPairs API error: {errors}")
    return dict(payload.get("result", {}))


def fetch_ticker(
    pair_ids: list[str],
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return the raw 'result' dict from Kraken Ticker endpoint for *pair_ids*.

    Handles batching internally so callers can pass any number of pairs.
    """
    if not pair_ids:
        return {}

    http = session or requests.Session()
    combined: dict[str, Any] = {}

    for start in range(0, len(pair_ids), _TICKER_BATCH_SIZE):
        batch = pair_ids[start : start + _TICKER_BATCH_SIZE]
        params = {"pair": ",".join(batch)}
        response = http.get(TICKER_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("error", [])
        if errors:
            LOGGER.warning("Kraken Ticker API error for batch: %s", errors)
            continue
        combined.update(payload.get("result", {}))

    return combined


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _normalize_kraken_quote(quote: str) -> str:
    """Strip Kraken's fiat currency Z-prefix convention (e.g. ZUSD → USD).

    Kraken internally prefixes fiat quote currencies with 'Z' (ZUSD, ZEUR, etc.)
    and crypto bases with 'X' (XXBT, XETH, etc.).  For eligibility checks we
    want the plain ISO code.
    """
    q = quote.upper()
    if len(q) > 1 and q.startswith("Z"):
        return q[1:]
    return q


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a numeric value that may be a string, list[str], None, or numeric.

    Kraken ticker fields often arrive as e.g. ["1.23456", 1] — take index 0.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return default
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default  # guard NaN
    except (TypeError, ValueError):
        return default


def _compute_spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0.0:
        return 0.0
    return (ask - bid) / mid * 100.0


def _compute_today_return_pct(last_price: float, open_price: float) -> float:
    if open_price <= 0.0:
        return 0.0
    return (last_price / open_price - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_pairs(raw_pairs: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the raw AssetPairs result into a flat list of row dicts.

    Each row contains the pair's metadata fields plus ``eligible`` and
    ``ineligible_reason``.  Ticker fields are left as ``None`` at this stage.
    """
    rows: list[dict[str, Any]] = []
    for pair_id, meta in raw_pairs.items():
        # Skip derived info entries (e.g. ".d" pairs)
        if not isinstance(meta, dict):
            continue

        wsname: str = meta.get("wsname", "") or ""
        altname: str = meta.get("altname", "") or ""
        base: str = meta.get("base", "") or ""
        quote: str = _normalize_kraken_quote(meta.get("quote", "") or "")
        status: str = (meta.get("status") or "").lower()

        row: dict[str, Any] = {
            "pair_id": pair_id,
            "altname": altname,
            "wsname": wsname,
            "base": base,
            "quote": quote,
            "status": status,
            "last_price": None,
            "bid": None,
            "ask": None,
            "spread_pct": None,
            "open_price": None,
            "today_return_pct": None,
            "volume_24h": None,
            "quote_volume_est": None,
            "eligible": False,
            "ineligible_reason": "",
        }

        # Determine eligibility
        reason = _eligibility_reason(quote=quote, status=status, wsname=wsname)
        row["eligible"] = reason == ""
        row["ineligible_reason"] = reason

        rows.append(row)

    return rows


def _eligibility_reason(quote: str, status: str, wsname: str) -> str:
    """Return an empty string if eligible, or a short reason string if not."""
    # Skip Kraken internal fee/info entries that lack wsname
    if not wsname:
        return "no_wsname"
    quote_normalized = _normalize_kraken_quote(quote)
    if quote_normalized not in USD_LIKE_QUOTES:
        return f"quote_not_usd_like:{quote_normalized}"
    if status and status not in TRADEABLE_STATUSES:
        return f"status_not_tradeable:{status}"
    return ""


# ---------------------------------------------------------------------------
# Ticker enrichment
# ---------------------------------------------------------------------------


def enrich_with_ticker(
    rows: list[dict[str, Any]],
    ticker_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fill in ticker fields for each row in-place using ``ticker_data``.

    Kraken returns ticker keyed by pair_id *or* altname depending on the query.
    We try both.
    """
    altname_to_ticker: dict[str, Any] = {}
    for key, val in ticker_data.items():
        altname_to_ticker[key] = val

    for row in rows:
        tdata = altname_to_ticker.get(row["pair_id"]) or altname_to_ticker.get(row["altname"])
        if tdata is None:
            continue

        # c = last trade price, b = bid, a = ask, o = today open, v = volume
        last_price = _safe_float(tdata.get("c"))
        bid = _safe_float(tdata.get("b"))
        ask = _safe_float(tdata.get("a"))
        open_price = _safe_float(tdata.get("o"))
        volume_24h = _safe_float(tdata.get("v"))

        spread_pct = _compute_spread_pct(bid, ask)
        today_return_pct = _compute_today_return_pct(last_price, open_price)
        quote_volume_est = volume_24h * last_price if last_price > 0.0 and volume_24h > 0.0 else 0.0

        row["last_price"] = last_price
        row["bid"] = bid
        row["ask"] = ask
        row["spread_pct"] = spread_pct
        row["open_price"] = open_price
        row["today_return_pct"] = today_return_pct
        row["volume_24h"] = volume_24h
        row["quote_volume_est"] = quote_volume_est

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_snapshot_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows to a CSV file, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a concise console summary of the snapshot."""
    total_raw = len(rows)
    eligible = [r for r in rows if r["eligible"]]
    print(f"\nKraken universe snapshot")
    print(f"  Raw pairs total  : {total_raw}")
    print(f"  Eligible (USD-like tradeable): {len(eligible)}")

    def _fmt(row: dict[str, Any], label_key: str, value_key: str) -> str:
        label = row.get(label_key) or row.get("pair_id") or "?"
        val = row.get(value_key)
        val_str = f"{val:,.4f}" if isinstance(val, float) else str(val)
        return f"    {label:<18} {val_str}"

    # Top 20 by today_return_pct
    by_return = sorted(
        [r for r in eligible if isinstance(r.get("today_return_pct"), float)],
        key=lambda r: r["today_return_pct"],
        reverse=True,
    )
    print("\n  Top 20 by today_return_pct:")
    for r in by_return[:20]:
        print(_fmt(r, "wsname", "today_return_pct"))

    # Top 20 by quote_volume_est
    by_volume = sorted(
        [r for r in eligible if isinstance(r.get("quote_volume_est"), float) and r["quote_volume_est"] > 0],
        key=lambda r: r["quote_volume_est"],
        reverse=True,
    )
    print("\n  Top 20 by quote_volume_est (USD):")
    for r in by_volume[:20]:
        print(_fmt(r, "wsname", "quote_volume_est"))

    # Top 20 lowest spread among liquid pairs (quote_volume_est > 0)
    by_spread = sorted(
        [r for r in eligible if isinstance(r.get("spread_pct"), float) and r.get("quote_volume_est", 0) > 0],
        key=lambda r: r["spread_pct"],
    )
    print("\n  Top 20 tightest spread (liquid pairs only):")
    for r in by_spread[:20]:
        print(_fmt(r, "wsname", "spread_pct"))

    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_kraken_universe_scan(
    output_path: Path = DEFAULT_SNAPSHOT_PATH,
    session: requests.Session | None = None,
    snapshot_dir: Path = DEFAULT_UNIVERSE_SNAPSHOT_DIR,
    meta_path: Path = UNIVERSE_META_PATH,
) -> list[dict[str, Any]]:
    """Fetch, enrich, print summary and write CSV.  Returns the full rows list.

    Also writes an immutable timestamped snapshot to *snapshot_dir* and a
    latest-metadata sidecar so other pipeline stages can read the
    universe_snapshot_id and data_cutoff_timestamp.
    """
    now = datetime.now(tz=timezone.utc)
    universe_snapshot_id = "univ_" + now.strftime("%Y%m%d_%H%M%S")
    # data_cutoff_timestamp: start of the most-recently-completed 15-min candle
    interval_s = 15 * 60
    data_cutoff_unix = (int(now.timestamp()) // interval_s) * interval_s
    data_cutoff_ts = datetime.fromtimestamp(data_cutoff_unix, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    collection_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    LOGGER.info("Fetching Kraken asset pairs…")
    raw_pairs = fetch_asset_pairs(session=session)

    rows = normalize_pairs(raw_pairs)
    LOGGER.info("Total raw pairs: %d", len(rows))

    eligible_rows = [r for r in rows if r["eligible"]]
    LOGGER.info("Eligible USD-like pairs: %d", len(eligible_rows))

    eligible_ids = [r["pair_id"] for r in eligible_rows]
    LOGGER.info("Fetching ticker data for %d pairs…", len(eligible_ids))
    ticker_data = fetch_ticker(eligible_ids, session=session)

    rows = enrich_with_ticker(rows, ticker_data)

    print_summary(rows)
    write_snapshot_csv(rows, output_path)

    # --- immutable timestamped snapshot ---
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts_filename = f"kraken_universe_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    immutable_path = snapshot_dir / ts_filename
    write_snapshot_csv(rows, immutable_path)
    LOGGER.info("Immutable universe snapshot: %s", immutable_path)

    # --- metadata sidecar ---
    meta = {
        "universe_snapshot_id": universe_snapshot_id,
        "data_cutoff_timestamp": data_cutoff_ts,
        "collection_timestamp": collection_ts,
        "snapshot_file": str(immutable_path),
        "latest_file": str(output_path),
        "row_count": len(rows),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Universe metadata written: %s", meta_path)

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_kraken_universe_scan()


if __name__ == "__main__":
    main()
