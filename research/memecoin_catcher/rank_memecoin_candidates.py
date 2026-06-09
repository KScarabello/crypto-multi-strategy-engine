"""Memecoin / explosion-candidate ranker — scanner/research only.

Reads data/kraken_universe_snapshot.csv (produced by fetch_kraken_universe.py),
applies liquidity and quality filters, scores each pair, and writes two CSVs:

  data/memecoin_candidates_latest.csv
      Only labeled candidates (HOT_MOVER / WATCH / DUMPING), sorted by label
      priority then explosion_score descending.  Includes OHLC placeholder
      columns (null for now) ready for a future enrichment step.

  data/memecoin_candidate_diagnostics.csv
      All eligible rows after fiat/stable exclusion including those filtered out,
      with an excluded_reason column explaining why each was dropped.

This module is NOT a trading script.
- Does not place orders.
- Does not require API keys.
- Does not import broker credentials or live config.
- Safe to run locally at any time.

Usage:
    python -m research.memecoin_catcher.rank_memecoin_candidates
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = Path("data/kraken_universe_snapshot.csv")
DEFAULT_CANDIDATES_PATH = Path("data/memecoin_candidates_latest.csv")
DEFAULT_DIAGNOSTICS_PATH = Path("data/memecoin_candidate_diagnostics.csv")

# ---------------------------------------------------------------------------
# Configurable filter thresholds — edit here or override via main()
# ---------------------------------------------------------------------------
MAX_SPREAD_PCT: float = 2.0
MIN_QUOTE_VOLUME_EST: float = 10_000.0
MIN_ABS_TODAY_RETURN_PCT: float = 2.0

# Label sort priority (lower = higher priority in output)
LABEL_ORDER: dict[str, int] = {"HOT_MOVER": 0, "WATCH": 1, "DUMPING": 2}

# ---------------------------------------------------------------------------
# Fiat / stablecoin / major-coin base symbols to exclude from candidates.
# This scanner targets smaller explosive coins; majors get their own scanner.
# Easy to extend — just add to the set.
# ---------------------------------------------------------------------------
FIAT_STABLE_BASES: frozenset[str] = frozenset({
    # USD-like stablecoins
    "USD", "USDT", "USDC", "DAI", "USD1", "PYUSD", "AUSD", "RLUSD",
    "USDG", "USDQ", "USDE", "USDS", "USDD", "USAT",
    # Fiat currencies
    "EUR", "EURC", "EUROP", "EURQ", "AUD", "GBP", "JPY", "CHF", "CAD",
    "MXNB", "BRL1", "TGBP", "QCAD",
    # Kraken Z-prefixed fiat internal codes
    "ZAUD", "ZCAD", "ZEUR", "ZGBP", "ZJPY", "ZUSD",
    # Other stable/wrapped assets
    "FIDD", "CASH", "FRNT", "XAUT", "PAXG",
    # Majors — excluded here, intended for a separate majors scanner
    "XXBT", "XBT", "XETH", "ETH", "XLTC", "LTC", "XXRP", "XRP",
})

# ---------------------------------------------------------------------------
# OHLC placeholder columns — null until a future enrichment step fills them
# ---------------------------------------------------------------------------
OHLC_PLACEHOLDER_COLUMNS: list[str] = [
    "ret_15m_pct",
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
    "breakout_24h",
    "ohlc_ready",
]

# ---------------------------------------------------------------------------
# Output column schemas
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS: list[str] = [
    "pair_id", "wsname", "base", "quote",
    "last_price", "spread_pct", "today_return_pct",
    "volume_24h", "quote_volume_est",
    "explosion_score", "scanner_label",
    "is_candidate", "is_hot_mover", "is_watch", "is_dumping",
    *OHLC_PLACEHOLDER_COLUMNS,
]

DIAGNOSTICS_COLUMNS: list[str] = [
    "pair_id", "wsname", "base", "quote",
    "last_price", "spread_pct", "today_return_pct",
    "volume_24h", "quote_volume_est",
    "explosion_score", "scanner_label",
    "is_candidate", "is_hot_mover", "is_watch", "is_dumping",
    "excluded_reason",
    *OHLC_PLACEHOLDER_COLUMNS,
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_snapshot(snapshot_path: Path = DEFAULT_SNAPSHOT_PATH) -> pd.DataFrame:
    """Read the universe snapshot CSV and return a DataFrame."""
    df = pd.read_csv(snapshot_path, dtype=str)
    # Coerce numeric columns
    for col in ("last_price", "bid", "ask", "spread_pct", "open_price",
                "today_return_pct", "volume_24h", "quote_volume_est"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["eligible"] = df["eligible"].map({"True": True, "False": False})
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that pass the basic universe quality gate.

    Returns a copy with only eligible, price-complete rows.
    """
    mask = (
        (df["eligible"] == True)  # noqa: E712
        & (df["last_price"] > 0)
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["open_price"] > 0)
        & df["spread_pct"].notna()
        & df["today_return_pct"].notna()
        & df["quote_volume_est"].notna()
    )
    return df[mask].copy()


def exclude_fiat_stable(
    df: pd.DataFrame,
    fiat_stable_bases: frozenset[str] = FIAT_STABLE_BASES,
) -> pd.DataFrame:
    """Drop rows whose base symbol is in the fiat/stablecoin exclusion set."""
    mask = ~df["base"].str.upper().isin(fiat_stable_bases)
    return df[mask].copy()


def apply_liquidity_filters(
    df: pd.DataFrame,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_quote_volume_est: float = MIN_QUOTE_VOLUME_EST,
    min_abs_today_return_pct: float = MIN_ABS_TODAY_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (candidates, excluded).

    Candidates pass all three liquidity/movement thresholds.
    Excluded rows have an `excluded_reason` explaining why they were dropped.
    """
    reasons: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        if row["spread_pct"] > max_spread_pct:
            parts.append(f"spread_pct>{max_spread_pct}")
        if row["quote_volume_est"] < min_quote_volume_est:
            parts.append(f"quote_volume_est<{min_quote_volume_est}")
        if abs(row["today_return_pct"]) < min_abs_today_return_pct:
            parts.append(f"abs_return<{min_abs_today_return_pct}")
        reasons.append("|".join(parts))

    df = df.copy()
    df["excluded_reason"] = reasons

    excluded = df[df["excluded_reason"] != ""].copy()
    candidates = df[df["excluded_reason"] == ""].copy()
    return candidates, excluded


# ---------------------------------------------------------------------------
# Scoring & labelling
# ---------------------------------------------------------------------------


def compute_explosion_score(df: pd.DataFrame) -> pd.Series:
    """Return a Series of explosion scores for the rows in *df*.

    Score = today_return_pct
      +5 if quote_volume_est >= 100_000
      +3 if quote_volume_est >= 25_000  (cumulative with +5 if also >=100k)
      +2 if spread_pct <= 0.25
      +1 if spread_pct <= 0.50
      -5 if spread_pct > 1.0
    """
    score = df["today_return_pct"].copy()
    score += (df["quote_volume_est"] >= 100_000).astype(float) * 5
    score += (df["quote_volume_est"] >= 25_000).astype(float) * 3
    score += (df["spread_pct"] <= 0.25).astype(float) * 2
    score += (df["spread_pct"] <= 0.50).astype(float) * 1
    score -= (df["spread_pct"] > 1.0).astype(float) * 5
    return score


def assign_scanner_label(df: pd.DataFrame) -> pd.Series:
    """Return a Series of scanner labels for each row."""
    labels = pd.Series("", index=df.index, dtype=str)

    hot = (
        (df["today_return_pct"] >= 5)
        & (df["quote_volume_est"] >= 25_000)
        & (df["spread_pct"] <= 1.0)
    )
    watch = (
        (df["today_return_pct"] >= 2)
        & (df["quote_volume_est"] >= 10_000)
        & (df["spread_pct"] <= 2.0)
    )
    dump = (
        (df["today_return_pct"] <= -5)
        & (df["quote_volume_est"] >= 25_000)
        & (df["spread_pct"] <= 1.0)
    )

    # Order matters: more specific label wins
    labels[watch] = "WATCH"
    labels[dump] = "DUMPING"
    labels[hot] = "HOT_MOVER"
    return labels


# ---------------------------------------------------------------------------
# Status flags and OHLC placeholders
# ---------------------------------------------------------------------------


def assign_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_candidate / is_hot_mover / is_watch / is_dumping boolean columns."""
    df = df.copy()
    label = df.get("scanner_label", pd.Series("", index=df.index))
    df["is_hot_mover"] = label == "HOT_MOVER"
    df["is_watch"] = label == "WATCH"
    df["is_dumping"] = label == "DUMPING"
    df["is_candidate"] = label.isin(["HOT_MOVER", "WATCH", "DUMPING"])
    return df


def add_ohlc_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Add null OHLC columns.  A future enrichment step will populate them."""
    df = df.copy()
    for col in ("ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
                "volume_ratio_1h", "volume_ratio_4h"):
        df[col] = float("nan")
    df["breakout_24h"] = False
    df["ohlc_ready"] = False
    return df


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def build_candidates_df(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return only labeled (HOT_MOVER/WATCH/DUMPING) rows sorted by
    label priority then explosion_score descending.

    Input *candidates* should already have explosion_score and scanner_label.
    """
    df = candidates.copy()
    if "explosion_score" not in df.columns:
        df["explosion_score"] = compute_explosion_score(df)
    if "scanner_label" not in df.columns:
        df["scanner_label"] = assign_scanner_label(df)

    df = assign_status_flags(df)
    df = add_ohlc_placeholders(df)

    # Keep only rows with an active label
    df = df[df["is_candidate"]].copy()

    df["_label_order"] = df["scanner_label"].map(LABEL_ORDER).fillna(99)
    df = df.sort_values(["_label_order", "explosion_score"], ascending=[True, False])
    df = df.drop(columns=["_label_order"])
    return df.reset_index(drop=True)[CANDIDATE_COLUMNS]


def build_diagnostics_df(
    candidates: pd.DataFrame,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """Combine scored candidates and excluded rows into a diagnostics DataFrame.

    Sorted by explosion_score descending (NaN at end), then today_return_pct.
    """
    cands = candidates.copy()
    if "excluded_reason" not in cands.columns:
        cands["excluded_reason"] = ""

    excl = excluded.copy()
    if "explosion_score" not in excl.columns:
        excl["explosion_score"] = float("nan")
    if "scanner_label" not in excl.columns:
        excl["scanner_label"] = ""

    combined = pd.concat([cands, excl], ignore_index=True)
    combined = assign_status_flags(combined)
    combined = add_ohlc_placeholders(combined)

    combined = combined.sort_values(
        ["explosion_score", "today_return_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return combined[DIAGNOSTICS_COLUMNS]


def write_candidates_csv(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_CANDIDATES_PATH,
) -> None:
    """Write the candidates DataFrame to CSV, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d candidate rows to %s", len(df), output_path)


def write_diagnostics_csv(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_DIAGNOSTICS_PATH,
) -> None:
    """Write the diagnostics DataFrame to CSV, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d diagnostic rows to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_candidate_summary(
    raw_count: int,
    diagnostics_count: int,
    candidates_df: pd.DataFrame,
) -> None:
    """Print a concise ranked summary to stdout."""
    n_hot = int(candidates_df["is_hot_mover"].sum())
    n_watch = int(candidates_df["is_watch"].sum())
    n_dump = int(candidates_df["is_dumping"].sum())

    print("\nMemecoin candidate ranker")
    print(f"  Rows loaded from snapshot  : {raw_count}")
    print(f"  Diagnostics rows written   : {diagnostics_count}")
    print(f"  Candidates written         : {len(candidates_df)}")
    print(f"    HOT_MOVER                : {n_hot}")
    print(f"    WATCH                    : {n_watch}")
    print(f"    DUMPING                  : {n_dump}")

    def _row_str(row: pd.Series) -> str:
        return (
            f"    {row['wsname']:<20} "
            f"ret={row['today_return_pct']:+.2f}%  "
            f"vol=${row['quote_volume_est']:>12,.0f}  "
            f"spread={row['spread_pct']:.4f}%  "
            f"score={row['explosion_score']:.1f}  "
            f"[{row['scanner_label']}]"
        )

    hot_watch = candidates_df[candidates_df["scanner_label"].isin(["HOT_MOVER", "WATCH"])]
    dumping = candidates_df[candidates_df["scanner_label"] == "DUMPING"]

    print(f"\n  Top {min(15, len(hot_watch))} HOT_MOVER / WATCH candidates:")
    for _, r in hot_watch.head(15).iterrows():
        print(_row_str(r))

    if len(dumping):
        print(f"\n  Top {min(15, len(dumping))} DUMPING candidates:")
        for _, r in dumping.head(15).iterrows():
            print(_row_str(r))
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_candidate_ranking(
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_quote_volume_est: float = MIN_QUOTE_VOLUME_EST,
    min_abs_today_return_pct: float = MIN_ABS_TODAY_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load → filter → score → print → write two CSVs.

    Returns (candidates_df, diagnostics_df).
    """
    LOGGER.info("Loading snapshot from %s", snapshot_path)
    raw = load_snapshot(snapshot_path)
    raw_count = len(raw)

    eligible = apply_universe_filters(raw)
    after_stable = exclude_fiat_stable(eligible)

    candidates, excluded = apply_liquidity_filters(
        after_stable,
        max_spread_pct=max_spread_pct,
        min_quote_volume_est=min_quote_volume_est,
        min_abs_today_return_pct=min_abs_today_return_pct,
    )

    candidates = candidates.copy()
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)

    candidates_df = build_candidates_df(candidates)
    diagnostics_df = build_diagnostics_df(candidates, excluded)

    print_candidate_summary(raw_count, len(diagnostics_df), candidates_df)

    write_candidates_csv(candidates_df, candidates_path)
    write_diagnostics_csv(diagnostics_df, diagnostics_path)
    return candidates_df, diagnostics_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_candidate_ranking()


if __name__ == "__main__":
    main()
