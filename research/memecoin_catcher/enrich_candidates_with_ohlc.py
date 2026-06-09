"""Enrich memecoin candidates with Kraken OHLC-derived explosion features.

Reads clean candidates from data/memecoin_candidates_latest.csv, fetches recent
public Kraken OHLC candles per candidate pair, computes short-window return and
volume features, and writes enriched output.

This module is scanner/research only.
- No private credentials.
- No order placement.
- No cron scheduling.

Usage:
    python -m research.memecoin_catcher.enrich_candidates_with_ohlc
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
KRAKEN_OHLC_URL = f"{KRAKEN_BASE_URL}/OHLC"

DEFAULT_INPUT_PATH = Path("data/memecoin_candidates_latest.csv")
DEFAULT_OUTPUT_PATH = Path("data/memecoin_candidates_latest.csv")
DEFAULT_AUDIT_OUTPUT_PATH = Path("data/memecoin_candidates_ohlc_latest.csv")

DEFAULT_INTERVAL_MINUTES = 15
ELIGIBLE_LABELS = frozenset({"HOT_MOVER", "WATCH", "DUMPING"})

OHLC_FEATURE_COLUMNS: list[str] = [
    "ret_15m_pct",
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
    "breakout_24h",
    "ohlc_ready",
    "long_explosion_score",
    "dump_score",
    "reversal_watch_score",
    "primary_ohlc_score",
    "ohlc_signal_type",
    "ohlc_explosion_score",
]

SIGNAL_PRIORITY: dict[str, int] = {
    "LONG_EXPLOSION": 0,
    "REVERSAL_WATCH": 1,
    "DUMPING": 2,
}


def fetch_ohlc(
    pair_id: str,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch raw Kraken OHLC payload for a pair and interval using public API."""
    http = session or requests.Session()
    response = http.get(
        KRAKEN_OHLC_URL,
        params={"pair": pair_id, "interval": int(interval)},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken OHLC API error for {pair_id}: {errors}")
    return payload


def parse_ohlc_response(payload: dict[str, Any], pair_id: str) -> pd.DataFrame:
    """Parse Kraken OHLC response payload into a normalized DataFrame."""
    result = payload.get("result", {})
    rows = result.get(pair_id)

    if rows is None:
        for key, value in result.items():
            if key == "last":
                continue
            rows = value
            break

    if rows is None:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    ohlc = pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    for col in ("time", "open", "high", "low", "close", "volume"):
        ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")
    ohlc = ohlc.dropna(subset=["time", "close", "high", "volume"]).copy()
    ohlc = ohlc.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return ohlc[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def select_complete_candles(
    ohlc_df: pd.DataFrame,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    now_ts: float | None = None,
) -> pd.DataFrame:
    """Drop an in-progress final candle if its window has not closed yet."""
    if ohlc_df.empty:
        return ohlc_df

    interval_seconds = int(interval_minutes) * 60
    current_ts = float(now_ts) if now_ts is not None else float(pd.Timestamp.now(tz="UTC").timestamp())
    last_candle_start = float(ohlc_df.iloc[-1]["time"])
    if last_candle_start + interval_seconds > current_ts:
        return ohlc_df.iloc[:-1].copy()
    return ohlc_df.copy()


def _safe_pct_return(closes: pd.Series, lookback: int) -> float:
    if len(closes) <= lookback:
        return float("nan")
    prev = float(closes.iloc[-(lookback + 1)])
    curr = float(closes.iloc[-1])
    if prev <= 0.0:
        return float("nan")
    return (curr / prev - 1.0) * 100.0


def compute_return_features(ohlc_df: pd.DataFrame) -> dict[str, float]:
    """Compute return features from complete OHLC candles."""
    closes = ohlc_df["close"] if not ohlc_df.empty else pd.Series(dtype=float)
    return {
        "ret_15m_pct": _safe_pct_return(closes, lookback=1),
        "ret_1h_pct": _safe_pct_return(closes, lookback=4),
        "ret_4h_pct": _safe_pct_return(closes, lookback=16),
        "ret_24h_pct": _safe_pct_return(closes, lookback=96),
    }


def _median_prior_rolling_volume(volumes: pd.Series, window: int) -> float:
    if len(volumes) < 2 * window:
        return float("nan")
    prior = volumes.iloc[:-window]
    prior_roll = prior.rolling(window=window).sum().dropna()
    if prior_roll.empty:
        return float("nan")
    median_vol = float(prior_roll.median())
    if median_vol <= 0.0:
        return float("nan")
    return median_vol


def compute_volume_features(ohlc_df: pd.DataFrame) -> dict[str, float]:
    """Compute recent-vs-baseline volume ratio features."""
    volumes = ohlc_df["volume"] if not ohlc_df.empty else pd.Series(dtype=float)

    ratio_1h = float("nan")
    ratio_4h = float("nan")

    if len(volumes) >= 4:
        recent_1h = float(volumes.iloc[-4:].sum())
        median_1h = _median_prior_rolling_volume(volumes, window=4)
        if np.isfinite(median_1h) and median_1h > 0.0:
            ratio_1h = recent_1h / median_1h

    if len(volumes) >= 16:
        recent_4h = float(volumes.iloc[-16:].sum())
        median_4h = _median_prior_rolling_volume(volumes, window=16)
        if np.isfinite(median_4h) and median_4h > 0.0:
            ratio_4h = recent_4h / median_4h

    return {
        "volume_ratio_1h": ratio_1h,
        "volume_ratio_4h": ratio_4h,
    }


def compute_breakout_features(ohlc_df: pd.DataFrame) -> dict[str, Any]:
    """Compute 24h breakout flag from complete candles.

    breakout_24h is True if latest close > max high of prior 96 candles.
    Returns <NA> when there is insufficient history.
    """
    if len(ohlc_df) < 97:
        return {"breakout_24h": pd.NA}

    latest_close = float(ohlc_df.iloc[-1]["close"])
    prior_24h_max_high = float(ohlc_df.iloc[-97:-1]["high"].max())
    return {"breakout_24h": bool(latest_close > prior_24h_max_high)}


def _safe_log_volume_term(volume_ratio_1h: float, weight: float) -> float:
    if np.isfinite(volume_ratio_1h) and volume_ratio_1h > 1.0:
        return float(weight * math.log(volume_ratio_1h))
    return 0.0


def compute_long_explosion_score(feature_row: dict[str, Any]) -> float:
    """Compute bullish long-side score; blank when near-term momentum is not bullish."""
    if not bool(feature_row.get("ohlc_ready", False)):
        return float("nan")

    ret_1h = float(feature_row["ret_1h_pct"])
    ret_4h = float(feature_row["ret_4h_pct"])
    ret_24h = float(feature_row["ret_24h_pct"])
    ratio_1h = float(feature_row["volume_ratio_1h"])
    breakout_24h = feature_row.get("breakout_24h", False)

    if ret_1h <= 0.0 and ret_4h <= 0.0:
        return float("nan")

    score = 0.35 * max(ret_1h, 0.0)
    score += 0.25 * max(ret_4h, 0.0)
    score += 0.20 * max(ret_24h, 0.0)
    if ratio_1h > 1.0 and ret_1h > 0.0:
        score += _safe_log_volume_term(ratio_1h, weight=5.0)
    if breakout_24h is True:
        score += 3.0
    return float(score)


def compute_dump_score(feature_row: dict[str, Any], scanner_label: str) -> float:
    """Compute bearish/capitulation score for DUMPING rows only."""
    if scanner_label != "DUMPING" or not bool(feature_row.get("ohlc_ready", False)):
        return float("nan")

    ret_1h = float(feature_row["ret_1h_pct"])
    ret_4h = float(feature_row["ret_4h_pct"])
    ret_24h = float(feature_row["ret_24h_pct"])
    ratio_1h = float(feature_row["volume_ratio_1h"])

    score = 0.35 * abs(min(ret_1h, 0.0))
    score += 0.25 * abs(min(ret_4h, 0.0))
    score += 0.20 * abs(min(ret_24h, 0.0))
    if ratio_1h > 1.0 and ret_1h < 0.0:
        score += _safe_log_volume_term(ratio_1h, weight=5.0)
    return float(score)


def compute_reversal_watch_score(feature_row: dict[str, Any]) -> float:
    """Compute reversal watch score for down-24h assets with bounce + unusual volume."""
    if not bool(feature_row.get("ohlc_ready", False)):
        return float("nan")

    ret_1h = float(feature_row["ret_1h_pct"])
    ret_4h = float(feature_row["ret_4h_pct"])
    ret_24h = float(feature_row["ret_24h_pct"])
    ratio_1h = float(feature_row["volume_ratio_1h"])

    if not (ret_24h < -5.0 and np.isfinite(ratio_1h) and ratio_1h > 2.0):
        return float("nan")

    score = 0.40 * max(ret_1h, 0.0)
    score += 0.20 * max(ret_4h, 0.0)
    score += 0.20 * abs(min(ret_24h, 0.0))
    score += _safe_log_volume_term(ratio_1h, weight=4.0)
    return float(score)


def choose_ohlc_signal_type(row: dict[str, Any]) -> str:
    """Classify OHLC signal type from label-aware scores."""
    label = str(row.get("scanner_label", ""))
    long_score = row.get("long_explosion_score", float("nan"))
    dump_score = row.get("dump_score", float("nan"))
    reversal_score = row.get("reversal_watch_score", float("nan"))

    if label in {"HOT_MOVER", "WATCH"} and pd.notna(long_score) and float(long_score) > 0.0:
        return "LONG_EXPLOSION"
    if pd.notna(reversal_score) and float(reversal_score) > 0.0:
        return "REVERSAL_WATCH"
    if label == "DUMPING" and pd.notna(dump_score) and float(dump_score) > 0.0:
        return "DUMPING"
    return ""


def compute_label_aware_scores(feature_row: dict[str, Any], scanner_label: str) -> dict[str, Any]:
    """Compute long/dump/reversal scores and primary score by label."""
    long_score = compute_long_explosion_score(feature_row)
    dump_score = compute_dump_score(feature_row, scanner_label=scanner_label)
    reversal_score = compute_reversal_watch_score(feature_row)

    primary = float("nan")
    if scanner_label in {"HOT_MOVER", "WATCH"}:
        primary = long_score
    elif scanner_label == "DUMPING":
        primary = dump_score

    score_row = {
        "long_explosion_score": long_score,
        "dump_score": dump_score,
        "reversal_watch_score": reversal_score,
        "primary_ohlc_score": primary,
        # Backward-compat alias
        "ohlc_explosion_score": primary,
    }
    score_row["ohlc_signal_type"] = choose_ohlc_signal_type({
        **feature_row,
        **score_row,
        "scanner_label": scanner_label,
    })
    return score_row


def enrich_candidate_row(
    row: pd.Series,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    fetcher: Callable[[str, int], dict[str, Any]] = fetch_ohlc,
) -> dict[str, Any]:
    """Enrich one candidate row with OHLC-derived fields."""
    pair_id = str(row.get("pair_id", "")).strip()

    enriched = {col: float("nan") for col in OHLC_FEATURE_COLUMNS}
    enriched["breakout_24h"] = pd.NA
    enriched["ohlc_ready"] = False
    enriched["ohlc_signal_type"] = ""

    if not pair_id:
        return enriched

    payload = fetcher(pair_id, int(interval))
    parsed = parse_ohlc_response(payload, pair_id=pair_id)
    complete = select_complete_candles(parsed, interval_minutes=int(interval))

    ret_features = compute_return_features(complete)
    vol_features = compute_volume_features(complete)
    breakout_features = compute_breakout_features(complete)

    enriched.update(ret_features)
    enriched.update(vol_features)
    enriched.update(breakout_features)

    required = [
        "ret_15m_pct",
        "ret_1h_pct",
        "ret_4h_pct",
        "ret_24h_pct",
        "volume_ratio_1h",
        "volume_ratio_4h",
    ]
    enriched["ohlc_ready"] = all(pd.notna(enriched.get(col)) for col in required)

    scanner_label = str(row.get("scanner_label", ""))
    label_scores = compute_label_aware_scores(enriched, scanner_label=scanner_label)
    enriched.update(label_scores)
    return enriched


def _ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OHLC_FEATURE_COLUMNS:
        if col not in out.columns:
            if col in ("breakout_24h", "ohlc_ready"):
                out[col] = False
            elif col == "ohlc_signal_type":
                out[col] = ""
            else:
                out[col] = float("nan")
    return out


def sort_enriched_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Sort clean candidate output by signal priority then primary score desc."""
    out = df.copy()
    out["_signal_priority"] = out["ohlc_signal_type"].map(SIGNAL_PRIORITY).fillna(99)
    out = out.sort_values(
        ["_signal_priority", "primary_ohlc_score"],
        ascending=[True, False],
        na_position="last",
    )
    return out.drop(columns=["_signal_priority"]).reset_index(drop=True)


def enrich_candidates_dataframe(
    df: pd.DataFrame,
    interval: int = DEFAULT_INTERVAL_MINUTES,
    fetcher: Callable[[str, int], dict[str, Any]] = fetch_ohlc,
) -> tuple[pd.DataFrame, int, int]:
    """Enrich only rows that are active candidates with supported labels."""
    out = _ensure_ohlc_columns(df)

    candidate_mask = (
        (out["is_candidate"] == True)  # noqa: E712
        & out["scanner_label"].isin(ELIGIBLE_LABELS)
    )
    candidate_indices = out.index[candidate_mask]

    success_count = 0
    insufficient_count = 0

    for idx in candidate_indices:
        row = out.loc[idx]
        try:
            features = enrich_candidate_row(row=row, interval=interval, fetcher=fetcher)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("OHLC enrichment failed for %s: %s", row.get("pair_id"), exc)
            features = {col: float("nan") for col in OHLC_FEATURE_COLUMNS}
            features["breakout_24h"] = pd.NA
            features["ohlc_ready"] = False

        for col, val in features.items():
            out.at[idx, col] = val

        if bool(features.get("ohlc_ready", False)):
            success_count += 1
        else:
            insufficient_count += 1

    return sort_enriched_candidates(out), success_count, insufficient_count


def write_enriched_candidates(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    audit_output_path: Path | None = DEFAULT_AUDIT_OUTPUT_PATH,
) -> None:
    """Write enriched candidates to primary and optional audit output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote enriched candidates to %s", output_path)

    if audit_output_path is not None:
        audit_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(audit_output_path, index=False)
        LOGGER.info("Wrote enriched candidates audit copy to %s", audit_output_path)


def _print_ranked_table(df: pd.DataFrame, title: str, score_col: str, n: int = 15) -> None:
    ranked = df[pd.notna(df[score_col])].sort_values(score_col, ascending=False)
    print(f"\n  Top {min(n, len(ranked))} {title}")
    for _, row in ranked.head(n).iterrows():
        wsname = row.get("wsname", "?")
        score = float(row.get(score_col, float("nan")))
        ret_1h = row.get("ret_1h_pct", float("nan"))
        ret_4h = row.get("ret_4h_pct", float("nan"))
        ret_24h = row.get("ret_24h_pct", float("nan"))
        vol_1h = row.get("volume_ratio_1h", float("nan"))
        print(
            f"    {wsname:<18} score={score:>8.3f} "
            f"r1h={ret_1h:>7.2f}% r4h={ret_4h:>7.2f}% r24h={ret_24h:>8.2f}% "
            f"vr1h={vol_1h:>6.2f}"
        )


def print_enrichment_summary(df: pd.DataFrame, success_count: int, insufficient_count: int) -> None:
    """Print required console summary and three label-aware top tables."""
    candidate_mask = (
        (df["is_candidate"] == True)  # noqa: E712
        & df["scanner_label"].isin(ELIGIBLE_LABELS)
    )
    candidate_count = int(candidate_mask.sum())

    print("\nMemecoin candidate OHLC enrichment")
    print(f"  candidates loaded                    : {candidate_count}")
    print(f"  candidates enriched successfully     : {success_count}")
    print(f"  candidates with insufficient OHLC    : {insufficient_count}")

    candidate_df = df[candidate_mask].copy()
    long_df = candidate_df[candidate_df["ohlc_signal_type"] == "LONG_EXPLOSION"]
    reversal_df = candidate_df[candidate_df["ohlc_signal_type"] == "REVERSAL_WATCH"]
    dump_df = candidate_df[candidate_df["ohlc_signal_type"] == "DUMPING"]

    _print_ranked_table(long_df, "long explosion candidates", score_col="long_explosion_score")
    _print_ranked_table(reversal_df, "reversal-watch candidates", score_col="reversal_watch_score")
    _print_ranked_table(dump_df, "dumping/capitulation candidates", score_col="dump_score")
    print()


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    audit_output_path: Path | None = DEFAULT_AUDIT_OUTPUT_PATH,
    interval: int = DEFAULT_INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Load candidates CSV, enrich with OHLC features, and write outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    df = pd.read_csv(input_path)
    enriched_df, success_count, insufficient_count = enrich_candidates_dataframe(
        df,
        interval=interval,
        fetcher=fetch_ohlc,
    )

    write_enriched_candidates(
        enriched_df,
        output_path=output_path,
        audit_output_path=audit_output_path,
    )
    print_enrichment_summary(enriched_df, success_count, insufficient_count)
    return enriched_df


if __name__ == "__main__":
    main()