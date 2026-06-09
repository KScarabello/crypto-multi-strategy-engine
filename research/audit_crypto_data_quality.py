"""Research-only data integrity and survivorship-bias audit for crypto backtests.

This script audits every local OHLCV file in the configured data directory and
produces two reports:

  reports/crypto_data_quality_symbol_report.csv   — per-symbol detailed metrics
  reports/crypto_data_quality_portfolio_summary.md — portfolio-level bias audit

The audit is READ-ONLY.  It never fetches new data and never modifies any
trading or execution code.

Usage
-----
    .venv/bin/python -m research.audit_crypto_data_quality

Options
-------
    --data-dir      Path to local OHLCV directory  (default: data/local)
    --timeframe     Bar timeframe                   (default: 4h)
    --backtest-start  Earliest date the backtest intends to use  (default: 2020-01-01)
    --min-lookback-bars  Minimum bars the strategy needs before its first signal
                         (default: 36, matching CrossSectionalMomentumStrategy default)
    --stale-days    Flag a symbol as STALE if latest bar is this many days older
                    than today                       (default: 14)
    --symbols       Explicit list of market symbols (e.g. BTC/USD ETH/USD).
                    If omitted, all *_<timeframe>.csv files in --data-dir are used.
    --output-dir    Report output directory          (default: reports)
    --no-markdown   Skip the Markdown portfolio summary
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import date, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

_TIMEFRAME_HOURS: dict[str, float] = {
    "1m": 1 / 60,
    "5m": 5 / 60,
    "15m": 15 / 60,
    "1h": 1.0,
    "4h": 4.0,
    "1d": 24.0,
    "d": 24.0,
    "daily": 24.0,
}


def timeframe_hours(tf: str) -> float:
    """Return the number of hours in *tf*, e.g. '4h' → 4.0."""
    key = tf.strip().lower()
    if key not in _TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe '{tf}'. Known: {sorted(_TIMEFRAME_HOURS)}")
    return _TIMEFRAME_HOURS[key]


def timeframe_freq(tf: str) -> str:
    """Return a pandas DateOffset frequency string for *tf*."""
    hours = timeframe_hours(tf)
    total_minutes = int(round(hours * 60))
    if total_minutes >= 60 and total_minutes % 60 == 0:
        return f"{total_minutes // 60}h"
    return f"{total_minutes}min"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditConfig:
    """Parameters controlling the audit."""

    data_dir: Path = Path("data/local")
    timeframe: str = "4h"
    backtest_start: str = "2020-01-01"
    min_lookback_bars: int = 36
    stale_days: int = 14
    output_dir: Path = Path("reports")
    symbols: tuple[str, ...] | None = None  # None → scan data_dir automatically

    @property
    def backtest_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.backtest_start, tz="UTC")

    @property
    def tf_hours(self) -> float:
        return timeframe_hours(self.timeframe)

    @property
    def tf_freq(self) -> str:
        return timeframe_freq(self.timeframe)

    @property
    def bars_per_day(self) -> float:
        return 24.0 / self.tf_hours


# ---------------------------------------------------------------------------
# Per-symbol result
# ---------------------------------------------------------------------------

@dataclass
class SymbolAuditResult:
    # Identification
    symbol: str
    source_path: str

    # Availability
    first_timestamp: str
    last_timestamp: str
    row_count: int
    expected_bars: int
    missing_bar_count: int
    missing_bar_pct: float
    largest_gap_hours: float
    duplicate_ts_count: int
    non_monotonic_count: int

    # Volume
    zero_volume_count: int
    zero_volume_pct: float

    # NaN counts per column
    nan_open: int
    nan_high: int
    nan_low: int
    nan_close: int
    nan_volume: int

    # Suspicious OHLC
    high_lt_low_count: int
    close_lte_zero_count: int
    open_lte_zero_count: int
    volume_lt_zero_count: int

    # Backtest alignment
    starts_after_backtest_start: bool
    bars_before_first_signal: int      # bars available before min_lookback satisfied
    has_enough_lookback: bool
    is_stale: bool

    # Label and notes
    data_quality_label: str
    notes: str


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAuditSummary:
    # Universe
    total_symbols_audited: int
    symbols_good: int
    symbols_limited_history: int
    symbols_gappy: int
    symbols_stale: int
    symbols_invalid: int
    symbols_needs_review: int

    # Earliest shared start
    earliest_shared_start: str          # date when ALL symbols have data
    symbols_constraining_start: list[str]

    # Survivorship
    using_current_universe_only: bool   # always True since we read live local files
    survivorship_risk_note: str

    # Backtest start alignment
    symbols_missing_full_backtest_history: list[str]
    symbols_insufficient_lookback: list[str]

    # Data gaps / stale
    symbols_with_large_gaps: list[str]
    symbols_stale_or_missing: list[str]

    # Engine behavior
    engine_drop_nonpositive_close: bool  # True → engine silently drops rows
    engine_ffill_missing: bool           # True → engine forward-fills
    engine_note: str


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def symbol_to_filename(symbol: str, timeframe: str) -> str:
    slug = symbol.strip().lower().replace("/", "-").replace(" ", "-")
    return f"{slug}_{timeframe}.csv"


def _load_ohlcv(path: Path) -> pd.DataFrame | None:
    """Load a CSV and parse timestamps.  Returns None if the file is unreadable."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Cannot read %s: %s", path, exc)
        return None
    if "timestamp" not in df.columns:
        LOGGER.warning("%s has no 'timestamp' column", path)
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _quality_label(result: SymbolAuditResult) -> str:
    """Assign a data quality label from the raw audit metrics."""
    # INVALID: structurally broken data
    if (
        result.row_count == 0
        or result.high_lt_low_count > 0
        or result.close_lte_zero_count > result.row_count * 0.01
        or result.volume_lt_zero_count > 0
        or result.nan_close > result.row_count * 0.05
    ):
        return "INVALID"

    # GAPPY: more than 5 % of expected bars are missing, or a single gap
    # is longer than 2 days (48 h) — materially longer than any normal bar
    if result.missing_bar_pct > 5.0 or result.largest_gap_hours > 48.0:
        return "GAPPY"

    # STALE: data hasn't been updated recently
    if result.is_stale:
        return "STALE"

    # LIMITED_HISTORY: symbol doesn't cover the configured backtest start date,
    # or doesn't have enough bars to satisfy the strategy lookback
    if result.starts_after_backtest_start or not result.has_enough_lookback:
        return "LIMITED_HISTORY"

    # NEEDS_REVIEW: minor issues present
    if (
        result.duplicate_ts_count > 0
        or result.non_monotonic_count > 0
        or result.zero_volume_pct > 1.0
        or result.nan_open + result.nan_high + result.nan_low + result.nan_volume > 0
    ):
        return "NEEDS_REVIEW"

    return "GOOD"


def _build_notes(result: SymbolAuditResult, cfg: AuditConfig) -> str:
    notes: list[str] = []

    if result.row_count == 0:
        notes.append("File is empty or unreadable.")
        return "; ".join(notes)

    if result.starts_after_backtest_start:
        notes.append(
            f"First bar {result.first_timestamp[:10]} is after configured backtest "
            f"start {cfg.backtest_start}; symbol was not yet listed on exchange at backtest start."
        )

    if not result.has_enough_lookback:
        notes.append(
            f"Only {result.bars_before_first_signal} bars before first usable signal "
            f"(need {cfg.min_lookback_bars}); strategy may have no valid trades early in history."
        )

    if result.missing_bar_pct > 0:
        notes.append(
            f"{result.missing_bar_count} missing {cfg.timeframe} bars "
            f"({result.missing_bar_pct:.1f}%); largest gap {result.largest_gap_hours:.0f}h."
        )

    if result.duplicate_ts_count > 0:
        notes.append(f"{result.duplicate_ts_count} duplicate timestamps detected.")

    if result.non_monotonic_count > 0:
        notes.append(f"{result.non_monotonic_count} non-monotonic timestamp steps.")

    if result.zero_volume_pct > 0:
        notes.append(
            f"{result.zero_volume_count} zero-volume bars ({result.zero_volume_pct:.1f}%)."
        )

    if result.high_lt_low_count > 0:
        notes.append(f"{result.high_lt_low_count} bars where high < low (INVALID OHLC).")

    if result.close_lte_zero_count > 0:
        notes.append(
            f"{result.close_lte_zero_count} bars with close <= 0; "
            "backtest engine silently drops these rows."
        )

    if result.volume_lt_zero_count > 0:
        notes.append(f"{result.volume_lt_zero_count} bars with volume < 0.")

    nan_total = result.nan_open + result.nan_high + result.nan_low + result.nan_close + result.nan_volume
    if nan_total > 0:
        notes.append(
            f"NaN values: open={result.nan_open} high={result.nan_high} "
            f"low={result.nan_low} close={result.nan_close} volume={result.nan_volume}."
        )

    if result.is_stale:
        notes.append(
            f"Last bar {result.last_timestamp[:10]} is more than {cfg.stale_days} days old; "
            "data may not cover recent price history."
        )

    return "; ".join(notes) if notes else "No issues detected."


def audit_symbol(
    symbol: str,
    path: Path,
    cfg: AuditConfig,
    reference_date: date | None = None,
) -> SymbolAuditResult:
    """Audit one OHLCV file and return a SymbolAuditResult.

    Args:
        symbol:         Market symbol string, e.g. 'BTC/USD'.
        path:           Path to the CSV file.
        cfg:            AuditConfig with backtest parameters.
        reference_date: Today's date for staleness check (injectable for tests).
    """
    today = reference_date or date.today()

    EMPTY = SymbolAuditResult(
        symbol=symbol,
        source_path=str(path),
        first_timestamp="",
        last_timestamp="",
        row_count=0,
        expected_bars=0,
        missing_bar_count=0,
        missing_bar_pct=0.0,
        largest_gap_hours=0.0,
        duplicate_ts_count=0,
        non_monotonic_count=0,
        zero_volume_count=0,
        zero_volume_pct=0.0,
        nan_open=0,
        nan_high=0,
        nan_low=0,
        nan_close=0,
        nan_volume=0,
        high_lt_low_count=0,
        close_lte_zero_count=0,
        open_lte_zero_count=0,
        volume_lt_zero_count=0,
        starts_after_backtest_start=True,
        bars_before_first_signal=0,
        has_enough_lookback=False,
        is_stale=True,
        data_quality_label="INVALID",
        notes="",
    )

    if not path.exists():
        EMPTY.notes = f"File not found: {path}"
        return EMPTY

    df = _load_ohlcv(path)
    if df is None or df.empty:
        EMPTY.notes = "File is empty or could not be parsed."
        return EMPTY

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"]

    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    row_count = len(df)

    # ---- Gap analysis --------------------------------------------------------
    expected_index = pd.date_range(
        start=first_ts, end=last_ts, freq=cfg.tf_freq, tz="UTC"
    )
    unique_ts = pd.DatetimeIndex(timestamps.drop_duplicates())
    missing_index = expected_index.difference(unique_ts)
    missing_bar_count = len(missing_index)
    expected_bars = len(expected_index)
    missing_bar_pct = (missing_bar_count / expected_bars * 100) if expected_bars > 0 else 0.0

    # Largest consecutive gap in hours
    diffs_hours = timestamps.diff().dropna().dt.total_seconds() / 3600.0
    largest_gap_hours = float(diffs_hours.max()) if len(diffs_hours) > 0 else 0.0

    # ---- Timestamp quality ---------------------------------------------------
    duplicate_ts_count = int(timestamps.duplicated().sum())
    non_monotonic_count = int((timestamps.diff().dropna() <= pd.Timedelta(0)).sum())

    # ---- Volume --------------------------------------------------------------
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        zero_volume_count = int((vol == 0).sum())
        zero_volume_pct = zero_volume_count / row_count * 100.0 if row_count else 0.0
        nan_volume = int(vol.isna().sum())
        volume_lt_zero_count = int((vol < 0).sum())
    else:
        zero_volume_count = zero_volume_pct = nan_volume = volume_lt_zero_count = 0

    # ---- NaN counts ----------------------------------------------------------
    def _nan(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(pd.to_numeric(df[col], errors="coerce").isna().sum())

    nan_open = _nan("open")
    nan_high = _nan("high")
    nan_low = _nan("low")
    nan_close = _nan("close")

    # ---- Suspicious OHLC -----------------------------------------------------
    high_lt_low_count = 0
    close_lte_zero_count = 0
    open_lte_zero_count = 0
    if all(c in df.columns for c in ["high", "low"]):
        h = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        valid_hl = h.notna() & lo.notna()
        high_lt_low_count = int((h[valid_hl] < lo[valid_hl]).sum())
    if "close" in df.columns:
        c = pd.to_numeric(df["close"], errors="coerce")
        close_lte_zero_count = int((c <= 0).sum())
    if "open" in df.columns:
        o = pd.to_numeric(df["open"], errors="coerce")
        open_lte_zero_count = int((o <= 0).sum())

    # ---- Backtest alignment --------------------------------------------------
    starts_after_backtest_start = first_ts > cfg.backtest_start_ts

    # How many bars are available after the lookback period is satisfied?
    # bars_before_first_signal = total rows that predate the min_lookback requirement
    bars_before_first_signal = max(0, row_count - cfg.min_lookback_bars)
    has_enough_lookback = row_count >= cfg.min_lookback_bars

    # ---- Staleness -----------------------------------------------------------
    last_date = last_ts.date() if hasattr(last_ts, "date") else date.fromisoformat(str(last_ts)[:10])
    days_since_last = (today - last_date).days
    is_stale = days_since_last > cfg.stale_days

    # ---- Assemble result -----------------------------------------------------
    result = SymbolAuditResult(
        symbol=symbol,
        source_path=str(path),
        first_timestamp=str(first_ts),
        last_timestamp=str(last_ts),
        row_count=row_count,
        expected_bars=expected_bars,
        missing_bar_count=missing_bar_count,
        missing_bar_pct=round(missing_bar_pct, 2),
        largest_gap_hours=round(largest_gap_hours, 1),
        duplicate_ts_count=duplicate_ts_count,
        non_monotonic_count=non_monotonic_count,
        zero_volume_count=zero_volume_count,
        zero_volume_pct=round(zero_volume_pct, 2),
        nan_open=nan_open,
        nan_high=nan_high,
        nan_low=nan_low,
        nan_close=nan_close,
        nan_volume=nan_volume,
        high_lt_low_count=high_lt_low_count,
        close_lte_zero_count=close_lte_zero_count,
        open_lte_zero_count=open_lte_zero_count,
        volume_lt_zero_count=volume_lt_zero_count,
        starts_after_backtest_start=starts_after_backtest_start,
        bars_before_first_signal=bars_before_first_signal,
        has_enough_lookback=has_enough_lookback,
        is_stale=is_stale,
        data_quality_label="",  # filled below
        notes="",
    )
    result.data_quality_label = _quality_label(result)
    result.notes = _build_notes(result, cfg)
    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_symbol_files(cfg: AuditConfig) -> list[tuple[str, Path]]:
    """Return (symbol, path) pairs to audit.

    If cfg.symbols is set, uses that explicit list (file must exist in data_dir).
    Otherwise discovers all *_{timeframe}.csv files automatically.
    """
    data_dir = Path(cfg.data_dir)
    tf = cfg.timeframe

    if cfg.symbols:
        pairs = []
        for sym in cfg.symbols:
            fname = symbol_to_filename(sym, tf)
            path = data_dir / fname
            pairs.append((sym, path))
        return pairs

    # Auto-discover
    paths = sorted(data_dir.glob(f"*_{tf}.csv"))
    pairs = []
    for path in paths:
        stem = path.stem[: -(len(tf) + 1)]  # strip "_4h"
        # Convert slug back to symbol: btc-usd → BTC/USD
        parts = stem.upper().rsplit("-", 1)
        symbol = "/".join(parts) if len(parts) == 2 else stem.upper()
        pairs.append((symbol, path))
    return pairs


# ---------------------------------------------------------------------------
# Portfolio-level audit
# ---------------------------------------------------------------------------

def audit_portfolio(results: list[SymbolAuditResult], cfg: AuditConfig) -> PortfolioAuditSummary:
    """Derive portfolio-level bias and quality metrics from per-symbol results."""
    label_counts: dict[str, int] = {
        "GOOD": 0, "LIMITED_HISTORY": 0, "GAPPY": 0,
        "STALE": 0, "INVALID": 0, "NEEDS_REVIEW": 0,
    }
    for r in results:
        label_counts[r.data_quality_label] = label_counts.get(r.data_quality_label, 0) + 1

    # Earliest shared start: latest first_timestamp across all non-empty symbols
    valid_firsts = [
        pd.Timestamp(r.first_timestamp, tz="UTC")
        for r in results
        if r.first_timestamp and r.first_timestamp not in ("", "MISSING", "EMPTY")
    ]
    if valid_firsts:
        earliest_shared = max(valid_firsts)
        constraining = [
            r.symbol for r in results
            if r.first_timestamp and r.first_timestamp not in ("", "MISSING", "EMPTY")
            and pd.Timestamp(r.first_timestamp, tz="UTC") == earliest_shared
        ]
    else:
        earliest_shared = pd.Timestamp(cfg.backtest_start, tz="UTC")
        constraining = []

    missing_history = [
        r.symbol for r in results if r.starts_after_backtest_start
    ]
    insufficient_lookback = [
        r.symbol for r in results if not r.has_enough_lookback
    ]
    large_gaps = [
        r.symbol for r in results
        if r.missing_bar_pct > 5.0 or r.largest_gap_hours > 72
    ]
    stale_or_missing = [
        r.symbol for r in results
        if r.is_stale or r.row_count == 0
    ]

    survivorship_note = (
        "This backtest uses only currently-available symbols fetched from the live "
        "exchange API. Coins that were delisted, rebranded, or went to zero before "
        "today are NOT represented in the data. Historical performance is therefore "
        "upward-biased: we are implicitly selecting only the survivors. "
        "Strategies that performed well on this universe may have partly succeeded "
        "because they avoided the delisted/collapsed coins that existed at the time."
    )

    engine_note = (
        "The backtest engine (backtest/engine.py) silently drops rows with "
        "close <= 0 (see _validate_ohlcv). It does NOT forward-fill missing bars — "
        "gaps simply reduce the number of bars available to the signal generator. "
        "Symbols with insufficient lookback history are silently excluded from "
        "portfolio selection at early timestamps via the min_history_bars eligibility "
        "filter in CrossSectionalMomentumStrategy."
    )

    return PortfolioAuditSummary(
        total_symbols_audited=len(results),
        symbols_good=label_counts.get("GOOD", 0),
        symbols_limited_history=label_counts.get("LIMITED_HISTORY", 0),
        symbols_gappy=label_counts.get("GAPPY", 0),
        symbols_stale=label_counts.get("STALE", 0),
        symbols_invalid=label_counts.get("INVALID", 0),
        symbols_needs_review=label_counts.get("NEEDS_REVIEW", 0),
        earliest_shared_start=earliest_shared.strftime("%Y-%m-%d"),
        symbols_constraining_start=constraining,
        using_current_universe_only=True,
        survivorship_risk_note=survivorship_note,
        symbols_missing_full_backtest_history=missing_history,
        symbols_insufficient_lookback=insufficient_lookback,
        symbols_with_large_gaps=large_gaps,
        symbols_stale_or_missing=stale_or_missing,
        engine_drop_nonpositive_close=True,
        engine_ffill_missing=False,
        engine_note=engine_note,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_symbol_csv(results: list[SymbolAuditResult], output_dir: Path) -> Path:
    """Write the per-symbol audit results to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "crypto_data_quality_symbol_report.csv"
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(path, index=False)
    LOGGER.info("Symbol report written to %s", path)
    return path


def write_portfolio_markdown(
    summary: PortfolioAuditSummary,
    results: list[SymbolAuditResult],
    cfg: AuditConfig,
    output_dir: Path,
) -> Path:
    """Write the portfolio-level audit summary as a Markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "crypto_data_quality_portfolio_summary.md"

    lines: list[str] = [
        "# Crypto Backtest Data Quality & Survivorship-Bias Audit",
        "",
        f"**Timeframe:** {cfg.timeframe}  ",
        f"**Configured backtest start:** {cfg.backtest_start}  ",
        f"**Strategy min lookback bars:** {cfg.min_lookback_bars}  ",
        f"**Data directory:** `{cfg.data_dir}`  ",
        f"**Symbols audited:** {summary.total_symbols_audited}  ",
        "",
        "---",
        "",
        "## 1. Universe Quality Summary",
        "",
        "| Label | Count |",
        "|---|---|",
        f"| GOOD | {summary.symbols_good} |",
        f"| LIMITED_HISTORY | {summary.symbols_limited_history} |",
        f"| GAPPY | {summary.symbols_gappy} |",
        f"| STALE | {summary.symbols_stale} |",
        f"| NEEDS_REVIEW | {summary.symbols_needs_review} |",
        f"| INVALID | {summary.symbols_invalid} |",
        "",
        "---",
        "",
        "## 2. Earliest Date Where All Symbols Have Usable Data",
        "",
        f"**{summary.earliest_shared_start}**",
        "",
    ]
    if summary.symbols_constraining_start:
        lines += [
            f"Symbols that constrain the joint start date: "
            f"{', '.join(summary.symbols_constraining_start)}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Symbols with Incomplete Backtest History",
        "",
        "> These symbols have a first bar that is *after* the configured backtest start.",
        "> Any analysis comparing them to BTC buy-and-hold from that date is unfair.",
        "",
    ]
    if summary.symbols_missing_full_backtest_history:
        for sym in sorted(summary.symbols_missing_full_backtest_history):
            r = next((x for x in results if x.symbol == sym), None)
            date_note = f" (first bar: {r.first_timestamp[:10]})" if r else ""
            lines.append(f"- {sym}{date_note}")
    else:
        lines.append("*None — all symbols cover the backtest start date.*")

    lines += [
        "",
        "---",
        "",
        "## 4. Survivorship Bias Risk",
        "",
        f"> {summary.survivorship_risk_note}",
        "",
        "### What this means for the backtest",
        "",
        "- The local OHLCV files are populated by fetching **currently-listed** Kraken symbols.",
        "- Coins that delisted, collapsed, or were replaced after listing are absent.",
        "- Example: POL/USD (formerly MATIC) starts only in 2024-09 — its pre-rebrand",
        "  history under MATIC/USD is not included.",
        "- If the strategy historically avoided delisted coins (e.g. via momentum score",
        "  declining before delisting), the measured edge may partly reflect this.",
        "- **To mitigate**: maintain a point-in-time universe snapshot, or explicitly",
        "  add delisted coins with their available history.",
        "",
        "---",
        "",
        "## 5. Symbols with Insufficient Lookback History",
        "",
        f"> The strategy requires **{cfg.min_lookback_bars} bars** of history before it can",
        "> generate a valid signal. Symbols below this threshold are silently excluded",
        "> from the portfolio at those early timestamps.",
        "",
    ]
    if summary.symbols_insufficient_lookback:
        for sym in sorted(summary.symbols_insufficient_lookback):
            r = next((x for x in results if x.symbol == sym), None)
            bars_note = f" ({r.row_count} bars total)" if r else ""
            lines.append(f"- {sym}{bars_note}")
    else:
        lines.append("*None — all symbols satisfy the minimum lookback.*")

    lines += [
        "",
        "---",
        "",
        "## 6. Symbols with Large Gaps or Stale Data",
        "",
        "### Large gaps (>5% missing bars or gap >72h)",
        "",
    ]
    if summary.symbols_with_large_gaps:
        for sym in sorted(summary.symbols_with_large_gaps):
            r = next((x for x in results if x.symbol == sym), None)
            if r:
                lines.append(
                    f"- {sym}: {r.missing_bar_count} missing bars "
                    f"({r.missing_bar_pct:.1f}%), largest gap {r.largest_gap_hours:.0f}h"
                )
    else:
        lines.append("*No symbols with large gaps detected.*")

    lines += [
        "",
        "### Stale or missing data",
        "",
    ]
    if summary.symbols_stale_or_missing:
        for sym in sorted(summary.symbols_stale_or_missing):
            r = next((x for x in results if x.symbol == sym), None)
            last_note = f" (last bar: {r.last_timestamp[:10]})" if r and r.last_timestamp else ""
            lines.append(f"- {sym}{last_note}")
    else:
        lines.append("*No stale symbols detected.*")

    lines += [
        "",
        "---",
        "",
        "## 7. Backtest Engine Behavior Notes",
        "",
        f"> {summary.engine_note}",
        "",
        "| Behavior | Status |",
        "|---|---|",
        f"| Silently drops close ≤ 0 rows | {'Yes ⚠️' if summary.engine_drop_nonpositive_close else 'No'} |",
        f"| Forward-fills missing bars | {'Yes ⚠️' if summary.engine_ffill_missing else 'No ✅'} |",
        "",
        "---",
        "",
        "## 8. Universe Size Over Time",
        "",
        "> The number of symbols with available data grows over the backtest window.",
        "> Early periods have fewer assets in the opportunity set.",
        "",
    ]

    # Build a simple year-by-year coverage table
    lines.append("| Year | Symbols with data |")
    lines.append("|---|---|")
    for yr in range(2020, 2027):
        yr_start = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        count = sum(
            1 for r in results
            if r.first_timestamp not in ("", "MISSING", "EMPTY")
            and pd.Timestamp(r.first_timestamp, tz="UTC") <= yr_start
        )
        lines.append(f"| {yr} | {count} |")

    lines += [
        "",
        "---",
        "",
        "## 9. Per-Symbol First/Last Bar Reference",
        "",
        "| Symbol | First Bar | Last Bar | Rows | Label |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.first_timestamp):
        first = r.first_timestamp[:10] if r.first_timestamp else "—"
        last = r.last_timestamp[:10] if r.last_timestamp else "—"
        lines.append(f"| {r.symbol} | {first} | {last} | {r.row_count} | {r.data_quality_label} |")

    lines += [
        "",
        "---",
        "",
        "## 10. Remaining Limitations (Even After This Audit)",
        "",
        "- **Intra-bar path unknown**: OHLCV only records open/high/low/close; actual",
        "  tick-by-tick path is not auditable.",
        "- **Wash trading / spoofed volume**: Exchange-reported volume may not reflect",
        "  genuine market depth, especially on alt-coins.",
        "- **Cross-exchange inconsistency**: Price and volume differ across exchanges.",
        "  All data here is Kraken-only; another exchange may show different history.",
        "- **Point-in-time universe**: Even with full history, we can only add coins",
        "  that exist today. Coins that peaked and died before today are not retrievable.",
        "- **Look-ahead in symbol selection**: The decision to include a coin in the",
        "  universe is made with knowledge of its current status, which is a subtle",
        "  form of look-ahead bias.",
        "- **Listing-date uncertainty**: Exchange APIs may not accurately report the",
        "  date a coin became actively tradable; early bars may have extreme spreads.",
        "",
        "---",
        "",
        "*Generated by `research/audit_crypto_data_quality.py` — research only.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Portfolio summary written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Console interpretation
# ---------------------------------------------------------------------------

def print_interpretation(
    results: list[SymbolAuditResult],
    summary: PortfolioAuditSummary,
    cfg: AuditConfig,
) -> None:
    bar = "=" * 70

    print(f"\n{bar}")
    print("  CRYPTO DATA QUALITY AUDIT — INTERPRETATION")
    print(f"{bar}")

    print(f"\n  Symbols audited:       {summary.total_symbols_audited}")
    print(f"  GOOD:                  {summary.symbols_good}")
    print(f"  LIMITED_HISTORY:       {summary.symbols_limited_history}")
    print(f"  GAPPY:                 {summary.symbols_gappy}")
    print(f"  STALE:                 {summary.symbols_stale}")
    print(f"  NEEDS_REVIEW:          {summary.symbols_needs_review}")
    print(f"  INVALID:               {summary.symbols_invalid}")

    print(f"\n  Earliest joint start:  {summary.earliest_shared_start}")
    if summary.symbols_constraining_start:
        print(f"  Constraining symbols:  {', '.join(summary.symbols_constraining_start)}")

    if summary.symbols_missing_full_backtest_history:
        print(
            f"\n  ⚠  {len(summary.symbols_missing_full_backtest_history)} symbol(s) don't cover "
            f"backtest start {cfg.backtest_start}:"
        )
        for sym in sorted(summary.symbols_missing_full_backtest_history):
            r = next(x for x in results if x.symbol == sym)
            print(f"     {sym}: first bar {r.first_timestamp[:10]}")

    if summary.symbols_with_large_gaps:
        print(f"\n  ⚠  {len(summary.symbols_with_large_gaps)} symbol(s) have large data gaps:")
        for sym in sorted(summary.symbols_with_large_gaps):
            r = next(x for x in results if x.symbol == sym)
            print(f"     {sym}: {r.missing_bar_count} missing bars, largest gap {r.largest_gap_hours:.0f}h")

    if summary.symbols_stale_or_missing:
        print(f"\n  ⚠  {len(summary.symbols_stale_or_missing)} symbol(s) have stale/missing data:")
        for sym in sorted(summary.symbols_stale_or_missing):
            r = next(x for x in results if x.symbol == sym)
            print(f"     {sym}: last bar {r.last_timestamp[:10] if r.last_timestamp else 'N/A'}")

    print(
        f"\n  Survivorship bias risk: HIGH — universe is built from currently-listed"
        f"\n  symbols only. Delisted/dead coins are absent from all history."
    )

    print(
        f"\n  Engine behaviour: drops close<=0 silently; does NOT forward-fill gaps."
        f"\n  Min-history filter excludes late-starting symbols early in the backtest."
    )

    # Check BTC-comparable history
    btc = next((r for r in results if "BTC" in r.symbol), None)
    if btc:
        short_symbols = [
            r.symbol for r in results
            if r.first_timestamp and r.first_timestamp not in ("", "MISSING", "EMPTY")
            and pd.Timestamp(r.first_timestamp, tz="UTC") > pd.Timestamp(btc.first_timestamp, tz="UTC")
        ]
        if short_symbols:
            print(
                f"\n  ⚠  {len(short_symbols)} symbol(s) have shorter history than BTC/USD; "
                f"direct buy-and-hold comparison is unfair for early years:"
            )
            for sym in sorted(short_symbols):
                r = next(x for x in results if x.symbol == sym)
                print(f"     {sym}: first bar {r.first_timestamp[:10]}")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-only crypto OHLCV data quality and survivorship-bias audit"
    )
    parser.add_argument("--data-dir", default="data/local", help="Local OHLCV directory")
    parser.add_argument("--timeframe", default="4h", help="Bar timeframe (e.g. 4h)")
    parser.add_argument("--backtest-start", default="2020-01-01", help="Earliest backtest date")
    parser.add_argument(
        "--min-lookback-bars", type=int, default=36,
        help="Strategy min history bars before first valid signal"
    )
    parser.add_argument(
        "--stale-days", type=int, default=14,
        help="Days since last bar before flagging as STALE"
    )
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--symbols", nargs="+", default=None, help="Explicit symbol list")
    parser.add_argument("--no-markdown", action="store_true", help="Skip Markdown report")
    return parser.parse_args()


def run_audit(cfg: AuditConfig) -> tuple[list[SymbolAuditResult], PortfolioAuditSummary]:
    """Run the full audit and return results (no file I/O — useful for tests)."""
    pairs = discover_symbol_files(cfg)
    if not pairs:
        LOGGER.warning("No symbol files found in %s for timeframe %s", cfg.data_dir, cfg.timeframe)
        return [], audit_portfolio([], cfg)

    results = []
    for symbol, path in pairs:
        LOGGER.info("Auditing %s from %s", symbol, path)
        result = audit_symbol(symbol=symbol, path=path, cfg=cfg)
        results.append(result)

    summary = audit_portfolio(results, cfg)
    return results, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()

    cfg = AuditConfig(
        data_dir=Path(args.data_dir),
        timeframe=args.timeframe,
        backtest_start=args.backtest_start,
        min_lookback_bars=args.min_lookback_bars,
        stale_days=args.stale_days,
        output_dir=Path(args.output_dir),
        symbols=tuple(args.symbols) if args.symbols else None,
    )

    results, summary = run_audit(cfg)
    if not results:
        print("No data files found. Run data/update_ohlcv.py first.")
        return

    csv_path = write_symbol_csv(results, cfg.output_dir)
    print(f"\nSymbol report:     {csv_path}")

    if not args.no_markdown:
        md_path = write_portfolio_markdown(summary, results, cfg, cfg.output_dir)
        print(f"Portfolio summary: {md_path}")

    print_interpretation(results, summary, cfg)


if __name__ == "__main__":
    main()
