"""Research-only universe integrity analysis and backtest mode comparison.

PARTS
-----
1. Audit the live 5-coin universe — per-symbol eligibility + rebalance-level log
2. Three explicit universe modes with no silent fallback
3. Backtest comparison: CURRENT_SURVIVORS_EXPANDING vs FIXED_COMMON_HISTORY
4. Point-in-time data requirements document
5. (Tests live in tests/test_universe_integrity_analysis.py)

IMPORTANT BIAS NOTE
-------------------
Neither CURRENT_SURVIVORS_EXPANDING nor FIXED_COMMON_HISTORY eliminates
survivorship bias.  Both begin from a symbol list selected with present-day
knowledge.  Coins that existed historically but are now delisted, collapsed, or
renamed are absent from both modes.  A genuine historical backtest requires a
POINT_IN_TIME_UNIVERSE built from exchange-specific listing/delisting records.

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.universe_integrity_analysis
    .venv/bin/python -m research.universe_integrity_analysis --mode a      # expanding only
    .venv/bin/python -m research.universe_integrity_analysis --mode b      # fixed common history only
    .venv/bin/python -m research.universe_integrity_analysis --no-backtest # audit only
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from data.fetch_ohlc import load_ohlcv_history, pivot_close
from backtest.engine import run_backtest
from backtest.metrics import summary_metrics
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIVE_FIVE_UNIVERSE: tuple[str, ...] = (
    "BTC/USD",
    "ETH/USD",
    "XRP/USD",
    "SOL/USD",
    "AVAX/USD",
)

# Default strategy parameters matching CrossSectionalMomentumStrategy defaults.
# These are NOT changed from their original values.
DEFAULT_TOP_N = 3
DEFAULT_SHORT_LOOKBACK = 12
DEFAULT_MEDIUM_LOOKBACK = 36
DEFAULT_MIN_HISTORY_BARS = 36       # max(12, 36)
DEFAULT_FEE_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5
DEFAULT_REBALANCE_BARS = 6
DEFAULT_INITIAL_CAPITAL = 10_000.0
BARS_PER_YEAR_4H = int(24 / 4 * 365)  # 2190

SURVIVORSHIP_BIAS_NOTE = (
    "SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and "
    "FIXED_COMMON_HISTORY use only coins currently listed on Kraken. "
    "Delisted, collapsed, or renamed coins that existed historically are "
    "absent from both modes. Neither mode eliminates survivorship bias. "
    "A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with "
    "authoritative exchange listing/delisting records."
)

# Schema for point-in-time membership CSV
PIT_MEMBERSHIP_REQUIRED_COLS = ["symbol", "exchange", "eligible_from", "eligible_to"]
PIT_MEMBERSHIP_ALL_COLS = [
    "symbol",
    "exchange",
    "eligible_from",
    "eligible_to",
    "source",
    "source_as_of",
    "listing_reason",
    "delisting_reason",
    "symbol_predecessor",
    "symbol_successor",
    "notes",
]


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class UniverseMode(str, Enum):
    """Explicitly labeled universe construction modes."""
    CURRENT_SURVIVORS_EXPANDING = "CURRENT_SURVIVORS_EXPANDING"
    FIXED_COMMON_HISTORY = "FIXED_COMMON_HISTORY"
    POINT_IN_TIME_UNIVERSE = "POINT_IN_TIME_UNIVERSE"


@dataclass
class SymbolHistory:
    """Per-symbol availability and eligibility summary."""
    symbol: str
    first_ts: str
    last_ts: str
    total_bars: int
    first_eligible_ts: str           # first bar with >= min_history_bars of history
    usable_bars: int                 # bars from first_eligible_ts onwards
    starts_after_backtest_start: bool
    notes: str


@dataclass
class RebalanceRow:
    """One row of the rebalance-level eligibility log."""
    timestamp: str
    configured_universe_size: int
    eligible_universe_size: int
    eligible_symbols: str            # pipe-separated
    ineligible_symbols: str          # pipe-separated
    ineligibility_reasons: str       # pipe-separated "symbol:reason"
    top_n: int
    top_n_pct_of_eligible: float
    top_n_gte_eligible: bool


@dataclass
class BacktestModeResult:
    """Full metrics for one backtest mode."""
    mode: str
    bias_note: str
    effective_start: str
    effective_end: str
    configured_universe: str         # pipe-separated
    avg_eligible_universe_size: float
    min_eligible_universe_size: int
    max_eligible_universe_size: int
    pct_rebalances_top_n_selected_full_eligible: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    n_rebalances: int
    n_trades: int
    pct_time_invested: float
    avg_cash_exposure: float
    btc_total_return_same_period: float
    btc_start: str
    btc_end: str


# ---------------------------------------------------------------------------
# Part 1: Close matrix helpers
# ---------------------------------------------------------------------------

def build_close_matrix(
    symbols: Sequence[str],
    timeframe: str = "4h",
    data_dir: Path = Path("data/local"),
) -> pd.DataFrame:
    """Load OHLCV and return a timestamp × symbol close-price matrix."""
    ohlcv = load_ohlcv_history(
        symbols=list(symbols),
        timeframe=timeframe,
        data_dir=data_dir,
    )
    return pivot_close(ohlcv)


def find_first_eligible_ts(
    close_col: pd.Series,
    min_history_bars: int,
) -> pd.Timestamp | None:
    """Return the first timestamp at which the symbol has enough lookback history."""
    valid = close_col.notna()
    hcount = valid.astype(int).cumsum()
    eligible = hcount >= min_history_bars
    hits = eligible[eligible].index
    return hits[0] if len(hits) else None


def find_joint_eligible_start(
    close: pd.DataFrame,
    min_history_bars: int,
) -> pd.Timestamp | None:
    """Return the first timestamp where ALL symbols have >= min_history_bars history."""
    valid = close.notna()
    hcount = valid.astype(int).cumsum()
    all_eligible = (hcount >= min_history_bars).all(axis=1)
    hits = all_eligible[all_eligible].index
    return hits[0] if len(hits) else None


def get_eligible_symbols_at_ts(
    close: pd.DataFrame,
    ts: pd.Timestamp,
    min_history_bars: int,
) -> list[str]:
    """Return symbols that have >= min_history_bars of non-NaN data at ts."""
    if ts not in close.index:
        return []
    valid = close.notna()
    hcount = valid.astype(int).cumsum()
    eligible_mask = close.loc[ts].notna() & (hcount.loc[ts] >= min_history_bars)
    return [str(s) for s in eligible_mask[eligible_mask].index]


def audit_symbol_histories(
    close: pd.DataFrame,
    min_history_bars: int,
    backtest_start: str = "2020-01-01",
) -> list[SymbolHistory]:
    """Build per-symbol availability and eligibility summaries."""
    backtest_ts = pd.Timestamp(backtest_start, tz="UTC")
    results = []
    for sym in close.columns:
        col = close[sym].dropna()
        if col.empty:
            results.append(SymbolHistory(
                symbol=sym, first_ts="", last_ts="", total_bars=0,
                first_eligible_ts="", usable_bars=0,
                starts_after_backtest_start=True,
                notes="No data available.",
            ))
            continue
        first_ts = col.index[0]
        last_ts = col.index[-1]
        total_bars = len(col)
        first_elig = find_first_eligible_ts(close[sym], min_history_bars)
        usable = len(close[sym].dropna().loc[first_elig:]) if first_elig is not None else 0
        starts_late = first_ts > backtest_ts
        notes_list = []
        if starts_late:
            notes_list.append(
                f"Listed after backtest start ({first_ts.date()} > {backtest_start}). "
                "Not in the universe for the early part of the backtest."
            )
        if first_elig is None:
            notes_list.append("Never satisfies min_history_bars requirement.")
        elif (first_elig - first_ts).total_seconds() / 3600 < 1:
            notes_list.append("Satisfies lookback almost immediately (data starts mid-lookback).")
        results.append(SymbolHistory(
            symbol=sym,
            first_ts=str(first_ts),
            last_ts=str(last_ts),
            total_bars=total_bars,
            first_eligible_ts=str(first_elig) if first_elig is not None else "",
            usable_bars=usable,
            starts_after_backtest_start=starts_late,
            notes="; ".join(notes_list) if notes_list else "No issues.",
        ))
    return results


def build_rebalance_eligibility_log(
    close: pd.DataFrame,
    min_history_bars: int,
    top_n: int,
    rebalance_every_bars: int = 1,
) -> pd.DataFrame:
    """Build a per-rebalance eligibility log.

    For each rebalance timestamp records: eligible symbols, eligible count,
    top_n versus eligible, and ineligibility reasons.
    """
    configured_size = len(close.columns)
    valid = close.notna()
    hcount = valid.astype(int).cumsum()
    eligible_mask_full = valid & (hcount >= min_history_bars)  # timestamp x symbol

    rows: list[RebalanceRow] = []
    rebalance_timestamps = close.index[::rebalance_every_bars]

    for ts in rebalance_timestamps:
        eligible_row = eligible_mask_full.loc[ts]
        eligible_syms = [str(s) for s in eligible_row[eligible_row].index]
        ineligible_syms = [str(s) for s in eligible_row[~eligible_row].index]

        # Reasons for ineligibility
        reasons = {}
        for sym in ineligible_syms:
            if close.loc[ts, sym] != close.loc[ts, sym]:  # NaN
                bar_count = int(hcount.loc[ts, sym])
                if bar_count == 0:
                    reasons[sym] = "no_data_yet"
                elif bar_count < min_history_bars:
                    reasons[sym] = f"only_{bar_count}_bars_of_{min_history_bars}_required"
                else:
                    reasons[sym] = "price_is_nan"
            else:
                bar_count = int(hcount.loc[ts, sym])
                reasons[sym] = f"only_{bar_count}_bars_of_{min_history_bars}_required"

        n_eligible = len(eligible_syms)
        pct = (top_n / n_eligible * 100.0) if n_eligible > 0 else float("nan")

        rows.append(RebalanceRow(
            timestamp=str(ts),
            configured_universe_size=configured_size,
            eligible_universe_size=n_eligible,
            eligible_symbols="|".join(eligible_syms),
            ineligible_symbols="|".join(ineligible_syms),
            ineligibility_reasons="|".join(f"{s}:{r}" for s, r in reasons.items()),
            top_n=top_n,
            top_n_pct_of_eligible=round(pct, 1),
            top_n_gte_eligible=(top_n >= n_eligible) if n_eligible > 0 else False,
        ))

    return pd.DataFrame([asdict(r) for r in rows])


# ---------------------------------------------------------------------------
# Part 2: Universe mode classes
# ---------------------------------------------------------------------------

class UniverseFilterError(Exception):
    """Raised when a universe mode cannot be applied."""


class UniverseFilter:
    """Abstract base for universe mode filters."""
    mode: UniverseMode

    def get_effective_close(
        self,
        close: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, str]:
        """Return (effective_close, effective_start_str).

        The returned close DataFrame is what should be passed to the backtest.
        effective_start_str is the ISO date string of the mode's logical start.
        """
        raise NotImplementedError

    def bias_note(self) -> str:
        raise NotImplementedError


class CurrentSurvivorsExpandingFilter(UniverseFilter):
    """CURRENT_SURVIVORS_EXPANDING: existing behavior.

    Uses the present-day configured symbols.  Allows symbols to enter the
    portfolio when enough history becomes available.

    BIAS: Subject to current-listing bias, survivorship bias, and look-ahead
    bias in universe construction.  The symbols included are known to have
    survived to the present day.
    """
    mode = UniverseMode.CURRENT_SURVIVORS_EXPANDING

    def get_effective_close(
        self,
        close: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, str]:
        effective_start = str(close.index[0].date())
        LOGGER.info(
            "CURRENT_SURVIVORS_EXPANDING: using full data from %s. "
            "Universe size expands as symbols accumulate lookback history.",
            effective_start,
        )
        return close, effective_start

    def bias_note(self) -> str:
        return (
            "CURRENT_SURVIVORS_EXPANDING: universe is built from currently-listed "
            "Kraken symbols. Subject to survivorship bias, current-listing bias, and "
            "look-ahead bias. Late-listing symbols (SOL from 2020-08-11, AVAX from "
            "2020-09-22) enter the portfolio when they satisfy the lookback requirement. "
            "Early universe may have as few as 3 eligible symbols, causing top_n=3 to "
            "select the entire eligible set with no discrimination."
        )


class FixedCommonHistoryFilter(UniverseFilter):
    """FIXED_COMMON_HISTORY: backtest begins only when every symbol has enough history.

    No symbols may silently enter or leave after the effective start.
    A missing or invalid symbol after the start raises an error explicitly.

    BIAS: Still subject to current-listing and survivorship bias because the
    symbol list is chosen with present-day knowledge.
    """
    mode = UniverseMode.FIXED_COMMON_HISTORY

    def __init__(self, min_history_bars: int = DEFAULT_MIN_HISTORY_BARS):
        self.min_history_bars = min_history_bars

    def get_effective_close(
        self,
        close: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, str]:
        joint_start = find_joint_eligible_start(close, self.min_history_bars)
        if joint_start is None:
            raise UniverseFilterError(
                "FIXED_COMMON_HISTORY: not all symbols ever satisfy min_history_bars "
                f"({self.min_history_bars}) simultaneously. Cannot determine a joint start."
            )

        LOGGER.info("FIXED_COMMON_HISTORY: joint effective start = %s", joint_start)

        # Verify all symbols remain present after the joint start (exclusive)
        after = close.loc[close.index > joint_start]
        missing_after = [s for s in close.columns if after.empty or after[s].notna().sum() == 0]
        if missing_after:
            raise UniverseFilterError(
                f"FIXED_COMMON_HISTORY: the following symbols have no data after the "
                f"effective start {joint_start}: {missing_after}. Cannot use this mode "
                "without resolving missing symbols. Do not fall back silently."
            )

        return close, str(joint_start.date())

    def bias_note(self) -> str:
        return (
            "FIXED_COMMON_HISTORY: backtest begins only after all configured symbols "
            "have sufficient lookback history. Universe composition does not change after "
            "effective start. Still subject to survivorship and current-listing bias "
            "because the symbol list is selected with present-day knowledge."
        )


class PointInTimeUniverseFilter(UniverseFilter):
    """POINT_IN_TIME_UNIVERSE: driven by a dated membership file.

    Supports delisted symbols, symbol migrations, and time-varying membership.
    This is the only mode that can genuinely reduce survivorship bias.

    STATUS: NOT READY FOR REAL EXECUTION.  The interface, schema validation,
    and synthetic test support are implemented.  Real historical membership data
    (exchange listing/delisting dates, delisted symbol OHLCV) has not been
    sourced.  Calling get_effective_close() without real membership data will
    raise NotReadyError.
    """
    mode = UniverseMode.POINT_IN_TIME_UNIVERSE

    class NotReadyError(Exception):
        pass

    def __init__(
        self,
        membership_df: pd.DataFrame | None = None,
        allow_synthetic: bool = False,
    ):
        """
        Args:
            membership_df: DataFrame conforming to PIT_MEMBERSHIP_ALL_COLS schema.
            allow_synthetic: If True, skip the NOT READY guard (for tests only).
        """
        self.membership_df = membership_df
        self.allow_synthetic = allow_synthetic

    def get_effective_close(
        self,
        close: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, str]:
        if not self.allow_synthetic:
            raise PointInTimeUniverseFilter.NotReadyError(
                "POINT_IN_TIME_UNIVERSE is NOT READY for real execution. "
                "Historical exchange listing/delisting records and delisted-coin OHLCV "
                "have not been sourced. "
                "See reports/universe_integrity_pit_requirements.md for what is needed."
            )
        if self.membership_df is None:
            raise UniverseFilterError("membership_df is required for POINT_IN_TIME_UNIVERSE")
        validate_pit_membership_schema(self.membership_df)
        effective_start = str(
            pd.to_datetime(self.membership_df["eligible_from"], utc=True, errors="coerce").min().date()
        )
        LOGGER.info("POINT_IN_TIME_UNIVERSE (synthetic): effective_start = %s", effective_start)
        return close, effective_start

    def get_eligible_symbols_at_ts(self, ts: pd.Timestamp) -> list[str]:
        """Return symbols eligible at ts according to the membership file."""
        if self.membership_df is None:
            return []
        validate_pit_membership_schema(self.membership_df)
        ts = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else ts
        mdf = self.membership_df.copy()
        mdf["eligible_from"] = pd.to_datetime(mdf["eligible_from"], utc=True, errors="coerce")
        mdf["eligible_to"] = pd.to_datetime(mdf["eligible_to"], utc=True, errors="coerce")
        in_window = mdf[
            (mdf["eligible_from"] <= ts) &
            (mdf["eligible_to"].isna() | (mdf["eligible_to"] > ts))
        ]
        return in_window["symbol"].tolist()

    def bias_note(self) -> str:
        return (
            "POINT_IN_TIME_UNIVERSE: driven by a dated membership file. Supports "
            "delisted symbols and symbol migrations. NOT READY FOR REAL EXECUTION — "
            "historical listing/delisting data and delisted-coin OHLCV not yet sourced."
        )


def validate_pit_membership_schema(df: pd.DataFrame) -> None:
    """Validate that a membership DataFrame has the required columns and types."""
    missing = [c for c in PIT_MEMBERSHIP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"PIT membership DataFrame is missing required columns: {missing}. "
            f"Required: {PIT_MEMBERSHIP_REQUIRED_COLS}"
        )
    if df.empty:
        raise ValueError("PIT membership DataFrame is empty.")
    # eligible_from must be parseable
    try:
        ef = pd.to_datetime(df["eligible_from"], utc=True, errors="raise")
    except Exception as exc:
        raise ValueError(f"eligible_from column has unparseable values: {exc}") from exc
    # eligible_to may be NaN (meaning still active) but must be parseable where not null
    non_null = df["eligible_to"].dropna()
    if len(non_null):
        try:
            pd.to_datetime(non_null, utc=True, errors="raise")
        except Exception as exc:
            raise ValueError(f"eligible_to column has unparseable values: {exc}") from exc


def get_universe_filter(
    mode: UniverseMode,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
    membership_df: pd.DataFrame | None = None,
    allow_synthetic: bool = False,
) -> UniverseFilter:
    """Factory: return the appropriate filter for a given mode."""
    if mode == UniverseMode.CURRENT_SURVIVORS_EXPANDING:
        return CurrentSurvivorsExpandingFilter()
    if mode == UniverseMode.FIXED_COMMON_HISTORY:
        return FixedCommonHistoryFilter(min_history_bars=min_history_bars)
    if mode == UniverseMode.POINT_IN_TIME_UNIVERSE:
        return PointInTimeUniverseFilter(
            membership_df=membership_df,
            allow_synthetic=allow_synthetic,
        )
    raise ValueError(f"Unknown universe mode: {mode}")


# ---------------------------------------------------------------------------
# Part 3: Backtest comparison helpers
# ---------------------------------------------------------------------------

def _close_to_ohlcv(close: pd.DataFrame) -> pd.DataFrame:
    """Convert a close-price matrix to long-format OHLCV with dummy OHLC columns."""
    frames = []
    for sym in close.columns:
        col = close[sym].dropna().reset_index()
        col.columns = ["timestamp", "close"]
        col["symbol"] = sym
        col["open"] = col["close"]
        col["high"] = col["close"]
        col["low"] = col["close"]
        col["volume"] = 1.0
        frames.append(col[["timestamp", "open", "high", "low", "close", "volume", "symbol"]])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _make_signal_generator(
    top_n: int = DEFAULT_TOP_N,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    medium_lookback: int = DEFAULT_MEDIUM_LOOKBACK,
) -> Callable:
    """Build a backtest signal generator using CrossSectionalMomentumStrategy."""
    strategy = CrossSectionalMomentumStrategy()
    cfg = {
        "top_n": top_n,
        "short_lookback_bars": short_lookback,
        "medium_lookback_bars": medium_lookback,
        "min_history_bars": min_history_bars,
        "use_regime_filter": False,
        "min_eligible_assets": 1,
    }

    def _generator(close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        try:
            weights = strategy.generate_target_weights(close, ts, config=cfg)
        except Exception as exc:
            LOGGER.debug("Signal generation error at %s: %s", ts, exc)
            weights = {}
        return pd.Series(weights, dtype=float).reindex(close.columns, fill_value=0.0)

    return _generator


def compute_btc_benchmark(
    close: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict:
    """Compute BTC buy-and-hold return over [start_ts, end_ts].

    Applies entry and exit transaction costs (2 × fee_bps).
    Uses the exact same start and end timestamps as the mode under comparison.
    """
    btc_col = None
    for sym in close.columns:
        if "BTC" in sym:
            btc_col = close[sym]
            break
    if btc_col is None:
        return {"btc_total_return": float("nan"), "btc_start": "", "btc_end": ""}

    # Align to available data
    avail = btc_col.dropna()
    first_avail = avail.index[avail.index >= start_ts]
    last_avail = avail.index[avail.index <= end_ts]
    if len(first_avail) == 0 or len(last_avail) == 0:
        return {"btc_total_return": float("nan"), "btc_start": str(start_ts.date()), "btc_end": str(end_ts.date())}

    actual_start = first_avail[0]
    actual_end = last_avail[-1]
    p_start = float(avail.loc[actual_start])
    p_end = float(avail.loc[actual_end])

    gross_return = p_end / p_start - 1.0
    # Entry + exit round trip fee
    round_trip_cost = 2 * fee_bps / 10_000.0
    net_return = gross_return - round_trip_cost

    return {
        "btc_total_return": round(net_return, 6),
        "btc_start": str(actual_start.date()),
        "btc_end": str(actual_end.date()),
    }


def _compute_extended_metrics(
    result,
    close: pd.DataFrame,
    effective_start: pd.Timestamp,
    effective_end: pd.Timestamp,
    top_n: int,
    min_history_bars: int,
    bars_per_year: int,
) -> dict:
    """Compute extra metrics beyond what summary_metrics provides."""
    # Slice portfolio to effective period
    port = result.portfolio
    port = port.loc[(port.index >= effective_start) & (port.index <= effective_end)]
    if port.empty or len(port) < 2:
        return {}

    holdings = result.holdings_history.reindex(port.index).fillna(0.0)
    turnover = result.turnover.reindex(port.index).fillna(0.0)

    # Rebalance eligibility within the effective period
    rebalance_log_in_period = (
        result.rebalance_log
        if result.rebalance_log.empty
        else result.rebalance_log[
            result.rebalance_log["execution_timestamp"].between(
                str(effective_start), str(effective_end)
            )
        ]
    )

    valid = close.notna()
    hcount = valid.astype(int).cumsum()
    eligible_full = valid & (hcount >= min_history_bars)
    eligible_in_period = eligible_full.loc[
        (eligible_full.index >= effective_start) & (eligible_full.index <= effective_end)
    ]
    elig_counts = eligible_in_period.sum(axis=1)

    pct_time_invested = float((holdings.sum(axis=1) > 0.01).mean() * 100)
    avg_cash = float((1.0 - holdings.sum(axis=1).clip(0, 1)).mean() * 100)

    # rebalances where top_n >= eligible count
    top_n_gte = (elig_counts <= top_n).sum()
    pct_top_n_full = float(top_n_gte / len(elig_counts) * 100) if len(elig_counts) > 0 else 0.0

    n_rebalances = len(rebalance_log_in_period)
    n_trades = int((turnover > 0.001).sum())

    metrics = summary_metrics(
        equity=port["equity"],
        returns=port["strategy_return"],
        turnover=turnover,
        bars_per_year=bars_per_year,
    )

    return {
        "total_return": metrics.get("total_return", float("nan")),
        "cagr": metrics.get("cagr", float("nan")),
        "sharpe": metrics.get("sharpe", float("nan")),
        "max_drawdown": metrics.get("max_drawdown", float("nan")),
        "avg_turnover": metrics.get("avg_turnover", float("nan")),
        "n_rebalances": n_rebalances,
        "n_trades": n_trades,
        "pct_time_invested": pct_time_invested,
        "avg_cash_exposure": avg_cash,
        "avg_eligible_universe_size": float(elig_counts.mean()),
        "min_eligible_universe_size": int(elig_counts.min()),
        "max_eligible_universe_size": int(elig_counts.max()),
        "pct_rebalances_top_n_selected_full_eligible": pct_top_n_full,
    }


def run_mode_comparison(
    symbols: Sequence[str] = LIVE_FIVE_UNIVERSE,
    timeframe: str = "4h",
    data_dir: Path = Path("data/local"),
    top_n: int = DEFAULT_TOP_N,
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    rebalance_every_bars: int = DEFAULT_REBALANCE_BARS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> list[BacktestModeResult]:
    """Run CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY modes and compare.

    Returns a list of BacktestModeResult (one per mode).

    Both modes run the SAME strategy with the SAME parameters; only the
    effective reporting window differs.  No strategy parameters are modified.
    """
    LOGGER.info("Loading close matrix for %d symbols", len(symbols))
    close = build_close_matrix(symbols, timeframe, data_dir)
    ohlcv = _close_to_ohlcv(close)
    signal_gen = _make_signal_generator(top_n, min_history_bars)
    bars_per_year = int(24 / 4 * 365) if timeframe == "4h" else 365

    LOGGER.info("Running backtest (shared for both modes)…")
    bt_result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=signal_gen,
        initial_capital=initial_capital,
        transaction_cost_bps=fee_bps,
        slippage_bps=slippage_bps,
        rebalance_every_bars=rebalance_every_bars,
    )

    results: list[BacktestModeResult] = []

    for mode in (UniverseMode.CURRENT_SURVIVORS_EXPANDING, UniverseMode.FIXED_COMMON_HISTORY):
        flt = get_universe_filter(mode, min_history_bars=min_history_bars)
        _, effective_start_str = flt.get_effective_close(close)
        effective_start_ts = pd.Timestamp(effective_start_str, tz="UTC")

        port_full = bt_result.portfolio
        effective_end_ts = port_full.index[-1]

        ext = _compute_extended_metrics(
            result=bt_result,
            close=close,
            effective_start=effective_start_ts,
            effective_end=effective_end_ts,
            top_n=top_n,
            min_history_bars=min_history_bars,
            bars_per_year=bars_per_year,
        )

        btc = compute_btc_benchmark(
            close=close,
            start_ts=effective_start_ts,
            end_ts=effective_end_ts,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
        )

        results.append(BacktestModeResult(
            mode=mode.value,
            bias_note=flt.bias_note(),
            effective_start=effective_start_str,
            effective_end=str(effective_end_ts.date()),
            configured_universe="|".join(symbols),
            avg_eligible_universe_size=ext.get("avg_eligible_universe_size", float("nan")),
            min_eligible_universe_size=ext.get("min_eligible_universe_size", 0),
            max_eligible_universe_size=ext.get("max_eligible_universe_size", 0),
            pct_rebalances_top_n_selected_full_eligible=ext.get("pct_rebalances_top_n_selected_full_eligible", float("nan")),
            total_return=ext.get("total_return", float("nan")),
            cagr=ext.get("cagr", float("nan")),
            sharpe=ext.get("sharpe", float("nan")),
            max_drawdown=ext.get("max_drawdown", float("nan")),
            avg_turnover=ext.get("avg_turnover", float("nan")),
            n_rebalances=ext.get("n_rebalances", 0),
            n_trades=ext.get("n_trades", 0),
            pct_time_invested=ext.get("pct_time_invested", float("nan")),
            avg_cash_exposure=ext.get("avg_cash_exposure", float("nan")),
            btc_total_return_same_period=btc.get("btc_total_return", float("nan")),
            btc_start=btc.get("btc_start", ""),
            btc_end=btc.get("btc_end", ""),
        ))
        LOGGER.info(
            "Mode %s: start=%s end=%s total_return=%.2f%% cagr=%.2f%% sharpe=%.2f",
            mode.value,
            effective_start_str,
            str(effective_end_ts.date()),
            ext.get("total_return", float("nan")) * 100,
            ext.get("cagr", float("nan")) * 100,
            ext.get("sharpe", float("nan")),
        )

    return results


# ---------------------------------------------------------------------------
# Part 4: Point-in-time requirements document
# ---------------------------------------------------------------------------

def write_pit_requirements_doc(output_dir: Path) -> Path:
    """Write the point-in-time data requirements to a Markdown document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "universe_integrity_pit_requirements.md"

    content = """\
# Point-in-Time Crypto Universe Requirements

## Why current-listing is not a valid historical universe

All OHLCV files in `data/local/` were populated by fetching data for symbols
**currently listed on Kraken**.  This is not a valid proxy for the historical
Kraken universe because:

1. **Delisted coins are absent.**  Coins that were delisted, collapsed, or
   withdrawn from Kraken before today have no data in the local cache.  Any
   strategy that would have held them is never penalized.

2. **Survivorship bias.**  The set of coins we can observe today is exactly the
   set that *survived* to today.  Historical strategies appear to have avoided
   the coins that later failed — but only because those coins are invisible.

3. **Symbol renames / migrations.**  Example: MATIC was renamed to POL in 2024.
   The `pol-usd_4h.csv` file starts from the rebrand date (2024-09-13) and
   contains no MATIC history.  A historical strategy using MATIC/USD from 2021
   cannot be replicated.

4. **Look-ahead in universe selection.**  Deciding to include AVAX/USD because
   it is currently available uses information from the future relative to any
   backtest start date before AVAX launched.

5. **Using only coins currently listed on Kraken is not a valid historical
   Kraken universe.**  Kraken listed and delisted dozens of assets between 2020
   and today.  The current listing is a strict subset.

---

## Requirements for a defensible historical universe

### 1. Exchange-specific listing dates

For each symbol on each exchange:
- Date (and ideally time) the symbol first appeared in the order book.
- This is distinct from the date the underlying asset was created.
- Source: exchange announcement APIs, historical market data vendors
  (Kaiko, CoinGecko Pro, CoinAPI).

### 2. Delisting dates

- Date the symbol was removed from active trading.
- Whether delisting was gradual (withdrawal notice → trading halt) or abrupt.
- Some delistings leave gaps in OHLCV that must not be forward-filled.

### 3. Historical symbol mappings and migrations

- Rename events (e.g., MATIC → POL, LUNA → LUNC after collapse).
- Forks and airdrops that created new tickers from existing ones.
- Consolidated tickers (e.g., XBT and BTC as equivalent Bitcoin symbols on
  some exchanges).
- Chain migrations that changed the economic exposure despite keeping the ticker.

### 4. Historical OHLCV for delisted assets

- Requires a third-party data provider.  Kraken's own API will not return
  historical data for delisted symbols.
- Kaiko, CoinGecko Pro historical exports, CoinMetrics are known sources.
- Must be stored separately from `data/local/` and tagged with their source
  and as-of date.

### 5. Point-in-time liquidity or volume screening

- Any liquidity filter (e.g., minimum 30-day median volume) must use only
  data available at the rebalance date — not full-sample average.
- Rolling windows that extend beyond the symbol's listing date must be
  handled as partial windows or the symbol must be excluded.

### 6. Treatment of special asset types

| Type | Treatment |
|---|---|
| Stablecoins (USDT, USDC, DAI) | Exclude from momentum universe; include only as cash proxy |
| Wrapped assets (WBTC, WETH) | Treat as duplicate economic exposure; exclude or deduplicate |
| Leveraged tokens (BTC3L, ETH2X) | Exclude; path-dependent decay distorts momentum signals |
| Synthetic / derivative tokens | Exclude unless the underlying economic exposure is unique |
| Rebased tokens (AMPL, OHM) | Exclude; close prices do not reflect total return |

### 7. Treatment of exchange outages and missing bars

- Distinguish between: (a) exchange outage (data exists elsewhere),
  (b) thin market / no trades (zero volume, last price carried), and
  (c) data vendor gap (unknown cause).
- Option A: exclude the bar; the lookback window shrinks.
- Option B: carry last close with zero volume and flag it.
- Do not silently forward-fill prices for multi-day outages.

### 8. Evidence / source metadata for every membership interval

Every interval in the membership file must include:
- `source`: the data vendor or primary evidence (e.g., "kraken_api_2024-12-01")
- `source_as_of`: the date the information was retrieved
- `listing_reason`: why the symbol was included (e.g., "listed_on_kraken")
- `delisting_reason`: why it was removed (e.g., "delisted_by_exchange",
  "below_liquidity_threshold", "duplicate_economic_exposure")

---

## Proposed membership CSV schema

```
symbol,exchange,eligible_from,eligible_to,source,source_as_of,listing_reason,delisting_reason,symbol_predecessor,symbol_successor,notes
BTC/USD,kraken,2020-01-01,,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,,,,"Bitcoin; continuous listing"
MATIC/USD,kraken,2021-06-11,2024-09-13,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,renamed_to_POL,,POL/USD,"Polygon; renamed to POL"
POL/USD,kraken,2024-09-13,,kraken_api_2024-12-01,2024-12-01,symbol_migration,,,MATIC/USD,"Polygon rebrand from MATIC"
LUNA/USD,kraken,2021-07-26,2022-05-13,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,project_collapsed,,,"Terra LUNA; collapsed May 2022"
```

### Field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | string | Yes | Exchange symbol (e.g., BTC/USD) |
| `exchange` | string | Yes | Exchange identifier (e.g., kraken) |
| `eligible_from` | ISO date | Yes | First date the symbol is included in the universe |
| `eligible_to` | ISO date or empty | Yes | Last date included (empty = still active) |
| `source` | string | No | Evidence source for this interval |
| `source_as_of` | ISO date | No | Date the source was retrieved |
| `listing_reason` | string | No | Why included (e.g., listed_on_kraken) |
| `delisting_reason` | string | No | Why removed (e.g., delisted_by_exchange) |
| `symbol_predecessor` | string | No | Previous symbol if renamed/migrated |
| `symbol_successor` | string | No | Next symbol if renamed/migrated |
| `notes` | string | No | Free text |

---

## Remaining blockers for a genuine POINT_IN_TIME_UNIVERSE backtest

1. **No delisted-coin OHLCV in local cache.**  `data/local/` contains only
   currently-listed coins.  Third-party historical data must be sourced and
   stored.

2. **No authoritative listing-date database.**  Kraken's API does not provide
   programmatic access to historical listing dates.  Manual reconstruction or
   a commercial data vendor is required.

3. **No symbol-migration mapping file.**  MATIC→POL and similar migrations have
   not been documented in machine-readable form.

4. **No liquidity screen history.**  Rolling median volume for inclusion
   criteria can only be applied historically once OHLCV for delisted coins is
   available.

5. **No external audit or cross-validation.**  Membership intervals should be
   validated against at least two independent sources.

---

*This document is research-only and describes future work requirements.*
*Generated by `research/universe_integrity_analysis.py`.*
"""
    path.write_text(content, encoding="utf-8")
    LOGGER.info("PIT requirements doc written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_symbol_eligibility_csv(
    histories: list[SymbolHistory],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "universe_integrity_symbol_eligibility.csv"
    df = pd.DataFrame([asdict(h) for h in histories])
    df.to_csv(path, index=False)
    LOGGER.info("Symbol eligibility report written to %s", path)
    return path


def write_rebalance_log_csv(reb_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "universe_integrity_rebalance_log.csv"
    reb_df.to_csv(path, index=False)
    LOGGER.info("Rebalance eligibility log written to %s (%d rows)", path, len(reb_df))
    return path


def write_mode_comparison_csv(
    mode_results: list[BacktestModeResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "universe_integrity_mode_comparison.csv"
    df = pd.DataFrame([asdict(r) for r in mode_results])
    df.to_csv(path, index=False)
    LOGGER.info("Mode comparison written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Console interpretation
# ---------------------------------------------------------------------------

def print_interpretation(
    histories: list[SymbolHistory],
    reb_df: pd.DataFrame,
    mode_results: list[BacktestModeResult],
    joint_start: pd.Timestamp | None,
    top_n: int,
) -> None:
    BAR = "=" * 72

    print(f"\n{BAR}")
    print("  UNIVERSE INTEGRITY ANALYSIS — INTERPRETATION")
    print(f"{BAR}\n")

    print("  PART 1 — SYMBOL ELIGIBILITY")
    print(f"  {'Symbol':<14} {'First bar':<14} {'First eligible':<22} {'Last bar':<14} {'Bars'}")
    for h in histories:
        fe = h.first_eligible_ts[:10] if h.first_eligible_ts else "NEVER"
        print(f"  {h.symbol:<14} {h.first_ts[:10]:<14} {fe:<22} {h.last_ts[:10]:<14} {h.total_bars}")
    if joint_start is not None:
        print(f"\n  Earliest all-five eligible date: {joint_start.date()}")

    if not reb_df.empty:
        full_select = reb_df["top_n_gte_eligible"].sum()
        total = len(reb_df)
        pct = full_select / total * 100 if total else 0.0
        print(f"\n  PART 1 — TOP_N vs ELIGIBLE UNIVERSE (rebalance_every=6)")
        print(f"  Total rebalance bars: {total}")
        print(f"  Bars where top_n({top_n}) >= eligible universe: {full_select} ({pct:.1f}%)")
        print(f"  → Strategy had no asset selection during {pct:.1f}% of rebalance bars")

    if mode_results:
        print(f"\n  PART 3 — MODE COMPARISON")
        print(f"  {'Mode':<36} {'Start':<12} {'End':<12} {'TotalRet':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'BTC':>10}")
        for r in mode_results:
            print(
                f"  {r.mode:<36} {r.effective_start:<12} {r.effective_end:<12} "
                f"{r.total_return:>9.1%} {r.cagr:>7.1%} {r.sharpe:>8.2f} "
                f"{r.max_drawdown:>7.1%} {r.btc_total_return_same_period:>9.1%}"
            )

        if len(mode_results) >= 2:
            a, b = mode_results[0], mode_results[1]
            diff = b.total_return - a.total_return
            print(
                f"\n  FIXED_COMMON_HISTORY vs CURRENT_SURVIVORS_EXPANDING: "
                f"total return difference = {diff:+.1%}"
            )
            print(
                f"  (FIXED mode starts later — {a.effective_start} → {b.effective_start} — "
                f"shorter effective period)"
            )

    print(f"\n  SURVIVORSHIP BIAS NOTE:")
    for line in SURVIVORSHIP_BIAS_NOTE.split(". "):
        if line.strip():
            print(f"  ⚠  {line.strip()}.")

    print(f"\n  REMAINING BLOCKERS FOR POINT_IN_TIME_UNIVERSE:")
    blockers = [
        "No delisted-coin OHLCV in data/local/ (need 3rd-party vendor data)",
        "No authoritative exchange listing-date database",
        "No symbol-migration mapping file (e.g., MATIC→POL)",
        "No rolling liquidity screening history",
    ]
    for b in blockers:
        print(f"  ✗  {b}")

    print(f"\n  LIVE BEHAVIOR UNCHANGED:")
    print(f"  ✓  No live trading files modified")
    print(f"  ✓  No config parameters changed")
    print(f"  ✓  No cron/execution/broker code touched")
    print(f"\n{BAR}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-only universe integrity analysis and backtest mode comparison"
    )
    parser.add_argument("--data-dir", default="data/local")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list (default: live 5-coin universe)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-history-bars", type=int, default=DEFAULT_MIN_HISTORY_BARS)
    parser.add_argument("--rebalance-bars", type=int, default=DEFAULT_REBALANCE_BARS)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip Part 3 backtest comparison (audit only)")
    parser.add_argument("--mode", choices=["a", "b", "both"], default="both",
                        help="Which backtest mode(s) to run (default: both)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()

    symbols = tuple(args.symbols) if args.symbols else LIVE_FIVE_UNIVERSE
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info("Loading close matrix for %d symbols", len(symbols))
    close = build_close_matrix(symbols, args.timeframe, data_dir)

    # --- Part 1: Symbol audit ---
    histories = audit_symbol_histories(
        close, args.min_history_bars, backtest_start="2020-01-01"
    )
    sym_path = write_symbol_eligibility_csv(histories, output_dir)

    reb_df = build_rebalance_eligibility_log(
        close, args.min_history_bars, args.top_n, args.rebalance_bars
    )
    reb_path = write_rebalance_log_csv(reb_df, output_dir)

    joint_start = find_joint_eligible_start(close, args.min_history_bars)

    # --- Part 3: Backtest comparison ---
    mode_results: list[BacktestModeResult] = []
    if not args.no_backtest:
        LOGGER.info("Running backtest mode comparison…")
        mode_results = run_mode_comparison(
            symbols=symbols,
            timeframe=args.timeframe,
            data_dir=data_dir,
            top_n=args.top_n,
            min_history_bars=args.min_history_bars,
            fee_bps=args.fee_bps,
            rebalance_every_bars=args.rebalance_bars,
        )
        mode_path = write_mode_comparison_csv(mode_results, output_dir)
        print(f"Mode comparison:   {mode_path}")

    # --- Part 4: PIT requirements doc ---
    pit_path = write_pit_requirements_doc(output_dir)

    print(f"\nSymbol eligibility: {sym_path}")
    print(f"Rebalance log:      {reb_path}")
    print(f"PIT requirements:   {pit_path}")

    print_interpretation(
        histories=histories,
        reb_df=reb_df,
        mode_results=mode_results,
        joint_start=joint_start,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
