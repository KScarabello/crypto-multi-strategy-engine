"""Tests for research/universe_integrity_analysis.py.

Synthetic fixtures cover all 10 required scenarios:
1. Symbol present from backtest start
2. Late-listed symbol
3. Delisted symbol (via PIT membership)
4. Symbol change / migration (via PIT membership)
5. Insufficient lookback history
6. Top_N larger than eligible universe
7. Missing symbol after FIXED_COMMON_HISTORY begins
8. BTC benchmark date alignment
9. No silent universe fallback between modes
10. No imports from live broker/exchange execution modules
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from research.universe_integrity_analysis import (
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_TOP_N,
    LIVE_FIVE_UNIVERSE,
    PIT_MEMBERSHIP_REQUIRED_COLS,
    SURVIVORSHIP_BIAS_NOTE,
    BacktestModeResult,
    CurrentSurvivorsExpandingFilter,
    FixedCommonHistoryFilter,
    PointInTimeUniverseFilter,
    UniverseFilterError,
    UniverseMode,
    audit_symbol_histories,
    build_rebalance_eligibility_log,
    compute_btc_benchmark,
    find_first_eligible_ts,
    find_joint_eligible_start,
    get_eligible_symbols_at_ts,
    get_universe_filter,
    validate_pit_membership_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close(symbols: list[str], n_bars: int = 200, start: str = "2020-01-01") -> pd.DataFrame:
    """Return a clean close-price matrix for all symbols from start."""
    idx = pd.date_range(start, periods=n_bars, freq="4h", tz="UTC")
    data = {sym: [100.0 + i * 0.01 for i in range(n_bars)] for sym in symbols}
    return pd.DataFrame(data, index=idx)


def _make_close_with_late(
    symbols: list[str],
    late_symbol: str,
    late_start: str,
    n_bars: int = 300,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Return close matrix where late_symbol starts at late_start."""
    idx = pd.date_range(start, periods=n_bars, freq="4h", tz="UTC")
    data = {}
    late_idx = pd.date_range(late_start, periods=n_bars, freq="4h", tz="UTC")
    for sym in symbols:
        if sym == late_symbol:
            col = pd.Series(float("nan"), index=idx)
            for ts in late_idx:
                if ts in idx:
                    col.loc[ts] = 100.0
            data[sym] = col
        else:
            data[sym] = [100.0 + i * 0.01 for i in range(n_bars)]
    return pd.DataFrame(data, index=idx)


def _make_pit_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal PIT membership DataFrame."""
    df = pd.DataFrame(rows)
    for col in PIT_MEMBERSHIP_REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


# ---------------------------------------------------------------------------
# Scenario 1 — Symbol present from backtest start
# ---------------------------------------------------------------------------

class TestSymbolFromStart:
    def test_first_eligible_ts_is_early(self):
        close = _make_close(["BTC/USD"], n_bars=200)
        ts = find_first_eligible_ts(close["BTC/USD"], min_history_bars=36)
        assert ts is not None
        # Should be within 36 bars of the start
        start = close.index[0]
        expected = close.index[35]
        assert ts == expected

    def test_audit_no_issues(self):
        close = _make_close(["BTC/USD"], n_bars=200)
        histories = audit_symbol_histories(close, min_history_bars=36)
        h = histories[0]
        assert h.symbol == "BTC/USD"
        assert h.starts_after_backtest_start is False
        assert h.total_bars == 200
        assert h.usable_bars == 200 - 36 + 1

    def test_expanding_filter_uses_full_history(self):
        close = _make_close(["BTC/USD", "ETH/USD"], n_bars=200)
        flt = CurrentSurvivorsExpandingFilter()
        out_close, start = flt.get_effective_close(close)
        assert len(out_close) == len(close)
        assert start == str(close.index[0].date())


# ---------------------------------------------------------------------------
# Scenario 2 — Late-listed symbol
# ---------------------------------------------------------------------------

class TestLateListedSymbol:
    def test_starts_after_backtest_start(self):
        # n_bars=2000 so that late_start="2021-01-01" falls inside the index
        close = _make_close_with_late(
            ["BTC/USD", "SOL/USD"], "SOL/USD", "2021-01-01", n_bars=2000
        )
        histories = audit_symbol_histories(close, min_history_bars=36, backtest_start="2020-01-01")
        sol = next(h for h in histories if h.symbol == "SOL/USD")
        assert sol.starts_after_backtest_start is True
        assert sol.notes != "No issues."

    def test_not_eligible_before_listing(self):
        close = _make_close_with_late(
            ["BTC/USD", "SOL/USD"], "SOL/USD", "2020-03-01", n_bars=500
        )
        # Early timestamp — SOL not yet listed
        early_ts = close.index[10]
        eligible = get_eligible_symbols_at_ts(close, early_ts, min_history_bars=36)
        assert "SOL/USD" not in eligible

    def test_eligible_after_listing_plus_lookback(self):
        # Use 3000 bars so 2020-03-01 AND 2020-04-01 are both in the index
        close = _make_close_with_late(
            ["BTC/USD", "SOL/USD"], "SOL/USD", "2020-03-01", n_bars=3000
        )
        late_ts = pd.Timestamp("2020-04-01", tz="UTC")
        eligible = get_eligible_symbols_at_ts(close, late_ts, min_history_bars=36)
        assert "SOL/USD" in eligible

    def test_joint_start_constrained_by_late_symbol(self):
        close = _make_close_with_late(
            ["BTC/USD", "SOL/USD"], "SOL/USD", "2020-06-01", n_bars=1000
        )
        joint = find_joint_eligible_start(close, min_history_bars=36)
        assert joint is not None
        # Joint start must be after 2020-06-01 + 36 * 4h = ~6 days
        assert joint > pd.Timestamp("2020-06-01", tz="UTC")


# ---------------------------------------------------------------------------
# Scenario 3 — Delisted symbol via PIT membership
# ---------------------------------------------------------------------------

class TestDelistedSymbol:
    def test_delisted_symbol_excluded_after_eligible_to(self):
        membership = _make_pit_df([
            {"symbol": "LUNA/USD", "exchange": "kraken",
             "eligible_from": "2021-01-01", "eligible_to": "2022-05-13"},
            {"symbol": "BTC/USD", "exchange": "kraken",
             "eligible_from": "2020-01-01", "eligible_to": ""},
        ])
        flt = PointInTimeUniverseFilter(membership_df=membership, allow_synthetic=True)
        # Before delisting
        ts_before = pd.Timestamp("2022-01-01", tz="UTC")
        eligible = flt.get_eligible_symbols_at_ts(ts_before)
        assert "LUNA/USD" in eligible
        # After delisting
        ts_after = pd.Timestamp("2022-06-01", tz="UTC")
        eligible_after = flt.get_eligible_symbols_at_ts(ts_after)
        assert "LUNA/USD" not in eligible_after

    def test_btc_still_active_after_luna_delisting(self):
        membership = _make_pit_df([
            {"symbol": "LUNA/USD", "exchange": "kraken",
             "eligible_from": "2021-01-01", "eligible_to": "2022-05-13"},
            {"symbol": "BTC/USD", "exchange": "kraken",
             "eligible_from": "2020-01-01", "eligible_to": ""},
        ])
        flt = PointInTimeUniverseFilter(membership_df=membership, allow_synthetic=True)
        ts_after = pd.Timestamp("2023-01-01", tz="UTC")
        eligible = flt.get_eligible_symbols_at_ts(ts_after)
        assert "BTC/USD" in eligible


# ---------------------------------------------------------------------------
# Scenario 4 — Symbol change / migration
# ---------------------------------------------------------------------------

class TestSymbolMigration:
    def test_predecessor_excluded_after_migration(self):
        membership = _make_pit_df([
            {"symbol": "MATIC/USD", "exchange": "kraken",
             "eligible_from": "2021-06-01", "eligible_to": "2024-09-13",
             "symbol_successor": "POL/USD"},
            {"symbol": "POL/USD", "exchange": "kraken",
             "eligible_from": "2024-09-13", "eligible_to": "",
             "symbol_predecessor": "MATIC/USD"},
        ])
        flt = PointInTimeUniverseFilter(membership_df=membership, allow_synthetic=True)
        ts_before = pd.Timestamp("2022-01-01", tz="UTC")
        ts_after = pd.Timestamp("2024-10-01", tz="UTC")
        eligible_before = flt.get_eligible_symbols_at_ts(ts_before)
        eligible_after = flt.get_eligible_symbols_at_ts(ts_after)
        assert "MATIC/USD" in eligible_before
        assert "MATIC/USD" not in eligible_after
        assert "POL/USD" in eligible_after

    def test_no_overlap_during_migration(self):
        membership = _make_pit_df([
            {"symbol": "MATIC/USD", "exchange": "kraken",
             "eligible_from": "2021-06-01", "eligible_to": "2024-09-13"},
            {"symbol": "POL/USD", "exchange": "kraken",
             "eligible_from": "2024-09-13", "eligible_to": ""},
        ])
        flt = PointInTimeUniverseFilter(membership_df=membership, allow_synthetic=True)
        # At exact migration date, POL starts; MATIC ends (eligible_to is exclusive)
        ts_migration = pd.Timestamp("2024-09-13", tz="UTC")
        eligible = flt.get_eligible_symbols_at_ts(ts_migration)
        assert "POL/USD" in eligible
        assert "MATIC/USD" not in eligible


# ---------------------------------------------------------------------------
# Scenario 5 — Insufficient lookback history
# ---------------------------------------------------------------------------

class TestInsufficientLookback:
    def test_symbol_with_fewer_bars_than_lookback(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="4h", tz="UTC")
        close = pd.DataFrame({"TINY/USD": [100.0] * 30}, index=idx)
        ts = find_first_eligible_ts(close["TINY/USD"], min_history_bars=36)
        assert ts is None

    def test_not_eligible_in_rebalance_log(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        # BTC has full history, TINY only last 10 bars
        btc = [100.0] * 100
        tiny = [float("nan")] * 90 + [100.0] * 10
        close = pd.DataFrame({"BTC/USD": btc, "TINY/USD": tiny}, index=idx)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=2)
        # Late rows should have TINY ineligible
        early_row = reb_df.iloc[0]
        assert "TINY/USD" not in early_row["eligible_symbols"]

    def test_audit_history_flags_insufficient(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="4h", tz="UTC")
        close = pd.DataFrame({"TINY/USD": [100.0] * 20}, index=idx)
        histories = audit_symbol_histories(close, min_history_bars=36)
        h = histories[0]
        assert h.first_eligible_ts == ""
        assert h.usable_bars == 0

    def test_joint_start_none_if_never_all_eligible(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0] * 30,
            "TINY/USD": [100.0] * 30,  # only 30 bars, less than 36
        }, index=idx)
        joint = find_joint_eligible_start(close, min_history_bars=36)
        assert joint is None


# ---------------------------------------------------------------------------
# Scenario 6 — Top_N >= eligible universe
# ---------------------------------------------------------------------------

class TestTopNGeEligible:
    def test_top_n_gte_eligible_flag(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        # Only BTC has full history; top_n=3 but eligible=1
        btc = [100.0] * 100
        close = pd.DataFrame({"BTC/USD": btc}, index=idx)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=3)
        # After 36 bars, BTC is eligible but count=1 <= top_n=3
        row = reb_df[reb_df["eligible_universe_size"] >= 1].iloc[0]
        assert row["top_n_gte_eligible"] is True or row["top_n_gte_eligible"] == True

    def test_pct_of_eligible_exceeds_100_when_top_n_gt_eligible(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        close = pd.DataFrame({"BTC/USD": [100.0] * 100}, index=idx)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=3)
        row = reb_df[reb_df["eligible_universe_size"] == 1].iloc[0]
        assert row["top_n_pct_of_eligible"] == pytest.approx(300.0)

    def test_three_symbols_with_top_n_equals_three(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0] * 100,
            "ETH/USD": [100.0] * 100,
            "XRP/USD": [100.0] * 100,
        }, index=idx)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=3)
        # After warm-up, all 3 eligible; top_n=3 = 100%
        late_rows = reb_df[reb_df["eligible_universe_size"] == 3]
        assert len(late_rows) > 0
        assert late_rows["top_n_gte_eligible"].all()


# ---------------------------------------------------------------------------
# Scenario 7 — Missing symbol after FIXED_COMMON_HISTORY begins
# ---------------------------------------------------------------------------

class TestMissingSymbolAfterFixedStart:
    def test_raises_if_symbol_vanishes_after_joint_start(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="4h", tz="UTC")
        btc = [100.0] * 200
        # SOL has exactly 36 valid bars (0..35), then all NaN.
        # joint_start = idx[35]; bars after idx[35] are all NaN for SOL.
        sol = [100.0] * 36 + [float("nan")] * 164
        close = pd.DataFrame({"BTC/USD": btc, "SOL/USD": sol}, index=idx)
        flt = FixedCommonHistoryFilter(min_history_bars=36)
        with pytest.raises(UniverseFilterError, match="no data after"):
            flt.get_effective_close(close)

    def test_raises_if_joint_start_unreachable(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0] * 30,
            "GHOST/USD": [float("nan")] * 30,
        }, index=idx)
        flt = FixedCommonHistoryFilter(min_history_bars=36)
        with pytest.raises(UniverseFilterError):
            flt.get_effective_close(close)


# ---------------------------------------------------------------------------
# Scenario 8 — BTC benchmark date alignment
# ---------------------------------------------------------------------------

class TestBenchmarkDateAlignment:
    def test_benchmark_uses_exact_start_end(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0 + i * 0.1 for i in range(500)],
        }, index=idx)
        start = idx[50]
        end = idx[499]
        result = compute_btc_benchmark(close, start, end, initial_capital=10000.0, fee_bps=10)
        assert result["btc_start"] == str(idx[50].date())
        assert result["btc_end"] == str(idx[499].date())

    def test_benchmark_return_is_correct(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        prices = [100.0] * 50 + [200.0] * 50  # price doubles
        close = pd.DataFrame({"BTC/USD": prices}, index=idx)
        start = idx[0]
        end = idx[99]
        result = compute_btc_benchmark(close, start, end, initial_capital=10000.0, fee_bps=0)
        # 100 → 200 = 100% return, no fees
        assert result["btc_total_return"] == pytest.approx(1.0, abs=0.001)

    def test_benchmark_applies_fee(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        prices = [100.0] * 100  # flat price
        close = pd.DataFrame({"BTC/USD": prices}, index=idx)
        start = idx[0]
        end = idx[99]
        result = compute_btc_benchmark(close, start, end, initial_capital=10000.0, fee_bps=50)
        # Gross return = 0; round-trip cost = 2 * 50 bps = 1%
        assert result["btc_total_return"] == pytest.approx(-0.01, abs=0.0001)

    def test_benchmark_nan_when_btc_missing(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        close = pd.DataFrame({"ETH/USD": [100.0] * 100}, index=idx)
        result = compute_btc_benchmark(close, idx[0], idx[-1])
        assert math.isnan(result["btc_total_return"])

    def test_benchmark_same_period_as_mode(self):
        """Benchmark dates must align with the mode effective period."""
        idx = pd.date_range("2020-01-01", periods=500, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0 + i for i in range(500)],
            "ETH/USD": [50.0 + i for i in range(500)],
        }, index=idx)
        # Fixed common history effective start = 36 bars in
        joint = find_joint_eligible_start(close, min_history_bars=36)
        end = idx[-1]
        result = compute_btc_benchmark(close, joint, end)
        assert result["btc_start"] == str(joint.date())


# ---------------------------------------------------------------------------
# Scenario 9 — No silent universe fallback
# ---------------------------------------------------------------------------

class TestNoSilentFallback:
    def test_get_universe_filter_raises_on_unknown_mode(self):
        with pytest.raises((ValueError, KeyError)):
            get_universe_filter("IMAGINARY_MODE")

    def test_pit_mode_raises_when_not_ready(self):
        """POINT_IN_TIME_UNIVERSE raises NotReadyError without allow_synthetic."""
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        close = pd.DataFrame({"BTC/USD": [100.0] * 100}, index=idx)
        membership = _make_pit_df([
            {"symbol": "BTC/USD", "exchange": "kraken",
             "eligible_from": "2020-01-01", "eligible_to": ""},
        ])
        flt = PointInTimeUniverseFilter(membership_df=membership, allow_synthetic=False)
        with pytest.raises(PointInTimeUniverseFilter.NotReadyError):
            flt.get_effective_close(close)

    def test_fixed_mode_does_not_fall_back_to_expanding(self):
        """FixedCommonHistoryFilter raises rather than silently using partial data."""
        idx = pd.date_range("2020-01-01", periods=20, freq="4h", tz="UTC")
        close = pd.DataFrame({
            "BTC/USD": [100.0] * 20,
            "GHOST/USD": [float("nan")] * 20,
        }, index=idx)
        flt = FixedCommonHistoryFilter(min_history_bars=36)
        with pytest.raises(UniverseFilterError):
            flt.get_effective_close(close)

    def test_expanding_and_fixed_are_different_modes(self):
        assert UniverseMode.CURRENT_SURVIVORS_EXPANDING != UniverseMode.FIXED_COMMON_HISTORY
        assert UniverseMode.FIXED_COMMON_HISTORY != UniverseMode.POINT_IN_TIME_UNIVERSE

    def test_bias_notes_are_distinct(self):
        exp = CurrentSurvivorsExpandingFilter()
        fix = FixedCommonHistoryFilter()
        pit = PointInTimeUniverseFilter(allow_synthetic=True)
        assert exp.bias_note() != fix.bias_note()
        assert fix.bias_note() != pit.bias_note()


# ---------------------------------------------------------------------------
# Scenario 10 — No imports from live broker/exchange execution modules
# ---------------------------------------------------------------------------

class TestNoLiveImports:
    def test_no_broker_imports(self):
        src = Path("research/universe_integrity_analysis.py").read_text()
        forbidden = ["from brokers", "import brokers", "from execution", "import execution",
                     "from live", "import live", "ccxt", "kraken.py"]
        for pattern in forbidden:
            assert pattern not in src, f"Found forbidden import: '{pattern}'"

    def test_no_live_trading_references(self):
        src = Path("research/universe_integrity_analysis.py").read_text()
        forbidden_patterns = ["place_order", "submit_order", "create_order",
                              "live_broker", "KrakenBroker"]
        for pattern in forbidden_patterns:
            assert pattern not in src, f"Found live trading reference: '{pattern}'"

    def test_survivorship_bias_note_present(self):
        assert len(SURVIVORSHIP_BIAS_NOTE) > 50
        assert "survivorship" in SURVIVORSHIP_BIAS_NOTE.lower()
        assert "POINT_IN_TIME" in SURVIVORSHIP_BIAS_NOTE


# ---------------------------------------------------------------------------
# PIT schema validation
# ---------------------------------------------------------------------------

class TestPITSchemaValidation:
    def test_valid_schema_passes(self):
        df = _make_pit_df([
            {"symbol": "BTC/USD", "exchange": "kraken",
             "eligible_from": "2020-01-01", "eligible_to": ""},
        ])
        validate_pit_membership_schema(df)  # should not raise

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"symbol": ["BTC/USD"], "exchange": ["kraken"]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_pit_membership_schema(df)

    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=PIT_MEMBERSHIP_REQUIRED_COLS)
        with pytest.raises(ValueError, match="empty"):
            validate_pit_membership_schema(df)

    def test_unparseable_eligible_from_raises(self):
        df = _make_pit_df([
            {"symbol": "BTC/USD", "exchange": "kraken",
             "eligible_from": "not-a-date", "eligible_to": ""},
        ])
        with pytest.raises(ValueError):
            validate_pit_membership_schema(df)


# ---------------------------------------------------------------------------
# Rebalance eligibility log
# ---------------------------------------------------------------------------

class TestRebalanceLog:
    def test_log_has_expected_columns(self):
        close = _make_close(["BTC/USD", "ETH/USD"], n_bars=100)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=2)
        for col in ["timestamp", "eligible_universe_size", "top_n_gte_eligible",
                    "top_n_pct_of_eligible", "ineligibility_reasons"]:
            assert col in reb_df.columns

    def test_log_length_matches_rebalance_period(self):
        close = _make_close(["BTC/USD"], n_bars=100)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=2, rebalance_every_bars=6)
        expected = math.ceil(100 / 6)
        assert len(reb_df) == expected

    def test_ineligible_reasons_recorded(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        btc = [100.0] * 100
        late = [float("nan")] * 80 + [100.0] * 20
        close = pd.DataFrame({"BTC/USD": btc, "LATE/USD": late}, index=idx)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=2)
        early_row = reb_df.iloc[0]
        assert "LATE/USD" in early_row["ineligible_symbols"]

    def test_eligible_symbols_populated_after_warmup(self):
        close = _make_close(["BTC/USD"], n_bars=100)
        reb_df = build_rebalance_eligibility_log(close, min_history_bars=36, top_n=1)
        late_rows = reb_df[reb_df["eligible_universe_size"] >= 1]
        assert len(late_rows) > 0
        row = late_rows.iloc[0]
        assert "BTC/USD" in row["eligible_symbols"]


# ---------------------------------------------------------------------------
# FixedCommonHistoryFilter
# ---------------------------------------------------------------------------

class TestFixedCommonHistoryFilter:
    def test_joint_start_correct(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="4h", tz="UTC")
        btc = [100.0] * 200
        sol = [float("nan")] * 60 + [100.0] * 140
        close = pd.DataFrame({"BTC/USD": btc, "SOL/USD": sol}, index=idx)
        flt = FixedCommonHistoryFilter(min_history_bars=36)
        _, effective_start = flt.get_effective_close(close)
        # Joint start: SOL has 36 bars starting from index 60+35 = index 95
        joint = find_joint_eligible_start(close, min_history_bars=36)
        assert effective_start == str(joint.date())

    def test_full_close_returned_not_sliced(self):
        """The filter returns the full close (not sliced) — slicing is at reporting."""
        close = _make_close(["BTC/USD", "ETH/USD"], n_bars=200)
        flt = FixedCommonHistoryFilter(min_history_bars=36)
        out_close, _ = flt.get_effective_close(close)
        assert len(out_close) == len(close)

    def test_bias_note_mentions_survivorship(self):
        flt = FixedCommonHistoryFilter()
        note = flt.bias_note()
        assert "survivorship" in note.lower()
        assert "current-listing" in note.lower()


# ---------------------------------------------------------------------------
# CurrentSurvivorsExpandingFilter
# ---------------------------------------------------------------------------

class TestCurrentSurvivorsExpandingFilter:
    def test_returns_full_close(self):
        close = _make_close(["BTC/USD", "ETH/USD"], n_bars=200)
        flt = CurrentSurvivorsExpandingFilter()
        out, start = flt.get_effective_close(close)
        assert len(out) == 200
        assert start == "2020-01-01"

    def test_bias_note_mentions_survivorship(self):
        flt = CurrentSurvivorsExpandingFilter()
        note = flt.bias_note()
        assert "survivorship" in note.lower()


# ---------------------------------------------------------------------------
# BTC benchmark extra edge cases
# ---------------------------------------------------------------------------

class TestBTCBenchmarkEdgeCases:
    def test_start_after_data_end_returns_nan(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        close = pd.DataFrame({"BTC/USD": [100.0] * 100}, index=idx)
        start = pd.Timestamp("2030-01-01", tz="UTC")
        end = pd.Timestamp("2031-01-01", tz="UTC")
        result = compute_btc_benchmark(close, start, end)
        assert math.isnan(result["btc_total_return"])
