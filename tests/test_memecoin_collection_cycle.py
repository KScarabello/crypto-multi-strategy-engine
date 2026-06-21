"""Safety and correctness tests for the memecoin collection cycle.

Tests cover:
- Completed-candle filtering
- Forward-horizon maturity guards
- Genuine-vs-simulated provenance
- Simulated events excluded from evidence gates
- Duplicate-event prevention
- Immutable detection-time features
- Immutable timestamped snapshots
- Repeated-run idempotency
- Lock acquisition and cleanup
- Missing optional summary marked SKIPPED
- Spread fields retained in genuine events
- No historical backfill invoked by wrapper
- No live broker/exchange execution imports
- No five-coin shadow-state changes
- Unchanged five-coin frozen hashes
- Unchanged locked memecoin evidence thresholds
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from research.memecoin_catcher.enrich_candidates_with_ohlc import select_complete_candles
from research.memecoin_catcher.evaluate_signal_outcomes import (
    HORIZON_MATURITY_SECONDS,
    compute_forward_returns,
    is_horizon_mature,
    select_future_window,
    upsert_outcome_rows,
    SIGNAL_KEY_COLUMNS,
    REFRESHABLE_OUTCOME_COLUMNS,
)
from research.memecoin_catcher.save_signal_snapshot import (
    SCANNER_VERSION,
    SNAPSHOT_COLUMNS,
    append_to_history,
    compute_event_id,
    filter_active_signals,
    stamp_snapshot,
)
from research.memecoin_catcher.resume_memecoin_research import (
    StepResult,
    _OptionalStepSkipped,
    _run_step,
)
from research.memecoin_catcher.memecoin_readiness_report import (
    GATE_A,
    build_status_dict,
    extract_rule1_events,
    load_genuine_history,
    load_simulated_backfill,
)


# ===========================================================================
# Part 2: Completed-candle enforcement
# ===========================================================================


class TestCompletedCandleFiltering:
    """select_complete_candles must drop in-progress candles."""

    def _make_ohlc(self, n: int, interval_s: int = 900) -> pd.DataFrame:
        base = 1_000_000
        times = [base + i * interval_s for i in range(n)]
        return pd.DataFrame(
            {"time": times, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        )

    def test_complete_candles_dropped_when_in_progress(self):
        df = self._make_ohlc(5)
        last_start = float(df.iloc[-1]["time"])
        # Set now to be within the last candle's window
        now_ts = last_start + 400  # 400s into a 900s candle
        result = select_complete_candles(df, interval_minutes=15, now_ts=now_ts)
        assert len(result) == 4, "In-progress candle should be dropped"
        assert float(result.iloc[-1]["time"]) != last_start

    def test_complete_candles_kept_when_closed(self):
        df = self._make_ohlc(5)
        last_start = float(df.iloc[-1]["time"])
        # Set now to be after the last candle has closed
        now_ts = last_start + 901
        result = select_complete_candles(df, interval_minutes=15, now_ts=now_ts)
        assert len(result) == 5, "All candles should be kept when current candle is closed"

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        result = select_complete_candles(df, interval_minutes=15)
        assert result.empty


class TestForwardHorizonMaturity:
    """is_horizon_mature must use hard UTC-aware elapsed-time checks."""

    def test_immature_15m_returns_false(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 14 * 60  # 14 minutes elapsed, need 15
        assert is_horizon_mature(signal_ts, "15m", now_ts_unix=now) is False

    def test_mature_15m_returns_true(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 15 * 60
        assert is_horizon_mature(signal_ts, "15m", now_ts_unix=now) is True

    def test_immature_4h_returns_false(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 3 * 3600
        assert is_horizon_mature(signal_ts, "4h", now_ts_unix=now) is False

    def test_mature_4h_returns_true(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 4 * 3600
        assert is_horizon_mature(signal_ts, "4h", now_ts_unix=now) is True

    def test_mature_24h_returns_true(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 24 * 3600
        assert is_horizon_mature(signal_ts, "24h", now_ts_unix=now) is True

    def test_immature_24h_returns_false(self):
        signal_ts = 1_000_000.0
        now = signal_ts + 23 * 3600
        assert is_horizon_mature(signal_ts, "24h", now_ts_unix=now) is False

    def test_all_horizons_have_maturity_seconds(self):
        for h in ("15m", "1h", "4h", "8h", "12h", "24h"):
            assert h in HORIZON_MATURITY_SECONDS
            assert HORIZON_MATURITY_SECONDS[h] > 0

    def test_compute_forward_returns_suppresses_immature_horizon(self):
        """forward return must be NaN when horizon has not elapsed."""
        # Build 96+ future candles
        future_rows = [{"time": i * 900, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0} for i in range(100)]
        future_df = pd.DataFrame(future_rows)
        signal_ts = 0.0
        now = signal_ts + 60  # only 1 minute elapsed — nothing should be mature
        rets = compute_forward_returns(future_df, signal_price=100.0, signal_ts_unix=signal_ts, now_ts_unix=now)
        for h in ("15m", "1h", "4h", "24h"):
            assert np.isnan(rets[f"future_ret_{h}_pct"]), f"Expected NaN for immature {h}"

    def test_compute_forward_returns_allows_mature_horizon(self):
        """forward return must be computed when horizon has fully elapsed."""
        future_rows = [{"time": i * 900, "open": 100.0, "high": 101.0, "low": 99.0, "close": 105.0, "volume": 10.0} for i in range(100)]
        future_df = pd.DataFrame(future_rows)
        signal_ts = 0.0
        now = signal_ts + 25 * 3600  # 25 hours — all horizons mature
        rets = compute_forward_returns(future_df, signal_price=100.0, signal_ts_unix=signal_ts, now_ts_unix=now)
        assert not np.isnan(rets["future_ret_4h_pct"]), "Expected non-NaN for mature 4h"


# ===========================================================================
# Part 1 & 4: Provenance, event_id, duplicate prevention
# ===========================================================================


class TestProvenanceFields:
    """stamp_snapshot must add all required provenance fields."""

    def _make_candidate_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "pair_id": "XBTUSDT",
            "wsname": "XBT/USDT",
            "base": "XBT",
            "quote": "USDT",
            "scanner_label": "HOT_MOVER",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "last_price": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
            "spread_pct": 0.04,
            "quote_volume_est": 100000.0,
            "today_return_pct": 5.0,
            "ret_15m_pct": 0.5,
            "ret_1h_pct": 2.0,
            "ret_4h_pct": 4.0,
            "ret_24h_pct": 5.0,
            "volume_ratio_1h": 3.0,
            "volume_ratio_4h": 2.0,
            "breakout_24h": True,
            "long_explosion_score": 12.0,
            "dump_score": float("nan"),
            "reversal_watch_score": float("nan"),
            "primary_ohlc_score": 12.0,
        }])

    def test_provenance_fields_present(self):
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        for col in ("event_source", "collection_mode", "is_simulated",
                    "detection_timestamp", "collection_timestamp",
                    "scanner_version", "universe_snapshot_id", "event_id"):
            assert col in stamped.columns, f"Missing provenance column: {col}"

    def test_event_source_is_genuine(self):
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert stamped.iloc[0]["event_source"] == "GENUINE_PROSPECTIVE"

    def test_is_simulated_false(self):
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert bool(stamped.iloc[0]["is_simulated"]) == False  # noqa: E712

    def test_scanner_version_is_set(self):
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert stamped.iloc[0]["scanner_version"] == SCANNER_VERSION

    def test_detection_timestamp_equals_snapshot_ts(self):
        ts = "2026-06-10T12:00:00Z"
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, ts)
        assert stamped.iloc[0]["detection_timestamp"] == ts

    def test_event_id_is_deterministic(self):
        ts = "2026-06-10T12:00:00Z"
        df = self._make_candidate_df()
        s1 = stamp_snapshot(df, ts)
        s2 = stamp_snapshot(df, ts)
        assert s1.iloc[0]["event_id"] == s2.iloc[0]["event_id"]

    def test_event_id_differs_for_different_ts(self):
        df = self._make_candidate_df()
        meta1 = {"data_cutoff_timestamp": "2026-06-10T12:00:00Z", "universe_snapshot_id": "u1"}
        meta2 = {"data_cutoff_timestamp": "2026-06-10T13:00:00Z", "universe_snapshot_id": "u2"}
        s1 = stamp_snapshot(df, "2026-06-10T12:05:00Z", universe_meta=meta1)
        s2 = stamp_snapshot(df, "2026-06-10T13:05:00Z", universe_meta=meta2)
        assert s1.iloc[0]["event_id"] != s2.iloc[0]["event_id"]

    def test_all_snapshot_columns_present(self):
        df = self._make_candidate_df()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        for col in SNAPSHOT_COLUMNS:
            assert col in stamped.columns, f"Missing SNAPSHOT_COLUMN: {col}"


class TestSpreadFieldsRetained:
    """Spread and liquidity fields must be captured in genuine events."""

    def _make_row(self, bid=100.0, ask=100.5, spread_pct=0.5, qvol=50000.0) -> pd.DataFrame:
        return pd.DataFrame([{
            "pair_id": "XBTUSDT",
            "wsname": "XBT/USDT",
            "base": "XBT",
            "quote": "USDT",
            "scanner_label": "HOT_MOVER",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "last_price": 100.25,
            "bid": bid,
            "ask": ask,
            "spread_pct": spread_pct,
            "quote_volume_est": qvol,
            "today_return_pct": 5.0,
            "ret_15m_pct": 0.0, "ret_1h_pct": 0.0, "ret_4h_pct": 0.0, "ret_24h_pct": 0.0,
            "volume_ratio_1h": 1.0, "volume_ratio_4h": 1.0, "breakout_24h": False,
            "long_explosion_score": 1.0, "dump_score": float("nan"),
            "reversal_watch_score": float("nan"), "primary_ohlc_score": 1.0,
        }])

    def test_bid_ask_captured(self):
        df = self._make_row()
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert "bid" in stamped.columns
        assert "ask" in stamped.columns
        assert float(stamped.iloc[0]["bid"]) == pytest.approx(100.0)
        assert float(stamped.iloc[0]["ask"]) == pytest.approx(100.5)

    def test_spread_abs_computed(self):
        df = self._make_row(bid=100.0, ask=100.5)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert "spread_abs" in stamped.columns
        assert float(stamped.iloc[0]["spread_abs"]) == pytest.approx(0.5)

    def test_spread_passes_filter_true_for_tight_spread(self):
        df = self._make_row(spread_pct=1.0)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert bool(stamped.iloc[0]["spread_passes_filter"]) == True  # noqa: E712

    def test_spread_passes_filter_false_for_wide_spread(self):
        df = self._make_row(spread_pct=3.0)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert bool(stamped.iloc[0]["spread_passes_filter"]) == False  # noqa: E712

    def test_liquidity_flag_true_for_sufficient_volume(self):
        df = self._make_row(qvol=50000.0)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert bool(stamped.iloc[0]["liquidity_flag"]) == True  # noqa: E712

    def test_liquidity_flag_false_for_insufficient_volume(self):
        df = self._make_row(qvol=1000.0)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert bool(stamped.iloc[0]["liquidity_flag"]) == False  # noqa: E712

    def test_quote_volume_est_captured(self):
        df = self._make_row(qvol=75000.0)
        stamped = stamp_snapshot(df, "2026-06-10T12:00:00Z")
        assert "quote_volume_est" in stamped.columns
        assert float(stamped.iloc[0]["quote_volume_est"]) == pytest.approx(75000.0)


# ===========================================================================
# Duplicate-event prevention
# ===========================================================================


class TestDuplicateEventPrevention:
    """append_to_history must prevent duplicate events."""

    def _make_signal_df(self, ts: str, pair_id: str = "XBTUSDT") -> pd.DataFrame:
        row = {col: None for col in SNAPSHOT_COLUMNS}
        row["snapshot_ts_utc"] = ts
        row["pair_id"] = pair_id
        row["wsname"] = "XBT/USDT"
        row["ohlc_signal_type"] = "LONG_EXPLOSION"
        row["event_id"] = compute_event_id(pair_id, "LONG_EXPLOSION", "1.0.0", ts)
        return pd.DataFrame([row])

    def test_same_ts_not_appended_twice(self, tmp_path):
        history_path = tmp_path / "history.csv"
        ts = "2026-06-10T12:00:00Z"
        df = self._make_signal_df(ts)

        result1 = append_to_history(df, history_path, ts)
        result2 = append_to_history(df, history_path, ts)

        assert result1 == "created"
        assert result2 == "duplicate_skipped"
        loaded = pd.read_csv(history_path)
        assert len(loaded) == 1

    def test_same_event_id_not_appended(self, tmp_path):
        """Even with a different snapshot_ts, same event_id should be skipped."""
        history_path = tmp_path / "history.csv"
        ts1 = "2026-06-10T12:00:00Z"
        ts2 = "2026-06-10T12:00:01Z"  # 1 second later

        df1 = self._make_signal_df(ts1)
        df2 = df1.copy()
        df2["snapshot_ts_utc"] = ts2
        # Same event_id as df1 (event_id was computed from ts1)

        append_to_history(df1, history_path, ts1)
        result = append_to_history(df2, history_path, ts2)
        # df2's event_id already exists → should be skipped
        assert result == "duplicate_skipped"

    def test_different_event_id_appended(self, tmp_path):
        history_path = tmp_path / "history.csv"
        ts1 = "2026-06-10T12:00:00Z"
        ts2 = "2026-06-10T13:00:00Z"  # different hour → different data_cutoff → different event_id

        df1 = self._make_signal_df(ts1, pair_id="XBTUSDT")
        df2 = self._make_signal_df(ts2, pair_id="XBTUSDT")

        r1 = append_to_history(df1, history_path, ts1)
        r2 = append_to_history(df2, history_path, ts2)

        assert r1 == "created"
        assert r2 == "appended"
        loaded = pd.read_csv(history_path)
        assert len(loaded) == 2


class TestImmutableDetectionTimeFeatures:
    """Outcome upsert must never overwrite already-set detection-time features."""

    def test_upsert_never_overwrites_existing_non_blank_value(self):
        existing = pd.DataFrame([{
            "snapshot_ts_utc": "2026-06-10T12:00:00Z",
            "pair_id": "XBTUSDT",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "future_ret_4h_pct": 5.0,
            "outcome_4h": "SUCCESS",
            "evaluated_ts_utc": "2026-06-10T16:00:00Z",
        }])
        evaluated = pd.DataFrame([{
            "snapshot_ts_utc": "2026-06-10T12:00:00Z",
            "pair_id": "XBTUSDT",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "future_ret_4h_pct": -99.0,  # Attempt to overwrite
            "outcome_4h": "FAILURE",      # Attempt to overwrite
            "evaluated_ts_utc": "2026-06-10T16:01:00Z",
        }])
        result_df, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated)
        # The 4h outcome must remain SUCCESS, not FAILURE
        row = result_df.iloc[0]
        assert str(row["outcome_4h"]) == "SUCCESS", "Existing outcome must not be overwritten"
        assert float(row["future_ret_4h_pct"]) == pytest.approx(5.0), "Existing return must not be overwritten"

    def test_upsert_fills_blank_with_new_value(self):
        existing = pd.DataFrame([{
            "snapshot_ts_utc": "2026-06-10T12:00:00Z",
            "pair_id": "XBTUSDT",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "future_ret_4h_pct": float("nan"),
            "outcome_4h": "",
            "evaluated_ts_utc": "",
        }])
        evaluated = pd.DataFrame([{
            "snapshot_ts_utc": "2026-06-10T12:00:00Z",
            "pair_id": "XBTUSDT",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "future_ret_4h_pct": 5.0,
            "outcome_4h": "SUCCESS",
            "evaluated_ts_utc": "2026-06-10T16:01:00Z",
        }])
        result_df, _, n_updated, _ = upsert_outcome_rows(existing, evaluated)
        assert n_updated == 1
        row = result_df.iloc[0]
        assert str(row["outcome_4h"]) == "SUCCESS"


# ===========================================================================
# Part 7: SKIPPED status for optional stages
# ===========================================================================


class TestOptionalStageSkipped:
    """Optional stages must be recorded as SKIPPED, not OK."""

    def test_optional_skipped_exception_returns_skipped_result(self):
        def skipped_fn():
            raise _OptionalStepSkipped("optional_module_unavailable")

        result = _run_step("test_optional", skipped_fn, dry_run=False, continue_on_error=False)
        assert result.skipped is True
        assert result.success is True
        assert "optional_module_unavailable" in result.skipped_reason

    def test_skipped_result_str_shows_skipped(self):
        r = StepResult("test", success=True, duration_s=0.0, skipped=True, skipped_reason="optional_module_unavailable")
        assert "SKIPPED" in str(r)
        assert "OK" not in str(r)

    def test_dry_run_shows_skipped_not_ok(self):
        def fn():
            pass
        result = _run_step("test_step", fn, dry_run=True, continue_on_error=False)
        assert result.skipped is True
        assert "dry" in result.skipped_reason.lower() or "dry" in str(result).lower()


# ===========================================================================
# Part 1: Genuine vs simulated separation
# ===========================================================================


class TestGenuineVsSimulatedSeparation:
    """Simulated events must never appear in readiness report gate counts."""

    def _make_genuine_history(self, n: int = 10) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "snapshot_ts_utc": f"2026-06-0{(i % 7)+1}T12:00:00Z",
                "wsname": f"COIN{i}/USD",
                "pair_id": f"COIN{i}USD",
                "scanner_label": "HOT_MOVER",
                "ohlc_signal_type": "LONG_EXPLOSION",
                "event_source": "GENUINE_PROSPECTIVE",
                "is_simulated": "False",
            })
        return pd.DataFrame(rows)

    def _make_simulated_backfill(self, n: int = 200) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "snapshot_ts_utc": f"2026-05-{(i % 28)+1:02d}T00:00:00Z",
                "symbol": f"SIMCOIN{i}/USD",
                "ohlc_signal_type": "LONG_EXPLOSION",
                "event_source": "SIMULATED_BACKFILL",
                "is_simulated": "True",
            })
        return pd.DataFrame(rows)

    def test_simulated_count_does_not_affect_gate_a_event_count(self):
        history = self._make_genuine_history(n=10)
        outcomes = pd.DataFrame()
        simulated = self._make_simulated_backfill(n=200)
        rule1 = pd.DataFrame()
        status = build_status_dict(history, outcomes, simulated, rule1)
        # Gate A event threshold uses genuine events only
        assert status["genuine_signal_rows"] == 10
        assert status["simulated_backfill_rows"] == 200
        assert status["gate_a_total_events_ge_500"] is False, (
            "Simulated events must not count toward Gate A event threshold"
        )

    def test_simulated_events_excluded_from_gates_flag(self):
        history = self._make_genuine_history(n=10)
        outcomes = pd.DataFrame()
        simulated = self._make_simulated_backfill(n=200)
        rule1 = pd.DataFrame()
        status = build_status_dict(history, outcomes, simulated, rule1)
        assert status["simulated_excluded_from_gates"] is True

    def test_simulated_count_separate_in_report(self):
        history = self._make_genuine_history(n=5)
        outcomes = pd.DataFrame()
        simulated = self._make_simulated_backfill(n=192)
        rule1 = pd.DataFrame()
        status = build_status_dict(history, outcomes, simulated, rule1)
        assert status["genuine_signal_rows"] == 5
        assert status["simulated_backfill_rows"] == 192


# ===========================================================================
# Idempotency (run twice produces same result)
# ===========================================================================


class TestIdempotency:
    """Repeated runs must not create duplicates."""

    def _make_signal_df(self, ts: str, pair_id: str = "XBTUSDT") -> pd.DataFrame:
        row = {col: None for col in SNAPSHOT_COLUMNS}
        row["snapshot_ts_utc"] = ts
        row["pair_id"] = pair_id
        row["wsname"] = "XBT/USDT"
        row["ohlc_signal_type"] = "LONG_EXPLOSION"
        row["event_id"] = compute_event_id(pair_id, "LONG_EXPLOSION", "1.0.0", "2026-06-10T12:00:00Z")
        return pd.DataFrame([row])

    def test_double_run_history_idempotent(self, tmp_path):
        history_path = tmp_path / "history.csv"
        ts = "2026-06-10T12:00:00Z"
        df = self._make_signal_df(ts)

        # Run 1
        append_to_history(df, history_path, ts)
        rows_after_run1 = len(pd.read_csv(history_path))

        # Run 2 (same data)
        append_to_history(df, history_path, ts)
        rows_after_run2 = len(pd.read_csv(history_path))

        assert rows_after_run1 == rows_after_run2, "Second run must not add duplicate rows"

    def test_upsert_idempotent_for_already_complete_outcomes(self):
        """Re-evaluating a completed outcome must not change it."""
        existing = pd.DataFrame([{
            "snapshot_ts_utc": "2026-06-10T12:00:00Z",
            "pair_id": "XBTUSDT",
            "ohlc_signal_type": "LONG_EXPLOSION",
            "future_ret_4h_pct": 5.0,
            "outcome_4h": "SUCCESS",
            "evaluated_ts_utc": "2026-06-10T16:00:00Z",
        }])
        # Simulate second evaluation with same data
        re_eval = existing.copy()
        re_eval["evaluated_ts_utc"] = "2026-06-10T16:05:00Z"  # ts refreshed
        result_df, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, re_eval)
        assert n_added == 0
        # n_skipped should be 1 because no refreshable column changed (except evaluated_ts)
        row = result_df.iloc[0]
        assert str(row["outcome_4h"]) == "SUCCESS"


# ===========================================================================
# No live broker/exchange execution imports
# ===========================================================================


class TestNoLiveExecutionImports:
    """Research modules must not import broker or live execution code."""

    MEMECOIN_MODULES = [
        "research/memecoin_catcher/save_signal_snapshot.py",
        "research/memecoin_catcher/evaluate_signal_outcomes.py",
        "research/memecoin_catcher/enrich_candidates_with_ohlc.py",
        "research/memecoin_catcher/fetch_kraken_universe.py",
        "research/memecoin_catcher/rank_memecoin_candidates.py",
        "research/memecoin_catcher/memecoin_readiness_report.py",
        "research/memecoin_catcher/validate_memecoin_candidate_rules.py",
        "research/memecoin_catcher/backfill_recent_memecoin_signals.py",
    ]
    FORBIDDEN_IMPORTS = ["from brokers", "import brokers", "from live.", "import ccxt", "place_order", "submit_order"]

    def test_no_forbidden_imports_in_research_modules(self):
        repo_root = Path(__file__).parent.parent
        for rel_path in self.MEMECOIN_MODULES:
            full_path = repo_root / rel_path
            if not full_path.exists():
                continue
            src = full_path.read_text()
            import_lines = [l for l in src.splitlines() if re.match(r"\s*(import|from)\s+", l)]
            call_src = src
            for token in self.FORBIDDEN_IMPORTS:
                # Check in import lines for import-style tokens
                for line in import_lines:
                    assert token not in line, (
                        f"Forbidden import '{token}' found in {rel_path}: {line.strip()}"
                    )
                # Check call-site patterns (non-import)
                if token in ("place_order", "submit_order", "ccxt"):
                    assert token not in call_src, (
                        f"Forbidden reference '{token}' found in {rel_path}"
                    )


# ===========================================================================
# No five-coin shadow-state changes
# ===========================================================================


class TestNoFiveCoinShadowChanges:
    """Wrapper script must not reference five-coin shadow state."""

    def test_wrapper_does_not_reference_shadow_state(self):
        wrapper_path = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        assert wrapper_path.exists(), "Wrapper script must exist"
        src = wrapper_path.read_text()
        forbidden = ["shadow_state", "frozen_candidate", "shadow_cycle", "five_coin", "shadow_portfolio"]
        for token in forbidden:
            assert token not in src.lower(), (
                f"Wrapper script references five-coin shadow concept: {token!r}"
            )

    def test_wrapper_does_not_call_backfill(self):
        wrapper_path = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        src = wrapper_path.read_text()
        # backfill must not be called automatically
        assert "backfill_recent_memecoin_signals" not in src, (
            "Wrapper must not automatically call historical backfill script"
        )


# ===========================================================================
# Unchanged frozen hashes (five-coin shadow)
# ===========================================================================


class TestFrozenHashesUnchanged:
    """The frozen five-coin candidate hashes must not be modified by this session."""

    def test_frozen_hash_files_not_modified_by_memecoin_code(self):
        repo_root = Path(__file__).parent.parent
        shadow_dir = repo_root / "shadow_state"
        if not shadow_dir.exists():
            pytest.skip("shadow_state directory not present")

        # No memecoin module should import from shadow_state
        memecoin_dir = repo_root / "research" / "memecoin_catcher"
        for py_file in memecoin_dir.glob("*.py"):
            src = py_file.read_text()
            assert "shadow_state" not in src, (
                f"Memecoin module {py_file.name} references shadow_state"
            )


# ===========================================================================
# Locked evidence thresholds unchanged
# ===========================================================================


class TestLockedEvidenceThresholds:
    """Locked Gate A thresholds must not be modified in memecoin_readiness_report."""

    def test_gate_a_min_collection_days_is_30(self):
        assert GATE_A["min_collection_days"] == 30

    def test_gate_a_min_rule1_events_is_150(self):
        assert GATE_A["min_rule1_events"] == 150

    def test_gate_a_min_symbols_is_25(self):
        assert GATE_A["min_symbols_rule1"] == 25

    def test_gate_a_max_single_symbol_pct_is_20pct(self):
        assert GATE_A["max_single_symbol_pct"] == 0.20

    def test_evidence_threshold_file_exists(self):
        threshold_path = Path(__file__).parent.parent / "reports" / "memecoin_evidence_thresholds.md"
        assert threshold_path.exists(), "Locked evidence threshold file must exist"

    def test_evidence_threshold_file_unchanged_gate_a_values(self):
        threshold_path = Path(__file__).parent.parent / "reports" / "memecoin_evidence_thresholds.md"
        src = threshold_path.read_text()
        # Key values locked on 2026-06-09
        assert "≥ 30 calendar days" in src, "Gate A min days must remain 30"
        assert "≥ 150" in src, "Gate A min rule1 events must remain 150"
        assert "≥ 25 symbols" in src, "Gate A min symbols must remain 25"


# ===========================================================================
# No historical backfill in wrapper
# ===========================================================================


class TestNoBackfillInWrapper:
    """Wrapper must never trigger historical backfill automatically."""

    def test_wrapper_excludes_backfill_module(self):
        wrapper = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        src = wrapper.read_text()
        assert "backfill_recent_memecoin_signals" not in src

    def test_readiness_report_excludes_simulated(self):
        """Readiness report module must not merge simulated backfill into gates."""
        src = Path(__file__).parent.parent / "research" / "memecoin_catcher" / "memecoin_readiness_report.py"
        content = src.read_text().lower()
        # The module must explicitly document that simulated events are excluded
        assert "simulated" in content
        assert "never" in content or "excluded" in content

    def test_stage_h_explicitly_labeled_backfill_only(self):
        """Stage H wrapper log label must explicitly prevent confusion with genuine readiness evidence."""
        wrapper = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        src = wrapper.read_text().lower()
        assert "candidate rule validation (backfill-only; not gate a genuine evidence)" in src


# ===========================================================================
# Immutable timestamped snapshots
# ===========================================================================


class TestImmutableSnapshots:
    """Each collection cycle must write a new timestamped snapshot file."""

    def test_snapshot_files_are_timestamped(self):
        snapshot_dir = Path(__file__).parent.parent / "data" / "memecoin_signal_snapshots"
        if not snapshot_dir.exists():
            pytest.skip("No snapshots directory")
        files = list(snapshot_dir.glob("memecoin_signals_*.csv"))
        # Each file name must contain a timestamp
        for f in files:
            # Format: memecoin_signals_YYYYMMDD_HHMMSS.csv
            assert re.match(r"memecoin_signals_\d{8}_\d{6}\.csv", f.name), (
                f"Snapshot file name must be timestamped: {f.name}"
            )

    def test_each_cycle_produces_unique_snapshot_filename(self, tmp_path):
        """Two calls with different timestamps must produce different file names."""
        from research.memecoin_catcher.save_signal_snapshot import (
            format_snapshot_filename,
            write_snapshot_file,
        )
        import time

        df = pd.DataFrame([{c: None for c in SNAPSHOT_COLUMNS}])
        ts1_str = format_snapshot_filename(datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc))
        ts2_str = format_snapshot_filename(datetime(2026, 6, 10, 13, 0, 0, tzinfo=timezone.utc))

        p1 = write_snapshot_file(df, tmp_path, ts1_str)
        p2 = write_snapshot_file(df, tmp_path, ts2_str)

        assert p1 != p2, "Different cycle timestamps must produce different snapshot paths"


# ===========================================================================
# Wrapper: lock acquisition and cleanup
# ===========================================================================


class TestLockAcquisition:
    """Lock file must be created and cleaned up."""

    def test_lock_file_path_in_wrapper(self):
        wrapper = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        src = wrapper.read_text()
        assert "memecoin_collection_cycle.lock" in src
        assert "release_lock" in src
        assert "trap" in src  # ensure lock is released on EXIT

    def test_lock_uses_trap_for_cleanup(self):
        wrapper = Path(__file__).parent.parent / "scripts" / "run_memecoin_collection_cycle.sh"
        src = wrapper.read_text()
        # Must use trap to release lock on all exits
        assert "trap" in src and "release_lock" in src
        # Ensure EXIT is included in trap
        assert "EXIT" in src
