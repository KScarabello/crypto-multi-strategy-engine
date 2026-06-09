"""Unit tests for research/memecoin_catcher/monitor_candidate_rule_forward.py.

Covers:
- Rule 1 signal selection
- Duplicate prevention in signal log
- Outcome update behavior
- tp10_else_4h return calculation
- Incomplete / missing outcome handling
- Summary metrics
- Missing column handling
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from research.memecoin_catcher.monitor_candidate_rule_forward import (
    CANDIDATE_RULE_NAME,
    OUTCOME_COLUMNS,
    SIGNAL_COLUMNS,
    SIGNAL_KEY_COLS,
    append_new_signals,
    build_summary,
    compute_tp10_else_4h,
    evaluate_outcomes,
    is_rule1_signal,
    load_outcome_log,
    load_signal_log,
    save_outcomes,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sig(
    symbol: str = "TEST/USD",
    pair_id: str = "TESTUSD",
    signal_ts: str = "2025-06-01T17:00:00Z",
    rule: str = CANDIDATE_RULE_NAME,
    **kwargs,
) -> dict[str, Any]:
    base = {
        "run_timestamp":       "2025-06-01T17:05:00Z",
        "signal_timestamp":    signal_ts,
        "symbol":              symbol,
        "pair_id":             pair_id,
        "signal_type":         "LONG_EXPLOSION",
        "primary_ohlc_score":  12.0,
        "r1h_pct":             1.5,
        "r4h_pct":             4.0,
        "r24h_pct":            10.0,
        "volume_ratio_1h":     8.0,
        "spread_pct":          float("nan"),
        "quote_volume_est":    float("nan"),
        "is_volume_climax":    True,
        "is_clean_continuation": False,
        "is_wide_spread":      False,
        "is_terminal_spike":   False,
        "is_overextended_24h": False,
        "is_rolling_over":     False,
        "candidate_rule_name": rule,
        "entry_price":         1.25,
    }
    base.update(kwargs)
    return base


def _outcome(
    symbol: str = "TEST/USD",
    pair_id: str = "TESTUSD",
    signal_ts: str = "2025-06-01T17:00:00Z",
    rule: str = CANDIDATE_RULE_NAME,
    ret_4h: float = 3.0,
    ret_24h: float = 5.0,
    mfe_4h: float = 8.0,
    complete_4h: bool = True,
    complete_24h: bool = True,
) -> dict[str, Any]:
    tp10 = compute_tp10_else_4h(ret_4h, mfe_4h)
    return {
        "symbol":              symbol,
        "pair_id":             pair_id,
        "signal_timestamp":    signal_ts,
        "candidate_rule_name": rule,
        "entry_price":         1.25,
        "future_ret_15m_pct":  0.5,
        "future_ret_1h_pct":   1.0,
        "future_ret_4h_pct":   ret_4h,
        "future_ret_24h_pct":  ret_24h,
        "max_favorable_4h_pct": mfe_4h,
        "max_adverse_4h_pct":  -2.0,
        "max_favorable_24h_pct": 12.0,
        "max_adverse_24h_pct": -5.0,
        "fixed_4h_return":     ret_4h,
        "tp10_else_4h_return": tp10,
        "fixed_24h_return":    ret_24h,
        "outcome_complete_4h": complete_4h,
        "outcome_complete_24h": complete_24h,
        "last_evaluated_utc":  "2025-06-01T21:00:00Z",
    }


# ---------------------------------------------------------------------------
# Rule 1 selection
# ---------------------------------------------------------------------------


class TestIsRule1Signal:
    def test_valid_rule1_signal(self):
        features = {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "is_volume_climax": True,
            "is_clean_continuation": False,
        }
        assert is_rule1_signal(features) is True

    def test_wrong_signal_type(self):
        features = {
            "ohlc_signal_type": "REVERSAL_WATCH",
            "is_volume_climax": True,
            "is_clean_continuation": False,
        }
        assert is_rule1_signal(features) is False

    def test_volume_climax_false(self):
        features = {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "is_volume_climax": False,
            "is_clean_continuation": False,
        }
        assert is_rule1_signal(features) is False

    def test_clean_continuation_true(self):
        features = {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "is_volume_climax": True,
            "is_clean_continuation": True,
        }
        assert is_rule1_signal(features) is False

    def test_missing_signal_type_returns_false(self):
        features = {"is_volume_climax": True, "is_clean_continuation": False}
        assert is_rule1_signal(features) is False

    def test_missing_volume_climax_defaults_false(self):
        features = {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "is_clean_continuation": False,
        }
        assert is_rule1_signal(features) is False

    def test_missing_cc_defaults_false_for_cc_condition(self):
        # is_clean_continuation absent defaults to False → CC=False condition met
        features = {
            "ohlc_signal_type": "LONG_EXPLOSION",
            "is_volume_climax": True,
        }
        assert is_rule1_signal(features) is True


# ---------------------------------------------------------------------------
# tp10_else_4h
# ---------------------------------------------------------------------------


class TestComputeTp10Else4h:
    def test_tp_triggered(self):
        assert compute_tp10_else_4h(4.0, 12.0) == pytest.approx(10.0)

    def test_tp_exactly_at_threshold(self):
        assert compute_tp10_else_4h(4.0, 10.0) == pytest.approx(10.0)

    def test_tp_not_triggered(self):
        assert compute_tp10_else_4h(4.0, 7.0) == pytest.approx(4.0)

    def test_negative_return_no_tp(self):
        assert compute_tp10_else_4h(-3.0, 4.0) == pytest.approx(-3.0)

    def test_nan_mfe_returns_raw(self):
        result = compute_tp10_else_4h(5.0, float("nan"))
        assert result == pytest.approx(5.0)

    def test_nan_ret_returns_nan(self):
        result = compute_tp10_else_4h(float("nan"), 15.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Duplicate prevention
# ---------------------------------------------------------------------------


class TestAppendNewSignals:
    def test_appends_new_signal(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        sigs = [_sig()]
        updated, n = append_new_signals(sigs, log, path)
        assert n == 1
        assert len(updated) == 1
        assert path.exists()

    def test_deduplicates_same_key(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        sigs = [_sig()]
        updated, _ = append_new_signals(sigs, log, path)
        # Try to append same signal again
        updated2, n = append_new_signals(sigs, updated, path)
        assert n == 0
        assert len(updated2) == 1

    def test_different_symbol_both_appended(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        sigs = [_sig(symbol="A/USD"), _sig(symbol="B/USD")]
        updated, n = append_new_signals(sigs, log, path)
        assert n == 2
        assert len(updated) == 2

    def test_same_symbol_different_timestamp(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        sigs = [
            _sig(signal_ts="2025-06-01T17:00:00Z"),
            _sig(signal_ts="2025-06-02T17:00:00Z"),
        ]
        updated, n = append_new_signals(sigs, log, path)
        assert n == 2

    def test_empty_signals_list(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        updated, n = append_new_signals([], log, path)
        assert n == 0
        assert not path.exists()

    def test_file_written_after_append(self, tmp_path):
        path = tmp_path / "signals.csv"
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        sigs = [_sig()]
        append_new_signals(sigs, log, path)
        reloaded = pd.read_csv(path)
        assert len(reloaded) == 1


# ---------------------------------------------------------------------------
# Outcome update behavior
# ---------------------------------------------------------------------------


class TestEvaluateOutcomes:
    def _make_ohlc(self, n: int = 100, start_price: float = 1.25) -> pd.DataFrame:
        """Return a minimal OHLC DataFrame with n rows."""
        times = [1748736000 + i * 900 for i in range(n)]  # 15-min steps
        prices = [start_price * (1 + 0.001 * i) for i in range(n)]
        return pd.DataFrame({
            "time":  times,
            "open":  prices,
            "high":  [p * 1.01 for p in prices],
            "low":   [p * 0.99 for p in prices],
            "close": prices,
            "volume": [100.0] * n,
        })

    def _make_fetcher(self, ohlc_df: pd.DataFrame):
        """Return a mock fetcher that always returns ohlc_df."""
        def _fetcher(pair_id, interval=15, since=None):
            return ohlc_df
        return _fetcher

    def test_adds_outcome_for_new_signal(self):
        ohlc_df = self._make_ohlc(n=120)
        fetcher = self._make_fetcher(ohlc_df)

        # Set signal_timestamp to time of first row
        sig_ts = pd.Timestamp(int(ohlc_df.iloc[0]["time"]), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        sig_df = pd.DataFrame([_sig(signal_ts=sig_ts)])
        outcome_df = pd.DataFrame(columns=OUTCOME_COLUMNS)

        updated, n = evaluate_outcomes(sig_df, outcome_df, fetcher=fetcher)
        assert n == 1
        assert len(updated) >= 1

    def test_skips_already_complete_24h(self):
        ohlc_df = self._make_ohlc(n=120)
        fetcher = self._make_fetcher(ohlc_df)

        sig_ts = "2025-06-01T17:00:00Z"
        sig_df = pd.DataFrame([_sig(signal_ts=sig_ts)])
        existing = pd.DataFrame([_outcome(
            signal_ts=sig_ts,
            complete_24h=True,
        )])
        _, n = evaluate_outcomes(sig_df, existing, fetcher=fetcher)
        assert n == 0  # already complete; not re-evaluated

    def test_updates_incomplete_4h_outcome(self):
        ohlc_df = self._make_ohlc(n=120)
        fetcher = self._make_fetcher(ohlc_df)

        sig_ts = pd.Timestamp(int(ohlc_df.iloc[0]["time"]), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        sig_df = pd.DataFrame([_sig(signal_ts=sig_ts)])
        # Existing outcome has 4h complete but 24h not
        existing = pd.DataFrame([_outcome(
            signal_ts=sig_ts,
            complete_4h=True,
            complete_24h=False,
        )])
        updated, n = evaluate_outcomes(sig_df, existing, fetcher=fetcher)
        assert n >= 1

    def test_graceful_on_empty_signal_log(self):
        outcome_df = pd.DataFrame(columns=OUTCOME_COLUMNS)
        sig_df = pd.DataFrame(columns=SIGNAL_COLUMNS)
        updated, n = evaluate_outcomes(sig_df, outcome_df)
        assert n == 0
        assert isinstance(updated, pd.DataFrame)

    def test_missing_entry_price_skips(self):
        ohlc_df = self._make_ohlc(n=120)
        fetcher = self._make_fetcher(ohlc_df)

        sig_ts = "2025-06-01T17:00:00Z"
        s = _sig(signal_ts=sig_ts)
        s["entry_price"] = float("nan")
        sig_df = pd.DataFrame([s])
        outcome_df = pd.DataFrame(columns=OUTCOME_COLUMNS)
        _, n = evaluate_outcomes(sig_df, outcome_df, fetcher=fetcher)
        assert n == 0

    def test_tp10_return_capped_in_outcome(self):
        """When MFE >= 10, tp10_else_4h_return must equal 10."""
        ohlc_df = self._make_ohlc(n=120, start_price=1.0)
        # Make highs much higher so MFE is > 10%
        ohlc_df["high"] = ohlc_df["high"] * 3.0
        fetcher = self._make_fetcher(ohlc_df)

        sig_ts = pd.Timestamp(int(ohlc_df.iloc[0]["time"]), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        sig_df = pd.DataFrame([_sig(signal_ts=sig_ts, entry_price=1.0)])
        outcome_df = pd.DataFrame(columns=OUTCOME_COLUMNS)
        updated, _ = evaluate_outcomes(sig_df, outcome_df, fetcher=fetcher)

        row = updated[updated["symbol"] == "TEST/USD"]
        if not row.empty:
            tp10 = pd.to_numeric(row["tp10_else_4h_return"].iloc[0], errors="coerce")
            mfe = pd.to_numeric(row["max_favorable_4h_pct"].iloc[0], errors="coerce")
            if not math.isnan(mfe) and mfe >= 10.0:
                assert tp10 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def _make_outcomes(self, n: int = 10, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets_4h = rng.normal(1, 5, n).tolist()
        mfe = [abs(float(rng.normal(4, 3))) for _ in range(n)]
        rows = [
            _outcome(
                symbol=f"TOK{i}/USD",
                signal_ts=f"2025-06-0{1 + i % 7}T{i % 24:02d}:00:00Z",
                ret_4h=r,
                mfe_4h=m,
                complete_4h=True,
                complete_24h=True,
            )
            for i, (r, m) in enumerate(zip(rets_4h, mfe))
        ]
        return pd.DataFrame(rows)

    def test_output_has_three_policies(self):
        df = self._make_outcomes(n=10)
        summary = build_summary(df)
        assert set(summary["exit_policy"].tolist()) == {"fixed_4h", "tp10_else_4h", "fixed_24h"}

    def test_event_count_matches(self):
        df = self._make_outcomes(n=8)
        summary = build_summary(df)
        for _, r in summary.iterrows():
            assert r["event_count"] == 8

    def test_avg_excl_best_le_avg(self):
        df = self._make_outcomes(n=10)
        summary = build_summary(df)
        for _, r in summary.iterrows():
            if r["completed_count"] > 1:
                assert r["avg_excl_best"] <= r["avg"] + 1e-9

    def test_win_rate_between_0_and_1(self):
        df = self._make_outcomes(n=10)
        summary = build_summary(df)
        for _, r in summary.iterrows():
            assert 0.0 <= r["win_rate"] <= 1.0

    def test_empty_outcome_df_returns_empty(self):
        df = pd.DataFrame(columns=OUTCOME_COLUMNS)
        summary = build_summary(df)
        assert summary.empty

    def test_fail_rates_correct(self):
        rets = [5.0, -4.0, -4.0, 2.0]  # 2 failures <= -3
        df = pd.DataFrame([
            _outcome(symbol=f"T{i}/USD", signal_ts=f"2025-06-01T{10+i:02d}:00:00Z",
                     ret_4h=r, mfe_4h=1.0, complete_4h=True, complete_24h=True)
            for i, r in enumerate(rets)
        ])
        summary = build_summary(df)
        f4h = summary[summary["exit_policy"] == "fixed_4h"].iloc[0]
        assert f4h["fail_rate_3"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_load_signal_log_missing_file(self, tmp_path):
        path = tmp_path / "does_not_exist.csv"
        df = load_signal_log(path)
        assert df.empty
        assert list(df.columns) == SIGNAL_COLUMNS

    def test_load_outcome_log_missing_file(self, tmp_path):
        path = tmp_path / "does_not_exist.csv"
        df = load_outcome_log(path)
        assert df.empty
        assert list(df.columns) == OUTCOME_COLUMNS

    def test_save_and_reload_outcomes(self, tmp_path):
        path = tmp_path / "outcomes.csv"
        df = pd.DataFrame([_outcome()])
        save_outcomes(df, path)
        assert path.exists()
        reloaded = load_outcome_log(path)
        assert len(reloaded) == 1

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "subdir" / "outcomes.csv"
        df = pd.DataFrame([_outcome()])
        save_outcomes(df, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Missing column handling
# ---------------------------------------------------------------------------


class TestMissingColumnHandling:
    def test_signal_log_missing_columns_filled(self, tmp_path):
        path = tmp_path / "signals.csv"
        # Signal dict missing some optional columns
        sparse = {
            "run_timestamp":       "2025-06-01T17:05:00Z",
            "signal_timestamp":    "2025-06-01T17:00:00Z",
            "symbol":              "TEST/USD",
            "candidate_rule_name": CANDIDATE_RULE_NAME,
        }
        log = pd.DataFrame(columns=SIGNAL_COLUMNS)
        updated, n = append_new_signals([sparse], log, path)
        assert n == 1
        assert "spread_pct" in updated.columns

    def test_build_summary_missing_tp10_column(self):
        df = pd.DataFrame([_outcome()])
        df = df.drop(columns=["tp10_else_4h_return"], errors="ignore")
        summary = build_summary(df)
        # tp10 policy absent — should not crash
        assert "fixed_4h" in summary["exit_policy"].values

    def test_outcome_log_coerces_numeric(self, tmp_path):
        path = tmp_path / "outcomes.csv"
        df = pd.DataFrame([_outcome()])
        save_outcomes(df, path)
        reloaded = load_outcome_log(path)
        assert pd.api.types.is_float_dtype(reloaded["future_ret_4h_pct"])
