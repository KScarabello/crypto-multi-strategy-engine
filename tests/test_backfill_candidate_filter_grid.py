"""Unit tests for research/memecoin_catcher/test_backfill_candidate_filter_grid.py.

Covers:
- Filter construction (volume-ratio, hour, overextension, combined, CC diagnostic)
- Hour filtering
- Volume-ratio filtering
- Overextension (ret_24h) filtering
- tp10_else_4h exit logic
- apply_exit_policy (vectorised)
- compute_stats (all metrics including top-1% exclusion)
- passes_validation criteria
- run_filter_grid integration
- Missing column handling
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from research.memecoin_catcher.test_backfill_candidate_filter_grid import (
    EXIT_POLICIES,
    HIGH_FAILURE_HOURS,
    MIN_N,
    SAFE_HOURS,
    TOP_PCT,
    apply_exit_policy,
    compute_stats,
    exit_fixed_4h,
    exit_tp10_else_4h,
    load_events,
    passes_validation,
    run_filter_grid,
    _make_filters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 60, seed: int = 42, **overrides) -> pd.DataFrame:
    """
    Build a minimal DataFrame that satisfies all filter/stats code paths.
    Deterministic via numpy seed.
    """
    rng = np.random.default_rng(seed)
    base = {
        "future_ret_4h_pct": rng.normal(0, 5, n).tolist(),
        "future_ret_24h_pct": rng.normal(0, 8, n).tolist(),
        "max_favorable_4h_pct": np.abs(rng.normal(3, 4, n)).tolist(),
        "max_adverse_4h_pct": (-np.abs(rng.normal(3, 4, n))).tolist(),
        "volume_ratio_1h": rng.uniform(0.5, 20, n).tolist(),
        "volume_ratio_4h": rng.uniform(0.5, 15, n).tolist(),
        "ret_1h_pct": rng.uniform(-3, 5, n).tolist(),
        "ret_4h_pct": rng.uniform(-5, 10, n).tolist(),
        "ret_24h_pct": rng.uniform(2, 50, n).tolist(),
        "is_volume_climax": ([True] * (n // 5) + [False] * (n - n // 5)),
        "is_clean_continuation": ([True] * (n // 3) + [False] * (n - n // 3)),
        "is_overextended_24h": [False] * n,
        "is_rolling_over": [False] * n,
        "is_wide_spread": [False] * n,
        "danger_terminal_spike": [False] * n,
        "hour_of_day_utc": [i % 24 for i in range(n)],
        "symbol": ["BTC/USD"] * n,
        "snapshot_ts_utc": [f"2025-01-01T{i % 24:02d}:00:00Z" for i in range(n)],
        "ohlc_signal_type": ["LONG_EXPLOSION"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------


class TestExitFixed4h:
    def test_returns_future_ret_4h(self):
        row = pd.Series({"future_ret_4h_pct": 3.7, "max_favorable_4h_pct": 15.0})
        assert exit_fixed_4h(row) == pytest.approx(3.7)

    def test_negative_return(self):
        row = pd.Series({"future_ret_4h_pct": -5.2, "max_favorable_4h_pct": 1.0})
        assert exit_fixed_4h(row) == pytest.approx(-5.2)


class TestExitTp10Else4h:
    def test_tp_triggered(self):
        row = pd.Series({"future_ret_4h_pct": 4.0, "max_favorable_4h_pct": 12.0})
        assert exit_tp10_else_4h(row) == pytest.approx(10.0)

    def test_tp_not_triggered_returns_4h(self):
        row = pd.Series({"future_ret_4h_pct": 4.0, "max_favorable_4h_pct": 7.0})
        assert exit_tp10_else_4h(row) == pytest.approx(4.0)

    def test_tp_exactly_at_threshold(self):
        row = pd.Series({"future_ret_4h_pct": 4.0, "max_favorable_4h_pct": 10.0})
        assert exit_tp10_else_4h(row) == pytest.approx(10.0)

    def test_tp_not_triggered_negative(self):
        row = pd.Series({"future_ret_4h_pct": -3.0, "max_favorable_4h_pct": 2.0})
        assert exit_tp10_else_4h(row) == pytest.approx(-3.0)

    def test_missing_mfe_falls_back_to_4h(self):
        row = pd.Series({"future_ret_4h_pct": 5.0})
        # No max_favorable_4h_pct key → should fall back to 4h return
        result = exit_tp10_else_4h(row)
        assert result == pytest.approx(5.0)


class TestApplyExitPolicy:
    def test_fixed_4h_vectorised(self):
        df = _make_df(n=10)
        result = apply_exit_policy(df, exit_fixed_4h)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            pd.to_numeric(df["future_ret_4h_pct"], errors="coerce").reset_index(drop=True),
        )

    def test_tp10_caps_at_10(self):
        df = _make_df(n=20)
        df["max_favorable_4h_pct"] = 15.0  # all exceed 10 → all capped
        result = apply_exit_policy(df, exit_tp10_else_4h)
        assert (result == 10.0).all()

    def test_tp10_falls_back_below_threshold(self):
        df = _make_df(n=20)
        df["max_favorable_4h_pct"] = 5.0  # none exceed 10 → raw returns
        result = apply_exit_policy(df, exit_tp10_else_4h)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            pd.to_numeric(df["future_ret_4h_pct"], errors="coerce").reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_basic_counts(self):
        s = pd.Series([5.0, -1.0, 3.0, -4.0, 2.0])
        stats = compute_stats(s)
        assert stats["n"] == 5

    def test_win_rate(self):
        s = pd.Series([1.0, -1.0, 2.0, -2.0])
        assert compute_stats(s)["win_rate"] == pytest.approx(0.5)

    def test_big_win_rate(self):
        s = pd.Series([6.0, 4.0, -2.0])
        assert compute_stats(s)["big_win_rate"] == pytest.approx(1 / 3, abs=1e-3)

    def test_fail_rate_3(self):
        s = pd.Series([1.0, -3.0, -5.0, 2.0])
        assert compute_stats(s)["fail_rate_3"] == pytest.approx(0.5)

    def test_fail_rate_5(self):
        s = pd.Series([1.0, -3.0, -5.0, 2.0])
        assert compute_stats(s)["fail_rate_5"] == pytest.approx(0.25)

    def test_avg_excl_best(self):
        s = pd.Series([100.0, 1.0, 2.0])
        stats = compute_stats(s)
        assert stats["avg_excl_best"] == pytest.approx(1.5)

    def test_avg_excl_best_is_le_avg(self):
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0, 5, 50))
        stats = compute_stats(s)
        assert stats["avg_excl_best"] <= stats["avg"] + 1e-9

    def test_top_1pct_exclusion_drops_highest(self):
        # 100 values; top 1% = the single value above the 99th percentile
        s = pd.Series([float(i) for i in range(100)])
        stats = compute_stats(s)
        # avg_excl_top1pct should be less than avg
        assert stats["avg_excl_top1pct"] < stats["avg"]

    def test_top_1pct_exclusion_positive_result(self):
        # Only one extreme outlier; avg_excl_top1pct should be well below avg
        s = pd.Series([1.0] * 99 + [10000.0])
        stats = compute_stats(s)
        assert stats["avg_excl_top1pct"] == pytest.approx(1.0, abs=0.01)

    def test_empty_series_returns_nan(self):
        stats = compute_stats(pd.Series(dtype=float))
        assert stats["n"] == 0
        assert math.isnan(stats["avg"])

    def test_nan_values_dropped(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        assert compute_stats(s)["n"] == 2

    def test_mfe_mae_included(self):
        s = pd.Series([1.0, 2.0, -1.0])
        mfe = pd.Series([3.0, 4.0, 1.0])
        mae = pd.Series([-1.0, -2.0, -3.0])
        stats = compute_stats(s, mfe=mfe, mae=mae)
        assert stats["avg_mfe"] == pytest.approx((3 + 4 + 1) / 3, abs=1e-3)
        assert stats["avg_mae"] == pytest.approx((-1 - 2 - 3) / 3)


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------


class TestFilterConstruction:
    def test_baseline_all_selects_everything(self):
        df = _make_df(n=30)
        filters = dict(_make_filters(df))
        mask = filters["baseline_all"](df)
        assert mask.all()

    def test_volume_ratio_filter_ge_5(self):
        df = _make_df(n=20)
        df["volume_ratio_1h"] = list(range(1, 21))  # 1..20
        filters = dict(_make_filters(df))
        mask = filters["vr1h_ge_5"](df)
        expected = df["volume_ratio_1h"] >= 5
        pd.testing.assert_series_equal(mask, expected)

    def test_volume_ratio_bounded_filter(self):
        df = _make_df(n=20)
        df["volume_ratio_1h"] = list(range(1, 21))
        filters = dict(_make_filters(df))
        mask = filters["vr1h_3_to_15"](df)
        expected = (df["volume_ratio_1h"] >= 3) & (df["volume_ratio_1h"] <= 15)
        pd.testing.assert_series_equal(mask, expected)

    def test_hour_safe_filter(self):
        df = _make_df(n=24)
        df["hour_of_day_utc"] = list(range(24))
        filters = dict(_make_filters(df))
        mask = filters["hour_safe_17_18_19"](df)
        expected = df["hour_of_day_utc"].isin(SAFE_HOURS)
        pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))

    def test_hour_safe_selects_exactly_3_hours(self):
        df = pd.DataFrame({"hour_of_day_utc": list(range(24)), "future_ret_4h_pct": [1.0] * 24})
        df["volume_ratio_1h"] = 5.0
        df["ret_24h_pct"] = 10.0
        df["is_volume_climax"] = False
        df["is_clean_continuation"] = True
        filters = dict(_make_filters(df))
        mask = filters["hour_safe_17_18_19"](df)
        assert mask.sum() == 3  # exactly hours 17, 18, 19

    def test_excl_high_failure_hours(self):
        df = _make_df(n=24)
        df["hour_of_day_utc"] = list(range(24))
        filters = dict(_make_filters(df))
        mask = filters["excl_high_failure_hours"](df)
        expected = ~df["hour_of_day_utc"].isin(HIGH_FAILURE_HOURS)
        pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))

    def test_ret24h_lt_25_filter(self):
        df = _make_df(n=20)
        df["ret_24h_pct"] = list(range(5, 105, 5))  # 5, 10, 15, ..., 100
        filters = dict(_make_filters(df))
        mask = filters["ret24h_lt_25"](df)
        expected = df["ret_24h_pct"] < 25
        pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))

    def test_ret24h_bounded_filter(self):
        df = _make_df(n=10)
        df["ret_24h_pct"] = [3, 7, 12, 20, 25, 30, 0, -5, 10, 22]
        filters = dict(_make_filters(df))
        mask = filters["ret24h_5_to_25"](df)
        expected = (df["ret_24h_pct"] >= 5) & (df["ret_24h_pct"] < 25)
        pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))

    def test_volume_climax_filter(self):
        df = _make_df(n=10)
        df["is_volume_climax"] = [True, False, True, False, True, False, False, True, False, True]
        filters = dict(_make_filters(df))
        mask = filters["volume_climax"](df)
        expected = pd.Series([True, False, True, False, True, False, False, True, False, True])
        pd.testing.assert_series_equal(mask.reset_index(drop=True), expected, check_names=False)

    def test_cc_true_filter(self):
        df = _make_df(n=6)
        df["is_clean_continuation"] = [True, False, True, True, False, False]
        filters = dict(_make_filters(df))
        mask = filters["cc_true"](df)
        assert list(mask.reset_index(drop=True)) == [True, False, True, True, False, False]

    def test_cc_false_is_complement_of_cc_true(self):
        df = _make_df(n=10)
        filters = dict(_make_filters(df))
        cc_t = filters["cc_true"](df)
        cc_f = filters["cc_false"](df)
        assert (cc_t & cc_f).sum() == 0
        assert (cc_t | cc_f).all()

    def test_combined_filter_subset_of_components(self):
        """Logical AND filters must select a subset of each component."""
        df = _make_df(n=60)
        filters = dict(_make_filters(df))
        combo = filters["vr1h_ge_5_AND_hour_17_18_19"](df)
        vr_only = filters["vr1h_ge_5"](df)
        hour_only = filters["hour_safe_17_18_19"](df)
        # combo must be a subset of both components
        assert (combo & ~vr_only).sum() == 0
        assert (combo & ~hour_only).sum() == 0


# ---------------------------------------------------------------------------
# Missing column handling
# ---------------------------------------------------------------------------


class TestMissingColumnHandling:
    def test_no_volume_ratio_column(self, caplog):
        df = _make_df(n=30)
        df = df.drop(columns=["volume_ratio_1h"], errors="ignore")
        # Should not raise; vr filters just absent
        import logging
        with caplog.at_level(logging.WARNING):
            filters = dict(_make_filters(df))
        assert "vr1h_ge_5" not in filters

    def test_no_hour_column(self, caplog):
        df = _make_df(n=30)
        df = df.drop(columns=["hour_of_day_utc"], errors="ignore")
        import logging
        with caplog.at_level(logging.WARNING):
            filters = dict(_make_filters(df))
        assert "hour_safe_17_18_19" not in filters

    def test_no_ret24h_column(self, caplog):
        df = _make_df(n=30)
        df = df.drop(columns=["ret_24h_pct"], errors="ignore")
        import logging
        with caplog.at_level(logging.WARNING):
            filters = dict(_make_filters(df))
        assert "ret24h_lt_25" not in filters

    def test_no_volume_climax_column(self, caplog):
        df = _make_df(n=30)
        df = df.drop(columns=["is_volume_climax"], errors="ignore")
        import logging
        with caplog.at_level(logging.WARNING):
            filters = dict(_make_filters(df))
        assert "volume_climax" not in filters

    def test_no_cc_column(self, caplog):
        df = _make_df(n=30)
        df = df.drop(columns=["is_clean_continuation"], errors="ignore")
        import logging
        with caplog.at_level(logging.WARNING):
            filters = dict(_make_filters(df))
        assert "cc_true" not in filters

    def test_missing_max_favorable_tp10_falls_back(self):
        df = _make_df(n=10)
        df = df.drop(columns=["max_favorable_4h_pct"], errors="ignore")
        # When MFE column is absent, TP never triggers → returns raw 4h returns
        result = apply_exit_policy(df, exit_tp10_else_4h)
        expected = pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Top 1% exclusion
# ---------------------------------------------------------------------------


class TestTopPctExclusion:
    def test_single_outlier_removed(self):
        """With 100 obs and one massive outlier, avg_excl_top1pct should be close to base."""
        s = pd.Series([1.0] * 99 + [9999.0])
        stats = compute_stats(s)
        assert stats["avg_excl_top1pct"] < 10.0  # outlier is gone

    def test_excl_top1pct_le_avg(self):
        rng = np.random.default_rng(17)
        s = pd.Series(rng.normal(0, 5, 100))
        stats = compute_stats(s)
        assert stats["avg_excl_top1pct"] <= stats["avg"] + 1e-9

    def test_small_series_doesnt_crash(self):
        s = pd.Series([1.0, 2.0, 3.0])
        stats = compute_stats(s)
        assert isinstance(stats["avg_excl_top1pct"], float)


# ---------------------------------------------------------------------------
# passes_validation
# ---------------------------------------------------------------------------


class TestPassesValidation:
    GOOD = {"n": 50, "median": 0.5, "avg_excl_best": 0.2, "fail_rate_3": 0.20, "fail_rate_5": 0.10}
    BASE_FAIL3 = 0.273
    BASE_FAIL5 = 0.153

    def test_all_criteria_pass(self):
        ok, failed = passes_validation(self.GOOD, self.BASE_FAIL3, self.BASE_FAIL5)
        assert ok
        assert failed == []

    def test_n_too_small(self):
        stats = {**self.GOOD, "n": 10}
        ok, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert not ok
        assert any("n=" in f for f in failed)

    def test_negative_median(self):
        stats = {**self.GOOD, "median": -0.1}
        ok, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert not ok
        assert any("median" in f for f in failed)

    def test_negative_avg_excl_best(self):
        stats = {**self.GOOD, "avg_excl_best": -0.1}
        ok, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert not ok
        assert any("avg_excl_best" in f for f in failed)

    def test_fail_rate_too_high(self):
        stats = {**self.GOOD, "fail_rate_3": 0.40}
        ok, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert not ok
        assert any("fail_rate_3" in f for f in failed)

    def test_severe_fail_rate_too_high(self):
        stats = {**self.GOOD, "fail_rate_5": 0.25}
        ok, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert not ok
        assert any("fail_rate_5" in f for f in failed)

    def test_multiple_failures_reported(self):
        stats = {**self.GOOD, "n": 5, "median": -1.0}
        _, failed = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert len(failed) >= 2

    def test_exactly_at_threshold_passes(self):
        stats = {**self.GOOD, "fail_rate_3": self.BASE_FAIL3, "fail_rate_5": self.BASE_FAIL5}
        ok, _ = passes_validation(stats, self.BASE_FAIL3, self.BASE_FAIL5)
        assert ok


# ---------------------------------------------------------------------------
# run_filter_grid integration
# ---------------------------------------------------------------------------


class TestRunFilterGrid:
    def test_output_has_expected_columns(self):
        df = _make_df(n=60)
        summary, _ = run_filter_grid(df)
        for col in ["filter_name", "exit_policy", "n", "avg", "median",
                    "win_rate", "fail_rate_3", "fail_rate_5",
                    "avg_excl_best", "avg_excl_top1pct", "passes_validation"]:
            assert col in summary.columns, f"Missing: {col}"

    def test_baseline_all_present(self):
        df = _make_df(n=60)
        summary, _ = run_filter_grid(df)
        assert "baseline_all" in summary["filter_name"].values

    def test_both_policies_present_for_each_filter(self):
        df = _make_df(n=60)
        summary, _ = run_filter_grid(df)
        for fname in summary["filter_name"].unique():
            policies_for = summary[summary["filter_name"] == fname]["exit_policy"].tolist()
            for p in EXIT_POLICIES:
                assert p in policies_for, f"Policy {p} missing for filter {fname}"

    def test_event_log_has_filter_columns(self):
        df = _make_df(n=60)
        _, event_log = run_filter_grid(df)
        # Should have at least one "in_filter__" column
        in_filter_cols = [c for c in event_log.columns if c.startswith("in_filter__")]
        assert len(in_filter_cols) > 0

    def test_filter_event_counts_consistent(self):
        """n in summary should match sum of in_filter__ mask."""
        df = _make_df(n=60)
        summary, event_log = run_filter_grid(df)
        for fname in summary["filter_name"].unique():
            col = f"in_filter__{fname}"
            if col not in event_log.columns:
                continue
            expected_n = int(event_log[col].fillna(False).sum())
            actual_n = int(summary[summary["filter_name"] == fname]["n"].iloc[0])
            assert actual_n == expected_n, f"n mismatch for {fname}: {actual_n} vs {expected_n}"

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        summary, event_log = run_filter_grid(df)
        assert summary.empty
        assert event_log.empty

    def test_passes_validation_column_is_bool(self):
        df = _make_df(n=60)
        summary, _ = run_filter_grid(df)
        assert summary["passes_validation"].dtype in (bool, object)

    def test_tp10_returns_capped_at_10(self):
        df = _make_df(n=30)
        df["max_favorable_4h_pct"] = 15.0  # all events hit TP
        summary, _ = run_filter_grid(df)
        tp10_baseline = summary[
            (summary["filter_name"] == "baseline_all") &
            (summary["exit_policy"] == "tp10_else_4h")
        ]
        assert not tp10_baseline.empty
        assert float(tp10_baseline["best"].iloc[0]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Hour filtering edge cases
# ---------------------------------------------------------------------------


class TestHourFiltering:
    def test_high_failure_hours_are_excluded(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0] * 24,
            "hour_of_day_utc": list(range(24)),
            "volume_ratio_1h": [5.0] * 24,
            "ret_24h_pct": [10.0] * 24,
            "is_volume_climax": [False] * 24,
            "is_clean_continuation": [True] * 24,
        })
        filters = dict(_make_filters(df))
        mask = filters["excl_high_failure_hours"](df)
        excluded = df[~mask]["hour_of_day_utc"].tolist()
        assert set(excluded) == HIGH_FAILURE_HOURS

    def test_safe_hours_only_includes_17_18_19(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0] * 24,
            "hour_of_day_utc": list(range(24)),
            "volume_ratio_1h": [5.0] * 24,
            "ret_24h_pct": [10.0] * 24,
            "is_volume_climax": [False] * 24,
            "is_clean_continuation": [True] * 24,
        })
        filters = dict(_make_filters(df))
        mask = filters["hour_safe_17_18_19"](df)
        included = set(df[mask]["hour_of_day_utc"].tolist())
        assert included == {17, 18, 19}

    def test_no_hour_column_graceful(self):
        df = _make_df(n=20)
        df = df.drop(columns=["hour_of_day_utc"], errors="ignore")
        # Should not raise
        filters = dict(_make_filters(df))
        assert "hour_safe_17_18_19" not in filters
