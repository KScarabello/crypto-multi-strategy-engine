"""Unit tests for research/memecoin_catcher/analyze_backfill_signal_traits.py.

Covers:
- Group assignment
- Boolean share calculation
- Threshold sweep calculations
- Outcome fields excluded from sweep filters (no lookahead)
- Minimum event count filtering
- Graceful handling of missing columns
- Empty / no-signal windows
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from research.memecoin_catcher.analyze_backfill_signal_traits import (
    OUTCOME_FIELDS,
    SWEEP_TRAIT_COLS,
    assign_groups,
    build_bool_share_table,
    build_group_summary,
    build_hour_summary,
    build_symbol_summary,
    build_threshold_sweep,
    load_events,
    _ret_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal event-log DataFrame from keyword-array columns."""
    n = len(next(iter(kwargs.values()))) if kwargs else 3
    base: dict = {
        "future_ret_4h_pct": [1.0] * n,
        "future_ret_24h_pct": [2.0] * n,
        "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * n,
        "symbol": ["BTC/USD"] * n,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


@pytest.fixture()
def basic_df() -> pd.DataFrame:
    """10-row DataFrame with varied outcomes for group tests."""
    returns = [10.0, 6.0, 3.0, 1.0, 0.5, 0.0, -1.0, -3.0, -5.0, -10.0]
    return _make_df(
        future_ret_4h_pct=returns,
        is_clean_continuation=[True, False, True, False, True, True, False, True, False, True],
        is_volume_climax=[False, True, False, False, True, False, False, True, False, True],
        is_overextended_24h=[True, False, False, False, False, True, False, False, True, False],
        is_rolling_over=[False] * 10,
        is_wide_spread=[False] * 10,
        danger_terminal_spike=[False] * 10,
        breakout_24h=[True, True, False, False, True, False, False, True, False, True],
        ret_1h_pct=[2.0, 1.0, 0.5, 0.3, -0.2, 0.0, -0.5, -1.0, -2.0, -3.0],
        ret_4h_pct=[8.0, 5.0, 2.0, 0.8, 0.3, -0.2, -1.5, -4.0, -6.0, -9.0],
        ret_24h_pct=[20.0, 15.0, 10.0, 5.0, 2.0, 0.0, -2.0, -5.0, -10.0, -15.0],
        volume_ratio_1h=[5.0, 8.0, 3.0, 2.0, 1.5, 1.2, 4.0, 2.5, 1.8, 6.0],
        primary_ohlc_score=[15.0, 12.0, 9.0, 7.0, 6.0, 5.0, 11.0, 8.0, 4.0, 13.0],
        long_explosion_score=[14.0, 11.0, 8.0, 6.5, 5.5, 4.0, 10.0, 7.0, 3.5, 12.0],
        max_favorable_4h_pct=[12.0, 8.0, 5.0, 2.0, 1.5, 1.0, 3.0, 1.5, 0.5, 2.0],
        max_adverse_4h_pct=[-0.5, -1.0, -1.5, -0.5, -2.0, -3.5, -3.0, -5.0, -7.0, -12.0],
    )


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------


class TestAssignGroups:
    def test_all_events_returns_complete_rows(self, basic_df):
        groups = assign_groups(basic_df)
        assert len(groups["all_events"]) == 10

    def test_big_winners_threshold(self, basic_df):
        groups = assign_groups(basic_df)
        bw = groups["big_winners_4h"]
        assert all(bw["future_ret_4h_pct"] >= 5.0)
        assert len(bw) == 2  # 10.0 and 6.0

    def test_winners_threshold(self, basic_df):
        groups = assign_groups(basic_df)
        w = groups["winners_4h"]
        assert all(w["future_ret_4h_pct"] > 0.0)
        assert len(w) == 5  # 10, 6, 3, 1, 0.5

    def test_losers_threshold(self, basic_df):
        groups = assign_groups(basic_df)
        lo = groups["losers_4h"]
        assert all(lo["future_ret_4h_pct"] <= 0.0)
        assert len(lo) == 5  # 0, -1, -3, -5, -10

    def test_failures_threshold(self, basic_df):
        groups = assign_groups(basic_df)
        f = groups["failures_4h"]
        assert all(f["future_ret_4h_pct"] <= -3.0)
        assert len(f) == 3  # -3, -5, -10

    def test_severe_failures_threshold(self, basic_df):
        groups = assign_groups(basic_df)
        sf = groups["severe_failures_4h"]
        assert all(sf["future_ret_4h_pct"] <= -5.0)
        assert len(sf) == 2  # -5, -10

    def test_drops_rows_with_nan_outcome(self):
        df = _make_df(future_ret_4h_pct=[1.0, float("nan"), -2.0])
        groups = assign_groups(df)
        assert len(groups["all_events"]) == 2

    def test_empty_dataframe_returns_empty_groups(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        groups = assign_groups(df)
        for name in ["all_events", "big_winners_4h", "winners_4h", "losers_4h"]:
            assert len(groups[name]) == 0

    def test_all_groups_are_subsets_of_all_events(self, basic_df):
        groups = assign_groups(basic_df)
        all_idx = set(groups["all_events"].index)
        for name, sub in groups.items():
            if name != "all_events":
                assert set(sub.index).issubset(all_idx), f"Group {name} not subset of all_events"


# ---------------------------------------------------------------------------
# Boolean share calculation
# ---------------------------------------------------------------------------


class TestBoolShareTable:
    def test_share_values_between_0_and_1(self, basic_df):
        groups = assign_groups(basic_df)
        share_df = build_bool_share_table(groups)
        valid = share_df["share"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_missing_bool_column_produces_nan(self):
        df = _make_df()  # no bool columns
        groups = assign_groups(df)
        share_df = build_bool_share_table(groups)
        # All shares should be NaN because bool columns absent
        assert share_df["share"].isna().all()

    def test_all_true_column_gives_share_1(self):
        df = _make_df(future_ret_4h_pct=[1.0, 2.0, 3.0], is_clean_continuation=[True, True, True])
        groups = assign_groups(df)
        share_df = build_bool_share_table(groups)
        cc_row = share_df[(share_df["group"] == "all_events") & (share_df["trait"] == "is_clean_continuation")]
        assert not cc_row.empty
        assert cc_row["share"].iloc[0] == pytest.approx(1.0)

    def test_all_false_column_gives_share_0(self):
        df = _make_df(future_ret_4h_pct=[1.0, 2.0], is_clean_continuation=[False, False])
        groups = assign_groups(df)
        share_df = build_bool_share_table(groups)
        cc_row = share_df[(share_df["group"] == "all_events") & (share_df["trait"] == "is_clean_continuation")]
        assert cc_row["share"].iloc[0] == pytest.approx(0.0)

    def test_group_names_in_output(self, basic_df):
        groups = assign_groups(basic_df)
        share_df = build_bool_share_table(groups)
        expected_groups = {"all_events", "big_winners_4h", "winners_4h", "losers_4h", "failures_4h", "severe_failures_4h"}
        assert expected_groups.issubset(set(share_df["group"].unique()))


# ---------------------------------------------------------------------------
# _ret_stats
# ---------------------------------------------------------------------------


class TestRetStats:
    def test_correct_win_rate(self):
        s = pd.Series([5.0, -1.0, 2.0, -3.0])
        stats = _ret_stats(s)
        assert stats["win_rate"] == pytest.approx(0.5)

    def test_big_win_rate(self):
        s = pd.Series([6.0, 4.0, -2.0])
        stats = _ret_stats(s)
        assert stats["big_win_rate"] == pytest.approx(1 / 3, abs=1e-3)

    def test_fail_rate_3(self):
        s = pd.Series([1.0, -3.0, -5.0, 2.0])
        stats = _ret_stats(s)
        assert stats["fail_rate_3"] == pytest.approx(0.5)

    def test_fail_rate_5(self):
        s = pd.Series([1.0, -3.0, -5.0, 2.0])
        stats = _ret_stats(s)
        assert stats["fail_rate_5"] == pytest.approx(0.25)

    def test_avg_excl_best(self):
        s = pd.Series([100.0, 1.0, 2.0])
        stats = _ret_stats(s)
        assert stats["avg_excl_best"] == pytest.approx(1.5)

    def test_empty_series_returns_nan(self):
        stats = _ret_stats(pd.Series(dtype=float))
        assert stats["n"] == 0
        assert math.isnan(stats["avg"])

    def test_nan_values_are_dropped(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        stats = _ret_stats(s)
        assert stats["n"] == 2
        assert stats["avg"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Threshold sweep — correctness
# ---------------------------------------------------------------------------


class TestThresholdSweep:
    def _make_sweep_df(self, n: int = 50) -> pd.DataFrame:
        """Generate a df with sufficient events for sweep tests."""
        import numpy as np
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 5, n).tolist()
        vr = rng.uniform(1.0, 10.0, n).tolist()
        r1h = rng.uniform(-3.0, 5.0, n).tolist()
        r4h = rng.uniform(-5.0, 10.0, n).tolist()
        r24h = rng.uniform(-10.0, 25.0, n).tolist()
        score = rng.uniform(5.0, 20.0, n).tolist()
        return pd.DataFrame({
            "future_ret_4h_pct": returns,
            "volume_ratio_1h": vr,
            "ret_1h_pct": r1h,
            "ret_4h_pct": r4h,
            "ret_24h_pct": r24h,
            "primary_ohlc_score": score,
            "long_explosion_score": score,
            "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * n,
            "symbol": ["BTC/USD"] * n,
        })

    def test_output_has_expected_columns(self):
        df = self._make_sweep_df()
        sweep = build_threshold_sweep(df, min_group_size=5)
        assert not sweep.empty
        for col in ["trait", "filter", "n", "avg", "median", "win_rate", "fail_rate_3"]:
            assert col in sweep.columns, f"Missing column: {col}"

    def test_min_group_size_enforced(self):
        df = self._make_sweep_df(n=50)
        sweep_tight = build_threshold_sweep(df, min_group_size=100)
        assert sweep_tight.empty

    def test_min_group_size_allows_rows(self):
        df = self._make_sweep_df(n=50)
        sweep = build_threshold_sweep(df, min_group_size=5)
        assert not sweep.empty

    def test_n_matches_actual_subset_size(self):
        df = self._make_sweep_df(n=50)
        sweep = build_threshold_sweep(df, min_group_size=5)
        for _, row in sweep.iterrows():
            # n must be a positive integer <= len(df)
            assert 1 <= row["n"] <= len(df)

    def test_avg_excl_best_less_than_or_equal_avg(self):
        """avg_excl_best should never exceed avg (removing best can only lower avg)."""
        df = self._make_sweep_df(n=60)
        sweep = build_threshold_sweep(df, min_group_size=5)
        valid = sweep[(sweep["n"] > 1) & sweep["avg_excl_best"].notna() & sweep["avg"].notna()]
        assert (valid["avg_excl_best"] <= valid["avg"] + 1e-9).all()

    def test_missing_trait_columns_silently_skipped(self):
        df = pd.DataFrame({"future_ret_4h_pct": [1.0] * 50, "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 50})
        # All SWEEP_TRAIT_COLS absent → empty sweep
        sweep = build_threshold_sweep(df, min_group_size=5)
        assert sweep.empty or len(sweep) == 0

    def test_spread_pct_all_nan_skipped(self):
        df = self._make_sweep_df()
        df["spread_pct"] = float("nan")
        sweep = build_threshold_sweep(df, min_group_size=5)
        # spread_pct rows should be absent or empty
        if not sweep.empty:
            assert "spread_pct" not in sweep["trait"].values


# ---------------------------------------------------------------------------
# Lookahead / no-outcome-fields in sweep
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_outcome_fields_not_in_sweep_trait_cols(self):
        """SWEEP_TRAIT_COLS must not contain any OUTCOME_FIELDS."""
        overlap = OUTCOME_FIELDS.intersection(SWEEP_TRAIT_COLS)
        assert not overlap, (
            f"Lookahead bias: SWEEP_TRAIT_COLS contains outcome field(s): {overlap}"
        )

    def test_outcome_fields_assertion_fires(self):
        """_sweep_thresholds_for_col raises AssertionError if given an outcome field."""
        from research.memecoin_catcher.analyze_backfill_signal_traits import _sweep_thresholds_for_col
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0] * 50,
            "max_favorable_4h_pct": [2.0] * 50,
        })
        with pytest.raises(AssertionError, match="outcome field"):
            _sweep_thresholds_for_col(df, "max_favorable_4h_pct", min_group_size=5)

    def test_build_threshold_sweep_assertion_on_outcome_field(self):
        """build_threshold_sweep assertion fires if SWEEP_TRAIT_COLS contains an outcome field."""
        import research.memecoin_catcher.analyze_backfill_signal_traits as mod
        original = mod.SWEEP_TRAIT_COLS[:]
        mod.SWEEP_TRAIT_COLS.append("future_ret_4h_pct")
        try:
            df = pd.DataFrame({"future_ret_4h_pct": [1.0] * 50, "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 50})
            with pytest.raises(AssertionError):
                build_threshold_sweep(df)
        finally:
            mod.SWEEP_TRAIT_COLS[:] = original


# ---------------------------------------------------------------------------
# Symbol summary
# ---------------------------------------------------------------------------


class TestSymbolSummary:
    def test_groups_by_symbol(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0, 2.0, -1.0, -2.0],
            "symbol": ["A", "A", "B", "B"],
            "is_clean_continuation": [True, False, True, True],
            "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 4,
        })
        sym = build_symbol_summary(df)
        assert set(sym["symbol"]) == {"A", "B"}
        a_row = sym[sym["symbol"] == "A"].iloc[0]
        assert a_row["n"] == 2
        assert a_row["avg"] == pytest.approx(1.5)

    def test_missing_future_ret_4h_dropped(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0, float("nan"), 2.0],
            "symbol": ["A", "A", "A"],
            "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 3,
        })
        sym = build_symbol_summary(df)
        assert sym.iloc[0]["n"] == 2

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float), "symbol": pd.Series(dtype=str)})
        sym = build_symbol_summary(df)
        assert sym.empty


# ---------------------------------------------------------------------------
# Hour-of-day summary
# ---------------------------------------------------------------------------


class TestHourSummary:
    def test_has_24_hours(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0, -1.0, 2.0, -2.0],
            "snapshot_ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T06:00:00Z",
                "2025-01-01T12:00:00Z",
                "2025-01-01T18:00:00Z",
            ],
            "symbol": ["BTC/USD"] * 4,
        })
        hour = build_hour_summary(df)
        assert len(hour) == 24

    def test_correct_stats_for_hour(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [4.0, 6.0, -1.0, -3.0],
            "snapshot_ts_utc": [
                "2025-01-01T10:00:00Z",
                "2025-01-01T10:00:00Z",
                "2025-01-01T22:00:00Z",
                "2025-01-01T22:00:00Z",
            ],
            "symbol": ["BTC/USD"] * 4,
        })
        hour = build_hour_summary(df)
        h10 = hour[hour["hour_utc"] == 10].iloc[0]
        assert h10["n"] == 2
        assert h10["avg"] == pytest.approx(5.0)

    def test_missing_timestamp_column_returns_empty(self):
        df = pd.DataFrame({"future_ret_4h_pct": [1.0, 2.0], "symbol": ["A", "A"]})
        hour = build_hour_summary(df)
        assert hour.empty

    def test_invalid_timestamp_handled(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0, 2.0],
            "snapshot_ts_utc": ["not-a-date", "also-not-a-date"],
            "symbol": ["A", "A"],
        })
        hour = build_hour_summary(df)
        # Should not raise — NaT hours are excluded
        assert isinstance(hour, pd.DataFrame)


# ---------------------------------------------------------------------------
# Group summary
# ---------------------------------------------------------------------------


class TestGroupSummary:
    def test_output_has_one_row_per_group(self, basic_df):
        groups = assign_groups(basic_df)
        gs = build_group_summary(groups)
        expected = {"all_events", "big_winners_4h", "winners_4h", "losers_4h", "failures_4h", "severe_failures_4h"}
        assert expected.issubset(set(gs["group"]))

    def test_missing_bool_column_produces_nan(self):
        df = _make_df(future_ret_4h_pct=[1.0, -1.0, 2.0])
        groups = assign_groups(df)
        gs = build_group_summary(groups)
        for col in ["is_clean_continuation", "is_volume_climax"]:
            assert f"share_{col}" in gs.columns
            all_row = gs[gs["group"] == "all_events"].iloc[0]
            assert math.isnan(all_row[f"share_{col}"])

    def test_numeric_traits_absent_produce_nan(self):
        df = _make_df(future_ret_4h_pct=[1.0, -1.0])
        groups = assign_groups(df)
        gs = build_group_summary(groups)
        assert "ret_1h_pct_avg" in gs.columns
        all_row = gs[gs["group"] == "all_events"].iloc[0]
        assert math.isnan(all_row["ret_1h_pct_avg"])


# ---------------------------------------------------------------------------
# Empty / no-signal windows
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_df_assign_groups(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        groups = assign_groups(df)
        for name, sub in groups.items():
            assert len(sub) == 0

    def test_empty_df_group_summary(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        groups = assign_groups(df)
        gs = build_group_summary(groups)
        assert len(gs) == 6  # one row per group, all NaN

    def test_empty_df_sweep(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        sweep = build_threshold_sweep(df, min_group_size=1)
        assert sweep.empty

    def test_empty_df_symbol_summary(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float), "symbol": pd.Series(dtype=str)})
        sym = build_symbol_summary(df)
        assert sym.empty

    def test_empty_df_hour_summary(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": pd.Series(dtype=float),
            "snapshot_ts_utc": pd.Series(dtype=str),
        })
        hour = build_hour_summary(df)
        assert hour.empty


# ---------------------------------------------------------------------------
# Graceful missing column handling
# ---------------------------------------------------------------------------


class TestMissingColumns:
    def test_missing_bool_flags_in_group_summary(self):
        df = _make_df(future_ret_4h_pct=[1.0, 2.0, -1.0])
        groups = assign_groups(df)
        gs = build_group_summary(groups)
        # Should not raise; share columns present but NaN
        assert "share_is_clean_continuation" in gs.columns

    def test_missing_volume_ratio_in_sweep(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0] * 50,
            "ret_1h_pct": [0.5] * 50,
            "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 50,
        })
        # volume_ratio_1h absent → should not appear in sweep
        sweep = build_threshold_sweep(df, min_group_size=5)
        if not sweep.empty:
            assert "volume_ratio_1h" not in sweep["trait"].values

    def test_spread_pct_all_nan_does_not_crash(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [1.0] * 50,
            "spread_pct": [float("nan")] * 50,
            "snapshot_ts_utc": ["2025-01-01T10:00:00Z"] * 50,
        })
        sweep = build_threshold_sweep(df, min_group_size=5)
        # Should not crash; spread_pct absent from sweep rows
        assert isinstance(sweep, pd.DataFrame)
