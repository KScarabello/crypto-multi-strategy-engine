"""Unit tests for research/memecoin_catcher/validate_memecoin_candidate_rules.py.

Covers:
- Each candidate rule selection
- tp10_else_4h exit logic
- apply_fee / slippage adjustment
- Chronological split logic
- Daily bucket logic
- Leave-one-symbol-out calculation
- Liquidity diagnostics
- Missing column handling
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from research.memecoin_catcher.validate_memecoin_candidate_rules import (
    CANDIDATE_RULES,
    FEE_BPS,
    MIN_EVENTS_FOR_BUCKET,
    SAFE_HOURS,
    apply_fee,
    apply_fixed_4h,
    apply_tp10_else_4h,
    build_fee_stress,
    build_liquidity_diagnostics,
    build_symbol_robustness,
    build_time_splits,
    build_validation_summary,
    compute_stats,
    load_events,
    main,
    rule1_mask,
    rule2_mask,
    rule3_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 60, seed: int = 42, **overrides) -> pd.DataFrame:
    """
    Build a minimal DataFrame with random (not sequential) flag assignment so
    candidate rule masks will always find matching events.
    """
    rng = np.random.default_rng(seed)
    base = {
        "future_ret_4h_pct":     rng.normal(0, 5, n).tolist(),
        "future_ret_24h_pct":    rng.normal(0, 8, n).tolist(),
        "max_favorable_4h_pct":  np.abs(rng.normal(3, 4, n)).tolist(),
        "max_adverse_4h_pct":    (-np.abs(rng.normal(3, 4, n))).tolist(),
        "volume_ratio_1h":       rng.uniform(3, 20, n).tolist(),  # ~80 % >= 5
        "ret_24h_pct":           rng.uniform(5, 35, n).tolist(),  # mix above/below 25
        # Random flag assignment so rules have overlap
        "is_volume_climax":      (rng.random(n) < 0.30).tolist(),   # 30 % True
        "is_clean_continuation": (rng.random(n) < 0.50).tolist(),   # 50 % True
        # Random hours so safe hours {17,18,19} appear ~12 % of events
        "hour_of_day_utc":       rng.integers(0, 24, n).tolist(),
        "day_index":             [i // 8 for i in range(n)],
        "date_utc":              [f"2025-01-0{1 + i // 8}" for i in range(n)],
        "symbol":                [f"TOK{i % 5}/USD" for i in range(n)],
        "snapshot_ts_utc":       [f"2025-01-0{1 + i // 24}T{i % 24:02d}:00:00Z" for i in range(n)],
        "ohlc_signal_type":      ["LONG_EXPLOSION"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_guaranteed_df(events_per_rule: int = 42, seed: int = 7) -> pd.DataFrame:
    """
    Build a DataFrame that guarantees at least *events_per_rule* matching
    events for every candidate rule, plus an equal number of noise rows.
    Useful for integration-style tests (time splits, fee stress, symbol robustness).
    """
    rng = np.random.default_rng(seed)

    def ret() -> float:
        return float(rng.normal(1, 4))

    def mfe() -> float:
        return float(abs(rng.normal(4, 3)))

    rows: list[dict] = []

    def _base(i: int, sym: str, **flags) -> dict:
        defaults = dict(
            future_ret_4h_pct=ret(),
            future_ret_24h_pct=ret(),
            max_favorable_4h_pct=mfe(),
            max_adverse_4h_pct=-abs(float(rng.normal(2, 2))),
            volume_ratio_1h=float(rng.uniform(5, 15)),
            ret_24h_pct=float(rng.uniform(5, 20)),
            day_index=i % 7,
            date_utc=f"2025-01-0{1 + i % 7}",
            symbol=sym,
            snapshot_ts_utc=f"2025-01-0{1 + i // 24}T{i % 24:02d}:00:00Z",
            ohlc_signal_type="LONG_EXPLOSION",
        )
        defaults.update(flags)  # flags override defaults
        return defaults

    # Rule 1: VC=True, CC=False
    for i in range(events_per_rule):
        rows.append(_base(i, f"R1_{i % 6}/USD",
                          is_volume_climax=True, is_clean_continuation=False,
                          hour_of_day_utc=(i % 20)))  # spread across non-safe hours too

    # Rule 2: VC=True, safe hour (17/18/19)
    for i in range(events_per_rule):
        rows.append(_base(i + 100, f"R2_{i % 6}/USD",
                          is_volume_climax=True, is_clean_continuation=True,
                          hour_of_day_utc=[17, 18, 19][i % 3],
                          volume_ratio_1h=float(rng.uniform(2, 4))))  # low VR to not overlap R3

    # Rule 3: VR1h>=5, safe hour, ret24<25 (VC=False to avoid overlap with R2)
    for i in range(events_per_rule):
        rows.append(_base(i + 200, f"R3_{i % 6}/USD",
                          is_volume_climax=False, is_clean_continuation=True,
                          hour_of_day_utc=[17, 18, 19][i % 3],
                          volume_ratio_1h=float(rng.uniform(6, 15)),
                          ret_24h_pct=float(rng.uniform(5, 20))))

    # Noise rows (match no candidate rule)
    for i in range(events_per_rule * 2):
        rows.append(_base(i + 300, "NOISE/USD",
                          is_volume_climax=False, is_clean_continuation=False,
                          hour_of_day_utc=5,
                          volume_ratio_1h=float(rng.uniform(1, 3)),
                          ret_24h_pct=float(rng.uniform(30, 50))))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rule masks
# ---------------------------------------------------------------------------


class TestRule1Mask:
    def test_requires_both_vc_true_and_cc_false(self):
        df = _make_df(n=4,
                      ohlc_signal_type=["LONG_EXPLOSION"] * 4,
                      is_volume_climax=[True, True, False, False],
                      is_clean_continuation=[False, True, False, True])
        m = rule1_mask(df)
        assert list(m) == [True, False, False, False]

    def test_requires_long_explosion(self):
        df = _make_df(n=4,
                      ohlc_signal_type=["LONG_EXPLOSION", "DUMPING", "LONG_EXPLOSION", "LONG_EXPLOSION"],
                      is_volume_climax=[True, True, True, False],
                      is_clean_continuation=[False, False, False, False])
        m = rule1_mask(df)
        assert list(m) == [True, False, True, False]

    def test_missing_ohlc_signal_type_returns_all_false(self):
        df = _make_df(n=4,
                      is_volume_climax=[True, True, True, True],
                      is_clean_continuation=[False, False, False, False]).drop(columns=["ohlc_signal_type"])
        m = rule1_mask(df)
        assert not m.any()

    def test_all_vc_true_cc_false(self):
        df = _make_df(n=3,
                      is_volume_climax=[True, True, True],
                      is_clean_continuation=[False, False, False])
        assert rule1_mask(df).all()

    def test_none_match(self):
        df = _make_df(n=3,
                      is_volume_climax=[False, False, False],
                      is_clean_continuation=[True, True, True])
        assert not rule1_mask(df).any()

    def test_missing_vc_returns_all_false(self):
        df = _make_df(n=5).drop(columns=["is_volume_climax"])
        assert not rule1_mask(df).any()

    def test_missing_cc_still_uses_vc(self):
        df = _make_df(n=5).drop(columns=["is_clean_continuation"])
        df["is_volume_climax"] = [True, True, False, True, False]
        m = rule1_mask(df)
        # CC condition skipped → result equals VC mask
        assert list(m) == [True, True, False, True, False]


class TestRule2Mask:
    def test_requires_vc_true_and_safe_hour(self):
        df = _make_df(n=4,
                      is_volume_climax=[True, True, False, True],
                      hour_of_day_utc=[17, 10, 17, 18])
        m = rule2_mask(df)
        assert list(m) == [True, False, False, True]

    def test_all_safe_hours_but_vc_false(self):
        df = _make_df(n=3,
                      is_volume_climax=[False, False, False],
                      hour_of_day_utc=[17, 18, 19])
        assert not rule2_mask(df).any()

    def test_missing_vc_returns_false(self):
        df = _make_df(n=3).drop(columns=["is_volume_climax"])
        assert not rule2_mask(df).any()

    def test_missing_hour_col_skips_hour_filter(self):
        df = _make_df(n=5).drop(columns=["hour_of_day_utc"])
        df["is_volume_climax"] = [True, True, False, True, False]
        m = rule2_mask(df)
        # Hour condition skipped → equals VC mask
        assert list(m) == [True, True, False, True, False]

    def test_safe_hours_set_is_exactly_17_18_19(self):
        assert SAFE_HOURS == frozenset({17, 18, 19})


class TestRule3Mask:
    def test_all_three_conditions(self):
        df = _make_df(n=4,
                      volume_ratio_1h=[6, 3, 6, 6],
                      hour_of_day_utc=[17, 17, 10, 18],
                      ret_24h_pct=[10, 10, 10, 30])
        m = rule3_mask(df)
        # row 0: vr>=5 ✓, hour=17 ✓, ret24<25 ✓ → True
        # row 1: vr=3 ✗ → False
        # row 2: hour=10 ✗ → False
        # row 3: ret24=30 ✗ → False
        assert list(m) == [True, False, False, False]

    def test_missing_vr_skips_vr_condition(self):
        df = _make_df(n=3,
                      hour_of_day_utc=[17, 18, 19],
                      ret_24h_pct=[10, 10, 10]).drop(columns=["volume_ratio_1h"])
        m = rule3_mask(df)
        assert m.all()

    def test_missing_ret24h_skips_condition(self):
        df = _make_df(n=3,
                      volume_ratio_1h=[6, 6, 6],
                      hour_of_day_utc=[17, 18, 19]).drop(columns=["ret_24h_pct"])
        m = rule3_mask(df)
        assert m.all()

    def test_ret24h_boundary(self):
        df = _make_df(n=2,
                      volume_ratio_1h=[6, 6],
                      hour_of_day_utc=[17, 17],
                      ret_24h_pct=[24.999, 25.0])
        m = rule3_mask(df)
        assert list(m) == [True, False]


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------


class TestApplyTp10Else4h:
    def test_tp_triggered_when_mfe_ge_10(self):
        df = pd.DataFrame({"future_ret_4h_pct": [3.0], "max_favorable_4h_pct": [12.0]})
        result = apply_tp10_else_4h(df)
        assert float(result.iloc[0]) == pytest.approx(10.0)

    def test_tp_not_triggered_when_mfe_lt_10(self):
        df = pd.DataFrame({"future_ret_4h_pct": [3.0], "max_favorable_4h_pct": [8.0]})
        assert float(apply_tp10_else_4h(df).iloc[0]) == pytest.approx(3.0)

    def test_missing_mfe_column_returns_raw_ret(self):
        df = pd.DataFrame({"future_ret_4h_pct": [5.0]})
        assert float(apply_tp10_else_4h(df).iloc[0]) == pytest.approx(5.0)

    def test_nan_mfe_falls_back(self):
        df = pd.DataFrame({"future_ret_4h_pct": [5.0], "max_favorable_4h_pct": [float("nan")]})
        assert float(apply_tp10_else_4h(df).iloc[0]) == pytest.approx(5.0)

    def test_tp_caps_multiple_rows(self):
        df = pd.DataFrame({
            "future_ret_4h_pct": [2.0, -1.0, 7.0],
            "max_favorable_4h_pct": [15.0, 4.0, 12.0],
        })
        result = apply_tp10_else_4h(df)
        assert list(result) == [10.0, -1.0, 10.0]

    def test_exactly_at_threshold(self):
        df = pd.DataFrame({"future_ret_4h_pct": [3.0], "max_favorable_4h_pct": [10.0]})
        assert float(apply_tp10_else_4h(df).iloc[0]) == pytest.approx(10.0)


class TestApplyFixed4h:
    def test_returns_raw_future_ret(self):
        df = pd.DataFrame({"future_ret_4h_pct": [2.5, -1.0, 0.0]})
        result = apply_fixed_4h(df)
        assert list(result) == pytest.approx([2.5, -1.0, 0.0])


# ---------------------------------------------------------------------------
# Fee / slippage
# ---------------------------------------------------------------------------


class TestApplyFee:
    def test_zero_fee_unchanged(self):
        s = pd.Series([3.0, -1.0, 2.0])
        pd.testing.assert_series_equal(apply_fee(s, 0), s)

    def test_100bps_fee_subtracts_1pct(self):
        s = pd.Series([3.0, -1.0])
        result = apply_fee(s, 100)
        assert list(result) == pytest.approx([2.0, -2.0])

    def test_50bps_fee(self):
        s = pd.Series([2.0])
        assert float(apply_fee(s, 50).iloc[0]) == pytest.approx(1.5)

    def test_fee_can_flip_winner_to_loser(self):
        s = pd.Series([0.4])
        result = apply_fee(s, 50)
        assert float(result.iloc[0]) < 0

    def test_fee_bps_list_contains_100(self):
        assert 100 in FEE_BPS


class TestBuildFeeStress:
    def test_output_has_all_fee_levels(self):
        df = _make_guaranteed_df()
        fee = build_fee_stress(df)
        for rule in CANDIDATE_RULES:
            rule_rows = fee[fee["rule"] == rule]
            assert set(rule_rows["fee_bps"].tolist()) == set(FEE_BPS), \
                f"Rule {rule} missing fee levels"
        for rule in CANDIDATE_RULES:
            rule_rows = fee[fee["rule"] == rule]
            assert set(rule_rows["fee_bps"].tolist()) == set(FEE_BPS)

    def test_avg_decreases_as_fee_increases(self):
        df = _make_guaranteed_df()
        fee = build_fee_stress(df)
        for rule in CANDIDATE_RULES:
            rule_rows = fee[fee["rule"] == rule].sort_values("fee_bps")
            if len(rule_rows) < 2:
                continue
            avgs = rule_rows["avg_net"].tolist()
            for i in range(len(avgs) - 1):
                assert avgs[i] >= avgs[i + 1] - 1e-9, f"avg_net not monotone for {rule}"

    def test_zero_fee_equals_gross_return(self):
        df = _make_guaranteed_df()
        fee = build_fee_stress(df)
        from research.memecoin_catcher.validate_memecoin_candidate_rules import build_validation_summary
        summary = build_validation_summary(df)
        for rule in CANDIDATE_RULES:
            zero_fee = fee[(fee["rule"] == rule) & (fee["fee_bps"] == 0)]
            if zero_fee.empty:
                continue
            gross = summary[summary["rule"] == rule]
            if gross.empty:
                continue
            assert float(zero_fee.iloc[0]["avg_net"]) == pytest.approx(
                float(gross.iloc[0]["avg"]), abs=1e-3
            )


# ---------------------------------------------------------------------------
# Chronological split logic
# ---------------------------------------------------------------------------


class TestBuildTimeSplits:
    def test_output_has_half_rows(self):
        df = _make_guaranteed_df()
        splits = build_time_splits(df)
        halves = splits[splits["split_type"] == "half"]
        assert len(halves) > 0

    def test_first_and_second_half_together_equal_total(self):
        df = _make_guaranteed_df()
        splits = build_time_splits(df)
        for rule in CANDIDATE_RULES:
            halves = splits[(splits["rule"] == rule) & (splits["split_type"] == "half")]
            if len(halves) < 2:
                continue
            total_n = halves["n"].sum()
            mask_fn, exit_fn = CANDIDATE_RULES[rule]
            completed = df.dropna(subset=["future_ret_4h_pct"])
            mask = mask_fn(completed).fillna(False)
            expected_n = int(mask.sum())
            assert total_n == expected_n

    def test_daily_buckets_present(self):
        df = _make_guaranteed_df()
        splits = build_time_splits(df)
        daily = splits[splits["split_type"] == "daily"]
        assert len(daily) > 0

    def test_rolling_2day_present(self):
        df = _make_guaranteed_df()
        splits = build_time_splits(df)
        rolling = splits[splits["split_type"] == "rolling_2day"]
        assert len(rolling) > 0

    def test_missing_day_index_returns_empty(self):
        df = _make_df(n=30).drop(columns=["day_index"])
        splits = build_time_splits(df)
        assert splits.empty

    def test_small_buckets_suppressed(self):
        """No bucket should appear with fewer than MIN_EVENTS_FOR_BUCKET events."""
        df = _make_guaranteed_df()
        splits = build_time_splits(df)
        if not splits.empty:
            for _, r in splits.iterrows():
                assert r["n"] >= MIN_EVENTS_FOR_BUCKET


# ---------------------------------------------------------------------------
# Leave-one-symbol-out / symbol robustness
# ---------------------------------------------------------------------------


class TestBuildSymbolRobustness:
    def test_output_contains_excl_rows(self):
        df = _make_guaranteed_df()
        rob = build_symbol_robustness(df)
        assert not rob.empty
        excl_rows = rob[rob["symbol"].str.startswith("EXCL_")]
        assert len(excl_rows) > 0

    def test_excl_top3_row_present(self):
        df = _make_guaranteed_df()
        rob = build_symbol_robustness(df)
        top3_rows = rob[rob["symbol"] == "EXCL_TOP3"]
        assert len(top3_rows) == len(CANDIDATE_RULES)

    def test_excl_top1_n_less_than_total(self):
        df = _make_guaranteed_df()
        rob = build_symbol_robustness(df)
        for rule in CANDIDATE_RULES:
            rule_rows = rob[rob["rule"] == rule]
            actual = rule_rows[~rule_rows["symbol"].str.startswith("EXCL_")]
            top1_sym = actual.iloc[0]["symbol"] if not actual.empty else None
            if top1_sym is None:
                continue
            excl1 = rule_rows[rule_rows["symbol"] == f"EXCL_{top1_sym}"]
            if excl1.empty:
                continue
            total_n = actual["n"].sum()
            excl1_n = int(excl1.iloc[0]["n"])
            assert excl1_n < total_n

    def test_pct_of_events_sums_to_1(self):
        df = _make_guaranteed_df()
        rob = build_symbol_robustness(df)
        for rule in CANDIDATE_RULES:
            rule_rows = rob[(rob["rule"] == rule) & ~rob["symbol"].str.startswith("EXCL_")]
            if rule_rows.empty:
                continue
            total_pct = rule_rows["pct_of_events"].sum()
            assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_missing_symbol_column_uses_pair_id(self):
        df = _make_guaranteed_df()
        df = df.drop(columns=["symbol"], errors="ignore")
        df["pair_id"] = [f"P{i % 3}USD" for i in range(len(df))]
        rob = build_symbol_robustness(df)
        assert not rob.empty


# ---------------------------------------------------------------------------
# Liquidity diagnostics
# ---------------------------------------------------------------------------


class TestBuildLiquidityDiagnostics:
    def test_all_rules_present(self):
        df = _make_df(n=40)
        liq = build_liquidity_diagnostics(df)
        assert set(liq["rule"].tolist()) == set(CANDIDATE_RULES.keys())

    def test_spread_unavailable_when_all_nan(self):
        df = _make_df(n=40)
        df["spread_pct"] = float("nan")
        liq = build_liquidity_diagnostics(df)
        for _, r in liq.iterrows():
            assert r["spread_available"] == False

    def test_spread_available_when_has_values(self):
        df = _make_guaranteed_df()
        df["spread_pct"] = 0.5
        # Only rules with n > 0 should have spread_available = True
        liq = build_liquidity_diagnostics(df)
        for _, r in liq.iterrows():
            if r["n"] > 0:
                assert r["spread_available"] == True

    def test_qvol_unavailable_when_absent(self):
        df = _make_df(n=40)
        assert "quote_volume_est" not in df.columns
        liq = build_liquidity_diagnostics(df)
        for _, r in liq.iterrows():
            assert r["qvol_available"] == False

    def test_qvol_pct_lt_10k_correct(self):
        df = _make_guaranteed_df()
        df["quote_volume_est"] = [5_000 if i % 2 == 0 else 50_000 for i in range(len(df))]
        liq = build_liquidity_diagnostics(df)
        for _, r in liq.iterrows():
            if r["n"] > 0 and r["qvol_available"]:
                assert 0.0 <= r["qvol_pct_lt_10k"] <= 1.0


# ---------------------------------------------------------------------------
# Compute stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_win_rate(self):
        s = pd.Series([1.0, -1.0, 2.0, -2.0])
        assert compute_stats(s)["win_rate"] == pytest.approx(0.5)

    def test_excl_best_le_avg(self):
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0, 5, 50))
        stats = compute_stats(s)
        assert stats["avg_excl_best"] <= stats["avg"] + 1e-9

    def test_excl_top1pct_correct(self):
        s = pd.Series([1.0] * 99 + [9999.0])
        stats = compute_stats(s)
        assert stats["avg_excl_top1pct"] == pytest.approx(1.0, abs=0.1)

    def test_empty_returns_nan(self):
        stats = compute_stats(pd.Series(dtype=float))
        assert stats["n"] == 0
        assert math.isnan(stats["avg"])

    def test_nan_values_dropped(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        assert compute_stats(s)["n"] == 2


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------


class TestBuildValidationSummary:
    def test_all_rules_in_output(self):
        df = _make_df(n=60)
        summary = build_validation_summary(df)
        all_rules = set(CANDIDATE_RULES.keys()) | {"baseline_all_fixed_4h", "baseline_all_tp10"}
        for rule in all_rules:
            assert rule in summary["rule"].values

    def test_candidate_flag_set_correctly(self):
        df = _make_df(n=60)
        summary = build_validation_summary(df)
        for rule in CANDIDATE_RULES:
            row = summary[summary["rule"] == rule]
            assert bool(row.iloc[0]["is_candidate"]) == True
        for rule in ["baseline_all_fixed_4h", "baseline_all_tp10"]:
            row = summary[summary["rule"] == rule]
            assert bool(row.iloc[0]["is_candidate"]) == False

    def test_stats_are_reasonable(self):
        df = _make_df(n=60)
        summary = build_validation_summary(df)
        for _, r in summary.iterrows():
            if r["n"] > 0:
                assert 0.0 <= r["win_rate"] <= 1.0
                assert 0.0 <= r["fail_rate_3"] <= 1.0

    def test_empty_completed_gives_zero_n(self):
        df = pd.DataFrame({"future_ret_4h_pct": pd.Series(dtype=float)})
        summary = build_validation_summary(df)
        assert (summary["n"] == 0).all()


# ---------------------------------------------------------------------------
# Missing column edge cases
# ---------------------------------------------------------------------------


class TestMissingColumns:
    def test_no_volume_climax_rule1_all_false(self):
        df = _make_df(n=20).drop(columns=["is_volume_climax"])
        assert not rule1_mask(df).any()

    def test_no_hour_col_rule2_skips_hour(self):
        df = _make_df(n=5,
                      is_volume_climax=[True] * 5).drop(columns=["hour_of_day_utc"])
        m = rule2_mask(df)
        # Hour skipped → result = VC mask
        assert m.all()

    def test_no_mfe_col_tp10_returns_raw(self):
        df = _make_df(n=10).drop(columns=["max_favorable_4h_pct"])
        result = apply_tp10_else_4h(df)
        expected = pd.to_numeric(df["future_ret_4h_pct"], errors="coerce")
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_liquidity_graceful_with_no_spread_or_qvol(self):
        df = _make_df(n=30)
        # Neither spread_pct nor quote_volume_est present
        liq = build_liquidity_diagnostics(df)
        assert not liq.empty
        for _, r in liq.iterrows():
            assert r["spread_available"] == False
            assert r["qvol_available"] == False


# ---------------------------------------------------------------------------
# Backfill-only labeling and provenance
# ---------------------------------------------------------------------------


class TestBackfillOnlyLabeling:
    def test_main_prints_backfill_only_warning(self, tmp_path, capsys):
        df = _make_guaranteed_df(events_per_rule=8)
        input_path = tmp_path / "memecoin_backfilled_signal_events.csv"
        df.to_csv(input_path, index=False)

        main(
            input_path=input_path,
            summary_path=tmp_path / "summary.csv",
            time_splits_path=tmp_path / "time_splits.csv",
            symbol_robust_path=tmp_path / "symbol.csv",
            fee_stress_path=tmp_path / "fee.csv",
            liquidity_path=tmp_path / "liq.csv",
        )

        out = capsys.readouterr().out.lower()
        assert "backfill-only validation" in out
        assert "not current genuine prospective evidence" in out
        assert "not used for gate a readiness" in out
        assert "backfill_n" in out

    def test_summary_csv_has_backfill_provenance_columns(self, tmp_path):
        df = _make_guaranteed_df(events_per_rule=8)
        input_path = tmp_path / "memecoin_backfilled_signal_events.csv"
        df.to_csv(input_path, index=False)

        main(
            input_path=input_path,
            summary_path=tmp_path / "summary.csv",
            time_splits_path=tmp_path / "time_splits.csv",
            symbol_robust_path=tmp_path / "symbol.csv",
            fee_stress_path=tmp_path / "fee.csv",
            liquidity_path=tmp_path / "liq.csv",
        )

        summary = pd.read_csv(tmp_path / "summary.csv")
        for col in [
            "backfill_n",
            "source_dataset_path",
            "sample_type",
            "scope_label",
            "used_for_gate_a_readiness",
            "is_current_genuine_evidence",
        ]:
            assert col in summary.columns

        assert set(summary["sample_type"].astype(str).unique()) == {"SIMULATED_BACKFILL_ONLY"}
        assert (summary["used_for_gate_a_readiness"] == False).all()  # noqa: E712
        assert (summary["is_current_genuine_evidence"] == False).all()  # noqa: E712
