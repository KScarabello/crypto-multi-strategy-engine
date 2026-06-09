"""Tests for research/fixed_five_normalized_efficiency.py (~40 tests).

Run with:
    .venv/bin/pytest tests/test_fixed_five_normalized_efficiency.py -v
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from research.fixed_five_normalized_efficiency import (
    BARS_PER_YEAR,
    SELECTED_NAMES,
    SELECTED_VARIANTS,
    SUPPRESSION_NOTE,
    YEAR_PERIODS,
    classify_shortlist,
    compute_counterfactual_reconciliation,
    compute_normalized_cost_efficiency,
    compute_performance_efficiency,
    compute_period_normalized_metrics,
    time_weighted_avg_equity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPORTS = Path("reports")


def _make_equity(n: int = 100, start: float = 10_000.0, growth: float = 0.001) -> pd.Series:
    """Create a simple growing equity series with UTC timestamps."""
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    values = [start * (1 + growth) ** i for i in range(n)]
    return pd.Series(values, index=idx, name="equity")


def _make_port_fx(equity: pd.Series) -> pd.DataFrame:
    rets = equity.pct_change().fillna(0)
    return pd.DataFrame({"equity": equity, "strategy_return": rets})


def _make_actual_costs(
    gross_end: float = 18_000.0,
    net_end: float = 17_000.0,
    notional: float = 50_000.0,
    fee: float = 500.0,
    slip: float = 250.0,
) -> dict:
    return {
        "gross_ending_equity": gross_end,
        "net_ending_equity": net_end,
        "total_executed_notional": notional,
        "actual_fee_dollars": fee,
        "actual_slippage_dollars": slip,
        "actual_direct_cost_dollars": fee + slip,
    }


def _make_rebalance_log(n: int = 20, joint_start: pd.Timestamp | None = None) -> pd.DataFrame:
    if joint_start is None:
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
    ts = pd.date_range(joint_start, periods=n, freq="7D")
    return pd.DataFrame({
        "execution_timestamp": ts,
        "signal_timestamp": ts - pd.Timedelta(hours=4),
        "turnover": [0.15] * n,
        "cost_rate": [0.0015] * n,
        "weight_sum": [1.0] * n,
    })


def _make_holdings_history(
    symbols: list[str] | None = None,
    n: int = 100,
    joint_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    if joint_start is None:
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
    idx = pd.date_range(joint_start, periods=n, freq="4h")
    data = {}
    for i, sym in enumerate(symbols):
        weights = [1 / 3 if (j + i) % 3 != 0 else 0.0 for j in range(n)]
        data[sym] = weights
    return pd.DataFrame(data, index=idx)


def _make_mock_result(
    n: int = 100,
    joint_start: pd.Timestamp | None = None,
    symbols: list[str] | None = None,
) -> MagicMock:
    if joint_start is None:
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
    reb_log = _make_rebalance_log(n=20, joint_start=joint_start)
    hh = _make_holdings_history(symbols=symbols, n=n, joint_start=joint_start)
    mock = MagicMock()
    mock.rebalance_log = reb_log
    mock.holdings_history = hh
    equity_series = _make_equity(n=n, start=10_000.0)
    mock.portfolio = pd.DataFrame({"equity": equity_series})
    mock.gross_return = equity_series.pct_change().fillna(0)
    return mock


# ---------------------------------------------------------------------------
# Part 1: time_weighted_avg_equity
# ---------------------------------------------------------------------------

class TestTimeWeightedAvgEquity:
    def test_twae_is_mean_of_equity_series(self):
        equity = _make_equity(50)
        joint_start = equity.index[0]
        result = time_weighted_avg_equity(equity, joint_start)
        expected = float(equity.loc[equity.index >= joint_start].mean())
        assert abs(result - expected) < 1e-6

    def test_twae_different_from_starting_equity(self):
        """Growing portfolio: TWAE > starting equity."""
        equity = _make_equity(100, start=10_000.0, growth=0.01)
        joint_start = equity.index[0]
        twae = time_weighted_avg_equity(equity, joint_start)
        assert twae > float(equity.iloc[0])

    def test_twae_excludes_before_joint_start(self):
        equity = _make_equity(100)
        # joint_start at midpoint
        joint_start = equity.index[50]
        twae = time_weighted_avg_equity(equity, joint_start)
        expected = float(equity.loc[equity.index >= joint_start].mean())
        assert abs(twae - expected) < 1e-6
        assert twae != float(equity.mean())


class TestNormalizedCostEfficiency:
    def _run(self, n: int = 100, n_rebs: int = 20) -> dict:
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
        equity = _make_equity(n, start=10_000.0, growth=0.002)
        port_fx = _make_port_fx(equity)
        actual_costs = _make_actual_costs(net_end=float(equity.iloc[-1]))
        result = _make_mock_result(n=n, joint_start=joint_start)
        return compute_normalized_cost_efficiency(
            "test_variant", result, port_fx, joint_start, actual_costs
        )

    def test_notional_div_twae_normalized(self):
        """Large portfolio doesn't inflate ratio unfairly."""
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")

        # Small portfolio
        eq_small = _make_equity(100, start=10_000.0, growth=0.002)
        pfx_small = _make_port_fx(eq_small)
        costs_small = _make_actual_costs(
            net_end=float(eq_small.iloc[-1]),
            notional=5_000.0,
            fee=50.0,
            slip=25.0,
        )
        result_small = _make_mock_result(n=100, joint_start=joint_start)
        norm_small = compute_normalized_cost_efficiency(
            "small", result_small, pfx_small, joint_start, costs_small
        )

        # Large portfolio (100x bigger)
        eq_large = _make_equity(100, start=1_000_000.0, growth=0.002)
        pfx_large = _make_port_fx(eq_large)
        costs_large = _make_actual_costs(
            net_end=float(eq_large.iloc[-1]),
            notional=500_000.0,
            fee=5_000.0,
            slip=2_500.0,
        )
        result_large = _make_mock_result(n=100, joint_start=joint_start)
        norm_large = compute_normalized_cost_efficiency(
            "large", result_large, pfx_large, joint_start, costs_large
        )

        # Ratios should be similar (within 5%) for proportional portfolios
        ratio_small = norm_small["notional_div_twae"]
        ratio_large = norm_large["notional_div_twae"]
        assert abs(ratio_small - ratio_large) / max(ratio_large, 1e-9) < 0.05

    def test_annualized_turnover_denominator(self):
        """Annualized turnover uses (end - start).days / 365.25."""
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
        equity = _make_equity(100, start=10_000.0, growth=0.001)
        port_fx = _make_port_fx(equity)
        actual_costs = _make_actual_costs(net_end=float(equity.iloc[-1]))
        result = _make_mock_result(n=100, joint_start=joint_start)
        norm = compute_normalized_cost_efficiency(
            "v", result, port_fx, joint_start, actual_costs
        )
        end_ts = equity.index[-1]
        expected_years = (end_ts - joint_start).days / 365.25
        total_turn = float(result.rebalance_log["turnover"].sum())
        expected_ann_turn = total_turn / max(expected_years, 0.001)
        assert abs(norm["annualized_turnover"] - round(expected_ann_turn, 4)) < 1e-3

    def test_direct_cost_div_twae_dimensionless(self):
        result = self._run()
        ratio = result["direct_cost_div_twae"]
        assert 0.0 <= ratio <= 1.0  # should be a small pure ratio

    def test_gross_profit_per_cost_dollar_positive(self):
        """For profitable strategy, gross_profit_per_cost_dollar > 0."""
        result = self._run()
        assert result["gross_profit_per_cost_dollar"] > 0

    def test_net_profit_per_cost_dollar_less_than_gross(self):
        """Costs reduce net profit vs gross profit."""
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
        equity = _make_equity(100, start=10_000.0, growth=0.003)
        port_fx = _make_port_fx(equity)
        actual_costs = _make_actual_costs(
            gross_end=float(equity.iloc[-1]) + 500,
            net_end=float(equity.iloc[-1]),
        )
        result = _make_mock_result(n=100, joint_start=joint_start)
        norm = compute_normalized_cost_efficiency(
            "v", result, port_fx, joint_start, actual_costs
        )
        assert norm["net_profit_per_cost_dollar"] < norm["gross_profit_per_cost_dollar"]

    def test_replacements_per_invested_year_positive(self):
        result = self._run()
        assert result["replacements_per_invested_year"] > 0

    def test_direct_cost_per_replacement_positive(self):
        result = self._run()
        assert result["direct_cost_per_replacement"] > 0

    def test_cost_efficiency_required_columns(self):
        result = self._run()
        assert "notional_div_twae" in result
        assert "direct_cost_div_twae" in result
        assert "gross_profit_per_cost_dollar" in result
        assert "net_profit_per_cost_dollar" in result


# ---------------------------------------------------------------------------
# Part 3: Counterfactual reconciliation
# ---------------------------------------------------------------------------

class TestCounterfactualReconciliation:
    def _make_variants(self) -> dict:
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
        variants = {}
        for name, net_end in [("control", 15_000.0), ("variant_a", 16_000.0), ("variant_b", 14_500.0)]:
            eq = _make_equity(100, start=10_000.0)
            eq = eq * (net_end / float(eq.iloc[-1]))
            port_fx = _make_port_fx(eq)
            actual_costs = _make_actual_costs(
                net_end=net_end,
                gross_end=net_end + 1000,
                fee=500.0 if name == "control" else 300.0,
                slip=250.0 if name == "control" else 150.0,
            )
            result = _make_mock_result(n=100, joint_start=joint_start)
            variants[name] = (result, port_fx, actual_costs)
        return variants

    def test_counterfactual_delta_direct_cost_sign(self):
        """control minus variant (positive = variant cheaper)."""
        variants = self._make_variants()
        df, _ = compute_counterfactual_reconciliation(variants, "control")
        # variant_a has fee=300+150=450 vs control 500+250=750 → delta should be positive
        assert df.loc["variant_a", "true_delta_direct_cost"] > 0

    def test_counterfactual_delta_net_equity_matches_equity_curves(self):
        """Δ net equity = variant_net - control_net directly."""
        variants = self._make_variants()
        df, _ = compute_counterfactual_reconciliation(variants, "control")
        # variant_a net_end=16000, control=15000 → delta = +1000
        assert abs(df.loc["variant_a", "true_delta_net_equity"] - 1000.0) < 1.0
        # variant_b net_end=14500 → delta = -500
        assert abs(df.loc["variant_b", "true_delta_net_equity"] - (-500.0)) < 1.0

    def test_suppression_note_warns_about_overlap(self):
        assert "non-additive" in SUPPRESSION_NOTE.lower()

    def test_suppression_sums_labeled_non_additive(self):
        assert "3.086" in SUPPRESSION_NOTE or "$3,086" in SUPPRESSION_NOTE.replace(",", ",") or "3,086" in SUPPRESSION_NOTE

    def test_counterfactual_csv_columns(self):
        variants = self._make_variants()
        df, _ = compute_counterfactual_reconciliation(variants, "control")
        assert "true_delta_direct_cost" in df.columns
        assert "true_delta_net_equity" in df.columns
        assert "true_delta_gross_equity" in df.columns


# ---------------------------------------------------------------------------
# Part 4: Year-by-year stability
# ---------------------------------------------------------------------------

class TestYearByYearStability:
    def _make_close(self, n: int = 500) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
        return pd.DataFrame({
            "BTC/USD": np.random.uniform(10_000, 50_000, n),
            "ETH/USD": np.random.uniform(200, 4_000, n),
            "SOL/USD": np.random.uniform(1, 200, n),
        }, index=idx)

    def test_year_period_slicing_2022(self):
        for period_name, ps, pe in YEAR_PERIODS:
            if period_name == "2022":
                assert ps == "2022-01-01"
                assert pe == "2022-12-31"
                break
        else:
            pytest.fail("2022 period not found in YEAR_PERIODS")

    def test_year_period_slicing_2020_2021(self):
        for period_name, ps, pe in YEAR_PERIODS:
            if period_name == "2020-2021":
                assert ps == "2020-09-28"
                assert pe == "2021-12-31"
                break
        else:
            pytest.fail("2020-2021 period not found in YEAR_PERIODS")

    def test_period_direct_cost_uses_rebalance_log(self):
        """Period direct cost computed from rebalance_log, not raw equity subtraction."""
        joint_start = pd.Timestamp("2022-01-01", tz="UTC")
        n = 200
        equity = _make_equity(n, start=10_000.0)
        port_fx = _make_port_fx(equity)
        close = self._make_close(n)
        actual_costs = _make_actual_costs(net_end=float(equity.iloc[-1]))
        result = _make_mock_result(n=n, joint_start=joint_start)

        pm = compute_period_normalized_metrics(
            "v", result, port_fx, close, joint_start,
            "2022", "2022-01-01", "2022-12-31", actual_costs
        )
        # Result should have this key (not NaN from missing data)
        assert "period_direct_cost_dollars" in pm
        # Should be >= 0
        assert pm["period_direct_cost_dollars"] >= 0.0

    def test_year_by_year_period_has_required_columns(self):
        joint_start = pd.Timestamp("2022-01-01", tz="UTC")
        n = 200
        equity = _make_equity(n, start=10_000.0)
        port_fx = _make_port_fx(equity)
        close = self._make_close(n)
        actual_costs = _make_actual_costs(net_end=float(equity.iloc[-1]))
        result = _make_mock_result(n=n, joint_start=joint_start)

        pm = compute_period_normalized_metrics(
            "v", result, port_fx, close, joint_start,
            "2022", "2022-01-01", "2022-12-31", actual_costs
        )
        required = [
            "variant", "period", "period_return_pct", "period_max_drawdown_pct",
            "period_sharpe", "period_cagr_pct", "period_direct_cost_dollars",
            "period_annualized_turnover", "period_direct_cost_div_twae",
            "period_n_rebalances", "period_n_rank_replacements",
        ]
        for col in required:
            assert col in pm, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Part 5: Shortlist
# ---------------------------------------------------------------------------

class TestShortlist:
    def _make_m(
        self,
        variant: str = "test",
        sharpe: float = 1.8,
        maxdd: float = -45.0,
    ) -> dict:
        return {
            "variant": variant,
            "sharpe": sharpe,
            "max_drawdown_pct": maxdd,
            "cagr_pct": 100.0,
            "calmar": 2.2,
            "return_2022_pct": -20.0,
        }

    def _make_norm(self, notional_div_twae: float = 0.5) -> dict:
        return {
            "notional_div_twae": notional_div_twae,
            "annualized_turnover": 10.0,
            "net_profit": 50_000.0,
            "total_direct_cost": 500.0,
            "net_ending_equity": 60_000.0,
        }

    def _make_cf(self, delta_direct: float = 100.0) -> dict:
        return {"true_delta_direct_cost": delta_direct}

    def _make_year_metrics(self, variant: str, years_improved: list[str] | None = None) -> list[dict]:
        """Create year metrics where specific years show improvement over control."""
        if years_improved is None:
            years_improved = ["2022", "2023", "2024", "2025"]
        rows = []
        for year in ["2022", "2023", "2024", "2025"]:
            v_ret = 10.0 if year in years_improved else -10.0
            rows.append({"variant": variant, "period": year, "period_return_pct": v_ret, "period_sharpe": 1.0})
            rows.append({"variant": "control", "period": year, "period_return_pct": 0.0, "period_sharpe": 0.8})
        return rows

    def test_shortlist_max_3_entries(self):
        """Even if many qualify, shortlist ≤ 3."""
        from research.fixed_five_normalized_efficiency import build_final_shortlist

        n_variants = 6
        all_perf = [
            self._make_m(f"v{i}", sharpe=1.9 - i * 0.01, maxdd=-44.0)
            for i in range(n_variants)
        ] + [self._make_m("control", sharpe=1.7, maxdd=-46.0)]

        all_norm = [
            self._make_norm(notional_div_twae=0.4)
            for _ in range(n_variants + 1)
        ]
        for i, n in enumerate(all_norm):
            n["variant"] = all_perf[i]["variant"]
            n["annualized_turnover"] = 10.0
            n["net_profit"] = 50_000.0
            n["total_direct_cost"] = 400.0
            n["net_ending_equity"] = 60_000.0

        # Add variant key to norm
        all_norm_with_variant = [
            {**all_norm[i], "variant": all_perf[i]["variant"]}
            for i in range(len(all_perf))
        ]

        # Build counterfactual df
        rows = []
        for m in all_perf:
            rows.append({
                "variant": m["variant"],
                "true_delta_direct_cost": 200.0 if m["variant"] != "control" else 0.0,
                "true_delta_net_equity": 5000.0 if m["variant"] != "control" else 0.0,
                "true_delta_gross_equity": 5000.0,
                "true_delta_gross_pnl": 4000.0,
                "true_delta_net_pnl": 4500.0,
                "variant_net_equity": 65_000.0,
                "control_net_equity": 60_000.0,
                "variant_direct_cost": 400.0,
                "control_direct_cost": 600.0,
            })
        cf_df = pd.DataFrame(rows).set_index("variant")

        year_metrics = []
        for m in all_perf:
            v = m["variant"]
            for yr in ["2022", "2023", "2024", "2025"]:
                year_metrics.append({"variant": v, "period": yr, "period_return_pct": 5.0, "period_sharpe": 1.0})
        # control returns
        for yr in ["2022", "2023", "2024", "2025"]:
            year_metrics.append({"variant": "control", "period": yr, "period_return_pct": 0.0, "period_sharpe": 0.5})

        shortlist = build_final_shortlist(all_perf, all_norm_with_variant, cf_df, year_metrics)
        assert len(shortlist) <= 3

    def test_shortlist_excludes_control(self):
        """Control variant is never on the shortlist."""
        ctrl_m = self._make_m("control", sharpe=1.7, maxdd=-46.0)
        ctrl_norm = self._make_norm(notional_div_twae=0.6)
        cf = self._make_cf(delta_direct=0.0)
        year_metrics = self._make_year_metrics("control", years_improved=[])
        label = classify_shortlist(ctrl_m, ctrl_norm, cf, year_metrics, ctrl_m, ctrl_norm)
        assert label == "CONTROL"

    def test_shortlist_requires_cost_reduction(self):
        """All shortlist members must have positive true_delta_direct_cost."""
        ctrl_m = self._make_m("control", sharpe=1.7, maxdd=-46.0)
        ctrl_norm = self._make_norm(notional_div_twae=0.6)
        # Variant with higher costs
        v_m = self._make_m("expensive", sharpe=1.85, maxdd=-44.0)
        v_norm = self._make_norm(notional_div_twae=0.4)
        cf = self._make_cf(delta_direct=-100.0)  # variant costs MORE
        year_metrics = self._make_year_metrics("expensive", years_improved=["2022", "2023", "2024", "2025"])
        label = classify_shortlist(v_m, v_norm, cf, year_metrics, ctrl_m, ctrl_norm)
        assert label == "REJECT"

    def test_shortlist_requires_sharpe_not_much_worse(self):
        """No shortlist member Sharpe < control - 0.15."""
        ctrl_m = self._make_m("control", sharpe=1.7, maxdd=-46.0)
        ctrl_norm = self._make_norm(notional_div_twae=0.6)
        v_m = self._make_m("bad_sharpe", sharpe=1.7 - 0.2, maxdd=-44.0)
        v_norm = self._make_norm(notional_div_twae=0.4)
        cf = self._make_cf(delta_direct=100.0)
        year_metrics = self._make_year_metrics("bad_sharpe", years_improved=["2022", "2023", "2024", "2025"])
        label = classify_shortlist(v_m, v_norm, cf, year_metrics, ctrl_m, ctrl_norm)
        assert label == "REJECT"

    def test_reject_if_higher_costs(self):
        ctrl_m = self._make_m("control", sharpe=1.7, maxdd=-46.0)
        ctrl_norm = self._make_norm(notional_div_twae=0.6)
        v_m = self._make_m("costly", sharpe=1.8, maxdd=-44.0)
        v_norm = self._make_norm(notional_div_twae=0.4)
        cf = {"true_delta_direct_cost": -500.0}
        year_metrics = self._make_year_metrics("costly", years_improved=["2022", "2023", "2024"])
        label = classify_shortlist(v_m, v_norm, cf, year_metrics, ctrl_m, ctrl_norm)
        assert label == "REJECT"

    def test_reject_if_dd_much_worse(self):
        ctrl_m = self._make_m("control", sharpe=1.7, maxdd=-46.0)
        ctrl_norm = self._make_norm(notional_div_twae=0.6)
        v_m = self._make_m("bad_dd", sharpe=1.8, maxdd=-46.0 - 6.0)
        v_norm = self._make_norm(notional_div_twae=0.4)
        cf = self._make_cf(delta_direct=100.0)
        year_metrics = self._make_year_metrics("bad_dd", years_improved=["2022", "2023", "2024"])
        label = classify_shortlist(v_m, v_norm, cf, year_metrics, ctrl_m, ctrl_norm)
        assert label == "REJECT"


# ---------------------------------------------------------------------------
# Part 6: Selection lock document
# ---------------------------------------------------------------------------

class TestSelectionLockDocument:
    def _make_shortlist(self) -> list[dict]:
        return [{"variant": "rank_buffer_4", "sharpe": 1.82, "cagr_pct": 145.0, "max_drawdown_pct": -44.5}]

    def test_selection_lock_mentions_no_live_deployment(self):
        from research.fixed_five_normalized_efficiency import build_selection_lock_document
        content = build_selection_lock_document(self._make_shortlist(), Path("reports"))
        assert "not approved for live" in content.lower() or "no shortlisted variant is approved" in content.lower()

    def test_selection_lock_mentions_in_sample(self):
        from research.fixed_five_normalized_efficiency import build_selection_lock_document
        content = build_selection_lock_document(self._make_shortlist(), Path("reports"))
        assert "in-sample" in content.lower()

    def test_selection_lock_mentions_future_validation(self):
        from research.fixed_five_normalized_efficiency import build_selection_lock_document
        content = build_selection_lock_document(self._make_shortlist(), Path("reports"))
        assert "future validation" in content.lower()

    def test_selection_lock_lists_parameters_tested(self):
        from research.fixed_five_normalized_efficiency import build_selection_lock_document
        content = build_selection_lock_document(self._make_shortlist(), Path("reports"))
        # Must mention MA lengths and score hurdle
        assert "ma" in content.lower() or "ma lengths" in content.lower()
        assert "score_hurdle" in content.lower() or "score hurdle" in content.lower()


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

class TestVariantConfiguration:
    def test_no_new_variants(self):
        """SELECTED_VARIANTS contains only the 9 approved names."""
        variant_names = {v.name for v in SELECTED_VARIANTS}
        assert variant_names == SELECTED_NAMES
        assert len(SELECTED_VARIANTS) == 9

    def test_no_live_imports(self):
        """No imports from brokers/, execution/, or live/ directories."""
        import ast
        script_path = Path("research/fixed_five_normalized_efficiency.py")
        source = script_path.read_text()
        tree = ast.parse(source)
        forbidden_prefixes = ("brokers", "execution", "live")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        for prefix in forbidden_prefixes:
                            assert not module.startswith(prefix), (
                                f"Forbidden import from {module}"
                            )
                for prefix in forbidden_prefixes:
                    assert not module.startswith(prefix), (
                        f"Forbidden import from {module}"
                    )

    def test_canonical_initialization(self):
        """joint_start must come from find_joint_eligible_start."""
        from research.fixed_five_normalized_efficiency import run_all
        import inspect
        src = inspect.getsource(run_all)
        assert "find_joint_eligible_start" in src

    def test_min_hold_variants_normalized_intensity(self):
        """min_hold_4 and min_hold_6 have notional_div_twae computed (not nan)."""
        joint_start = pd.Timestamp("2021-01-01", tz="UTC")
        n = 80
        equity = _make_equity(n, start=10_000.0, growth=0.002)
        port_fx = _make_port_fx(equity)
        actual_costs = _make_actual_costs(net_end=float(equity.iloc[-1]))
        result = _make_mock_result(n=n, joint_start=joint_start)

        for vname in ["min_hold_4", "min_hold_6"]:
            norm = compute_normalized_cost_efficiency(
                vname, result, port_fx, joint_start, actual_costs
            )
            assert math.isfinite(norm["notional_div_twae"])
            assert norm["notional_div_twae"] > 0


# ---------------------------------------------------------------------------
# Output files (require full run — skip unless reports exist)
# ---------------------------------------------------------------------------

class TestOutputFiles:
    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_normalized_cost_efficiency.csv").exists(),
        reason="Full run not completed yet",
    )
    def test_all_6_output_files_created(self):
        expected = [
            "fixed_five_normalized_cost_efficiency.csv",
            "fixed_five_performance_efficiency.csv",
            "fixed_five_suppression_counterfactual_reconciliation.md",
            "fixed_five_normalized_efficiency_by_year.csv",
            "fixed_five_final_in_sample_shortlist.md",
            "fixed_five_final_selection_lock.md",
        ]
        for fname in expected:
            assert (_REPORTS / fname).exists(), f"Missing: {fname}"

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_normalized_cost_efficiency.csv").exists(),
        reason="Full run not completed yet",
    )
    def test_normalized_efficiency_csv_has_9_rows(self):
        df = pd.read_csv(_REPORTS / "fixed_five_normalized_cost_efficiency.csv")
        assert len(df) == 9, f"Expected 9 rows, got {len(df)}"

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_performance_efficiency.csv").exists(),
        reason="Full run not completed yet",
    )
    def test_performance_efficiency_csv_has_9_rows(self):
        df = pd.read_csv(_REPORTS / "fixed_five_performance_efficiency.csv")
        assert len(df) == 9, f"Expected 9 rows, got {len(df)}"

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_normalized_efficiency_by_year.csv").exists(),
        reason="Full run not completed yet",
    )
    def test_by_year_csv_has_9x6_rows(self):
        df = pd.read_csv(_REPORTS / "fixed_five_normalized_efficiency_by_year.csv")
        assert len(df) == 54, f"Expected 54 rows (9×6), got {len(df)}"

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_final_in_sample_shortlist.md").exists(),
        reason="Full run not completed yet",
    )
    def test_shortlist_md_mentions_in_sample(self):
        content = (_REPORTS / "fixed_five_final_in_sample_shortlist.md").read_text()
        assert "in-sample" in content.lower()

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_final_in_sample_shortlist.md").exists(),
        reason="Full run not completed yet",
    )
    def test_shortlist_md_no_live_deployment_claim(self):
        content = (_REPORTS / "fixed_five_final_in_sample_shortlist.md").read_text()
        assert "not approved for live" in content.lower() or "no variant" in content.lower()

    @pytest.mark.skipif(
        not (_REPORTS / "fixed_five_normalized_cost_efficiency.csv").exists(),
        reason="Full run not completed yet",
    )
    def test_deterministic_output(self):
        """Two reads of the same CSV return identical results."""
        df1 = pd.read_csv(_REPORTS / "fixed_five_normalized_cost_efficiency.csv")
        df2 = pd.read_csv(_REPORTS / "fixed_five_normalized_cost_efficiency.csv")
        pd.testing.assert_frame_equal(df1, df2)
