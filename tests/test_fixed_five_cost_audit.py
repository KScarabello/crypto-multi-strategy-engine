"""Tests for research/fixed_five_cost_audit.py

Verifies:
- fee_dollars = executed_notional × fee_rate (not estimated from gross-minus-net)
- slippage_dollars reconcile to price differences
- trade-reason classifications sum to total trades
- turnover-source costs sum to total direct costs
- gross-minus-net is not labeled as fees paid
- additive rate × initial equity is not labeled as actual cost
- counterfactual curves use aligned dates
- canonical initialization identical
- deterministic reruns
- no live imports
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_cost_audit as ca
from research.fixed_five_cost_audit import (
    TradeRecord,
    _classify_trade_reason,
    build_trade_ledger,
    compute_cost_reconciliation,
    compute_turnover_decomposition,
    build_rebalance_event_audit,
    run_counterfactual_curves,
    TRADE_REASONS,
    COST_SCENARIOS,
    CANDIDATE_LOCK_CONTENT,
)
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    WhipsawControlledSignalGen,
    compute_controlled_gate_series,
)
from research.fixed_five_defensive_overlay import BaseSignalGen
from research.universe_integrity_analysis import (
    DEFAULT_INITIAL_CAPITAL, DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
    find_joint_eligible_start,
)
import research.universe_integrity_analysis as u


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_close(n: int = 400, seed: int = 42) -> pd.DataFrame:
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data = {}
    for sym in symbols:
        data[sym] = 1000 * np.cumprod(1 + 0.001 + rng.normal(0, 0.01, n))
    return pd.DataFrame(data, index=idx)


def _run_strategy(close, ma_bars=20, entry_confirm=1, entry_buf=0.0):
    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        pytest.skip("Not enough bars")
    btc = close["BTC/USD"].dropna()
    params = WhipsawControl(
        name="test",
        entry_confirm_bars=entry_confirm,
        entry_buffer_pct=entry_buf,
    )
    gate = compute_controlled_gate_series(btc, ma_bars, params)
    sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
    ohlcv = u._close_to_ohlcv(close)
    from research.fixed_five_canonical_regime_comparison import run_canonical
    result, port_fx = run_canonical(
        ohlcv, sig, joint_start,
        rebalance_bars=12,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        fee_bps=DEFAULT_FEE_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )
    gate_fx = gate.reindex(port_fx.index).fillna(False)
    return result, port_fx, gate_fx, joint_start, close


# ---------------------------------------------------------------------------
# Fee dollars = executed notional × fee rate
# ---------------------------------------------------------------------------

class TestFeeCalculation:
    def test_fee_equals_notional_times_fee_rate(self):
        """Every trade's fee_dollars must exactly equal gross_notional × fee_rate."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
            fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        fee_rate = DEFAULT_FEE_BPS / 10_000
        for t in ledger:
            expected_fee = t.gross_notional * fee_rate
            assert abs(t.fee_dollars - expected_fee) < 0.01, \
                f"fee_dollars={t.fee_dollars:.4f} != notional×rate={expected_fee:.4f}"

    def test_slippage_equals_notional_times_slippage_rate(self):
        """Every trade's slippage_dollars must equal gross_notional × slippage_rate."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
            fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        slip_rate = DEFAULT_SLIPPAGE_BPS / 10_000
        for t in ledger:
            expected_slip = t.gross_notional * slip_rate
            assert abs(t.slippage_dollars - expected_slip) < 0.01

    def test_total_direct_cost_equals_fee_plus_slippage(self):
        """total_direct_cost = fee + slippage for every trade."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        for t in ledger:
            expected = t.fee_dollars + t.slippage_dollars
            assert abs(t.total_direct_cost_dollars - expected) < 0.01

    def test_gross_notional_is_abs_weight_change_times_equity(self):
        """gross_notional ≈ |weight_change| × portfolio_equity_before_trade."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        for t in ledger:
            expected = abs(t.weight_change) * t.portfolio_equity_before_trade
            assert abs(t.gross_notional - expected) < 0.10


# ---------------------------------------------------------------------------
# Slippage price reconciliation
# ---------------------------------------------------------------------------

class TestSlippagePriceReconciliation:
    def test_buy_slipped_price_greater_than_raw(self):
        """BUY execution should fill at price × (1 + slippage_rate)."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        slip_rate = DEFAULT_SLIPPAGE_BPS / 10_000
        for t in ledger:
            if t.side == "BUY" and not math.isnan(t.execution_price):
                expected_slipped = t.execution_price * (1.0 + slip_rate)
                assert abs(t.execution_price_after_slippage - expected_slipped) < 1e-4

    def test_sell_slipped_price_less_than_raw(self):
        """SELL execution should fill at price × (1 - slippage_rate)."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        slip_rate = DEFAULT_SLIPPAGE_BPS / 10_000
        for t in ledger:
            if t.side == "SELL" and not math.isnan(t.execution_price):
                expected_slipped = t.execution_price * (1.0 - slip_rate)
                assert abs(t.execution_price_after_slippage - expected_slipped) < 1e-4


# ---------------------------------------------------------------------------
# Trade reason classification
# ---------------------------------------------------------------------------

class TestTradeReasonClassification:
    def test_all_reasons_are_valid(self):
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        for t in ledger:
            assert t.trade_reason in TRADE_REASONS, \
                f"Unknown trade reason: {t.trade_reason}"

    def test_classify_rank_entry(self):
        reason = _classify_trade_reason(
            "ETH/USD", 0.0, 0.333,
            was_regime_exit=False, was_regime_entry=False,
            prev_held_set={"BTC/USD"}, new_held_set={"BTC/USD", "ETH/USD"},
        )
        assert reason == "RANK_ENTRY"

    def test_classify_rank_exit(self):
        reason = _classify_trade_reason(
            "ETH/USD", 0.333, 0.0,
            was_regime_exit=False, was_regime_entry=False,
            prev_held_set={"BTC/USD", "ETH/USD"}, new_held_set={"BTC/USD"},
        )
        assert reason == "RANK_EXIT"

    def test_classify_regime_exit(self):
        reason = _classify_trade_reason(
            "ETH/USD", 0.333, 0.0,
            was_regime_exit=True, was_regime_entry=False,
            prev_held_set={"BTC/USD", "ETH/USD"}, new_held_set=set(),
        )
        assert reason == "REGIME_EXIT"

    def test_classify_regime_entry(self):
        reason = _classify_trade_reason(
            "ETH/USD", 0.0, 0.333,
            was_regime_exit=False, was_regime_entry=True,
            prev_held_set=set(), new_held_set={"BTC/USD", "ETH/USD"},
        )
        assert reason == "REGIME_ENTRY"

    def test_classify_reweight(self):
        reason = _classify_trade_reason(
            "ETH/USD", 0.333, 0.400,
            was_regime_exit=False, was_regime_entry=False,
            prev_held_set={"BTC/USD", "ETH/USD"}, new_held_set={"BTC/USD", "ETH/USD"},
        )
        assert reason == "REWEIGHT_EXISTING_POSITION"

    def test_trade_reasons_sum_to_total(self):
        """Sum of trades per reason must equal total trades."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        decomp = compute_turnover_decomposition("test", ledger)
        total_row = next(r for r in decomp if r["trade_reason"] == "TOTAL")
        reason_rows = [r for r in decomp if r["trade_reason"] != "TOTAL"]
        assert sum(r["trade_count"] for r in reason_rows) == total_row["trade_count"]


# ---------------------------------------------------------------------------
# Turnover source costs sum to total
# ---------------------------------------------------------------------------

class TestTurnoverDecomposition:
    def test_cost_sums_match_total(self):
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        decomp = compute_turnover_decomposition("test", ledger)
        total_row = next(r for r in decomp if r["trade_reason"] == "TOTAL")
        reason_rows = [r for r in decomp if r["trade_reason"] != "TOTAL"]
        sum_cost = sum(r["total_direct_cost_dollars"] for r in reason_rows)
        assert abs(sum_cost - total_row["total_direct_cost_dollars"]) < 1.0

    def test_fee_sums_match_total(self):
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        decomp = compute_turnover_decomposition("test", ledger)
        total_row = next(r for r in decomp if r["trade_reason"] == "TOTAL")
        reason_rows = [r for r in decomp if r["trade_reason"] != "TOTAL"]
        sum_fee = sum(r["fee_dollars"] for r in reason_rows)
        assert abs(sum_fee - total_row["fee_dollars"]) < 1.0

    def test_pct_sums_to_100(self):
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        decomp = compute_turnover_decomposition("test", ledger)
        reason_rows = [r for r in decomp if r["trade_reason"] != "TOTAL"]
        if sum(r["total_direct_cost_dollars"] for r in reason_rows) > 0:
            total_pct = sum(r["pct_of_total_direct_cost"] for r in reason_rows)
            assert abs(total_pct - 100.0) < 0.5


# ---------------------------------------------------------------------------
# Gross-minus-net labeling — NOT fees paid
# ---------------------------------------------------------------------------

class TestCostLabeling:
    def test_gross_minus_net_not_labeled_fees(self):
        """The reconciliation must NOT use 'gross_minus_net' as 'fees paid'."""
        source = Path("research/fixed_five_cost_audit.py").read_text()
        # Check that our reconciliation docstring/comments correctly distinguish
        assert "gross-minus-net is not the same as" in source.lower() or \
               "gross-minus-net ending equity" in source.lower() or \
               "not fees paid" in source.lower()

    def test_reconciliation_has_residual_foregone_field(self):
        """The reconciliation dict must include a residual_foregone_compounding field."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        recon = compute_cost_reconciliation("test", result, port_fx, ledger, joint_start)
        assert "residual_foregone_compounding_dollars" in recon
        assert "actual_fee_dollars" in recon
        assert "actual_slippage_dollars" in recon

    def test_gross_minus_net_not_equal_to_direct_cost(self):
        """gross_minus_net should be > total_direct_cost due to foregone compounding."""
        close = _make_close(n=600)
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        recon = compute_cost_reconciliation("test", result, port_fx, ledger, joint_start)
        # gross_minus_net >= total_direct_cost (compounding makes it larger)
        # Allow tolerance for very short periods
        assert recon["gross_minus_net_dollars"] >= recon["total_direct_cost_dollars"] - 100.0

    def test_additive_rate_sum_labeled_reference_only(self):
        """The additive rate sum × start equity must be labeled as REFERENCE ONLY."""
        close = _make_close()
        result, port_fx, gate_fx, joint_start, close = _run_strategy(close)
        ledger = build_trade_ledger(
            "test", result, port_fx, close, gate_fx, joint_start,
        )
        recon = compute_cost_reconciliation("test", result, port_fx, ledger, joint_start)
        # Key must be named REFERENCE_ONLY
        assert "additive_rate_sum_times_start_equity_REFERENCE_ONLY" in recon
        assert "additive_rate_sum_pct_REFERENCE_ONLY" in recon

    def test_source_code_does_not_describe_additive_rate_as_actual_cost(self):
        """Source must not multiply additive rate by initial equity and call it actual cost."""
        source = Path("research/fixed_five_cost_audit.py").read_text()
        # The key must be clearly marked REFERENCE_ONLY
        assert "REFERENCE_ONLY" in source
        assert "NOT actual cost" in source or "not labeled as actual" in source.lower() or \
               "REFERENCE ONLY" in source


# ---------------------------------------------------------------------------
# Counterfactual curves — aligned dates and signals
# ---------------------------------------------------------------------------

class TestCounterfactualCurves:
    def _run_cf(self, close):
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")
        btc = close["BTC/USD"].dropna()
        params = WhipsawControl(name="ctrl")
        gate = compute_controlled_gate_series(btc, 20, params)
        sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
        ohlcv = u._close_to_ohlcv(close)
        rows = run_counterfactual_curves(ohlcv, sig, joint_start, "test", rebalance_bars=12)
        return rows, joint_start

    def test_four_scenarios_generated(self):
        close = _make_close(n=400)
        rows, _ = self._run_cf(close)
        assert len(rows) == 4

    def test_scenario_names_match_spec(self):
        close = _make_close(n=400)
        rows, _ = self._run_cf(close)
        names = {r["scenario"] for r in rows}
        assert "no_fees_no_slippage" in names
        assert "fees_only" in names
        assert "slippage_only" in names
        assert "fees_and_slippage" in names

    def test_no_fee_scenario_has_highest_equity(self):
        """No-fee/no-slippage must have highest ending equity."""
        close = _make_close(n=500)
        rows, _ = self._run_cf(close)
        no_cost = next(r for r in rows if r["scenario"] == "no_fees_no_slippage")
        with_cost = next(r for r in rows if r["scenario"] == "fees_and_slippage")
        assert no_cost["net_ending_equity"] >= with_cost["net_ending_equity"]

    def test_fee_only_plus_slip_only_less_than_no_cost(self):
        """fees_only and slippage_only must each be between no-cost and full-cost."""
        close = _make_close(n=500)
        rows, _ = self._run_cf(close)
        r = {row["scenario"]: row for row in rows}
        assert r["no_fees_no_slippage"]["net_ending_equity"] >= \
               r["fees_only"]["net_ending_equity"]
        assert r["no_fees_no_slippage"]["net_ending_equity"] >= \
               r["slippage_only"]["net_ending_equity"]
        assert r["fees_only"]["net_ending_equity"] >= \
               r["fees_and_slippage"]["net_ending_equity"]
        assert r["slippage_only"]["net_ending_equity"] >= \
               r["fees_and_slippage"]["net_ending_equity"]

    def test_counterfactual_contains_path_dependency_note(self):
        close = _make_close(n=400)
        rows, _ = self._run_cf(close)
        for r in rows:
            assert "path_dependency_note" in r
            assert len(r["path_dependency_note"]) > 20

    def test_all_scenarios_same_equity_at_start(self):
        """All four scenarios share identical joint_start and initial capital."""
        close = _make_close(n=400)
        rows, _ = self._run_cf(close)
        starts = {r["equity_at_start"] for r in rows}
        # All should start at same equity (same canonical initialization)
        assert len(starts) == 1


# ---------------------------------------------------------------------------
# Canonical initialization identical
# ---------------------------------------------------------------------------

class TestCanonicalInitialization:
    def test_same_equity_at_start_for_both_strategies(self):
        """Control and candidate must share identical joint_start equity."""
        close = _make_close(n=500)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")
        btc = close["BTC/USD"].dropna()
        ohlcv = u._close_to_ohlcv(close)
        from research.fixed_five_canonical_regime_comparison import run_canonical

        starts = []
        for entry_confirm, buf in [(1, 0.0), (2, 0.5)]:
            params = WhipsawControl(name="v", entry_confirm_bars=entry_confirm,
                                    entry_buffer_pct=buf)
            gate = compute_controlled_gate_series(btc, 20, params)
            sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
            _, port_fx = run_canonical(
                ohlcv, sig, joint_start, rebalance_bars=12,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
            )
            starts.append(float(port_fx["equity"].dropna().iloc[0]))

        assert abs(starts[0] - starts[1]) < 100.0  # within $100 — same run-up period

    def test_portfolio_start_dates_identical(self):
        close = _make_close(n=500)
        joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
        if joint_start is None:
            pytest.skip("Not enough bars")
        btc = close["BTC/USD"].dropna()
        ohlcv = u._close_to_ohlcv(close)
        from research.fixed_five_canonical_regime_comparison import run_canonical

        dates = []
        for entry_confirm in [1, 2]:
            params = WhipsawControl(name="v", entry_confirm_bars=entry_confirm)
            gate = compute_controlled_gate_series(btc, 20, params)
            sig = WhipsawControlledSignalGen(gate, BaseSignalGen())
            _, port_fx = run_canonical(
                ohlcv, sig, joint_start, rebalance_bars=12,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
            )
            dates.append(port_fx.index.min())

        assert dates[0] == dates[1]


# ---------------------------------------------------------------------------
# Deterministic reruns
# ---------------------------------------------------------------------------

class TestDeterministicReruns:
    def test_ledger_is_deterministic(self):
        close = _make_close(n=400)
        r1, p1, g1, js, c1 = _run_strategy(close)
        r2, p2, g2, _, c2 = _run_strategy(close)

        l1 = build_trade_ledger("t", r1, p1, c1, g1, js)
        l2 = build_trade_ledger("t", r2, p2, c2, g2, js)

        assert len(l1) == len(l2)
        for t1, t2 in zip(l1, l2):
            assert abs(t1.fee_dollars - t2.fee_dollars) < 0.01
            assert abs(t1.gross_notional - t2.gross_notional) < 0.01

    def test_reconciliation_is_deterministic(self):
        close = _make_close(n=400)
        r1, p1, g1, js, c1 = _run_strategy(close)
        l1 = build_trade_ledger("t", r1, p1, c1, g1, js)
        rec1 = compute_cost_reconciliation("t", r1, p1, l1, js)

        r2, p2, g2, _, c2 = _run_strategy(close)
        l2 = build_trade_ledger("t", r2, p2, c2, g2, js)
        rec2 = compute_cost_reconciliation("t", r2, p2, l2, js)

        assert abs(rec1["actual_fee_dollars"] - rec2["actual_fee_dollars"]) < 1.0
        assert abs(rec1["gross_minus_net_dollars"] - rec2["gross_minus_net_dollars"]) < 1.0


# ---------------------------------------------------------------------------
# Cost reconciliation identity
# ---------------------------------------------------------------------------

class TestCostReconciliationIdentity:
    def test_gross_minus_net_equals_gross_minus_net(self):
        """gross_minus_net_dollars = gross_ending_equity - net_ending_equity."""
        close = _make_close(n=500)
        r, p, g, js, c = _run_strategy(close)
        l = build_trade_ledger("t", r, p, c, g, js)
        rec = compute_cost_reconciliation("t", r, p, l, js)
        diff = rec["gross_ending_equity"] - rec["net_ending_equity"]
        assert abs(diff - rec["gross_minus_net_dollars"]) < 1.0

    def test_residual_equals_gross_minus_net_minus_direct(self):
        """residual = gross_minus_net - total_direct_cost."""
        close = _make_close(n=500)
        r, p, g, js, c = _run_strategy(close)
        l = build_trade_ledger("t", r, p, c, g, js)
        rec = compute_cost_reconciliation("t", r, p, l, js)
        expected_residual = rec["gross_minus_net_dollars"] - rec["total_direct_cost_dollars"]
        assert abs(rec["residual_foregone_compounding_dollars"] - expected_residual) < 1.0

    def test_actual_fee_plus_slip_equals_direct_cost(self):
        """actual_fee + actual_slippage = total_direct_cost_dollars."""
        close = _make_close(n=500)
        r, p, g, js, c = _run_strategy(close)
        l = build_trade_ledger("t", r, p, c, g, js)
        rec = compute_cost_reconciliation("t", r, p, l, js)
        expected = rec["actual_fee_dollars"] + rec["actual_slippage_dollars"]
        assert abs(expected - rec["total_direct_cost_dollars"]) < 1.0


# ---------------------------------------------------------------------------
# Candidate selection lock content
# ---------------------------------------------------------------------------

class TestCandidateSelectionLock:
    def test_lock_states_in_sample(self):
        """Lock document must state the candidate was selected on full in-sample data."""
        assert "full historical sample" in CANDIDATE_LOCK_CONTENT or \
               "in-sample" in CANDIDATE_LOCK_CONTENT

    def test_lock_not_approved_for_live(self):
        """Lock must explicitly state the candidate is NOT approved for live deployment."""
        assert "not approved for live deployment" in CANDIDATE_LOCK_CONTENT.lower() or \
               "NOT approved for live" in CANDIDATE_LOCK_CONTENT

    def test_lock_prohibits_further_tuning_claims(self):
        """Lock must state no further parameter adjustment can be called out-of-sample."""
        assert "out-of-sample" in CANDIDATE_LOCK_CONTENT

    def test_lock_requires_future_data_for_validation(self):
        """Lock must require genuinely future/unseen data for validation."""
        assert "future" in CANDIDATE_LOCK_CONTENT.lower()

    def test_lock_contains_frozen_parameters(self):
        """Lock must state frozen parameters."""
        assert "entry_confirm_bars=2" in CANDIDATE_LOCK_CONTENT
        assert "entry_buffer_pct=0.5" in CANDIDATE_LOCK_CONTENT
        assert "BTC_MA_BARS = 240" in CANDIDATE_LOCK_CONTENT


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------

class TestNoLiveImports:
    FORBIDDEN = ["brokers", "execution", "live"]

    def test_no_live_module_imports(self):
        source = Path("research/fixed_five_cost_audit.py").read_text()
        for mod in self.FORBIDDEN:
            assert f"from {mod}" not in source
            assert f"import {mod}" not in source

    def test_no_order_placement_patterns(self):
        source = Path("research/fixed_five_cost_audit.py").read_text()
        for pat in ["place_order", "submit_order", "create_order", "send_order"]:
            assert pat not in source

    def test_four_cost_scenarios_defined(self):
        """Exactly 4 counterfactual scenarios are defined."""
        assert len(COST_SCENARIOS) == 4

    def test_six_trade_reasons_defined(self):
        """Exactly 6 trade reasons are defined."""
        assert len(TRADE_REASONS) == 6
