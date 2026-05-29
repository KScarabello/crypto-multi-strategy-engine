"""Tests for exploratory BTC crash-filter re-entry research."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from research.btc_crash_filter_reentry_research import (
    BtcCrashFilterReentryResearchConfig,
    CrashFilterReentryCandidateSpec,
    ReentryRuleSpec,
    _base_filter_map,
    _build_base_long_cash_target_positions,
    _build_reentry_condition,
    _reentry_ma_reclaim,
    _reentry_max_forced_duration,
    _reentry_rebound_from_low,
    _reentry_short_momentum_recovery,
    run_btc_crash_filter_reentry_research,
    run_single_crash_filter_reentry_candidate,
)


def _toy_close(periods: int = 1500) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="4h", tz="UTC")
    up = [100.0 + i * 0.22 for i in range(periods // 3)]
    down = [up[-1] - i * 0.35 for i in range(periods // 3)]
    rebound = [down[-1] + i * 0.42 for i in range(periods - 2 * (periods // 3))]
    return pd.DataFrame({"BTC/USD": up + down + rebound}, index=index, dtype=float)


def test_positions_are_only_zero_or_one() -> None:
    close = _toy_close()
    base_filter = _base_filter_map()["combo_drawdown_severe"]
    candidate = CrashFilterReentryCandidateSpec(
        name="test",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="short_momentum_recovery", short_momentum_recovery=True),
    )

    artifacts = run_single_crash_filter_reentry_candidate(
        close_prices=close,
        bars_per_year=2190,
        base_filter=base_filter,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=12,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert set(artifacts.signals["position"].dropna().astype(float).unique().tolist()).issubset({0.0, 1.0})


def test_reentry_only_overrides_when_base_is_long() -> None:
    close = _toy_close()
    base_filter = _base_filter_map()["combo_all"]
    candidate = CrashFilterReentryCandidateSpec(
        name="test",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="combined", short_momentum_recovery=True, rebound_window=60, rebound_threshold=0.10, ma_window=60, max_forced_cash_bars=36),
    )

    artifacts = run_single_crash_filter_reentry_candidate(
        close_prices=close,
        bars_per_year=2190,
        base_filter=base_filter,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=12,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    sig = artifacts.signals

    assert (~sig.loc[sig["forced_cash_overridden_by_reentry"], "base_target"].eq(1.0)).sum() == 0


def test_reentry_does_not_create_long_when_base_cash() -> None:
    close = _toy_close()
    base_filter = _base_filter_map()["combo_drawdown_severe"]
    candidate = CrashFilterReentryCandidateSpec(
        name="test",
        base_filter_name="combo_drawdown_severe",
        reentry=ReentryRuleSpec(reentry_type="combined", short_momentum_recovery=True, rebound_window=60, rebound_threshold=0.10, ma_window=60, max_forced_cash_bars=36),
    )

    artifacts = run_single_crash_filter_reentry_candidate(
        close_prices=close,
        bars_per_year=2190,
        base_filter=base_filter,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=12,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    sig = artifacts.signals
    base_cash = sig["base_target"] == 0.0
    assert ((sig.loc[base_cash, "target_position"] <= 0.0)).all()


def test_short_momentum_recovery_reentry_triggers() -> None:
    close = _toy_close(periods=700)
    cond = _reentry_short_momentum_recovery(close, short_lookback_bars=60)
    assert bool(cond.any())


def test_rebound_from_low_reentry_triggers() -> None:
    close = _toy_close(periods=700)
    cond = _reentry_rebound_from_low(close, rebound_window=60, rebound_threshold=0.08)
    assert bool(cond.any())


def test_moving_average_reclaim_reentry_triggers() -> None:
    close = _toy_close(periods=700)
    cond = _reentry_ma_reclaim(close, ma_window=60)
    assert bool(cond.any())


def test_max_forced_cash_duration_reentry_triggers() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    eligible = pd.Series(False, index=idx)
    eligible.iloc[10:80] = True
    cond = _reentry_max_forced_duration(eligible_forced=eligible, max_forced_cash_bars=24)
    assert bool(cond.any())
    assert bool(cond.iloc[33])


def test_positions_lagged_and_costs_applied() -> None:
    close = _toy_close()
    base_filter = _base_filter_map()["combo_all"]
    candidate = CrashFilterReentryCandidateSpec(
        name="test",
        base_filter_name="combo_all",
        reentry=ReentryRuleSpec(reentry_type="short_momentum_recovery", short_momentum_recovery=True),
    )
    artifacts = run_single_crash_filter_reentry_candidate(
        close_prices=close,
        bars_per_year=2190,
        base_filter=base_filter,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )
    sig = artifacts.signals

    pd.testing.assert_series_equal(sig["position"], sig["target_position"].shift(1).fillna(0.0), check_names=False)
    expected_cost = sig["position_change"] * (15.0 / 10_000.0)
    assert (sig["transaction_cost"] - expected_cost).abs().max() < 1e-12


def test_output_columns_exist() -> None:
    close = _toy_close(periods=1800)
    config = BtcCrashFilterReentryResearchConfig()
    summary, by_year, events = run_btc_crash_filter_reentry_research(
        config=config,
        close_prices=close,
        save_csv=False,
    )

    summary_required = {
        "candidate_name",
        "base_filter_name",
        "reentry_type",
        "reentry_parameters",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_month_return",
        "number_of_trades",
        "percent_time_long",
        "percent_time_cash",
        "forced_cash_periods",
        "percent_time_forced_cash",
        "reentry_events",
        "percent_forced_cash_overridden_by_reentry",
        "total_transaction_costs",
        "vs_btc_buy_hold_total_return",
        "vs_btc_buy_hold_sharpe",
        "vs_btc_buy_hold_max_drawdown",
        "vs_btc_long_cash_total_return",
        "vs_btc_long_cash_sharpe",
        "vs_btc_long_cash_max_drawdown",
    }
    by_year_required = {
        "year",
        "candidate_name",
        "strategy_return",
        "btc_buy_hold_return",
        "btc_long_cash_return",
        "excess_return_vs_btc",
        "excess_return_vs_long_cash",
        "sharpe",
        "max_drawdown",
        "worst_month_return",
        "percent_time_long",
        "percent_time_cash",
        "percent_time_forced_cash",
        "reentry_events",
        "number_of_forced_cash_events",
    }
    event_required = {
        "candidate_name",
        "event_start_time",
        "event_end_time",
        "trigger_type",
        "reentry_trigger_type",
        "btc_return_during_event",
        "baseline_long_cash_return_during_event",
        "strategy_return_during_event",
        "avoided_loss_or_missed_gain_vs_baseline",
        "holding_period_bars",
        "was_reentered_early",
    }
    assert summary_required.issubset(set(summary.columns))
    assert by_year_required.issubset(set(by_year.columns))
    assert event_required.issubset(set(events.columns))


def test_runner_does_not_import_live_modules() -> None:
    runner_path = Path("research/run_btc_crash_filter_reentry_research.py")
    module_ast = ast.parse(runner_path.read_text(encoding="utf-8"))

    imported_modules: list[str] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert all(not mod.startswith("live") for mod in imported_modules)
    assert all("live." not in mod for mod in imported_modules)
