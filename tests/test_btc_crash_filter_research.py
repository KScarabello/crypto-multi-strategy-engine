"""Tests for exploratory BTC crash-filter overlay research module."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from research.btc_crash_filter_research import (
    BtcCrashFilterResearchConfig,
    CrashFilterCandidateSpec,
    DrawdownFilterSpec,
    SevereReturnFilterSpec,
    VolatilityFilterSpec,
    _build_base_long_cash_target_positions,
    _drawdown_force_cash,
    _severe_return_force_cash,
    _volatility_force_cash,
    run_btc_crash_filter_research,
    run_single_crash_filter_candidate,
)


def _toy_close(periods: int = 1200) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="4h", tz="UTC")
    up = [100.0 + i * 0.25 for i in range(periods // 3)]
    down = [up[-1] - i * 0.35 for i in range(periods // 3)]
    rebound = [down[-1] + i * 0.45 for i in range(periods - 2 * (periods // 3))]
    return pd.DataFrame({"BTC/USD": up + down + rebound}, index=index, dtype=float)


def test_positions_are_only_zero_or_one() -> None:
    close = _toy_close()
    candidate = CrashFilterCandidateSpec(name="baseline")
    artifacts = run_single_crash_filter_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=12,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    unique_values = set(artifacts.signals["position"].dropna().astype(float).unique().tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_filters_can_only_force_long_to_cash_not_create_new_longs() -> None:
    close = _toy_close()
    candidate = CrashFilterCandidateSpec(
        name="dd",
        drawdown_filter=DrawdownFilterSpec(rolling_high_window=180, drawdown_threshold=-0.15),
    )
    artifacts = run_single_crash_filter_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=12,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    signals = artifacts.signals
    assert ((signals["target_position"] <= signals["base_target"]) | (signals["base_target"] == 0.0)).all()


def test_positions_are_lagged_before_returns() -> None:
    close = _toy_close()
    candidate = CrashFilterCandidateSpec(name="baseline")
    artifacts = run_single_crash_filter_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate=candidate,
        base_short_lookback=20,
        base_medium_lookback=40,
        rebalance_every_bars=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    signals = artifacts.signals
    pd.testing.assert_series_equal(
        signals["position"],
        signals["target_position"].shift(1).fillna(0.0),
        check_names=False,
    )


def test_cash_returns_zero_before_costs() -> None:
    close = _toy_close()
    candidate = CrashFilterCandidateSpec(name="baseline")
    artifacts = run_single_crash_filter_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate=candidate,
        base_short_lookback=60,
        base_medium_lookback=240,
        rebalance_every_bars=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    signals = artifacts.signals
    cash_mask = (signals["position"] == 0.0) & (signals["position_change"] < 1e-12)
    if cash_mask.any():
        assert signals.loc[cash_mask, "gross_return"].abs().max() < 1e-12


def test_transaction_costs_applied_when_position_changes() -> None:
    close = _toy_close()
    candidate = CrashFilterCandidateSpec(name="baseline")
    artifacts = run_single_crash_filter_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate=candidate,
        base_short_lookback=20,
        base_medium_lookback=40,
        rebalance_every_bars=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    signals = artifacts.signals
    expected_cost = signals["position_change"] * (15.0 / 10_000.0)
    assert (signals["transaction_cost"] - expected_cost).abs().max() < 1e-12


def test_drawdown_filter_triggers_when_breached() -> None:
    close = _toy_close(periods=300)
    force = _drawdown_force_cash(close, DrawdownFilterSpec(rolling_high_window=60, drawdown_threshold=-0.10))
    assert bool(force.any())


def test_volatility_filter_requires_vol_spike_and_negative_short_momentum() -> None:
    close = _toy_close(periods=600)
    force = _volatility_force_cash(
        close=close,
        short_lookback_bars=60,
        spec=VolatilityFilterSpec(vol_window=60, vol_multiplier=1.5),
    )

    px = close.iloc[:, 0].astype(float)
    btc_returns = px.pct_change().fillna(0.0)
    realized_vol = btc_returns.rolling(window=60, min_periods=60).std(ddof=0)
    rolling_median_vol = realized_vol.rolling(window=60, min_periods=1).median()
    short_momentum = px.pct_change(60)

    triggered = force.fillna(False)
    if bool(triggered.any()):
        assert (realized_vol.loc[triggered] > 1.5 * rolling_median_vol.loc[triggered]).all()
        assert (short_momentum.loc[triggered] < 0.0).all()


def test_severe_return_filter_applies_cooldown() -> None:
    close = _toy_close(periods=400)
    force = _severe_return_force_cash(
        close=close,
        spec=SevereReturnFilterSpec(recent_return_window=12, severe_return_threshold=-0.03, cooldown_bars=24),
    )
    assert bool(force.any())


def test_forced_cash_percentage_reported() -> None:
    close = _toy_close(periods=1600)
    config = BtcCrashFilterResearchConfig(
        candidates=(
            CrashFilterCandidateSpec(name="baseline"),
            CrashFilterCandidateSpec(
                name="dd",
                drawdown_filter=DrawdownFilterSpec(rolling_high_window=180, drawdown_threshold=-0.15),
            ),
        )
    )

    summary, _, _ = run_btc_crash_filter_research(config=config, close_prices=close, save_csv=False)
    assert "percent_time_forced_cash" in summary.columns


def test_output_columns_exist() -> None:
    close = _toy_close(periods=1800)
    config = BtcCrashFilterResearchConfig()

    summary, by_year, events = run_btc_crash_filter_research(
        config=config,
        close_prices=close,
        save_csv=False,
    )

    summary_required = {
        "candidate_name",
        "base_short_lookback",
        "base_medium_lookback",
        "rebalance_every",
        "filter_type",
        "filter_parameters",
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
        "number_of_forced_cash_events",
    }
    events_required = {
        "candidate_name",
        "event_start_time",
        "event_end_time",
        "trigger_type",
        "btc_return_during_event",
        "baseline_long_cash_return_during_event",
        "strategy_return_during_event",
        "avoided_loss_or_missed_gain_vs_baseline",
        "holding_period_bars",
    }

    assert summary_required.issubset(set(summary.columns))
    assert by_year_required.issubset(set(by_year.columns))
    assert events_required.issubset(set(events.columns))


def test_runner_does_not_import_live_modules() -> None:
    runner_path = Path("research/run_btc_crash_filter_research.py")
    module_ast = ast.parse(runner_path.read_text(encoding="utf-8"))

    imported_modules: list[str] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert all(not mod.startswith("live") for mod in imported_modules)
    assert all("live." not in mod for mod in imported_modules)
