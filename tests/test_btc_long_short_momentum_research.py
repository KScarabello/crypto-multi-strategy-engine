"""Tests for exploratory BTC long/short momentum research module."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from research.btc_long_short_momentum_research import (
    BtcLongShortResearchConfig,
    _build_target_positions,
    run_btc_long_short_momentum_research,
    run_single_long_short_candidate,
)


def _toy_close(periods: int = 1200) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="4h", tz="UTC")
    up = [100.0 + i * 0.25 for i in range(periods // 3)]
    down = [up[-1] - i * 0.35 for i in range(periods // 3)]
    rebound = [down[-1] + i * 0.45 for i in range(periods - 2 * (periods // 3))]
    return pd.DataFrame({"BTC/USD": up + down + rebound}, index=index, dtype=float)


def test_positions_only_use_minus_one_zero_plus_one() -> None:
    close = _toy_close()
    artifacts = run_single_long_short_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate_name="test",
        short_lookback_bars=60,
        medium_lookback_bars=240,
        rebalance_every_bars=12,
        short_threshold=0.0,
        medium_threshold=0.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    unique_values = set(artifacts.signals["position"].dropna().astype(float).unique().tolist())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


def test_positions_are_lagged_before_returns() -> None:
    close = _toy_close()
    artifacts = run_single_long_short_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate_name="test",
        short_lookback_bars=20,
        medium_lookback_bars=40,
        rebalance_every_bars=1,
        short_threshold=0.0,
        medium_threshold=0.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    signals = artifacts.signals
    pd.testing.assert_series_equal(
        signals["position"],
        signals["target_position"].shift(1).fillna(0.0),
        check_names=False,
    )


def test_short_long_cash_return_mechanics_before_costs() -> None:
    close = _toy_close()
    artifacts = run_single_long_short_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate_name="test",
        short_lookback_bars=60,
        medium_lookback_bars=240,
        rebalance_every_bars=1,
        short_threshold=0.0,
        medium_threshold=0.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    signals = artifacts.signals

    long_mask = (signals["position"] == 1.0) & (signals["position_change"] < 1e-12)
    short_mask = (signals["position"] == -1.0) & (signals["position_change"] < 1e-12)
    cash_mask = (signals["position"] == 0.0) & (signals["position_change"] < 1e-12)

    if long_mask.any():
        assert (signals.loc[long_mask, "gross_return"] - signals.loc[long_mask, "btc_return"]).abs().max() < 1e-12
    if short_mask.any():
        assert (signals.loc[short_mask, "gross_return"] + signals.loc[short_mask, "btc_return"]).abs().max() < 1e-12
    if cash_mask.any():
        assert signals.loc[cash_mask, "gross_return"].abs().max() < 1e-12


def test_transaction_costs_apply_when_position_changes() -> None:
    close = _toy_close()
    artifacts = run_single_long_short_candidate(
        close_prices=close,
        bars_per_year=2190,
        candidate_name="test",
        short_lookback_bars=20,
        medium_lookback_bars=40,
        rebalance_every_bars=1,
        short_threshold=0.0,
        medium_threshold=0.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    signals = artifacts.signals
    per_turnover_cost = 15.0 / 10_000.0
    expected_cost = signals["position_change"] * per_turnover_cost
    assert (signals["transaction_cost"] - expected_cost).abs().max() < 1e-12
    assert float(signals.loc[signals["position_change"] > 1e-12, "transaction_cost"].sum()) > 0.0


def test_conservative_thresholds_have_fewer_or_equal_short_periods() -> None:
    close = _toy_close()

    symmetric = _build_target_positions(
        close=close,
        short_lookback_bars=60,
        medium_lookback_bars=240,
        rebalance_every_bars=12,
        short_threshold=0.0,
        medium_threshold=0.0,
    )
    conservative = _build_target_positions(
        close=close,
        short_lookback_bars=60,
        medium_lookback_bars=240,
        rebalance_every_bars=12,
        short_threshold=-0.08,
        medium_threshold=-0.15,
    )

    assert int((conservative < 0.0).sum()) <= int((symmetric < 0.0).sum())


def test_output_columns_exist() -> None:
    close = _toy_close(periods=1800)
    config = BtcLongShortResearchConfig(
        short_lookback_bars=60,
        medium_lookback_bars=240,
        rebalance_every_bars=12,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    summary, by_year, short_trades = run_btc_long_short_momentum_research(
        config=config,
        close_prices=close,
        save_csv=False,
    )

    summary_required = {
        "candidate_name",
        "short_lookback",
        "medium_lookback",
        "rebalance_every",
        "short_threshold",
        "medium_threshold",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_month_return",
        "number_of_trades",
        "number_of_long_entries",
        "number_of_short_entries",
        "number_of_cash_periods",
        "percent_time_long",
        "percent_time_short",
        "percent_time_cash",
        "average_long_trade_return",
        "average_short_trade_return",
        "short_trade_win_rate",
        "worst_short_trade_return",
        "best_short_trade_return",
        "total_transaction_costs",
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
        "percent_time_long",
        "percent_time_short",
        "percent_time_cash",
        "short_side_pnl_contribution",
        "long_side_pnl_contribution",
    }
    short_trade_required = {
        "candidate_name",
        "short_entry_time",
        "short_exit_time",
        "short_trade_return",
        "btc_return_during_short",
        "holding_period_bars",
        "exit_reason",
    }

    assert summary_required.issubset(set(summary.columns))
    assert by_year_required.issubset(set(by_year.columns))
    assert short_trade_required.issubset(set(short_trades.columns))


def test_runner_does_not_import_live_modules() -> None:
    runner_path = Path("research/run_btc_long_short_momentum_research.py")
    module_ast = ast.parse(runner_path.read_text(encoding="utf-8"))

    imported_modules: list[str] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert all(not mod.startswith("live") for mod in imported_modules)
    assert all("live." not in mod for mod in imported_modules)
