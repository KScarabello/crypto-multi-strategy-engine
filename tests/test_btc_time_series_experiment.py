"""Tests for BTC-only time-series momentum experiment module."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from research.btc_time_series import (
    BtcTimeSeriesConfig,
    build_btc_ts_momentum_signal_generator,
    build_btc_buy_and_hold_signal_generator,
    calculate_btc_combined_momentum,
    format_btc_experiment_summary,
    run_btc_time_series_experiment,
)
from research.run_expanded_universe_experiment import close_prices_to_ohlcv


def _toy_close(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
    prices = [100.0 + i * 0.3 for i in range(180)] + [154.0 - i * 0.5 for i in range(periods - 180)]
    return pd.DataFrame({"BTC/USD": prices}, index=index, dtype=float)


def test_calculate_btc_combined_momentum_weighted_average() -> None:
    index = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    close = pd.DataFrame({"BTC/USD": [100.0 + i for i in range(220)]}, index=index)

    combined = calculate_btc_combined_momentum(
        close_prices=close,
        short_lookback_bars=42,
        medium_lookback_bars=180,
        short_weight=0.5,
        medium_weight=0.5,
    )

    expected_short = close["BTC/USD"].pct_change(42)
    expected_medium = close["BTC/USD"].pct_change(180)
    expected = 0.5 * expected_short + 0.5 * expected_medium
    pd.testing.assert_series_equal(combined, expected)


def test_combined_momentum_has_no_lookahead_bias() -> None:
    close = _toy_close(periods=320)
    changed = close.copy()
    changed.loc[changed.index[-1], "BTC/USD"] = 1_000_000.0

    base_signal = calculate_btc_combined_momentum(close_prices=close)
    changed_signal = calculate_btc_combined_momentum(close_prices=changed)

    pd.testing.assert_series_equal(base_signal.iloc[:-1], changed_signal.iloc[:-1])


def test_btc_ts_signal_generator_switches_between_btc_and_cash() -> None:
    close = _toy_close(periods=320)
    config = BtcTimeSeriesConfig(
        short_lookback_bars=42,
        medium_lookback_bars=180,
        short_weight=0.5,
        medium_weight=0.5,
    )
    generator = build_btc_ts_momentum_signal_generator(close_prices=close, config=config)

    warmup_ts = close.index[120]
    uptrend_ts = close.index[210]
    downtrend_ts = close.index[-1]

    warmup_weights = generator(close, warmup_ts)
    uptrend_weights = generator(close, uptrend_ts)
    downtrend_weights = generator(close, downtrend_ts)

    assert warmup_weights["BTC/USD"] == pytest.approx(0.0)
    assert uptrend_weights["BTC/USD"] == pytest.approx(1.0)
    assert downtrend_weights["BTC/USD"] == pytest.approx(0.0)


def test_run_btc_time_series_experiment_outputs_required_metrics() -> None:
    close = _toy_close(periods=320)
    config = BtcTimeSeriesConfig(
        short_lookback_bars=42,
        medium_lookback_bars=180,
        rebalance_every_bars=6,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    report = run_btc_time_series_experiment(config=config, close_prices=close)

    assert set(report["strategy"]) == {"btc_buy_and_hold", "btc_time_series_momentum"}
    required_cols = {
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "total_return",
        "num_trades",
        "percent_time_invested",
        "vs_buy_hold_total_return",
    }
    assert required_cols.issubset(set(report.columns))

    ts_row = report.loc[report["strategy"] == "btc_time_series_momentum"].iloc[0]
    assert float(ts_row["percent_time_invested"]) >= 0.0
    assert float(ts_row["percent_time_invested"]) <= 1.0
    assert float(ts_row["num_trades"]) >= 0.0


def test_signal_execution_alignment_uses_one_bar_delay() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    # Construct a series where lookback signals become defined and positive mid-sample.
    close = pd.DataFrame({"BTC/USD": [100, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100]}, index=index)
    config = BtcTimeSeriesConfig(short_lookback_bars=1, medium_lookback_bars=2, rebalance_every_bars=1)

    ohlcv = close_prices_to_ohlcv(close)
    execution = run_backtest(
        ohlcv=ohlcv,
        signal_generator=build_btc_ts_momentum_signal_generator(close_prices=close, config=config),
        initial_capital=10_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        rebalance_every_bars=1,
    )

    assert not execution.rebalance_log.empty
    nonzero = execution.rebalance_log.loc[execution.rebalance_log["turnover"] > 1e-12]
    assert not nonzero.empty
    first_exec = pd.to_datetime(nonzero["execution_timestamp"].iloc[0], utc=True)
    # With lookbacks 1 and 2, first valid signal appears at index[2], then executes next bar.
    assert first_exec == index[3]


def test_buy_and_hold_turnover_cost_is_initial_entry_only() -> None:
    close = _toy_close(periods=320)
    ohlcv = close_prices_to_ohlcv(close)
    execution = run_backtest(
        ohlcv=ohlcv,
        signal_generator=build_btc_buy_and_hold_signal_generator(symbol="BTC/USD"),
        initial_capital=10_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        rebalance_every_bars=1,
    )

    turnover = execution.turnover.reindex(execution.portfolio.index).fillna(0.0)
    assert int((turnover > 1e-12).sum()) == 1
    assert float(turnover.sum()) == pytest.approx(1.0)


def test_buy_and_hold_reporting_uses_effective_rebalance_count() -> None:
    close = _toy_close(periods=320)
    report = run_btc_time_series_experiment(close_prices=close)
    summary = format_btc_experiment_summary(report)

    row = summary.loc[summary["strategy"] == "btc_buy_and_hold"].iloc[0]
    assert int(row["num_rebalances"]) == 1
    assert int(row["num_trades"]) == 1
