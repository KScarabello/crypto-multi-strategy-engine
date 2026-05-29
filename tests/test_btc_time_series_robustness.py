"""Tests for BTC time-series yearly robustness report."""

from __future__ import annotations

import pandas as pd
import pytest

from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_experiment
from research.btc_time_series_robustness import run_btc_time_series_robustness


def _toy_close_cross_year() -> pd.DataFrame:
    index = pd.date_range("2023-07-01", periods=3200, freq="4h", tz="UTC")
    prices = [100.0 + i * 0.1 for i in range(1600)] + [260.0 - i * 0.08 for i in range(1600)]
    return pd.DataFrame({"BTC/USD": prices}, index=index, dtype=float)


def test_robustness_contains_expected_strategies_and_years() -> None:
    close = _toy_close_cross_year()
    report = run_btc_time_series_robustness(close_prices=close, save_csv=False)

    expected_strategies = {
        "btc_buy_and_hold",
        "btc_ts_candidate_60_240_12",
        "btc_ts_baseline_42_180_6",
    }
    assert expected_strategies.issubset(set(report["strategy"]))

    years_in_data = {str(y) for y in sorted(close.index.year.unique().tolist())}
    years_in_report = set(report.loc[report["year"] != "FULL", "year"].astype(str))
    assert years_in_data.issubset(years_in_report)
    assert "FULL" in set(report["year"].astype(str))


def test_robustness_output_required_columns() -> None:
    close = _toy_close_cross_year()
    report = run_btc_time_series_robustness(close_prices=close, save_csv=False)

    required = {
        "year",
        "strategy",
        "total_return",
        "annual_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "percent_time_invested",
    }
    assert required.issubset(set(report.columns))


def test_robustness_full_period_matches_direct_experiment_baseline() -> None:
    close = _toy_close_cross_year()
    baseline = BtcTimeSeriesConfig(short_lookback_bars=42, medium_lookback_bars=180, rebalance_every_bars=6)

    direct = run_btc_time_series_experiment(config=baseline, close_prices=close)
    direct_ts = direct.loc[direct["strategy"] == "btc_time_series_momentum"].iloc[0]

    report = run_btc_time_series_robustness(close_prices=close, save_csv=False)
    robust_full = report.loc[
        (report["year"] == "FULL") & (report["strategy"] == "btc_ts_baseline_42_180_6")
    ].iloc[0]

    assert float(robust_full["total_return"]) == pytest.approx(float(direct_ts["total_return"]), rel=1e-9, abs=1e-9)
    assert float(robust_full["cagr"]) == pytest.approx(float(direct_ts["cagr"]), rel=1e-9, abs=1e-9)
    assert float(robust_full["sharpe"]) == pytest.approx(float(direct_ts["sharpe"]), rel=1e-9, abs=1e-9)
    assert float(robust_full["max_drawdown"]) == pytest.approx(float(direct_ts["max_drawdown"]), rel=1e-9, abs=1e-9)
