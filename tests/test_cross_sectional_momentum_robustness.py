"""Tests for CS momentum yearly robustness report."""

from __future__ import annotations

import pandas as pd
import pytest

from research.cross_sectional_momentum_robustness import run_cross_sectional_momentum_robustness


def _toy_close_cross_year() -> pd.DataFrame:
    index = pd.date_range("2023-07-01", periods=3200, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "BTC/USD": [100.0 + i * 0.08 for i in range(3200)],
            "ETH/USD": [80.0 + i * 0.06 for i in range(3200)],
            "SOL/USD": [40.0 + i * 0.10 for i in range(3200)],
            "XRP/USD": [20.0 + i * 0.03 for i in range(3200)],
            "AVAX/USD": [30.0 + i * 0.07 for i in range(3200)],
            "ADA/USD": [10.0 + i * 0.02 for i in range(3200)],
        },
        index=index,
        dtype=float,
    )


def test_cs_robustness_contains_expected_configs() -> None:
    close = _toy_close_cross_year()
    report = run_cross_sectional_momentum_robustness(
        close_prices=close,
        symbols=tuple(close.columns),
        save_csv=False,
    )

    expected = {
        "equal_weight_dynamic_benchmark",
        "cs_42_180_top3_reb6",
        "cs_180_top5_reb12",
        "cs_42_180_top3_reb12",
        "cs_180_top8_reb42",
        "cs_42_180_top5_reb42",
    }
    assert expected.issubset(set(report["strategy"]))


def test_cs_robustness_required_columns() -> None:
    close = _toy_close_cross_year()
    report = run_cross_sectional_momentum_robustness(
        close_prices=close,
        symbols=tuple(close.columns),
        save_csv=False,
    )
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
        "turnover",
        "percent_time_invested",
        "worst_drawdown_start",
        "worst_drawdown_end",
    }
    assert required.issubset(set(report.columns))


def test_cs_robustness_full_period_matches_existing_best_config_tolerance() -> None:
    report = run_cross_sectional_momentum_robustness(save_csv=False)
    full_row = report.loc[
        (report["year"] == "FULL") & (report["strategy"] == "cs_42_180_top3_reb6")
    ].iloc[0]

    expected_total_return = 52.84909076496341
    expected_cagr = 0.8678777848375017
    expected_sharpe = 1.1256930185920193
    expected_max_drawdown = -0.8463819300859621

    assert float(full_row["total_return"]) == pytest.approx(expected_total_return, rel=1e-9, abs=1e-9)
    assert float(full_row["cagr"]) == pytest.approx(expected_cagr, rel=1e-9, abs=1e-9)
    assert float(full_row["sharpe"]) == pytest.approx(expected_sharpe, rel=1e-9, abs=1e-9)
    assert float(full_row["max_drawdown"]) == pytest.approx(expected_max_drawdown, rel=1e-9, abs=1e-9)
