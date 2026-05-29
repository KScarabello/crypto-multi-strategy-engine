"""Tests for focused sleeve blend robustness sweep."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import summary_metrics
from research.sleeve_blend_experiment import run_sleeve_backtests
from research.sleeve_blend_robustness import (
    FOCUSED_BLEND_CS_CANDIDATES,
    build_focused_blend_allocation_grid,
    run_sleeve_blend_robustness,
)


def _toy_universe_close(periods: int = 1200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
    data: dict[str, list[float]] = {}
    symbols = [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "ADA/USD",
        "DOGE/USD",
        "LINK/USD",
        "AVAX/USD",
        "LTC/USD",
        "ATOM/USD",
    ]
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 12.0
        slope = 0.12 + idx * 0.015
        cycle = 0.45 + idx * 0.02
        prices = [base + slope * step + cycle * ((step + idx * 7) % 17) for step in range(periods)]
        data[symbol] = prices
    return pd.DataFrame(data, index=index, dtype=float)


def test_focused_allocation_grid_is_correct() -> None:
    assert build_focused_blend_allocation_grid() == (
        (1.0, 0.0),
        (0.95, 0.05),
        (0.9, 0.1),
        (0.85, 0.15),
        (0.8, 0.2),
        (0.75, 0.25),
        (0.7, 0.3),
    )


def test_blend_robustness_required_columns() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_robustness(close_prices=close, save_csv=False)

    required = {
        "year",
        "btc_strategy",
        "cs_strategy",
        "blend_name",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "sleeve_return_correlation",
        "delta_total_return",
        "delta_cagr",
        "delta_sharpe",
        "delta_max_drawdown",
    }
    assert required.issubset(set(report.columns))


def test_blend_robustness_100_0_matches_locked_btc_ts_within_tolerance() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_robustness(close_prices=close, save_csv=False)
    _, bars_per_year, btc_result, _ = run_sleeve_backtests(close_prices=close)
    btc_metrics = summary_metrics(
        equity=btc_result.portfolio["equity"],
        returns=btc_result.portfolio["strategy_return"],
        bars_per_year=bars_per_year,
    )

    row = report.loc[
        (report["cs_strategy"] == FOCUSED_BLEND_CS_CANDIDATES[0].name)
        & ((report["btc_weight"] - 1.0).abs() < 1e-12)
        & (report["cs_weight"].abs() < 1e-12)
    ].iloc[0]

    assert float(row["total_return"]) == pytest.approx(float(btc_metrics["total_return"]), rel=1e-9, abs=1e-9)
    assert float(row["cagr"]) == pytest.approx(float(btc_metrics["cagr"]), rel=1e-9, abs=1e-9)
    assert float(row["sharpe"]) == pytest.approx(float(btc_metrics["sharpe"]), rel=1e-9, abs=1e-9)
    assert float(row["max_drawdown"]) == pytest.approx(float(btc_metrics["max_drawdown"]), rel=1e-9, abs=1e-9)


def test_blend_robustness_deltas_are_relative_to_btc_ts() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_robustness(close_prices=close, save_csv=False)

    baseline_row = report.loc[
        (report["cs_strategy"] == FOCUSED_BLEND_CS_CANDIDATES[0].name)
        & ((report["btc_weight"] - 1.0).abs() < 1e-12)
        & (report["cs_weight"].abs() < 1e-12)
    ].iloc[0]
    assert float(baseline_row["delta_total_return"]) == pytest.approx(0.0, abs=1e-12)
    assert float(baseline_row["delta_cagr"]) == pytest.approx(0.0, abs=1e-12)
    assert float(baseline_row["delta_sharpe"]) == pytest.approx(0.0, abs=1e-12)
    assert float(baseline_row["delta_max_drawdown"]) == pytest.approx(0.0, abs=1e-12)

    comparison_row = report.loc[
        (report["cs_strategy"] == FOCUSED_BLEND_CS_CANDIDATES[0].name)
        & ((report["btc_weight"] - 0.9).abs() < 1e-12)
        & ((report["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]
    assert float(comparison_row["delta_total_return"]) == pytest.approx(
        float(comparison_row["total_return"]) - float(baseline_row["total_return"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(comparison_row["delta_cagr"]) == pytest.approx(
        float(comparison_row["cagr"]) - float(baseline_row["cagr"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(comparison_row["delta_sharpe"]) == pytest.approx(
        float(comparison_row["sharpe"]) - float(baseline_row["sharpe"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(comparison_row["delta_max_drawdown"]) == pytest.approx(
        float(comparison_row["max_drawdown"]) - float(baseline_row["max_drawdown"]),
        rel=1e-9,
        abs=1e-9,
    )