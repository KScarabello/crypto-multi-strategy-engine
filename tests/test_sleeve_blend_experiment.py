"""Tests for BTC plus CS sleeve blend experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import summary_metrics
from research.sleeve_blend_experiment import (
    DEFAULT_BLEND_CS_CANDIDATES,
    align_sleeve_return_streams,
    build_blend_allocation_grid,
    run_sleeve_backtests,
    run_sleeve_blend_experiment,
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


def test_align_sleeve_return_streams_uses_common_timestamps() -> None:
    index_a = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    index_b = pd.date_range("2024-01-01 08:00:00", periods=5, freq="4h", tz="UTC")
    btc_returns = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], index=index_a)
    cs_returns = pd.Series([0.0, -0.01, -0.02, -0.03, -0.04], index=index_b)

    aligned = align_sleeve_return_streams(btc_returns=btc_returns, cs_returns=cs_returns)

    assert list(aligned.index) == list(index_a.intersection(index_b))
    assert list(aligned.columns) == ["btc_sleeve_return", "cs_sleeve_return"]


def test_blend_allocations_sum_to_one() -> None:
    allocations = build_blend_allocation_grid()
    assert all(abs(btc_weight + cs_weight - 1.0) < 1e-12 for btc_weight, cs_weight in allocations)

    with pytest.raises(ValueError, match="sum to 1.0"):
        build_blend_allocation_grid(((0.8, 0.3),))


def test_blend_output_required_columns() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_experiment(close_prices=close, save_csv=False)

    required = {
        "year",
        "btc_strategy",
        "cs_strategy",
        "blend_name",
        "btc_weight",
        "cs_weight",
        "total_return",
        "annual_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "sleeve_return_correlation",
    }
    assert required.issubset(set(report.columns))


def test_blend_100_0_matches_locked_btc_ts_within_tolerance() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_experiment(close_prices=close, save_csv=False)
    _, bars_per_year, btc_result, _ = run_sleeve_backtests(close_prices=close)
    btc_metrics = summary_metrics(
        equity=btc_result.portfolio["equity"],
        returns=btc_result.portfolio["strategy_return"],
        bars_per_year=bars_per_year,
    )

    row = report.loc[
        (report["year"] == "FULL")
        & (report["cs_strategy"] == DEFAULT_BLEND_CS_CANDIDATES[0].name)
        & ((report["btc_weight"] - 1.0).abs() < 1e-12)
        & (report["cs_weight"].abs() < 1e-12)
    ].iloc[0]

    assert float(row["total_return"]) == pytest.approx(float(btc_metrics["total_return"]), rel=1e-9, abs=1e-9)
    assert float(row["cagr"]) == pytest.approx(float(btc_metrics["cagr"]), rel=1e-9, abs=1e-9)
    assert float(row["sharpe"]) == pytest.approx(float(btc_metrics["sharpe"]), rel=1e-9, abs=1e-9)
    assert float(row["max_drawdown"]) == pytest.approx(float(btc_metrics["max_drawdown"]), rel=1e-9, abs=1e-9)


def test_blend_0_100_matches_selected_cs_candidate_within_tolerance() -> None:
    close = _toy_universe_close()
    report = run_sleeve_blend_experiment(close_prices=close, save_csv=False)
    _, bars_per_year, _, cs_results = run_sleeve_backtests(close_prices=close)
    candidate_name = DEFAULT_BLEND_CS_CANDIDATES[1].name
    cs_result = cs_results[candidate_name]
    cs_metrics = summary_metrics(
        equity=cs_result.portfolio["equity"],
        returns=cs_result.portfolio["strategy_return"],
        bars_per_year=bars_per_year,
    )

    row = report.loc[
        (report["year"] == "FULL")
        & (report["cs_strategy"] == candidate_name)
        & (report["btc_weight"].abs() < 1e-12)
        & ((report["cs_weight"] - 1.0).abs() < 1e-12)
    ].iloc[0]

    assert float(row["total_return"]) == pytest.approx(float(cs_metrics["total_return"]), rel=1e-9, abs=1e-9)
    assert float(row["cagr"]) == pytest.approx(float(cs_metrics["cagr"]), rel=1e-9, abs=1e-9)
    assert float(row["sharpe"]) == pytest.approx(float(cs_metrics["sharpe"]), rel=1e-9, abs=1e-9)
    assert float(row["max_drawdown"]) == pytest.approx(float(cs_metrics["max_drawdown"]), rel=1e-9, abs=1e-9)