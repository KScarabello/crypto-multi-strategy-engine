"""Tests for yearly robustness report of selected sleeve blends."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import summary_metrics
from research.sleeve_blend_experiment import align_sleeve_return_streams, run_sleeve_backtests
from research.sleeve_blend_yearly_robustness import (
    YEARLY_BLEND_CS_CANDIDATE,
    YEARLY_BLEND_ALLOCATIONS,
    run_sleeve_blend_yearly_robustness,
)


def _toy_universe_close_cross_year(periods: int = 3200) -> pd.DataFrame:
    index = pd.date_range("2023-07-01", periods=periods, freq="4h", tz="UTC")
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
    split = periods // 2
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 8.0
        up = [base + (0.12 + idx * 0.01) * step for step in range(split)]
        down = [up[-1] - (0.09 + idx * 0.008) * step for step in range(periods - split)]
        data[symbol] = up + down
    return pd.DataFrame(data, index=index, dtype=float)


def test_yearly_robustness_contains_requested_blends() -> None:
    close = _toy_universe_close_cross_year()
    report = run_sleeve_blend_yearly_robustness(close_prices=close, save_csv=False)

    expected_allocations = {tuple(map(float, weights)) for weights in YEARLY_BLEND_ALLOCATIONS}
    actual_allocations = {
        (float(row.btc_weight), float(row.cs_weight))
        for row in report.loc[report["year"] == "FULL", ["btc_weight", "cs_weight"]].itertuples(index=False)
    }
    assert expected_allocations == actual_allocations


def test_yearly_robustness_required_columns() -> None:
    close = _toy_universe_close_cross_year()
    report = run_sleeve_blend_yearly_robustness(close_prices=close, save_csv=False)

    required = {
        "year",
        "btc_weight",
        "cs_weight",
        "cs_candidate",
        "total_return",
        "annual_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_month_return",
        "ending_equity",
        "growth_of_1",
    }
    assert required.issubset(set(report.columns))


def test_yearly_robustness_100_0_matches_btc_ts_yearly_and_full_within_tolerance() -> None:
    close = _toy_universe_close_cross_year()
    report = run_sleeve_blend_yearly_robustness(close_prices=close, save_csv=False)
    _, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(YEARLY_BLEND_CS_CANDIDATE,),
        close_prices=close,
    )
    aligned = align_sleeve_return_streams(
        btc_returns=btc_result.portfolio["strategy_return"],
        cs_returns=cs_results[YEARLY_BLEND_CS_CANDIDATE.name].portfolio["strategy_return"],
    )

    for year in [*sorted(aligned.index.year.unique().tolist()), "FULL"]:
        if year == "FULL":
            sliced = aligned
            report_year = "FULL"
        else:
            sliced = aligned.loc[aligned.index.year == int(year)]
            report_year = str(year)

        btc_returns = sliced["btc_sleeve_return"]
        equity = 10_000.0 * (1.0 + btc_returns).cumprod()
        metrics = summary_metrics(equity=equity, returns=btc_returns, bars_per_year=bars_per_year)

        row = report.loc[
            (report["year"].astype(str) == report_year)
            & ((report["btc_weight"] - 1.0).abs() < 1e-12)
            & (report["cs_weight"].abs() < 1e-12)
        ].iloc[0]

        assert float(row["total_return"]) == pytest.approx(float(metrics["total_return"]), rel=1e-9, abs=1e-9)
        assert float(row["sharpe"]) == pytest.approx(float(metrics["sharpe"]), rel=1e-9, abs=1e-9)
        assert float(row["max_drawdown"]) == pytest.approx(float(metrics["max_drawdown"]), rel=1e-9, abs=1e-9)


def test_yearly_rows_are_sliced_from_full_aligned_stream_not_restarted() -> None:
    close = _toy_universe_close_cross_year()
    report = run_sleeve_blend_yearly_robustness(close_prices=close, save_csv=False)
    _, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(YEARLY_BLEND_CS_CANDIDATE,),
        close_prices=close,
    )
    aligned = align_sleeve_return_streams(
        btc_returns=btc_result.portfolio["strategy_return"],
        cs_returns=cs_results[YEARLY_BLEND_CS_CANDIDATE.name].portfolio["strategy_return"],
    )
    year = sorted(aligned.index.year.unique().tolist())[0]
    sliced = aligned.loc[aligned.index.year == int(year)]
    blend_returns = sliced["btc_sleeve_return"] * 0.9 + sliced["cs_sleeve_return"] * 0.1
    equity = 10_000.0 * (1.0 + blend_returns).cumprod()
    metrics = summary_metrics(equity=equity, returns=blend_returns, bars_per_year=bars_per_year)

    row = report.loc[
        (report["year"].astype(str) == str(year))
        & ((report["btc_weight"] - 0.9).abs() < 1e-12)
        & ((report["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]

    assert float(row["total_return"]) == pytest.approx(float(metrics["total_return"]), rel=1e-9, abs=1e-9)
    assert float(row["annualized_volatility"]) == pytest.approx(float(metrics["annualized_volatility"]), rel=1e-9, abs=1e-9)
    assert float(row["sharpe"]) == pytest.approx(float(metrics["sharpe"]), rel=1e-9, abs=1e-9)