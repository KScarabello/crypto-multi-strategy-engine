"""Tests for research-only sleeve blend gating experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from research.sleeve_blend_experiment import align_sleeve_return_streams, run_sleeve_backtests, run_sleeve_blend_experiment
from research.sleeve_blend_gating_experiment import (
    DEFAULT_GATE_SPECS,
    GATING_BASE_ALLOCATIONS,
    GATING_CS_CANDIDATE,
    build_cs_gate_series,
    build_gated_blend_stream,
    run_sleeve_blend_gating_experiment,
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


def test_gating_never_changes_100_btc_baseline() -> None:
    close = _toy_universe_close_cross_year()
    full_report, _ = run_sleeve_blend_gating_experiment(close_prices=close, save_csv=False)
    baseline_rows = full_report.loc[(full_report["btc_weight"] - 1.0).abs() < 1e-12]

    assert len(baseline_rows) == 1
    baseline = baseline_rows.iloc[0]
    assert float(baseline["cs_weight"]) == pytest.approx(0.0, abs=1e-12)
    assert float(baseline["percent_time_cs_enabled"]) == pytest.approx(0.0, abs=1e-12)


def test_cs_weight_is_zero_when_gate_is_off() -> None:
    close = _toy_universe_close_cross_year()
    close_frame, _, btc_result, cs_results = run_sleeve_backtests(cs_candidates=(GATING_CS_CANDIDATE,), close_prices=close)
    aligned = align_sleeve_return_streams(
        btc_returns=btc_result.portfolio["strategy_return"],
        cs_returns=cs_results[GATING_CS_CANDIDATE.name].portfolio["strategy_return"],
    )
    gate_series = build_cs_gate_series(
        gate_spec=DEFAULT_GATE_SPECS[1],
        aligned_index=aligned.index,
        btc_close=close_frame["BTC/USD"].reindex(aligned.index),
        btc_invested=btc_result.holdings_history["BTC/USD"].reindex(aligned.index).fillna(0.0),
    )
    stream = build_gated_blend_stream(aligned, gate_series=gate_series, btc_weight=0.85, cs_weight=0.15)

    off_rows = stream.loc[stream["cs_gate"] < 0.5]
    assert not off_rows.empty
    assert (off_rows["effective_cs_weight"].abs() < 1e-12).all()


def test_percent_time_cs_enabled_between_zero_and_one() -> None:
    close = _toy_universe_close_cross_year()
    full_report, yearly_report = run_sleeve_blend_gating_experiment(close_prices=close, save_csv=False)

    assert ((full_report["percent_time_cs_enabled"] >= 0.0) & (full_report["percent_time_cs_enabled"] <= 1.0)).all()
    assert ((yearly_report["percent_time_cs_enabled"] >= 0.0) & (yearly_report["percent_time_cs_enabled"] <= 1.0)).all()


def test_gating_output_required_columns() -> None:
    close = _toy_universe_close_cross_year()
    full_report, yearly_report = run_sleeve_blend_gating_experiment(close_prices=close, save_csv=False)

    required = {
        "year",
        "gate_rule",
        "btc_weight",
        "cs_weight",
        "cs_candidate",
        "total_return",
        "annual_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "worst_month_return",
        "percent_time_cs_enabled",
        "sleeve_return_correlation",
    }
    assert required.issubset(set(full_report.columns))
    assert required.issubset(set(yearly_report.columns))


def test_always_on_versions_match_prior_blend_results_within_tolerance() -> None:
    close = _toy_universe_close_cross_year()
    full_report, _ = run_sleeve_blend_gating_experiment(close_prices=close, save_csv=False)
    prior = run_sleeve_blend_experiment(
        cs_candidates=(GATING_CS_CANDIDATE,),
        allocations=GATING_BASE_ALLOCATIONS,
        close_prices=close,
        save_csv=False,
    )
    prior_full = prior.loc[prior["year"] == "FULL"].copy()

    for btc_weight, cs_weight in GATING_BASE_ALLOCATIONS[1:]:
        current = full_report.loc[
            (full_report["gate_rule"] == "ALWAYS_ON")
            & ((full_report["btc_weight"] - float(btc_weight)).abs() < 1e-12)
            & ((full_report["cs_weight"] - float(cs_weight)).abs() < 1e-12)
        ].iloc[0]
        old = prior_full.loc[
            (prior_full["cs_strategy"] == GATING_CS_CANDIDATE.name)
            & ((prior_full["btc_weight"] - float(btc_weight)).abs() < 1e-12)
            & ((prior_full["cs_weight"] - float(cs_weight)).abs() < 1e-12)
        ].iloc[0]

        assert float(current["total_return"]) == pytest.approx(float(old["total_return"]), rel=1e-9, abs=1e-9)
        assert float(current["cagr"]) == pytest.approx(float(old["cagr"]), rel=1e-9, abs=1e-9)
        assert float(current["sharpe"]) == pytest.approx(float(old["sharpe"]), rel=1e-9, abs=1e-9)
        assert float(current["max_drawdown"]) == pytest.approx(float(old["max_drawdown"]), rel=1e-9, abs=1e-9)