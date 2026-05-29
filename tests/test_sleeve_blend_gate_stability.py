"""Tests for sleeve blend gate parameter stability sweep."""

from __future__ import annotations

import pandas as pd
import pytest

from research.sleeve_blend_gate_stability import (
    STABILITY_MA_LENGTHS,
    run_sleeve_blend_gate_stability,
)


def _toy_universe_close_cross_year(periods: int = 3200) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=periods, freq="4h", tz="UTC")
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


@pytest.fixture(scope="module")
def stability_panel() -> pd.DataFrame:
    close = _toy_universe_close_cross_year()
    panel, _ = run_sleeve_blend_gate_stability(close_prices=close, save_csv=False)
    return panel


def test_all_requested_ma_lengths_included(stability_panel: pd.DataFrame) -> None:
    ma_rows = stability_panel.loc[stability_panel["gate_name"].str.startswith("BTC_ABOVE_MA_")]
    observed = set(int(v) for v in ma_rows["ma_length"].dropna().astype(int).unique().tolist())
    assert set(STABILITY_MA_LENGTHS).issubset(observed)


def test_always_on_rows_present(stability_panel: pd.DataFrame) -> None:
    always_on = stability_panel.loc[stability_panel["gate_name"] == "ALWAYS_ON"]
    assert len(always_on) == 4


def test_same_weight_deltas_against_always_on(stability_panel: pd.DataFrame) -> None:
    sample = stability_panel.loc[
        (stability_panel["gate_name"] == "BTC_TS_INVESTED")
        & ((stability_panel["btc_weight"] - 0.9).abs() < 1e-12)
        & ((stability_panel["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]
    always = stability_panel.loc[
        (stability_panel["gate_name"] == "ALWAYS_ON")
        & ((stability_panel["btc_weight"] - 0.9).abs() < 1e-12)
        & ((stability_panel["cs_weight"] - 0.1).abs() < 1e-12)
    ].iloc[0]

    assert float(sample["delta_sharpe_vs_always_on_same_weight"]) == pytest.approx(
        float(sample["sharpe"]) - float(always["sharpe"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(sample["delta_max_drawdown_vs_always_on_same_weight"]) == pytest.approx(
        float(sample["max_drawdown"]) - float(always["max_drawdown"]),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(sample["delta_2022_return_vs_always_on_same_weight"]) == pytest.approx(
        float(sample["2022_return"]) - float(always["2022_return"]),
        rel=1e-9,
        abs=1e-9,
    )


def test_stability_output_required_columns(stability_panel: pd.DataFrame) -> None:
    required = {
        "gate_name",
        "ma_length",
        "btc_weight",
        "cs_weight",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "worst_year_return",
        "worst_month_return",
        "2022_return",
        "2022_max_drawdown",
        "count_negative_years",
        "percent_time_cs_enabled",
        "delta_sharpe_vs_always_on_same_weight",
        "delta_max_drawdown_vs_always_on_same_weight",
        "delta_2022_return_vs_always_on_same_weight",
        "delta_sharpe_vs_btc_only",
        "delta_max_drawdown_vs_btc_only",
    }
    assert required.issubset(set(stability_panel.columns))
