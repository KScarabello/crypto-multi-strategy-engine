"""Focused tests for the expanded-universe research runner."""

from __future__ import annotations

import pandas as pd
import pytest

import research.run_expanded_universe_experiment as expanded_experiment
from research.run_expanded_universe_experiment import (
    EXPANDED_UNIVERSE_20,
    LOOKBACK_CONFIGS,
    _build_experiment_plan,
    _rebalance_signal_generator,
    build_universal_eligibility_mask,
    run_expanded_universe_experiment,
    symbol_to_local_filename,
)


def _toy_close_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "BTC/USD": [100 + i for i in range(12)],
            "ETH/USD": [50 + 2 * i for i in range(12)],
            "SOL/USD": [20 + i for i in range(12)],
        },
        index=index,
        dtype=float,
    )


def test_expanded_universe_constant_has_20_symbols() -> None:
    """The expanded universe should contain exactly 20 symbols."""
    assert len(EXPANDED_UNIVERSE_20) == 20
    assert "BTC/USD" in EXPANDED_UNIVERSE_20
    assert "XRP/USD" in EXPANDED_UNIVERSE_20


def test_local_filename_mapping_works() -> None:
    """Symbol-to-filename mapping should match local OHLCV naming."""
    assert symbol_to_local_filename("BTC/USD", "4h") == "btc-usd_4h.csv"


def test_universal_eligibility_blocks_before_enough_history() -> None:
    """Eligibility should remain false until the required lookback is satisfied."""
    index = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    close = pd.DataFrame({"BTC/USD": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    mask = build_universal_eligibility_mask(close, min_required_lookback=3)

    assert bool(mask.iloc[0, 0]) is False
    assert bool(mask.iloc[1, 0]) is False
    assert bool(mask.iloc[2, 0]) is False
    assert bool(mask.iloc[3, 0]) is True
    assert bool(mask.iloc[4, 0]) is True


def test_universal_eligibility_blocks_missing_required_window_data() -> None:
    """Eligibility should fail when the trailing window contains a missing close."""
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    close = pd.DataFrame({"BTC/USD": [1.0, 2.0, 3.0, float("nan"), 5.0, 6.0]}, index=index)

    mask = build_universal_eligibility_mask(close, min_required_lookback=3)

    assert bool(mask.iloc[3, 0]) is False
    assert bool(mask.iloc[4, 0]) is False
    assert bool(mask.iloc[5, 0]) is False


def test_toy_experiment_runs_without_real_data_files() -> None:
    """A toy in-memory run should complete without loading local files."""
    close = _toy_close_prices()

    report = run_expanded_universe_experiment(
        symbols=tuple(close.columns),
        universe="toy",
        timeframe="4h",
        close_prices=close,
        save_csv=False,
        lookback_config_names=("short_42_only",),
        rebalance_bars=(2,),
        top_ns=(2,),
        cost_configs=((10, 5),),
    )

    assert not report.empty
    assert set(report["universe"]) == {"toy"}
    assert set(report["lookback_config"]) == {"short_42_only"}
    assert set(report["rebalance_bars"]) == {2}
    assert set(report["fee_bps"]) == {10}
    assert set(report["slippage_bps"]) == {5}
    assert set(report["symbols_count"]) == {3}
    assert {"ts_momentum", "ts_reversal", "cs_momentum"}.issubset(set(report["strategy_name"]))
    assert report["calmar"].dropna().ge(0).all()


def test_ts_signal_generator_uses_current_row_only_for_non_stateful_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-stateful time-series variants should not receive growing history slices."""
    close = _toy_close_prices()
    eligibility = build_universal_eligibility_mask(close, min_required_lookback=0)
    captured_lengths: list[int] = []

    def fake_time_series_momentum_weights(momentum_signal: pd.DataFrame) -> pd.DataFrame:
        captured_lengths.append(len(momentum_signal.index))
        return pd.DataFrame(0.0, index=momentum_signal.index, columns=momentum_signal.columns)

    monkeypatch.setattr(expanded_experiment, "time_series_momentum_weights", fake_time_series_momentum_weights)

    signal_generator = _rebalance_signal_generator(
        strategy_name="ts_momentum",
        lookback_config=LOOKBACK_CONFIGS["short_42_only"],
        close_prices=close,
        eligibility_mask=eligibility,
    )

    signal_generator(close, close.index[-1])

    assert captured_lengths == [1]


def test_toy_experiment_respects_max_runs() -> None:
    """The experiment runner should stop after the requested number of planned runs."""
    close = _toy_close_prices()

    report = run_expanded_universe_experiment(
        symbols=tuple(close.columns),
        universe="toy",
        timeframe="4h",
        close_prices=close,
        save_csv=False,
        lookback_config_names=("short_42_only",),
        rebalance_bars=(2,),
        top_ns=(2,),
        cost_configs=((10, 5),),
        max_runs=2,
    )

    assert len(report) == 2


def test_smoke_mode_slices_to_last_window_rows() -> None:
    """Smoke mode should limit execution to the last configured row window."""
    index = pd.date_range("2024-01-01", periods=2000, freq="4h", tz="UTC")
    close = pd.DataFrame(
        {
            "AAVE/USD": [100 + i for i in range(2000)],
            "ADA/USD": [50 + i for i in range(2000)],
            "APT/USD": [20 + i for i in range(2000)],
            "ARB/USD": [10 + i for i in range(2000)],
            "ATOM/USD": [30 + i for i in range(2000)],
            "AVAX/USD": [40 + i for i in range(2000)],
            "BCH/USD": [60 + i for i in range(2000)],
        },
        index=index,
        dtype=float,
    )

    report = run_expanded_universe_experiment(
        symbols=tuple(close.columns),
        universe="toy_smoke",
        timeframe="4h",
        close_prices=close,
        save_csv=False,
        smoke=True,
        max_runs=1,
    )

    expected_start = str(index[-1500])
    assert len(report) == 1
    assert set(report["symbols_count"]) == {6}
    assert set(report["start_date"]) == {expected_start}


def test_default_plan_excludes_stateful_variants_and_flag_includes_them() -> None:
    """Stateful variants should be excluded by default and included only when requested."""
    lookbacks = (LOOKBACK_CONFIGS["short_42_only"],)

    default_plan = _build_experiment_plan(
        lookback_configs=lookbacks,
        rebalance_bars=(6,),
        top_ns=(3,),
        cost_configs=((10, 5),),
        include_stateful=False,
    )
    default_strategies = {item["strategy_name"] for item in default_plan}
    assert "ts_momentum_exit_signal" not in default_strategies
    assert "ts_momentum_entry_exit" not in default_strategies
    assert {"ts_momentum", "ts_reversal", "ts_momentum_entry_filter", "cs_momentum"} == default_strategies

    stateful_plan = _build_experiment_plan(
        lookback_configs=lookbacks,
        rebalance_bars=(6,),
        top_ns=(3,),
        cost_configs=((10, 5),),
        include_stateful=True,
    )
    stateful_strategies = {item["strategy_name"] for item in stateful_plan}
    assert "ts_momentum_exit_signal" in stateful_strategies
    assert "ts_momentum_entry_exit" in stateful_strategies