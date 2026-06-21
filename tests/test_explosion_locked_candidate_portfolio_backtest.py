"""Focused tests for locked-candidate portfolio backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.explosion_delayed_entry_study import DelayedEntryStudyConfig, build_second_pass_signals
from research.explosion_locked_candidate_portfolio_backtest import (
    LockedSignalConfig,
    PortfolioScenario,
    _baseline_entries,
    _candidate_entries,
    _build_scenarios,
    run_portfolio_backtest,
    run_portfolio_simulation,
)


def _make_symbol_df(
    symbol: str,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    highs = highs if highs is not None else [c * 1.01 for c in closes]
    lows = lows if lows is not None else [c * 0.99 for c in closes]
    volumes = volumes if volumes is not None else [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "symbol": symbol,
        }
    )


def _dataset() -> pd.DataFrame:
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113])
    aaa = _make_symbol_df(
        "AAA/USD",
        [100, 112, 116, 118, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111],
        highs=[101, 113, 117, 119, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112],
        lows=[99, 111, 113, 116, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109],
    )
    bbb = _make_symbol_df(
        "BBB/USD",
        [100, 112, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103],
        highs=[101, 113, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104],
        lows=[99, 110, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101],
    )
    full = pd.concat([btc, aaa, bbb], ignore_index=True)
    return full.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _signals_entries(full: pd.DataFrame) -> pd.DataFrame:
    signals = build_second_pass_signals(full, DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0))
    return _candidate_entries(signals, full, LockedSignalConfig())


def test_frozen_signal_rule_is_unchanged() -> None:
    cfg = LockedSignalConfig()
    assert cfg.follow_rule == "HIGH_BREAK_100_BPS"
    assert cfg.hold_bars == 6
    assert cfg.wait_bars_after_event == 1


def test_next_open_entry_and_six_candle_exit_alignment() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    row = entries.iloc[0]

    assert row["entry_timestamp"] == row["observation_timestamp"] + pd.Timedelta(hours=4)
    assert row["exit_timestamp"] == row["entry_timestamp"] + pd.Timedelta(hours=24)


def test_equal_weight_portfolio_accounting_and_idle_cash() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    scenario = PortfolioScenario(
        name="T",
        max_open_positions=999999,
        max_total_exposure=1.0,
        symbol_cooldown_bars=0,
        cost_bps=50,
    )
    trades, curve = run_portfolio_simulation(entries, full, scenario, initial_capital=1.0)

    assert not curve.empty
    assert (curve["cash"] >= -1e-9).all()
    assert (curve["equity"] > 0).all()
    assert len(trades) >= 1


def test_max_position_handling() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    scenario = PortfolioScenario(
        name="MAX1",
        max_open_positions=1,
        max_total_exposure=1.0,
        symbol_cooldown_bars=0,
        cost_bps=50,
    )
    _, curve = run_portfolio_simulation(entries, full, scenario, initial_capital=1.0)
    assert int(curve["open_positions"].max()) <= 1


def test_exposure_limits() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    scenario = PortfolioScenario(
        name="EX25",
        max_open_positions=10,
        max_total_exposure=0.25,
        symbol_cooldown_bars=0,
        cost_bps=50,
    )
    _, curve = run_portfolio_simulation(entries, full, scenario, initial_capital=1.0)
    assert (curve["exposure"] <= 0.251 + 1e-9).all()


def test_symbol_cooldowns() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    scenario = PortfolioScenario(
        name="CD6",
        max_open_positions=999999,
        max_total_exposure=1.0,
        symbol_cooldown_bars=6,
        cost_bps=50,
    )
    trades, _ = run_portfolio_simulation(entries, full, scenario, initial_capital=1.0)
    if trades.empty:
        return
    for _, g in trades.sort_values(["symbol", "entry_timestamp"]).groupby("symbol", sort=False):
        if len(g) <= 1:
            continue
        deltas = g["entry_timestamp"].diff().dropna()
        assert (deltas >= pd.Timedelta(hours=24)).all()


def test_cost_application_reduces_returns() -> None:
    full = _dataset()
    entries = _signals_entries(full)
    s50 = PortfolioScenario("C50", 999999, 1.0, 0, 50)
    s200 = PortfolioScenario("C200", 999999, 1.0, 0, 200)
    t50, _ = run_portfolio_simulation(entries, full, s50, initial_capital=1.0)
    t200, _ = run_portfolio_simulation(entries, full, s200, initial_capital=1.0)

    assert t50["net_return_pct"].mean() >= t200["net_return_pct"].mean()


def test_benchmark_baseline_feasible() -> None:
    full = _dataset()
    signals = build_second_pass_signals(full, DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0))
    base = _baseline_entries(signals, full, hold_bars=6)
    assert not base.empty


def test_no_lookahead_in_locked_entry_selection() -> None:
    full = _dataset()
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)

    s1 = build_second_pass_signals(full, cfg)
    e1 = _candidate_entries(s1, full, LockedSignalConfig())

    changed = full.copy()
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-03T00:00:00Z")), "close"] = 500.0
    s2 = build_second_pass_signals(changed, cfg)
    e2 = _candidate_entries(s2, changed, LockedSignalConfig())

    cols = ["event_id", "symbol", "entry_timestamp", "exit_timestamp"]
    pd.testing.assert_frame_equal(e1[cols].reset_index(drop=True), e2[cols].reset_index(drop=True))


def test_end_to_end_portfolio_outputs(tmp_path: Path) -> None:
    full = _dataset()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for symbol in sorted(full["symbol"].unique()):
        sdf = full[full["symbol"] == symbol]
        (data_dir / f"{symbol.lower().replace('/', '-')}_4h.csv").write_text(
            sdf[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(index=False)
        )

    out_dir = tmp_path / "portfolio"
    result = run_portfolio_backtest(data_dir=data_dir, output_dir=out_dir)

    assert (out_dir / "portfolio_trades.csv").exists()
    assert (out_dir / "portfolio_equity_curves.csv").exists()
    assert (out_dir / "portfolio_summary.csv").exists()
    assert (out_dir / "portfolio_by_year.csv").exists()
    assert (out_dir / "portfolio_by_month.csv").exists()
    assert (out_dir / "portfolio_by_btc_regime.csv").exists()
    assert (out_dir / "portfolio_by_liquidity.csv").exists()
    assert (out_dir / "portfolio_cost_sensitivity.csv").exists()
    assert (out_dir / "portfolio_benchmark_comparison.csv").exists()
    assert (out_dir / "portfolio_drawdown_summary.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert isinstance(result, dict)
