"""Second-pass unit tests for research/explosion_delayed_entry_study.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.explosion_delayed_entry_study import (
    DelayedEntryStudyConfig,
    build_second_pass_signals,
    build_second_pass_trades,
    build_train_test_summary,
    detect_functional_equivalents,
    evaluate_follow_through,
    run_delayed_entry_study,
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


def _study_frame() -> pd.DataFrame:
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])

    # Follow-through path.
    aaa = _make_symbol_df(
        "AAA/USD",
        [100, 112, 118, 120, 122, 121, 120, 119, 118, 117, 116, 115],
        highs=[101, 113, 119, 121, 123, 122, 121, 120, 119, 118, 117, 116],
        lows=[99, 111, 112, 118, 120, 119, 118, 117, 116, 115, 114, 113],
    )

    # No follow-through path, useful for short and risk-exit checks.
    bbb = _make_symbol_df(
        "BBB/USD",
        [100, 112, 109, 106, 104, 102, 101, 100, 99, 98, 97, 96],
        highs=[101, 113, 110, 107, 105, 103, 102, 101, 100, 99, 98, 97],
        lows=[99, 108, 107, 104, 102, 100, 99, 98, 97, 96, 95, 94],
    )

    # Deeper pullback path.
    ccc = _make_symbol_df(
        "CCC/USD",
        [100, 112, 106, 104, 103, 102, 101, 100, 99, 98, 97, 96],
        highs=[101, 113, 108, 106, 104, 103, 102, 101, 100, 99, 98, 97],
        lows=[99, 107, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
    )

    full = pd.concat([btc, aaa, bbb, ccc], ignore_index=True)
    return full.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_strict_follow_through_threshold_logic() -> None:
    assert evaluate_follow_through("HIGH_BREAK_0_BPS", 100.0, 110.0, 109.0, 110.1, 2.0, 0.5)
    assert not evaluate_follow_through("HIGH_BREAK_25_BPS", 100.0, 110.0, 109.0, 110.2, 2.0, 0.5)
    assert evaluate_follow_through("HIGH_BREAK_25_BPS", 100.0, 110.0, 109.0, 110.3, 2.0, 0.5)
    assert evaluate_follow_through("CLOSE_ABOVE_EXPLOSION_HIGH", 100.0, 110.0, 110.1, 110.1, 2.0, 0.5)
    assert not evaluate_follow_through("CLOSE_ABOVE_EXPLOSION_CLOSE_50_BPS", 100.0, 110.0, 100.4, 111.0, 2.0, 0.5)


def test_delayed_entry_uses_observation_close_not_explosion_close() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    signals = build_second_pass_signals(full, cfg)
    trades = build_second_pass_trades(signals, full, cfg)

    row = trades[
        (trades["setup"] == "BUY_AFTER_FOLLOW_THROUGH_STRICT")
        & (trades["variant"] == "HIGH_BREAK_0_BPS")
        & (trades["symbol"] == "AAA/USD")
        & (trades["hold_bars"] == 1)
    ].iloc[0]

    assert row["entry_timestamp"] == row["observation_timestamp"]
    assert row["entry_timestamp"] == row["event_timestamp"] + pd.Timedelta(hours=4)


def test_immediate_entry_risk_management_exits_failures_at_observation_close() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    signals = build_second_pass_signals(full, cfg)
    trades = build_second_pass_trades(signals, full, cfg)

    row = trades[
        (trades["setup"] == "BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH")
        & (trades["variant"] == "HIGH_BREAK_0_BPS")
        & (trades["symbol"] == "BBB/USD")
        & (trades["hold_bars"] == 6)
    ].iloc[0]

    assert row["entry_timestamp"] == row["event_timestamp"]
    assert bool(row["forced_exit_observation"]) is True
    assert row["exit_timestamp"] == row["observation_timestamp"]


def test_event_time_filter_is_no_lookahead() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    base = _study_frame()
    changed = base.copy()

    # mutate only candles far after observation; event-time filters should stay stable
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T16:00:00Z")), "close"] = 999.0
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T16:00:00Z")), "high"] = 1000.0

    s1 = build_second_pass_signals(base, cfg)
    s2 = build_second_pass_signals(changed, cfg)

    for col in [
        "event_filter_EXCL_LOW_LIQUIDITY",
        "event_filter_EXCL_EXTREME_OVEREXTENSION",
        "event_filter_EXCL_BTC_DOWN_OR_SIDEWAYS",
        "event_filter_EXCL_HIGH_VOL_LOW_VOLUME",
        "event_filter_CONSERVATIVE_COMBINED",
    ]:
        a = s1.loc[s1["symbol"] == "AAA/USD", col].iloc[0]
        b = s2.loc[s2["symbol"] == "AAA/USD", col].iloc[0]
        assert bool(a) == bool(b)


def test_pullback_threshold_assignment() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    signals = build_second_pass_signals(full, cfg)
    row = signals.loc[signals["symbol"] == "CCC/USD"].iloc[0]

    assert row["pullback_depth_1c_pct"] < -5.0
    assert bool(row["deep_pullback_5pct"]) is True
    assert bool(row["deep_pullback_10pct"]) is False


def test_cost_application_across_threshold_variants() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    signals = build_second_pass_signals(full, cfg)
    trades = build_second_pass_trades(signals, full, cfg)

    row = trades[(trades["setup"] == "BUY_AT_EXPLOSION") & (trades["symbol"] == "AAA/USD") & (trades["hold_bars"] == 1)].iloc[0]
    assert abs(float(row["net_return_pct_50"]) - (float(row["gross_return_pct"]) - 0.5)) < 1e-12
    assert abs(float(row["net_return_pct_100"]) - (float(row["gross_return_pct"]) - 1.0)) < 1e-12


def test_insufficient_future_candles_are_skipped() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103])
    alt = _make_symbol_df("ZZZ/USD", [100, 112, 109, 108], highs=[101, 113, 110, 109], lows=[99, 108, 107, 106])
    full = pd.concat([btc, alt], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    signals = build_second_pass_signals(full, cfg)
    trades = build_second_pass_trades(signals, full, cfg)

    # Full-horizon setups should be skipped due insufficient candles; early-exit
    # risk variants may still appear because they close at observation.
    assert trades[
        (trades["symbol"] == "ZZZ/USD")
        & (trades["hold_bars"] == 6)
        & (trades["setup"] == "BUY_AT_EXPLOSION")
    ].empty
    assert trades[
        (trades["symbol"] == "ZZZ/USD")
        & (trades["hold_bars"] == 6)
        & (trades["setup"] == "BUY_AFTER_FOLLOW_THROUGH_STRICT")
    ].empty


def test_train_test_summary_includes_degradation_columns() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0, train_fraction=0.5)

    later = _make_symbol_df(
        "DDD/USD",
        [100, 112, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105],
        highs=[101, 113, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106],
        lows=[99, 111, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103],
        start="2026-01-03T00:00:00Z",
    )
    full = pd.concat([_study_frame(), later], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    signals = build_second_pass_signals(full, cfg)
    trades = build_second_pass_trades(signals, full, cfg)
    summary = build_train_test_summary(trades, cfg, total_events=int(signals["event_id"].nunique()))

    assert not summary.empty
    assert "test_train_degradation_pct_50" in summary.columns


def test_duplicate_variant_detection() -> None:
    data = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2", "e2"],
            "entry_timestamp": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True),
            "exit_timestamp": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"], utc=True),
            "gross_return_pct": [1.0, 1.0, -1.0, -1.0],
            "setup": ["A", "B", "A", "B"],
            "variant": ["V1", "V2", "V1", "V2"],
            "hold_bars": [1, 1, 1, 1],
            "direction": ["LONG", "LONG", "LONG", "LONG"],
        }
    )
    eq = detect_functional_equivalents(data)
    assert len(eq) == 1


def test_end_to_end_second_pass_outputs(tmp_path: Path) -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for symbol in sorted(full["symbol"].unique()):
        f = full[full["symbol"] == symbol]
        out = data_dir / f"{symbol.lower().replace('/', '-')}_4h.csv"
        f[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out, index=False)

    out_dir = tmp_path / "reports"
    result = run_delayed_entry_study(cfg=cfg, data_dir=data_dir, output_dir=out_dir)

    second_pass = out_dir / "second_pass"
    assert (second_pass / "second_pass_trades.csv").exists()
    assert (second_pass / "second_pass_strategy_summary.csv").exists()
    assert (second_pass / "follow_through_threshold_sensitivity.csv").exists()
    assert (second_pass / "event_time_filter_summary.csv").exists()
    assert (second_pass / "risk_management_exit_summary.csv").exists()
    assert (second_pass / "pullback_variant_summary.csv").exists()
    assert (second_pass / "train_test_summary.csv").exists()
    assert (second_pass / "top_second_pass_setups.json").exists()
    assert (second_pass / "summary.md").exists()
    assert not result["signals"].empty
