"""Unit tests for research/explosion_delayed_entry_study.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.explosion_delayed_entry_study import (
    DelayedEntryStudyConfig,
    build_delayed_entry_signals,
    build_delayed_entry_trades,
    build_pullback_bucket_summary,
    build_strategy_summary,
    build_train_test_summary,
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
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103, 104, 105, 106, 107])

    # Strong follow-through after the explosion.
    a = _make_symbol_df(
        "AAA/USD",
        [100, 112, 118, 117, 116, 115, 114, 113],
        highs=[101, 113, 119, 118, 117, 116, 115, 114],
        lows=[99, 110, 111, 115, 114, 113, 112, 111],
    )

    # No follow-through, then favorable short drift.
    b = _make_symbol_df(
        "BBB/USD",
        [100, 112, 110, 108, 106, 104, 103, 102],
        highs=[101, 113, 111, 109, 107, 105, 104, 103],
        lows=[99, 109, 108, 106, 104, 102, 101, 100],
    )

    # Mild pullback only.
    c = _make_symbol_df(
        "CCC/USD",
        [100, 112, 114, 113, 112, 111, 110, 109],
        highs=[101, 113, 115, 114, 113, 112, 111, 110],
        lows=[99, 111.5, 111.4, 112, 111, 110, 109, 108],
    )

    # Deep pullback.
    d = _make_symbol_df(
        "DDD/USD",
        [100, 112, 107, 106, 105, 104, 103, 102],
        highs=[101, 113, 108, 107, 106, 105, 104, 103],
        lows=[99, 108.5, 104, 103, 102, 101, 100, 99],
    )

    full = pd.concat([btc, a, b, c, d], ignore_index=True)
    return full.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_delayed_entry_timestamp_alignment_and_follow_through() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()

    signals = build_delayed_entry_signals(full, cfg)
    row = signals.loc[signals["symbol"] == "AAA/USD"].iloc[0]

    assert row["entry_timestamp"] == row["timestamp"] + pd.Timedelta(hours=4)
    assert row["signal_delay_bars"] == 1
    assert bool(row["follow_through_1c"]) is True
    assert bool(row["no_follow_through_1c"]) is False


def test_delayed_entry_uses_only_first_post_event_candle() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    base = _study_frame()
    changed = base.copy()

    # Alter a later candle only; the entry decision should stay identical.
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T12:00:00Z")), "close"] = 500.0
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T12:00:00Z")), "high"] = 510.0
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T12:00:00Z")), "low"] = 490.0

    base_signals = build_delayed_entry_signals(base, cfg)
    changed_signals = build_delayed_entry_signals(changed, cfg)

    base_row = base_signals.loc[base_signals["symbol"] == "AAA/USD"].iloc[0]
    changed_row = changed_signals.loc[changed_signals["symbol"] == "AAA/USD"].iloc[0]

    assert bool(base_row["follow_through_1c"]) == bool(changed_row["follow_through_1c"])
    assert bool(base_row["deep_pullback_1c"]) == bool(changed_row["deep_pullback_1c"])
    assert base_row["pullback_bucket_1c"] == changed_row["pullback_bucket_1c"]


def test_short_return_is_positive_after_no_follow_through() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()
    signals = build_delayed_entry_signals(full, cfg)
    trades = build_delayed_entry_trades(signals, full, cfg)

    short_trade = trades[(trades["setup"] == "SHORT_AFTER_NO_FOLLOW_THROUGH") & (trades["symbol"] == "BBB/USD") & (trades["hold_bars"] == 1)].iloc[0]

    assert short_trade["direction"] == "SHORT"
    assert short_trade["gross_return_pct"] > 0.0


def test_cost_application_subtracts_expected_bps() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()
    signals = build_delayed_entry_signals(full, cfg)
    trades = build_delayed_entry_trades(signals, full, cfg)

    row = trades[(trades["setup"] == "BUY_AT_EXPLOSION") & (trades["symbol"] == "AAA/USD") & (trades["hold_bars"] == 1)].iloc[0]
    assert abs(float(row["net_return_pct_50"]) - (float(row["gross_return_pct"]) - 0.5)) < 1e-12
    assert bool(row["win_50"]) == (float(row["net_return_pct_50"]) > 0.0)


@pytest.mark.parametrize(
    ("symbol", "decision_low", "expected_bucket"),
    [
        ("NO_PULLBACK/USD", 112.0, "NO_PULLBACK"),
        ("SHALLOW/USD", 110.8, "SHALLOW"),
        ("MODERATE/USD", 108.5, "MODERATE"),
        ("DEEP/USD", 105.0, "DEEP"),
    ],
)
def test_pullback_buckets_are_classified_from_first_post_event_candle(symbol: str, decision_low: float, expected_bucket: str) -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103, 104, 105])
    closes = [100, 112, 114, 113, 112, 111]
    highs = [101, 113, 115, 114, 113, 112]
    lows = [99, 111, decision_low, 112, 111, 110]
    alt = _make_symbol_df(symbol, closes, highs=highs, lows=lows)
    full = pd.concat([btc, alt], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    signals = build_delayed_entry_signals(full, cfg)
    row = signals.iloc[0]

    assert row["pullback_bucket_1c"] == expected_bucket


def test_long_only_exclusion_filter_reduces_baseline_trade_count() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    full = _study_frame()
    signals = build_delayed_entry_signals(full, cfg)
    trades = build_delayed_entry_trades(signals, full, cfg)
    summary = build_strategy_summary(trades, cfg, total_event_count=int(signals["event_id"].nunique()))

    baseline = summary[(summary["setup"] == "BUY_AT_EXPLOSION") & (summary["hold_bars"] == 1)].iloc[0]
    filtered = summary[(summary["setup"] == "AVOID_NO_FOLLOW_THROUGH") & (summary["hold_bars"] == 1)].iloc[0]

    assert filtered["n_events"] < baseline["n_events"]
    assert filtered["selection_rate"] < baseline["selection_rate"]


def test_insufficient_future_candles_are_dropped_for_long_holds() -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103])
    alt = _make_symbol_df("ZZZ/USD", [100, 112, 110, 108], highs=[101, 113, 111, 109], lows=[99, 109, 108, 107])
    full = pd.concat([btc, alt], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    signals = build_delayed_entry_signals(full, cfg)
    trades = build_delayed_entry_trades(signals, full, cfg)

    assert trades[(trades["symbol"] == "ZZZ/USD") & (trades["hold_bars"] == 6)].empty


def test_train_test_summary_and_end_to_end_run(tmp_path: Path) -> None:
    cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0, train_fraction=0.5)
    later_btc = _make_symbol_df("BTC/USD", [108, 109, 110, 111, 112, 113, 114, 115], start="2026-01-03T00:00:00Z")
    later_alt = _make_symbol_df(
        "EEE/USD",
        [100, 112, 111, 110, 109, 108, 107, 106],
        highs=[101, 113, 112, 111, 110, 109, 108, 107],
        lows=[99, 109, 108, 107, 106, 105, 104, 103],
        start="2026-01-03T00:00:00Z",
    )
    full = pd.concat([_study_frame(), later_btc, later_alt], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    signals = build_delayed_entry_signals(full, cfg)
    trades = build_delayed_entry_trades(signals, full, cfg)

    train_test = build_train_test_summary(trades, cfg, total_event_count=int(signals["event_id"].nunique()))
    assert not train_test.empty
    assert set(train_test["split"].unique()) <= {"train", "test"}

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for symbol in sorted(full["symbol"].unique()):
        symbol_df = full[full["symbol"] == symbol].copy()
        symbol_file = data_dir / f"{symbol.lower().replace('/', '-')}_4h.csv"
        symbol_df[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(symbol_file, index=False)

    out = tmp_path / "delayed-entry"
    result = run_delayed_entry_study(cfg=cfg, data_dir=data_dir, output_dir=out)
    assert (out / "summary.md").exists()
    assert not result["signals"].empty
