"""Unit tests for research/explosion_reversal_event_study.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.explosion_reversal_event_study import (
    EventStudyConfig,
    apply_cooldown,
    build_bucket_summary,
    build_event_time_features,
    compute_forward_outcomes,
    detect_explosion_events,
    join_btc_context,
    load_ohlcv_data,
    label_outcomes,
)


def _make_symbol_df(symbol: str, closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=len(closes), freq="4h", tz="UTC")
    highs = highs if highs is not None else [c * 1.01 for c in closes]
    lows = lows if lows is not None else [c * 0.99 for c in closes]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(closes),
            "symbol": symbol,
        }
    )


def test_detect_explosion_events_finds_large_jump() -> None:
    cfg = EventStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    df = _make_symbol_df("AAA/USD", [100, 101, 102, 120, 121, 122])
    feats = build_event_time_features(df, cfg)
    events = detect_explosion_events(feats, cfg)

    assert not events.empty
    assert (events["ret_4h_pct"] >= 5.0).any()


def test_apply_cooldown_deduplicates_nearby_events() -> None:
    ts = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T04:00:00Z",
        "2026-01-01T16:00:00Z",
    ])
    events = pd.DataFrame({"symbol": ["AAA/USD", "AAA/USD", "AAA/USD"], "timestamp": ts})

    dedup = apply_cooldown(events, cooldown_bars=3)
    assert len(dedup) == 2


def test_forward_return_and_excursions_are_correct() -> None:
    cfg = EventStudyConfig(min_dollar_volume=0.0)

    # Event at first bar: close=100
    # Next bar (4h horizon): close=110, high=115, low=95
    # 24h window bars include up to index 6.
    closes = [100, 110, 108, 112, 105, 107, 120]
    highs = [101, 115, 110, 113, 107, 109, 121]
    lows = [99, 95, 100, 104, 98, 103, 110]

    full = _make_symbol_df("AAA/USD", closes=closes, highs=highs, lows=lows)
    events = full.iloc[[0]].copy()
    events["event_id"] = ["e1"]

    out = compute_forward_outcomes(events, full)
    assert len(out) == 1

    row = out.iloc[0]
    # 4h return: 110/100 - 1 = 10%
    assert abs(float(row["future_ret_4h_pct"]) - 10.0) < 1e-9
    # MFE_4h from high 115 => +15%
    assert abs(float(row["max_favorable_4h_pct"]) - 15.0) < 1e-9
    # MAE_4h from low 95 => -5%
    assert abs(float(row["max_adverse_4h_pct"]) + 5.0) < 1e-9


def test_btc_context_join_aligns_on_timestamp() -> None:
    cfg = EventStudyConfig(btc_symbol="BTC/USD")
    btc = _make_symbol_df("BTC/USD", [100, 104, 108, 112])
    alt = _make_symbol_df("ALT/USD", [10, 11, 9, 10])
    full = pd.concat([btc, alt], ignore_index=True).sort_values(["symbol", "timestamp"])

    events = alt.iloc[[1]].copy()
    joined = join_btc_context(events, full, cfg)

    assert len(joined) == 1
    # BTC return at ts index 1 should be approx +4%
    assert abs(float(joined.iloc[0]["btc_ret_4h_pct"]) - 4.0) < 1e-9


def test_event_time_features_do_not_use_future_data() -> None:
    cfg = EventStudyConfig(min_dollar_volume=0.0)
    base = _make_symbol_df("AAA/USD", [100, 101, 102, 103, 104, 105, 106])
    changed = base.copy()
    # Change a future bar only
    changed.loc[6, "close"] = 500.0
    changed.loc[6, "high"] = 510.0
    changed.loc[6, "low"] = 490.0

    f1 = build_event_time_features(base, cfg)
    f2 = build_event_time_features(changed, cfg)

    # Compare a pre-change index (index 4) for no-look-ahead features
    cols = ["ret_4h_pct", "ret_zscore", "volume_spike_ratio", "move_pct_rank_symbol", "volatility_expansion"]
    for c in cols:
        v1 = f1.loc[4, c]
        v2 = f2.loc[4, c]
        if pd.isna(v1) and pd.isna(v2):
            continue
        assert abs(float(v1) - float(v2)) < 1e-12, f"Feature {c} changed due to future bar"


def test_bucket_summary_flags_small_samples() -> None:
    cfg = EventStudyConfig(min_bucket_events=50)
    events = pd.DataFrame(
        {
            "explosion_size_bucket": ["Q1", "Q2", "Q1", "Q2"],
            "volume_spike_bucket": ["Q1", "Q2", "Q1", "Q2"],
            "liquidity_bucket": ["SMALL", "LARGE", "SMALL", "LARGE"],
            "btc_trend_bucket": ["UP", "DOWN", "UP", "DOWN"],
            "new_high_within_4h": [1.0, 0.0, 1.0, 0.0],
            "pullback_depth_bucket": ["SHALLOW", "DEEP", "SHALLOW", "DEEP"],
            "volatility_bucket": ["Q1", "Q2", "Q1", "Q2"],
            "price_bucket": ["LOW", "MID", "LOW", "MID"],
            "small_mid_large_proxy": ["SMALL", "LARGE", "SMALL", "LARGE"],
            "outcome_label": ["CONTINUATION", "REVERSAL", "CHOP", "REVERSAL"],
            "future_ret_4h_pct": [2.0, -3.0, 0.0, -1.0],
            "net_future_ret_4h_bps_50": [1.5, -3.5, -0.5, -1.5],
        }
    )

    summary = build_bucket_summary(events, cfg)
    assert not summary.empty
    assert summary["small_sample_warning"].all()


def test_optional_missing_ohlcv_files_are_skipped(tmp_path: Path) -> None:
    # Valid file
    valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 120.0],
        }
    )
    valid.to_csv(tmp_path / "abc-usd_4h.csv", index=False)

    # Invalid file (missing columns)
    invalid = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "close": [1.0]})
    invalid.to_csv(tmp_path / "bad-usd_4h.csv", index=False)

    loaded = load_ohlcv_data(tmp_path)
    assert not loaded.empty
    assert set(loaded["symbol"].unique()) == {"ABC/USD"}


def test_label_outcomes_creates_three_class_labels() -> None:
    cfg = EventStudyConfig(label_cost_bps=50, continuation_return_threshold_pct=1.0, reversal_return_threshold_pct=-1.0)
    df = pd.DataFrame(
        {
            "future_ret_4h_pct": [2.0, -2.0, 0.1],
            "future_ret_24h_pct": [3.0, -3.0, 0.0],
            "max_favorable_4h_pct": [2.5, 0.2, 0.5],
            "max_adverse_4h_pct": [-0.5, -4.0, -0.7],
            "ret_4h_pct": [9.0, 8.5, 8.2],
            "volume_spike_ratio": [3.0, 2.0, 1.5],
            "liquidity_rank_universe": [0.9, 0.3, 0.6],
            "rolling_vol_4h_pct": [1.0, 2.0, 1.5],
            "close": [5.0, 2.0, 1.0],
            "first_pullback_depth_24h_pct": [-1.0, -6.0, np.nan],
        }
    )

    labeled = label_outcomes(df, cfg)
    assert "outcome_label" in labeled.columns
    assert set(labeled["outcome_label"].unique()).issubset({"CONTINUATION", "REVERSAL", "CHOP"})
