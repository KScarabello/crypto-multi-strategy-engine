"""Focused tests for locked candidate robustness validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.explosion_delayed_entry_study import DelayedEntryStudyConfig, build_second_pass_signals
from research.explosion_locked_candidate_validation import (
    COST_BPS,
    LockedCandidateConfig,
    _build_candidate_trades,
    _candidate_signals,
    build_benchmark_comparison,
    build_cost_sensitivity_summary,
    build_overlap_concurrency_summary,
    build_symbol_concentration_summary,
    run_locked_candidate_validation,
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
    btc = _make_symbol_df("BTC/USD", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    aaa = _make_symbol_df(
        "AAA/USD",
        [100, 112, 116, 118, 120, 119, 118, 117, 116, 115],
        highs=[101, 113, 117, 119, 121, 120, 119, 118, 117, 116],
        lows=[99, 111, 113, 116, 118, 117, 116, 115, 114, 113],
    )
    bbb = _make_symbol_df(
        "BBB/USD",
        [100, 112, 115, 114, 113, 112, 111, 110, 109, 108],
        highs=[101, 113, 116, 115, 114, 113, 112, 111, 110, 109],
        lows=[99, 110, 112, 111, 110, 109, 108, 107, 106, 105],
    )
    return pd.concat([btc, aaa, bbb], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_locked_candidate_is_frozen_to_expected_rule() -> None:
    cfg = LockedCandidateConfig()
    assert cfg.follow_rule == "HIGH_BREAK_100_BPS"
    assert cfg.hold_bars == 6


def test_next_open_entry_alignment_when_available() -> None:
    full = _dataset()
    study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    signals = build_second_pass_signals(full, study_cfg)
    locked = _candidate_signals(signals, LockedCandidateConfig())
    trades = _build_candidate_trades(locked, full, LockedCandidateConfig())

    next_open = trades[trades["assumption"] == "ENTRY_NEXT_OPEN"].iloc[0]
    assert next_open["entry_timestamp"] == next_open["observation_timestamp"] + pd.Timedelta(hours=4)


def test_cost_sensitivity_calculations_present_for_all_requested_costs() -> None:
    full = _dataset()
    study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    signals = build_second_pass_signals(full, study_cfg)
    locked = _candidate_signals(signals, LockedCandidateConfig())
    trades = _build_candidate_trades(locked, full, LockedCandidateConfig())

    summary = build_cost_sensitivity_summary(trades)
    assert not summary.empty
    row = summary.iloc[0]
    for bps in COST_BPS:
        assert f"net_return_mean_pct_{bps}" in summary.columns
        assert pd.notna(row[f"net_return_mean_pct_{bps}"])


def test_symbol_exclusion_diagnostics_have_top_k_rows() -> None:
    full = _dataset()
    study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    signals = build_second_pass_signals(full, study_cfg)
    locked = _candidate_signals(signals, LockedCandidateConfig())
    trades = _build_candidate_trades(locked, full, LockedCandidateConfig())

    s = build_symbol_concentration_summary(trades)
    assert not s.empty
    assert (s.get("excluded_top_k", pd.Series(dtype=float)).fillna(-1).isin([0, 1, 3, 5])).any()


def test_overlap_concurrency_diagnostics_include_required_scenarios() -> None:
    full = _dataset()
    study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    signals = build_second_pass_signals(full, study_cfg)
    locked = _candidate_signals(signals, LockedCandidateConfig())
    trades = _build_candidate_trades(locked, full, LockedCandidateConfig())

    c = build_overlap_concurrency_summary(trades, LockedCandidateConfig())
    scenarios = set(c["scenario"].tolist())
    assert "all_signals" in scenarios
    assert "max_1_position_per_timestamp" in scenarios
    assert "max_3_positions_per_timestamp" in scenarios
    assert "symbol_cooldown" in scenarios


def test_benchmark_comparison_has_candidate_and_baseline() -> None:
    full = _dataset()
    study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    signals = build_second_pass_signals(full, study_cfg)
    locked = _candidate_signals(signals, LockedCandidateConfig())
    trades = _build_candidate_trades(locked, full, LockedCandidateConfig())

    b = build_benchmark_comparison(trades, signals, full, LockedCandidateConfig())
    labels = set(b["benchmark"].tolist())
    assert "LOCKED_CANDIDATE" in labels
    assert "BUY_AT_EXPLOSION_BASELINE" in labels


def test_locked_candidate_validation_no_lookahead_guard() -> None:
    cfg = LockedCandidateConfig()
    full = _dataset()

    base_study_cfg = DelayedEntryStudyConfig(ret_4h_threshold_pct=5.0, ret_zscore_threshold=10.0, min_dollar_volume=0.0)
    s1 = build_second_pass_signals(full, base_study_cfg)

    changed = full.copy()
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T08:00:00Z")), "close"] = 400.0
    changed.loc[(changed["symbol"] == "AAA/USD") & (changed["timestamp"] == pd.Timestamp("2026-01-02T08:00:00Z")), "high"] = 410.0

    s2 = build_second_pass_signals(changed, base_study_cfg)

    lock1 = _candidate_signals(s1, cfg)
    lock2 = _candidate_signals(s2, cfg)

    cols = ["event_id", "symbol", "timestamp", "observation_timestamp"]
    p1 = lock1[cols].sort_values(cols).reset_index(drop=True)
    p2 = lock2[cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(p1, p2)


def test_end_to_end_locked_validation_outputs(tmp_path: Path) -> None:
    full = _dataset()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for symbol in sorted(full["symbol"].unique()):
        sdf = full[full["symbol"] == symbol]
        (data_dir / f"{symbol.lower().replace('/', '-')}_4h.csv").write_text(
            sdf[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(index=False)
        )

    out_dir = tmp_path / "reports" / "locked"
    run_locked_candidate_validation(data_dir=data_dir, output_dir=out_dir)

    assert (out_dir / "locked_candidate_trades.csv").exists()
    assert (out_dir / "execution_assumption_summary.csv").exists()
    assert (out_dir / "yearly_summary.csv").exists()
    assert (out_dir / "train_test_holdout_summary.csv").exists()
    assert (out_dir / "symbol_concentration_summary.csv").exists()
    assert (out_dir / "overlap_concurrency_summary.csv").exists()
    assert (out_dir / "btc_regime_summary.csv").exists()
    assert (out_dir / "liquidity_summary.csv").exists()
    assert (out_dir / "cost_sensitivity_summary.csv").exists()
    assert (out_dir / "benchmark_comparison.csv").exists()
    assert (out_dir / "summary.md").exists()
