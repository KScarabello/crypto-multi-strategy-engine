"""Tests for research/memecoin_catcher/enrich_candidates_with_ohlc.py.

Unit tests only; no live network calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.enrich_candidates_with_ohlc import (
    compute_breakout_features,
    compute_dump_score,
    compute_label_aware_scores,
    compute_long_explosion_score,
    compute_reversal_watch_score,
    compute_return_features,
    compute_volume_features,
    enrich_candidate_row,
    enrich_candidates_dataframe,
    parse_ohlc_response,
    sort_enriched_candidates,
)


def _make_ohlc_df(n: int, close_start: float = 100.0, close_step: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i in range(n):
        close = close_start + close_step * i
        rows.append(
            {
                "time": float(1_700_000_000 + i * 900),
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


def _make_payload_for_pair(pair_id: str, ohlc_df: pd.DataFrame) -> dict:
    rows = []
    for _, r in ohlc_df.iterrows():
        rows.append(
            [
                int(r["time"]),
                str(r["open"]),
                str(r["high"]),
                str(r["low"]),
                str(r["close"]),
                str(r["close"]),
                str(r["volume"]),
                "1",
            ]
        )
    return {"error": [], "result": {pair_id: rows, "last": "0"}}


def test_compute_return_features_from_synthetic_candles() -> None:
    ohlc = _make_ohlc_df(100, close_start=100.0, close_step=1.0)
    features = compute_return_features(ohlc)

    closes = ohlc["close"]
    expected_15m = (closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0
    expected_1h = (closes.iloc[-1] / closes.iloc[-5] - 1.0) * 100.0
    expected_4h = (closes.iloc[-1] / closes.iloc[-17] - 1.0) * 100.0
    expected_24h = (closes.iloc[-1] / closes.iloc[-97] - 1.0) * 100.0

    assert features["ret_15m_pct"] == pytest.approx(expected_15m)
    assert features["ret_1h_pct"] == pytest.approx(expected_1h)
    assert features["ret_4h_pct"] == pytest.approx(expected_4h)
    assert features["ret_24h_pct"] == pytest.approx(expected_24h)


def test_insufficient_candle_history_sets_ohlc_ready_false() -> None:
    row = pd.Series({"pair_id": "TESTUSD"})
    short_ohlc = _make_ohlc_df(30)
    payload = _make_payload_for_pair("TESTUSD", short_ohlc)

    def fake_fetcher(pair_id: str, interval: int) -> dict:
        assert pair_id == "TESTUSD"
        assert interval == 15
        return payload

    features = enrich_candidate_row(row=row, interval=15, fetcher=fake_fetcher)
    assert features["ohlc_ready"] is False
    assert pd.isna(features["ret_24h_pct"])
    assert pd.isna(features["volume_ratio_4h"])


def test_compute_volume_features_ratio_calculation() -> None:
    ohlc = _make_ohlc_df(40)
    ohlc.loc[:, "volume"] = 10.0
    ohlc.loc[ohlc.index[-4:], "volume"] = 20.0
    ohlc.loc[ohlc.index[-16:], "volume"] = 20.0

    features = compute_volume_features(ohlc)
    assert features["volume_ratio_1h"] == pytest.approx(2.0)
    assert features["volume_ratio_4h"] == pytest.approx(2.0)


def test_compute_breakout_features_true_false() -> None:
    base = _make_ohlc_df(100, close_start=10.0, close_step=0.01)
    true_df = base.copy()
    true_df.loc[true_df.index[-1], "close"] = float(true_df.iloc[-97:-1]["high"].max() + 1.0)
    true_flag = compute_breakout_features(true_df)
    assert true_flag["breakout_24h"] is True

    false_df = base.copy()
    false_df.loc[false_df.index[-1], "close"] = float(false_df.iloc[-97:-1]["high"].max() - 1.0)
    false_flag = compute_breakout_features(false_df)
    assert false_flag["breakout_24h"] is False


def test_negative_returns_huge_volume_not_high_long_score() -> None:
    row = {
        "ret_1h_pct": -8.0,
        "ret_4h_pct": -12.0,
        "ret_24h_pct": -25.0,
        "volume_ratio_1h": 500.0,
        "breakout_24h": False,
        "ohlc_ready": True,
    }
    score = compute_long_explosion_score(row)
    assert pd.isna(score)


def test_dumping_negative_returns_huge_volume_gets_high_dump_score() -> None:
    row = {
        "ret_1h_pct": -8.0,
        "ret_4h_pct": -12.0,
        "ret_24h_pct": -25.0,
        "volume_ratio_1h": 50.0,
        "breakout_24h": False,
        "ohlc_ready": True,
    }
    score = compute_dump_score(row, scanner_label="DUMPING")
    assert pd.notna(score)
    assert float(score) > 10.0


def test_reversal_watch_score_when_down_24h_but_bouncing_with_volume() -> None:
    row = {
        "ret_1h_pct": 3.0,
        "ret_4h_pct": 1.5,
        "ret_24h_pct": -18.0,
        "volume_ratio_1h": 3.5,
        "breakout_24h": False,
        "ohlc_ready": True,
    }
    score = compute_reversal_watch_score(row)
    assert pd.notna(score)
    assert float(score) > 0.0


def test_primary_ohlc_score_uses_long_for_hot_mover_watch() -> None:
    row = {
        "ret_1h_pct": 5.0,
        "ret_4h_pct": 7.0,
        "ret_24h_pct": 20.0,
        "volume_ratio_1h": 2.0,
        "breakout_24h": True,
        "ohlc_ready": True,
    }
    hot_scores = compute_label_aware_scores(row, scanner_label="HOT_MOVER")
    watch_scores = compute_label_aware_scores(row, scanner_label="WATCH")
    assert hot_scores["primary_ohlc_score"] == pytest.approx(hot_scores["long_explosion_score"])
    assert watch_scores["primary_ohlc_score"] == pytest.approx(watch_scores["long_explosion_score"])


def test_primary_ohlc_score_uses_dump_for_dumping() -> None:
    row = {
        "ret_1h_pct": -4.0,
        "ret_4h_pct": -9.0,
        "ret_24h_pct": -30.0,
        "volume_ratio_1h": 4.0,
        "breakout_24h": False,
        "ohlc_ready": True,
    }
    scores = compute_label_aware_scores(row, scanner_label="DUMPING")
    assert scores["primary_ohlc_score"] == pytest.approx(scores["dump_score"])


def test_sorting_uses_signal_priority_then_primary_score() -> None:
    df = pd.DataFrame(
        [
            {"pair_id": "DUMP_A", "ohlc_signal_type": "DUMPING", "primary_ohlc_score": 20.0},
            {"pair_id": "LONG_A", "ohlc_signal_type": "LONG_EXPLOSION", "primary_ohlc_score": 8.0},
            {"pair_id": "REV_A", "ohlc_signal_type": "REVERSAL_WATCH", "primary_ohlc_score": 30.0},
            {"pair_id": "LONG_B", "ohlc_signal_type": "LONG_EXPLOSION", "primary_ohlc_score": 12.0},
        ]
    )
    out = sort_enriched_candidates(df)
    assert out["pair_id"].tolist() == ["LONG_B", "LONG_A", "REV_A", "DUMP_A"]


def test_parse_ohlc_response_accepts_non_pair_key() -> None:
    payload = {
        "error": [],
        "result": {
            "XALTUSD": [[1700000000, "1", "2", "0.5", "1.5", "1.5", "10", "5"]],
            "last": "1700000000",
        },
    }
    out = parse_ohlc_response(payload, pair_id="DIFFERENTKEY")
    assert len(out) == 1
    assert out.iloc[0]["close"] == pytest.approx(1.5)


def test_enrich_candidates_dataframe_uses_mock_fetcher_and_filters_candidates() -> None:
    df = pd.DataFrame(
        [
            {
                "pair_id": "CANDUSD",
                "scanner_label": "HOT_MOVER",
                "is_candidate": True,
            },
            {
                "pair_id": "NOTCANDUSD",
                "scanner_label": "",
                "is_candidate": False,
            },
        ]
    )

    full_ohlc = _make_ohlc_df(120)
    payload = _make_payload_for_pair("CANDUSD", full_ohlc)
    calls: list[tuple[str, int]] = []

    def fake_fetcher(pair_id: str, interval: int) -> dict:
        calls.append((pair_id, interval))
        return payload

    out, success_count, insufficient_count = enrich_candidates_dataframe(
        df,
        interval=15,
        fetcher=fake_fetcher,
    )

    assert calls == [("CANDUSD", 15)]
    assert success_count == 1
    assert insufficient_count == 0
    assert out.loc[out["pair_id"] == "CANDUSD", "ohlc_ready"].iloc[0] == True  # noqa: E712
    assert pd.isna(out.loc[out["pair_id"] == "NOTCANDUSD", "ret_1h_pct"]).iloc[0]


def test_module_does_not_import_live_or_broker_modules() -> None:
    module_path = Path("research/memecoin_catcher/enrich_candidates_with_ohlc.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert all(not m.startswith("live") for m in imported)
    assert all(not m.startswith("brokers") for m in imported)
    assert all("config" not in m for m in imported)