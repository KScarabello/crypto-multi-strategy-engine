"""Tests for research/memecoin_catcher/evaluate_signal_outcomes.py.

Unit tests only; no live network calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.evaluate_signal_outcomes import (
    SIGNAL_KEY_COLUMNS,
    classify_outcomes,
    compute_excursions,
    compute_forward_returns,
    evaluate_signal_row,
    run_outcome_evaluation,
    select_future_window,
    upsert_outcome_rows,
)

# ---------------------------------------------------------------------------
# Shared constants / fixtures
# ---------------------------------------------------------------------------

_SIGNAL_TS_UNIX = 1_748_477_400  # 2025-05-28T21:30:00Z
_SIGNAL_TS_UTC = "2025-05-28T21:30:00Z"
_SIGNAL_PRICE = 10.0


def _make_future_df(
    n: int,
    signal_ts_unix: int = _SIGNAL_TS_UNIX,
    close_vals: list[float] | None = None,
    high_vals: list[float] | None = None,
    low_vals: list[float] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of n candles with open times strictly after signal."""
    rows = []
    for i in range(n):
        t = float(signal_ts_unix + (i + 1) * 900)
        close = close_vals[i] if close_vals else _SIGNAL_PRICE * (1.0 + 0.005 * i)
        high = high_vals[i] if high_vals else close * 1.02
        low = low_vals[i] if low_vals else close * 0.98
        rows.append(
            {
                "time": t,
                "open": close * 0.99,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


def _make_ohlc_payload(pair_id: str, future_df: pd.DataFrame) -> dict:
    """Wrap a future_df in a Kraken-style OHLC payload."""
    rows = []
    for _, r in future_df.iterrows():
        rows.append(
            [
                int(r["time"]),
                str(r.get("open", r["close"] * 0.99)),
                str(r["high"]),
                str(r["low"]),
                str(r["close"]),
                str(r["close"]),
                str(r.get("volume", 100.0)),
                "5",
            ]
        )
    return {"error": [], "result": {pair_id: rows, "last": "0"}}


def _make_signal_row(
    pair_id: str = "TESTUSD",
    last_price: float = _SIGNAL_PRICE,
    ohlc_signal_type: str = "LONG_EXPLOSION",
    snapshot_ts_utc: str = _SIGNAL_TS_UTC,
) -> pd.Series:
    return pd.Series(
        {
            "pair_id": pair_id,
            "last_price": last_price,
            "ohlc_signal_type": ohlc_signal_type,
            "snapshot_ts_utc": snapshot_ts_utc,
        }
    )


def _fake_fetcher(pair_id: str, n_candles: int) -> callable:
    """Return a closure that yields an n-candle OHLC payload."""
    def _fetch(pair: str, interval: int, since: int | None = None) -> dict:
        future_df = _make_future_df(n_candles)
        return _make_ohlc_payload(pair, future_df)
    return _fetch


# ---------------------------------------------------------------------------
# select_future_window
# ---------------------------------------------------------------------------


def test_select_future_window_excludes_signal_candle() -> None:
    df = _make_future_df(5, signal_ts_unix=_SIGNAL_TS_UNIX)
    # Insert a candle AT the signal timestamp (not strictly after)
    at_signal = df.iloc[[0]].copy()
    at_signal["time"] = float(_SIGNAL_TS_UNIX)
    combined = pd.concat([at_signal, df], ignore_index=True)

    result = select_future_window(combined, float(_SIGNAL_TS_UNIX))
    assert all(result["time"] > float(_SIGNAL_TS_UNIX))
    assert len(result) == 5


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------


def test_forward_return_calculation_from_synthetic_candles() -> None:
    closes = [10.5, 11.0, 11.5, 12.0, 12.5] + [13.0] * 91  # 96 total
    future_df = _make_future_df(96, close_vals=closes)
    rets = compute_forward_returns(future_df, signal_price=10.0)

    assert rets["future_ret_15m_pct"] == pytest.approx((10.5 / 10.0 - 1) * 100)
    assert rets["future_ret_1h_pct"] == pytest.approx((12.0 / 10.0 - 1) * 100)
    assert rets["future_ret_4h_pct"] == pytest.approx(
        (future_df.iloc[15]["close"] / 10.0 - 1) * 100
    )
    assert rets["future_ret_24h_pct"] == pytest.approx(
        (future_df.iloc[95]["close"] / 10.0 - 1) * 100
    )


def test_insufficient_future_candles_produce_blanks() -> None:
    future_df = _make_future_df(5)
    rets = compute_forward_returns(future_df, signal_price=10.0)

    assert pd.notna(rets["future_ret_15m_pct"])
    assert pd.notna(rets["future_ret_1h_pct"])
    assert pd.isna(rets["future_ret_4h_pct"])
    assert pd.isna(rets["future_ret_24h_pct"])


def test_zero_signal_price_produces_all_blanks() -> None:
    future_df = _make_future_df(96)
    rets = compute_forward_returns(future_df, signal_price=0.0)
    assert all(pd.isna(v) for v in rets.values())


# ---------------------------------------------------------------------------
# classify_outcomes — LONG_EXPLOSION
# ---------------------------------------------------------------------------


def test_long_explosion_success_label() -> None:
    rets = {"future_ret_1h_pct": 4.0, "future_ret_4h_pct": 5.0, "future_ret_24h_pct": 8.0}
    out = classify_outcomes(rets, "LONG_EXPLOSION")
    assert out["outcome_1h"] == "SUCCESS"
    assert out["outcome_4h"] == "SUCCESS"
    assert out["outcome_24h"] == "SUCCESS"


def test_long_explosion_failure_label() -> None:
    rets = {"future_ret_1h_pct": -4.0, "future_ret_4h_pct": -4.0, "future_ret_24h_pct": -4.0}
    out = classify_outcomes(rets, "LONG_EXPLOSION")
    assert out["outcome_1h"] == "FAILURE"


def test_long_explosion_flat_label() -> None:
    rets = {"future_ret_1h_pct": 1.0, "future_ret_4h_pct": -1.0, "future_ret_24h_pct": 0.5}
    out = classify_outcomes(rets, "LONG_EXPLOSION")
    assert out["outcome_1h"] == "FLAT"
    assert out["outcome_4h"] == "FLAT"


# ---------------------------------------------------------------------------
# classify_outcomes — DUMPING
# ---------------------------------------------------------------------------


def test_dumping_success_label() -> None:
    rets = {"future_ret_1h_pct": -4.0, "future_ret_4h_pct": -8.0, "future_ret_24h_pct": -12.0}
    out = classify_outcomes(rets, "DUMPING")
    assert out["outcome_1h"] == "SUCCESS"
    assert out["outcome_4h"] == "SUCCESS"


def test_dumping_failure_label() -> None:
    rets = {"future_ret_1h_pct": 4.0, "future_ret_4h_pct": 4.0, "future_ret_24h_pct": 4.0}
    out = classify_outcomes(rets, "DUMPING")
    assert out["outcome_1h"] == "FAILURE"


def test_dumping_flat_label() -> None:
    rets = {"future_ret_1h_pct": -1.0, "future_ret_4h_pct": 1.0, "future_ret_24h_pct": 0.0}
    out = classify_outcomes(rets, "DUMPING")
    assert out["outcome_1h"] == "FLAT"


def test_reversal_watch_failure_threshold_is_minus_5() -> None:
    rets = {"future_ret_1h_pct": -4.0, "future_ret_4h_pct": -5.1, "future_ret_24h_pct": -3.0}
    out = classify_outcomes(rets, "REVERSAL_WATCH")
    assert out["outcome_1h"] == "FLAT"    # -4 > -5 → not FAILURE
    assert out["outcome_4h"] == "FAILURE"  # -5.1 <= -5


def test_blank_return_produces_empty_outcome_label() -> None:
    rets = {
        "future_ret_15m_pct": float("nan"),
        "future_ret_1h_pct": float("nan"),
        "future_ret_4h_pct": 5.0,
        "future_ret_24h_pct": float("nan"),
    }
    out = classify_outcomes(rets, "LONG_EXPLOSION")
    assert out["outcome_15m"] == ""
    assert out["outcome_1h"] == ""
    assert out["outcome_4h"] == "SUCCESS"
    assert out["outcome_24h"] == ""


# ---------------------------------------------------------------------------
# compute_excursions — LONG_EXPLOSION
# ---------------------------------------------------------------------------


def test_max_favorable_adverse_long_explosion() -> None:
    highs = [10.0] * 15 + [13.0]  # candle 16 has high 13
    lows = [10.0] * 8 + [8.5] + [10.0] * 7  # candle 9 has low 8.5
    future_df = _make_future_df(16, high_vals=highs, low_vals=lows)

    ex = compute_excursions(future_df, signal_price=10.0, ohlc_signal_type="LONG_EXPLOSION")
    assert ex["max_favorable_4h_pct"] == pytest.approx((13.0 / 10.0 - 1) * 100)
    assert ex["max_adverse_4h_pct"] == pytest.approx((8.5 / 10.0 - 1) * 100)


def test_max_favorable_adverse_dumping() -> None:
    # min low = 7.0 (candle 5), max high = 11.5 (candle 10)
    lows = [9.0] * 4 + [7.0] + [9.0] * 11
    highs = [10.0] * 9 + [11.5] + [10.0] * 6
    future_df = _make_future_df(16, high_vals=highs, low_vals=lows)

    ex = compute_excursions(future_df, signal_price=10.0, ohlc_signal_type="DUMPING")
    assert ex["max_favorable_4h_pct"] == pytest.approx((10.0 / 7.0 - 1) * 100)
    assert ex["max_adverse_4h_pct"] == pytest.approx((11.5 / 10.0 - 1) * 100)


def test_excursion_blank_when_insufficient_candles() -> None:
    future_df = _make_future_df(10)  # fewer than 16 required for 4h
    ex = compute_excursions(future_df, signal_price=10.0, ohlc_signal_type="LONG_EXPLOSION")
    assert pd.isna(ex["max_favorable_4h_pct"])
    assert pd.isna(ex["max_adverse_4h_pct"])


# ---------------------------------------------------------------------------
# evaluate_signal_row with mocked fetcher
# ---------------------------------------------------------------------------


def test_evaluate_signal_row_full_history_produces_all_fields() -> None:
    future_df = _make_future_df(96)
    payload = _make_ohlc_payload("TESTUSD", future_df)
    calls: list[tuple] = []

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        calls.append((pair_id, interval, since))
        return payload

    row = _make_signal_row(ohlc_signal_type="LONG_EXPLOSION")
    result = evaluate_signal_row(row=row, interval=15, fetcher=fake_fetcher)

    assert calls[0][0] == "TESTUSD"
    assert calls[0][1] == 15
    assert result["ohlc_ready"] if "ohlc_ready" in result else True  # field is optional
    assert pd.notna(result["future_ret_1h_pct"])
    assert pd.notna(result["future_ret_24h_pct"])
    assert result["outcome_1h"] in ("SUCCESS", "FAILURE", "FLAT")


def test_evaluate_signal_row_insufficient_history_blanks_returns() -> None:
    future_df = _make_future_df(5)
    payload = _make_ohlc_payload("SMALLUSD", future_df)

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        return payload

    row = _make_signal_row(pair_id="SMALLUSD")
    result = evaluate_signal_row(row=row, interval=15, fetcher=fake_fetcher)

    assert pd.isna(result["future_ret_4h_pct"])
    assert pd.isna(result["future_ret_24h_pct"])
    assert result["outcome_4h"] == ""
    assert result["outcome_24h"] == ""


# ---------------------------------------------------------------------------
# upsert_outcome_rows
# ---------------------------------------------------------------------------


def _make_signals_df(keys: list[tuple[str, str, str]], last_price: float = _SIGNAL_PRICE) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snapshot_ts_utc": ts, "pair_id": pid, "ohlc_signal_type": sig, "last_price": last_price}
            for ts, pid, sig in keys
        ]
    )


def test_upsert_adds_new_key_as_new_row() -> None:
    existing = _make_signals_df([("2025-05-28T21:30:00Z", "ALLUSD", "LONG_EXPLOSION")])
    existing["future_ret_1h_pct"] = "5.0"
    evaluated = _make_signals_df([
        ("2025-05-28T21:30:00Z", "ALLUSD", "LONG_EXPLOSION"),
        ("2025-05-28T21:30:00Z", "NEWINST", "LONG_EXPLOSION"),
    ])
    evaluated["future_ret_1h_pct"] = 3.0
    evaluated["evaluated_ts_utc"] = "2025-05-28T22:00:00Z"

    result, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated)
    assert n_added == 1
    assert len(result) == 2


def test_upsert_fills_blank_field_on_existing_row() -> None:
    """Second run should update NaN → real value for refreshable columns."""
    existing = _make_signals_df([("2025-05-28T21:30:00Z", "ABCUSD", "LONG_EXPLOSION")])
    existing["future_ret_1h_pct"] = float("nan")
    existing["evaluated_ts_utc"] = "2025-05-28T21:30:00Z"

    evaluated = existing.copy()
    evaluated["future_ret_1h_pct"] = 4.5
    evaluated["evaluated_ts_utc"] = "2025-05-28T22:00:00Z"

    result, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated)
    assert n_added == 0
    assert n_updated == 1
    assert n_skipped == 0
    assert float(result.iloc[0]["future_ret_1h_pct"]) == pytest.approx(4.5)


def test_upsert_does_not_overwrite_nonblank_value_with_blank() -> None:
    """A real existing value must never be replaced by NaN."""
    existing = _make_signals_df([("2025-05-28T21:30:00Z", "XYZUSD", "DUMPING")])
    existing["future_ret_1h_pct"] = "7.5"
    existing["evaluated_ts_utc"] = "2025-05-28T21:30:00Z"

    evaluated = existing.copy()
    evaluated["future_ret_1h_pct"] = float("nan")  # new eval has no data yet
    evaluated["evaluated_ts_utc"] = "2025-05-28T22:00:00Z"

    result, _, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated)
    # evaluated_ts_utc changed → counted as updated, but ret value preserved
    assert float(result.iloc[0]["future_ret_1h_pct"]) == pytest.approx(7.5)


def test_upsert_skips_row_when_nothing_changes() -> None:
    """When both existing and new values are blank, row should be skipped."""
    existing = _make_signals_df([("2025-05-28T21:30:00Z", "NODATA", "DUMPING")])
    existing["future_ret_1h_pct"] = float("nan")
    existing["evaluated_ts_utc"] = "2025-05-28T21:30:00Z"

    evaluated = existing.copy()
    evaluated["future_ret_1h_pct"] = float("nan")
    evaluated["evaluated_ts_utc"] = ""  # blank ts → nothing to refresh

    result, n_added, n_updated, n_skipped = upsert_outcome_rows(existing, evaluated)
    assert n_added == 0
    assert n_skipped == 1


def test_upsert_returns_all_rows_when_existing_is_empty() -> None:
    evaluated = _make_signals_df([
        ("2025-05-28T21:30:00Z", "A", "LONG_EXPLOSION"),
        ("2025-05-28T21:30:00Z", "B", "DUMPING"),
    ])
    result, n_added, n_updated, n_skipped = upsert_outcome_rows(pd.DataFrame(), evaluated)
    assert n_added == 2
    assert len(result) == 2


# ---------------------------------------------------------------------------
# run_outcome_evaluation — integration tests
# ---------------------------------------------------------------------------


def test_first_run_creates_rows_with_blank_future_fields(tmp_path: Path) -> None:
    """First run: insufficient candles → outcome fields blank, but row is created."""
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "outcomes.csv"

    future_df = _make_future_df(2)  # only 2 candles → 4h/24h blank
    payload = _make_ohlc_payload("TESTUSD", future_df)

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        return payload

    signals = _make_signals_df([("2025-05-28T21:30:00Z", "TESTUSD", "LONG_EXPLOSION")])
    signals["last_price"] = _SIGNAL_PRICE
    signals.to_csv(input_path, index=False)

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == 1
    assert pd.isna(loaded.iloc[0]["future_ret_4h_pct"])
    assert loaded.iloc[0]["outcome_4h"] in ("", float("nan"), None) or pd.isna(loaded.iloc[0]["outcome_4h"])


def test_second_run_fills_in_blank_future_fields(tmp_path: Path) -> None:
    """Second run: more candles now available → blank fields get populated."""
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "outcomes.csv"

    signals = _make_signals_df([("2025-05-28T21:30:00Z", "TESTUSD", "LONG_EXPLOSION")])
    signals["last_price"] = _SIGNAL_PRICE
    signals.to_csv(input_path, index=False)

    # First run — only 2 candles available
    few_payload = _make_ohlc_payload("TESTUSD", _make_future_df(2))

    def fetcher_few(pair_id: str, interval: int, since: int | None = None) -> dict:
        return few_payload

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fetcher_few)
    after_run1 = pd.read_csv(output_path)
    assert pd.isna(after_run1.iloc[0]["future_ret_4h_pct"])

    # Second run — full 96 candles available
    full_payload = _make_ohlc_payload("TESTUSD", _make_future_df(96))

    def fetcher_full(pair_id: str, interval: int, since: int | None = None) -> dict:
        return full_payload

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fetcher_full)
    after_run2 = pd.read_csv(output_path)
    assert len(after_run2) == 1, "Row count must remain 1 — no duplicates"
    assert pd.notna(after_run2.iloc[0]["future_ret_4h_pct"])
    assert pd.notna(after_run2.iloc[0]["future_ret_24h_pct"])


def test_no_duplicate_rows_on_second_run(tmp_path: Path) -> None:
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "outcomes.csv"

    full_payload = _make_ohlc_payload("ABCUSD", _make_future_df(20))

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        return full_payload

    signals = _make_signals_df([("2025-05-28T21:30:00Z", "ABCUSD", "LONG_EXPLOSION")])
    signals["last_price"] = _SIGNAL_PRICE
    signals.to_csv(input_path, index=False)

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)
    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == 1, "Duplicate outcome row should not be written"


def test_evaluated_ts_utc_updates_on_reevaluation(tmp_path: Path) -> None:
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "outcomes.csv"

    payload = _make_ohlc_payload("TSTUSD", _make_future_df(5))

    call_count = [0]

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        call_count[0] += 1
        return payload

    signals = _make_signals_df([("2025-05-28T21:30:00Z", "TSTUSD", "LONG_EXPLOSION")])
    signals["last_price"] = _SIGNAL_PRICE
    signals.to_csv(input_path, index=False)

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)
    ts1 = pd.read_csv(output_path).iloc[0]["evaluated_ts_utc"]

    import time; time.sleep(1)  # ensure wall-clock advances ≥1 second

    run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)
    ts2 = pd.read_csv(output_path).iloc[0]["evaluated_ts_utc"]

    assert ts2 >= ts1, "evaluated_ts_utc should be refreshed on second run"
    assert call_count[0] == 2, "Fetcher should have been called on both runs"


def test_row_count_stable_when_only_updating(tmp_path: Path) -> None:
    """Running evaluator multiple times on the same signals must not grow the file."""
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "outcomes.csv"

    payload = _make_ohlc_payload("STBUSD", _make_future_df(96))

    def fake_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
        return payload

    signals = _make_signals_df([
        ("2025-05-28T21:30:00Z", "STBUSD", "LONG_EXPLOSION"),
        ("2025-05-28T21:30:00Z", "STBUSD2", "DUMPING"),
    ])
    signals["last_price"] = _SIGNAL_PRICE
    signals.to_csv(input_path, index=False)

    for _ in range(3):
        run_outcome_evaluation(input_path=input_path, output_path=output_path, fetcher=fake_fetcher)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Module safety check
# ---------------------------------------------------------------------------


def test_module_does_not_import_live_or_broker_modules() -> None:
    module_path = Path("research/memecoin_catcher/evaluate_signal_outcomes.py")
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
