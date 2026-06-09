"""Tests for research/memecoin_catcher/backfill_recent_memecoin_signals.py."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.backfill_recent_memecoin_signals import (
    MIN_LOOKBACK_CANDLES,
    VALID_SNAPSHOT_INTERVALS,
    _approx_scanner_label,
    _compute_diagnostic_flags,
    build_backfill_event_log,
    build_backfill_summary,
    apply_exit_strategies,
    compute_features_at_index,
    compute_outcomes_at_index,
    fetch_symbol_ohlc,
    load_candidate_symbols,
    run_backfill,
    simulate_snapshots_for_symbol,
    OHLC_INTERVAL_SECONDS,
    STRATEGY_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic OHLC data
# ---------------------------------------------------------------------------


def _make_ohlc(n: int, base_price: float = 1.0, trend: float = 0.001) -> pd.DataFrame:
    """
    Build n rows of synthetic 15-min OHLC candles with a gentle uptrend.
    Candle open times start at unix timestamp 0 (epoch) for simplicity.
    """
    times = [i * OHLC_INTERVAL_SECONDS for i in range(n)]
    closes = [base_price * (1 + trend) ** i for i in range(n)]
    rows = []
    for i, (t, c) in enumerate(zip(times, closes)):
        rows.append({
            "time": float(t),
            "open": float(closes[i - 1] if i > 0 else c),
            "high": float(c * 1.005),
            "low": float(c * 0.995),
            "close": float(c),
            "volume": 1000.0 + i * 10,
        })
    return pd.DataFrame(rows)


def _make_bearish_ohlc(n: int, base_price: float = 1.0) -> pd.DataFrame:
    """OHLC where price trends down — 24h return will be negative."""
    times = [i * OHLC_INTERVAL_SECONDS for i in range(n)]
    closes = [base_price * (0.999) ** i for i in range(n)]
    rows = []
    for i, (t, c) in enumerate(zip(times, closes)):
        rows.append({
            "time": float(t),
            "open": float(closes[i - 1] if i > 0 else c),
            "high": float(c * 1.002),
            "low": float(c * 0.998),
            "close": float(c),
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _make_volatile_ohlc(n: int, base_price: float = 1.0, spike_at: int = 100) -> pd.DataFrame:
    """OHLC with a big spike at candle spike_at — useful for MFE/MAE tests."""
    rows = []
    for i in range(n):
        t = float(i * OHLC_INTERVAL_SECONDS)
        c = base_price * 1.1 if i == spike_at else base_price
        rows.append({
            "time": t,
            "open": float(base_price),
            "high": float(c * 1.02),
            "low": float(base_price * 0.98),
            "close": float(c),
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _approx_scanner_label
# ---------------------------------------------------------------------------


def test_approx_label_hot_mover():
    assert _approx_scanner_label(5.0) == "HOT_MOVER"
    assert _approx_scanner_label(10.0) == "HOT_MOVER"


def test_approx_label_watch():
    assert _approx_scanner_label(2.0) == "WATCH"
    assert _approx_scanner_label(4.9) == "WATCH"


def test_approx_label_empty_below_threshold():
    assert _approx_scanner_label(1.9) == ""
    assert _approx_scanner_label(0.0) == ""
    assert _approx_scanner_label(-5.0) == ""


def test_approx_label_nan():
    assert _approx_scanner_label(float("nan")) == ""


# ---------------------------------------------------------------------------
# _compute_diagnostic_flags
# ---------------------------------------------------------------------------


def test_diagnostic_flags_clean_continuation():
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": 2.0, "ret_4h_pct": 3.0,
                "ret_24h_pct": 10.0, "volume_ratio_4h": 5.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["is_clean_continuation"] == True


def test_diagnostic_flags_not_clean_when_1h_negative():
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": -1.0, "ret_4h_pct": 3.0,
                "ret_24h_pct": 10.0, "volume_ratio_4h": 5.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["is_clean_continuation"] == False


def test_diagnostic_flags_overextended():
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": 2.0, "ret_4h_pct": 3.0,
                "ret_24h_pct": 30.0, "volume_ratio_4h": 5.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["is_overextended_24h"] == True


def test_diagnostic_flags_volume_climax():
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": 2.0, "ret_4h_pct": 3.0,
                "ret_24h_pct": 10.0, "volume_ratio_4h": 15.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["is_volume_climax"] == True


def test_diagnostic_flags_spread_unavailable():
    """Spread is not available from OHLC — is_wide_spread must be False (not unknown)."""
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": 2.0, "ret_4h_pct": 3.0,
                "ret_24h_pct": 10.0, "volume_ratio_4h": 5.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["is_wide_spread"] == False  # unknown, treated as not wide


def test_diagnostic_danger_terminal_spike_disabled():
    """Terminal spike detection requires live spread — must be False."""
    features = {"ret_15m_pct": 1.0, "ret_1h_pct": -5.0, "ret_4h_pct": 10.0,
                "ret_24h_pct": 20.0, "volume_ratio_4h": 15.0, "spread_pct": 5.0}
    flags = _compute_diagnostic_flags(features)
    assert flags["danger_terminal_spike"] == False


# ---------------------------------------------------------------------------
# compute_features_at_index — no-lookahead guarantee
# ---------------------------------------------------------------------------


def test_no_lookahead_features_use_only_past_candles():
    """
    Features at idx=99 must not use candle 100 or later.
    We verify by placing a massive spike at candle 100 that would
    contaminate ret_15m_pct if lookahead occurred.
    """
    n = 200
    ohlc = _make_ohlc(n, base_price=1.0, trend=0.001)
    # Inject a 1000x spike at candle 100
    ohlc.loc[100, "close"] = 1000.0
    ohlc.loc[100, "high"] = 1000.0

    features_at_99 = compute_features_at_index(ohlc, idx=99)
    features_at_100 = compute_features_at_index(ohlc, idx=100)

    # At idx=99, the spike candle (100) must NOT be included
    if features_at_99 is not None:
        assert features_at_99["ret_15m_pct"] < 100.0, (
            "ret_15m at idx=99 includes future spike — lookahead bias detected!"
        )

    # At idx=100, the spike IS the latest candle, so ret_15m should be huge
    if features_at_100 is not None:
        assert features_at_100["ret_15m_pct"] > 100.0


def test_features_return_none_when_insufficient_candles():
    ohlc = _make_ohlc(MIN_LOOKBACK_CANDLES - 1)  # one too few
    result = compute_features_at_index(ohlc, idx=MIN_LOOKBACK_CANDLES - 2)
    assert result is None


def test_features_return_none_for_small_idx():
    ohlc = _make_ohlc(200)
    # idx < MIN_LOOKBACK_CANDLES - 1 must return None
    result = compute_features_at_index(ohlc, idx=5)
    assert result is None


def test_features_contain_expected_keys():
    ohlc = _make_ohlc(200, trend=0.002)
    result = compute_features_at_index(ohlc, idx=MIN_LOOKBACK_CANDLES)
    if result is not None:
        for key in ("ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
                    "volume_ratio_1h", "entry_price", "snapshot_ts_utc",
                    "is_clean_continuation", "is_wide_spread", "ohlc_signal_type"):
            assert key in result, f"Missing key: {key}"


def test_features_snapshot_ts_after_candle_close():
    """snapshot_ts_unix must equal candle[idx].time + interval_seconds."""
    ohlc = _make_ohlc(200, trend=0.003)
    idx = 150
    result = compute_features_at_index(ohlc, idx=idx)
    if result is not None:
        expected_ts = float(ohlc.iloc[idx]["time"]) + OHLC_INTERVAL_SECONDS
        assert result["snapshot_ts_unix"] == pytest.approx(expected_ts)


def test_features_bearish_trend_returns_none():
    """A deeply bearish trend should not produce LONG_EXPLOSION (scanner label empty)."""
    ohlc = _make_bearish_ohlc(200)
    # Most snapshots should return None (no HOT_MOVER/WATCH label)
    results = [compute_features_at_index(ohlc, idx=i) for i in range(96, 180, 4)]
    valid = [r for r in results if r is not None]
    long_explosions = [r for r in valid if r.get("ohlc_signal_type") == "LONG_EXPLOSION"]
    assert len(long_explosions) == 0


# ---------------------------------------------------------------------------
# compute_outcomes_at_index — forward returns
# ---------------------------------------------------------------------------


def test_forward_return_calculation():
    """
    Build OHLC where close at candle i = base * (1+r)^i.
    At idx=99, signal_price = close[99].
    future_ret_4h should equal close[115] / close[99] - 1 (16 candles at 15-min = 4h).
    """
    r = 0.01  # 1% per candle
    ohlc = _make_ohlc(300, base_price=1.0, trend=r)
    idx = 99
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)

    expected_4h = (float(ohlc.iloc[idx + 16]["close"]) / signal_price - 1.0) * 100.0
    assert "future_ret_4h_pct" in outcomes
    assert outcomes["future_ret_4h_pct"] == pytest.approx(expected_4h, rel=1e-4)


def test_forward_return_1h():
    r = 0.005
    ohlc = _make_ohlc(250, base_price=1.0, trend=r)
    idx = 100
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)

    expected_1h = (float(ohlc.iloc[idx + 4]["close"]) / signal_price - 1.0) * 100.0
    assert outcomes["future_ret_1h_pct"] == pytest.approx(expected_1h, rel=1e-4)


def test_forward_return_nan_when_insufficient_future_candles():
    """If there aren't enough future candles for 24h, future_ret_24h_pct must be NaN."""
    ohlc = _make_ohlc(200, trend=0.002)
    idx = 195  # only 4 future candles left (200 - 196)
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)
    assert math.isnan(outcomes["future_ret_24h_pct"])


def test_max_favorable_excursion():
    """
    Inject a spike into the future window; max_favorable_4h should capture it.
    """
    n = 300
    ohlc = _make_ohlc(n, base_price=1.0, trend=0.0)
    idx = 100
    # Inject a 20% spike 8 candles into the future (within 4h window)
    ohlc.loc[idx + 8, "high"] = float(ohlc.iloc[idx]["close"]) * 1.20
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)
    assert outcomes["max_favorable_4h_pct"] == pytest.approx(20.0, abs=0.5)


def test_max_adverse_excursion():
    """
    Inject a dip; max_adverse_4h should be negative and capture it.
    """
    n = 300
    ohlc = _make_ohlc(n, base_price=1.0, trend=0.0)
    idx = 100
    # Inject a -10% dip 5 candles into the future (within 4h window)
    ohlc.loc[idx + 5, "low"] = float(ohlc.iloc[idx]["close"]) * 0.90
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)
    assert outcomes["max_adverse_4h_pct"] == pytest.approx(-10.0, abs=0.5)


def test_max_favorable_nan_when_fewer_than_16_future_candles():
    """4h excursion requires 16 future candles; NaN if fewer available."""
    n = 120
    ohlc = _make_ohlc(n, trend=0.001)
    idx = 110  # only 9 future candles
    signal_price = float(ohlc.iloc[idx]["close"])
    outcomes = compute_outcomes_at_index(ohlc, idx=idx, signal_price=signal_price)
    assert math.isnan(outcomes["max_favorable_4h_pct"])


# ---------------------------------------------------------------------------
# simulate_snapshots_for_symbol
# ---------------------------------------------------------------------------


def test_simulate_returns_empty_for_too_short_ohlc():
    ohlc = _make_ohlc(MIN_LOOKBACK_CANDLES - 1)
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=4)
    assert events == []


def test_simulate_returns_empty_for_bearish_trend():
    ohlc = _make_bearish_ohlc(250)
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=4)
    assert events == []


def test_simulate_events_have_required_keys():
    ohlc = _make_ohlc(300, trend=0.003)
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=4)
    for event in events:
        assert "symbol" in event
        assert "pair_id" in event
        assert "snapshot_ts_utc" in event
        assert "entry_price" in event
        assert "is_clean_continuation" in event
        assert "ohlc_signal_type" in event


def test_simulate_only_long_explosion_events():
    ohlc = _make_ohlc(300, trend=0.003)
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=4)
    for event in events:
        assert event["ohlc_signal_type"] == "LONG_EXPLOSION"


def test_simulate_snapshot_step_respected():
    """Events must be spaced at least snapshot_step candles apart in time."""
    ohlc = _make_ohlc(300, trend=0.003)
    step = 16  # 4h snapshots
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=step)
    if len(events) >= 2:
        ts_list = sorted(e["snapshot_ts_unix"] for e in events)
        for a, b in zip(ts_list, ts_list[1:]):
            gap_seconds = b - a
            # Must be at least step * 15-min apart
            assert gap_seconds >= step * OHLC_INTERVAL_SECONDS - 1


def test_simulate_entry_price_equals_candle_close():
    """entry_price must equal the close of the last complete candle at snapshot time."""
    ohlc = _make_ohlc(300, trend=0.003)
    events = simulate_snapshots_for_symbol("TESTUSDT", "TEST/USDT", ohlc, snapshot_step=4)
    for event in events:
        # snapshot_ts_unix = candle.time + interval_seconds → candle.time = snap_ts - interval
        snap_ts = event["snapshot_ts_unix"]
        candle_time = snap_ts - OHLC_INTERVAL_SECONDS
        matching = ohlc[ohlc["time"] == candle_time]
        if not matching.empty:
            expected_close = float(matching.iloc[0]["close"])
            assert event["entry_price"] == pytest.approx(expected_close, rel=1e-6)


# ---------------------------------------------------------------------------
# load_candidate_symbols
# ---------------------------------------------------------------------------


def test_load_candidate_symbols_from_csv(tmp_path):
    df = pd.DataFrame({
        "pair_id": ["ALLOUSD", "SWARMSUSD"],
        "wsname": ["ALLO/USD", "SWARMS/USD"],
        "scanner_label": ["HOT_MOVER", "WATCH"],
    })
    p = tmp_path / "candidates.csv"
    df.to_csv(p, index=False)
    pairs = load_candidate_symbols(candidates_path=p)
    assert len(pairs) == 2
    assert ("ALLOUSD", "ALLO/USD") in pairs


def test_load_candidate_symbols_deduplicates(tmp_path):
    df = pd.DataFrame({
        "pair_id": ["ALLOUSD", "ALLOUSD"],
        "wsname": ["ALLO/USD", "ALLO/USD"],
    })
    p = tmp_path / "candidates.csv"
    df.to_csv(p, index=False)
    pairs = load_candidate_symbols(candidates_path=p)
    assert len(pairs) == 1


def test_load_candidate_symbols_limit(tmp_path):
    df = pd.DataFrame({
        "pair_id": [f"TOK{i}USD" for i in range(10)],
        "wsname": [f"TOK{i}/USD" for i in range(10)],
    })
    p = tmp_path / "candidates.csv"
    df.to_csv(p, index=False)
    pairs = load_candidate_symbols(candidates_path=p, limit=3)
    assert len(pairs) == 3


def test_load_candidate_symbols_override():
    pairs = load_candidate_symbols(symbols_override=["ALLO/USD", "SWARMS/USD"])
    assert len(pairs) == 2
    assert pairs[0][1] == "ALLO/USD"


def test_load_candidate_symbols_missing_file():
    pairs = load_candidate_symbols(candidates_path=Path("/nonexistent/path.csv"))
    assert pairs == []


# ---------------------------------------------------------------------------
# fetch_symbol_ohlc — with mock fetcher
# ---------------------------------------------------------------------------


def _mock_fetcher(pair_id: str, interval: int, since: int | None = None) -> dict:
    """Return a minimal valid Kraken OHLC payload."""
    t0 = 1_700_000_000
    rows = [
        [t0 + i * 900, "1.0", "1.01", "0.99", "1.00", "1.00", "1000", 10]
        for i in range(200)
    ]
    return {"error": [], "result": {pair_id: rows, "last": t0 + 200 * 900}}


def test_fetch_symbol_ohlc_returns_dataframe():
    df = fetch_symbol_ohlc("ALLOUSD", fetcher=_mock_fetcher)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "close" in df.columns


def test_fetch_symbol_ohlc_returns_empty_on_error():
    def bad_fetcher(pair_id, interval, since=None):
        raise ConnectionError("network error")

    df = fetch_symbol_ohlc("ALLOUSD", fetcher=bad_fetcher)
    assert df.empty


# ---------------------------------------------------------------------------
# run_backfill — with mock fetcher
# ---------------------------------------------------------------------------


def test_run_backfill_with_mock_fetcher(tmp_path):
    candidates = pd.DataFrame({
        "pair_id": ["ALLOUSD"],
        "wsname": ["ALLO/USD"],
        "scanner_label": ["HOT_MOVER"],
    })
    p = tmp_path / "candidates.csv"
    candidates.to_csv(p, index=False)

    df = run_backfill(
        days=7,
        snapshot_interval="4h",
        candidates_path=p,
        fetcher=_mock_fetcher,
    )

    # The mock has flat prices (0% return), so no HOT_MOVER events expected
    # (ret_24h ≈ 0%). It should return empty or a df with 0 events.
    assert isinstance(df, pd.DataFrame)


def test_run_backfill_missing_candidates_file(tmp_path):
    df = run_backfill(
        days=7,
        snapshot_interval="1h",
        candidates_path=tmp_path / "nonexistent.csv",
        fetcher=_mock_fetcher,
    )
    assert df.empty


def test_run_backfill_symbol_override(tmp_path):
    candidates = pd.DataFrame({"pair_id": ["OTHER"], "wsname": ["OTHER/USD"]})
    p = tmp_path / "candidates.csv"
    candidates.to_csv(p, index=False)

    df = run_backfill(
        days=7,
        snapshot_interval="4h",
        candidates_path=p,
        symbols_override=["ALLOUSD"],
        fetcher=_mock_fetcher,
    )
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# apply_exit_strategies
# ---------------------------------------------------------------------------


def _make_signal_df(**kwargs) -> pd.DataFrame:
    base = {
        "symbol": "TEST/USD",
        "snapshot_ts_utc": "2026-01-01T00:00:00Z",
        "entry_price": 1.0,
        "future_ret_4h_pct": 3.0,
        "future_ret_24h_pct": 5.0,
        "max_favorable_4h_pct": 5.0,
        "max_adverse_4h_pct": -2.0,
        "max_favorable_24h_pct": 7.0,
        "max_adverse_24h_pct": -3.0,
        "is_clean_continuation": True,
        "is_wide_spread": False,
        "danger_terminal_spike": False,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_apply_exit_strategies_adds_all_columns():
    df = _make_signal_df()
    out = apply_exit_strategies(df)
    for s in STRATEGY_NAMES:
        assert f"ret_{s}" in out.columns
        assert f"trigger_{s}" in out.columns


def test_apply_exit_strategies_tp10_triggered():
    df = _make_signal_df(max_favorable_4h_pct=12.0, future_ret_4h_pct=5.0)
    out = apply_exit_strategies(df)
    assert out.iloc[0]["ret_clean_continuation_tp10_else_4h"] == pytest.approx(10.0)


def test_apply_exit_strategies_non_cc_gets_nan_for_cc_strategies():
    df = _make_signal_df(is_clean_continuation=False)
    out = apply_exit_strategies(df)
    assert math.isnan(out.iloc[0]["ret_clean_continuation_fixed_4h"])
    # Baseline strategy still gets a value
    assert not math.isnan(out.iloc[0]["ret_baseline_long_explosion_fixed_4h"])


# ---------------------------------------------------------------------------
# build_backfill_summary
# ---------------------------------------------------------------------------


def test_build_backfill_summary_has_all_strategies():
    df = _make_signal_df()
    with_strats = apply_exit_strategies(df)
    summary = build_backfill_summary(with_strats)
    for s in STRATEGY_NAMES:
        assert s in summary["strategy"].values


def test_build_backfill_summary_subsets():
    df = _make_signal_df()
    with_strats = apply_exit_strategies(df)
    summary = build_backfill_summary(with_strats)
    subsets = set(summary["subset"].unique())
    assert "all" in subsets
    assert "excl_ALLO" in subsets
    assert "excl_best_trade" in subsets


# ---------------------------------------------------------------------------
# build_backfill_event_log
# ---------------------------------------------------------------------------


def test_build_backfill_event_log_shape():
    df = pd.DataFrame([
        {"symbol": "A/USD", "snapshot_ts_utc": "2026-01-01T00:00:00Z",
         "entry_price": 1.0, "future_ret_4h_pct": 2.0, "future_ret_24h_pct": 3.0,
         "max_favorable_4h_pct": 5.0, "max_adverse_4h_pct": -1.0,
         "max_favorable_24h_pct": 6.0, "max_adverse_24h_pct": -2.0,
         "is_clean_continuation": True, "is_wide_spread": False,
         "danger_terminal_spike": False},
        {"symbol": "B/USD", "snapshot_ts_utc": "2026-01-02T00:00:00Z",
         "entry_price": 2.0, "future_ret_4h_pct": -1.0, "future_ret_24h_pct": -2.0,
         "max_favorable_4h_pct": 3.0, "max_adverse_4h_pct": -4.0,
         "max_favorable_24h_pct": 3.0, "max_adverse_24h_pct": -4.0,
         "is_clean_continuation": True, "is_wide_spread": False,
         "danger_terminal_spike": False},
    ])
    with_strats = apply_exit_strategies(df)
    event_log = build_backfill_event_log(with_strats)
    assert len(event_log) == 2
    for s in STRATEGY_NAMES:
        assert f"ret_{s}" in event_log.columns


# ---------------------------------------------------------------------------
# No live trading imports
# ---------------------------------------------------------------------------


def test_no_broker_or_live_imports():
    import ast
    src = Path("research/memecoin_catcher/backfill_recent_memecoin_signals.py").read_text()
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    forbidden = ["brokers", "live_trading", "place_order", "ccxt", "execution"]
    for token in forbidden:
        matches = [n for n in names if token in n]
        assert not matches, f"Forbidden import: {token!r} in {matches}"
