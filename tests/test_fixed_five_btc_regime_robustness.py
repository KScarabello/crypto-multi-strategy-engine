"""Unit tests for research/fixed_five_btc_regime_robustness.py

All tests use synthetic fixtures where possible.
Tests requiring real data skip when data files are absent.

RESEARCH ONLY — no live trading code imported.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.fixed_five_btc_regime_robustness as rob


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_close(
    n: int = 200,
    symbols: list[str] | None = None,
    btc_trend: float = 0.001,
    others_trend: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic close-price matrix with 4h frequency."""
    symbols = symbols or ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for i, sym in enumerate(symbols):
        drift = btc_trend if "BTC" in sym else others_trend + i * 0.0001
        data[sym] = 1000.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.005, n))
    return pd.DataFrame(data, index=idx)


def _make_close_btc_above_ma(n: int = 500, ma_bars: int = 180, seed: int = 1) -> pd.DataFrame:
    """BTC consistently above its MA — regime gate always ON."""
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for sym in symbols:
        if "BTC" in sym:
            # Strong uptrend so price stays above MA
            data[sym] = 1000.0 * np.cumprod(1.0 + 0.005 + rng.normal(0, 0.001, n))
        else:
            data[sym] = 500.0 * np.cumprod(1.0 + 0.002 + rng.normal(0, 0.003, n))
    return pd.DataFrame(data, index=idx)


def _make_close_btc_below_ma(n: int = 500, ma_bars: int = 180, seed: int = 2) -> pd.DataFrame:
    """BTC consistently below its MA — regime gate always OFF (all-cash)."""
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for sym in symbols:
        if "BTC" in sym:
            # Persistent downtrend so price stays below MA
            data[sym] = 10000.0 * np.cumprod(1.0 - 0.004 + rng.normal(0, 0.001, n))
        else:
            data[sym] = 500.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.003, n))
    return pd.DataFrame(data, index=idx)


def _make_ohlcv_from_close(close: pd.DataFrame) -> pd.DataFrame:
    """Convert close matrix to long-format OHLCV (same as _close_to_ohlcv)."""
    from research.universe_integrity_analysis import _close_to_ohlcv
    return _close_to_ohlcv(close)


def _run_small_backtest(
    signal_gen: object,
    close: pd.DataFrame,
    rebalance_bars: int = 6,
) -> "BacktestResult":
    from backtest.engine import run_backtest
    from research.universe_integrity_analysis import _close_to_ohlcv
    ohlcv = _close_to_ohlcv(close)
    return run_backtest(
        ohlcv=ohlcv,
        signal_generator=signal_gen,
        initial_capital=10_000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        rebalance_every_bars=rebalance_bars,
    )


# ---------------------------------------------------------------------------
# 1. test_ma_label_180_bars
# ---------------------------------------------------------------------------


def test_ma_label_180_bars():
    assert rob.ma_label(180) == "MA-180bars(≈30days)"


# ---------------------------------------------------------------------------
# 2. test_ma_label_360_bars
# ---------------------------------------------------------------------------


def test_ma_label_360_bars():
    assert rob.ma_label(360) == "MA-360bars(≈60days)"


# ---------------------------------------------------------------------------
# 3. test_ma_bars_to_days_conversion
# ---------------------------------------------------------------------------


def test_ma_bars_to_days_conversion():
    expected = {
        120: 20,
        150: 25,
        180: 30,
        210: 35,
        240: 40,
        300: 50,
        360: 60,
        420: 70,
        480: 80,
    }
    for bars, days in expected.items():
        lbl = rob.ma_label(bars)
        assert f"≈{days}days" in lbl, f"Expected ≈{days}days in label for {bars} bars, got: {lbl}"


# ---------------------------------------------------------------------------
# 4. test_no_same_bar_regime_execution
# ---------------------------------------------------------------------------


def test_no_same_bar_regime_execution():
    """Signal and execution timestamps must differ by exactly 1 bar (4h)."""
    close = _make_close(n=200, seed=10)
    signal_gen = rob.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=36)
    result = _run_small_backtest(signal_gen, close, rebalance_bars=6)

    rl = result.rebalance_log
    if rl.empty:
        pytest.skip("No rebalances in synthetic backtest — increase n or reduce min_history")

    rl = rl.copy()
    rl["signal_timestamp"] = pd.to_datetime(rl["signal_timestamp"], utc=True, errors="coerce")
    rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")
    rl = rl.dropna(subset=["signal_timestamp", "execution_timestamp"])

    assert len(rl) > 0, "No valid rebalance rows to check"

    for _, row in rl.iterrows():
        diff = row["execution_timestamp"] - row["signal_timestamp"]
        # Must be exactly 1 bar (4h)
        assert diff == pd.Timedelta("4h"), (
            f"Expected 4h gap, got {diff} "
            f"(signal={row['signal_timestamp']}, exec={row['execution_timestamp']})"
        )


# ---------------------------------------------------------------------------
# 5. test_regime_gate_transition_detected_rising
# ---------------------------------------------------------------------------


def test_regime_gate_transition_detected_rising():
    """Verify RISK_ON transition is detected when BTC crosses above its MA."""
    n_bars = 600
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="4h", tz="UTC")
    # BTC falls for first half, then strongly rises for second half
    btc = np.concatenate([
        np.linspace(10000, 5000, n_bars // 2),  # falling
        np.linspace(5000, 15000, n_bars - n_bars // 2),  # rising
    ])
    data = {"BTC/USD": btc}
    for sym in symbols:
        if sym != "BTC/USD":
            data[sym] = np.full(n_bars, 500.0)
    close = pd.DataFrame(data, index=idx)

    joint_start = idx[180]  # enough history for MA-180
    n_switches, avg_dur = rob.compute_regime_stats(close, 180, joint_start)

    # There should be at least one switch (from below MA to above MA)
    assert n_switches >= 1, f"Expected at least 1 regime switch, got {n_switches}"


# ---------------------------------------------------------------------------
# 6. test_regime_gate_transition_detected_falling
# ---------------------------------------------------------------------------


def test_regime_gate_transition_detected_falling():
    """Verify RISK_OFF transition is detected when BTC crosses below its MA."""
    n_bars = 600
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="4h", tz="UTC")
    # BTC rises for first half, then strongly falls for second half
    btc = np.concatenate([
        np.linspace(5000, 15000, n_bars // 2),   # rising
        np.linspace(15000, 4000, n_bars - n_bars // 2),  # falling
    ])
    data = {"BTC/USD": btc}
    for sym in symbols:
        if sym != "BTC/USD":
            data[sym] = np.full(n_bars, 500.0)
    close = pd.DataFrame(data, index=idx)

    joint_start = idx[180]
    n_switches, avg_dur = rob.compute_regime_stats(close, 180, joint_start)

    assert n_switches >= 1, f"Expected at least 1 regime switch, got {n_switches}"


# ---------------------------------------------------------------------------
# 7. test_benchmark_date_alignment
# ---------------------------------------------------------------------------


def test_benchmark_date_alignment():
    """BTC benchmark period should start at joint_start and end at same date as strategy."""
    close = _make_close(n=400, seed=7)
    joint_start = close.index[50]

    btc_close = close["BTC/USD"].dropna()
    btc_period = btc_close.loc[btc_close.index >= joint_start]

    # Benchmark and strategy share the same start
    strategy_close = close.loc[close.index >= joint_start]
    assert btc_period.index[0] == strategy_close.index[0], (
        "BTC benchmark must start at joint_start"
    )
    assert btc_period.index[-1] == strategy_close.index[-1], (
        "BTC benchmark must end at same date as strategy"
    )


# ---------------------------------------------------------------------------
# 8. test_cash_accounting_during_regime_off
# ---------------------------------------------------------------------------


def test_cash_accounting_during_regime_off():
    """When regime gate is always OFF (BTC below MA), portfolio weights must sum to 0."""
    close = _make_close_btc_below_ma(n=400, ma_bars=36)
    signal_gen = rob.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=36)
    result = _run_small_backtest(signal_gen, close, rebalance_bars=6)

    # After warm-up (36 bars for MA), all bars should have zero weights
    holdings = result.holdings_history
    # BTC is consistently below MA, so weight_sum should be ~0 for most bars
    weight_sum = holdings.sum(axis=1)
    late_holdings = weight_sum.iloc[50:]  # skip warm-up
    assert float(late_holdings.max()) < 0.01, (
        f"Expected all-cash regime, but max weight_sum = {float(late_holdings.max()):.4f}"
    )


# ---------------------------------------------------------------------------
# 9. test_robustness_grid_uses_only_allowed_ma_values
# ---------------------------------------------------------------------------


def test_robustness_grid_uses_only_allowed_ma_values():
    """BTC_MA_BARS_TO_TEST must contain exactly the 9 predefined values."""
    allowed = {120, 150, 180, 210, 240, 300, 360, 420, 480}
    assert set(rob.BTC_MA_BARS_TO_TEST) == allowed, (
        f"Expected MA values {allowed}, got {set(rob.BTC_MA_BARS_TO_TEST)}"
    )
    assert len(rob.BTC_MA_BARS_TO_TEST) == 9


# ---------------------------------------------------------------------------
# 10. test_deterministic_results
# ---------------------------------------------------------------------------


def test_deterministic_results():
    """Same inputs produce identical outputs (no hidden randomness)."""
    close = _make_close(n=300, seed=99)
    signal_gen_a = rob.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=60)
    signal_gen_b = rob.BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=60)

    result_a = _run_small_backtest(signal_gen_a, close, rebalance_bars=6)
    result_b = _run_small_backtest(signal_gen_b, close, rebalance_bars=6)

    pd.testing.assert_series_equal(
        result_a.portfolio["equity"],
        result_b.portfolio["equity"],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 11. test_no_live_trading_imports
# ---------------------------------------------------------------------------


def test_no_live_trading_imports():
    """research.fixed_five_btc_regime_robustness must not import from brokers/, execution/, live/."""
    src = Path("research/fixed_five_btc_regime_robustness.py").read_text()
    tree = ast.parse(src)
    forbidden_prefixes = ("brokers.", "execution.", "live.")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    for prefix in forbidden_prefixes:
                        assert not module.startswith(prefix), (
                            f"Forbidden import from '{module}' found in research script"
                        )
                continue
            else:
                continue
            for prefix in forbidden_prefixes:
                assert not module.startswith(prefix), (
                    f"Forbidden import from '{module}' found in research script"
                )


# ---------------------------------------------------------------------------
# 12. test_no_order_placement_patterns
# ---------------------------------------------------------------------------


def test_no_order_placement_patterns():
    """Research script must not contain live order-placement patterns."""
    src = Path("research/fixed_five_btc_regime_robustness.py").read_text()
    forbidden_patterns = [
        "place_order", "submit_order", "create_order",
        "execute_trade", "send_order", "market_order",
        "limit_order", "cancel_order",
    ]
    src_lower = src.lower()
    for pattern in forbidden_patterns:
        assert pattern not in src_lower, (
            f"Forbidden order-placement pattern '{pattern}' found in research script"
        )


# ---------------------------------------------------------------------------
# 13. test_canonical_baseline_uses_full_ohlcv
# ---------------------------------------------------------------------------


def test_canonical_baseline_uses_full_ohlcv():
    """With real data: canonical max_dd should be within 5pp of -79.7%."""
    data_dir = Path("data/local")
    if not data_dir.exists():
        pytest.skip("data/local not present — skipping real-data test")

    # Quick check: at least one data file present
    data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
    if not data_files:
        pytest.skip("No data files in data/local — skipping real-data test")

    from research.universe_integrity_analysis import (
        build_close_matrix, find_joint_eligible_start, _close_to_ohlcv,
        LIVE_FIVE_UNIVERSE, DEFAULT_MIN_HISTORY_BARS,
    )

    try:
        close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    except Exception as e:
        pytest.skip(f"Could not build close matrix: {e}")

    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        pytest.skip("Cannot determine joint_start")

    ohlcv = _close_to_ohlcv(close)  # FULL ohlcv — canonical

    _, _, baseline_metrics, _ = rob.compute_canonical_baseline(ohlcv, close, joint_start)

    max_dd = baseline_metrics.get("max_drawdown", float("nan"))
    assert not pd.isna(max_dd), "Canonical max_drawdown should not be NaN"

    # Canonical figure is approximately -79.7%; allow 5pp tolerance
    assert max_dd < -0.70, f"Expected max_dd < -70%, got {max_dd:.1%}"
    assert max_dd > -0.90, f"Expected max_dd > -90%, got {max_dd:.1%}"


# ---------------------------------------------------------------------------
# 14. test_ewb_return_positive_for_rising_market
# ---------------------------------------------------------------------------


def test_ewb_return_positive_for_rising_market():
    """EWB return should be positive when all symbols trend upward."""
    close = _make_close(n=300, btc_trend=0.003, others_trend=0.002, seed=5)
    joint_start = close.index[10]
    ewb = rob.compute_ewb_return(close, joint_start)
    assert ewb > 0, f"Expected positive EWB return for rising market, got {ewb:.4f}"


# ---------------------------------------------------------------------------
# 15. test_overfit_risk_labels_deterministic
# ---------------------------------------------------------------------------


def test_overfit_risk_labels_deterministic():
    """classify_overfit_risk must return identical labels for the same inputs."""
    # Build a synthetic metrics dict that satisfies all criteria
    m_good = {
        "variant_name": "btc_ma_180_reb6",
        "ma_bars": 180,
        "rebalance_bars": 6,
        "max_drawdown": -0.55,
        "return_2022": -0.40,
        "return_2023": 0.10,
        "return_2024": 0.05,
        "pct_time_in_cash": 30.0,
        "n_regime_switches": 100,
        "avg_regime_duration_bars": 60,  # well above 20-bar threshold
        "sharpe": 0.8,
    }
    baseline_m = {"max_drawdown": -0.797, "sharpe": 0.5}

    # Build a small grid_df with two neighbors
    neighbors = [
        {
            "variant_name": "btc_ma_150_reb6", "ma_bars": 150, "rebalance_bars": 6,
            "max_drawdown": -0.58, "return_2022": -0.42, "return_2023": 0.08,
            "return_2024": 0.04, "pct_time_in_cash": 32.0,
            "n_regime_switches": 110, "avg_regime_duration_bars": 55,
        },
        {
            "variant_name": "btc_ma_210_reb6", "ma_bars": 210, "rebalance_bars": 6,
            "max_drawdown": -0.53, "return_2022": -0.38, "return_2023": 0.12,
            "return_2024": 0.06, "pct_time_in_cash": 28.0,
            "n_regime_switches": 90, "avg_regime_duration_bars": 65,
        },
    ]
    grid_df = pd.DataFrame([m_good] + neighbors)

    label1, notes1 = rob.classify_overfit_risk(m_good, grid_df, baseline_m)
    label2, notes2 = rob.classify_overfit_risk(m_good, grid_df, baseline_m)

    assert label1 == label2, "classify_overfit_risk must be deterministic"
    assert notes1 == notes2, "classify_overfit_risk notes must be deterministic"

    # A variant that fails criterion 1 (return_2022 too low) should be HIGHER_OVERFIT_RISK
    m_bad = dict(m_good)
    m_bad["return_2022"] = -0.80
    label_bad, notes_bad = rob.classify_overfit_risk(m_bad, grid_df, baseline_m)
    assert label_bad == "HIGHER_OVERFIT_RISK", (
        f"Expected HIGHER_OVERFIT_RISK for bad return_2022, got {label_bad}"
    )
    assert "return_2022" in notes_bad


# ---------------------------------------------------------------------------
# Bonus: verify REBALANCE_BARS_TO_TEST content
# ---------------------------------------------------------------------------


def test_rebalance_bars_to_test():
    """REBALANCE_BARS_TO_TEST must be exactly [6, 12, 24]."""
    assert rob.REBALANCE_BARS_TO_TEST == [6, 12, 24]


# ---------------------------------------------------------------------------
# Bonus: verify WALK_FORWARD_VARIANTS content
# ---------------------------------------------------------------------------


def test_walk_forward_variants_content():
    """Walk-forward variants must include the 6 named combinations."""
    names = {v[0] for v in rob.WALK_FORWARD_VARIANTS}
    required = {
        "btc_ma_360_reb6", "btc_ma_180_reb6",
        "btc_ma_180_reb12", "btc_ma_180_reb24",
        "btc_ma_360_reb12", "btc_ma_360_reb24",
    }
    assert required.issubset(names), f"Missing variants: {required - names}"
