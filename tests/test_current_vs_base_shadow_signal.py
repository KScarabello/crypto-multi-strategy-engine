from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from research import current_vs_base_shadow_signal as shadow


def test_no_execute_orders_import_or_call() -> None:
    source = inspect.getsource(shadow)
    assert "execute_orders" not in source
    assert "live.execute_orders" not in source
    assert "subprocess.run" in source


def test_no_writes_to_old_repo(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None):
        seen["cwd"] = cwd
        seen["cmd"] = cmd

        class Result:
            returncode = 0
            stdout = json.dumps(
                {
                    "timestamp": "2026-06-10T00:00:00+00:00",
                    "data_fresh": True,
                    "is_rebalance_bar": True,
                    "target_weights": {"BTC/USD": 0.25, "ETH/USD": 0.25, "XRP/USD": 0.25},
                    "cash_weight": 0.25,
                    "total_risky_weight": 0.75,
                    "selected_symbols": ["BTC/USD", "ETH/USD", "XRP/USD"],
                    "warnings": [],
                    "blocked": False,
                }
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(shadow.subprocess, "run", fake_run)
    payload = shadow._current_bot_snapshot()
    assert seen["cwd"] == shadow.OLD_REPO_ROOT
    assert payload["blocked"] is False
    assert payload["cash_weight"] == 0.25


def test_base_target_weights_sum_to_one(monkeypatch) -> None:
    close = pd.DataFrame(
        {
            "BTC/USD": [100.0, 110.0, 120.0],
            "ETH/USD": [10.0, 11.0, 12.0],
            "XRP/USD": [1.0, 1.1, 1.2],
            "SOL/USD": [20.0, 21.0, 22.0],
            "AVAX/USD": [30.0, 31.0, 32.0],
        },
        index=pd.to_datetime(
            ["2026-06-10T00:00:00Z", "2026-06-10T04:00:00Z", "2026-06-10T08:00:00Z"],
            utc=True,
        ),
    )

    class FakeHoldings:
        holdings_history = pd.DataFrame({"BTC/USD": [0.0, 1.0, 1.0]}, index=close.index)

    class FakeBtcResult:
        pass

    fake_result = FakeBtcResult()
    fake_result.holdings_history = pd.DataFrame({"BTC/USD": [0.0, 1.0, 1.0]}, index=close.index)

    monkeypatch.setattr(shadow, "load_universe_close_prices", lambda **kwargs: close)
    monkeypatch.setattr(shadow, "run_btc_time_series_backtests", lambda **kwargs: (close[["BTC/USD"]], 2190, {"btc_time_series_momentum": fake_result}))

    class FakeBtcSignalGen:
        def __call__(self, close_matrix, timestamp):
            row = pd.Series(0.0, index=close_matrix.columns, dtype=float)
            row.loc["BTC/USD"] = 1.0
            return row

    class FakeCsSignalGen:
        def __call__(self, close_matrix, timestamp):
            row = pd.Series(0.0, index=close_matrix.columns, dtype=float)
            row.loc[["ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]] = 0.25
            return row

    monkeypatch.setattr(shadow, "build_btc_ts_momentum_signal_generator", lambda **kwargs: FakeBtcSignalGen())
    monkeypatch.setattr(shadow, "_rebalance_signal_generator", lambda **kwargs: FakeCsSignalGen())
    monkeypatch.setattr(shadow, "build_universal_eligibility_mask", lambda close_prices, min_required_lookback: pd.DataFrame(True, index=close.index, columns=close.columns))
    monkeypatch.setattr(shadow, "_is_data_fresh", lambda latest_timestamp, timeframe=shadow.BASE_TIMEFRAME: True)

    snapshot = shadow._base_snapshot()
    assert abs(sum(snapshot["target_weights"].values()) - 1.0) < 1e-9


def test_old_target_parser_using_fixture(monkeypatch) -> None:
    payload = {
        "timestamp": "2026-06-10T20:00:00+00:00",
        "data_fresh": True,
        "is_rebalance_bar": True,
        "target_weights": {"BTC/USD": 0.25, "ETH/USD": 0.25, "XRP/USD": 0.25},
        "cash_weight": 0.25,
        "total_risky_weight": 0.75,
        "selected_symbols": ["BTC/USD", "ETH/USD", "XRP/USD"],
        "warnings": [],
        "blocked": False,
    }

    monkeypatch.setattr(shadow, "_run_current_bot_preview", lambda data_dir=shadow.CURRENT_BOT_DATA_DIR: payload)
    snapshot = shadow._current_bot_snapshot()
    assert snapshot["timestamp"] == pd.Timestamp("2026-06-10T20:00:00+00:00")
    assert snapshot["cash_weight"] == 0.25
    assert snapshot["target_weights"]["BTC/USD"] == 0.25


def test_target_difference_calculation() -> None:
    current = {
        "strategy": "current_live_bot",
        "timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "data_fresh": True,
        "is_rebalance_point": True,
        "target_weights": {"BTC/USD": 0.25, "ETH/USD": 0.25, "XRP/USD": 0.25, "CASH": 0.25},
        "cash_weight": 0.25,
        "total_risky_weight": 0.75,
        "btc_exposure": 0.25,
        "alt_exposure": 0.5,
        "symbols": ["BTC/USD", "ETH/USD", "XRP/USD"],
        "warnings": [],
        "blocked": False,
    }
    base = {
        "strategy": "canonical_base_strategy",
        "timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "data_fresh": True,
        "is_rebalance_point": True,
        "btc_gate_on": True,
        "cs_gate_on": True,
        "target_weights": {"BTC/USD": 0.75, "SOL/USD": 0.25, "CASH": 0.0},
        "cash_weight": 0.0,
        "total_risky_weight": 1.0,
        "btc_exposure": 0.75,
        "alt_exposure": 0.25,
        "symbols": ["BTC/USD", "SOL/USD"],
        "warnings": [],
        "blocked": False,
    }

    result = shadow.compare_current_vs_base(current, base)
    assert abs(result.comparison_row["btc_exposure_difference"] - 0.5) < 1e-9
    assert abs(result.comparison_row["cash_difference"] - (-0.25)) < 1e-9
    assert "SOL/USD" in result.target_differences["symbol"].tolist()


def test_stale_data_warning() -> None:
    old_ts = pd.Timestamp("2020-01-01T00:00:00+00:00")
    assert shadow._is_data_fresh(old_ts, "4h") is False


def test_history_append_behavior(tmp_path) -> None:
    history_path = tmp_path / "history.csv"
    row = {
        "run_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "latest_data_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "current_data_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "base_data_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "current_data_fresh": True,
        "base_data_fresh": True,
        "current_is_rebalance_point": True,
        "base_is_rebalance_point": True,
        "base_btc_gate_on": True,
        "base_cs_gate_on": True,
        "current_btc_exposure": 0.25,
        "base_btc_exposure": 0.75,
        "btc_exposure_difference": 0.5,
        "current_alt_exposure": 0.5,
        "base_alt_exposure": 0.25,
        "alt_exposure_difference": -0.25,
        "current_cash_weight": 0.25,
        "base_cash_weight": 0.0,
        "cash_difference": -0.25,
        "current_symbols": "BTC/USD,ETH/USD",
        "base_symbols": "BTC/USD,SOL/USD",
        "current_only_symbols": "ETH/USD",
        "base_only_symbols": "SOL/USD",
        "current_blocked": False,
        "base_blocked": False,
        "warnings": "",
    }

    shadow.append_history(history_path, row)
    shadow.append_history(history_path, row)
    history = pd.read_csv(history_path)
    assert len(history) == 2


def test_blocked_old_bot_target_preview_handling(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("preview blocked")

    monkeypatch.setattr(shadow, "_run_current_bot_preview", boom)
    snapshot = shadow._current_bot_snapshot()
    assert snapshot["blocked"] is True
    assert snapshot["cash_weight"] == 1.0
