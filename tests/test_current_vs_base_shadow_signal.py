from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from research import current_vs_base_shadow_signal as shadow


def _write_ohlcv_csv(path: Path, timestamps: list[str]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * len(timestamps),
            "high": [1.0] * len(timestamps),
            "low": [1.0] * len(timestamps),
            "close": [1.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
            "symbol": ["BTC/USD"] * len(timestamps),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_no_execute_orders_import_or_call() -> None:
    source = inspect.getsource(shadow)
    assert "execute_orders" not in source
    assert "live.execute_orders" not in source
    assert "subprocess.run" in source


def test_no_mac_specific_repo_path_in_logic() -> None:
    source = inspect.getsource(shadow)
    assert "/Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy" not in source


def test_current_repo_dir_explicit_resolution() -> None:
    repo_dir = Path("/home/trader/repos/crypto-momentum-strategy")
    data_dir = repo_dir / "data/local"
    resolved = shadow._resolve_current_repo_dir(current_repo_dir=repo_dir, current_data_dir=data_dir)
    assert resolved == repo_dir


def test_current_repo_dir_inferred_from_current_data_dir() -> None:
    data_dir = Path("/home/trader/repos/crypto-momentum-strategy/data/local")
    inferred = shadow._resolve_current_repo_dir(current_repo_dir=None, current_data_dir=data_dir)
    assert inferred == Path("/home/trader/repos/crypto-momentum-strategy")


def test_current_repo_dir_inferred_for_mac_style_path() -> None:
    data_dir = Path("/Users/example/Desktop/Repos/crypto/crypto-momentum-strategy/data/local")
    inferred = shadow._resolve_current_repo_dir(current_repo_dir=None, current_data_dir=data_dir)
    assert inferred == Path("/Users/example/Desktop/Repos/crypto/crypto-momentum-strategy")


def test_current_repo_dir_inference_requires_expected_layout() -> None:
    data_dir = Path("/home/trader/data")
    try:
        shadow._resolve_current_repo_dir(current_repo_dir=None, current_data_dir=data_dir)
    except ValueError as exc:
        assert "--current-repo-dir" in str(exc)
    else:
        raise AssertionError("expected ValueError when repo dir cannot be inferred")


def test_no_writes_to_old_repo(monkeypatch) -> None:
    seen = {}
    linux_repo = Path("/home/trader/repos/crypto-momentum-strategy")

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
    payload = shadow._current_bot_snapshot(current_repo_dir=linux_repo, data_dir=linux_repo / "data/local")
    assert seen["cwd"] == linux_repo
    assert payload["blocked"] is False
    assert payload["cash_weight"] == 0.25


def test_linux_style_repo_path_not_blocked(monkeypatch) -> None:
    linux_repo = Path("/home/trader/repos/crypto-momentum-strategy")

    def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None):
        class Result:
            returncode = 0
            stdout = json.dumps(
                {
                    "timestamp": "2026-06-22T04:00:00+00:00",
                    "data_fresh": True,
                    "is_rebalance_bar": True,
                    "target_weights": {"BTC/USD": 0.25, "ETH/USD": 0.25, "SOL/USD": 0.25, "CASH": 0.25},
                    "cash_weight": 0.25,
                    "total_risky_weight": 0.75,
                    "selected_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
                    "warnings": [],
                    "blocked": False,
                }
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(shadow.subprocess, "run", fake_run)
    snapshot = shadow._current_bot_snapshot(
        current_repo_dir=linux_repo,
        data_dir=linux_repo / "data/local",
    )
    assert snapshot["blocked"] is False
    assert snapshot["source_repo_path"] == str(linux_repo)


def test_data_freshness_calculation(tmp_path) -> None:
    data_dir = tmp_path / "data"
    csv_path = data_dir / "btc-usd_4h.csv"
    _write_ohlcv_csv(
        csv_path,
        [
            "2026-06-21T20:00:00+00:00",
            "2026-06-22T00:00:00+00:00",
        ],
    )
    audit = shadow._audit_data_source("TEST_SOURCE", data_dir, ["BTC/USD"])
    assert bool(audit.loc[0, "fresh_enough"]) is True
    assert audit.loc[0, "warning_reason"] == ""
    assert audit.loc[0, "expected_bar_interval_hours"] == 4


def test_stale_data_warning_behavior() -> None:
    current = {
        "strategy": "current_live_bot",
        "timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "data_fresh": False,
        "is_rebalance_point": False,
        "target_weights": {"BTC/USD": 0.25, "ETH/USD": 0.25, "XRP/USD": 0.25, "CASH": 0.25},
        "cash_weight": 0.25,
        "total_risky_weight": 0.75,
        "btc_exposure": 0.25,
        "alt_exposure": 0.5,
        "symbols": ["BTC/USD", "ETH/USD", "XRP/USD"],
        "warnings": ["current bot data is stale"],
        "blocked": False,
        "source_label": shadow.CURRENT_BOT_PREVIEW_LABEL,
        "source_path": str(shadow.CURRENT_BOT_DATA_DIR),
    }
    old_research = {
        "strategy": "old_strategy_on_research_data_preview",
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
        "source_label": shadow.OLD_RESEARCH_PREVIEW_LABEL,
        "source_path": str(shadow.OLD_RESEARCH_DATA_DIR),
    }
    base = {
        "strategy": "canonical_base_strategy",
        "timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "data_fresh": False,
        "is_rebalance_point": False,
        "btc_gate_on": False,
        "cs_gate_on": False,
        "target_weights": {"CASH": 1.0},
        "cash_weight": 1.0,
        "total_risky_weight": 0.0,
        "btc_exposure": 0.0,
        "alt_exposure": 0.0,
        "symbols": [],
        "warnings": ["canonical base data is stale"],
        "blocked": False,
        "source_label": shadow.CANONICAL_BASE_LABEL,
        "source_path": str(shadow.BASE_DATA_DIR),
    }

    result = shadow.compare_current_vs_base(current, old_research, base)
    assert "stale data warning" in result.comparison_row["warnings"]
    assert result.comparison_row["blocker_status"] == "WARNINGS_ONLY"
    assert "old_research_preview_target_weights" in result.comparison_row


def test_source_selection_logic() -> None:
    freshness = pd.DataFrame(
        [
            {"source_label": "ACTUAL_OLD_BOT_DATA", "source_path": "a", "latest_timestamp": pd.Timestamp("2026-06-01T00:00:00+00:00"), "fresh_enough": False},
            {"source_label": "ACTUAL_OLD_BOT_DATA", "source_path": "a", "latest_timestamp": pd.Timestamp("2026-06-01T00:00:00+00:00"), "fresh_enough": False},
            {"source_label": "OLD_STRATEGY_ON_RESEARCH_DATA_PREVIEW", "source_path": "b", "latest_timestamp": pd.Timestamp("2026-06-21T20:00:00+00:00"), "fresh_enough": True},
            {"source_label": "OLD_STRATEGY_ON_RESEARCH_DATA_PREVIEW", "source_path": "b", "latest_timestamp": pd.Timestamp("2026-06-21T20:00:00+00:00"), "fresh_enough": True},
            {"source_label": "CANONICAL_BASE_TARGET", "source_path": "c", "latest_timestamp": pd.Timestamp("2026-06-21T20:00:00+00:00"), "fresh_enough": True},
        ]
    )
    selection = shadow._best_source_from_freshness(freshness)
    assert selection["preferred_old_strategy_preview_source"] == "OLD_STRATEGY_ON_RESEARCH_DATA_PREVIEW"
    assert selection["actual_old_bot_target_source"]["all_fresh"] is False


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

    monkeypatch.setattr(
        shadow,
        "_run_current_bot_preview",
        lambda current_repo_dir, data_dir=shadow.CURRENT_BOT_DATA_DIR: payload,
    )
    snapshot = shadow._current_bot_snapshot(
        current_repo_dir=Path("/home/trader/repos/crypto-momentum-strategy"),
        data_dir=Path("/home/trader/repos/crypto-momentum-strategy/data/local"),
    )
    assert snapshot["timestamp"] == pd.Timestamp("2026-06-10T20:00:00+00:00")
    assert snapshot["cash_weight"] == 0.25
    assert snapshot["target_weights"]["BTC/USD"] == 0.25


def test_old_strategy_on_research_data_preview(monkeypatch) -> None:
    payload = {
        "timestamp": "2026-06-22T00:00:00+00:00",
        "data_fresh": True,
        "is_rebalance_bar": True,
        "target_weights": {"BTC/USD": 0.25, "SOL/USD": 0.25, "XRP/USD": 0.25},
        "cash_weight": 0.25,
        "total_risky_weight": 0.75,
        "selected_symbols": ["BTC/USD", "SOL/USD", "XRP/USD"],
        "warnings": [],
        "blocked": False,
    }

    monkeypatch.setattr(
        shadow,
        "_run_current_bot_preview",
        lambda current_repo_dir, data_dir=shadow.OLD_RESEARCH_DATA_DIR: payload,
    )
    snapshot = shadow._old_strategy_on_research_data_snapshot(
        current_repo_dir=Path("/home/trader/repos/crypto-momentum-strategy"),
    )
    assert snapshot["source_label"] == shadow.OLD_RESEARCH_PREVIEW_LABEL
    assert snapshot["target_weights"]["SOL/USD"] == 0.25


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
        "source_label": shadow.CURRENT_BOT_PREVIEW_LABEL,
        "source_path": str(shadow.CURRENT_BOT_DATA_DIR),
    }
    old_research = {
        "strategy": "old_strategy_on_research_data_preview",
        "timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "data_fresh": True,
        "is_rebalance_point": True,
        "target_weights": {"BTC/USD": 0.25, "SOL/USD": 0.25, "XRP/USD": 0.25, "CASH": 0.25},
        "cash_weight": 0.25,
        "total_risky_weight": 0.75,
        "btc_exposure": 0.25,
        "alt_exposure": 0.5,
        "symbols": ["BTC/USD", "SOL/USD", "XRP/USD"],
        "warnings": [],
        "blocked": False,
        "source_label": shadow.OLD_RESEARCH_PREVIEW_LABEL,
        "source_path": str(shadow.OLD_RESEARCH_DATA_DIR),
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
        "source_label": shadow.CANONICAL_BASE_LABEL,
        "source_path": str(shadow.BASE_DATA_DIR),
    }

    result = shadow.compare_current_vs_base(current, old_research, base)
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
        "current_source_label": shadow.CURRENT_BOT_PREVIEW_LABEL,
        "current_source_path": str(shadow.CURRENT_BOT_DATA_DIR),
        "old_research_source_label": shadow.OLD_RESEARCH_PREVIEW_LABEL,
        "old_research_source_path": str(shadow.OLD_RESEARCH_DATA_DIR),
        "old_research_preview_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "old_research_preview_target_weights": json.dumps({"BTC/USD": 0.25, "SOL/USD": 0.25, "XRP/USD": 0.25, "CASH": 0.25}),
        "old_research_preview_btc_exposure": 0.25,
        "old_research_preview_alt_exposure": 0.5,
        "old_research_preview_cash_weight": 0.25,
        "old_research_preview_blocked": False,
        "base_source_label": shadow.CANONICAL_BASE_LABEL,
        "base_source_path": str(shadow.BASE_DATA_DIR),
        "blocker_status": "CLEAR",
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
    snapshot = shadow._current_bot_snapshot(
        current_repo_dir=Path("/home/trader/repos/crypto-momentum-strategy"),
        data_dir=Path("/home/trader/repos/crypto-momentum-strategy/data/local"),
    )
    assert snapshot["blocked"] is True
    assert snapshot["cash_weight"] == 1.0


def test_history_append_includes_sources(tmp_path) -> None:
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
        "current_source_label": shadow.CURRENT_BOT_PREVIEW_LABEL,
        "current_source_path": str(shadow.CURRENT_BOT_DATA_DIR),
        "old_research_source_label": shadow.OLD_RESEARCH_PREVIEW_LABEL,
        "old_research_source_path": str(shadow.OLD_RESEARCH_DATA_DIR),
        "old_research_preview_timestamp": pd.Timestamp("2026-06-10T20:00:00+00:00"),
        "old_research_preview_target_weights": json.dumps({"BTC/USD": 0.25, "SOL/USD": 0.25, "XRP/USD": 0.25, "CASH": 0.25}),
        "old_research_preview_btc_exposure": 0.25,
        "old_research_preview_alt_exposure": 0.5,
        "old_research_preview_cash_weight": 0.25,
        "old_research_preview_blocked": False,
        "base_source_label": shadow.CANONICAL_BASE_LABEL,
        "base_source_path": str(shadow.BASE_DATA_DIR),
        "blocker_status": "CLEAR",
        "warnings": "",
    }

    shadow.append_history(history_path, row)
    history = pd.read_csv(history_path)
    assert len(history) == 1
    assert "current_source_label" in history.columns
    assert "old_research_source_label" in history.columns
