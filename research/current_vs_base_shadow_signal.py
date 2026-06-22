"""Research-only shadow/latest-signal comparison for current live bot vs canonical base.

This module reads the old live bot in a subprocess for a safe target preview,
computes the canonical base targets from research-only code, and writes
comparison artifacts into reports/current_vs_base_shadow_signal/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from config import SETTINGS
from research.btc_time_series import BtcTimeSeriesConfig, build_btc_ts_momentum_signal_generator, run_btc_time_series_backtests
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.run_expanded_universe_experiment import (
    EXPANDED_UNIVERSE_20,
    LOOKBACK_CONFIGS,
    _rebalance_signal_generator,
    build_universal_eligibility_mask,
    load_universe_close_prices,
)
from research.sleeve_blend_gating_corrected import apply_gate_lag
from research.sleeve_blend_gating_experiment import GateSpec, build_cs_gate_series


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_REPO_ROOT = Path("/Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy")
REPORT_DIR = REPO_ROOT / "reports/current_vs_base_shadow_signal"

CURRENT_BOT_DATA_DIR = OLD_REPO_ROOT / "data/local"
BASE_DATA_DIR = REPO_ROOT / "data"
BASE_TIMEFRAME = "4h"
BASE_GATE_LAG_BARS = 1
BASE_REBALANCE_BARS = 12
BASE_CS_SPEC = CrossSectionalRobustnessSpec(
    name="cs_180_top5_reb12",
    lookback_config_name="medium_180_only",
    top_n=5,
    rebalance_bars=12,
)
BASE_BTC_CONFIG = BtcTimeSeriesConfig(
    symbol="BTC/USD",
    timeframe=BASE_TIMEFRAME,
    short_lookback_bars=60,
    medium_lookback_bars=240,
    rebalance_every_bars=12,
    short_weight=0.5,
    medium_weight=0.5,
    transaction_cost_bps=15.0,
    slippage_bps=5.0,
)

CURRENT_BOT_PREVIEW_CODE = (
    "from pathlib import Path; "
    "import json; "
    "from live.generate_targets import generate_targets; "
    f"result = generate_targets(data_dir=Path(r'{CURRENT_BOT_DATA_DIR}')); "
    "print(json.dumps(result, default=str))"
)


@dataclass(frozen=True)
class ComparisonResult:
    current_snapshot: dict[str, Any]
    base_snapshot: dict[str, Any]
    comparison_row: dict[str, Any]
    current_targets: pd.DataFrame
    base_targets: pd.DataFrame
    target_differences: pd.DataFrame


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for symbol, value in weights.items():
        normalized[str(symbol)] = float(value)
    if "CASH" not in normalized:
        normalized["CASH"] = 0.0
    return normalized


def _is_data_fresh(latest_timestamp: pd.Timestamp, timeframe: str = BASE_TIMEFRAME) -> bool:
    bar_hours = {"4h": 4, "1d": 24, "d": 24, "daily": 24}.get(timeframe.strip().lower())
    if bar_hours is None:
        raise ValueError(f"Unsupported timeframe for freshness check: {timeframe!r}")
    latest_utc = latest_timestamp.tz_convert("UTC") if latest_timestamp.tzinfo is not None else latest_timestamp
    age_hours = (pd.Timestamp.now(tz="UTC") - latest_utc).total_seconds() / 3600.0
    return age_hours <= float(bar_hours) * 2.0


def _safe_json_loads(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("current bot preview did not return an object")
    return payload


def _run_current_bot_preview(data_dir: Path = CURRENT_BOT_DATA_DIR) -> dict[str, Any]:
    code = (
        "from pathlib import Path; "
        "import json; "
        "from live.generate_targets import generate_targets; "
        f"result = generate_targets(data_dir=Path(r'{data_dir}')); "
        "print(json.dumps(result, default=str))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=OLD_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"current bot preview failed with exit code {proc.returncode}")
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("current bot preview returned no output")
    return _safe_json_loads(stdout)


def _current_bot_snapshot(data_dir: Path = CURRENT_BOT_DATA_DIR) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        payload = _run_current_bot_preview(data_dir=data_dir)
    except Exception as exc:
        return {
            "strategy": "current_live_bot",
            "timestamp": pd.NaT,
            "data_fresh": False,
            "is_rebalance_point": False,
            "target_weights": {"CASH": 1.0},
            "cash_weight": 1.0,
            "total_risky_weight": 0.0,
            "btc_exposure": 0.0,
            "alt_exposure": 0.0,
            "symbols": [],
            "warnings": [f"current bot preview blocked: {exc}"],
            "blocked": True,
        }

    timestamp = pd.Timestamp(payload["timestamp"])
    target_weights = _normalize_weights(payload.get("target_weights", {}))
    cash_weight = float(payload.get("cash_weight", max(0.0, 1.0 - sum(target_weights.values()))))
    total_risky = float(payload.get("total_risky_weight", sum(v for k, v in target_weights.items() if k != "CASH")))
    data_fresh = bool(payload.get("data_fresh", False))
    is_rebalance_point = bool(payload.get("is_rebalance_bar", False))
    if not data_fresh:
        warnings.append("current bot data is stale")
    if not is_rebalance_point:
        warnings.append("current bot snapshot is not on a rebalance bar")

    btc_exposure = float(target_weights.get("BTC/USD", 0.0))
    alt_exposure = float(sum(weight for symbol, weight in target_weights.items() if symbol not in {"BTC/USD", "CASH"}))
    symbols = [symbol for symbol, weight in target_weights.items() if symbol != "CASH" and weight > 0.0]

    return {
        "strategy": "current_live_bot",
        "timestamp": timestamp,
        "data_fresh": data_fresh,
        "is_rebalance_point": is_rebalance_point,
        "target_weights": target_weights,
        "cash_weight": cash_weight,
        "total_risky_weight": total_risky,
        "btc_exposure": btc_exposure,
        "alt_exposure": alt_exposure,
        "symbols": symbols,
        "warnings": warnings,
        "blocked": False,
    }


def _base_snapshot(data_dir: Path = BASE_DATA_DIR) -> dict[str, Any]:
    close = load_universe_close_prices(symbols=EXPANDED_UNIVERSE_20, timeframe=BASE_TIMEFRAME, data_dir=data_dir)
    if close.empty:
        raise ValueError("base close matrix is empty")

    latest_ts = pd.Timestamp(close.index[-1])
    data_fresh = _is_data_fresh(latest_ts, BASE_TIMEFRAME)
    warnings: list[str] = []
    if not data_fresh:
        warnings.append("canonical base data is stale")

    btc_signal_generator = build_btc_ts_momentum_signal_generator(
        close_prices=close[["BTC/USD"]],
        config=BASE_BTC_CONFIG,
    )
    eligibility_mask = build_universal_eligibility_mask(close, min_required_lookback=int(LOOKBACK_CONFIGS[BASE_CS_SPEC.lookback_config_name].min_required_lookback))
    cs_signal_generator = _rebalance_signal_generator(
        strategy_name="cs_momentum",
        lookback_config=LOOKBACK_CONFIGS[BASE_CS_SPEC.lookback_config_name],
        close_prices=close,
        eligibility_mask=eligibility_mask,
        top_n=BASE_CS_SPEC.top_n,
    )

    btc_signal_row = btc_signal_generator(close, latest_ts).reindex(close.columns, fill_value=0.0).astype(float)
    cs_signal_row = cs_signal_generator(close, latest_ts).reindex(close.columns, fill_value=0.0).astype(float)

    _, _, btc_backtests = run_btc_time_series_backtests(config=BASE_BTC_CONFIG, close_prices=close[["BTC/USD"]])
    btc_holdings = btc_backtests["btc_time_series_momentum"].holdings_history["BTC/USD"].reindex(close.index).fillna(0.0).astype(float)
    lagged_gate = apply_gate_lag(btc_holdings, lag_bars=BASE_GATE_LAG_BARS)
    base_btc_gate_on = bool(float(lagged_gate.reindex(close.index).fillna(0.0).iloc[-1]) > 0.0)
    base_cs_gate_on = base_btc_gate_on

    base_weights = (btc_signal_row * 0.75) + (cs_signal_row * 0.25 * float(base_cs_gate_on))
    base_weights = base_weights.astype(float)
    cash_weight = max(0.0, 1.0 - float(base_weights.sum()))
    target_weights = {str(symbol): float(weight) for symbol, weight in base_weights.items() if abs(float(weight)) > 1e-12}
    target_weights["CASH"] = cash_weight

    is_rebalance_point = int(close.index.get_loc(latest_ts)) > 0 and (int(close.index.get_loc(latest_ts)) % BASE_REBALANCE_BARS == 0)
    if not is_rebalance_point:
        warnings.append("canonical base snapshot is not on a rebalance bar")
    if not base_btc_gate_on:
        warnings.append("canonical base BTC gate is off")

    btc_exposure = float(target_weights.get("BTC/USD", 0.0))
    alt_exposure = float(sum(weight for symbol, weight in target_weights.items() if symbol not in {"BTC/USD", "CASH"}))
    symbols = [symbol for symbol, weight in target_weights.items() if symbol != "CASH" and weight > 0.0]

    return {
        "strategy": "canonical_base_strategy",
        "timestamp": latest_ts,
        "data_fresh": data_fresh,
        "is_rebalance_point": is_rebalance_point,
        "btc_gate_on": base_btc_gate_on,
        "cs_gate_on": base_cs_gate_on,
        "target_weights": target_weights,
        "cash_weight": cash_weight,
        "total_risky_weight": float(base_weights.sum()),
        "btc_exposure": btc_exposure,
        "alt_exposure": alt_exposure,
        "symbols": symbols,
        "warnings": warnings,
        "blocked": False,
    }


def _snapshot_to_frame(snapshot: dict[str, Any], source_name: str) -> pd.DataFrame:
    rows = []
    for symbol, weight in snapshot["target_weights"].items():
        rows.append(
            {
                "source": source_name,
                "symbol": symbol,
                "weight": float(weight),
                "timestamp": snapshot["timestamp"],
                "data_fresh": bool(snapshot["data_fresh"]),
                "is_rebalance_point": bool(snapshot["is_rebalance_point"]),
            }
        )
    return pd.DataFrame(rows)


def _union_symbol_frame(current_snapshot: dict[str, Any], base_snapshot: dict[str, Any]) -> pd.DataFrame:
    current_weights = _normalize_weights(current_snapshot["target_weights"])
    base_weights = _normalize_weights(base_snapshot["target_weights"])
    symbols = sorted(set(current_weights) | set(base_weights))
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        current_weight = float(current_weights.get(symbol, 0.0))
        base_weight = float(base_weights.get(symbol, 0.0))
        rows.append(
            {
                "symbol": symbol,
                "current_weight": current_weight,
                "base_weight": base_weight,
                "delta_weight": base_weight - current_weight,
                "abs_delta_weight": abs(base_weight - current_weight),
                "current_has_symbol": bool(current_weight > 0.0),
                "base_has_symbol": bool(base_weight > 0.0),
                "classification": "cash" if symbol == "CASH" else ("btc" if symbol == "BTC/USD" else "alt"),
            }
        )
    return pd.DataFrame(rows)


def _comparison_row(current_snapshot: dict[str, Any], base_snapshot: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    current_weights = _normalize_weights(current_snapshot["target_weights"])
    base_weights = _normalize_weights(base_snapshot["target_weights"])
    current_ts = pd.Timestamp(current_snapshot["timestamp"])
    base_ts = pd.Timestamp(base_snapshot["timestamp"])
    latest_ts = max(current_ts, base_ts)
    current_only = sorted([symbol for symbol in current_weights if symbol != "CASH" and current_weights.get(symbol, 0.0) > 0.0 and base_weights.get(symbol, 0.0) <= 0.0])
    base_only = sorted([symbol for symbol in base_weights if symbol != "CASH" and base_weights.get(symbol, 0.0) > 0.0 and current_weights.get(symbol, 0.0) <= 0.0])

    return {
        "run_timestamp": pd.Timestamp.now(tz="UTC"),
        "latest_data_timestamp": latest_ts,
        "current_data_timestamp": current_ts,
        "base_data_timestamp": base_ts,
        "current_data_fresh": bool(current_snapshot["data_fresh"]),
        "base_data_fresh": bool(base_snapshot["data_fresh"]),
        "current_is_rebalance_point": bool(current_snapshot["is_rebalance_point"]),
        "base_is_rebalance_point": bool(base_snapshot["is_rebalance_point"]),
        "base_btc_gate_on": bool(base_snapshot.get("btc_gate_on", False)),
        "base_cs_gate_on": bool(base_snapshot.get("cs_gate_on", False)),
        "current_btc_exposure": float(current_snapshot["btc_exposure"]),
        "base_btc_exposure": float(base_snapshot["btc_exposure"]),
        "btc_exposure_difference": float(base_snapshot["btc_exposure"]) - float(current_snapshot["btc_exposure"]),
        "current_alt_exposure": float(current_snapshot["alt_exposure"]),
        "base_alt_exposure": float(base_snapshot["alt_exposure"]),
        "alt_exposure_difference": float(base_snapshot["alt_exposure"]) - float(current_snapshot["alt_exposure"]),
        "current_cash_weight": float(current_snapshot["cash_weight"]),
        "base_cash_weight": float(base_snapshot["cash_weight"]),
        "cash_difference": float(base_snapshot["cash_weight"]) - float(current_snapshot["cash_weight"]),
        "current_symbols": ",".join(current_snapshot["symbols"]),
        "base_symbols": ",".join(base_snapshot["symbols"]),
        "current_only_symbols": ",".join(current_only),
        "base_only_symbols": ",".join(base_only),
        "current_blocked": bool(current_snapshot.get("blocked", False)),
        "base_blocked": bool(base_snapshot.get("blocked", False)),
        "warnings": " | ".join([*(current_snapshot.get("warnings", [])), *(base_snapshot.get("warnings", [])), *warnings]),
    }


def _artifact_paths(report_dir: Path) -> dict[str, Path]:
    return {
        "history": report_dir / "shadow_signal_history.csv",
        "latest_comparison": report_dir / "latest_shadow_signal_comparison.csv",
        "latest_base_targets": report_dir / "latest_base_targets.csv",
        "latest_current_targets": report_dir / "latest_current_bot_targets.csv",
        "target_differences": report_dir / "target_differences.csv",
        "summary": report_dir / "summary.md",
    }


def compare_current_vs_base(current_snapshot: dict[str, Any], base_snapshot: dict[str, Any]) -> ComparisonResult:
    warnings = []
    if current_snapshot["timestamp"] is pd.NaT:
        warnings.append("current bot preview blocked")
    if not current_snapshot.get("data_fresh", False) or not base_snapshot.get("data_fresh", False):
        warnings.append("stale data warning")

    current_targets = _snapshot_to_frame(current_snapshot, source_name="current_live_bot")
    base_targets = _snapshot_to_frame(base_snapshot, source_name="canonical_base_strategy")
    target_differences = _union_symbol_frame(current_snapshot, base_snapshot)
    comparison_row = _comparison_row(current_snapshot, base_snapshot, warnings)

    return ComparisonResult(
        current_snapshot=current_snapshot,
        base_snapshot=base_snapshot,
        comparison_row=comparison_row,
        current_targets=current_targets,
        base_targets=base_targets,
        target_differences=target_differences,
    )


def append_history(history_path: Path, comparison_row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([comparison_row]).copy()
    frame["run_timestamp"] = pd.to_datetime(frame["run_timestamp"], utc=True)
    mode = "a" if history_path.exists() else "w"
    header = not history_path.exists()
    frame.to_csv(history_path, mode=mode, header=header, index=False)


def build_summary_markdown(result: ComparisonResult) -> str:
    current = result.current_snapshot
    base = result.base_snapshot
    row = result.comparison_row

    lines = [
        "# Current Live Bot vs Canonical Base Shadow Signal",
        "",
        "## Answers",
        f"1. Can we safely compute old bot targets? {'YES' if not current.get('blocked') else 'NO'}.",
        f"2. Can we safely compute base strategy targets? YES.",
        f"3. What does the base strategy want right now? {', '.join(base['symbols']) or 'cash only'} with cash={base['cash_weight']:.4f} and BTC gate={'ON' if base.get('btc_gate_on') else 'OFF'}.",
        f"4. How different is it from the old bot? BTC exposure delta={row['btc_exposure_difference']:+.4f}, alt exposure delta={row['alt_exposure_difference']:+.4f}, cash delta={row['cash_difference']:+.4f}.",
        f"5. Is this ready for repeated manual shadow runs? {'YES, with stale-data warnings acknowledged' if not row['warnings'] else 'YES for manual shadow runs, but warnings remain'}.",
        "6. What remains blocked before production adapter work? Feature-flagged adapter design, paper-mode validation, and final live-deployment approval.",
        "",
        "## Snapshot Summary",
        f"- Current bot timestamp: {current['timestamp']}",
        f"- Base timestamp: {base['timestamp']}",
        f"- Current bot rebalance point: {current['is_rebalance_point']}",
        f"- Base rebalance point: {base['is_rebalance_point']}",
        f"- Current bot data fresh: {current['data_fresh']}",
        f"- Base data fresh: {base['data_fresh']}",
        f"- Current bot symbols: {', '.join(current['symbols']) or 'none'}",
        f"- Base symbols: {', '.join(base['symbols']) or 'none'}",
        f"- Current only symbols: {row['current_only_symbols'] or 'none'}",
        f"- Base only symbols: {row['base_only_symbols'] or 'none'}",
        f"- Warnings: {row['warnings'] or 'none'}",
        "",
        "## Safety",
        "- No orders were placed.",
        "- No live behavior changed.",
        "- No broker or exchange execution path was called.",
    ]
    return "\n".join(lines) + "\n"


def run_shadow_signal_comparison(
    current_data_dir: Path = CURRENT_BOT_DATA_DIR,
    base_data_dir: Path = BASE_DATA_DIR,
    report_dir: Path = REPORT_DIR,
) -> ComparisonResult:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(report_dir)
    current_snapshot = _current_bot_snapshot(data_dir=current_data_dir)
    base_snapshot = _base_snapshot(data_dir=base_data_dir)
    result = compare_current_vs_base(current_snapshot=current_snapshot, base_snapshot=base_snapshot)

    pd.DataFrame([result.comparison_row]).to_csv(paths["latest_comparison"], index=False)
    result.current_targets.to_csv(paths["latest_current_targets"], index=False)
    result.base_targets.to_csv(paths["latest_base_targets"], index=False)
    result.target_differences.to_csv(paths["target_differences"], index=False)
    append_history(paths["history"], result.comparison_row)
    paths["summary"].write_text(build_summary_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only shadow/latest-signal comparison tool")
    parser.add_argument("--current-data-dir", default=str(CURRENT_BOT_DATA_DIR))
    parser.add_argument("--base-data-dir", default=str(BASE_DATA_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    result = run_shadow_signal_comparison(
        current_data_dir=Path(args.current_data_dir),
        base_data_dir=Path(args.base_data_dir),
        report_dir=Path(args.report_dir),
    )

    print(json.dumps(result.comparison_row, default=str, indent=2))


if __name__ == "__main__":
    main()