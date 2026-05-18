"""Run expanded-universe strategy experiments on local 4h OHLCV data.

This research runner intentionally stays inside the data/backtest surface.
It reuses the existing time-series strategy helpers, cross-sectional momentum
strategy class, and backtest engine while enforcing a shared universal
eligibility layer in the runner itself.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.signals import calculate_momentum_signal, calculate_overextension_signal
from research.strategy_variants import (
    momentum_with_entry_filter_and_exit_signal_weights,
    momentum_with_entry_filter_weights,
    momentum_with_exit_signal_weights,
    short_term_reversal_weights,
    time_series_momentum_weights,
)
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy


LOGGER = logging.getLogger(__name__)

EXPANDED_UNIVERSE_20: tuple[str, ...] = (
    "AAVE/USD",
    "ADA/USD",
    "APT/USD",
    "ARB/USD",
    "ATOM/USD",
    "AVAX/USD",
    "BCH/USD",
    "BTC/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "INJ/USD",
    "LINK/USD",
    "LTC/USD",
    "NEAR/USD",
    "OP/USD",
    "POL/USD",
    "SOL/USD",
    "UNI/USD",
    "XRP/USD",
)

DEFAULT_OUTPUT_PATH = Path("research/results/expanded_universe_experiment_metrics.csv")


@dataclass(frozen=True)
class LookbackConfig:
    """Lookback windows shared across strategy variants."""

    name: str
    momentum_windows: tuple[int, ...]
    momentum_weights: tuple[float, ...]
    reversal_windows: tuple[int, ...]
    reversal_weights: tuple[float, ...]

    @property
    def overextension_window(self) -> int:
        return min(self.momentum_windows)

    @property
    def min_required_lookback(self) -> int:
        return max((*self.momentum_windows, *self.reversal_windows))


LOOKBACK_CONFIGS: dict[str, LookbackConfig] = {
    "short_42_only": LookbackConfig(
        name="short_42_only",
        momentum_windows=(42,),
        momentum_weights=(1.0,),
        reversal_windows=(42,),
        reversal_weights=(1.0,),
    ),
    "medium_180_only": LookbackConfig(
        name="medium_180_only",
        momentum_windows=(180,),
        momentum_weights=(1.0,),
        reversal_windows=(180,),
        reversal_weights=(1.0,),
    ),
    "short_42_medium_180": LookbackConfig(
        name="short_42_medium_180",
        momentum_windows=(42, 180),
        momentum_weights=(0.5, 0.5),
        reversal_windows=(42, 180),
        reversal_weights=(0.5, 0.5),
    ),
    "short_42_medium_180_long_360": LookbackConfig(
        name="short_42_medium_180_long_360",
        momentum_windows=(42, 180, 360),
        momentum_weights=(0.4, 0.4, 0.2),
        reversal_windows=(42, 180, 360),
        reversal_weights=(0.4, 0.4, 0.2),
    ),
}

DEFAULT_REBALANCE_BARS: tuple[int, ...] = (6, 12, 42)
DEFAULT_TOP_NS: tuple[int, ...] = (3, 5, 8)
DEFAULT_COST_CONFIGS: tuple[tuple[int, int], ...] = ((10, 5), (20, 10))
STATEFUL_TS_STRATEGIES = {"ts_momentum_exit_signal", "ts_momentum_entry_exit"}
STATEFUL_TS_STRATEGY_ORDER: tuple[str, ...] = (
    "ts_momentum_exit_signal",
    "ts_momentum_entry_exit",
)
DEFAULT_FIRST_PASS_STRATEGIES: tuple[str, ...] = (
    "ts_momentum",
    "ts_reversal",
    "ts_momentum_entry_filter",
    "cs_momentum",
)
SMOKE_WINDOW_ROWS = 1500


def configure_logging() -> None:
    """Configure research logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def bars_per_year_for_timeframe(timeframe: str) -> int:
    """Return a simple annualization basis for crypto timeframes."""
    key = timeframe.strip().lower()
    if key == "4h":
        return 365 * 6
    if key in {"1d", "d", "daily"}:
        return 365
    raise ValueError(f"Unsupported timeframe for annualization: {timeframe}")


def symbol_to_local_filename(symbol: str, timeframe: str) -> str:
    """Convert BTC/USD and 4h into btc-usd_4h.csv."""
    return symbol.strip().lower().replace("/", "-").replace(" ", "-") + f"_{timeframe}.csv"


def _validate_close_prices(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a wide close-price matrix."""
    if not isinstance(close_prices, pd.DataFrame):
        raise TypeError("close_prices must be a pandas DataFrame")
    if close_prices.empty:
        raise ValueError("close_prices is empty")
    if close_prices.columns.empty:
        raise ValueError("close_prices must have at least one symbol column")

    clean = close_prices.sort_index().copy()
    clean.index = pd.to_datetime(clean.index, utc=True)
    clean = clean.astype(float)
    return clean


def close_prices_to_ohlcv(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide close-price matrix into long-format OHLCV rows."""
    clean = _validate_close_prices(close_prices)
    long_rows: list[pd.DataFrame] = []

    for symbol in clean.columns:
        series = clean[symbol].dropna()
        if series.empty:
            continue
        frame = pd.DataFrame(
            {
                "timestamp": series.index,
                "open": series.values,
                "high": series.values,
                "low": series.values,
                "close": series.values,
                "volume": 1.0,
                "symbol": symbol,
            }
        )
        long_rows.append(frame)

    if not long_rows:
        raise ValueError("close_prices did not contain any valid observations")

    return pd.concat(long_rows, axis=0, ignore_index=True)


def load_universe_close_prices(
    symbols: Sequence[str],
    timeframe: str,
    data_dir: Path = SETTINGS.data_dir,
) -> pd.DataFrame:
    """Load local OHLCV data for the universe and pivot to close prices."""
    ohlcv = load_ohlcv_history(symbols=tuple(symbols), timeframe=timeframe, data_dir=data_dir)
    return pivot_close(ohlcv)


def build_universal_eligibility_mask(close_prices: pd.DataFrame, min_required_lookback: int) -> pd.DataFrame:
    """Return a timestamp x symbol eligibility mask for a shared lookback requirement."""
    if min_required_lookback < 0:
        raise ValueError("min_required_lookback must be non-negative")

    close = _validate_close_prices(close_prices)
    valid = close.notna()
    window = min_required_lookback + 1
    rolling_valid = valid.rolling(window=window, min_periods=window).sum()
    eligible = valid & (rolling_valid == window)
    return eligible


def eligible_symbols_at_timestamp(
    eligibility_mask: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> list[str]:
    """Return eligible symbols at a rebalance timestamp."""
    if timestamp not in eligibility_mask.index:
        raise KeyError(f"timestamp not found in eligibility mask: {timestamp}")
    row = eligibility_mask.loc[timestamp]
    return row[row].index.tolist()


def _weighted_return_signal(
    close_prices: pd.DataFrame,
    windows: Sequence[int],
    weights: Sequence[float],
) -> pd.DataFrame:
    """Build a weighted trailing-return signal frame from the requested windows."""
    if len(windows) != len(weights):
        raise ValueError("windows and weights must have the same length")
    if not windows:
        raise ValueError("windows must not be empty")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    combined: pd.DataFrame | None = None
    for window, weight in zip(windows, weights, strict=True):
        signal = calculate_momentum_signal(close_prices, lookback_bars=int(window))
        scaled = signal * (float(weight) / total_weight)
        combined = scaled if combined is None else combined.add(scaled, fill_value=0.0)

    return combined if combined is not None else pd.DataFrame(index=close_prices.index, columns=close_prices.columns)


def _cross_sectional_config_for_lookback(
    lookback_config: LookbackConfig,
    top_n: int,
) -> dict[str, Any] | None:
    """Map a lookback config to CrossSectionalMomentumStrategy settings.

    TODO: the three-window lookback config does not fit the current cross-sectional
    momentum API cleanly. Keep it out of the first pass rather than forcing an
    awkward abstraction.
    """
    if len(lookback_config.momentum_windows) == 3:
        return None

    if len(lookback_config.momentum_windows) == 1:
        window = int(lookback_config.momentum_windows[0])
        if lookback_config.name == "short_42_only":
            short_weight, medium_weight = 1.0, 0.0
        else:
            short_weight, medium_weight = 0.0, 1.0
        short_window = medium_window = window
    else:
        short_window = int(lookback_config.momentum_windows[0])
        medium_window = int(lookback_config.momentum_windows[1])
        short_weight = float(lookback_config.momentum_weights[0])
        medium_weight = float(lookback_config.momentum_weights[1])

    return {
        "top_n": int(top_n),
        "short_lookback_bars": short_window,
        "medium_lookback_bars": medium_window,
        "short_weight": short_weight,
        "medium_weight": medium_weight,
        "min_history_bars": lookback_config.min_required_lookback,
        "min_eligible_assets": 1,
        "use_regime_filter": False,
    }


def _rebalance_signal_generator(
    strategy_name: str,
    lookback_config: LookbackConfig,
    close_prices: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    top_n: int | None = None,
    entry_overextension_threshold: float = 0.15,
    exit_overextension_threshold: float = 0.30,
) -> Callable[[pd.DataFrame, pd.Timestamp], pd.Series]:
    """Build a rebalance signal generator that only sees eligible symbols."""

    momentum_signal = _weighted_return_signal(
        close_prices=close_prices,
        windows=lookback_config.momentum_windows,
        weights=lookback_config.momentum_weights,
    )
    reversal_signal = _weighted_return_signal(
        close_prices=close_prices,
        windows=lookback_config.reversal_windows,
        weights=lookback_config.reversal_weights,
    )
    overextension_signal = calculate_overextension_signal(
        close_prices,
        lookback_bars=lookback_config.overextension_window,
    )

    cs_strategy = CrossSectionalMomentumStrategy()
    cs_config = None
    if strategy_name == "cs_momentum":
        if top_n is None:
            raise ValueError("top_n is required for cross-sectional momentum")
        cs_config = _cross_sectional_config_for_lookback(lookback_config, top_n=top_n)
        if cs_config is None:
            raise ValueError("cross-sectional momentum is not supported for this lookback config")

    def signal_generator(close: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        if timestamp not in close.index:
            raise KeyError(f"timestamp not found in close matrix: {timestamp}")

        eligible_symbols = eligible_symbols_at_timestamp(eligibility_mask, timestamp)
        if not eligible_symbols:
            return pd.Series(dtype=float)

        subset = close.loc[:timestamp, eligible_symbols]
        if subset.empty or subset.columns.empty:
            return pd.Series(dtype=float)

        momentum_current = momentum_signal.loc[[timestamp], eligible_symbols]
        reversal_current = reversal_signal.loc[[timestamp], eligible_symbols]
        overextension_current = overextension_signal.loc[[timestamp], eligible_symbols]

        if strategy_name == "ts_momentum":
            weights = time_series_momentum_weights(momentum_current)
        elif strategy_name == "ts_reversal":
            weights = short_term_reversal_weights(reversal_current)
        elif strategy_name == "ts_momentum_entry_filter":
            weights = momentum_with_entry_filter_weights(
                momentum_current,
                overextension_current,
                entry_overextension_threshold=entry_overextension_threshold,
            )
        elif strategy_name == "ts_momentum_exit_signal":
            weights = momentum_with_exit_signal_weights(
                momentum_signal.loc[:timestamp, eligible_symbols],
                overextension_signal.loc[:timestamp, eligible_symbols],
                exit_overextension_threshold=exit_overextension_threshold,
            )
        elif strategy_name == "ts_momentum_entry_exit":
            weights = momentum_with_entry_filter_and_exit_signal_weights(
                momentum_signal.loc[:timestamp, eligible_symbols],
                overextension_signal.loc[:timestamp, eligible_symbols],
                entry_overextension_threshold=entry_overextension_threshold,
                exit_overextension_threshold=exit_overextension_threshold,
            )
        elif strategy_name == "cs_momentum":
            assert cs_config is not None
            weights = pd.Series(
                cs_strategy.generate_target_weights(
                    close_prices=subset,
                    timestamp=timestamp,
                    config=cs_config,
                )
            )
        else:
            raise ValueError(f"Unsupported strategy_name: {strategy_name}")

        if isinstance(weights, pd.DataFrame):
            row = weights.loc[timestamp] if timestamp in weights.index else weights.iloc[-1]
        else:
            row = weights

        return row.reindex(close.columns, fill_value=0.0).astype(float)

    return signal_generator


def _bars_to_include(signal_close: pd.DataFrame, execution: BacktestResult, bars_per_year: int) -> dict[str, float | str | int | None]:
    """Compute a metrics row from a backtest result."""
    portfolio = execution.portfolio
    metrics = summary_metrics(
        equity=portfolio["equity"],
        returns=portfolio["strategy_return"],
        turnover=execution.turnover.reindex(portfolio.index).fillna(0.0),
        bars_per_year=bars_per_year,
    )

    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    calmar = None if max_drawdown >= 0 else cagr / abs(max_drawdown)

    avg_positions = float(execution.holdings_history.gt(0.0).sum(axis=1).mean())
    turnover_value = float(metrics.get("total_turnover", 0.0))
    start_date = str(signal_close.index.min())
    end_date = str(signal_close.index.max())

    return {
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": cagr,
        "annualized_vol": float(metrics.get("annualized_volatility", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover": turnover_value,
        "num_rebalances": int(len(execution.rebalance_log)),
        "avg_positions": avg_positions,
        "start_date": start_date,
        "end_date": end_date,
    }


def _run_one_experiment(
    close_prices: pd.DataFrame,
    ohlcv: pd.DataFrame,
    universe: str,
    timeframe: str,
    strategy_name: str,
    lookback_config: LookbackConfig,
    rebalance_bars: int,
    fee_bps: int,
    slippage_bps: int,
    top_n: int | None,
) -> dict[str, Any] | None:
    """Run one strategy/config combination and return a metrics row."""
    if strategy_name == "cs_momentum" and len(lookback_config.momentum_windows) == 3:
        LOGGER.info(
            "Skipping %s with %s: cross-sectional momentum does not yet support the three-window config",
            strategy_name,
            lookback_config.name,
        )
        return None

    eligibility_mask = build_universal_eligibility_mask(
        close_prices=close_prices,
        min_required_lookback=lookback_config.min_required_lookback,
    )
    signal_generator = _rebalance_signal_generator(
        strategy_name=strategy_name,
        lookback_config=lookback_config,
        close_prices=close_prices,
        eligibility_mask=eligibility_mask,
        top_n=top_n,
    )

    execution = run_backtest(
        ohlcv=ohlcv,
        signal_generator=signal_generator,
        initial_capital=SETTINGS.initial_capital,
        transaction_cost_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        rebalance_every_bars=int(rebalance_bars),
    )

    bars_per_year = bars_per_year_for_timeframe(timeframe)
    metrics = _bars_to_include(close_prices, execution, bars_per_year=bars_per_year)

    notes = "universal eligibility enforced; one-bar delayed execution"
    if strategy_name.startswith("ts_momentum"):
        notes = f"{notes}; time-series helper"
    elif strategy_name == "ts_reversal":
        notes = f"{notes}; reversal helper"
    elif strategy_name == "cs_momentum":
        notes = f"{notes}; CrossSectionalMomentumStrategy"

    row: dict[str, Any] = {
        "strategy_name": strategy_name,
        "universe": universe,
        "rebalance_bars": int(rebalance_bars),
        "lookback_config": lookback_config.name,
        "top_n": int(top_n) if top_n is not None else None,
        "fee_bps": int(fee_bps),
        "slippage_bps": int(slippage_bps),
        "min_required_lookback": int(lookback_config.min_required_lookback),
        "symbols_count": int(close_prices.shape[1]),
        "notes": notes,
    }
    row.update(metrics)
    return row


def _prepare_ohlcv_and_close(
    symbols: Sequence[str],
    timeframe: str,
    data_dir: Path,
    close_prices: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or synthesize OHLCV data and the matching close matrix."""
    if close_prices is not None:
        close = _validate_close_prices(close_prices)
        ohlcv = close_prices_to_ohlcv(close)
        return ohlcv, close

    ohlcv = load_ohlcv_history(symbols=tuple(symbols), timeframe=timeframe, data_dir=data_dir)
    close = pivot_close(ohlcv)
    return ohlcv, close


def _build_experiment_plan(
    lookback_configs: Sequence[LookbackConfig],
    rebalance_bars: Sequence[int],
    top_ns: Sequence[int],
    cost_configs: Sequence[tuple[int, int]],
    include_stateful: bool = False,
) -> list[dict[str, Any]]:
    """Build the full experiment plan before execution."""
    plan: list[dict[str, Any]] = []
    ts_strategies: tuple[str, ...] = (
        DEFAULT_FIRST_PASS_STRATEGIES[:-1] + STATEFUL_TS_STRATEGY_ORDER
        if include_stateful
        else DEFAULT_FIRST_PASS_STRATEGIES[:-1]
    )

    # TODO: Stateful exit-signal variants need an incremental/stateful
    # implementation before inclusion in the full expanded-universe grid.

    for lookback_config in lookback_configs:
        for rebalance in rebalance_bars:
            for fee_bps, slippage_bps in cost_configs:
                for strategy_name in ts_strategies:
                    plan.append(
                        {
                            "strategy_name": strategy_name,
                            "lookback_config": lookback_config,
                            "rebalance_bars": int(rebalance),
                            "fee_bps": int(fee_bps),
                            "slippage_bps": int(slippage_bps),
                            "top_n": None,
                        }
                    )

                for top_n in top_ns:
                    if _cross_sectional_config_for_lookback(lookback_config, top_n=int(top_n)) is None:
                        continue
                    plan.append(
                        {
                            "strategy_name": "cs_momentum",
                            "lookback_config": lookback_config,
                            "rebalance_bars": int(rebalance),
                            "fee_bps": int(fee_bps),
                            "slippage_bps": int(slippage_bps),
                            "top_n": int(top_n),
                        }
                    )

    return plan


def _apply_smoke_mode(
    symbols: Sequence[str],
    lookback_config_names: Sequence[str] | None,
    rebalance_bars: Sequence[int],
    top_ns: Sequence[int],
    cost_configs: Sequence[tuple[int, int]],
) -> tuple[Sequence[str], Sequence[str] | None, Sequence[int], Sequence[int], Sequence[tuple[int, int]]]:
    """Shrink experiment scope for quick smoke testing."""
    smoke_symbols = tuple(symbols[:6])
    smoke_lookbacks = lookback_config_names or ("short_42_only",)
    return smoke_symbols, smoke_lookbacks, (6,), (3,), ((10, 5),)


def run_expanded_universe_experiment(
    symbols: Sequence[str] = EXPANDED_UNIVERSE_20,
    universe: str = "expanded_20",
    timeframe: str = "4h",
    data_dir: Path = SETTINGS.data_dir,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
    close_prices: pd.DataFrame | None = None,
    lookback_config_names: Sequence[str] | None = None,
    rebalance_bars: Sequence[int] = DEFAULT_REBALANCE_BARS,
    top_ns: Sequence[int] = DEFAULT_TOP_NS,
    cost_configs: Sequence[tuple[int, int]] = DEFAULT_COST_CONFIGS,
    smoke: bool = False,
    max_runs: int | None = None,
    include_stateful: bool = False,
) -> pd.DataFrame:
    """Run the expanded-universe research grid and return a metrics table."""
    if smoke:
        symbols, lookback_config_names, rebalance_bars, top_ns, cost_configs = _apply_smoke_mode(
            symbols=symbols,
            lookback_config_names=lookback_config_names,
            rebalance_bars=rebalance_bars,
            top_ns=top_ns,
            cost_configs=cost_configs,
        )

    ohlcv, close = _prepare_ohlcv_and_close(symbols=symbols, timeframe=timeframe, data_dir=data_dir, close_prices=close_prices)

    if smoke:
        close = close.reindex(columns=list(symbols))
        close = close.iloc[-SMOKE_WINDOW_ROWS:].copy()
        ohlcv = close_prices_to_ohlcv(close)

    LOGGER.info(
        "Experiment data prepared: smoke=%s close_shape=%s symbols=%d first_timestamp=%s last_timestamp=%s",
        smoke,
        close.shape,
        close.shape[1],
        close.index.min(),
        close.index.max(),
    )

    selected_configs = (
        {name: LOOKBACK_CONFIGS[name] for name in lookback_config_names}
        if lookback_config_names is not None
        else LOOKBACK_CONFIGS
    )

    plan = _build_experiment_plan(
        lookback_configs=tuple(selected_configs.values()),
        rebalance_bars=rebalance_bars,
        top_ns=top_ns,
        cost_configs=cost_configs,
        include_stateful=include_stateful,
    )
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive when provided")
        plan = plan[:max_runs]

    LOGGER.info("Planned expanded-universe experiment runs: %d", len(plan))

    rows: list[dict[str, Any]] = []
    total_runs = len(plan)
    for run_index, plan_item in enumerate(plan, start=1):
        LOGGER.info(
            "Starting run %d/%d | strategy=%s lookback=%s rebalance_bars=%d top_n=%s fee_bps=%d slippage_bps=%d",
            run_index,
            total_runs,
            plan_item["strategy_name"],
            plan_item["lookback_config"].name,
            plan_item["rebalance_bars"],
            plan_item["top_n"],
            plan_item["fee_bps"],
            plan_item["slippage_bps"],
        )
        started_at = time.perf_counter()
        row = _run_one_experiment(
            close_prices=close,
            ohlcv=ohlcv,
            universe=universe,
            timeframe=timeframe,
            strategy_name=plan_item["strategy_name"],
            lookback_config=plan_item["lookback_config"],
            rebalance_bars=plan_item["rebalance_bars"],
            fee_bps=plan_item["fee_bps"],
            slippage_bps=plan_item["slippage_bps"],
            top_n=plan_item["top_n"],
        )
        elapsed_seconds = time.perf_counter() - started_at
        if row is not None:
            if plan_item["top_n"] is not None:
                row["notes"] = f"{row['notes']}; top_n={plan_item['top_n']}"
            rows.append(row)
            LOGGER.info(
                "Completed run %d/%d in %.2fs | sharpe=%.4f cagr=%.4f total_return=%.4f",
                run_index,
                total_runs,
                elapsed_seconds,
                float(row["sharpe"]),
                float(row["cagr"]),
                float(row["total_return"]),
            )
        else:
            LOGGER.info("Skipped run %d/%d in %.2fs", run_index, total_runs, elapsed_seconds)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No experiment runs were produced")

    report = report[
        [
            "strategy_name",
            "universe",
            "rebalance_bars",
            "lookback_config",
            "top_n",
            "fee_bps",
            "slippage_bps",
            "min_required_lookback",
            "start_date",
            "end_date",
            "symbols_count",
            "total_return",
            "cagr",
            "annualized_vol",
            "sharpe",
            "max_drawdown",
            "calmar",
            "turnover",
            "num_rebalances",
            "avg_positions",
            "notes",
        ]
    ]
    report = report.sort_values(["sharpe", "cagr"], ascending=[False, False]).reset_index(drop=True)

    if save_csv:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    print(f"\nExperiment runs: {len(report)}")
    print("\nTop 10 by Sharpe")
    print("=" * 120)
    print(report.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 by CAGR")
    print("=" * 120)
    print(report.sort_values("cagr", ascending=False).head(10).to_string(index=False))

    print("\nData source: Binance public USDT spot klines normalized to /USD symbols")
    print("Known source gap: 2020-02-19 12:00 UTC missing for 9 older symbols")

    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run expanded-universe research experiments")
    parser.add_argument("--symbols", nargs="+", default=list(EXPANDED_UNIVERSE_20), help="Universe symbols")
    parser.add_argument("--expanded-universe", action="store_true", help="Use the built-in expanded universe")
    parser.add_argument("--timeframe", default="4h", help="Data timeframe")
    parser.add_argument("--data-dir", default=str(SETTINGS.data_dir), help="Local OHLCV directory")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Metrics CSV output path")
    parser.add_argument("--no-save-csv", action="store_true", help="Do not write the CSV output")
    parser.add_argument("--lookback-configs", nargs="+", default=None, help="Optional subset of lookback config names")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke-test subset of the experiment grid")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of planned runs to execute")
    parser.add_argument("--include-stateful", action="store_true", help="Include expensive stateful exit-signal variants")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the expanded-universe experiment."""
    configure_logging()
    args = parse_args()

    symbols = EXPANDED_UNIVERSE_20 if args.expanded_universe else tuple(args.symbols)
    run_expanded_universe_experiment(
        symbols=symbols,
        timeframe=args.timeframe,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output_path),
        save_csv=not args.no_save_csv,
        lookback_config_names=tuple(args.lookback_configs) if args.lookback_configs else None,
        smoke=bool(args.smoke),
        max_runs=args.max_runs,
        include_stateful=bool(args.include_stateful),
    )


if __name__ == "__main__":
    main()