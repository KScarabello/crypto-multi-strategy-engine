"""BTC-only time-series momentum parameter sweep research helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.btc_time_series import BtcTimeSeriesConfig, run_btc_time_series_experiment


DEFAULT_SHORT_LOOKBACKS: tuple[int, ...] = (24, 42, 60, 90)
DEFAULT_MEDIUM_LOOKBACKS: tuple[int, ...] = (120, 180, 240, 360)
DEFAULT_REBALANCE_EVERY: tuple[int, ...] = (3, 6, 12)
DEFAULT_OUTPUT_PATH = Path("reports/btc_time_series_sweep.csv")


def _validate_positive_int_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate a non-empty sequence of positive integers."""
    if not values:
        raise ValueError(f"{name} must not be empty")

    normalized: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError(f"{name} must contain positive integers")
        normalized.append(ivalue)
    return tuple(normalized)


def build_btc_sweep_parameter_grid(
    short_lookbacks: Sequence[int] = DEFAULT_SHORT_LOOKBACKS,
    medium_lookbacks: Sequence[int] = DEFAULT_MEDIUM_LOOKBACKS,
    rebalance_every: Sequence[int] = DEFAULT_REBALANCE_EVERY,
) -> list[tuple[int, int, int]]:
    """Build valid (short, medium, rebalance) combinations for sweep.

    Only includes combinations where short_lookback < medium_lookback.
    """
    shorts = _validate_positive_int_sequence(short_lookbacks, "short_lookbacks")
    mediums = _validate_positive_int_sequence(medium_lookbacks, "medium_lookbacks")
    rebalances = _validate_positive_int_sequence(rebalance_every, "rebalance_every")

    combos = [
        (short_lb, medium_lb, rebalance_bars)
        for short_lb in shorts
        for medium_lb in mediums
        for rebalance_bars in rebalances
        if short_lb < medium_lb
    ]
    if not combos:
        raise ValueError("No valid parameter combinations: require short_lookback < medium_lookback")

    return combos


def _load_btc_close_prices(symbol: str, timeframe: str, data_dir: Path) -> pd.DataFrame:
    """Load BTC close prices once for all parameter sweep runs."""
    ohlcv = load_ohlcv_history(symbols=(symbol,), timeframe=timeframe, data_dir=data_dir)
    close = pivot_close(ohlcv)
    if symbol not in close.columns:
        raise ValueError(f"Loaded close prices do not include {symbol}")
    return close[[symbol]].copy().sort_index()


def run_btc_time_series_sweep(
    short_lookbacks: Sequence[int] = DEFAULT_SHORT_LOOKBACKS,
    medium_lookbacks: Sequence[int] = DEFAULT_MEDIUM_LOOKBACKS,
    rebalance_every: Sequence[int] = DEFAULT_REBALANCE_EVERY,
    symbol: str = "BTC/USD",
    timeframe: str = "4h",
    short_weight: float = 0.5,
    medium_weight: float = 0.5,
    initial_capital: float = SETTINGS.initial_capital,
    transaction_cost_bps: float = SETTINGS.transaction_cost_bps,
    slippage_bps: float = SETTINGS.slippage_bps,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run BTC-only time-series momentum sweep vs BTC buy-and-hold."""
    combos = build_btc_sweep_parameter_grid(
        short_lookbacks=short_lookbacks,
        medium_lookbacks=medium_lookbacks,
        rebalance_every=rebalance_every,
    )

    close = close_prices if close_prices is not None else _load_btc_close_prices(symbol, timeframe, data_dir)
    close = close.sort_index()

    rows: list[dict[str, float | int]] = []
    for short_lb, medium_lb, rebalance_bars in combos:
        config = BtcTimeSeriesConfig(
            symbol=symbol,
            timeframe=timeframe,
            short_lookback_bars=int(short_lb),
            medium_lookback_bars=int(medium_lb),
            short_weight=float(short_weight),
            medium_weight=float(medium_weight),
            rebalance_every_bars=int(rebalance_bars),
            initial_capital=float(initial_capital),
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
        )

        report = run_btc_time_series_experiment(config=config, close_prices=close)
        ts_row = report.loc[report["strategy"] == "btc_time_series_momentum"].iloc[0]
        rows.append(
            {
                "short_lookback": int(short_lb),
                "medium_lookback": int(medium_lb),
                "rebalance_every": int(rebalance_bars),
                "total_return": float(ts_row["total_return"]),
                "cagr": float(ts_row["cagr"]),
                "annualized_volatility": float(ts_row["annualized_volatility"]),
                "sharpe": float(ts_row["sharpe"]),
                "max_drawdown": float(ts_row["max_drawdown"]),
                "num_trades": int(ts_row["num_trades"]),
                "percent_time_invested": float(ts_row["percent_time_invested"]),
                "vs_buy_hold_total_return": float(ts_row["vs_buy_hold_total_return"]),
                "vs_buy_hold_cagr": float(ts_row["vs_buy_hold_cagr"]),
                "vs_buy_hold_sharpe": float(ts_row["vs_buy_hold_sharpe"]),
                "vs_buy_hold_max_drawdown": float(ts_row["vs_buy_hold_max_drawdown"]),
            }
        )

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result
