"""Run research comparison: time-series momentum vs short-term reversal overlays."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from data.fetch_ohlc import load_ohlcv_history, pivot_close
from research.run_backtest import bars_per_year_for_timeframe
from research.signals import (
    calculate_momentum_signal,
    calculate_overextension_signal,
)
from research.strategy_variants import (
    momentum_with_entry_filter_and_exit_signal_weights,
    momentum_with_entry_filter_weights,
    momentum_with_exit_signal_weights,
    short_term_reversal_weights,
    time_series_momentum_weights,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_UNIVERSE: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "AVAX/USD")
DEFAULT_OUTPUT_PATH = Path("research/results/ts_momentum_reversal_metrics.csv")


def configure_logging() -> None:
    """Configure research logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _normalize_timeframe(timeframe: str | None) -> str:
    """Normalize timeframe and default to 4h for this research workflow."""
    candidate = (timeframe or "").strip().lower()
    if candidate in {"4h", "1d", "d", "daily"}:
        return "1d" if candidate in {"1d", "d", "daily"} else "4h"
    return "4h"


def _align_weights_to_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Shift target weights by one bar to avoid lookahead execution."""
    executed = weights.shift(1).fillna(0.0)
    return executed.reindex(asset_returns.index).fillna(0.0)


def _simulate_from_weights(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, pd.Series]:
    """Convert target weights into net returns/equity using one-bar delayed execution."""
    asset_returns = close.pct_change().fillna(0.0)
    executed_weights = _align_weights_to_returns(target_weights, asset_returns)

    gross_returns = (executed_weights * asset_returns).sum(axis=1)
    turnover = executed_weights.diff().abs().sum(axis=1).fillna(0.0)
    total_cost_bps = float(transaction_cost_bps) + float(slippage_bps)
    cost_rate = turnover * (total_cost_bps / 10_000.0)
    net_returns = gross_returns - cost_rate

    equity = float(initial_capital) * (1.0 + net_returns).cumprod()
    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "equity": equity,
    }


def _format_metrics_row(strategy_name: str, metrics: dict[str, float], final_equity: float) -> dict[str, float | str]:
    """Build one output row from computed summary metrics."""
    return {
        "strategy": strategy_name,
        "cagr": float(metrics.get("cagr", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "avg_turnover": float(metrics.get("avg_turnover", 0.0)),
        "total_turnover": float(metrics.get("total_turnover", 0.0)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "final_equity": float(final_equity),
    }


def run_research(
    symbols: tuple[str, ...] = DEFAULT_UNIVERSE,
    timeframe: str = "4h",
    output_path: Path = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
    momentum_lookback_bars: int = 336,
    overextension_lookback_bars: int = 42,
    entry_overextension_threshold: float = 0.15,
    exit_overextension_threshold: float = 0.30,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Run strategy comparison for momentum and reversal variants."""
    tf = _normalize_timeframe(timeframe)
    bars_per_year = bars_per_year_for_timeframe(tf)

    LOGGER.info("Loading OHLCV for %d symbols at %s", len(symbols), tf)
    ohlcv = load_ohlcv_history(
        symbols=symbols,
        timeframe=tf,
        data_dir=SETTINGS.data_dir,
    )
    close = pivot_close(ohlcv)

    momentum_signal = calculate_momentum_signal(close, lookback_bars=momentum_lookback_bars)
    overextension_signal = calculate_overextension_signal(
        close,
        lookback_bars=overextension_lookback_bars,
    )

    weight_variants: dict[str, pd.DataFrame] = {
        "momentum_only": time_series_momentum_weights(momentum_signal),
        "short_term_reversal_only": short_term_reversal_weights(overextension_signal),
        "momentum_with_entry_filter": momentum_with_entry_filter_weights(
            momentum_signal,
            overextension_signal,
            entry_overextension_threshold=entry_overextension_threshold,
        ),
        "momentum_with_exit_signal": momentum_with_exit_signal_weights(
            momentum_signal,
            overextension_signal,
            exit_overextension_threshold=exit_overextension_threshold,
        ),
        "momentum_with_entry_and_exit": momentum_with_entry_filter_and_exit_signal_weights(
            momentum_signal,
            overextension_signal,
            entry_overextension_threshold=entry_overextension_threshold,
            exit_overextension_threshold=exit_overextension_threshold,
        ),
    }

    rows: list[dict[str, float | str]] = []
    for strategy_name, target_weights in weight_variants.items():
        simulation = _simulate_from_weights(
            close=close,
            target_weights=target_weights,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        metrics = summary_metrics(
            equity=simulation["equity"],
            returns=simulation["net_returns"],
            turnover=simulation["turnover"],
            bars_per_year=bars_per_year,
        )
        rows.append(
            _format_metrics_row(
                strategy_name=strategy_name,
                metrics=metrics,
                final_equity=float(simulation["equity"].iloc[-1]),
            )
        )

    comparison = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    print("\nTS Momentum + Reversal Research Comparison")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path, index=False)
        LOGGER.info("Saved research comparison to %s", output_path)

    return comparison


def main() -> None:
    """CLI entrypoint for momentum/reversal research comparison."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run modular time-series momentum and reversal strategy research",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_UNIVERSE),
        help="Universe symbols (default: BTC/USD ETH/USD SOL/USD XRP/USD AVAX/USD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4h",
        help="Data timeframe (default: 4h)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path for strategy metrics",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Disable writing metrics CSV",
    )
    parser.add_argument("--momentum-lookback", type=int, default=336)
    parser.add_argument("--overextension-lookback", type=int, default=42)
    parser.add_argument("--entry-threshold", type=float, default=0.15)
    parser.add_argument("--exit-threshold", type=float, default=0.30)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()

    run_research(
        symbols=tuple(args.symbols),
        timeframe=args.timeframe,
        output_path=args.output_path,
        save_csv=not args.no_save_csv,
        momentum_lookback_bars=args.momentum_lookback,
        overextension_lookback_bars=args.overextension_lookback,
        entry_overextension_threshold=args.entry_threshold,
        exit_overextension_threshold=args.exit_threshold,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )


if __name__ == "__main__":
    main()
