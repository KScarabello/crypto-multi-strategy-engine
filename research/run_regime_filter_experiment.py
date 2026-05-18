"""Run focused regime-filter experiments on expanded-universe momentum candidates."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from backtest.engine import run_backtest
from config import SETTINGS
from research.run_expanded_universe_experiment import (
    EXPANDED_UNIVERSE_20,
    LOOKBACK_CONFIGS,
    _bars_to_include,
    _rebalance_signal_generator,
    _validate_close_prices,
    bars_per_year_for_timeframe,
    build_universal_eligibility_mask,
    close_prices_to_ohlcv,
    load_universe_close_prices,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("research/results/regime_filter_experiment_metrics.csv")
REGIME_FILTERS: tuple[str, ...] = (
    "none",
    "btc_momentum_180",
    "btc_ma_360",
    "equal_weight_momentum_180",
    "equal_weight_ma_360",
)
COST_CONFIGS: tuple[tuple[int, int], ...] = ((10, 5), (20, 10))


@dataclass(frozen=True)
class CandidateStrategy:
    """One baseline strategy candidate to test with regime overlays."""

    strategy_name: str
    rebalance_bars: int
    lookback_config_name: str
    top_n: int | None = None


CANDIDATE_STRATEGIES: tuple[CandidateStrategy, ...] = (
    CandidateStrategy("cs_momentum", rebalance_bars=6, lookback_config_name="short_42_medium_180", top_n=3),
    CandidateStrategy("cs_momentum", rebalance_bars=12, lookback_config_name="medium_180_only", top_n=5),
    CandidateStrategy("cs_momentum", rebalance_bars=42, lookback_config_name="medium_180_only", top_n=8),
    CandidateStrategy("ts_momentum", rebalance_bars=42, lookback_config_name="medium_180_only", top_n=None),
    CandidateStrategy("ts_momentum_entry_filter", rebalance_bars=42, lookback_config_name="short_42_medium_180", top_n=None),
)


def configure_logging() -> None:
    """Configure readable logging for research runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _regime_signal_none(index: pd.Index) -> pd.Series:
    """Always risk-on regime."""
    return pd.Series(True, index=index, dtype=bool)


def _btc_momentum_filter(close_prices: pd.DataFrame, lookback_bars: int = 180) -> pd.Series:
    """Risk-on when BTC trailing return is positive."""
    btc = close_prices.get("BTC/USD")
    if btc is None:
        return pd.Series(False, index=close_prices.index, dtype=bool)
    return (btc.pct_change(lookback_bars) > 0).fillna(False)


def _btc_ma_filter(close_prices: pd.DataFrame, ma_bars: int = 360) -> pd.Series:
    """Risk-on when BTC is above its moving average."""
    btc = close_prices.get("BTC/USD")
    if btc is None:
        return pd.Series(False, index=close_prices.index, dtype=bool)
    ma = btc.rolling(window=ma_bars, min_periods=ma_bars).mean()
    return (btc > ma).fillna(False)


def _equal_weight_index(close_prices: pd.DataFrame, eligibility_mask: pd.DataFrame | None = None) -> pd.Series:
    """Construct equal-weight index from available returns without forward-filling pre-listing history."""
    returns = close_prices.pct_change()
    if eligibility_mask is not None:
        returns = returns.where(eligibility_mask)
    ew_returns = returns.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + ew_returns).cumprod()


def _equal_weight_momentum_filter(
    close_prices: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    lookback_bars: int = 180,
) -> pd.Series:
    """Risk-on when equal-weight index trailing return is positive."""
    ew_index = _equal_weight_index(close_prices, eligibility_mask=eligibility_mask)
    return (ew_index.pct_change(lookback_bars) > 0).fillna(False)


def _equal_weight_ma_filter(
    close_prices: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    ma_bars: int = 360,
) -> pd.Series:
    """Risk-on when equal-weight index is above moving average."""
    ew_index = _equal_weight_index(close_prices, eligibility_mask=eligibility_mask)
    ma = ew_index.rolling(window=ma_bars, min_periods=ma_bars).mean()
    return (ew_index > ma).fillna(False)


def build_regime_mask(
    regime_filter_name: str,
    close_prices: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
) -> pd.Series:
    """Build one risk-on mask aligned to close_prices index."""
    if regime_filter_name == "none":
        return _regime_signal_none(close_prices.index)
    if regime_filter_name == "btc_momentum_180":
        return _btc_momentum_filter(close_prices, lookback_bars=180)
    if regime_filter_name == "btc_ma_360":
        return _btc_ma_filter(close_prices, ma_bars=360)
    if regime_filter_name == "equal_weight_momentum_180":
        return _equal_weight_momentum_filter(close_prices, eligibility_mask=eligibility_mask, lookback_bars=180)
    if regime_filter_name == "equal_weight_ma_360":
        return _equal_weight_ma_filter(close_prices, eligibility_mask=eligibility_mask, ma_bars=360)
    raise ValueError(f"Unsupported regime filter: {regime_filter_name}")


def apply_regime_filter_to_weights(weights: pd.Series, risk_on: bool) -> pd.Series:
    """Convert target weights to all-cash when regime is risk-off."""
    if risk_on:
        return weights
    return pd.Series(0.0, index=weights.index, dtype=float)


def _regime_gated_signal_generator(
    base_signal_generator: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
    regime_mask: pd.Series,
) -> Callable[[pd.DataFrame, pd.Timestamp], pd.Series]:
    """Wrap a base signal generator with risk-on/risk-off gating."""

    def signal_generator(close: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        base_weights = base_signal_generator(close, timestamp)
        risk_on = bool(regime_mask.get(timestamp, False))
        return apply_regime_filter_to_weights(base_weights, risk_on=risk_on)

    return signal_generator


def _baseline_name(candidate: CandidateStrategy, fee_bps: int, slippage_bps: int) -> str:
    """Create a stable baseline strategy identifier."""
    top_n = "na" if candidate.top_n is None else str(candidate.top_n)
    return (
        f"{candidate.strategy_name}|lookback={candidate.lookback_config_name}|"
        f"rebalance={candidate.rebalance_bars}|top_n={top_n}|fee={fee_bps}|slippage={slippage_bps}"
    )


def _risk_on_fraction_from_rebalances(rebalance_log: pd.DataFrame, regime_mask: pd.Series) -> float:
    """Compute fraction of rebalance signals that occurred in risk-on state."""
    if rebalance_log.empty or "signal_timestamp" not in rebalance_log.columns:
        return 0.0

    signal_ts = pd.to_datetime(rebalance_log["signal_timestamp"], utc=True, errors="coerce").dropna()
    if signal_ts.empty:
        return 0.0

    risk_on = regime_mask.reindex(signal_ts).fillna(False).astype(bool)
    return float(risk_on.mean())


def run_regime_filter_experiment(
    symbols: Sequence[str] = EXPANDED_UNIVERSE_20,
    timeframe: str = "4h",
    data_dir: Path = SETTINGS.data_dir,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
    close_prices: pd.DataFrame | None = None,
    candidates: Sequence[CandidateStrategy] = CANDIDATE_STRATEGIES,
    cost_configs: Sequence[tuple[int, int]] = COST_CONFIGS,
    regime_filters: Sequence[str] = REGIME_FILTERS,
) -> pd.DataFrame:
    """Run focused regime/risk-filter experiments for selected baseline strategies."""
    if close_prices is None:
        close = load_universe_close_prices(symbols=symbols, timeframe=timeframe, data_dir=data_dir)
    else:
        close = _validate_close_prices(close_prices)

    ohlcv = close_prices_to_ohlcv(close)
    bars_per_year = bars_per_year_for_timeframe(timeframe)
    rows: list[dict[str, Any]] = []

    total_runs = len(candidates) * len(cost_configs) * len(regime_filters)
    LOGGER.info("Planned regime-filter runs: %d", total_runs)

    run_idx = 0
    # Universal eligibility remains mandatory for all strategy candidates.
    for candidate in candidates:
        lookback_cfg = LOOKBACK_CONFIGS[candidate.lookback_config_name]
        eligibility = build_universal_eligibility_mask(close, min_required_lookback=lookback_cfg.min_required_lookback)

        base_signal_generator = _rebalance_signal_generator(
            strategy_name=candidate.strategy_name,
            lookback_config=lookback_cfg,
            close_prices=close,
            eligibility_mask=eligibility,
            top_n=candidate.top_n,
        )

        for fee_bps, slippage_bps in cost_configs:
            baseline_name = _baseline_name(candidate, fee_bps=fee_bps, slippage_bps=slippage_bps)

            for regime_filter_name in regime_filters:
                run_idx += 1
                LOGGER.info(
                    "Starting run %d/%d | baseline=%s regime=%s",
                    run_idx,
                    total_runs,
                    baseline_name,
                    regime_filter_name,
                )

                regime_mask = build_regime_mask(
                    regime_filter_name=regime_filter_name,
                    close_prices=close,
                    eligibility_mask=eligibility,
                )
                gated_signal_generator = _regime_gated_signal_generator(
                    base_signal_generator=base_signal_generator,
                    regime_mask=regime_mask,
                )

                execution = run_backtest(
                    ohlcv=ohlcv,
                    signal_generator=gated_signal_generator,
                    initial_capital=SETTINGS.initial_capital,
                    transaction_cost_bps=float(fee_bps),
                    slippage_bps=float(slippage_bps),
                    rebalance_every_bars=int(candidate.rebalance_bars),
                )

                metrics = _bars_to_include(close, execution, bars_per_year=bars_per_year)
                row: dict[str, Any] = {
                    "strategy_name": candidate.strategy_name,
                    "baseline_strategy_name": baseline_name,
                    "regime_filter_name": regime_filter_name,
                    "universe": "expanded_20",
                    "rebalance_bars": int(candidate.rebalance_bars),
                    "lookback_config": candidate.lookback_config_name,
                    "top_n": candidate.top_n,
                    "fee_bps": int(fee_bps),
                    "slippage_bps": int(slippage_bps),
                    "risk_on_fraction": _risk_on_fraction_from_rebalances(execution.rebalance_log, regime_mask),
                }
                row.update(metrics)
                rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No regime-filter runs were produced")

    baseline_mdd = (
        report.loc[report["regime_filter_name"] == "none", ["baseline_strategy_name", "max_drawdown"]]
        .rename(columns={"max_drawdown": "baseline_max_drawdown"})
        .drop_duplicates(subset=["baseline_strategy_name"])
    )
    report = report.merge(baseline_mdd, on="baseline_strategy_name", how="left")
    report["max_drawdown_improvement"] = report["max_drawdown"] - report["baseline_max_drawdown"]

    ordered_columns = [
        "strategy_name",
        "baseline_strategy_name",
        "regime_filter_name",
        "universe",
        "rebalance_bars",
        "lookback_config",
        "top_n",
        "fee_bps",
        "slippage_bps",
        "risk_on_fraction",
        "total_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "turnover",
        "num_rebalances",
        "avg_positions",
        "start_date",
        "end_date",
        "max_drawdown_improvement",
        "baseline_max_drawdown",
    ]
    report = report[ordered_columns].sort_values(["sharpe", "cagr"], ascending=[False, False]).reset_index(drop=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    print(f"\nRegime-filter runs: {len(report)}")
    print("\nTop 10 by Sharpe")
    print("=" * 120)
    print(report.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 by max_drawdown improvement")
    print("=" * 120)
    print(report.sort_values("max_drawdown_improvement", ascending=False).head(10).to_string(index=False))

    print("\nBest result per base strategy")
    print("=" * 120)
    best_by_base = report.sort_values("sharpe", ascending=False).groupby("baseline_strategy_name", as_index=False).first()
    print(best_by_base.to_string(index=False))

    print("\nData source: Binance public USDT spot klines normalized to /USD symbols")
    print("Known source gap: 2020-02-19 12:00 UTC missing for 9 older symbols")

    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for regime-filter experiment runner."""
    parser = argparse.ArgumentParser(description="Run focused regime-filter experiments")
    parser.add_argument("--symbols", nargs="+", default=list(EXPANDED_UNIVERSE_20), help="Universe symbols")
    parser.add_argument("--expanded-universe", action="store_true", help="Use built-in expanded 20-symbol universe")
    parser.add_argument("--timeframe", default="4h", help="Data timeframe")
    parser.add_argument("--data-dir", default=str(SETTINGS.data_dir), help="Local OHLCV directory")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="CSV output path")
    parser.add_argument("--no-save-csv", action="store_true", help="Disable CSV output")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for focused regime-filter experiments."""
    configure_logging()
    args = parse_args()

    symbols = EXPANDED_UNIVERSE_20 if args.expanded_universe else tuple(args.symbols)
    run_regime_filter_experiment(
        symbols=symbols,
        timeframe=args.timeframe,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output_path),
        save_csv=not args.no_save_csv,
    )


if __name__ == "__main__":
    main()
