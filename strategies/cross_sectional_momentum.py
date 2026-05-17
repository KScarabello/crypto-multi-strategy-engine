"""Cross-sectional momentum strategy.

This module ports the pure strategy logic from the legacy single-strategy repo
into the multi-strategy `Strategy` interface.

Behavioral assumptions mirrored from the legacy momentum implementation:
- Momentum score is a weighted average of short and medium lookback returns.
- Ranking is cross-sectional at a single decision timestamp.
- Selection is top-N among eligible symbols.
- Eligibility requires valid price at the decision timestamp and minimum history.
- Optional BTC moving-average regime filter can force all-cash (risk-off).
- Target construction is equal-weight across selected symbols, optionally capped.

Execution timing (including one-bar-delayed execution) is intentionally not
implemented here; that belongs to the backtest/live orchestration layers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import Strategy, StrategyState


DEFAULT_CONFIG: dict[str, Any] = {
    "top_n": 2,
    "short_lookback_bars": 12,
    "medium_lookback_bars": 36,
    "short_weight": 0.5,
    "medium_weight": 0.5,
    "use_regime_filter": False,
    "btc_symbol": "BTC/USD",
    "regime_lookback_bars": 30,
    "min_history_bars": None,
    "min_eligible_assets": 1,
    "max_position_weight": None,
    "max_gross_exposure": None,
    "eligible_symbols": None,
    "min_median_volume": None,
    "volume": None,
    "include_cash_symbol": False,
    "cash_symbol": "USD",
}


def _clean_close_prices(close: pd.DataFrame) -> pd.DataFrame:
    """Return close prices with non-positive values treated as missing.

    Legacy engine behavior drops rows with non-positive close before pivoting.
    At strategy level we cannot drop per-row OHLCV, so we mirror that behavior
    by marking non-positive closes as NaN and letting eligibility filters handle
    the resulting missing history.
    """
    if close.empty:
        raise ValueError("close price matrix is empty")

    clean = close.sort_index().astype(float).copy()
    clean[clean <= 0] = float("nan")
    return clean


def compute_return_over_lookback(close: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    """Compute percentage return over a trailing lookback window."""
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if close.empty:
        raise ValueError("close price matrix is empty")
    return close.pct_change(periods=lookback_bars)


def compute_momentum_score(
    close: pd.DataFrame,
    short_lookback_bars: int,
    medium_lookback_bars: int,
    short_weight: float = 0.5,
    medium_weight: float = 0.5,
) -> pd.DataFrame:
    """Compute weighted momentum score from short and medium lookback returns."""
    if short_lookback_bars <= 0 or medium_lookback_bars <= 0:
        raise ValueError("lookback values must be positive")
    if short_weight < 0 or medium_weight < 0:
        raise ValueError("weights must be non-negative")

    weight_sum = short_weight + medium_weight
    if weight_sum <= 0:
        raise ValueError("sum of weights must be positive")

    normalized_short = short_weight / weight_sum
    normalized_medium = medium_weight / weight_sum

    short_ret = compute_return_over_lookback(close, lookback_bars=short_lookback_bars)
    medium_ret = compute_return_over_lookback(close, lookback_bars=medium_lookback_bars)

    return (normalized_short * short_ret) + (normalized_medium * medium_ret)


def rank_symbols_for_date(
    momentum_score: pd.DataFrame,
    rebalance_timestamp: pd.Timestamp,
    top_n: int,
) -> list[str]:
    """Rank symbols by momentum at one timestamp and return top selections."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rebalance_timestamp not in momentum_score.index:
        raise KeyError(f"rebalance_timestamp not found in score index: {rebalance_timestamp}")

    row = momentum_score.loc[rebalance_timestamp]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    ranked = row.dropna().sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


def check_regime_filter(
    close: pd.DataFrame,
    rebalance_timestamp: pd.Timestamp,
    btc_symbol: str = "BTC/USD",
    ma_lookback_bars: int = 30,
) -> bool:
    """Return True when BTC is above its moving average at the decision timestamp."""
    if ma_lookback_bars <= 0:
        raise ValueError("ma_lookback_bars must be positive")
    if btc_symbol not in close.columns:
        raise KeyError(f"BTC symbol not found in close matrix: {btc_symbol}")
    if rebalance_timestamp not in close.index:
        raise KeyError(f"rebalance_timestamp not found in close index: {rebalance_timestamp}")

    btc_close = close[btc_symbol]
    btc_ma = btc_close.rolling(window=ma_lookback_bars, min_periods=ma_lookback_bars).mean()

    current_close = btc_close.loc[rebalance_timestamp]
    current_ma = btc_ma.loc[rebalance_timestamp]

    if pd.isna(current_close) or pd.isna(current_ma):
        return False

    return bool(current_close > current_ma)


def _build_target_weights(symbols: pd.Index, selected: list[str]) -> pd.Series:
    """Build equal-weight long-only target weights for selected symbols."""
    weights = pd.Series(0.0, index=symbols, dtype=float)
    if selected:
        weights.loc[selected] = 1.0 / len(selected)
    return weights


def _apply_position_weight_cap(target_weights: pd.Series, max_position_weight: float | None) -> pd.Series:
    """Cap individual asset weights and leave any remainder in cash."""
    if max_position_weight is None:
        return target_weights
    if max_position_weight <= 0 or max_position_weight > 1.0:
        raise ValueError("max_position_weight must be within (0, 1] when provided")
    return target_weights.clip(lower=0.0, upper=max_position_weight)


def _apply_gross_exposure_cap(target_weights: pd.Series, max_gross_exposure: float | None) -> pd.Series:
    """Scale the full portfolio down to a maximum gross exposure."""
    if max_gross_exposure is None:
        return target_weights
    if max_gross_exposure <= 0 or max_gross_exposure > 1.0:
        raise ValueError("max_gross_exposure must be within (0, 1] when provided")

    total_weight = float(target_weights.sum())
    if total_weight <= 0 or total_weight <= max_gross_exposure:
        return target_weights
    return target_weights * (max_gross_exposure / total_weight)


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller config onto strategy defaults."""
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


class CrossSectionalMomentumStrategy(Strategy):
    """Long-only cross-sectional momentum strategy.

    Output convention: weights for risky assets only (cash is implicit as
    `1 - sum(risky_weights)`). Optionally, set `include_cash_symbol=True` in
    config to include an explicit cash symbol key.

    Supported config keys:
    - top_n
    - short_lookback_bars
    - medium_lookback_bars
    - short_weight
    - medium_weight
    - use_regime_filter
    - btc_symbol
    - regime_lookback_bars
    - min_history_bars (defaults to max(short_lookback_bars, medium_lookback_bars))
    - min_eligible_assets
    - eligible_symbols (optional symbol allowlist)
    - min_median_volume and volume (optional parity with legacy eligibility)
    - max_position_weight
    - max_gross_exposure
    - include_cash_symbol
    - cash_symbol
    """

    def generate_target_weights(
        self,
        close_prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        state: StrategyState | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        cfg = _merged_config(config)
        _ = state  # Reserved for future state-aware strategy variants.

        top_n = int(cfg["top_n"])
        short_lookback_bars = int(cfg["short_lookback_bars"])
        medium_lookback_bars = int(cfg["medium_lookback_bars"])
        short_weight = float(cfg["short_weight"])
        medium_weight = float(cfg["medium_weight"])
        use_regime_filter = bool(cfg["use_regime_filter"])
        btc_symbol = str(cfg["btc_symbol"])
        regime_lookback_bars = int(cfg["regime_lookback_bars"])
        min_eligible_assets = int(cfg["min_eligible_assets"])
        max_position_weight = cfg["max_position_weight"]
        max_gross_exposure = cfg["max_gross_exposure"]
        include_cash_symbol = bool(cfg["include_cash_symbol"])
        cash_symbol = str(cfg["cash_symbol"])

        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if min_eligible_assets <= 0:
            raise ValueError("min_eligible_assets must be positive")

        close = _clean_close_prices(close_prices)
        ts = pd.Timestamp(timestamp)
        if ts not in close.index:
            raise KeyError(f"timestamp not found in close index: {ts}")

        min_history_bars_cfg = cfg["min_history_bars"]
        if min_history_bars_cfg is None:
            min_history_bars = max(short_lookback_bars, medium_lookback_bars)
        else:
            min_history_bars = int(min_history_bars_cfg)
            if min_history_bars <= 0:
                raise ValueError("min_history_bars must be positive when provided")

        momentum_score = compute_momentum_score(
            close=close,
            short_lookback_bars=short_lookback_bars,
            medium_lookback_bars=medium_lookback_bars,
            short_weight=short_weight,
            medium_weight=medium_weight,
        )

        valid_price = close.notna()
        history_count = valid_price.astype(int).cumsum()
        eligible_mask = valid_price.loc[ts] & (history_count.loc[ts] >= min_history_bars)

        eligible_symbols_cfg = cfg["eligible_symbols"]
        if eligible_symbols_cfg is not None:
            allowlist = set(str(symbol) for symbol in eligible_symbols_cfg)
            eligible_mask = eligible_mask & eligible_mask.index.to_series().isin(allowlist)

        min_median_volume = cfg["min_median_volume"]
        if min_median_volume is not None:
            volume = cfg["volume"]
            if volume is None:
                raise ValueError("config['volume'] is required when min_median_volume is provided")
            volume_frame = volume.sort_index().astype(float).reindex(index=close.index, columns=close.columns)
            rolling_median = volume_frame.rolling(
                window=min_history_bars,
                min_periods=min_history_bars,
            ).median()
            eligible_mask = eligible_mask & (rolling_median.loc[ts] >= float(min_median_volume))

        eligible_symbols = close.columns[eligible_mask.values]

        risk_on = True
        if use_regime_filter:
            risk_on = check_regime_filter(
                close=close,
                rebalance_timestamp=ts,
                btc_symbol=btc_symbol,
                ma_lookback_bars=regime_lookback_bars,
            )

        selected: list[str] = []
        if risk_on and len(eligible_symbols) >= min_eligible_assets:
            ranked = rank_symbols_for_date(
                momentum_score=momentum_score,
                rebalance_timestamp=ts,
                top_n=len(close.columns),
            )
            eligible_set = set(eligible_symbols)
            selected = [symbol for symbol in ranked if symbol in eligible_set][:top_n]

        target_weights = _build_target_weights(close.columns, selected)
        target_weights = _apply_position_weight_cap(target_weights, max_position_weight=max_position_weight)
        target_weights = _apply_gross_exposure_cap(target_weights, max_gross_exposure=max_gross_exposure)

        weights_dict = {
            symbol: float(weight)
            for symbol, weight in target_weights.items()
            if float(weight) > 0.0
        }

        if include_cash_symbol:
            cash_weight = max(0.0, 1.0 - float(target_weights.sum()))
            if cash_weight > 0:
                weights_dict[cash_symbol] = cash_weight

        self.validate_weights(weights_dict)
        return weights_dict
