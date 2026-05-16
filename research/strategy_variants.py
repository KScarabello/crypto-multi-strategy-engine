"""Weight-construction variants for momentum and reversal research experiments."""

from __future__ import annotations

import pandas as pd


def _validate_signal_frame(signal: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate input signal frame used by strategy constructors."""
    if not isinstance(signal, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if signal.empty:
        raise ValueError(f"{name} is empty")
    return signal.astype(float)


def _equal_weight_row(symbols: pd.Index, selected: set[str]) -> pd.Series:
    """Build one long-only equal-weight row from a selected symbol set."""
    row = pd.Series(0.0, index=symbols, dtype=float)
    if selected:
        w = 1.0 / len(selected)
        row.loc[list(selected)] = w
    return row


def _empty_weight_frame(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Return a zero-filled timestamp x symbol weight matrix."""
    return pd.DataFrame(0.0, index=index, columns=columns, dtype=float)


def time_series_momentum_weights(momentum_signal: pd.DataFrame) -> pd.DataFrame:
    """Long assets with positive medium-term momentum, equal-weighted.

    Rule:
    - Hold symbol at time t if momentum_signal[t, symbol] > 0
    - Allocate equal weight across held symbols
    - If none qualify, hold cash (all-zero row)
    """
    momentum = _validate_signal_frame(momentum_signal, "momentum_signal")
    weights = _empty_weight_frame(momentum.index, momentum.columns)

    for ts, row in momentum.iterrows():
        selected = set(row[row > 0.0].dropna().index)
        weights.loc[ts] = _equal_weight_row(momentum.columns, selected)

    return weights


def short_term_reversal_weights(
    overextension_signal: pd.DataFrame,
    pullback_threshold: float = 0.0,
    max_positions: int | None = None,
) -> pd.DataFrame:
    """Experimental standalone reversal rule based on recent pullbacks.

    Rule:
    - Select symbols whose short-term return <= pullback_threshold
    - Optionally keep only the most negative max_positions symbols
    - Equal weight selected symbols; otherwise hold cash
    """
    over = _validate_signal_frame(overextension_signal, "overextension_signal")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive when provided")

    weights = _empty_weight_frame(over.index, over.columns)
    for ts, row in over.iterrows():
        candidates = row[row <= pullback_threshold].dropna().sort_values(ascending=True)
        if max_positions is not None:
            candidates = candidates.head(max_positions)
        selected = set(candidates.index)
        weights.loc[ts] = _equal_weight_row(over.columns, selected)

    return weights


def momentum_with_entry_filter_weights(
    momentum_signal: pd.DataFrame,
    overextension_signal: pd.DataFrame,
    entry_overextension_threshold: float = 0.15,
) -> pd.DataFrame:
    """Momentum entries gated by overextension threshold.

    Rule:
    - Enter/hold symbol at time t if momentum > 0 and overextension < entry threshold
    - Equal weight selected symbols, else hold cash
    """
    momentum = _validate_signal_frame(momentum_signal, "momentum_signal")
    over = _validate_signal_frame(overextension_signal, "overextension_signal")
    if not momentum.index.equals(over.index) or not momentum.columns.equals(over.columns):
        raise ValueError("momentum_signal and overextension_signal must have identical index/columns")

    weights = _empty_weight_frame(momentum.index, momentum.columns)
    for ts in momentum.index:
        mom_row = momentum.loc[ts]
        over_row = over.loc[ts]
        selected = set(
            mom_row[(mom_row > 0.0) & (over_row < entry_overextension_threshold)].dropna().index
        )
        weights.loc[ts] = _equal_weight_row(momentum.columns, selected)

    return weights


def momentum_with_exit_signal_weights(
    momentum_signal: pd.DataFrame,
    overextension_signal: pd.DataFrame,
    exit_overextension_threshold: float = 0.30,
) -> pd.DataFrame:
    """Momentum entries with stateful overextension exits.

    Rule:
    - New entries: momentum > 0
    - Existing holding exits when momentum < 0 or overextension > exit threshold
    - Exit and re-entry are not allowed on the same bar for the same symbol
    - Equal weight across surviving holdings
    """
    momentum = _validate_signal_frame(momentum_signal, "momentum_signal")
    over = _validate_signal_frame(overextension_signal, "overextension_signal")
    if not momentum.index.equals(over.index) or not momentum.columns.equals(over.columns):
        raise ValueError("momentum_signal and overextension_signal must have identical index/columns")

    weights = _empty_weight_frame(momentum.index, momentum.columns)
    held: set[str] = set()

    for ts in momentum.index:
        mom_row = momentum.loc[ts]
        over_row = over.loc[ts]

        exited = {
            symbol
            for symbol in held
            if (mom_row.get(symbol, float("nan")) < 0.0)
            or (over_row.get(symbol, float("nan")) > exit_overextension_threshold)
        }
        held -= exited

        entries = {
            symbol
            for symbol in momentum.columns
            if (symbol not in held)
            and (symbol not in exited)
            and (mom_row.get(symbol, float("nan")) > 0.0)
        }
        held |= entries

        weights.loc[ts] = _equal_weight_row(momentum.columns, held)

    return weights


def momentum_with_entry_filter_and_exit_signal_weights(
    momentum_signal: pd.DataFrame,
    overextension_signal: pd.DataFrame,
    entry_overextension_threshold: float = 0.15,
    exit_overextension_threshold: float = 0.30,
) -> pd.DataFrame:
    """Stateful momentum strategy with separate entry and exit overextension rules.

    Rule:
    - New entries: momentum > 0 and overextension < entry threshold
    - Existing holdings continue unless momentum < 0 or overextension > exit threshold
    - Uses different entry and exit thresholds to reduce churn risk
    - Equal weight across surviving holdings
    """
    momentum = _validate_signal_frame(momentum_signal, "momentum_signal")
    over = _validate_signal_frame(overextension_signal, "overextension_signal")
    if not momentum.index.equals(over.index) or not momentum.columns.equals(over.columns):
        raise ValueError("momentum_signal and overextension_signal must have identical index/columns")
    if entry_overextension_threshold >= exit_overextension_threshold:
        raise ValueError("entry_overextension_threshold should be lower than exit_overextension_threshold")

    weights = _empty_weight_frame(momentum.index, momentum.columns)
    held: set[str] = set()

    for ts in momentum.index:
        mom_row = momentum.loc[ts]
        over_row = over.loc[ts]

        exited = {
            symbol
            for symbol in held
            if (mom_row.get(symbol, float("nan")) < 0.0)
            or (over_row.get(symbol, float("nan")) > exit_overextension_threshold)
        }
        held -= exited

        entries = {
            symbol
            for symbol in momentum.columns
            if (symbol not in held)
            and (symbol not in exited)
            and (mom_row.get(symbol, float("nan")) > 0.0)
            and (over_row.get(symbol, float("nan")) < entry_overextension_threshold)
        }
        held |= entries

        weights.loc[ts] = _equal_weight_row(momentum.columns, held)

    return weights
