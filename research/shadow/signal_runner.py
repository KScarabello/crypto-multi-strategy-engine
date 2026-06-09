"""Per-bar signal generation — no live orders, no exchange connectivity."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timezone

import pandas as pd

from research.shadow.specs import (
    FROZEN_CANDIDATES,
    SPEC_HASHES,
    PROSPECTIVE_START,
    SpecificationError,
    assert_spec_integrity,
)
from research.shadow.portfolio import ShadowPortfolio
from strategies.cross_sectional_momentum import compute_momentum_score


@dataclass
class BarDecision:
    candidate_name: str
    spec_hash: str
    decision_ts: str
    execution_ts: str
    data_cutoff_ts: str
    btc_regime_state: bool
    btc_price: float
    btc_ma_value: float
    should_rebalance: bool
    current_holdings: dict[str, float]
    target_holdings: dict[str, float]
    momentum_scores: dict[str, float]
    momentum_ranks: dict[str, int]
    hypothetical_trades: list[dict]
    bar_count_at_decision: int
    rebalance_bar_index: int
    is_prospective: bool


def _bar_freq(close: pd.DataFrame) -> pd.Timedelta | None:
    """Infer bar frequency from index."""
    if len(close.index) < 2:
        return None
    diffs = close.index[1:] - close.index[:-1]
    return diffs.median()


def compute_gate_at_bar(
    btc_close: pd.Series,
    btc_ma_bars: int,
    entry_confirm_bars: int,
    exit_confirm_bars: int,
    entry_buffer_pct: float,
    last_gate: bool,
    gate_entry_confirm_count: int,
) -> tuple[bool, int, float, float]:
    """
    Returns: (new_gate_state, new_confirm_count, btc_price, ma_value)
    Uses only btc_close (already sliced to <= bar_ts).
    """
    if len(btc_close) == 0:
        return False, 0, float("nan"), float("nan")

    btc_price = float(btc_close.iloc[-1])

    if len(btc_close) < btc_ma_bars:
        return False, 0, btc_price, float("nan")

    ma_val = float(btc_close.iloc[-btc_ma_bars:].mean())
    entry_threshold = ma_val * (1 + entry_buffer_pct / 100.0)
    exit_threshold = ma_val  # exit_buffer_pct=0

    above_threshold = btc_price >= entry_threshold
    below_exit = btc_price < exit_threshold

    if last_gate:
        # Risk-on: exit immediately on confirmed bar below MA
        if exit_confirm_bars <= 1:
            new_gate = not below_exit
        else:
            new_gate = not below_exit
        new_count = 0
        return new_gate, new_count, btc_price, ma_val
    else:
        # Risk-off: need entry_confirm_bars consecutive bars above threshold
        if above_threshold:
            new_count = gate_entry_confirm_count + 1
            if new_count >= entry_confirm_bars:
                return True, 0, btc_price, ma_val
            else:
                return False, new_count, btc_price, ma_val
        else:
            return False, 0, btc_price, ma_val


def _get_momentum_scores_and_ranks(
    close: pd.DataFrame,
    bar_ts: pd.Timestamp,
    universe: list[str],
    short_lookback_bars: int,
    medium_lookback_bars: int,
    min_history_bars: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute momentum scores and ranks using only data up to bar_ts."""
    avail_symbols = [s for s in universe if s in close.columns]
    if not avail_symbols:
        return {}, {}

    sub = close[avail_symbols].loc[close.index <= bar_ts]

    if len(sub) < min_history_bars:
        return {}, {}

    try:
        scores_df = compute_momentum_score(
            close=sub,
            short_lookback_bars=short_lookback_bars,
            medium_lookback_bars=medium_lookback_bars,
        )
    except Exception:
        return {}, {}

    if bar_ts not in scores_df.index:
        return {}, {}

    row = scores_df.loc[bar_ts]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    scores = {}
    for sym in avail_symbols:
        val = row.get(sym, float("nan"))
        if not (isinstance(val, float) and math.isnan(val)):
            scores[sym] = float(val)

    # rank: 1 = best momentum
    sorted_syms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {sym: i + 1 for i, (sym, _) in enumerate(sorted_syms)}
    return scores, ranks


def _apply_rank_stability(
    current_holdings: dict[str, float],
    top_n_selected: list[str],
    scores: dict[str, float],
    ranks: dict[str, int],
    rank_stability: dict,
    portfolio: ShadowPortfolio,
    top_n: int,
    risk_on: bool,
) -> tuple[list[str], int]:
    """Apply rank-stability rules. Returns (final_selected, rank_replacements_this_bar)."""
    if not risk_on:
        return [], 0

    rank_buffer = rank_stability.get("rank_buffer", 0)
    challenger_confirm_rebs = rank_stability.get("challenger_confirm_rebs", 0)
    min_hold_rebs = rank_stability.get("min_hold_rebs", 0)

    incumbents = [s for s in current_holdings if current_holdings[s] > 0]

    # Start with top-N selection
    final_selected = list(top_n_selected)

    # Apply rank_buffer: retain incumbents whose rank <= top_n + rank_buffer
    if rank_buffer > 0:
        for inc in incumbents:
            if inc not in final_selected:
                inc_rank = ranks.get(inc, 9999)
                if inc_rank <= top_n + rank_buffer:
                    # Retain incumbent; drop worst challenger if needed
                    if len(final_selected) >= top_n:
                        # Find the lowest-ranked in final_selected that's not an incumbent
                        non_incumbents_in_sel = [
                            s for s in final_selected if s not in incumbents
                        ]
                        if non_incumbents_in_sel:
                            worst = max(
                                non_incumbents_in_sel,
                                key=lambda s: ranks.get(s, 9999),
                            )
                            final_selected = [s for s in final_selected if s != worst]
                    final_selected.append(inc)

    # Apply min_hold: don't drop incumbents that haven't held long enough
    if min_hold_rebs > 0:
        for inc in incumbents:
            age = portfolio.hold_age.get(inc, 0)
            if age < min_hold_rebs and inc not in final_selected:
                if len(final_selected) >= top_n:
                    non_incumbents_in_sel = [
                        s for s in final_selected if s not in incumbents
                    ]
                    if non_incumbents_in_sel:
                        worst = max(
                            non_incumbents_in_sel,
                            key=lambda s: ranks.get(s, 9999),
                        )
                        final_selected = [s for s in final_selected if s != worst]
                final_selected.append(inc)

    # Apply challenger_confirm: new entrants must appear in top_n+buffer for N consecutive rebs
    if challenger_confirm_rebs > 0:
        confirmed_new = []
        for sym in final_selected:
            if sym in incumbents:
                confirmed_new.append(sym)
            else:
                streak = portfolio.challenger_streak.get(sym, 0) + 1
                if streak >= challenger_confirm_rebs:
                    confirmed_new.append(sym)
                # else: not yet confirmed — keep incumbent if possible
        # Fill remaining slots with top-ranked unconfirmed if needed
        slots = top_n - len(confirmed_new)
        if slots > 0:
            for sym in top_n_selected:
                if sym not in confirmed_new and slots > 0:
                    confirmed_new.append(sym)
                    slots -= 1
        final_selected = confirmed_new[:top_n]

    # Trim to top_n
    final_selected = final_selected[:top_n]

    # Count replacements
    prev_set = set(incumbents)
    new_set = set(final_selected)
    replacements = len(new_set - prev_set)

    return final_selected, replacements


def _update_stability_state(
    portfolio: ShadowPortfolio,
    new_selected: list[str],
    should_rebalance: bool,
) -> None:
    """Update hold_age, challenger_streak, cooldown in-place."""
    if not should_rebalance:
        return

    incumbents = set(s for s in portfolio.holdings if portfolio.holdings.get(s, 0) > 0)
    new_set = set(new_selected)

    new_hold_age = {}
    new_challenger_streak = {}

    for sym in new_set:
        if sym in incumbents:
            new_hold_age[sym] = portfolio.hold_age.get(sym, 0) + 1
            new_challenger_streak[sym] = 0
        else:
            new_hold_age[sym] = 1
            streak = portfolio.challenger_streak.get(sym, 0) + 1
            new_challenger_streak[sym] = streak

    for sym in incumbents - new_set:
        cd = portfolio.cooldown.get(sym, 0)
        if cd > 0:
            portfolio.cooldown[sym] = cd - 1

    portfolio.hold_age = new_hold_age
    portfolio.challenger_streak = new_challenger_streak


def _generate_strategy_signal(
    candidate_name: str,
    spec: dict,
    bar_ts: pd.Timestamp,
    close: pd.DataFrame,
    bar_count: int,
    portfolio: ShadowPortfolio,
) -> tuple[dict[str, float], bool, bool, float, float, dict[str, float], dict[str, int]]:
    """
    Returns: (target_weights, btc_regime, should_rebalance, btc_price, btc_ma,
              momentum_scores, momentum_ranks)
    """
    regime = spec["regime"]
    rank_stability = spec.get("rank_stability") or {}

    btc_ma_bars = regime["btc_ma_bars"]
    entry_confirm_bars = regime["entry_confirm_bars"]
    exit_confirm_bars = regime["exit_confirm_bars"]
    entry_buffer_pct = regime["entry_buffer_pct"]
    rebalance_bars = regime["rebalance_bars"]
    top_n = regime["top_n"]
    short_lookback_bars = regime["short_lookback_bars"]
    medium_lookback_bars = regime["medium_lookback_bars"]
    min_history_bars = regime["min_history_bars"]
    universe = regime["universe"]

    btc_col = "BTC/USD"
    btc_series = close[btc_col].loc[close.index <= bar_ts].dropna() if btc_col in close.columns else pd.Series(dtype=float)

    new_gate, new_confirm_count, btc_price, ma_val = compute_gate_at_bar(
        btc_close=btc_series,
        btc_ma_bars=btc_ma_bars,
        entry_confirm_bars=entry_confirm_bars,
        exit_confirm_bars=exit_confirm_bars,
        entry_buffer_pct=entry_buffer_pct,
        last_gate=portfolio.last_gate,
        gate_entry_confirm_count=portfolio.gate_entry_confirm_count,
    )
    portfolio.gate_entry_confirm_count = new_confirm_count
    portfolio.last_gate = new_gate

    should_rebalance = (bar_count > 0) and (bar_count % rebalance_bars == 0)

    scores, ranks = _get_momentum_scores_and_ranks(
        close=close,
        bar_ts=bar_ts,
        universe=universe,
        short_lookback_bars=short_lookback_bars,
        medium_lookback_bars=medium_lookback_bars,
        min_history_bars=min_history_bars,
    )

    if not new_gate:
        target_weights: dict[str, float] = {}
        return target_weights, new_gate, should_rebalance, btc_price, ma_val, scores, ranks

    if not should_rebalance and portfolio.holdings:
        return dict(portfolio.holdings), new_gate, should_rebalance, btc_price, ma_val, scores, ranks

    if not scores:
        target_weights = {}
        return target_weights, new_gate, should_rebalance, btc_price, ma_val, scores, ranks

    # Top-N raw selection
    sorted_syms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n_selected = [s for s, _ in sorted_syms[:top_n]]

    # Apply rank stability
    final_selected, replacements = _apply_rank_stability(
        current_holdings=portfolio.holdings,
        top_n_selected=top_n_selected,
        scores=scores,
        ranks=ranks,
        rank_stability=rank_stability,
        portfolio=portfolio,
        top_n=top_n,
        risk_on=new_gate,
    )

    portfolio.rank_replacements += replacements
    _update_stability_state(portfolio, final_selected, should_rebalance)

    if final_selected:
        w = 1.0 / len(final_selected)
        target_weights = {s: w for s in final_selected}
    else:
        target_weights = {}

    return target_weights, new_gate, should_rebalance, btc_price, ma_val, scores, ranks


def run_bar(
    candidate_name: str,
    bar_ts: pd.Timestamp,
    close: pd.DataFrame,
    portfolio: ShadowPortfolio,
    spec_hash: str,
) -> BarDecision:
    """Generate a signal for one bar. No live orders.

    Returns a BarDecision with target weights and hypothetical trades.
    Does NOT update portfolio equity — caller does that after getting next bar price.
    """
    assert_spec_integrity(candidate_name)

    if spec_hash != SPEC_HASHES[candidate_name]:
        raise SpecificationError(
            f"spec_hash mismatch for {candidate_name!r}: "
            f"expected {SPEC_HASHES[candidate_name]!r}, got {spec_hash!r}"
        )

    spec = FROZEN_CANDIDATES[candidate_name]

    # Slice data: only use data up to and including bar_ts
    close_up_to = close.loc[close.index <= bar_ts].copy()
    data_cutoff_ts = str(close_up_to.index[-1]) if len(close_up_to) > 0 else str(bar_ts)

    # Determine execution_ts (one bar later)
    freq = _bar_freq(close)
    if freq is not None:
        execution_ts = bar_ts + freq
    else:
        execution_ts = bar_ts + pd.Timedelta(hours=4)

    prospective_start = pd.Timestamp(PROSPECTIVE_START)
    is_prospective = bar_ts >= prospective_start

    bar_count = portfolio.bar_count + 1
    regime_param = spec.get("regime")
    rebalance_bars = regime_param["rebalance_bars"] if regime_param else 12

    universe = regime_param["universe"] if regime_param else ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]

    btc_price = float("nan")
    btc_ma_value = float("nan")
    btc_regime_state = False
    should_rebalance = False
    momentum_scores: dict[str, float] = {}
    momentum_ranks: dict[str, int] = {}
    current_holdings = dict(portfolio.holdings)

    if candidate_name == "btc_buyhold":
        # BTC buy-and-hold: first bar buy, never rebalance
        if portfolio.bar_count == 0:
            target_holdings = {"BTC/USD": 1.0}
        else:
            target_holdings = dict(portfolio.holdings) if portfolio.holdings else {"BTC/USD": 1.0}
        btc_series = close_up_to["BTC/USD"].dropna() if "BTC/USD" in close_up_to.columns else pd.Series(dtype=float)
        btc_price = float(btc_series.iloc[-1]) if len(btc_series) > 0 else float("nan")
        btc_regime_state = True
        should_rebalance = portfolio.bar_count == 0

    elif candidate_name == "ewb_buyhold":
        # Equal-weight buy-and-hold: first bar buy, never rebalance
        if portfolio.bar_count == 0:
            w = 1.0 / len(universe)
            target_holdings = {s: w for s in universe}
        else:
            target_holdings = dict(portfolio.holdings) if portfolio.holdings else {s: 1.0 / len(universe) for s in universe}
        btc_series = close_up_to["BTC/USD"].dropna() if "BTC/USD" in close_up_to.columns else pd.Series(dtype=float)
        btc_price = float(btc_series.iloc[-1]) if len(btc_series) > 0 else float("nan")
        btc_regime_state = True
        should_rebalance = portfolio.bar_count == 0

    else:
        # Strategy candidates
        (
            target_holdings,
            btc_regime_state,
            should_rebalance,
            btc_price,
            btc_ma_value,
            momentum_scores,
            momentum_ranks,
        ) = _generate_strategy_signal(
            candidate_name=candidate_name,
            spec=spec,
            bar_ts=bar_ts,
            close=close_up_to,
            bar_count=bar_count,
            portfolio=portfolio,
        )

    # Compute hypothetical trades
    hypothetical_trades = []
    fee_bps = regime_param["fee_bps"] if regime_param else 10
    slippage_bps = regime_param["slippage_bps"] if regime_param else 5

    if should_rebalance or portfolio.bar_count == 0:
        all_symbols = set(list(current_holdings.keys()) + list(target_holdings.keys()))
        for sym in all_symbols:
            old_w = current_holdings.get(sym, 0.0)
            new_w = target_holdings.get(sym, 0.0)
            delta = new_w - old_w
            if abs(delta) > 1e-6:
                notional = abs(delta) * portfolio.equity
                fee_est = notional * fee_bps / 10_000.0
                slip_est = notional * slippage_bps / 10_000.0
                hypothetical_trades.append({
                    "symbol": sym,
                    "side": "buy" if delta > 0 else "sell",
                    "weight_change": delta,
                    "notional_estimate": notional,
                    "fee_est": fee_est,
                    "slip_est": slip_est,
                })

    rebalance_bar_index = portfolio.rebalance_count + (1 if should_rebalance else 0)

    return BarDecision(
        candidate_name=candidate_name,
        spec_hash=spec_hash,
        decision_ts=str(bar_ts),
        execution_ts=str(execution_ts),
        data_cutoff_ts=data_cutoff_ts,
        btc_regime_state=btc_regime_state,
        btc_price=btc_price,
        btc_ma_value=btc_ma_value,
        should_rebalance=should_rebalance,
        current_holdings=current_holdings,
        target_holdings=target_holdings,
        momentum_scores=momentum_scores,
        momentum_ranks=momentum_ranks,
        hypothetical_trades=hypothetical_trades,
        bar_count_at_decision=bar_count,
        rebalance_bar_index=rebalance_bar_index,
        is_prospective=is_prospective,
    )
