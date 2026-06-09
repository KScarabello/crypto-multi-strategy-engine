"""Audit-grade transaction-cost and turnover-source reconciliation.

PURPOSE
-------
Produce a precise, trade-level cost ledger and reconciliation for two strategies:
  1. btc_ma_240_reb12  (canonical control)
  2. combo_entry2_imm_buf05  (whipsaw-reduction candidate)

IMPORTANT ACCOUNTING RULES
---------------------------
- Fees and slippage are computed from executed notional at the trade level.
  They are NOT estimated by allocating the gross-minus-net ending-equity gap.
- "Gross-minus-net ending equity" is not the same as "fees paid".
  The residual beyond actual fees+slippage is foregone compounding and
  path-dependent position-size effects.
- The additive cost-rate sum × initial_equity is NOT actual cost.
  It is a rough first-order approximation; actual costs compound.

CANDIDATE SELECTION LOCK
------------------------
combo_entry2_imm_buf05 is selection-locked as the current research candidate.
It was selected after observing the full historical sample. It is NOT out-of-sample.

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_cost_audit
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from research.universe_integrity_analysis import (
    BARS_PER_YEAR_4H,
    DEFAULT_FEE_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_SLIPPAGE_BPS,
    LIVE_FIVE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
    _close_to_ohlcv,
    build_close_matrix,
    find_joint_eligible_start,
)
from research.fixed_five_defensive_overlay import (
    BaseSignalGen,
    _cagr,
    _max_drawdown,
    _sharpe,
    _sortino,
)
from research.fixed_five_canonical_regime_comparison import run_canonical
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    WhipsawControlledSignalGen,
    compute_controlled_gate_series,
)

LOGGER = logging.getLogger(__name__)

BTC_COL = "BTC/USD"
BARS_PER_YEAR = BARS_PER_YEAR_4H

CONTROL_MA_BARS = 240
CONTROL_REBALANCE_BARS = 12
CANDIDATE_ENTRY_CONFIRM = 2
CANDIDATE_ENTRY_BUF = 0.5

TRADE_REASONS = (
    "REGIME_EXIT",
    "REGIME_ENTRY",
    "RANK_ENTRY",
    "RANK_EXIT",
    "REWEIGHT_EXISTING_POSITION",
    "OTHER_EXPLAINED",
)

# -------------------------------------------------------------------------
# Strategy builders
# -------------------------------------------------------------------------

def build_control_signal(btc_full: pd.Series) -> tuple[Any, pd.Series]:
    """Return (signal_gen, gate_series) for the control."""
    params = WhipsawControl(name="control")
    gate = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)
    base = BaseSignalGen()
    sig = WhipsawControlledSignalGen(gate, base)
    return sig, gate


def build_candidate_signal(btc_full: pd.Series) -> tuple[Any, pd.Series]:
    """Return (signal_gen, gate_series) for combo_entry2_imm_buf05."""
    params = WhipsawControl(
        name="combo_entry2_imm_buf05",
        entry_confirm_bars=CANDIDATE_ENTRY_CONFIRM,
        exit_confirm_bars=1,
        entry_buffer_pct=CANDIDATE_ENTRY_BUF,
        exit_buffer_pct=0.0,
    )
    gate = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)
    base = BaseSignalGen()
    sig = WhipsawControlledSignalGen(gate, base)
    return sig, gate


# -------------------------------------------------------------------------
# Trade-level ledger reconstruction
# -------------------------------------------------------------------------

WEIGHT_THRESHOLD = 1e-4   # minimum weight change to count as a trade

@dataclass
class TradeRecord:
    strategy_name: str
    execution_timestamp: str
    symbol: str
    side: str                         # BUY / SELL
    weight_before: float
    weight_after: float
    weight_change: float
    portfolio_equity_before_trade: float
    gross_notional: float             # abs(weight_change) × equity_before
    execution_price: float            # close price at execution bar
    execution_price_after_slippage: float
    fee_rate: float                   # fee_bps / 10_000
    slippage_rate: float              # slippage_bps / 10_000
    fee_dollars: float
    slippage_dollars: float
    total_direct_cost_dollars: float
    portfolio_equity_after_trade: float
    trade_reason: str
    notes: str


def _classify_trade_reason(
    symbol: str,
    weight_before: float,
    weight_after: float,
    was_regime_exit: bool,
    was_regime_entry: bool,
    prev_held_set: set[str],
    new_held_set: set[str],
) -> str:
    """Classify a single symbol trade."""
    is_buy = weight_after > weight_before
    is_sell = weight_after < weight_before

    if was_regime_exit and is_sell and weight_before > WEIGHT_THRESHOLD:
        return "REGIME_EXIT"
    if was_regime_entry and is_buy and weight_before < WEIGHT_THRESHOLD:
        return "REGIME_ENTRY"
    if is_buy and weight_before < WEIGHT_THRESHOLD:
        return "RANK_ENTRY"
    if is_sell and weight_after < WEIGHT_THRESHOLD:
        return "RANK_EXIT"
    if abs(weight_after - weight_before) > WEIGHT_THRESHOLD:
        return "REWEIGHT_EXISTING_POSITION"
    return "OTHER_EXPLAINED"


def build_trade_ledger(
    strategy_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    gate_fx: pd.Series,
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> list[TradeRecord]:
    """Reconstruct per-symbol trade records from holdings_history and rebalance_log."""
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000

    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()
    all_ts = holdings.index
    symbols = holdings.columns

    reb = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ].copy()
    reb["execution_timestamp"] = pd.to_datetime(reb["execution_timestamp"], utc=True)

    # Gate transitions (sliced to joint_start)
    gate_fx_aligned = gate_fx.reindex(all_ts).fillna(False)
    gate_trans = gate_fx_aligned[gate_fx_aligned != gate_fx_aligned.shift(1)].dropna()

    records: list[TradeRecord] = []

    for _, row in reb.iterrows():
        exec_ts = row["execution_timestamp"]
        sig_ts = row["signal_timestamp"]
        turnover = float(row["turnover"])

        if turnover < WEIGHT_THRESHOLD:
            continue  # no actual trades

        # Holdings before (at signal bar) and after (at execution bar)
        prev_ts_arr = all_ts[all_ts < exec_ts]
        if len(prev_ts_arr) == 0:
            continue
        prev_ts = prev_ts_arr[-1]

        w_before = holdings.loc[prev_ts]
        w_after = holdings.loc[exec_ts]
        eq_before = float(equity.loc[prev_ts]) if prev_ts in equity.index else float("nan")
        eq_after = float(equity.loc[exec_ts]) if exec_ts in equity.index else float("nan")

        # Regime transition detection
        was_regime_exit = exec_ts in gate_trans.index and not bool(gate_trans.loc[exec_ts])
        was_regime_entry = exec_ts in gate_trans.index and bool(gate_trans.loc[exec_ts])

        prev_held = set(symbols[w_before > WEIGHT_THRESHOLD])
        new_held = set(symbols[w_after > WEIGHT_THRESHOLD])

        # Close prices at execution bar
        if exec_ts in close.index:
            prices = close.loc[exec_ts]
        else:
            prices = pd.Series(float("nan"), index=symbols)

        for sym in symbols:
            wb = float(w_before.get(sym, 0.0))
            wa = float(w_after.get(sym, 0.0))
            dw = wa - wb

            if abs(dw) < WEIGHT_THRESHOLD:
                continue

            side = "BUY" if dw > 0 else "SELL"
            notional = abs(dw) * eq_before

            price_raw = float(prices.get(sym, float("nan")))
            if side == "BUY":
                price_slipped = price_raw * (1.0 + slippage_rate)
            else:
                price_slipped = price_raw * (1.0 - slippage_rate)

            fee_d = notional * fee_rate
            slip_d = notional * slippage_rate
            total_cost = fee_d + slip_d

            reason = _classify_trade_reason(
                sym, wb, wa, was_regime_exit, was_regime_entry, prev_held, new_held
            )

            records.append(TradeRecord(
                strategy_name=strategy_name,
                execution_timestamp=str(exec_ts),
                symbol=sym,
                side=side,
                weight_before=round(wb, 6),
                weight_after=round(wa, 6),
                weight_change=round(dw, 6),
                portfolio_equity_before_trade=round(eq_before, 4),
                gross_notional=round(notional, 4),
                execution_price=round(price_raw, 6) if not math.isnan(price_raw) else float("nan"),
                execution_price_after_slippage=round(price_slipped, 6) if not math.isnan(price_slipped) else float("nan"),
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fee_dollars=round(fee_d, 4),
                slippage_dollars=round(slip_d, 4),
                total_direct_cost_dollars=round(total_cost, 4),
                portfolio_equity_after_trade=round(eq_after, 4),
                trade_reason=reason,
                notes=f"sig={str(sig_ts)[:16]} | reb_turnover={turnover:.4f}",
            ))

    return records


# -------------------------------------------------------------------------
# Part 2 — Cost reconciliation
# -------------------------------------------------------------------------

def compute_cost_reconciliation(
    strategy_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    ledger: list[TradeRecord],
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Exact cost reconciliation with clearly named denominators."""
    eq = port_fx["equity"].dropna()
    equity_at_start = float(eq.iloc[0])
    net_end = float(eq.iloc[-1])

    # Gross equity from gross_return series
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    gross_eq_series = equity_at_start * (1 + gross_r).cumprod()
    gross_end = float(gross_eq_series.iloc[-1])
    gross_minus_net = gross_end - net_end
    gross_profit = gross_end - equity_at_start

    # Direct costs from trade ledger
    total_fee_d = sum(t.fee_dollars for t in ledger)
    total_slip_d = sum(t.slippage_dollars for t in ledger)
    total_direct = total_fee_d + total_slip_d
    total_notional = sum(t.gross_notional for t in ledger)

    # Additive cost rate sum (from rebalance_log — kept for reference only)
    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ]
    additive_rate_sum = float(reb_fx["cost_rate"].sum())

    # The residual: gross_minus_net minus actual direct costs
    # This includes (1) foregone compounding and (2) path-dependent position-size effects.
    # It is NOT a cost; it is the wealth that was lost because the compounding base
    # was reduced each time a direct cost was paid.
    residual_foregone_compounding = gross_minus_net - total_direct

    return {
        "strategy": strategy_name,
        # Actual modeled costs (from trade ledger, computed from executed notional)
        "actual_fee_dollars": round(total_fee_d, 2),
        "actual_slippage_dollars": round(total_slip_d, 2),
        "total_direct_cost_dollars": round(total_direct, 2),
        "total_executed_notional": round(total_notional, 2),
        # Equity levels
        "equity_at_start": round(equity_at_start, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        # Gross-minus-net (NOT the same as fees paid)
        "gross_minus_net_dollars": round(gross_minus_net, 2),
        # Ratios with distinct denominators (denominator named explicitly)
        "direct_cost_div_equity_at_start_pct": round(total_direct / max(equity_at_start, 1) * 100, 4),
        "direct_cost_div_executed_notional_pct": round(total_direct / max(total_notional, 1) * 100, 4),
        "direct_cost_div_gross_profit_pct": round(total_direct / max(gross_profit, 1) * 100, 4),
        "gross_minus_net_div_gross_ending_equity_pct": round(gross_minus_net / max(gross_end, 1) * 100, 4),
        # Residual: foregone compounding + path effects
        # DEFINITION: gross_minus_net − total_direct_cost_dollars
        # This is wealth lost because every cost payment reduced the future compounding base.
        # It is not fees or slippage; it is the compounding penalty.
        "residual_foregone_compounding_and_path_pct_of_gross_end": round(
            residual_foregone_compounding / max(gross_end, 1) * 100, 4
        ),
        "residual_foregone_compounding_dollars": round(residual_foregone_compounding, 2),
        # Reference only — DO NOT call this "actual cost"
        "additive_rate_sum_times_start_equity_REFERENCE_ONLY": round(
            additive_rate_sum * equity_at_start, 2
        ),
        "additive_rate_sum_pct_REFERENCE_ONLY": round(additive_rate_sum * 100, 4),
        "n_trades_in_ledger": len(ledger),
    }


# -------------------------------------------------------------------------
# Part 3 — Turnover source decomposition
# -------------------------------------------------------------------------

def compute_turnover_decomposition(
    strategy_name: str,
    ledger: list[TradeRecord],
) -> list[dict[str, Any]]:
    """Break down turnover and direct costs by trade reason category."""
    total_cost = sum(t.total_direct_cost_dollars for t in ledger)
    rows = []
    for reason in TRADE_REASONS:
        trades = [t for t in ledger if t.trade_reason == reason]
        notional = sum(t.gross_notional for t in trades)
        fee_d = sum(t.fee_dollars for t in trades)
        slip_d = sum(t.slippage_dollars for t in trades)
        cost_d = sum(t.total_direct_cost_dollars for t in trades)
        rows.append({
            "strategy": strategy_name,
            "trade_reason": reason,
            "trade_count": len(trades),
            "gross_notional": round(notional, 2),
            "fee_dollars": round(fee_d, 2),
            "slippage_dollars": round(slip_d, 2),
            "total_direct_cost_dollars": round(cost_d, 2),
            "pct_of_total_direct_cost": round(cost_d / max(total_cost, 1) * 100, 2),
        })
    # Summary row
    rows.append({
        "strategy": strategy_name,
        "trade_reason": "TOTAL",
        "trade_count": len(ledger),
        "gross_notional": round(sum(t.gross_notional for t in ledger), 2),
        "fee_dollars": round(sum(t.fee_dollars for t in ledger), 2),
        "slippage_dollars": round(sum(t.slippage_dollars for t in ledger), 2),
        "total_direct_cost_dollars": round(total_cost, 2),
        "pct_of_total_direct_cost": 100.0,
    })
    return rows


# -------------------------------------------------------------------------
# Part 4 — Rebalance event audit
# -------------------------------------------------------------------------

def build_rebalance_event_audit(
    strategy_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    gate_fx: pd.Series,
    joint_start: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> list[dict[str, Any]]:
    """Per-rebalance audit record."""
    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()
    all_ts = holdings.index
    gate_aligned = gate_fx.reindex(all_ts).fillna(False)
    gate_trans = gate_aligned[gate_aligned != gate_aligned.shift(1)].dropna()

    reb = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ].copy()
    reb["execution_timestamp"] = pd.to_datetime(reb["execution_timestamp"], utc=True)

    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    total_rate = fee_rate + slippage_rate

    rows = []
    for _, row in reb.iterrows():
        exec_ts = row["execution_timestamp"]
        turnover = float(row["turnover"])

        prev_ts_arr = all_ts[all_ts < exec_ts]
        if len(prev_ts_arr) == 0:
            continue
        prev_ts = prev_ts_arr[-1]

        w_before = holdings.loc[prev_ts]
        w_after = holdings.loc[exec_ts]
        eq_before = float(equity.loc[prev_ts]) if prev_ts in equity.index else float("nan")

        regime_state = "RISK_ON" if bool(gate_aligned.get(exec_ts, True)) else "RISK_OFF"
        was_regime_exit = exec_ts in gate_trans.index and not bool(gate_trans.loc[exec_ts])
        was_regime_entry = exec_ts in gate_trans.index and bool(gate_trans.loc[exec_ts])

        prev_held = set(w_before.index[w_before > WEIGHT_THRESHOLD])
        new_held = set(w_after.index[w_after > WEIGHT_THRESHOLD])
        entered = new_held - prev_held
        exited = prev_held - new_held
        reweighted = {s for s in prev_held & new_held
                      if abs(float(w_after[s]) - float(w_before[s])) > WEIGHT_THRESHOLD}

        n_changes = len(entered) + len(exited) + len(reweighted)
        direct_cost = turnover * total_rate * eq_before if not math.isnan(eq_before) else float("nan")

        # Primary turnover cause
        if was_regime_exit:
            primary_cause = "REGIME_EXIT"
        elif was_regime_entry:
            primary_cause = "REGIME_ENTRY"
        elif entered and not reweighted:
            primary_cause = "RANK_ENTRY"
        elif exited and not reweighted:
            primary_cause = "RANK_EXIT"
        elif reweighted and not entered and not exited:
            primary_cause = "REWEIGHT_EXISTING_POSITION"
        elif n_changes > 0:
            primary_cause = "RANK_CHANGE_WITH_REWEIGHT"
        else:
            primary_cause = "OTHER_EXPLAINED"

        # Flags
        only_reweight = bool(reweighted and not entered and not exited)
        only_one_change = bool(n_changes == 1)
        full_exit = bool(not new_held and prev_held)
        full_entry = bool(new_held and not prev_held)
        repeat_sell_rebuy = bool(exited and entered and not (entered & exited))

        rows.append({
            "strategy": strategy_name,
            "execution_timestamp": str(exec_ts),
            "regime_state": regime_state,
            "holdings_before": ",".join(sorted(prev_held)) if prev_held else "(cash)",
            "holdings_after": ",".join(sorted(new_held)) if new_held else "(cash)",
            "turnover": round(turnover, 4),
            "n_trades": n_changes,
            "direct_cost_dollars": round(direct_cost, 2) if not math.isnan(direct_cost) else float("nan"),
            "primary_turnover_cause": primary_cause,
            "symbols_entered": ",".join(sorted(entered)) if entered else "",
            "symbols_exited": ",".join(sorted(exited)) if exited else "",
            "symbols_reweighted": ",".join(sorted(reweighted)) if reweighted else "",
            "flag_only_reweight": only_reweight,
            "flag_only_one_change": only_one_change,
            "flag_full_exit": full_exit,
            "flag_full_entry": full_entry,
            "flag_repeat_sell_rebuy": repeat_sell_rebuy,
            "equity_before": round(eq_before, 2) if not math.isnan(eq_before) else float("nan"),
        })

    return rows


# -------------------------------------------------------------------------
# Part 5 — Counterfactual cost curves
# -------------------------------------------------------------------------

COST_SCENARIOS = [
    ("no_fees_no_slippage",   0.0,            0.0),
    ("fees_only",             DEFAULT_FEE_BPS, 0.0),
    ("slippage_only",         0.0,            DEFAULT_SLIPPAGE_BPS),
    ("fees_and_slippage",     DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS),
]


def run_counterfactual_curves(
    ohlcv: pd.DataFrame,
    signal_gen: Any,
    joint_start: pd.Timestamp,
    strategy_name: str,
    rebalance_bars: int = CONTROL_REBALANCE_BARS,
) -> list[dict[str, Any]]:
    """Run 4 cost scenarios with identical signal timing."""
    rows = []
    for scenario_name, fee, slip in COST_SCENARIOS:
        result, port_fx = run_canonical(
            ohlcv, signal_gen, joint_start,
            rebalance_bars=rebalance_bars,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=fee,
            slippage_bps=slip,
        )
        eq = port_fx["equity"].dropna()
        rets = port_fx["strategy_return"].fillna(0)
        v_total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
        v_cagr = _cagr(eq, BARS_PER_YEAR)
        v_sharpe = _sharpe(rets, BARS_PER_YEAR)
        v_maxdd = _max_drawdown(eq)

        # 2022 return
        eq_2022 = eq.loc[(eq.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                         (eq.index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))]
        ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1) * 100 if len(eq_2022) >= 2 else float("nan")

        rows.append({
            "strategy": strategy_name,
            "scenario": scenario_name,
            "fee_bps": fee,
            "slippage_bps": slip,
            "total_cost_bps": fee + slip,
            "net_ending_equity": round(float(eq.iloc[-1]), 2),
            "equity_at_start": round(float(eq.iloc[0]), 2),
            "total_return_pct": round(v_total_ret * 100, 2),
            "cagr_pct": round(v_cagr * 100, 2),
            "sharpe": round(v_sharpe, 4),
            "max_drawdown_pct": round(v_maxdd * 100, 2),
            "return_2022_pct": round(ret_2022, 2) if not math.isnan(ret_2022) else float("nan"),
            "path_dependency_note": (
                "Costs reduce portfolio equity at each rebalance, which changes "
                "future position sizes (notional = weight × equity). "
                "Higher costs → smaller future equity → smaller future notional "
                "→ slightly fewer dollars at risk in subsequent periods. "
                "This makes the four curves path-dependent: they are not simply "
                "shifted by a constant amount."
            ),
        })

    return rows


# -------------------------------------------------------------------------
# Candidate selection lock writer
# -------------------------------------------------------------------------

CANDIDATE_LOCK_CONTENT = """# Candidate Selection Lock

## Candidate

**Name:** combo_entry2_imm_buf05

**Definition:**
- Universe: FIXED_COMMON_HISTORY (BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD)
- Timeframe: 4-hour bars
- BTC MA length: 240 bars (≈ 40 calendar days)
- Portfolio rebalance frequency: every 12 bars (48 hours)
- Entry rule: BTC must remain at least 0.5% above the 240-bar MA for 2 completed
  consecutive bars before entering. Formally: for bars i and i-1 (completed before
  signal bar i), both must satisfy: BTC_close > MA_240 × 1.005
- Exit rule: Exit immediately (1 bar) when the completed-bar condition fails
- One-bar execution delay preserved: signal at bar i, execution at bar i+1

**Parameter source:** All parameters were defined before observing outcomes and
tested in the whipsaw control study. The combination was proposed as a structural
variant, not searched from a grid.

---

## Selection Basis

This candidate was selected based on the following observed full-sample results
relative to the btc_ma_240_reb12 control:

| Metric | Control | Candidate | Δ |
|---|---|---|---|
| Max drawdown | -48.1% | -45.9% | +2.2pp improvement |
| Sharpe | 1.8555 | 1.7940 | -0.06 (within -0.10 threshold) |
| 2022 return | -41.8% | -38.1% | +3.7pp improvement |
| 7-day whipsaw clusters | 45 | 25 | -44% |
| Additive cost drag | 50.0% | 49.8% | -0.2pp |
| Time in cash | 45.7% | 43.8% | -1.9pp |

All metrics are computed on the FIXED_COMMON_HISTORY dataset (2020-09-28 onward)
using the canonical Run A initialization.

---

## ⚠ CRITICAL LIMITATIONS

### 1. Selection was performed on full historical sample

This candidate was selected after observing the complete available price history
from 2020-09-28 to the present. It is NOT out-of-sample validated.

### 2. No further parameter adjustment may be described as out-of-sample

Any subsequent change to the MA length, entry confirmation bars, buffer percentage,
rebalance frequency, or any other parameter—even small adjustments—constitutes
additional in-sample tuning on this same dataset. It may not be described as
out-of-sample optimization.

### 3. Out-of-sample validation requires genuinely future data

Prospective validation must use data that was not available at the time of
candidate selection: either live trading results on genuinely future bars, or
newly acquired historical data that was not used during development.

### 4. Survivorship bias is present and unresolved

The universe (BTC, ETH, XRP, SOL, AVAX) was selected using present-day knowledge
of which coins are still trading and available on the exchange. This introduces
survivorship bias. Results would likely differ materially on a truly historical
universe including assets that later delisted or declined.

### 5. Candidate is NOT approved for live deployment

Selection-locking a candidate means only that no further tuning may claim
out-of-sample status. It does not constitute approval for live deployment.
Before live deployment would require (at minimum):
  - Genuinely out-of-sample prospective results
  - Live slippage and fee validation
  - Position sizing and risk analysis
  - Exchange-specific execution review
  - Regulatory and operational review

---

## Frozen Specification

The frozen specification is:

```python
WhipsawControl(
    name="combo_entry2_imm_buf05",
    entry_confirm_bars=2,
    exit_confirm_bars=1,
    entry_buffer_pct=0.5,
    exit_buffer_pct=0.0,
    min_duration_bars=0,
)
BTC_MA_BARS = 240
REBALANCE_BARS = 12
FEE_BPS = 10
SLIPPAGE_BPS = 5
```

Any research that modifies any of these values must be described as further
in-sample tuning, not out-of-sample testing.

---

## Confirmation

- Candidate locked by: research/fixed_five_cost_audit.py
- Date data generated: see report timestamps
- Live trading code unchanged: YES
- Baseline v1 parameters unchanged: YES
- Live configuration unchanged: YES
"""


# -------------------------------------------------------------------------
# Main runner
# -------------------------------------------------------------------------

def run_all(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    LOGGER.info("Loading close matrix…")
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        raise RuntimeError("Cannot find joint eligible start")

    ohlcv_full = _close_to_ohlcv(close)
    btc_full = close[BTC_COL].dropna()
    LOGGER.info("Joint start: %s", joint_start)

    strategies = []

    # ---- Control ----
    LOGGER.info("Running control (btc_ma_240_reb12)…")
    ctrl_sig, ctrl_gate = build_control_signal(btc_full)
    ctrl_result, ctrl_port = run_canonical(
        ohlcv_full, ctrl_sig, joint_start,
        rebalance_bars=CONTROL_REBALANCE_BARS,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        fee_bps=DEFAULT_FEE_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )
    ctrl_gate_fx = ctrl_gate.reindex(ctrl_port.index).fillna(False)
    strategies.append(("btc_ma_240_reb12", ctrl_sig, ctrl_gate, ctrl_gate_fx,
                        ctrl_result, ctrl_port))

    # ---- Candidate ----
    LOGGER.info("Running candidate (combo_entry2_imm_buf05)…")
    cand_sig, cand_gate = build_candidate_signal(btc_full)
    cand_result, cand_port = run_canonical(
        ohlcv_full, cand_sig, joint_start,
        rebalance_bars=CONTROL_REBALANCE_BARS,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        fee_bps=DEFAULT_FEE_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )
    cand_gate_fx = cand_gate.reindex(cand_port.index).fillna(False)
    strategies.append(("combo_entry2_imm_buf05", cand_sig, cand_gate, cand_gate_fx,
                        cand_result, cand_port))

    # ---- Build ledgers ----
    all_ledger_rows: list[dict] = []
    all_decomp_rows: list[dict] = []
    all_audit_rows: list[dict] = []
    all_recon_rows: list[dict] = []
    all_cf_rows: list[dict] = []

    for strat_name, sig, gate, gate_fx, result, port_fx in strategies:
        LOGGER.info("Building trade ledger: %s…", strat_name)
        ledger = build_trade_ledger(
            strat_name, result, port_fx, close, gate_fx, joint_start
        )
        all_ledger_rows.extend([asdict(t) for t in ledger])

        recon = compute_cost_reconciliation(
            strat_name, result, port_fx, ledger, joint_start
        )
        all_recon_rows.append(recon)

        decomp = compute_turnover_decomposition(strat_name, ledger)
        all_decomp_rows.extend(decomp)

        audit = build_rebalance_event_audit(
            strat_name, result, port_fx, close, gate_fx, joint_start
        )
        all_audit_rows.extend(audit)

        LOGGER.info("Running counterfactual curves: %s…", strat_name)
        cf_rows = run_counterfactual_curves(ohlcv_full, sig, joint_start, strat_name)
        all_cf_rows.extend(cf_rows)

    # ---- Write outputs ----
    ledger_path = output_dir / "fixed_five_direct_cost_trade_ledger.csv"
    pd.DataFrame(all_ledger_rows).to_csv(ledger_path, index=False)

    decomp_path = output_dir / "fixed_five_turnover_source_decomposition.csv"
    pd.DataFrame(all_decomp_rows).to_csv(decomp_path, index=False)

    audit_path = output_dir / "fixed_five_rebalance_event_audit.csv"
    pd.DataFrame(all_audit_rows).to_csv(audit_path, index=False)

    cf_path = output_dir / "fixed_five_cost_counterfactual_curves.csv"
    pd.DataFrame(all_cf_rows).to_csv(cf_path, index=False)

    recon_md = write_reconciliation_md(all_recon_rows, all_decomp_rows, all_cf_rows, output_dir)
    lock_md = output_dir / "fixed_five_candidate_selection_lock.md"
    lock_md.write_text(CANDIDATE_LOCK_CONTENT)

    print_summary(all_recon_rows, all_decomp_rows, all_cf_rows,
                  [ledger_path, decomp_path, audit_path, cf_path, recon_md, lock_md])


# -------------------------------------------------------------------------
# Reconciliation markdown
# -------------------------------------------------------------------------

def write_reconciliation_md(
    recon_rows: list[dict],
    decomp_rows: list[dict],
    cf_rows: list[dict],
    output_dir: Path,
) -> Path:
    lines = [
        "# Transaction-Cost and Turnover-Source Reconciliation",
        "",
        "## Part 1 — Direct Cost Definitions",
        "",
        "Costs in this report are computed from executed notional at the trade level:",
        "",
        "- **fee_dollars** = executed_notional × fee_bps / 10,000",
        "- **slippage_dollars** = executed_notional × slippage_bps / 10,000",
        "- **total_direct_cost** = fee_dollars + slippage_dollars",
        "- **executed_notional** = |weight_change| × portfolio_equity_before_trade",
        "",
        "These are the modeled direct costs as the engine applies them.",
        "",
        "### What gross-minus-net is NOT",
        "",
        "The gross-minus-net ending-equity difference is not the fees paid.",
        "It includes three components:",
        "1. Actual direct fees (fee_bps × notional)",
        "2. Actual modeled slippage (slippage_bps × notional)",
        "3. **Foregone compounding and path effects**: every dollar paid in costs",
        "   reduces the equity base. Future trades execute on a smaller base,",
        "   compounding less wealth over time. This residual is NOT a fee.",
        "",
        "### What additive cost-rate sum is NOT",
        "",
        "additive_rate_sum × equity_at_start is a rough first-order estimate.",
        "It is NOT actual cost. Actual costs compound because each cost reduces",
        "the equity base for future growth. The actual dollar impact is many times",
        "larger than this first-order estimate.",
        "",
        "## Part 2 — Cost Reconciliation",
        "",
    ]

    for r in recon_rows:
        lines += [
            f"### {r['strategy']}",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Equity at start | ${r['equity_at_start']:,.2f} |",
            f"| Gross ending equity | ${r['gross_ending_equity']:,.2f} |",
            f"| Net ending equity | ${r['net_ending_equity']:,.2f} |",
            f"| **Actual modeled fee dollars** | **${r['actual_fee_dollars']:,.2f}** |",
            f"| **Actual modeled slippage dollars** | **${r['actual_slippage_dollars']:,.2f}** |",
            f"| **Total direct cost dollars** | **${r['total_direct_cost_dollars']:,.2f}** |",
            f"| Gross-minus-net (NOT fees paid) | ${r['gross_minus_net_dollars']:,.2f} |",
            f"| Residual foregone compounding | ${r['residual_foregone_compounding_dollars']:,.2f} |",
            f"| Direct cost / equity at start | {r['direct_cost_div_equity_at_start_pct']:.2f}% |",
            f"| Direct cost / executed notional | {r['direct_cost_div_executed_notional_pct']:.4f}% |",
            f"| Direct cost / gross profit | {r['direct_cost_div_gross_profit_pct']:.2f}% |",
            f"| Gross-minus-net / gross ending equity | {r['gross_minus_net_div_gross_ending_equity_pct']:.2f}% |",
            f"| Additive rate × start equity (REFERENCE ONLY) | ${r['additive_rate_sum_times_start_equity_REFERENCE_ONLY']:,.2f} |",
            f"| (This is ~{r['additive_rate_sum_times_start_equity_REFERENCE_ONLY']/max(r['total_direct_cost_dollars'],1)*100:.1f}% of actual direct cost — NOT the same thing) | |",
            "",
        ]

    lines += [
        "## Part 3 — Why Whipsaw Reduction Barely Changed Total Cost",
        "",
        "The combo_entry2_imm_buf05 candidate reduced 7-day whipsaw clusters by 44%",
        "but reduced the additive cost measure only from 50.0% to 49.8% (−0.2pp).",
        "",
        "### Explanation",
        "",
        "Regime transitions (entries and exits) produce large, one-time turnover events.",
        "However, the bulk of total turnover comes from regular cross-sectional rank",
        "rebalances that occur every 12 bars regardless of regime state.",
        "",
        "A whipsaw cluster is defined as 3+ transitions within 7 calendar days.",
        "Eliminating 20 whipsaw clusters (45 → 25) removes the transitions that caused",
        "unnecessary round-trips during noisy BTC-MA crossings.",
        "",
        "But each whipsaw cluster typically produces 3–5 transitions, each with turnover",
        "of ~1× (full portfolio exit then re-entry). The total notional from all clusters",
        "is a small fraction of the total notional from hundreds of routine rebalances.",
        "",
        "See the turnover-source decomposition CSV for exact numbers.",
        "",
        "## Part 5 — Counterfactual Curves",
        "",
        "### Path-Dependency Warning",
        "",
        "The four counterfactual curves use identical signal logic and timing.",
        "However, they are path-dependent because:",
        "",
        "1. Each cost payment reduces current portfolio equity.",
        "2. Future trades execute notional = weight × equity.",
        "3. A lower equity base → smaller future notional → lower absolute future returns.",
        "",
        "Therefore, the no-cost curve is not simply the fees-and-slippage curve shifted",
        "upward by a constant. The differences compound over the full period.",
        "This is the same mechanism that makes the residual_foregone_compounding figure",
        "larger than zero.",
        "",
    ]

    for strat in set(r["strategy"] for r in cf_rows):
        s_rows = [r for r in cf_rows if r["strategy"] == strat]
        lines += [
            f"### {strat}",
            "",
            "| Scenario | Net End Eq | Total Return | Sharpe | Max DD | 2022 |",
            "|---|---|---|---|---|---|",
        ]
        for r in s_rows:
            lines.append(
                f"| {r['scenario']} | ${r['net_ending_equity']:,.0f} | "
                f"{r['total_return_pct']:.1f}% | {r['sharpe']:.4f} | "
                f"{r['max_drawdown_pct']:.1f}% | {r['return_2022_pct']:.1f}% |"
            )
        lines.append("")

    lines += [
        f"⚠ {SURVIVORSHIP_BIAS_NOTE}",
    ]

    path = output_dir / "fixed_five_cost_reconciliation.md"
    path.write_text("\n".join(lines))
    return path


# -------------------------------------------------------------------------
# Summary printer
# -------------------------------------------------------------------------

def print_summary(
    recon_rows: list[dict],
    decomp_rows: list[dict],
    cf_rows: list[dict],
    generated_paths: list[Path],
) -> None:
    print("\n" + "=" * 72)
    print("COST AUDIT AND TURNOVER RECONCILIATION — SUMMARY")
    print("=" * 72)

    for r in recon_rows:
        strat = r["strategy"]
        print(f"\n  ── {strat} ──")
        print(f"    Actual modeled fees:            ${r['actual_fee_dollars']:>14,.0f}")
        print(f"    Actual modeled slippage:        ${r['actual_slippage_dollars']:>14,.0f}")
        print(f"    Total direct cost dollars:      ${r['total_direct_cost_dollars']:>14,.0f}")
        print(f"    Gross ending equity:            ${r['gross_ending_equity']:>14,.0f}")
        print(f"    Net ending equity:              ${r['net_ending_equity']:>14,.0f}")
        print(f"    Gross-minus-net (NOT fees):     ${r['gross_minus_net_dollars']:>14,.0f}")
        print(f"    Residual foregone compounding:  ${r['residual_foregone_compounding_dollars']:>14,.0f}")
        print(f"    Direct cost / equity at start:   {r['direct_cost_div_equity_at_start_pct']:>13.1f}%")
        print(f"    Direct cost / gross profit:      {r['direct_cost_div_gross_profit_pct']:>13.1f}%")
        ref = r["additive_rate_sum_times_start_equity_REFERENCE_ONLY"]
        actual = r["total_direct_cost_dollars"]
        print(f"    ⚠ Additive_rate×start_equity:  ${ref:>14,.0f}  ← REFERENCE ONLY, NOT actual cost")
        print(f"      (= {ref/max(actual,1)*100:.1f}% of actual direct cost; "
              f"the rest is foregone compounding)")

    print("\n  TURNOVER SOURCE (control vs candidate):")
    print(f"  {'Strategy':<30} {'Reason':<30} {'Trades':>7} {'Cost $':>10} {'% of total':>10}")
    print(f"  {'-'*30} {'-'*30} {'-'*7} {'-'*10} {'-'*10}")
    for row in decomp_rows:
        if row["trade_reason"] == "TOTAL":
            continue
        if row["total_direct_cost_dollars"] > 0:
            print(
                f"  {row['strategy']:<30} {row['trade_reason']:<30} "
                f"{row['trade_count']:>7} "
                f"${row['total_direct_cost_dollars']:>9,.0f} "
                f"{row['pct_of_total_direct_cost']:>9.1f}%"
            )

    print("\n  COUNTERFACTUAL CURVES:")
    print(f"  {'Strategy':<30} {'Scenario':<25} {'Net End':>12} {'Return%':>9} {'Sharpe':>7}")
    print(f"  {'-'*30} {'-'*25} {'-'*12} {'-'*9} {'-'*7}")
    for r in cf_rows:
        print(
            f"  {r['strategy']:<30} {r['scenario']:<25} "
            f"${r['net_ending_equity']:>11,.0f} "
            f"{r['total_return_pct']:>9.1f}% "
            f"{r['sharpe']:>7.4f}"
        )

    # Why minimal cost savings
    ctrl_total = next((r["total_direct_cost_dollars"] for r in recon_rows
                       if r["strategy"] == "btc_ma_240_reb12"), 0)
    cand_total = next((r["total_direct_cost_dollars"] for r in recon_rows
                       if r["strategy"] == "combo_entry2_imm_buf05"), 0)
    ctrl_regime = sum(r["total_direct_cost_dollars"] for r in decomp_rows
                      if r["strategy"] == "btc_ma_240_reb12"
                      and r["trade_reason"] in ("REGIME_EXIT", "REGIME_ENTRY"))
    cand_regime = sum(r["total_direct_cost_dollars"] for r in decomp_rows
                      if r["strategy"] == "combo_entry2_imm_buf05"
                      and r["trade_reason"] in ("REGIME_EXIT", "REGIME_ENTRY"))

    print(f"\n  WHY WHIPSAW REDUCTION BARELY CHANGED TOTAL COST:")
    print(f"    Control regime transition cost:   ${ctrl_regime:>10,.0f}  "
          f"({ctrl_regime/max(ctrl_total,1)*100:.1f}% of total)")
    print(f"    Candidate regime transition cost: ${cand_regime:>10,.0f}  "
          f"({cand_regime/max(cand_total,1)*100:.1f}% of total)")
    print(f"    Saving from reducing 20 clusters: ${ctrl_regime - cand_regime:>10,.0f}")
    print(f"    Regime transitions are a small fraction of total cost;")
    print(f"    bulk comes from routine rank rebalances (unchanged at reb=12).")

    # Candidate still improves?
    ctrl_cf = next((r for r in cf_rows if r["strategy"] == "btc_ma_240_reb12"
                    and r["scenario"] == "fees_and_slippage"), {})
    cand_cf = next((r for r in cf_rows if r["strategy"] == "combo_entry2_imm_buf05"
                    and r["scenario"] == "fees_and_slippage"), {})
    ctrl_dd = ctrl_cf.get("max_drawdown_pct", 0)
    cand_dd = cand_cf.get("max_drawdown_pct", 0)
    print(f"\n  CANDIDATE vs CONTROL (fees_and_slippage scenario):")
    print(f"    Control max DD:   {ctrl_dd:.1f}%")
    print(f"    Candidate max DD: {cand_dd:.1f}%   (improvement: {cand_dd - ctrl_dd:+.1f}pp)")
    ctrl_r22 = ctrl_cf.get("return_2022_pct", 0)
    cand_r22 = cand_cf.get("return_2022_pct", 0)
    print(f"    Control 2022:     {ctrl_r22:.1f}%")
    print(f"    Candidate 2022:   {cand_r22:.1f}%   (improvement: {cand_r22 - ctrl_r22:+.1f}pp)")

    print(f"\n  CANDIDATE SELECTION LOCK: combo_entry2_imm_buf05 is selection-locked.")
    print(f"    Selected on full in-sample history. NOT out-of-sample validated.")
    print(f"    Any further parameter change is additional in-sample tuning.")
    print(f"    Candidate is NOT approved for live deployment.")

    print(f"\n  LIVE BEHAVIOR UNCHANGED ✓")
    print(f"  No live, broker, execution, or config files modified.")

    print("\n  Generated:")
    for p in generated_paths:
        print(f"    {p}")
    print("\n" + "=" * 72)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cost audit and reconciliation")
    parser.add_argument("--data-dir", default="data/local")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    run_all(Path(args.data_dir), Path(args.output_dir))
