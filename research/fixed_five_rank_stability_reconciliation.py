"""Reconciliation script for the rank-stability study.

Addresses three confirmed bugs:
  Bug 1: min_hold timing — LABEL ERROR, implementation is correct
  Bug 2: Cost accounting — gross_minus_net × fraction is NOT actual fees
  Bug 3: Opportunity cost — pct_helped ignores cost-avoided component

Produces 5 output files:
  reports/fixed_five_min_hold_timing_audit.csv
  reports/fixed_five_rank_stability_direct_costs.csv
  reports/fixed_five_suppression_opportunity_cost.csv
  reports/fixed_five_rank_stability_corrected_comparison.csv
  reports/fixed_five_rank_stability_reconciliation.md

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_rank_stability_reconciliation
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from research.universe_integrity_analysis import (
    LIVE_FIVE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
    _close_to_ohlcv,
    build_close_matrix,
    find_joint_eligible_start,
)
from research.fixed_five_defensive_overlay import (
    _cagr,
    _max_drawdown,
    _sharpe,
    _sortino,
    _worst_calendar_year,
    _worst_month_return,
    _period_return,
)
from research.fixed_five_canonical_regime_comparison import run_canonical, PERIODS
from research.fixed_five_rank_stability import (
    RankStabilityParams,
    RankStabilitySignalGen,
    VARIANTS,
    compute_holding_episodes,
    holding_duration_summary,
    build_stability_signal,
    build_frozen_candidate_gate,
    CONTROL_MA_BARS,
    CONTROL_REBALANCE_BARS,
    TOP_N,
    FEE_BPS,
    SLIPPAGE_BPS,
    MIN_HISTORY_BARS,
)
from strategies.cross_sectional_momentum import compute_momentum_score

LOGGER = logging.getLogger(__name__)

BARS_PER_YEAR = 2190  # 4h bars per year
SHORT_LOOKBACK = 12
MEDIUM_LOOKBACK = 36
INITIAL_CAPITAL = 10_000.0
BTC_COL = "BTC/USD"

SELECTED_VARIANT_NAMES = {
    "control",
    "rank_buffer_4",
    "challenger_confirm_3",
    "score_hurdle_025",
    "score_hurdle_100",
    "min_hold_4",
    "min_hold_6",
    "combo_buf4_hurdle025",
    "combo_buf4_conf2_hold2",
}

SELECTED_VARIANTS: list[RankStabilityParams] = [
    v for v in VARIANTS if v.name in SELECTED_VARIANT_NAMES
]


# ---------------------------------------------------------------------------
# Part 1: Timing audit
# ---------------------------------------------------------------------------

def compute_min_hold_timing_audit() -> pd.DataFrame:
    """Document min_hold_rebs timing semantics for all min_hold variants.

    Each unit of min_hold_rebs is one scheduled portfolio rebalance.
    Rebalances occur every CONTROL_REBALANCE_BARS × 4 hours = 48 hours = 2 calendar days.
    """
    rebalance_interval_bars = CONTROL_REBALANCE_BARS  # 12
    rebalance_interval_hours = rebalance_interval_bars * 4  # 48 hours

    example_entry_ts = pd.Timestamp("2022-01-10 00:00:00", tz="UTC")

    rows = []
    for min_hold_rebs in [0, 1, 2, 3, 4, 5, 6]:
        min_hold_hours = min_hold_rebs * rebalance_interval_hours
        min_hold_days = min_hold_hours / 24
        earliest_replacement = example_entry_ts + pd.Timedelta(hours=min_hold_hours)

        if min_hold_rebs == 6:
            prior_label_error = "YES - prior print said 3 days for min_hold_6; correct is 12 days"
        elif min_hold_rebs in [2, 4]:
            prior_label_error = "YES - prior print was wrong"
        else:
            prior_label_error = "N/A"

        rows.append({
            "min_hold_rebs": min_hold_rebs,
            "rebalance_interval_bars": rebalance_interval_bars,
            "rebalance_interval_hours": rebalance_interval_hours,
            "min_hold_hours": min_hold_hours,
            "min_hold_days": min_hold_days,
            "example_entry_ts": str(example_entry_ts),
            "earliest_replacement_ts": str(earliest_replacement),
            "hold_unit": "scheduled_portfolio_rebalances",
            "implementation_status": "CORRECT" if min_hold_rebs > 0 else "N/A",
            "prior_label_error": prior_label_error,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2: Actual direct cost computation
# ---------------------------------------------------------------------------

def compute_actual_direct_costs(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    fee_bps: float,
    slippage_bps: float,
) -> dict:
    """Compute actual fees and slippage from executed notional per trade.

    fee_dollars = sum_over_trades(|weight_change| × equity_before × fee_bps/10_000)
    slippage_dollars = sum_over_trades(|weight_change| × equity_before × slippage_bps/10_000)

    gross-minus-net is NOT fees. The residual beyond actual costs = foregone compounding.
    """
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000

    holdings = result.holdings_history
    equity = result.portfolio["equity"].ffill()

    reb = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ].copy()
    reb["execution_timestamp"] = pd.to_datetime(reb["execution_timestamp"], utc=True)

    total_fee = 0.0
    total_slip = 0.0
    total_notional = 0.0

    for _, row in reb.iterrows():
        exec_ts = row["execution_timestamp"]
        turnover = float(row["turnover"])
        if turnover < 1e-6:
            continue

        all_ts = holdings.index
        prev_ts_arr = all_ts[all_ts < exec_ts]
        if len(prev_ts_arr) == 0:
            continue
        prev_ts = prev_ts_arr[-1]

        w_before = holdings.loc[prev_ts]
        w_after = holdings.loc[exec_ts]
        eq_before = float(equity.loc[prev_ts]) if prev_ts in equity.index else float("nan")
        if not np.isfinite(eq_before):
            continue

        for sym in holdings.columns:
            wb = float(w_before.get(sym, 0.0))
            wa = float(w_after.get(sym, 0.0))
            dw = abs(wa - wb)
            if dw < 1e-6:
                continue
            notional = dw * eq_before
            total_fee += notional * fee_rate
            total_slip += notional * slippage_rate
            total_notional += notional

    # Gross/net equity
    eq = port_fx["equity"].dropna()
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    eq_start = float(eq.iloc[0])
    gross_eq = eq_start * (1 + gross_r).cumprod()
    gross_end = float(gross_eq.iloc[-1])
    net_end = float(eq.iloc[-1])
    gross_minus_net = gross_end - net_end

    actual_direct_cost = total_fee + total_slip
    foregone_compounding = gross_minus_net - actual_direct_cost

    return {
        "total_executed_notional": round(total_notional, 2),
        "actual_fee_dollars": round(total_fee, 2),
        "actual_slippage_dollars": round(total_slip, 2),
        "actual_direct_cost_dollars": round(actual_direct_cost, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        "gross_minus_net_dollars": round(gross_minus_net, 2),
        "residual_foregone_compounding": round(foregone_compounding, 2),
        "note": "gross_minus_net != fees paid; residual = foregone compounding + path effects",
    }


# ---------------------------------------------------------------------------
# Part 3: Corrected opportunity cost analysis
# ---------------------------------------------------------------------------

def enrich_suppressed_events_corrected(
    suppressed_events: list[dict],
    close: pd.DataFrame,
    rebalance_timestamps: list[pd.Timestamp],
    equity_series: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    top_n: int = 3,
) -> list[dict]:
    """Enrich suppressed events with correct opportunity cost analysis.

    For each suppressed replacement:
    - incumbent_return_next_reb: return of held symbol until next rebalance
    - challenger_return_next_reb: return of suppressed entrant until next rebalance
    - raw_return_diff: incumbent - challenger (positive = suppression helped in raw return)
    - cost_avoided: transaction cost not paid (round trip: entry + exit)
    - net_economic_benefit: raw_return_diff × position_notional + cost_avoided
    - suppression_was_beneficial_after_costs: net_economic_benefit > 0
    """
    enriched = []
    reb_list = sorted(rebalance_timestamps)
    weight = 1.0 / top_n
    total_cost_rate = (fee_bps + slippage_bps) / 10_000

    for ev in suppressed_events:
        ev = dict(ev)
        ts = ev["timestamp"]
        incumbent = ev["incumbent"]
        challenger = ev["challenger"]

        # Next rebalance timestamp
        future_rebs = [t for t in reb_list if t > ts]
        next_reb_ts = future_rebs[0] if future_rebs else None

        def _safe_ret(sym: str, t_start: pd.Timestamp, t_end: pd.Timestamp | None) -> float | None:
            if t_end is None:
                return None
            if t_start not in close.index or t_end not in close.index:
                return None
            if sym not in close.columns:
                return None
            p0 = float(close.loc[t_start, sym])
            p1 = float(close.loc[t_end, sym])
            if p0 <= 0:
                return None
            return round(float(p1 / p0 - 1), 6)

        inc_ret = _safe_ret(incumbent, ts, next_reb_ts)
        chal_ret = _safe_ret(challenger, ts, next_reb_ts)

        ev["incumbent_return_next_reb"] = inc_ret
        ev["challenger_return_next_reb"] = chal_ret

        # Cost avoided: one-leg entry cost (buy challenger) + one-leg exit cost (sell incumbent)
        eq_at_ts = float(equity_series.get(ts, equity_series.iloc[0]))
        notional = weight * eq_at_ts
        # cost_avoided = entry cost of round trip (not paying to swap)
        cost_avoided = notional * total_cost_rate
        ev["cost_avoided"] = round(cost_avoided, 4)

        # Raw return difference
        if inc_ret is not None and chal_ret is not None:
            raw_return_diff = inc_ret - chal_ret
            ev["raw_return_diff"] = round(raw_return_diff, 6)
            # Net economic benefit in dollars
            net_economic_benefit = raw_return_diff * notional + cost_avoided
            ev["net_economic_benefit"] = round(net_economic_benefit, 4)
            ev["suppression_was_beneficial_after_costs"] = bool(net_economic_benefit > 0)
            ev["raw_return_diff_pct_of_position"] = round(raw_return_diff * 100, 4)
            ev["net_benefit_pct_of_position"] = round(
                net_economic_benefit / max(notional, 1) * 100, 4
            )
        else:
            ev["raw_return_diff"] = None
            ev["net_economic_benefit"] = None
            ev["suppression_was_beneficial_after_costs"] = None
            ev["raw_return_diff_pct_of_position"] = None
            ev["net_benefit_pct_of_position"] = None

        enriched.append(ev)

    return enriched


def suppressed_events_corrected_summary(enriched: list[dict]) -> dict:
    """Summarise corrected suppressed events."""
    n = len(enriched)
    if n == 0:
        return {
            "total_suppressions": 0,
            "pct_helped_before_costs": float("nan"),
            "pct_beneficial_after_costs": float("nan"),
            "mean_raw_return_diff": float("nan"),
            "median_raw_return_diff": float("nan"),
            "mean_net_benefit_pct_of_position": float("nan"),
            "median_net_benefit_pct_of_position": float("nan"),
            "total_cost_avoided": 0.0,
            "total_return_gain_from_beneficial": 0.0,
            "total_return_loss_from_harmful": 0.0,
        }

    raw_diffs = [e["raw_return_diff"] for e in enriched if e.get("raw_return_diff") is not None]
    net_benefits = [e["net_economic_benefit"] for e in enriched
                    if e.get("net_economic_benefit") is not None]
    net_benefit_pcts = [e["net_benefit_pct_of_position"] for e in enriched
                        if e.get("net_benefit_pct_of_position") is not None]
    cost_vals = [e["cost_avoided"] for e in enriched if e.get("cost_avoided") is not None]
    beneficial = [e for e in enriched
                  if e.get("suppression_was_beneficial_after_costs") is True]
    harmful = [e for e in enriched
               if e.get("suppression_was_beneficial_after_costs") is False]
    helped_before = [e for e in enriched
                     if e.get("raw_return_diff") is not None and e["raw_return_diff"] > 0]

    n_with_data = len(raw_diffs)
    pct_helped = round(len(helped_before) / n_with_data * 100, 2) if n_with_data > 0 else float("nan")
    pct_after = round(len(beneficial) / n_with_data * 100, 2) if n_with_data > 0 else float("nan")

    gain_from_beneficial = sum(
        e["net_economic_benefit"] for e in beneficial if e.get("net_economic_benefit") is not None
    )
    loss_from_harmful = sum(
        e["net_economic_benefit"] for e in harmful if e.get("net_economic_benefit") is not None
    )

    return {
        "total_suppressions": n,
        "pct_helped_before_costs": pct_helped,
        "pct_beneficial_after_costs": pct_after,
        "mean_raw_return_diff": round(float(np.mean(raw_diffs)) * 100, 4) if raw_diffs else float("nan"),
        "median_raw_return_diff": round(float(np.median(raw_diffs)) * 100, 4) if raw_diffs else float("nan"),
        "mean_net_benefit_pct_of_position": round(float(np.mean(net_benefit_pcts)), 4) if net_benefit_pcts else float("nan"),
        "median_net_benefit_pct_of_position": round(float(np.median(net_benefit_pcts)), 4) if net_benefit_pcts else float("nan"),
        "total_cost_avoided": round(sum(cost_vals), 2) if cost_vals else 0.0,
        "total_return_gain_from_beneficial": round(gain_from_beneficial, 2),
        "total_return_loss_from_harmful": round(loss_from_harmful, 2),
    }


# ---------------------------------------------------------------------------
# Part 4: Corrected RANK_STABILITY_IMPROVEMENT classification
# ---------------------------------------------------------------------------

def classify_rank_stability_improvement_corrected(
    m: dict,
    control: dict,
) -> str:
    """Returns RANK_STABILITY_IMPROVEMENT | MIXED_TRADEOFF | WEAK_EVIDENCE | UNFAVORABLE.

    Uses actual direct costs (not additive_cost_pct) for cost criterion.
    """
    ctrl_cost = float(control.get("actual_direct_cost_dollars", 0) or 0)
    v_cost = float(m.get("actual_direct_cost_dollars", ctrl_cost) or ctrl_cost)
    cost_reduction = (ctrl_cost - v_cost) / max(ctrl_cost, 1) if ctrl_cost > 0 else 0

    ctrl_n_rep = float(control.get("n_rank_replacements", 1) or 1)
    v_n_rep = float(m.get("n_rank_replacements", ctrl_n_rep) or ctrl_n_rep)
    rep_reduction = (ctrl_n_rep - v_n_rep) / ctrl_n_rep if ctrl_n_rep > 0 else 0

    # max_drawdown_pct is negative. dd_worsened < 0 means variant is worse.
    ctrl_dd = float(control.get("max_drawdown_pct", -100) or -100)
    v_dd = float(m.get("max_drawdown_pct", -100) or -100)
    dd_worsened = v_dd - ctrl_dd  # negative = variant is worse

    ctrl_sharpe = float(control.get("sharpe", 0) or 0)
    v_sharpe = float(m.get("sharpe", 0) or 0)

    ctrl_2022 = float(control.get("return_2022_pct", 0) or 0)
    v_2022 = float(m.get("return_2022_pct", ctrl_2022) or ctrl_2022)

    ctrl_cash = float(control.get("pct_time_cash", 0) or 0)
    v_cash = float(m.get("pct_time_cash", 0) or 0)

    c1 = cost_reduction >= 0.25
    c2 = rep_reduction >= 0.35
    c4 = v_sharpe >= ctrl_sharpe - 0.10
    c5 = dd_worsened >= -5.0
    c6 = v_2022 >= ctrl_2022 - 3.0
    c7 = v_cash <= ctrl_cash + 15.0

    if all([c1, c2, c4, c5, c6, c7]):
        return "RANK_STABILITY_IMPROVEMENT"
    elif v_sharpe < ctrl_sharpe - 0.15 or dd_worsened < -8.0:
        return "UNFAVORABLE"
    elif c1 or c2:
        return "MIXED_TRADEOFF"
    else:
        return "WEAK_EVIDENCE"


# ---------------------------------------------------------------------------
# Corrected full-period metrics (adds actual_direct_cost to existing metrics)
# ---------------------------------------------------------------------------

def compute_corrected_metrics(
    variant_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    gate: pd.Series,
    actual_costs: dict,
) -> dict:
    """Compute full-period metrics including corrected actual cost fields."""
    eq = port_fx["equity"].dropna()
    rets = port_fx["strategy_return"].fillna(0)
    end_ts = eq.index[-1]

    v_total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    v_cagr = _cagr(eq, BARS_PER_YEAR)
    v_sharpe = _sharpe(rets, BARS_PER_YEAR)
    v_sortino = _sortino(rets, BARS_PER_YEAR)
    v_maxdd = _max_drawdown(eq)
    calmar = v_cagr / abs(v_maxdd) if v_maxdd < 0 else float("nan")
    worst_yr = _worst_calendar_year(eq)
    worst_mo = _worst_month_return(eq)

    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ]
    years = (end_ts - joint_start).days / 365.25
    ann_turn = float(reb_fx["turnover"].sum()) / max(years, 0.001)
    n_rebalances = len(reb_fx)
    additive_cost_pct = float(reb_fx["cost_rate"].sum() * 100)

    # n_rank_replacements
    gate_on_times = set(gate[gate].index.tolist())
    n_rank_replacements = int(
        reb_fx[
            (reb_fx["turnover"] > 0.01) &
            (reb_fx["execution_timestamp"].isin(gate_on_times) |
             reb_fx["signal_timestamp"].isin(gate_on_times))
        ].shape[0]
    )

    # Time in cash
    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_cash = float((holdings.sum(axis=1) < 0.01).mean() * 100)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)

    # Regime switches
    gate_fx = gate.reindex(port_fx.index).ffill().fillna(False)
    transitions = gate_fx[gate_fx != gate_fx.shift(1)].dropna()
    n_regime_switches = len(transitions)

    # 2022 return
    eq_2022 = eq.loc[(eq.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                     (eq.index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))]
    ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1) * 100 if len(eq_2022) >= 2 else float("nan")

    m: dict[str, Any] = {
        "variant": variant_name,
        "total_return_pct": round(v_total_ret * 100, 2),
        "cagr_pct": round(v_cagr * 100, 2),
        "sharpe": round(v_sharpe, 4),
        "sortino": round(v_sortino, 4) if not math.isnan(v_sortino) else float("nan"),
        "max_drawdown_pct": round(v_maxdd * 100, 2),
        "calmar": round(calmar, 4) if not math.isnan(calmar) else float("nan"),
        "worst_year_pct": round(worst_yr * 100, 2) if not math.isnan(worst_yr) else float("nan"),
        "worst_month_pct": round(worst_mo * 100, 2) if not math.isnan(worst_mo) else float("nan"),
        "return_2022_pct": round(ret_2022, 2) if not math.isnan(ret_2022) else float("nan"),
        "avg_annual_turnover": round(ann_turn, 4),
        "n_rebalances": n_rebalances,
        "n_rank_replacements": n_rank_replacements,
        "additive_cost_pct": round(additive_cost_pct, 4),
        "pct_time_invested": round(pct_invested, 2),
        "pct_time_cash": round(pct_cash, 2),
        "n_regime_switches": n_regime_switches,
        # Corrected actual costs
        "actual_fee_dollars": actual_costs["actual_fee_dollars"],
        "actual_slippage_dollars": actual_costs["actual_slippage_dollars"],
        "actual_direct_cost_dollars": actual_costs["actual_direct_cost_dollars"],
        "gross_minus_net_dollars": actual_costs["gross_minus_net_dollars"],
        "residual_foregone_compounding": actual_costs["residual_foregone_compounding"],
    }
    return m


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def write_reconciliation_md(
    timing_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    supp_summary: dict,
    control_actual_costs: dict,
    output_dir: Path,
) -> Path:
    """Write the reconciliation markdown report."""
    ctrl = comparison_df[comparison_df["variant"] == "control"]
    ctrl_row = ctrl.iloc[0].to_dict() if not ctrl.empty else {}
    ctrl_fee = control_actual_costs.get("actual_fee_dollars", float("nan"))
    ctrl_slip = control_actual_costs.get("actual_slippage_dollars", float("nan"))
    ctrl_direct = control_actual_costs.get("actual_direct_cost_dollars", float("nan"))
    ctrl_gmn = control_actual_costs.get("gross_minus_net_dollars", float("nan"))
    ctrl_foregone = control_actual_costs.get("residual_foregone_compounding", float("nan"))

    wrong_fee = round(ctrl_gmn * (FEE_BPS / (FEE_BPS + SLIPPAGE_BPS)), 2) if math.isfinite(ctrl_gmn) else float("nan")
    wrong_slip = round(ctrl_gmn * (SLIPPAGE_BPS / (FEE_BPS + SLIPPAGE_BPS)), 2) if math.isfinite(ctrl_gmn) else float("nan")

    candidates = comparison_df[
        comparison_df.get("rank_stability_label_corrected", pd.Series()) == "RANK_STABILITY_IMPROVEMENT"
    ]["variant"].tolist() if "rank_stability_label_corrected" in comparison_df.columns else []

    lines = [
        "# Rank Stability Study — Reconciliation Report",
        "",
        "This document reconciles three confirmed bugs found in the original",
        "`fixed_five_rank_stability.py` output.",
        "",
        "---",
        "",
        "## 1. Min-Hold Timing Reconciliation",
        "",
        "### Finding: LABEL ERROR — Implementation Is Correct",
        "",
        "The implementation correctly counts **scheduled portfolio rebalances**.",
        "Each rebalance interval = 12 bars × 4 hours = **48 hours = 2 calendar days**.",
        "The `hold_age` counter increments once per rebalance call.",
        "",
        "| min_hold_rebs | Rebalance Interval | min_hold_hours | min_hold_days |",
        "|---|---|---|---|",
    ]

    for _, row in timing_df[timing_df["min_hold_rebs"] > 0].iterrows():
        lines.append(
            f"| {int(row['min_hold_rebs'])} | {int(row['rebalance_interval_hours'])}h | "
            f"{int(row['min_hold_hours'])}h | {row['min_hold_days']:.0f} days |"
        )

    lines += [
        "",
        "**min_hold_6 = 6 rebalances × 48 hours = 288 hours = 12 calendar days, NOT 3 days.**",
        "",
        "The prior print summary contained a **label error**. The implementation was correct.",
        "The label was wrong because it printed `min_hold_rebs × 12h` instead of",
        "`min_hold_rebs × 48h`.",
        "",
        "Concrete example: entry at 2022-01-10 00:00 UTC, min_hold_6 allows",
        "earliest replacement at 2022-01-22 00:00 UTC (12 days later).",
        "",
        "---",
        "",
        "## 2. Direct-Cost Accounting Correction",
        "",
        "### The Bug",
        "",
        "The original `rank_stability.py` computed:",
        "",
        "```python",
        "gross_minus_net = gross_end - net_end",
        "fee_dollars = gross_minus_net * (fee_bps / total_bps)   # WRONG",
        "slippage_dollars = gross_minus_net * (slippage_bps / total_bps)  # WRONG",
        "```",
        "",
        "This allocates the gross-minus-net wealth gap by the fee:slippage ratio.",
        f"For the control: ${ctrl_gmn:,.2f} × (10/15) = ${wrong_fee:,.2f} ← reported as 'fees'",
        f"                 ${ctrl_gmn:,.2f} × (5/15)  = ${wrong_slip:,.2f} ← reported as 'slippage'",
        "",
        "### The Fix",
        "",
        "Actual costs must be computed per trade:",
        "",
        "```python",
        "fee_dollars = sum(abs(weight_change_i) × equity_before_i × fee_bps / 10_000)",
        "slippage_dollars = sum(abs(weight_change_i) × equity_before_i × slippage_bps / 10_000)",
        "```",
        "",
        "### Corrected Figures (control — combo_entry2_imm_buf05 strict)",
        "",
        f"| Metric | WRONG (prior) | CORRECT (actual) |",
        f"|---|---|---|",
        f"| Fee dollars | ${wrong_fee:,.2f} | ${ctrl_fee:,.2f} |",
        f"| Slippage dollars | ${wrong_slip:,.2f} | ${ctrl_slip:,.2f} |",
        f"| Total direct cost | ${wrong_fee+wrong_slip:,.2f} | ${ctrl_direct:,.2f} |",
        f"| Gross-minus-net | ${ctrl_gmn:,.2f} | ${ctrl_gmn:,.2f} (same — not a bug) |",
        f"| Residual (foregone compounding) | N/A | ${ctrl_foregone:,.2f} |",
        "",
        "**The $942K/$471K figures were an allocation of gross-minus-net wealth by the",
        "fee:slippage rate ratio. This is NOT actual fees paid. Actual fees are computed",
        f"from executed notional and equal ${ctrl_fee:,.2f}.**",
        "",
        "gross-minus-net ≠ fees paid. The residual beyond actual direct costs is foregone",
        "compounding: wealth lost because each cost payment reduced the future compounding base.",
        "",
        "---",
        "",
        "## 3. Opportunity Cost Interpretation",
        "",
        "### The Incomplete Metric",
        "",
        "The original `pct_helped` (39.75%) means: **39.75% of suppressed replacements had the",
        "incumbent outperform the challenger in raw return over the next rebalance.**",
        "",
        "This is NOT the net economic benefit because it ignores:",
        "1. The transaction cost avoided by suppressing the trade",
        "2. The magnitude of the return difference vs the cost",
        "",
        "### Corrected Analysis",
        "",
        "net_economic_benefit_i = (incumbent_return − challenger_return) × position_notional + cost_avoided",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total suppressions | {supp_summary['total_suppressions']} |",
        f"| pct_helped_before_costs (raw return only) | {supp_summary['pct_helped_before_costs']}% |",
        f"| pct_beneficial_after_costs (net economic benefit > 0) | {supp_summary['pct_beneficial_after_costs']}% |",
        f"| Mean raw return diff (% of position) | {supp_summary['mean_raw_return_diff']}% |",
        f"| Mean net economic benefit (% of position) | {supp_summary['mean_net_benefit_pct_of_position']}% |",
        f"| Total cost avoided | ${supp_summary['total_cost_avoided']:,.2f} |",
        f"| Total return gain from beneficial suppressions | ${supp_summary['total_return_gain_from_beneficial']:,.2f} |",
        f"| Total return loss from harmful suppressions | ${supp_summary['total_return_loss_from_harmful']:,.2f} |",
        "",
        "When the challenger outperforms by less than the transaction costs avoided,",
        "suppression is still net-beneficial.",
        "",
        "---",
        "",
        "## 4. Corrected Comparison Table (9 Variants)",
        "",
    ]

    if not comparison_df.empty:
        display_cols = [
            "variant", "sharpe", "max_drawdown_pct", "return_2022_pct",
            "actual_fee_dollars", "actual_direct_cost_dollars",
            "n_rank_replacements", "rank_stability_label_corrected",
        ]
        available = [c for c in display_cols if c in comparison_df.columns]
        lines.append("| " + " | ".join(available) + " |")
        lines.append("|" + "|".join(["---"] * len(available)) + "|")
        for _, row in comparison_df.iterrows():
            vals = []
            for c in available:
                v = row.get(c, "")
                if isinstance(v, float) and not math.isnan(v):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "---",
        "",
        "## 5. Revised RANK_STABILITY_IMPROVEMENT List",
        "",
        "Criteria (using corrected actual direct costs):",
        "1. Actual direct costs reduced by ≥ 25% vs control",
        "2. Rank replacements reduced by ≥ 35%",
        "3. Sharpe within 0.10 of control OR improved",
        "4. Max drawdown not worsened by > 5pp",
        "5. 2022 return not materially worse (< 3pp worse)",
        "6. NOT achieved by staying predominantly in cash",
        "7. Regime exit never delayed (structural guarantee)",
        "8. One-bar execution delay preserved (structural guarantee)",
        "9. No future information used (structural guarantee)",
        "",
        f"**RANK_STABILITY_IMPROVEMENT variants: {', '.join(candidates) if candidates else 'None found'}**",
        "",
        "---",
        "",
        "## Notes",
        "",
        f"⚠ **{SURVIVORSHIP_BIAS_NOTE}**",
        "",
        "**In-sample disclaimer:** All results are in-sample. combo_entry2_imm_buf05 was",
        "selected after observing the full historical sample and remains the frozen candidate.",
        "No parameter changes in this reconciliation constitute out-of-sample validation.",
        "",
        "**LIVE BEHAVIOR UNCHANGED:** No live trading files, configuration, cron, or exchange",
        "code was modified by this reconciliation.",
        "",
    ]

    path = output_dir / "fixed_five_rank_stability_reconciliation.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --- Part 1: Timing audit ---
    LOGGER.info("Part 1: Timing audit...")
    timing_df = compute_min_hold_timing_audit()
    timing_path = output_dir / "fixed_five_min_hold_timing_audit.csv"
    timing_df.to_csv(timing_path, index=False)
    LOGGER.info("Saved: %s", timing_path)

    # --- Load data ---
    LOGGER.info("Loading close matrix...")
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, MIN_HISTORY_BARS)
    if joint_start is None:
        raise RuntimeError("Cannot find joint eligible start")
    LOGGER.info("Joint start: %s", joint_start)

    ohlcv_full = _close_to_ohlcv(close)
    btc_full = close[BTC_COL].dropna()

    # Build frozen candidate gate
    gate_full = build_frozen_candidate_gate(btc_full)
    gate_fx = gate_full.reindex(
        gate_full.index[gate_full.index >= joint_start]
    ).fillna(False)

    # Pre-compute momentum scores
    LOGGER.info("Pre-computing momentum scores...")
    scores_df = compute_momentum_score(close, SHORT_LOOKBACK, MEDIUM_LOOKBACK)

    LOGGER.info("Running %d selected rank-stability variants...", len(SELECTED_VARIANTS))

    all_metrics: list[dict] = []
    all_cost_rows: list[dict] = []
    all_suppressed: list[dict] = []
    control_metrics: dict | None = None
    control_actual_costs: dict | None = None

    for params in SELECTED_VARIANTS:
        LOGGER.info("  %s...", params.name)
        sig_gen = build_stability_signal(gate_full, params, scores_df)
        result, port_fx = run_canonical(
            ohlcv_full, sig_gen, joint_start,
            rebalance_bars=CONTROL_REBALANCE_BARS,
            initial_capital=INITIAL_CAPITAL,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )

        # --- Part 2: Actual costs ---
        actual_costs = compute_actual_direct_costs(
            result, port_fx, joint_start, FEE_BPS, SLIPPAGE_BPS
        )

        # Combine into cost row
        cost_row = {
            "variant": params.name,
            **actual_costs,
        }
        all_cost_rows.append(cost_row)

        # Full metrics with actual costs
        m = compute_corrected_metrics(
            params.name, result, port_fx, joint_start, gate_fx, actual_costs
        )
        m["description"] = params.description
        m["rank_buffer"] = params.rank_buffer
        m["challenger_confirm_rebs"] = params.challenger_confirm_rebs
        m["score_hurdle"] = params.score_hurdle
        m["min_hold_rebs"] = params.min_hold_rebs
        m["cooldown_rebs"] = params.cooldown_rebs

        if params.name == "control":
            control_metrics = m.copy()
            control_actual_costs = actual_costs.copy()

        all_metrics.append(m)

        # --- Part 3: Corrected suppressed events ---
        eq_series = port_fx["equity"].dropna()
        reb_ts_list = result.rebalance_log["execution_timestamp"].dropna().tolist()
        enriched = enrich_suppressed_events_corrected(
            sig_gen.suppressed_events, close, reb_ts_list, eq_series, FEE_BPS, SLIPPAGE_BPS
        )
        for ev in enriched:
            ev["variant"] = params.name
        all_suppressed.extend(enriched)

    # Add corrected classification labels
    for m in all_metrics:
        if m["variant"] == "control":
            m["rank_stability_label_corrected"] = "CONTROL"
            continue
        if control_metrics is not None:
            m["rank_stability_label_corrected"] = classify_rank_stability_improvement_corrected(
                m, control_metrics
            )
        else:
            m["rank_stability_label_corrected"] = "UNKNOWN"

    # Suppressed events summary
    supp_summary = suppressed_events_corrected_summary(all_suppressed)

    # Write outputs
    costs_df = pd.DataFrame(all_cost_rows)
    costs_path = output_dir / "fixed_five_rank_stability_direct_costs.csv"
    costs_df.to_csv(costs_path, index=False)
    LOGGER.info("Saved: %s", costs_path)

    supp_df = pd.DataFrame(all_suppressed) if all_suppressed else pd.DataFrame(
        columns=["variant", "timestamp", "incumbent", "challenger",
                 "incumbent_return_next_reb", "challenger_return_next_reb",
                 "raw_return_diff", "cost_avoided", "net_economic_benefit",
                 "suppression_was_beneficial_after_costs"]
    )
    supp_path = output_dir / "fixed_five_suppression_opportunity_cost.csv"
    supp_df.to_csv(supp_path, index=False)
    LOGGER.info("Saved: %s", supp_path)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_path = output_dir / "fixed_five_rank_stability_corrected_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    LOGGER.info("Saved: %s", comparison_path)

    md_path = write_reconciliation_md(
        timing_df,
        costs_df,
        comparison_df,
        supp_summary,
        control_actual_costs or {},
        output_dir,
    )
    LOGGER.info("Saved: %s", md_path)

    # --- Final print ---
    ctrl = control_metrics or {}
    ctrl_costs = control_actual_costs or {}
    candidates = [
        m["variant"] for m in all_metrics
        if m.get("rank_stability_label_corrected") == "RANK_STABILITY_IMPROVEMENT"
    ]

    files_created = [
        timing_path, costs_path, supp_path, comparison_path, md_path
    ]

    print("\n=== RANK STABILITY RECONCILIATION COMPLETE ===")
    print("\nFiles created or changed:")
    for f in files_created:
        print(f"  {f}")

    print(f"\nTests: run .venv/bin/pytest tests/test_fixed_five_rank_stability_reconciliation.py -v")

    print("\n=== PART 1: MIN-HOLD TIMING ===")
    print(f"min_hold_2 = 2 rebalances × 48h = 4 days (NOT 2 days)")
    print(f"min_hold_4 = 4 rebalances × 48h = 8 days (NOT 4 days)")
    print(f"min_hold_6 = 6 rebalances × 48h = 12 days (NOT 3 days)")
    print("Implementation was: CORRECT")
    print("Label was: WRONG")

    print("\n=== PART 2: DIRECT COST CORRECTION ===")
    print("Bug: fee_dollars was computed as gross_minus_net × (fee_bps / total_bps)")
    print("Fix: fee_dollars = sum(|Δweight| × equity_before × fee_bps/10_000)")
    print()
    print("Control (combo_entry2_imm_buf05 strict):")
    gmn = ctrl_costs.get("gross_minus_net_dollars", float("nan"))
    fee_d = ctrl_costs.get("actual_fee_dollars", float("nan"))
    slip_d = ctrl_costs.get("actual_slippage_dollars", float("nan"))
    direct_d = ctrl_costs.get("actual_direct_cost_dollars", float("nan"))
    foregone_d = ctrl_costs.get("residual_foregone_compounding", float("nan"))
    notional_d = ctrl_costs.get("total_executed_notional", float("nan"))
    gross_end = ctrl_costs.get("gross_ending_equity", float("nan"))
    net_end_v = ctrl_costs.get("net_ending_equity", float("nan"))
    print(f"  Total executed notional: ${notional_d:,.2f}" if math.isfinite(notional_d) else "  Total executed notional: N/A")
    print(f"  CORRECTED actual fee dollars: ${fee_d:,.2f}" if math.isfinite(fee_d) else "  CORRECTED actual fee dollars: N/A")
    print(f"  CORRECTED actual slippage dollars: ${slip_d:,.2f}" if math.isfinite(slip_d) else "  CORRECTED actual slippage dollars: N/A")
    print(f"  CORRECTED total direct cost: ${direct_d:,.2f}" if math.isfinite(direct_d) else "  CORRECTED total direct cost: N/A")
    print(f"  Gross ending equity: ${gross_end:,.2f}" if math.isfinite(gross_end) else "  Gross ending equity: N/A")
    print(f"  Net ending equity: ${net_end_v:,.2f}" if math.isfinite(net_end_v) else "  Net ending equity: N/A")
    print(f"  Gross-minus-net: ${gmn:,.2f}" if math.isfinite(gmn) else "  Gross-minus-net: N/A")
    print(f"  Residual foregone compounding: ${foregone_d:,.2f}" if math.isfinite(foregone_d) else "  Residual foregone compounding: N/A")
    wrong_fee = round(gmn * (FEE_BPS / (FEE_BPS + SLIPPAGE_BPS)), 0) if math.isfinite(gmn) else float("nan")
    wrong_slip = round(gmn * (SLIPPAGE_BPS / (FEE_BPS + SLIPPAGE_BPS)), 0) if math.isfinite(gmn) else float("nan")
    print(f"  WRONG prior figures: fee=${wrong_fee:,.0f}, slippage=${wrong_slip:,.0f} (were gross_minus_net × 2/3 and 1/3)" if math.isfinite(wrong_fee) else "  WRONG prior figures: N/A")

    print("\n=== PART 3: OPPORTUNITY COST CORRECTION ===")
    print("39.75% 'helped' means: incumbent beat challenger in raw return (BEFORE costs)")
    print("After adjusting for transaction costs avoided:")
    print(f"  pct_beneficial_after_costs: {supp_summary['pct_beneficial_after_costs']}%")
    print(f"  avg_net_benefit per suppression (% of position): {supp_summary['mean_net_benefit_pct_of_position']}%")
    print(f"  total_cost_avoided across all suppressions: ${supp_summary['total_cost_avoided']:,.2f}")
    print(f"  total_return_gain_from_beneficial: ${supp_summary['total_return_gain_from_beneficial']:,.2f}")
    print(f"  total_return_loss_from_harmful: ${supp_summary['total_return_loss_from_harmful']:,.2f}")

    print("\n=== CORRECTED COMPARISON (9 variants) ===")
    if not comparison_df.empty:
        display_cols = ["variant", "actual_fee_dollars", "actual_direct_cost_dollars",
                        "sharpe", "max_drawdown_pct", "return_2022_pct",
                        "rank_stability_label_corrected"]
        avail = [c for c in display_cols if c in comparison_df.columns]
        header = " | ".join(f"{c:>30}" if "variant" not in c and "label" not in c else f"{c:<35}" for c in avail)
        print(header)
        for _, row in comparison_df.iterrows():
            parts = []
            for c in avail:
                v = row.get(c, "")
                if isinstance(v, float) and not math.isnan(v):
                    parts.append(f"{v:>10.2f}")
                else:
                    parts.append(f"{str(v):<35}")
            print(" | ".join(parts))

    print("\n=== MIN_HOLD_4 AND MIN_HOLD_6 STILL QUALIFY? ===")
    for m in all_metrics:
        if m["variant"] in ("min_hold_4", "min_hold_6"):
            label = m.get("rank_stability_label_corrected", "UNKNOWN")
            print(f"  {m['variant']}: {label}")

    print("\n=== IN-SAMPLE NOTE ===")
    print("All results are in-sample. No new parameters tested.")

    print("\n=== LIVE BEHAVIOR UNCHANGED ===")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
