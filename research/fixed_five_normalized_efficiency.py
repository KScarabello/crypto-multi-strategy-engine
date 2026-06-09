"""Normalized cost efficiency analysis for FIXED_FIVE rank-stability variants.

Parts:
  1. Normalized cost efficiency (notional / TWAE, cost / TWAE)
  2. Performance efficiency (Sharpe, Calmar per unit of turnover)
  3. Suppression counterfactual reconciliation
  4. Year-by-year stability
  5. Final in-sample shortlist (≤ 3 variants)
  6. Selection lock document

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_normalized_efficiency
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from research.universe_integrity_analysis import (
    BARS_PER_YEAR_4H,
    DEFAULT_FEE_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_SHORT_LOOKBACK,
    DEFAULT_MEDIUM_LOOKBACK,
    DEFAULT_TOP_N,
    LIVE_FIVE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
    _close_to_ohlcv,
    build_close_matrix,
    find_joint_eligible_start,
)
from research.fixed_five_defensive_overlay import (
    BaseSignalGen,
    _max_drawdown,
    _cagr,
    _sharpe,
    _sortino,
    _worst_calendar_year,
    _worst_month_return,
    _period_return,
)
from research.fixed_five_canonical_regime_comparison import run_canonical, PERIODS
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    compute_controlled_gate_series,
)
from research.fixed_five_rank_stability import (
    RankStabilityParams,
    RankStabilitySignalGen,
    VARIANTS,
    build_stability_signal,
    build_frozen_candidate_gate,
    CONTROL_MA_BARS,
    CONTROL_REBALANCE_BARS,
    TOP_N,
    FEE_BPS,
    SLIPPAGE_BPS,
    INITIAL_CAPITAL,
    MIN_HISTORY_BARS,
    SHORT_LOOKBACK,
    MEDIUM_LOOKBACK,
)
from research.fixed_five_rank_stability_reconciliation import (
    compute_actual_direct_costs,
)
from strategies.cross_sectional_momentum import compute_momentum_score

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_COL = "BTC/USD"
BARS_PER_YEAR = BARS_PER_YEAR_4H  # 2190
REBALANCE_BARS_PER_YEAR = BARS_PER_YEAR / CONTROL_REBALANCE_BARS  # 182.5

SELECTED_NAMES = {
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
    v for v in VARIANTS if v.name in SELECTED_NAMES
]

YEAR_PERIODS = [
    ("2020-2021", "2020-09-28", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", None),
]

SUPPRESSION_NOTE = """
Event-level suppression diagnostic sums ($3.086M avoided costs, $10.974M return gains)
are NOT realizable portfolio savings. These figures are non-additive for three reasons:

1. Overlap: Multiple suppression events may be recorded at the same or adjacent signal
   timestamps. The portfolio cannot simultaneously realize both the cost saving and the
   return difference for overlapping events.

2. Path dependence: The return difference is estimated as a fraction of portfolio equity
   at the signal timestamp. But if the incumbent performs differently from the challenger,
   the portfolio's equity path changes, altering all subsequent position sizes.

3. Cross-variant double-counting: Suppressions from different variants (score_hurdle_025,
   score_hurdle_100, combo_buf4_hurdle025, etc.) may record the same replacement events
   under different names. Summing across variants double-counts the same economic event.

The ONLY reliable measure of economic benefit is:
  TRUE_DELTA_NET_EQUITY = variant_net_ending_equity - control_net_ending_equity

This is computed directly from the backtest equity curves and requires no decomposition.
"""


# ---------------------------------------------------------------------------
# Part 1: Normalized cost efficiency
# ---------------------------------------------------------------------------

def time_weighted_avg_equity(equity: pd.Series, joint_start: pd.Timestamp) -> float:
    """Arithmetic mean of all equity values from joint_start onward."""
    eq = equity.dropna()
    eq = eq.loc[eq.index >= joint_start]
    return float(eq.mean())


def compute_normalized_cost_efficiency(
    variant: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    actual_costs: dict,
) -> dict:
    """Compute normalized cost efficiency metrics for one variant."""
    equity = port_fx["equity"].dropna()
    end_ts = equity.index[-1]

    twae = time_weighted_avg_equity(equity, joint_start)
    starting_equity = float(equity.iloc[0])
    avg_daily_equity = float(equity.resample("1D").mean().mean())

    gross_end = float(actual_costs["gross_ending_equity"])
    net_end = float(actual_costs["net_ending_equity"])
    total_notional = float(actual_costs["total_executed_notional"])
    fee_dollars = float(actual_costs["actual_fee_dollars"])
    slip_dollars = float(actual_costs["actual_slippage_dollars"])
    total_direct_cost = float(actual_costs["actual_direct_cost_dollars"])

    years = (end_ts - joint_start).days / 365.25

    reb = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ]
    ann_turnover = float(reb["turnover"].sum()) / max(years, 0.001)
    n_rank_replacements = int((reb["turnover"] > 0.01).sum())

    notional_div_twae = total_notional / max(twae, 1.0)
    direct_cost_div_twae = total_direct_cost / max(twae, 1.0)

    gross_profit = gross_end - starting_equity
    net_profit = net_end - starting_equity

    direct_cost_div_gross_profit = total_direct_cost / max(gross_profit, 1.0)
    direct_cost_div_gross_equity = total_direct_cost / max(gross_end, 1.0)
    direct_cost_div_net_equity = total_direct_cost / max(net_end, 1.0)

    gross_profit_per_cost_dollar = gross_profit / max(total_direct_cost, 1.0)
    net_profit_per_cost_dollar = net_profit / max(total_direct_cost, 1.0)

    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)
    years_invested = pct_invested * years / 100.0
    replacements_per_invested_year = n_rank_replacements / max(years_invested, 0.001)
    direct_cost_per_replacement = total_direct_cost / max(n_rank_replacements, 1)

    return {
        "variant": variant,
        "starting_equity": round(starting_equity, 2),
        "avg_daily_equity": round(avg_daily_equity, 2),
        "time_weighted_avg_equity": round(twae, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        "total_executed_notional": round(total_notional, 2),
        "actual_fee_dollars": round(fee_dollars, 2),
        "actual_slippage_dollars": round(slip_dollars, 2),
        "total_direct_cost": round(total_direct_cost, 2),
        "years": round(years, 4),
        "annualized_turnover": round(ann_turnover, 4),
        "notional_div_twae": round(notional_div_twae, 6),
        "direct_cost_div_twae": round(direct_cost_div_twae, 6),
        "gross_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "direct_cost_div_gross_profit": round(direct_cost_div_gross_profit, 6),
        "direct_cost_div_gross_equity": round(direct_cost_div_gross_equity, 6),
        "direct_cost_div_net_equity": round(direct_cost_div_net_equity, 6),
        "gross_profit_per_cost_dollar": round(gross_profit_per_cost_dollar, 4),
        "net_profit_per_cost_dollar": round(net_profit_per_cost_dollar, 4),
        "n_rank_replacements": n_rank_replacements,
        "pct_time_invested": round(pct_invested, 2),
        "years_invested": round(years_invested, 4),
        "replacements_per_invested_year": round(replacements_per_invested_year, 4),
        "direct_cost_per_replacement": round(direct_cost_per_replacement, 2),
    }


# ---------------------------------------------------------------------------
# Part 2: Performance efficiency
# ---------------------------------------------------------------------------

def _compute_median_holding_days(holdings_history: pd.DataFrame, joint_start: pd.Timestamp) -> float:
    """Compute median holding episode duration in days."""
    hh = holdings_history.copy()
    hh = hh.loc[hh.index >= joint_start]
    durations_days: list[float] = []

    for sym in hh.columns:
        series = hh[sym]
        in_pos = series > 0.001
        transitions = in_pos.astype(int).diff().fillna(0)
        entries = transitions[transitions == 1].index.tolist()
        exits = transitions[transitions == -1].index.tolist()

        for entry_ts in entries:
            future_exits = [t for t in exits if t > entry_ts]
            exit_ts = future_exits[0] if future_exits else series.index[-1]
            duration_days = (exit_ts - entry_ts).total_seconds() / 86400.0
            durations_days.append(duration_days)

    if not durations_days:
        return float("nan")
    return float(np.median(durations_days))


def compute_performance_efficiency(
    variant: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    actual_costs: dict,
    norm_efficiency: dict,
    control_metrics: dict | None = None,
) -> dict:
    """Compute performance efficiency metrics for one variant."""
    equity = port_fx["equity"].dropna()
    rets = port_fx["strategy_return"].fillna(0)
    end_ts = equity.index[-1]

    total_return_pct = float(equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr_pct = _cagr(equity, BARS_PER_YEAR) * 100
    sharpe = _sharpe(rets, BARS_PER_YEAR)
    sortino = _sortino(rets, BARS_PER_YEAR)
    maxdd_pct = _max_drawdown(equity) * 100
    calmar = (cagr_pct / 100) / abs(maxdd_pct / 100) if maxdd_pct < 0 else float("nan")
    worst_yr_pct = _worst_calendar_year(equity) * 100
    worst_mo_pct = _worst_month_return(equity) * 100

    eq_2022 = equity.loc[
        (equity.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
        (equity.index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))
    ]
    ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1) * 100 if len(eq_2022) >= 2 else float("nan")

    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)
    pct_cash = float((holdings.sum(axis=1) < 0.01).mean() * 100)

    median_holding_days = _compute_median_holding_days(result.holdings_history, joint_start)

    n_rank_replacements = norm_efficiency["n_rank_replacements"]
    annualized_turnover = norm_efficiency["annualized_turnover"]
    net_profit = norm_efficiency["net_profit"]
    total_direct_cost = norm_efficiency["total_direct_cost"]
    net_end = norm_efficiency["net_ending_equity"]

    cagr_per_turnover = (cagr_pct / 100) / max(annualized_turnover, 0.001)
    sharpe_per_turnover = sharpe / max(annualized_turnover, 0.001)
    calmar_per_turnover = calmar / max(annualized_turnover, 0.001) if math.isfinite(calmar) else float("nan")
    net_profit_per_replacement = net_profit / max(n_rank_replacements, 1)

    row: dict[str, Any] = {
        "variant": variant,
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4) if math.isfinite(sortino) else float("nan"),
        "max_drawdown_pct": round(maxdd_pct, 2),
        "calmar": round(calmar, 4) if math.isfinite(calmar) else float("nan"),
        "worst_year_pct": round(worst_yr_pct, 2) if math.isfinite(worst_yr_pct) else float("nan"),
        "worst_month_pct": round(worst_mo_pct, 2) if math.isfinite(worst_mo_pct) else float("nan"),
        "return_2022_pct": round(ret_2022, 2) if math.isfinite(ret_2022) else float("nan"),
        "pct_time_invested": round(pct_invested, 2),
        "pct_time_cash": round(pct_cash, 2),
        "median_holding_days": round(median_holding_days, 2) if math.isfinite(median_holding_days) else float("nan"),
        "n_rank_replacements": n_rank_replacements,
        "annualized_turnover": round(annualized_turnover, 4),
        "cagr_per_turnover": round(cagr_per_turnover, 6),
        "sharpe_per_turnover": round(sharpe_per_turnover, 6),
        "calmar_per_turnover": round(calmar_per_turnover, 6) if math.isfinite(calmar_per_turnover) else float("nan"),
        "net_profit_per_replacement": round(net_profit_per_replacement, 2),
    }

    if control_metrics is not None:
        control_net = control_metrics.get("net_ending_equity", net_end)
        control_cost = control_metrics.get("total_direct_cost", 0.0)
        control_turnover = control_metrics.get("annualized_turnover", annualized_turnover)
        row["net_equity_vs_control_dollars"] = round(net_end - control_net, 2)
        row["direct_cost_vs_control_dollars"] = round(total_direct_cost - control_cost, 2)
        row["turnover_vs_control"] = round(annualized_turnover - control_turnover, 4)
    else:
        row["net_equity_vs_control_dollars"] = float("nan")
        row["direct_cost_vs_control_dollars"] = float("nan")
        row["turnover_vs_control"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Part 3: Suppression counterfactual reconciliation
# ---------------------------------------------------------------------------

def compute_counterfactual_reconciliation(
    variants_results: dict[str, tuple[BacktestResult, pd.DataFrame, dict]],
    control_name: str = "control",
) -> tuple[pd.DataFrame, dict]:
    """Compute true portfolio-level differences between each variant and the control.

    Returns:
    - comparison_df: DataFrame with true_delta_direct_cost, true_delta_net_equity, etc.
    - suppression_note: explanation of why event-level sums are non-additive
    """
    ctrl_result, ctrl_port_fx, ctrl_costs = variants_results[control_name]
    ctrl_net_end = float(ctrl_costs["net_ending_equity"])
    ctrl_gross_end = float(ctrl_costs["gross_ending_equity"])
    ctrl_direct = float(ctrl_costs["actual_direct_cost_dollars"])
    ctrl_start = float(ctrl_port_fx["equity"].dropna().iloc[0])

    rows = []
    for variant, (result, port_fx, actual_costs) in variants_results.items():
        v_net_end = float(actual_costs["net_ending_equity"])
        v_gross_end = float(actual_costs["gross_ending_equity"])
        v_direct = float(actual_costs["actual_direct_cost_dollars"])
        v_start = float(port_fx["equity"].dropna().iloc[0])

        true_delta_direct_cost = ctrl_direct - v_direct
        true_delta_net_equity = v_net_end - ctrl_net_end
        true_delta_gross_equity = v_gross_end - ctrl_gross_end
        true_delta_gross_pnl = (v_gross_end - v_start) - (ctrl_gross_end - ctrl_start)
        true_delta_net_pnl = (v_net_end - v_start) - (ctrl_net_end - ctrl_start)

        rows.append({
            "variant": variant,
            "variant_net_equity": round(v_net_end, 2),
            "control_net_equity": round(ctrl_net_end, 2),
            "variant_direct_cost": round(v_direct, 2),
            "control_direct_cost": round(ctrl_direct, 2),
            "true_delta_direct_cost": round(true_delta_direct_cost, 2),
            "true_delta_net_equity": round(true_delta_net_equity, 2),
            "true_delta_gross_equity": round(true_delta_gross_equity, 2),
            "true_delta_gross_pnl": round(true_delta_gross_pnl, 2),
            "true_delta_net_pnl": round(true_delta_net_pnl, 2),
        })

    comparison_df = pd.DataFrame(rows).set_index("variant")
    suppression_note = {
        "warning": "Event-level sums are non-additive diagnostic figures only.",
        "prior_avoided_costs_diagnostic": "$3,086,242 — cannot be realized as portfolio profit",
        "prior_return_gains_diagnostic": "$10,973,593 — cannot be realized as portfolio profit",
        "reason_1_overlap": (
            "Multiple suppression events may overlap in time; "
            "portfolio cannot simultaneously earn both saving and return diff"
        ),
        "reason_2_path_dependence": (
            "Return diff estimated at signal timestamp; equity path changes "
            "if incumbent and challenger perform differently, altering all subsequent sizes"
        ),
        "reason_3_cross_variant_doublecounting": (
            "Same replacement event may appear in multiple variants under different names"
        ),
        "true_measure": "TRUE_DELTA_NET_EQUITY = variant_net_ending_equity - control_net_ending_equity",
    }
    return comparison_df, suppression_note


# ---------------------------------------------------------------------------
# Part 4: Year-by-year stability
# ---------------------------------------------------------------------------

def compute_period_normalized_metrics(
    variant: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    period_name: str,
    period_start_str: str | None,
    period_end_str: str | None,
    actual_costs: dict,
) -> dict:
    """Compute normalized metrics for one (variant, period)."""
    equity = port_fx["equity"].dropna()
    rets = port_fx["strategy_return"].fillna(0)

    period_start_ts = (
        pd.Timestamp(period_start_str, tz="UTC")
        if period_start_str else equity.index[0]
    )
    period_end_ts = (
        pd.Timestamp(period_end_str, tz="UTC").replace(hour=23, minute=59, second=59)
        if period_end_str else equity.index[-1]
    )
    period_start_ts = max(period_start_ts, equity.index[0])
    period_end_ts = min(period_end_ts, equity.index[-1])

    eq_p = equity.loc[(equity.index >= period_start_ts) & (equity.index <= period_end_ts)]
    rets_p = rets.loc[(rets.index >= period_start_ts) & (rets.index <= period_end_ts)]

    if len(eq_p) < 2:
        period_years = max((period_end_ts - period_start_ts).days / 365.25, 0.001)
        return {
            "variant": variant,
            "period": period_name,
            "period_return_pct": float("nan"),
            "period_max_drawdown_pct": float("nan"),
            "period_sharpe": float("nan"),
            "period_cagr_pct": float("nan"),
            "period_direct_cost_dollars": 0.0,
            "period_annualized_turnover": 0.0,
            "period_direct_cost_div_twae": float("nan"),
            "period_direct_cost_div_gross_profit": float("nan"),
            "period_n_rebalances": 0,
            "period_n_rank_replacements": 0,
        }

    period_return_pct = float(eq_p.iloc[-1] / eq_p.iloc[0] - 1) * 100
    period_maxdd_pct = _max_drawdown(eq_p) * 100
    period_sharpe = _sharpe(rets_p, BARS_PER_YEAR)
    period_years = max((eq_p.index[-1] - eq_p.index[0]).days / 365.25, 0.001)
    period_cagr_pct = _cagr(eq_p, BARS_PER_YEAR) * 100

    reb = result.rebalance_log.copy()
    reb["execution_timestamp"] = pd.to_datetime(reb["execution_timestamp"], utc=True)
    reb_period = reb[
        (reb["execution_timestamp"] >= period_start_ts) &
        (reb["execution_timestamp"] <= period_end_ts)
    ]

    period_n_rebalances = len(reb_period)
    period_n_rank_replacements = int((reb_period["turnover"] > 0.01).sum())
    period_ann_turnover = float(reb_period["turnover"].sum()) / max(period_years, 0.001)

    # Period direct cost: sum cost_rate × equity at execution time
    period_twae = float(eq_p.mean()) if not eq_p.empty else 1.0
    period_direct_cost = 0.0
    cost_rate_per_bps = (FEE_BPS + SLIPPAGE_BPS) / 10_000
    eq_full = port_fx["equity"].dropna()
    for _, row in reb_period.iterrows():
        exec_ts = row["execution_timestamp"]
        turnover = float(row["turnover"])
        if turnover < 1e-6:
            continue
        if exec_ts in eq_full.index:
            eq_at = float(eq_full.loc[exec_ts])
        else:
            eq_at = period_twae
        period_direct_cost += turnover * eq_at * cost_rate_per_bps

    period_direct_cost_div_twae = period_direct_cost / max(period_twae, 1.0)
    period_gross_profit = period_return_pct / 100.0 * float(eq_p.iloc[0])
    period_direct_cost_div_gross_profit = (
        period_direct_cost / max(period_gross_profit, 1.0)
        if period_gross_profit > 0 else float("nan")
    )

    return {
        "variant": variant,
        "period": period_name,
        "period_return_pct": round(period_return_pct, 2),
        "period_max_drawdown_pct": round(period_maxdd_pct, 2),
        "period_sharpe": round(period_sharpe, 4),
        "period_cagr_pct": round(period_cagr_pct, 2),
        "period_direct_cost_dollars": round(period_direct_cost, 2),
        "period_annualized_turnover": round(period_ann_turnover, 4),
        "period_direct_cost_div_twae": round(period_direct_cost_div_twae, 6),
        "period_direct_cost_div_gross_profit": (
            round(period_direct_cost_div_gross_profit, 6)
            if math.isfinite(period_direct_cost_div_gross_profit) else float("nan")
        ),
        "period_n_rebalances": period_n_rebalances,
        "period_n_rank_replacements": period_n_rank_replacements,
    }


# ---------------------------------------------------------------------------
# Part 5: Final in-sample shortlist
# ---------------------------------------------------------------------------

def classify_shortlist(
    m: dict,
    norm: dict,
    counterfactual: dict,
    year_metrics: list[dict],
    control_m: dict,
    control_norm: dict,
) -> str:
    """Returns: SHORTLIST | ALTERNATE | REJECT"""
    variant = m["variant"]
    if variant == "control":
        return "CONTROL"

    ctrl_sharpe = control_m.get("sharpe", 0.0) or 0.0
    ctrl_dd = control_m.get("max_drawdown_pct", -100.0) or -100.0
    ctrl_norm_intensity = control_norm.get("notional_div_twae", 1.0) or 1.0

    v_sharpe = m.get("sharpe", 0.0) or 0.0
    v_dd = m.get("max_drawdown_pct", -100.0) or -100.0
    v_notional_div_twae = norm.get("notional_div_twae", 1.0) or 1.0
    true_delta_direct_cost = counterfactual.get("true_delta_direct_cost", 0.0) or 0.0

    # Hard rejects
    if v_dd < ctrl_dd - 5.0:
        return "REJECT"
    if v_sharpe < ctrl_sharpe - 0.15:
        return "REJECT"
    if true_delta_direct_cost < 0:
        return "REJECT"

    # Year-by-year improvement check
    year_map: dict[str, dict] = {}
    ctrl_year_map: dict[str, dict] = {}
    for ym in year_metrics:
        if ym["variant"] == variant:
            year_map[ym["period"]] = ym
        elif ym["variant"] == "control":
            ctrl_year_map[ym["period"]] = ym

    check_years = ["2022", "2023", "2024", "2025"]
    n_improved = 0
    for yr in check_years:
        v_yr = year_map.get(yr, {})
        c_yr = ctrl_year_map.get(yr, {})
        v_ret = v_yr.get("period_return_pct", float("nan"))
        c_ret = c_yr.get("period_return_pct", float("nan"))
        v_calmar = v_yr.get("period_sharpe", float("nan"))
        c_calmar = c_yr.get("period_sharpe", float("nan"))
        if not math.isnan(v_ret) and not math.isnan(c_ret) and v_ret > c_ret:
            n_improved += 1
        elif not math.isnan(v_calmar) and not math.isnan(c_calmar) and v_calmar > c_calmar:
            n_improved += 1

    if n_improved <= 1:
        return "REJECT"

    # All criteria for SHORTLIST
    intensity_ok = (v_notional_div_twae < ctrl_norm_intensity) or (v_sharpe >= ctrl_sharpe)
    sharpe_ok = v_sharpe >= ctrl_sharpe - 0.10
    year_ok = n_improved >= 2
    cost_ok = true_delta_direct_cost > 0
    dd_ok = v_dd >= ctrl_dd - 5.0

    # Operational simplicity: only simple rule combinations
    simple = True  # All 9 selected variants are operationally simple

    if intensity_ok and sharpe_ok and year_ok and cost_ok and dd_ok and simple:
        return "SHORTLIST"
    elif sharpe_ok and cost_ok and dd_ok:
        return "ALTERNATE"
    else:
        return "REJECT"


def build_final_shortlist(
    all_perf: list[dict],
    all_norm: list[dict],
    counterfactual_df: pd.DataFrame,
    all_year_metrics: list[dict],
) -> list[dict]:
    """Build and rank the final shortlist (≤ 3 variants)."""
    norm_map = {r["variant"]: r for r in all_norm}
    perf_map = {r["variant"]: r for r in all_perf}
    control_m = perf_map.get("control", {})
    control_norm = norm_map.get("control", {})

    results = []
    for m in all_perf:
        v = m["variant"]
        norm = norm_map.get(v, {})
        cf = counterfactual_df.loc[v].to_dict() if v in counterfactual_df.index else {}
        label = classify_shortlist(m, norm, cf, all_year_metrics, control_m, control_norm)
        results.append({**m, **norm, "shortlist_label": label})

    shortlist = [r for r in results if r["shortlist_label"] == "SHORTLIST"]

    # Sort by Sharpe descending, then by true_delta_direct_cost desc
    def sort_key(r: dict) -> tuple:
        sh = r.get("sharpe", 0.0) or 0.0
        cf = counterfactual_df.loc[r["variant"]].get("true_delta_direct_cost", 0.0) if r["variant"] in counterfactual_df.index else 0.0
        return (-sh, -cf)

    shortlist.sort(key=sort_key)
    return shortlist[:3]


# ---------------------------------------------------------------------------
# Part 6: Selection lock document
# ---------------------------------------------------------------------------

def build_selection_lock_document(
    shortlist: list[dict],
    output_dir: Path,
) -> str:
    best = shortlist[0]["variant"] if shortlist else "None qualified"
    shortlist_lines = "\n".join(
        f"  {i+1}. {r['variant']} — Sharpe {r.get('sharpe', 'N/A')}, "
        f"CAGR {r.get('cagr_pct', 'N/A')}%, MaxDD {r.get('max_drawdown_pct', 'N/A')}%"
        for i, r in enumerate(shortlist)
    ) or "  (No variant met all shortlist criteria)"

    content = f"""# Final Research Selection Lock
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## Frozen Regime Candidate
combo_entry2_imm_buf05 (BTC MA-240bars, rebalance every 12 bars,
entry_confirm=2, entry_buffer=0.5%, immediate exit)

## All Parameters Tested (Full In-Sample History)

### Regime gate variants (tested in whipsaw study):
- MA lengths: 180, 240, 300, 360, 420, 480 bars
- Rebalance: 6, 12, 24 bars
- Entry confirmation: 0, 2, 3 bars
- Exit confirmation: 0, 1, 2, 3 bars
- Entry buffer: 0%, 0.5%, 1.0%
- Exit buffer: 0%, 0.5%, 1.0%
- Min regime duration: 0, 3, 6, 12 bars
- 4 predefined combinations

### Rank stability variants (tested in rank stability study):
- rank_buffer: 4, 5 (or better)
- challenger_confirm_rebs: 2, 3
- score_hurdle: 0.025, 0.050, 0.100 raw score units
- min_hold_rebs: 2, 4, 6
- cooldown_rebs: 1, 2, 3
- 4 predefined combinations

## Final Rank-Stability Shortlist (In-Sample Only)
{shortlist_lines}

## Status Declarations

1. combo_entry2_imm_buf05 is the frozen regime candidate.
   Selected after observing the full available historical sample.
   NOT out-of-sample validated. NOT approved for live deployment.

2. All rank-stability variants on the shortlist are additional in-sample
   results. Any further parameter change is additional in-sample tuning.

3. The complete in-sample parameter sweep now includes regime gate rules,
   rank-stability rules, MA lengths, rebalance frequencies, and their
   combinations. No further parameter search can be considered unbiased.

4. Future validation must use:
   - Genuinely unseen data (newly acquired historical data)
   - Prospective shadow trading
   - OR live deployment with a separate validation budget

5. No shortlisted variant is approved for live capital deployment.

6. Baseline v1 and all live trading behavior remain unchanged.

## Evidence Summary

- Baseline canonical (no regime gate): Sharpe 1.21, MaxDD -79.7%, CAGR 91.9%
- Frozen candidate (combo_entry2_imm_buf05): Sharpe 1.794, MaxDD -45.9%, CAGR 141.8%
- Best shortlisted rank-stability variant: {best}
- Improvement type: reduced transaction costs, broader score-gap filtering,
  or challenger confirmation, not parameter tuning of lookbacks or signals

{SURVIVORSHIP_BIAS_NOTE}
"""
    return content


# ---------------------------------------------------------------------------
# Counterfactual markdown report
# ---------------------------------------------------------------------------

def build_counterfactual_md(
    counterfactual_df: pd.DataFrame,
    suppression_note: dict,
) -> str:
    rows_md = []
    for variant, row in counterfactual_df.iterrows():
        delta_net = row["true_delta_net_equity"]
        delta_cost = row["true_delta_direct_cost"]
        sign_net = "+" if delta_net >= 0 else ""
        sign_cost = "+" if delta_cost >= 0 else ""
        rows_md.append(
            f"| {variant} | ${row['variant_net_equity']:,.0f} | ${row['control_net_equity']:,.0f} "
            f"| {sign_net}${delta_net:,.0f} | {sign_cost}${delta_cost:,.0f} |"
        )

    table = "\n".join(rows_md)
    content = f"""# Suppression Counterfactual Reconciliation
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## WARNING: Event-Level Sums Are Non-Additive

{SUPPRESSION_NOTE}

## Prior Study Diagnostic Figures (FOR REFERENCE ONLY)

| Figure | Value | Status |
|--------|-------|--------|
| Avoided transaction costs | $3,086,242 | DIAGNOSTIC — non-additive, not realizable |
| Gains from beneficial suppressions | $10,973,593 | DIAGNOSTIC — non-additive, not realizable |

These figures were computed as event-level sums across potentially overlapping
suppression events. They CANNOT be added together and CANNOT be treated as
portfolio profit.

## True Realizable Differences (from equity curve comparison)

| Variant | Variant Net Equity | Control Net Equity | Δ Net Equity | Δ Direct Cost (saved) |
|---------|-------------------|-------------------|-------------|----------------------|
{table}

Notes:
- Δ Net Equity > 0 means the variant finished with more money than the control
- Δ Direct Cost (saved) > 0 means the variant spent LESS on transaction costs
- These figures are computed directly from backtest equity curves
- No decomposition required; no double-counting possible

## Conclusion

The ONLY reliable economic comparison is the equity curve endpoint comparison shown
above. All further analysis of "avoided costs" or "suppression gains" is diagnostic
and must not be presented as realized portfolio benefit.
"""
    return content


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    LOGGER.info("Loading close matrix…")
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, MIN_HISTORY_BARS)
    if joint_start is None:
        raise RuntimeError("Cannot find joint eligible start")

    ohlcv_full = _close_to_ohlcv(close)
    btc_full = close[BTC_COL].dropna()

    LOGGER.info("Joint start: %s", joint_start)

    gate_full = build_frozen_candidate_gate(btc_full)

    LOGGER.info("Pre-computing momentum scores…")
    scores_df = compute_momentum_score(close, SHORT_LOOKBACK, MEDIUM_LOOKBACK)

    LOGGER.info("Running %d selected rank-stability variants…", len(SELECTED_VARIANTS))

    all_norm_efficiency: list[dict] = []
    all_perf_efficiency: list[dict] = []
    all_year_metrics: list[dict] = []
    variants_results: dict[str, tuple[BacktestResult, pd.DataFrame, dict]] = {}

    control_norm: dict | None = None
    control_perf: dict | None = None

    for params in SELECTED_VARIANTS:
        LOGGER.info("  Running variant: %s…", params.name)
        sig_gen = build_stability_signal(gate_full, params, scores_df)
        result, port_fx = run_canonical(
            ohlcv_full, sig_gen, joint_start,
            rebalance_bars=CONTROL_REBALANCE_BARS,
            initial_capital=INITIAL_CAPITAL,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )

        actual_costs = compute_actual_direct_costs(
            result, port_fx, joint_start, FEE_BPS, SLIPPAGE_BPS
        )

        variants_results[params.name] = (result, port_fx, actual_costs)

        # Part 1
        norm = compute_normalized_cost_efficiency(
            params.name, result, port_fx, joint_start, actual_costs
        )
        all_norm_efficiency.append(norm)
        if params.name == "control":
            control_norm = norm

        # Part 2
        perf = compute_performance_efficiency(
            params.name, result, port_fx, joint_start, actual_costs, norm,
            control_metrics=control_norm,
        )
        all_perf_efficiency.append(perf)
        if params.name == "control":
            control_perf = perf

        # Part 4
        for period_name, ps, pe in YEAR_PERIODS:
            pm = compute_period_normalized_metrics(
                params.name, result, port_fx, close, joint_start,
                period_name, ps, pe, actual_costs,
            )
            all_year_metrics.append(pm)

    # Part 3
    LOGGER.info("Computing counterfactual reconciliation…")
    counterfactual_df, suppression_note = compute_counterfactual_reconciliation(
        variants_results, control_name="control"
    )

    # Part 5
    LOGGER.info("Building final shortlist…")
    shortlist = build_final_shortlist(
        all_perf_efficiency, all_norm_efficiency, counterfactual_df, all_year_metrics
    )

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    norm_df = pd.DataFrame(all_norm_efficiency)
    perf_df = pd.DataFrame(all_perf_efficiency)
    year_df = pd.DataFrame(all_year_metrics)

    out1 = output_dir / "fixed_five_normalized_cost_efficiency.csv"
    out2 = output_dir / "fixed_five_performance_efficiency.csv"
    out3 = output_dir / "fixed_five_suppression_counterfactual_reconciliation.md"
    out4 = output_dir / "fixed_five_normalized_efficiency_by_year.csv"
    out5 = output_dir / "fixed_five_final_in_sample_shortlist.md"
    out6 = output_dir / "fixed_five_final_selection_lock.md"

    norm_df.to_csv(out1, index=False)
    perf_df.to_csv(out2, index=False)

    counterfactual_md = build_counterfactual_md(counterfactual_df, suppression_note)
    out3.write_text(counterfactual_md)

    year_df.to_csv(out4, index=False)

    # Shortlist markdown
    norm_map = {r["variant"]: r for r in all_norm_efficiency}
    ctrl_norm_intensity = norm_map.get("control", {}).get("notional_div_twae", float("nan"))
    ctrl_sharpe = (control_perf or {}).get("sharpe", float("nan"))

    shortlist_body = []
    for i, r in enumerate(shortlist, 1):
        v = r["variant"]
        cf = counterfactual_df.loc[v] if v in counterfactual_df.index else pd.Series()
        delta_net = cf.get("true_delta_net_equity", float("nan"))
        delta_cost = cf.get("true_delta_direct_cost", float("nan"))
        shortlist_body.append(
            f"{i}. **{v}** — "
            f"Sharpe {r.get('sharpe', 'N/A')}, "
            f"CAGR {r.get('cagr_pct', 'N/A')}%, "
            f"MaxDD {r.get('max_drawdown_pct', 'N/A')}%, "
            f"Δ net equity ${delta_net:+,.0f}, "
            f"Δ direct cost ${delta_cost:+,.0f}"
        )

    if not shortlist_body:
        shortlist_body = ["No variant met all shortlist criteria"]

    shortlist_md = f"""# Final In-Sample Rank-Stability Shortlist
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## IMPORTANT: These are IN-SAMPLE results only

All shortlisted variants were evaluated on the same historical data used to
select the frozen regime candidate (combo_entry2_imm_buf05). No variant on
this shortlist is approved for live capital deployment.

## Control Baseline
- Sharpe: {ctrl_sharpe}
- Normalized trading intensity (notional/TWAE): {ctrl_norm_intensity}

## Shortlisted Variants (≤ 3)

{chr(10).join(shortlist_body)}

## Shortlist Criteria Applied
1. Normalized trading intensity lower than control OR Sharpe ≥ control_sharpe
2. Sharpe ≥ control_sharpe - 0.10
3. Improvement in ≥ 2 of (2022, 2023, 2024, 2025) years
4. Actual direct costs lower than control (true_delta_direct_cost > 0)
5. Operationally simple rule
6. Max drawdown not worse than control + 5pp

## Status
- All results: IN-SAMPLE ONLY
- Frozen regime candidate: combo_entry2_imm_buf05 (unchanged)
- No shortlisted variant is approved for live capital deployment
- Future validation requires genuinely unseen data or prospective shadow trading
"""
    out5.write_text(shortlist_md)

    lock_content = build_selection_lock_document(shortlist, output_dir)
    out6.write_text(lock_content)

    LOGGER.info("All outputs written to %s", output_dir)

    # ---------------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------------
    print("\n=== NORMALIZED EFFICIENCY STUDY COMPLETE ===\n")
    print("Files created:")
    for f in [out1, out2, out3, out4, out5, out6]:
        print(f"  {f}")

    print("\n=== NORMALIZED TRADING INTENSITY (notional / time-weighted avg equity) ===")
    ctrl_intensity = norm_map.get("control", {}).get("notional_div_twae", 1.0)
    for r in all_norm_efficiency:
        v = r["variant"]
        ni = r["notional_div_twae"]
        if v == "control":
            print(f"control:               {ni:.4f}")
        else:
            pct_diff = (ni / max(ctrl_intensity, 1e-9) - 1) * 100
            sign = "+" if pct_diff >= 0 else ""
            print(f"{v:30s} {ni:.4f}  [{sign}{pct_diff:.1f}% vs control]")

    print("\n=== MIN_HOLD NORMALIZED EFFICIENCY ===")
    for name in ["min_hold_4", "min_hold_6"]:
        nr = norm_map.get(name, {})
        ni = nr.get("notional_div_twae", float("nan"))
        if math.isfinite(ni):
            pct_diff = (ni / max(ctrl_intensity, 1e-9) - 1) * 100
            label = "more" if pct_diff > 0 else "less"
            print(f"{name} normalized intensity vs control: {pct_diff:+.1f}% [{label} intense]")
    mh4 = norm_map.get("min_hold_4", {}).get("notional_div_twae", ctrl_intensity)
    mh6 = norm_map.get("min_hold_6", {}).get("notional_div_twae", ctrl_intensity)
    avg_mh = (mh4 + mh6) / 2
    conclusion = "MORE" if avg_mh > ctrl_intensity else "LESS"
    print(f"Conclusion: min_hold_4 and min_hold_6 are {conclusion} cost-efficient than control")
    print("when measured by direct cost relative to portfolio size.")

    print("\n=== SUPPRESSION COUNTERFACTUAL NOTE ===")
    print("Prior $3.086M and $10.974M figures are NON-ADDITIVE DIAGNOSTIC EXPOSURE ESTIMATES.")
    print("They overlap in time and cannot be realized simultaneously.")
    print("True realizable differences (from equity curve comparison):")
    for variant, row in counterfactual_df.iterrows():
        if variant == "control":
            continue
        dn = row["true_delta_net_equity"]
        dc = row["true_delta_direct_cost"]
        print(f"  {variant:30s} Δnet_equity = ${dn:+,.0f} | Δdirect_cost = ${dc:+,.0f}")

    print("\n=== YEAR-BY-YEAR STABILITY ===")
    pivot_sharpe = year_df.pivot_table(
        index="variant", columns="period", values="period_sharpe", aggfunc="first"
    )
    print(pivot_sharpe.to_string())

    print("\n=== FINAL SHORTLIST (≤3 variants) ===")
    if shortlist:
        for i, r in enumerate(shortlist, 1):
            v = r["variant"]
            cf = counterfactual_df.loc[v] if v in counterfactual_df.index else pd.Series()
            dn = cf.get("true_delta_net_equity", float("nan"))
            dc = cf.get("true_delta_direct_cost", float("nan"))
            print(f"{i}. {v} — Sharpe {r.get('sharpe', 'N/A')}, Δnet ${dn:+,.0f}, Δcost ${dc:+,.0f}")
    else:
        print("No variant met all shortlist criteria")

    print("\nAll shortlist results are IN-SAMPLE ONLY.")
    print("combo_entry2_imm_buf05 frozen candidate remains unchanged.")
    print("No shortlist variant is approved for live deployment.")

    print("\n=== LIVE BEHAVIOR UNCHANGED ===")
    print("No live trading files, configuration, cron, or exchange code was modified.")


if __name__ == "__main__":
    run_all()
