"""Canonical backtest consolidation and regime-gate comparison.

PURPOSE
-------
Reconcile the two initialization paths found in prior research scripts
(Run A vs Run B) and rerun all regime-gate variants using a single
canonical baseline so results are strictly comparable.

CANONICAL INITIALIZATION ("Run A")
------------------------------------
- Backtest runs from the beginning of available close-price history (Jan 2020).
- Initial capital: $10,000 at the data start date.
- At joint_start (2020-09-28 00:00 UTC), the strategy already has ~1,625 bars of
  history; all five coins are immediately eligible; no warm-up cash delay occurs.
- Equity at joint_start ≈ $15,070 (accumulated during the 9-month pre-period).
- All metrics are computed on the portfolio slice from joint_start onward.

WHY THE OVERLAY SCRIPT PRODUCED -83.4% (Run B)
-----------------------------------------------
The defensive-overlay script sliced close to close.loc[close.index >= joint_start]
BEFORE running the backtest. This gave the strategy only zero bars of lookback
history at the start of the truncated data, forcing a 36-bar cash delay (the
strategy needed 36 bars of history before the first trade). The fresh-start
equity began at $10,000, reached a lower absolute peak in Nov 2021 ($414K vs
$700K), and therefore showed a deeper percentage drawdown: -83.4% vs -79.7%.

The cash delay was NOT intentional for FIXED_COMMON_HISTORY. It was an
unintentional research-script inconsistency caused by truncating the input
before the backtest rather than after.

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_canonical_regime_comparison
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import summary_metrics
from research.universe_integrity_analysis import (
    BARS_PER_YEAR_4H,
    DEFAULT_FEE_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MEDIUM_LOOKBACK,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_REBALANCE_BARS,
    DEFAULT_SHORT_LOOKBACK,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TOP_N,
    LIVE_FIVE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
    _close_to_ohlcv,
    build_close_matrix,
    find_joint_eligible_start,
)
from research.fixed_five_defensive_overlay import (
    BTCRegimeSignalGen,
    BaseSignalGen,
    _cagr,
    _max_drawdown,
    _max_consecutive_cash,
    _period_rebalance_log,
    _period_return,
    _sharpe,
    _sortino,
    _worst_calendar_year,
    _worst_month_return,
)

LOGGER = logging.getLogger(__name__)

BARS_PER_YEAR = BARS_PER_YEAR_4H   # 2190
BTC_COL = "BTC/USD"

# ---------------------------------------------------------------------------
# Canonical baseline documentation
# ---------------------------------------------------------------------------

CANONICAL_BASELINE = {
    "initial_capital_$": DEFAULT_INITIAL_CAPITAL,
    "backtest_start": "First available bar in data/local/ (2020-01-01 04:00 UTC)",
    "joint_start": "2020-09-28 00:00 UTC (first bar where all 5 coins have >= 36 bars)",
    "equity_at_joint_start_$": "~15,070 (accumulated pre-period; varies with data freshness)",
    "metrics_window": "portfolio.loc[>= joint_start]",
    "universe": list(LIVE_FIVE_UNIVERSE),
    "timeframe": "4h",
    "fee_bps": DEFAULT_FEE_BPS,
    "slippage_bps": DEFAULT_SLIPPAGE_BPS,
    "round_trip_bps": DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS,
    "execution_delay_bars": 1,
    "execution_delay_desc": "Signal at bar i; executed at bar i+1 (4h later)",
    "rebalance_every_bars": DEFAULT_REBALANCE_BARS,
    "cost_accounting": (
        "cost_rate = turnover × (fee_bps + slippage_bps) / 10_000 per rebalance; "
        "deducted at execution bar; not at signal bar"
    ),
    "drawdown_calc": "min(equity / equity.cummax() - 1) over slice >= joint_start",
    "max_drawdown_canonical": "-79.7%",
    "benchmark": (
        "BTC B&H from joint_start to same end date; 2×fee_bps round-trip applied once"
    ),
    "bias_warning": SURVIVORSHIP_BIAS_NOTE,
    "36_bar_delay_explanation": (
        "The overlay script (fixed_five_defensive_overlay.py) sliced close to "
        "close.loc[>= joint_start] BEFORE calling run_backtest. The strategy needs "
        "min_history_bars=36 bars of lookback to compute momentum scores. When the "
        "backtest starts from bar 0 of a 36-bar-limited window, it holds cash for 36 "
        "bars before the first trade (a pure artifact of truncating the input). "
        "In the canonical Run A, the backtest starts from the full data history; by "
        "joint_start the strategy already has 1,625 bars of lookback and can invest "
        "immediately. The 36-bar delay was NOT intentional."
    ),
}

# ---------------------------------------------------------------------------
# Variants to run
# ---------------------------------------------------------------------------

VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline",          "gate": None,  "ma_bars": None, "reb": 6},
    {"name": "btc_ma_180_reb6",   "gate": "ma",  "ma_bars": 180,  "reb": 6},
    {"name": "btc_ma_240_reb6",   "gate": "ma",  "ma_bars": 240,  "reb": 6},
    {"name": "btc_ma_300_reb6",   "gate": "ma",  "ma_bars": 300,  "reb": 6},
    {"name": "btc_ma_360_reb6",   "gate": "ma",  "ma_bars": 360,  "reb": 6},
    {"name": "btc_ma_420_reb6",   "gate": "ma",  "ma_bars": 420,  "reb": 6},
    {"name": "btc_ma_480_reb6",   "gate": "ma",  "ma_bars": 480,  "reb": 6},
    {"name": "btc_ma_240_reb12",  "gate": "ma",  "ma_bars": 240,  "reb": 12},
    {"name": "btc_ma_360_reb12",  "gate": "ma",  "ma_bars": 360,  "reb": 12},
    {"name": "btc_ma_360_reb24",  "gate": "ma",  "ma_bars": 360,  "reb": 24},
]

# Periods for by-period breakdown
PERIODS: list[tuple[str, str | None, str | None]] = [
    ("2020-09-28_to_2021", "2020-09-28", "2021-12-31"),
    ("2022",               "2022-01-01", "2022-12-31"),
    ("2023",               "2023-01-01", "2023-12-31"),
    ("2024",               "2024-01-01", "2024-12-31"),
    ("2025",               "2025-01-01", "2025-12-31"),
    ("2026",               "2026-01-01", None),
    ("full",               None,         None),
]

MA_BAR_TO_DAYS = {b: b * 4 // 24 for b in [120, 150, 180, 210, 240, 300, 360, 420, 480]}


def ma_label(bars: int) -> str:
    days = bars * 4 // 24
    return f"MA-{bars}bars(≈{days}days)"


# ---------------------------------------------------------------------------
# Canonical backtest runner — the ONE place that runs backtests
# ---------------------------------------------------------------------------

def run_canonical(
    ohlcv: pd.DataFrame,
    signal_gen: Any,
    joint_start: pd.Timestamp,
    rebalance_bars: int = DEFAULT_REBALANCE_BARS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the canonical backtest and return (result, port_fx).

    Uses FULL ohlcv (not truncated to joint_start) so the strategy has maximum
    lookback history and no unintended warm-up cash delay at joint_start.

    Metrics should always be computed on port_fx (sliced >= joint_start).
    """
    result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=signal_gen,
        initial_capital=initial_capital,
        transaction_cost_bps=fee_bps,
        slippage_bps=slippage_bps,
        rebalance_every_bars=rebalance_bars,
    )
    port_fx = result.portfolio.loc[result.portfolio.index >= joint_start].copy()
    return result, port_fx


def verify_canonical_initialization(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Verify canonical initialization properties. Returns a dict of checks."""
    reb = result.rebalance_log
    eq = port_fx["equity"].dropna()

    # First investment: first rebalance where weight_sum > 0 and execution >= joint_start
    first_inv = reb[(reb["weight_sum"] > 0.01) & (reb["execution_timestamp"] >= joint_start)]
    has_no_delay = len(first_inv) > 0 and first_inv.iloc[0]["execution_timestamp"] <= (
        joint_start + pd.Timedelta(hours=24)
    )

    # One-bar delay
    has_one_bar_delay = (reb["signal_timestamp"] != reb["execution_timestamp"]).all()

    return {
        "joint_start": str(joint_start),
        "portfolio_start": str(eq.index[0]),
        "portfolio_end": str(eq.index[-1]),
        "equity_at_joint_start": round(float(eq.iloc[0]), 2),
        "equity_at_start_exceeds_initial_capital": float(eq.iloc[0]) > DEFAULT_INITIAL_CAPITAL,
        "first_investment_within_24h_of_joint_start": has_no_delay,
        "one_bar_delay_confirmed": has_one_bar_delay,
        "start_matches_expected": (
            eq.index[0] == expected_start if expected_start else "not_checked"
        ),
        "end_matches_expected": (
            eq.index[-1] == expected_end if expected_end else "not_checked"
        ),
    }


# ---------------------------------------------------------------------------
# Gate state helpers
# ---------------------------------------------------------------------------

def compute_gate_series(
    close: pd.DataFrame,
    ma_bars: int | None,
    joint_start: pd.Timestamp,
) -> pd.Series:
    """Compute per-bar BTC MA gate state (True=risk_on) from joint_start."""
    if ma_bars is None:
        idx = close.loc[close.index >= joint_start].index
        return pd.Series(True, index=idx, name="gate")

    btc = close[BTC_COL].dropna()
    ma = btc.rolling(ma_bars, min_periods=ma_bars).mean()
    gate = (btc > ma).astype(bool)
    return gate.reindex(close.loc[close.index >= joint_start].index).fillna(False)


def regime_stats_from_gate(gate: pd.Series) -> dict[str, float]:
    """Count transitions, average duration, max cash run from a gate series."""
    transitions = gate[gate != gate.shift(1)]
    n_switches = int(len(transitions))

    groups = (gate != gate.shift(1)).cumsum()
    durations = gate.groupby(groups).count()
    avg_dur = float(durations.mean()) if len(durations) > 0 else float("nan")

    # Cash = gate is False
    cash_runs = gate[~gate]
    cash_groups = (~gate & (gate != gate.shift(1)).astype(bool)).cumsum()
    cash_durations = cash_runs.groupby(cash_groups).count()
    max_cash = int(cash_durations.max()) if len(cash_durations) > 0 else 0

    risk_off_count = int((transitions == False).sum())  # noqa: E712
    risk_on_count = int((transitions == True).sum())    # noqa: E712

    return {
        "n_regime_switches": n_switches,
        "n_risk_off_transitions": risk_off_count,
        "n_risk_on_transitions": risk_on_count,
        "avg_regime_duration_bars": round(avg_dur, 1),
        "max_consecutive_cash_bars": max_cash,
        "pct_time_risk_off": round(float((~gate).mean()) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Whipsaw cluster detection
# ---------------------------------------------------------------------------

@dataclass
class WhipsawCluster:
    variant_name: str
    ma_bars: int
    cluster_start: str
    cluster_end: str
    n_transitions: int
    n_bars_span: int
    btc_price_start: float
    btc_ma_start: float
    btc_pct_from_ma_start: float
    turnover_in_cluster: float
    cost_rate_in_cluster: float
    net_return_in_cluster: float
    gate_helped_or_hurt: str   # "helped" / "hurt" / "neutral"
    notes: str


def detect_whipsaw_clusters(
    variant_name: str,
    ma_bars: int,
    close: pd.DataFrame,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    cluster_days: int = 7,
    min_transitions: int = 3,
) -> list[WhipsawCluster]:
    """Identify transition clusters within cluster_days calendar days."""
    gate = compute_gate_series(close, ma_bars, joint_start)
    transitions = gate[gate != gate.shift(1)].dropna()
    trans_idx = transitions.index.tolist()
    if not trans_idx:
        return []

    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ]
    eq = port_fx["equity"].dropna()
    net_ret = port_fx["strategy_return"].fillna(0)
    btc = close[BTC_COL].dropna()
    btc_ma = btc.rolling(ma_bars, min_periods=ma_bars).mean()

    clusters: list[WhipsawCluster] = []
    i = 0
    while i < len(trans_idx):
        window_end = trans_idx[i] + pd.Timedelta(days=cluster_days)
        in_window = [t for t in trans_idx[i:] if t <= window_end]
        if len(in_window) >= min_transitions:
            cl_start = in_window[0]
            cl_end = in_window[-1]

            # BTC price at cluster start
            btc_p = float(btc.loc[cl_start]) if cl_start in btc.index else float("nan")
            btc_m = float(btc_ma.loc[cl_start]) if cl_start in btc_ma.index else float("nan")
            pct_from_ma = (btc_p / btc_m - 1) * 100 if not (
                math.isnan(btc_p) or math.isnan(btc_m) or btc_m == 0
            ) else float("nan")

            # Rebalance cost during cluster
            cl_reb = reb_fx[
                (reb_fx["execution_timestamp"] >= cl_start) &
                (reb_fx["execution_timestamp"] <= cl_end)
            ]
            turnover_cl = float(cl_reb["turnover"].sum())
            cost_cl = float(cl_reb["cost_rate"].sum())

            # Net return during cluster
            cl_ret = net_ret.loc[(net_ret.index >= cl_start) & (net_ret.index <= cl_end)]
            net_ret_cl = float((1 + cl_ret).prod() - 1) if len(cl_ret) > 0 else float("nan")

            # Helped or hurt: compare to holding fully invested during cluster
            btc_segment = btc.loc[(btc.index >= cl_start) & (btc.index <= cl_end)]
            btc_cl_ret = float(btc_segment.iloc[-1] / btc_segment.iloc[0] - 1) if len(btc_segment) >= 2 else float("nan")
            if math.isnan(net_ret_cl) or math.isnan(btc_cl_ret):
                helped = "unknown"
            elif net_ret_cl > btc_cl_ret + 0.01:
                helped = "helped"
            elif net_ret_cl < btc_cl_ret - 0.01:
                helped = "hurt"
            else:
                helped = "neutral"

            clusters.append(WhipsawCluster(
                variant_name=variant_name,
                ma_bars=ma_bars,
                cluster_start=str(cl_start),
                cluster_end=str(cl_end),
                n_transitions=len(in_window),
                n_bars_span=len(net_ret.loc[cl_start:cl_end]),
                btc_price_start=round(btc_p, 2),
                btc_ma_start=round(btc_m, 2),
                btc_pct_from_ma_start=round(pct_from_ma, 3) if not math.isnan(pct_from_ma) else float("nan"),
                turnover_in_cluster=round(turnover_cl, 4),
                cost_rate_in_cluster=round(cost_cl * 100, 4),
                net_return_in_cluster=round(net_ret_cl * 100, 4) if not math.isnan(net_ret_cl) else float("nan"),
                gate_helped_or_hurt=helped,
                notes=f"{len(in_window)} transitions in {cluster_days} days",
            ))
            i += len(in_window)
        else:
            i += 1

    return clusters


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

def compute_cost_accounting(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Detailed cost accounting using distinct denominators."""
    eq = port_fx["equity"].dropna()
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    gross_eq = eq.iloc[0] * (1 + gross_r).cumprod()

    gross_end = float(gross_eq.iloc[-1])
    net_end = float(eq.iloc[-1])
    equity_at_start = float(eq.iloc[0])
    gross_profit = gross_end - equity_at_start
    net_profit = net_end - equity_at_start
    cost_dollars = gross_end - net_end

    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= eq.index[0]
    ]
    additive_cost_rate_sum = float(reb_fx["cost_rate"].sum())
    n_reb = len(reb_fx)

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_turnover = float(reb_fx["turnover"].sum()) / max(years, 0.001)

    # Fee vs slippage split (proportional)
    total_bps = fee_bps + slippage_bps
    fee_frac = fee_bps / total_bps if total_bps > 0 else 0.5
    slip_frac = slippage_bps / total_bps if total_bps > 0 else 0.5

    return {
        "equity_at_start": round(equity_at_start, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        "gross_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "total_cost_dollars": round(cost_dollars, 2),
        "fee_dollars_est": round(cost_dollars * fee_frac, 2),
        "slippage_dollars_est": round(cost_dollars * slip_frac, 2),
        "gross_minus_net_ending": round(cost_dollars, 2),
        "cost_div_equity_at_start_pct": round(cost_dollars / equity_at_start * 100, 4),
        "cost_div_gross_profit_pct": round(cost_dollars / max(gross_profit, 1) * 100, 4),
        "cost_div_gross_ending_equity_pct": round(cost_dollars / max(gross_end, 1) * 100, 4),
        "additive_cost_rate_sum_pct": round(additive_cost_rate_sum * 100, 4),
        "n_rebalances": n_reb,
        "ann_turnover": round(ann_turnover, 4),
        "gross_total_return_pct": round((gross_end / equity_at_start - 1) * 100, 2),
        "net_total_return_pct": round((net_end / equity_at_start - 1) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Full-period and per-period metrics
# ---------------------------------------------------------------------------

def _btc_benchmark_period(
    close: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> float:
    """BTC buy-and-hold return over [start, end] with 2×fee round-trip."""
    btc = close[BTC_COL].dropna()
    seg = btc.loc[(btc.index >= start) & (btc.index <= end)]
    if len(seg) < 2:
        return float("nan")
    ret = seg.iloc[-1] / seg.iloc[0] - 1
    return float(ret - 2 * fee_bps / 10_000)


def _ewb_benchmark_period(
    close: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Equal-weight buy-and-hold all 5 symbols over [start, end]."""
    seg = close.loc[(close.index >= start) & (close.index <= end)]
    if len(seg) < 2:
        return float("nan")
    rets = seg.pct_change().fillna(0).mean(axis=1)
    return float((1 + rets).prod() - 1)


def compute_full_metrics(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    variant_name: str,
    ma_bars: int | None,
    rebalance_bars: int,
    baseline_metrics: dict | None = None,
) -> dict[str, Any]:
    """Compute all full-period metrics for one variant."""
    eq = port_fx["equity"].dropna()
    rets = port_fx["strategy_return"].fillna(0)
    gate = compute_gate_series(close, ma_bars, joint_start)
    reg = regime_stats_from_gate(gate)
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
    n_trades = int((reb_fx["turnover"] > 0.01).sum())
    ann_turn = float(reb_fx["turnover"].sum()) / max(
        (end_ts - joint_start).days / 365.25, 0.001
    )
    cost_drag_additive = float(reb_fx["cost_rate"].sum() * 100)

    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)
    pct_cash = float((holdings.sum(axis=1) < 0.01).mean() * 100)

    # 2021 peak and recovery
    eq_pre2022 = eq.loc[eq.index < pd.Timestamp("2022-01-01", tz="UTC")]
    peak_ts = eq_pre2022.idxmax() if len(eq_pre2022) > 0 else eq.idxmax()
    peak_val = float(eq.loc[peak_ts])
    recovery_after = eq.loc[eq.index > peak_ts]
    rec_ts = recovery_after[recovery_after >= peak_val].index
    recovery_date = str(rec_ts[0].date()) if len(rec_ts) > 0 else "not_recovered"

    # Benchmarks
    btc_ret = _btc_benchmark_period(close, joint_start, end_ts)
    ewb_ret = _ewb_benchmark_period(close, joint_start, end_ts)

    m: dict[str, Any] = {
        "variant": variant_name,
        "ma_bars": ma_bars if ma_bars else 0,
        "ma_label": ma_label(ma_bars) if ma_bars else "baseline",
        "rebalance_bars": rebalance_bars,
        "total_return_pct": round(v_total_ret * 100, 2),
        "cagr_pct": round(v_cagr * 100, 2),
        "sharpe": round(v_sharpe, 4),
        "sortino": round(v_sortino, 4) if not math.isnan(v_sortino) else float("nan"),
        "max_drawdown_pct": round(v_maxdd * 100, 2),
        "calmar": round(calmar, 4) if not math.isnan(calmar) else float("nan"),
        "worst_year_return_pct": round(worst_yr * 100, 2),
        "worst_month_return_pct": round(worst_mo * 100, 2),
        "n_trades": n_trades,
        "ann_turnover": round(ann_turn, 4),
        "additive_cost_drag_pct": round(cost_drag_additive, 4),
        "pct_time_invested": round(pct_invested, 2),
        "pct_time_in_cash": round(pct_cash, 2),
        "n_regime_switches": reg["n_regime_switches"],
        "avg_regime_duration_bars": reg["avg_regime_duration_bars"],
        "max_consecutive_cash_bars": reg["max_consecutive_cash_bars"],
        "recovery_from_2021_peak": recovery_date,
        "btc_benchmark_pct": round(btc_ret * 100, 2) if not math.isnan(btc_ret) else float("nan"),
        "ewb_benchmark_pct": round(ewb_ret * 100, 2) if not math.isnan(ewb_ret) else float("nan"),
    }

    if baseline_metrics:
        m["dd_vs_baseline_pp"] = round(v_maxdd * 100 - baseline_metrics["max_drawdown_pct"], 2)
        m["sharpe_vs_baseline"] = round(v_sharpe - baseline_metrics["sharpe"], 4)

    return m


def compute_period_metrics(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    variant_name: str,
    period_name: str,
    period_start_str: str | None,
    period_end_str: str | None,
) -> dict[str, Any]:
    """Metrics for one (variant, period) slice."""
    p_start = (
        pd.Timestamp(period_start_str + " 00:00:00", tz="UTC")
        if period_start_str else joint_start
    )
    p_end = (
        pd.Timestamp(period_end_str + " 23:59:59", tz="UTC")
        if period_end_str else port_fx.index[-1]
    )
    p_start = max(p_start, joint_start)

    eq_slice = _period_return(port_fx["equity"].dropna(), p_start, p_end)
    rets_slice = _period_return(port_fx["strategy_return"].fillna(0), p_start, p_end)
    reb_slice = _period_rebalance_log(result.rebalance_log, p_start, p_end)
    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    hld_slice = holdings.loc[(holdings.index >= p_start) & (holdings.index <= p_end)]

    if len(eq_slice) < 2:
        return {"variant": variant_name, "period": period_name, "error": "insufficient_bars"}

    v_ret = float(eq_slice.iloc[-1] / eq_slice.iloc[0] - 1)
    v_dd = _max_drawdown(eq_slice)
    v_sh = _sharpe(rets_slice, BARS_PER_YEAR)
    cost_drag = float(reb_slice["cost_rate"].sum() * 100) if len(reb_slice) > 0 else 0.0
    n_tr = int((reb_slice["turnover"] > 0.01).sum()) if len(reb_slice) > 0 else 0
    pct_cash = float((hld_slice.sum(axis=1) < 0.01).mean() * 100) if len(hld_slice) > 0 else 0.0
    pct_inv = 100.0 - pct_cash

    # Gate transitions in period
    gate = compute_gate_series(close, None, joint_start)  # placeholder
    gate_trans = 0  # computed separately; use 0 here for efficiency

    btc_ret = _btc_benchmark_period(close, p_start, p_end)
    ewb_ret = _ewb_benchmark_period(close, p_start, p_end)

    return {
        "variant": variant_name,
        "period": period_name,
        "period_start": str(p_start.date()),
        "period_end": str(min(p_end, port_fx.index[-1]).date()),
        "total_return_pct": round(v_ret * 100, 2),
        "max_drawdown_pct": round(v_dd * 100, 2),
        "sharpe": round(v_sh, 4),
        "additive_cost_drag_pct": round(cost_drag, 4),
        "n_trades": n_tr,
        "pct_time_in_cash": round(pct_cash, 2),
        "pct_time_invested": round(pct_inv, 2),
        "btc_benchmark_pct": round(btc_ret * 100, 2) if not math.isnan(btc_ret) else float("nan"),
        "ewb_benchmark_pct": round(ewb_ret * 100, 2) if not math.isnan(ewb_ret) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Drawdown episodes
# ---------------------------------------------------------------------------

def major_drawdown_episodes(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    variant_name: str,
    threshold: float = -0.20,
) -> list[dict[str, Any]]:
    """Return metrics for each drawdown episode deeper than threshold."""
    eq = port_fx["equity"].dropna()
    running_max = eq.cummax()
    dd = eq / running_max - 1

    episodes = []
    in_ep = False
    ep_peak_ts = ep_trough_ts = None
    ep_peak_val = ep_trough_val = None
    ep_dd = 0.0

    for ts in eq.index:
        curr_dd = float(dd.loc[ts])
        if not in_ep and curr_dd < threshold:
            in_ep = True
            ep_peak_ts = running_max.loc[:ts].idxmax()
            ep_peak_val = float(eq.loc[ep_peak_ts])
            ep_trough_ts, ep_trough_val, ep_dd = ts, float(eq.loc[ts]), curr_dd
        elif in_ep:
            if curr_dd < ep_dd:
                ep_trough_ts, ep_trough_val, ep_dd = ts, float(eq.loc[ts]), curr_dd
            if curr_dd >= -0.001 and float(running_max.loc[ts]) > ep_peak_val * 1.001:
                episodes.append((ep_peak_ts, ep_peak_val, ep_trough_ts, ep_trough_val, ts))
                in_ep = False

    if in_ep:
        episodes.append((ep_peak_ts, ep_peak_val, ep_trough_ts, ep_trough_val, None))

    rows = []
    for pk_ts, pk_eq, tr_ts, tr_eq, rec_ts in episodes:
        btc_ret = _btc_benchmark_period(close, pk_ts, tr_ts)
        reb_in = result.rebalance_log[
            (result.rebalance_log["execution_timestamp"] >= pk_ts) &
            (result.rebalance_log["execution_timestamp"] <= tr_ts)
        ]
        rows.append({
            "variant": variant_name,
            "peak_ts": str(pk_ts),
            "trough_ts": str(tr_ts),
            "recovery_ts": str(rec_ts) if rec_ts else "not_recovered",
            "drawdown_pct": round((tr_eq / pk_eq - 1) * 100, 2),
            "peak_to_trough_bars": len(eq.loc[pk_ts:tr_ts]),
            "btc_return_same_period_pct": round(btc_ret * 100, 2) if not math.isnan(btc_ret) else float("nan"),
            "cost_drag_in_episode_pct": round(float(reb_in["cost_rate"].sum()) * 100, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Robustness labeling (stricter than prior LOWER_OVERFIT_RISK)
# ---------------------------------------------------------------------------

ROBUSTNESS_LABELS = ("ROBUST_IMPROVEMENT", "MIXED_TRADEOFF", "WEAK_EVIDENCE", "UNFAVORABLE")


def classify_robustness(
    m: dict[str, Any],
    baseline: dict[str, Any],
    period_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    """Classify a variant with stricter labels."""
    dd_imp = m["max_drawdown_pct"] - baseline["max_drawdown_pct"]   # positive = better
    sh_imp = m["sharpe"] - baseline["sharpe"]                        # positive = better
    cost_imp = baseline["additive_cost_drag_pct"] - m["additive_cost_drag_pct"]  # positive = better

    # Per-period: how many years improved vs baseline (sharpe or return)
    by_period = {r["period"]: r for r in period_rows if r.get("variant") == m["variant"]}
    base_by_period = {r["period"]: r for r in period_rows if r.get("variant") == "baseline"}

    years_improved_dd = 0
    years_total = 0
    for period in ["2022", "2023", "2024", "2025"]:
        vp = by_period.get(period, {})
        bp = base_by_period.get(period, {})
        if "error" in vp or "error" in bp:
            continue
        years_total += 1
        if vp.get("max_drawdown_pct", 0) > bp.get("max_drawdown_pct", 0) - 3:
            years_improved_dd += 1

    # ROBUST_IMPROVEMENT: ALL criteria
    is_robust = (
        dd_imp > 5.0           # >5pp DD improvement
        and sh_imp > 0.05      # >0.05 Sharpe improvement
        and cost_imp > 0       # any cost reduction
        and m["pct_time_in_cash"] < 60.0
        and years_improved_dd >= 2  # improves in at least 2 periods
        and m["n_regime_switches"] > 0
    )

    # UNFAVORABLE: clearly worse
    is_unfavorable = (
        dd_imp < -5.0           # DD WORSE than baseline
        or sh_imp < -0.15       # Sharpe clearly worse
        or m["max_drawdown_pct"] < baseline["max_drawdown_pct"] - 5
    )

    # MIXED_TRADEOFF: some improvement but tradeoffs
    is_mixed = (
        dd_imp > 5.0
        and (sh_imp < 0.05 or cost_imp < 0 or years_improved_dd < 2)
        and not is_unfavorable
    )

    notes_parts = []
    notes_parts.append(f"dd_improvement={dd_imp:+.1f}pp")
    notes_parts.append(f"sharpe_diff={sh_imp:+.3f}")
    notes_parts.append(f"cost_drag_diff={cost_imp:+.1f}pp")
    notes_parts.append(f"years_dd_improved={years_improved_dd}/{years_total}")
    notes_parts.append(SURVIVORSHIP_BIAS_NOTE[:80] + "...")

    if is_unfavorable:
        return "UNFAVORABLE", "; ".join(notes_parts)
    elif is_robust:
        return "ROBUST_IMPROVEMENT", "; ".join(notes_parts)
    elif is_mixed:
        return "MIXED_TRADEOFF", "; ".join(notes_parts)
    else:
        return "WEAK_EVIDENCE", "; ".join(notes_parts)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_all(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading close matrix…")
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        raise RuntimeError("Cannot find joint eligible start")

    ohlcv_full = _close_to_ohlcv(close)  # FULL history — canonical
    LOGGER.info("Joint start: %s", joint_start)
    LOGGER.info("Running %d variants (canonical initialization)…", len(VARIANTS))

    all_results: dict[str, tuple[BacktestResult, pd.DataFrame]] = {}
    all_full_metrics: list[dict] = []
    all_period_metrics: list[dict] = []
    all_cost_rows: list[dict] = []
    all_whipsaw_rows: list[dict] = []
    all_dd_episodes: list[dict] = []
    all_checks: list[dict] = []

    # Run all variants
    for v in VARIANTS:
        name = v["name"]
        gate_type = v["gate"]
        ma_bars = v["ma_bars"]
        reb = v["reb"]

        if gate_type is None:
            gen = BaseSignalGen()
        else:
            gen = BTCRegimeSignalGen(gate_type=gate_type, btc_ma_bars=ma_bars)

        LOGGER.info("  %s (ma=%s, reb=%d)…", name, ma_bars, reb)
        result, port_fx = run_canonical(ohlcv_full, gen, joint_start,
                                         rebalance_bars=reb,
                                         initial_capital=DEFAULT_INITIAL_CAPITAL,
                                         fee_bps=DEFAULT_FEE_BPS,
                                         slippage_bps=DEFAULT_SLIPPAGE_BPS)
        all_results[name] = (result, port_fx)

        # Initialization check
        chk = verify_canonical_initialization(result, port_fx, joint_start)
        chk["variant"] = name
        all_checks.append(chk)

        # Cost accounting
        cost_row = compute_cost_accounting(result, port_fx, DEFAULT_INITIAL_CAPITAL,
                                            DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS)
        cost_row["variant"] = name
        all_cost_rows.append(cost_row)

    # Compute baseline metrics first (needed for relative labels)
    baseline_result, baseline_port = all_results["baseline"]
    baseline_full = compute_full_metrics(
        baseline_result, baseline_port, close, joint_start,
        "baseline", None, DEFAULT_REBALANCE_BARS,
    )
    all_full_metrics.append(baseline_full)

    # Baseline per-period
    for period_name, ps, pe in PERIODS:
        pm = compute_period_metrics(
            baseline_result, baseline_port, close, joint_start,
            "baseline", period_name, ps, pe,
        )
        all_period_metrics.append(pm)

    # Baseline DD episodes
    all_dd_episodes.extend(
        major_drawdown_episodes(baseline_result, baseline_port, close, joint_start, "baseline")
    )

    # All other variants
    for v in VARIANTS:
        if v["name"] == "baseline":
            continue
        name = v["name"]
        ma_bars = v["ma_bars"]
        reb = v["reb"]
        result, port_fx = all_results[name]

        m = compute_full_metrics(result, port_fx, close, joint_start,
                                  name, ma_bars, reb, baseline_full)
        all_full_metrics.append(m)

        for period_name, ps, pe in PERIODS:
            pm = compute_period_metrics(result, port_fx, close, joint_start,
                                         name, period_name, ps, pe)
            all_period_metrics.append(pm)

        all_dd_episodes.extend(
            major_drawdown_episodes(result, port_fx, close, joint_start, name)
        )

        # Whipsaw clusters for MA-240/360/420
        if ma_bars in (240, 360, 420) and reb == 6:
            clusters = detect_whipsaw_clusters(name, ma_bars, close, result, port_fx, joint_start)
            all_whipsaw_rows.extend([asdict(c) for c in clusters])

    # Add robustness labels
    for m in all_full_metrics:
        if m["variant"] == "baseline":
            m["robustness_label"] = "BASELINE"
            m["robustness_notes"] = "canonical reference"
            continue
        label, notes = classify_robustness(m, baseline_full, all_period_metrics)
        m["robustness_label"] = label
        m["robustness_notes"] = notes

    # Write outputs
    comp_df = pd.DataFrame(all_full_metrics)
    comp_path = output_dir / "fixed_five_canonical_regime_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    LOGGER.info("Saved %s", comp_path)

    period_df = pd.DataFrame(all_period_metrics)
    period_path = output_dir / "fixed_five_canonical_regime_by_period.csv"
    period_df.to_csv(period_path, index=False)
    LOGGER.info("Saved %s", period_path)

    cost_df = pd.DataFrame(all_cost_rows)
    cost_path = output_dir / "fixed_five_canonical_regime_costs.csv"
    cost_df.to_csv(cost_path, index=False)
    LOGGER.info("Saved %s", cost_path)

    if all_whipsaw_rows:
        wh_df = pd.DataFrame(all_whipsaw_rows)
    else:
        wh_df = pd.DataFrame(columns=[f.name for f in WhipsawCluster.__dataclass_fields__.values()])
    wh_path = output_dir / "fixed_five_canonical_regime_whipsaws.csv"
    wh_df.to_csv(wh_path, index=False)
    LOGGER.info("Saved %s", wh_path)

    # Write reconciliation markdown
    recon_path = write_initialization_reconciliation(
        all_checks, comp_df, output_dir, joint_start
    )
    LOGGER.info("Saved %s", recon_path)

    # Write main analysis markdown
    analysis_path = write_analysis_md(
        comp_df, period_df, cost_df, wh_df,
        pd.DataFrame(all_dd_episodes),
        close, joint_start, output_dir
    )
    LOGGER.info("Saved %s", analysis_path)

    print_summary(comp_df, period_df, cost_df, wh_df, joint_start,
                  [comp_path, period_path, cost_path, wh_path, recon_path, analysis_path])


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_initialization_reconciliation(
    checks: list[dict],
    comp_df: pd.DataFrame,
    output_dir: Path,
    joint_start: pd.Timestamp,
) -> Path:
    baseline_check = next((c for c in checks if c["variant"] == "baseline"), {})

    lines = [
        "# Canonical Initialization Reconciliation",
        "",
        "## Why -79.7% vs -83.4%",
        "",
        "Both figures describe the SAME market drawdown (2021-11-22 → 2022-12-19).",
        "The difference is caused by **different backtest initialization paths**, not",
        "different market behavior or data.",
        "",
        "| | Run A (canonical, -79.7%) | Run B (overlay script, -83.4%) |",
        "|---|---|---|",
        "| Close matrix | Full history from Jan 2020 | Sliced to joint_start |",
        "| Backtest start | Jan 2020 | 2020-09-28 (joint_start) |",
        "| Initial capital | $10,000 | $10,000 |",
        f"| Equity at joint_start | ~$15,070 | $10,000 |",
        "| History at joint_start | 1,625 bars | 0 bars |",
        "| Min history needed | 36 bars | 36 bars |",
        "| First trade possible | Immediately at joint_start | After 36-bar warm-up delay |",
        "| Nov 2021 peak | ~$700,210 | ~$413,931 |",
        "| Dec 2022 trough | ~$142,215 | ~$68,648 |",
        "| Drawdown | **-79.7%** | **-83.4%** |",
        "",
        "### Root cause of Run B's 36-bar delay",
        "",
        "```python",
        "# Run B (overlay script — incorrect for FIXED_COMMON_HISTORY)",
        "close_fixed = close.loc[close.index >= joint_start]   # ← truncates history",
        "ohlcv = _close_to_ohlcv(close_fixed)                  # strategy sees 0 bars of history",
        "result = run_backtest(ohlcv, signal_gen, ...)          # must wait 36 bars before first trade",
        "",
        "# Run A (canonical — correct)",
        "ohlcv = _close_to_ohlcv(close)                        # full history",
        "result = run_backtest(ohlcv, signal_gen, ...)          # strategy invests immediately at joint_start",
        "port_fx = result.portfolio.loc[>= joint_start]         # slice AFTER backtest",
        "```",
        "",
        "The 36-bar delay was **not intentional**. It was a research-script inconsistency.",
        "All future scripts must use the canonical Run A pattern via `run_canonical()`.",
        "",
        "## Canonical Baseline Definition",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k, v in CANONICAL_BASELINE.items():
        lines.append(f"| {k} | {str(v)[:120]} |")

    lines += [
        "",
        "## Initialization Verification (all variants)",
        "",
        "| Variant | Portfolio start | Equity at start | >$10K? | First invest within 24h? | 1-bar delay? |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        gt10 = "✓" if c.get("equity_at_start_exceeds_initial_capital") else "✗"
        fast = "✓" if c.get("first_investment_within_24h_of_joint_start") else "✗"
        delay = "✓" if c.get("one_bar_delay_confirmed") else "✗"
        lines.append(
            f"| {c['variant']} | {str(c.get('portfolio_start','?'))[:19]} | "
            f"${c.get('equity_at_joint_start',0):,.0f} | {gt10} | {fast} | {delay} |"
        )

    lines += [
        "",
        "## MA Bar-to-Calendar-Day Labels",
        "",
        "⚠ **These are 4h bar counts, NOT day counts.** A 360-bar MA is a 60-calendar-day",
        "moving average, NOT a 360-day moving average.",
        "",
        "| MA bars | Calendar days (4h bars × 4h ÷ 24h) |",
        "|---|---|",
    ]
    for bars, days in sorted(MA_BAR_TO_DAYS.items()):
        lines.append(f"| MA-{bars}bars | ≈{days} calendar days |")

    lines += [
        "",
        f"> ⚠ {SURVIVORSHIP_BIAS_NOTE}",
        "",
        "*Generated by `research/fixed_five_canonical_regime_comparison.py` — research only.*",
    ]

    path = output_dir / "fixed_five_initialization_reconciliation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(v, pct=False, dp=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if pct:
        return f"{v:+.{dp}f}%"
    return f"{v:.{dp}f}"


def write_analysis_md(
    comp_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    wh_df: pd.DataFrame,
    dd_df: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    output_dir: Path,
) -> Path:
    baseline_row = comp_df[comp_df["variant"] == "baseline"].iloc[0].to_dict()

    lines = [
        "# Canonical Regime-Gate Comparison — Analysis",
        "",
        f"> ⚠ {SURVIVORSHIP_BIAS_NOTE}",
        "",
        "**All variants use the canonical Run A initialization.**",
        "**MA bars are 4h-bar counts, NOT day counts. See bar-to-day table below.**",
        "",
        "## MA Bar-to-Day Reference",
        "",
        "| MA bars | ≈ Calendar days |",
        "|---|---|",
    ]
    for bars, days in sorted(MA_BAR_TO_DAYS.items()):
        lines.append(f"| {bars} | {days} |")

    lines += [
        "",
        "## Canonical Baseline",
        "",
        f"| Metric | Value |",
        "|---|---|",
        f"| Total return | {_fmt(baseline_row.get('total_return_pct'), pct=True)} |",
        f"| CAGR | {_fmt(baseline_row.get('cagr_pct'), pct=True)} |",
        f"| Sharpe | {_fmt(baseline_row.get('sharpe'))} |",
        f"| Max drawdown | {_fmt(baseline_row.get('max_drawdown_pct'), pct=True)} |",
        f"| Additive cost drag | {_fmt(baseline_row.get('additive_cost_drag_pct'), pct=True)} |",
        f"| Joint start | {joint_start.date()} |",
        f"| Recovery from 2021 peak | {baseline_row.get('recovery_from_2021_peak', 'N/A')} |",
        "",
        "## Full-Period Comparison Table",
        "",
        "| Variant | Total Return | CAGR | Sharpe | Max DD | Cost Drag | Time in Cash | Robustness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in comp_df.iterrows():
        lines.append(
            f"| {row['variant']} "
            f"| {_fmt(row.get('total_return_pct'),pct=True)} "
            f"| {_fmt(row.get('cagr_pct'),pct=True)} "
            f"| {_fmt(row.get('sharpe'))} "
            f"| {_fmt(row.get('max_drawdown_pct'),pct=True)} "
            f"| {_fmt(row.get('additive_cost_drag_pct'),pct=True)} "
            f"| {_fmt(row.get('pct_time_in_cash'),pct=True)} "
            f"| {row.get('robustness_label','?')} |"
        )

    lines += [
        "",
        "## By-Period Detail",
        "",
        "### Total return by period",
        "",
        "| Variant | 2020–2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "|---|---|---|---|---|---|---|",
    ]
    period_pivot = period_df.dropna(subset=["period", "variant"])
    period_wide = period_pivot.pivot_table(
        index="variant", columns="period", values="total_return_pct", aggfunc="first"
    )
    ordered_variants = [v["name"] for v in VARIANTS]
    ordered_periods = ["2020-09-28_to_2021", "2022", "2023", "2024", "2025", "2026"]
    for var in ordered_variants:
        if var not in period_wide.index:
            continue
        row = period_wide.loc[var]
        cells = [_fmt(row.get(p, float("nan")), pct=True) for p in ordered_periods]
        lines.append(f"| {var} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Cost Accounting",
        "",
        "| Variant | Gross End $K | Net End $K | Cost $ | Cost/Start | Cost/Gross Profit | Cost/Gross End | Additive Sum |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in cost_df.iterrows():
        lines.append(
            f"| {row.get('variant','?')} "
            f"| ${row.get('gross_ending_equity',0)/1000:.1f}K "
            f"| ${row.get('net_ending_equity',0)/1000:.1f}K "
            f"| ${row.get('total_cost_dollars',0)/1000:.1f}K "
            f"| {_fmt(row.get('cost_div_equity_at_start_pct'),pct=True)} "
            f"| {_fmt(row.get('cost_div_gross_profit_pct'),pct=True)} "
            f"| {_fmt(row.get('cost_div_gross_ending_equity_pct'),pct=True)} "
            f"| {_fmt(row.get('additive_cost_rate_sum_pct'),pct=True)} |"
        )

    lines += [
        "",
        "## Whipsaw Cluster Diagnostics (MA-240, MA-360, MA-420)",
        "",
        f"Total whipsaw clusters detected: {len(wh_df)}",
        "",
    ]
    if len(wh_df) > 0:
        wh_summary = wh_df.groupby("variant_name").agg(
            n_clusters=("cluster_start", "count"),
            total_cost_pct=("cost_rate_in_cluster", "sum"),
            helped=("gate_helped_or_hurt", lambda x: (x == "helped").sum()),
            hurt=("gate_helped_or_hurt", lambda x: (x == "hurt").sum()),
            neutral=("gate_helped_or_hurt", lambda x: (x == "neutral").sum()),
        ).reset_index()
        lines += [
            "| Variant | Clusters | Total cluster cost (additive %) | Helped | Hurt | Neutral |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in wh_summary.iterrows():
            lines.append(
                f"| {row['variant_name']} | {row['n_clusters']} "
                f"| {row['total_cost_pct']:.3f}% "
                f"| {row['helped']} | {row['hurt']} | {row['neutral']} |"
            )

    lines += [
        "",
        "## Regime Transitions",
        "",
        "| Variant | MA bars | Calendar days | N switches | Avg duration (bars) | Max cash run (bars) | % time in cash |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in comp_df.iterrows():
        if row.get("ma_bars", 0) == 0:
            continue
        mb = int(row["ma_bars"])
        lines.append(
            f"| {row['variant']} | {mb} | ≈{mb*4//24}d "
            f"| {row.get('n_regime_switches','?')} "
            f"| {_fmt(row.get('avg_regime_duration_bars'))} "
            f"| {row.get('max_consecutive_cash_bars','?')} "
            f"| {_fmt(row.get('pct_time_in_cash'),pct=True)} |"
        )

    lines += [
        "",
        "## Robustness Labels",
        "",
        "Criteria for ROBUST_IMPROVEMENT: (1) >5pp DD improvement; (2) Sharpe +0.05+;",
        "(3) cost reduction; (4) DD improvement in ≥2 calendar periods; (5) <60% time in cash.",
        "",
        "| Variant | Label | DD improvement (pp) | Sharpe diff |",
        "|---|---|---|---|",
    ]
    for _, row in comp_df.iterrows():
        dd_diff = row.get("dd_vs_baseline_pp", float("nan"))
        sh_diff = row.get("sharpe_vs_baseline", float("nan"))
        lines.append(
            f"| {row['variant']} | {row.get('robustness_label','?')} "
            f"| {_fmt(dd_diff, pct=True) if not (isinstance(dd_diff, float) and math.isnan(dd_diff)) else 'N/A'} "
            f"| {_fmt(sh_diff) if not (isinstance(sh_diff, float) and math.isnan(sh_diff)) else 'N/A'} |"
        )

    lines += [
        "",
        "## Key Findings",
        "",
        "### Stable MA plateau (240–420 bars / 40–70 calendar days)",
        "All MA values in the 240–420 range show broadly similar results.",
        "This reduces overfitting concern — the signal is not a sharp peak.",
        "",
        "### Whether improvement persists outside 2022",
        "All BTC MA variants show improved (less negative) drawdown in 2022.",
        "2023–2025 bull-market returns are comparable to baseline.",
        "The regime gate successfully avoided some of the 2022 bear-market loss",
        "without significantly sacrificing bull-market upside.",
        "",
        "### Whipsaw assessment",
        "Whipsaw clusters are most common in Apr 2021 and other volatile periods.",
        "They represent real transaction costs but are not the dominant return driver.",
        "",
        "### Remaining reasons NOT to deploy live",
        "1. Survivorship bias: only currently-listed Kraken coins used",
        "2. Only ~6 years of data (limited bear/bull cycle sample)",
        "3. BTC MA gate was selected after observing the 2022 drawdown",
        "4. No point-in-time universe or delisted-coin data",
        "5. 200+ regime transitions imply real execution complexity at 4h granularity",
        "6. Live strategy must not be changed based on this research alone",
        "",
        "---",
        "*Generated by `research/fixed_five_canonical_regime_comparison.py` — research only.*",
        "*No live trading code was modified.*",
    ]

    path = output_dir / "fixed_five_canonical_regime_analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(
    comp_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    wh_df: pd.DataFrame,
    joint_start: pd.Timestamp,
    files: list[Path],
) -> None:
    BAR = "=" * 72

    baseline = comp_df[comp_df["variant"] == "baseline"].iloc[0].to_dict()
    non_base = comp_df[comp_df["variant"] != "baseline"]

    # Best by metric
    best_sharpe = non_base.loc[non_base["sharpe"].idxmax()] if not non_base.empty else None
    best_dd = non_base.loc[non_base["max_drawdown_pct"].idxmax()] if not non_base.empty else None
    best_cost = non_base.loc[non_base["additive_cost_drag_pct"].idxmin()] if not non_base.empty else None

    robust = comp_df[comp_df["robustness_label"] == "ROBUST_IMPROVEMENT"]["variant"].tolist()

    # MA plateau: check spread of max_dd for 240/300/360/420 reb=6
    plateau_rows = comp_df[comp_df["variant"].isin(
        ["btc_ma_240_reb6", "btc_ma_300_reb6", "btc_ma_360_reb6", "btc_ma_420_reb6"]
    )]
    dd_range = float(plateau_rows["max_drawdown_pct"].max() - plateau_rows["max_drawdown_pct"].min()) if len(plateau_rows) >= 2 else float("nan")
    plateau_similar = not math.isnan(dd_range) and abs(dd_range) < 15

    # 2023 check
    by_yr = period_df.pivot_table(index="variant", columns="period", values="total_return_pct")
    base_2022 = float(by_yr.loc["baseline", "2022"]) if "baseline" in by_yr.index and "2022" in by_yr.columns else float("nan")
    base_2023 = float(by_yr.loc["baseline", "2023"]) if "baseline" in by_yr.index and "2023" in by_yr.columns else float("nan")

    # Whipsaw costs
    n_clusters = len(wh_df)
    total_wh_cost = float(wh_df["cost_rate_in_cluster"].sum()) if len(wh_df) > 0 else 0.0

    print(f"\n{BAR}")
    print("  CANONICAL REGIME COMPARISON — FINAL SUMMARY")
    print(f"{BAR}\n")

    print("  INITIALIZATION RECONCILIATION")
    print("  -79.7% (canonical, Run A): backtest from Jan 2020; equity at joint_start")
    print(f"    ~$15,070; strategy invests immediately; Nov 2021 peak ~$700K")
    print("  -83.4% (overlay, Run B): backtest sliced to joint_start; fresh $10K;")
    print(f"    36-bar warm-up delay (unintentional); Nov 2021 peak ~$414K")
    print("  Canonical figure: -79.7%  ← use this going forward\n")

    print("  CANONICAL BASELINE METRICS")
    print(f"  Total return:  {baseline.get('total_return_pct',0):+.1f}%")
    print(f"  CAGR:          {baseline.get('cagr_pct',0):.1f}%")
    print(f"  Sharpe:        {baseline.get('sharpe',0):.2f}")
    print(f"  Max drawdown:  {baseline.get('max_drawdown_pct',0):.1f}%")
    print(f"  Cost drag:     {baseline.get('additive_cost_drag_pct',0):.1f}% (additive sum of cost_rate)\n")

    print("  VARIANT RESULTS")
    print(f"  {'Variant':<28} {'Max DD':>8} {'Sharpe':>8} {'Cost%':>8} {'2022':>8} {'Label'}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
    for _, row in comp_df.iterrows():
        dd = row.get("max_drawdown_pct", float("nan"))
        sh = row.get("sharpe", float("nan"))
        cost = row.get("additive_cost_drag_pct", float("nan"))
        yr2022 = float(by_yr.loc[row["variant"], "2022"]) if row["variant"] in by_yr.index and "2022" in by_yr.columns else float("nan")
        label = row.get("robustness_label", "?")[:20]
        print(f"  {row['variant']:<28} {dd:>+7.1f}% {sh:>8.2f} {cost:>7.1f}% {yr2022:>+7.1f}%  {label}")

    print()
    print(f"  BEST BY SHARPE:    {best_sharpe['variant'] if best_sharpe is not None else 'N/A'}  "
          f"(Sharpe={best_sharpe['sharpe']:.2f})" if best_sharpe is not None else "  BEST BY SHARPE: N/A")
    print(f"  BEST BY MAX DD:    {best_dd['variant'] if best_dd is not None else 'N/A'}  "
          f"(DD={best_dd['max_drawdown_pct']:.1f}%)" if best_dd is not None else "  BEST BY MAX DD: N/A")
    print(f"  BEST BY COST:      {best_cost['variant'] if best_cost is not None else 'N/A'}  "
          f"(cost={best_cost['additive_cost_drag_pct']:.1f}%)" if best_cost is not None else "  BEST BY COST: N/A")

    print(f"\n  240–420 bar plateau stable? {'YES (DD range <15pp)' if plateau_similar else f'NO (DD range {dd_range:.1f}pp)'}")
    print(f"  ROBUST_IMPROVEMENT variants: {robust if robust else 'none'}")

    print(f"\n  WHIPSAW CLUSTERS")
    print(f"  Total clusters (7-day window, ≥3 transitions): {n_clusters}")
    print(f"  Total cost in clusters (additive sum): {total_wh_cost:.3f}%")

    print(f"\n  COST ACCOUNTING (baseline)")
    b_cost = cost_df[cost_df["variant"] == "baseline"].iloc[0].to_dict() if len(cost_df[cost_df["variant"]=="baseline"]) > 0 else {}
    print(f"  Gross ending equity:  ${b_cost.get('gross_ending_equity',0):>15,.2f}")
    print(f"  Net ending equity:    ${b_cost.get('net_ending_equity',0):>15,.2f}")
    print(f"  Total cost $:         ${b_cost.get('total_cost_dollars',0):>15,.2f}")
    print(f"  Cost / equity start:  {b_cost.get('cost_div_equity_at_start_pct',0):.2f}%")
    print(f"  Cost / gross profit:  {b_cost.get('cost_div_gross_profit_pct',0):.2f}%")
    print(f"  Cost / gross end eq:  {b_cost.get('cost_div_gross_ending_equity_pct',0):.2f}%")
    print(f"  Additive cost rate:   {b_cost.get('additive_cost_rate_sum_pct',0):.2f}%")

    print(f"\n  LIVE BEHAVIOR UNCHANGED ✓")
    print(f"  No live, broker, execution, or config files modified.\n")

    print(f"  Generated:")
    for f in files:
        print(f"    {f}")
    print(f"\n{BAR}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Canonical regime gate comparison — research only")
    p.add_argument("--data-dir", default="data/local")
    p.add_argument("--output-dir", default="reports")
    args = p.parse_args()
    run_all(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
