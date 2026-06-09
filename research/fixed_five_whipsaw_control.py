"""Research-only regime-whipsaw reduction study for BTC MA-240 / reb-12 candidate.

PURPOSE
-------
Test predefined whipsaw-control rules on top of the btc_ma_240_reb12 variant
to determine whether any rule materially reduces regime-switching noise without
materially worsening drawdown protection or Sharpe ratio.

CANONICAL CONTROL
-----------------
BTC MA-240bars (≈40 calendar days), rebalance every 12 four-hour bars.
All other parameters identical to the canonical baseline:
  - FIXED_COMMON_HISTORY universe (BTC/ETH/XRP/SOL/AVAX)
  - joint_start: 2020-09-28 00:00 UTC
  - fee_bps=10, slippage_bps=5, one-bar execution delay
  - initial_capital=$10,000, backtest starts from full data history

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_whipsaw_control
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
    DEFAULT_REBALANCE_BARS,
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
    _max_consecutive_cash,
    _period_rebalance_log,
    _period_return,
    _sharpe,
    _sortino,
    _worst_calendar_year,
    _worst_month_return,
)
from research.fixed_five_canonical_regime_comparison import (
    run_canonical,
    verify_canonical_initialization,
    _btc_benchmark_period,
    _ewb_benchmark_period,
    PERIODS,
)

LOGGER = logging.getLogger(__name__)

BTC_COL = "BTC/USD"
BARS_PER_YEAR = BARS_PER_YEAR_4H  # 2190

# Frozen candidate parameters
CONTROL_MA_BARS = 240
CONTROL_REBALANCE_BARS = 12

# MA label for reports
CONTROL_MA_LABEL = f"MA-{CONTROL_MA_BARS}bars(≈{CONTROL_MA_BARS * 4 // 24}days)"


# ---------------------------------------------------------------------------
# Whipsaw control parameters
# ---------------------------------------------------------------------------

@dataclass
class WhipsawControl:
    """Parameters for one whipsaw-control variant."""
    name: str
    entry_confirm_bars: int = 1    # Consecutive bars above MA needed to enter (1=immediate)
    exit_confirm_bars: int = 1     # Consecutive bars below MA needed to exit (1=immediate)
    entry_buffer_pct: float = 0.0  # Enter only when BTC > MA × (1 + buf/100)
    exit_buffer_pct: float = 0.0   # Exit only when BTC < MA × (1 - buf/100)
    min_duration_bars: int = 0     # Min bars between transitions (0=no lock)
    description: str = ""

    def __post_init__(self) -> None:
        if self.entry_confirm_bars < 1:
            raise ValueError("entry_confirm_bars must be >= 1")
        if self.exit_confirm_bars < 1:
            raise ValueError("exit_confirm_bars must be >= 1")
        if self.min_duration_bars < 0:
            raise ValueError("min_duration_bars must be >= 0")


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: list[WhipsawControl] = [
    # ---- Control (raw MA, no whipsaw reduction) ----------------------------
    WhipsawControl(
        name="control",
        description="Raw MA-240 gate, no whipsaw controls. Identical to btc_ma_240_reb12.",
    ),

    # ---- Part A: Entry confirmation ----------------------------------------
    WhipsawControl(
        name="entry_conf_2bar",
        entry_confirm_bars=2,
        description="Enter after 2 consecutive bars above MA; exit immediately.",
    ),
    WhipsawControl(
        name="entry_conf_3bar",
        entry_confirm_bars=3,
        description="Enter after 3 consecutive bars above MA; exit immediately.",
    ),

    # ---- Part B: Exit confirmation -----------------------------------------
    WhipsawControl(
        name="exit_conf_2bar",
        exit_confirm_bars=2,
        description="Enter immediately; exit after 2 consecutive bars below MA.",
    ),
    WhipsawControl(
        name="exit_conf_3bar",
        exit_confirm_bars=3,
        description="Enter immediately; exit after 3 consecutive bars below MA.",
    ),

    # ---- Part C: Symmetric confirmation ------------------------------------
    WhipsawControl(
        name="sym_conf_2bar",
        entry_confirm_bars=2,
        exit_confirm_bars=2,
        description="2-bar symmetric confirmation for both entry and exit.",
    ),
    WhipsawControl(
        name="sym_conf_3bar",
        entry_confirm_bars=3,
        exit_confirm_bars=3,
        description="3-bar symmetric confirmation for both entry and exit.",
    ),

    # ---- Part D: Hysteresis ------------------------------------------------
    WhipsawControl(
        name="hyst_05_05",
        entry_buffer_pct=0.5,
        exit_buffer_pct=0.5,
        description="Enter above MA+0.5%; exit below MA-0.5%.",
    ),
    WhipsawControl(
        name="hyst_10_05",
        entry_buffer_pct=1.0,
        exit_buffer_pct=0.5,
        description="Enter above MA+1.0%; exit below MA-0.5%.",
    ),
    WhipsawControl(
        name="hyst_10_10",
        entry_buffer_pct=1.0,
        exit_buffer_pct=1.0,
        description="Enter above MA+1.0%; exit below MA-1.0%.",
    ),

    # ---- Part E: Minimum regime duration -----------------------------------
    WhipsawControl(
        name="min_dur_3bar",
        min_duration_bars=3,
        description="Min 3 bars between regime transitions.",
    ),
    WhipsawControl(
        name="min_dur_6bar",
        min_duration_bars=6,
        description="Min 6 bars between regime transitions.",
    ),
    WhipsawControl(
        name="min_dur_12bar",
        min_duration_bars=12,
        description="Min 12 bars between regime transitions.",
    ),

    # ---- Part 3 combinations -----------------------------------------------
    WhipsawControl(
        name="combo_sym2_hyst05",
        entry_confirm_bars=2,
        exit_confirm_bars=2,
        entry_buffer_pct=0.5,
        exit_buffer_pct=0.5,
        description="2-bar symmetric confirmation plus 0.5% hysteresis band.",
    ),
    WhipsawControl(
        name="combo_sym2_mindur6",
        entry_confirm_bars=2,
        exit_confirm_bars=2,
        min_duration_bars=6,
        description="2-bar symmetric confirmation plus 6-bar minimum duration.",
    ),
    WhipsawControl(
        name="combo_hyst05_mindur6",
        entry_buffer_pct=0.5,
        exit_buffer_pct=0.5,
        min_duration_bars=6,
        description="0.5% hysteresis band plus 6-bar minimum duration.",
    ),
    WhipsawControl(
        name="combo_entry2_imm_buf05",
        entry_confirm_bars=2,
        exit_confirm_bars=1,
        entry_buffer_pct=0.5,
        exit_buffer_pct=0.0,
        description="2-bar entry confirmation plus immediate exit plus 0.5% entry buffer.",
    ),
]


# ---------------------------------------------------------------------------
# Controlled gate computation (bar-by-bar, no look-ahead)
# ---------------------------------------------------------------------------

def compute_controlled_gate_series(
    btc_close: pd.Series,
    ma_bars: int,
    params: WhipsawControl,
) -> pd.Series:
    """Compute the controlled gate series bar-by-bar.

    The gate starts as False (risk-off). Uses only information available
    at each bar (no future data). Returns a boolean Series indexed like btc_close.

    State machine:
    - When risk-off: count consecutive bars where BTC > MA × (1 + entry_buf).
      After entry_confirm_bars consecutive confirming bars → transition to risk-on.
    - When risk-on: count consecutive bars where BTC < MA × (1 - exit_buf).
      After exit_confirm_bars consecutive confirming bars → transition to risk-off.
    - After any transition, lock for min_duration_bars bars.
      A transition at bar i: next transition allowed only at bar i + min_duration_bars.
    - Consecutive counter resets whenever the raw signal breaks.
    """
    btc = btc_close.dropna()
    ma = btc.rolling(ma_bars, min_periods=ma_bars).mean()

    entry_confirm = params.entry_confirm_bars
    exit_confirm = params.exit_confirm_bars
    entry_buf = params.entry_buffer_pct
    exit_buf = params.exit_buffer_pct
    min_dur = params.min_duration_bars

    gate_state = False
    confirming_count = 0
    last_transition_bar = -(min_dur + 1)  # Allow immediate first entry

    gate_values = np.zeros(len(btc), dtype=bool)

    for i, (ts, btc_val) in enumerate(btc.items()):
        ma_val = ma.iloc[i]

        if math.isnan(ma_val) or ma_val <= 0 or math.isnan(btc_val):
            gate_values[i] = gate_state
            continue

        locked = (i - last_transition_bar) < min_dur

        if not locked:
            if not gate_state:
                # Risk-off: checking for entry
                if btc_val > ma_val * (1.0 + entry_buf / 100.0):
                    confirming_count += 1
                    if confirming_count >= entry_confirm:
                        gate_state = True
                        last_transition_bar = i
                        confirming_count = 0
                else:
                    confirming_count = 0
            else:
                # Risk-on: checking for exit
                if btc_val < ma_val * (1.0 - exit_buf / 100.0):
                    confirming_count += 1
                    if confirming_count >= exit_confirm:
                        gate_state = False
                        last_transition_bar = i
                        confirming_count = 0
                else:
                    confirming_count = 0

        gate_values[i] = gate_state

    return pd.Series(gate_values, index=btc.index, name="gate")


# ---------------------------------------------------------------------------
# Signal generator using precomputed gate
# ---------------------------------------------------------------------------

class WhipsawControlledSignalGen:
    """BTC MA gate with whipsaw controls. Gate is precomputed from full history."""

    def __init__(self, gate_series: pd.Series, base_gen: BaseSignalGen) -> None:
        self._gate = gate_series
        self._base = base_gen

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        try:
            gate_val = bool(self._gate.get(ts, False))
            if gate_val:
                return self._base(close, ts)
            return pd.Series(0.0, index=close.columns)
        except Exception:
            return pd.Series(0.0, index=close.columns)


# ---------------------------------------------------------------------------
# Gate transition event analysis
# ---------------------------------------------------------------------------

@dataclass
class GateTransitionEvent:
    variant_name: str
    transition_type: str          # "entry" or "exit"
    raw_transition_ts: str
    controlled_transition_ts: str
    delay_bars: int
    delay_hours: float
    btc_return_during_delay_pct: float
    ewb_return_during_delay_pct: float
    notes: str


def build_gate_transition_events(
    variant_name: str,
    raw_gate: pd.Series,
    controlled_gate: pd.Series,
    close: pd.DataFrame,
) -> list[GateTransitionEvent]:
    """Compare raw and controlled gate transitions. Compute delay and missed/extra returns.

    For each controlled transition, find the nearest raw transition of the same type
    that preceded it. Delay = controlled_ts - raw_ts.

    Returns list of GateTransitionEvent records.
    """
    btc = close[BTC_COL].dropna()

    # Align to common index
    common_idx = raw_gate.index.intersection(controlled_gate.index)
    rg = raw_gate.reindex(common_idx).fillna(False)
    cg = controlled_gate.reindex(common_idx).fillna(False)

    # Raw transitions: (index, "entry"|"exit")
    raw_entries = rg[(rg) & (~rg.shift(1).fillna(False))].index
    raw_exits = rg[(~rg) & (rg.shift(1).fillna(False))].index

    # Controlled transitions
    ctrl_entries = cg[(cg) & (~cg.shift(1).fillna(False))].index
    ctrl_exits = cg[(~cg) & (cg.shift(1).fillna(False))].index

    events: list[GateTransitionEvent] = []

    def _find_preceding_raw(raw_times, ctrl_time):
        """Last raw transition that occurred at or before ctrl_time."""
        candidates = raw_times[raw_times <= ctrl_time]
        return candidates[-1] if len(candidates) > 0 else None

    def _compute_returns(start_ts, end_ts):
        if start_ts == end_ts or start_ts is None or end_ts is None:
            return 0.0, 0.0
        btc_seg = btc.loc[(btc.index >= start_ts) & (btc.index <= end_ts)]
        all_seg = close.loc[(close.index >= start_ts) & (close.index <= end_ts)]
        btc_ret = float(btc_seg.iloc[-1] / btc_seg.iloc[0] - 1) * 100 if len(btc_seg) >= 2 else 0.0
        ewb_ret = float(((1 + all_seg.pct_change().fillna(0)).prod(axis=0).mean()) - 1) * 100 if len(all_seg) >= 2 else 0.0
        return btc_ret, ewb_ret

    for ctrl_ts in ctrl_entries:
        raw_ts = _find_preceding_raw(raw_entries, ctrl_ts)
        if raw_ts is None:
            continue
        delay = len(common_idx[(common_idx >= raw_ts) & (common_idx < ctrl_ts)])
        delay_hrs = delay * 4.0
        btc_ret, ewb_ret = _compute_returns(raw_ts, ctrl_ts)
        events.append(GateTransitionEvent(
            variant_name=variant_name,
            transition_type="entry",
            raw_transition_ts=str(raw_ts),
            controlled_transition_ts=str(ctrl_ts),
            delay_bars=delay,
            delay_hours=delay_hrs,
            btc_return_during_delay_pct=round(btc_ret, 4),
            ewb_return_during_delay_pct=round(ewb_ret, 4),
            notes=f"entry delayed {delay} bars ({delay_hrs:.1f}h)",
        ))

    for ctrl_ts in ctrl_exits:
        raw_ts = _find_preceding_raw(raw_exits, ctrl_ts)
        if raw_ts is None:
            continue
        delay = len(common_idx[(common_idx >= raw_ts) & (common_idx < ctrl_ts)])
        delay_hrs = delay * 4.0
        btc_ret, ewb_ret = _compute_returns(raw_ts, ctrl_ts)
        events.append(GateTransitionEvent(
            variant_name=variant_name,
            transition_type="exit",
            raw_transition_ts=str(raw_ts),
            controlled_transition_ts=str(ctrl_ts),
            delay_bars=delay,
            delay_hours=delay_hrs,
            btc_return_during_delay_pct=round(btc_ret, 4),
            ewb_return_during_delay_pct=round(ewb_ret, 4),
            notes=f"exit delayed {delay} bars ({delay_hrs:.1f}h)",
        ))

    return events


# ---------------------------------------------------------------------------
# Cluster counting for controlled gate
# ---------------------------------------------------------------------------

def count_whipsaw_clusters(
    gate: pd.Series,
    window_days: int,
    min_transitions: int = 3,
) -> int:
    """Count clusters where >= min_transitions gate transitions occur within window_days.

    Uses a sliding-window approach identical to the canonical comparison script.
    """
    transitions = gate[gate != gate.shift(1)].dropna()
    trans_idx = transitions.index.tolist()
    if not trans_idx:
        return 0

    cluster_count = 0
    i = 0
    while i < len(trans_idx):
        window_end = trans_idx[i] + pd.Timedelta(days=window_days)
        in_window = [t for t in trans_idx[i:] if t <= window_end]
        if len(in_window) >= min_transitions:
            cluster_count += 1
            i += len(in_window)
        else:
            i += 1

    return cluster_count


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

def compute_full_metrics(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    controlled_gate: pd.Series,
    params: WhipsawControl,
    control_metrics: dict | None = None,
) -> dict[str, Any]:
    """Compute all full-period metrics for one whipsaw control variant."""
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
    n_trades = int((reb_fx["turnover"] > 0.01).sum())
    years = (end_ts - joint_start).days / 365.25
    ann_turn = float(reb_fx["turnover"].sum()) / max(years, 0.001)
    cost_drag_additive = float(reb_fx["cost_rate"].sum() * 100)

    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)
    pct_cash = float((holdings.sum(axis=1) < 0.01).mean() * 100)

    # Regime stats from controlled gate (sliced to joint_start)
    gate_fx = controlled_gate.reindex(port_fx.index).fillna(False)
    transitions = gate_fx[gate_fx != gate_fx.shift(1)].dropna()
    n_switches = len(transitions)
    n_3day = count_whipsaw_clusters(gate_fx, window_days=3)
    n_7day = count_whipsaw_clusters(gate_fx, window_days=7)

    # Benchmarks
    btc_ret = _btc_benchmark_period(close, joint_start, end_ts)
    ewb_ret = _ewb_benchmark_period(close, joint_start, end_ts)

    # 2022 return from by-period slice
    eq_2022 = eq.loc[(eq.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                     (eq.index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))]
    ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1) * 100 if len(eq_2022) >= 2 else float("nan")

    m: dict[str, Any] = {
        "variant": params.name,
        "description": params.description,
        "entry_confirm_bars": params.entry_confirm_bars,
        "exit_confirm_bars": params.exit_confirm_bars,
        "entry_buffer_pct": params.entry_buffer_pct,
        "exit_buffer_pct": params.exit_buffer_pct,
        "min_duration_bars": params.min_duration_bars,
        "total_return_pct": round(v_total_ret * 100, 2),
        "cagr_pct": round(v_cagr * 100, 2),
        "sharpe": round(v_sharpe, 4),
        "sortino": round(v_sortino, 4) if not math.isnan(v_sortino) else float("nan"),
        "max_drawdown_pct": round(v_maxdd * 100, 2),
        "calmar": round(calmar, 4) if not math.isnan(calmar) else float("nan"),
        "worst_year_return_pct": round(worst_yr * 100, 2),
        "worst_month_return_pct": round(worst_mo * 100, 2),
        "return_2022_pct": round(ret_2022, 2) if not math.isnan(ret_2022) else float("nan"),
        "n_trades": n_trades,
        "ann_turnover": round(ann_turn, 4),
        "additive_cost_drag_pct": round(cost_drag_additive, 4),
        "pct_time_invested": round(pct_invested, 2),
        "pct_time_in_cash": round(pct_cash, 2),
        "n_regime_switches": n_switches,
        "n_3day_whipsaw_clusters": n_3day,
        "n_7day_whipsaw_clusters": n_7day,
        "btc_benchmark_pct": round(btc_ret * 100, 2) if not math.isnan(btc_ret) else float("nan"),
        "ewb_benchmark_pct": round(ewb_ret * 100, 2) if not math.isnan(ewb_ret) else float("nan"),
    }

    if control_metrics:
        m["dd_vs_control_pp"] = round(v_maxdd * 100 - control_metrics["max_drawdown_pct"], 2)
        m["sharpe_vs_control"] = round(v_sharpe - control_metrics["sharpe"], 4)
        m["clusters_7day_vs_control"] = n_7day - control_metrics["n_7day_whipsaw_clusters"]
        m["clusters_3day_vs_control"] = n_3day - control_metrics["n_3day_whipsaw_clusters"]
        m["cost_vs_control_pp"] = round(
            cost_drag_additive - control_metrics["additive_cost_drag_pct"], 4
        )

    return m


def compute_period_metrics(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    controlled_gate: pd.Series,
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

    # Regime transitions in period
    gate_fx = controlled_gate.reindex(
        controlled_gate.index[(controlled_gate.index >= p_start) & (controlled_gate.index <= p_end)]
    ).fillna(False)
    n_trans = int(len(gate_fx[gate_fx != gate_fx.shift(1)].dropna()))
    n_3day = count_whipsaw_clusters(gate_fx, window_days=3)
    n_7day = count_whipsaw_clusters(gate_fx, window_days=7)

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
        "n_regime_transitions": n_trans,
        "n_3day_whipsaw_clusters": n_3day,
        "n_7day_whipsaw_clusters": n_7day,
        "btc_benchmark_pct": round(btc_ret * 100, 2) if not math.isnan(btc_ret) else float("nan"),
        "ewb_benchmark_pct": round(ewb_ret * 100, 2) if not math.isnan(ewb_ret) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cost accounting (distinct denominators and cost categories)
# ---------------------------------------------------------------------------

def compute_cost_accounting(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    params: WhipsawControl,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Detailed cost accounting with explicitly distinct denominators.

    Distinguishes:
    1. fee_dollars_est     – portion of gross-minus-net attributable to fees
    2. slippage_dollars_est – portion attributable to slippage
    3. additive_cost_rate_sum_pct – sum of (turnover × bps/10000); additive, not compounded
    4. gross_minus_net_ending – actual wealth reduction from all costs (incl. compounding)
    5. foregone_compounding – gross_minus_net minus simple-additive cost estimate

    Note: gross_minus_net > simple_additive because costs reduce future compounding base.
    """
    eq = port_fx["equity"].dropna()
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    gross_eq = eq.iloc[0] * (1 + gross_r).cumprod()

    gross_end = float(gross_eq.iloc[-1])
    net_end = float(eq.iloc[-1])
    equity_at_start = float(eq.iloc[0])
    gross_minus_net = gross_end - net_end
    gross_profit = gross_end - equity_at_start

    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= eq.index[0]
    ]
    additive_cost_rate_sum = float(reb_fx["cost_rate"].sum())
    n_reb = len(reb_fx)

    total_bps = fee_bps + slippage_bps
    fee_frac = fee_bps / total_bps if total_bps > 0 else 0.5
    slip_frac = slippage_bps / total_bps if total_bps > 0 else 0.5

    # Simple additive cost estimate (no compounding effect)
    simple_cost_estimate = equity_at_start * additive_cost_rate_sum
    foregone_compounding = gross_minus_net - simple_cost_estimate

    return {
        "variant": params.name,
        # Equity levels
        "equity_at_start": round(equity_at_start, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        "gross_total_return_pct": round((gross_end / equity_at_start - 1) * 100, 2),
        "net_total_return_pct": round((net_end / equity_at_start - 1) * 100, 2),
        # Cost decomposition — dollar amounts
        "gross_minus_net_dollars": round(gross_minus_net, 2),
        "fee_dollars_est": round(gross_minus_net * fee_frac, 2),
        "slippage_dollars_est": round(gross_minus_net * slip_frac, 2),
        # Additive cost rate sum (sum of per-rebalance cost_rate; NOT compounded)
        "additive_cost_rate_sum_pct": round(additive_cost_rate_sum * 100, 4),
        # Simple additive estimate vs actual gross-minus-net
        "simple_cost_estimate_dollars": round(simple_cost_estimate, 2),
        "foregone_compounding_dollars": round(foregone_compounding, 2),
        # Ratios (clearly named by denominator)
        "cost_div_equity_at_start_pct": round(gross_minus_net / equity_at_start * 100, 4),
        "cost_div_gross_profit_pct": round(gross_minus_net / max(gross_profit, 1) * 100, 4),
        "cost_div_gross_ending_equity_pct": round(gross_minus_net / max(gross_end, 1) * 100, 4),
        # Trading stats
        "n_rebalances": n_reb,
    }


# ---------------------------------------------------------------------------
# WHIPSAW_IMPROVEMENT classification
# ---------------------------------------------------------------------------

def classify_whipsaw_improvement(
    m: dict[str, Any],
    control_metrics: dict[str, Any],
) -> tuple[str, str]:
    """Apply PART 5 strict WHIPSAW_IMPROVEMENT criteria.

    WHIPSAW_IMPROVEMENT requires ALL 8 criteria:
    1. Reduces 7-day clusters by >= 25%
    2. Reduces direct transaction costs (additive cost drag)
    3. Sharpe within -0.10 of control or better
    4. Max drawdown does not worsen by more than 5pp
    5. Does not materially worsen 2022 protection (within 5pp)
    6. Not simply staying in cash >80% of time
    7. One-bar delay preserved (structural guarantee)
    8. No future information (structural guarantee)
    """
    ctrl_clusters = control_metrics.get("n_7day_whipsaw_clusters", 1)
    v_clusters = m.get("n_7day_whipsaw_clusters", ctrl_clusters)
    cluster_reduction_pct = (ctrl_clusters - v_clusters) / max(ctrl_clusters, 1) * 100

    ctrl_cost = control_metrics.get("additive_cost_drag_pct", 0)
    v_cost = m.get("additive_cost_drag_pct", ctrl_cost)
    cost_reduction = ctrl_cost - v_cost  # positive = better

    ctrl_sharpe = control_metrics.get("sharpe", 0)
    v_sharpe = m.get("sharpe", 0)
    sharpe_diff = v_sharpe - ctrl_sharpe

    ctrl_dd = control_metrics.get("max_drawdown_pct", -100)
    v_dd = m.get("max_drawdown_pct", -100)
    dd_diff = v_dd - ctrl_dd  # positive = better (less negative)

    ctrl_2022 = control_metrics.get("return_2022_pct", 0)
    v_2022 = m.get("return_2022_pct", ctrl_2022)
    r2022_diff = v_2022 - ctrl_2022  # positive = better

    pct_cash = m.get("pct_time_in_cash", 0)

    notes_parts = [
        f"cluster_reduction={cluster_reduction_pct:+.1f}%",
        f"cost_reduction={cost_reduction:+.2f}pp",
        f"sharpe_diff={sharpe_diff:+.3f}",
        f"dd_diff={dd_diff:+.1f}pp",
        f"r2022_diff={r2022_diff:+.1f}pp",
        f"pct_in_cash={pct_cash:.1f}%",
        f"bias_note={SURVIVORSHIP_BIAS_NOTE[:60]}...",
    ]

    criteria = {
        "cluster_reduction_ge_25pct": cluster_reduction_pct >= 25.0,
        "cost_reduced": cost_reduction > 0,
        "sharpe_within_0_10_of_control": sharpe_diff >= -0.10,
        "dd_not_worse_than_5pp": dd_diff >= -5.0,
        "r2022_not_worse_than_5pp": r2022_diff >= -5.0,
        "not_predominantly_in_cash": pct_cash < 80.0,
    }
    # Criteria 7 (one-bar delay) and 8 (no future info) are structural guarantees.

    all_pass = all(criteria.values())
    label = "WHIPSAW_IMPROVEMENT" if all_pass else "NOT_WHIPSAW_IMPROVEMENT"
    failed = [k for k, v in criteria.items() if not v]
    if failed:
        notes_parts.append(f"failed_criteria={','.join(failed)}")

    return label, "; ".join(notes_parts)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

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
    LOGGER.info("Running %d whipsaw-control variants…", len(VARIANTS))

    all_full_metrics: list[dict] = []
    all_period_metrics: list[dict] = []
    all_cost_rows: list[dict] = []
    all_event_rows: list[dict] = []

    # Precompute raw (control) gate for delay analysis
    control_params = VARIANTS[0]
    raw_gate_full = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, control_params)
    raw_gate_fx = raw_gate_full.reindex(
        raw_gate_full.index[raw_gate_full.index >= joint_start]
    ).fillna(False)

    # Store (result, port_fx, gate) for each variant
    variant_outputs: dict[str, tuple[BacktestResult, pd.DataFrame, pd.Series]] = {}
    control_metrics: dict | None = None

    for params in VARIANTS:
        LOGGER.info("  %s…", params.name)

        # Compute controlled gate from full btc history
        gate_full = compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)
        gate_fx = gate_full.reindex(
            gate_full.index[gate_full.index >= joint_start]
        ).fillna(False)

        base = BaseSignalGen()
        sig_gen = WhipsawControlledSignalGen(gate_full, base)

        result, port_fx = run_canonical(
            ohlcv_full, sig_gen, joint_start,
            rebalance_bars=CONTROL_REBALANCE_BARS,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            fee_bps=DEFAULT_FEE_BPS,
            slippage_bps=DEFAULT_SLIPPAGE_BPS,
        )
        variant_outputs[params.name] = (result, port_fx, gate_fx)

        # Full metrics (control first so comparisons work)
        m = compute_full_metrics(
            result, port_fx, close, joint_start,
            gate_fx, params,
            control_metrics=control_metrics,
        )
        if params.name == "control":
            control_metrics = m.copy()

        all_full_metrics.append(m)

        # Per-period metrics
        for period_name, ps, pe in PERIODS:
            pm = compute_period_metrics(
                result, port_fx, close, joint_start,
                gate_fx, params.name, period_name, ps, pe,
            )
            all_period_metrics.append(pm)

        # Cost accounting
        cost_row = compute_cost_accounting(
            result, port_fx, params, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
        )
        all_cost_rows.append(cost_row)

        # Transition events (delay analysis vs raw control gate)
        events = build_gate_transition_events(
            params.name, raw_gate_fx, gate_fx, close
        )
        all_event_rows.extend([asdict(e) for e in events])

    # Add WHIPSAW_IMPROVEMENT labels
    for m in all_full_metrics:
        if m["variant"] == "control":
            m["whipsaw_label"] = "CONTROL"
            m["whipsaw_notes"] = "canonical btc_ma_240_reb12 reference"
            continue
        label, notes = classify_whipsaw_improvement(m, control_metrics)
        m["whipsaw_label"] = label
        m["whipsaw_notes"] = notes

    # Compute delay summary stats per variant
    event_df = pd.DataFrame(all_event_rows) if all_event_rows else pd.DataFrame()
    if not event_df.empty:
        for m in all_full_metrics:
            v = m["variant"]
            v_events = event_df[event_df["variant_name"] == v]
            entries = v_events[v_events["transition_type"] == "entry"]
            exits = v_events[v_events["transition_type"] == "exit"]
            m["avg_entry_delay_bars"] = round(float(entries["delay_bars"].mean()), 2) if len(entries) > 0 else 0.0
            m["avg_exit_delay_bars"] = round(float(exits["delay_bars"].mean()), 2) if len(exits) > 0 else 0.0
            m["avg_btc_return_missed_entry_pct"] = round(float(entries["btc_return_during_delay_pct"].mean()), 4) if len(entries) > 0 else 0.0
            m["avg_ewb_return_extra_loss_exit_pct"] = round(float(exits["ewb_return_during_delay_pct"].mean()), 4) if len(exits) > 0 else 0.0

    # Write outputs
    comp_df = pd.DataFrame(all_full_metrics)
    comp_path = output_dir / "fixed_five_whipsaw_control_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    period_df = pd.DataFrame(all_period_metrics)
    period_path = output_dir / "fixed_five_whipsaw_control_by_period.csv"
    period_df.to_csv(period_path, index=False)

    cost_df = pd.DataFrame(all_cost_rows)
    cost_path = output_dir / "fixed_five_whipsaw_cost_accounting.csv"
    cost_df.to_csv(cost_path, index=False)

    if not event_df.empty:
        events_path = output_dir / "fixed_five_whipsaw_control_events.csv"
        event_df.to_csv(events_path, index=False)
    else:
        events_path = output_dir / "fixed_five_whipsaw_control_events.csv"
        pd.DataFrame(columns=[f.name for f in GateTransitionEvent.__dataclass_fields__.values()]).to_csv(events_path, index=False)

    # Write analysis markdown
    analysis_path = write_analysis_md(
        comp_df, period_df, cost_df, event_df, close, joint_start, output_dir
    )

    print_summary(comp_df, period_df, cost_df, event_df, joint_start, control_metrics)

    print("\nGenerated:")
    for p in [comp_path, period_path, cost_path, events_path, analysis_path]:
        print(f"  {p}")
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Analysis markdown writer
# ---------------------------------------------------------------------------

def write_analysis_md(
    comp_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    event_df: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    output_dir: Path,
) -> Path:
    lines = [
        "# Regime Whipsaw Reduction Study",
        "",
        "## Control Configuration",
        "",
        f"- Universe: FIXED_COMMON_HISTORY ({', '.join(LIVE_FIVE_UNIVERSE)})",
        f"- Joint start: {joint_start.date()}",
        f"- BTC MA: {CONTROL_MA_LABEL}",
        f"- Rebalance: every {CONTROL_REBALANCE_BARS} bars (48 hours)",
        "- Execution delay: 1 bar (4 hours)",
        f"- Fees: {DEFAULT_FEE_BPS} bps | Slippage: {DEFAULT_SLIPPAGE_BPS} bps",
        f"- ⚠ {SURVIVORSHIP_BIAS_NOTE}",
        "",
        "## Full-Period Results",
        "",
    ]

    if not comp_df.empty:
        cols = ["variant", "total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct",
                "return_2022_pct", "n_7day_whipsaw_clusters", "additive_cost_drag_pct",
                "pct_time_in_cash", "n_regime_switches", "whipsaw_label"]
        display_cols = [c for c in cols if c in comp_df.columns]
        lines.append("")
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
        for _, row in comp_df.iterrows():
            vals = []
            for c in display_cols:
                v = row.get(c, "")
                if isinstance(v, float) and not math.isnan(v):
                    vals.append(f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Cost Accounting Definitions",
        "",
        "The following cost concepts are explicitly distinguished:",
        "",
        "1. **fee_dollars_est** = gross_minus_net × fee_bps / (fee_bps + slippage_bps)",
        "   The estimated dollar amount attributable to exchange fees.",
        "",
        "2. **slippage_dollars_est** = gross_minus_net × slippage_bps / (fee_bps + slippage_bps)",
        "   The estimated dollar amount attributable to modeled slippage.",
        "",
        "3. **additive_cost_rate_sum_pct** = sum(turnover × total_bps / 10000) × 100",
        "   Additive sum of per-rebalance cost rates. Does NOT include compounding effects.",
        "",
        "4. **gross_minus_net_dollars** = gross_ending_equity − net_ending_equity",
        "   Actual dollar wealth reduction from all costs, including foregone compounding.",
        "   This is LARGER than the additive estimate because costs reduce the future compounding base.",
        "",
        "5. **foregone_compounding_dollars** = gross_minus_net − (equity_at_start × additive_rate)",
        "   Additional wealth lost because costs reduced the base for future returns.",
        "   This is NOT fees or slippage directly — it is the compounding penalty.",
        "",
        "Do not describe gross_minus_net as 'fees paid'. It includes slippage and compounding effects.",
        "",
        "## WHIPSAW_IMPROVEMENT Criteria (Part 5)",
        "",
        "A variant receives WHIPSAW_IMPROVEMENT only if ALL criteria are satisfied:",
        "",
        "1. 7-day whipsaw clusters reduced by >= 25% vs control",
        "2. Additive cost drag reduced",
        "3. Sharpe within -0.10 of control (or better)",
        "4. Max drawdown does not worsen by more than 5pp vs control",
        "5. 2022 return does not worsen by more than 5pp vs control",
        "6. Not predominantly in cash (time-in-cash < 80%)",
        "7. One-bar execution delay preserved (structural guarantee)",
        "8. No future information used (structural guarantee)",
        "",
    ]

    if not comp_df.empty:
        wi_count = int((comp_df["whipsaw_label"] == "WHIPSAW_IMPROVEMENT").sum())
        lines += [
            f"**Variants meeting WHIPSAW_IMPROVEMENT: {wi_count} of {len(comp_df) - 1}**",
            "",
        ]

    lines += [
        "## Entry/Exit Delay Analysis",
        "",
        "Delays are measured in 4-hour bars relative to the raw MA crossing.",
        "A delay > 0 means the controlled gate entered/exited later than the raw gate.",
        "",
    ]

    if not event_df.empty:
        for vname in comp_df["variant"].tolist():
            if vname == "control":
                continue
            vev = event_df[event_df["variant_name"] == vname]
            entries = vev[vev["transition_type"] == "entry"]
            exits = vev[vev["transition_type"] == "exit"]
            avg_ent = float(entries["delay_bars"].mean()) if len(entries) > 0 else 0.0
            avg_ext = float(exits["delay_bars"].mean()) if len(exits) > 0 else 0.0
            avg_missed = float(entries["btc_return_during_delay_pct"].mean()) if len(entries) > 0 else 0.0
            avg_extra = float(exits["ewb_return_during_delay_pct"].mean()) if len(exits) > 0 else 0.0
            lines.append(
                f"- **{vname}**: avg entry delay={avg_ent:.1f} bars, "
                f"avg exit delay={avg_ext:.1f} bars, "
                f"avg missed BTC entry return={avg_missed:+.2f}%, "
                f"avg extra EWB exit return={avg_extra:+.2f}%"
            )

    lines += [
        "",
        f"⚠ **Survivorship bias warning**: {SURVIVORSHIP_BIAS_NOTE}",
        "These results use a universe selected from present-day knowledge. "
        "No conclusions should be extrapolated to out-of-sample crypto universe performance.",
    ]

    path = output_dir / "fixed_five_whipsaw_control_analysis.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    comp_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    event_df: pd.DataFrame,
    joint_start: pd.Timestamp,
    control_metrics: dict | None,
) -> None:
    print("\n" + "=" * 72)
    print("REGIME WHIPSAW REDUCTION STUDY — SUMMARY")
    print("=" * 72)

    print(f"\n  Control: {CONTROL_MA_LABEL}, rebalance every {CONTROL_REBALANCE_BARS} bars")
    print(f"  Joint start: {joint_start.date()}")

    if control_metrics:
        print(f"\n  CANONICAL CONTROL (btc_ma_240_reb12) METRICS")
        print(f"    Total return:        {control_metrics['total_return_pct']:>8.1f}%")
        print(f"    CAGR:                {control_metrics['cagr_pct']:>8.1f}%")
        print(f"    Sharpe:              {control_metrics['sharpe']:>8.4f}")
        print(f"    Max drawdown:        {control_metrics['max_drawdown_pct']:>8.1f}%")
        print(f"    2022 return:         {control_metrics['return_2022_pct']:>8.1f}%")
        print(f"    7-day clusters:      {control_metrics['n_7day_whipsaw_clusters']:>8d}")
        print(f"    3-day clusters:      {control_metrics['n_3day_whipsaw_clusters']:>8d}")
        print(f"    Regime switches:     {control_metrics['n_regime_switches']:>8d}")
        print(f"    Additive cost:       {control_metrics['additive_cost_drag_pct']:>8.1f}%")
        print(f"    Time in cash:        {control_metrics['pct_time_in_cash']:>8.1f}%")

    if not comp_df.empty:
        non_ctrl = comp_df[comp_df["variant"] != "control"]

        print(f"\n  VARIANT RESULTS")
        print(f"  {'Variant':<30} {'MaxDD':>7} {'Sharpe':>7} {'Cost%':>7} {'7dClust':>8} "
              f"{'2022':>7} Label")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*24}")
        for _, r in comp_df.iterrows():
            label = r.get("whipsaw_label", "")
            print(
                f"  {r['variant']:<30} "
                f"{r.get('max_drawdown_pct', float('nan')):>7.1f}% "
                f"{r.get('sharpe', float('nan')):>7.3f} "
                f"{r.get('additive_cost_drag_pct', float('nan')):>7.1f}% "
                f"{r.get('n_7day_whipsaw_clusters', 0):>8d} "
                f"{r.get('return_2022_pct', float('nan')):>7.1f}% "
                f"{label}"
            )

        wi = comp_df[comp_df["whipsaw_label"] == "WHIPSAW_IMPROVEMENT"]
        print(f"\n  WHIPSAW_IMPROVEMENT variants: {len(wi)}")
        if len(wi) > 0:
            for _, r in wi.iterrows():
                print(f"    ✓ {r['variant']}")
        else:
            print("    None — no variant met all 8 criteria")

        # Best by individual metrics
        if len(non_ctrl) > 0:
            best_sharpe = non_ctrl.loc[non_ctrl["sharpe"].idxmax()]
            best_dd = non_ctrl.loc[non_ctrl["max_drawdown_pct"].idxmax()]
            best_cost = non_ctrl.loc[non_ctrl["additive_cost_drag_pct"].idxmin()]
            best_clusters = non_ctrl.loc[non_ctrl["n_7day_whipsaw_clusters"].idxmin()]
            print(f"\n  BEST BY INDIVIDUAL METRIC:")
            print(f"    Best Sharpe:      {best_sharpe['variant']} ({best_sharpe['sharpe']:.4f})")
            print(f"    Best max DD:      {best_dd['variant']} ({best_dd['max_drawdown_pct']:.1f}%)")
            print(f"    Best cost:        {best_cost['variant']} ({best_cost['additive_cost_drag_pct']:.1f}%)")
            print(f"    Fewest clusters:  {best_clusters['variant']} ({best_clusters['n_7day_whipsaw_clusters']})")

    if not event_df.empty:
        non_ctrl_ev = event_df[event_df["variant_name"] != "control"]
        entries = non_ctrl_ev[non_ctrl_ev["transition_type"] == "entry"]
        exits = non_ctrl_ev[non_ctrl_ev["transition_type"] == "exit"]
        if len(entries) > 0:
            print(f"\n  ENTRY DELAY SUMMARY (across all non-control variants)")
            print(f"    Avg entry delay:  {float(entries['delay_bars'].mean()):.1f} bars")
            print(f"    Max entry delay:  {int(entries['delay_bars'].max())} bars")
            missed = float(entries["btc_return_during_delay_pct"].mean())
            print(f"    Avg BTC return missed during entry delay: {missed:+.2f}%")
        if len(exits) > 0:
            print(f"\n  EXIT DELAY SUMMARY (across all non-control variants)")
            print(f"    Avg exit delay:   {float(exits['delay_bars'].mean()):.1f} bars")
            print(f"    Max exit delay:   {int(exits['delay_bars'].max())} bars")
            extra = float(exits["ewb_return_during_delay_pct"].mean())
            print(f"    Avg EWB return during delayed exit: {extra:+.2f}%")

    if not cost_df.empty:
        ctrl_cost = cost_df[cost_df["variant"] == "control"]
        if len(ctrl_cost) > 0:
            c = ctrl_cost.iloc[0]
            print(f"\n  COST ACCOUNTING (control)")
            print(f"    Gross ending equity:        ${c['gross_ending_equity']:>14,.0f}")
            print(f"    Net ending equity:          ${c['net_ending_equity']:>14,.0f}")
            print(f"    Gross minus net (all costs):${c['gross_minus_net_dollars']:>14,.0f}")
            print(f"    Fee dollars (est):          ${c['fee_dollars_est']:>14,.0f}")
            print(f"    Slippage dollars (est):     ${c['slippage_dollars_est']:>14,.0f}")
            print(f"    Additive cost rate sum:      {c['additive_cost_rate_sum_pct']:>13.1f}%")
            print(f"    Foregone compounding:       ${c['foregone_compounding_dollars']:>14,.0f}")
            print(f"    Note: gross_minus_net != fees_paid; includes slippage and compounding penalty")

    print("\n  LIVE BEHAVIOR UNCHANGED ✓")
    print("  No live, broker, execution, or config files modified.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regime whipsaw reduction study")
    parser.add_argument("--data-dir", default="data/local")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    run_all(Path(args.data_dir), Path(args.output_dir))
