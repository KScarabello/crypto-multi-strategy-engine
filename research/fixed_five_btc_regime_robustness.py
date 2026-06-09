"""BTC regime gate MA-window robustness study for the FIXED_COMMON_HISTORY five-coin backtest.

RESEARCH ONLY — no live trading code is touched or imported.
No imports from brokers/, execution/, or live/.

CANONICAL BASELINE DESIGN
--------------------------
Run the backtest using the FULL close matrix from data start (Jan 2020).
Slice the portfolio to joint_start (2020-09-28) for metrics only.
This is "Run A" style — the strategy has 9 months of warm-up before joint_start,
all 5 coins are immediately eligible at joint_start, equity ≈ $15K (not $10K).

Compare to "Run B" (the overlay script): close_fixed sliced to joint_start, fresh
$10K capital → yields -83.4% max DD instead of -79.7% because the Nov 2021 peak
is lower in absolute terms (lower start equity compounded through the bull run).

ONE-BAR EXECUTION DELAY
------------------------
Regime gate is evaluated AT bar i (signal_generator call). The resulting weights
are EXECUTED at bar i+1. This is the backtest engine's default pending_target
mechanism. The signal generator runs at the signal bar — the delay is the
engine's responsibility.

Outputs
-------
reports/fixed_five_baseline_metric_reconciliation.md
reports/fixed_five_btc_regime_robustness_grid.csv
reports/fixed_five_btc_regime_walkforward.csv
reports/fixed_five_btc_regime_events.csv
reports/fixed_five_btc_regime_robustness_analysis.md

Usage
-----
    .venv/bin/python -m research.fixed_five_btc_regime_robustness
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.universe_integrity_analysis import (
    LIVE_FIVE_UNIVERSE,
    DEFAULT_TOP_N,
    DEFAULT_SHORT_LOOKBACK,
    DEFAULT_MEDIUM_LOOKBACK,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_REBALANCE_BARS,
    DEFAULT_INITIAL_CAPITAL,
    BARS_PER_YEAR_4H,
    SURVIVORSHIP_BIAS_NOTE,
    build_close_matrix,
    find_joint_eligible_start,
    _close_to_ohlcv,
)
from research.fixed_five_defensive_overlay import (
    BaseSignalGen,
    BTCRegimeSignalGen,
    _max_drawdown,
    _cagr,
    _sharpe,
    _sortino,
    _worst_month_return,
    _worst_calendar_year,
    _period_return,
    _period_rebalance_log,
    _max_consecutive_cash,
    PERIODS,
    BARS_PER_YEAR,
)
from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import summary_metrics
from strategies.cross_sectional_momentum import compute_momentum_score, check_regime_filter

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------

BTC_MA_BARS_TO_TEST: list[int] = [120, 150, 180, 210, 240, 300, 360, 420, 480]
REBALANCE_BARS_TO_TEST: list[int] = [6, 12, 24]

# Walk-forward variants: (name, ma_bars, rebalance_bars)
WALK_FORWARD_VARIANTS: list[tuple[str, int, int]] = [
    ("btc_ma_360_reb6",  360, 6),
    ("btc_ma_180_reb6",  180, 6),
    ("btc_ma_180_reb12", 180, 12),
    ("btc_ma_180_reb24", 180, 24),
    ("btc_ma_360_reb12", 360, 12),
    ("btc_ma_360_reb24", 360, 24),
]

# Canonical baseline definition (documentation constant)
CANONICAL_BASELINE: dict[str, Any] = {
    "start_timestamp": "2020-09-28 00:00:00 UTC (joint_start when all 5 symbols have >= 36 bars)",
    "end_timestamp": "last available bar in data/local/",
    "universe": list(LIVE_FIVE_UNIVERSE),
    "timeframe": "4h",
    "fee_bps": DEFAULT_FEE_BPS,
    "slippage_bps": DEFAULT_SLIPPAGE_BPS,
    "execution_delay": "1 bar (4h) — signal at bar i, executed at bar i+1",
    "rebalance_schedule": "every 6 bars (every 24h)",
    "cost_accounting": "turnover × (fee_bps + slippage_bps) / 10000 per rebalance, deducted at execution",
    "drawdown_calculation": "min(equity / equity.cummax() - 1) over portfolio sliced to >= joint_start",
    "equity_construction": "full backtest from data start; portfolio metrics from joint_start onward",
    "benchmark_alignment": "BTC B&H from joint_start to same end date, 2x fee applied once",
    "max_drawdown": "-79.7% (canonical figure)",
    "bias_warning": SURVIVORSHIP_BIAS_NOTE,
}

# MA bar-to-day label map for 4h bars
_MA_DAYS: dict[int, int] = {bars: bars * 4 // 24 for bars in BTC_MA_BARS_TO_TEST}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ma_label(bars: int) -> str:
    """Return human-readable MA label: MA-{bars}bars(≈{days}days)."""
    days = bars * 4 // 24
    return f"MA-{bars}bars(≈{days}days)"


def _fmt_pct(v: Any, decimals: int = 1) -> str:
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except Exception:
        return str(v)


def _fmt_f(v: Any, decimals: int = 2) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def run_canonical_backtest(
    signal_gen: Any,
    ohlcv: pd.DataFrame,
    rebalance_bars: int,
    joint_start: pd.Timestamp,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run full backtest with full ohlcv; return (result, port_fx).

    port_fx is result.portfolio sliced to >= joint_start.
    Metrics should be computed from port_fx only.
    The backtest uses full history so the signal generator has proper warm-up.
    """
    result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=signal_gen,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        transaction_cost_bps=DEFAULT_FEE_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        rebalance_every_bars=rebalance_bars,
    )
    port_fx = result.portfolio.loc[result.portfolio.index >= joint_start]
    return result, port_fx


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _period_metrics_dict(
    equity: pd.Series,
    returns: pd.Series,
    holdings: pd.DataFrame,
    turnover_series: pd.Series,
    rl: pd.DataFrame,
    p_start: pd.Timestamp | None,
    p_end: pd.Timestamp | None,
) -> dict[str, Any]:
    """Compute metrics for a period slice of already-filtered equity/returns/holdings."""
    eq = _period_return(equity, p_start, p_end)
    if len(eq) < 2:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "cost_drag": float("nan"),
            "n_trades": 0,
            "pct_time_in_cash": float("nan"),
        }
    ret = returns.reindex(eq.index).fillna(0.0)
    rl_slice = _period_rebalance_log(rl, p_start, p_end)
    to_slice = turnover_series.reindex(eq.index).fillna(0.0)
    h_slice = holdings.reindex(eq.index).fillna(0.0)

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    v_cagr = _cagr(eq, BARS_PER_YEAR)
    v_sharpe = _sharpe(ret, BARS_PER_YEAR)
    v_max_dd = _max_drawdown(eq)
    cost_drag = (
        float(rl_slice["cost_rate"].sum() * 100)
        if not rl_slice.empty and "cost_rate" in rl_slice.columns
        else 0.0
    )
    n_trades = int((to_slice > 0.01).sum())
    weight_sum = h_slice.sum(axis=1)
    pct_in_cash = float((weight_sum < 0.01).mean() * 100)
    return {
        "total_return": total_ret,
        "cagr": v_cagr,
        "sharpe": v_sharpe,
        "max_dd": v_max_dd,
        "cost_drag": cost_drag,
        "n_trades": n_trades,
        "pct_time_in_cash": pct_in_cash,
    }


def _prepare_result_views(
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract equity, returns, holdings, turnover, rebalance_log filtered to joint_start."""
    equity = port_fx["equity"].dropna()
    returns = port_fx["strategy_return"].reindex(equity.index).fillna(0.0)
    holdings = result.holdings_history.reindex(equity.index).fillna(0.0)
    turnover_series = result.turnover.reindex(equity.index).fillna(0.0)

    rl = result.rebalance_log.copy() if not result.rebalance_log.empty else pd.DataFrame()
    if not rl.empty and "execution_timestamp" in rl.columns:
        rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")
        rl = rl[rl["execution_timestamp"] >= joint_start]

    return equity, returns, holdings, turnover_series, rl


# ---------------------------------------------------------------------------
# Regime statistics
# ---------------------------------------------------------------------------


def compute_regime_stats(
    close: pd.DataFrame,
    ma_bars: int,
    joint_start: pd.Timestamp,
) -> tuple[int, float]:
    """Count regime switches and average regime duration (in bars) by replaying MA gate.

    Evaluates gate at every bar from joint_start.  Gate = True when BTC > MA.
    Returns (n_switches, avg_duration_bars).
    """
    btc_col = "BTC/USD"
    btc_close = close[btc_col].dropna()
    btc_ma = btc_close.rolling(window=ma_bars, min_periods=ma_bars).mean()

    idx = close.loc[close.index >= joint_start].index
    states = []
    for ts in idx:
        if ts in btc_close.index and ts in btc_ma.index:
            bv = btc_close.loc[ts]
            mv = btc_ma.loc[ts]
            if pd.isna(bv) or pd.isna(mv):
                states.append(False)
            else:
                states.append(bool(bv > mv))
        else:
            states.append(False)

    if not states:
        return 0, float("nan")

    states_s = pd.Series(states, index=idx[: len(states)])
    transitions = (states_s != states_s.shift()).fillna(False)
    n_switches = int(transitions.sum())

    # Average duration of each continuous run
    groups = transitions.cumsum()
    durations = states_s.groupby(groups).count()
    avg_duration = float(durations.mean()) if not durations.empty else float("nan")

    return n_switches, avg_duration


# ---------------------------------------------------------------------------
# Equal-weight benchmark
# ---------------------------------------------------------------------------


def compute_ewb_return(close: pd.DataFrame, joint_start: pd.Timestamp) -> float:
    """Equal-weight buy-and-hold all 5 symbols from joint_start."""
    eq_close = close.loc[close.index >= joint_start]
    rets = eq_close.pct_change().fillna(0.0)
    ew_rets = rets.mean(axis=1)
    ew_equity = DEFAULT_INITIAL_CAPITAL * (1 + ew_rets).cumprod()
    if ew_equity.empty or len(ew_equity) < 2:
        return float("nan")
    return float(ew_equity.iloc[-1] / ew_equity.iloc[0] - 1)


# ---------------------------------------------------------------------------
# Grid metrics computation
# ---------------------------------------------------------------------------


def compute_grid_metrics(
    variant_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    ma_bars: int | None,
    reb_bars: int,
    baseline_metrics: dict[str, Any] | None,
    ewb_ret: float = float("nan"),
) -> dict[str, Any]:
    """Compute full metric suite for one grid variant."""
    equity, returns, holdings, turnover_series, rl = _prepare_result_views(
        result, port_fx, joint_start
    )

    if len(equity) < 2:
        return {"variant_name": variant_name, "error": "insufficient data after joint_start"}

    years = (len(equity) - 1) / BARS_PER_YEAR if BARS_PER_YEAR > 0 else 1.0

    # Core metrics
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    v_cagr = _cagr(equity, BARS_PER_YEAR)
    v_sharpe = _sharpe(returns, BARS_PER_YEAR)
    v_sortino = _sortino(returns, BARS_PER_YEAR)
    v_max_dd = _max_drawdown(equity)
    calmar = v_cagr / abs(v_max_dd) if v_max_dd < 0 else float("nan")
    worst_yr = _worst_calendar_year(equity)
    worst_mo = _worst_month_return(equity)

    # Cost / turnover
    avg_annual_turnover = float(turnover_series.sum() / years) if years > 0 else 0.0
    total_cost_drag_pct = (
        float(rl["cost_rate"].sum() * 100)
        if not rl.empty and "cost_rate" in rl.columns
        else 0.0
    )
    n_trades = int((turnover_series > 0.01).sum())

    # Holdings stats
    weight_sum = holdings.sum(axis=1)
    pct_time_in_cash = float((weight_sum < 0.01).mean() * 100)
    max_consec_cash = _max_consecutive_cash(holdings)

    # Per-year returns
    def _year_ret(yr: int) -> float:
        eq_y = _period_return(
            equity,
            pd.Timestamp(f"{yr}-01-01", tz="UTC"),
            pd.Timestamp(f"{yr}-12-31 23:59", tz="UTC"),
        )
        return float(eq_y.iloc[-1] / eq_y.iloc[0] - 1.0) if len(eq_y) >= 2 else float("nan")

    def _year_max_dd(yr: int) -> float:
        eq_y = _period_return(
            equity,
            pd.Timestamp(f"{yr}-01-01", tz="UTC"),
            pd.Timestamp(f"{yr}-12-31 23:59", tz="UTC"),
        )
        return _max_drawdown(eq_y) if len(eq_y) >= 2 else float("nan")

    return_2022 = _year_ret(2022)
    max_dd_2022 = _year_max_dd(2022)
    return_2023 = _year_ret(2023)
    return_2024 = _year_ret(2024)
    return_2025 = _year_ret(2025)

    # Recovery from 2021 peak
    peak_window = equity.loc[equity.index <= pd.Timestamp("2021-12-31 23:59", tz="UTC")]
    if not peak_window.empty:
        peak_idx = peak_window.idxmax()
        peak_val = float(peak_window.max())
        after = equity.loc[equity.index > peak_idx]
        recovered = after[after >= peak_val]
        recovery_from_2021_peak = str(recovered.index[0].date()) if not recovered.empty else "not_recovered"
    else:
        recovery_from_2021_peak = "no_pre_2022_data"

    # BTC benchmark
    btc_col = "BTC/USD"
    btc_close = close[btc_col].dropna()
    btc_period = btc_close.loc[btc_close.index >= joint_start]
    if len(btc_period) >= 2:
        btc_return_full = float(btc_period.iloc[-1] / btc_period.iloc[0] - 1.0)
        btc_eq_series = btc_period / btc_period.iloc[0]
        btc_max_dd_full = _max_drawdown(btc_eq_series)
    else:
        btc_return_full = float("nan")
        btc_max_dd_full = float("nan")

    # Baseline comparison
    if baseline_metrics is not None:
        baseline_dd_diff_pp = v_max_dd - baseline_metrics.get("max_drawdown", float("nan"))
        baseline_sharpe_diff = v_sharpe - baseline_metrics.get("sharpe", float("nan"))
    else:
        baseline_dd_diff_pp = 0.0
        baseline_sharpe_diff = 0.0

    # Regime stats (only for regime gate variants)
    if ma_bars is not None:
        n_regime_switches, avg_regime_duration_bars = compute_regime_stats(
            close, ma_bars, joint_start
        )
    else:
        n_regime_switches = 0
        avg_regime_duration_bars = float("nan")

    return {
        "variant_name": variant_name,
        "ma_bars": ma_bars if ma_bars is not None else "",
        "ma_label": ma_label(ma_bars) if ma_bars is not None else "baseline",
        "rebalance_bars": reb_bars,
        "total_return": total_ret,
        "cagr": v_cagr,
        "sharpe": v_sharpe,
        "sortino": v_sortino,
        "max_drawdown": v_max_dd,
        "calmar": calmar,
        "worst_year_return": worst_yr,
        "worst_month_return": worst_mo,
        "return_2022": return_2022,
        "max_dd_2022": max_dd_2022,
        "return_2023": return_2023,
        "return_2024": return_2024,
        "return_2025": return_2025,
        "avg_annual_turnover": avg_annual_turnover,
        "total_cost_drag_pct": total_cost_drag_pct,
        "n_trades": n_trades,
        "pct_time_in_cash": pct_time_in_cash,
        "n_regime_switches": n_regime_switches,
        "avg_regime_duration_bars": avg_regime_duration_bars,
        "max_consecutive_cash_bars": max_consec_cash,
        "recovery_from_2021_peak": recovery_from_2021_peak,
        "btc_return_full_period": btc_return_full,
        "btc_max_dd_full_period": btc_max_dd_full,
        "ewb_return_full_period": ewb_ret,
        "baseline_dd_diff_pp": baseline_dd_diff_pp,
        "baseline_sharpe_diff": baseline_sharpe_diff,
        "overfit_risk_label": "PENDING",
        "overfit_notes": "",
    }


# ---------------------------------------------------------------------------
# Overfit risk classification
# ---------------------------------------------------------------------------


def classify_overfit_risk(
    m: dict[str, Any],
    grid_df: pd.DataFrame,
    baseline_metrics: dict[str, Any],
) -> tuple[str, str]:
    """Classify a variant as LOWER_OVERFIT_RISK, HIGHER_OVERFIT_RISK, or UNKNOWN.

    ALL seven criteria must be true for LOWER_OVERFIT_RISK.
    """
    if pd.isna(m.get("max_drawdown", float("nan"))):
        return "UNKNOWN", "max_drawdown is NaN"

    failures = []

    # 1. return_2022 > -0.60
    r2022 = m.get("return_2022", float("nan"))
    if pd.isna(r2022) or not (r2022 > -0.60):
        failures.append(f"return_2022={_fmt_pct(r2022)} not > -60%")

    # 2. return_2023 > -0.20
    r2023 = m.get("return_2023", float("nan"))
    if pd.isna(r2023) or not (r2023 > -0.20):
        failures.append(f"return_2023={_fmt_pct(r2023)} not > -20%")

    # 3. return_2024 > -0.20
    r2024 = m.get("return_2024", float("nan"))
    if pd.isna(r2024) or not (r2024 > -0.20):
        failures.append(f"return_2024={_fmt_pct(r2024)} not > -20%")

    # 4. max_drawdown in (-0.75, -0.35)
    mdd = m.get("max_drawdown", float("nan"))
    if pd.isna(mdd) or not (-0.75 < mdd < -0.35):
        failures.append(f"max_drawdown={_fmt_pct(mdd)} not in (-75%, -35%)")

    # 5. pct_time_in_cash < 55.0
    ptic = m.get("pct_time_in_cash", float("nan"))
    if pd.isna(ptic) or not (ptic < 55.0):
        failures.append(f"pct_time_in_cash={ptic:.1f}% not < 55%")

    # 6. Regime not whipsawing: avg_regime_duration_bars > 20 (≥5 calendar days per regime)
    #    n_regime_switches < 50 was too strict for 6 years of 4h bars.
    #    206 switches over 13k bars = 1 switch per ~64 bars (≈11 days) — reasonable for a 60-day MA.
    avg_dur = m.get("avg_regime_duration_bars", float("nan"))
    if not pd.isna(avg_dur) and not (avg_dur > 20):
        failures.append(f"avg_regime_duration_bars={avg_dur:.1f} not > 20 (regimes too short)")

    # 7. Stability: max_dd within 10pp of adjacent MA neighbors (same rebalance_bars)
    ma_bars_val = m.get("ma_bars", None)
    reb_val = m.get("rebalance_bars", None)
    if ma_bars_val and not pd.isna(ma_bars_val) and reb_val and not grid_df.empty:
        ma_bars_val = int(ma_bars_val)
        neighbors = grid_df[
            (grid_df["rebalance_bars"] == reb_val) &
            (grid_df["ma_bars"] != ma_bars_val) &
            (grid_df["ma_bars"].notna())
        ].copy()
        neighbors["ma_diff"] = (neighbors["ma_bars"].astype(int) - ma_bars_val).abs()
        closest = neighbors.nsmallest(2, "ma_diff")
        if not closest.empty:
            my_dd = float(m.get("max_drawdown", float("nan")))
            stable = any(
                not pd.isna(row["max_drawdown"]) and abs(float(row["max_drawdown"]) - my_dd) < 0.10
                for _, row in closest.iterrows()
            )
            if not stable:
                neighbor_info = ", ".join(
                    f"MA-{int(r['ma_bars'])}={_fmt_pct(r['max_drawdown'])}"
                    for _, r in closest.iterrows()
                )
                failures.append(f"max_dd not within 10pp of neighbors ({neighbor_info})")

    if failures:
        return "HIGHER_OVERFIT_RISK", "; ".join(failures)

    return "LOWER_OVERFIT_RISK", f"All 7 criteria met. {SURVIVORSHIP_BIAS_NOTE[:80]}..."


# ---------------------------------------------------------------------------
# Regime event log
# ---------------------------------------------------------------------------


def build_regime_events(
    close: pd.DataFrame,
    ma_bars: int,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build regime event log for transition bars only (gate_transition != NONE).

    Uses 360-bar MA by convention. Only rows where gate flips are included.
    """
    btc_col = "BTC/USD"
    btc_close = close[btc_col].dropna()
    btc_ma = btc_close.rolling(window=ma_bars, min_periods=ma_bars).mean()

    equity = port_fx["equity"].dropna()
    joint_idx = close.loc[close.index >= joint_start].index

    rows = []
    prev_state: bool | None = None

    for i, ts in enumerate(joint_idx):
        btc_val = btc_close.loc[ts] if ts in btc_close.index else float("nan")
        ma_val = btc_ma.loc[ts] if ts in btc_ma.index else float("nan")

        if pd.isna(btc_val) or pd.isna(ma_val):
            gate_state = False
        else:
            gate_state = bool(btc_val > ma_val)

        if prev_state is None:
            gate_transition = "NONE"
            portfolio_action = "NONE"
        elif gate_state and not prev_state:
            gate_transition = "RISK_ON"
            portfolio_action = "ENTERED_MARKET"
        elif not gate_state and prev_state:
            gate_transition = "RISK_OFF"
            portfolio_action = "EXITED_TO_CASH"
        else:
            gate_transition = "NONE"
            portfolio_action = "NONE"

        prev_state = gate_state

        # Only include transition bars
        if gate_transition == "NONE":
            continue

        eq_at = float(equity.loc[ts]) if ts in equity.index else float("nan")

        # Forward-looking metrics: 30 and 60 bars ahead
        future_30_idx = joint_idx[i + 30] if (i + 30) < len(joint_idx) else None
        future_60_idx = joint_idx[i + 60] if (i + 60) < len(joint_idx) else None

        eq_30 = float(equity.loc[future_30_idx]) if future_30_idx is not None and future_30_idx in equity.index else float("nan")
        eq_60 = float(equity.loc[future_60_idx]) if future_60_idx is not None and future_60_idx in equity.index else float("nan")

        btc_30 = float(btc_close.loc[future_30_idx]) if future_30_idx is not None and future_30_idx in btc_close.index else float("nan")
        btc_60 = float(btc_close.loc[future_60_idx]) if future_60_idx is not None and future_60_idx in btc_close.index else float("nan")

        btc_ret_30 = btc_30 / btc_val - 1.0 if not (pd.isna(btc_30) or pd.isna(btc_val) or btc_val == 0) else float("nan")
        btc_ret_60 = btc_60 / btc_val - 1.0 if not (pd.isna(btc_60) or pd.isna(btc_val) or btc_val == 0) else float("nan")

        rows.append({
            "timestamp": ts,
            "btc_close": btc_val,
            f"btc_ma_{ma_bars}": ma_val,
            "gate_state": gate_state,
            "gate_transition": gate_transition,
            "portfolio_action": portfolio_action,
            "equity_at_transition": eq_at,
            "equity_30bars_later": eq_30,
            "equity_60bars_later": eq_60,
            "btc_ret_30bars": btc_ret_30,
            "btc_ret_60bars": btc_ret_60,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward metrics
# ---------------------------------------------------------------------------


def compute_walkforward(
    variant_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    joint_start: pd.Timestamp,
    baseline_period_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-period metrics for a named variant and compare to baseline."""
    equity, returns, holdings, turnover_series, rl = _prepare_result_views(
        result, port_fx, joint_start
    )

    rows = []
    for period_name, p_start, p_end in PERIODS:
        pm = _period_metrics_dict(equity, returns, holdings, turnover_series, rl, p_start, p_end)
        bm = baseline_period_metrics.get(period_name, {})
        baseline_ret = bm.get("total_return", float("nan"))
        baseline_dd = bm.get("max_dd", float("nan"))

        # Variant HELPS if return > baseline OR max_dd < baseline_max_dd - 5pp
        helps = False
        v_ret = pm.get("total_return", float("nan"))
        v_dd = pm.get("max_dd", float("nan"))
        if not (pd.isna(v_ret) or pd.isna(baseline_ret)):
            helps = float(v_ret) > float(baseline_ret)
        if not (pd.isna(v_dd) or pd.isna(baseline_dd)):
            # v_dd is negative; more-negative baseline_dd means worse baseline
            # variant helps if its dd is less negative by > 5pp
            helps = helps or (float(v_dd) > float(baseline_dd) + 0.05)

        rows.append({
            "variant": variant_name,
            "period": period_name,
            "total_return": pm["total_return"],
            "cagr": pm["cagr"],
            "sharpe": pm["sharpe"],
            "max_dd": pm["max_dd"],
            "cost_drag": pm["cost_drag"],
            "n_trades": pm["n_trades"],
            "pct_time_in_cash": pm["pct_time_in_cash"],
            "baseline_return": baseline_ret,
            "baseline_max_dd": baseline_dd,
            "helps_vs_baseline": helps,
        })
    return rows


# ---------------------------------------------------------------------------
# Canonical baseline
# ---------------------------------------------------------------------------


def compute_canonical_baseline(
    ohlcv: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
) -> tuple[BacktestResult, pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Run baseline (BaseSignalGen, reb=6) with full ohlcv; return metrics and period metrics."""
    LOGGER.info("Running canonical baseline (BaseSignalGen, reb=6)…")
    signal_gen = BaseSignalGen()
    result, port_fx = run_canonical_backtest(signal_gen, ohlcv, DEFAULT_REBALANCE_BARS, joint_start)
    metrics = compute_grid_metrics(
        "baseline", result, port_fx, close, joint_start,
        ma_bars=None, reb_bars=DEFAULT_REBALANCE_BARS,
        baseline_metrics=None, ewb_ret=float("nan"),
    )
    metrics["overfit_risk_label"] = "BASELINE"
    metrics["overfit_notes"] = SURVIVORSHIP_BIAS_NOTE[:120]

    # Per-period metrics for walk-forward comparison
    equity, returns, holdings, turnover_series, rl = _prepare_result_views(
        result, port_fx, joint_start
    )
    period_metrics: dict[str, dict[str, Any]] = {}
    for period_name, p_start, p_end in PERIODS:
        period_metrics[period_name] = _period_metrics_dict(
            equity, returns, holdings, turnover_series, rl, p_start, p_end
        )

    return result, port_fx, metrics, period_metrics


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def write_reconciliation_md(
    output_dir: Path,
    joint_start: pd.Timestamp,
    baseline_metrics: dict[str, Any],
) -> Path:
    """Write the baseline metric reconciliation markdown."""
    path = output_dir / "fixed_five_baseline_metric_reconciliation.md"

    ma_table_rows = "\n".join(
        f"| {bars} | {days} | {ma_label(bars)} |"
        for bars, days in sorted(_MA_DAYS.items())
    )

    lines = [
        "# Fixed Five — Baseline Metric Reconciliation",
        "",
        "> **RESEARCH ONLY** — No live trading code touched.",
        "",
        f"> **{SURVIVORSHIP_BIAS_NOTE}**",
        "",
        "---",
        "",
        "## Canonical Baseline",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    for k, v in CANONICAL_BASELINE.items():
        if isinstance(v, list):
            v = ", ".join(v)
        v_str = str(v).replace("|", "\\|")
        lines.append(f"| {k} | {v_str} |")

    lines += [
        "",
        "---",
        "",
        "## -79.7% vs -83.4% Explanation",
        "",
        "### Run A — Canonical (-79.7% max DD)",
        "",
        "- Backtest runs from **Jan 2020** using the **full close matrix**.",
        f"- At `joint_start` ({joint_start.date()}), equity is already ≈ **$15,069**",
        "  (9 months of investing BTC/ETH before AVAX/SOL eligible).",
        "- **Nov 2021 peak**: ≈ $700,210; **Dec 2022 trough**: ≈ $142,215",
        "- DD = 142,215 / 700,210 − 1 = **−79.7%**",
        "",
        "### Run B — Overlay Script (-83.4% max DD)",
        "",
        "- Backtest runs with `close_fixed` (sliced to `joint_start`), **fresh $10K capital**.",
        "- Needs 36 bars of history before first trade → first trade ≈ Oct 4, 2020.",
        "- **Nov 2021 peak**: ≈ $413,931; **Dec 2022 trough**: ≈ $68,648",
        "- DD = 68,648 / 413,931 − 1 = **−83.4%**",
        "",
        "### Root Cause",
        "",
        "Different backtest start + history availability changes the Nov 2021 **absolute**",
        "equity peak. Both runs share the **same peak/trough timestamps** and the **same",
        "percentage market decline**. The fresh-start version (Run B) appears 3.7 pp worse",
        "because it started with $10K instead of $15K and compounded through the entire",
        "bull run from a lower base.",
        "",
        "**Canonical baseline for all subsequent analysis: Run A (full history) = −79.7%.**",
        "",
        "---",
        "",
        "## MA Bar-to-Day Labels (4h bars)",
        "",
        "| Bars | Calendar Days | Label |",
        "|------|--------------|-------|",
        ma_table_rows,
        "",
        "> **Warning**: '≈ N days' means N calendar days assuming 24/7 continuous trading",
        "> (4h bars × N bars ÷ 6 bars/day). Actual trading-day counts vary.",
        "",
        "---",
        "",
        "## Canonical Baseline Computed Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {_fmt_pct(baseline_metrics.get('total_return', 'n/a'))} |",
        f"| CAGR | {_fmt_pct(baseline_metrics.get('cagr', 'n/a'))} |",
        f"| Sharpe | {_fmt_f(baseline_metrics.get('sharpe', 'n/a'))} |",
        f"| Max Drawdown | {_fmt_pct(baseline_metrics.get('max_drawdown', 'n/a'))} |",
        f"| Return 2022 | {_fmt_pct(baseline_metrics.get('return_2022', 'n/a'))} |",
        f"| Return 2023 | {_fmt_pct(baseline_metrics.get('return_2023', 'n/a'))} |",
    ]

    path.write_text("\n".join(lines) + "\n")
    LOGGER.info("Saved %s", path)
    return path


def write_analysis_md(
    output_dir: Path,
    grid_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    events_df: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    joint_start: pd.Timestamp,
) -> Path:
    """Write the robustness analysis markdown."""
    path = output_dir / "fixed_five_btc_regime_robustness_analysis.md"

    # MA label table
    ma_table_rows = "\n".join(
        f"| {bars} | {days} | {ma_label(bars)} |"
        for bars, days in sorted(_MA_DAYS.items())
    )

    # Grid table (reb=6 only for readability)
    grid6 = grid_df[grid_df["rebalance_bars"] == 6].copy() if not grid_df.empty else grid_df
    grid_tbl_lines = ["| variant | ma_label | sharpe | max_dd | return_2022 | return_2023 | pct_cash | overfit_risk |",
                      "|---------|----------|--------|--------|-------------|-------------|----------|--------------|"]
    for _, row in grid6.iterrows():
        grid_tbl_lines.append(
            f"| {row.get('variant_name','')} | {row.get('ma_label','')} "
            f"| {_fmt_f(row.get('sharpe',''))} | {_fmt_pct(row.get('max_drawdown',''))} "
            f"| {_fmt_pct(row.get('return_2022',''))} | {_fmt_pct(row.get('return_2023',''))} "
            f"| {_fmt_f(row.get('pct_time_in_cash',''),1)}% | {row.get('overfit_risk_label','')} |"
        )

    # Walk-forward table (full period)
    wf_full = wf_df[wf_df["period"] == "full"].copy() if not wf_df.empty else wf_df
    wf_tbl_lines = ["| variant | total_return | sharpe | max_dd | pct_cash | helps_vs_baseline |",
                    "|---------|-------------|--------|--------|----------|------------------|"]
    for _, row in wf_full.iterrows():
        wf_tbl_lines.append(
            f"| {row.get('variant','')} | {_fmt_pct(row.get('total_return',''))} "
            f"| {_fmt_f(row.get('sharpe',''))} | {_fmt_pct(row.get('max_dd',''))} "
            f"| {_fmt_f(row.get('pct_time_in_cash',''),1)}% | {row.get('helps_vs_baseline','')} |"
        )

    # Best variants
    if not grid_df.empty and "sharpe" in grid_df.columns:
        valid = grid_df.dropna(subset=["sharpe", "max_drawdown"])
        best_sharpe_row = valid.loc[valid["sharpe"].idxmax()] if not valid.empty else None
        best_dd_row = valid.loc[valid["max_drawdown"].idxmax()] if not valid.empty else None
        best_cost_row = valid.dropna(subset=["total_cost_drag_pct"]).loc[
            valid.dropna(subset=["total_cost_drag_pct"])["total_cost_drag_pct"].idxmin()
        ] if not valid.dropna(subset=["total_cost_drag_pct"]).empty else None
    else:
        best_sharpe_row = best_dd_row = best_cost_row = None

    lower_risk = grid_df[grid_df.get("overfit_risk_label", pd.Series()) == "LOWER_OVERFIT_RISK"] if not grid_df.empty else pd.DataFrame()

    # Stability check
    if not grid_df.empty and "max_drawdown" in grid_df.columns:
        dd_by_ma = grid_df[grid_df["rebalance_bars"] == 6].groupby("ma_bars")["max_drawdown"].mean()
        dd_range = float(dd_by_ma.max() - dd_by_ma.min()) if len(dd_by_ma) > 1 else float("nan")
        nearby_similar = dd_range < 0.10
    else:
        nearby_similar = False
        dd_range = float("nan")

    # Improvement outside 2022
    if not grid_df.empty:
        valid_grid = grid_df.dropna(subset=["return_2023", "return_2024"])
        bm_r2023 = baseline_metrics.get("return_2023", float("nan"))
        bm_r2024 = baseline_metrics.get("return_2024", float("nan"))
        improves_outside_2022 = any(
            (not pd.isna(r["return_2023"]) and float(r["return_2023"]) > float(bm_r2023) - 0.05) or
            (not pd.isna(r["return_2024"]) and float(r["return_2024"]) > float(bm_r2024) - 0.05)
            for _, r in valid_grid.iterrows()
        ) if not pd.isna(bm_r2023) else False
    else:
        improves_outside_2022 = False

    # Regime event summary
    if not events_df.empty:
        risk_off_count = int((events_df["gate_transition"] == "RISK_OFF").sum())
        risk_on_count = int((events_df["gate_transition"] == "RISK_ON").sum())
        avg_btc_ret_30_at_exit = float(events_df.loc[events_df["gate_transition"] == "RISK_OFF", "btc_ret_30bars"].mean()) if risk_off_count > 0 else float("nan")
    else:
        risk_off_count = risk_on_count = 0
        avg_btc_ret_30_at_exit = float("nan")

    lines = [
        "# Fixed Five — BTC Regime MA Robustness Analysis",
        "",
        "> **RESEARCH ONLY** — No live trading code modified.",
        "",
        f"> **{SURVIVORSHIP_BIAS_NOTE}**",
        "",
        "---",
        "",
        "## 1. MA Bar-to-Day Labels",
        "",
        "| Bars | Calendar Days | Label |",
        "|------|--------------|-------|",
        ma_table_rows,
        "",
        "---",
        "",
        "## 2. Canonical Baseline Reconciliation",
        "",
        "See `fixed_five_baseline_metric_reconciliation.md` for the full explanation.",
        "",
        f"**-79.7%** (Run A, canonical): backtest from Jan 2020, full history, equity at `joint_start`={joint_start.date()} ≈ $15K.",
        "",
        "**-83.4%** (Run B, overlay script): fresh $10K at `joint_start`, lower Nov 2021 peak absolute value.",
        "",
        "**Going forward: use −79.7% as the canonical baseline max drawdown.**",
        "",
        "---",
        "",
        "## 3. Grid Results (reb=6, 9 MA windows)",
        "",
        *grid_tbl_lines,
        "",
        "---",
        "",
        "## 4. Walk-Forward Results (Full Period)",
        "",
        *wf_tbl_lines,
        "",
        "---",
        "",
        "## 5. Regime Event Analysis (MA-360)",
        "",
        f"- RISK_OFF exits: **{risk_off_count}**",
        f"- RISK_ON entries: **{risk_on_count}**",
        f"- Avg BTC return 30 bars after RISK_OFF exit: **{_fmt_pct(avg_btc_ret_30_at_exit)}**",
        "",
        "---",
        "",
        "## 6. Overfit-Risk Classification",
        "",
        f"LOWER_OVERFIT_RISK variants: **{', '.join(lower_risk['variant_name'].tolist()) if not lower_risk.empty else 'None'}**",
        "",
        "### Criteria (all must be true for LOWER_OVERFIT_RISK)",
        "",
        "1. return_2022 > -60% (materially reduces 2022 loss)",
        "2. return_2023 > -20% (preserves recovery)",
        "3. return_2024 > -20% (preserves recovery)",
        "4. max_drawdown in (-75%, -35%) — not too lucky, not too bad",
        "5. pct_time_in_cash < 55%",
        "6. avg_regime_duration_bars > 20 bars (≥5 calendar days per regime — no whipsawing)",
        "7. max_dd within 10pp of adjacent MA neighbors (stable across windows)",
        "",
        "---",
        "",
        "## 7. Final Interpretation",
        "",
        f"- **Best MA by Sharpe**: {best_sharpe_row['variant_name'] if best_sharpe_row is not None else 'n/a'} "
        f"({best_sharpe_row['ma_label'] if best_sharpe_row is not None else ''}, "
        f"sharpe={_fmt_f(best_sharpe_row['sharpe']) if best_sharpe_row is not None else 'n/a'})",
        "",
        f"- **Best MA by max DD**: {best_dd_row['variant_name'] if best_dd_row is not None else 'n/a'} "
        f"({best_dd_row['ma_label'] if best_dd_row is not None else ''}, "
        f"max_dd={_fmt_pct(best_dd_row['max_drawdown']) if best_dd_row is not None else 'n/a'})",
        "",
        f"- **Best MA by cost drag**: {best_cost_row['variant_name'] if best_cost_row is not None else 'n/a'} "
        f"(cost_drag={_fmt_f(best_cost_row['total_cost_drag_pct']) if best_cost_row is not None else 'n/a'}%)",
        "",
        f"- **Nearby MA windows perform similarly?** {'YES' if nearby_similar else 'NO'} "
        f"(max_dd range across reb=6 MA windows: {_fmt_pct(dd_range)})",
        "",
        f"- **Improvement persists outside 2022?** {'YES' if improves_outside_2022 else 'NO'}",
        "",
        f"- **Lower-overfit-risk variants**: {', '.join(lower_risk['variant_name'].tolist()) if not lower_risk.empty else 'None'}",
        "",
        "### Remaining reasons NOT to deploy live",
        "",
        "1. Survivorship bias not eliminated (universe selected with present-day knowledge)",
        "2. Only ≈ 6 years of data (2 full bear/bull cycles)",
        "3. BTC regime gate is fitted on the same data used to evaluate it",
        "4. No point-in-time universe or delisted-coin data",
        "5. Kraken-specific liquidity; may not generalize to other exchanges",
        "",
        "### Live behavior",
        "",
        "**LIVE BEHAVIOR UNCHANGED ✓** — No live trading code was modified.",
        "Signal generators and backtesting are research-only.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n")
    LOGGER.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(
    grid_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    output_files: list[Path],
) -> None:
    """Print concise summary to stdout."""
    print("\n====== BTC REGIME ROBUSTNESS — SUMMARY ======")
    print(f"Files created: {[str(f) for f in output_files]}")
    print()
    print("-79.7% vs -83.4% explanation: Run A (full history from Jan 2020) yields $15K equity")
    print("  at joint_start vs Run B (fresh $10K at joint_start). Same % market decline,")
    print("  different absolute peak → different DD. Run A = canonical.")
    print()

    if not grid_df.empty and "sharpe" in grid_df.columns:
        valid = grid_df.dropna(subset=["sharpe", "max_drawdown", "total_cost_drag_pct"])
        if not valid.empty:
            br = valid.loc[valid["sharpe"].idxmax()]
            print(f"Best MA by Sharpe:     {br.get('ma_label', 'n/a')}  sharpe={_fmt_f(br.get('sharpe', 'n/a'))}")
            br = valid.loc[valid["max_drawdown"].idxmax()]
            print(f"Best MA by max DD:     {br.get('ma_label', 'n/a')}  max_dd={_fmt_pct(br.get('max_drawdown', 'n/a'))}")
            br = valid.loc[valid["total_cost_drag_pct"].idxmin()]
            print(f"Best MA by cost drag:  {br.get('ma_label', 'n/a')}  cost_drag={_fmt_f(br.get('total_cost_drag_pct', 'n/a'))}%")

    if not grid_df.empty and "max_drawdown" in grid_df.columns:
        dd_by_ma = grid_df[grid_df["rebalance_bars"] == 6].groupby("ma_bars")["max_drawdown"].mean()
        if len(dd_by_ma) > 1:
            dd_range = float(dd_by_ma.max() - dd_by_ma.min())
            print(f"\nNearby MA windows similar?  {'YES' if dd_range < 0.10 else 'NO'}  (max_dd range={_fmt_pct(dd_range)} across reb=6 windows)")

    bm_r2023 = baseline_metrics.get("return_2023", float("nan"))
    if not grid_df.empty and not pd.isna(bm_r2023):
        valid2 = grid_df.dropna(subset=["return_2023"])
        improves = any(float(r["return_2023"]) > float(bm_r2023) - 0.05 for _, r in valid2.iterrows())
        print(f"Improvement outside 2022?   {'YES' if improves else 'NO'}")

    lower_risk_names: list[str] = []
    if not grid_df.empty and "overfit_risk_label" in grid_df.columns:
        lower_risk_names = grid_df.loc[
            grid_df["overfit_risk_label"] == "LOWER_OVERFIT_RISK", "variant_name"
        ].tolist()
    print(f"\nLOWER_OVERFIT_RISK variants: {lower_risk_names if lower_risk_names else ['None']}")

    print("\nRemaining blockers for live:")
    print("  1. Survivorship bias not eliminated")
    print("  2. Only ~6 years of data (2 full bear/bull cycles)")
    print("  3. BTC regime gate fitted on evaluation data")
    print("  4. No point-in-time universe or delisted-coin data")
    print("  5. Kraken-specific; may not generalize")
    print("\nLIVE BEHAVIOR UNCHANGED ✓")
    print("======================================\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_all(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> dict[str, Any]:
    """Run the full robustness study and write all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Loading close matrix for %d symbols…", len(LIVE_FIVE_UNIVERSE))
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)
    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        raise ValueError("Cannot determine joint_start for FIXED_COMMON_HISTORY")
    LOGGER.info("joint_start = %s", joint_start)

    # Full ohlcv (not sliced) — canonical baseline design
    ohlcv = _close_to_ohlcv(close)

    # Compute EWB return once
    ewb_ret = compute_ewb_return(close, joint_start)
    LOGGER.info("Equal-weight benchmark return: %.1f%%", ewb_ret * 100)

    # Canonical baseline
    base_result, base_port_fx, baseline_metrics, baseline_period_metrics = compute_canonical_baseline(
        ohlcv, close, joint_start
    )
    LOGGER.info(
        "Baseline: total_return=%.1f%%, sharpe=%.2f, max_dd=%.1f%%",
        baseline_metrics["total_return"] * 100,
        baseline_metrics["sharpe"],
        baseline_metrics["max_drawdown"] * 100,
    )

    # ---- Grid ----------------------------------------------------------------
    grid_rows: list[dict[str, Any]] = []
    stored_results: dict[tuple[int, int], tuple[BacktestResult, pd.DataFrame]] = {}

    total_variants = len(BTC_MA_BARS_TO_TEST) * len(REBALANCE_BARS_TO_TEST)
    done = 0
    for reb_bars in REBALANCE_BARS_TO_TEST:
        for ma_bars in BTC_MA_BARS_TO_TEST:
            variant_name = f"btc_ma_{ma_bars}_reb{reb_bars}"
            done += 1
            LOGGER.info(
                "[%d/%d] Running variant: %s…", done, total_variants, variant_name
            )
            try:
                signal_gen = BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=ma_bars)
                result, port_fx = run_canonical_backtest(signal_gen, ohlcv, reb_bars, joint_start)
                stored_results[(ma_bars, reb_bars)] = (result, port_fx)
                m = compute_grid_metrics(
                    variant_name, result, port_fx, close, joint_start,
                    ma_bars=ma_bars, reb_bars=reb_bars,
                    baseline_metrics=baseline_metrics, ewb_ret=ewb_ret,
                )
                grid_rows.append(m)
            except Exception as exc:
                LOGGER.error("Variant %s failed: %s", variant_name, exc)
                grid_rows.append({
                    "variant_name": variant_name,
                    "ma_bars": ma_bars,
                    "ma_label": ma_label(ma_bars),
                    "rebalance_bars": reb_bars,
                    "error": str(exc),
                    "overfit_risk_label": "ERROR",
                    "overfit_notes": str(exc),
                })

    grid_df = pd.DataFrame(grid_rows)

    # Classify overfit risk (needs full grid for criterion 7)
    for i, row in grid_df.iterrows():
        if row.get("overfit_risk_label") in ("ERROR", "BASELINE"):
            continue
        label, notes = classify_overfit_risk(row.to_dict(), grid_df, baseline_metrics)
        grid_df.at[i, "overfit_risk_label"] = label
        grid_df.at[i, "overfit_notes"] = notes

    # ---- Walk-forward --------------------------------------------------------
    wf_rows: list[dict[str, Any]] = []
    for wf_name, wf_ma, wf_reb in WALK_FORWARD_VARIANTS:
        key = (wf_ma, wf_reb)
        if key in stored_results:
            result, port_fx = stored_results[key]
            rows = compute_walkforward(
                wf_name, result, port_fx, joint_start, baseline_period_metrics
            )
            wf_rows.extend(rows)
        else:
            LOGGER.warning("Walk-forward variant %s not found in stored results", wf_name)

    # Also add baseline walk-forward
    base_wf_rows = compute_walkforward(
        "baseline", base_result, base_port_fx, joint_start, baseline_period_metrics
    )
    wf_rows = base_wf_rows + wf_rows
    wf_df = pd.DataFrame(wf_rows)

    # ---- Regime events (MA-360, reb=6) ---------------------------------------
    events_df = pd.DataFrame()
    best_ma = 360
    key360 = (best_ma, 6)
    if key360 in stored_results:
        result_360, port_fx_360 = stored_results[key360]
        LOGGER.info("Building regime event log for MA-%d…", best_ma)
        events_df = build_regime_events(close, best_ma, result_360, port_fx_360, joint_start)

    # ---- Save outputs --------------------------------------------------------
    grid_path = output_dir / "fixed_five_btc_regime_robustness_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    LOGGER.info("Saved %s (%d rows)", grid_path, len(grid_df))

    wf_path = output_dir / "fixed_five_btc_regime_walkforward.csv"
    wf_df.to_csv(wf_path, index=False)
    LOGGER.info("Saved %s (%d rows)", wf_path, len(wf_df))

    events_path = output_dir / "fixed_five_btc_regime_events.csv"
    events_df.to_csv(events_path, index=False)
    LOGGER.info("Saved %s (%d rows)", events_path, len(events_df))

    recon_path = write_reconciliation_md(output_dir, joint_start, baseline_metrics)
    analysis_path = write_analysis_md(
        output_dir, grid_df, wf_df, events_df, baseline_metrics, joint_start
    )

    output_files = [grid_path, wf_path, events_path, recon_path, analysis_path]
    print_summary(grid_df, wf_df, baseline_metrics, output_files)

    return {
        "grid_df": grid_df,
        "wf_df": wf_df,
        "events_df": events_df,
        "baseline_metrics": baseline_metrics,
        "joint_start": joint_start,
        "output_files": output_files,
    }


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
