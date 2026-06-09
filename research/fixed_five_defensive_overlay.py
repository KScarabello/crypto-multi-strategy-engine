"""Research-only defensive overlay and turnover-reduction analysis for
FIXED_COMMON_HISTORY five-coin backtest (starting 2020-09-28).

Tests various defensive overlays (absolute momentum filter, BTC regime filter,
reduced rebalance frequency, turnover buffer, minimum trade threshold) and
their combinations against the baseline CrossSectionalMomentum strategy.

RESEARCH ONLY — no live trading code is touched or imported.

Outputs
-------
reports/fixed_five_gross_vs_net_cost_analysis.csv
reports/fixed_five_defensive_overlay_comparison.csv
reports/fixed_five_defensive_overlay_by_period.csv
reports/fixed_five_defensive_overlay_analysis.md

Usage
-----
    .venv/bin/python -m research.fixed_five_defensive_overlay
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
    _make_signal_generator,
)
from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import summary_metrics
from strategies.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
    compute_momentum_score,
    check_regime_filter,
)

LOGGER = logging.getLogger(__name__)

BARS_PER_YEAR = BARS_PER_YEAR_4H  # 2190 for 4h bars

# ---------------------------------------------------------------------------
# Period definitions
# ---------------------------------------------------------------------------

PERIODS: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = [
    ("2020-09-28_to_2021", pd.Timestamp("2020-09-28", tz="UTC"), pd.Timestamp("2021-12-31 23:59", tz="UTC")),
    ("2022",               pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:59", tz="UTC")),
    ("2023",               pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:59", tz="UTC")),
    ("2024",               pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59", tz="UTC")),
    ("2025",               pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59", tz="UTC")),
    ("2026",               pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-12-31 23:59", tz="UTC")),
    ("full",               None, None),
]

# ---------------------------------------------------------------------------
# Signal Generator Classes
# ---------------------------------------------------------------------------


class BaseSignalGen:
    """Baseline CrossSectionalMomentum: top-3 equal-weight."""

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        min_history: int = DEFAULT_MIN_HISTORY_BARS,
        short_lb: int = DEFAULT_SHORT_LOOKBACK,
        med_lb: int = DEFAULT_MEDIUM_LOOKBACK,
    ) -> None:
        self._strategy = CrossSectionalMomentumStrategy()
        self._cfg: dict[str, Any] = {
            "top_n": top_n,
            "short_lookback_bars": short_lb,
            "medium_lookback_bars": med_lb,
            "min_history_bars": min_history,
            "use_regime_filter": False,
            "min_eligible_assets": 1,
        }

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        try:
            w = self._strategy.generate_target_weights(close, ts, config=self._cfg)
        except Exception:
            w = {}
        return pd.Series(w, dtype=float).reindex(close.columns, fill_value=0.0)


class AbsMomSignalGen:
    """Absolute momentum filter: only hold symbols with positive momentum score."""

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        min_history: int = DEFAULT_MIN_HISTORY_BARS,
        short_lb: int = DEFAULT_SHORT_LOOKBACK,
        med_lb: int = DEFAULT_MEDIUM_LOOKBACK,
    ) -> None:
        self._top_n = top_n
        self._min_history = min_history
        self._short_lb = short_lb
        self._med_lb = med_lb

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        try:
            scores = compute_momentum_score(close, self._short_lb, self._med_lb)
            valid_price = close.notna()
            history_count = valid_price.astype(int).cumsum()
            eligible_mask = valid_price.loc[ts] & (history_count.loc[ts] >= self._min_history)
            eligible_syms = close.columns[eligible_mask]

            if ts not in scores.index:
                return pd.Series(0.0, index=close.columns)

            score_at_ts = scores.loc[ts, eligible_syms].dropna()
            positive = score_at_ts[score_at_ts > 0].sort_values(ascending=False)
            selected = positive.index[: self._top_n].tolist()

            weights = pd.Series(0.0, index=close.columns)
            if selected:
                weights.loc[selected] = 1.0 / len(selected)
            return weights
        except Exception:
            return pd.Series(0.0, index=close.columns)


class BTCRegimeSignalGen:
    """BTC regime gate — MA or momentum check. When gate=False: all cash."""

    def __init__(
        self,
        gate_type: str,
        btc_ma_bars: int | None,
        top_n: int = DEFAULT_TOP_N,
        min_history: int = DEFAULT_MIN_HISTORY_BARS,
        short_lb: int = DEFAULT_SHORT_LOOKBACK,
        med_lb: int = DEFAULT_MEDIUM_LOOKBACK,
    ) -> None:
        self._gate_type = gate_type
        self._btc_ma_bars = btc_ma_bars
        self._top_n = top_n
        self._min_history = min_history
        self._short_lb = short_lb
        self._med_lb = med_lb
        self._base_gen = BaseSignalGen(top_n=top_n, min_history=min_history, short_lb=short_lb, med_lb=med_lb)

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        btc_col = "BTC/USD"
        try:
            if self._gate_type == "ma":
                risk_on = check_regime_filter(
                    close, ts, btc_symbol=btc_col, ma_lookback_bars=self._btc_ma_bars
                )
            else:  # mom
                scores = compute_momentum_score(close, self._short_lb, self._med_lb)
                if ts in scores.index and btc_col in scores.columns:
                    btc_score = scores.loc[ts, btc_col]
                    risk_on = (not pd.isna(btc_score)) and (btc_score > 0)
                else:
                    risk_on = False

            if not risk_on:
                return pd.Series(0.0, index=close.columns)

            return self._base_gen(close, ts)
        except Exception:
            return pd.Series(0.0, index=close.columns)


class TurnoverBufferSignalGen:
    """Keep existing holdings if still in top-buffer_top_n (and optionally pass abs mom)."""

    def __init__(
        self,
        base_gen: Any,
        buffer_top_n: int = 4,
        use_abs_mom: bool = False,
        top_n: int = DEFAULT_TOP_N,
        min_history: int = DEFAULT_MIN_HISTORY_BARS,
        short_lb: int = DEFAULT_SHORT_LOOKBACK,
        med_lb: int = DEFAULT_MEDIUM_LOOKBACK,
    ) -> None:
        self._base_gen = base_gen
        self._buffer_top_n = buffer_top_n
        self._use_abs_mom = use_abs_mom
        self._top_n = top_n
        self._min_history = min_history
        self._short_lb = short_lb
        self._med_lb = med_lb
        self.last_target: pd.Series | None = None

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        try:
            scores = compute_momentum_score(close, self._short_lb, self._med_lb)
            valid_price = close.notna()
            history_count = valid_price.astype(int).cumsum()
            eligible_mask = valid_price.loc[ts] & (history_count.loc[ts] >= self._min_history)
            eligible_syms = close.columns[eligible_mask]

            if ts not in scores.index:
                weights = pd.Series(0.0, index=close.columns)
                self.last_target = weights.copy()
                return weights

            score_at_ts = scores.loc[ts, eligible_syms].dropna().sort_values(ascending=False)
            buffer_set = set(score_at_ts.index[: self._buffer_top_n].tolist())

            if self.last_target is None:
                top_n_syms = score_at_ts.index[: self._top_n].tolist()
                selected = top_n_syms
            else:
                currently_held = {
                    sym
                    for sym in self.last_target.index
                    if self.last_target.get(sym, 0.0) > 0.01
                }
                if self._use_abs_mom:
                    pos_mom = set(score_at_ts[score_at_ts > 0].index.tolist())
                    keepable = buffer_set & pos_mom
                else:
                    keepable = buffer_set
                kept = currently_held & keepable
                needed = self._top_n - len(kept)
                new_candidates = [s for s in score_at_ts.index if s not in kept]
                if self._use_abs_mom:
                    new_candidates = [
                        s for s in new_candidates if score_at_ts.get(s, float("-inf")) > 0
                    ]
                added = new_candidates[: max(needed, 0)]
                selected = list(kept) + added

            weights = pd.Series(0.0, index=close.columns)
            if selected:
                weights.loc[selected] = 1.0 / len(selected)
            self.last_target = weights.copy()
            return weights
        except Exception:
            weights = pd.Series(0.0, index=close.columns)
            self.last_target = weights.copy()
            return weights


class MinTradeSignalGen:
    """Suppress trades smaller than threshold; renormalize result."""

    def __init__(self, base_gen: Any, threshold: float, top_n: int = DEFAULT_TOP_N) -> None:
        self._base_gen = base_gen
        self._threshold = threshold
        self._top_n = top_n
        self.last_target: pd.Series | None = None

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        raw = self._base_gen(close, ts)
        if self.last_target is None:
            self.last_target = raw.copy()
            return raw

        adjusted = raw.copy()
        for sym in raw.index:
            change = abs(raw.loc[sym] - self.last_target.get(sym, 0.0))
            if change < self._threshold:
                adjusted.loc[sym] = self.last_target.get(sym, 0.0)

        total = adjusted.sum()
        if total > 1.0 + 1e-9:
            adjusted = adjusted / total
        adjusted = adjusted.clip(lower=0.0)

        self.last_target = adjusted.copy()
        return adjusted


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _sortino(returns: pd.Series, bars_per_year: int) -> float:
    neg = returns[returns < 0]
    if len(neg) == 0:
        return float("nan")
    downside_std = neg.std(ddof=0) * math.sqrt(bars_per_year)
    if downside_std <= 0:
        return float("nan")
    return float(returns.mean() * bars_per_year / downside_std)


def _max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return 0.0
    running_max = clean.cummax()
    dd = clean / running_max - 1.0
    return float(dd.min())


def _cagr(equity: pd.Series, bars_per_year: int) -> float:
    clean = equity.dropna()
    if len(clean) < 2:
        return 0.0
    years = (len(clean) - 1) / bars_per_year
    if years <= 0:
        return 0.0
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series, bars_per_year: int) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    std = clean.std(ddof=0)
    if math.isclose(float(std), 0.0):
        return 0.0
    return float((clean.mean() / std) * math.sqrt(bars_per_year))


def _worst_month_return(equity: pd.Series) -> float:
    try:
        monthly = equity.resample("ME").last()
    except Exception:
        monthly = equity.resample("M").last()
    monthly_ret = monthly.pct_change().dropna()
    if monthly_ret.empty:
        return float("nan")
    return float(monthly_ret.min())


def _worst_calendar_year(equity: pd.Series) -> float:
    yearly = equity.resample("YE").last()
    yr = yearly.pct_change().dropna()
    if yr.empty:
        return float("nan")
    return float(yr.min())


def _period_return(equity: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    """Slice equity to period."""
    e = equity.dropna()
    if start is not None:
        e = e.loc[e.index >= start]
    if end is not None:
        e = e.loc[e.index <= end]
    return e


def _period_rebalance_log(
    rebalance_log: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    if rebalance_log.empty:
        return rebalance_log
    rl = rebalance_log.copy()
    if "execution_timestamp" in rl.columns:
        rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")
        if start is not None:
            rl = rl[rl["execution_timestamp"] >= start]
        if end is not None:
            rl = rl[rl["execution_timestamp"] <= end]
    return rl


def _recovery_date(equity: pd.Series, peak_ts: pd.Timestamp, peak_equity: float) -> str:
    after = equity.loc[equity.index > peak_ts]
    recovered = after[after >= peak_equity]
    if recovered.empty:
        return "not_recovered"
    return str(recovered.index[0].date())


def _max_consecutive_cash(holdings_history: pd.DataFrame) -> int:
    weight_sum = holdings_history.sum(axis=1)
    in_cash = (weight_sum < 0.01).astype(int)
    max_run = 0
    current_run = 0
    for v in in_cash:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Full metrics computation per variant
# ---------------------------------------------------------------------------


def compute_variant_metrics(
    result: BacktestResult,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Compute full suite of metrics for a single variant."""
    portfolio = result.portfolio
    equity = portfolio["equity"].dropna()
    returns = portfolio["strategy_return"].reindex(equity.index).fillna(0.0)
    holdings = result.holdings_history.reindex(equity.index).fillna(0.0)
    rl = result.rebalance_log.copy() if not result.rebalance_log.empty else pd.DataFrame()

    # Ensure execution_timestamp is parsed
    if not rl.empty and "execution_timestamp" in rl.columns:
        rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")

    # Basic metrics
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else float("nan")
    v_cagr = _cagr(equity, BARS_PER_YEAR)
    v_sharpe = _sharpe(returns, BARS_PER_YEAR)
    v_sortino = _sortino(returns, BARS_PER_YEAR)
    v_max_dd = _max_drawdown(equity)
    calmar = v_cagr / abs(v_max_dd) if v_max_dd < 0 else float("nan")
    worst_yr = _worst_calendar_year(equity)
    worst_mo = _worst_month_return(equity)

    # Turnover / cost
    years = (len(equity) - 1) / BARS_PER_YEAR if len(equity) > 1 else 1.0
    turnover_series = result.turnover.reindex(equity.index).fillna(0.0)
    avg_annual_turnover = float(turnover_series.sum() / years) if years > 0 else 0.0
    total_cost_drag_pct = float(rl["cost_rate"].sum() * 100) if not rl.empty and "cost_rate" in rl.columns else 0.0
    n_trades = int((turnover_series > 0.01).sum())

    # Holdings stats
    weight_sum = holdings.sum(axis=1)
    pct_time_invested = float((weight_sum > 0.01).mean() * 100)
    pct_time_in_cash = float((weight_sum < 0.01).mean() * 100)
    max_consec_cash = _max_consecutive_cash(holdings)

    # 2022 metrics
    eq_2022 = _period_return(equity, pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:59", tz="UTC"))
    ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1.0) if len(eq_2022) >= 2 else float("nan")
    max_dd_2022 = _max_drawdown(eq_2022) if len(eq_2022) >= 2 else float("nan")

    # 2023 return (for robust candidate check)
    eq_2023 = _period_return(equity, pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:59", tz="UTC"))
    ret_2023 = float(eq_2023.iloc[-1] / eq_2023.iloc[0] - 1.0) if len(eq_2023) >= 2 else float("nan")

    # Recovery from 2021 peak
    peak_window = equity.loc[equity.index <= pd.Timestamp("2021-12-31 23:59", tz="UTC")]
    if not peak_window.empty:
        peak_idx = peak_window.idxmax()
        peak_val = float(peak_window.max())
        recovery = _recovery_date(equity, peak_idx, peak_val)
    else:
        recovery = "not_recovered"

    # BTC benchmark metrics
    btc_col = "BTC/USD"
    btc_eq = close[btc_col].dropna()
    btc_eq_period = btc_eq.loc[btc_eq.index >= joint_start]
    if len(btc_eq_period) >= 2:
        btc_return_full = float(btc_eq_period.iloc[-1] / btc_eq_period.iloc[0] - 1.0)
        btc_equity_series = btc_eq_period / btc_eq_period.iloc[0]
        btc_max_dd = _max_drawdown(btc_equity_series)
    else:
        btc_return_full = float("nan")
        btc_max_dd = float("nan")

    return {
        "total_return": total_ret,
        "cagr": v_cagr,
        "sharpe": v_sharpe,
        "sortino": v_sortino,
        "max_drawdown": v_max_dd,
        "calmar": calmar,
        "worst_year_return": worst_yr,
        "worst_month_return": worst_mo,
        "avg_annual_turnover": avg_annual_turnover,
        "total_cost_drag_pct": total_cost_drag_pct,
        "n_trades": n_trades,
        "pct_time_invested": pct_time_invested,
        "pct_time_in_cash": pct_time_in_cash,
        "max_consecutive_cash_bars": max_consec_cash,
        "return_2022": ret_2022,
        "max_dd_2022": max_dd_2022,
        "return_2023": ret_2023,
        "recovery_from_2021_peak": recovery,
        "btc_return_full_period": btc_return_full,
        "btc_max_dd_full_period": btc_max_dd,
    }


def compute_period_metrics(
    result: BacktestResult,
    period_name: str,
    period_start: pd.Timestamp | None,
    period_end: pd.Timestamp | None,
) -> dict[str, Any]:
    """Compute metrics for a single time period slice."""
    equity = result.portfolio["equity"].dropna()
    returns = result.portfolio["strategy_return"].reindex(equity.index).fillna(0.0)
    rl = result.rebalance_log.copy() if not result.rebalance_log.empty else pd.DataFrame()

    if not rl.empty and "execution_timestamp" in rl.columns:
        rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")

    eq_slice = _period_return(equity, period_start, period_end)
    if len(eq_slice) < 2:
        return {
            "period": period_name,
            "total_return": float("nan"),
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "cost_drag": float("nan"),
            "n_trades": 0,
            "pct_time_in_cash": float("nan"),
        }

    ret_slice = returns.reindex(eq_slice.index).fillna(0.0)
    rl_slice = _period_rebalance_log(rl, period_start, period_end)
    turnover_slice = result.turnover.reindex(eq_slice.index).fillna(0.0)
    holdings_slice = result.holdings_history.reindex(eq_slice.index).fillna(0.0)

    total_ret = float(eq_slice.iloc[-1] / eq_slice.iloc[0] - 1.0)
    v_cagr = _cagr(eq_slice, BARS_PER_YEAR)
    v_sharpe = _sharpe(ret_slice, BARS_PER_YEAR)
    v_max_dd = _max_drawdown(eq_slice)
    cost_drag = float(rl_slice["cost_rate"].sum() * 100) if not rl_slice.empty and "cost_rate" in rl_slice.columns else 0.0
    n_trades = int((turnover_slice > 0.01).sum())
    weight_sum = holdings_slice.sum(axis=1)
    pct_in_cash = float((weight_sum < 0.01).mean() * 100)

    return {
        "period": period_name,
        "total_return": total_ret,
        "cagr": v_cagr,
        "sharpe": v_sharpe,
        "max_dd": v_max_dd,
        "cost_drag": cost_drag,
        "n_trades": n_trades,
        "pct_time_in_cash": pct_in_cash,
    }


# ---------------------------------------------------------------------------
# Part 1 — Gross vs Net Baseline Analysis
# ---------------------------------------------------------------------------


def run_gross_vs_net_analysis(
    result: BacktestResult,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Build the gross vs net cost analysis DataFrame."""
    portfolio = result.portfolio
    equity = portfolio["equity"].dropna()
    gross_ret = result.gross_return.reindex(equity.index).fillna(0.0)

    # Build gross equity curve
    gross_equity = equity.iloc[0] * (1 + gross_ret).cumprod()
    net_equity = equity

    gross_total = float(gross_equity.iloc[-1] / gross_equity.iloc[0] - 1.0)
    net_total = float(net_equity.iloc[-1] / net_equity.iloc[0] - 1.0)

    gross_cagr = _cagr(gross_equity, BARS_PER_YEAR)
    net_cagr = _cagr(net_equity, BARS_PER_YEAR)

    gross_returns = gross_equity.pct_change().fillna(0.0)
    net_returns = portfolio["strategy_return"].reindex(equity.index).fillna(0.0)

    gross_sharpe = _sharpe(gross_returns, BARS_PER_YEAR)
    net_sharpe = _sharpe(net_returns, BARS_PER_YEAR)

    gross_max_dd = _max_drawdown(gross_equity)
    net_max_dd = _max_drawdown(net_equity)

    gross_end = float(gross_equity.iloc[-1])
    net_end = float(net_equity.iloc[-1])
    diff_end = gross_end - net_end
    cost_pct_of_initial = diff_end / initial_capital * 100
    cost_pct_of_gross_end = diff_end / gross_end * 100

    # Rebalance log analysis
    rl = result.rebalance_log.copy() if not result.rebalance_log.empty else pd.DataFrame()
    if not rl.empty and "execution_timestamp" in rl.columns:
        rl["execution_timestamp"] = pd.to_datetime(rl["execution_timestamp"], utc=True, errors="coerce")

    # Turnover by year
    turnover_by_year = {}
    trades_by_year = {}
    if not rl.empty:
        rl["year"] = rl["execution_timestamp"].dt.year
        for yr, grp in rl.groupby("year"):
            turnover_by_year[yr] = float(grp["turnover"].sum())
            trades_by_year[yr] = int((grp["turnover"] > 0.01).sum())

    pct_zero_turnover = float((result.turnover.reindex(equity.index).fillna(0.0) < 0.01).mean() * 100)

    # 21.7% cost drag clarification: drawdown period 2021-11-22 to 2022-12-19
    drawdown_start = pd.Timestamp("2021-11-22", tz="UTC")
    drawdown_end = pd.Timestamp("2022-12-19", tz="UTC")
    if not rl.empty and "execution_timestamp" in rl.columns and "cost_rate" in rl.columns:
        mask = (rl["execution_timestamp"] >= drawdown_start) & (rl["execution_timestamp"] <= drawdown_end)
        dd_cost_sum = float(rl.loc[mask, "cost_rate"].sum()) * 100
    else:
        dd_cost_sum = float("nan")

    rows = [
        ("gross_total_return",        gross_total,          net_total,           gross_total - net_total),
        ("cagr",                      gross_cagr,           net_cagr,            gross_cagr - net_cagr),
        ("sharpe",                    gross_sharpe,         net_sharpe,          gross_sharpe - net_sharpe),
        ("max_drawdown",              gross_max_dd,         net_max_dd,          net_max_dd - gross_max_dd),
        ("ending_equity_$",           gross_end,            net_end,             diff_end),
        ("cost_pct_of_initial_cap",   cost_pct_of_initial,  cost_pct_of_initial, 0.0),
        ("cost_pct_of_gross_end",     cost_pct_of_gross_end, cost_pct_of_gross_end, 0.0),
        ("pct_rebalances_zero_turnover", pct_zero_turnover, pct_zero_turnover,   0.0),
        ("drawdown_period_cost_drag_pct_2021_11_22_to_2022_12_19", dd_cost_sum, dd_cost_sum, 0.0),
    ]

    df = pd.DataFrame(rows, columns=["metric", "gross_value", "net_value", "difference"])
    return df


# ---------------------------------------------------------------------------
# Robust Candidate Classification
# ---------------------------------------------------------------------------


def classify_robust_candidate(
    m: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[str, str]:
    """Return (ROBUST_CANDIDATE or NOT_SELECTED, reason)."""
    reasons_fail = []

    # 1. max_drawdown materially better (>5pp improvement)
    # More-negative baseline vs less-negative variant → positive improvement
    dd_improvement = m["max_drawdown"] - baseline["max_drawdown"]
    if not (dd_improvement > 0.05):
        reasons_fail.append(f"max_dd not materially better ({m['max_drawdown']:.1%} vs baseline {baseline['max_drawdown']:.1%})")

    # 2. sharpe >= baseline_sharpe - 0.1
    if not (m["sharpe"] >= baseline["sharpe"] - 0.1):
        reasons_fail.append(f"sharpe too low ({m['sharpe']:.2f} vs threshold {baseline['sharpe'] - 0.1:.2f})")

    # 3. reduces turnover or cost
    reduces_cost = (
        m["avg_annual_turnover"] <= baseline["avg_annual_turnover"] or
        m["total_cost_drag_pct"] <= baseline["total_cost_drag_pct"]
    )
    if not reduces_cost:
        reasons_fail.append("does not reduce turnover or cost")

    # 4. return_2022 > baseline_return_2022 AND return_2023 > -30%
    if not (m["return_2022"] > baseline["return_2022"]):
        reasons_fail.append(f"2022 return not better ({m['return_2022']:.1%} vs {baseline['return_2022']:.1%})")
    if not (m.get("return_2023", float("nan")) > -0.30):
        reasons_fail.append(f"2023 return < -30% ({m.get('return_2023', float('nan')):.1%})")

    # 5. pct_time_in_cash < 50%
    if not (m["pct_time_in_cash"] < 50.0):
        reasons_fail.append(f"too much time in cash ({m['pct_time_in_cash']:.1f}%)")

    # 7. n_trades > 0
    if not (m["n_trades"] > 0):
        reasons_fail.append("zero trades (all-cash)")

    if reasons_fail:
        return "NOT_SELECTED", "; ".join(reasons_fail)

    note = f"SURVIVORSHIP_BIAS: {SURVIVORSHIP_BIAS_NOTE[:80]}..."
    return "ROBUST_CANDIDATE", note


# ---------------------------------------------------------------------------
# Variant definitions (factory functions)
# ---------------------------------------------------------------------------


def _make_variants(
    base_gen_factory: Any,
    abs_mom_factory: Any,
    btc_ma180_factory: Any,
    btc_ma360_factory: Any,
    btc_mom_factory: Any,
    buffer_factory: Any,
    min25_factory: Any,
    min5_factory: Any,
    min10_factory: Any,
    abs_mom_buf_factory: Any,
    btc_ma180_buf_factory: Any,
) -> list[dict[str, Any]]:
    return [
        {"name": "baseline",               "gen_factory": base_gen_factory,       "rebalance_bars": 6},
        {"name": "abs_mom",                "gen_factory": abs_mom_factory,        "rebalance_bars": 6},
        {"name": "btc_ma_180",             "gen_factory": btc_ma180_factory,      "rebalance_bars": 6},
        {"name": "btc_ma_360",             "gen_factory": btc_ma360_factory,      "rebalance_bars": 6},
        {"name": "btc_mom_pos",            "gen_factory": btc_mom_factory,        "rebalance_bars": 6},
        {"name": "reb_12",                 "gen_factory": base_gen_factory,       "rebalance_bars": 12},
        {"name": "reb_18",                 "gen_factory": base_gen_factory,       "rebalance_bars": 18},
        {"name": "reb_24",                 "gen_factory": base_gen_factory,       "rebalance_bars": 24},
        {"name": "reb_42",                 "gen_factory": base_gen_factory,       "rebalance_bars": 42},
        {"name": "turnover_buffer",        "gen_factory": buffer_factory,         "rebalance_bars": 6},
        {"name": "min_trade_2_5",          "gen_factory": min25_factory,          "rebalance_bars": 6},
        {"name": "min_trade_5",            "gen_factory": min5_factory,           "rebalance_bars": 6},
        {"name": "min_trade_10",           "gen_factory": min10_factory,          "rebalance_bars": 6},
        # Combinations
        {"name": "abs_mom_reb12",          "gen_factory": abs_mom_factory,        "rebalance_bars": 12},
        {"name": "abs_mom_reb24",          "gen_factory": abs_mom_factory,        "rebalance_bars": 24},
        {"name": "btc_ma180_reb12",        "gen_factory": btc_ma180_factory,      "rebalance_bars": 12},
        {"name": "btc_ma180_reb24",        "gen_factory": btc_ma180_factory,      "rebalance_bars": 24},
        {"name": "abs_mom_buffer_reb12",   "gen_factory": abs_mom_buf_factory,    "rebalance_bars": 12},
        {"name": "btc_ma180_buffer_reb12", "gen_factory": btc_ma180_buf_factory,  "rebalance_bars": 12},
    ]


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run_all_variants(
    data_dir: Path = Path("data/local"),
    output_dir: Path = Path("reports"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all variants and return (comparison_df, by_period_df, gross_vs_net_df)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading close matrix for %d symbols…", len(LIVE_FIVE_UNIVERSE))
    close = build_close_matrix(LIVE_FIVE_UNIVERSE, "4h", data_dir)

    joint_start = find_joint_eligible_start(close, DEFAULT_MIN_HISTORY_BARS)
    if joint_start is None:
        raise ValueError("Cannot find joint eligible start for FIXED_COMMON_HISTORY")

    LOGGER.info("FIXED_COMMON_HISTORY joint start: %s", joint_start)

    # Restrict close to fixed common history window
    close_fixed = close.loc[close.index >= joint_start]
    ohlcv = _close_to_ohlcv(close_fixed)

    # ---------- Part 1: Gross vs Net Baseline ----------
    LOGGER.info("Running baseline backtest for gross vs net analysis…")
    baseline_gen = _make_signal_generator(
        DEFAULT_TOP_N, DEFAULT_MIN_HISTORY_BARS, DEFAULT_SHORT_LOOKBACK, DEFAULT_MEDIUM_LOOKBACK
    )
    baseline_result = run_backtest(
        ohlcv=ohlcv,
        signal_generator=baseline_gen,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        transaction_cost_bps=DEFAULT_FEE_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        rebalance_every_bars=DEFAULT_REBALANCE_BARS,
    )

    gross_net_df = run_gross_vs_net_analysis(baseline_result, DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS)
    gross_net_path = output_dir / "fixed_five_gross_vs_net_cost_analysis.csv"
    gross_net_df.to_csv(gross_net_path, index=False)
    LOGGER.info("Saved %s", gross_net_path)

    # ---------- Part 2-5: All Variants ----------

    # Define factory lambdas (fresh instance per run)
    def base_gen_factory():
        return BaseSignalGen()

    def abs_mom_factory():
        return AbsMomSignalGen()

    def btc_ma180_factory():
        return BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=180)

    def btc_ma360_factory():
        return BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=360)

    def btc_mom_factory():
        return BTCRegimeSignalGen(gate_type="mom", btc_ma_bars=None)

    def buffer_factory():
        return TurnoverBufferSignalGen(base_gen=BaseSignalGen(), buffer_top_n=4, use_abs_mom=False)

    def min25_factory():
        return MinTradeSignalGen(base_gen=BaseSignalGen(), threshold=0.025)

    def min5_factory():
        return MinTradeSignalGen(base_gen=BaseSignalGen(), threshold=0.05)

    def min10_factory():
        return MinTradeSignalGen(base_gen=BaseSignalGen(), threshold=0.10)

    def abs_mom_buf_factory():
        return TurnoverBufferSignalGen(base_gen=AbsMomSignalGen(), buffer_top_n=4, use_abs_mom=True)

    def btc_ma180_buf_factory():
        return TurnoverBufferSignalGen(
            base_gen=BTCRegimeSignalGen(gate_type="ma", btc_ma_bars=180),
            buffer_top_n=4,
            use_abs_mom=False,
        )

    variants = _make_variants(
        base_gen_factory, abs_mom_factory, btc_ma180_factory,
        btc_ma360_factory, btc_mom_factory, buffer_factory,
        min25_factory, min5_factory, min10_factory,
        abs_mom_buf_factory, btc_ma180_buf_factory,
    )

    all_metrics: list[dict[str, Any]] = []
    all_period_metrics: list[dict[str, Any]] = []
    all_results: dict[str, BacktestResult] = {}

    for variant in variants:
        name = variant["name"]
        reb_bars = variant["rebalance_bars"]
        gen = variant["gen_factory"]()
        LOGGER.info("Running variant: %s (reb_bars=%d)…", name, reb_bars)

        try:
            result = run_backtest(
                ohlcv=ohlcv,
                signal_generator=gen,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                transaction_cost_bps=DEFAULT_FEE_BPS,
                slippage_bps=DEFAULT_SLIPPAGE_BPS,
                rebalance_every_bars=reb_bars,
            )
            all_results[name] = result

            m = compute_variant_metrics(
                result, close_fixed, joint_start, DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
            )
            m["name"] = name
            all_metrics.append(m)

            for period_name, p_start, p_end in PERIODS:
                pm = compute_period_metrics(result, period_name, p_start, p_end)
                pm["variant"] = name
                all_period_metrics.append(pm)

        except Exception as exc:
            LOGGER.error("Variant %s failed: %s", name, exc)
            all_metrics.append({"name": name, "error": str(exc)})

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)
    baseline_row = comparison_df[comparison_df["name"] == "baseline"].iloc[0].to_dict() if "baseline" in comparison_df["name"].values else {}

    # Classify robust candidates
    robust_labels = []
    notes_list = []
    for _, row in comparison_df.iterrows():
        if "error" in row or pd.isna(row.get("max_drawdown", float("nan"))):
            robust_labels.append("ERROR")
            notes_list.append(row.get("error", "unknown error"))
        elif row["name"] == "baseline":
            robust_labels.append("BASELINE")
            notes_list.append(f"{SURVIVORSHIP_BIAS_NOTE[:60]}…")
        else:
            label, note = classify_robust_candidate(row.to_dict(), baseline_row)
            robust_labels.append(label)
            notes_list.append(note)

    comparison_df["robust_candidate"] = robust_labels
    comparison_df["notes"] = notes_list

    # Reorder columns
    desired_cols = [
        "name", "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "worst_year_return", "worst_month_return", "avg_annual_turnover", "total_cost_drag_pct",
        "n_trades", "pct_time_invested", "pct_time_in_cash", "max_consecutive_cash_bars",
        "return_2022", "max_dd_2022", "recovery_from_2021_peak",
        "btc_return_full_period", "btc_max_dd_full_period", "robust_candidate", "notes",
    ]
    available_cols = [c for c in desired_cols if c in comparison_df.columns]
    extra_cols = [c for c in comparison_df.columns if c not in desired_cols]
    comparison_df = comparison_df[available_cols + extra_cols]

    comp_path = output_dir / "fixed_five_defensive_overlay_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    LOGGER.info("Saved %s", comp_path)

    # Build by-period DataFrame
    period_df = pd.DataFrame(all_period_metrics)
    period_cols = ["variant", "period", "total_return", "cagr", "sharpe", "max_dd", "cost_drag", "n_trades", "pct_time_in_cash"]
    available_period_cols = [c for c in period_cols if c in period_df.columns]
    period_df = period_df[available_period_cols]

    period_path = output_dir / "fixed_five_defensive_overlay_by_period.csv"
    period_df.to_csv(period_path, index=False)
    LOGGER.info("Saved %s", period_path)

    return comparison_df, period_df, gross_net_df


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


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


def write_analysis_markdown(
    comparison_df: pd.DataFrame,
    period_df: pd.DataFrame,
    gross_net_df: pd.DataFrame,
    output_dir: Path = Path("reports"),
) -> Path:
    """Write the Markdown narrative analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fixed_five_defensive_overlay_analysis.md"

    baseline_row = comparison_df[comparison_df["name"] == "baseline"].iloc[0] if "baseline" in comparison_df["name"].values else None
    robust_candidates = comparison_df[comparison_df["robust_candidate"] == "ROBUST_CANDIDATE"]

    # Best by max DD improvement
    non_baseline = comparison_df[comparison_df["name"] != "baseline"].copy()
    if not non_baseline.empty and "max_drawdown" in non_baseline.columns:
        non_baseline = non_baseline.dropna(subset=["max_drawdown"])
        if not non_baseline.empty:
            best_dd_idx = non_baseline["max_drawdown"].idxmax()  # least negative = best
            best_dd_var = non_baseline.loc[best_dd_idx]
        else:
            best_dd_var = None
    else:
        best_dd_var = None

    # Best independent (non-combination) variant
    independent_names = {"abs_mom", "btc_ma_180", "btc_ma_360", "btc_mom_pos", "reb_12", "reb_18", "reb_24", "reb_42",
                         "turnover_buffer", "min_trade_2_5", "min_trade_5", "min_trade_10"}
    combination_names = {"abs_mom_reb12", "abs_mom_reb24", "btc_ma180_reb12", "btc_ma180_reb24",
                         "abs_mom_buffer_reb12", "btc_ma180_buffer_reb12"}
    indep_df = non_baseline[non_baseline["name"].isin(independent_names)].dropna(subset=["max_drawdown"])
    combo_df = non_baseline[non_baseline["name"].isin(combination_names)].dropna(subset=["max_drawdown"])

    best_indep = indep_df.loc[indep_df["max_drawdown"].idxmax()] if not indep_df.empty else None
    best_combo = combo_df.loc[combo_df["max_drawdown"].idxmax()] if not combo_df.empty else None

    baseline_max_dd = float(baseline_row["max_drawdown"]) if baseline_row is not None and "max_drawdown" in baseline_row else float("nan")
    any_beats_797 = any(float(r["max_drawdown"]) > -0.797 for _, r in non_baseline.iterrows() if not pd.isna(r.get("max_drawdown", float("nan"))))

    # Improvement persists outside 2022?
    if best_indep is not None and "return_2023" in best_indep:
        persists_note = (
            f"Best independent ({best_indep['name']}): "
            f"2022 return {_fmt_pct(best_indep.get('return_2022'))}, "
            f"2023 return {_fmt_pct(best_indep.get('return_2023', 'n/a'))}."
        )
    else:
        persists_note = "Insufficient data to assess persistence."

    lines = [
        "# Fixed Five — Defensive Overlay & Turnover-Reduction Analysis",
        "",
        "> **RESEARCH ONLY** — No live trading code was modified. No imports from "
        "`brokers/`, `execution/`, or `live/`. All changes are isolated to this "
        "research script.",
        "",
        f"> **{SURVIVORSHIP_BIAS_NOTE}**",
        "",
        "---",
        "",
        "## 1. Cost Drag Definition",
        "",
        "The **21.7% cost drag** figure cited in earlier analyses refers to the **sum of "
        "`cost_rate` values in the `rebalance_log` during the 2021-11-22 → 2022-12-19 "
        "drawdown period**. Each `cost_rate = turnover × (fee_bps + slippage_bps) / 10000`. "
        "The sum approximates total return lost to costs over that period (additive, not "
        "compounded). It is _not_ the difference between gross and net ending equity — "
        "it is the sum of per-rebalance cost fractions over a specific sub-period.",
        "",
        "---",
        "",
        "## 2. Gross vs Net Baseline",
        "",
    ]

    # Gross vs net table
    if not gross_net_df.empty:
        lines.append("| Metric | Gross | Net | Difference |")
        lines.append("|--------|-------|-----|------------|")
        for _, row in gross_net_df.iterrows():
            lines.append(f"| {row['metric']} | {_fmt_f(row['gross_value'], 4)} | {_fmt_f(row['net_value'], 4)} | {_fmt_f(row['difference'], 4)} |")
    lines.append("")

    lines += [
        "---",
        "",
        "## 3. Variant Summary",
        "",
    ]

    # Summary table
    if not comparison_df.empty:
        show_cols = ["name", "total_return", "cagr", "sharpe", "max_drawdown", "avg_annual_turnover",
                     "total_cost_drag_pct", "pct_time_in_cash", "return_2022", "robust_candidate"]
        tbl_cols = [c for c in show_cols if c in comparison_df.columns]
        header = "| " + " | ".join(tbl_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(tbl_cols)) + " |"
        lines.append(header)
        lines.append(separator)
        for _, row in comparison_df.iterrows():
            cells = []
            for c in tbl_cols:
                v = row.get(c, "")
                if c in ("total_return", "cagr", "max_drawdown", "return_2022"):
                    cells.append(_fmt_pct(v))
                elif c in ("sharpe", "avg_annual_turnover"):
                    cells.append(_fmt_f(v, 2))
                elif c in ("total_cost_drag_pct", "pct_time_in_cash"):
                    cells.append(f"{_fmt_f(v, 1)}%")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "---",
        "",
        "## 4. Robust Candidates",
        "",
        "A variant is **ROBUST_CANDIDATE** only if: (1) max drawdown improves by >5pp over baseline, "
        "(2) Sharpe ≥ baseline − 0.10, (3) reduces turnover or cost, (4) 2022 return better than baseline "
        "AND 2023 return > −30%, (5) <50% time in cash, (6) n_trades > 0.",
        "",
    ]

    if robust_candidates.empty:
        lines.append("**No robust candidates found.** All variants either failed to materially reduce "
                     "drawdown while preserving Sharpe, or exhibited unrealistic cash-switching behavior.")
    else:
        lines.append(f"Found **{len(robust_candidates)} robust candidate(s)**:")
        lines.append("")
        for _, row in robust_candidates.iterrows():
            lines.append(
                f"- **{row['name']}**: max_dd={_fmt_pct(row.get('max_drawdown'))}, "
                f"sharpe={_fmt_f(row.get('sharpe'))}, 2022={_fmt_pct(row.get('return_2022'))}, "
                f"pct_cash={_fmt_f(row.get('pct_time_in_cash', 0), 1)}%"
            )
    lines.append("")

    lines += [
        "---",
        "",
        "## 5. Interpretation",
        "",
    ]

    if any_beats_797:
        lines.append(f"✅ **At least one variant materially reduced the −79.7% max drawdown** "
                     f"(baseline max_dd={_fmt_pct(baseline_max_dd)}).")
    else:
        lines.append(f"❌ **No variant materially reduced the −79.7% baseline max drawdown** "
                     f"(baseline={_fmt_pct(baseline_max_dd)}). The deep 2022 bear market drove "
                     "losses that regime filters and turnover reduction alone could not prevent entirely.")
    lines.append("")

    if best_indep is not None:
        lines.append(
            f"**Best independent variant:** `{best_indep['name']}` — "
            f"max_dd={_fmt_pct(best_indep.get('max_drawdown'))}, "
            f"sharpe={_fmt_f(best_indep.get('sharpe'))}, "
            f"2022={_fmt_pct(best_indep.get('return_2022'))}."
        )
    else:
        lines.append("**Best independent variant:** N/A")
    lines.append("")

    if best_combo is not None:
        lines.append(
            f"**Best predefined combination:** `{best_combo['name']}` — "
            f"max_dd={_fmt_pct(best_combo.get('max_drawdown'))}, "
            f"sharpe={_fmt_f(best_combo.get('sharpe'))}, "
            f"2022={_fmt_pct(best_combo.get('return_2022'))}."
        )
    else:
        lines.append("**Best predefined combination:** N/A")
    lines.append("")

    lines.append(f"**Persistence outside 2022:** {persists_note}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "**RESEARCH ONLY. No live trading code was modified.** This script imports "
        "only from `research/`, `backtest/`, `strategies/`, and `data/` modules. "
        "Observed improvements in 2022 may not persist in future bear markets of "
        "different character. The FIXED_COMMON_HISTORY universe retains survivorship "
        "bias from present-day symbol selection."
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(
    comparison_df: pd.DataFrame,
    gross_net_df: pd.DataFrame,
    output_files: list[Path],
) -> None:
    baseline = comparison_df[comparison_df["name"] == "baseline"].iloc[0] if "baseline" in comparison_df["name"].values else None
    robust = comparison_df[comparison_df["robust_candidate"] == "ROBUST_CANDIDATE"]

    non_baseline = comparison_df[comparison_df["name"] != "baseline"].dropna(subset=["max_drawdown"] if "max_drawdown" in comparison_df.columns else [])
    best_indep_row = None
    best_combo_row = None
    independent_names = {"abs_mom", "btc_ma_180", "btc_ma_360", "btc_mom_pos", "reb_12", "reb_18", "reb_24", "reb_42",
                         "turnover_buffer", "min_trade_2_5", "min_trade_5", "min_trade_10"}
    combination_names = {"abs_mom_reb12", "abs_mom_reb24", "btc_ma180_reb12", "btc_ma180_reb24",
                         "abs_mom_buffer_reb12", "btc_ma180_buffer_reb12"}
    if not non_baseline.empty and "max_drawdown" in non_baseline.columns:
        indep = non_baseline[non_baseline["name"].isin(independent_names)]
        combo = non_baseline[non_baseline["name"].isin(combination_names)]
        if not indep.empty:
            best_indep_row = indep.loc[indep["max_drawdown"].idxmax()]
        if not combo.empty:
            best_combo_row = combo.loc[combo["max_drawdown"].idxmax()]

    any_beats = False
    if not non_baseline.empty and "max_drawdown" in non_baseline.columns:
        any_beats = any(float(v) > -0.797 for v in non_baseline["max_drawdown"].dropna())

    print("\n" + "=" * 72)
    print("FIXED FIVE — DEFENSIVE OVERLAY RESEARCH SUMMARY")
    print("=" * 72)

    print("\n── 21.7% Cost Drag Definition ──────────────────────────────────────")
    print("  Sum of cost_rate values in rebalance_log during the 2021-11-22 →")
    print("  2022-12-19 drawdown period (additive, not compounded).")
    print("  cost_rate = turnover × (fee_bps + slippage_bps) / 10000.")

    if not gross_net_df.empty:
        print("\n── Gross vs Net Baseline ────────────────────────────────────────────")
        for _, row in gross_net_df.iterrows():
            print(f"  {row['metric']:<50s} gross={_fmt_f(row['gross_value'], 4)}  net={_fmt_f(row['net_value'], 4)}  diff={_fmt_f(row['difference'], 4)}")

    if baseline is not None:
        print("\n── Baseline Metrics ─────────────────────────────────────────────────")
        print(f"  total_return : {_fmt_pct(baseline.get('total_return'))}")
        print(f"  cagr         : {_fmt_pct(baseline.get('cagr'))}")
        print(f"  sharpe       : {_fmt_f(baseline.get('sharpe'))}")
        print(f"  max_drawdown : {_fmt_pct(baseline.get('max_drawdown'))}")
        print(f"  return_2022  : {_fmt_pct(baseline.get('return_2022'))}")

    print("\n── Best Variants (by max DD improvement) ────────────────────────────")
    if best_indep_row is not None:
        print(f"  Best independent : {best_indep_row['name']:<25s}  "
              f"max_dd={_fmt_pct(best_indep_row.get('max_drawdown'))}  "
              f"sharpe={_fmt_f(best_indep_row.get('sharpe'))}  "
              f"2022={_fmt_pct(best_indep_row.get('return_2022'))}")
    else:
        print("  Best independent : N/A")
    if best_combo_row is not None:
        print(f"  Best combination : {best_combo_row['name']:<25s}  "
              f"max_dd={_fmt_pct(best_combo_row.get('max_drawdown'))}  "
              f"sharpe={_fmt_f(best_combo_row.get('sharpe'))}  "
              f"2022={_fmt_pct(best_combo_row.get('return_2022'))}")
    else:
        print("  Best combination : N/A")

    print("\n── Key Questions ────────────────────────────────────────────────────")
    print(f"  Any variant beats −79.7% max DD?  {'YES ✅' if any_beats else 'NO ❌'}")
    print(f"  Robust candidates: {len(robust)}")

    print("\n── LIVE BEHAVIOR UNCHANGED ✓ ────────────────────────────────────────")
    print("  No live, broker, execution, or config files were modified.")

    print("\n── Generated Files ──────────────────────────────────────────────────")
    for p in output_files:
        print(f"  {p}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path("data/local")
    output_dir = Path("reports")

    comparison_df, period_df, gross_net_df = run_all_variants(data_dir, output_dir)
    md_path = write_analysis_markdown(comparison_df, period_df, gross_net_df, output_dir)

    output_files = [
        output_dir / "fixed_five_gross_vs_net_cost_analysis.csv",
        output_dir / "fixed_five_defensive_overlay_comparison.csv",
        output_dir / "fixed_five_defensive_overlay_by_period.csv",
        md_path,
    ]
    print_summary(comparison_df, gross_net_df, output_files)


if __name__ == "__main__":
    main()
