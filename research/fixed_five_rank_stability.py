"""Rank-stability and turnover-control study for FIXED_FIVE combo_entry2_imm_buf05.

PURPOSE
-------
Research-only study investigating whether rank-stability rules (rank buffer,
challenger confirmation, score hurdle, minimum hold, cooldown) can reduce
rank-replacement turnover costs without degrading risk-adjusted performance.

FROZEN CONTROL
--------------
combo_entry2_imm_buf05:  BTC MA-240 bars, rebalance every 12 bars,
entry_confirm_bars=2, exit_confirm_bars=1, entry_buffer_pct=0.5
TOP_N=3 equal-weight; strict top-3 replacement baseline.

SCORE UNIT DOCUMENTATION
-------------------------
One raw score unit = weighted average of short and medium lookback returns
(both 0.5 weight).  A score of 0.025 means the coin returned ~2.5% more
than the alternative over the lookback period.  These are NOT percentage
points; they are decimal return values.

RESEARCH ONLY — no live trading code is touched or imported.

Usage
-----
    .venv/bin/python -m research.fixed_five_rank_stability
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
from research.universe_integrity_analysis import (
    BARS_PER_YEAR_4H,
    DEFAULT_FEE_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MIN_HISTORY_BARS,
    DEFAULT_REBALANCE_BARS,
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
    _period_rebalance_log,
)
from research.fixed_five_canonical_regime_comparison import run_canonical, PERIODS
from research.fixed_five_whipsaw_control import (
    WhipsawControl,
    WhipsawControlledSignalGen,
    compute_controlled_gate_series,
    count_whipsaw_clusters,
)
from strategies.cross_sectional_momentum import compute_momentum_score

LOGGER = logging.getLogger(__name__)

BTC_COL = "BTC/USD"
BARS_PER_YEAR = BARS_PER_YEAR_4H  # 2190

# Frozen candidate parameters
CONTROL_MA_BARS = 240
CONTROL_REBALANCE_BARS = 12
TOP_N = 3
SHORT_LOOKBACK = 12   # bars
MEDIUM_LOOKBACK = 36  # bars
FEE_BPS = 10
SLIPPAGE_BPS = 5
INITIAL_CAPITAL = 10_000.0
MIN_HISTORY_BARS = 36


# ---------------------------------------------------------------------------
# RankStabilityParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class RankStabilityParams:
    """Parameters for one rank-stability variant."""
    name: str
    description: str = ""
    rank_buffer: int = 0              # 0=strict top-N; 4=retain if rank<=4; 5=retain if rank<=5
    challenger_confirm_rebs: int = 0  # require challenger in top-N for N consecutive rebalances
    score_hurdle: float = 0.0         # min raw-score-unit advantage challenger needs over incumbent
    min_hold_rebs: int = 0            # min rebalances to hold before rank-replacement (regime exit overrides)
    cooldown_rebs: int = 0            # rebalances before repurchasing sold symbol via rank replacement


# ---------------------------------------------------------------------------
# Variant definitions (18 total)
# ---------------------------------------------------------------------------

VARIANTS: list[RankStabilityParams] = [
    # Control (combo_entry2_imm_buf05 with strict replacement)
    RankStabilityParams("control", "combo_entry2_imm_buf05 strict top-3 replacement"),

    # Part 2A — RANK_BUFFER
    RankStabilityParams("rank_buffer_4", "retain if rank<=4", rank_buffer=4),
    RankStabilityParams("rank_buffer_5", "retain if rank<=5", rank_buffer=5),

    # Part 2B — CHALLENGER_CONFIRMATION
    RankStabilityParams("challenger_confirm_2", "challenger must be in top-3 for 2 prior rebalances", challenger_confirm_rebs=2),
    RankStabilityParams("challenger_confirm_3", "challenger must be in top-3 for 3 prior rebalances", challenger_confirm_rebs=3),

    # Part 2C — SCORE_ADVANTAGE_HURDLE
    RankStabilityParams("score_hurdle_025", "score hurdle 0.025 raw units", score_hurdle=0.025),
    RankStabilityParams("score_hurdle_050", "score hurdle 0.050 raw units", score_hurdle=0.050),
    RankStabilityParams("score_hurdle_100", "score hurdle 0.100 raw units", score_hurdle=0.100),

    # Part 2D — MINIMUM_HOLDING_PERIOD
    RankStabilityParams("min_hold_2", "min hold 2 rebalances before rank replacement", min_hold_rebs=2),
    RankStabilityParams("min_hold_4", "min hold 4 rebalances before rank replacement", min_hold_rebs=4),
    RankStabilityParams("min_hold_6", "min hold 6 rebalances before rank replacement", min_hold_rebs=6),

    # Part 2E — REPLACEMENT_COOLDOWN
    RankStabilityParams("cooldown_1", "cooldown 1 rebalance after rank-replacement sell", cooldown_rebs=1),
    RankStabilityParams("cooldown_2", "cooldown 2 rebalances after rank-replacement sell", cooldown_rebs=2),
    RankStabilityParams("cooldown_3", "cooldown 3 rebalances after rank-replacement sell", cooldown_rebs=3),

    # Part 3 — LIMITED COMBINATIONS (only these 4)
    RankStabilityParams("combo_buf4_conf2", "rank_buffer=4 + challenger_confirm=2", rank_buffer=4, challenger_confirm_rebs=2),
    RankStabilityParams("combo_buf4_hurdle025", "rank_buffer=4 + score_hurdle=0.025", rank_buffer=4, score_hurdle=0.025),
    RankStabilityParams("combo_conf2_hurdle025", "challenger_confirm=2 + score_hurdle=0.025", challenger_confirm_rebs=2, score_hurdle=0.025),
    RankStabilityParams("combo_buf4_conf2_hold2", "rank_buffer=4 + challenger_confirm=2 + min_hold=2", rank_buffer=4, challenger_confirm_rebs=2, min_hold_rebs=2),
]


# ---------------------------------------------------------------------------
# RankStabilitySignalGen
# ---------------------------------------------------------------------------

class RankStabilitySignalGen:
    """Stateful signal generator applying rank-stability rules on top of a gate."""

    def __init__(self, gate_series: pd.Series, params: RankStabilityParams) -> None:
        self._gate = gate_series
        self._params = params
        # State
        self._current_held: set[str] = set()
        self._hold_age: dict[str, int] = {}           # symbol -> rebalances held
        self._challenger_streak: dict[str, int] = {}  # symbol -> consecutive rebalances in top-N while not held
        self._cooldown: dict[str, int] = {}           # symbol -> remaining cooldown count
        self._last_gate: bool = False
        self._scores_df: pd.DataFrame | None = None

        # Suppressed replacement events (filled in post-hoc with forward returns)
        self.suppressed_events: list[dict] = []

    def set_scores(self, scores_df: pd.DataFrame) -> None:
        """Pre-set full-history momentum scores to avoid recomputing."""
        self._scores_df = scores_df

    def __call__(self, close: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        # 1. Check gate
        gate_val = bool(self._gate.get(ts, False))

        # 2. If gate just turned OFF: clear state, return zeros (regime exit overrides everything)
        if not gate_val:
            self._current_held.clear()
            self._hold_age.clear()
            self._challenger_streak.clear()
            # Cooldowns cleared on regime-off so symbol may be purchased immediately on regime-on
            self._cooldown.clear()
            self._last_gate = False
            return pd.Series(0.0, index=close.columns)

        # 3. If gate just turned ON: fresh entry
        if not self._last_gate and gate_val:
            self._current_held.clear()
            self._hold_age.clear()
            self._challenger_streak.clear()
            self._cooldown.clear()
        self._last_gate = True

        # 4. Get scores at ts
        if self._scores_df is not None:
            scores_available = self._scores_df.loc[self._scores_df.index <= ts]
            if scores_available.empty:
                return pd.Series(0.0, index=close.columns)
            score_row = scores_available.iloc[-1]
        else:
            score_row = compute_momentum_score(
                close.loc[close.index <= ts], SHORT_LOOKBACK, MEDIUM_LOOKBACK
            ).iloc[-1]

        # 5. Get eligible symbols (non-NaN score, enough history)
        eligible = score_row.dropna()
        close_up_to = close.loc[close.index <= ts]
        eligible_syms = [s for s in eligible.index
                        if close_up_to[s].dropna().shape[0] >= MIN_HISTORY_BARS]
        if not eligible_syms:
            return pd.Series(0.0, index=close.columns)

        scores = eligible.loc[eligible_syms].sort_values(ascending=False)
        # Rankings: 1=best
        ranked = scores.rank(ascending=False, method="first").astype(int)

        # 6. Compute "strict top-N" (what base strategy would do)
        strict_top = set(scores.head(TOP_N).index.tolist())

        # 7. Decrement cooldowns
        for sym in list(self._cooldown.keys()):
            self._cooldown[sym] -= 1
            if self._cooldown[sym] <= 0:
                del self._cooldown[sym]

        # 8. Apply stability rules to determine actual target holdings
        incumbents_to_keep: set[str] = set()
        incumbents_to_replace: set[str] = set()

        for sym in list(self._current_held):
            sym_rank = int(ranked.get(sym, 999))
            can_replace = True

            # min_hold_rebs: must have held for at least this many rebalances
            if self._params.min_hold_rebs > 0:
                age = self._hold_age.get(sym, 0)
                if age < self._params.min_hold_rebs:
                    can_replace = False

            # rank_buffer: retain if still within buffer rank
            if can_replace and self._params.rank_buffer > 0:
                if sym_rank <= self._params.rank_buffer:
                    can_replace = False

            if can_replace:
                incumbents_to_replace.add(sym)
            else:
                incumbents_to_keep.add(sym)

        # Start target with kept incumbents
        target_held = incumbents_to_keep.copy()

        # 8b. Fill remaining slots with top-N challengers
        available_slots = TOP_N - len(target_held)
        if available_slots > 0:
            challenger_candidates = [s for s in scores.index
                                     if s not in target_held
                                     and s not in self._cooldown]

            filled = 0
            for challenger in challenger_candidates:
                if filled >= available_slots:
                    break

                challenger_rank = int(ranked.get(challenger, 999))

                # Check challenger_confirm_rebs — only applies when replacing incumbents
                # Fresh fills (no incumbents to replace) bypass confirmation so
                # initial portfolio construction is not blocked.
                if self._params.challenger_confirm_rebs > 0 and incumbents_to_replace:
                    streak = self._challenger_streak.get(challenger, 0)
                    if streak < self._params.challenger_confirm_rebs:
                        continue

                # Check score_hurdle against incumbents being replaced
                if self._params.score_hurdle > 0 and incumbents_to_replace:
                    challenger_score = float(scores.get(challenger, -999))
                    incumbent_we_displace = [s for s in incumbents_to_replace if s not in target_held]
                    if incumbent_we_displace:
                        best_incumbent = max(incumbent_we_displace,
                                           key=lambda s: float(scores.get(s, -999)))
                        incumbent_score = float(scores.get(best_incumbent, -999))
                        advantage = challenger_score - incumbent_score
                        if advantage < self._params.score_hurdle:
                            self.suppressed_events.append({
                                "timestamp": ts,
                                "incumbent": best_incumbent,
                                "challenger": challenger,
                                "incumbent_rank": int(ranked.get(best_incumbent, 999)),
                                "challenger_rank": challenger_rank,
                                "incumbent_score": round(incumbent_score, 6),
                                "challenger_score": round(challenger_score, 6),
                                "score_advantage": round(advantage, 6),
                                "reason": "score_hurdle",
                                "incumbent_return_next_reb": None,
                                "challenger_return_next_reb": None,
                                "incumbent_return_7d": None,
                                "challenger_return_7d": None,
                                "cost_avoided": None,
                            })
                            continue

                target_held.add(challenger)
                filled += 1

            # If slots still unfilled, try incumbents being replaced
            for sym in sorted(incumbents_to_replace, key=lambda s: float(scores.get(s, -999)), reverse=True):
                if len(target_held) >= TOP_N:
                    break
                target_held.add(sym)

        # Also add incumbents_to_replace if still have space
        for sym in incumbents_to_replace:
            if sym not in target_held and len(target_held) < TOP_N:
                target_held.add(sym)

        # 9. Update challenger streaks
        for sym in list(ranked.index):
            if sym in target_held or sym not in strict_top:
                if sym in self._challenger_streak and sym not in strict_top:
                    del self._challenger_streak[sym]
            else:
                # sym is in strict_top but NOT in target_held (challenger)
                self._challenger_streak[sym] = self._challenger_streak.get(sym, 0) + 1

        # Clear streaks for symbols no longer in strict_top
        for sym in list(self._challenger_streak.keys()):
            if sym not in strict_top:
                del self._challenger_streak[sym]

        # 10. Record sells due to rank replacement (for cooldown tracking)
        newly_sold = self._current_held - target_held
        for sym in newly_sold:
            if self._params.cooldown_rebs > 0:
                self._cooldown[sym] = self._params.cooldown_rebs

        # 11. Update hold ages
        new_entries = target_held - self._current_held
        for sym in target_held:
            if sym in new_entries:
                self._hold_age[sym] = 0
            else:
                self._hold_age[sym] = self._hold_age.get(sym, 0) + 1
        for sym in list(self._hold_age.keys()):
            if sym not in target_held:
                del self._hold_age[sym]

        # 12. Update current held
        self._current_held = target_held

        # 13. Build target weights (equal weight)
        weights = pd.Series(0.0, index=close.columns)
        if target_held:
            w = 1.0 / len(target_held)
            for sym in target_held:
                if sym in weights.index:
                    weights[sym] = w

        return weights


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_stability_signal(
    gate_series: pd.Series,
    params: RankStabilityParams,
    scores_df: pd.DataFrame,
) -> RankStabilitySignalGen:
    """Build a rank stability signal generator for a given params set."""
    gen = RankStabilitySignalGen(gate_series, params)
    gen.set_scores(scores_df)
    return gen


def build_frozen_candidate_gate(btc_full: pd.Series) -> pd.Series:
    """Build the gate series for combo_entry2_imm_buf05."""
    params = WhipsawControl(
        name="combo_entry2_imm_buf05",
        entry_confirm_bars=2, exit_confirm_bars=1,
        entry_buffer_pct=0.5, exit_buffer_pct=0.0,
    )
    return compute_controlled_gate_series(btc_full, CONTROL_MA_BARS, params)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_variant_metrics_full(
    strategy_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    gate: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    control_metrics: dict | None = None,
) -> dict:
    """Compute all full-period metrics for one rank-stability variant."""
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

    # Fee and slippage dollars
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    gross_eq = eq.iloc[0] * (1 + gross_r).cumprod()
    gross_end = float(gross_eq.iloc[-1])
    net_end = float(eq.iloc[-1])
    gross_minus_net = gross_end - net_end
    total_bps = fee_bps + slippage_bps
    fee_frac = fee_bps / total_bps if total_bps > 0 else 0.5
    slip_frac = slippage_bps / total_bps if total_bps > 0 else 0.5
    fee_dollars = gross_minus_net * fee_frac
    slippage_dollars = gross_minus_net * slip_frac

    # n_rank_replacements: rebalances where turnover > 0.01 and regime was ON
    gate_on_times = set(gate[gate].index.tolist())
    n_rank_replacements = int(
        reb_fx[
            (reb_fx["turnover"] > 0.01) &
            (reb_fx["execution_timestamp"].isin(gate_on_times) |
             reb_fx["signal_timestamp"].isin(gate_on_times))
        ].shape[0]
    )

    # Time invested
    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    pct_invested = float(holdings.sum(axis=1).mean() * 100)
    pct_cash = float((holdings.sum(axis=1) < 0.01).mean() * 100)

    # Regime switches
    gate_fx = gate.reindex(port_fx.index).ffill().fillna(False)
    transitions = gate_fx[gate_fx != gate_fx.shift(1)].dropna()
    n_regime_switches = len(transitions)

    # 2022 return
    eq_2022 = eq.loc[(eq.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                     (eq.index <= pd.Timestamp("2022-12-31 23:59:59", tz="UTC"))]
    ret_2022 = float(eq_2022.iloc[-1] / eq_2022.iloc[0] - 1) * 100 if len(eq_2022) >= 2 else float("nan")

    m: dict[str, Any] = {
        "variant": strategy_name,
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
        "fee_dollars": round(fee_dollars, 2),
        "slippage_dollars": round(slippage_dollars, 2),
        "gross_minus_net_dollars": round(gross_minus_net, 2),
        "n_rebalances": n_rebalances,
        "n_rank_replacements": n_rank_replacements,
        "additive_cost_pct": round(additive_cost_pct, 4),
        "pct_time_invested": round(pct_invested, 2),
        "pct_time_cash": round(pct_cash, 2),
        "n_regime_switches": n_regime_switches,
    }

    if control_metrics:
        ctrl_sharpe = control_metrics.get("sharpe", 0) or 0
        ctrl_dd = control_metrics.get("max_drawdown_pct", -100) or -100
        ctrl_cost = control_metrics.get("additive_cost_pct", 0) or 0
        m["sharpe_vs_control"] = round(v_sharpe - ctrl_sharpe, 4)
        m["dd_vs_control_pp"] = round(v_maxdd * 100 - ctrl_dd, 2)
        m["cost_reduction_pct"] = round(
            (ctrl_cost - additive_cost_pct) / max(abs(ctrl_cost), 1e-9) * 100, 2
        ) if ctrl_cost != 0 else 0.0

    return m


# ---------------------------------------------------------------------------
# Holding duration analysis
# ---------------------------------------------------------------------------

def compute_holding_episodes(
    holdings_history: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
) -> list[dict]:
    """Extract holding episodes from holdings_history.

    Episode: weight transitions from ~0 to >0 (entry) and >0 to ~0 (exit).
    Returns list of dicts with symbol, entry_ts, exit_ts, duration_bars,
    duration_days, episode_gross_return.
    """
    episodes: list[dict] = []
    hh = holdings_history.reindex(close.index).fillna(0)
    hh = hh.loc[hh.index >= joint_start]

    for sym in hh.columns:
        series = hh[sym]
        in_position = series > 0.001
        transitions = in_position.astype(int).diff().fillna(0)
        entries = transitions[transitions == 1].index.tolist()
        exits = transitions[transitions == -1].index.tolist()

        for entry_ts in entries:
            # Find next exit after entry
            future_exits = [t for t in exits if t > entry_ts]
            if future_exits:
                exit_ts = future_exits[0]
            else:
                exit_ts = series.index[-1]  # held to end

            # Duration
            entry_loc = hh.index.get_loc(entry_ts)
            exit_loc = hh.index.get_loc(exit_ts)
            duration_bars = exit_loc - entry_loc
            duration_days = duration_bars * 4 / 24.0  # 4h bars

            # Episode gross return
            if sym in close.columns and entry_ts in close.index and exit_ts in close.index:
                p_entry = float(close.loc[entry_ts, sym])
                p_exit = float(close.loc[exit_ts, sym])
                gross_ret = (p_exit / p_entry - 1) if p_entry > 0 else float("nan")
            else:
                gross_ret = float("nan")

            episodes.append({
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "duration_bars": duration_bars,
                "duration_days": round(duration_days, 2),
                "episode_gross_return": round(gross_ret, 6) if not math.isnan(gross_ret) else float("nan"),
            })

    return episodes


def holding_duration_summary(episodes: list[dict]) -> dict:
    """Summarize holding duration stats from episodes list."""
    if not episodes:
        return {
            "n_episodes": 0,
            "median_duration_bars": float("nan"),
            "median_duration_days": float("nan"),
            "mean_duration_days": float("nan"),
            "pct_lt_4days": float("nan"),
            "pct_lt_7days": float("nan"),
        }
    durations_bars = [e["duration_bars"] for e in episodes]
    durations_days = [e["duration_days"] for e in episodes]
    n = len(durations_days)
    return {
        "n_episodes": n,
        "median_duration_bars": round(float(np.median(durations_bars)), 1),
        "median_duration_days": round(float(np.median(durations_days)), 2),
        "mean_duration_days": round(float(np.mean(durations_days)), 2),
        "pct_lt_4days": round(sum(1 for d in durations_days if d < 4) / n * 100, 2),
        "pct_lt_7days": round(sum(1 for d in durations_days if d < 7) / n * 100, 2),
    }


# ---------------------------------------------------------------------------
# Suppressed events opportunity cost enrichment
# ---------------------------------------------------------------------------

def enrich_suppressed_events(
    suppressed_events: list[dict],
    close: pd.DataFrame,
    rebalance_timestamps: list[pd.Timestamp],
    equity_series: pd.Series,
    fee_bps: float,
    slippage_bps: float,
) -> list[dict]:
    """Enrich suppressed replacement events with forward returns and cost avoided."""
    enriched = []
    reb_list = sorted(rebalance_timestamps)

    for ev in suppressed_events:
        ev = dict(ev)
        ts = ev["timestamp"]
        incumbent = ev["incumbent"]
        challenger = ev["challenger"]

        # Find next rebalance timestamp
        future_rebs = [t for t in reb_list if t > ts]
        next_reb_ts = future_rebs[0] if future_rebs else None

        # +7 days
        ts_7d = ts + pd.Timedelta(days=7)
        # Find nearest close index to ts_7d
        idx_7d = close.index[close.index >= ts_7d]
        ts_7d_actual = idx_7d[0] if len(idx_7d) > 0 else None

        def _safe_ret(sym, t_start, t_end):
            if t_start not in close.index or t_end is None or t_end not in close.index:
                return None
            if sym not in close.columns:
                return None
            p0 = float(close.loc[t_start, sym])
            p1 = float(close.loc[t_end, sym])
            if p0 <= 0:
                return None
            return round(float(p1 / p0 - 1), 6)

        ev["incumbent_return_next_reb"] = _safe_ret(incumbent, ts, next_reb_ts) if next_reb_ts else None
        ev["challenger_return_next_reb"] = _safe_ret(challenger, ts, next_reb_ts) if next_reb_ts else None
        ev["incumbent_return_7d"] = _safe_ret(incumbent, ts, ts_7d_actual)
        ev["challenger_return_7d"] = _safe_ret(challenger, ts, ts_7d_actual)

        # Cost avoided (entry cost only — one leg)
        eq_at_ts = float(equity_series.get(ts, equity_series.iloc[0]))
        weight = 1.0 / TOP_N
        notional = weight * eq_at_ts
        cost_avoided = notional * (fee_bps + slippage_bps) / 10000.0
        ev["cost_avoided"] = round(cost_avoided, 4)

        # return_gained_or_lost: positive = suppression helped (incumbent outperformed)
        if ev["incumbent_return_next_reb"] is not None and ev["challenger_return_next_reb"] is not None:
            ev["return_gained_or_lost"] = round(
                ev["incumbent_return_next_reb"] - ev["challenger_return_next_reb"], 6
            )
        else:
            ev["return_gained_or_lost"] = None

        enriched.append(ev)

    return enriched


def suppressed_events_summary(enriched: list[dict]) -> dict:
    """Summarise all enriched suppressed events across all variants."""
    n = len(enriched)
    if n == 0:
        return {
            "total_suppressions": 0,
            "pct_helped": float("nan"),
            "avg_cost_avoided": float("nan"),
            "total_cost_avoided": 0.0,
            "avg_return_gained": float("nan"),
        }

    helped = [e for e in enriched if e.get("return_gained_or_lost") is not None
              and e["return_gained_or_lost"] > 0]
    cost_vals = [e["cost_avoided"] for e in enriched if e.get("cost_avoided") is not None]
    ret_vals = [e["return_gained_or_lost"] for e in enriched if e.get("return_gained_or_lost") is not None]

    return {
        "total_suppressions": n,
        "pct_helped": round(len(helped) / n * 100, 2) if n > 0 else float("nan"),
        "avg_cost_avoided": round(float(np.mean(cost_vals)), 4) if cost_vals else float("nan"),
        "total_cost_avoided": round(float(sum(cost_vals)), 2) if cost_vals else 0.0,
        "avg_return_gained": round(float(np.mean(ret_vals)) * 100, 4) if ret_vals else float("nan"),
    }


# ---------------------------------------------------------------------------
# By-period metrics
# ---------------------------------------------------------------------------

def compute_period_metrics(
    variant_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    close: pd.DataFrame,
    joint_start: pd.Timestamp,
    period_name: str,
    period_start_str: str | None,
    period_end_str: str | None,
) -> dict:
    """Compute metrics for one (variant, period) slice."""
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

    if len(eq_slice) < 2:
        return {"variant": variant_name, "period": period_name, "error": "insufficient_bars"}

    v_ret = float(eq_slice.iloc[-1] / eq_slice.iloc[0] - 1)
    v_dd = _max_drawdown(eq_slice)
    v_sh = _sharpe(rets_slice, BARS_PER_YEAR)
    n_trades = int((reb_slice["turnover"] > 0.01).sum()) if len(reb_slice) > 0 else 0
    cost_drag = float(reb_slice["cost_rate"].sum() * 100) if len(reb_slice) > 0 else 0.0

    holdings = result.holdings_history.reindex(port_fx.index).fillna(0)
    hld_slice = holdings.loc[(holdings.index >= p_start) & (holdings.index <= p_end)]
    pct_cash = float((hld_slice.sum(axis=1) < 0.01).mean() * 100) if len(hld_slice) > 0 else 0.0

    return {
        "variant": variant_name,
        "period": period_name,
        "period_start": str(p_start.date()),
        "period_end": str(min(p_end, port_fx.index[-1]).date()),
        "total_return_pct": round(v_ret * 100, 2),
        "max_drawdown_pct": round(v_dd * 100, 2),
        "sharpe": round(v_sh, 4),
        "additive_cost_drag_pct": round(cost_drag, 4),
        "n_trades": n_trades,
        "pct_time_in_cash": round(pct_cash, 2),
    }


# ---------------------------------------------------------------------------
# RANK_STABILITY_IMPROVEMENT classification
# ---------------------------------------------------------------------------

def classify_rank_stability_improvement(metrics: dict, control_metrics: dict) -> str:
    """Classify a variant's rank stability impact.

    Returns: RANK_STABILITY_IMPROVEMENT | MIXED_TRADEOFF | WEAK_EVIDENCE | UNFAVORABLE
    """
    ctrl_cost = float(control_metrics.get("additive_cost_pct", 0) or 0)
    v_cost = float(metrics.get("additive_cost_pct", ctrl_cost) or ctrl_cost)
    cost_reduction_pct = (ctrl_cost - v_cost) / max(abs(ctrl_cost), 1e-9) * 100

    ctrl_replacements = float(control_metrics.get("n_rank_replacements", 1) or 1)
    v_replacements = float(metrics.get("n_rank_replacements", ctrl_replacements) or ctrl_replacements)
    replacement_reduction_pct = (ctrl_replacements - v_replacements) / max(ctrl_replacements, 1) * 100

    ctrl_sharpe = float(control_metrics.get("sharpe", 0) or 0)
    v_sharpe = float(metrics.get("sharpe", 0) or 0)
    sharpe_diff = v_sharpe - ctrl_sharpe

    ctrl_dd = float(control_metrics.get("max_drawdown_pct", -100) or -100)
    v_dd = float(metrics.get("max_drawdown_pct", -100) or -100)
    dd_diff = v_dd - ctrl_dd  # positive = less negative = better

    ctrl_2022 = float(control_metrics.get("return_2022_pct", 0) or 0)
    v_2022 = float(metrics.get("return_2022_pct", ctrl_2022) or ctrl_2022)
    r2022_diff = v_2022 - ctrl_2022

    pct_cash = float(metrics.get("pct_time_cash", 0) or 0)

    # All criteria for RANK_STABILITY_IMPROVEMENT
    criteria = {
        "cost_reduced_25pct": cost_reduction_pct >= 25.0,
        "replacements_reduced_35pct": replacement_reduction_pct >= 35.0,
        "sharpe_within_0_10": sharpe_diff >= -0.10,
        "dd_not_worse_5pp": dd_diff >= -5.0,
        "r2022_not_worse_3pp": r2022_diff >= -3.0,
        "not_predominantly_cash": pct_cash < 80.0,
        # Structural guarantees: one-bar execution delay preserved, no future info
        # These are always true by design of this module
    }

    all_pass = all(criteria.values())
    if all_pass:
        return "RANK_STABILITY_IMPROVEMENT"

    failed = [k for k, v in criteria.items() if not v]
    n_failed = len(failed)

    # UNFAVORABLE: worsens drawdown significantly or sharpe badly hurt
    if dd_diff < -5.0 or sharpe_diff < -0.20:
        return "UNFAVORABLE"

    # MIXED_TRADEOFF: cost reduced but something else meaningfully hurt
    if cost_reduction_pct >= 10.0 and n_failed <= 2:
        return "MIXED_TRADEOFF"

    return "WEAK_EVIDENCE"


# ---------------------------------------------------------------------------
# Cost accounting CSV
# ---------------------------------------------------------------------------

def compute_cost_row(
    variant_name: str,
    result: BacktestResult,
    port_fx: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    joint_start: pd.Timestamp,
) -> dict:
    """Detailed cost accounting row for one variant."""
    eq = port_fx["equity"].dropna()
    equity_at_start = float(eq.iloc[0])
    gross_r = result.gross_return.reindex(eq.index).fillna(0)
    gross_eq = equity_at_start * (1 + gross_r).cumprod()
    gross_end = float(gross_eq.iloc[-1])
    net_end = float(eq.iloc[-1])
    gross_minus_net = gross_end - net_end

    total_bps = fee_bps + slippage_bps
    fee_frac = fee_bps / total_bps if total_bps > 0 else 0.5
    slip_frac = slippage_bps / total_bps if total_bps > 0 else 0.5

    reb_fx = result.rebalance_log[
        result.rebalance_log["execution_timestamp"] >= joint_start
    ]
    additive_sum = float(reb_fx["cost_rate"].sum())
    n_reb = len(reb_fx)
    ann_turn = float(reb_fx["turnover"].sum())
    simple_cost_est = equity_at_start * additive_sum
    foregone_comp = gross_minus_net - simple_cost_est

    return {
        "variant": variant_name,
        "equity_at_start": round(equity_at_start, 2),
        "gross_ending_equity": round(gross_end, 2),
        "net_ending_equity": round(net_end, 2),
        "gross_minus_net_dollars": round(gross_minus_net, 2),
        "fee_dollars_est": round(gross_minus_net * fee_frac, 2),
        "slippage_dollars_est": round(gross_minus_net * slip_frac, 2),
        "additive_cost_rate_sum_pct": round(additive_sum * 100, 4),
        "simple_cost_estimate_dollars": round(simple_cost_est, 2),
        "foregone_compounding_dollars": round(foregone_comp, 2),
        "n_rebalances": n_reb,
        "total_turnover": round(ann_turn, 4),
    }


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def write_analysis_md(
    comp_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    suppressed_df: pd.DataFrame,
    joint_start: pd.Timestamp,
    control_metrics: dict,
    supp_summary: dict,
    output_dir: Path,
) -> Path:
    candidates = comp_df[comp_df["rank_stability_label"] == "RANK_STABILITY_IMPROVEMENT"]["variant"].tolist()
    best_indep = _find_best_variant(comp_df, exclude_combos=True)
    best_combo = _find_best_variant(comp_df, combos_only=True)

    lines = [
        "# Rank Stability and Turnover Control Study",
        "",
        "## Score Unit Documentation",
        "",
        "One raw score unit = weighted average of short and medium lookback returns (both 0.5 weight).",
        "A score of 0.025 means the coin returned ~2.5% more than the alternative over the lookback",
        "period. These are NOT percentage points; they are decimal return values.",
        "",
        "## Control Configuration",
        "",
        f"- Universe: FIXED_COMMON_HISTORY ({', '.join(LIVE_FIVE_UNIVERSE)})",
        f"- Joint start: {joint_start.date()}",
        "- Frozen candidate: combo_entry2_imm_buf05",
        "- BTC MA: 240 bars (≈40 calendar days), rebalance every 12 bars",
        "- entry_confirm_bars=2, exit_confirm_bars=1, entry_buffer_pct=0.5",
        "- TOP_N=3 equal-weight",
        "- Execution delay: 1 bar (4 hours)",
        f"- Fees: {FEE_BPS} bps | Slippage: {SLIPPAGE_BPS} bps",
        f"- ⚠ {SURVIVORSHIP_BIAS_NOTE}",
        "",
        "## Control Confirmation",
        "",
    ]

    for k, v in control_metrics.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Full-Period Results Summary", ""]

    if not comp_df.empty:
        display_cols = ["variant", "total_return_pct", "cagr_pct", "sharpe",
                        "max_drawdown_pct", "return_2022_pct", "additive_cost_pct",
                        "n_rank_replacements", "avg_annual_turnover", "rank_stability_label"]
        available = [c for c in display_cols if c in comp_df.columns]
        lines.append("| " + " | ".join(available) + " |")
        lines.append("|" + "|".join(["---"] * len(available)) + "|")
        for _, row in comp_df.iterrows():
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
        "## RANK_STABILITY_IMPROVEMENT Candidates",
        "",
        (", ".join(candidates) if candidates else "None found"),
        "",
        f"## Best Independent Rule: {best_indep}",
        "",
        f"## Best Predefined Combination: {best_combo}",
        "",
        "## Opportunity Cost Analysis",
        "",
        f"- Total suppressions: {supp_summary['total_suppressions']}",
        f"- % suppressions that helped (incumbent outperformed): {supp_summary['pct_helped']}%",
        f"- Avg cost avoided per suppression: ${supp_summary['avg_cost_avoided']}",
        f"- Total cost avoided: ${supp_summary['total_cost_avoided']:,.2f}",
        f"- Avg return gained/lost: {supp_summary['avg_return_gained']}%",
        "",
        "## In-Sample Note",
        "",
        "All results are in-sample. combo_entry2_imm_buf05 was selected from the full historical",
        "sample and remains the frozen candidate. No parameter changes in this task constitute",
        "out-of-sample validation.",
        "",
        "## RANK_STABILITY_IMPROVEMENT Classification Criteria",
        "",
        "All criteria must be true:",
        "1. Direct costs reduced by >= 25%",
        "2. Rank replacements reduced by >= 35%",
        "3. Sharpe within 0.10 of control OR improved",
        "4. Max drawdown not worsened by > 5pp",
        "5. 2022 return not materially worse (< 3pp worse)",
        "6. NOT achieved by staying permanently in cash",
        "7. Regime exit never delayed (structural guarantee)",
        "8. One-bar execution delay preserved (structural guarantee)",
        "9. No future information used (structural guarantee)",
        "",
    ]

    path = output_dir / "fixed_five_rank_stability_analysis.md"
    path.write_text("\n".join(lines))
    return path


def _find_best_variant(comp_df: pd.DataFrame, exclude_combos: bool = False, combos_only: bool = False) -> str:
    """Find best variant by Sharpe, filtered by type."""
    if comp_df.empty:
        return "N/A"
    df = comp_df[comp_df["variant"] != "control"].copy()
    if exclude_combos:
        df = df[~df["variant"].str.startswith("combo_")]
    if combos_only:
        df = df[df["variant"].str.startswith("combo_")]
    if df.empty:
        return "N/A"
    best = df.loc[df["sharpe"].idxmax()]
    return f"{best['variant']} (Sharpe={best['sharpe']:.3f})"


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

    # Build frozen candidate gate
    gate_full = build_frozen_candidate_gate(btc_full)
    gate_fx = gate_full.reindex(
        gate_full.index[gate_full.index >= joint_start]
    ).fillna(False)

    # Pre-compute full-history momentum scores once
    LOGGER.info("Pre-computing momentum scores…")
    scores_df = compute_momentum_score(close, SHORT_LOOKBACK, MEDIUM_LOOKBACK)

    LOGGER.info("Running %d rank-stability variants…", len(VARIANTS))

    all_full_metrics: list[dict] = []
    all_period_metrics: list[dict] = []
    all_cost_rows: list[dict] = []
    all_suppressed: list[dict] = []
    all_holding_summaries: list[dict] = []
    control_metrics: dict | None = None

    for params in VARIANTS:
        LOGGER.info("  %s…", params.name)
        sig_gen = build_stability_signal(gate_full, params, scores_df)
        result, port_fx = run_canonical(
            ohlcv_full, sig_gen, joint_start,
            rebalance_bars=CONTROL_REBALANCE_BARS,
            initial_capital=INITIAL_CAPITAL,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )

        m = compute_variant_metrics_full(
            params.name, result, port_fx, close, joint_start,
            gate_fx, FEE_BPS, SLIPPAGE_BPS,
            control_metrics=control_metrics,
        )
        m["description"] = params.description
        m["rank_buffer"] = params.rank_buffer
        m["challenger_confirm_rebs"] = params.challenger_confirm_rebs
        m["score_hurdle"] = params.score_hurdle
        m["min_hold_rebs"] = params.min_hold_rebs
        m["cooldown_rebs"] = params.cooldown_rebs

        if params.name == "control":
            control_metrics = m.copy()

        all_full_metrics.append(m)

        # Per-period metrics
        for period_name, ps, pe in PERIODS:
            pm = compute_period_metrics(
                params.name, result, port_fx, close, joint_start, period_name, ps, pe,
            )
            all_period_metrics.append(pm)

        # Cost accounting
        cost_row = compute_cost_row(
            params.name, result, port_fx, FEE_BPS, SLIPPAGE_BPS, joint_start
        )
        all_cost_rows.append(cost_row)

        # Holding episodes
        episodes = compute_holding_episodes(result.holdings_history, close, joint_start)
        dur_summary = holding_duration_summary(episodes)
        dur_summary["variant"] = params.name
        all_holding_summaries.append(dur_summary)

        # Enrich suppressed events
        eq_series = port_fx["equity"].dropna()
        reb_ts_list = result.rebalance_log["execution_timestamp"].dropna().tolist()
        enriched = enrich_suppressed_events(
            sig_gen.suppressed_events, close, reb_ts_list, eq_series, FEE_BPS, SLIPPAGE_BPS
        )
        for ev in enriched:
            ev["variant"] = params.name
        all_suppressed.extend(enriched)

    # Add classification labels
    for m in all_full_metrics:
        if m["variant"] == "control":
            m["rank_stability_label"] = "CONTROL"
            continue
        if control_metrics is not None:
            m["rank_stability_label"] = classify_rank_stability_improvement(m, control_metrics)
        else:
            m["rank_stability_label"] = "UNKNOWN"

    # Add median holding duration to full metrics
    hold_by_variant = {h["variant"]: h for h in all_holding_summaries}
    for m in all_full_metrics:
        dur = hold_by_variant.get(m["variant"], {})
        m["median_holding_days"] = dur.get("median_duration_days", float("nan"))

    # Suppressed events summary
    supp_summary = suppressed_events_summary(all_suppressed)

    # Write outputs
    comp_df = pd.DataFrame(all_full_metrics)
    comp_path = output_dir / "fixed_five_rank_stability_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    LOGGER.info("Saved: %s", comp_path)

    period_df = pd.DataFrame(all_period_metrics)
    period_path = output_dir / "fixed_five_rank_stability_by_period.csv"
    period_df.to_csv(period_path, index=False)
    LOGGER.info("Saved: %s", period_path)

    supp_df = pd.DataFrame(all_suppressed) if all_suppressed else pd.DataFrame()
    supp_path = output_dir / "fixed_five_suppressed_replacement_events.csv"
    supp_df.to_csv(supp_path, index=False)
    LOGGER.info("Saved: %s", supp_path)

    cost_df = pd.DataFrame(all_cost_rows)
    cost_path = output_dir / "fixed_five_rank_stability_costs.csv"
    cost_df.to_csv(cost_path, index=False)
    LOGGER.info("Saved: %s", cost_path)

    md_path = write_analysis_md(
        comp_df, period_df, cost_df, supp_df,
        joint_start, control_metrics or {}, supp_summary, output_dir,
    )
    LOGGER.info("Saved: %s", md_path)

    # Print final summary
    ctrl = control_metrics or {}
    candidates = comp_df[comp_df.get("rank_stability_label", pd.Series()) == "RANK_STABILITY_IMPROVEMENT"]["variant"].tolist() if "rank_stability_label" in comp_df.columns else []
    best_indep = _find_best_variant(comp_df, exclude_combos=True)
    best_combo = _find_best_variant(comp_df, combos_only=True)

    print("\n=== RANK STABILITY STUDY COMPLETE ===")
    print(f"Files created: [{comp_path}, {period_path}, {supp_path}, {cost_path}, {md_path}]")
    print("Tests: run manually")

    print("\n=== FROZEN CONTROL (combo_entry2_imm_buf05, strict replacement) ===")
    print(f"Total return: {ctrl.get('total_return_pct', 'N/A')}%")
    print(f"CAGR: {ctrl.get('cagr_pct', 'N/A')}%")
    print(f"Sharpe: {ctrl.get('sharpe', 'N/A')}")
    print(f"Sortino: {ctrl.get('sortino', 'N/A')}")
    print(f"Max drawdown: {ctrl.get('max_drawdown_pct', 'N/A')}%")
    print(f"Calmar: {ctrl.get('calmar', 'N/A')}")
    print(f"2022 return: {ctrl.get('return_2022_pct', 'N/A')}%")
    print(f"Turnover (annual): {ctrl.get('avg_annual_turnover', 'N/A')}")
    fee_d = ctrl.get('fee_dollars', float('nan'))
    slip_d = ctrl.get('slippage_dollars', float('nan'))
    print(f"Fee $: ${fee_d:,.2f}" if not (isinstance(fee_d, float) and math.isnan(fee_d)) else "Fee $: N/A")
    print(f"Slippage $: ${slip_d:,.2f}" if not (isinstance(slip_d, float) and math.isnan(slip_d)) else "Slippage $: N/A")
    print(f"Rank replacements: {ctrl.get('n_rank_replacements', 'N/A')}")
    print(f"Median holding duration: {ctrl.get('median_holding_days', 'N/A')} days")

    print(f"\n=== BEST INDEPENDENT RANK STABILITY RULE ===")
    print(best_indep)

    print(f"\n=== BEST PREDEFINED COMBINATION ===")
    print(best_combo)

    print(f"\n=== RANK_STABILITY_IMPROVEMENT CANDIDATES ===")
    print(", ".join(candidates) if candidates else "None found")

    print(f"\n=== OPPORTUNITY COST SUMMARY ===")
    print(f"Total suppressions: {supp_summary['total_suppressions']}")
    print(f"% suppressions that helped (incumbent outperformed): {supp_summary['pct_helped']}%")
    print(f"Avg cost avoided per suppression: ${supp_summary['avg_cost_avoided']}")
    print(f"Total cost avoided: ${supp_summary['total_cost_avoided']:,.2f}")
    print(f"Avg return gained/lost: {supp_summary['avg_return_gained']}%")

    print("\n=== IN-SAMPLE NOTE ===")
    print("All results are in-sample. combo_entry2_imm_buf05 was selected from the full")
    print("historical sample and remains frozen. No changes here constitute out-of-sample")
    print("validation.")

    print("\n=== LIVE BEHAVIOR UNCHANGED ===")
    print("No live trading files, configuration, cron, or exchange code was modified.")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
