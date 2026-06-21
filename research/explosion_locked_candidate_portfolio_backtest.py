"""Research-only portfolio backtest for the locked explosion continuation candidate.

Frozen signal (not optimized in this pass):
- BUY_AFTER_FOLLOW_THROUGH_STRICT
- HIGH_BREAK_100_BPS
- wait one 4h observation candle
- enter at next candle open when available
- hold 6 candles
- long-only

This module evaluates portfolio construction realism only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.explosion_delayed_entry_study import DelayedEntryStudyConfig, build_second_pass_signals  # noqa: E402
from research.explosion_reversal_event_study import load_ohlcv_data  # noqa: E402


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/explosion_delayed_entry_study/portfolio_backtest")
COST_BPS = (50, 100, 150, 200)


@dataclass(frozen=True)
class LockedSignalConfig:
    follow_rule: str = "HIGH_BREAK_100_BPS"
    hold_bars: int = 6
    wait_bars_after_event: int = 1


@dataclass(frozen=True)
class PortfolioScenario:
    name: str
    max_open_positions: int
    max_total_exposure: float
    symbol_cooldown_bars: int
    cost_bps: int
    rank_method: str = "event_time_rank"
    liquidity_guard: bool = False


def _symbol_frames(full_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        s: g.sort_values("timestamp").reset_index(drop=True)
        for s, g in full_df.groupby("symbol", sort=False)
    }


def _candidate_entries(signals: pd.DataFrame, full_df: pd.DataFrame, cfg: LockedSignalConfig) -> pd.DataFrame:
    col = f"follow_{cfg.follow_rule}"
    chosen = signals[signals[col].astype(bool)].copy()
    if chosen.empty:
        return pd.DataFrame()

    by_symbol = _symbol_frames(full_df)
    rows: list[dict[str, object]] = []

    for _, row in chosen.iterrows():
        sym = str(row["symbol"])
        sdf = by_symbol.get(sym)
        if sdf is None:
            continue

        event_ts = pd.Timestamp(row["timestamp"])
        idx = sdf.index[sdf["timestamp"] == event_ts]
        if len(idx) == 0:
            continue

        event_pos = int(idx[0])
        obs_pos = int(row["observation_pos"])
        entry_pos = obs_pos + 1
        exit_pos = entry_pos + cfg.hold_bars
        if entry_pos >= len(sdf) or exit_pos >= len(sdf):
            continue

        entry_ts = pd.Timestamp(sdf.iloc[entry_pos]["timestamp"])
        exit_ts = pd.Timestamp(sdf.iloc[exit_pos]["timestamp"])
        entry_open = float(sdf.iloc[entry_pos]["open"])
        exit_close = float(sdf.iloc[exit_pos]["close"])

        # Event-time ranking score only.
        liq_rank = float(row.get("liquidity_rank_universe", np.nan))
        vol_spike = float(row.get("volume_spike_ratio", np.nan))
        size_move = float(row.get("ret_4h_pct", np.nan))
        btc_bucket = str(row.get("btc_trend_bucket", "UNKNOWN"))
        btc_bonus = 1.0 if btc_bucket == "UP" else (0.3 if btc_bucket == "SIDEWAYS" else 0.0)

        rank_score = (
            (0.40 * (liq_rank if not pd.isna(liq_rank) else 0.0))
            + (0.30 * min(max(vol_spike, 0.0), 8.0) / 8.0 if not pd.isna(vol_spike) else 0.0)
            + (0.20 * min(max(size_move, 0.0), 25.0) / 25.0 if not pd.isna(size_move) else 0.0)
            + (0.10 * btc_bonus)
        )

        rows.append(
            {
                "event_id": row["event_id"],
                "symbol": sym,
                "event_timestamp": event_ts,
                "observation_timestamp": pd.Timestamp(row["observation_timestamp"]),
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "entry_price": entry_open,
                "exit_price_ref": exit_close,
                "hold_bars": cfg.hold_bars,
                "liquidity_rank_universe": liq_rank,
                "volume_spike_ratio": vol_spike,
                "ret_4h_pct": size_move,
                "btc_trend_bucket": btc_bucket,
                "btc_rolling_vol_pct": float(row.get("btc_rolling_vol_pct", np.nan)),
                "liquidity_bucket": row.get("liquidity_bucket", "UNKNOWN"),
                "dollar_volume": float(row.get("dollar_volume", np.nan)),
                "rank_score": float(rank_score),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["entry_timestamp", "rank_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def _baseline_entries(signals: pd.DataFrame, full_df: pd.DataFrame, hold_bars: int) -> pd.DataFrame:
    # BUY_AT_EXPLOSION baseline, portfolio-realistic entry at next candle open.
    if signals.empty:
        return pd.DataFrame()

    by_symbol = _symbol_frames(full_df)
    rows: list[dict[str, object]] = []

    for _, row in signals.iterrows():
        sym = str(row["symbol"])
        sdf = by_symbol.get(sym)
        if sdf is None:
            continue
        event_ts = pd.Timestamp(row["timestamp"])
        idx = sdf.index[sdf["timestamp"] == event_ts]
        if len(idx) == 0:
            continue

        event_pos = int(idx[0])
        entry_pos = event_pos + 1
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(sdf) or exit_pos >= len(sdf):
            continue

        rows.append(
            {
                "event_id": row["event_id"],
                "symbol": sym,
                "event_timestamp": event_ts,
                "entry_timestamp": pd.Timestamp(sdf.iloc[entry_pos]["timestamp"]),
                "exit_timestamp": pd.Timestamp(sdf.iloc[exit_pos]["timestamp"]),
                "entry_price": float(sdf.iloc[entry_pos]["open"]),
                "exit_price_ref": float(sdf.iloc[exit_pos]["close"]),
            }
        )

    return pd.DataFrame(rows)


def _price_lookup(full_df: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    out: dict[tuple[str, pd.Timestamp], float] = {}
    for _, r in full_df.iterrows():
        out[(str(r["symbol"]), pd.Timestamp(r["timestamp"]))] = float(r["close"])
    return out


def _timeline(full_df: pd.DataFrame, entries: pd.DataFrame) -> list[pd.Timestamp]:
    ts = pd.to_datetime(full_df["timestamp"], utc=True).dropna().unique().tolist()
    if not entries.empty:
        ts.extend(pd.to_datetime(entries["entry_timestamp"], utc=True).dropna().tolist())
        ts.extend(pd.to_datetime(entries["exit_timestamp"], utc=True).dropna().tolist())
    return sorted(pd.unique(pd.Series(ts)))


def _apply_portfolio_constraints(
    pending: pd.DataFrame,
    open_positions: list[dict[str, object]],
    equity: float,
    scenario: PortfolioScenario,
    last_exit_by_symbol: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    if pending.empty:
        return pending

    # Cooldown filter.
    if scenario.symbol_cooldown_bars > 0:
        cooldown = pd.Timedelta(hours=4 * scenario.symbol_cooldown_bars)
        keep = []
        for _, r in pending.iterrows():
            sym = str(r["symbol"])
            ts = pd.Timestamp(r["entry_timestamp"])
            last = last_exit_by_symbol.get(sym)
            if last is None or (ts - last) >= cooldown:
                keep.append(r)
        pending = pd.DataFrame(keep)
        if pending.empty:
            return pending

    # Max position filter.
    cap = scenario.max_open_positions
    if cap < 999999:
        slots = max(cap - len(open_positions), 0)
        if slots <= 0:
            return pending.iloc[0:0].copy()
        pending = pending.sort_values(["rank_score", "liquidity_rank_universe", "symbol"], ascending=[False, False, True]).head(slots)

    # Exposure filter.
    current_notional = float(sum(float(p["units"]) * float(p["entry_price"]) for p in open_positions))
    max_notional = float(equity * scenario.max_total_exposure)
    available = max(max_notional - current_notional, 0.0)
    if available <= 0.0:
        return pending.iloc[0:0].copy()

    pending = pending.copy()
    pending["alloc_notional"] = available / max(len(pending), 1)
    return pending[pending["alloc_notional"] > 0.0].copy()


def run_portfolio_simulation(
    entries: pd.DataFrame,
    full_df: pd.DataFrame,
    scenario: PortfolioScenario,
    initial_capital: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if entries.empty:
        return pd.DataFrame(), pd.DataFrame()

    entries = entries.sort_values(["entry_timestamp", "rank_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    lookup = _price_lookup(full_df)
    times = _timeline(full_df, entries)

    fee_rate = scenario.cost_bps / 20_000.0  # half on entry, half on exit
    cash = initial_capital
    open_positions: list[dict[str, object]] = []
    last_exit_by_symbol: dict[str, pd.Timestamp] = {}

    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for ts in times:
        ts = pd.Timestamp(ts)

        # Exit positions at scheduled timestamp (close-based).
        remaining: list[dict[str, object]] = []
        for pos in open_positions:
            if pd.Timestamp(pos["exit_timestamp"]) == ts:
                px = lookup.get((str(pos["symbol"]), ts), float(pos["exit_price_ref"]))
                gross_notional = float(pos["units"]) * float(px)
                proceeds = gross_notional * (1.0 - fee_rate)
                cash += proceeds
                gross_ret = (px / float(pos["entry_price"]) - 1.0) * 100.0
                net_ret = (proceeds / float(pos["cash_out"]) - 1.0) * 100.0
                trade_rows.append(
                    {
                        **pos,
                        "exit_price": float(px),
                        "gross_return_pct": float(gross_ret),
                        "net_return_pct": float(net_ret),
                        "cost_bps": int(scenario.cost_bps),
                    }
                )
                last_exit_by_symbol[str(pos["symbol"])] = ts
            else:
                remaining.append(pos)
        open_positions = remaining

        # Enter new positions at this timestamp.
        pending = entries[entries["entry_timestamp"] == ts].copy()
        pending = _apply_portfolio_constraints(pending, open_positions, cash + sum(float(p["units"]) * lookup.get((str(p["symbol"]), ts), float(p["entry_price"])) for p in open_positions), scenario, last_exit_by_symbol)

        for _, r in pending.iterrows():
            alloc = float(r["alloc_notional"])
            entry_px = float(r["entry_price"])
            if alloc <= 0 or entry_px <= 0:
                continue

            # Liquidity realism flag at 5% participation.
            dv = float(r.get("dollar_volume", np.nan))
            liquidity_flag = bool((not pd.isna(dv)) and alloc > 0.05 * dv)
            if scenario.liquidity_guard and liquidity_flag:
                continue

            cash_out = alloc * (1.0 + fee_rate)
            if cash_out > cash:
                continue

            units = alloc / entry_px
            cash -= cash_out
            open_positions.append(
                {
                    "scenario": scenario.name,
                    "symbol": str(r["symbol"]),
                    "event_id": str(r["event_id"]),
                    "event_timestamp": pd.Timestamp(r["event_timestamp"]),
                    "entry_timestamp": ts,
                    "exit_timestamp": pd.Timestamp(r["exit_timestamp"]),
                    "entry_price": entry_px,
                    "exit_price_ref": float(r["exit_price_ref"]),
                    "units": float(units),
                    "cash_out": float(cash_out),
                    "entry_notional": float(alloc),
                    "liquidity_flag": liquidity_flag,
                    "btc_trend_bucket": r.get("btc_trend_bucket", "UNKNOWN"),
                    "liquidity_bucket": r.get("liquidity_bucket", "UNKNOWN"),
                    "rank_score": float(r.get("rank_score", np.nan)),
                    "max_open_positions": scenario.max_open_positions,
                    "max_total_exposure": scenario.max_total_exposure,
                    "symbol_cooldown_bars": scenario.symbol_cooldown_bars,
                }
            )

        # Mark-to-market equity at close.
        mtm = float(sum(float(p["units"]) * lookup.get((str(p["symbol"]), ts), float(p["entry_price"])) for p in open_positions))
        equity = float(cash + mtm)
        equity_rows.append(
            {
                "timestamp": ts,
                "scenario": scenario.name,
                "cash": float(cash),
                "market_value": mtm,
                "equity": equity,
                "open_positions": int(len(open_positions)),
                "exposure": float(mtm / equity) if equity > 0 else np.nan,
            }
        )

    trades = pd.DataFrame(trade_rows)
    curve = pd.DataFrame(equity_rows)
    return trades, curve


def _compute_drawdown(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    c = curve.sort_values("timestamp").copy()
    c["peak"] = c["equity"].cummax()
    c["drawdown_pct"] = (c["equity"] / c["peak"] - 1.0) * 100.0
    return c


def _portfolio_summary(curve: pd.DataFrame, trades: pd.DataFrame, scenario: PortfolioScenario) -> dict[str, object]:
    if curve.empty:
        return {}

    c = _compute_drawdown(curve)
    start = float(c["equity"].iloc[0])
    end = float(c["equity"].iloc[-1])
    total_return = (end / start - 1.0) * 100.0 if start > 0 else np.nan

    days = (pd.Timestamp(c["timestamp"].max()) - pd.Timestamp(c["timestamp"].min())).total_seconds() / 86400.0
    years = max(days / 365.25, 1e-9)
    cagr = ((end / start) ** (1.0 / years) - 1.0) * 100.0 if start > 0 else np.nan

    rets = c["equity"].pct_change().dropna()
    vol_ann = float(rets.std() * np.sqrt(6 * 365)) * 100.0 if not rets.empty else np.nan
    sharpe_like = float(rets.mean() / rets.std() * np.sqrt(6 * 365)) if not rets.empty and rets.std() > 0 else np.nan

    avg_exposure = float(c["exposure"].mean())
    time_in_market = float((c["open_positions"] > 0).mean())
    avg_open = float(c["open_positions"].mean())
    max_open = int(c["open_positions"].max())

    n_trades = int(len(trades))
    win_rate = float((trades["net_return_pct"] > 0).mean()) if n_trades else np.nan
    avg_trade = float(trades["net_return_pct"].mean()) if n_trades else np.nan
    med_trade = float(trades["net_return_pct"].median()) if n_trades else np.nan

    turnover = float((trades["entry_notional"].sum() * 2.0) / c["equity"].mean()) if n_trades else 0.0

    return {
        "scenario": scenario.name,
        "cost_bps": scenario.cost_bps,
        "max_open_positions": scenario.max_open_positions,
        "max_total_exposure": scenario.max_total_exposure,
        "symbol_cooldown_bars": scenario.symbol_cooldown_bars,
        "liquidity_guard": scenario.liquidity_guard,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "annualized_volatility_pct": vol_ann,
        "sharpe_like": sharpe_like,
        "max_drawdown_pct": float(c["drawdown_pct"].min()),
        "average_exposure": avg_exposure,
        "time_in_market": time_in_market,
        "average_open_positions": avg_open,
        "max_open_positions_realized": max_open,
        "turnover": turnover,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade_return_pct": avg_trade,
        "median_trade_return_pct": med_trade,
        "liquidity_flag_rate": float(trades["liquidity_flag"].mean()) if n_trades else np.nan,
    }


def _grouped_perf(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    if trades.empty or by not in trades.columns:
        return pd.DataFrame()
    rows = []
    for k, g in trades.groupby(by, dropna=False, sort=False):
        rows.append(
            {
                by: k,
                "n_trades": int(len(g)),
                "win_rate": float((g["net_return_pct"] > 0).mean()),
                "avg_trade_return_pct": float(g["net_return_pct"].mean()),
                "median_trade_return_pct": float(g["net_return_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def _yearly_curve_returns(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    c = curve.copy()
    c["year"] = pd.to_datetime(c["timestamp"], utc=True).dt.year
    rows = []
    for y, g in c.groupby("year"):
        s = float(g["equity"].iloc[0])
        e = float(g["equity"].iloc[-1])
        rows.append({"year": int(y), "return_pct": (e / s - 1.0) * 100.0})
    return pd.DataFrame(rows)


def _monthly_curve_returns(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    c = curve.copy()
    dt = pd.to_datetime(c["timestamp"], utc=True)
    c["month"] = dt.dt.to_period("M").astype(str)
    rows = []
    for m, g in c.groupby("month"):
        s = float(g["equity"].iloc[0])
        e = float(g["equity"].iloc[-1])
        rows.append({"month": m, "return_pct": (e / s - 1.0) * 100.0})
    return pd.DataFrame(rows)


def _build_scenarios() -> list[PortfolioScenario]:
    scenarios: list[PortfolioScenario] = []
    # One-factor sweeps around a fixed anchor to avoid a large Cartesian explosion.
    # Anchor: unlimited positions, 100% exposure, no cooldown.
    scenarios.append(
        PortfolioScenario(
            name="LOCKED_ANCHOR_MAXUNL_EX100_CD0_C50",
            max_open_positions=999999,
            max_total_exposure=1.0,
            symbol_cooldown_bars=0,
            cost_bps=50,
            liquidity_guard=False,
        )
    )

    for m in [1, 3, 5, 10, 999999]:
        scenarios.append(
            PortfolioScenario(
                name=f"LOCKED_MAX{('UNL' if m >= 999999 else m)}_EX100_CD0_C50",
                max_open_positions=m,
                max_total_exposure=1.0,
                symbol_cooldown_bars=0,
                cost_bps=50,
                liquidity_guard=False,
            )
        )

    for ex in [0.25, 0.50, 1.00]:
        scenarios.append(
            PortfolioScenario(
                name=f"LOCKED_MAXUNL_EX{int(ex*100)}_CD0_C50",
                max_open_positions=999999,
                max_total_exposure=ex,
                symbol_cooldown_bars=0,
                cost_bps=50,
                liquidity_guard=False,
            )
        )

    for cd in [0, 1, 3, 6]:
        scenarios.append(
            PortfolioScenario(
                name=f"LOCKED_MAXUNL_EX100_CD{cd}_C50",
                max_open_positions=999999,
                max_total_exposure=1.0,
                symbol_cooldown_bars=cd,
                cost_bps=50,
                liquidity_guard=False,
            )
        )

    for bps in COST_BPS:
        scenarios.append(
            PortfolioScenario(
                name=f"LOCKED_MAXUNL_EX100_CD0_C{bps}",
                max_open_positions=999999,
                max_total_exposure=1.0,
                symbol_cooldown_bars=0,
                cost_bps=bps,
                liquidity_guard=False,
            )
        )

    # Liquidity realism stress scenario.
    scenarios.append(
        PortfolioScenario(
            name="LOCKED_MAXUNL_EX100_CD0_C50_LIQGUARD",
            max_open_positions=999999,
            max_total_exposure=1.0,
            symbol_cooldown_bars=0,
            cost_bps=50,
            liquidity_guard=True,
        )
    )
    # Explicit equal-weight all-signals anchor.
    scenarios.append(
        PortfolioScenario(
            name="LOCKED_EQUAL_WEIGHT_ALL_SIGNALS_C50",
            max_open_positions=999999,
            max_total_exposure=1.0,
            symbol_cooldown_bars=0,
            cost_bps=50,
            liquidity_guard=False,
        )
    )

    dedup: dict[str, PortfolioScenario] = {}
    for s in scenarios:
        dedup[s.name] = s
    return list(dedup.values())


def _btc_buy_hold_benchmark(full_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, cost_bps: int) -> dict[str, object] | None:
    btc = full_df[full_df["symbol"] == "BTC/USD"].sort_values("timestamp").copy()
    btc = btc[(btc["timestamp"] >= start_ts) & (btc["timestamp"] <= end_ts)]
    if len(btc) < 2:
        return None

    entry = float(btc.iloc[0]["open"])
    exitp = float(btc.iloc[-1]["close"])
    gross = (exitp / entry - 1.0) * 100.0
    net = gross - (cost_bps / 100.0)
    return {
        "benchmark": "BTC_BUY_AND_HOLD",
        "cost_bps": cost_bps,
        "total_return_pct": net,
        "gross_return_pct": gross,
        "n_points": int(len(btc)),
    }


def _random_portfolio_benchmark(entries: pd.DataFrame, full_df: pd.DataFrame, cost_bps: int, seed: int = 7) -> dict[str, object] | None:
    if entries.empty:
        return None

    rng = np.random.default_rng(seed)
    by_symbol = _symbol_frames(full_df)

    rows = []
    for sym, count in entries.groupby("symbol").size().items():
        sdf = by_symbol.get(sym)
        if sdf is None:
            continue
        max_start = len(sdf) - 8
        if max_start <= 1:
            continue
        choices = rng.choice(np.arange(1, max_start), size=min(int(count), max_start - 1), replace=False)
        for i in choices:
            entry_pos = int(i)
            exit_pos = entry_pos + 6
            if exit_pos >= len(sdf):
                continue
            entry = float(sdf.iloc[entry_pos]["open"])
            exitp = float(sdf.iloc[exit_pos]["close"])
            rows.append((exitp / entry - 1.0) * 100.0 - (cost_bps / 100.0))

    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return {
        "benchmark": "RANDOM_SIGNAL_PORTFOLIO",
        "cost_bps": cost_bps,
        "total_return_pct": float(np.mean(arr)),
        "gross_return_pct": float(np.mean(arr) + (cost_bps / 100.0)),
        "n_points": int(len(arr)),
    }


def run_portfolio_backtest(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    locked_cfg: LockedSignalConfig | None = None,
) -> dict[str, pd.DataFrame]:
    locked_cfg = locked_cfg or LockedSignalConfig()

    full_df = load_ohlcv_data(data_dir)
    study_cfg = DelayedEntryStudyConfig()
    signals = build_second_pass_signals(full_df, study_cfg)

    locked_entries = _candidate_entries(signals, full_df, locked_cfg)
    baseline_entries = _baseline_entries(signals, full_df, locked_cfg.hold_bars)

    scenarios = _build_scenarios()

    all_trades = []
    all_curves = []
    all_summaries = []

    for s in scenarios:
        trades, curve = run_portfolio_simulation(locked_entries, full_df, s)
        if trades.empty or curve.empty:
            continue
        all_trades.append(trades)
        all_curves.append(curve)
        all_summaries.append(_portfolio_summary(curve, trades, s))

    portfolio_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    portfolio_curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    portfolio_summary = pd.DataFrame(all_summaries)

    # Year/month summaries from default anchor scenario.
    anchor = portfolio_curves[portfolio_curves["scenario"] == "LOCKED_EQUAL_WEIGHT_ALL_SIGNALS_C50"].copy()
    if anchor.empty and not portfolio_curves.empty:
        anchor = portfolio_curves[portfolio_curves["scenario"] == portfolio_curves["scenario"].iloc[0]].copy()
    anchor_trades = portfolio_trades[portfolio_trades["scenario"] == (anchor["scenario"].iloc[0] if not anchor.empty else "")].copy()

    by_year = _yearly_curve_returns(anchor)
    by_month = _monthly_curve_returns(anchor)
    by_btc = _grouped_perf(anchor_trades, "btc_trend_bucket")
    by_liq = _grouped_perf(anchor_trades, "liquidity_bucket")

    cost_sensitivity = portfolio_summary.groupby("cost_bps", as_index=False).agg(
        avg_total_return_pct=("total_return_pct", "mean"),
        best_total_return_pct=("total_return_pct", "max"),
        avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
    ) if not portfolio_summary.empty else pd.DataFrame()

    # Benchmarks at cost 50 for comparability.
    bench_rows = [{"benchmark": "CASH", "cost_bps": 50, "total_return_pct": 0.0, "gross_return_pct": 0.0, "n_points": 0}]
    if not anchor.empty:
        st = pd.Timestamp(anchor["timestamp"].min())
        en = pd.Timestamp(anchor["timestamp"].max())
        b = _btc_buy_hold_benchmark(full_df, st, en, cost_bps=50)
        if b:
            bench_rows.append(b)
    rb = _random_portfolio_benchmark(locked_entries, full_df, cost_bps=50)
    if rb:
        bench_rows.append(rb)

    if not baseline_entries.empty:
        base_scenario = PortfolioScenario(
            name="BUY_AT_EXPLOSION_BASELINE_C50",
            max_open_positions=999999,
            max_total_exposure=1.0,
            symbol_cooldown_bars=0,
            cost_bps=50,
        )
        btr, bcv = run_portfolio_simulation(
            baseline_entries.assign(rank_score=0.0, btc_trend_bucket="UNKNOWN", liquidity_bucket="UNKNOWN"),
            full_df,
            base_scenario,
        )
        if not bcv.empty:
            bsum = _portfolio_summary(bcv, btr, base_scenario)
            bench_rows.append(
                {
                    "benchmark": "BUY_AT_EXPLOSION_BASELINE",
                    "cost_bps": 50,
                    "total_return_pct": float(bsum.get("total_return_pct", np.nan)),
                    "gross_return_pct": float(np.nan),
                    "n_points": int(len(bcv)),
                }
            )

    benchmark_comp = pd.DataFrame(bench_rows)

    if not portfolio_summary.empty:
        anchor_row = portfolio_summary[portfolio_summary["scenario"] == "LOCKED_ANCHOR_MAXUNL_EX100_CD0_C50"]
        if not anchor_row.empty:
            benchmark_comp = pd.concat(
                [
                    benchmark_comp,
                    pd.DataFrame(
                        [
                            {
                                "benchmark": "LOCKED_PORTFOLIO_ANCHOR",
                                "cost_bps": 50,
                                "total_return_pct": float(anchor_row.iloc[0]["total_return_pct"]),
                                "gross_return_pct": float(np.nan),
                                "n_points": int(len(anchor)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    drawdown_summary = pd.DataFrame()
    if not portfolio_curves.empty:
        dd_rows = []
        for sc, g in portfolio_curves.groupby("scenario", sort=False):
            d = _compute_drawdown(g)
            if d.empty:
                continue
            dd_rows.append(
                {
                    "scenario": sc,
                    "max_drawdown_pct": float(d["drawdown_pct"].min()),
                    "time_at_new_high": float((d["equity"] >= d["peak"]).mean()),
                }
            )
        drawdown_summary = pd.DataFrame(dd_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_trades.to_csv(output_dir / "portfolio_trades.csv", index=False)
    portfolio_curves.to_csv(output_dir / "portfolio_equity_curves.csv", index=False)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    by_year.to_csv(output_dir / "portfolio_by_year.csv", index=False)
    by_month.to_csv(output_dir / "portfolio_by_month.csv", index=False)
    by_btc.to_csv(output_dir / "portfolio_by_btc_regime.csv", index=False)
    by_liq.to_csv(output_dir / "portfolio_by_liquidity.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "portfolio_cost_sensitivity.csv", index=False)
    benchmark_comp.to_csv(output_dir / "portfolio_benchmark_comparison.csv", index=False)
    drawdown_summary.to_csv(output_dir / "portfolio_drawdown_summary.csv", index=False)

    write_summary_markdown(
        portfolio_summary=portfolio_summary,
        by_year=by_year,
        by_btc=by_btc,
        cost_sensitivity=cost_sensitivity,
        benchmark_comp=benchmark_comp,
        drawdown_summary=drawdown_summary,
        output_path=output_dir / "summary.md",
    )

    return {
        "portfolio_trades": portfolio_trades,
        "portfolio_equity_curves": portfolio_curves,
        "portfolio_summary": portfolio_summary,
        "portfolio_by_year": by_year,
        "portfolio_by_month": by_month,
        "portfolio_by_btc_regime": by_btc,
        "portfolio_by_liquidity": by_liq,
        "portfolio_cost_sensitivity": cost_sensitivity,
        "portfolio_benchmark_comparison": benchmark_comp,
        "portfolio_drawdown_summary": drawdown_summary,
    }


def write_summary_markdown(
    portfolio_summary: pd.DataFrame,
    by_year: pd.DataFrame,
    by_btc: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    benchmark_comp: pd.DataFrame,
    drawdown_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Locked Candidate Portfolio Backtest (Research-Only)",
        "",
        "Frozen signal:",
        "- BUY_AFTER_FOLLOW_THROUGH_STRICT",
        "- HIGH_BREAK_100_BPS",
        "- wait one 4h candle",
        "- next-open entry",
        "- hold 6 candles",
        "",
        "## 1) Does the locked candidate survive as a portfolio?",
    ]

    if portfolio_summary.empty:
        lines.append("- No portfolio scenarios produced trades.")
    else:
        best = portfolio_summary.sort_values("total_return_pct", ascending=False).iloc[0]
        lines.append(
            f"- Best scenario {best['scenario']} total_return={best['total_return_pct']:+.2f}% "
            f"max_dd={best['max_drawdown_pct']:+.2f}%"
        )

    lines += [
        "",
        "## 2) Does it beat BTC buy-and-hold?",
    ]
    btc = benchmark_comp[benchmark_comp["benchmark"] == "BTC_BUY_AND_HOLD"] if not benchmark_comp.empty else pd.DataFrame()
    lock = benchmark_comp[benchmark_comp["benchmark"] == "LOCKED_PORTFOLIO_ANCHOR"] if not benchmark_comp.empty else pd.DataFrame()
    if not btc.empty:
        lines.append(f"- BTC benchmark total_return={btc.iloc[0]['total_return_pct']:+.2f}%")
    else:
        lines.append("- BTC benchmark unavailable for period.")

    lines += [
        "",
        "## 3) Does it beat BUY_AT_EXPLOSION portfolio baseline?",
    ]
    base = benchmark_comp[benchmark_comp["benchmark"] == "BUY_AT_EXPLOSION_BASELINE"] if not benchmark_comp.empty else pd.DataFrame()
    if not base.empty:
        lines.append(f"- BUY_AT_EXPLOSION baseline total_return={base.iloc[0]['total_return_pct']:+.2f}%")
    else:
        lines.append("- BUY_AT_EXPLOSION baseline unavailable.")

    lines += [
        "",
        "## 4) Is edge destroyed by position limits?",
    ]
    if not portfolio_summary.empty:
        limited = portfolio_summary[portfolio_summary["max_open_positions"].isin([1, 3, 5, 10])]
        if not limited.empty:
            lines.append(f"- Position-limit scenarios evaluated: {len(limited)}")

    lines += [
        "",
        "## 5) Is edge destroyed by 100/150/200 bps costs?",
    ]
    if not cost_sensitivity.empty:
        for _, r in cost_sensitivity.sort_values("cost_bps").iterrows():
            lines.append(f"- {int(r['cost_bps'])} bps: avg_total_return={r['avg_total_return_pct']:+.2f}%")

    lines += [
        "",
        "## 6) Are drawdowns acceptable?",
    ]
    if not drawdown_summary.empty:
        lines.append(f"- Best max drawdown among scenarios: {drawdown_summary['max_drawdown_pct'].max():+.2f}%")
        lines.append(f"- Worst max drawdown among scenarios: {drawdown_summary['max_drawdown_pct'].min():+.2f}%")

    lines += [
        "",
        "## 7) Is performance concentrated in a few periods?",
        f"- Year rows: {len(by_year)}",
        "",
        "## 8) Paper-trading readiness vs exploratory",
    ]

    if not portfolio_summary.empty:
        robust = (portfolio_summary["total_return_pct"] > 0).mean()
        verdict = "candidate may be paper-trading candidate with caution" if robust > 0.5 else "still exploratory"
        lines.append(f"- Preliminary verdict: {verdict}.")

    lines += [
        "",
        "## Safety Confirmation",
        "- Research-only portfolio study.",
        "- No live trading behavior changed.",
        "- No scheduler, broker/exchange execution, credentials, launchd, allocation, or production config changes.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locked candidate portfolio backtest (research-only)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = run_portfolio_backtest(data_dir=args.data_dir, output_dir=args.output_dir)

    summary = out["portfolio_summary"]
    print("=" * 80)
    print("LOCKED CANDIDATE PORTFOLIO BACKTEST (RESEARCH-ONLY)")
    print("=" * 80)
    print(f"Scenarios: {len(summary)}")
    if not summary.empty:
        best = summary.sort_values("total_return_pct", ascending=False).iloc[0]
        print(
            f"Best scenario: {best['scenario']} | total={best['total_return_pct']:+.2f}% | "
            f"CAGR={best['cagr_pct']:+.2f}% | maxDD={best['max_drawdown_pct']:+.2f}%"
        )
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
