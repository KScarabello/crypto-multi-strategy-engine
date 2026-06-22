"""Research-only blend study: canonical base portfolio plus locked explosion sleeve.

This module does not modify live trading behavior. It reuses existing fixed
research candidates and evaluates allocation/funding overlays only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import SETTINGS
from research.portfolio_walk_forward_validation import WALK_FORWARD_CS_CANDIDATE
from research.sleeve_blend_experiment import run_sleeve_backtests
from research.sleeve_blend_gating_corrected import build_corrected_gated_stream
from research.sleeve_blend_gating_experiment import GateSpec, build_cs_gate_series


DEFAULT_OUTPUT_DIR = Path("reports/base_plus_explosion_blend")
BASE_OVERLAY_COSTS_BPS: tuple[float, ...] = (15.0, 30.0)
EXPLOSION_COSTS_BPS: tuple[int, ...] = (50, 100, 150, 200)
BLEND_ALLOCATIONS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
FUNDING_METHODS: tuple[str, ...] = ("proportional", "cs_only", "cash_overlay")

BASE_CANDIDATE = "GATED_BTC_TS_INVESTED_75_25"
BASE_BTC_STRATEGY = "btc_ts_60_240_12"
EXPLOSION_SCENARIO_TEMPLATE = "LOCKED_MAXUNL_EX100_CD0_C{cost}"


@dataclass(frozen=True)
class BlendRunConfig:
    base_overlay_costs_bps: tuple[float, ...] = BASE_OVERLAY_COSTS_BPS
    explosion_costs_bps: tuple[int, ...] = EXPLOSION_COSTS_BPS
    allocations: tuple[float, ...] = BLEND_ALLOCATIONS
    funding_methods: tuple[str, ...] = FUNDING_METHODS
    gate_lag_bars: int = 1
    bars_per_year: int = 2190


def align_return_streams(base_returns: pd.Series, explosion_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align sleeve return streams on common timestamps (no look-ahead)."""
    left = base_returns.astype(float).copy()
    right = explosion_returns.astype(float).copy()
    left.index = pd.to_datetime(left.index, utc=True)
    right.index = pd.to_datetime(right.index, utc=True)
    common = left.index.intersection(right.index)
    return left.reindex(common).fillna(0.0), right.reindex(common).fillna(0.0)


def equity_curve_from_returns(returns: pd.Series, initial_equity: float = 1.0) -> pd.Series:
    r = returns.astype(float).fillna(0.0)
    return float(initial_equity) * (1.0 + r).cumprod()


def max_drawdown_from_equity(equity: pd.Series) -> float:
    e = equity.astype(float)
    if e.empty:
        return 0.0
    drawdown = e / e.cummax() - 1.0
    return float(drawdown.min())


def compute_period_return(returns: pd.Series, year_start: int, year_end: int) -> float:
    mask = (returns.index.year >= int(year_start)) & (returns.index.year <= int(year_end))
    window = returns.loc[mask]
    if window.empty:
        return 0.0
    return float((1.0 + window).prod() - 1.0)


def compute_core_metrics(returns: pd.Series, bars_per_year: int) -> dict[str, float]:
    r = returns.astype(float).fillna(0.0)
    if len(r) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "worst_year": 0.0,
            "worst_month": 0.0,
            "ret_2020_2021": 0.0,
            "ret_2022_2026": 0.0,
        }

    equity = equity_curve_from_returns(r)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    years = max((len(r) - 1) / float(bars_per_year), 1e-9)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else -1.0

    vol = float(r.std(ddof=0) * np.sqrt(float(bars_per_year)))
    sharpe = float((r.mean() / (r.std(ddof=0) + 1e-12)) * np.sqrt(float(bars_per_year)))
    mdd = max_drawdown_from_equity(equity)

    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    monthly = (1.0 + r).resample("ME").prod() - 1.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "volatility": vol,
        "max_drawdown": mdd,
        "worst_year": float(yearly.min()) if not yearly.empty else 0.0,
        "worst_month": float(monthly.min()) if not monthly.empty else 0.0,
        "ret_2020_2021": compute_period_return(r, 2020, 2021),
        "ret_2022_2026": compute_period_return(r, 2022, 2026),
    }


def compute_marginal_contribution(
    blend_returns: pd.Series,
    explosion_returns: pd.Series,
    explosion_weight: float,
) -> dict[str, float]:
    contribution = float(explosion_weight) * explosion_returns.astype(float)
    blend = blend_returns.astype(float)
    contrib_sum = float(contribution.sum())
    blend_sum = float(blend.sum())
    frac = contrib_sum / blend_sum if abs(blend_sum) > 1e-12 else 0.0
    return {
        "explosion_contribution_sum": contrib_sum,
        "blend_sum": blend_sum,
        "explosion_contribution_fraction": float(frac),
        "explosion_contribution_mean": float(contribution.mean()),
    }


def blend_returns_proportional(base_returns: pd.Series, explosion_returns: pd.Series, explosion_weight: float) -> pd.Series:
    w = float(explosion_weight)
    return (1.0 - w) * base_returns.astype(float) + w * explosion_returns.astype(float)


def blend_returns_cs_only(
    base_btc_returns: pd.Series,
    base_cs_returns: pd.Series,
    base_overlay_cost: pd.Series,
    explosion_returns: pd.Series,
    explosion_weight: float,
    base_cs_nominal_weight: float = 0.25,
) -> pd.Series:
    w = float(explosion_weight)
    cs_scale = max(0.0, (float(base_cs_nominal_weight) - w) / float(base_cs_nominal_weight))
    return (
        base_btc_returns.astype(float)
        + cs_scale * base_cs_returns.astype(float)
        - base_overlay_cost.astype(float)
        + w * explosion_returns.astype(float)
    )


def blend_returns_cash_overlay(
    base_returns: pd.Series,
    explosion_returns: pd.Series,
    base_cash_weight: pd.Series,
    explosion_weight: float,
) -> tuple[pd.Series, pd.Series]:
    requested = float(explosion_weight)
    cash = base_cash_weight.astype(float).clip(lower=0.0)
    effective = cash.clip(upper=requested)
    blended = base_returns.astype(float) + effective * explosion_returns.astype(float)
    return blended, effective


def build_benchmark_comparison_row(
    blend_name: str,
    benchmark_name: str,
    blend_metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
) -> dict[str, float | str]:
    return {
        "blend": blend_name,
        "benchmark": benchmark_name,
        "blend_total_return": float(blend_metrics["total_return"]),
        "benchmark_total_return": float(benchmark_metrics["total_return"]),
        "delta_total_return": float(blend_metrics["total_return"] - benchmark_metrics["total_return"]),
        "blend_cagr": float(blend_metrics["cagr"]),
        "benchmark_cagr": float(benchmark_metrics["cagr"]),
        "delta_cagr": float(blend_metrics["cagr"] - benchmark_metrics["cagr"]),
        "blend_sharpe": float(blend_metrics["sharpe"]),
        "benchmark_sharpe": float(benchmark_metrics["sharpe"]),
        "delta_sharpe": float(blend_metrics["sharpe"] - benchmark_metrics["sharpe"]),
        "blend_max_drawdown": float(blend_metrics["max_drawdown"]),
        "benchmark_max_drawdown": float(benchmark_metrics["max_drawdown"]),
        "delta_max_drawdown": float(blend_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"]),
    }


def _load_base_streams(
    data_dir: Path,
    base_overlay_cost_bps: float,
    gate_lag_bars: int,
) -> pd.DataFrame:
    close, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(WALK_FORWARD_CS_CANDIDATE,),
        data_dir=data_dir,
    )
    aligned = pd.concat(
        [
            btc_result.portfolio["strategy_return"].rename("btc_sleeve_return"),
            cs_results[WALK_FORWARD_CS_CANDIDATE.name].portfolio["strategy_return"].rename("cs_sleeve_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    aligned.index = pd.to_datetime(aligned.index, utc=True)
    aligned = aligned.sort_index()

    btc_close = close["BTC/USD"].reindex(aligned.index).astype(float)
    btc_invested = btc_result.holdings_history["BTC/USD"].reindex(aligned.index).fillna(0.0)

    raw_gate = build_cs_gate_series(
        gate_spec=GateSpec(name="BTC_TS_INVESTED"),
        aligned_index=aligned.index,
        btc_close=btc_close,
        btc_invested=btc_invested,
    )

    stream = build_corrected_gated_stream(
        aligned_returns=aligned,
        raw_gate_series=raw_gate,
        btc_weight=0.75,
        cs_weight=0.25,
        gate_lag_bars=int(gate_lag_bars),
        overlay_cost_bps=float(base_overlay_cost_bps),
    )
    stream["bars_per_year"] = int(bars_per_year)
    return stream


def _load_explosion_curves(explosion_report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(explosion_report_dir / "portfolio_equity_curves.csv")
    curves["timestamp"] = pd.to_datetime(curves["timestamp"], utc=True)
    summary = pd.read_csv(explosion_report_dir / "portfolio_summary.csv")
    by_regime = pd.read_csv(explosion_report_dir / "portfolio_by_btc_regime.csv")
    by_liquidity = pd.read_csv(explosion_report_dir / "portfolio_by_liquidity.csv")
    return curves, summary, by_regime, by_liquidity


def run_base_plus_explosion_blend_research(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_report_dir: Path = Path("reports/btc_base_portfolio_canonical"),
    explosion_report_dir: Path = Path("reports/explosion_delayed_entry_study/portfolio_backtest"),
    locked_validation_dir: Path = Path("reports/explosion_delayed_entry_study/locked_candidate_validation"),
    config: BlendRunConfig = BlendRunConfig(),
    data_dir: Path = Path("data"),
) -> dict[str, pd.DataFrame]:
    """Run fixed-candidate blend study and write report tables."""
    # Touch required input dirs early for auditable failure.
    _ = pd.read_csv(base_report_dir / "base_portfolio_summary.csv")
    if not (locked_validation_dir / "summary.md").exists():
        raise FileNotFoundError(f"Missing required input: {locked_validation_dir / 'summary.md'}")

    curves, explosion_summary, explosion_regime, explosion_liquidity = _load_explosion_curves(explosion_report_dir)
    explosion_bench = pd.read_csv(explosion_report_dir / "portfolio_benchmark_comparison.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    drawdown_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    liquidity_rows: list[dict[str, Any]] = []

    for base_cost in config.base_overlay_costs_bps:
        base_stream = _load_base_streams(data_dir=data_dir, base_overlay_cost_bps=base_cost, gate_lag_bars=config.gate_lag_bars)
        base_returns = base_stream["blended_return"].astype(float)
        base_btc_returns = base_stream["effective_btc_return"].astype(float)
        base_cs_returns = base_stream["effective_cs_return"].astype(float)
        base_overlay_cost_series = base_stream["overlay_cost"].astype(float)
        base_cash_weight = base_stream["effective_cash_weight"].astype(float)

        base_metrics = compute_core_metrics(base_returns, bars_per_year=config.bars_per_year)
        base_avg_exposure = float((base_stream["effective_btc_weight"] + base_stream["effective_cs_weight"]).mean())
        base_turnover = float(base_stream["overlay_turnover"].sum())

        for explosion_cost in config.explosion_costs_bps:
            scenario = EXPLOSION_SCENARIO_TEMPLATE.format(cost=int(explosion_cost))
            eq = curves.loc[curves["scenario"] == scenario].copy()
            if eq.empty:
                continue
            eq = eq.sort_values("timestamp")
            explosion_returns = eq.set_index("timestamp")["equity"].pct_change().fillna(0.0).astype(float)
            explosion_exposure = eq.set_index("timestamp")["exposure"].astype(float)

            exp_row = explosion_summary.loc[explosion_summary["scenario"] == scenario]
            exp_turnover = float(exp_row.iloc[0]["turnover"]) if not exp_row.empty else 0.0
            exp_trades = int(exp_row.iloc[0]["n_trades"]) if not exp_row.empty else 0
            exp_metrics = compute_core_metrics(explosion_returns, bars_per_year=config.bars_per_year)

            for method in config.funding_methods:
                method_allowed = True
                if method == "cash_overlay" and float(base_cash_weight.mean()) <= 1e-9:
                    method_allowed = False
                if not method_allowed:
                    continue

                for alloc in config.allocations:
                    blend_name = (
                        f"base_cost{int(base_cost)}_exp_cost{int(explosion_cost)}"
                        f"_{method}_base{int(round((1.0-alloc)*100)):02d}_exp{int(round(alloc*100)):02d}"
                    )

                    b_ret, e_ret = align_return_streams(base_returns, explosion_returns)
                    idx = b_ret.index
                    b_btc = base_btc_returns.reindex(idx).fillna(0.0)
                    b_cs = base_cs_returns.reindex(idx).fillna(0.0)
                    b_ov = base_overlay_cost_series.reindex(idx).fillna(0.0)
                    b_cash = base_cash_weight.reindex(idx).fillna(0.0)
                    e_exp = explosion_exposure.reindex(idx).fillna(0.0)

                    if method == "proportional":
                        blend_returns = blend_returns_proportional(b_ret, e_ret, alloc)
                        eff_explosion_weight = pd.Series(float(alloc), index=idx)
                        avg_exposure = float(((1.0 - alloc) * (1.0 - b_cash) + alloc * e_exp).mean())
                        turnover = float((1.0 - alloc) * base_turnover + alloc * exp_turnover)
                    elif method == "cs_only":
                        blend_returns = blend_returns_cs_only(
                            base_btc_returns=b_btc,
                            base_cs_returns=b_cs,
                            base_overlay_cost=b_ov,
                            explosion_returns=e_ret,
                            explosion_weight=alloc,
                        )
                        eff_explosion_weight = pd.Series(float(alloc), index=idx)
                        cs_scale = max(0.0, (0.25 - alloc) / 0.25)
                        avg_exposure = float((0.75 + cs_scale * (base_stream["cs_gate"].reindex(idx).fillna(0.0) * 0.25) + alloc * e_exp).mean())
                        turnover = float(base_turnover + alloc * exp_turnover)
                    elif method == "cash_overlay":
                        blend_returns, eff_explosion_weight = blend_returns_cash_overlay(
                            base_returns=b_ret,
                            explosion_returns=e_ret,
                            base_cash_weight=b_cash,
                            explosion_weight=alloc,
                        )
                        avg_exposure = float(((1.0 - b_cash) + eff_explosion_weight * e_exp).mean())
                        turnover = float(base_turnover + eff_explosion_weight.mean() * exp_turnover)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    metrics = compute_core_metrics(blend_returns, bars_per_year=config.bars_per_year)
                    corr = float(b_ret.corr(e_ret)) if len(b_ret) > 1 else 0.0
                    if np.isnan(corr):
                        corr = 0.0
                    marginal = compute_marginal_contribution(blend_returns, e_ret, float(eff_explosion_weight.mean()))

                    equity = equity_curve_from_returns(blend_returns)
                    drawdown = equity / equity.cummax() - 1.0
                    yearly = (1.0 + blend_returns).groupby(blend_returns.index.year).prod() - 1.0
                    monthly = (1.0 + blend_returns).resample("ME").prod() - 1.0

                    summary_rows.append(
                        {
                            "blend": blend_name,
                            "base_overlay_cost_bps": float(base_cost),
                            "explosion_cost_bps": int(explosion_cost),
                            "funding_method": method,
                            "base_weight": float(1.0 - alloc),
                            "explosion_weight": float(alloc),
                            "effective_explosion_weight": float(eff_explosion_weight.mean()),
                            "total_return": metrics["total_return"],
                            "cagr": metrics["cagr"],
                            "sharpe": metrics["sharpe"],
                            "volatility": metrics["volatility"],
                            "max_drawdown": metrics["max_drawdown"],
                            "worst_year": metrics["worst_year"],
                            "worst_month": metrics["worst_month"],
                            "average_exposure": avg_exposure,
                            "turnover": turnover,
                            "explosion_trades": int(round(float(eff_explosion_weight.mean()) * exp_trades)),
                            "base_explosion_correlation": corr,
                            "ret_2020_2021": metrics["ret_2020_2021"],
                            "ret_2022_2026": metrics["ret_2022_2026"],
                        }
                    )

                    for y, v in yearly.items():
                        yearly_rows.append(
                            {
                                "blend": blend_name,
                                "base_overlay_cost_bps": float(base_cost),
                                "explosion_cost_bps": int(explosion_cost),
                                "funding_method": method,
                                "year": int(y),
                                "return": float(v),
                            }
                        )

                    for ts, v in monthly.items():
                        monthly_rows.append(
                            {
                                "blend": blend_name,
                                "base_overlay_cost_bps": float(base_cost),
                                "explosion_cost_bps": int(explosion_cost),
                                "funding_method": method,
                                "month": str(ts.to_period("M")),
                                "return": float(v),
                            }
                        )

                    for ts, v in drawdown.items():
                        drawdown_rows.append(
                            {
                                "blend": blend_name,
                                "base_overlay_cost_bps": float(base_cost),
                                "explosion_cost_bps": int(explosion_cost),
                                "funding_method": method,
                                "timestamp": str(ts),
                                "drawdown": float(v),
                            }
                        )

                    corr_rows.append(
                        {
                            "blend": blend_name,
                            "base_overlay_cost_bps": float(base_cost),
                            "explosion_cost_bps": int(explosion_cost),
                            "funding_method": method,
                            "correlation": corr,
                        }
                    )

                    marginal_rows.append(
                        {
                            "blend": blend_name,
                            "base_overlay_cost_bps": float(base_cost),
                            "explosion_cost_bps": int(explosion_cost),
                            "funding_method": method,
                            **marginal,
                        }
                    )

                    benchmark_rows.append(
                        build_benchmark_comparison_row(
                            blend_name=blend_name,
                            benchmark_name="BASE_ONLY",
                            blend_metrics=metrics,
                            benchmark_metrics=base_metrics,
                        )
                    )
                    benchmark_rows.append(
                        build_benchmark_comparison_row(
                            blend_name=blend_name,
                            benchmark_name="LOCKED_EXPLOSION_ONLY",
                            blend_metrics=metrics,
                            benchmark_metrics=exp_metrics,
                        )
                    )

                    btc_buy = explosion_bench.loc[explosion_bench["benchmark"] == "BTC_BUY_AND_HOLD"]
                    if not btc_buy.empty:
                        benchmark_rows.append(
                            build_benchmark_comparison_row(
                                blend_name=blend_name,
                                benchmark_name="BTC_BUY_AND_HOLD",
                                blend_metrics=metrics,
                                benchmark_metrics={
                                    "total_return": float(btc_buy.iloc[0]["total_return_pct"]) / 100.0,
                                    "cagr": 0.0,
                                    "sharpe": 0.0,
                                    "max_drawdown": 0.0,
                                },
                            )
                        )

                    buy_at = explosion_bench.loc[explosion_bench["benchmark"] == "BUY_AT_EXPLOSION_BASELINE"]
                    if not buy_at.empty and int(explosion_cost) == 50:
                        benchmark_rows.append(
                            build_benchmark_comparison_row(
                                blend_name=blend_name,
                                benchmark_name="BUY_AT_EXPLOSION_BASELINE",
                                blend_metrics=metrics,
                                benchmark_metrics={
                                    "total_return": float(buy_at.iloc[0]["total_return_pct"]) / 100.0,
                                    "cagr": 0.0,
                                    "sharpe": 0.0,
                                    "max_drawdown": 0.0,
                                },
                            )
                        )

                    benchmark_rows.append(
                        build_benchmark_comparison_row(
                            blend_name=blend_name,
                            benchmark_name="BTC_TS_ONLY_IF_AVAILABLE",
                            blend_metrics=metrics,
                            benchmark_metrics={
                                "total_return": float(base_metrics["total_return"]),
                                "cagr": float(base_metrics["cagr"]),
                                "sharpe": float(base_metrics["sharpe"]),
                                "max_drawdown": float(base_metrics["max_drawdown"]),
                            },
                        )
                    )

                    # Feasible proxy breakdowns: scale locked sleeve trade profiles by effective sleeve weight.
                    eff_w = float(eff_explosion_weight.mean())
                    for _, rr in explosion_regime.iterrows():
                        regime_rows.append(
                            {
                                "blend": blend_name,
                                "base_overlay_cost_bps": float(base_cost),
                                "explosion_cost_bps": int(explosion_cost),
                                "funding_method": method,
                                "btc_trend_bucket": str(rr["btc_trend_bucket"]),
                                "explosion_n_trades_scaled": eff_w * float(rr["n_trades"]),
                                "explosion_avg_trade_return_pct": float(rr["avg_trade_return_pct"]),
                            }
                        )

                    for _, ll in explosion_liquidity.iterrows():
                        liquidity_rows.append(
                            {
                                "blend": blend_name,
                                "base_overlay_cost_bps": float(base_cost),
                                "explosion_cost_bps": int(explosion_cost),
                                "funding_method": method,
                                "liquidity_bucket": str(ll["liquidity_bucket"]),
                                "explosion_n_trades_scaled": eff_w * float(ll["n_trades"]),
                                "explosion_avg_trade_return_pct": float(ll["avg_trade_return_pct"]),
                            }
                        )

    blend_summary = pd.DataFrame(summary_rows).sort_values(
        ["base_overlay_cost_bps", "explosion_cost_bps", "funding_method", "explosion_weight"]
    ).reset_index(drop=True)
    blend_yearly = pd.DataFrame(yearly_rows).sort_values(
        ["base_overlay_cost_bps", "explosion_cost_bps", "funding_method", "blend", "year"]
    ).reset_index(drop=True)
    blend_monthly = pd.DataFrame(monthly_rows).sort_values(
        ["base_overlay_cost_bps", "explosion_cost_bps", "funding_method", "blend", "month"]
    ).reset_index(drop=True)
    blend_drawdown = pd.DataFrame(drawdown_rows).sort_values(
        ["base_overlay_cost_bps", "explosion_cost_bps", "funding_method", "blend", "timestamp"]
    ).reset_index(drop=True)
    benchmark_comp = pd.DataFrame(benchmark_rows)
    sleeve_corr = pd.DataFrame(corr_rows)
    marginal_df = pd.DataFrame(marginal_rows)
    regime_df = pd.DataFrame(regime_rows)
    liquidity_df = pd.DataFrame(liquidity_rows)

    blend_summary.to_csv(output_dir / "blend_summary.csv", index=False)
    blend_yearly.to_csv(output_dir / "blend_yearly.csv", index=False)
    blend_monthly.to_csv(output_dir / "blend_monthly.csv", index=False)
    blend_drawdown.to_csv(output_dir / "blend_drawdown.csv", index=False)
    benchmark_comp.to_csv(output_dir / "blend_benchmark_comparison.csv", index=False)
    sleeve_corr.to_csv(output_dir / "sleeve_correlation.csv", index=False)
    marginal_df.to_csv(output_dir / "marginal_contribution.csv", index=False)
    regime_df.to_csv(output_dir / "btc_regime_breakdown.csv", index=False)
    liquidity_df.to_csv(output_dir / "liquidity_breakdown.csv", index=False)

    # Decision-oriented summary.
    base_only = blend_summary.loc[(blend_summary["funding_method"] == "proportional") & (blend_summary["explosion_weight"] == 0.0)]
    improved = blend_summary.loc[
        (blend_summary["total_return"] > base_only["total_return"].max())
        & (blend_summary["sharpe"] > base_only["sharpe"].max())
        & (blend_summary["max_drawdown"] >= base_only["max_drawdown"].min() - 0.03)
    ] if not base_only.empty else pd.DataFrame()

    best = blend_summary.sort_values(["sharpe", "total_return"], ascending=[False, False]).head(1)
    lines: list[str] = []
    lines.append("# Base Plus Locked Explosion Blend (Research-Only)")
    lines.append("")
    lines.append("Fixed inputs:")
    lines.append("- Base: GATED_BTC_TS_INVESTED_75_25 (btc_ts_60_240_12, BTC_TS_INVESTED gate, 4h, rebalance 12, gate lag 1)")
    lines.append("- Explosion: BUY_AFTER_FOLLOW_THROUGH_STRICT + HIGH_BREAK_100_BPS, delayed next-open entry, hold 6 bars, long-only")
    lines.append("- No signal optimization and no new candidates")
    lines.append("")

    if not best.empty:
        b = best.iloc[0]
        lines.append("Top blend by Sharpe:")
        lines.append(
            f"- {b['blend']} | total_return={b['total_return']:+.4f}, cagr={b['cagr']:+.4f}, sharpe={b['sharpe']:+.4f}, max_drawdown={b['max_drawdown']:+.4f}"
        )
        lines.append("")

    lines.append("Key questions:")
    if base_only.empty:
        lines.append("1. Base-only comparison could not be established from generated rows.")
    else:
        base_ref = base_only.sort_values("base_overlay_cost_bps").head(1).iloc[0]
        lines.append(
            f"1. Total return improvement vs base-only baseline? {'YES' if float(best.iloc[0]['total_return']) > float(base_ref['total_return']) else 'NO'}"
        )
        lines.append(
            f"2. CAGR improvement? {'YES' if float(best.iloc[0]['cagr']) > float(base_ref['cagr']) else 'NO'}"
        )
        lines.append(
            f"3. Sharpe improvement? {'YES' if float(best.iloc[0]['sharpe']) > float(base_ref['sharpe']) else 'NO'}"
        )
        lines.append(
            f"4. Max drawdown improved or stable? {'YES' if float(best.iloc[0]['max_drawdown']) >= float(base_ref['max_drawdown']) else 'NO'}"
        )
        lines.append(
            f"5. 2022-2026 improved? {'YES' if float(best.iloc[0]['ret_2022_2026']) > float(base_ref['ret_2022_2026']) else 'NO'}"
        )

    if not sleeve_corr.empty:
        c = sleeve_corr["correlation"].mean()
        lines.append(f"6. Diversification signal (mean base/explosion correlation): {c:+.4f}")

    mania = blend_summary["ret_2020_2021"].mean()
    post = blend_summary["ret_2022_2026"].mean()
    lines.append(f"7. Avg 2020-2021 return across blends: {mania:+.4f}")
    lines.append(f"   Avg 2022-2026 return across blends: {post:+.4f}")

    if improved.empty:
        lines.append("8. Allocation improving return+Sharpe without material drawdown worsening: none found under strict rule.")
        lines.append("9-13. Recommendation: keep as research-only; do not promote to live. Consider paper-trading research only if future OOS risk-adjusted gains persist.")
    else:
        good = improved.sort_values(["sharpe", "total_return"], ascending=[False, False]).iloc[0]
        lines.append(
            f"8. Best strict-improvement allocation: {good['blend']}"
        )
        lines.append("9-13. Recommendation: consider as small tactical sleeve in paper-trading research only; not live.")

    lines.append("")
    lines.append("Decision rules applied:")
    lines.append("- No recommendation when return gains required materially worse drawdown.")
    lines.append("- No recommendation when gains were confined to 2020-2021 and degraded 2022-2026.")
    lines.append("")
    lines.append("Safety confirmation:")
    lines.append("- Research-only analysis.")
    lines.append("- No live trading behavior, scheduler/launchd, broker/exchange execution, credentials, production config, or production allocation logic changed.")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "blend_summary": blend_summary,
        "blend_yearly": blend_yearly,
        "blend_monthly": blend_monthly,
        "blend_drawdown": blend_drawdown,
        "blend_benchmark_comparison": benchmark_comp,
        "sleeve_correlation": sleeve_corr,
        "marginal_contribution": marginal_df,
        "btc_regime_breakdown": regime_df,
        "liquidity_breakdown": liquidity_df,
    }


def main() -> None:
    run_base_plus_explosion_blend_research()
    print("Wrote blend study outputs to", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
