"""Research-only walk-forward validation for BTC/CS portfolio candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.metrics import summary_metrics
from config import SETTINGS
from research.cross_sectional_momentum_robustness import CrossSectionalRobustnessSpec
from research.sleeve_blend_experiment import run_sleeve_backtests
from research.sleeve_blend_gating_corrected import build_corrected_gated_stream
from research.sleeve_blend_gating_experiment import GateSpec, build_cs_gate_series


DEFAULT_OUTPUT_PATH = Path("portfolio_walk_forward_validation.csv")
DEFAULT_RANKINGS_OUTPUT_PATH = Path("portfolio_walk_forward_candidate_rankings.csv")
DEFAULT_FIXED_COMPARISON_OUTPUT_PATH = Path("portfolio_fixed_candidate_comparison.csv")
WALK_FORWARD_OVERLAY_COST_BPS: tuple[float, ...] = (15.0, 30.0)
WALK_FORWARD_GATE_LAG_BARS = 1
WALK_FORWARD_FIXED_CANDIDATE = "GATED_BTC_TS_INVESTED_75_25"
WALK_FORWARD_CS_CANDIDATE = CrossSectionalRobustnessSpec(
    name="cs_180_top5_reb12",
    lookback_config_name="medium_180_only",
    top_n=5,
    rebalance_bars=12,
)

CANDIDATE_NAMES: tuple[str, ...] = (
    "BTC_ONLY",
    "ALWAYS_ON_90_10",
    "ALWAYS_ON_85_15",
    "ALWAYS_ON_75_25",
    "GATED_BTC_TS_INVESTED_90_10",
    "GATED_BTC_TS_INVESTED_85_15",
    "GATED_BTC_TS_INVESTED_75_25",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    btc_weight: float
    cs_weight: float
    gate_name: str
    gated: bool


WALK_FORWARD_CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(name="BTC_ONLY", btc_weight=1.0, cs_weight=0.0, gate_name="ALWAYS_ON", gated=False),
    CandidateSpec(name="ALWAYS_ON_90_10", btc_weight=0.9, cs_weight=0.1, gate_name="ALWAYS_ON", gated=False),
    CandidateSpec(name="ALWAYS_ON_85_15", btc_weight=0.85, cs_weight=0.15, gate_name="ALWAYS_ON", gated=False),
    CandidateSpec(name="ALWAYS_ON_75_25", btc_weight=0.75, cs_weight=0.25, gate_name="ALWAYS_ON", gated=False),
    CandidateSpec(name="GATED_BTC_TS_INVESTED_90_10", btc_weight=0.9, cs_weight=0.1, gate_name="BTC_TS_INVESTED", gated=True),
    CandidateSpec(name="GATED_BTC_TS_INVESTED_85_15", btc_weight=0.85, cs_weight=0.15, gate_name="BTC_TS_INVESTED", gated=True),
    CandidateSpec(name="GATED_BTC_TS_INVESTED_75_25", btc_weight=0.75, cs_weight=0.25, gate_name="BTC_TS_INVESTED", gated=True),
)


def _worst_month_return(returns: pd.Series) -> float:
    monthly = (1.0 + returns.dropna().astype(float)).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float(monthly.min())


def _annual_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).groupby(clean.index.year).prod() - 1.0


def compute_period_metrics(returns: pd.Series, bars_per_year: int) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "worst_month_return": 0.0,
            "count_negative_years": 0.0,
            "annualized_volatility": 0.0,
        }

    equity = pd.Series(1.0, index=clean.index, dtype=float) * (1.0 + clean).cumprod()
    metrics = summary_metrics(equity=equity, returns=clean, bars_per_year=bars_per_year)
    annual_returns = _annual_returns(clean)

    return {
        "total_return": float(metrics.get("total_return", 0.0)),
        "annual_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "worst_month_return": _worst_month_return(clean),
        "count_negative_years": float((annual_returns < 0.0).sum()),
        "annualized_volatility": float(metrics.get("annualized_volatility", 0.0)),
    }


def build_walk_forward_splits(index: pd.Index, start_test_year: int = 2022) -> list[dict[str, pd.Timestamp]]:
    ts_index = pd.to_datetime(index, utc=True)
    years = sorted(pd.Index(ts_index.year).unique().tolist())
    splits: list[dict[str, pd.Timestamp]] = []

    for test_year in years:
        if int(test_year) < int(start_test_year):
            continue

        train_mask = ts_index.year <= int(test_year) - 1
        test_mask = ts_index.year == int(test_year)

        if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 2:
            continue

        train_index = ts_index[train_mask]
        test_index = ts_index[test_mask]

        splits.append(
            {
                "train_start": pd.Timestamp(train_index.min()),
                "train_end": pd.Timestamp(train_index.max()),
                "test_start": pd.Timestamp(test_index.min()),
                "test_end": pd.Timestamp(test_index.max()),
            }
        )

    return splits


def select_candidate_from_training(
    train_metrics: pd.DataFrame,
    drawdown_floor: float = -0.50,
    fallback_candidate: str = "BTC_ONLY",
) -> str:
    if train_metrics.empty:
        return fallback_candidate

    btc_row = train_metrics.loc[train_metrics["candidate_name"] == "BTC_ONLY"]
    if btc_row.empty:
        return fallback_candidate

    btc_negative_years = float(btc_row.iloc[0]["train_count_negative_years"])

    filtered = train_metrics.loc[
        (train_metrics["train_max_drawdown"] >= float(drawdown_floor))
        & (train_metrics["train_count_negative_years"] <= btc_negative_years)
    ]

    if filtered.empty:
        return fallback_candidate

    return str(filtered.sort_values(["train_sharpe", "train_total_return"], ascending=[False, False]).iloc[0]["candidate_name"])


def build_training_candidate_ranking(
    train_metrics: pd.DataFrame,
    selected_candidate: str,
    overlay_cost_bps: float,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    drawdown_floor: float = -0.50,
) -> pd.DataFrame:
    """Build per-split training ranking diagnostics for all candidates.

    Rank convention:
    - Passing candidates receive rank 1..N by training Sharpe (desc), then
      training total return (desc).
    - Failed candidates are retained with selection_rank=NaN.
    """
    ranking = train_metrics.copy()
    if ranking.empty:
        return ranking

    btc_row = ranking.loc[ranking["candidate_name"] == "BTC_ONLY"]
    btc_negative_years = (
        float(btc_row.iloc[0]["train_count_negative_years"])
        if not btc_row.empty
        else float("inf")
    )

    ranking["passes_max_drawdown_filter"] = ranking["train_max_drawdown"] >= float(drawdown_floor)
    ranking["passes_negative_years_filter"] = ranking["train_count_negative_years"] <= btc_negative_years
    ranking["passes_all_filters"] = ranking["passes_max_drawdown_filter"] & ranking["passes_negative_years_filter"]

    passing_sorted = ranking.loc[ranking["passes_all_filters"]].sort_values(
        ["train_sharpe", "train_total_return"],
        ascending=[False, False],
    )
    rank_map = {
        str(name): float(i)
        for i, name in enumerate(passing_sorted["candidate_name"].tolist(), start=1)
    }
    ranking["selection_rank"] = ranking["candidate_name"].astype(str).map(rank_map)

    ranking["candidate"] = ranking["candidate_name"].astype(str)
    ranking["selected_candidate"] = str(selected_candidate)
    ranking["is_selected"] = ranking["candidate"] == str(selected_candidate)
    ranking["overlay_cost_bps"] = float(overlay_cost_bps)
    ranking["train_start"] = str(train_start)
    ranking["train_end"] = str(train_end)
    ranking["test_start"] = str(test_start)
    ranking["test_end"] = str(test_end)

    ordered_cols = [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "overlay_cost_bps",
        "candidate",
        "train_total_return",
        "train_cagr",
        "train_sharpe",
        "train_max_drawdown",
        "train_count_negative_years",
        "passes_max_drawdown_filter",
        "passes_negative_years_filter",
        "passes_all_filters",
        "selected_candidate",
        "is_selected",
        "selection_rank",
    ]
    return ranking[ordered_cols].copy()


def _build_candidate_return_streams(
    overlay_cost_bps_values: Sequence[float] = WALK_FORWARD_OVERLAY_COST_BPS,
    gate_lag_bars: int = WALK_FORWARD_GATE_LAG_BARS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
) -> tuple[int, dict[float, dict[str, pd.Series]]]:
    close, bars_per_year, btc_result, cs_results = run_sleeve_backtests(
        cs_candidates=(WALK_FORWARD_CS_CANDIDATE,),
        close_prices=close_prices,
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

    raw_gates: dict[str, pd.Series] = {
        "ALWAYS_ON": pd.Series(1.0, index=aligned.index, dtype=float),
        "BTC_TS_INVESTED": build_cs_gate_series(
            gate_spec=GateSpec(name="BTC_TS_INVESTED"),
            aligned_index=aligned.index,
            btc_close=btc_close,
            btc_invested=btc_invested,
        ),
    }

    scenario_streams: dict[float, dict[str, pd.Series]] = {}
    for overlay_cost_bps in tuple(float(x) for x in overlay_cost_bps_values):
        streams_for_overlay: dict[str, pd.Series] = {}
        for spec in WALK_FORWARD_CANDIDATE_SPECS:
            raw_gate = raw_gates[spec.gate_name]
            stream = build_corrected_gated_stream(
                aligned_returns=aligned,
                raw_gate_series=raw_gate,
                btc_weight=spec.btc_weight,
                cs_weight=spec.cs_weight,
                gate_lag_bars=int(gate_lag_bars),
                overlay_cost_bps=float(overlay_cost_bps),
            )
            streams_for_overlay[spec.name] = stream["blended_return"].astype(float)

        scenario_streams[float(overlay_cost_bps)] = streams_for_overlay

    return bars_per_year, scenario_streams


def run_walk_forward_for_scenario(
    candidate_returns: dict[str, pd.Series],
    bars_per_year: int,
    overlay_cost_bps: float,
    splits: list[dict[str, pd.Timestamp]] | None = None,
    drawdown_floor: float = -0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "BTC_ONLY" not in candidate_returns:
        raise ValueError("candidate_returns must include BTC_ONLY")
    if WALK_FORWARD_FIXED_CANDIDATE not in candidate_returns:
        raise ValueError(f"candidate_returns must include fixed candidate: {WALK_FORWARD_FIXED_CANDIDATE}")

    index = next(iter(candidate_returns.values())).index
    for series in candidate_returns.values():
        if not series.index.equals(index):
            raise ValueError("All candidate return streams must share the same index")

    wf_splits = splits if splits is not None else build_walk_forward_splits(index)
    rows: list[dict[str, Any]] = []
    ranking_rows: list[pd.DataFrame] = []
    stitched_returns_selector: list[pd.Series] = []
    stitched_returns_fixed: list[pd.Series] = []
    test_beats_return = 0
    test_beats_sharpe = 0
    test_beats_drawdown = 0
    selector_beats_fixed_return = 0
    selector_beats_fixed_sharpe = 0
    selector_beats_fixed_drawdown = 0

    for split in wf_splits:
        train_start = pd.Timestamp(split["train_start"])
        train_end = pd.Timestamp(split["train_end"])
        test_start = pd.Timestamp(split["test_start"])
        test_end = pd.Timestamp(split["test_end"])

        train_mask = (index >= train_start) & (index <= train_end)
        test_mask = (index >= test_start) & (index <= test_end)

        train_metric_rows: list[dict[str, Any]] = []
        for candidate_name, series in candidate_returns.items():
            train_metrics = compute_period_metrics(series.loc[train_mask], bars_per_year=bars_per_year)
            train_metric_rows.append(
                {
                    "candidate_name": candidate_name,
                    "train_total_return": train_metrics["total_return"],
                    "train_cagr": train_metrics["cagr"],
                    "train_sharpe": train_metrics["sharpe"],
                    "train_max_drawdown": train_metrics["max_drawdown"],
                    "train_count_negative_years": train_metrics["count_negative_years"],
                }
            )

        train_metrics_df = pd.DataFrame(train_metric_rows)
        selected_candidate = select_candidate_from_training(
            train_metrics=train_metrics_df,
            drawdown_floor=drawdown_floor,
            fallback_candidate="BTC_ONLY",
        )

        selected_train = train_metrics_df.loc[train_metrics_df["candidate_name"] == selected_candidate].iloc[0]
        ranking_rows.append(
            build_training_candidate_ranking(
                train_metrics=train_metrics_df,
                selected_candidate=selected_candidate,
                overlay_cost_bps=float(overlay_cost_bps),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                drawdown_floor=drawdown_floor,
            )
        )

        selected_test_metrics = compute_period_metrics(
            candidate_returns[selected_candidate].loc[test_mask],
            bars_per_year=bars_per_year,
        )
        fixed_test_metrics = compute_period_metrics(
            candidate_returns[WALK_FORWARD_FIXED_CANDIDATE].loc[test_mask],
            bars_per_year=bars_per_year,
        )
        btc_only_test_metrics = compute_period_metrics(
            candidate_returns["BTC_ONLY"].loc[test_mask],
            bars_per_year=bars_per_year,
        )

        beats_return = selected_test_metrics["total_return"] > btc_only_test_metrics["total_return"]
        beats_sharpe = selected_test_metrics["sharpe"] > btc_only_test_metrics["sharpe"]
        beats_drawdown = selected_test_metrics["max_drawdown"] > btc_only_test_metrics["max_drawdown"]

        test_beats_return += int(beats_return)
        test_beats_sharpe += int(beats_sharpe)
        test_beats_drawdown += int(beats_drawdown)

        beats_fixed_return = selected_test_metrics["total_return"] > fixed_test_metrics["total_return"]
        beats_fixed_sharpe = selected_test_metrics["sharpe"] > fixed_test_metrics["sharpe"]
        beats_fixed_drawdown = selected_test_metrics["max_drawdown"] > fixed_test_metrics["max_drawdown"]

        selector_beats_fixed_return += int(beats_fixed_return)
        selector_beats_fixed_sharpe += int(beats_fixed_sharpe)
        selector_beats_fixed_drawdown += int(beats_fixed_drawdown)

        rows.append(
            {
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "selected_candidate": selected_candidate,
                "overlay_cost_bps": float(overlay_cost_bps),
                "train_total_return": float(selected_train["train_total_return"]),
                "train_cagr": float(selected_train["train_cagr"]),
                "train_sharpe": float(selected_train["train_sharpe"]),
                "train_max_drawdown": float(selected_train["train_max_drawdown"]),
                "train_count_negative_years": int(selected_train["train_count_negative_years"]),
                "test_total_return": float(selected_test_metrics["total_return"]),
                "test_annual_return": float(selected_test_metrics["annual_return"]),
                "test_sharpe": float(selected_test_metrics["sharpe"]),
                "test_max_drawdown": float(selected_test_metrics["max_drawdown"]),
                "test_worst_month_return": float(selected_test_metrics["worst_month_return"]),
                "fixed_candidate": WALK_FORWARD_FIXED_CANDIDATE,
                "fixed_test_total_return": float(fixed_test_metrics["total_return"]),
                "fixed_test_annual_return": float(fixed_test_metrics["annual_return"]),
                "fixed_test_sharpe": float(fixed_test_metrics["sharpe"]),
                "fixed_test_max_drawdown": float(fixed_test_metrics["max_drawdown"]),
                "fixed_test_worst_month_return": float(fixed_test_metrics["worst_month_return"]),
                "btc_only_test_total_return": float(btc_only_test_metrics["total_return"]),
                "btc_only_test_annual_return": float(btc_only_test_metrics["annual_return"]),
                "btc_only_test_sharpe": float(btc_only_test_metrics["sharpe"]),
                "btc_only_test_max_drawdown": float(btc_only_test_metrics["max_drawdown"]),
                "btc_only_test_worst_month_return": float(btc_only_test_metrics["worst_month_return"]),
                "selector_excess_return_vs_fixed": float(selected_test_metrics["total_return"] - fixed_test_metrics["total_return"]),
                "selector_excess_annual_return_vs_fixed": float(selected_test_metrics["annual_return"] - fixed_test_metrics["annual_return"]),
                "selector_excess_sharpe_vs_fixed": float(selected_test_metrics["sharpe"] - fixed_test_metrics["sharpe"]),
                "selector_drawdown_delta_vs_fixed": float(selected_test_metrics["max_drawdown"] - fixed_test_metrics["max_drawdown"]),
                "selector_worst_month_delta_vs_fixed": float(selected_test_metrics["worst_month_return"] - fixed_test_metrics["worst_month_return"]),
                "test_excess_return_vs_btc": float(selected_test_metrics["total_return"] - btc_only_test_metrics["total_return"]),
                "test_excess_annual_return_vs_btc": float(selected_test_metrics["annual_return"] - btc_only_test_metrics["annual_return"]),
                "test_excess_sharpe_vs_btc": float(selected_test_metrics["sharpe"] - btc_only_test_metrics["sharpe"]),
                "test_drawdown_delta_vs_btc": float(selected_test_metrics["max_drawdown"] - btc_only_test_metrics["max_drawdown"]),
                "test_worst_month_delta_vs_btc": float(selected_test_metrics["worst_month_return"] - btc_only_test_metrics["worst_month_return"]),
                "selector_beats_fixed_return": bool(beats_fixed_return),
                "selector_beats_fixed_sharpe": bool(beats_fixed_sharpe),
                "selector_beats_fixed_drawdown": bool(beats_fixed_drawdown),
                "test_beats_btc_only_return": bool(beats_return),
                "test_beats_btc_only_sharpe": bool(beats_sharpe),
                "test_beats_btc_only_drawdown": bool(beats_drawdown),
            }
        )

        selected_test_returns = candidate_returns[selected_candidate].loc[test_mask].astype(float)
        fixed_test_returns = candidate_returns[WALK_FORWARD_FIXED_CANDIDATE].loc[test_mask].astype(float)
        stitched_returns_selector.append(selected_test_returns)
        stitched_returns_fixed.append(fixed_test_returns)

    step_report = pd.DataFrame(rows)
    if step_report.empty:
        raise ValueError("No walk-forward rows produced")
    ranking_report = pd.concat(ranking_rows, axis=0, ignore_index=True)

    selector_combined = pd.concat(stitched_returns_selector, axis=0).sort_index()
    fixed_combined = pd.concat(stitched_returns_fixed, axis=0).sort_index()
    selector_aggregate_metrics = compute_period_metrics(selector_combined, bars_per_year=bars_per_year)
    fixed_aggregate_metrics = compute_period_metrics(fixed_combined, bars_per_year=bars_per_year)

    aggregate_report = pd.DataFrame(
        [
            {
                "overlay_cost_bps": float(overlay_cost_bps),
                "selector_compounded_test_total_return": float(selector_aggregate_metrics["total_return"]),
                "selector_walk_forward_sharpe": float(selector_aggregate_metrics["sharpe"]),
                "fixed_compounded_test_total_return": float(fixed_aggregate_metrics["total_return"]),
                "fixed_walk_forward_sharpe": float(fixed_aggregate_metrics["sharpe"]),
                "selector_excess_compounded_return_vs_fixed": float(selector_aggregate_metrics["total_return"] - fixed_aggregate_metrics["total_return"]),
                "selector_excess_walk_forward_sharpe_vs_fixed": float(selector_aggregate_metrics["sharpe"] - fixed_aggregate_metrics["sharpe"]),
                "selector_beats_fixed_return_years": int(selector_beats_fixed_return),
                "selector_beats_fixed_sharpe_years": int(selector_beats_fixed_sharpe),
                "selector_beats_fixed_drawdown_years": int(selector_beats_fixed_drawdown),
                "compounded_walk_forward_total_return": float(selector_aggregate_metrics["total_return"]),
                "walk_forward_cagr": float(selector_aggregate_metrics["cagr"]),
                "walk_forward_annualized_volatility": float(selector_aggregate_metrics["annualized_volatility"]),
                "walk_forward_sharpe": float(selector_aggregate_metrics["sharpe"]),
                "walk_forward_max_drawdown": float(selector_aggregate_metrics["max_drawdown"]),
                "number_of_test_years_beating_btc_only_on_return": int(test_beats_return),
                "number_of_test_years_beating_btc_only_on_sharpe": int(test_beats_sharpe),
                "number_of_test_years_with_better_max_drawdown_than_btc_only": int(test_beats_drawdown),
            }
        ]
    )

    return step_report, aggregate_report, ranking_report


def build_fixed_candidate_comparison_for_scenario(
    candidate_returns: dict[str, pd.Series],
    bars_per_year: int,
    overlay_cost_bps: float,
    splits: list[dict[str, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "BTC_ONLY" not in candidate_returns:
        raise ValueError("candidate_returns must include BTC_ONLY")

    index = next(iter(candidate_returns.values())).index
    for series in candidate_returns.values():
        if not series.index.equals(index):
            raise ValueError("All candidate return streams must share the same index")

    wf_splits = splits if splits is not None else build_walk_forward_splits(index)
    candidate_order = [name for name in CANDIDATE_NAMES if name in candidate_returns]
    candidate_order.extend(name for name in candidate_returns if name not in candidate_order)

    rows: list[dict[str, Any]] = []
    candidate_stitched_returns: dict[str, list[pd.Series]] = {name: [] for name in candidate_order}

    for split in wf_splits:
        test_start = pd.Timestamp(split["test_start"])
        test_end = pd.Timestamp(split["test_end"])
        test_mask = (index >= test_start) & (index <= test_end)

        btc_only_test_metrics = compute_period_metrics(
            candidate_returns["BTC_ONLY"].loc[test_mask],
            bars_per_year=bars_per_year,
        )

        for candidate_name in candidate_order:
            candidate_series = candidate_returns[candidate_name].loc[test_mask].astype(float)
            candidate_metrics = compute_period_metrics(candidate_series, bars_per_year=bars_per_year)

            beats_return = candidate_metrics["total_return"] > btc_only_test_metrics["total_return"]
            beats_sharpe = candidate_metrics["sharpe"] > btc_only_test_metrics["sharpe"]
            beats_drawdown = candidate_metrics["max_drawdown"] > btc_only_test_metrics["max_drawdown"]

            rows.append(
                {
                    "overlay_cost_bps": float(overlay_cost_bps),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "candidate": candidate_name,
                    "test_total_return": float(candidate_metrics["total_return"]),
                    "test_annual_return": float(candidate_metrics["annual_return"]),
                    "test_sharpe": float(candidate_metrics["sharpe"]),
                    "test_max_drawdown": float(candidate_metrics["max_drawdown"]),
                    "test_worst_month_return": float(candidate_metrics["worst_month_return"]),
                    "btc_only_test_total_return": float(btc_only_test_metrics["total_return"]),
                    "btc_only_test_annual_return": float(btc_only_test_metrics["annual_return"]),
                    "btc_only_test_sharpe": float(btc_only_test_metrics["sharpe"]),
                    "btc_only_test_max_drawdown": float(btc_only_test_metrics["max_drawdown"]),
                    "btc_only_test_worst_month_return": float(btc_only_test_metrics["worst_month_return"]),
                    "excess_return_vs_btc": float(candidate_metrics["total_return"] - btc_only_test_metrics["total_return"]),
                    "excess_annual_return_vs_btc": float(candidate_metrics["annual_return"] - btc_only_test_metrics["annual_return"]),
                    "excess_sharpe_vs_btc": float(candidate_metrics["sharpe"] - btc_only_test_metrics["sharpe"]),
                    "drawdown_delta_vs_btc": float(candidate_metrics["max_drawdown"] - btc_only_test_metrics["max_drawdown"]),
                    "worst_month_delta_vs_btc": float(candidate_metrics["worst_month_return"] - btc_only_test_metrics["worst_month_return"]),
                    "beats_btc_return": bool(beats_return),
                    "beats_btc_sharpe": bool(beats_sharpe),
                    "beats_btc_drawdown": bool(beats_drawdown),
                }
            )

            candidate_stitched_returns[candidate_name].append(candidate_series)

    fixed_comparison_report = pd.DataFrame(rows)
    if fixed_comparison_report.empty:
        raise ValueError("No fixed-candidate comparison rows produced")

    aggregate_rows: list[dict[str, Any]] = []
    grouped = fixed_comparison_report.groupby(["overlay_cost_bps", "candidate"], dropna=False)
    for (overlay, candidate), group in grouped:
        stitched = pd.concat(candidate_stitched_returns[str(candidate)], axis=0).sort_index()
        stitched_metrics = compute_period_metrics(stitched, bars_per_year=bars_per_year)
        number_of_windows = int(len(group))
        beats_return_years = int(group["beats_btc_return"].astype(bool).sum())
        beats_sharpe_years = int(group["beats_btc_sharpe"].astype(bool).sum())
        beats_drawdown_years = int(group["beats_btc_drawdown"].astype(bool).sum())

        aggregate_rows.append(
            {
                "overlay_cost_bps": float(overlay),
                "candidate": str(candidate),
                "compounded_test_total_return": float(stitched_metrics["total_return"]),
                "walk_forward_sharpe": float(stitched_metrics["sharpe"]),
                "max_test_drawdown": float(stitched_metrics["max_drawdown"]),
                "worst_test_month": float(stitched_metrics["worst_month_return"]),
                "avg_excess_return_vs_btc": float(group["excess_return_vs_btc"].mean()),
                "avg_excess_sharpe_vs_btc": float(group["excess_sharpe_vs_btc"].mean()),
                "avg_drawdown_delta_vs_btc": float(group["drawdown_delta_vs_btc"].mean()),
                "beats_btc_return_years": beats_return_years,
                "beats_btc_sharpe_years": beats_sharpe_years,
                "beats_btc_drawdown_years": beats_drawdown_years,
                "number_of_test_windows": number_of_windows,
                "negative_test_years": int((group["test_total_return"] < 0.0).sum()),
                "percent_years_beating_btc_return": float(beats_return_years / number_of_windows if number_of_windows else 0.0),
                "percent_years_beating_btc_sharpe": float(beats_sharpe_years / number_of_windows if number_of_windows else 0.0),
                "percent_years_beating_btc_drawdown": float(beats_drawdown_years / number_of_windows if number_of_windows else 0.0),
            }
        )

    fixed_aggregate_report = pd.DataFrame(aggregate_rows).sort_values(
        ["overlay_cost_bps", "candidate"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return fixed_comparison_report, fixed_aggregate_report


def run_portfolio_walk_forward_validation(
    overlay_cost_bps_values: Sequence[float] = WALK_FORWARD_OVERLAY_COST_BPS,
    gate_lag_bars: int = WALK_FORWARD_GATE_LAG_BARS,
    close_prices: pd.DataFrame | None = None,
    data_dir: Path = SETTINGS.data_dir,
    save_csv: bool = True,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    ranking_output_path: Path = DEFAULT_RANKINGS_OUTPUT_PATH,
    fixed_comparison_output_path: Path = DEFAULT_FIXED_COMPARISON_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars_per_year, scenario_streams = _build_candidate_return_streams(
        overlay_cost_bps_values=overlay_cost_bps_values,
        gate_lag_bars=gate_lag_bars,
        close_prices=close_prices,
        data_dir=data_dir,
    )

    all_steps: list[pd.DataFrame] = []
    all_aggregates: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    all_fixed_comparisons: list[pd.DataFrame] = []
    all_fixed_aggregates: list[pd.DataFrame] = []
    for overlay_cost_bps, candidate_returns in scenario_streams.items():
        step_report, aggregate_report, ranking_report = run_walk_forward_for_scenario(
            candidate_returns=candidate_returns,
            bars_per_year=bars_per_year,
            overlay_cost_bps=float(overlay_cost_bps),
        )
        fixed_comparison_report, fixed_aggregate_report = build_fixed_candidate_comparison_for_scenario(
            candidate_returns=candidate_returns,
            bars_per_year=bars_per_year,
            overlay_cost_bps=float(overlay_cost_bps),
        )
        all_steps.append(step_report)
        all_aggregates.append(aggregate_report)
        all_rankings.append(ranking_report)
        all_fixed_comparisons.append(fixed_comparison_report)
        all_fixed_aggregates.append(fixed_aggregate_report)

    steps = pd.concat(all_steps, axis=0, ignore_index=True)
    aggregates = pd.concat(all_aggregates, axis=0, ignore_index=True)
    rankings = pd.concat(all_rankings, axis=0, ignore_index=True)
    fixed_comparison = pd.concat(all_fixed_comparisons, axis=0, ignore_index=True)
    fixed_aggregates = pd.concat(all_fixed_aggregates, axis=0, ignore_index=True)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_output_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        steps.to_csv(output_path, index=False)
        rankings.to_csv(ranking_output_path, index=False)
        fixed_comparison.to_csv(fixed_comparison_output_path, index=False)

    return steps, aggregates, rankings, fixed_comparison, fixed_aggregates
