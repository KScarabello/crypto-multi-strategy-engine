"""Tests for portfolio walk-forward validation."""

from __future__ import annotations

import pandas as pd

from research.portfolio_walk_forward_validation import (
    CANDIDATE_NAMES,
    build_fixed_candidate_comparison_for_scenario,
    compute_period_metrics,
    build_walk_forward_splits,
    run_walk_forward_for_scenario,
    select_candidate_from_training,
)


def _synthetic_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", "2026-08-01", freq="4h", tz="UTC")


def _candidate_returns_for_logic() -> dict[str, pd.Series]:
    idx = _synthetic_index()

    # BTC baseline: moderate, stable.
    btc = pd.Series(0.0001, index=idx, dtype=float)

    # Strong in-train (<=2024), weak in late years.
    gated_7525 = pd.Series(0.0, index=idx, dtype=float)
    gated_7525.loc[idx.year <= 2024] = 0.0002
    gated_7525.loc[idx.year >= 2025] = -0.00015

    # Slightly weaker in-train, stronger in 2025+.
    always_8515 = pd.Series(0.0, index=idx, dtype=float)
    always_8515.loc[idx.year <= 2024] = 0.00012
    always_8515.loc[idx.year >= 2025] = 0.0002

    # Intentionally poor to trigger filters.
    poor = pd.Series(-0.0005, index=idx, dtype=float)

    return {
        "BTC_ONLY": btc,
        "ALWAYS_ON_90_10": always_8515 * 0.9,
        "ALWAYS_ON_85_15": always_8515,
        "ALWAYS_ON_75_25": always_8515 * 0.8,
        "GATED_BTC_TS_INVESTED_90_10": poor,
        "GATED_BTC_TS_INVESTED_85_15": poor,
        "GATED_BTC_TS_INVESTED_75_25": gated_7525,
    }


def test_train_test_periods_do_not_overlap() -> None:
    splits = build_walk_forward_splits(_synthetic_index(), start_test_year=2022)
    assert splits
    for split in splits:
        assert split["train_end"] < split["test_start"]


def test_training_windows_end_before_test_windows_start() -> None:
    splits = build_walk_forward_splits(_synthetic_index(), start_test_year=2022)
    assert splits
    for split in splits:
        assert split["train_start"] <= split["train_end"]
        assert split["train_end"] < split["test_start"]
        assert split["test_start"] <= split["test_end"]


def test_selection_uses_training_only_metrics() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    splits = build_walk_forward_splits(_synthetic_index(), start_test_year=2022)
    first_split = splits[0]
    train_start = pd.Timestamp(first_split["train_start"])
    train_end = pd.Timestamp(first_split["train_end"])

    idx = next(iter(candidate_returns.values())).index
    train_mask = (idx >= train_start) & (idx <= train_end)
    train_rows = []
    for name, series in candidate_returns.items():
        metrics = compute_period_metrics(series.loc[train_mask], bars_per_year=2190)
        train_rows.append(
            {
                "candidate_name": name,
                "train_sharpe": metrics["sharpe"],
                "train_total_return": metrics["total_return"],
                "train_max_drawdown": metrics["max_drawdown"],
                "train_count_negative_years": metrics["count_negative_years"],
            }
        )
    expected = select_candidate_from_training(pd.DataFrame(train_rows), drawdown_floor=-0.50)

    early = steps.loc[steps["test_start"].str.startswith("2022")]
    assert not early.empty
    assert str(early.iloc[0]["selected_candidate"]) == expected


def test_fallback_to_btc_only_when_filters_eliminate_candidates() -> None:
    train_metrics = pd.DataFrame(
        [
            {
                "candidate_name": "BTC_ONLY",
                "train_sharpe": 0.1,
                "train_total_return": 0.05,
                "train_max_drawdown": -0.9,
                "train_count_negative_years": 5,
            },
            {
                "candidate_name": "ALWAYS_ON_75_25",
                "train_sharpe": 1.2,
                "train_total_return": 0.3,
                "train_max_drawdown": -0.8,
                "train_count_negative_years": 6,
            },
        ]
    )
    selected = select_candidate_from_training(train_metrics, drawdown_floor=-0.5, fallback_candidate="BTC_ONLY")
    assert selected == "BTC_ONLY"


def test_required_output_columns_present() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, rankings = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=30.0,
    )

    required = {
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "selected_candidate",
        "overlay_cost_bps",
        "train_total_return",
        "train_cagr",
        "train_sharpe",
        "train_max_drawdown",
        "train_count_negative_years",
        "test_total_return",
        "test_annual_return",
        "test_sharpe",
        "test_max_drawdown",
        "test_worst_month_return",
        "fixed_candidate",
        "fixed_test_total_return",
        "fixed_test_annual_return",
        "fixed_test_sharpe",
        "fixed_test_max_drawdown",
        "fixed_test_worst_month_return",
        "selector_excess_return_vs_fixed",
        "selector_excess_annual_return_vs_fixed",
        "selector_excess_sharpe_vs_fixed",
        "selector_drawdown_delta_vs_fixed",
        "selector_worst_month_delta_vs_fixed",
        "selector_beats_fixed_return",
        "selector_beats_fixed_sharpe",
        "selector_beats_fixed_drawdown",
        "btc_only_test_total_return",
        "btc_only_test_annual_return",
        "btc_only_test_sharpe",
        "btc_only_test_max_drawdown",
        "btc_only_test_worst_month_return",
        "test_excess_return_vs_btc",
        "test_excess_annual_return_vs_btc",
        "test_excess_sharpe_vs_btc",
        "test_drawdown_delta_vs_btc",
        "test_worst_month_delta_vs_btc",
        "test_beats_btc_only_return",
        "test_beats_btc_only_sharpe",
        "test_beats_btc_only_drawdown",
    }
    assert required.issubset(set(steps.columns))

    ranking_required = {
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
    }
    assert ranking_required.issubset(set(rankings.columns))


def test_selected_vs_btc_delta_columns_are_consistent() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )
    row = steps.iloc[0]

    assert float(row["test_excess_return_vs_btc"]) == float(row["test_total_return"] - row["btc_only_test_total_return"])
    assert float(row["test_excess_annual_return_vs_btc"]) == float(row["test_annual_return"] - row["btc_only_test_annual_return"])
    assert float(row["test_excess_sharpe_vs_btc"]) == float(row["test_sharpe"] - row["btc_only_test_sharpe"])
    assert float(row["test_drawdown_delta_vs_btc"]) == float(row["test_max_drawdown"] - row["btc_only_test_max_drawdown"])
    assert float(row["test_worst_month_delta_vs_btc"]) == float(row["test_worst_month_return"] - row["btc_only_test_worst_month_return"])


def test_selector_vs_fixed_delta_columns_are_consistent() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )
    row = steps.iloc[0]

    assert float(row["selector_excess_return_vs_fixed"]) == float(row["test_total_return"] - row["fixed_test_total_return"])
    assert float(row["selector_excess_annual_return_vs_fixed"]) == float(row["test_annual_return"] - row["fixed_test_annual_return"])
    assert float(row["selector_excess_sharpe_vs_fixed"]) == float(row["test_sharpe"] - row["fixed_test_sharpe"])
    assert float(row["selector_drawdown_delta_vs_fixed"]) == float(row["test_max_drawdown"] - row["fixed_test_max_drawdown"])
    assert float(row["selector_worst_month_delta_vs_fixed"]) == float(row["test_worst_month_return"] - row["fixed_test_worst_month_return"])


def test_selector_vs_fixed_boolean_flags_are_consistent() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=30.0,
    )
    row = steps.iloc[0]

    assert bool(row["selector_beats_fixed_return"]) == bool(row["test_total_return"] > row["fixed_test_total_return"])
    assert bool(row["selector_beats_fixed_sharpe"]) == bool(row["test_sharpe"] > row["fixed_test_sharpe"])
    assert bool(row["selector_beats_fixed_drawdown"]) == bool(row["test_max_drawdown"] > row["fixed_test_max_drawdown"])


def test_selector_vs_fixed_deltas_zero_when_selected_equals_fixed() -> None:
    idx = _synthetic_index()
    fixed = pd.Series(0.0002, index=idx, dtype=float)
    poor = pd.Series(-0.0004, index=idx, dtype=float)
    btc = pd.Series(0.00005, index=idx, dtype=float)
    candidate_returns = {
        "BTC_ONLY": btc,
        "ALWAYS_ON_90_10": poor,
        "ALWAYS_ON_85_15": poor,
        "ALWAYS_ON_75_25": poor,
        "GATED_BTC_TS_INVESTED_90_10": poor,
        "GATED_BTC_TS_INVESTED_85_15": poor,
        "GATED_BTC_TS_INVESTED_75_25": fixed,
    }

    steps, _, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    assert (steps["selected_candidate"] == steps["fixed_candidate"]).all()
    assert (steps["selector_excess_return_vs_fixed"].abs() < 1e-12).all()
    assert (steps["selector_excess_annual_return_vs_fixed"].abs() < 1e-12).all()
    assert (steps["selector_excess_sharpe_vs_fixed"].abs() < 1e-12).all()
    assert (steps["selector_drawdown_delta_vs_fixed"].abs() < 1e-12).all()
    assert (steps["selector_worst_month_delta_vs_fixed"].abs() < 1e-12).all()


def test_candidate_ranking_contains_all_candidates_and_one_selected_per_split() -> None:
    candidate_returns = _candidate_returns_for_logic()
    steps, _, rankings = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=30.0,
    )

    expected_candidates = set(candidate_returns.keys())
    group_cols = ["overlay_cost_bps", "train_start", "train_end", "test_start", "test_end"]
    grouped = rankings.groupby(group_cols, dropna=False)

    assert len(grouped) == len(steps)
    for _, group in grouped:
        assert set(group["candidate"].tolist()) == expected_candidates
        assert int(group["is_selected"].sum()) == 1
        selected_name = str(group.loc[group["is_selected"], "candidate"].iloc[0])
        declared_name = str(group["selected_candidate"].iloc[0])
        assert selected_name == declared_name


def test_failed_candidates_are_kept_in_ranking_output() -> None:
    candidate_returns = _candidate_returns_for_logic()
    _, _, rankings = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    failed = rankings.loc[~rankings["passes_all_filters"]]
    assert not failed.empty
    assert failed["selection_rank"].isna().all()


def test_aggregate_metrics_are_produced() -> None:
    candidate_returns = _candidate_returns_for_logic()
    _, aggregates, _ = run_walk_forward_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    assert len(aggregates) == 1
    row = aggregates.iloc[0]
    assert "compounded_walk_forward_total_return" in aggregates.columns
    assert "walk_forward_sharpe" in aggregates.columns
    assert "fixed_compounded_test_total_return" in aggregates.columns
    assert "fixed_walk_forward_sharpe" in aggregates.columns
    assert "selector_compounded_test_total_return" in aggregates.columns
    assert "selector_walk_forward_sharpe" in aggregates.columns
    assert "selector_excess_compounded_return_vs_fixed" in aggregates.columns
    assert "selector_excess_walk_forward_sharpe_vs_fixed" in aggregates.columns
    assert "selector_beats_fixed_return_years" in aggregates.columns
    assert "selector_beats_fixed_sharpe_years" in aggregates.columns
    assert "selector_beats_fixed_drawdown_years" in aggregates.columns
    assert "number_of_test_years_beating_btc_only_on_return" in aggregates.columns
    assert float(row["overlay_cost_bps"]) == 15.0


def test_fixed_candidate_comparison_rows_cover_all_candidates_per_window() -> None:
    candidate_returns = _candidate_returns_for_logic()
    splits = build_walk_forward_splits(_synthetic_index(), start_test_year=2022)

    comparison, _ = build_fixed_candidate_comparison_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
        splits=splits,
    )

    expected_rows = len(splits) * len(candidate_returns)
    assert len(comparison) == expected_rows
    assert set(comparison["candidate"].tolist()) == set(candidate_returns.keys())

    grouped = comparison.groupby(["overlay_cost_bps", "test_start", "test_end"], dropna=False)
    assert len(grouped) == len(splits)
    for _, group in grouped:
        assert set(group["candidate"].tolist()) == set(candidate_returns.keys())


def test_fixed_candidate_comparison_required_columns_present() -> None:
    candidate_returns = _candidate_returns_for_logic()
    comparison, _ = build_fixed_candidate_comparison_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=30.0,
    )

    required = {
        "overlay_cost_bps",
        "test_start",
        "test_end",
        "candidate",
        "test_total_return",
        "test_annual_return",
        "test_sharpe",
        "test_max_drawdown",
        "test_worst_month_return",
        "btc_only_test_total_return",
        "btc_only_test_annual_return",
        "btc_only_test_sharpe",
        "btc_only_test_max_drawdown",
        "btc_only_test_worst_month_return",
        "excess_return_vs_btc",
        "excess_annual_return_vs_btc",
        "excess_sharpe_vs_btc",
        "drawdown_delta_vs_btc",
        "worst_month_delta_vs_btc",
        "beats_btc_return",
        "beats_btc_sharpe",
        "beats_btc_drawdown",
    }
    assert required.issubset(set(comparison.columns))


def test_fixed_candidate_btc_only_rows_have_zero_deltas() -> None:
    candidate_returns = _candidate_returns_for_logic()
    comparison, _ = build_fixed_candidate_comparison_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    btc_rows = comparison.loc[comparison["candidate"] == "BTC_ONLY"]
    assert not btc_rows.empty
    assert (btc_rows["excess_return_vs_btc"].abs() < 1e-12).all()
    assert (btc_rows["excess_annual_return_vs_btc"].abs() < 1e-12).all()
    assert (btc_rows["excess_sharpe_vs_btc"].abs() < 1e-12).all()
    assert (btc_rows["drawdown_delta_vs_btc"].abs() < 1e-12).all()
    assert (btc_rows["worst_month_delta_vs_btc"].abs() < 1e-12).all()
    assert (~btc_rows["beats_btc_return"].astype(bool)).all()
    assert (~btc_rows["beats_btc_sharpe"].astype(bool)).all()
    assert (~btc_rows["beats_btc_drawdown"].astype(bool)).all()


def test_fixed_candidate_delta_and_boolean_columns_are_consistent() -> None:
    candidate_returns = _candidate_returns_for_logic()
    comparison, _ = build_fixed_candidate_comparison_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=30.0,
    )

    row = comparison.loc[comparison["candidate"] == "ALWAYS_ON_85_15"].iloc[0]
    assert float(row["excess_return_vs_btc"]) == float(row["test_total_return"] - row["btc_only_test_total_return"])
    assert float(row["excess_annual_return_vs_btc"]) == float(row["test_annual_return"] - row["btc_only_test_annual_return"])
    assert float(row["excess_sharpe_vs_btc"]) == float(row["test_sharpe"] - row["btc_only_test_sharpe"])
    assert float(row["drawdown_delta_vs_btc"]) == float(row["test_max_drawdown"] - row["btc_only_test_max_drawdown"])
    assert float(row["worst_month_delta_vs_btc"]) == float(row["test_worst_month_return"] - row["btc_only_test_worst_month_return"])
    assert bool(row["beats_btc_return"]) == bool(row["test_total_return"] > row["btc_only_test_total_return"])
    assert bool(row["beats_btc_sharpe"]) == bool(row["test_sharpe"] > row["btc_only_test_sharpe"])
    assert bool(row["beats_btc_drawdown"]) == bool(row["test_max_drawdown"] > row["btc_only_test_max_drawdown"])


def test_fixed_candidate_aggregate_has_expected_rows_and_consistent_counts() -> None:
    candidate_returns = _candidate_returns_for_logic()
    comparison, aggregates = build_fixed_candidate_comparison_for_scenario(
        candidate_returns=candidate_returns,
        bars_per_year=2190,
        overlay_cost_bps=15.0,
    )

    expected_candidates = [name for name in CANDIDATE_NAMES if name in candidate_returns]
    assert len(aggregates) == len(expected_candidates)
    assert set(aggregates["candidate"].tolist()) == set(expected_candidates)

    required = {
        "overlay_cost_bps",
        "candidate",
        "compounded_test_total_return",
        "walk_forward_sharpe",
        "max_test_drawdown",
        "worst_test_month",
        "avg_excess_return_vs_btc",
        "avg_excess_sharpe_vs_btc",
        "avg_drawdown_delta_vs_btc",
        "beats_btc_return_years",
        "beats_btc_sharpe_years",
        "beats_btc_drawdown_years",
        "number_of_test_windows",
        "negative_test_years",
        "percent_years_beating_btc_return",
        "percent_years_beating_btc_sharpe",
        "percent_years_beating_btc_drawdown",
    }
    assert required.issubset(set(aggregates.columns))

    for _, aggregate_row in aggregates.iterrows():
        candidate = str(aggregate_row["candidate"])
        candidate_rows = comparison.loc[comparison["candidate"] == candidate]
        assert int(aggregate_row["number_of_test_windows"]) == int(len(candidate_rows))
        assert int(aggregate_row["beats_btc_return_years"]) == int(candidate_rows["beats_btc_return"].astype(bool).sum())
        assert int(aggregate_row["beats_btc_sharpe_years"]) == int(candidate_rows["beats_btc_sharpe"].astype(bool).sum())
        assert int(aggregate_row["beats_btc_drawdown_years"]) == int(candidate_rows["beats_btc_drawdown"].astype(bool).sum())
