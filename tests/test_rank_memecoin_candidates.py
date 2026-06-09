"""Tests for research/memecoin_catcher/rank_memecoin_candidates.py

All unit-level — no network calls, no file I/O except tmp_path fixtures.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from research.memecoin_catcher.rank_memecoin_candidates import (
    apply_liquidity_filters,
    apply_universe_filters,
    assign_scanner_label,
    assign_status_flags,
    add_ohlc_placeholders,
    build_candidates_df,
    build_diagnostics_df,
    compute_explosion_score,
    exclude_fiat_stable,
    load_snapshot,
    write_candidates_csv,
    write_diagnostics_csv,
    FIAT_STABLE_BASES,
    CANDIDATE_COLUMNS,
    DIAGNOSTICS_COLUMNS,
    OHLC_PLACEHOLDER_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    eligible: bool = True,
    last_price: float = 1.0,
    bid: float = 0.99,
    ask: float = 1.01,
    open_price: float = 0.95,
    spread_pct: float | None = 2.0,
    today_return_pct: float | None = 5.27,
    volume_24h: float = 50_000.0,
    quote_volume_est: float | None = 50_000.0,
    base: str = "SOL",
    quote: str = "USD",
    pair_id: str = "SOLUSD",
    wsname: str = "SOL/USD",
    status: str = "online",
    ineligible_reason: str = "",
) -> dict:
    return dict(
        pair_id=pair_id,
        altname=pair_id,
        wsname=wsname,
        base=base,
        quote=quote,
        status=status,
        last_price=last_price,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        open_price=open_price,
        today_return_pct=today_return_pct,
        volume_24h=volume_24h,
        quote_volume_est=quote_volume_est,
        eligible=eligible,
        ineligible_reason=ineligible_reason,
    )


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# apply_universe_filters
# ---------------------------------------------------------------------------


def test_universe_filter_keeps_complete_eligible_row() -> None:
    df = _df(_make_row())
    result = apply_universe_filters(df)
    assert len(result) == 1


def test_universe_filter_drops_ineligible() -> None:
    df = _df(_make_row(eligible=False, ineligible_reason="quote_not_usd_like:EUR"))
    result = apply_universe_filters(df)
    assert len(result) == 0


def test_universe_filter_drops_zero_last_price() -> None:
    df = _df(_make_row(last_price=0.0))
    result = apply_universe_filters(df)
    assert len(result) == 0


def test_universe_filter_drops_null_spread() -> None:
    df = _df(_make_row(spread_pct=None))
    result = apply_universe_filters(df)
    assert len(result) == 0


def test_universe_filter_drops_null_today_return() -> None:
    df = _df(_make_row(today_return_pct=None))
    result = apply_universe_filters(df)
    assert len(result) == 0


def test_universe_filter_drops_null_quote_volume() -> None:
    df = _df(_make_row(quote_volume_est=None))
    result = apply_universe_filters(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# exclude_fiat_stable
# ---------------------------------------------------------------------------


def test_exclude_fiat_stable_drops_usdt_base() -> None:
    df = _df(_make_row(base="USDT", wsname="USDT/USD"))
    result = exclude_fiat_stable(df)
    assert len(result) == 0


def test_exclude_fiat_stable_drops_eur_base() -> None:
    df = _df(_make_row(base="EUR", wsname="EUR/USD"))
    result = exclude_fiat_stable(df)
    assert len(result) == 0


def test_exclude_fiat_stable_case_insensitive() -> None:
    df = _df(_make_row(base="usdc", wsname="usdc/USD"))
    result = exclude_fiat_stable(df)
    assert len(result) == 0


def test_exclude_fiat_stable_keeps_crypto() -> None:
    df = _df(_make_row(base="SOL", wsname="SOL/USD"))
    result = exclude_fiat_stable(df)
    assert len(result) == 1


def test_fiat_stable_bases_covers_required_symbols() -> None:
    required = {"USD", "USDT", "USDC", "DAI", "EUR", "EURC", "GBP", "AUD"}
    assert required.issubset(FIAT_STABLE_BASES)


def test_fiat_stable_bases_excludes_btc_aliases() -> None:
    for sym in ("XBT", "XXBT", "ETH", "XETH", "LTC", "XLTC", "XRP", "XXRP"):
        assert sym in FIAT_STABLE_BASES, f"{sym} should be in FIAT_STABLE_BASES"


def test_fiat_stable_bases_excludes_z_prefixed_fiat() -> None:
    for sym in ("ZUSD", "ZEUR", "ZGBP", "ZAUD", "ZCAD", "ZJPY"):
        assert sym in FIAT_STABLE_BASES, f"{sym} should be in FIAT_STABLE_BASES"


# ---------------------------------------------------------------------------
# apply_liquidity_filters
# ---------------------------------------------------------------------------


def test_liquidity_filter_passes_good_row() -> None:
    df = _df(_make_row(spread_pct=0.5, quote_volume_est=50_000, today_return_pct=10.0))
    candidates, excluded = apply_liquidity_filters(df)
    assert len(candidates) == 1
    assert len(excluded) == 0


def test_liquidity_filter_excludes_huge_spread() -> None:
    df = _df(_make_row(spread_pct=5.0, quote_volume_est=50_000, today_return_pct=10.0))
    candidates, excluded = apply_liquidity_filters(df)
    assert len(candidates) == 0
    assert len(excluded) == 1
    assert "spread_pct" in excluded.iloc[0]["excluded_reason"]


def test_liquidity_filter_excludes_low_volume() -> None:
    df = _df(_make_row(spread_pct=0.5, quote_volume_est=500, today_return_pct=10.0))
    candidates, excluded = apply_liquidity_filters(df)
    assert len(candidates) == 0
    assert "quote_volume_est" in excluded.iloc[0]["excluded_reason"]


def test_liquidity_filter_excludes_low_abs_return() -> None:
    df = _df(_make_row(spread_pct=0.5, quote_volume_est=50_000, today_return_pct=0.5))
    candidates, excluded = apply_liquidity_filters(df)
    assert len(candidates) == 0
    assert "abs_return" in excluded.iloc[0]["excluded_reason"]


def test_liquidity_filter_negative_return_passes_abs_threshold() -> None:
    df = _df(_make_row(spread_pct=0.5, quote_volume_est=50_000, today_return_pct=-10.0))
    candidates, excluded = apply_liquidity_filters(df)
    assert len(candidates) == 1


def test_excluded_rows_have_excluded_reason() -> None:
    rows = [
        _make_row(pair_id="A", wsname="A/USD", spread_pct=5.0, quote_volume_est=50_000, today_return_pct=10.0),
        _make_row(pair_id="B", wsname="B/USD", spread_pct=0.5, quote_volume_est=500, today_return_pct=10.0),
        _make_row(pair_id="C", wsname="C/USD", spread_pct=0.5, quote_volume_est=50_000, today_return_pct=0.1),
    ]
    _, excluded = apply_liquidity_filters(_df(*rows))
    for _, row in excluded.iterrows():
        assert row["excluded_reason"] != "", f"Row {row['pair_id']} missing excluded_reason"


# ---------------------------------------------------------------------------
# assign_status_flags
# ---------------------------------------------------------------------------


def test_status_flags_hot_mover() -> None:
    df = _df(_make_row())
    df["scanner_label"] = "HOT_MOVER"
    result = assign_status_flags(df)
    assert result.iloc[0]["is_hot_mover"] == True   # noqa: E712
    assert result.iloc[0]["is_watch"] == False       # noqa: E712
    assert result.iloc[0]["is_dumping"] == False     # noqa: E712
    assert result.iloc[0]["is_candidate"] == True    # noqa: E712


def test_status_flags_watch() -> None:
    df = _df(_make_row())
    df["scanner_label"] = "WATCH"
    result = assign_status_flags(df)
    assert result.iloc[0]["is_watch"] == True        # noqa: E712
    assert result.iloc[0]["is_hot_mover"] == False   # noqa: E712
    assert result.iloc[0]["is_candidate"] == True    # noqa: E712


def test_status_flags_dumping() -> None:
    df = _df(_make_row())
    df["scanner_label"] = "DUMPING"
    result = assign_status_flags(df)
    assert result.iloc[0]["is_dumping"] == True      # noqa: E712
    assert result.iloc[0]["is_candidate"] == True    # noqa: E712


def test_status_flags_blank_label_not_candidate() -> None:
    df = _df(_make_row())
    df["scanner_label"] = ""
    result = assign_status_flags(df)
    assert result.iloc[0]["is_candidate"] == False   # noqa: E712
    assert result.iloc[0]["is_hot_mover"] == False   # noqa: E712
    assert result.iloc[0]["is_watch"] == False       # noqa: E712
    assert result.iloc[0]["is_dumping"] == False     # noqa: E712


# ---------------------------------------------------------------------------
# add_ohlc_placeholders
# ---------------------------------------------------------------------------


def test_ohlc_placeholders_all_present() -> None:
    df = _df(_make_row())
    result = add_ohlc_placeholders(df)
    for col in OHLC_PLACEHOLDER_COLUMNS:
        assert col in result.columns, f"Missing OHLC placeholder: {col}"


def test_ohlc_numeric_placeholders_are_null() -> None:
    df = _df(_make_row())
    result = add_ohlc_placeholders(df)
    for col in ("ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "ret_24h_pct",
                "volume_ratio_1h", "volume_ratio_4h"):
        assert pd.isna(result.iloc[0][col]), f"{col} should be null"


def test_ohlc_bool_placeholders_are_false() -> None:
    df = _df(_make_row())
    result = add_ohlc_placeholders(df)
    assert result.iloc[0]["breakout_24h"] == False   # noqa: E712
    assert result.iloc[0]["ohlc_ready"] == False     # noqa: E712


# ---------------------------------------------------------------------------
# assign_scanner_label
# ---------------------------------------------------------------------------


def test_label_hot_mover() -> None:
    df = _df(_make_row(today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.5))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] == "HOT_MOVER"


def test_label_watch() -> None:
    df = _df(_make_row(today_return_pct=3.0, quote_volume_est=15_000, spread_pct=1.5))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] == "WATCH"


def test_label_dumping() -> None:
    df = _df(_make_row(today_return_pct=-8.0, quote_volume_est=50_000, spread_pct=0.5))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] == "DUMPING"


def test_label_hot_mover_takes_priority_over_watch() -> None:
    # today_return_pct >= 5, quote_volume_est >= 25000, spread_pct <= 1.0 → HOT_MOVER
    df = _df(_make_row(today_return_pct=7.0, quote_volume_est=30_000, spread_pct=0.8))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] == "HOT_MOVER"


def test_label_blank_for_borderline() -> None:
    # return < 2%, volume low, spread high → no label
    df = _df(_make_row(today_return_pct=1.0, quote_volume_est=5_000, spread_pct=3.0))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] == ""


def test_label_dumping_not_applied_for_high_spread() -> None:
    # spread > 1.0 → should not get DUMPING
    df = _df(_make_row(today_return_pct=-10.0, quote_volume_est=50_000, spread_pct=1.5))
    labels = assign_scanner_label(df)
    assert labels.iloc[0] != "DUMPING"


# ---------------------------------------------------------------------------
# compute_explosion_score
# ---------------------------------------------------------------------------


def test_explosion_score_base_is_today_return() -> None:
    df = _df(_make_row(today_return_pct=5.0, quote_volume_est=1_000, spread_pct=1.5))
    score = compute_explosion_score(df).iloc[0]
    # No volume bonuses, no spread bonuses, spread>1 so -5
    assert score == pytest.approx(5.0 - 5.0)


def test_explosion_score_volume_bonuses() -> None:
    df = _df(_make_row(today_return_pct=0.0, quote_volume_est=200_000, spread_pct=0.1))
    score = compute_explosion_score(df).iloc[0]
    # +5 (>=100k) + +3 (>=25k) + +2 (<=0.25) + +1 (<=0.50)
    assert score == pytest.approx(0.0 + 5 + 3 + 2 + 1)


def test_explosion_score_penalty_for_wide_spread() -> None:
    df = _df(_make_row(today_return_pct=10.0, quote_volume_est=1_000, spread_pct=2.0))
    score = compute_explosion_score(df).iloc[0]
    assert score == pytest.approx(10.0 - 5.0)


# ---------------------------------------------------------------------------
# build_candidates_df
# ---------------------------------------------------------------------------


def _make_scored_candidates(*rows: dict) -> pd.DataFrame:
    """Helper: run liquidity filters + scoring on rows."""
    df = _df(*rows)
    candidates, _ = apply_liquidity_filters(df)
    candidates = candidates.copy()
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)
    return candidates


def test_candidates_df_contains_only_labeled_rows() -> None:
    rows = [
        _make_row(pair_id="HOT", wsname="HOT/USD", today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.4),
        _make_row(pair_id="UNLABELED", wsname="UNLABELED/USD", today_return_pct=3.0, quote_volume_est=50_000, spread_pct=0.4),
    ]
    # UNLABELED has return 3% + vol >= 10k + spread <=2 => WATCH, so both labeled here
    # Let's make one with blank label (high spread -> no WATCH)
    rows = [
        _make_row(pair_id="HOT", wsname="HOT/USD", today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.4),
        _make_row(pair_id="NOLABEL", wsname="NOLABEL/USD", today_return_pct=3.0, quote_volume_est=50_000, spread_pct=0.4),
    ]
    scored = _make_scored_candidates(*rows)
    # Manually blank one label
    scored.loc[scored["pair_id"] == "NOLABEL", "scanner_label"] = ""
    out = build_candidates_df(scored)
    assert all(out["scanner_label"].isin(["HOT_MOVER", "WATCH", "DUMPING"]))
    assert "NOLABEL" not in out["pair_id"].values


def test_candidates_df_no_excluded_reason_column() -> None:
    """Candidates file should not expose excluded_reason."""
    scored = _make_scored_candidates(
        _make_row(today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    assert "excluded_reason" not in out.columns


def test_candidates_df_has_all_required_columns() -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    for col in CANDIDATE_COLUMNS:
        assert col in out.columns, f"Missing candidate column: {col}"


def test_candidates_df_has_ohlc_placeholder_columns() -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=10.0, quote_volume_est=50_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    for col in OHLC_PLACEHOLDER_COLUMNS:
        assert col in out.columns, f"Missing OHLC placeholder: {col}"


def test_candidates_df_sorted_by_label_priority_then_score() -> None:
    rows = [
        _make_row(pair_id="W1", wsname="W1/USD", today_return_pct=3.0, quote_volume_est=15_000, spread_pct=1.5),
        _make_row(pair_id="H1", wsname="H1/USD", today_return_pct=7.0, quote_volume_est=30_000, spread_pct=0.4),
        _make_row(pair_id="D1", wsname="D1/USD", today_return_pct=-8.0, quote_volume_est=30_000, spread_pct=0.4),
        _make_row(pair_id="H2", wsname="H2/USD", today_return_pct=15.0, quote_volume_est=50_000, spread_pct=0.2),
        _make_row(pair_id="W2", wsname="W2/USD", today_return_pct=4.0, quote_volume_est=12_000, spread_pct=0.9),
    ]
    scored = _make_scored_candidates(*rows)
    out = build_candidates_df(scored)
    labels = out["scanner_label"].tolist()
    # All HOT_MOVERs before all WATCHes before all DUMPINGs
    from research.memecoin_catcher.rank_memecoin_candidates import LABEL_ORDER
    order_vals = [LABEL_ORDER[lbl] for lbl in labels]
    assert order_vals == sorted(order_vals)
    # Within each label, score is descending
    for label in ("HOT_MOVER", "WATCH", "DUMPING"):
        grp = out[out["scanner_label"] == label]["explosion_score"].tolist()
        assert grp == sorted(grp, reverse=True), f"{label} group not sorted"


def test_btc_eth_xrp_ltc_aliases_not_candidates(tmp_path: Path) -> None:
    """Majors excluded from FIAT_STABLE_BASES must never appear as candidates."""
    major_rows = [
        _make_row(pair_id="XBTUSD", wsname="XBT/USD", base="XBT", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
        _make_row(pair_id="XXBTUSD", wsname="XBT/USD", base="XXBT", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
        _make_row(pair_id="ETHUSD", wsname="ETH/USD", base="ETH", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
        _make_row(pair_id="XETHUSD", wsname="ETH/USD", base="XETH", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
        _make_row(pair_id="XRPUSD", wsname="XRP/USD", base="XRP", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
        _make_row(pair_id="LTCUSD", wsname="LTC/USD", base="LTC", today_return_pct=10.0, quote_volume_est=5_000_000, spread_pct=0.1),
    ]
    df = _df(*major_rows)
    after_stable = exclude_fiat_stable(df)
    assert len(after_stable) == 0, "Major coin aliases should be excluded"


# ---------------------------------------------------------------------------
# build_diagnostics_df
# ---------------------------------------------------------------------------


def test_diagnostics_includes_excluded_rows() -> None:
    good = _make_row(pair_id="GOOD", wsname="GOOD/USD", spread_pct=0.5, quote_volume_est=50_000, today_return_pct=10.0)
    bad = _make_row(pair_id="BAD", wsname="BAD/USD", spread_pct=5.0, quote_volume_est=50_000, today_return_pct=10.0)
    df = _df(good, bad)
    candidates, excluded = apply_liquidity_filters(df)
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)
    diag = build_diagnostics_df(candidates, excluded)
    pair_ids = diag["pair_id"].tolist()
    assert "GOOD" in pair_ids
    assert "BAD" in pair_ids


def test_diagnostics_excluded_rows_have_reason() -> None:
    bad = _make_row(pair_id="BAD", wsname="BAD/USD", spread_pct=5.0, quote_volume_est=50_000, today_return_pct=10.0)
    df = _df(bad)
    candidates, excluded = apply_liquidity_filters(df)
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)
    diag = build_diagnostics_df(candidates, excluded)
    bad_row = diag[diag["pair_id"] == "BAD"].iloc[0]
    assert bad_row["excluded_reason"] != ""


def test_diagnostics_has_all_required_columns() -> None:
    df = _df(_make_row(spread_pct=0.5, quote_volume_est=50_000, today_return_pct=10.0))
    candidates, excluded = apply_liquidity_filters(df)
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)
    diag = build_diagnostics_df(candidates, excluded)
    for col in DIAGNOSTICS_COLUMNS:
        assert col in diag.columns, f"Missing diagnostics column: {col}"


# ---------------------------------------------------------------------------
# write_candidates_csv / write_diagnostics_csv
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path: Path) -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=5.0, quote_volume_est=30_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    dest = tmp_path / "out.csv"
    write_candidates_csv(out, dest)
    assert dest.exists()


def test_write_candidates_csv_excludes_excluded_reason_column(tmp_path: Path) -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=5.0, quote_volume_est=30_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    dest = tmp_path / "out.csv"
    write_candidates_csv(out, dest)
    header = dest.read_text().splitlines()[0]
    assert "excluded_reason" not in header


def test_write_diagnostics_csv_includes_excluded_reason(tmp_path: Path) -> None:
    df = _df(_make_row(spread_pct=5.0, quote_volume_est=50_000, today_return_pct=10.0))
    candidates, excluded = apply_liquidity_filters(df)
    candidates["explosion_score"] = compute_explosion_score(candidates)
    candidates["scanner_label"] = assign_scanner_label(candidates)
    diag = build_diagnostics_df(candidates, excluded)
    dest = tmp_path / "diag.csv"
    write_diagnostics_csv(diag, dest)
    header = dest.read_text().splitlines()[0]
    assert "excluded_reason" in header


def test_write_csv_has_correct_candidate_columns(tmp_path: Path) -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=5.0, quote_volume_est=30_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    dest = tmp_path / "out.csv"
    write_candidates_csv(out, dest)
    header = dest.read_text().splitlines()[0]
    for col in CANDIDATE_COLUMNS:
        assert col in header


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    scored = _make_scored_candidates(
        _make_row(today_return_pct=5.0, quote_volume_est=30_000, spread_pct=0.4)
    )
    out = build_candidates_df(scored)
    dest = tmp_path / "deep" / "nested" / "out.csv"
    write_candidates_csv(out, dest)
    assert dest.exists()


# ---------------------------------------------------------------------------
# load_snapshot (round-trip via tmp CSV)
# ---------------------------------------------------------------------------


def test_load_snapshot_round_trip(tmp_path: Path) -> None:
    import csv as _csv
    from research.memecoin_catcher.fetch_kraken_universe import OUTPUT_COLUMNS as SNAP_COLS
    dest = tmp_path / "snap.csv"
    row = _make_row()
    with dest.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=SNAP_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    df = load_snapshot(dest)
    assert len(df) == 1
    assert df.iloc[0]["last_price"] == pytest.approx(1.0)
    assert df.iloc[0]["eligible"] == True  # noqa: E712 — numpy bool comparison


# ---------------------------------------------------------------------------
# Module safety check
# ---------------------------------------------------------------------------


def test_module_does_not_import_live_or_broker_modules() -> None:
    module_path = Path("research/memecoin_catcher/rank_memecoin_candidates.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert all(not m.startswith("live") for m in imported)
    assert all(not m.startswith("brokers") for m in imported)
    assert all("config" not in m for m in imported)
