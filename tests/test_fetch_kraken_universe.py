"""Tests for research/memecoin_catcher/fetch_kraken_universe.py

All tests are unit-level — no network calls are made.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research.memecoin_catcher.fetch_kraken_universe import (
    _compute_spread_pct,
    _compute_today_return_pct,
    _eligibility_reason,
    _safe_float,
    enrich_with_ticker,
    normalize_pairs,
    write_snapshot_csv,
    OUTPUT_COLUMNS,
)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


def test_safe_float_with_string() -> None:
    assert _safe_float("1.2345") == pytest.approx(1.2345)


def test_safe_float_with_list_of_strings() -> None:
    # Kraken sends e.g. ["1.23456", 1] for ticker fields
    assert _safe_float(["98000.5", 1]) == pytest.approx(98000.5)


def test_safe_float_with_none_returns_default() -> None:
    assert _safe_float(None) == 0.0
    assert _safe_float(None, default=-1.0) == -1.0


def test_safe_float_with_empty_list_returns_default() -> None:
    assert _safe_float([]) == 0.0


def test_safe_float_with_non_numeric_string() -> None:
    assert _safe_float("bad") == 0.0


def test_safe_float_with_numeric() -> None:
    assert _safe_float(42.0) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Spread calculation
# ---------------------------------------------------------------------------


def test_spread_pct_basic() -> None:
    # bid=99, ask=101, mid=100, spread = 2/100*100 = 2.0
    result = _compute_spread_pct(bid=99.0, ask=101.0)
    assert result == pytest.approx(2.0)


def test_spread_pct_zero_mid_returns_zero() -> None:
    assert _compute_spread_pct(bid=0.0, ask=0.0) == 0.0


def test_spread_pct_tight_market() -> None:
    # bid=99900, ask=100100, mid=100000 → spread = 200/100000 * 100 = 0.2
    spread = _compute_spread_pct(bid=99_900.0, ask=100_100.0)
    assert spread == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# today_return_pct calculation
# ---------------------------------------------------------------------------


def test_today_return_pct_basic() -> None:
    result = _compute_today_return_pct(last_price=110.0, open_price=100.0)
    assert result == pytest.approx(10.0)


def test_today_return_pct_zero_open_returns_zero() -> None:
    assert _compute_today_return_pct(last_price=100.0, open_price=0.0) == 0.0


def test_today_return_pct_negative() -> None:
    result = _compute_today_return_pct(last_price=90.0, open_price=100.0)
    assert result == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Quote filtering / eligibility
# ---------------------------------------------------------------------------


def test_eligibility_usd_quote_online() -> None:
    assert _eligibility_reason(quote="USD", status="online", wsname="BTC/USD") == ""


def test_eligibility_usdt_quote() -> None:
    assert _eligibility_reason(quote="USDT", status="online", wsname="SOL/USDT") == ""


def test_eligibility_usdc_quote() -> None:
    assert _eligibility_reason(quote="USDC", status="", wsname="ETH/USDC") == ""


def test_ineligible_non_usd_like_quote() -> None:
    reason = _eligibility_reason(quote="EUR", status="online", wsname="BTC/EUR")
    assert "quote_not_usd_like" in reason


def test_ineligible_cancel_only_status() -> None:
    reason = _eligibility_reason(quote="USD", status="cancel_only", wsname="XYZ/USD")
    assert "status_not_tradeable" in reason


def test_ineligible_no_wsname() -> None:
    reason = _eligibility_reason(quote="USD", status="online", wsname="")
    assert reason == "no_wsname"


# ---------------------------------------------------------------------------
# normalize_pairs
# ---------------------------------------------------------------------------

_RAW_PAIRS_FIXTURE = {
    # Kraken uses Z-prefixed fiat (ZUSD, ZEUR) and X-prefixed crypto (XXBT, XETH)
    "XXBTZUSD": {
        "altname": "XBTUSD",
        "wsname": "XBT/USD",
        "base": "XXBT",
        "quote": "ZUSD",   # real Kraken code — module normalises to USD
        "status": "online",
    },
    "XETHZUSD": {
        "altname": "ETHUSD",
        "wsname": "ETH/USD",
        "base": "XETH",
        "quote": "ZUSD",   # real Kraken code — module normalises to USD
        "status": "online",
    },
    "XETHZEUR": {
        "altname": "ETHEUR",
        "wsname": "ETH/EUR",
        "base": "XETH",
        "quote": "ZEUR",   # real Kraken code — module normalises to EUR
        "status": "online",
    },
    "XXBTUSDT": {
        "altname": "XBTUSDT",
        "wsname": "XBT/USDT",
        "base": "XXBT",
        "quote": "USDT",   # no Z-prefix (stablecoin)
        "status": "online",
    },
    "CANCELONLY": {
        "altname": "CANCELONLY",
        "wsname": "XYZ/USD",
        "base": "XYZ",
        "quote": "ZUSD",   # USD-like quote but non-tradeable status
        "status": "cancel_only",
    },
    "NODOTINFO": "not_a_dict_should_be_ignored",
}


def test_normalize_pairs_returns_list_of_dicts() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)


def test_normalize_pairs_skips_non_dict_values() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    pair_ids = {r["pair_id"] for r in rows}
    assert "NODOTINFO" not in pair_ids


def test_normalize_pairs_usd_quote_pair_is_eligible() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    eth_usd = next(r for r in rows if r["pair_id"] == "XETHZUSD")
    assert eth_usd["eligible"] is True
    assert eth_usd["ineligible_reason"] == ""


def test_normalize_pairs_eur_quote_pair_is_ineligible() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    eth_eur = next(r for r in rows if r["pair_id"] == "XETHZEUR")
    assert eth_eur["eligible"] is False
    assert eth_eur["ineligible_reason"] != ""


def test_normalize_pairs_cancel_only_is_ineligible() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    cancel = next(r for r in rows if r["pair_id"] == "CANCELONLY")
    assert cancel["eligible"] is False
    assert "status_not_tradeable" in cancel["ineligible_reason"]


def test_all_ineligible_rows_have_ineligible_reason() -> None:
    rows = normalize_pairs(_RAW_PAIRS_FIXTURE)
    for row in rows:
        if not row["eligible"]:
            assert row["ineligible_reason"] != "", f"Row {row['pair_id']} missing ineligible_reason"


# ---------------------------------------------------------------------------
# enrich_with_ticker
# ---------------------------------------------------------------------------

_TICKER_FIXTURE: dict[str, dict] = {
    "XXBTZUSD": {
        "c": ["98500.0", 1],
        "b": ["98490.0", 1],
        "a": ["98510.0", 1],
        "o": "95000.0",
        "v": ["1234.5", "1234.5"],
    },
    "XBTUSDT": {
        "c": ["98400.0", 1],
        "b": ["98390.0", 1],
        "a": ["98420.0", 1],
        "o": "94000.0",
        "v": ["500.0", "500.0"],
    },
}


def _make_rows() -> list[dict]:
    return normalize_pairs(_RAW_PAIRS_FIXTURE)


def test_enrich_fills_last_price() -> None:
    rows = _make_rows()
    rows = enrich_with_ticker(rows, _TICKER_FIXTURE)
    btc_row = next(r for r in rows if r["pair_id"] == "XXBTZUSD")
    assert btc_row["last_price"] == pytest.approx(98500.0)


def test_enrich_fills_spread_pct() -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    btc_row = next(r for r in rows if r["pair_id"] == "XXBTZUSD")
    expected = _compute_spread_pct(98490.0, 98510.0)
    assert btc_row["spread_pct"] == pytest.approx(expected)


def test_enrich_fills_today_return_pct() -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    btc_row = next(r for r in rows if r["pair_id"] == "XXBTZUSD")
    expected = _compute_today_return_pct(98500.0, 95000.0)
    assert btc_row["today_return_pct"] == pytest.approx(expected)


def test_enrich_fills_quote_volume_est() -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    btc_row = next(r for r in rows if r["pair_id"] == "XXBTZUSD")
    assert btc_row["quote_volume_est"] == pytest.approx(1234.5 * 98500.0)


def test_enrich_tolerates_missing_ticker_for_pair() -> None:
    rows = _make_rows()
    rows = enrich_with_ticker(rows, {})  # empty ticker
    for row in rows:
        assert row["last_price"] is None


def test_enrich_tolerates_none_values_in_ticker() -> None:
    bad_ticker = {
        "XXBTZUSD": {
            "c": None,
            "b": None,
            "a": None,
            "o": None,
            "v": None,
        }
    }
    rows = enrich_with_ticker(_make_rows(), bad_ticker)
    btc_row = next(r for r in rows if r["pair_id"] == "XXBTZUSD")
    assert btc_row["last_price"] == pytest.approx(0.0)
    assert btc_row["spread_pct"] == pytest.approx(0.0)
    assert btc_row["today_return_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# write_snapshot_csv
# ---------------------------------------------------------------------------


def test_write_snapshot_csv_creates_file(tmp_path: Path) -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    output = tmp_path / "test_snapshot.csv"
    write_snapshot_csv(rows, output)
    assert output.exists()


def test_write_snapshot_csv_has_correct_columns(tmp_path: Path) -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    output = tmp_path / "test_snapshot.csv"
    write_snapshot_csv(rows, output)
    header = output.read_text().splitlines()[0]
    for col in OUTPUT_COLUMNS:
        assert col in header


def test_write_snapshot_csv_creates_parent_dirs(tmp_path: Path) -> None:
    rows = enrich_with_ticker(_make_rows(), _TICKER_FIXTURE)
    output = tmp_path / "nested" / "dir" / "snapshot.csv"
    write_snapshot_csv(rows, output)
    assert output.exists()


# ---------------------------------------------------------------------------
# Module safety check
# ---------------------------------------------------------------------------


def test_module_does_not_import_live_or_broker_modules() -> None:
    module_path = Path("research/memecoin_catcher/fetch_kraken_universe.py")
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
