"""Focused tests for Binance public historical kline provider."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from data.providers.binance_public import (
    BINANCE_KLINE_COLUMNS,
    build_monthly_kline_url,
    iterate_months,
    merge_ohlcv_frames,
    parse_binance_timestamp,
    read_binance_kline_zip,
    symbol_to_binance_pair,
)


def test_symbol_to_binance_pair() -> None:
    """Normalized USD symbols should map to USDT spot pairs."""
    assert symbol_to_binance_pair("BTC/USD") == "BTCUSDT"
    assert symbol_to_binance_pair("ETH/USD") == "ETHUSDT"
    assert symbol_to_binance_pair("SOL/USD") == "SOLUSDT"


def test_build_monthly_kline_url() -> None:
    """Monthly URL format should match data.binance.vision spot path."""
    url = build_monthly_kline_url("BTCUSDT", "4h", "2020-01")
    assert url == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/4h/BTCUSDT-4h-2020-01.zip"
    )


def test_parse_binance_timestamp_milliseconds() -> None:
    """Millisecond timestamps should parse as UTC correctly."""
    series = pd.Series([1577836800000])
    parsed = parse_binance_timestamp(series)
    assert parsed.iloc[0] == pd.Timestamp("2020-01-01T00:00:00Z")


def test_parse_binance_timestamp_microseconds() -> None:
    """Very large timestamps should parse as microseconds defensively."""
    series = pd.Series([1577836800000000])
    parsed = parse_binance_timestamp(series)
    assert parsed.iloc[0] == pd.Timestamp("2020-01-01T00:00:00Z")


def test_read_binance_kline_zip_normalizes_columns(tmp_path: Path) -> None:
    """ZIP reader should normalize schema, deduplicate by timestamp, and sort."""
    zip_path = tmp_path / "BTCUSDT-4h-2020-01.zip"
    csv_name = "BTCUSDT-4h-2020-01.csv"

    row_old = [1577836800000, 100, 110, 90, 105, 1000, 0, 0, 0, 0, 0, 0]
    row_new = [1577836800000, 101, 111, 91, 106, 1001, 0, 0, 0, 0, 0, 0]
    row_next = [1577851200000, 106, 116, 96, 112, 1200, 0, 0, 0, 0, 0, 0]

    csv_text = "\n".join(
        ",".join(str(v) for v in row)
        for row in [row_next, row_old, row_new]
    )

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_text)

    df = read_binance_kline_zip(zip_path)

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing

    first = df.iloc[0]
    assert first["timestamp"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert first["close"] == pytest.approx(106.0)


def test_merge_ohlcv_frames_deduplicates_and_sorts() -> None:
    """Merge should keep latest duplicate timestamp and return sorted output."""
    existing = pd.DataFrame(
        {
            "timestamp": [2000, 1000],
            "open": [2.0, 1.0],
            "high": [2.2, 1.2],
            "low": [1.8, 0.8],
            "close": [2.0, 1.0],
            "volume": [20.0, 10.0],
        }
    )
    downloaded = pd.DataFrame(
        {
            "timestamp": [1000, 3000],
            "open": [1.5, 3.0],
            "high": [1.6, 3.2],
            "low": [1.4, 2.8],
            "close": [1.5, 3.0],
            "volume": [15.0, 30.0],
        }
    )

    merged = merge_ohlcv_frames(existing=existing, downloaded=downloaded)

    assert len(merged) == 3
    assert merged["timestamp"].is_monotonic_increasing
    row_1000 = merged.loc[merged["timestamp"] == pd.to_datetime(1000, unit="ms", utc=True)].iloc[0]
    assert row_1000["close"] == pytest.approx(1.5)


def test_iterate_months_inclusive() -> None:
    """Month iterator should include both start and end months."""
    months = list(iterate_months("2020-01", "2020-03"))
    assert months == ["2020-01", "2020-02", "2020-03"]


def test_binance_columns_constant_order() -> None:
    """Kline column constant should define the full Binance CSV schema."""
    assert BINANCE_KLINE_COLUMNS == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
