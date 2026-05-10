"""Tests for data loading and fetching utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from data.fetch_ohlc import (
    _normalize_columns,
    _sanitize_symbol,
    _to_ccxt_symbol,
    _to_ccxt_timeframe,
    _to_quote_symbol,
    pivot_close,
)


def test_sanitize_symbol() -> None:
    """Test symbol sanitization for filenames."""
    assert _sanitize_symbol("BTC/USD") == "btc-usd"
    assert _sanitize_symbol("ETH/USDT") == "eth-usdt"
    assert _sanitize_symbol("SOL / USD") == "sol---usd"


def test_to_ccxt_symbol() -> None:
    """Test symbol conversion to CCXT format."""
    assert _to_ccxt_symbol("BTC/USD") == "BTC/USD"
    assert _to_ccxt_symbol("btc/usd") == "BTC/USD"
    assert _to_ccxt_symbol("BTC-USD") == "BTC/USD"


def test_to_ccxt_symbol_invalid() -> None:
    """Test invalid symbol format."""
    with pytest.raises(ValueError, match="Unsupported symbol format"):
        _to_ccxt_symbol("INVALID")


def test_to_ccxt_timeframe() -> None:
    """Test timeframe conversion."""
    assert _to_ccxt_timeframe("4h") == "4h"
    assert _to_ccxt_timeframe("1d") == "1d"
    assert _to_ccxt_timeframe("daily") == "1d"
    assert _to_ccxt_timeframe("4H") == "4h"


def test_to_ccxt_timeframe_invalid() -> None:
    """Test invalid timeframe."""
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        _to_ccxt_timeframe("5m")


def test_to_quote_symbol() -> None:
    """Test quote symbol extraction."""
    base, quote = _to_quote_symbol("BTC/USD")
    assert base == "BTC"
    assert quote == "USD"


def test_normalize_columns() -> None:
    """Test column normalization."""
    df = pd.DataFrame({
        "Date": [1, 2, 3],
        "Pair": ["BTC/USD", "BTC/USD", "BTC/USD"],
        "Open": [100.0, 101.0, 102.0],
        "Close": [101.0, 102.0, 103.0],
    })
    
    normalized = _normalize_columns(df)
    assert "timestamp" in normalized.columns
    assert "symbol" in normalized.columns
    assert "open" in normalized.columns
    assert "close" in normalized.columns
    assert "Date" not in normalized.columns
    assert "Pair" not in normalized.columns


def test_pivot_close() -> None:
    """Test close price matrix pivoting."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC").tolist() * 2,
        "symbol": ["BTC/USD"] * 4 + ["ETH/USD"] * 4,
        "close": [100.0, 101.0, 102.0, 103.0, 50.0, 51.0, 52.0, 53.0],
    })
    
    close_matrix = pivot_close(df)
    assert close_matrix.shape == (4, 2)
    assert list(close_matrix.columns) == ["BTC/USD", "ETH/USD"]
    assert close_matrix.loc[df["timestamp"].iloc[0], "BTC/USD"] == 100.0
    assert close_matrix.loc[df["timestamp"].iloc[4], "ETH/USD"] == 50.0
