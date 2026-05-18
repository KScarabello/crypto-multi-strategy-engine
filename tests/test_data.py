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
from data.update_ohlcv import (
    fetch_ohlcv_paginated,
    merge_ohlcv_frames,
    symbol_to_local_filename,
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


def test_symbol_to_local_filename() -> None:
    """Test symbol to local CSV filename mapping."""
    path = symbol_to_local_filename("BTC/USD", timeframe="4h")
    assert str(path).endswith("data/local/btc-usd_4h.csv")


def test_fetch_ohlcv_paginated_combines_batches() -> None:
    """Test paginated downloader combines sequential non-empty batches."""

    class DummyExchange:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                return [
                    [1000, 1.0, 1.1, 0.9, 1.0, 10.0],
                    [2000, 1.0, 1.1, 0.9, 1.0, 10.0],
                ]
            if self.calls == 2:
                return [[3000, 1.0, 1.1, 0.9, 1.0, 10.0]]
            return []

    exchange = DummyExchange()
    df = fetch_ohlcv_paginated(
        exchange=exchange,
        symbol="BTC/USD",
        timeframe="4h",
        since_ms=1000,
        limit=720,
        max_batches=10,
        sleep_seconds=0.0,
    )

    assert len(df) == 3
    assert df["timestamp"].tolist() == [1000, 2000, 3000]
    assert int(df.attrs["batches_completed"]) == 3


def test_fetch_ohlcv_paginated_stops_on_empty_batch() -> None:
    """Test paginated downloader stops immediately on empty batch."""

    class DummyExchange:
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
            return []

    df = fetch_ohlcv_paginated(
        exchange=DummyExchange(),
        symbol="BTC/USD",
        timeframe="4h",
        since_ms=1000,
        limit=720,
        max_batches=10,
        sleep_seconds=0.0,
    )

    assert df.empty
    assert int(df.attrs["batches_completed"]) == 1


def test_fetch_ohlcv_paginated_stops_when_timestamp_not_advancing() -> None:
    """Test paginated downloader stops when the last timestamp does not advance."""

    class DummyExchange:
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
            return [[since, 1.0, 1.1, 0.9, 1.0, 10.0]]

    df = fetch_ohlcv_paginated(
        exchange=DummyExchange(),
        symbol="BTC/USD",
        timeframe="4h",
        since_ms=1000,
        limit=720,
        max_batches=10,
        sleep_seconds=0.0,
    )

    assert len(df) == 1
    assert df["timestamp"].iloc[0] == 1000
    assert int(df.attrs["batches_completed"]) == 1


def test_merge_ohlcv_frames_deduplicates_and_sorts() -> None:
    """Test merge keeps latest duplicate timestamp and sorts ascending."""
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
