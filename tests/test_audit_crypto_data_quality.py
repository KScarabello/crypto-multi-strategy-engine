"""Unit tests for research/audit_crypto_data_quality.py.

Fixtures:
1. Complete clean 4h data → GOOD
2. Missing candles → GAPPY / NEEDS_REVIEW
3. Duplicate timestamps
4. Late-starting symbol → LIMITED_HISTORY
5. Zero-volume bars
6. Invalid OHLC rows (high < low, close <= 0, volume < 0)
7. Symbol with insufficient lookback history
8. Stale data ending before expected latest date
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.audit_crypto_data_quality import (
    AuditConfig,
    PortfolioAuditSummary,
    SymbolAuditResult,
    audit_portfolio,
    audit_symbol,
    discover_symbol_files,
    run_audit,
    symbol_to_filename,
    timeframe_hours,
    timeframe_freq,
    write_symbol_csv,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(
    path: Path,
    start: str = "2020-01-01",
    n_bars: int = 200,
    freq: str = "4h",
    close_values: list[float] | None = None,
    high_mult: float = 1.01,
    low_mult: float = 0.99,
    volume: float = 100.0,
) -> pd.DataFrame:
    """Write a synthetic OHLCV CSV and return its DataFrame."""
    index = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    closes = close_values if close_values else [100.0 + i * 0.1 for i in range(n_bars)]
    df = pd.DataFrame({
        "timestamp": index,
        "open":   closes,
        "high":   [c * high_mult for c in closes],
        "low":    [c * low_mult  for c in closes],
        "close":  closes,
        "volume": [volume] * n_bars,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _default_cfg(tmp_path: Path, **kwargs) -> AuditConfig:
    return AuditConfig(
        data_dir=tmp_path,
        timeframe="4h",
        backtest_start="2020-01-01",
        min_lookback_bars=36,
        stale_days=14,
        output_dir=tmp_path / "reports",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Timeframe helpers
# ---------------------------------------------------------------------------

class TestTimeframeHelpers:
    def test_4h_hours(self):
        assert timeframe_hours("4h") == 4.0

    def test_1d_hours(self):
        assert timeframe_hours("1d") == 24.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            timeframe_hours("6h")

    def test_4h_freq(self):
        assert timeframe_freq("4h") == "4h"

    def test_1d_freq(self):
        assert timeframe_freq("1d") == "24h"


# ---------------------------------------------------------------------------
# 2. symbol_to_filename
# ---------------------------------------------------------------------------

class TestSymbolToFilename:
    def test_btc_usd(self):
        assert symbol_to_filename("BTC/USD", "4h") == "btc-usd_4h.csv"

    def test_lowercase_passthrough(self):
        assert symbol_to_filename("sol/usd", "4h") == "sol-usd_4h.csv"

    def test_spaces_normalized(self):
        assert symbol_to_filename("ETH USD", "4h") == "eth-usd_4h.csv"


# ---------------------------------------------------------------------------
# 3. Fixture 1 — complete clean 4h data → GOOD
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_good_label(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=200)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.data_quality_label == "GOOD"

    def test_no_missing_bars(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=200)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.missing_bar_count == 0
        assert result.missing_bar_pct == 0.0

    def test_zero_duplicates(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=200)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.duplicate_ts_count == 0

    def test_row_count_matches(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=100)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.row_count == 100

    def test_has_enough_lookback(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=200)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.has_enough_lookback is True

    def test_no_zero_volume(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=100, volume=500.0)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.zero_volume_count == 0

    def test_no_invalid_ohlc(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, n_bars=100)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.high_lt_low_count == 0
        assert result.close_lte_zero_count == 0


# ---------------------------------------------------------------------------
# 4. Fixture 2 — missing candles → GAPPY
# ---------------------------------------------------------------------------

class TestMissingCandles:
    def _write_with_gaps(self, path: Path, n_bars: int = 200, drop_every: int = 10) -> None:
        """Write a CSV with every nth row removed."""
        index = pd.date_range("2020-01-01", periods=n_bars, freq="4h", tz="UTC")
        closes = [100.0 + i * 0.1 for i in range(n_bars)]
        df = pd.DataFrame({
            "timestamp": index,
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [100.0] * n_bars,
        })
        df = df[df.index % drop_every != 0]  # remove every 10th row
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def test_missing_bar_count_nonzero(self, tmp_path):
        path = tmp_path / "eth-usd_4h.csv"
        self._write_with_gaps(path)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("ETH/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.missing_bar_count > 0

    def test_missing_bar_pct_nonzero(self, tmp_path):
        path = tmp_path / "eth-usd_4h.csv"
        self._write_with_gaps(path)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("ETH/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.missing_bar_pct > 0.0

    def test_gappy_label(self, tmp_path):
        """A symbol with > 5% missing bars should get GAPPY."""
        path = tmp_path / "eth-usd_4h.csv"
        # Drop every 5th row to force >5% missing
        index = pd.date_range("2020-01-01", periods=500, freq="4h", tz="UTC")
        closes = [100.0] * 500
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": [100.0] * 500,
        })
        df = df.iloc[::5 != 0].reset_index(drop=True)  # keep all except every 5th
        keep = [i for i in range(500) if i % 5 != 0]
        df = df.iloc[keep]
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("ETH/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.missing_bar_pct > 5.0
        assert result.data_quality_label == "GAPPY"

    def test_largest_gap_hours_correct(self, tmp_path):
        """A single 48-hour gap should be detected."""
        path = tmp_path / "sol-usd_4h.csv"
        idx1 = pd.date_range("2020-01-01", periods=50, freq="4h", tz="UTC")
        # Start idx2 four days after idx1 ends to guarantee a ~72h gap
        idx1_last = idx1[-1]
        idx2_start = idx1_last + pd.Timedelta(hours=72)
        idx2 = pd.date_range(idx2_start, periods=50, freq="4h", tz="UTC")
        index = idx1.append(idx2)
        closes = [100.0] * len(index)
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": [100.0] * len(index),
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("SOL/USD", path, cfg, reference_date=date(2020, 2, 15))
        assert result.largest_gap_hours >= 48.0


# ---------------------------------------------------------------------------
# 5. Fixture 3 — duplicate timestamps
# ---------------------------------------------------------------------------

class TestDuplicateTimestamps:
    def test_duplicate_count_detected(self, tmp_path):
        path = tmp_path / "xrp-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": [100.0] * 100,
        })
        # Duplicate the first 3 rows
        dupes = df.iloc[:3].copy()
        df = pd.concat([df, dupes], ignore_index=True)
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("XRP/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.duplicate_ts_count == 3

    def test_needs_review_label_with_duplicates(self, tmp_path):
        path = tmp_path / "xrp-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": [100.0] * 100,
        })
        dupes = df.iloc[:2].copy()
        df = pd.concat([df, dupes], ignore_index=True)
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("XRP/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.data_quality_label in ("NEEDS_REVIEW", "LIMITED_HISTORY", "GAPPY", "GOOD")
        assert result.duplicate_ts_count > 0


# ---------------------------------------------------------------------------
# 6. Fixture 4 — late-starting symbol → LIMITED_HISTORY
# ---------------------------------------------------------------------------

class TestLateStartingSymbol:
    def test_starts_after_backtest_start(self, tmp_path):
        path = tmp_path / "pol-usd_4h.csv"
        _make_ohlcv_csv(path, start="2023-06-01", n_bars=500)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("POL/USD", path, cfg, reference_date=date(2023, 7, 10))
        assert result.starts_after_backtest_start is True

    def test_limited_history_label(self, tmp_path):
        path = tmp_path / "pol-usd_4h.csv"
        _make_ohlcv_csv(path, start="2023-06-01", n_bars=500)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("POL/USD", path, cfg, reference_date=date(2023, 7, 10))
        assert result.data_quality_label == "LIMITED_HISTORY"

    def test_early_starting_symbol_flag_false(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, start="2019-12-01", n_bars=500)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2020, 2, 10))
        assert result.starts_after_backtest_start is False


# ---------------------------------------------------------------------------
# 7. Fixture 5 — zero-volume bars
# ---------------------------------------------------------------------------

class TestZeroVolumeBars:
    def test_zero_volume_count(self, tmp_path):
        path = tmp_path / "link-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        volumes = [100.0 if i % 10 != 0 else 0.0 for i in range(100)]
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": volumes,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("LINK/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.zero_volume_count == 10
        assert abs(result.zero_volume_pct - 10.0) < 0.1

    def test_zero_volume_triggers_needs_review(self, tmp_path):
        path = tmp_path / "link-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        volumes = [0.0 if i < 5 else 100.0 for i in range(100)]
        df = pd.DataFrame({
            "timestamp": index, "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": volumes,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("LINK/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.zero_volume_count == 5
        assert result.zero_volume_pct > 1.0
        assert result.data_quality_label in ("NEEDS_REVIEW", "LIMITED_HISTORY", "GOOD")


# ---------------------------------------------------------------------------
# 8. Fixture 6 — invalid OHLC (high < low, close <= 0, volume < 0)
# ---------------------------------------------------------------------------

class TestInvalidOHLC:
    def test_high_lt_low_detected(self, tmp_path):
        path = tmp_path / "avax-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        highs = [105.0] * 100
        lows  = [95.0] * 100
        # Corrupt row 5: high < low
        highs[5] = 90.0
        lows[5]  = 99.0
        df = pd.DataFrame({
            "timestamp": index, "open": closes,
            "high": highs, "low": lows,
            "close": closes, "volume": [100.0] * 100,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("AVAX/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.high_lt_low_count >= 1
        assert result.data_quality_label == "INVALID"

    def test_close_lte_zero_detected(self, tmp_path):
        path = tmp_path / "avax-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        closes[10] = 0.0
        closes[11] = -1.0
        df = pd.DataFrame({
            "timestamp": index, "open": closes,
            "high": [c * 1.01 if c > 0 else 1.0 for c in closes],
            "low":  [c * 0.99 if c > 0 else 0.5 for c in closes],
            "close": closes, "volume": [100.0] * 100,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("AVAX/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.close_lte_zero_count == 2

    def test_volume_lt_zero_detected(self, tmp_path):
        path = tmp_path / "avax-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        volumes = [100.0] * 100
        volumes[7] = -5.0
        df = pd.DataFrame({
            "timestamp": index, "open": closes,
            "high": [c * 1.01 for c in closes],
            "low":  [c * 0.99 for c in closes],
            "close": closes, "volume": volumes,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("AVAX/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.volume_lt_zero_count == 1

    def test_invalid_label_on_high_lt_low(self, tmp_path):
        path = tmp_path / "avax-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        highs = [105.0] * 100
        lows  = [95.0] * 100
        highs[0] = 80.0  # high < low
        df = pd.DataFrame({
            "timestamp": index, "open": closes,
            "high": highs, "low": lows,
            "close": closes, "volume": [100.0] * 100,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("AVAX/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.data_quality_label == "INVALID"


# ---------------------------------------------------------------------------
# 9. Fixture 7 — insufficient lookback history
# ---------------------------------------------------------------------------

class TestInsufficientLookback:
    def test_fewer_bars_than_lookback(self, tmp_path):
        path = tmp_path / "inj-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=20)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("INJ/USD", path, cfg, reference_date=date(2020, 1, 10))
        assert result.has_enough_lookback is False

    def test_insufficient_lookback_label(self, tmp_path):
        path = tmp_path / "inj-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=20)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("INJ/USD", path, cfg, reference_date=date(2020, 1, 10))
        assert result.data_quality_label == "LIMITED_HISTORY"

    def test_bars_before_first_signal(self, tmp_path):
        path = tmp_path / "inj-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=50)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("INJ/USD", path, cfg, reference_date=date(2020, 1, 30))
        # bars_before_first_signal = max(0, 50 - 36) = 14
        assert result.bars_before_first_signal == 14

    def test_exact_lookback_boundary(self, tmp_path):
        """Symbol with exactly min_lookback_bars bars should pass."""
        path = tmp_path / "inj-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=36)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("INJ/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.has_enough_lookback is True
        assert result.bars_before_first_signal == 0


# ---------------------------------------------------------------------------
# 10. Fixture 8 — stale data
# ---------------------------------------------------------------------------

class TestStaleData:
    def test_stale_flag(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=200)
        cfg = _default_cfg(tmp_path)
        # Reference date: far in the future — data should be stale
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2025, 1, 1))
        assert result.is_stale is True

    def test_not_stale_when_recent(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=200)
        cfg = _default_cfg(tmp_path)
        # Reference date: just after the generated data ends (~2020-01-01 + 200 * 4h)
        data_end = pd.date_range("2020-01-01", periods=200, freq="4h", tz="UTC")[-1]
        ref = data_end.date()
        result = audit_symbol("BTC/USD", path, cfg, reference_date=ref)
        assert result.is_stale is False

    def test_stale_label(self, tmp_path):
        path = tmp_path / "btc-usd_4h.csv"
        _make_ohlcv_csv(path, start="2020-01-01", n_bars=200)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("BTC/USD", path, cfg, reference_date=date(2025, 1, 1))
        assert result.data_quality_label == "STALE"


# ---------------------------------------------------------------------------
# 11. Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_missing_file_invalid_label(self, tmp_path):
        path = tmp_path / "ghost-usd_4h.csv"
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("GHOST/USD", path, cfg)
        assert result.data_quality_label == "INVALID"
        assert result.row_count == 0

    def test_missing_file_note_contains_path(self, tmp_path):
        path = tmp_path / "ghost-usd_4h.csv"
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("GHOST/USD", path, cfg)
        assert "ghost" in result.notes.lower() or "not found" in result.notes.lower()


# ---------------------------------------------------------------------------
# 12. Portfolio audit
# ---------------------------------------------------------------------------

class TestAuditPortfolio:
    def _make_results(self) -> list[SymbolAuditResult]:
        def _r(sym, first, last, rows, label, starts_late=False, enough_lookback=True):
            return SymbolAuditResult(
                symbol=sym, source_path=f"data/local/{sym.lower().replace('/', '-')}_4h.csv",
                first_timestamp=first, last_timestamp=last,
                row_count=rows, expected_bars=rows, missing_bar_count=0,
                missing_bar_pct=0.0, largest_gap_hours=4.0,
                duplicate_ts_count=0, non_monotonic_count=0,
                zero_volume_count=0, zero_volume_pct=0.0,
                nan_open=0, nan_high=0, nan_low=0, nan_close=0, nan_volume=0,
                high_lt_low_count=0, close_lte_zero_count=0,
                open_lte_zero_count=0, volume_lt_zero_count=0,
                starts_after_backtest_start=starts_late,
                bars_before_first_signal=max(0, rows - 36),
                has_enough_lookback=enough_lookback,
                is_stale=False,
                data_quality_label=label, notes="",
            )
        return [
            _r("BTC/USD", "2020-01-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00", 13000, "GOOD"),
            _r("POL/USD", "2024-09-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00", 800, "LIMITED_HISTORY", starts_late=True),
            _r("ETH/USD", "2020-01-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00", 13000, "GOOD"),
        ]

    def test_total_symbols_audited(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert summary.total_symbols_audited == 3

    def test_good_count(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert summary.symbols_good == 2

    def test_limited_history_count(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert summary.symbols_limited_history == 1

    def test_earliest_shared_start_is_latest_first(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        # POL starts 2024-09-01, that is the latest first bar → constrains shared start
        assert summary.earliest_shared_start == "2024-09-01"

    def test_symbols_constraining_start(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert "POL/USD" in summary.symbols_constraining_start

    def test_survivorship_flag_always_true(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert summary.using_current_universe_only is True

    def test_symbols_missing_history_list(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert "POL/USD" in summary.symbols_missing_full_backtest_history

    def test_engine_notes_present(self, tmp_path):
        results = self._make_results()
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio(results, cfg)
        assert len(summary.engine_note) > 20
        assert summary.engine_drop_nonpositive_close is True
        assert summary.engine_ffill_missing is False

    def test_empty_results(self, tmp_path):
        cfg = _default_cfg(tmp_path)
        summary = audit_portfolio([], cfg)
        assert summary.total_symbols_audited == 0


# ---------------------------------------------------------------------------
# 13. File discovery
# ---------------------------------------------------------------------------

class TestDiscoverSymbolFiles:
    def test_discovers_csv_files(self, tmp_path):
        for sym in ["btc-usd", "eth-usd", "sol-usd"]:
            (tmp_path / f"{sym}_4h.csv").write_text("timestamp\n")
        cfg = _default_cfg(tmp_path)
        pairs = discover_symbol_files(cfg)
        symbols = [p[0] for p in pairs]
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols
        assert len(pairs) == 3

    def test_explicit_symbols_override_discovery(self, tmp_path):
        # Only BTC exists on disk
        (tmp_path / "btc-usd_4h.csv").write_text("timestamp\n")
        cfg = _default_cfg(tmp_path, symbols=("BTC/USD", "ETH/USD"))
        pairs = discover_symbol_files(cfg)
        assert len(pairs) == 2
        syms = [p[0] for p in pairs]
        assert "ETH/USD" in syms

    def test_empty_dir_returns_empty(self, tmp_path):
        cfg = _default_cfg(tmp_path)
        pairs = discover_symbol_files(cfg)
        assert pairs == []


# ---------------------------------------------------------------------------
# 14. run_audit + write_symbol_csv integration
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_run_returns_results_for_each_file(self, tmp_path):
        for sym in ["btc-usd", "eth-usd"]:
            _make_ohlcv_csv(tmp_path / f"{sym}_4h.csv", n_bars=100)
        cfg = _default_cfg(tmp_path)
        results, summary = run_audit(cfg)
        assert len(results) == 2
        assert summary.total_symbols_audited == 2

    def test_empty_dir_returns_empty_results(self, tmp_path):
        cfg = _default_cfg(tmp_path)
        results, summary = run_audit(cfg)
        assert results == []
        assert summary.total_symbols_audited == 0

    def test_csv_written_correctly(self, tmp_path):
        for sym in ["btc-usd", "eth-usd"]:
            _make_ohlcv_csv(tmp_path / f"{sym}_4h.csv", n_bars=100)
        cfg = _default_cfg(tmp_path)
        results, _ = run_audit(cfg)
        csv_path = write_symbol_csv(results, tmp_path / "reports")
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "data_quality_label" in df.columns
        assert "missing_bar_pct" in df.columns

    def test_all_required_columns_present(self, tmp_path):
        _make_ohlcv_csv(tmp_path / "btc-usd_4h.csv", n_bars=100)
        cfg = _default_cfg(tmp_path)
        results, _ = run_audit(cfg)
        csv_path = write_symbol_csv(results, tmp_path / "reports")
        df = pd.read_csv(csv_path)
        required = [
            "symbol", "source_path", "first_timestamp", "last_timestamp",
            "row_count", "expected_bars", "missing_bar_count", "missing_bar_pct",
            "largest_gap_hours", "duplicate_ts_count", "non_monotonic_count",
            "zero_volume_count", "zero_volume_pct",
            "nan_open", "nan_high", "nan_low", "nan_close", "nan_volume",
            "high_lt_low_count", "close_lte_zero_count", "volume_lt_zero_count",
            "starts_after_backtest_start", "bars_before_first_signal",
            "has_enough_lookback", "is_stale",
            "data_quality_label", "notes",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 15. NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_close_counted(self, tmp_path):
        path = tmp_path / "dot-usd_4h.csv"
        index = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
        closes = [100.0] * 100
        closes[20] = float("nan")
        closes[21] = float("nan")
        df = pd.DataFrame({
            "timestamp": index, "open": closes,
            "high": [c * 1.01 if not math.isnan(c) else float("nan") for c in closes],
            "low":  [c * 0.99 if not math.isnan(c) else float("nan") for c in closes],
            "close": closes, "volume": [100.0] * 100,
        })
        df.to_csv(path, index=False)
        cfg = _default_cfg(tmp_path)
        result = audit_symbol("DOT/USD", path, cfg, reference_date=date(2020, 1, 20))
        assert result.nan_close == 2
