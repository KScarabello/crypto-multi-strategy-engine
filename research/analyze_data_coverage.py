"""Analyze local OHLCV coverage for the expanded crypto universe.

This script inspects local 4h CSV files under data/local/ and reports:
- row counts
- first/latest timestamps
- expected 4h bars
- missing bars
- duplicate timestamps
- rough coverage percentage

Run:
    .venv/bin/python -m research.analyze_data_coverage
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/local")
TIMEFRAME_HOURS = 4

EXPANDED_UNIVERSE_20 = [
    "AAVE/USD",
    "ADA/USD",
    "APT/USD",
    "ARB/USD",
    "ATOM/USD",
    "AVAX/USD",
    "BCH/USD",
    "BTC/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "INJ/USD",
    "LINK/USD",
    "LTC/USD",
    "NEAR/USD",
    "OP/USD",
    "POL/USD",
    "SOL/USD",
    "UNI/USD",
    "XRP/USD",
]


@dataclass(frozen=True)
class CoverageRow:
    symbol: str
    file: str
    rows: int
    first_timestamp: str
    latest_timestamp: str
    expected_bars: int
    missing_bars: int
    duplicate_timestamps: int
    coverage_pct: float
    status: str


def symbol_to_filename(symbol: str) -> str:
    """Convert BTC/USD to btc-usd_4h.csv."""
    return symbol.lower().replace("/", "-") + "_4h.csv"


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} is missing required column: timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def classify_status(rows: int, coverage_pct: float, missing_bars: int) -> str:
    if rows == 0:
        return "empty"

    if coverage_pct < 95:
        return "gappy"

    if missing_bars > 0:
        return "minor_gaps"

    return "ok"


def analyze_symbol(symbol: str) -> CoverageRow:
    filename = symbol_to_filename(symbol)
    path = DATA_DIR / filename

    if not path.exists():
        return CoverageRow(
            symbol=symbol,
            file=str(path),
            rows=0,
            first_timestamp="MISSING",
            latest_timestamp="MISSING",
            expected_bars=0,
            missing_bars=0,
            duplicate_timestamps=0,
            coverage_pct=0.0,
            status="missing_file",
        )

    df = load_ohlcv(path)

    if df.empty:
        return CoverageRow(
            symbol=symbol,
            file=str(path),
            rows=0,
            first_timestamp="EMPTY",
            latest_timestamp="EMPTY",
            expected_bars=0,
            missing_bars=0,
            duplicate_timestamps=0,
            coverage_pct=0.0,
            status="empty",
        )

    timestamps = df["timestamp"]
    first = timestamps.iloc[0]
    latest = timestamps.iloc[-1]

    duplicate_timestamps = int(timestamps.duplicated().sum())

    expected_index = pd.date_range(
        start=first,
        end=latest,
        freq=f"{TIMEFRAME_HOURS}h",
        tz="UTC",
    )

    actual_unique_timestamps = pd.DatetimeIndex(timestamps.drop_duplicates())
    missing_bars = int(len(expected_index.difference(actual_unique_timestamps)))
    expected_bars = int(len(expected_index))

    coverage_pct = (
        len(actual_unique_timestamps) / expected_bars * 100 if expected_bars else 0.0
    )

    status = classify_status(
        rows=len(df),
        coverage_pct=coverage_pct,
        missing_bars=missing_bars,
    )

    return CoverageRow(
        symbol=symbol,
        file=str(path),
        rows=int(len(df)),
        first_timestamp=str(first),
        latest_timestamp=str(latest),
        expected_bars=expected_bars,
        missing_bars=missing_bars,
        duplicate_timestamps=duplicate_timestamps,
        coverage_pct=round(coverage_pct, 2),
        status=status,
    )


def main() -> None:
    rows = [analyze_symbol(symbol) for symbol in EXPANDED_UNIVERSE_20]

    report = pd.DataFrame([row.__dict__ for row in rows])

    display_columns = [
        "symbol",
        "rows",
        "first_timestamp",
        "latest_timestamp",
        "expected_bars",
        "missing_bars",
        "duplicate_timestamps",
        "coverage_pct",
        "status",
    ]

    print("\nExpanded universe OHLCV coverage")
    print("=" * 120)
    print(report[display_columns].to_string(index=False))

    print("\nStatus counts")
    print("=" * 120)
    print(report["status"].value_counts().to_string())

    if (report["status"] != "ok").any():
        print("\nSymbols needing review")
        print("=" * 120)
        needs_review = report.loc[report["status"] != "ok", display_columns]
        print(needs_review.to_string(index=False))


if __name__ == "__main__":
    main()