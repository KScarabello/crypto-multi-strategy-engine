"""Public-safe project configuration defaults for multi-strategy framework.

This file is intentionally safe for open-source publication and includes
demo placeholders only.

Private production or proprietary research parameters should live in
config/private.yaml (gitignored). If present, settings from that file will
override the public defaults below at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration values for demo/public usage."""

    # Trading universe used for signal generation and order execution
    trading_symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD")
    # Broader data-refresh universe (includes candidates not yet live-tradable)
    data_symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD")
    timeframe: str = "4h"

    # Backtesting parameters
    initial_capital: float = 10_000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    min_history_bars: int | None = 36
    min_eligible_assets: int = 1
    min_median_volume: float | None = None
    max_turnover_per_rebalance: float | None = None
    max_position_weight: float | None = None
    max_gross_exposure: float | None = None

    # Rebalancing schedule
    rebalance_every_bars: int = 1
    rebalance_hour_utc: int = 0

    # Data loading and downloading
    data_dir: Path = Path("data/local")
    use_downloader: bool = False
    historical_data_provider: str = "kraken"
    historical_fallback_provider: str | None = None
    historical_exchange_name: str | None = None
    historical_since: str | None = None
    historical_max_batches: int | None = 5
    historical_max_rows: int | None = None
    historical_limit_per_request: int = 720
    historical_request_pause_seconds: float = 0.1
    historical_overwrite: bool = False

    # Output
    output_dir: Path = Path("outputs")

    # Live trading
    broker_source: str = "mock"  # "mock" or "kraken"
    min_order_notional_usd: float = 10.0


SETTINGS = Settings()

# Optional private override: if config/private.yaml is present in environment,
# it can override settings. For now, we only support Python-based config_private.
try:
    from config.private import SETTINGS as PRIVATE_SETTINGS  # type: ignore[import-not-found]
except (ModuleNotFoundError, ImportError):
    PRIVATE_SETTINGS = None

if PRIVATE_SETTINGS is not None:
    SETTINGS = PRIVATE_SETTINGS


def _as_symbol_tuple(value: object) -> tuple[str, ...]:
    """Normalize configured symbol containers to tuple[str, ...]."""
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (str(value),)


def get_trading_symbols() -> tuple[str, ...]:
    """Return the configured trading universe for signals and execution."""
    symbols = _as_symbol_tuple(getattr(SETTINGS, "trading_symbols", None))
    if symbols:
        return symbols
    raise ValueError("No trading symbols configured")


def get_data_symbols() -> tuple[str, ...]:
    """Return the configured data refresh universe."""
    symbols = _as_symbol_tuple(getattr(SETTINGS, "data_symbols", None))
    if symbols:
        return symbols
    # Fall back to trading symbols
    return get_trading_symbols()
