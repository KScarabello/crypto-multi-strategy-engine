"""Broker-state abstraction: Account equity and position tracking.

This module provides a broker interface for retrieving account state
from mock presets (for testing) or real Kraken API (for production).

Transferred from crypto-momentum-strategy and adapted for multi-strategy use.

Features:
- Mock/dry-run mode (default): Load preset demo data for testing
- Real Kraken mode: Fetch actual account equity and holdings via Kraken API
- Symbol normalization: Converts Kraken holdings to strategy format (BASE/USD)
- Cash tracking: Separates crypto holdings from USD balance

This is for state retrieval only—no order placement or network activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import get_trading_symbols


MIN_POSITION_NOTIONAL_USD = 0.01


@dataclass(frozen=True)
class Position:
    """Represents one asset position at current market value."""

    symbol: str
    quantity: float
    value_usd: float  # dollar exposure (quantity * current_price)


@dataclass(frozen=True)
class AccountState:
    """Represents full account state at a point in time."""

    equity: float  # total account equity in USD
    positions: dict[str, float]  # symbol -> dollar exposure
    available_cash: float = 0.0  # broker-reported free USD cash


def normalize_symbol(symbol: str, target_format: str = "BASE/QUOTE") -> str:
    """Normalize symbol to strategy format (e.g., 'BTC/USD' from 'BTC' or 'BTCUSD').

    Args:
        symbol: Input symbol in any format.
        target_format: Expected format pattern (default: "BASE/QUOTE").

    Returns:
        Normalized symbol string.
    """
    s = symbol.strip().upper()
    # Handle common variations
    if "/" in s:
        return s  # Already in BASE/QUOTE format
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}/USD"
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT"
    # Default: assume it's a base symbol, append /USD
    return f"{s}/USD"


def load_account_state_mock(
    equity: float = 100_000.0,
    use_preset: str = "baseline",
) -> AccountState:
    """Load account state from a mock demo (no network calls).

    Args:
        equity: Simulated account equity in USD.
        use_preset: Which demo preset to load ("baseline", "minimal", "empty").

    Returns:
        AccountState with mock data.
    """
    if equity <= 0:
        raise ValueError("equity must be positive")

    presets = {
        "baseline": {
            "BTC/USD": 25_000.0,
            "ETH/USD": 25_000.0,
            "XRP/USD": 25_000.0,
            # Cash: 25_000.0 (implicit)
        },
        "minimal": {
            "BTC/USD": 30_000.0,
            "ETH/USD": 20_000.0,
            # XRP empty, Cash: 50_000.0 (implicit)
        },
        "empty": {
            # All cash
        },
    }

    if use_preset not in presets:
        raise ValueError(f"Unknown preset '{use_preset}'. Choose from: {list(presets.keys())}")

    positions = presets[use_preset]
    total_risky = sum(positions.values())
    if total_risky > equity:
        raise ValueError(f"Preset positions (${total_risky}) exceed equity (${equity})")

    available_cash = max(0.0, float(equity) - float(total_risky))
    return AccountState(equity=equity, positions=positions, available_cash=available_cash)


def load_account_state_kraken(
    api_key: str | None = None,
    api_secret: str | None = None,
    api_passphrase: str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> AccountState:
    """Load account state from Kraken Spot API.

    Fetches current account equity and holdings, normalizing Kraken symbols
    to strategy format (e.g., "BTC" → "BTC/USD").

    Args:
        api_key: Kraken API key.
        api_secret: Kraken API secret.
        api_passphrase: Kraken API passphrase (not typically needed for spot trading).
        symbols: Tuple of supported symbols (defaults to trading symbols).

    Returns:
        AccountState with account equity and normalized positions.

    Raises:
        ValueError: If credentials are missing or invalid.
        ImportError: If ccxt library is not installed.
        Exception: If Kraken API call fails.
    """
    if not api_key or not api_secret:
        raise ValueError("Kraken loader requires api_key and api_secret")

    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt library required for Kraken support. "
            "Install with: pip install ccxt"
        )

    # Initialize Kraken exchange
    try:
        kraken = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase or "",
            "enableRateLimit": True,
        })
    except Exception as e:
        raise ValueError(f"Failed to initialize Kraken client: {e}")

    # Fetch account balances
    try:
        balances = kraken.fetch_balance()
    except Exception as e:
        raise Exception(f"Failed to fetch Kraken account balance: {e}")

    # Extract total balances by asset from Kraken response.
    total = balances.get("total", {})
    if not total:
        raise ValueError("Could not determine account equity from Kraken balances")

    # USD cash balance is explicit cash, not total account equity.
    usd_cash = float(total.get("USD", 0.0) or 0.0)

    # Build normalized position mapping using the same valuation source.
    positions = _normalize_kraken_holdings(
        kraken=kraken,
        total_balances=total,
        supported_symbols=symbols,
    )
    available_cash = _extract_kraken_available_cash_usd(balances)

    # Total account equity = USD cash + market value of supported crypto holdings.
    equity_usd = usd_cash + float(sum(positions.values()))
    if equity_usd <= 0:
        raise ValueError(
            f"Invalid Kraken equity computed from cash+positions: cash={usd_cash}, positions={sum(positions.values())}"
        )

    return AccountState(
        equity=equity_usd,
        positions=positions,
        available_cash=available_cash,
    )


def _extract_kraken_available_cash_usd(balances: dict[str, object]) -> float:
    """Extract broker-reported free USD cash from Kraken balance payload."""
    free_balances = balances.get("free", {})
    if not isinstance(free_balances, dict):
        return 0.0

    # Kraken may expose USD as USD or ZUSD depending on endpoint/account mapping.
    for key in ("USD", "ZUSD"):
        value = free_balances.get(key)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue

    return 0.0


def _normalize_kraken_holdings(
    kraken,
    total_balances: dict[str, float],
    supported_symbols: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Convert Kraken holdings dict to normalized strategy positions.

        Kraken returns total balances as a mapping, e.g.:
            {"BTC": 1.5, "ETH": 0.2, "USD": 100.0, ...}

    This function:
    1. Extracts total balances per supported asset
    2. Maps assets to strategy format (e.g., "BTC" → "BTC/USD")
    3. Computes USD value at current Kraken prices
    4. Filters dust balances below a minimum USD notional threshold
    5. Returns symbol → USD exposure mapping

    Args:
        kraken: Initialized Kraken ccxt client.
        total_balances: Kraken total balances dict from fetch_balance()["total"].
        supported_symbols: Tuple of supported symbols (defaults to trading symbols).

    Returns:
        Dictionary mapping normalized symbols (e.g., "BTC/USD") to USD exposure.
    """
    # Strategy universe base assets (from symbols like BTC/USD).
    universe = supported_symbols or get_trading_symbols()
    strategy_symbols = {sym.split("/")[0].upper() for sym in universe}
    positions = {}

    price_symbols = []
    quantities: dict[str, float] = {}

    # Build symbol quantities only for supported strategy assets.
    for symbol, qty in total_balances.items():
        if symbol not in strategy_symbols:
            continue

        total_qty = float(qty or 0.0)
        if total_qty <= 1e-8:
            continue

        strategy_symbol = f"{symbol}/USD"
        quantities[strategy_symbol] = total_qty
        price_symbols.append(strategy_symbol)

    if not price_symbols:
        return positions

    # One consistent ticker snapshot for all supported holdings.
    try:
        tickers = kraken.fetch_tickers(price_symbols)
    except Exception as e:
        raise ValueError(f"Failed to fetch Kraken tickers for holdings valuation: {e}")

    for strategy_symbol, qty in quantities.items():
        ticker = tickers.get(strategy_symbol)
        if not ticker:
            raise ValueError(f"Missing ticker for supported asset mapping: {strategy_symbol}")

        price_usd = ticker.get("last")
        if price_usd is None:
            price_usd = ticker.get("close")
        if price_usd is None:
            raise ValueError(f"No last/close price available for {strategy_symbol}")

        usd_exposure = float(qty) * float(price_usd)
        if usd_exposure < MIN_POSITION_NOTIONAL_USD:
            continue

        positions[strategy_symbol] = usd_exposure

    return positions


def load_account_state(
    source: str = "mock",
    api_key: str | None = None,
    api_secret: str | None = None,
    api_passphrase: str | None = None,
    symbols: tuple[str, ...] | None = None,
    loader: Callable[[], AccountState] | None = None,
) -> AccountState:
    """Load account state from mock or Kraken.

    This is the main entry point for retrieving broker state.
    Supports two modes:
    1. Mock/dry-run (default): Load preset demo data
    2. Real Kraken: Load actual account state via Kraken API

    Args:
        source: Source type ("mock" or "real"). Default: "mock".
        api_key: Kraken API key (required if source="real").
        api_secret: Kraken API secret (required if source="real").
        api_passphrase: Kraken API passphrase (optional).
        symbols: Tuple of supported symbols (defaults to trading symbols).
        loader: Custom callable that returns AccountState (overrides all above).

    Returns:
        AccountState from the selected source.

    Raises:
        ValueError: If parameters are invalid.
        ImportError: If ccxt is not installed (for real mode).
        Exception: If API call fails.
    """
    # Custom loader takes priority
    if loader is not None:
        return loader()

    # Route by source
    source_lower = source.lower().strip()
    if source_lower == "mock":
        return load_account_state_mock(use_preset="baseline")
    elif source_lower == "real":
        return load_account_state_kraken(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            symbols=symbols,
        )
    else:
        raise ValueError(f"Unknown source '{source}'. Choose from: mock, real")
