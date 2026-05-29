# Crypto Multi-Strategy Framework

A modular, production-ready framework for cryptocurrency research, backtesting, and live trading across multiple strategies.

**Status**: Early release. Reusable infrastructure transferred from `crypto-momentum-strategy` and adapted for multi-strategy use.

---

## Purpose

This repository provides a clean foundation for:

- **Strategy Research**: Develop and test multiple trading strategies with consistent interfaces
- **Backtesting**: Execution-aware simulation with realistic cost modeling and position constraints
- **Live Trading**: Safe, modular execution with dry-run defaults and scheduler integration
- **Data Management**: Incremental OHLCV updates with Kraken integration

## Design Philosophy

- **Separation of Concerns**: Data, strategy, backtest, and execution are independently testable
- **Multi-Strategy Ready**: Common `Strategy` interface enables blending and comparison
- **Dry-Run First**: Mock broker mode and safe defaults; live trading requires explicit opt-in
- **Deterministic**: All simulations are reproducible and logged
- **Modular**: Import and reuse individual components without full framework

---

## Repository Structure

```
crypto-multi-strategy/
├── config/                    # Configuration and settings
│   ├── __init__.py           # Public config defaults + private override logic
│   └── private.py            # (git-ignored) Private overrides
│
├── data/                      # Data fetching, loading, and validation
│   ├── fetch_ohlc.py         # OHLCV loading from local/remote sources
│   ├── update_ohlcv.py       # Incremental CSV update utilities
│   └── local/                # Local OHLCV CSV cache (git-ignored)
│
├── brokers/                   # Broker abstraction and account state
│   └── kraken.py             # Kraken API wrapper + mock presets
│
├── execution/                 # Order planning and execution
│   └── orders.py             # Trade sizing, planning, validation
│
├── backtest/                  # Backtesting engine and metrics
│   ├── engine.py             # Generic backtest simulation runner
│   └── metrics.py            # CAGR, Sharpe, max drawdown, etc.
│
├── live/                      # Live trading orchestration
│   ├── notifications.py      # Email alert utilities
│   └── (placeholder for scheduler integration)
│
├── strategies/                # Strategy implementations
│   ├── base.py               # Abstract Strategy interface
│   └── (momentum.py, reversal.py, etc. - to be implemented)
│
├── tests/                     # Unit tests for all modules
│   ├── test_data.py
│   ├── test_execution.py
│   ├── test_metrics.py
│   └── test_strategies.py
│
├── logs/                      # Execution logs (git-ignored)
├── outputs/                   # Backtest results and analysis
├── .env.example               # Example environment variables
├── .gitignore                 # Excludes secrets, data, logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## What Was Transferred from `crypto-momentum-strategy`

### Data Module (`data/`)
- **fetch_ohlc.py**: OHLCV loading from local CSV, remote CCXT (Kraken), or CryptoCompare
  - Symbol normalization (BTC/USD format)
  - Incremental Kraken updates via watermarking
  - Data validation (non-positive prices, duplicates, NaNs)
  - Close price matrix pivoting

- **update_ohlcv.py**: Command-line entry point for refreshing data

### Backtest Module (`backtest/`)
- **metrics.py**: Reusable performance calculations
  - Total return, CAGR, Sharpe ratio, max drawdown
  - Annualized volatility
  - Turnover statistics

- **engine.py**: Generic execution-aware simulator
  - Flexible signal generation via callbacks
  - One-bar-delayed execution
  - Transaction costs and slippage
  - Position and turnover constraints
  - Detailed rebalance logging

### Brokers Module (`brokers/`)
- **kraken.py**: Broker abstraction with mock and real modes
  - Mock presets (baseline, minimal, empty) for testing
  - Real Kraken API integration via CCXT
  - Symbol normalization and position valuation
  - Available cash tracking

### Execution Module (`execution/`)
- **orders.py**: Trade planning and sizing
  - Order representation (symbol, side, delta)
  - Plan trades from current to target weights
  - Minimum notional filters
  - Formatted output

### Config Module (`config/`)
- **__init__.py**: Settings dataclass with public defaults
  - Optional private override pattern
  - Trading universe, timeframe, backtest parameters
  - Data paths and provider selection
  - Cost assumptions (slippage, transaction costs)

### Live Module (`live/`)
- **notifications.py**: Email alerts for trade activity
  - SMTP integration (Gmail, custom servers)
  - Simple trade notification template
  - Environment-based credential loading

### Tests
- Unit tests for data, metrics, execution, and strategies
- No integration/live tests to avoid external dependencies

---

## What Was NOT Transferred

### From `crypto-momentum-strategy`
- **Momentum-specific logic** (`strategy/momentum.py`): Only used as reference; not copied
- **Momentum backtest runner**: Replaced with generic engine accepting signal callbacks
- **Single-strategy research scripts**: Not copied to keep codebase clean
- **Live trading state files**: .signals/, .pending/ state tracking (can be adapted as needed)
- **Private configuration or API keys**: Only public examples provided

### Design Differences
- New repo is **not strategy-opinionated**: No embedded momentum scoring
- Common **Strategy interface**: All strategies inherit from same base and output normalized weights
- **Flexible signal generator**: Backtest accepts any callable that maps (close, timestamp) → weights
- Simplified **config management**: No momentum-specific parameters in defaults

---

## Getting Started

### Installation

```bash
# Clone the repo
git clone <repo_url>
cd crypto-multi-strategy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings (optional for mock mode)
# Leave API keys commented out for dry-run
```

### Quick Test

```bash
# Run tests to verify installation
pytest tests/

# Example data functions
python -c "from data.fetch_ohlc import pivot_close; help(pivot_close)"

# Example order planning
python -c "
from execution.orders import plan_trades
equity = 100_000
current = {'BTC/USD': 25_000}
target = {'BTC/USD': 0.3}
orders = plan_trades(equity, current, target)
print(f'{len(orders)} trade(s) planned')
"
```

---

## Usage Examples

## Research Safety Note

- `research/btc_long_short_momentum_research.py` and `research/run_btc_long_short_momentum_research.py` are exploratory research-only modules.
- They evaluate synthetic unlevered BTC short exposure for analysis and are not wired into live execution.
- No live shorting behavior should be inferred from these reports; liquidation/risk-of-ruin effects are not modeled in this first pass.
- `research/btc_crash_filter_research.py` and `research/run_btc_crash_filter_research.py` are also exploratory research-only modules.
- Crash-filter overlays are evaluated for defensive timing research and are not deployable live strategy code.
- `research/btc_crash_filter_reentry_research.py` and `research/run_btc_crash_filter_reentry_research.py` are exploratory re-entry studies on top of crash-filter logic.
- Re-entry results are research artifacts only and are not live-trading deployment recommendations.

### Loading Historical Data

```python
from data.fetch_ohlc import load_ohlcv_history
from pathlib import Path

symbols = ("BTC/USD", "ETH/USD", "SOL/USD")
ohlcv = load_ohlcv_history(
    symbols=symbols,
    timeframe="4h",
    data_dir=Path("data/local")
)
print(f"Loaded {len(ohlcv)} rows across {ohlcv['symbol'].nunique()} symbols")
```

### Running a Backtest

```python
import pandas as pd
from backtest.engine import run_backtest
from backtest.metrics import summary_metrics

# Define a simple signal generator
def my_signal(close: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, float]:
    """Equal-weight all symbols."""
    n = len(close.columns)
    return {symbol: 1.0 / n for symbol in close.columns}

# Run backtest
result = run_backtest(
    ohlcv=ohlcv,
    signal_generator=my_signal,
    initial_capital=10_000.0,
    transaction_cost_bps=10.0,
)

# Calculate metrics
metrics = summary_metrics(
    equity=result.portfolio["equity"],
    bars_per_year=6 * 24,  # 4h bars
    returns=result.portfolio["strategy_return"],
    turnover=result.turnover,
)
print(f"CAGR: {metrics['cagr']:.2%}")
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### Implementing a Strategy

```python
from strategies.base import Strategy
import pandas as pd

class MyMomentumStrategy(Strategy):
    def generate_target_weights(
        self, close_prices, timestamp, state=None, config=None
    ):
        config = config or {}
        lookback_bars = config.get("lookback_bars", 20)
        top_n = config.get("top_n", 3)
        
        # Compute returns
        returns = close_prices.pct_change(periods=lookback_bars).iloc[-1]
        
        # Select top performers
        selected = returns.nlargest(top_n).index.tolist()
        weight = 1.0 / len(selected) if selected else 0.0
        
        return {symbol: weight for symbol in selected}

# Use in backtest
strategy = MyMomentumStrategy()
result = run_backtest(
    ohlcv=ohlcv,
    signal_generator=lambda close, ts: strategy.generate_target_weights(close, ts),
)
```

### Getting Account State

```python
from brokers.kraken import load_account_state

# Mock mode (dry-run)
state = load_account_state(source="mock")
print(f"Account Equity: ${state.equity:,.2f}")
print(f"Positions: {state.positions}")

# Real Kraken mode (requires API keys in environment)
# state = load_account_state(
#     source="real",
#     api_key=os.environ["KRAKEN_API_KEY"],
#     api_secret=os.environ["KRAKEN_API_SECRET"]
# )
```

### Planning Trades

```python
from execution.orders import plan_trades, print_trade_plan

orders = plan_trades(
    equity=state.equity,
    current_positions=state.positions,
    target_weights={"BTC/USD": 0.3, "ETH/USD": 0.3, "SOL/USD": 0.2},
    min_trade_notional=10.0,
)
print_trade_plan(orders, equity=state.equity, title="REBALANCE PLAN")
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=.
```

---

## Configuration Management

### Public Config (checked in)

`config/__init__.py` contains safe defaults:
- Demo symbol universes
- Backtest parameters (slippage, costs)
- Data paths and timeframe defaults

### Private Config (git-ignored)

Create `config/private.py` for sensitive overrides:

```python
from pathlib import Path
from config import Settings

SETTINGS = Settings(
    trading_symbols=("BTC/USD", "ETH/USD"),
    initial_capital=50_000.0,
    # ... other overrides
)
```

### Environment Variables

Create `.env` for API keys and credentials (git-ignored):

```bash
KRAKEN_API_KEY=your_key_here
EMAIL_USERNAME=alerts@example.com
```

Load with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("KRAKEN_API_KEY")
```

---

## Development

### Adding a New Strategy

1. Create `strategies/my_strategy.py`
2. Inherit from `Strategy` base class
3. Implement `generate_target_weights()`
4. Add tests in `tests/test_my_strategy.py`

### Adding Data Sources

Extend `data/fetch_ohlc.py`:
- Implement new `downloader()` function returning normalized OHLCV
- Use existing `_validate_and_clean()` to standardize output
- Integrate via `load_ohlcv_history()` or `build_historical_downloader()`

### Extending Metrics

Add functions to `backtest/metrics.py`:
- Follow existing pattern (clean input, compute, return float/dict)
- Use `_clean_equity()` and `_clean_returns()` helpers
- Add tests in `tests/test_metrics.py`

---

## Assumptions & Constraints

### Data
- Close prices must be positive (zero/negative filtered as invalid)
- Timestamps must be timezone-aware UTC
- Duplicates are dropped (keeping last)
- Long-format OHLCV with columns: timestamp, open, high, low, close, volume, symbol

### Strategies
- Target weights must sum to ≤ 1.0 (remainder is cash)
- No short selling (weights ≥ 0)
- Per-symbol weights ≤ 1.0 (position weight cap)

### Backtesting
- One-bar-delayed execution (signal generated bar T, executed bar T+1)
- Uniform transaction costs across all trades
- Linear slippage (not order-book dependent)
- Gross return before costs is tracked separately

### Live Trading
- Mock/dry-run is the default safe mode
- Real trading requires explicit `--live` flag and env credentials
- Email alerts optional (skip if EMAIL_* vars not set)

---

## What to Do Next

### Short Term
- Implement concrete strategy (e.g., momentum, reversal)
- Add CLI for common tasks (backtest, live mode)
- Expand broker support (Binance, dYdX, etc.)
- Add reporting/visualization utilities

### Medium Term
- Add portfolio constraint optimizer
- Implement factor exposure tracking
- Build risk management module (VaR, CVaR)
- Add multi-strategy ensemble tools

### Long Term
- Production deployment infrastructure
- Database logging of live trades and signals
- Sentiment/alternative data integration
- ML-based strategy development tools

---

## Architecture Decisions

### Why Separate Broker/Execution from Strategy?
- Strategy is algorithm-agnostic
- Broker abstraction allows testing without API calls
- Execution can be mocked/dry-run independently

### Why Generic Backtest Engine?
- Avoids momentum-specific hard-coded logic
- Callable signal generator is simpler than strategy registry
- Easier to reason about what's being simulated

### Why Not Copy Momentum Directly?
- New repo should serve multiple strategies, not just momentum
- Clean slate prevents momentum-specific assumptions from creeping into common code
- Easier to add reversal, ensemble, ML strategies as independent modules

---

## References

### Original Momentum Strategy
The `crypto-momentum-strategy` repo remains read-only reference material:
- Check it for detailed momentum implementation patterns
- Reference its tests and research scripts
- Extract/adapt specific utilities as needed

### Data Sources
- **Kraken**: Via CCXT (default)
- **CryptoCompare**: Public API (no keys required)
- **Local CSV**: data/local/ directory

### External Libraries
- `pandas`: Data manipulation and time series
- `numpy`: Numerical computation
- `ccxt`: Exchange API wrapper
- `pytest`: Testing framework
- `python-dotenv`: Environment variable management

---

## License

[Specify your license here, e.g., MIT, Apache 2.0]

---

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new code
4. Ensure `pytest` passes
5. Submit a pull request

---

## Security Notes

⚠️ **Never commit:**
- API keys or credentials
- Private configuration files
- Real secrets (.env files)
- Local data with sensitive symbol lists

Use `.gitignore` and `.env.example` to maintain safety.

---

## Support

For questions or issues:
- Check existing GitHub issues
- Review test examples for usage patterns
- Consult crypto-momentum-strategy for momentum strategy details
