# Research Data Refresh Audit

## Command Run
- ./.venv/bin/python -c "from pathlib import Path; from data.update_ohlcv import download_all_symbols, EXPANDED_UNIVERSE_20; download_all_symbols(symbols=EXPANDED_UNIVERSE_20, timeframe='4h', data_dir=Path('data'))"

## Results
- Latest timestamp before: 2026-06-10 00:00:00+00:00
- Latest timestamp after: 2026-06-22 00:00:00+00:00
- Symbols refreshed: 20
- Refreshed symbols: AAVE/USD, ADA/USD, APT/USD, ARB/USD, ATOM/USD, AVAX/USD, BCH/USD, BTC/USD, DOGE/USD, DOT/USD, ETH/USD, INJ/USD, LINK/USD, LTC/USD, NEAR/USD, OP/USD, POL/USD, SOL/USD, UNI/USD, XRP/USD
- Canonical base data fresh enough for 4h comparator: True
- Symbols still stale: none
- Errors: none observed from command execution
