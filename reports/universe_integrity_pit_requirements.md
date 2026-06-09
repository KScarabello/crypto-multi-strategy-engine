# Point-in-Time Crypto Universe Requirements

## Why current-listing is not a valid historical universe

All OHLCV files in `data/local/` were populated by fetching data for symbols
**currently listed on Kraken**.  This is not a valid proxy for the historical
Kraken universe because:

1. **Delisted coins are absent.**  Coins that were delisted, collapsed, or
   withdrawn from Kraken before today have no data in the local cache.  Any
   strategy that would have held them is never penalized.

2. **Survivorship bias.**  The set of coins we can observe today is exactly the
   set that *survived* to today.  Historical strategies appear to have avoided
   the coins that later failed — but only because those coins are invisible.

3. **Symbol renames / migrations.**  Example: MATIC was renamed to POL in 2024.
   The `pol-usd_4h.csv` file starts from the rebrand date (2024-09-13) and
   contains no MATIC history.  A historical strategy using MATIC/USD from 2021
   cannot be replicated.

4. **Look-ahead in universe selection.**  Deciding to include AVAX/USD because
   it is currently available uses information from the future relative to any
   backtest start date before AVAX launched.

5. **Using only coins currently listed on Kraken is not a valid historical
   Kraken universe.**  Kraken listed and delisted dozens of assets between 2020
   and today.  The current listing is a strict subset.

---

## Requirements for a defensible historical universe

### 1. Exchange-specific listing dates

For each symbol on each exchange:
- Date (and ideally time) the symbol first appeared in the order book.
- This is distinct from the date the underlying asset was created.
- Source: exchange announcement APIs, historical market data vendors
  (Kaiko, CoinGecko Pro, CoinAPI).

### 2. Delisting dates

- Date the symbol was removed from active trading.
- Whether delisting was gradual (withdrawal notice → trading halt) or abrupt.
- Some delistings leave gaps in OHLCV that must not be forward-filled.

### 3. Historical symbol mappings and migrations

- Rename events (e.g., MATIC → POL, LUNA → LUNC after collapse).
- Forks and airdrops that created new tickers from existing ones.
- Consolidated tickers (e.g., XBT and BTC as equivalent Bitcoin symbols on
  some exchanges).
- Chain migrations that changed the economic exposure despite keeping the ticker.

### 4. Historical OHLCV for delisted assets

- Requires a third-party data provider.  Kraken's own API will not return
  historical data for delisted symbols.
- Kaiko, CoinGecko Pro historical exports, CoinMetrics are known sources.
- Must be stored separately from `data/local/` and tagged with their source
  and as-of date.

### 5. Point-in-time liquidity or volume screening

- Any liquidity filter (e.g., minimum 30-day median volume) must use only
  data available at the rebalance date — not full-sample average.
- Rolling windows that extend beyond the symbol's listing date must be
  handled as partial windows or the symbol must be excluded.

### 6. Treatment of special asset types

| Type | Treatment |
|---|---|
| Stablecoins (USDT, USDC, DAI) | Exclude from momentum universe; include only as cash proxy |
| Wrapped assets (WBTC, WETH) | Treat as duplicate economic exposure; exclude or deduplicate |
| Leveraged tokens (BTC3L, ETH2X) | Exclude; path-dependent decay distorts momentum signals |
| Synthetic / derivative tokens | Exclude unless the underlying economic exposure is unique |
| Rebased tokens (AMPL, OHM) | Exclude; close prices do not reflect total return |

### 7. Treatment of exchange outages and missing bars

- Distinguish between: (a) exchange outage (data exists elsewhere),
  (b) thin market / no trades (zero volume, last price carried), and
  (c) data vendor gap (unknown cause).
- Option A: exclude the bar; the lookback window shrinks.
- Option B: carry last close with zero volume and flag it.
- Do not silently forward-fill prices for multi-day outages.

### 8. Evidence / source metadata for every membership interval

Every interval in the membership file must include:
- `source`: the data vendor or primary evidence (e.g., "kraken_api_2024-12-01")
- `source_as_of`: the date the information was retrieved
- `listing_reason`: why the symbol was included (e.g., "listed_on_kraken")
- `delisting_reason`: why it was removed (e.g., "delisted_by_exchange",
  "below_liquidity_threshold", "duplicate_economic_exposure")

---

## Proposed membership CSV schema

```
symbol,exchange,eligible_from,eligible_to,source,source_as_of,listing_reason,delisting_reason,symbol_predecessor,symbol_successor,notes
BTC/USD,kraken,2020-01-01,,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,,,,"Bitcoin; continuous listing"
MATIC/USD,kraken,2021-06-11,2024-09-13,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,renamed_to_POL,,POL/USD,"Polygon; renamed to POL"
POL/USD,kraken,2024-09-13,,kraken_api_2024-12-01,2024-12-01,symbol_migration,,,MATIC/USD,"Polygon rebrand from MATIC"
LUNA/USD,kraken,2021-07-26,2022-05-13,kraken_api_2024-12-01,2024-12-01,listed_on_kraken,project_collapsed,,,"Terra LUNA; collapsed May 2022"
```

### Field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | string | Yes | Exchange symbol (e.g., BTC/USD) |
| `exchange` | string | Yes | Exchange identifier (e.g., kraken) |
| `eligible_from` | ISO date | Yes | First date the symbol is included in the universe |
| `eligible_to` | ISO date or empty | Yes | Last date included (empty = still active) |
| `source` | string | No | Evidence source for this interval |
| `source_as_of` | ISO date | No | Date the source was retrieved |
| `listing_reason` | string | No | Why included (e.g., listed_on_kraken) |
| `delisting_reason` | string | No | Why removed (e.g., delisted_by_exchange) |
| `symbol_predecessor` | string | No | Previous symbol if renamed/migrated |
| `symbol_successor` | string | No | Next symbol if renamed/migrated |
| `notes` | string | No | Free text |

---

## Remaining blockers for a genuine POINT_IN_TIME_UNIVERSE backtest

1. **No delisted-coin OHLCV in local cache.**  `data/local/` contains only
   currently-listed coins.  Third-party historical data must be sourced and
   stored.

2. **No authoritative listing-date database.**  Kraken's API does not provide
   programmatic access to historical listing dates.  Manual reconstruction or
   a commercial data vendor is required.

3. **No symbol-migration mapping file.**  MATIC→POL and similar migrations have
   not been documented in machine-readable form.

4. **No liquidity screen history.**  Rolling median volume for inclusion
   criteria can only be applied historically once OHLCV for delisted coins is
   available.

5. **No external audit or cross-validation.**  Membership intervals should be
   validated against at least two independent sources.

---

*This document is research-only and describes future work requirements.*
*Generated by `research/universe_integrity_analysis.py`.*
