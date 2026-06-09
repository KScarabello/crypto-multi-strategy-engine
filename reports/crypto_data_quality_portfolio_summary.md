# Crypto Backtest Data Quality & Survivorship-Bias Audit

**Timeframe:** 4h  
**Configured backtest start:** 2020-01-01  
**Strategy min lookback bars:** 36  
**Data directory:** `data/local`  
**Symbols audited:** 20  

---

## 1. Universe Quality Summary

| Label | Count |
|---|---|
| GOOD | 0 |
| LIMITED_HISTORY | 0 |
| GAPPY | 0 |
| STALE | 20 |
| NEEDS_REVIEW | 0 |
| INVALID | 0 |

---

## 2. Earliest Date Where All Symbols Have Usable Data

**2024-09-13**

Symbols that constrain the joint start date: POL/USD

---

## 3. Symbols with Incomplete Backtest History

> These symbols have a first bar that is *after* the configured backtest start.
> Any analysis comparing them to BTC buy-and-hold from that date is unfair.

- AAVE/USD (first bar: 2020-10-15)
- APT/USD (first bar: 2022-10-19)
- ARB/USD (first bar: 2023-03-23)
- AVAX/USD (first bar: 2020-09-22)
- DOT/USD (first bar: 2020-08-18)
- INJ/USD (first bar: 2020-10-21)
- NEAR/USD (first bar: 2020-10-14)
- OP/USD (first bar: 2022-06-01)
- POL/USD (first bar: 2024-09-13)
- SOL/USD (first bar: 2020-08-11)
- UNI/USD (first bar: 2020-09-17)

---

## 4. Survivorship Bias Risk

> This backtest uses only currently-available symbols fetched from the live exchange API. Coins that were delisted, rebranded, or went to zero before today are NOT represented in the data. Historical performance is therefore upward-biased: we are implicitly selecting only the survivors. Strategies that performed well on this universe may have partly succeeded because they avoided the delisted/collapsed coins that existed at the time.

### What this means for the backtest

- The local OHLCV files are populated by fetching **currently-listed** Kraken symbols.
- Coins that delisted, collapsed, or were replaced after listing are absent.
- Example: POL/USD (formerly MATIC) starts only in 2024-09 — its pre-rebrand
  history under MATIC/USD is not included.
- If the strategy historically avoided delisted coins (e.g. via momentum score
  declining before delisting), the measured edge may partly reflect this.
- **To mitigate**: maintain a point-in-time universe snapshot, or explicitly
  add delisted coins with their available history.

---

## 5. Symbols with Insufficient Lookback History

> The strategy requires **36 bars** of history before it can
> generate a valid signal. Symbols below this threshold are silently excluded
> from the portfolio at those early timestamps.

*None — all symbols satisfy the minimum lookback.*

---

## 6. Symbols with Large Gaps or Stale Data

### Large gaps (>5% missing bars or gap >72h)

*No symbols with large gaps detected.*

### Stale or missing data

- AAVE/USD (last bar: 2026-05-17)
- ADA/USD (last bar: 2026-05-17)
- APT/USD (last bar: 2026-05-17)
- ARB/USD (last bar: 2026-05-17)
- ATOM/USD (last bar: 2026-05-17)
- AVAX/USD (last bar: 2026-05-17)
- BCH/USD (last bar: 2026-05-17)
- BTC/USD (last bar: 2026-05-18)
- DOGE/USD (last bar: 2026-05-17)
- DOT/USD (last bar: 2026-05-17)
- ETH/USD (last bar: 2026-05-18)
- INJ/USD (last bar: 2026-05-17)
- LINK/USD (last bar: 2026-05-17)
- LTC/USD (last bar: 2026-05-17)
- NEAR/USD (last bar: 2026-05-17)
- OP/USD (last bar: 2026-05-17)
- POL/USD (last bar: 2026-05-17)
- SOL/USD (last bar: 2026-05-17)
- UNI/USD (last bar: 2026-05-17)
- XRP/USD (last bar: 2026-05-17)

---

## 7. Backtest Engine Behavior Notes

> The backtest engine (backtest/engine.py) silently drops rows with close <= 0 (see _validate_ohlcv). It does NOT forward-fill missing bars — gaps simply reduce the number of bars available to the signal generator. Symbols with insufficient lookback history are silently excluded from portfolio selection at early timestamps via the min_history_bars eligibility filter in CrossSectionalMomentumStrategy.

| Behavior | Status |
|---|---|
| Silently drops close ≤ 0 rows | Yes ⚠️ |
| Forward-fills missing bars | No ✅ |

---

## 8. Universe Size Over Time

> The number of symbols with available data grows over the backtest window.
> Early periods have fewer assets in the opportunity set.

| Year | Symbols with data |
|---|---|
| 2020 | 9 |
| 2021 | 16 |
| 2022 | 16 |
| 2023 | 18 |
| 2024 | 19 |
| 2025 | 20 |
| 2026 | 20 |

---

## 9. Per-Symbol First/Last Bar Reference

| Symbol | First Bar | Last Bar | Rows | Label |
|---|---|---|---|---|
| ADA/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| ATOM/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| BCH/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| BTC/USD | 2020-01-01 | 2026-05-18 | 13974 | STALE |
| DOGE/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| ETH/USD | 2020-01-01 | 2026-05-18 | 13974 | STALE |
| LINK/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| LTC/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| XRP/USD | 2020-01-01 | 2026-05-17 | 13969 | STALE |
| SOL/USD | 2020-08-11 | 2026-05-17 | 12631 | STALE |
| DOT/USD | 2020-08-18 | 2026-05-17 | 12585 | STALE |
| UNI/USD | 2020-09-17 | 2026-05-17 | 12410 | STALE |
| AVAX/USD | 2020-09-22 | 2026-05-17 | 12379 | STALE |
| NEAR/USD | 2020-10-14 | 2026-05-17 | 12247 | STALE |
| AAVE/USD | 2020-10-15 | 2026-05-17 | 12242 | STALE |
| INJ/USD | 2020-10-21 | 2026-05-17 | 12205 | STALE |
| OP/USD | 2022-06-01 | 2026-05-17 | 8676 | STALE |
| APT/USD | 2022-10-19 | 2026-05-17 | 7838 | STALE |
| ARB/USD | 2023-03-23 | 2026-05-17 | 6905 | STALE |
| POL/USD | 2024-09-13 | 2026-05-17 | 3666 | STALE |

---

## 10. Remaining Limitations (Even After This Audit)

- **Intra-bar path unknown**: OHLCV only records open/high/low/close; actual
  tick-by-tick path is not auditable.
- **Wash trading / spoofed volume**: Exchange-reported volume may not reflect
  genuine market depth, especially on alt-coins.
- **Cross-exchange inconsistency**: Price and volume differ across exchanges.
  All data here is Kraken-only; another exchange may show different history.
- **Point-in-time universe**: Even with full history, we can only add coins
  that exist today. Coins that peaked and died before today are not retrievable.
- **Look-ahead in symbol selection**: The decision to include a coin in the
  universe is made with knowledge of its current status, which is a subtle
  form of look-ahead bias.
- **Listing-date uncertainty**: Exchange APIs may not accurately report the
  date a coin became actively tradable; early bars may have extreme spreads.

---

*Generated by `research/audit_crypto_data_quality.py` — research only.*