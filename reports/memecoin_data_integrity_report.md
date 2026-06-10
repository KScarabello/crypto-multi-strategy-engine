# Memecoin Catcher — Data Integrity Report
Generated: 2026-06-09

## Audit Results

| Check | Result | Detail |
|---|---|---|
| 1. Earliest timestamp | 2026-05-31 06:00 UTC | |
| 2. Latest timestamp | 2026-06-06 05:00 UTC | |
| 3. Distinct collection days | 7 | 2026-05-31 through 2026-06-06 |
| 4. Missing collection periods | **NONE** | All 7 days consecutive |
| 5. Duplicate events (symbol+ts) | **0** | Clean |
| 6. Duplicate symbols/timestamps | **0** | Clean |
| 7. Incomplete 4h forward windows | 17 / 1 451 (1.2%) | Near cutoff events |
| 8. Incomplete 24h forward windows | 100 / 1 451 (6.9%) | Near cutoff events |
| 9. Look-ahead leakage in features | **NOT DETECTED** | See notes |
| 10. Stale/partial candles | **POSSIBLE** | See notes |
| 11. Survivorship/listing bias | **PRESENT** | Only currently-listed Kraken pairs |
| 12. Delisted coins missing | **YES** | Cannot be measured without historical pair list |
| 13. Liquidity/volume coverage | **INCOMPLETE** | Spread/fill data absent from rule subsets |
| 14. Spreads/slippage modeled | **NO** | Fee stress only; no spread simulation |
| 15. Single symbol/day dominance | **PARTIAL** | Jun 1 = 27.7% of events; acceptable for full set |

---

## Notes

### Feature Look-Ahead (item 9)
Entry-classification features (`ohlc_signal_type`, `is_volume_climax`,
`is_clean_continuation`, `volume_ratio_1h/4h`, `ret_*_pct`) are computed at
snapshot time and stored at detection. Forward-return columns (`future_ret_*`,
`max_favorable_*`, `max_adverse_*`) are added afterward via
`backfill_recent_memecoin_signals.py` using OHLC fetched after the fact. 
**No evidence of in-sample feature leakage**, but the backfill script must be
reviewed to confirm: (a) OHLC bars are complete/closed at evaluation time, and
(b) the entry price used is the snapshot price, not a future price.

### Stale/Partial Candles (item 10)
The backfill script fetches 1h and 4h OHLC from Kraken. The most-recent bar at
fetch time may be incomplete. The script's handling of the current (open) bar
has not been independently verified. This is a known risk for events captured
within the last 1–4 hours of the collection window.

### Survivorship Bias (item 11–12)
All data comes from pairs currently listed on Kraken. Pairs delisted between
data collection and now are missing. For a 7-day window this risk is low, but
it cannot be ruled out. Any coin that was delisted after it exploded would be
excluded from the negative-return tail, creating a mild positive bias.

### Liquidity and Fill Modeling (items 13–14)
The liquidity diagnostics CSV shows `spread_available = False` for all three
candidate rules, meaning spread data was not available for the backfilled subset.
Actual entry cost on illiquid memecoins could be 50–300 bps (0.5–3%). Fee stress
tests show:
- Rule 1 remains positive through 100 bps (avg +1.25%, median +0.27%)
- Rule 1 goes negative at 150 bps median
- Rule 2 turns median-negative at 50 bps
- Rule 3 goes negative at any non-zero fee level

### Day Concentration (item 15)
June 1 contains 402 / 1 451 events (27.7%). Within rule 1 (n=78), June 1 has
15 events (19.2%) with mean +3.74% — the best single day. The time-split test
shows rule 1 is positive in both halves, but daily variance is high.

---

## Data Readiness Label

> **EXPLORATORY_ONLY**

Justification:
- Only 7 days of data
- Fewer than 100 events in leading rule
- Liquidity and spread costs not modeled
- Survivorship bias cannot be ruled out
- No out-of-sample window available
- No walk-forward or cross-validated test
