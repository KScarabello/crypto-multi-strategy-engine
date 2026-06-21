# Memecoin Catcher — Readiness Status Report
Generated: 2026-06-21T01:06:16Z

## Data Provenance
- Genuine prospective events and simulated backfill events are **separate files**.
- Simulated events are **never** counted toward evidence gates.

## Event Counts
| Metric | Value |
|---|---|
| Genuine signal rows | 5706 |
| Simulated backfill rows (EXCLUDED from gates) | 192 |
| Newest genuine detection timestamp | 2026-06-21T01:05:59Z |
| Oldest genuine detection timestamp | 2026-05-29T04:18:14Z |
| Distinct genuine collection days | 15 |
| Distinct symbols (genuine) | 429 |
| Duplicate event count | 0 |

## Outcome Completeness (Genuine Events)
| Horizon | Complete | Incomplete |
|---|---|---|
| 15m | 5694 | 12 |
| 1h | 5694 | 12 |
| 4h | 5481 | 225 |
| 24h | 5139 | 567 |

## Rule 1 Status
Definition: `LONG_EXPLOSION + is_volume_climax=True + is_clean_continuation=False`

> **Note:** `is_volume_climax` = `volume_ratio_4h > 10`; `is_clean_continuation` = `ret_15m>0 AND ret_1h>0 AND ret_4h>0` — derived from detection-time columns.

| Metric | Value |
|---|---|
| Rule 1 total events (any horizon) | 348 |
| Rule 1 matured-4h events (used by Gate A) | 336 |
| Rule 1 date range | 2026-05-29T04:18:14Z -> 2026-06-20T21:05:36Z |
| spread_pct available (detection-time, matured-4h events) | 336/336 (100.0%) |
| bid/ask/spread_abs in candidates pipeline | False |
| Spread field used | spread_pct (Kraken ticker detection-time) |

## Gate A Status (→ VALIDATION_READY)
**Pass count: 4 / 10**
*(Gate A uses Rule 1 matured-4h events; simulated rows are excluded)*

| Criterion | Threshold | Pass |
|---|---|---|
| Collection days | ≥ 30 | ❌ (15) |
| Total genuine events | ≥ 500 | ✓ (5706) |
| Rule 1 events | ≥ 150 | ✓ (336) |
| Rule 1 symbols | ≥ 25 | ✓ |
| Top symbol ≤ 20% | ≤ 20% | ✓ |
| Top day ≤ 20% | ≤ 20% | ❌ |
| Rule 1 positive median | > 0 | ❌ |
| Rule 1 positive mean @ 50 bps | > 0 | ❌ |
| Both time-split medians positive | Yes | ❌ |
| Excl top-3 symbols: positive mean | > 0 | ❌ |

## Current Readiness Label
> **EXPLORATORY_ONLY**

## Safety Confirmation
- Simulated backfill excluded from gates: ✓
- No live execution capability: ✓
- Five-coin shadow state not modified: ✓
