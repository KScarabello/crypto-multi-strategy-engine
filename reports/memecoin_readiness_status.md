# Memecoin Catcher — Readiness Status Report
Generated: 2026-06-10T04:23:39Z

## Data Provenance
- Genuine prospective events and simulated backfill events are **separate files**.
- Simulated events are **never** counted toward evidence gates.

## Event Counts
| Metric | Value |
|---|---|
| Genuine signal rows | 189 |
| Simulated backfill rows (EXCLUDED from gates) | 192 |
| Newest genuine detection timestamp | 2026-06-10T04:08:38Z |
| Oldest genuine detection timestamp | 2026-05-29T04:18:14Z |
| Distinct genuine collection days | 5 |
| Distinct symbols (genuine) | 144 |
| Duplicate event count | 0 |

## Outcome Completeness (Genuine Events)
| Horizon | Complete | Incomplete |
|---|---|---|
| 15m | 172 | 17 |
| 1h | 172 | 17 |
| 4h | 166 | 23 |
| 24h | 166 | 23 |

## Rule 1 Status
Definition: `LONG_EXPLOSION + is_volume_climax=True + is_clean_continuation=False`

> **Note:** `is_volume_climax` = `volume_ratio_4h > 10`; `is_clean_continuation` = `ret_15m>0 AND ret_1h>0 AND ret_4h>0` — derived from detection-time columns.

| Metric | Value |
|---|---|
| Rule 1 total events (any horizon) | 21 |
| Rule 1 matured-4h events (used by Gate A) | 17 |
| Rule 1 date range | 2026-05-29T04:18:14Z -> 2026-06-01T04:43:46Z |
| spread_pct available (detection-time, matured-4h events) | 17/17 (100.0%) |
| bid/ask/spread_abs in candidates pipeline | False |
| Spread field used | spread_pct (Kraken ticker detection-time) |

## Gate A Status (→ VALIDATION_READY)
**Pass count: 1 / 10**
*(Gate A uses Rule 1 matured-4h events; simulated rows are excluded)*

| Criterion | Threshold | Pass |
|---|---|---|
| Collection days | ≥ 30 | ❌ (5) |
| Total genuine events | ≥ 500 | ❌ (189) |
| Rule 1 events | ≥ 150 | ❌ (17) |
| Rule 1 symbols | ≥ 25 | ❌ |
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
