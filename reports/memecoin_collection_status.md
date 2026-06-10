# Memecoin Catcher — Collection Status Report
Generated: 2026-06-09

---

## Is Collection Automated?
**NO.** There is no LaunchAgent, cron job, or background daemon for memecoin
event collection. All snapshots have been captured by manual invocation of
`resume_memecoin_research.py`.

---

## Last Successful Run
- **Timestamp:** 2026-06-05 22:21:39 UTC (log: `logs/memecoin_resume.log`)
- **Outcomes evaluated:** 166 total rows (102 pre-existing + 64 new)
- **Newest snapshot file:** `data/memecoin_signal_snapshots/memecoin_signals_20260606_052101.csv`
- **Newest event timestamp in history:** 2026-06-06T05:21:01Z (UTC offset timestamp)
- **Gap to now (2026-06-09):** ~3 days of missing collection

---

## Snapshot File History
| File | Rows | Captured (approx) |
|---|---|---|
| memecoin_signals_20260529_041814.csv | 21 | 2026-05-28 21:18 UTC |
| memecoin_signals_20260530_041743.csv | 50 | 2026-05-29 21:17 UTC |
| memecoin_signals_20260601_044346.csv | 31 | 2026-05-31 21:43 UTC |
| memecoin_signals_20260606_052101.csv | 64 | 2026-06-05 22:21 UTC |

4 snapshots, ~4 distinct collection days.

---

## Completed-Candle Protection
The backfill script (`backfill_recent_memecoin_signals.py`) fetches Kraken OHLC
for completed bars. **RISK:** Events captured within the last 4h before a
collection run may use a partially-formed bar for the max_favorable / max_adverse
calculation. This has not been independently verified in the code.

---

## Duplicate-Event Protection
The history file (`data/memecoin_signal_history.csv`) is appended-to by
`save_signal_snapshot.py`. Deduplication relies on `snapshot_ts_utc + pair_id`
uniqueness. Confirmed: **0 duplicate (symbol+ts) rows** in current backfilled set.

---

## Live-Order Capability
**NONE.** The collection pipeline uses:
1. `fetch_kraken_universe.py` — public REST, read-only
2. `rank_memecoin_candidates.py` — pure computation
3. `enrich_candidates_with_ohlc.py` — public OHLC REST, read-only
4. `save_signal_snapshot.py` — writes local CSV only
5. `evaluate_signal_outcomes.py` — fetches historical OHLC, read-only

No authenticated Kraken API calls. No order placement possible.

---

## Proposed Minimal Safe Collection Cycle (if manual collection resumes)

**Frequency:** Once per day, manually, during a stable market window (avoid
collection within 1h of a major news event if possible).

**Commands (in order):**
```bash
# Step 1: evaluate outcomes for previous snapshots
.venv/bin/python -m research.memecoin_catcher.evaluate_signal_outcomes

# Step 2 (all-in-one): fetch universe + rank + enrich + snapshot + evaluate
.venv/bin/python -m research.memecoin_catcher.resume_memecoin_research

# Step 3 (weekly): backfill forward returns for all stored snapshots
.venv/bin/python -m research.memecoin_catcher.backfill_recent_memecoin_signals

# Step 4 (weekly): regenerate validation reports
.venv/bin/python -m research.memecoin_catcher.validate_memecoin_candidate_rules
```

**Storage:**
- Events: `data/memecoin_backfilled_signal_events.csv` (append-only)
- Snapshots: `data/memecoin_signal_snapshots/` (timestamped, never overwritten)
- History: `data/memecoin_signal_history.csv` (append-only)
- Outcomes: `data/memecoin_signal_outcomes.csv` (updated in place)
- Logs: `logs/memecoin_resume.log`

**Completely separate from five-coin shadow trading:** the collection commands
do not touch `shadow_state/`, `brokers/`, `execution/`, `live/`, or any
five-coin strategy files.

---

## Next Collection Step
Run `resume_memecoin_research.py` to close the 3-day gap, then follow with
`backfill_recent_memecoin_signals.py` to add forward returns for new snapshots.
Do **not** automate until Gate A evidence thresholds are met (see
`memecoin_evidence_thresholds.md`).
