# Memecoin Evidence Audit (Current)
Generated: 2026-06-21 UTC

## Executive Summary
- Genuine-only Rule 1 evidence currently shows 336 matured 4h events from 5706 genuine signals.
- Gate A remains 4/10, so readiness correctly stays EXPLORATORY_ONLY.
- The 336 vs 16 mismatch is explained by different input datasets and different evaluation intent:
	- 336 comes from genuine prospective outcomes in data/memecoin_signal_outcomes.csv.
	- 16 comes from a small historical backfill subset in data/memecoin_backfilled_signal_events.csv used by candidate rule validation.
- Candidate validation is useful as backfill research context, but it is not a readiness-evidence counter and should not be interpreted as current genuine sample size.

## Scope And Safety
- Analysis/reporting only.
- No strategy rules, thresholds, entry/exit logic, classification logic, live trading code, Kraken execution code, or five-coin shadow state were modified.
- No historical backfill, no live trading, no order placement, no credential sourcing were run.

## Files Inspected
- data/memecoin_signal_history.csv
- data/memecoin_signal_outcomes.csv
- data/memecoin_backfilled_signal_events.csv
- reports/memecoin_readiness_status.csv
- reports/memecoin_readiness_status.md
- reports/memecoin_candidate_rule_validation_summary.csv
- reports/memecoin_candidate_rule_time_splits.csv
- reports/memecoin_candidate_validation_report.md
- logs/memecoin_collection_cycle.log
- scripts/status_memecoin_launchagent.sh
- scripts/run_memecoin_collection_cycle.sh
- research/memecoin_catcher/memecoin_readiness_report.py
- research/memecoin_catcher/validate_memecoin_candidate_rules.py
- memecoin_readiness_status.md (workspace root): not present

## Provenance: Which File Drives Which Number
- status_memecoin_launchagent.sh:
	- Reads data/memecoin_signal_history.csv and data/memecoin_signal_outcomes.csv.
	- Computes Rule 1 matured 4h as:
		- ohlc_signal_type == LONG_EXPLOSION
		- is_volume_climax == True
		- is_clean_continuation == False
		- future_ret_4h_pct not null
	- Reads Gate A and readiness label from reports/memecoin_readiness_status.csv.

- readiness reporting (research/memecoin_catcher/memecoin_readiness_report.py):
	- Primary evidence input: data/memecoin_signal_history.csv and data/memecoin_signal_outcomes.csv.
	- Simulated backfill file is loaded for counting/provenance only and explicitly excluded from gates.
	- Gate A uses Rule 1 matured 4h events from genuine outcomes.

- candidate validation (research/memecoin_catcher/validate_memecoin_candidate_rules.py):
	- Input is data/memecoin_backfilled_signal_events.csv.
	- Rule1 mask is is_volume_climax True and is_clean_continuation False (no LONG_EXPLOSION filter in this module).
	- Evaluates backfill events with completed future_ret_4h_pct and tp10_else_4h scoring.

## Genuine-Only Rule 1 Recompute (Canonical)
Locked Rule 1 definition used:
- signal_type == LONG_EXPLOSION
- is_volume_climax == True
- is_clean_continuation == False

Computed from data/memecoin_signal_outcomes.csv (joined context from canonical genuine pipeline):

| Metric | Value |
|---|---:|
| Genuine history rows | 5706 |
| Rule 1 total events | 348 |
| Rule 1 mature 15m | 348 |
| Rule 1 mature 1h | 348 |
| Rule 1 mature 4h | 336 |
| Rule 1 mature 24h | 303 |
| Avg 4h return | -0.7876% |
| Median 4h return | -0.8818% |
| Win rate | 0.3899 |
| Fail rate at -3% | 0.2708 |
| Fail rate at -5% | 0.1310 |
| Avg net after 50 bps | -1.2876% |
| Median net after 50 bps | -1.3818% |
| Avg net after 100 bps | -1.7876% |
| Median net after 100 bps | -1.8818% |

Concentration and stability:
- Top symbol concentration: SYN/USD, 14 of 336 (4.17%).
- Top day concentration within Rule 1 matured 4h: 72 of 336 (21.43%), day 2026-06-15.
- Time-of-day concentration within Rule 1 matured 4h: hour 18 UTC, 40 of 336 (11.90%).
- First-half median (readiness day-index split): -1.6375%.
- Second-half median: -0.0212%.
- Excluding top 3 symbols:
	- n = 304
	- mean = -1.2450%
	- median = -0.9895%

Interpretation:
- The second half is less negative than the first half (directional improvement), but both central tendency and net-of-cost metrics are still below zero.
- Evidence remains inconclusive for promotion beyond exploratory.

## Gate A Recompute (Genuine-Only)

| Criterion | Threshold | Current | Pass |
|---|---|---:|:---:|
| Collection days | >= 30 | 15 | No |
| Total genuine events | >= 500 | 5706 | Yes |
| Rule 1 matured 4h events | >= 150 | 336 | Yes |
| Rule 1 symbols | >= 25 | 126 | Yes |
| Top symbol share (Rule 1) | <= 20% | 4.17% | Yes |
| Top day share (full genuine history) | <= 20% | 22.34% | No |
| Rule 1 positive median | > 0 | -0.8818% | No |
| Rule 1 positive mean at 50 bps | > 0 | -1.2876% | No |
| Both time-split medians positive | Yes | first -1.6375%, second -0.0212% | No |
| Excl top-3 symbols positive mean | > 0 | -1.2450% | No |

Result: 4/10 (matches current readiness output).

## Mismatch Explanation: 336 vs 16
The numbers are not counting the same population.

- 336 is Rule 1 matured 4h count from genuine prospective outcomes (data/memecoin_signal_outcomes.csv), used by readiness and surfaced by status helper.
- 16 is candidate-validation n for rule1_volume_climax_cc_false_tp10 from backfill events (data/memecoin_backfilled_signal_events.csv), where:
	- total backfill rows = 192
	- completed 4h outcomes in backfill = 177
	- rule1 subset in backfill completed 4h = 16
	- backfill sample covers only 6 unique dates

Additional nuance:
- The candidate validation Rule 1 mask in validate_memecoin_candidate_rules.py does not require LONG_EXPLOSION; despite that broader mask, its sample remains small because it runs on the small backfill file.
- Therefore, candidate validation n is a research-context sample size, not the production genuine evidence count.

## Trustworthiness Of Candidate Validation Block
- Trustworthy for what it is: backfill-only robustness exploration over data/memecoin_backfilled_signal_events.csv.
- Not trustworthy as a readiness-evidence counter for current genuine data.
- Risk today: readers can misinterpret candidate-validation console output as if it were current genuine evidence.

Minimal reporting fix for later task (not implemented here):
- Add an explicit console header line in Stage H and validate_memecoin_candidate_rules.py:
	- Data source path
	- sample scope (backfill-only)
	- not comparable to Gate A genuine counts
- Optionally rename printed n field to backfill_n to reduce confusion.

## Why Readiness Remains EXPLORATORY_ONLY
- Gate A is 4/10 because five criteria fail:
	- insufficient collection-day span
	- excessive top-day concentration
	- negative Rule 1 median
	- negative Rule 1 mean after 50 bps
	- negative mean when excluding top-3 symbols
- Blocking issue is not event count; it is distribution concentration and weak net performance in genuine Rule 1 outcomes.

## Next Recommended Analysis Step
- Continue accumulating genuine prospective days and rerun this audit after the next 7-14 days to check:
	- top-day concentration decay below 20%
	- median and net mean crossing above zero
	- stability of ex-top3 mean and split medians

## Explicit No-Change Confirmation
- No strategy behavior changed.
- No live behavior changed.
- No shadow behavior changed.
