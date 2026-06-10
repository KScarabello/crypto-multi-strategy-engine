# Memecoin Catcher — Current State Report
Generated: 2026-06-09  (based on data through 2026-06-06)

## Status
Research sleeve only. No live orders. No connection to five-coin shadow framework.

---

## 1. Relevant Files

### Research scripts (`research/memecoin_catcher/`)
| File | Purpose |
|---|---|
| `validate_memecoin_candidate_rules.py` | Robustness validation for top-3 entry rules |
| `exit_policy_experiment.py` | Compares 10 exit policies on rule-1 events |
| `backfill_recent_memecoin_signals.py` | Backfills OHLC + forward returns for stored snapshots |
| `analyze_backfill_signal_traits.py` | Trait sweeps and group summaries on backfilled events |
| `candidate_strategy_comparison.py` | Multi-rule comparison with event-level log |
| `clean_continuation_event_log.py` | Builds event log for CC-filtered subset |
| `enrich_candidates_with_ohlc.py` | Fetches OHLC from Kraken for candidate enrichment |
| `evaluate_signal_outcomes.py` | Evaluates outcome for each saved snapshot signal |
| `fetch_kraken_universe.py` | Fetches full Kraken pair universe + ticker |
| `monitor_candidate_rule_forward.py` | Forward-monitors candidate rules |
| `rank_memecoin_candidates.py` | Ranks and filters Kraken universe to explosion candidates |
| `resume_memecoin_research.py` | One-shot resume: evaluate outcomes + fresh snapshot |
| `save_signal_snapshot.py` | Saves a timestamped snapshot of ranked candidates |
| `test_backfill_candidate_filter_grid.py` | Grid search over filter combinations on backfill |

### Data files
| File | Rows | Description |
|---|---|---|
| `data/memecoin_backfilled_signal_events.csv` | 1 451 | **Primary research dataset** — LONG_EXPLOSION events + forward returns |
| `data/memecoin_signal_history.csv` | 166 | Live-collected snapshot history |
| `data/memecoin_signal_outcomes.csv` | 166 | Evaluated outcomes for live-collected snapshots |
| `data/memecoin_signal_snapshots/` | 4 files | Raw snapshots: 2026-05-29, 30, 06-01, 06-06 |
| `data/memecoin_candidates_latest.csv` | 67 | Most recent candidate list |
| `data/memecoin_rule_experiment_summary_latest.csv` | 8 | Rule experiment summary |

### Report files
| File | Description |
|---|---|
| `reports/memecoin_candidate_rule_validation_summary.csv` | Main rule validation metrics |
| `reports/memecoin_candidate_rule_time_splits.csv` | Time-split stability |
| `reports/memecoin_candidate_rule_symbol_robustness.csv` | Symbol leave-one-out |
| `reports/memecoin_candidate_rule_fee_stress.csv` | Fee/slippage sensitivity |
| `reports/memecoin_candidate_rule_liquidity_diagnostics.csv` | Spread/volume (incomplete — see §8) |
| `reports/memecoin_exit_policy_summary.csv` | 10-policy exit comparison |
| `reports/memecoin_backfill_candidate_strategy_summary.csv` | Backfill rule comparison |
| `reports/memecoin_backfill_filter_grid_summary.csv` | Filter grid search results |

### Tests (`tests/`)
15 test files covering all memecoin research modules.

---

## 2. Latest Generated Reports
All key reports were regenerated on **2026-06-05** via the validation script
and exit policy experiment. Reports are current against the backfilled dataset.

---

## 3. Current Candidate Rules

| # | Rule Label | Definition |
|---|---|---|
| 1 | `rule1_volume_climax_cc_false_tp10` | `ohlc_signal_type == LONG_EXPLOSION` AND `is_volume_climax == True` AND `is_clean_continuation == False`. Exit: tp10_else_4h. **Leading rule.** |
| 2 | `rule2_volume_climax_safe_hours_tp10` | `is_volume_climax == True` AND hour_of_day_utc ∈ {17,18,19}. Exit: tp10_else_4h. |
| 3 | `rule3_vr5_safe_hours_ret24lt25_tp10` | `volume_ratio_4h ≥ 5` AND safe hours AND `ret_24h_pct < 25`. Exit: tp10_else_4h. |

Baseline rules (for comparison, not candidates):
- `baseline_all_fixed_4h`, `baseline_all_tp10`, `baseline_volume_climax_tp10`, `baseline_safe_hours_tp10`

---

## 4. Current Exit Variants (tested on rule-1 events, n=20)

| Policy | Avg | Median | WR | fail≤-3 | Avg excl best |
|---|---|---|---|---|---|
| fixed_4h_exit | +2.82% | +1.07% | 0.60 | 0.050 | +0.65% |
| take_profit_10_else_4h | +1.29% | +1.07% | 0.60 | 0.050 | +0.84% |
| take_profit_5_else_4h | +0.85% | +1.07% | 0.60 | 0.050 | +0.63% |
| stop_loss_3_else_4h | +0.19% | -0.05% | 0.50 | 0.350 | -0.14% |
| fixed_24h_exit | -0.41% | -1.51% | 0.25 | 0.350 | -2.31% |
| trailing_stop_after_up_5 | +4.88% | +1.07% | 0.60 | 0.050 | +1.00% *(approx)* |

**Leading exit: `fixed_4h_exit`** has best median and lowest failure rate.
`tp10_else_4h` is a capped variant; it is still positive but limits best outcomes.
Stop-loss at 3% is harmful (high fail rate; cuts normal drawdowns prematurely).

---

## 5. Available Data Date Range

| Source | Earliest | Latest | Days |
|---|---|---|---|
| Backfilled events | 2026-05-31 06:00 UTC | 2026-06-06 05:00 UTC | 7 |
| Live snapshots | 2026-05-28 (snap) | 2026-06-05 (snap) | 4 snapshots, ~7 unique days |

**The seven-day limitation still applies.** There are 7 distinct calendar days in
the backfilled dataset (2026-05-31 through 2026-06-06 inclusive).

---

## 6. Total Events and Events by Rule

| Dataset | Events |
|---|---|
| All backfilled events (LONG_EXPLOSION) | 1 451 |
| Rule 1: volume_climax AND NOT CC | 80 total, **78 with complete 4h outcome** |
| Rule 2: volume_climax + safe hours | 33 |
| Rule 3: vr4h≥5 + safe hours + ret24<25 | 59 |

Rule-1 symbol breakdown (n=78): 16 distinct symbols, top 5 each ≤15% of events.
No single symbol exceeds 15% of rule-1 events.

---

## 7. Test Status

**Full suite (2026-06-09):**
- 1 327 passed, 1 282 warnings, **1 failed** (pre-existing unrelated BTC robustness tolerance test)
- `test_resume_memecoin_research.py::test_no_broker_imports` — **FIXED** (false positive; 
  `credentials` appeared only in docstring saying "No private credentials". Test now checks
  import lines only. Passes after fix.)
- All 248 core memecoin tests pass.

---

## 8. Unfinished or Inconsistent Work

1. **Liquidity diagnostics incomplete**: `memecoin_candidate_rule_liquidity_diagnostics.csv`
   shows `spread_available=False` and all NaN spread/volume values for all three rules.
   The backfill data does not carry `spread_pct` or `quote_volume_est` columns reliably
   for these subsets. Spread and fill-risk modeling is missing.

2. **Exit policy experiment uses only n=20**: The exit policy is evaluated on 
   `memecoin_clean_continuation_event_log.csv` (20 completed 4h events), not the full 
   backfilled dataset. Results are directionally useful but very noisy.

3. **No automated collection**: Collection is manual. Last run: 2026-06-05 22:21 UTC.
   No LaunchAgent or cron job exists for memecoin research. Gap since last collection:
   ~3 days (to 2026-06-09).

4. **24h forward returns have 100 NaN rows** in backfilled dataset (6.9%) — these events
   were near the collection cutoff and do not have complete 24h windows.

5. **`summarize_signal_outcomes` stub missing** — logged as warning in resume log; 
   summary step is skipped silently.

6. **Safe-hours filter (rule 2 & 3) shows mixed hour results**: Hour 17 is negative
   for both rules; hours 18-19 are positive for rule 2 but all negative for rule 3.

---

## 9. Seven-Day Limitation Status
**Confirmed still applies.** The backfilled dataset spans exactly 7 calendar days.
The forward returns on rule-1 events use up to 6 days of completed outcomes.
This is insufficient for robust validation — see Part 3 data readiness label.

---

## 10. Live Execution Confirmation
- Zero memecoin references found in `deployment/`, `live/`, or `shadow_state/`.
- No import of `brokers`, `live_trading`, `place_order`, or `ccxt` in any
  `research/memecoin_catcher/` script.
- The `fetch_kraken_universe.py` module fetches public ticker data only (read-only REST).
- The `resume_memecoin_research.py` script explicitly documents "No trading. No orders.
  No private credentials."
- **No live orders can be placed from this research sleeve.**
