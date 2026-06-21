# Reversal Watch Frozen Candidate Spec (Research-Only)
Generated: 2026-06-21 UTC

## Candidate Identity
- Candidate name: REVERSAL_WATCH_FIXED_4H_RESEARCH
- Candidate status: RESEARCH_ONLY
- Approval status: NOT_LIVE_READY
- Trading status: NOT_APPROVED_FOR_TRADING

## Scope And Safety
- This document defines a research specification only.
- It does not change strategy, execution, scheduling, or launch behavior.
- It does not authorize trading.

## Dataset Scope
- Use genuine prospective memecoin data only:
  - data/memecoin_signal_history.csv
  - data/memecoin_signal_outcomes.csv
- Explicitly excluded from this candidate spec:
  - data/memecoin_backfilled_signal_events.csv

## Frozen Predicate
- Canonical signal column: ohlc_signal_type
- Frozen cohort predicate: ohlc_signal_type == REVERSAL_WATCH

## Outcomes To Track
- Primary outcome:
  - future_ret_4h_pct
- Secondary outcomes (when present in genuine outcomes data):
  - future_ret_15m_pct
  - future_ret_1h_pct
  - future_ret_24h_pct
  - max_favorable_4h_pct
  - max_adverse_4h_pct
  - max_favorable_24h_pct
  - max_adverse_24h_pct

## Cost Assumptions To Track
- 0 bps
- 50 bps
- 100 bps
- 150 bps
- 200 bps

## Current Evidence Snapshot (Latest Genuine Audit)
Source: reports/memecoin_reversal_watch_readiness_audit.md

- Total events: 319
- Mature 15m count: 318
- Mature 1h count: 318
- Mature 4h count: 315
- Mature 24h count: 293
- Avg 4h return: +0.5827%
- Median 4h return: +0.3908%
- Win rate: 0.5587
- Cost-adjusted win rate at 50 bps: 0.4952
- Fail rate <= -3%: 0.1619
- Fail rate <= -5%: 0.1079

Interpretation snapshot:
- Positive gross mean/median in current sample.
- Cost-adjusted win rate at 50 bps is near break-even.
- Tail-loss rates remain material.
- Candidate remains research-only.

## Diagnostic Research Criteria (Not Live Thresholds)
Important:
- These are diagnostic criteria for continued research quality checks only.
- They are not official live trading thresholds and must not be interpreted as deployment gates.

Suggested diagnostics:
- Minimum collection days: >= 30 distinct UTC dates with genuine events.
- Minimum mature 4h cohort events: >= 300.
- Minimum cohort symbols: >= 40.
- Top symbol concentration cap: <= 20% of mature 4h cohort.
- Top day concentration cap: <= 20% of mature 4h cohort.
- Positive median 4h return: > 0.
- Positive average net return after 50 bps: > 0.
- Positive median net return after 50 bps: > 0.
- Both time halves positive median: first and second half medians > 0.
- Excluding top 3 symbols remains positive: mean net return > 0.
- Fail-rate guardrails for ongoing research quality:
  - fail rate <= -3% not persistently rising.
  - fail rate <= -5% not persistently rising.

## Prospective Tracking Plan
- Use existing collection outputs only; no new scheduler required.
- No live trading integration.
- Recompute candidate metrics after sufficient new genuine events accumulate.
- Track side-by-side against:
  - Rule 1 locked cohort
  - LONG_EXPLOSION overall
  - DUMPING
- Re-review after larger sample growth in collection days and mature 4h events.

## Candidate Invalidation Conditions (Research)
The candidate should be deprioritized if one or more of the following persist:
- Median 4h return turns and stays negative.
- Cost-adjusted win rate at 50 bps stays near or below 50%.
- Tail-loss profile worsens (fail rates <= -3% or <= -5% trend up).
- Concentration risk increases (top symbol/day remains elevated without diversification).

## Explicit Non-Implementation Statement
- This spec does not alter Rule 1, Reversal Watch definition, evidence thresholds, entry/exit logic, or signal classification.
- This spec does not alter launch agents, collector cadence, Kraken execution, live trading behavior, or five-coin shadow behavior.
