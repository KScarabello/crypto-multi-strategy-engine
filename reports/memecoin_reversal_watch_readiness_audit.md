# Memecoin Reversal Watch Readiness Audit (Genuine-Only)
Generated: 2026-06-21 UTC

## Scope
- Analysis/reporting only.
- No strategy or threshold changes.
- No live execution changes.
- No backfill execution.

## Inputs
- data/memecoin_signal_history.csv
- data/memecoin_signal_outcomes.csv
- reports/memecoin_evidence_audit_current.md
- reports/memecoin_readiness_status.md
- Explicitly excluded: data/memecoin_backfilled_signal_events.csv

## Canonical Predicate Used
- Signal column in outcomes: ohlc_signal_type
- Reversal Watch predicate: ohlc_signal_type == REVERSAL_WATCH

## Reversal Watch Metrics (Genuine Prospective Only)
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

## Context Baseline (Not For Trading Decision)
- LONG_EXPLOSION mature 4h n: 4649
- LONG_EXPLOSION avg 4h: -0.2824%
- LONG_EXPLOSION median 4h: -0.5528%
- LONG_EXPLOSION win rate: 0.3833
- LONG_EXPLOSION cost-adjusted win rate at 50 bps: 0.3054

## Interpretation
- Reversal Watch looks stronger than LONG_EXPLOSION on gross and median 4h outcomes in the current genuine sample.
- Cost-adjusted win rate at 50 bps is near break-even (0.4952), not clearly robust.
- Tail risk remains meaningful (16.2% at <= -3%, 10.8% at <= -5%).
- Conclusion: keep as a research candidate for dedicated robustness work; not sufficient for trading promotion.

## Safety Confirmation
- Genuine-only analysis performed.
- Backfill dataset excluded from metric computation.
- No changes to strategy logic, automation, launch agents, execution code, or shadow state.
