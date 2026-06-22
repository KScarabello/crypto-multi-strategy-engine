# Current Live Bot vs Canonical Base Shadow Signal

## Answers
1. Can we safely compute old bot targets? YES.
2. Can we safely compute base strategy targets? YES.
3. What does the base strategy want right now? cash only with cash=1.0000 and BTC gate=OFF.
4. How different is it from the old bot? BTC exposure delta=-0.2500, alt exposure delta=-0.5000, cash delta=+0.7500.
5. Is this ready for repeated manual shadow runs? YES for manual shadow runs, but warnings remain.
6. What remains blocked before production adapter work? Feature-flagged adapter design, paper-mode validation, and final live-deployment approval.

## Snapshot Summary
- Current bot timestamp: 2026-05-17 04:00:00+00:00
- Base timestamp: 2026-06-10 00:00:00+00:00
- Current bot rebalance point: False
- Base rebalance point: False
- Current bot data fresh: False
- Base data fresh: False
- Current bot symbols: BTC/USD, SOL/USD, XRP/USD
- Base symbols: none
- Current only symbols: BTC/USD,SOL/USD,XRP/USD
- Base only symbols: none
- Warnings: current bot data is stale | current bot snapshot is not on a rebalance bar | canonical base data is stale | canonical base snapshot is not on a rebalance bar | canonical base BTC gate is off | stale data warning

## Safety
- No orders were placed.
- No live behavior changed.
- No broker or exchange execution path was called.
