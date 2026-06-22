# Current Live Bot vs Canonical Base Audit

This audit compares the read-only current live bot inventory and old live bot repo against the canonical research base strategy.

## Bottom line
- The old bot appears worse/different in design relative to the canonical base strategy.
- The current bot is a cross-sectional momentum rotator with a disabled BTC regime overlay; the canonical base is a BTC-led gated sleeve structure.
- The canonical base strategy can be shadow-tested safely in this research repo.
- No orders were placed and no live behavior changed.

## Key findings
- Current bot: 4h universe rotation across five symbols, top-3 selection, 75% gross exposure cap, one-bar delayed execution, disabled BTC gate.
- Canonical base: GATED_BTC_TS_INVESTED_75_25, btc_ts_60_240_12 base sleeve, 25% BTC-gated CS sleeve, 4h bars, rebalance every 12 bars, 1-bar gate lag.
- They are not interchangeable without an adapter and a staged migration plan.

## Recommended deployment sequence
1. Current live bot inventory.
2. Canonical base latest-signal audit.
3. Shadow mode in this research repo or a separate shadow runner.
4. Compare shadow targets to old bot targets for multiple cycles.
5. Feature-flagged adapter in the old/live repo.
6. Dry-run or paper mode.
7. Small-cap live mode only after review.
8. Rollback plan ready before any live cutover.

## Safety
- No changes were made to the old/live bot repo.
- No orders were placed.
- No scheduler, execution, credentials, or production config files were modified.