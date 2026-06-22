# Replacement Plan

## Preconditions
- Shadow mode has matched the canonical base target generation repeatedly.
- Target differences versus the current bot are fully explained.
- Stakeholders approve a controlled migration.

## Controlled replacement sequence
1. Add a feature-flagged adapter in the old repo to emit canonical-base targets in parallel with the current bot logic.
2. Keep the existing bot behavior as the default path.
3. Run the adapter in dry-run or paper mode first.
4. Compare outputs, notional sizes, and execution timing for multiple cycles.
5. Enable small-cap live mode only after review.
6. Expand only if paper/live outcomes remain consistent with expectations.

## What would eventually change
- A live adapter or selector in the old repo.
- Possibly the private config values if the canonical base becomes the production design.
- Possibly the scheduler only if the new design requires a different cadence.

## What should not change until final deployment approval
- Credentials and env files.
- Broker/exchange execution code.
- Pending-signal state handling.
- System scheduler/launchd/cron/plist/systemd artifacts.
- Production config defaults.