# Rollback Plan

## Goal
Restore the previous live bot behavior immediately if the canonical-base migration produces mismatched targets, bad fills, or unexpected behavior.

## Rollback controls
1. Keep the current live bot path as the default until the new path is proven.
2. Preserve the feature flag or adapter switch that selects the canonical path.
3. Keep the existing pending-signal and execution flow intact.
4. Maintain the ability to disable the new path without redeploying broker logic.

## Rollback triggers
- Target mismatch beyond the agreed tolerance.
- Unexpected exposure, cash, or timing drift.
- Any execution anomaly.
- Any data freshness or schedule mismatch.

## Rollback action
- Flip the feature flag back to the original live bot path.
- Stop the shadow/paper path.
- Re-run the current bot only after the issue is understood.
- Do not modify broker or scheduler internals during rollback.