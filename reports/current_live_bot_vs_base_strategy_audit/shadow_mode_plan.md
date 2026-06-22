# Shadow Mode Plan

## Objective
Run the canonical base strategy in read-only shadow mode and compare its targets against the current live bot targets without modifying live execution.

## Steps
1. Capture the current live bot inventory and store the exact target-generation assumptions.
2. Reconstruct the canonical base latest-signal audit from reports/btc_base_portfolio_canonical/.
3. Build or use a research-only shadow runner in this repo that emits canonical targets only.
4. Compare shadow targets to the old bot targets across multiple rebalance cycles.
5. Record differences in symbol selection, weights, cash, and timing.
6. Verify one-bar delay and rebalance-bar alignment before any production discussion.

## Acceptance criteria
- Shadow targets are deterministic and repeatable.
- No live files, schedules, credentials, or broker code are changed.
- Differences are understood and documented before any adapter work.

## Constraints
- No orders.
- No live execution.
- No changes to the old repo.
- No scheduler or production config edits.