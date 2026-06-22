# Comparison Summary

## Answers to the requested questions
1. The old/current live bot appears to be a cross-sectional momentum rotator across BTC/USD, ETH/USD, XRP/USD, SOL/USD, and AVAX/USD, with a 75% gross exposure cap and one-bar delayed execution.
2. The canonical base strategy is GATED_BTC_TS_INVESTED_75_25: a 75% BTC time-series sleeve plus a 25% BTC-gated cross-sectional sleeve.
3. Yes. They are fundamentally different in design.
4. No. The current bot does not actively use BTC time-series timing in its live config; the BTC regime filter exists but is disabled.
5. Yes. The current bot does use a cross-sectional sleeve-like ranking model.
6. No. The current bot does not have an active BTC gate.
7. Yes, but only as a cash cushion/exposure cap. It is not the same as the canonical BTC-gated regime behavior.
8. The current bot is effectively daily on the 20:00 UTC bar despite a 4h cron wake-up; the canonical base is a 4h strategy rebalance every 12 bars with a 1-bar gate lag.
9. The current bot allocates roughly equal weight across the top 3 names subject to caps; the canonical base uses a fixed 75/25 sleeve split with the CS sleeve only active when BTC_TS_INVESTED is on.
10. To run the canonical base in shadow mode, the research repo needs a shadow runner that reads the canonical base report, reconstructs latest signals, and writes target weights without touching live execution or broker state.
11. A safe replacement would require a feature-flagged adapter in the old repo, parity tests, paper-mode validation, and multiple shadow cycles before any live use.
12. Production files that would eventually need controlled change include the old repo's live configuration and adapter path, but not until final deployment review.
13. Do not touch .env, config_private.py, live execution modules, scheduler files, pending-signal state, or production config until final deployment approval.
14. Required tests include target parity, one-bar delay behavior, rebalance-bar detection, gate activation, allocation math, and paper-mode cycle checks.
15. Safest sequence: inventory capture, canonical latest-signal audit, shadow mode in research or separate runner, compare shadow targets to old bot targets for multiple cycles, add feature-flagged adapter, dry-run/paper mode, then only consider small-cap live mode with rollback prepared.

## Conclusion
The old bot appears worse/different in design relative to the canonical base because it is a cross-sectional momentum rotator with disabled BTC gating, while the canonical base is a BTC-led gated sleeve structure.

The base strategy can be shadow-tested safely in the research repo without changing live behavior.

Immediate next step: build a read-only shadow runner that emits canonical base targets and compares them to the current live bot targets over several cycles.

Safety confirmation: no orders were placed and no live behavior changed.