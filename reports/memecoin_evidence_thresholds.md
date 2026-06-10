# Memecoin Catcher — Evidence Threshold Specification
Locked: 2026-06-09
Status: PRE-DATA — thresholds set before seeing future data; must not be relaxed.

---

## Purpose
These thresholds define the minimum evidence required before a candidate rule
can be promoted to any stage beyond EXPLORATORY_ONLY. They are recorded now,
before additional data is collected, so they cannot be moved retroactively to
make a candidate pass.

---

## Promotion Gates

### Gate A: EXPLORATORY → VALIDATION_READY
Required to begin rigorous robustness testing.

| Threshold | Minimum Requirement | Rationale |
|---|---|---|
| Collection days | ≥ 30 calendar days | One full market-regime sample |
| Total events (full dataset) | ≥ 500 events | Enough for stratified analysis |
| Events per candidate rule | ≥ 150 complete 4h-outcome events | Power for win-rate confidence interval ±6% |
| Distinct symbols | ≥ 25 symbols | Diversification check |
| Max single-symbol contribution | ≤ 20% of events | No one coin dominates |
| Max single-day contribution | ≤ 20% of events | No single market event dominates |
| Positive median return (tp10_else_4h) | Yes | No negative-median candidates |
| Positive mean return at 50 bps fees | Yes | Must survive realistic costs |
| Both time-split medians positive | Yes | Temporal stability required |
| Excl top-3 symbols: positive mean | Yes | Symbol-robustness check |

### Gate B: VALIDATION_READY → PAPER_TRADING_READY
Required before any live (simulated) order placement.

| Threshold | Minimum Requirement |
|---|---|
| Collection days | ≥ 90 calendar days (3 months) |
| Events per rule | ≥ 500 complete 4h-outcome events |
| Distinct symbols | ≥ 50 symbols |
| Max single-symbol contribution | ≤ 10% |
| Max top-5% event contribution | ≤ 30% of total profit |
| Win rate (net of 75 bps RT cost) | ≥ 0.52 |
| Profit factor (gross wins / gross losses, net of costs) | ≥ 1.3 |
| Required time-split stability | All rolling-2-day windows positive median |
| Required symbol-split stability | Excl top-5 symbols: positive mean and median |
| Out-of-sample test window | ≥ 14 days held out, evaluated after rule finalized |
| Spread modeling | Actual spread data available; median spread ≤ 1.5% |
| Volume/liquidity | 90th percentile quote volume ≥ $10 000 at signal time |
| Benchmark comparison | Must outperform `baseline_volume_climax_tp10` |

### Gate C: PAPER_TRADING_READY → LIVE_READY
Not in scope for this task. Will require separate specification.

---

## Current Status vs. Gate A

| Threshold | Required | Current | Pass? |
|---|---|---|---|
| Collection days | ≥ 30 | 7 | ❌ |
| Total events | ≥ 500 | 1 451 | ✓ |
| Events per rule (rule 1) | ≥ 150 | 78 | ❌ |
| Distinct symbols (rule 1) | ≥ 25 | 16 | ❌ |
| Max single-symbol pct (rule 1) | ≤ 20% | 14.1% (DEGEN) | ✓ |
| Max single-day pct (full set) | ≤ 20% | 27.7% (Jun 1) | ❌ |
| Positive median (rule 1) | Yes | +1.27% | ✓ |
| Positive mean at 50 bps (rule 1) | Yes | +1.75% | ✓ |
| Both time-split medians positive | Yes | Yes | ✓ |
| Excl top-3 symbols: positive mean | Yes | +0.87% | ✓ |

**Result: 5/10 Gate A thresholds met. NOT VALIDATION_READY.**

---

## Notes

1. These thresholds apply to **all** candidate rules independently. A rule that
   passes on some metrics but fails any single hard threshold is not promoted.
2. Thresholds may only be tightened, not loosened, once locked.
3. Any new rule variants proposed after this date must also meet these thresholds
   before promotion. No rule may be back-tested to fit these thresholds.
4. The 90-day collection period for Gate B does not need to be continuous, but
   must cover at least two distinct 30-day windows and include both bull and
   sideways/bear market conditions.
5. Spread data requirement: the liquidity diagnostics CSV must show
   `spread_available = True` before Gate B is passed.
