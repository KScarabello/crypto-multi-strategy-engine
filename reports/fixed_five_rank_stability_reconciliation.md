# Rank Stability Study — Reconciliation Report

This document reconciles three confirmed bugs found in the original
`fixed_five_rank_stability.py` output.

---

## 1. Min-Hold Timing Reconciliation

### Finding: LABEL ERROR — Implementation Is Correct

The implementation correctly counts **scheduled portfolio rebalances**.
Each rebalance interval = 12 bars × 4 hours = **48 hours = 2 calendar days**.
The `hold_age` counter increments once per rebalance call.

| min_hold_rebs | Rebalance Interval | min_hold_hours | min_hold_days |
|---|---|---|---|
| 1 | 48h | 48h | 2 days |
| 2 | 48h | 96h | 4 days |
| 3 | 48h | 144h | 6 days |
| 4 | 48h | 192h | 8 days |
| 5 | 48h | 240h | 10 days |
| 6 | 48h | 288h | 12 days |

**min_hold_6 = 6 rebalances × 48 hours = 288 hours = 12 calendar days, NOT 3 days.**

The prior print summary contained a **label error**. The implementation was correct.
The label was wrong because it printed `min_hold_rebs × 12h` instead of
`min_hold_rebs × 48h`.

Concrete example: entry at 2022-01-10 00:00 UTC, min_hold_6 allows
earliest replacement at 2022-01-22 00:00 UTC (12 days later).

---

## 2. Direct-Cost Accounting Correction

### The Bug

The original `rank_stability.py` computed:

```python
gross_minus_net = gross_end - net_end
fee_dollars = gross_minus_net * (fee_bps / total_bps)   # WRONG
slippage_dollars = gross_minus_net * (slippage_bps / total_bps)  # WRONG
```

This allocates the gross-minus-net wealth gap by the fee:slippage ratio.
For the control: $1,414,160.94 × (10/15) = $942,773.96 ← reported as 'fees'
                 $1,414,160.94 × (5/15)  = $471,386.98 ← reported as 'slippage'

### The Fix

Actual costs must be computed per trade:

```python
fee_dollars = sum(abs(weight_change_i) × equity_before_i × fee_bps / 10_000)
slippage_dollars = sum(abs(weight_change_i) × equity_before_i × slippage_bps / 10_000)
```

### Corrected Figures (control — combo_entry2_imm_buf05 strict)

| Metric | WRONG (prior) | CORRECT (actual) |
|---|---|---|
| Fee dollars | $942,773.96 | $405,708.48 |
| Slippage dollars | $471,386.98 | $202,854.24 |
| Total direct cost | $1,414,160.94 | $608,562.72 |
| Gross-minus-net | $1,414,160.94 | $1,414,160.94 (same — not a bug) |
| Residual (foregone compounding) | N/A | $805,598.22 |

**The $942K/$471K figures were an allocation of gross-minus-net wealth by the
fee:slippage rate ratio. This is NOT actual fees paid. Actual fees are computed
from executed notional and equal $405,708.48.**

gross-minus-net ≠ fees paid. The residual beyond actual direct costs is foregone
compounding: wealth lost because each cost payment reduced the future compounding base.

---

## 3. Opportunity Cost Interpretation

### The Incomplete Metric

The original `pct_helped` (39.75%) means: **39.75% of suppressed replacements had the
incumbent outperform the challenger in raw return over the next rebalance.**

This is NOT the net economic benefit because it ignores:
1. The transaction cost avoided by suppressing the trade
2. The magnitude of the return difference vs the cost

### Corrected Analysis

net_economic_benefit_i = (incumbent_return − challenger_return) × position_notional + cost_avoided

| Metric | Value |
|---|---|
| Total suppressions | 5309 |
| pct_helped_before_costs (raw return only) | 39.93% |
| pct_beneficial_after_costs (net economic benefit > 0) | 66.3% |
| Mean raw return diff (% of position) | nan% |
| Mean net economic benefit (% of position) | nan% |
| Total cost avoided | $3,086,242.49 |
| Total return gain from beneficial suppressions | $10,973,592.56 |
| Total return loss from harmful suppressions | $nan |

When the challenger outperforms by less than the transaction costs avoided,
suppression is still net-beneficial.

---

## 4. Corrected Comparison Table (9 Variants)

| variant | sharpe | max_drawdown_pct | return_2022_pct | actual_fee_dollars | actual_direct_cost_dollars | n_rank_replacements | rank_stability_label_corrected |
|---|---|---|---|---|---|---|---|
| control | 1.79 | -45.92 | -38.07 | 405708.48 | 608562.72 | 363 | CONTROL |
| rank_buffer_4 | 1.76 | -46.66 | -39.58 | 188427.43 | 282641.15 | 215 | RANK_STABILITY_IMPROVEMENT |
| challenger_confirm_3 | 1.72 | -42.12 | -34.23 | 151136.67 | 226705.00 | 138 | RANK_STABILITY_IMPROVEMENT |
| score_hurdle_025 | 1.77 | -50.54 | -32.69 | 192294.93 | 288442.40 | 96 | RANK_STABILITY_IMPROVEMENT |
| score_hurdle_100 | 1.72 | -47.67 | -32.69 | 118453.12 | 177679.68 | 57 | RANK_STABILITY_IMPROVEMENT |
| min_hold_4 | 1.94 | -48.05 | -38.97 | 442044.17 | 663066.26 | 227 | MIXED_TRADEOFF |
| min_hold_6 | 1.94 | -45.33 | -35.45 | 398474.38 | 597711.57 | 174 | MIXED_TRADEOFF |
| combo_buf4_hurdle025 | 1.75 | -47.85 | -32.45 | 170283.45 | 255425.18 | 187 | RANK_STABILITY_IMPROVEMENT |
| combo_buf4_conf2_hold2 | 1.76 | -45.98 | -35.01 | 145105.87 | 217658.80 | 141 | RANK_STABILITY_IMPROVEMENT |

---

## 5. Revised RANK_STABILITY_IMPROVEMENT List

Criteria (using corrected actual direct costs):
1. Actual direct costs reduced by ≥ 25% vs control
2. Rank replacements reduced by ≥ 35%
3. Sharpe within 0.10 of control OR improved
4. Max drawdown not worsened by > 5pp
5. 2022 return not materially worse (< 3pp worse)
6. NOT achieved by staying predominantly in cash
7. Regime exit never delayed (structural guarantee)
8. One-bar execution delay preserved (structural guarantee)
9. No future information used (structural guarantee)

**RANK_STABILITY_IMPROVEMENT variants: rank_buffer_4, challenger_confirm_3, score_hurdle_025, score_hurdle_100, combo_buf4_hurdle025, combo_buf4_conf2_hold2**

---

## Notes

⚠ **SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.**

**In-sample disclaimer:** All results are in-sample. combo_entry2_imm_buf05 was
selected after observing the full historical sample and remains the frozen candidate.
No parameter changes in this reconciliation constitute out-of-sample validation.

**LIVE BEHAVIOR UNCHANGED:** No live trading files, configuration, cron, or exchange
code was modified by this reconciliation.
