# Transaction-Cost and Turnover-Source Reconciliation

## Part 1 — Direct Cost Definitions

Costs in this report are computed from executed notional at the trade level:

- **fee_dollars** = executed_notional × fee_bps / 10,000
- **slippage_dollars** = executed_notional × slippage_bps / 10,000
- **total_direct_cost** = fee_dollars + slippage_dollars
- **executed_notional** = |weight_change| × portfolio_equity_before_trade

These are the modeled direct costs as the engine applies them.

### What gross-minus-net is NOT

The gross-minus-net ending-equity difference is not the fees paid.
It includes three components:
1. Actual direct fees (fee_bps × notional)
2. Actual modeled slippage (slippage_bps × notional)
3. **Foregone compounding and path effects**: every dollar paid in costs
   reduces the equity base. Future trades execute on a smaller base,
   compounding less wealth over time. This residual is NOT a fee.

### What additive cost-rate sum is NOT

additive_rate_sum × equity_at_start is a rough first-order estimate.
It is NOT actual cost. Actual costs compound because each cost reduces
the equity base for future growth. The actual dollar impact is many times
larger than this first-order estimate.

## Part 2 — Cost Reconciliation

### btc_ma_240_reb12

| Item | Value |
|---|---|
| Equity at start | $15,515.11 |
| Gross ending equity | $4,697,258.30 |
| Net ending equity | $2,846,274.20 |
| **Actual modeled fee dollars** | **$507,450.96** |
| **Actual modeled slippage dollars** | **$253,725.48** |
| **Total direct cost dollars** | **$761,176.44** |
| Gross-minus-net (NOT fees paid) | $1,850,984.10 |
| Residual foregone compounding | $1,089,807.66 |
| Direct cost / equity at start | 4906.03% |
| Direct cost / executed notional | 0.1500% |
| Direct cost / gross profit | 16.26% |
| Gross-minus-net / gross ending equity | 39.41% |
| Additive rate × start equity (REFERENCE ONLY) | $7,765.31 |
| (This is ~1.0% of actual direct cost — NOT the same thing) | |

### combo_entry2_imm_buf05

| Item | Value |
|---|---|
| Equity at start | $15,077.76 |
| Gross ending equity | $3,605,951.45 |
| Net ending equity | $2,191,790.51 |
| **Actual modeled fee dollars** | **$405,708.48** |
| **Actual modeled slippage dollars** | **$202,854.24** |
| **Total direct cost dollars** | **$608,562.72** |
| Gross-minus-net (NOT fees paid) | $1,414,160.94 |
| Residual foregone compounding | $805,598.22 |
| Direct cost / equity at start | 4036.16% |
| Direct cost / executed notional | 0.1500% |
| Direct cost / gross profit | 16.95% |
| Gross-minus-net / gross ending equity | 39.22% |
| Additive rate × start equity (REFERENCE ONLY) | $7,501.18 |
| (This is ~1.2% of actual direct cost — NOT the same thing) | |

## Part 3 — Why Whipsaw Reduction Barely Changed Total Cost

The combo_entry2_imm_buf05 candidate reduced 7-day whipsaw clusters by 44%
but reduced the additive cost measure only from 50.0% to 49.8% (−0.2pp).

### Explanation

Regime transitions (entries and exits) produce large, one-time turnover events.
However, the bulk of total turnover comes from regular cross-sectional rank
rebalances that occur every 12 bars regardless of regime state.

A whipsaw cluster is defined as 3+ transitions within 7 calendar days.
Eliminating 20 whipsaw clusters (45 → 25) removes the transitions that caused
unnecessary round-trips during noisy BTC-MA crossings.

But each whipsaw cluster typically produces 3–5 transitions, each with turnover
of ~1× (full portfolio exit then re-entry). The total notional from all clusters
is a small fraction of the total notional from hundreds of routine rebalances.

See the turnover-source decomposition CSV for exact numbers.

## Part 5 — Counterfactual Curves

### Path-Dependency Warning

The four counterfactual curves use identical signal logic and timing.
However, they are path-dependent because:

1. Each cost payment reduces current portfolio equity.
2. Future trades execute notional = weight × equity.
3. A lower equity base → smaller future notional → lower absolute future returns.

Therefore, the no-cost curve is not simply the fees-and-slippage curve shifted
upward by a constant. The differences compound over the full period.
This is the same mechanism that makes the residual_foregone_compounding figure
larger than zero.

### btc_ma_240_reb12

| Scenario | Net End Eq | Total Return | Sharpe | Max DD | 2022 |
|---|---|---|---|---|---|
| no_fees_no_slippage | $4,787,553 | 30175.4% | 2.0057 | -45.3% | -38.4% |
| fees_only | $3,385,232 | 21580.8% | 1.9056 | -47.2% | -40.7% |
| slippage_only | $4,025,941 | 25521.1% | 1.9557 | -46.3% | -39.5% |
| fees_and_slippage | $2,846,274 | 18245.2% | 1.8555 | -48.1% | -41.8% |

### combo_entry2_imm_buf05

| Scenario | Net End Eq | Total Return | Sharpe | Max DD | 2022 |
|---|---|---|---|---|---|
| no_fees_no_slippage | $3,686,166 | 23815.7% | 1.9440 | -44.7% | -34.7% |
| fees_only | $2,606,697 | 17062.0% | 1.8439 | -45.5% | -37.0% |
| slippage_only | $3,099,910 | 20160.1% | 1.8939 | -45.1% | -35.8% |
| fees_and_slippage | $2,191,791 | 14436.6% | 1.7937 | -45.9% | -38.1% |

⚠ SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.