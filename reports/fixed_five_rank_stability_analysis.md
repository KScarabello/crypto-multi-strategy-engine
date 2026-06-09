# Rank Stability and Turnover Control Study

## Score Unit Documentation

One raw score unit = weighted average of short and medium lookback returns (both 0.5 weight).
A score of 0.025 means the coin returned ~2.5% more than the alternative over the lookback
period. These are NOT percentage points; they are decimal return values.

## Control Configuration

- Universe: FIXED_COMMON_HISTORY (BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD)
- Joint start: 2020-09-28
- Frozen candidate: combo_entry2_imm_buf05
- BTC MA: 240 bars (≈40 calendar days), rebalance every 12 bars
- entry_confirm_bars=2, exit_confirm_bars=1, entry_buffer_pct=0.5
- TOP_N=3 equal-weight
- Execution delay: 1 bar (4 hours)
- Fees: 10 bps | Slippage: 5 bps
- ⚠ SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.

## Control Confirmation

- variant: control
- total_return_pct: 14436.58
- cagr_pct: 141.84
- sharpe: 1.7937
- sortino: 1.7663
- max_drawdown_pct: -45.92
- calmar: 3.0887
- worst_year_pct: -38.07
- worst_month_pct: -20.77
- return_2022_pct: -38.07
- avg_annual_turnover: 58.8636
- fee_dollars: 942773.96
- slippage_dollars: 471386.98
- gross_minus_net_dollars: 1414160.94
- n_rebalances: 1029
- n_rank_replacements: 363
- additive_cost_pct: 49.75
- pct_time_invested: 52.22
- pct_time_cash: 47.78
- n_regime_switches: 157
- description: combo_entry2_imm_buf05 strict top-3 replacement
- rank_buffer: 0
- challenger_confirm_rebs: 0
- score_hurdle: 0.0
- min_hold_rebs: 0
- cooldown_rebs: 0

## Full-Period Results Summary

| variant | total_return_pct | cagr_pct | sharpe | max_drawdown_pct | return_2022_pct | additive_cost_pct | n_rank_replacements | avg_annual_turnover | rank_stability_label |
|---|---|---|---|---|---|---|---|---|---|
| control | 14436.58 | 141.84 | 1.79 | -45.92 | -38.07 | 49.75 | 363 | 58.86 | CONTROL |
| rank_buffer_4 | 12111.29 | 134.48 | 1.76 | -46.66 | -39.58 | 30.95 | 215 | 36.62 | RANK_STABILITY_IMPROVEMENT |
| rank_buffer_5 | 5833.27 | 106.30 | 1.56 | -47.67 | -32.69 | 14.25 | 48 | 16.86 | UNFAVORABLE |
| challenger_confirm_2 | 5451.49 | 103.88 | 1.51 | -45.98 | -38.44 | 28.65 | 184 | 33.90 | UNFAVORABLE |
| challenger_confirm_3 | 11395.85 | 131.98 | 1.72 | -42.12 | -34.23 | 23.75 | 138 | 28.10 | RANK_STABILITY_IMPROVEMENT |
| score_hurdle_025 | 15517.41 | 144.94 | 1.77 | -50.54 | -32.69 | 19.95 | 96 | 23.60 | RANK_STABILITY_IMPROVEMENT |
| score_hurdle_050 | 13042.67 | 137.55 | 1.70 | -51.95 | -32.69 | 17.25 | 75 | 20.41 | UNFAVORABLE |
| score_hurdle_100 | 12413.43 | 135.50 | 1.72 | -47.67 | -32.69 | 15.25 | 57 | 18.04 | RANK_STABILITY_IMPROVEMENT |
| min_hold_2 | 12035.58 | 134.22 | 1.74 | -48.12 | -36.56 | 40.55 | 284 | 47.98 | MIXED_TRADEOFF |
| min_hold_4 | 22623.23 | 161.78 | 1.94 | -48.05 | -38.97 | 33.45 | 227 | 39.58 | RANK_STABILITY_IMPROVEMENT |
| min_hold_6 | 24552.48 | 165.59 | 1.94 | -45.33 | -35.45 | 27.75 | 174 | 32.83 | RANK_STABILITY_IMPROVEMENT |
| cooldown_1 | 14436.58 | 141.84 | 1.79 | -45.92 | -38.07 | 49.75 | 363 | 58.86 | WEAK_EVIDENCE |
| cooldown_2 | 11931.37 | 133.86 | 1.74 | -46.41 | -40.21 | 43.15 | 310 | 51.05 | MIXED_TRADEOFF |
| cooldown_3 | 9811.53 | 125.96 | 1.69 | -45.10 | -34.16 | 40.15 | 295 | 47.51 | WEAK_EVIDENCE |
| combo_buf4_conf2 | 9167.05 | 123.28 | 1.67 | -46.19 | -38.52 | 24.25 | 148 | 28.69 | MIXED_TRADEOFF |
| combo_buf4_hurdle025 | 12114.12 | 134.49 | 1.75 | -47.85 | -32.45 | 28.15 | 187 | 33.31 | RANK_STABILITY_IMPROVEMENT |
| combo_conf2_hurdle025 | 10677.48 | 129.34 | 1.67 | -51.25 | -32.69 | 18.35 | 87 | 21.71 | UNFAVORABLE |
| combo_buf4_conf2_hold2 | 12548.16 | 135.94 | 1.76 | -45.98 | -35.01 | 23.55 | 141 | 27.86 | RANK_STABILITY_IMPROVEMENT |

## RANK_STABILITY_IMPROVEMENT Candidates

rank_buffer_4, challenger_confirm_3, score_hurdle_025, score_hurdle_100, min_hold_4, min_hold_6, combo_buf4_hurdle025, combo_buf4_conf2_hold2

## Best Independent Rule: min_hold_6 (Sharpe=1.940)

## Best Predefined Combination: combo_buf4_conf2_hold2 (Sharpe=1.760)

## Opportunity Cost Analysis

- Total suppressions: 8330
- % suppressions that helped (incumbent outperformed): 39.75%
- Avg cost avoided per suppression: $578.7132
- Total cost avoided: $4,820,681.13
- Avg return gained/lost: nan%

## In-Sample Note

All results are in-sample. combo_entry2_imm_buf05 was selected from the full historical
sample and remains the frozen candidate. No parameter changes in this task constitute
out-of-sample validation.

## RANK_STABILITY_IMPROVEMENT Classification Criteria

All criteria must be true:
1. Direct costs reduced by >= 25%
2. Rank replacements reduced by >= 35%
3. Sharpe within 0.10 of control OR improved
4. Max drawdown not worsened by > 5pp
5. 2022 return not materially worse (< 3pp worse)
6. NOT achieved by staying permanently in cash
7. Regime exit never delayed (structural guarantee)
8. One-bar execution delay preserved (structural guarantee)
9. No future information used (structural guarantee)
