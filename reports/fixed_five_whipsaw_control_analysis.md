# Regime Whipsaw Reduction Study

## Control Configuration

- Universe: FIXED_COMMON_HISTORY (BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD)
- Joint start: 2020-09-28
- BTC MA: MA-240bars(≈40days)
- Rebalance: every 12 bars (48 hours)
- Execution delay: 1 bar (4 hours)
- Fees: 10 bps | Slippage: 5 bps
- ⚠ SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.

## Full-Period Results


| variant | total_return_pct | cagr_pct | sharpe | max_drawdown_pct | return_2022_pct | n_7day_whipsaw_clusters | additive_cost_drag_pct | pct_time_in_cash | n_regime_switches | whipsaw_label |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 18245.18 | 152.03 | 1.86 | -48.07 | -41.77 | 45 | 50.05 | 45.74 | 285 | CONTROL |
| entry_conf_2bar | 15689.38 | 145.41 | 1.81 | -48.07 | -41.77 | 39 | 50.15 | 46.51 | 215 | NOT_WHIPSAW_IMPROVEMENT |
| entry_conf_3bar | 19782.75 | 155.65 | 1.89 | -47.34 | -39.41 | 31 | 50.35 | 47.19 | 183 | NOT_WHIPSAW_IMPROVEMENT |
| exit_conf_2bar | 17720.54 | 150.74 | 1.84 | -57.47 | -52.30 | 32 | 50.35 | 45.25 | 178 | NOT_WHIPSAW_IMPROVEMENT |
| exit_conf_3bar | 17410.63 | 149.96 | 1.83 | -57.47 | -52.30 | 28 | 50.25 | 45.06 | 156 | NOT_WHIPSAW_IMPROVEMENT |
| sym_conf_2bar | 16404.66 | 147.35 | 1.82 | -57.47 | -52.30 | 27 | 50.45 | 45.64 | 156 | NOT_WHIPSAW_IMPROVEMENT |
| sym_conf_3bar | 16464.48 | 147.51 | 1.82 | -56.87 | -50.23 | 16 | 50.15 | 45.64 | 120 | NOT_WHIPSAW_IMPROVEMENT |
| hyst_05_05 | 16431.50 | 147.42 | 1.82 | -50.26 | -44.05 | 21 | 50.55 | 46.22 | 145 | NOT_WHIPSAW_IMPROVEMENT |
| hyst_10_05 | 13169.90 | 137.96 | 1.76 | -54.53 | -49.28 | 17 | 49.45 | 47.10 | 129 | NOT_WHIPSAW_IMPROVEMENT |
| hyst_10_10 | 13304.90 | 138.39 | 1.76 | -57.24 | -52.30 | 13 | 49.25 | 46.51 | 112 | NOT_WHIPSAW_IMPROVEMENT |
| min_dur_3bar | 19736.15 | 155.55 | 1.88 | -48.07 | -41.93 | 41 | 50.65 | 46.22 | 219 | NOT_WHIPSAW_IMPROVEMENT |
| min_dur_6bar | 20778.64 | 157.88 | 1.90 | -53.53 | -48.65 | 35 | 51.95 | 46.81 | 189 | NOT_WHIPSAW_IMPROVEMENT |
| min_dur_12bar | 14190.98 | 141.11 | 1.79 | -46.50 | -39.60 | 23 | 56.35 | 46.61 | 147 | NOT_WHIPSAW_IMPROVEMENT |
| combo_sym2_hyst05 | 15858.47 | 145.88 | 1.81 | -52.97 | -47.53 | 12 | 49.95 | 46.13 | 108 | NOT_WHIPSAW_IMPROVEMENT |
| combo_sym2_mindur6 | 14077.63 | 140.77 | 1.77 | -57.80 | -52.54 | 21 | 52.25 | 45.93 | 136 | NOT_WHIPSAW_IMPROVEMENT |
| combo_hyst05_mindur6 | 14657.80 | 142.49 | 1.79 | -56.23 | -50.76 | 18 | 51.35 | 46.32 | 139 | NOT_WHIPSAW_IMPROVEMENT |
| combo_entry2_imm_buf05 | 14436.58 | 141.84 | 1.79 | -45.92 | -38.07 | 25 | 49.75 | 47.78 | 157 | WHIPSAW_IMPROVEMENT |

## Cost Accounting Definitions

The following cost concepts are explicitly distinguished:

1. **fee_dollars_est** = gross_minus_net × fee_bps / (fee_bps + slippage_bps)
   The estimated dollar amount attributable to exchange fees.

2. **slippage_dollars_est** = gross_minus_net × slippage_bps / (fee_bps + slippage_bps)
   The estimated dollar amount attributable to modeled slippage.

3. **additive_cost_rate_sum_pct** = sum(turnover × total_bps / 10000) × 100
   Additive sum of per-rebalance cost rates. Does NOT include compounding effects.

4. **gross_minus_net_dollars** = gross_ending_equity − net_ending_equity
   Actual dollar wealth reduction from all costs, including foregone compounding.
   This is LARGER than the additive estimate because costs reduce the future compounding base.

5. **foregone_compounding_dollars** = gross_minus_net − (equity_at_start × additive_rate)
   Additional wealth lost because costs reduced the base for future returns.
   This is NOT fees or slippage directly — it is the compounding penalty.

Do not describe gross_minus_net as 'fees paid'. It includes slippage and compounding effects.

## WHIPSAW_IMPROVEMENT Criteria (Part 5)

A variant receives WHIPSAW_IMPROVEMENT only if ALL criteria are satisfied:

1. 7-day whipsaw clusters reduced by >= 25% vs control
2. Additive cost drag reduced
3. Sharpe within -0.10 of control (or better)
4. Max drawdown does not worsen by more than 5pp vs control
5. 2022 return does not worsen by more than 5pp vs control
6. Not predominantly in cash (time-in-cash < 80%)
7. One-bar execution delay preserved (structural guarantee)
8. No future information used (structural guarantee)

**Variants meeting WHIPSAW_IMPROVEMENT: 1 of 16**

## Entry/Exit Delay Analysis

Delays are measured in 4-hour bars relative to the raw MA crossing.
A delay > 0 means the controlled gate entered/exited later than the raw gate.

- **entry_conf_2bar**: avg entry delay=0.0 bars, avg exit delay=0.0 bars, avg missed BTC entry return=+0.00%, avg extra EWB exit return=+0.00%
- **entry_conf_3bar**: avg entry delay=0.0 bars, avg exit delay=0.0 bars, avg missed BTC entry return=+0.00%, avg extra EWB exit return=+0.00%
- **exit_conf_2bar**: avg entry delay=0.0 bars, avg exit delay=1.0 bars, avg missed BTC entry return=-0.03%, avg extra EWB exit return=-0.25%
- **exit_conf_3bar**: avg entry delay=0.0 bars, avg exit delay=2.0 bars, avg missed BTC entry return=-0.06%, avg extra EWB exit return=-0.82%
- **sym_conf_2bar**: avg entry delay=0.0 bars, avg exit delay=1.0 bars, avg missed BTC entry return=-0.03%, avg extra EWB exit return=-0.18%
- **sym_conf_3bar**: avg entry delay=0.0 bars, avg exit delay=2.0 bars, avg missed BTC entry return=-0.05%, avg extra EWB exit return=-0.75%
- **hyst_05_05**: avg entry delay=0.0 bars, avg exit delay=0.3 bars, avg missed BTC entry return=-0.01%, avg extra EWB exit return=-0.19%
- **hyst_10_05**: avg entry delay=0.0 bars, avg exit delay=0.3 bars, avg missed BTC entry return=-0.01%, avg extra EWB exit return=-0.22%
- **hyst_10_10**: avg entry delay=0.0 bars, avg exit delay=0.8 bars, avg missed BTC entry return=-0.02%, avg extra EWB exit return=-0.84%
- **min_dur_3bar**: avg entry delay=0.0 bars, avg exit delay=0.2 bars, avg missed BTC entry return=-0.01%, avg extra EWB exit return=-0.16%
- **min_dur_6bar**: avg entry delay=0.0 bars, avg exit delay=0.7 bars, avg missed BTC entry return=-0.02%, avg extra EWB exit return=-0.49%
- **min_dur_12bar**: avg entry delay=0.1 bars, avg exit delay=2.2 bars, avg missed BTC entry return=-0.07%, avg extra EWB exit return=-0.58%
- **combo_sym2_hyst05**: avg entry delay=0.0 bars, avg exit delay=1.4 bars, avg missed BTC entry return=-0.04%, avg extra EWB exit return=-0.68%
- **combo_sym2_mindur6**: avg entry delay=0.0 bars, avg exit delay=1.6 bars, avg missed BTC entry return=-0.05%, avg extra EWB exit return=-0.67%
- **combo_hyst05_mindur6**: avg entry delay=0.0 bars, avg exit delay=0.7 bars, avg missed BTC entry return=-0.03%, avg extra EWB exit return=-0.54%
- **combo_entry2_imm_buf05**: avg entry delay=0.0 bars, avg exit delay=0.0 bars, avg missed BTC entry return=+0.00%, avg extra EWB exit return=+0.00%

⚠ **Survivorship bias warning**: SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.
These results use a universe selected from present-day knowledge. No conclusions should be extrapolated to out-of-sample crypto universe performance.