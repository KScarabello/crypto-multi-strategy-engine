# Final Research Selection Lock
Generated: 2026-06-09 02:34 UTC

## Frozen Regime Candidate
combo_entry2_imm_buf05 (BTC MA-240bars, rebalance every 12 bars,
entry_confirm=2, entry_buffer=0.5%, immediate exit)

## All Parameters Tested (Full In-Sample History)

### Regime gate variants (tested in whipsaw study):
- MA lengths: 180, 240, 300, 360, 420, 480 bars
- Rebalance: 6, 12, 24 bars
- Entry confirmation: 0, 2, 3 bars
- Exit confirmation: 0, 1, 2, 3 bars
- Entry buffer: 0%, 0.5%, 1.0%
- Exit buffer: 0%, 0.5%, 1.0%
- Min regime duration: 0, 3, 6, 12 bars
- 4 predefined combinations

### Rank stability variants (tested in rank stability study):
- rank_buffer: 4, 5 (or better)
- challenger_confirm_rebs: 2, 3
- score_hurdle: 0.025, 0.050, 0.100 raw score units
- min_hold_rebs: 2, 4, 6
- cooldown_rebs: 1, 2, 3
- 4 predefined combinations

## Final Rank-Stability Shortlist (In-Sample Only)
  1. min_hold_6 — Sharpe 1.9405, CAGR 165.59%, MaxDD -45.33%
  2. combo_buf4_conf2_hold2 — Sharpe 1.7596, CAGR 135.94%, MaxDD -45.98%
  3. rank_buffer_4 — Sharpe 1.757, CAGR 134.48%, MaxDD -46.66%

## Status Declarations

1. combo_entry2_imm_buf05 is the frozen regime candidate.
   Selected after observing the full available historical sample.
   NOT out-of-sample validated. NOT approved for live deployment.

2. All rank-stability variants on the shortlist are additional in-sample
   results. Any further parameter change is additional in-sample tuning.

3. The complete in-sample parameter sweep now includes regime gate rules,
   rank-stability rules, MA lengths, rebalance frequencies, and their
   combinations. No further parameter search can be considered unbiased.

4. Future validation must use:
   - Genuinely unseen data (newly acquired historical data)
   - Prospective shadow trading
   - OR live deployment with a separate validation budget

5. No shortlisted variant is approved for live capital deployment.

6. Baseline v1 and all live trading behavior remain unchanged.

## Evidence Summary

- Baseline canonical (no regime gate): Sharpe 1.21, MaxDD -79.7%, CAGR 91.9%
- Frozen candidate (combo_entry2_imm_buf05): Sharpe 1.794, MaxDD -45.9%, CAGR 141.8%
- Best shortlisted rank-stability variant: min_hold_6
- Improvement type: reduced transaction costs, broader score-gap filtering,
  or challenger confirmation, not parameter tuning of lookbacks or signals

SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.
