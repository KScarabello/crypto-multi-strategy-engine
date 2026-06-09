# Suppression Counterfactual Reconciliation
Generated: 2026-06-09 02:34 UTC

## WARNING: Event-Level Sums Are Non-Additive


Event-level suppression diagnostic sums ($3.086M avoided costs, $10.974M return gains)
are NOT realizable portfolio savings. These figures are non-additive for three reasons:

1. Overlap: Multiple suppression events may be recorded at the same or adjacent signal
   timestamps. The portfolio cannot simultaneously realize both the cost saving and the
   return difference for overlapping events.

2. Path dependence: The return difference is estimated as a fraction of portfolio equity
   at the signal timestamp. But if the incumbent performs differently from the challenger,
   the portfolio's equity path changes, altering all subsequent position sizes.

3. Cross-variant double-counting: Suppressions from different variants (score_hurdle_025,
   score_hurdle_100, combo_buf4_hurdle025, etc.) may record the same replacement events
   under different names. Summing across variants double-counts the same economic event.

The ONLY reliable measure of economic benefit is:
  TRUE_DELTA_NET_EQUITY = variant_net_ending_equity - control_net_ending_equity

This is computed directly from the backtest equity curves and requires no decomposition.


## Prior Study Diagnostic Figures (FOR REFERENCE ONLY)

| Figure | Value | Status |
|--------|-------|--------|
| Avoided transaction costs | $3,086,242 | DIAGNOSTIC — non-additive, not realizable |
| Gains from beneficial suppressions | $10,973,593 | DIAGNOSTIC — non-additive, not realizable |

These figures were computed as event-level sums across potentially overlapping
suppression events. They CANNOT be added together and CANNOT be treated as
portfolio profit.

## True Realizable Differences (from equity curve comparison)

| Variant | Variant Net Equity | Control Net Equity | Δ Net Equity | Δ Direct Cost (saved) |
|---------|-------------------|-------------------|-------------|----------------------|
| control | $2,191,791 | $2,191,791 | +$0 | +$0 |
| rank_buffer_4 | $1,724,139 | $2,191,791 | $-467,651 | +$325,922 |
| challenger_confirm_3 | $1,595,839 | $2,191,791 | $-595,951 | +$381,858 |
| score_hurdle_025 | $2,428,113 | $2,191,791 | +$236,322 | +$320,120 |
| score_hurdle_100 | $1,945,522 | $2,191,791 | $-246,269 | +$430,883 |
| min_hold_4 | $3,473,587 | $2,191,791 | +$1,281,796 | $-54,504 |
| min_hold_6 | $3,768,502 | $2,191,791 | +$1,576,711 | +$10,851 |
| combo_buf4_hurdle025 | $1,724,540 | $2,191,791 | $-467,250 | +$353,138 |
| combo_buf4_conf2_hold2 | $1,785,823 | $2,191,791 | $-405,968 | +$390,904 |

Notes:
- Δ Net Equity > 0 means the variant finished with more money than the control
- Δ Direct Cost (saved) > 0 means the variant spent LESS on transaction costs
- These figures are computed directly from backtest equity curves
- No decomposition required; no double-counting possible

## Conclusion

The ONLY reliable economic comparison is the equity curve endpoint comparison shown
above. All further analysis of "avoided costs" or "suppression gains" is diagnostic
and must not be presented as realized portfolio benefit.
