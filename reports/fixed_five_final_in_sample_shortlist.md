# Final In-Sample Rank-Stability Shortlist
Generated: 2026-06-09 02:34 UTC

## IMPORTANT: These are IN-SAMPLE results only

All shortlisted variants were evaluated on the same historical data used to
select the frozen regime candidate (combo_entry2_imm_buf05). No variant on
this shortlist is approved for live capital deployment.

## Control Baseline
- Sharpe: 1.7937
- Normalized trading intensity (notional/TWAE): 339.949614

## Shortlisted Variants (≤ 3)

1. **min_hold_6** — Sharpe 1.9405, CAGR 165.59%, MaxDD -45.33%, Δ net equity $+1,576,711, Δ direct cost $+10,851
2. **combo_buf4_conf2_hold2** — Sharpe 1.7596, CAGR 135.94%, MaxDD -45.98%, Δ net equity $-405,968, Δ direct cost $+390,904
3. **rank_buffer_4** — Sharpe 1.757, CAGR 134.48%, MaxDD -46.66%, Δ net equity $-467,651, Δ direct cost $+325,922

## Shortlist Criteria Applied
1. Normalized trading intensity lower than control OR Sharpe ≥ control_sharpe
2. Sharpe ≥ control_sharpe - 0.10
3. Improvement in ≥ 2 of (2022, 2023, 2024, 2025) years
4. Actual direct costs lower than control (true_delta_direct_cost > 0)
5. Operationally simple rule
6. Max drawdown not worse than control + 5pp

## Status
- All results: IN-SAMPLE ONLY
- Frozen regime candidate: combo_entry2_imm_buf05 (unchanged)
- No shortlisted variant is approved for live capital deployment
- Future validation requires genuinely unseen data or prospective shadow trading
