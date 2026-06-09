# Rank Churn Analysis — combo_entry2_imm_buf05

> SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.

## Part 1 — Trade Summary

| Metric | Value |
|---|---|
| Total rebalances | 1029 |
| Total trades generated | 995 |
| Total fee dollars | $405,708.48 |
| Total slippage dollars | $202,854.24 |
| Total direct costs | $608,562.72 |

## Part 2 — Rank-3/4 Boundary Statistics

| Metric | Value |
|---|---|
| total_rebalances | 1029 |
| total_boundary_swaps | 322 |
| pct_rebalances_with_swap | 31.2900 |
| mean_score_gap | 0.0188 |
| median_score_gap | 0.0124 |
| p5_score_gap | 0.0013 |
| p10_score_gap | 0.0023 |
| p25_score_gap | 0.0053 |
| p50_score_gap | 0.0124 |
| p75_score_gap | 0.0253 |
| p90_score_gap | 0.0427 |
| p95_score_gap | 0.0590 |
| direct_costs_boundary_swaps | 291704.4600 |
| pct_total_costs_from_boundary_swaps | 47.9300 |
| displaced_returns_within_1reb_pct | 48.4000 |
| displaced_returns_within_2reb_pct | 66.1000 |
| displaced_returns_within_3reb_pct | 78.3000 |

## Part 3 — Round-Trip Churn (7 and 30-day horizons)

| Horizon | Count | Total Cost | % Helped (avoided losses) |
|---|---|---|---|
| Within 7 days | 273 | $329,915.58 | 31.9% |
| Within 30 days | 430 | $520,179.12 | 37.2% |

## Part 4 — Holding Duration

Median holding duration: **4.0 days**

| Bucket | Count | Mean Return % | Total Fees |
|---|---|---|---|
| 14+ days | 60 | 39.75% | $46,720.09 |
| 2-7 days | 346 | -2.17% | $277,857.58 |
| 7-14 days | 92 | 9.59% | $80,779.28 |
| < 1 day | 1 | -3.53% | $745.42 |

## Part 5 — Score Stability and Replacement Quality

| Metric | Value |
|---|---|
| mean_score_autocorr | 0.5541 |
| median_score_autocorr | 0.5357 |
| mean_rank_autocorr | 0.4276 |
| median_rank_autocorr | 0.4351 |
| mean_score_advantage | 0.0468 |
| median_score_advantage | 0.0322 |
| p25_score_advantage | 0.0149 |
| p75_score_advantage | 0.0613 |
| pct_adv_lt_0.25pp | 2.3000 |
| pct_adv_lt_0.5pp | 6.2000 |
| pct_adv_lt_1.0pp | 14.9000 |
| pct_adv_lt_2.0pp | 32.4000 |
| pct_entrant_outperformed_7d | 53.8000 |

## Part 6 — Reweighting vs Replacement

Since this is a pure equal-weight strategy, PARTIAL_RETAINED and EQUAL_WEIGHT_NORMALIZATION
trades should be zero or near-zero. REGIME_TRANSITION and FULL_RANK_REPLACEMENT account
for all meaningful turnover.

| Trade Type | Count | Total Cost | % of Total |
|---|---|---|---|
| FULL_RANK_REPLACEMENT | 986 | $601,650.74 | 98.9% |
| REGIME_TRANSITION | 9 | $6,911.98 | 1.1% |

## Part 7 — Symbol Churn Leaders

| Symbol | Entries | Exits | Round Trips | % Turnover | Avg Hold (days) | Episode Win Rate |
|---|---|---|---|---|---|---|
| AVAX/USD | 100 | 100 | 99 | 21.5% | 5.7 | 38% |
| XRP/USD | 102 | 101 | 101 | 20.9% | 5.2 | 40% |
| BTC/USD | 104 | 103 | 103 | 20.4% | 6.9 | 53% |
| SOL/USD | 97 | 97 | 96 | 18.9% | 7.0 | 39% |
| ETH/USD | 96 | 95 | 95 | 18.3% | 7.6 | 58% |

## Genuine Rotation or Ranking Noise?

**Verdict: Ranking noise**

### Evidence

- Rank-3/4 boundary swaps occur in **31.3%** of rebalances
- Boundary swaps account for **47.9%** of total direct costs
- Mean score advantage of entrant over displaced: **0.0468** (raw return units)
- Median score advantage: **0.0322**
- % of replacements where entrant outperformed displaced in next 7 days: **53.8%**

### Interpretation

The data suggests **ranking noise** is a significant contributor to turnover costs.
Small score differences at the rank-3/4 boundary produce frequent costly swaps
without consistent return benefit. Consider a score-gap filter or hold-buffer rule.

---
*RESEARCH ONLY — SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.*
