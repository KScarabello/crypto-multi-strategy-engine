# Fixed Five — Defensive Overlay & Turnover-Reduction Analysis

> **RESEARCH ONLY** — No live trading code was modified. No imports from `brokers/`, `execution/`, or `live/`. All changes are isolated to this research script.

> **SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.**

---

## 1. Cost Drag Definition

The **21.7% cost drag** figure cited in earlier analyses refers to the **sum of `cost_rate` values in the `rebalance_log` during the 2021-11-22 → 2022-12-19 drawdown period**. Each `cost_rate = turnover × (fee_bps + slippage_bps) / 10000`. The sum approximates total return lost to costs over that period (additive, not compounded). It is _not_ the difference between gross and net ending equity — it is the sum of per-rebalance cost fractions over a specific sub-period.

---

## 2. Gross vs Net Baseline

| Metric | Gross | Net | Difference |
|--------|-------|-----|------------|
| gross_total_return | 67.0374 | 19.7418 | 47.2957 |
| cagr | 1.1139 | 0.7123 | 0.4016 |
| sharpe | 1.3316 | 1.0705 | 0.2610 |
| max_drawdown | -0.7903 | -0.8342 | -0.0439 |
| ending_equity_$ | 680374.4195 | 207417.5020 | 472956.9175 |
| cost_pct_of_initial_cap | 4729.5692 | 4729.5692 | 0.0000 |
| cost_pct_of_gross_end | 69.5142 | 69.5142 | 0.0000 |
| pct_rebalances_zero_turnover | 91.1403 | 91.1403 | 0.0000 |
| drawdown_period_cost_drag_pct_2021_11_22_to_2022_12_19 | 23.3000 | 23.3000 | 0.0000 |

---

## 3. Variant Summary

| name | total_return | cagr | sharpe | max_drawdown | avg_annual_turnover | total_cost_drag_pct | pct_time_in_cash | return_2022 | robust_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1974.2% | 71.2% | 1.07 | -83.4% | 140.42 | 118.8% | 0.3% | -79.8% | BASELINE |
| abs_mom | 6723.1% | 111.5% | 1.35 | -75.7% | 201.02 | 170.0% | 26.5% | -59.4% | NOT_SELECTED |
| btc_ma_180 | 3702.1% | 90.7% | 1.42 | -59.5% | 93.77 | 79.3% | 48.7% | -51.8% | ROBUST_CANDIDATE |
| btc_ma_360 | 4047.6% | 93.6% | 1.44 | -53.5% | 85.55 | 72.4% | 48.3% | -47.4% | ROBUST_CANDIDATE |
| btc_mom_pos | 289.2% | 27.3% | 0.71 | -82.3% | 143.67 | 121.5% | 47.2% | -75.5% | NOT_SELECTED |
| reb_12 | 2538.1% | 78.7% | 1.12 | -81.6% | 97.26 | 82.3% | 0.3% | -78.3% | NOT_SELECTED |
| reb_18 | 2210.3% | 74.5% | 1.09 | -84.2% | 70.53 | 59.7% | 0.3% | -79.9% | NOT_SELECTED |
| reb_24 | 2928.9% | 83.1% | 1.15 | -84.7% | 57.88 | 49.0% | 0.4% | -80.2% | NOT_SELECTED |
| reb_42 | 1931.6% | 70.6% | 1.06 | -85.4% | 40.14 | 34.0% | 0.3% | -81.7% | NOT_SELECTED |
| turnover_buffer | 4534.9% | 97.5% | 1.25 | -82.6% | 62.49 | 52.8% | 0.3% | -76.0% | NOT_SELECTED |
| min_trade_2_5 | 1974.2% | 71.2% | 1.07 | -83.4% | 140.42 | 118.8% | 0.3% | -79.8% | NOT_SELECTED |
| min_trade_5 | 1974.2% | 71.2% | 1.07 | -83.4% | 140.42 | 118.8% | 0.3% | -79.8% | NOT_SELECTED |
| min_trade_10 | 1974.2% | 71.2% | 1.07 | -83.4% | 140.42 | 118.8% | 0.3% | -79.8% | NOT_SELECTED |
| abs_mom_reb12 | 1300.7% | 59.7% | 0.99 | -88.9% | 129.60 | 109.6% | 26.7% | -81.1% | NOT_SELECTED |
| abs_mom_reb24 | 2066.4% | 72.6% | 1.09 | -90.6% | 75.68 | 64.0% | 26.0% | -85.2% | NOT_SELECTED |
| btc_ma180_reb12 | 3828.0% | 91.8% | 1.43 | -55.8% | 65.92 | 55.8% | 48.9% | -51.5% | ROBUST_CANDIDATE |
| btc_ma180_reb24 | 3602.1% | 89.8% | 1.38 | -64.2% | 39.67 | 33.6% | 49.2% | -61.1% | ROBUST_CANDIDATE |
| abs_mom_buffer_reb12 | 1474.1% | 63.0% | 1.02 | -88.7% | 113.28 | 95.8% | 26.7% | -80.7% | NOT_SELECTED |
| btc_ma180_buffer_reb12 | 2867.9% | 82.5% | 1.16 | -81.7% | 45.94 | 38.9% | 0.3% | -73.5% | NOT_SELECTED |

---

## 4. Robust Candidates

A variant is **ROBUST_CANDIDATE** only if: (1) max drawdown improves by >5pp over baseline, (2) Sharpe ≥ baseline − 0.10, (3) reduces turnover or cost, (4) 2022 return better than baseline AND 2023 return > −30%, (5) <50% time in cash, (6) n_trades > 0.

Found **4 robust candidate(s)**:

- **btc_ma_180**: max_dd=-59.5%, sharpe=1.42, 2022=-51.8%, pct_cash=48.7%
- **btc_ma_360**: max_dd=-53.5%, sharpe=1.44, 2022=-47.4%, pct_cash=48.3%
- **btc_ma180_reb12**: max_dd=-55.8%, sharpe=1.43, 2022=-51.5%, pct_cash=48.9%
- **btc_ma180_reb24**: max_dd=-64.2%, sharpe=1.38, 2022=-61.1%, pct_cash=49.2%

---

## 5. Interpretation

✅ **At least one variant materially reduced the −79.7% max drawdown** (baseline max_dd=-83.4%).

**Best independent variant:** `btc_ma_360` — max_dd=-53.5%, sharpe=1.44, 2022=-47.4%.

**Best predefined combination:** `btc_ma180_reb12` — max_dd=-55.8%, sharpe=1.43, 2022=-51.5%.

**Persistence outside 2022:** Best independent (btc_ma_360): 2022 return -47.4%, 2023 return 228.8%.

---

## Disclaimer

**RESEARCH ONLY. No live trading code was modified.** This script imports only from `research/`, `backtest/`, `strategies/`, and `data/` modules. Observed improvements in 2022 may not persist in future bear markets of different character. The FIXED_COMMON_HISTORY universe retains survivorship bias from present-day symbol selection.