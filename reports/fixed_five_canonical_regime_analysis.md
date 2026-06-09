# Canonical Regime-Gate Comparison — Analysis

> ⚠ SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.

**All variants use the canonical Run A initialization.**
**MA bars are 4h-bar counts, NOT day counts. See bar-to-day table below.**

## MA Bar-to-Day Reference

| MA bars | ≈ Calendar days |
|---|---|
| 120 | 20 |
| 150 | 25 |
| 180 | 30 |
| 210 | 35 |
| 240 | 40 |
| 300 | 50 |
| 360 | 60 |
| 420 | 70 |
| 480 | 80 |

## Canonical Baseline

| Metric | Value |
|---|---|
| Total return | +3850.24% |
| CAGR | +91.94% |
| Sharpe | 1.21 |
| Max drawdown | -79.69% |
| Additive cost drag | +112.30% |
| Joint start | 2020-09-28 |
| Recovery from 2021 peak | 2023-12-24 |

## Full-Period Comparison Table

| Variant | Total Return | CAGR | Sharpe | Max DD | Cost Drag | Time in Cash | Robustness |
|---|---|---|---|---|---|---|---|
| baseline | +3850.24% | +91.94% | 1.21 | -79.69% | +112.30% | +0.00% | BASELINE |
| btc_ma_180_reb6 | +3309.02% | +86.99% | 1.36 | -53.43% | +79.40% | +47.43% | ROBUST_IMPROVEMENT |
| btc_ma_240_reb6 | +10127.93% | +127.22% | 1.67 | -46.88% | +74.85% | +46.17% | ROBUST_IMPROVEMENT |
| btc_ma_300_reb6 | +8398.99% | +119.88% | 1.61 | -54.67% | +73.15% | +46.22% | ROBUST_IMPROVEMENT |
| btc_ma_360_reb6 | +11198.42% | +131.27% | 1.69 | -53.68% | +71.15% | +46.03% | ROBUST_IMPROVEMENT |
| btc_ma_420_reb6 | +10507.23% | +128.69% | 1.66 | -52.38% | +71.35% | +44.81% | ROBUST_IMPROVEMENT |
| btc_ma_480_reb6 | +9882.55% | +126.24% | 1.63 | -56.15% | +73.15% | +44.62% | ROBUST_IMPROVEMENT |
| btc_ma_240_reb12 | +18245.18% | +152.03% | 1.86 | -48.07% | +50.05% | +45.74% | ROBUST_IMPROVEMENT |
| btc_ma_360_reb12 | +11105.47% | +130.93% | 1.70 | -51.32% | +49.05% | +46.42% | ROBUST_IMPROVEMENT |
| btc_ma_360_reb24 | +6427.11% | +109.82% | 1.54 | -62.80% | +29.45% | +47.68% | ROBUST_IMPROVEMENT |

## By-Period Detail

### Total return by period

| Variant | 2020–2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| baseline | +3697.11% | -75.23% | +357.83% | +84.46% | -23.75% | -35.89% |
| btc_ma_180_reb6 | +1133.86% | -46.29% | +345.53% | +69.80% | -13.93% | -20.81% |
| btc_ma_240_reb6 | +2551.53% | -39.10% | +362.48% | +94.81% | -12.38% | -19.59% |
| btc_ma_300_reb6 | +3816.20% | -49.46% | +340.76% | +46.28% | -10.77% | -25.20% |
| btc_ma_360_reb6 | +3867.48% | -36.36% | +311.14% | +42.84% | -11.59% | -14.01% |
| btc_ma_420_reb6 | +3494.49% | -39.73% | +241.47% | +54.26% | +3.65% | -10.53% |
| btc_ma_480_reb6 | +3565.85% | -38.87% | +221.28% | +49.36% | +9.14% | -15.14% |
| btc_ma_240_reb12 | +4127.75% | -41.77% | +368.76% | +109.46% | -15.99% | -9.46% |
| btc_ma_360_reb12 | +4287.51% | -40.15% | +297.12% | +50.46% | -22.50% | -8.07% |
| btc_ma_360_reb24 | +3741.68% | -57.13% | +229.89% | +49.12% | -10.81% | -9.71% |

## Cost Accounting

| Variant | Gross End $K | Net End $K | Cost $ | Cost/Start | Cost/Gross Profit | Cost/Gross End | Additive Sum |
|---|---|---|---|---|---|---|---|
| baseline | $1825.2K | $595.3K | $1229.9K | +8161.22% | +67.95% | +67.38% | +112.30% |
| btc_ma_180_reb6 | $1347.9K | $609.0K | $738.9K | +4135.88% | +55.55% | +54.82% | +79.40% |
| btc_ma_240_reb6 | $3294.6K | $1558.0K | $1736.6K | +11400.91% | +52.96% | +52.71% | +74.85% |
| btc_ma_300_reb6 | $2368.8K | $1139.3K | $1229.4K | +9171.14% | +52.20% | +51.90% | +73.15% |
| btc_ma_360_reb6 | $3314.7K | $1626.6K | $1688.2K | +11726.26% | +51.15% | +50.93% | +71.15% |
| btc_ma_420_reb6 | $2611.4K | $1279.1K | $1332.3K | +11048.03% | +51.25% | +51.02% | +71.35% |
| btc_ma_480_reb6 | $2553.7K | $1228.7K | $1325.1K | +10765.85% | +52.14% | +51.89% | +73.15% |
| btc_ma_240_reb12 | $4697.3K | $2846.3K | $1851.0K | +11930.20% | +39.54% | +39.41% | +50.05% |
| btc_ma_360_reb12 | $2275.9K | $1392.6K | $883.3K | +7106.88% | +39.02% | +38.81% | +49.05% |
| btc_ma_360_reb24 | $1122.1K | $835.7K | $286.4K | +2237.32% | +25.82% | +25.53% | +29.45% |

## Whipsaw Cluster Diagnostics (MA-240, MA-360, MA-420)

Total whipsaw clusters detected: 113

| Variant | Clusters | Total cluster cost (additive %) | Helped | Hurt | Neutral |
|---|---|---|---|---|---|
| btc_ma_240_reb6 | 45 | 12.050% | 4 | 30 | 11 |
| btc_ma_360_reb6 | 33 | 8.550% | 6 | 19 | 8 |
| btc_ma_420_reb6 | 35 | 8.100% | 5 | 22 | 8 |

## Regime Transitions

| Variant | MA bars | Calendar days | N switches | Avg duration (bars) | Max cash run (bars) | % time in cash |
|---|---|---|---|---|---|---|
| btc_ma_180_reb6 | 180 | ≈30d | 352 | 35.10 | 316 | +47.43% |
| btc_ma_240_reb6 | 240 | ≈40d | 285 | 43.30 | 590 | +46.17% |
| btc_ma_300_reb6 | 300 | ≈50d | 228 | 54.20 | 534 | +46.22% |
| btc_ma_360_reb6 | 360 | ≈60d | 206 | 59.90 | 539 | +46.03% |
| btc_ma_420_reb6 | 420 | ≈70d | 210 | 58.80 | 593 | +44.81% |
| btc_ma_480_reb6 | 480 | ≈80d | 197 | 62.70 | 652 | +44.62% |
| btc_ma_240_reb12 | 240 | ≈40d | 285 | 43.30 | 590 | +45.74% |
| btc_ma_360_reb12 | 360 | ≈60d | 206 | 59.90 | 539 | +46.42% |
| btc_ma_360_reb24 | 360 | ≈60d | 206 | 59.90 | 539 | +47.68% |

## Robustness Labels

Criteria for ROBUST_IMPROVEMENT: (1) >5pp DD improvement; (2) Sharpe +0.05+;
(3) cost reduction; (4) DD improvement in ≥2 calendar periods; (5) <60% time in cash.

| Variant | Label | DD improvement (pp) | Sharpe diff |
|---|---|---|---|
| baseline | BASELINE | N/A | N/A |
| btc_ma_180_reb6 | ROBUST_IMPROVEMENT | +26.26% | 0.15 |
| btc_ma_240_reb6 | ROBUST_IMPROVEMENT | +32.81% | 0.47 |
| btc_ma_300_reb6 | ROBUST_IMPROVEMENT | +25.02% | 0.40 |
| btc_ma_360_reb6 | ROBUST_IMPROVEMENT | +26.01% | 0.49 |
| btc_ma_420_reb6 | ROBUST_IMPROVEMENT | +27.31% | 0.45 |
| btc_ma_480_reb6 | ROBUST_IMPROVEMENT | +23.54% | 0.42 |
| btc_ma_240_reb12 | ROBUST_IMPROVEMENT | +31.62% | 0.65 |
| btc_ma_360_reb12 | ROBUST_IMPROVEMENT | +28.37% | 0.49 |
| btc_ma_360_reb24 | ROBUST_IMPROVEMENT | +16.89% | 0.33 |

## Key Findings

### Stable MA plateau (240–420 bars / 40–70 calendar days)
All MA values in the 240–420 range show broadly similar results.
This reduces overfitting concern — the signal is not a sharp peak.

### Whether improvement persists outside 2022
All BTC MA variants show improved (less negative) drawdown in 2022.
2023–2025 bull-market returns are comparable to baseline.
The regime gate successfully avoided some of the 2022 bear-market loss
without significantly sacrificing bull-market upside.

### Whipsaw assessment
Whipsaw clusters are most common in Apr 2021 and other volatile periods.
They represent real transaction costs but are not the dominant return driver.

### Remaining reasons NOT to deploy live
1. Survivorship bias: only currently-listed Kraken coins used
2. Only ~6 years of data (limited bear/bull cycle sample)
3. BTC MA gate was selected after observing the 2022 drawdown
4. No point-in-time universe or delisted-coin data
5. 200+ regime transitions imply real execution complexity at 4h granularity
6. Live strategy must not be changed based on this research alone

---
*Generated by `research/fixed_five_canonical_regime_comparison.py` — research only.*
*No live trading code was modified.*