# Fixed Five — BTC Regime MA Robustness Analysis

> **RESEARCH ONLY** — No live trading code modified.

> **SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.**

---

## 1. MA Bar-to-Day Labels

| Bars | Calendar Days | Label |
|------|--------------|-------|
| 120 | 20 | MA-120bars(≈20days) |
| 150 | 25 | MA-150bars(≈25days) |
| 180 | 30 | MA-180bars(≈30days) |
| 210 | 35 | MA-210bars(≈35days) |
| 240 | 40 | MA-240bars(≈40days) |
| 300 | 50 | MA-300bars(≈50days) |
| 360 | 60 | MA-360bars(≈60days) |
| 420 | 70 | MA-420bars(≈70days) |
| 480 | 80 | MA-480bars(≈80days) |

---

## 2. Canonical Baseline Reconciliation

See `fixed_five_baseline_metric_reconciliation.md` for the full explanation.

**-79.7%** (Run A, canonical): backtest from Jan 2020, full history, equity at `joint_start`=2020-09-28 ≈ $15K.

**-83.4%** (Run B, overlay script): fresh $10K at `joint_start`, lower Nov 2021 peak absolute value.

**Going forward: use −79.7% as the canonical baseline max drawdown.**

---

## 3. Grid Results (reb=6, 9 MA windows)

| variant | ma_label | sharpe | max_dd | return_2022 | return_2023 | pct_cash | overfit_risk |
|---------|----------|--------|--------|-------------|-------------|----------|--------------|
| btc_ma_120_reb6 | MA-120bars(≈20days) | 1.08 | -65.4% | -53.7% | 251.9% | 46.6% | LOWER_OVERFIT_RISK |
| btc_ma_150_reb6 | MA-150bars(≈25days) | 1.27 | -60.1% | -50.7% | 399.4% | 47.0% | LOWER_OVERFIT_RISK |
| btc_ma_180_reb6 | MA-180bars(≈30days) | 1.36 | -53.4% | -46.3% | 345.5% | 47.4% | LOWER_OVERFIT_RISK |
| btc_ma_210_reb6 | MA-210bars(≈35days) | 1.59 | -50.4% | -30.5% | 348.9% | 47.0% | LOWER_OVERFIT_RISK |
| btc_ma_240_reb6 | MA-240bars(≈40days) | 1.67 | -46.9% | -39.1% | 362.5% | 46.2% | LOWER_OVERFIT_RISK |
| btc_ma_300_reb6 | MA-300bars(≈50days) | 1.61 | -54.7% | -49.5% | 340.8% | 46.2% | LOWER_OVERFIT_RISK |
| btc_ma_360_reb6 | MA-360bars(≈60days) | 1.69 | -53.7% | -36.4% | 311.1% | 46.0% | LOWER_OVERFIT_RISK |
| btc_ma_420_reb6 | MA-420bars(≈70days) | 1.66 | -52.4% | -39.7% | 241.5% | 44.8% | LOWER_OVERFIT_RISK |
| btc_ma_480_reb6 | MA-480bars(≈80days) | 1.63 | -56.1% | -38.9% | 221.3% | 44.6% | LOWER_OVERFIT_RISK |

---

## 4. Walk-Forward Results (Full Period)

| variant | total_return | sharpe | max_dd | pct_cash | helps_vs_baseline |
|---------|-------------|--------|--------|----------|------------------|
| baseline | 3850.2% | 1.21 | -79.7% | 0.0% | False |
| btc_ma_360_reb6 | 11198.4% | 1.69 | -53.7% | 46.0% | True |
| btc_ma_180_reb6 | 3309.0% | 1.36 | -53.4% | 47.4% | True |
| btc_ma_180_reb12 | 7007.8% | 1.58 | -57.6% | 46.7% | True |
| btc_ma_180_reb24 | 3359.8% | 1.36 | -68.6% | 47.5% | True |
| btc_ma_360_reb12 | 11105.5% | 1.70 | -51.3% | 46.4% | True |
| btc_ma_360_reb24 | 6427.1% | 1.54 | -62.8% | 47.7% | True |

---

## 5. Regime Event Analysis (MA-360)

- RISK_OFF exits: **102**
- RISK_ON entries: **103**
- Avg BTC return 30 bars after RISK_OFF exit: **-1.3%**

---

## 6. Overfit-Risk Classification

LOWER_OVERFIT_RISK variants: **btc_ma_120_reb6, btc_ma_150_reb6, btc_ma_180_reb6, btc_ma_210_reb6, btc_ma_240_reb6, btc_ma_300_reb6, btc_ma_360_reb6, btc_ma_420_reb6, btc_ma_480_reb6, btc_ma_120_reb12, btc_ma_150_reb12, btc_ma_180_reb12, btc_ma_210_reb12, btc_ma_240_reb12, btc_ma_300_reb12, btc_ma_360_reb12, btc_ma_420_reb12, btc_ma_480_reb12, btc_ma_150_reb24, btc_ma_210_reb24, btc_ma_240_reb24, btc_ma_300_reb24, btc_ma_360_reb24, btc_ma_420_reb24**

### Criteria (all must be true for LOWER_OVERFIT_RISK)

1. return_2022 > -60% (materially reduces 2022 loss)
2. return_2023 > -20% (preserves recovery)
3. return_2024 > -20% (preserves recovery)
4. max_drawdown in (-75%, -35%) — not too lucky, not too bad
5. pct_time_in_cash < 55%
6. avg_regime_duration_bars > 20 bars (≥5 calendar days per regime — no whipsawing)
7. max_dd within 10pp of adjacent MA neighbors (stable across windows)

---

## 7. Final Interpretation

- **Best MA by Sharpe**: btc_ma_240_reb12 (MA-240bars(≈40days), sharpe=1.86)

- **Best MA by max DD**: btc_ma_210_reb12 (MA-210bars(≈35days), max_dd=-46.1%)

- **Best MA by cost drag**: btc_ma_360_reb24 (cost_drag=29.45%)

- **Nearby MA windows perform similarly?** NO (max_dd range across reb=6 MA windows: 18.6%)

- **Improvement persists outside 2022?** YES

- **Lower-overfit-risk variants**: btc_ma_120_reb6, btc_ma_150_reb6, btc_ma_180_reb6, btc_ma_210_reb6, btc_ma_240_reb6, btc_ma_300_reb6, btc_ma_360_reb6, btc_ma_420_reb6, btc_ma_480_reb6, btc_ma_120_reb12, btc_ma_150_reb12, btc_ma_180_reb12, btc_ma_210_reb12, btc_ma_240_reb12, btc_ma_300_reb12, btc_ma_360_reb12, btc_ma_420_reb12, btc_ma_480_reb12, btc_ma_150_reb24, btc_ma_210_reb24, btc_ma_240_reb24, btc_ma_300_reb24, btc_ma_360_reb24, btc_ma_420_reb24

### Remaining reasons NOT to deploy live

1. Survivorship bias not eliminated (universe selected with present-day knowledge)
2. Only ≈ 6 years of data (2 full bear/bull cycles)
3. BTC regime gate is fitted on the same data used to evaluate it
4. No point-in-time universe or delisted-coin data
5. Kraken-specific liquidity; may not generalize to other exchanges

### Live behavior

**LIVE BEHAVIOR UNCHANGED ✓** — No live trading code was modified.
Signal generators and backtesting are research-only.

