# Fixed Five — Baseline Metric Reconciliation

> **RESEARCH ONLY** — No live trading code touched.

> **SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.**

---

## Canonical Baseline

| Field | Value |
|-------|-------|
| start_timestamp | 2020-09-28 00:00:00 UTC (joint_start when all 5 symbols have >= 36 bars) |
| end_timestamp | last available bar in data/local/ |
| universe | BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD |
| timeframe | 4h |
| fee_bps | 10 |
| slippage_bps | 5 |
| execution_delay | 1 bar (4h) — signal at bar i, executed at bar i+1 |
| rebalance_schedule | every 6 bars (every 24h) |
| cost_accounting | turnover × (fee_bps + slippage_bps) / 10000 per rebalance, deducted at execution |
| drawdown_calculation | min(equity / equity.cummax() - 1) over portfolio sliced to >= joint_start |
| equity_construction | full backtest from data start; portfolio metrics from joint_start onward |
| benchmark_alignment | BTC B&H from joint_start to same end date, 2x fee applied once |
| max_drawdown | -79.7% (canonical figure) |
| bias_warning | SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records. |

---

## -79.7% vs -83.4% Explanation

### Run A — Canonical (-79.7% max DD)

- Backtest runs from **Jan 2020** using the **full close matrix**.
- At `joint_start` (2020-09-28), equity is already ≈ **$15,069**
  (9 months of investing BTC/ETH before AVAX/SOL eligible).
- **Nov 2021 peak**: ≈ $700,210; **Dec 2022 trough**: ≈ $142,215
- DD = 142,215 / 700,210 − 1 = **−79.7%**

### Run B — Overlay Script (-83.4% max DD)

- Backtest runs with `close_fixed` (sliced to `joint_start`), **fresh $10K capital**.
- Needs 36 bars of history before first trade → first trade ≈ Oct 4, 2020.
- **Nov 2021 peak**: ≈ $413,931; **Dec 2022 trough**: ≈ $68,648
- DD = 68,648 / 413,931 − 1 = **−83.4%**

### Root Cause

Different backtest start + history availability changes the Nov 2021 **absolute**
equity peak. Both runs share the **same peak/trough timestamps** and the **same
percentage market decline**. The fresh-start version (Run B) appears 3.7 pp worse
because it started with $10K instead of $15K and compounded through the entire
bull run from a lower base.

**Canonical baseline for all subsequent analysis: Run A (full history) = −79.7%.**

---

## MA Bar-to-Day Labels (4h bars)

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

> **Warning**: '≈ N days' means N calendar days assuming 24/7 continuous trading
> (4h bars × N bars ÷ 6 bars/day). Actual trading-day counts vary.

---

## Canonical Baseline Computed Metrics

| Metric | Value |
|--------|-------|
| Total Return | 3850.2% |
| CAGR | 91.9% |
| Sharpe | 1.21 |
| Max Drawdown | -79.7% |
| Return 2022 | -75.2% |
| Return 2023 | 357.8% |
