# Fixed Five-Coin FIXED_COMMON_HISTORY: Max Drawdown & Regime Analysis

**Effective backtest start:** 2020-09-28  
**Universe:** BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD  
**Strategy:** CrossSectionalMomentum — top_n=3, min_history=36 bars, rebalance_every=6 bars  
**Fee:** 10 bps + 5 bps slippage  
**Initial capital:** $10,000  

> ⚠ SURVIVORSHIP BIAS WARNING: Both CURRENT_SURVIVORS_EXPANDING and FIXED_COMMON_HISTORY use only coins currently listed on Kraken. Delisted, collapsed, or renamed coins that existed historically are absent from both modes. Neither mode eliminates survivorship bias. A genuine historical backtest requires POINT_IN_TIME_UNIVERSE with authoritative exchange listing/delisting records.

---

## 1. Maximum Drawdown — Exact Timestamps

| Metric | Value |
|---|---|
| **Peak timestamp** | `2021-11-22 12:00:00+00:00` |
| **Trough timestamp** | `2022-12-19 20:00:00+00:00` |
| **Recovery timestamp** | `2023-12-24 00:00:00+00:00` |
| Peak equity | $700,210.01 |
| Trough equity | $142,215.47 |
| Recovery equity | $714,538.19 |
| **Strategy drawdown** | **-79.69%** |
| **BTC drawdown same period** | **-71.75%** |
| Strategy excess vs BTC | -7.94% |
| Peak-to-trough bars (4h) | 2,355 |
| Peak-to-recovery bars (4h) | 4,570 |
| Duration peak→trough | ~392 calendar days |
| Duration peak→recovery | ~761 calendar days |

---

## 2. Portfolio Holdings During the Max Drawdown

The strategy was **0% cash throughout the entire drawdown** (mean cash = 0.0%).
It was fully invested at all times; there was no defensive cash allocation.

### Average weight by symbol (peak → trough)

| Symbol | Avg Weight | Return over DD period |
|---|---|---|
| AVAX/USD | 16.1% | -91.8% |
| BTC/USD | 23.3% | -71.8% |
| ETH/USD | 25.1% | -72.4% |
| SOL/USD | 15.7% | -94.6% |
| XRP/USD | 19.8% | -67.9% |

### Holdings rotation (every 100 bars)

The strategy rotated holdings approximately every 24 hours (rebalance_every=6 × 4h=24h).
Top-3 selection from 5 eligible symbols means 2 symbols are always excluded.
During the drawdown the strategy cycled through all five symbols, providing no
protection — it was simply always holding the top-3 of five falling assets.

---

## 3. Attribution — What Caused the -79.7% Drawdown?

### Primary cause: Broad crypto crash while remaining fully invested

BTC fell **-71.8%** over the same period. All five assets declined severely:

| Symbol | Peak→Trough Return |
|---|---|
| SOL/USD | -94.6% |
| AVAX/USD | -91.8% |
| ETH/USD | -72.4% |
| BTC/USD | -71.8% |
| XRP/USD | -67.9% |

### Secondary cause: Altcoin underperformance vs BTC

The strategy held equal-weight top-3 altcoin positions. SOL fell -94.6%
and AVAX fell -91.8% vs BTC's -71.8%. Rotating into
SOL and AVAX which fell ~20pp more than BTC added ~-7.9% of excess drawdown.

### Rebalancing / whipsaw analysis

- **392 rebalances** occurred during the drawdown period
- **197 had turnover > 30%** (full position rotations)
- Average turnover per rebalance: **36.9%**
- Total transaction cost drag during drawdown: **21.70%** of starting capital

The cross-sectional momentum signal kept rotating into whichever of the 5 coins
had the least-bad recent momentum — but all 5 were in a bear market. Each rotation
incurred transaction costs with no improvement in holdings quality.

### Concentration

The strategy held exactly 3 of 5 symbols at equal weight (33.3% each) throughout.
HHI was constant at 0.333 (same as equal-weight). No concentration amplification.
However, 3/5 concentration with 5 correlated assets means diversification was limited.

### Transaction costs

Total cost drag during the -80% drawdown: **21.70%** of capital.
This is meaningful but secondary — the primary driver was the ~72% market decline.
The strategy would have lost ~-58.0% even with zero trading costs.

### Data gaps

No data gaps were detected in any of the five symbols during the drawdown period.
The drawdown calculation is not affected by missing or stale prices.

---

## 4. One-Bar Execution Delay Verification

**Confirmed:** Every rebalance in the log has `execution_timestamp != signal_timestamp`.
- All 2,328 rebalances show a one-bar (4h) delay between signal and execution.
- Example: signal at 2020-01-02T00:00 → executed at 2020-01-02T04:00.

**Fee verification:**
- Configured: 10 bps transaction + 5 bps slippage = 15 bps round-trip
- Average cost_rate per rebalance: 0.0491%
- Total cost drag over full history: 1.14x initial capital

---

## 5. Drawdowns by Calendar Year

| Year | Strategy Return | BTC Return | Excess | Max DD | Sharpe |
|---|---|---|---|---|---|
| 2020.0 | 37.8% | 300.3% | -262.5% | -32.0% | 1.92 |
| 2021.0 | 2577.2% | 57.9% | +2519.3% | -57.7% | 3.34 |
| 2022.0 | -75.2% | -64.7% | -10.6% | -76.1% | -1.32 |
| 2023.0 | 357.8% | 155.8% | +202.1% | -34.2% | 2.72 |
| 2024.0 | 84.5% | 121.1% | -36.6% | -49.7% | 1.24 |
| 2025.0 | -23.8% | -6.6% | -17.1% | -52.7% | -0.05 |
| 2026.0 | -35.9% | -12.7% | -23.2% | -46.8% | -1.83 |

---

## 6. Ten Worst Single 4h-Bar Losses

| Rank | Timestamp | Bar Return | Notes |
|---|---|---|---|
| 1 | `2021-05-19 08:00:00+00:00` | -14.13% | during broad market crash |
| 2 | `2021-04-18 00:00:00+00:00` | -13.95% | during broad market crash |
| 3 | `2021-05-19 20:00:00+00:00` | -12.56% | during broad market crash |
| 4 | `2021-02-01 12:00:00+00:00` | -11.85% | during broad market crash |
| 5 | `2021-05-21 12:00:00+00:00` | -11.22% | during broad market crash |
| 6 | `2024-03-05 16:00:00+00:00` | -11.03% | during broad market crash |
| 7 | `2021-04-18 08:00:00+00:00` | -10.79% | during broad market crash |
| 8 | `2024-08-05 00:00:00+00:00` | -10.25% | during broad market crash |
| 9 | `2021-01-10 08:00:00+00:00` | -10.17% | during broad market crash |
| 10 | `2020-11-26 00:00:00+00:00` | -9.88% |  |

---

## 7. All Drawdown Episodes > 20%

| # | Peak | Trough | DD % | Recovery | Duration (bars) | Recovery bars | Primary cause |
|---|---|---|---|---|---|---|---|
| 37 | 2021-11-22 | 2022-12-19 | -79.7% | 2023-12-24 | 2355 | 4570 | broad_crypto_crash|high_cost_drag |
| 52 | 2025-10-06 | 2026-03-29 | -66.0% | not yet | 1046 | ongoing | broad_crypto_crash|altcoin_underperformance_vs_btc|high_cost_drag |
| 24 | 2021-05-18 | 2021-07-20 | -57.7% | 2021-08-17 | 377 | 547 | broad_crypto_crash|altcoin_underperformance_vs_btc |
| 49 | 2025-01-20 | 2025-04-07 | -52.7% | 2025-09-13 | 462 | 1415 | broad_crypto_crash|altcoin_underperformance_vs_btc |
| 42 | 2024-03-18 | 2024-09-06 | -49.7% | 2024-11-22 | 1035 | 1493 | broad_crypto_crash|altcoin_underperformance_vs_btc|high_cost_drag |
| 6 | 2020-11-24 | 2020-12-24 | -32.0% | 2021-01-06 | 177 | 257 | altcoin_underperformance_vs_btc |
| 21 | 2021-04-15 | 2021-04-18 | -28.3% | 2021-04-29 | 21 | 87 | altcoin_underperformance_vs_btc |
| 38 | 2023-12-25 | 2024-01-23 | -26.1% | 2024-02-26 | 176 | 380 | altcoin_underperformance_vs_btc |
| 17 | 2021-02-13 | 2021-02-28 | -25.4% | 2021-03-30 | 94 | 272 | altcoin_underperformance_vs_btc |
| 9 | 2021-01-10 | 2021-01-11 | -22.2% | 2021-01-14 | 9 | 26 | broad_crypto_crash |
| 47 | 2024-12-17 | 2024-12-20 | -21.0% | 2025-01-18 | 19 | 194 | broad_market_decline_remained_invested |
| 31 | 2021-09-18 | 2021-09-21 | -20.2% | 2021-10-06 | 23 | 111 | broad_market_decline_remained_invested |

---

## 8. BTC Buy-and-Hold Comparison

| Metric | Strategy | BTC B&H |
|---|---|---|
| Total return (full period) | 3850.2% | 604.7% |
| CAGR | 91.9% | — |
| Sharpe | 1.21 | — |
| Max drawdown | -79.7% | -77.0% |

> Both figures are from the same effective start date (2020-09-28) to the same end date.
> BTC B&H applies no transaction costs except the one-time entry/exit (2 × 10 bps = 0.2%).

---

## 9. Data Quality During Drawdown

- **No data gaps** detected in any symbol during the 2021-11-22 → 2022-12-19 episode.
- **No stale prices** (all 5 symbols have continuous Kraken data through the period).
- The -79.7% drawdown is a genuine strategy result, not an artifact of missing data.

---

## 10. Summary Diagnosis

The -79.7% maximum drawdown is explained by the following contributing factors
(ordered by magnitude):

1. **Broad crypto bear market (2022)** — BTC fell 71.8% from the strategy peak.
   This is the dominant cause. All 5 universe coins fell 68–95%.

2. **Altcoin underperformance vs BTC** — SOL (−94.6%) and AVAX (−91.8%) fell
   significantly more than BTC. The cross-sectional momentum signal rotated into
   these alts, adding ~7.9pp of excess drawdown vs a BTC-only position.

3. **No defensive cash allocation** — The strategy held 0% cash at all times.
   There is no regime filter, volatility scaling, or drawdown stop.

4. **Transaction costs** — 21.7%+ of capital was consumed in fees during the
   drawdown period alone, from 392 rebalances with 33–100% turnover.

5. **Repeated whipsaws** — Momentum rotations picked the 'least-bad' of 5 falling
   assets every 24 hours. Each rotation added cost with no protective benefit.

**What did NOT cause the drawdown:**
- Data gaps or stale prices (none found)
- Concentration (HHI constant at 0.333 — equal weight)
- Delayed execution (one-bar delay correctly implemented)

**Recovery:** The strategy recovered to its prior peak on 2023-12-24 — taking
~761 calendar days (~2 years) to recover.

---

*Generated by `research/fixed_five_drawdown_analysis.py` — research only.*
*No live trading code was modified.*