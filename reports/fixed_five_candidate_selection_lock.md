# Candidate Selection Lock

## Candidate

**Name:** combo_entry2_imm_buf05

**Definition:**
- Universe: FIXED_COMMON_HISTORY (BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD)
- Timeframe: 4-hour bars
- BTC MA length: 240 bars (≈ 40 calendar days)
- Portfolio rebalance frequency: every 12 bars (48 hours)
- Entry rule: BTC must remain at least 0.5% above the 240-bar MA for 2 completed
  consecutive bars before entering. Formally: for bars i and i-1 (completed before
  signal bar i), both must satisfy: BTC_close > MA_240 × 1.005
- Exit rule: Exit immediately (1 bar) when the completed-bar condition fails
- One-bar execution delay preserved: signal at bar i, execution at bar i+1

**Parameter source:** All parameters were defined before observing outcomes and
tested in the whipsaw control study. The combination was proposed as a structural
variant, not searched from a grid.

---

## Selection Basis

This candidate was selected based on the following observed full-sample results
relative to the btc_ma_240_reb12 control:

| Metric | Control | Candidate | Δ |
|---|---|---|---|
| Max drawdown | -48.1% | -45.9% | +2.2pp improvement |
| Sharpe | 1.8555 | 1.7940 | -0.06 (within -0.10 threshold) |
| 2022 return | -41.8% | -38.1% | +3.7pp improvement |
| 7-day whipsaw clusters | 45 | 25 | -44% |
| Additive cost drag | 50.0% | 49.8% | -0.2pp |
| Time in cash | 45.7% | 43.8% | -1.9pp |

All metrics are computed on the FIXED_COMMON_HISTORY dataset (2020-09-28 onward)
using the canonical Run A initialization.

---

## ⚠ CRITICAL LIMITATIONS

### 1. Selection was performed on full historical sample

This candidate was selected after observing the complete available price history
from 2020-09-28 to the present. It is NOT out-of-sample validated.

### 2. No further parameter adjustment may be described as out-of-sample

Any subsequent change to the MA length, entry confirmation bars, buffer percentage,
rebalance frequency, or any other parameter—even small adjustments—constitutes
additional in-sample tuning on this same dataset. It may not be described as
out-of-sample optimization.

### 3. Out-of-sample validation requires genuinely future data

Prospective validation must use data that was not available at the time of
candidate selection: either live trading results on genuinely future bars, or
newly acquired historical data that was not used during development.

### 4. Survivorship bias is present and unresolved

The universe (BTC, ETH, XRP, SOL, AVAX) was selected using present-day knowledge
of which coins are still trading and available on the exchange. This introduces
survivorship bias. Results would likely differ materially on a truly historical
universe including assets that later delisted or declined.

### 5. Candidate is NOT approved for live deployment

Selection-locking a candidate means only that no further tuning may claim
out-of-sample status. It does not constitute approval for live deployment.
Before live deployment would require (at minimum):
  - Genuinely out-of-sample prospective results
  - Live slippage and fee validation
  - Position sizing and risk analysis
  - Exchange-specific execution review
  - Regulatory and operational review

---

## Frozen Specification

The frozen specification is:

```python
WhipsawControl(
    name="combo_entry2_imm_buf05",
    entry_confirm_bars=2,
    exit_confirm_bars=1,
    entry_buffer_pct=0.5,
    exit_buffer_pct=0.0,
    min_duration_bars=0,
)
BTC_MA_BARS = 240
REBALANCE_BARS = 12
FEE_BPS = 10
SLIPPAGE_BPS = 5
```

Any research that modifies any of these values must be described as further
in-sample tuning, not out-of-sample testing.

---

## Confirmation

- Candidate locked by: research/fixed_five_cost_audit.py
- Date data generated: see report timestamps
- Live trading code unchanged: YES
- Baseline v1 parameters unchanged: YES
- Live configuration unchanged: YES
