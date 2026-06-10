# Shadow Validation Monitor Report

**Generated:** 2026-06-10T01:56:27.821305+00:00
**Prospective start:** 2026-06-09T00:00:00+00:00

## 1. Shadow Validation Status
- Bars processed (min/max across candidates): 6 / 6

> ⚠️ **IMMATURE DATA**: Only 6 observations. Need ≥30 for reliable statistics.

## 2. Per-Candidate Performance

| Candidate | Bars | Return% | MaxDD% | Sharpe | Rebalances | Fees+Slip$ | Regime On% |
|-----------|------|---------|--------|--------|------------|------------|------------|
| control | 6 | 0.00% | 0.00% | — | 0 | $0.00 | 0.0% |
| min_hold_6 | 6 | 0.00% | 0.00% | — | 0 | $0.00 | 0.0% |
| rank_buffer_4 | 6 | 0.00% | 0.00% | — | 0 | $0.00 | 0.0% |
| combo_buf4_conf2_hold2 | 6 | 0.00% | 0.00% | — | 0 | $0.00 | 0.0% |
| btc_buyhold | 6 | -1.66% | 3.28% | *-9.18* | 1 | $15.00 | 100.0% |
| ewb_buyhold | 6 | -1.28% | 3.35% | *-6.71* | 1 | $15.00 | 100.0% |

> *Sharpe values in italics are statistically immature (< 30 observations).*

## 3. Regime State Summary
- **control**: current gate=risk-OFF, transitions=0, rebalances=0
- **min_hold_6**: current gate=risk-OFF, transitions=0, rebalances=0
- **rank_buffer_4**: current gate=risk-OFF, transitions=0, rebalances=0
- **combo_buf4_conf2_hold2**: current gate=risk-OFF, transitions=0, rebalances=0

## 4. Candidate Delta vs Control

| Candidate | Return% | Delta vs Control |
|-----------|---------|-----------------|
| min_hold_6 | 0.00% | +0.00% |
| rank_buffer_4 | 0.00% | +0.00% |
| combo_buf4_conf2_hold2 | 0.00% | +0.00% |
| btc_buyhold | -1.66% | -1.66% |
| ewb_buyhold | -1.28% | -1.28% |

## 5. Benchmark Comparison
- BTC buy-and-hold: -1.66%
- EW5 buy-and-hold: -1.28%

## 6. Frozen Spec Hashes
- `control`: `f61c113027647b8c`
- `min_hold_6`: `19ecc97f4f1947fe`
- `rank_buffer_4`: `944d0f18a064ba64`
- `combo_buf4_conf2_hold2`: `a40bcfdec5c352b9`
- `btc_buyhold`: `14f3f97bc887e2ba`
- `ewb_buyhold`: `9651c1dfb3e47c9b`
