# Memecoin Catcher — Candidate Validation Report
Generated: 2026-06-09  |  Data: 2026-05-31 → 2026-06-06 (7 days)
Input: data/memecoin_backfilled_signal_events.csv (1 451 events)

---

## Rule 1: `LONG_EXPLOSION + volume_climax + NOT clean_continuation`  ★ Leading Rule

### Identity
- `ohlc_signal_type == LONG_EXPLOSION`
- `is_volume_climax == True`
- `is_clean_continuation == False`
- Exit: `tp10_else_4h` (take +10% if touched within 4 h, else exit at 4h return)

### Summary Stats (tp10_else_4h, n=78, 16 symbols, 6 days)
| Metric | Value |
|---|---|
| Event count | 78 |
| Symbol count | 16 |
| Date range | 2026-05-31 → 2026-06-05 |
| Mean return | +2.25% |
| Median return | +1.27% |
| Win rate (>0) | 0.615 |
| Fail rate (<-3%) | 0.192 |
| Fail rate (<-5%) | 0.141 |
| Worst event | -22.29% |
| Best event | +10.00% (capped) |
| MAE mean | -6.24% |
| MFE mean | +9.84% |
| TP10 hit rate | 34.6% |
| Mean excl single best | +2.14% |
| Median excl single best | +1.17% |
| Mean excl top 1% | -1.86% |
| Profit from top-1 event | 3.2% |
| Profit from top-3 events | 9.6% |
| Profit from top-5 events | 16.0% |
| Profit from top-10 events | 32.0% |

**Note on mean excl top 1%:** With n=78, top 1% = < 1 event, so this value reflects
trimming the very highest returner. Mean drops to -1.86% indicating the TP cap
concentrates value in a small number of events that hit TP10.

### Multi-Horizon Raw Returns (no TP cap)
| Horizon | n | Mean | Median | Win Rate |
|---|---|---|---|---|
| 1h | 78 | +0.23% | -0.21% | 0.474 |
| 4h | 78 | +0.62% | -0.26% | 0.487 |
| 24h | 73 | +0.48% | -4.14% | 0.397 |

**Takeaway:** Without TP, median 4h return is -0.26% (below zero). The tp10_else_4h
strategy works by capturing the large positive tail events (+10%), not by having a
positive median raw return. This is inherently a skewed-return strategy.

### Performance by Day
| Date | n | Mean | Median |
|---|---|---|---|
| 2026-05-31 | 3 | +0.53% | -3.31% |
| 2026-06-01 | 15 | +3.74% | +10.00% |
| 2026-06-02 | 15 | +1.56% | -0.08% |
| 2026-06-03 | 24 | +1.07% | +0.97% |
| 2026-06-04 | 15 | +2.14% | +0.44% |
| 2026-06-05 | 6 | +6.07% | +10.00% |

High daily variance. Jun 1 and Jun 5 are strong; Jun 31 and Jun 2 weak.

### Performance by Symbol (top contributors)
| Symbol | n | Mean | Median |
|---|---|---|---|
| BABY/USD | 5 | +10.00% | +10.00% |
| ESPORTS/USD | 9 | +5.79% | +10.00% |
| SLX/USD | 9 | +2.66% | +10.00% |
| DEGEN/USD | 11 | +2.45% | +1.17% |
| CCD/USD | 9 | +2.24% | +0.44% |
| ENA/USD | 8 | -0.96% | -0.54% |
| OPN/USD | 7 | -0.78% | -6.13% |

### Time-Split Stability
| Split | n | Mean | Median | Fail≤-3% |
|---|---|---|---|---|
| First half (days 0–2) | 37 | +2.45% | +1.84% | 0.189 |
| Second half (days 3–5) | 41 | +2.06% | +0.86% | 0.195 |

✓ Positive median in **both** halves.

### Fee/Slippage Stress (rule 1)
| Fee bps | Avg net | Median net | Win rate | Fail≤-3% |
|---|---|---|---|---|
| 0 | +2.25% | +1.27% | 0.615 | 0.192 |
| 25 | +2.00% | +1.02% | 0.590 | 0.192 |
| 50 | +1.75% | +0.77% | 0.551 | 0.205 |
| 100 | +1.25% | +0.27% | 0.513 | 0.231 |
| 150 | +0.75% | -0.23% | 0.474 | 0.244 |
| 200 | +0.25% | -0.73% | 0.436 | 0.282 |

Median turns negative at ~125 bps. Memecoins on Kraken typically carry
50–150 bps spread + 26 bps taker fee = 76–176 bps round-trip. **Edge is thin.**

---

## Rule 2: `volume_climax + safe hours (17–19 UTC) + tp10_else_4h`

| Metric | Value |
|---|---|
| Event count | 33 |
| Mean | +2.44% |
| Median | +0.44% |
| Win rate | 0.576 |
| Fail≤-3% | 0.212 |
| Mean excl best | +2.21% |
| Excl top-3 symbols | +0.56% |

Time-split: positive median both halves. Symbol-dependent: SLX (3 events, all TP10) 
drives 91% of contribution when included. Very small sample.

---

## Rule 3: `volume_ratio_4h ≥ 5 + safe hours + ret_24h < 25%`

| Metric | Value |
|---|---|
| Event count | 59 |
| Mean | +0.37% |
| Median | -0.49% |
| Win rate | 0.441 |
| Fail≤-3% | 0.136 |
| Mean excl best | +0.20% |
| Excl top-3 symbols | -0.35% |

⚠ Negative median overall. Positive median only in second half (not stable).
Turns negative at any fee level. **Not a viable candidate without improvement.**

---

## Baseline Comparisons

| Rule | n | Mean | Median | Win rate |
|---|---|---|---|---|
| All events, fixed_4h | 1 434 | -0.59% | -0.95% | 0.365 |
| All events, tp10 | 1 434 | -0.46% | -0.82% | 0.390 |
| volume_climax only, tp10 | 117 | +1.53% | +0.77% | 0.556 |
| safe_hours only, tp10 | 230 | +0.33% | -0.36% | 0.461 |

Rule 1 (+2.25% mean) materially outperforms volume_climax alone (+1.53%) and
all-event baselines. The CC=False filter adds value on top of volume_climax.

---

## Verdict

- **Rule 1 is the leading candidate.** Positive in both time splits, diversified across
  16 symbols, and positive through 100 bps fees.
- **Rule 2 is promising but heavily symbol-dependent** with only 33 events.
- **Rule 3 does not pass current bar** — median negative, fee-sensitive.
- **No rule is ready for paper trading.** The seven-day data window is insufficient.
