# Research Plan: Time-Series Momentum + Reversal Filter

## 1. Research Hypothesis

A crypto time-series momentum strategy may perform better when paired with a short-term reversal or overextension filter.

The core idea is to distinguish between:

- Emerging strength: an asset has positive medium-term momentum and may continue trending.
- Overextended strength: an asset has recently moved too far too fast and may be vulnerable to short-term reversal.

The hypothesis is that buying assets with strong 8-week momentum, while avoiding or reducing exposure to assets that are overextended on a 1-2 week basis, can improve risk-adjusted returns compared with time-series momentum alone.

## 2. Strategy Variants to Compare

### Variant A: Time-Series Momentum Only

Buy or hold an asset when its 8-week return is positive.

### Variant B: Short-Term Reversal Only

Buy or hold an asset when its recent 1-2 week return suggests short-term weakness or mean-reversion opportunity.

### Variant C: Time-Series Momentum + Reversal Entry Filter

Buy or hold an asset only when:

- 8-week return is positive
- 1-week return is not excessively overextended

### Variant D: Time-Series Momentum + Reversal Exit Signal

- Enter based on momentum.
- Exit existing holdings on trend break or strong overextension.

### Variant E: Time-Series Momentum + Entry Filter + Exit Signal

- Use overextension as an entry gate for new positions.
- Use a separate, higher overextension threshold for exits from existing positions.

## 3. Initial Signal Definitions

The framework uses trailing returns as reusable signal primitives.

- Momentum (8-week):

```text
momentum_8w = price_today / price_8_weeks_ago - 1
```

- Overextension (1-week):

```text
overextension_1w = price_today / price_1_week_ago - 1
```

- Optional overextension (2-week):

```text
overextension_2w = price_today / price_2_weeks_ago - 1
```

## 4. Implemented Research Framework

The modular implementation is now in place:

- Signals module: research/signals.py
- Strategy variants module: research/strategy_variants.py
- Runner script: research/run_ts_momentum_reversal_research.py

### Implemented Variant Rules

- Momentum only: hold assets where momentum is positive; equal weight.
- Short-term reversal only: hold assets with short-term pullback (negative recent return); equal weight.
- Momentum + entry filter: require momentum positive and overextension below entry threshold.
- Momentum + exit signal: enter on momentum; exit existing holdings when momentum turns negative or overextension crosses exit threshold.
- Momentum + entry filter + exit signal: separate entry and exit thresholds; existing holdings are managed independently from new entries.

## 5. Initial Research Assumptions (Not Optimized)

These are starter values for exploration and are not optimized final parameters.

- Timeframe: 4-hour bars
- Momentum window: 8 weeks (336 bars)
- Overextension window: 1 week (42 bars)
- Optional overextension window: 2 weeks (84 bars)
- Entry overextension threshold: 15%
- Exit overextension threshold: 30%

## 6. Universe and Expansion Plan

Initial universe:

- BTC/USD
- ETH/USD
- SOL/USD
- XRP/USD
- AVAX/USD

The research code is written so symbol count can expand (for example to around 20 assets) without changing strategy logic.
