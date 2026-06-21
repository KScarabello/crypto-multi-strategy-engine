# Locked Candidate Validation (Research-Only)

Frozen candidate:
- BUY_AFTER_FOLLOW_THROUGH_STRICT
- HIGH_BREAK_100_BPS
- one-candle observation delay
- hold 6 candles

## 1) Does the locked candidate survive entry-at-next-open execution?
- Observation-close net@50: +0.9960%; next-open net@50: +1.0659%.

## 2) Does it remain positive after 100/150/200 bps?
- net@100: +0.5659%, net@150: +0.0659%, net@200: -0.4341%.

## 3) Is it stable year by year?
- Year rows: 7

## 4) Is it dominated by one year, one coin, or one regime?
- Symbol rows: 25
- BTC regime rows: 5

## 5) Does it survive excluding top contributors?
- Excluding top 0: net@50=+1.0659%, net@100=+0.5659%.
- Excluding top 1: net@50=+0.8646%, net@100=+0.3646%.
- Excluding top 3: net@50=+0.6885%, net@100=+0.1885%.
- Excluding top 5: net@50=+0.5633%, net@100=+0.0633%.

## 6) Does it work only during favorable BTC regimes?
- See btc_regime_summary.csv (rows=5).

## 7) Does it beat random/placebo entries?
- net_return_pct_50: candidate-placebo=+1.3742%, p(placebo>=candidate)=0.000.
- net_return_pct_100: candidate-placebo=+1.3742%, p(placebo>=candidate)=0.000.

## 8) Does it beat BUY_AT_EXPLOSION after realistic costs?
- Candidate net@50=+1.0659% vs baseline net@50=+0.5978%.

## 9) Prototype-ready or still exploratory?
- Preliminary verdict: candidate is stronger but still needs further out-of-sample monitoring.

## Safety Confirmation
- Research-only validation pass.
- No live trading behavior changed.
- No scheduler/launchd, broker/exchange, credentials, allocation, or production config changes.
