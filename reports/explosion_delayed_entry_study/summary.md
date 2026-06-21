# Explosion Delayed-Entry Study (Research-Only)

## Scope
- Research only.
- No live trading, order execution, scheduler, launchd, broker, or production config changes.
- Decisions are made using the first 4h candle after an explosion only.
- All trade rows are delayed-entry diagnostics; they are not production signals.

## Setups Tested
- BUY_AT_EXPLOSION baseline.
- BUY_AFTER_FOLLOW_THROUGH.
- BUY_IF_NO_DEEP_PULLBACK.
- SHORT_AFTER_NO_FOLLOW_THROUGH, research-only.
- AVOID_NO_FOLLOW_THROUGH as a long-only exclusion filter.
- Pullback continuation / reversal buckets.

## Cost / Hold Grid
- Hold periods: 1, 2, 6 candles.
- Cost assumptions (bps): 0, 10, 25, 50, 100.
- Canonical ranking cost: 50 bps.

## Event Coverage
- Explosion events: 20211
- Tradable delayed rows: 185229
- Date range: 2020-01-03 08:00:00+00:00 -> 2026-06-07 20:00:00+00:00

## Signal Definitions
- Follow-through: first post-explosion close > explosion high.
- Deep pullback: first post-explosion low <= 2.00% below the explosion close.
- Pullback buckets are based on the first post-explosion low relative to the explosion close.

## Strategy Summary
- AVOID_NO_FOLLOW_THROUGH / 1c: n=7385, gross=+0.2900%, net@50bps=-0.2100%, win@50bps=0.417
- AVOID_NO_FOLLOW_THROUGH / 2c: n=7385, gross=+0.3539%, net@50bps=-0.1461%, win@50bps=0.452
- AVOID_NO_FOLLOW_THROUGH / 6c: n=7385, gross=+1.0315%, net@50bps=+0.5315%, win@50bps=0.482
- BUY_AFTER_FOLLOW_THROUGH / 1c: n=7385, gross=+0.2900%, net@50bps=-0.2100%, win@50bps=0.417
- BUY_AFTER_FOLLOW_THROUGH / 2c: n=7385, gross=+0.3539%, net@50bps=-0.1461%, win@50bps=0.452
- BUY_AFTER_FOLLOW_THROUGH / 6c: n=7385, gross=+1.0315%, net@50bps=+0.5315%, win@50bps=0.482
- BUY_AT_EXPLOSION / 1c: n=20211, gross=+0.3358%, net@50bps=-0.1642%, win@50bps=0.443
- BUY_AT_EXPLOSION / 2c: n=20211, gross=+0.6986%, net@50bps=+0.1986%, win@50bps=0.498
- BUY_AT_EXPLOSION / 6c: n=20211, gross=+0.8116%, net@50bps=+0.3116%, win@50bps=0.484
- BUY_IF_NO_DEEP_PULLBACK / 1c: n=13936, gross=+0.0867%, net@50bps=-0.4133%, win@50bps=0.382
- BUY_IF_NO_DEEP_PULLBACK / 2c: n=13936, gross=+0.0618%, net@50bps=-0.4382%, win@50bps=0.398
- BUY_IF_NO_DEEP_PULLBACK / 6c: n=13936, gross=+0.2890%, net@50bps=-0.2110%, win@50bps=0.431

## BTC Regime Buckets
- AVOID_NO_FOLLOW_THROUGH / 1c / DOWN: n=456, net@50bps=+1.5337%
- AVOID_NO_FOLLOW_THROUGH / 1c / SIDEWAYS: n=1056, net@50bps=+0.0463%
- AVOID_NO_FOLLOW_THROUGH / 1c / UP: n=5873, net@50bps=-0.3915%
- AVOID_NO_FOLLOW_THROUGH / 2c / DOWN: n=456, net@50bps=+2.1170%
- AVOID_NO_FOLLOW_THROUGH / 2c / SIDEWAYS: n=1056, net@50bps=+0.4124%
- AVOID_NO_FOLLOW_THROUGH / 2c / UP: n=5873, net@50bps=-0.4222%
- AVOID_NO_FOLLOW_THROUGH / 6c / DOWN: n=456, net@50bps=+5.2019%
- AVOID_NO_FOLLOW_THROUGH / 6c / SIDEWAYS: n=1056, net@50bps=-0.4553%
- AVOID_NO_FOLLOW_THROUGH / 6c / UP: n=5873, net@50bps=+0.3464%
- BUY_AFTER_FOLLOW_THROUGH / 1c / DOWN: n=456, net@50bps=+1.5337%
- BUY_AFTER_FOLLOW_THROUGH / 1c / SIDEWAYS: n=1056, net@50bps=+0.0463%
- BUY_AFTER_FOLLOW_THROUGH / 1c / UP: n=5873, net@50bps=-0.3915%

## Liquidity Buckets
- AVOID_NO_FOLLOW_THROUGH / 1c / LARGE: n=3093, net@50bps=-0.2420%
- AVOID_NO_FOLLOW_THROUGH / 1c / MID: n=2489, net@50bps=-0.2267%
- AVOID_NO_FOLLOW_THROUGH / 1c / SMALL: n=1803, net@50bps=-0.1322%
- AVOID_NO_FOLLOW_THROUGH / 2c / LARGE: n=3093, net@50bps=-0.2296%
- AVOID_NO_FOLLOW_THROUGH / 2c / MID: n=2489, net@50bps=-0.1220%
- AVOID_NO_FOLLOW_THROUGH / 2c / SMALL: n=1803, net@50bps=-0.0358%
- AVOID_NO_FOLLOW_THROUGH / 6c / LARGE: n=3093, net@50bps=+0.1847%
- AVOID_NO_FOLLOW_THROUGH / 6c / MID: n=2489, net@50bps=+0.6367%
- AVOID_NO_FOLLOW_THROUGH / 6c / SMALL: n=1803, net@50bps=+0.9813%
- BUY_AFTER_FOLLOW_THROUGH / 1c / LARGE: n=3093, net@50bps=-0.2420%
- BUY_AFTER_FOLLOW_THROUGH / 1c / MID: n=2489, net@50bps=-0.2267%
- BUY_AFTER_FOLLOW_THROUGH / 1c / SMALL: n=1803, net@50bps=-0.1322%

## Pullback Buckets
- SHALLOW: n=12860, follow-through=0.450, reversal=0.550, avg_pullback=-0.8577%
- MODERATE: n=4591, follow-through=0.128, reversal=0.872, avg_pullback=-3.2251%
- DEEP: n=1684, follow-through=0.062, reversal=0.938, avg_pullback=-7.2685%
- NO_PULLBACK: n=1076, follow-through=0.840, reversal=0.160, avg_pullback=+0.0196%

## Train / Test Robustness
- AVOID_NO_FOLLOW_THROUGH / 1c / test: n=2309, gross=-0.3308%, net@50bps=-0.8308%
- AVOID_NO_FOLLOW_THROUGH / 1c / train: n=5076, gross=+0.5724%, net@50bps=+0.0724%
- AVOID_NO_FOLLOW_THROUGH / 2c / test: n=2309, gross=-0.4594%, net@50bps=-0.9594%
- AVOID_NO_FOLLOW_THROUGH / 2c / train: n=5076, gross=+0.7239%, net@50bps=+0.2239%
- AVOID_NO_FOLLOW_THROUGH / 6c / test: n=2309, gross=+1.3185%, net@50bps=+0.8185%
- AVOID_NO_FOLLOW_THROUGH / 6c / train: n=5076, gross=+0.9010%, net@50bps=+0.4010%
- BUY_AFTER_FOLLOW_THROUGH / 1c / test: n=2309, gross=-0.3308%, net@50bps=-0.8308%
- BUY_AFTER_FOLLOW_THROUGH / 1c / train: n=5076, gross=+0.5724%, net@50bps=+0.0724%
- BUY_AFTER_FOLLOW_THROUGH / 2c / test: n=2309, gross=-0.4594%, net@50bps=-0.9594%
- BUY_AFTER_FOLLOW_THROUGH / 2c / train: n=5076, gross=+0.7239%, net@50bps=+0.2239%
- BUY_AFTER_FOLLOW_THROUGH / 6c / test: n=2309, gross=+1.3185%, net@50bps=+0.8185%
- BUY_AFTER_FOLLOW_THROUGH / 6c / train: n=5076, gross=+0.9010%, net@50bps=+0.4010%
- BUY_AT_EXPLOSION / 1c / test: n=5628, gross=+0.5907%, net@50bps=+0.0907%
- BUY_AT_EXPLOSION / 1c / train: n=14583, gross=+0.2374%, net@50bps=-0.2626%
- BUY_AT_EXPLOSION / 2c / test: n=5628, gross=+0.1537%, net@50bps=-0.3463%
- BUY_AT_EXPLOSION / 2c / train: n=14583, gross=+0.9088%, net@50bps=+0.4088%
- BUY_AT_EXPLOSION / 6c / test: n=5628, gross=-0.2107%, net@50bps=-0.7107%
- BUY_AT_EXPLOSION / 6c / train: n=14583, gross=+1.2061%, net@50bps=+0.7061%
- BUY_IF_NO_DEEP_PULLBACK / 1c / test: n=4690, gross=-0.5188%, net@50bps=-1.0188%
- BUY_IF_NO_DEEP_PULLBACK / 1c / train: n=9246, gross=+0.3938%, net@50bps=-0.1062%
- BUY_IF_NO_DEEP_PULLBACK / 2c / test: n=4690, gross=-0.6521%, net@50bps=-1.1521%
- BUY_IF_NO_DEEP_PULLBACK / 2c / train: n=9246, gross=+0.4239%, net@50bps=-0.0761%
- BUY_IF_NO_DEEP_PULLBACK / 6c / test: n=4690, gross=-0.1833%, net@50bps=-0.6833%
- BUY_IF_NO_DEEP_PULLBACK / 6c / train: n=9246, gross=+0.5285%, net@50bps=+0.0285%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 1c / test: n=3319, gross=+0.4994%, net@50bps=-0.0006%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 1c / train: n=9507, gross=-0.7350%, net@50bps=-1.2350%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 2c / test: n=3319, gross=+0.4502%, net@50bps=-0.0498%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 2c / train: n=9507, gross=-1.0483%, net@50bps=-1.5483%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 6c / test: n=3319, gross=+1.3361%, net@50bps=+0.8361%
- SHORT_AFTER_NO_FOLLOW_THROUGH / 6c / train: n=9507, gross=-0.9485%, net@50bps=-1.4485%

## Top Delayed-Entry Setups
- AVOID_NO_FOLLOW_THROUGH / 6c: n=7385, gross=+1.0315%, net@50bps=+0.5315%, win@50bps=0.482
- BUY_AFTER_FOLLOW_THROUGH / 6c: n=7385, gross=+1.0315%, net@50bps=+0.5315%, win@50bps=0.482
- BUY_AT_EXPLOSION / 6c: n=20211, gross=+0.8116%, net@50bps=+0.3116%, win@50bps=0.484
- BUY_AT_EXPLOSION / 2c: n=20211, gross=+0.6986%, net@50bps=+0.1986%, win@50bps=0.498
- AVOID_NO_FOLLOW_THROUGH / 2c: n=7385, gross=+0.3539%, net@50bps=-0.1461%, win@50bps=0.452
- BUY_AFTER_FOLLOW_THROUGH / 2c: n=7385, gross=+0.3539%, net@50bps=-0.1461%, win@50bps=0.452
- BUY_AT_EXPLOSION / 1c: n=20211, gross=+0.3358%, net@50bps=-0.1642%, win@50bps=0.443
- AVOID_NO_FOLLOW_THROUGH / 1c: n=7385, gross=+0.2900%, net@50bps=-0.2100%, win@50bps=0.417

## Safety Confirmation
- No live trading behavior changed.
- No broker/exchange execution logic changed.
- No scheduled jobs, launchd files, or production config changed.
