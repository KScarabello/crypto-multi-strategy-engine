# Explosion Delayed-Entry Study - Second Pass (Research-Only)

## Scope
- Research only.
- No live trading, scheduler, launchd, broker/exchange execution, or production config changes.
- Uses 4h OHLCV only.
- Event-time filters are restricted to event-time-available data.

## Coverage
- Events: 3977
- Trades: 337920
- Date range: 2020-01-03 08:00:00+00:00 -> 2026-06-07 20:00:00+00:00
- Hold periods: 1, 2, 6
- Cost grid (bps): 10, 25, 50, 100

## Q1: Does stricter follow-through improve delayed-entry results?
- Best strict delayed rule: HIGH_BREAK_100_BPS / 6c, net@50bps=+0.9960%, net@100bps=+0.4960%

## Q2: Is immediate entry + exit-on-no-follow-through better than buy-all explosions?
- Best risk-exit: HIGH_BREAK_25_BPS / 6c net@50bps=+0.4910%
- Best baseline: BASELINE / 6c net@50bps=+0.5978%

## Q3: Is delayed confirmation better than immediate entry?
- Delayed best net@50bps=+0.9960% vs baseline best net@50bps=+0.5978%

## Q4: Can event-time filters improve baseline without look-ahead?
- Best event-time filter: EXCL_EXTREME_OVEREXTENSION / 6c net@50bps=+0.5300%
- Event-time filter rules use only BTC regime/returns, explosion-size bucket, volume-spike bucket, liquidity bucket, and volatility bucket.

## Q5: Deep pullback as avoid/exit vs dip-buy?
- Shallow-only delayed best net@50bps=+0.4311%
- Avoid-deep immediate-exit best net@50bps=+0.6118%

## Q6: Any setups positive after 50/100 bps?
- Positive at 50 bps: 39 rows
- Positive at 100 bps: 12 rows

## Q7: Any setups survive holdout/test period?
- Test rows with positive net@50bps: 9

## Q8: BTC regime / liquidity dependence
- Regime and liquidity subgroup averages were computed in generated CSV summaries.
- BTC regime grouped rows: 21
- Liquidity grouped rows: 21

## Q9: Duplicate or functionally equivalent variants
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_15PCT:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_20PCT:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_5PCT:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION:BASELINE:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:CLOSE_ABOVE_EXPLOSION_CLOSE_25_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:CLOSE_ABOVE_EXPLOSION_CLOSE_50_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:CLOSE_ABOVE_EXPLOSION_HIGH:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:HIGH_BREAK_0_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:HIGH_BREAK_100_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:HIGH_BREAK_25_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:HIGH_BREAK_50_BPS:1 (identical_event_entry_exit_return_signature)
- AVOID_DEEP_PULLBACK:IMMEDIATE_EXIT_THR_10PCT:1 == BUY_AT_EXPLOSION_EXIT_ON_NO_FOLLOW_THROUGH:VOL_ADJUSTED_BREAK:1 (identical_event_entry_exit_return_signature)

## Q10: Prototype readiness vs exploratory status
- Some holdout-positive rows exist, but require stability review before any prototype promotion.

## Top Setups
- BUY_AFTER_FOLLOW_THROUGH_STRICT / HIGH_BREAK_100_BPS / 6c / LONG: n=1688, net@50bps=+0.9960%, net@100bps=+0.4960%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / VOL_ADJUSTED_BREAK / 6c / LONG: n=1729, net@50bps=+0.8624%, net@100bps=+0.3624%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / HIGH_BREAK_50_BPS / 6c / LONG: n=2177, net@50bps=+0.8346%, net@100bps=+0.3346%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / CLOSE_ABOVE_EXPLOSION_HIGH / 6c / LONG: n=1295, net@50bps=+0.7800%, net@100bps=+0.2800%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / HIGH_BREAK_25_BPS / 6c / LONG: n=2435, net@50bps=+0.7730%, net@100bps=+0.2730%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / HIGH_BREAK_0_BPS / 6c / LONG: n=2726, net@50bps=+0.6455%, net@100bps=+0.1455%
- AVOID_DEEP_PULLBACK / IMMEDIATE_EXIT_THR_20PCT / 6c / LONG: n=3977, net@50bps=+0.6118%, net@100bps=+0.1118%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / CLOSE_ABOVE_EXPLOSION_CLOSE_50_BPS / 6c / LONG: n=1598, net@50bps=+0.6053%, net@100bps=+0.1053%
- BUY_AT_EXPLOSION / BASELINE / 6c / LONG: n=3977, net@50bps=+0.5978%, net@100bps=+0.0978%
- BUY_AFTER_FOLLOW_THROUGH_STRICT / CLOSE_ABOVE_EXPLOSION_CLOSE_25_BPS / 6c / LONG: n=1790, net@50bps=+0.5596%, net@100bps=+0.0596%

## Safety Confirmation
- SHORT_AFTER_NO_FOLLOW_THROUGH_STRICT remains research-only and not live-ready.
- No live trading behavior changed.
- No scheduler/launchd, broker/exchange, credentials, or production config changes were made.
