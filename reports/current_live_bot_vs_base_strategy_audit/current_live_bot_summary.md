# Current Live Bot Summary

Read-only inventory and old-repo inspection indicate the current live bot is a cross-sectional momentum rotator, not a BTC time-series sleeve strategy.

## Apparent behavior
- Universe: BTC/USD, ETH/USD, XRP/USD, SOL/USD, AVAX/USD
- Ranking model: weighted short- and medium-lookback momentum
- Live config: 42-bar short lookback, 180-bar medium lookback, equal weights, top_n=3
- Timeframe: 4h bars
- Rebalance check: cron wakes every 4 hours, but the bot only acts on the configured 20:00 UTC rebalance bar
- Exposure: 75% gross exposure cap, 0.5 per-position cap, leaving roughly 25% cash when fully invested
- BTC regime filter exists in code, but is disabled in config_private.py, so BTC is not currently used as an active timing gate
- Execution is one-bar delayed through the pending-signal flow

## Relevant evidence
- Inventory summary: /tmp/current_live_bot_inventory/live_bot_strategy_summary.md
- Schedule summary: /tmp/current_live_bot_inventory/live_bot_schedule_summary.md
- Old repo config: /Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy/config_private.py
- Old repo target generation: /Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy/live/generate_targets.py

## Bottom line
The live bot is a cross-sectional, equal-weight momentum rotation with a cash cushion and optional BTC regime overlay that is currently disabled. It is materially different from the canonical base sleeve.