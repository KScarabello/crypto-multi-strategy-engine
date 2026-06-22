# Current Live Bot vs Canonical Base Shadow Signal

## Answers
1. Why were stale-data warnings emitted? The old repo data/local files and the research repo data are older than the 8-hour freshness threshold for 4h logic.
2. Which repo/data source is stale? Current live bot repo data/local is stale; research repo data is also stale but fresher than the old repo.
3. Is research repo data fresh? YES.
4. Is old live bot repo data fresh? YES.
5. Can actual old bot targets be trusted right now? YES.
6. Can canonical base targets be trusted right now? YES.
7. If old bot data is stale, can we safely preview old bot logic using research data? YES, but the preview is also stale on current data and fresh.
8. What does the base strategy want right now? cash only with cash=1.0000 and BTC gate=OFF.
9. What does the old bot want right now, if safely computable? BTC/USD, ETH/USD, SOL/USD with cash=0.2500 from ACTUAL_OLD_BOT_TARGET. Research-data preview: BTC/USD, ETH/USD, SOL/USD with cash=0.2500.
10. Is this ready for repeated manual shadow runs? YES, because the tool now distinguishes actual old data, research-data preview, and canonical base data.
11. Is this ready for production adapter work? NO, because stale-data warnings remain.

## Snapshot Summary
- Current bot timestamp: 2026-06-22 00:00:00+00:00
- Base timestamp: 2026-06-22 00:00:00+00:00
- Current bot rebalance point: False
- Base rebalance point: False
- Current bot data fresh: True
- Base data fresh: True
- Current bot source: ACTUAL_OLD_BOT_TARGET @ /Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy/data/local
- Old research preview source: OLD_STRATEGY_ON_RESEARCH_DATA_PREVIEW @ /Users/kimscarabello/Repos/crypto/crypto-multi-strategy/data
- Canonical base source: CANONICAL_BASE_TARGET @ /Users/kimscarabello/Repos/crypto/crypto-multi-strategy/data
- Preferred old strategy preview source: OLD_STRATEGY_ON_RESEARCH_DATA_PREVIEW
- Old research preview weights: {"AVAX/USD": 0.0, "BTC/USD": 0.25, "ETH/USD": 0.25, "SOL/USD": 0.25, "XRP/USD": 0.0, "CASH": 0.0}
- Current bot symbols: BTC/USD, ETH/USD, SOL/USD
- Base symbols: none
- Current only symbols: BTC/USD,ETH/USD,SOL/USD
- Base only symbols: none
- Warnings: current bot snapshot is not on a rebalance bar | canonical base snapshot is not on a rebalance bar | canonical base BTC gate is off

## Manual Run
- `./.venv/bin/python research/current_vs_base_shadow_signal.py --current-data-dir /Users/kimscarabello/Desktop/Repos/crypto/crypto-momentum-strategy/data/local --base-data-dir /Users/kimscarabello/Repos/crypto/crypto-multi-strategy/data --report-dir /Users/kimscarabello/Repos/crypto/crypto-multi-strategy/reports/current_vs_base_shadow_signal`

## Safety
- No orders were placed.
- No live behavior changed.
- No broker or exchange execution path was called.
