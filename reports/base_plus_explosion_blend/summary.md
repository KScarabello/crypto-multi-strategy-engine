# Base Plus Locked Explosion Blend (Research-Only)

Fixed inputs:
- Base: GATED_BTC_TS_INVESTED_75_25 (btc_ts_60_240_12, BTC_TS_INVESTED gate, 4h, rebalance 12, gate lag 1)
- Explosion: BUY_AFTER_FOLLOW_THROUGH_STRICT + HIGH_BREAK_100_BPS, delayed next-open entry, hold 6 bars, long-only
- No signal optimization and no new candidates

Top blend by Sharpe:
- base_cost15_exp_cost50_proportional_base80_exp20 | total_return=+32.2603, cagr=+0.7227, sharpe=+1.6279, max_drawdown=-0.3343

Key questions:
1. Total return improvement vs base-only baseline? NO
2. CAGR improvement? NO
3. Sharpe improvement? YES
4. Max drawdown improved or stable? YES
5. 2022-2026 improved? NO
6. Diversification signal (mean base/explosion correlation): +0.2183
7. Avg 2020-2021 return across blends: +8.4755
   Avg 2022-2026 return across blends: +2.7677
8. Best strict-improvement allocation: base_cost15_exp_cost50_cash_overlay_base95_exp05
9-13. Recommendation: consider as small tactical sleeve in paper-trading research only; not live.

Decision rules applied:
- No recommendation when return gains required materially worse drawdown.
- No recommendation when gains were confined to 2020-2021 and degraded 2022-2026.

Safety confirmation:
- Research-only analysis.
- No live trading behavior, scheduler/launchd, broker/exchange execution, credentials, production config, or production allocation logic changed.