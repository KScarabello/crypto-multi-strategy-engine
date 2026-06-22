# Canonical Base Strategy Summary

The canonical research base strategy is GATED_BTC_TS_INVESTED_75_25.

## Strategy definition
- Underlying BTC sleeve: btc_ts_60_240_12
- Cross-sectional sleeve: cs_180_top5_reb12
- Allocation: 75% BTC time-series sleeve, 25% cross-sectional sleeve
- Gate: the CS sleeve is active only when BTC_TS_INVESTED is on
- Timing: 4h bars
- Rebalance frequency: every 12 bars
- Gate lag in the walk-forward framework: 1 bar

## Reported performance
From the canonical report under reports/btc_base_portfolio_canonical/:
- Overlay 15 bps: total_return=+48.5494, CAGR=+0.8327, Sharpe=+1.5757, max_drawdown=-0.3749
- Overlay 30 bps: total_return=+44.4815, CAGR=+0.8085, Sharpe=+1.5459, max_drawdown=-0.3843

## Interpretation
This is a fixed research base sleeve with explicit BTC time-series timing and a BTC-dependent cross-sectional overlay. It is not a pure cross-sectional momentum rotator, and it is not the same design as the current live bot.