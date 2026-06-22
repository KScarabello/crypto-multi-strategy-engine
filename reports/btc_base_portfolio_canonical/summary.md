# BTC Base Portfolio Canonical Report (Research-Only)

## 1) What exact strategy was evaluated?
- Candidate: GATED_BTC_TS_INVESTED_75_25
- Underlying BTC sleeve: btc_ts_60_240_12
- Cross-sectional sleeve: cs_180_top5_reb12

## 2) What are its rules?
- 75% BTC TS sleeve, 25% CS sleeve
- CS sleeve active only when BTC_TS_INVESTED gate is ON
- 4h bars, rebalance_every_bars = 12
- Gate lag in walk-forward framework: 1 bar

## 3) What date range/data was used?
- Overlay 15 bps: 2020-01-01 04:00:00+00:00 -> 2026-06-10 00:00:00+00:00 (data_dir=data)
- Overlay 30 bps: 2020-01-01 04:00:00+00:00 -> 2026-06-10 00:00:00+00:00 (data_dir=data)

## 4) What costs/slippage assumptions were used?
- Base frictions: transaction_cost_bps = 10, slippage_bps = 5
- Overlay costs (walk-forward): 15 bps and 30 bps

## 5) What are total return, CAGR, Sharpe, max drawdown, worst year, and worst month?
- Overlay 15 bps: total_return=+48.5494, CAGR=+0.8327, Sharpe=+1.5757, max_drawdown=-0.3749, worst_year=-0.2360, worst_month=-0.2023
- Overlay 30 bps: total_return=+44.4815, CAGR=+0.8085, Sharpe=+1.5459, max_drawdown=-0.3843, worst_year=-0.2474, worst_month=-0.2041

## 6) How does it compare to BTC buy-and-hold?
- Overlay 15 bps: delta_total_return=+27.6318, delta_sharpe=+0.2191, delta_max_drawdown=+0.0349
- Overlay 30 bps: delta_total_return=+23.5639, delta_sharpe=+0.1893, delta_max_drawdown=+0.0256

## 7) How does it compare to BTC_TS_ONLY if available?
- Overlay 15 bps: delta_total_return=+27.6318, delta_sharpe=+0.2191, delta_max_drawdown=+0.0349
- Overlay 30 bps: delta_total_return=+23.5639, delta_sharpe=+0.1893, delta_max_drawdown=+0.0256

## 8) How does it compare to the cross-sectional sleeve alone if available?
- Overlay 15 bps: delta_total_return=+20.2737, delta_sharpe=+0.5405, delta_max_drawdown=+0.4285
- Overlay 30 bps: delta_total_return=+16.2057, delta_sharpe=+0.5107, delta_max_drawdown=+0.4192

## 9) How stable is it across train/test or walk-forward windows?
- Overlay 15 bps: windows=5, avg_test_return=+0.6161, avg_test_sharpe=+0.7179, avg_test_max_drawdown=-0.2382, beats_btc(return/sharpe/dd)=2/2/1
- Overlay 30 bps: windows=5, avg_test_return=+0.5962, avg_test_sharpe=+0.6720, avg_test_max_drawdown=-0.2434, beats_btc(return/sharpe/dd)=2/2/1

## 10) Should this be treated as the base portfolio for future blend testing?
- Yes. This is the fixed, auditable base candidate used in the walk-forward framework and is appropriate as the base sleeve before adding the locked explosion continuation candidate.

## Safety Confirmation
- Research-only reporting run.
- No live trading behavior, scheduler/launchd, broker/exchange execution, credentials, production config, or allocation logic was modified.