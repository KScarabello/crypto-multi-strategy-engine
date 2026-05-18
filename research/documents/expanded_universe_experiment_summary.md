# Expanded Universe Crypto Momentum Experiment Summary

Date: 2026-05-17  
Repo: crypto-multi-strategy  
Experiment output: `research/results/expanded_universe_experiment_metrics.csv`

## Purpose

This experiment tested whether the crypto strategy research framework works on a larger 20-coin universe using multi-year 4h data.

The goal was not to find a final production strategy. The goal was to compare first-pass strategy families under shared data, execution, cost, and eligibility assumptions.

## Data

Universe: 20 Binance-backed crypto symbols normalized to `/USD` names:

- AAVE/USD
- ADA/USD
- APT/USD
- ARB/USD
- ATOM/USD
- AVAX/USD
- BCH/USD
- BTC/USD
- DOGE/USD
- DOT/USD
- ETH/USD
- INJ/USD
- LINK/USD
- LTC/USD
- NEAR/USD
- OP/USD
- POL/USD
- SOL/USD
- UNI/USD
- XRP/USD

Data source: Binance public spot kline archive using USDT quote pairs, normalized into local `/USD` symbol names for compatibility.

Timeframe: 4h OHLCV bars.

Date range: approximately 2020-01-01 through 2026-05-18, with newer coins entering only once data exists.

Known data caveat:

- Binance source data is missing one shared 4h candle at `2020-02-19 12:00 UTC` for 9 older symbols.
- The known gap is small and was left unfilled for honesty.

## Backtest Assumptions

All strategies used:

- Universal eligibility
- One-bar delayed execution
- Transaction costs
- No live trading behavior
- No forward-fill before a symbol exists

Universal eligibility means a symbol can only receive a signal/allocation once it has:

1. a valid current close price,
2. enough prior valid bars for the strategy lookback,
3. no missing required data in the lookback window,
4. naturally existing data by that timestamp.

## Strategy Families Tested

First-pass strategy families:

1. Time-series momentum
2. Time-series reversal
3. Time-series momentum with entry filter
4. Cross-sectional momentum

Stateful exit-signal variants were excluded from the default experiment because they need a more efficient incremental/stateful implementation before being included in the full grid.

Cross-sectional momentum plus reversal/overextension filter remains a TODO because it does not fit the current cross-sectional strategy API cleanly yet.

## Experiment Grid

Total runs: 126

Rebalance cadence:

- 6 bars
- 12 bars
- 42 bars

Lookback configurations:

- `short_42_only`
- `medium_180_only`
- `short_42_medium_180`
- `short_42_medium_180_long_360`

Cross-sectional top-N values:

- 3
- 5
- 8

Cost assumptions:

- 10 bps fee + 5 bps slippage
- 20 bps fee + 10 bps slippage

## Headline Results

### Best overall by Sharpe and CAGR

Strategy: cross-sectional momentum  
Rebalance: every 6 bars  
Lookback: `short_42_medium_180`  
Top N: 3  
Costs: 10 bps fee + 5 bps slippage

Metrics:

- Sharpe: 1.1257
- CAGR: 86.79%
- Total return: 52.85x
- Max drawdown: -84.64%
- Turnover: 825.00

This was the strongest headline result, but it had high turnover and very large drawdown.

### Best time-series momentum result

Strategy: time-series momentum  
Rebalance: every 42 bars  
Lookback: `medium_180_only`  
Costs: 10 bps fee + 5 bps slippage

Metrics:

- Sharpe: 1.0682
- CAGR: 71.44%
- Total return: 30.16x
- Max drawdown: -79.87%
- Turnover: 238.52

This is one of the more interesting practical candidates because it has strong performance with much lower turnover than the daily cross-sectional winner.

### Best entry-filter result

Strategy: time-series momentum with entry filter  
Rebalance: every 42 bars  
Lookback: `short_42_medium_180`  
Costs: 10 bps fee + 5 bps slippage

Metrics:

- Sharpe: 0.9308
- CAGR: 52.34%
- Total return: 13.67x
- Max drawdown: -74.73%
- Turnover: 340.93

This underperformed the best pure momentum results but had a somewhat better drawdown profile.

### Best reversal result

Strategy: time-series reversal  
Rebalance: every 42 bars  
Lookback: `short_42_only`  
Costs: 10 bps fee + 5 bps slippage

Metrics:

- Sharpe: 0.4397
- CAGR: 1.78%
- Total return: 0.12x
- Max drawdown: -90.05%
- Turnover: 392.29

Reversal alone did not look promising in this first-pass test.

## Main Interpretation

Momentum is the dominant signal in this experiment.

Cross-sectional momentum produced the best headline results, especially with combined short/medium lookbacks and smaller top-N portfolios.

Time-series momentum also produced strong results, especially with the 180-bar medium lookback and slower 42-bar rebalancing.

Reversal alone performed poorly and does not currently look like a standalone strategy candidate.

The 180-bar medium lookback appears more promising than the 42-bar-only lookback.

## Cost Sensitivity

The experiment included two cost assumptions.

Average results by cost bucket:

### 10 bps fee + 5 bps slippage

- Mean Sharpe: 0.6529
- Mean CAGR: 27.29%
- Max Sharpe: 1.1257
- Max CAGR: 86.79%

### 20 bps fee + 10 bps slippage

- Mean Sharpe: 0.4703
- Mean CAGR: 10.74%
- Max Sharpe: 1.0004
- Max CAGR: 62.07%

Costs matter significantly, especially for high-turnover daily strategies. However, the better-performing strategy families did not completely collapse under the higher-cost assumption.

## Lower-Turnover Candidates

The most interesting lower-turnover candidates were:

### Candidate 1

Strategy: time-series momentum  
Rebalance: 42 bars  
Lookback: `medium_180_only`  
Costs: 10 bps fee + 5 bps slippage

- Sharpe: 1.0682
- CAGR: 71.44%
- Max drawdown: -79.87%
- Turnover: 238.52

### Candidate 2

Strategy: cross-sectional momentum  
Rebalance: 42 bars  
Lookback: `short_42_medium_180`  
Top N: 5  
Costs: 10 bps fee + 5 bps slippage

- Sharpe: 1.0193
- CAGR: 66.40%
- Max drawdown: -79.63%
- Turnover: 251.40

### Candidate 3

Strategy: cross-sectional momentum  
Rebalance: 42 bars  
Lookback: `medium_180_only`  
Top N: 8  
Costs: 20 bps fee + 10 bps slippage

- Sharpe: 0.9671
- CAGR: 58.09%
- Max drawdown: -79.93%
- Turnover: 149.25

Candidate 3 is notable because it survived the higher-cost assumption while keeping turnover relatively low.

## Key Concern

The major unresolved problem is drawdown.

Top-performing strategies still experienced maximum drawdowns around -75% to -85%.

This is too severe for a practical live strategy unless the position sizing, volatility targeting, or risk filter substantially reduces crash exposure.

## Next Research Direction

The next experiment should focus on risk control, not more alpha variants.

Recommended next test:

Add a market regime or crash filter to the best candidate strategies.

Possible filters:

1. BTC moving-average regime filter
2. BTC medium-term momentum filter
3. Equal-weight universe moving-average filter
4. Equal-weight universe medium-term momentum filter
5. Volatility targeting or exposure scaling

Goal:

Reduce max drawdown meaningfully while preserving as much Sharpe and CAGR as possible.

Target next experiment candidates:

1. `cs_momentum`, rebalance 6, `short_42_medium_180`, top_n 3
2. `cs_momentum`, rebalance 12, `medium_180_only`, top_n 5
3. `cs_momentum`, rebalance 42, `medium_180_only`, top_n 8
4. `ts_momentum`, rebalance 42, `medium_180_only`
5. `ts_momentum_entry_filter`, rebalance 42, `short_42_medium_180`

## Current Conclusion

The expanded-universe experiment found a meaningful momentum signal across the 20-coin Binance-backed universe.

The most promising strategy families are:

1. Cross-sectional momentum
2. Time-series momentum

The least promising first-pass family is:

1. Time-series reversal alone

The next serious research question is whether risk controls can reduce drawdown enough to make the momentum signal practically usable.