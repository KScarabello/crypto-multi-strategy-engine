# BTC-Only Trend Sleeve Research Specification (Plan Only)

## 1) Scope and Constraints

Objective: evaluate a BTC-only sleeve that decides between BTC and cash.

This document is a research plan only. It does not authorize implementation, tuning, live deployment, or strategy selection.

Hard constraints:
- Use the existing fixed BTC historical data source already used by current BTC research.
- Completed candles only.
- One-bar execution delay for all entries/exits and exposure changes.
- Realistic fees and slippage in both gross and net accounting.
- Cash is an explicit position (0% BTC exposure).
- No leverage.
- Identical benchmark date range across all BTC variants.
- Walk-forward reporting required.
- Parameter-neighborhood stability checks required.
- No live trading changes.
- Explicit comparison against the frozen five-coin sleeve.

## 2) Candidate Approaches (Predefined, Small Set)

The experiment compares only the following predefined approaches.

1. BTC buy-and-hold benchmark
- Rule: 100% BTC from first tradable bar; never exit.

2. BTC above medium-term moving average
- Rule: BTC exposure = 100% when close > SMA(240 bars), else 0%.
- Confirm: 2 completed bars above/below threshold before switching state.

3. BTC positive weighted time-series momentum
- Rule: exposure = 100% when weighted sum of lookback returns is positive, else 0%.
- Weighted lookbacks (bars): 24, 72, 168 with weights 0.5, 0.3, 0.2.
- Confirm: 1 completed bar state confirmation.

4. BTC trailing-price breakout
- Rule: exposure = 100% when close >= rolling max of prior N bars; else 0% when close <= rolling min of prior M bars.
- Predefined windows: N = 120, M = 120.
- Confirm: immediate on completed-bar signal, still executed with one-bar delay.

5. Volatility-targeted BTC trend exposure (feasibility variant)
- Base signal: approach 2 (SMA-240 gate).
- Target annualized volatility: 20% using realized vol over 72 bars.
- Exposure cap: [0%, 100%] (no leverage). If target implies >100%, clamp to 100%.
- If vol estimate unavailable, hold cash until estimate matures.

## 3) Fixed Trading Frictions and Accounting

Use one friction model shared by all variants:
- Fee: 0.40% per side.
- Slippage: 0.10% per side.
- Turnover computed from target exposure changes.

Required outputs per variant:
- Gross return/equity series.
- Net return/equity series.
- Cumulative fees, slippage, and turnover.

## 4) Walk-Forward Design (Predefined)

Use fixed windows (no search):
- Train window: 365 calendar days.
- Test window: 90 calendar days.
- Step: 90 calendar days.

Notes:
- Parameters above are frozen for the study and are not to be optimized.
- Walk-forward is for robustness reporting, not parameter selection.

## 5) Parameter-Neighborhood Stability Checks

Run local sensitivity checks only around fixed defaults:
- SMA length neighborhood: 216, 240, 264.
- Breakout neighborhood: 96, 120, 144.
- Momentum lookback neighborhood (centered proportionally):
  - Short: 20, 24, 28
  - Medium: 60, 72, 84
  - Long: 140, 168, 196
- Vol-target neighborhood: 15%, 20%, 25%.

Decision rule for interpretation:
- Stability is acceptable if ranking and qualitative conclusions do not invert from single-step neighborhood changes.
- This is a robustness screen, not an optimizer.

## 6) Comparison Against Existing Five-Coin Sleeve

Required comparative table dimensions:
- Period-matched net CAGR.
- Max drawdown.
- Net Sharpe (with maturity caveat if short sample).
- Trade count and turnover.
- Fee/slippage drag.
- Time in cash.

The five-coin comparator is the currently frozen five-coin sleeve configuration and benchmark reporting set; no changes to that sleeve are permitted.

## 7) Proposed Files (Future Implementation Targets)

Proposed (not yet created by this spec):
- `research/btc_only/sleeve_signals.py`
- `research/btc_only/sleeve_backtest.py`
- `research/btc_only/sleeve_walkforward.py`
- `research/btc_only/sleeve_stability.py`
- `tests/test_btc_only_sleeve_signals.py`
- `tests/test_btc_only_sleeve_backtest.py`
- `tests/test_btc_only_walkforward.py`
- `reports/btc_only_trend_sleeve_summary.csv`
- `reports/btc_only_trend_sleeve_walkforward.csv`
- `reports/btc_only_trend_sleeve_stability.csv`
- `reports/btc_only_vs_fixed_five_comparison.csv`

## 8) Required Test Coverage (Future)

Minimum tests to pass before any interpretation:
- Completed-candle enforcement and one-bar delay correctness.
- Cash-position transitions and no-leverage cap enforcement.
- Friction application parity (gross vs net math consistency).
- Walk-forward split boundaries and no look-ahead leakage.
- Stability-report generation with fixed neighborhoods.
- Benchmark date alignment with BTC buy-and-hold and fixed five-coin comparison.

## 9) Non-Goals

- No parameter grid search.
- No winner selection.
- No strategy retuning of frozen five-coin candidates.
- No live/deployment modifications.
