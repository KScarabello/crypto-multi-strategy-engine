# Locked Candidate Portfolio Backtest (Research-Only)

Frozen signal:
- BUY_AFTER_FOLLOW_THROUGH_STRICT
- HIGH_BREAK_100_BPS
- wait one 4h candle
- next-open entry
- hold 6 candles

## 1) Does the locked candidate survive as a portfolio?
- Best scenario LOCKED_MAXUNL_EX50_CD0_C50 total_return=+745.08% max_dd=-55.70%

## 2) Does it beat BTC buy-and-hold?
- BTC benchmark total_return=+757.56%

## 3) Does it beat BUY_AT_EXPLOSION portfolio baseline?
- BUY_AT_EXPLOSION baseline total_return=+1063.07%

## 4) Is edge destroyed by position limits?
- Position-limit scenarios evaluated: 3

## 5) Is edge destroyed by 100/150/200 bps costs?
- 50 bps: avg_total_return=+289.60%
- 100 bps: avg_total_return=+59.08%
- 150 bps: avg_total_return=-36.51%
- 200 bps: avg_total_return=-72.18%

## 6) Are drawdowns acceptable?
- Best max drawdown among scenarios: -32.70%
- Worst max drawdown among scenarios: -89.58%

## 7) Is performance concentrated in a few periods?
- Year rows: 7

## 8) Paper-trading readiness vs exploratory
- Preliminary verdict: candidate may be paper-trading candidate with caution.

## Safety Confirmation
- Research-only portfolio study.
- No live trading behavior changed.
- No scheduler, broker/exchange execution, credentials, launchd, allocation, or production config changes.
