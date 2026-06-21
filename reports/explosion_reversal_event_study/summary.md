# Explosion Reversal Event Study (Research-Only)

## Scope
- Research only.
- No live trading or scheduling changes.
- Event-time features use only information available at detection time.
- Follow-through and pullback traits are explicitly post-event diagnostics.

## Explosion Definition (Configurable)
- ret_4h_pct >= 8.00 OR ret_zscore >= 2.50
- min dollar volume >= 75000
- cooldown: 3 bars (4h bars)

## Data Availability Notes
- OHLCV resolution in local files is 4h.
- 15m/1h features and outcomes are unavailable from current OHLCV resolution and are reported as NaN.
- Unavailable from current data: future_ret_15m_pct, future_ret_1h_pct, btc_ret_1h_pct

## Overall Event Counts
- Event count: 3977
- Date range: 2020-01-03 08:00:00+00:00 -> 2026-06-07 20:00:00+00:00
- Avg forward 4h return: +0.1361%
- Median forward 4h return: +0.0000%
- Avg net forward 4h return (50 bps): -0.3639%
- Continuation rate: 0.539
- Reversal rate: 0.290
- Chop rate: 0.171

## Robustness Summary
- train: n=2784, continuation=0.571, reversal=0.278, avg_net_4h=-0.3796%
- test: n=1193, continuation=0.465, reversal=0.319, avg_net_4h=-0.3273%

## Top Reversal-Linked Trait Buckets
- follow_through=0.0: n=1251, reversal_rate=0.571, avg_net_4h=-2.1228%
- pullback_depth=DEEP: n=592, reversal_rate=0.461, avg_net_4h=-3.3146%
- pullback_depth=MODERATE: n=2096, reversal_rate=0.370, avg_net_4h=-0.6721%
- btc_regime=SIDEWAYS: n=988, reversal_rate=0.334, avg_net_4h=-0.4476%
- price=MICRO: n=562, reversal_rate=0.320, avg_net_4h=-0.2130%
- volume_spike=(0.188, 2.059]: n=992, reversal_rate=0.320, avg_net_4h=-0.5177%
- liquidity=SMALL: n=778, reversal_rate=0.314, avg_net_4h=-0.3277%
- size_proxy=SMALL: n=778, reversal_rate=0.314, avg_net_4h=-0.3277%

## Top Continuation-Linked Trait Buckets
- volatility=(2.958, 26.444]: n=991, continuation_rate=0.731, avg_net_4h=-0.6199%
- explosion_size=(8.85, 128.188]: n=994, continuation_rate=0.720, avg_net_4h=-0.5742%
- follow_through=1.0: n=2726, continuation_rate=0.680, avg_net_4h=+0.4433%
- pullback_depth=NO_PULLBACK_OBS: n=1289, continuation_rate=0.643, avg_net_4h=+1.4923%
- btc_regime=DOWN: n=647, continuation_rate=0.634, avg_net_4h=-0.5375%
- explosion_size=(5.929, 8.85]: n=994, continuation_rate=0.620, avg_net_4h=-0.3370%
- volume_spike=(4.402, 225.911]: n=992, continuation_rate=0.607, avg_net_4h=-0.2624%
- volatility=(1.898, 2.958]: n=991, continuation_rate=0.596, avg_net_4h=-0.2921%

## Preliminary Interpretation
- Explosion outcomes depend strongly on context and bucket definitions rather than being uniformly one-sided.
- Buckets with small sample warnings should be treated as unstable.
- Gross-vs-net comparison can materially reduce apparent edge.

## What To Test Next (Research-Only)
- Add delayed-entry variants (e.g., post-pullback) with strict no-look-ahead event-time features.
- Stress-test top buckets across longer holdout windows and symbol-universe subsets.
- Recheck robustness as new genuine 4h events accumulate.

## Safety Confirmation
- No live execution behavior changed.
- No scheduler/launchd changes.
- No production strategy logic changed.
