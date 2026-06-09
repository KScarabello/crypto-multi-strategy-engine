# Shadow Validation Framework

Research-only prospective shadow validation for the crypto momentum strategy.

## Purpose and Constraints

This framework **paper-trades** four candidate strategy variants and two benchmarks,
starting from `2026-06-09T00:00:00+00:00`, to gather prospective out-of-sample evidence.

**Critical constraints:**
- No imports from `brokers/`, `execution/`, or `live/`
- No orders sent to Kraken or any exchange
- No live configuration, cron, or account state modified
- Shadow state isolated in `shadow_state/` (never under `live/` or `config/`)

## One Shadow Cycle (One Bar)

```bash
.venv/bin/python -m research.shadow.runner --bar 2026-06-09T04:00:00+00:00
```

## Backfill from Official Start

```bash
.venv/bin/python -m research.shadow.runner --from-start --start 2026-06-09T00:00:00+00:00
```

## View Current Results

```bash
.venv/bin/python -m research.shadow.monitor
```

Report written to: `reports/shadow_monitor_report.md`

## Verify Candidate Hashes

```bash
.venv/bin/python -m research.shadow.specs
```

## Confirm No Live Execution

To verify no live execution occurs:
1. Check that `research/shadow/` has no imports from `brokers/`, `execution/`, `live/`
2. Run: `grep -r "from brokers\|from execution\|from live\|import brokers\|import execution\|import live" research/shadow/`
3. Run: `grep -r "kraken" research/shadow/` — should return nothing
4. State files are in `shadow_state/`, never in `live/`, `.signals/`, or `.state/`

## State File Locations

```
shadow_state/
  portfolios/{candidate_name}.json    # current portfolio state per candidate
  ledger/{candidate_name}.jsonl       # append-only bar-by-bar evidence
  decisions/                          # combined decision log (future use)
```

## How to Add a Bar Manually

1. Ensure close price data is available in `data/local/`
2. Run: `.venv/bin/python -m research.shadow.runner --bar <ISO_TIMESTAMP>`
3. Processing is idempotent — running the same bar twice is safe
4. Use `--dry-run` to preview without writing state

## Candidates

| Name | Description |
|------|-------------|
| `control` | Strict top-3 replacement (baseline) |
| `min_hold_6` | Minimum 6-rebalance hold (~12 calendar days) |
| `rank_buffer_4` | Retain incumbent while rank ≤ 4 |
| `combo_buf4_conf2_hold2` | rank_buffer=4 + challenger_confirm=2 + min_hold=2 |
| `btc_buyhold` | BTC/USD buy-and-hold benchmark |
| `ewb_buyhold` | Equal-weight five-coin benchmark |
