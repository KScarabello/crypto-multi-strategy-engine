#!/usr/bin/env bash
set -euo pipefail

cd /Users/kimscarabello/Desktop/Repos/crypto/crypto-multi-strategy

echo "[$(date -u)] Updating old outcomes..."
.venv/bin/python -m research.memecoin_catcher.evaluate_signal_outcomes --recompute

echo "[$(date -u)] Fetching Kraken universe..."
.venv/bin/python -m research.memecoin_catcher.fetch_kraken_universe

echo "[$(date -u)] Ranking candidates..."
.venv/bin/python -m research.memecoin_catcher.rank_memecoin_candidates

echo "[$(date -u)] Enriching candidates with OHLC..."
.venv/bin/python -m research.memecoin_catcher.enrich_candidates_with_ohlc

echo "[$(date -u)] Saving signal snapshot..."
.venv/bin/python -m research.memecoin_catcher.save_signal_snapshot

echo "[$(date -u)] Evaluating outcomes again..."
.venv/bin/python -m research.memecoin_catcher.evaluate_signal_outcomes --recompute

echo "[$(date -u)] Done."
