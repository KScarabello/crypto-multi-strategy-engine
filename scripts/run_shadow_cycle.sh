#!/usr/bin/env bash
# ============================================================================
# scripts/run_shadow_cycle.sh
#
# Research-only shadow validation cycle.
#
# Order of operations:
#   1. Acquire a non-overlapping lock (PID file)
#   2. Refresh OHLCV data via the canonical incremental updater
#   3. Abort if data refresh fails
#   4. Run shadow runner  (--from-start, idempotent, candle-guard active)
#   5. Run shadow monitor (append-only monitoring report)
#   6. Append timestamped stdout/stderr to logs/shadow_cycle.log
#   7. Release lock on every exit path (trap)
#
# Safety guarantees:
#   - Never sources Kraken API credentials
#   - Never imports or invokes live order execution
#   - Never resets shadow state automatically
#   - Never modifies frozen candidate specifications or hashes
#   - Returns nonzero exit code if any step fails
#
# Usage:
#   bash scripts/run_shadow_cycle.sh
#   # or, from an absolute path when scheduling:
#   /abs/path/to/repo/scripts/run_shadow_cycle.sh
# ============================================================================
set -euo pipefail

# ─── Paths (all absolute so scheduling works from any working directory) ──────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"
LOCK_FILE="${REPO_DIR}/shadow_state/shadow_cycle.lock"
LOG_FILE="${REPO_DIR}/logs/shadow_cycle.log"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/shadow_state"

# ─── Logging helper (stdout + log file) ──────────────────────────────────────
log() {
    local msg="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

# ─── Lock: PID file, non-overlapping ─────────────────────────────────────────
if [ -f "${LOCK_FILE}" ]; then
    LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
    if [ -n "${LOCK_PID}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
        log "[LOCK] Another shadow cycle (PID ${LOCK_PID}) is already running. Exiting."
        exit 1
    fi
    log "[LOCK] Stale lock file found (PID ${LOCK_PID:-unknown} no longer running). Proceeding."
fi
echo $$ > "${LOCK_FILE}"

# Release lock on every exit path
cleanup() {
    rm -f "${LOCK_FILE}"
}
trap cleanup EXIT

# ─── Change to repo root (required for Python module imports) ─────────────────
cd "${REPO_DIR}"

log "[START] shadow_cycle starting (PID $$)"

# ─── Step 1: Refresh OHLCV data ──────────────────────────────────────────────
# Uses the canonical incremental updater for BTC/USD ETH/USD XRP/USD SOL/USD AVAX/USD.
# get_data_symbols() in config returns the fixed-five universe.
# No --symbols flag = incremental mode (no full history backfill).
log "[DATA] Refreshing OHLCV data..."
if ! "${PYTHON}" -m data.update_ohlcv >> "${LOG_FILE}" 2>&1; then
    log "[ERROR] Data refresh failed — aborting shadow cycle."
    exit 1
fi
log "[DATA] OHLCV refresh complete."

# ─── Step 2: Shadow validation runner ────────────────────────────────────────
# --from-start is idempotent: already-processed bars are skipped by the ledger guard.
# Incomplete candles are rejected by the completed-candle guard inside the runner.
# Does NOT reset shadow state; does NOT touch frozen specs.
log "[SHADOW] Running shadow validation (--from-start)..."
if ! "${PYTHON}" -m research.shadow.runner --from-start >> "${LOG_FILE}" 2>&1; then
    log "[ERROR] Shadow runner failed."
    exit 1
fi
log "[SHADOW] Shadow runner complete."

# ─── Step 3: Monitoring report ───────────────────────────────────────────────
log "[MONITOR] Generating monitoring report..."
if ! "${PYTHON}" -m research.shadow.monitor >> "${LOG_FILE}" 2>&1; then
    log "[ERROR] Monitor generation failed."
    exit 1
fi
log "[MONITOR] Report written to reports/shadow_monitor_report.md"

# ─── Step 4: Summary (newest bar + observation count) ────────────────────────
"${PYTHON}" - <<'PYEOF' | tee -a "${LOG_FILE}"
import json
from pathlib import Path

state_dir = Path("shadow_state/ledger")
control_ledger = state_dir / "control.jsonl"

if not control_ledger.exists():
    print("[SUMMARY] No control ledger found.")
else:
    raw = control_ledger.read_text().strip()
    lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if not lines:
        print("[SUMMARY] No observations yet.")
    else:
        newest = lines[-1]["decision_ts"]
        count = len(lines)
        print(f"[SUMMARY] Newest completed bar : {newest}")
        print(f"[SUMMARY] Shadow observations  : {count}  (control ledger)")
PYEOF

log "[DONE] shadow_cycle complete."
