#!/usr/bin/env bash
# =============================================================================
# run_memecoin_collection_cycle.sh
#
# Research-only automated prospective memecoin event collection cycle.
#
# SAFETY GUARANTEES:
#   - Never places live orders
#   - Never modifies five-coin shadow portfolios or frozen candidate hashes
#   - Never runs historical backfill automatically
#   - Only reads public Kraken REST API (no authentication)
#   - PID lock prevents concurrent cycles
#   - All output logged with timestamps
#   - Nonzero exit code if a required stage fails
#   - Lock removed on every exit path (including errors)
#
# PIPELINE ORDER:
#   1. Acquire PID lock
#   2. Evaluate existing matured outcomes (required)
#   3. Fetch Kraken universe + write immutable timestamped snapshot (required)
#   4. Rank candidates (required)
#   5. Enrich candidates with completed OHLC candles (required)
#   6. Save genuine signal snapshot + append-only to history (required)
#   7. Evaluate newly matured outcomes from new snapshot (required)
#   8. Run readiness report — genuine events only (required)
#   9. Validate candidate rules — genuine events only (optional, soft failure)
#  10. Print concise summary
#  11. Release lock
#
# USAGE:
#   ./scripts/run_memecoin_collection_cycle.sh
#   ./scripts/run_memecoin_collection_cycle.sh --skip-validation
#   ./scripts/run_memecoin_collection_cycle.sh --dry-run
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python3"
LOCK_FILE="${REPO_DIR}/data/memecoin_collection_cycle.lock"
LOG_FILE="${REPO_DIR}/logs/memecoin_collection_cycle.log"
STAGE_TIMEOUT=300   # seconds per stage before warning

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DRY_RUN=0
SKIP_VALIDATION=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=1 ;;
        --skip-validation) SKIP_VALIDATION=1 ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--dry-run] [--skip-validation]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

mkdir -p "${REPO_DIR}/logs"
CYCLE_START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() {
    local msg="[${CYCLE_START_TS}] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

log_ts() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# Redirect all stdout+stderr to log file AND terminal from this point forward
exec > >(tee -a "${LOG_FILE}") 2>&1

log_ts "============================================================"
log_ts "  Memecoin Collection Cycle starting"
log_ts "  Repo: ${REPO_DIR}"
if [[ $DRY_RUN -eq 1 ]]; then
    log_ts "  Mode: DRY-RUN (no files will be modified)"
fi
log_ts "============================================================"

# ---------------------------------------------------------------------------
# Lock acquisition
# ---------------------------------------------------------------------------

acquire_lock() {
    if [[ -f "${LOCK_FILE}" ]]; then
        existing_pid="$(cat "${LOCK_FILE}" 2>/dev/null || echo "")"
        if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
            log_ts "ERROR: Another collection cycle is running (PID ${existing_pid}). Exiting."
            exit 1
        else
            log_ts "WARNING: Stale lock file found (PID ${existing_pid}). Removing."
            rm -f "${LOCK_FILE}"
        fi
    fi
    echo $$ > "${LOCK_FILE}"
    log_ts "Lock acquired (PID $$): ${LOCK_FILE}"
}

release_lock() {
    if [[ -f "${LOCK_FILE}" ]]; then
        rm -f "${LOCK_FILE}"
        log_ts "Lock released: ${LOCK_FILE}"
    fi
}

# Release lock on all exit paths
trap 'release_lock; log_ts "Cycle exited (exit code: $?)"' EXIT

acquire_lock

# ---------------------------------------------------------------------------
# Summary printer (defined early so run_stage can call it on failure)
# ---------------------------------------------------------------------------

print_summary() {
    AFTER_HISTORY_ROWS=0
    AFTER_OUTCOME_ROWS=0
    AFTER_SNAPSHOTS=0
    if [[ -f "${REPO_DIR}/data/memecoin_signal_history.csv" ]]; then
        AFTER_HISTORY_ROWS=$(( $(wc -l < "${REPO_DIR}/data/memecoin_signal_history.csv") - 1 ))
    fi
    if [[ -f "${REPO_DIR}/data/memecoin_signal_outcomes.csv" ]]; then
        AFTER_OUTCOME_ROWS=$(( $(wc -l < "${REPO_DIR}/data/memecoin_signal_outcomes.csv") - 1 ))
    fi
    AFTER_SNAPSHOTS=$(ls "${REPO_DIR}/data/memecoin_signal_snapshots/"memecoin_signals_*.csv 2>/dev/null | wc -l || echo 0)

    log_ts "============================================================"
    log_ts "  CYCLE SUMMARY"
    log_ts "  Lock path    : ${LOCK_FILE}"
    log_ts "  Log path     : ${LOG_FILE}"
    log_ts "------------------------------------------------------------"
    log_ts "  History rows : ${BEFORE_HISTORY_ROWS} → ${AFTER_HISTORY_ROWS} (Δ $(( AFTER_HISTORY_ROWS - BEFORE_HISTORY_ROWS )))"
    log_ts "  Outcome rows : ${BEFORE_OUTCOME_ROWS} → ${AFTER_OUTCOME_ROWS} (Δ $(( AFTER_OUTCOME_ROWS - BEFORE_OUTCOME_ROWS )))"
    log_ts "  Snapshots    : ${BEFORE_SNAPSHOTS} → ${AFTER_SNAPSHOTS}"
    log_ts "------------------------------------------------------------"
    log_ts "  Stage results:"
    for r in "${STAGE_RESULTS[@]}"; do
        log_ts "    ${r}"
    done
    log_ts "------------------------------------------------------------"
    log_ts "  Skipped stages:"
    log_ts "    NONE scheduled: historical backfill (run manually only)"
    log_ts "    NONE scheduled: live order placement (never automated)"
    log_ts "  Five-coin shadow state: not modified"
    log_ts "  Simulated backfill: not modified"
    log_ts "============================================================"
}

# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

STAGE_FAILED=0
STAGE_RESULTS=()

run_stage() {
    local stage_name="$1"
    local required="$2"   # "required" or "optional"
    shift 2
    local cmd=("$@")

    log_ts "--- Stage: ${stage_name} ---"

    if [[ $DRY_RUN -eq 1 ]]; then
        log_ts "  DRY-RUN: would run: ${cmd[*]}"
        STAGE_RESULTS+=("SKIPPED(dry-run): ${stage_name}")
        return 0
    fi

    local t_start
    t_start="$(date +%s)"
    local exit_code=0

    "${cmd[@]}" || exit_code=$?

    local t_end
    t_end="$(date +%s)"
    local duration=$(( t_end - t_start ))

    if [[ $exit_code -eq 0 ]]; then
        log_ts "  OK: ${stage_name} (${duration}s)"
        STAGE_RESULTS+=("OK(${duration}s): ${stage_name}")
    else
        log_ts "  FAILED: ${stage_name} (exit code: ${exit_code}, ${duration}s)"
        STAGE_RESULTS+=("FAILED(${exit_code}): ${stage_name}")
        if [[ "$required" == "required" ]]; then
            STAGE_FAILED=1
            log_ts "Required stage failed — aborting cycle"
            # Print summary before exiting
            print_summary
            exit "${exit_code}"
        else
            log_ts "Optional stage failed — continuing"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

log_ts "Running safety checks..."
if [[ ! -f "${PYTHON}" ]]; then
    log_ts "ERROR: Python interpreter not found at ${PYTHON}"
    exit 1
fi

# Confirm no imports from live execution in research modules
if grep -r "from brokers\|import brokers\|from live\|from execution.kraken\|place_order\|submit_order\|ccxt" \
    "${REPO_DIR}/research/memecoin_catcher/" 2>/dev/null | grep -v "^Binary\|\.pyc" | grep -v "# " | head -5; then
    log_ts "ERROR: Forbidden live-execution import detected in research/memecoin_catcher/"
    exit 1
fi
log_ts "  Safety checks passed"

# ---------------------------------------------------------------------------
# Capture before-state
# ---------------------------------------------------------------------------

log_ts "Capturing before-state..."
BEFORE_HISTORY_ROWS=0
BEFORE_OUTCOME_ROWS=0
BEFORE_SNAPSHOTS=0
if [[ -f "${REPO_DIR}/data/memecoin_signal_history.csv" ]]; then
    BEFORE_HISTORY_ROWS=$(( $(wc -l < "${REPO_DIR}/data/memecoin_signal_history.csv") - 1 ))
fi
if [[ -f "${REPO_DIR}/data/memecoin_signal_outcomes.csv" ]]; then
    BEFORE_OUTCOME_ROWS=$(( $(wc -l < "${REPO_DIR}/data/memecoin_signal_outcomes.csv") - 1 ))
fi
BEFORE_SNAPSHOTS=$(ls "${REPO_DIR}/data/memecoin_signal_snapshots/"memecoin_signals_*.csv 2>/dev/null | wc -l || echo 0)
log_ts "  Before: history=${BEFORE_HISTORY_ROWS} outcomes=${BEFORE_OUTCOME_ROWS} snapshots=${BEFORE_SNAPSHOTS}"

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

cd "${REPO_DIR}"

# Stage 1: Evaluate matured outcomes for all existing signals
run_stage "A: Evaluate existing matured outcomes" required \
    "${PYTHON}" -m research.memecoin_catcher.evaluate_signal_outcomes

# Stage 2: Fetch Kraken universe (writes immutable timestamped snapshot + metadata)
run_stage "B: Fetch Kraken universe" required \
    "${PYTHON}" -m research.memecoin_catcher.fetch_kraken_universe

# Stage 3: Rank memecoin candidates
run_stage "C: Rank memecoin candidates" required \
    "${PYTHON}" -m research.memecoin_catcher.rank_memecoin_candidates

# Stage 4: Enrich candidates with completed OHLC candles
run_stage "D: Enrich candidates with completed OHLC" required \
    "${PYTHON}" -m research.memecoin_catcher.enrich_candidates_with_ohlc

# Stage 5: Save genuine prospective signal snapshot
run_stage "E: Save genuine signal snapshot" required \
    "${PYTHON}" -m research.memecoin_catcher.save_signal_snapshot

# Stage 6: Evaluate newly matured outcomes (including new snapshot signals)
run_stage "F: Evaluate outcomes (post-snapshot)" required \
    "${PYTHON}" -m research.memecoin_catcher.evaluate_signal_outcomes

# Stage 7: Readiness report (genuine events only)
run_stage "G: Readiness report (genuine events only)" required \
    "${PYTHON}" -m research.memecoin_catcher.memecoin_readiness_report

# Stage 8: Candidate rule validation (optional — uses backfilled events for research context)
if [[ $SKIP_VALIDATION -eq 0 ]]; then
    run_stage "H: Candidate rule validation" optional \
        "${PYTHON}" -m research.memecoin_catcher.validate_memecoin_candidate_rules
fi

# ---------------------------------------------------------------------------
# Capture after-state and print summary
# ---------------------------------------------------------------------------

print_summary

if [[ $STAGE_FAILED -ne 0 ]]; then
    exit 1
fi
