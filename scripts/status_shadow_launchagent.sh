#!/usr/bin/env bash
# ============================================================================
# scripts/status_shadow_launchagent.sh
#
# Display the current status of the shadow-cycle LaunchAgent.
#
# Shows:
#   - Registration: LOADED or NOT LOADED
#   - Runtime state: running or idle
#   - PID (only when the job is actively executing)
#   - Last exit status (when available)
#   - Recent entries from logs/shadow_cycle.log
#   - Current shadow observation count (from control ledger)
#
# Loaded-state determination:
#   Uses "launchctl print gui/<uid>/<label>" as the authoritative check.
#   Exit status 0 means LOADED.  A dash in "launchctl list" PID column means
#   the job is idle/scheduled, not unloaded — this script does NOT use the
#   presence or absence of a PID to determine registration state.
#
# Usage:
#   bash scripts/status_shadow_launchagent.sh
# ============================================================================
set -euo pipefail

LABEL="com.kim.crypto-shadow-cycle"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CYCLE_LOG="${REPO_DIR}/logs/shadow_cycle.log"
STDOUT_LOG="${REPO_DIR}/logs/shadow_launchd_stdout.log"
STDERR_LOG="${REPO_DIR}/logs/shadow_launchd_stderr.log"
LEDGER="${REPO_DIR}/shadow_state/ledger/control.jsonl"
PYTHON="${REPO_DIR}/.venv/bin/python"
UID_VAL="$(id -u)"
DOMAIN_TARGET="gui/${UID_VAL}/${LABEL}"

echo "=== Shadow Cycle LaunchAgent Status ==="
echo "Label  : ${LABEL}"
echo "Domain : ${DOMAIN_TARGET}"
echo ""

# ─── Authoritative loaded-state check ────────────────────────────────────────
# launchctl print exits 0 if the service is registered, nonzero if not.
# A dash in "launchctl list" PID column = idle/waiting, NOT unloaded.
PRINT_OUTPUT=""
if PRINT_OUTPUT="$(launchctl print "${DOMAIN_TARGET}" 2>&1)"; then
    REGISTRATION="LOADED"
else
    REGISTRATION="NOT LOADED"
fi

echo "--- Registration ---"
if [ "${REGISTRATION}" = "LOADED" ]; then
    echo "  [LOADED] Agent is registered with launchd and will run on schedule."
    echo ""

    # Runtime state from launchctl print output
    STATE_LINE="$(echo "${PRINT_OUTPUT}" | grep 'state\s*=' || echo "")"
    if echo "${STATE_LINE}" | grep -qE "state[[:space:]]*=[[:space:]]*running$"; then
        echo "  Runtime state : running"
        # Extract PID from launchctl list when actually running
        PID_LINE="$(launchctl list 2>/dev/null | grep "${LABEL}" || echo "")"
        PID_VAL="$(echo "${PID_LINE}" | awk '{print $1}')"
        if [ -n "${PID_VAL}" ] && [ "${PID_VAL}" != "-" ]; then
            echo "  PID           : ${PID_VAL}"
        fi
    else
        echo "  Runtime state : idle (scheduled, waiting for next trigger)"
        echo "  Note: A dash in launchctl list PID column is normal for a scheduled job."
    fi

    # Last exit status from launchctl print
    EXIT_LINE="$(echo "${PRINT_OUTPUT}" | grep 'last exit code' || echo "")"
    if [ -n "${EXIT_LINE}" ]; then
        echo "  ${EXIT_LINE# }"
    fi

    echo ""
    echo "--- launchctl list entry ---"
    launchctl list 2>/dev/null | grep "${LABEL}" || echo "  (not yet shown in list)"
    echo "  Columns: PID | LastExitStatus | Label"
else
    echo "  [NOT LOADED] Agent is not registered with launchd."
    echo "  Install with: bash scripts/install_shadow_launchagent.sh"
fi
echo ""

# ─── Installed plist ─────────────────────────────────────────────────────────
INSTALLED="${HOME}/Library/LaunchAgents/${LABEL}.plist"
if [ -f "${INSTALLED}" ]; then
    echo "--- Installed plist ---"
    echo "  ${INSTALLED}  (present)"
    echo ""
else
    echo "--- Installed plist ---"
    echo "  ${INSTALLED}  (NOT found)"
    echo ""
fi

# ─── Recent cycle log entries ────────────────────────────────────────────────
echo "--- Last 20 lines of logs/shadow_cycle.log ---"
if [ -f "${CYCLE_LOG}" ]; then
    tail -20 "${CYCLE_LOG}"
else
    echo "  (log not found: ${CYCLE_LOG})"
fi
echo ""

# ─── LaunchAgent stdout/stderr logs ──────────────────────────────────────────
if [ -f "${STDOUT_LOG}" ] && [ -s "${STDOUT_LOG}" ]; then
    echo "--- Last 10 lines of shadow_launchd_stdout.log ---"
    tail -10 "${STDOUT_LOG}"
    echo ""
fi
if [ -f "${STDERR_LOG}" ] && [ -s "${STDERR_LOG}" ]; then
    echo "--- Last 10 lines of shadow_launchd_stderr.log (errors) ---"
    tail -10 "${STDERR_LOG}"
    echo ""
fi

# ─── Shadow observation count ────────────────────────────────────────────────
echo "--- Shadow observations ---"
if [ -f "${LEDGER}" ]; then
    "${PYTHON}" - <<'PYEOF'
import json
from pathlib import Path
import os

ledger_dir = Path(os.environ.get("LEDGER_DIR", "shadow_state/ledger"))
candidates = ["control", "min_hold_6", "rank_buffer_4", "combo_buf4_conf2_hold2",
              "btc_buyhold", "ewb_buyhold"]

for name in candidates:
    path = ledger_dir / f"{name}.jsonl"
    if not path.exists():
        print(f"  {name:<28}  0 observations")
        continue
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        print(f"  {name:<28}  0 observations")
        continue
    last = json.loads(lines[-1])
    print(f"  {name:<28}  {len(lines):>3} observations  latest={last['decision_ts']}")
PYEOF
else
    echo "  Control ledger not found: ${LEDGER}"
    echo "  Run the shadow cycle first: bash scripts/run_shadow_cycle.sh"
fi

echo ""
echo "=== End Status ==="
