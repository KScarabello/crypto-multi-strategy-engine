#!/usr/bin/env bash
# ============================================================================
# scripts/status_shadow_launchagent.sh
#
# Display the current status of the shadow-cycle LaunchAgent.
#
# Shows:
#   - Whether the agent is loaded in launchd
#   - launchctl list entry (PID and last exit code)
#   - Recent entries from logs/shadow_cycle.log
#   - Current shadow observation count (from control ledger)
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

echo "=== Shadow Cycle LaunchAgent Status ==="
echo "Label : ${LABEL}"
echo ""

# ─── launchd registration ────────────────────────────────────────────────────
echo "--- launchctl list ---"
if launchctl list | grep -q "${LABEL}" 2>/dev/null; then
    echo "[LOADED] Agent is registered with launchd:"
    launchctl list | grep "${LABEL}" || true
    echo ""
    echo "  Columns: PID | LastExitStatus | Label"
    echo "  PID=- means not currently running (scheduled, waiting for trigger)"
    echo "  LastExitStatus=0 means last run succeeded"
else
    echo "[NOT LOADED] Agent is not currently registered. Install with:"
    echo "  bash scripts/install_shadow_launchagent.sh"
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
import json, sys
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
