#!/usr/bin/env bash
# ============================================================================
# scripts/uninstall_shadow_launchagent.sh
#
# Safely removes the research-only shadow-cycle LaunchAgent.
#
# This script:
#   - Unloads/bootouts only the shadow-cycle agent
#   - Removes only the shadow-cycle plist from ~/Library/LaunchAgents/
#   - Does NOT delete shadow state, reports, logs, or any data
#   - Does NOT affect the live trading bot, cron, or any other agents
#   - Does NOT reset or regenerate shadow state
#
# Usage:
#   bash scripts/uninstall_shadow_launchagent.sh
# ============================================================================
set -euo pipefail

LABEL="com.kim.crypto-shadow-cycle"
PLIST_NAME="${LABEL}.plist"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
INSTALLED_PLIST="${LAUNCH_AGENTS_DIR}/${PLIST_NAME}"
UID_VAL="$(id -u)"

echo "=== Shadow Cycle LaunchAgent Uninstaller ==="
echo "Label    : ${LABEL}"
echo "Plist    : ${INSTALLED_PLIST}"
echo ""

# ─── Safety: refuse to run as root ───────────────────────────────────────────
if [ "${UID_VAL}" -eq 0 ]; then
    echo "[ERROR] Do not run this as root."
    exit 1
fi

# ─── Unload the agent if currently loaded ────────────────────────────────────
if launchctl list | grep -q "${LABEL}" 2>/dev/null; then
    echo "[UNLOAD] Unloading ${LABEL}..."
    launchctl bootout "gui/${UID_VAL}/${LABEL}" 2>/dev/null || \
        launchctl unload "${INSTALLED_PLIST}" 2>/dev/null || \
        echo "[WARN] bootout/unload returned nonzero — agent may already be stopped."
    echo "[UNLOAD] Agent unloaded."
else
    echo "[INFO] Agent is not currently loaded (nothing to unload)."
fi

# ─── Remove the installed plist ──────────────────────────────────────────────
if [ -f "${INSTALLED_PLIST}" ]; then
    rm "${INSTALLED_PLIST}"
    echo "[REMOVE] Removed: ${INSTALLED_PLIST}"
else
    echo "[INFO] Installed plist not found (already removed)."
fi

echo ""
echo "=== NOT removed (preserved) ==="
echo "  shadow_state/      — shadow portfolios and ledgers"
echo "  logs/              — cycle logs and launchd logs"
echo "  reports/           — monitoring reports"
echo "  deployment/launchd/ — repository plist template (untouched)"
echo ""
echo "[DONE] Uninstall complete. No live behavior was changed."
