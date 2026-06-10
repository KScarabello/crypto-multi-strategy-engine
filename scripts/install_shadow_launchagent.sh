#!/usr/bin/env bash
# ============================================================================
# scripts/install_shadow_launchagent.sh
#
# Installs the research-only shadow-cycle LaunchAgent for the current user.
#
# Steps:
#   1. Verify this is NOT being run as root
#   2. Derive the absolute repository path from this script's location
#   3. Substitute __REPO_DIR__ in the plist template
#   4. Create ~/Library/LaunchAgents if needed
#   5. Validate the resulting plist with plutil
#   6. Unload any existing copy (safe no-op if not loaded)
#   7. Install (copy) the plist to ~/Library/LaunchAgents/
#   8. Bootstrap / load the agent
#   9. Print verification commands
#
# This script is RESEARCH-ONLY:
#   - It installs only the shadow-cycle wrapper
#   - It never sources Kraken credentials
#   - It never modifies the live strategy, cron jobs, or Baseline v1
#   - It never resets or regenerates shadow state
#   - It does not modify frozen candidate specifications or hashes
#
# Usage:
#   bash scripts/install_shadow_launchagent.sh
# ============================================================================
set -euo pipefail

LABEL="com.kim.crypto-shadow-cycle"
PLIST_NAME="${LABEL}.plist"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${REPO_DIR}/deployment/launchd/${PLIST_NAME}"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
INSTALLED_PLIST="${LAUNCH_AGENTS_DIR}/${PLIST_NAME}"
UID_VAL="$(id -u)"

echo "=== Shadow Cycle LaunchAgent Installer ==="
echo "Label        : ${LABEL}"
echo "Repo         : ${REPO_DIR}"
echo "Template     : ${TEMPLATE}"
echo "Install to   : ${INSTALLED_PLIST}"
echo ""

# ─── Safety: refuse to run as root ───────────────────────────────────────────
if [ "${UID_VAL}" -eq 0 ]; then
    echo "[ERROR] Do not run this as root. LaunchAgents must be installed as a regular user."
    exit 1
fi

# ─── Safety: verify template exists ──────────────────────────────────────────
if [ ! -f "${TEMPLATE}" ]; then
    echo "[ERROR] Plist template not found: ${TEMPLATE}"
    exit 1
fi

# ─── Safety: verify wrapper script exists ────────────────────────────────────
WRAPPER="${REPO_DIR}/scripts/run_shadow_cycle.sh"
if [ ! -f "${WRAPPER}" ]; then
    echo "[ERROR] Wrapper script not found: ${WRAPPER}"
    exit 1
fi

# ─── Safety: refuse to install if live trading plist is present ──────────────
if grep -q "live.runner\|execute_order\|place_order\|kraken_secret\|api_secret" "${TEMPLATE}" 2>/dev/null; then
    echo "[ERROR] Template contains forbidden live-trading content. Aborting."
    exit 1
fi

# ─── Step 1: Substitute __REPO_DIR__ in a temp copy ─────────────────────────
TMP_PLIST="$(mktemp /tmp/${LABEL}.XXXXXX.plist)"
trap "rm -f '${TMP_PLIST}'" EXIT

sed "s|__REPO_DIR__|${REPO_DIR}|g" "${TEMPLATE}" > "${TMP_PLIST}"

# Verify no placeholder remains
if grep -q "__REPO_DIR__" "${TMP_PLIST}"; then
    echo "[ERROR] Substitution failed — __REPO_DIR__ still present in output."
    exit 1
fi

# ─── Step 2: Validate with plutil ────────────────────────────────────────────
echo "[VALIDATE] Running plutil -lint on generated plist..."
if ! plutil -lint "${TMP_PLIST}"; then
    echo "[ERROR] plutil validation failed. Aborting installation."
    exit 1
fi
echo "[VALIDATE] plist is valid."
echo ""

# ─── Step 3: Show diff if already installed ──────────────────────────────────
if [ -f "${INSTALLED_PLIST}" ]; then
    echo "[INFO] Existing installed plist found. Diff:"
    diff "${INSTALLED_PLIST}" "${TMP_PLIST}" || true
    echo ""
fi

# ─── Step 4: Create ~/Library/LaunchAgents if needed ─────────────────────────
mkdir -p "${LAUNCH_AGENTS_DIR}"

# ─── Step 5: Unload any currently loaded copy ────────────────────────────────
if launchctl list | grep -q "${LABEL}" 2>/dev/null; then
    echo "[UNLOAD] Unloading existing agent..."
    launchctl bootout "gui/${UID_VAL}/${LABEL}" 2>/dev/null || \
        launchctl unload "${INSTALLED_PLIST}" 2>/dev/null || \
        echo "[WARN] Could not unload existing agent (may not be loaded). Continuing."
fi

# ─── Step 6: Install plist ───────────────────────────────────────────────────
cp "${TMP_PLIST}" "${INSTALLED_PLIST}"
echo "[INSTALL] Plist installed to: ${INSTALLED_PLIST}"

# ─── Step 7: Bootstrap / load ────────────────────────────────────────────────
echo "[LOAD] Bootstrapping agent..."
if ! launchctl bootstrap "gui/${UID_VAL}" "${INSTALLED_PLIST}" 2>/dev/null; then
    # Fallback for environments where bootstrap may fail on first attempt
    launchctl load "${INSTALLED_PLIST}" 2>/dev/null || true
fi

# Give launchd a moment
sleep 1

# ─── Step 8: Verify loaded ────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
if launchctl list | grep -q "${LABEL}"; then
    launchctl list | grep "${LABEL}"
    echo "[OK] Agent is loaded and registered with launchd."
else
    echo "[WARN] Agent may not appear in launchctl list immediately. Check with:"
    echo "  bash scripts/status_shadow_launchagent.sh"
fi

echo ""
echo "=== Next Steps ==="
echo "  Check status    : bash scripts/status_shadow_launchagent.sh"
echo "  View cycle log  : tail -f ${REPO_DIR}/logs/shadow_cycle.log"
echo "  View stdout log : tail -f ${REPO_DIR}/logs/shadow_launchd_stdout.log"
echo "  Uninstall       : bash scripts/uninstall_shadow_launchagent.sh"
echo ""
echo "  The agent will run automatically at:"
echo "    01:10, 05:10, 09:10, 13:10, 17:10, 21:10  (MST / America/Phoenix)"
echo "  corresponding to 10 minutes after each 4h UTC candle boundary."
echo ""
echo "[DONE] Installation complete. No live behavior was changed."
