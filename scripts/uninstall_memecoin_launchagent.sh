#!/usr/bin/env bash
# =============================================================================
# uninstall_memecoin_launchagent.sh
#
# Unloads and removes the memecoin collection-cycle LaunchAgent.
#
# WHAT IT DOES:
#   1. Unloads the LaunchAgent from launchctl.
#   2. Removes the plist from ~/Library/LaunchAgents/.
#   3. Removes the generated plist from deployment/launchd/ (if present).
#
# WHAT IT DOES NOT DO:
#   - Does not delete logs or data files.
#   - Does not touch genuine signal history.
#   - Does not affect the five-coin shadow LaunchAgent.
#   - Does not run as root.
#
# USAGE:
#   bash scripts/uninstall_memecoin_launchagent.sh
# =============================================================================

set -euo pipefail

LABEL="com.kim.memecoin-collection-cycle"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
GENERATED="${REPO_DIR}/deployment/launchd/${LABEL}.generated.plist"

echo "=== Memecoin LaunchAgent uninstaller ==="
echo "  Label   : ${LABEL}"

if [[ "$(id -u)" -eq 0 ]]; then
    echo "ERROR: Do not run as root."
    exit 1
fi

# Safety: do not remove the shadow LaunchAgent by accident
SHADOW_LABEL="com.kim.crypto-shadow-cycle"
if [[ "${LABEL}" == "${SHADOW_LABEL}" ]]; then
    echo "ERROR: Label collision with shadow agent. Aborting."
    exit 1
fi

# Unload if loaded
if launchctl list 2>/dev/null | grep -q "${LABEL}"; then
    echo "  Unloading ${LABEL}..."
    launchctl unload "${INSTALL_DEST}" 2>/dev/null || true
    echo "  Unloaded."
else
    echo "  ${LABEL} is not loaded (nothing to unload)."
fi

# Remove installed plist
if [[ -f "${INSTALL_DEST}" ]]; then
    rm -f "${INSTALL_DEST}"
    echo "  Removed: ${INSTALL_DEST}"
else
    echo "  Not found (already removed): ${INSTALL_DEST}"
fi

# Remove generated plist from repo (optional cleanup)
if [[ -f "${GENERATED}" ]]; then
    rm -f "${GENERATED}"
    echo "  Removed generated plist: ${GENERATED}"
fi

echo ""
echo "=== Uninstall complete ==="
echo "  Logs and data files are preserved."
echo "  To reinstall: bash scripts/install_memecoin_launchagent.sh"
