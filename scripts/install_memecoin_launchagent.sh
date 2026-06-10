#!/usr/bin/env bash
# =============================================================================
# install_memecoin_launchagent.sh
#
# Installs the memecoin collection-cycle LaunchAgent for the current user.
#
# WHAT IT DOES:
#   1. Resolves the repo's absolute path.
#   2. Substitutes __REPO_DIR__ in the plist template.
#   3. Validates the generated plist with plutil.
#   4. Copies it to ~/Library/LaunchAgents/.
#   5. Loads it with launchctl (schedules but does NOT run immediately).
#
# WHAT IT DOES NOT DO:
#   - Does not modify Rule 1 or any strategy parameter.
#   - Does not modify exit policies or evidence thresholds.
#   - Does not source Kraken credentials.
#   - Does not invoke historical backfill.
#   - Does not reset or truncate genuine event history.
#   - Does not touch the five-coin shadow LaunchAgent.
#   - Does not run as root.
#
# USAGE:
#   bash scripts/install_memecoin_launchagent.sh
# =============================================================================

set -euo pipefail

LABEL="com.kim.memecoin-collection-cycle"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="${REPO_DIR}/deployment/launchd/${LABEL}.plist"
GENERATED="${REPO_DIR}/deployment/launchd/${LABEL}.generated.plist"
INSTALL_DIR="${HOME}/Library/LaunchAgents"
INSTALL_DEST="${INSTALL_DIR}/${LABEL}.plist"
LOG_DIR="${REPO_DIR}/logs"

echo "=== Memecoin LaunchAgent installer ==="
echo "  Repo      : ${REPO_DIR}"
echo "  Label     : ${LABEL}"
echo "  Template  : ${TEMPLATE}"
echo "  Installs  : ${INSTALL_DEST}"
echo ""

# ---- Safety checks -------------------------------------------------------

if [[ "$(id -u)" -eq 0 ]]; then
    echo "ERROR: Do not run as root. LaunchAgents must be installed by the target user."
    exit 1
fi

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "ERROR: Plist template not found: ${TEMPLATE}"
    exit 1
fi

WRAPPER="${REPO_DIR}/scripts/run_memecoin_collection_cycle.sh"
if [[ ! -f "${WRAPPER}" ]]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER}"
    exit 1
fi

# ---- Substitute __REPO_DIR__ ---------------------------------------------

sed "s|__REPO_DIR__|${REPO_DIR}|g" "${TEMPLATE}" > "${GENERATED}"
echo "  Generated plist: ${GENERATED}"

# ---- Validate with plutil ------------------------------------------------

echo ""
echo "Running plutil validation..."
if plutil -lint "${GENERATED}"; then
    echo "  plutil: OK"
else
    echo "ERROR: plist validation failed. Aborting."
    rm -f "${GENERATED}"
    exit 1
fi

# ---- Double-check safety invariants in the generated plist ---------------
# Inspect ProgramArguments specifically — do NOT grep the whole file for
# keywords like "backfill" or "shadow", because the plist template contains
# harmless XML comments ("Never invokes historical backfill",
# "Never modifies five-coin shadow portfolios") that are not executable
# references and would cause false-positive aborts.

GENERATED_PLIST="${GENERATED}" python3 <<'PYEOF'
import os, sys, plistlib

plist_path = os.environ["GENERATED_PLIST"]
with open(plist_path, "rb") as fh:
    plist = plistlib.load(fh)

prog_args = plist.get("ProgramArguments", [])
REQUIRED_WRAPPER = "run_memecoin_collection_cycle.sh"

FORBIDDEN_BACKFILL = [
    "backfill_recent_memecoin_signals",
    "research.memecoin_catcher.backfill",
    "memecoin_backfilled_signal_events",
]
FORBIDDEN_SHADOW = [
    "run_shadow_cycle",
    "research.shadow",
    "shadow_cycle",
    "status_shadow_launchagent",
    "install_shadow_launchagent",
]
FORBIDDEN_CREDENTIALS = [
    "place_order", "submit_order", "kraken_api",
    "api_key", "api_secret", "secret_key", "private_key",
]

# 1. ProgramArguments must invoke the exact wrapper.
if not any(REQUIRED_WRAPPER in arg for arg in prog_args):
    print(f"ERROR: ProgramArguments does not invoke {REQUIRED_WRAPPER}: {prog_args}")
    sys.exit(1)

for arg in prog_args:
    # 2. No backfill executable.
    for token in FORBIDDEN_BACKFILL:
        if token in arg:
            print(f"ERROR: Forbidden backfill executable '{token}' in ProgramArguments.")
            sys.exit(2)
    if "backfill" in arg.lower():
        print(f"ERROR: Unexpected backfill reference in ProgramArguments: {arg!r}")
        sys.exit(2)

    # 3. No shadow executable.
    for token in FORBIDDEN_SHADOW:
        if token in arg:
            print(f"ERROR: Forbidden shadow executable '{token}' in ProgramArguments.")
            sys.exit(3)

    # 4. No live-trading or credential executable.
    for token in FORBIDDEN_CREDENTIALS:
        if token in arg.lower():
            print(f"ERROR: Forbidden credential/live-trading token '{token}' in ProgramArguments.")
            sys.exit(4)

print("  ProgramArguments: OK")
PYEOF
if [[ $? -ne 0 ]]; then
    echo "ERROR: ProgramArguments safety check failed. Aborting."
    rm -f "${GENERATED}"
    exit 1
fi

echo "  Safety checks: OK"

# ---- Ensure log directory exists ----------------------------------------

mkdir -p "${LOG_DIR}"
echo "  Log directory: ${LOG_DIR}"

# ---- Check for existing installation ------------------------------------

if launchctl list 2>/dev/null | grep -q "${LABEL}"; then
    echo ""
    echo "WARNING: ${LABEL} is already loaded."
    echo "Unloading existing agent before reinstalling..."
    launchctl unload "${INSTALL_DEST}" 2>/dev/null || true
fi

# ---- Install and load ---------------------------------------------------

mkdir -p "${INSTALL_DIR}"
cp "${GENERATED}" "${INSTALL_DEST}"
echo "  Copied to: ${INSTALL_DEST}"

launchctl load "${INSTALL_DEST}"
echo "  Loaded with launchctl."

# ---- Report -------------------------------------------------------------

echo ""
echo "=== Installation complete ==="
echo "  Label     : ${LABEL}"
echo "  Schedule  : every hour at :05 (America/Phoenix local time)"
echo "  Wrapper   : ${WRAPPER}"
echo "  Cycle log : ${LOG_DIR}/memecoin_collection_cycle.log"
echo "  Launchd stdout : ${LOG_DIR}/memecoin_launchd_stdout.log"
echo "  Launchd stderr : ${LOG_DIR}/memecoin_launchd_stderr.log"
echo ""
echo "  Status : bash scripts/status_memecoin_launchagent.sh"
echo "  Uninstall: bash scripts/uninstall_memecoin_launchagent.sh"
echo ""
echo "NOTE: RunAtLoad=false — first run will occur at the next :05 mark."
