#!/usr/bin/env bash
# =============================================================================
# status_memecoin_launchagent.sh
#
# Shows the current status of the memecoin collection-cycle LaunchAgent
# and a summary of the latest genuine prospective signal data.
#
# USAGE:
#   bash scripts/status_memecoin_launchagent.sh
# =============================================================================

set -uo pipefail

LABEL="com.kim.memecoin-collection-cycle"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
CYCLE_LOG="${REPO_DIR}/logs/memecoin_collection_cycle.log"
STDOUT_LOG="${REPO_DIR}/logs/memecoin_launchd_stdout.log"
PYTHON="${REPO_DIR}/.venv/bin/python3"

echo "============================================================"
echo "  MEMECOIN COLLECTION-CYCLE STATUS"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# ---- LaunchAgent loaded/not-loaded ----------------------------------------

if launchctl list 2>/dev/null | grep -q "${LABEL}"; then
    echo "  LaunchAgent : LOADED"
    RAW_STATUS="$(launchctl list 2>/dev/null | grep "${LABEL}" || true)"
    LAST_EXIT="$(echo "${RAW_STATUS}" | awk '{print $2}')"
    PID_COL="$(echo "${RAW_STATUS}" | awk '{print $1}')"
    if [[ "${PID_COL}" != "-" && -n "${PID_COL}" ]]; then
        echo "  Running     : YES (PID ${PID_COL})"
    else
        echo "  Running     : idle"
    fi
    echo "  Last exit   : ${LAST_EXIT:-unknown}"
else
    echo "  LaunchAgent : NOT LOADED"
    echo "  Install with: bash scripts/install_memecoin_launchagent.sh"
fi

# ---- Signal history stats -------------------------------------------------

echo ""
echo "  --- Genuine signal data ---"
HISTORY="${REPO_DIR}/data/memecoin_signal_history.csv"
OUTCOMES="${REPO_DIR}/data/memecoin_signal_outcomes.csv"

if [[ -f "${HISTORY}" && -f "${OUTCOMES}" && -x "${PYTHON}" ]]; then
    REPO_DIR="${REPO_DIR}" "${PYTHON}" - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("REPO_DIR", "."))

import pandas as pd
from pathlib import Path

REPO = Path(os.environ.get("REPO_DIR", "."))
hist_path = REPO / "data/memecoin_signal_history.csv"
out_path  = REPO / "data/memecoin_signal_outcomes.csv"

try:
    hist = pd.read_csv(hist_path, dtype=str)
    out  = pd.read_csv(out_path,  dtype=str)

    n_genuine = len(hist)
    newest_ts = hist["snapshot_ts_utc"].max() if "snapshot_ts_utc" in hist.columns else "unknown"

    # Rule 1: LE + vc=True + cc=False, matured 4h
    for c in ["volume_ratio_4h", "ret_15m_pct", "ret_1h_pct", "ret_4h_pct", "future_ret_4h_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "is_volume_climax" not in out.columns and "volume_ratio_4h" in out.columns:
        out["is_volume_climax"] = out["volume_ratio_4h"] > 10
    if "is_clean_continuation" not in out.columns and all(
        c in out.columns for c in ["ret_15m_pct","ret_1h_pct","ret_4h_pct"]
    ):
        out["is_clean_continuation"] = (
            (out["ret_15m_pct"] > 0) & (out["ret_1h_pct"] > 0) & (out["ret_4h_pct"] > 0)
        )

    r1_matured = 0
    if all(c in out.columns for c in ["ohlc_signal_type","is_volume_climax","is_clean_continuation","future_ret_4h_pct"]):
        vc = out["is_volume_climax"].astype(str).str.lower().isin({"true","1"})
        cc = out["is_clean_continuation"].astype(str).str.lower().isin({"false","0"})
        le = out["ohlc_signal_type"] == "LONG_EXPLOSION"
        r1_matured = int((le & vc & cc & out["future_ret_4h_pct"].notna()).sum())

    # Gate A from readiness report if available
    gate_a = "N/A"
    readiness = "N/A"
    status_csv = REPO / "reports/memecoin_readiness_status.csv"
    if status_csv.exists():
        try:
            s = pd.read_csv(status_csv)
            gate_a = f"{int(s['gate_a_pass_count'].iloc[0])}/{int(s['gate_a_total'].iloc[0])}"
            readiness = str(s['readiness_label'].iloc[0])
        except Exception:
            pass

    print(f"  Genuine signals     : {n_genuine}")
    print(f"  Newest genuine ts   : {newest_ts}")
    print(f"  Rule 1 mature-4h    : {r1_matured}")
    print(f"  Gate A              : {gate_a}")
    print(f"  Readiness           : {readiness}")
except Exception as e:
    print(f"  (Could not read data: {e})")
PYEOF
else
    echo "  (data files or Python not available)"
fi

# ---- Log tail -------------------------------------------------------------

echo ""
echo "  --- Last 10 lines of cycle log ---"
if [[ -f "${CYCLE_LOG}" ]]; then
    tail -10 "${CYCLE_LOG}" | sed 's/^/  /'
else
    echo "  (no cycle log yet: ${CYCLE_LOG})"
fi

echo ""
echo "============================================================"
echo "  Cycle log  : ${CYCLE_LOG}"
echo "  LaunchD out: ${STDOUT_LOG}"
echo "  Schedule   : every hour at :05 (America/Phoenix local time)"
echo "============================================================"
