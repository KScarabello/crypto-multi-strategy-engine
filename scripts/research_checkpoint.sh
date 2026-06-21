#!/usr/bin/env bash
# =============================================================================
# research_checkpoint.sh
#
# Research-only checkpoint helper for study iterations.
#
# This script is intentionally conservative: it only stages and commits
# study-scoped research, tests, and report artifacts. It is not intended for
# committing live trading or production/scheduling changes.
#
# Usage:
#   scripts/research_checkpoint.sh <study_name> "<commit message>"
#   scripts/research_checkpoint.sh --dry-run <study_name> "<commit message>"
#   scripts/research_checkpoint.sh --push <study_name> "<commit message>"
#
# Example:
#   scripts/research_checkpoint.sh explosion_delayed_entry_study \
#     "Add explosion delayed-entry study"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN=0
DO_PUSH=0
POSITIONAL=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=1
            ;;
        --push)
            DO_PUSH=1
            ;;
        *)
            POSITIONAL+=("$arg")
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
    echo "Usage: scripts/research_checkpoint.sh [--dry-run] [--push] <study_name> \"<commit message>\"" >&2
    exit 1
fi

STUDY_NAME="${POSITIONAL[0]}"
COMMIT_MESSAGE="${POSITIONAL[1]}"

if [[ ! "$STUDY_NAME" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "ERROR: study_name must match ^[A-Za-z0-9_]+$" >&2
    exit 1
fi

if [[ -z "$COMMIT_MESSAGE" ]]; then
    echo "ERROR: commit message cannot be empty" >&2
    exit 1
fi

cd "$REPO_DIR"

RESEARCH_FILE="research/${STUDY_NAME}.py"
TEST_FILE="tests/test_${STUDY_NAME}.py"
REPORT_DIR="reports/${STUDY_NAME}"
TEST_CMD=".venv/bin/python -m pytest ${TEST_FILE}"

KNOWN_RUNTIME_PATHS=(
    "data/universe_snapshots/latest_universe_meta.json"
    "reports/memecoin_readiness_status.md"
    "logs/"
    "data/memecoin_signal_snapshots/"
    "data/memecoin_signal_history.csv"
    "data/memecoin_signal_outcomes.csv"
    "data/memecoin_candidates_latest.csv"
    "data/memecoin_candidates_ohlc_latest.csv"
    "data/memecoin_backfilled_signal_events.csv"
)

is_expected_path() {
    local path="$1"
    [[ "$path" == "$RESEARCH_FILE" ]] && return 0
    [[ "$path" == "$TEST_FILE" ]] && return 0
    [[ "$path" == "$REPORT_DIR" ]] && return 0
    [[ "$path" == "$REPORT_DIR/"* ]] && return 0
    return 1
}

is_runtime_generated_path() {
    local path="$1"
    local p
    for p in "${KNOWN_RUNTIME_PATHS[@]}"; do
        if [[ "$p" == */ ]]; then
            [[ "$path" == "$p"* ]] && return 0
        else
            [[ "$path" == "$p" ]] && return 0
        fi
    done
    return 1
}

is_prohibited_path() {
    local path="$1"
    case "$path" in
        live/*|execution/*|brokers/*)
            return 0
            ;;
        deployment/launchd/*|*.plist)
            return 0
            ;;
        config/*|shadow_state/*)
            return 0
            ;;
        .env|.env.*|secrets/*|credentials/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

classify_status_paths() {
    local line path old_path new_path
    CHANGED_PATHS=()

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        path="${line:3}"

        if [[ "$path" == *" -> "* ]]; then
            old_path="${path%% -> *}"
            new_path="${path##* -> }"
            CHANGED_PATHS+=("$old_path" "$new_path")
        else
            CHANGED_PATHS+=("$path")
        fi
    done < <(git status --porcelain=v1)
}

echo "============================================================"
echo "Research Checkpoint Precheck"
echo "Repo : ${REPO_DIR}"
echo "Mode : $([[ $DRY_RUN -eq 1 ]] && echo "DRY-RUN" || echo "APPLY")"
echo "Study: ${STUDY_NAME}"
echo "============================================================"
echo "Branch: $(git branch --show-current)"
git status --short

if [[ ! -f "$RESEARCH_FILE" ]]; then
    echo "ERROR: missing expected research file: ${RESEARCH_FILE}" >&2
    exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: missing expected test file: ${TEST_FILE}" >&2
    exit 1
fi

if [[ ! -d "$REPORT_DIR" ]]; then
    echo "ERROR: missing expected report directory: ${REPORT_DIR}" >&2
    exit 1
fi

classify_status_paths

UNEXPECTED_PATHS=()
RUNTIME_PATHS=()
PROHIBITED_PATHS=()

if [[ ${#CHANGED_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: no changes detected; nothing to checkpoint." >&2
    exit 1
fi

for p in "${CHANGED_PATHS[@]}"; do
    [[ -z "$p" ]] && continue

    if is_prohibited_path "$p"; then
        PROHIBITED_PATHS+=("$p")
        continue
    fi

    if is_expected_path "$p"; then
        continue
    fi

    if is_runtime_generated_path "$p"; then
        RUNTIME_PATHS+=("$p")
        continue
    fi

    UNEXPECTED_PATHS+=("$p")
done

if [[ ${#PROHIBITED_PATHS[@]} -gt 0 ]]; then
    echo "ERROR: refusing to continue because prohibited paths have changes:" >&2
    printf '  - %s\n' "${PROHIBITED_PATHS[@]}" >&2
    exit 1
fi

if [[ ${#UNEXPECTED_PATHS[@]} -gt 0 ]]; then
    echo "ERROR: refusing to continue because non-study paths are changed:" >&2
    printf '  - %s\n' "${UNEXPECTED_PATHS[@]}" >&2
    echo "Allowed changed paths are only:" >&2
    echo "  - ${RESEARCH_FILE}" >&2
    echo "  - ${TEST_FILE}" >&2
    echo "  - ${REPORT_DIR}/" >&2
    echo "Known runtime/generated paths are tolerated but never auto-staged." >&2
    exit 1
fi

if [[ ${#RUNTIME_PATHS[@]} -gt 0 ]]; then
    echo "Known runtime/generated changes detected (will NOT be committed):"
    printf '  - %s\n' "${RUNTIME_PATHS[@]}"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY-RUN: would run focused test command:"
    echo "  ${TEST_CMD}"
    echo "DRY-RUN: would stage only:"
    echo "  ${RESEARCH_FILE}"
    echo "  ${TEST_FILE}"
    echo "  ${REPORT_DIR}/"
    echo "DRY-RUN: would show staged diff summary via: git diff --cached --stat"
    echo "DRY-RUN: would commit with message: ${COMMIT_MESSAGE}"
    if [[ $DO_PUSH -eq 1 ]]; then
        echo "DRY-RUN: would push current branch"
    else
        echo "DRY-RUN: push not requested"
    fi
    exit 0
fi

echo "Running focused test: ${TEST_CMD}"
.venv/bin/python -m pytest "${TEST_FILE}"

echo "Staging study files only..."
git add -- "${RESEARCH_FILE}" "${TEST_FILE}" "${REPORT_DIR}"

STAGED_PATHS=()
while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    STAGED_PATHS+=("$p")
done < <(git diff --cached --name-only)

for p in "${STAGED_PATHS[@]}"; do
    if is_prohibited_path "$p"; then
        echo "ERROR: staged prohibited path detected: ${p}" >&2
        exit 1
    fi
    if ! is_expected_path "$p"; then
        echo "ERROR: staged non-study path detected: ${p}" >&2
        exit 1
    fi
done

echo "Staged diff summary:"
git diff --cached --stat

git commit -m "${COMMIT_MESSAGE}"

NEW_HASH="$(git rev-parse --short HEAD)"
echo "Checkpoint commit created: ${NEW_HASH}"

if [[ $DO_PUSH -eq 1 ]]; then
    echo "Pushing current branch..."
    git push
else
    echo "Push skipped (use --push to push automatically)."
fi
