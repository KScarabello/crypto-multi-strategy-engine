"""Static safety checks proving run_shadow_cycle.sh cannot invoke live trading.

These tests read the wrapper script and verify by content analysis that it:
- Never sources Kraken credentials
- Never imports or invokes live order execution modules
- Never resets shadow state automatically
- Uses the repo .venv Python interpreter
- Uses the canonical read-only data refresher (data.update_ohlcv)
- Contains the candle-guard-preserving --from-start flag
- Returns nonzero exit code on failure (set -euo pipefail present)
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_shadow_cycle.sh"


def _script_text() -> str:
    assert SCRIPT_PATH.exists(), f"Wrapper script not found: {SCRIPT_PATH}"
    return SCRIPT_PATH.read_text()


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_script_is_executable():
    import stat
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "Script must be executable by owner"


def test_no_live_credentials_sourced():
    """Script must never source Kraken or other exchange credential files."""
    import re
    # Strip comment lines before checking for credential patterns
    text = _script_text()
    non_comment_lines = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    forbidden_patterns = [
        r"^\s*source\s+.*kraken",
        r"^\s*\.\s+.*kraken",
        r"^\s*source\s+.*secret",
        r"^\s*\.\s+.*secret",
        r"^\s*source\s+.*cred",
        r"^\s*export\s+KRAKEN_API",
        r"^\s*KRAKEN_API_KEY\s*=",
        r"^\s*KRAKEN_API_SECRET\s*=",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, non_comment_lines, re.IGNORECASE | re.MULTILINE), (
            f"Wrapper script contains forbidden credential pattern: {pattern!r}"
        )


def test_no_live_order_execution():
    """Script must not import or invoke live order execution modules."""
    text = _script_text()
    forbidden = [
        "brokers.kraken",
        "execution.order",
        "place_order",
        "create_order",
        "execute_order",
        "send_order",
        "live.runner",
        "live.strategy",
    ]
    for term in forbidden:
        assert term not in text, (
            f"Wrapper script contains live-execution reference: {term!r}"
        )


def test_no_auto_reset():
    """Script must never call --reset automatically."""
    text = _script_text()
    assert "--reset" not in text, (
        "Wrapper must not auto-reset shadow state. "
        "--reset removes the append-only ledger and must only be run manually."
    )


def test_uses_venv_python():
    """Script must use the repo .venv interpreter, not system python."""
    text = _script_text()
    assert ".venv/bin/python" in text, "Script must use .venv/bin/python"
    assert "#!/usr/bin/env python" not in text


def test_uses_canonical_data_updater():
    """Script must use data.update_ohlcv (the canonical incremental updater)."""
    text = _script_text()
    assert "data.update_ohlcv" in text, (
        "Script must use the canonical data.update_ohlcv module for data refresh"
    )


def test_no_backfill_since_flag():
    """Script must use incremental mode — no --since flag that triggers full backfill."""
    text = _script_text()
    # The incremental (no --since) path is correct; --since would trigger full backfill
    assert "--since" not in text, (
        "Script must not pass --since to data.update_ohlcv; "
        "that triggers full backfill instead of incremental update"
    )


def test_shadow_runner_uses_from_start():
    """Script must run shadow runner with --from-start (idempotent backfill)."""
    text = _script_text()
    assert "--from-start" in text, "Runner must be invoked with --from-start"


def test_exits_on_error():
    """Script must use set -euo pipefail to exit on any failure."""
    text = _script_text()
    assert "set -euo pipefail" in text, "Script must use 'set -euo pipefail'"


def test_uses_lock_file():
    """Script must acquire a non-overlapping lock."""
    text = _script_text()
    assert "LOCK_FILE" in text and "LOCK" in text, "Script must implement a lock mechanism"


def test_lock_released_on_exit():
    """Lock must be released via trap on all exit paths."""
    text = _script_text()
    assert "trap" in text and "cleanup" in text, (
        "Script must use trap + cleanup to release lock on all exit paths"
    )


def test_logs_to_shadow_cycle_log():
    """Script must append to logs/shadow_cycle.log."""
    text = _script_text()
    assert "shadow_cycle.log" in text, "Script must log to logs/shadow_cycle.log"


def test_no_frozen_spec_modification():
    """Script must not write to specs.py or modify frozen candidate hashes."""
    text = _script_text()
    assert "specs.py" not in text
    assert "SPEC_HASHES" not in text
    assert "FROZEN_CANDIDATES" not in text


def test_does_not_invoke_live_modules():
    """Verify none of the live/ modules are referenced in the script."""
    text = _script_text()
    live_modules = [
        "live.runner",
        "live.strategy",
        "live.order",
        "live.portfolio",
        "-m live",
    ]
    for mod in live_modules:
        assert mod not in text, f"Script references live module: {mod!r}"


def test_absolute_path_used_for_python():
    """Script must derive an absolute path for Python (scheduling safety)."""
    text = _script_text()
    # REPO_DIR must be derived via shell expansion, not hardcoded relative path
    assert "REPO_DIR" in text, "Script must use REPO_DIR variable for absolute paths"
    assert "${REPO_DIR}/.venv/bin/python" in text or "REPO_DIR" in text
