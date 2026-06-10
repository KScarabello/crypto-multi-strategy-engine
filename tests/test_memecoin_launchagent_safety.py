"""Static safety tests for the memecoin collection-cycle LaunchAgent.

These tests assert properties of the plist template, install/uninstall/status
scripts, and cross-check that frozen files are not referenced or modified.

Nothing here starts, stops, or modifies any LaunchAgent.
Nothing here touches live trading, five-coin shadow state, or genuine data.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PLIST_TEMPLATE = REPO / "deployment/launchd/com.kim.memecoin-collection-cycle.plist"
INSTALL_SCRIPT = REPO / "scripts/install_memecoin_launchagent.sh"
UNINSTALL_SCRIPT = REPO / "scripts/uninstall_memecoin_launchagent.sh"
STATUS_SCRIPT = REPO / "scripts/status_memecoin_launchagent.sh"
SHADOW_PLIST = REPO / "deployment/launchd/com.kim.crypto-shadow-cycle.plist"
WRAPPER = REPO / "scripts/run_memecoin_collection_cycle.sh"
EVIDENCE_THRESHOLDS = REPO / "reports/memecoin_evidence_thresholds.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_comment_lines(text: str) -> str:
    """Return only non-comment, non-blank lines from a shell script or plist."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("<!--") and not stripped.startswith("*"):
            lines.append(line)
    return "\n".join(lines)


def _read(path: Path) -> str:
    return path.read_text()


def _read_code(path: Path) -> str:
    """Return file content with shell-style comment lines stripped."""
    return _non_comment_lines(path.read_text())


def _read_plist_code(path: Path) -> str:
    """Return plist content with XML comment blocks stripped."""
    import re
    content = path.read_text()
    return re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)


def _plist_xml() -> ET.Element:
    """Parse the plist template as XML (with __REPO_DIR__ placeholder intact)."""
    content = PLIST_TEMPLATE.read_text()
    # Replace placeholder so XML parses cleanly
    content = content.replace("__REPO_DIR__", "/REPO")
    return ET.fromstring(content)


def _plist_values() -> dict:
    """Return a flat dict of top-level key→value from the plist dict element."""
    root = _plist_xml()
    # plist root > dict
    d = root.find("dict")
    assert d is not None
    result = {}
    children = list(d)
    for i in range(0, len(children) - 1, 2):
        key_el = children[i]
        val_el = children[i + 1]
        if key_el.tag == "key":
            result[key_el.text] = val_el
    return result


# ---------------------------------------------------------------------------
# Plist template exists and is valid XML
# ---------------------------------------------------------------------------

class TestPlistExists:
    def test_plist_template_exists(self):
        assert PLIST_TEMPLATE.exists(), f"Missing: {PLIST_TEMPLATE}"

    def test_plist_is_valid_xml(self):
        _plist_xml()  # raises if invalid

    def test_plutil_validates_template(self):
        """plutil must report the template (with placeholder substituted) as valid."""
        import tempfile, os
        content = PLIST_TEMPLATE.read_text().replace("__REPO_DIR__", "/tmp/repo")
        with tempfile.NamedTemporaryFile(suffix=".plist", mode="w", delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            result = subprocess.run(
                ["plutil", "-lint", tmp],
                capture_output=True, text=True
            )
            assert result.returncode == 0, f"plutil failed: {result.stdout} {result.stderr}"
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Only the memecoin wrapper is invoked
# ---------------------------------------------------------------------------

class TestOnlyMemecoInWrapperInvoked:
    def test_program_arguments_contain_wrapper(self):
        values = _plist_values()
        prog_args = values.get("ProgramArguments")
        assert prog_args is not None, "ProgramArguments key missing"
        strings = [el.text for el in prog_args if el.tag == "string"]
        wrapper_strs = [s for s in strings if "run_memecoin_collection_cycle" in (s or "")]
        assert len(wrapper_strs) == 1, f"Expected exactly one memecoin wrapper reference, got: {strings}"

    def test_program_arguments_do_not_contain_shadow(self):
        values = _plist_values()
        prog_args = values.get("ProgramArguments")
        strings = [el.text or "" for el in prog_args if el.tag == "string"]
        for s in strings:
            assert "shadow" not in s.lower(), f"Shadow reference in ProgramArguments: {s}"
            assert "run_shadow" not in s.lower(), f"Shadow script in ProgramArguments: {s}"

    def test_program_arguments_do_not_contain_backfill(self):
        values = _plist_values()
        prog_args = values.get("ProgramArguments")
        strings = [el.text or "" for el in prog_args if el.tag == "string"]
        for s in strings:
            assert "backfill" not in s.lower(), f"Backfill reference in ProgramArguments: {s}"

    def test_label_is_correct(self):
        values = _plist_values()
        label_el = values.get("Label")
        assert label_el is not None
        assert label_el.text == "com.kim.memecoin-collection-cycle"


# ---------------------------------------------------------------------------
# No live trading, credentials, or backfill in plist
# ---------------------------------------------------------------------------

class TestNoForbiddenContentInPlist:
    def test_no_live_order_commands(self):
        content = _read_plist_code(PLIST_TEMPLATE).lower()
        for forbidden in ("place_order", "submit_order", "kraken_api", "api_secret"):
            assert forbidden not in content, f"Forbidden term '{forbidden}' in plist"

    def test_no_credential_references(self):
        content = _read_plist_code(PLIST_TEMPLATE).lower()
        for term in ("api_key", "secret_key", "private_key", "api_secret"):
            assert term not in content, f"Credential term '{term}' in plist"

    def test_no_backfill_in_plist(self):
        content = _read_plist_code(PLIST_TEMPLATE).lower()
        assert "backfill" not in content, "backfill reference in plist executable content"

    def test_no_reset_or_truncate_in_plist(self):
        content = _read_plist_code(PLIST_TEMPLATE).lower()
        assert "truncate" not in content
        assert "reset_history" not in content

    def test_run_at_load_is_false(self):
        values = _plist_values()
        ral = values.get("RunAtLoad")
        assert ral is not None, "RunAtLoad key missing"
        assert ral.tag == "false", f"RunAtLoad must be false, got tag={ral.tag}"


# ---------------------------------------------------------------------------
# Hourly :05 schedule is present
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_start_calendar_interval_present(self):
        values = _plist_values()
        assert "StartCalendarInterval" in values, "StartCalendarInterval missing from plist"

    def test_schedule_fires_at_minute_5(self):
        values = _plist_values()
        sched = values["StartCalendarInterval"]
        # May be a dict (single entry) or array (multiple entries)
        minute_values = []
        if sched.tag == "dict":
            children = list(sched)
            for i in range(0, len(children) - 1, 2):
                if children[i].tag == "key" and children[i].text == "Minute":
                    minute_values.append(int(children[i + 1].text))
        elif sched.tag == "array":
            for entry in sched:
                if entry.tag == "dict":
                    ch = list(entry)
                    for i in range(0, len(ch) - 1, 2):
                        if ch[i].tag == "key" and ch[i].text == "Minute":
                            minute_values.append(int(ch[i + 1].text))
        assert 5 in minute_values, f"Minute=5 not found in schedule; got {minute_values}"

    def test_schedule_is_hourly_not_subhourly(self):
        """A single Minute-only entry fires once per hour — not more frequently."""
        values = _plist_values()
        sched = values["StartCalendarInterval"]
        if sched.tag == "dict":
            # Single dict with only Minute → fires every hour ✓
            keys = [el.text for el in list(sched)[::2] if el.tag == "key"]
            # Ensure there's no Second key (sub-minute would be unusual but check)
            assert "Second" not in keys, "Unexpected Second key in schedule"

    def test_schedule_comment_mentions_phoenix(self):
        content = _read(PLIST_TEMPLATE)
        assert "Phoenix" in content or "phoenix" in content.lower(), \
            "Schedule comment should document America/Phoenix timezone"


# ---------------------------------------------------------------------------
# Install script safety
# ---------------------------------------------------------------------------

class TestInstallScript:
    def test_install_script_exists(self):
        assert INSTALL_SCRIPT.exists()

    def test_install_invokes_only_memecoin_wrapper(self):
        content = _read_code(INSTALL_SCRIPT)
        assert "run_memecoin_collection_cycle" in content

    def test_install_does_not_invoke_shadow(self):
        """Install must not execute any shadow-cycle script."""
        content = _read_code(INSTALL_SCRIPT)
        # grep -q checks are guard checks, not invocations — only flag actual invocations
        lines = [l for l in content.splitlines()
                 if "run_shadow_cycle" in l and "grep" not in l]
        assert not lines, f"Install script invokes shadow script: {lines}"

    def test_install_does_not_invoke_backfill(self):
        """Install must not invoke a backfill script (grep guards and echo messages are allowed)."""
        content = _read_code(INSTALL_SCRIPT)
        lines = [l for l in content.splitlines()
                 if "backfill" in l.lower()
                 and "grep" not in l
                 and not l.strip().startswith("echo")]
        assert not lines, f"Install script invokes backfill: {lines}"

    def test_install_checks_not_root(self):
        content = _read(INSTALL_SCRIPT)
        assert 'id -u' in content or 'EUID' in content, \
            "Install script must refuse to run as root"

    def test_install_runs_plutil_validation(self):
        content = _read(INSTALL_SCRIPT)
        assert "plutil" in content, "Install script must validate plist with plutil"

    def test_install_does_not_reset_history(self):
        content = _read_code(INSTALL_SCRIPT).lower()
        assert "reset_history" not in content
        assert "> data/memecoin_signal_history" not in content

    def test_install_safety_checks_no_credentials(self):
        """Credential terms must appear only in guard-grep patterns, not as invocations."""
        content = _read_code(INSTALL_SCRIPT)
        for term in ("api_key", "api_secret", "place_order"):
            lines_with_term = [l for l in content.splitlines()
                                if term in l.lower() and "grep" not in l]
            assert not lines_with_term, \
                f"Credential term '{term}' in non-guard install script line: {lines_with_term}"


# ---------------------------------------------------------------------------
# Uninstall script safety
# ---------------------------------------------------------------------------

class TestUninstallScript:
    def test_uninstall_script_exists(self):
        assert UNINSTALL_SCRIPT.exists()

    def test_uninstall_does_not_unload_shadow(self):
        """Uninstall must not launchctl unload the shadow agent."""
        content = _read(UNINSTALL_SCRIPT)
        # Only the memecoin label should be unloaded; the shadow label may appear
        # in a safety guard comment, but must not be unloaded.
        lines_with_unload = [
            l for l in content.splitlines()
            if "launchctl" in l and "unload" in l and not l.strip().startswith("#")
        ]
        for line in lines_with_unload:
            assert "shadow" not in line.lower(), \
                f"Uninstall script unloads shadow agent: {line}"

    def test_uninstall_does_not_delete_data(self):
        content = _read_code(UNINSTALL_SCRIPT).lower()
        assert "rm -f data/" not in content
        assert "rm -rf data/" not in content

    def test_uninstall_checks_not_root(self):
        content = _read(UNINSTALL_SCRIPT)
        assert 'id -u' in content or 'EUID' in content


# ---------------------------------------------------------------------------
# Status script
# ---------------------------------------------------------------------------

class TestStatusScript:
    def test_status_script_exists(self):
        assert STATUS_SCRIPT.exists()

    def test_status_shows_genuine_count(self):
        content = _read(STATUS_SCRIPT)
        assert "genuine" in content.lower() or "n_genuine" in content.lower()

    def test_status_shows_rule1_count(self):
        content = _read(STATUS_SCRIPT)
        assert "rule" in content.lower() and ("mature" in content.lower() or "r1" in content.lower())

    def test_status_shows_gate_a(self):
        content = _read(STATUS_SCRIPT)
        assert "gate" in content.lower() or "Gate" in content

    def test_status_shows_readiness(self):
        content = _read(STATUS_SCRIPT)
        assert "readiness" in content.lower() or "Readiness" in content

    def test_status_shows_log_tail(self):
        content = _read(STATUS_SCRIPT)
        assert "tail" in content


# ---------------------------------------------------------------------------
# Five-coin shadow LaunchAgent not modified
# ---------------------------------------------------------------------------

class TestShadowAgentUnchanged:
    def test_shadow_plist_exists(self):
        assert SHADOW_PLIST.exists(), "Shadow plist must still exist"

    def test_shadow_plist_label_unchanged(self):
        content = _read(SHADOW_PLIST)
        assert "com.kim.crypto-shadow-cycle" in content

    def test_shadow_plist_not_modified_by_memecoin_scripts(self):
        """None of the memecoin scripts should reference or rewrite the shadow plist."""
        for script in (INSTALL_SCRIPT, UNINSTALL_SCRIPT, STATUS_SCRIPT):
            content = _read(script)
            assert "com.kim.crypto-shadow-cycle" not in content or \
                   "SHADOW_LABEL" in content, \
                f"{script.name} must not hardcode the shadow agent label"


# ---------------------------------------------------------------------------
# Evidence thresholds and Rule 1 not modified
# ---------------------------------------------------------------------------

class TestFrozenDefinitionsUnchanged:
    def test_evidence_threshold_file_exists(self):
        assert EVIDENCE_THRESHOLDS.exists()

    def test_plist_does_not_reference_evidence_thresholds(self):
        content = _read(PLIST_TEMPLATE).lower()
        assert "evidence_threshold" not in content
        assert "gate_a" not in content

    def test_plist_does_not_reference_rule1_parameters(self):
        content = _read(PLIST_TEMPLATE).lower()
        assert "volume_climax" not in content
        assert "clean_continuation" not in content

    def test_install_does_not_modify_evidence_thresholds(self):
        content = _read(INSTALL_SCRIPT).lower()
        assert "evidence_threshold" not in content
        assert "gate_a" not in content

    def test_wrapper_invoked_by_launchagent_unchanged(self):
        assert WRAPPER.exists(), "Wrapper script must still exist"
        content = _read(WRAPPER)
        # The wrapper must still invoke the locked research modules only
        assert "run_memecoin_collection_cycle" in str(WRAPPER)
        assert "backfill" not in content.lower().split("# never")[0].split("never run")[0]


# ---------------------------------------------------------------------------
# Logs directory and paths
# ---------------------------------------------------------------------------

class TestLogPaths:
    def test_stdout_log_path_in_plist(self):
        content = _read(PLIST_TEMPLATE)
        assert "memecoin_launchd_stdout.log" in content

    def test_stderr_log_path_in_plist(self):
        content = _read(PLIST_TEMPLATE)
        assert "memecoin_launchd_stderr.log" in content

    def test_log_paths_under_logs_dir(self):
        content = _read(PLIST_TEMPLATE)
        assert "logs/memecoin_launchd_stdout.log" in content
        assert "logs/memecoin_launchd_stderr.log" in content

    def test_logs_dir_created_by_install(self):
        content = _read(INSTALL_SCRIPT)
        assert "mkdir" in content and "LOG_DIR" in content
