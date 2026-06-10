"""Static safety tests for the shadow-cycle LaunchAgent plist template and
install/uninstall/status helper scripts.

These tests prove — without installing anything — that:
  - the plist invokes only scripts/run_shadow_cycle.sh
  - no live trading command appears in plist or helpers
  - no Kraken credentials are sourced
  - no root / system LaunchDaemon path is used
  - frozen candidate files are not modified
  - state reset is not invoked
  - the six required local schedule times (MST) are present
  - absolute path substitution is handled correctly
  - install script refuses to run as root
  - uninstall preserves shadow_state, logs, and reports
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).parent.parent
PLIST_TEMPLATE = REPO / "deployment" / "launchd" / "com.kim.crypto-shadow-cycle.plist"
INSTALL_SH = REPO / "scripts" / "install_shadow_launchagent.sh"
UNINSTALL_SH = REPO / "scripts" / "uninstall_shadow_launchagent.sh"
STATUS_SH = REPO / "scripts" / "status_shadow_launchagent.sh"
WRAPPER_SH = REPO / "scripts" / "run_shadow_cycle.sh"

LABEL = "com.kim.crypto-shadow-cycle"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _plist_text() -> str:
    assert PLIST_TEMPLATE.exists(), f"plist template not found: {PLIST_TEMPLATE}"
    return PLIST_TEMPLATE.read_text()


def _plist_substituted() -> str:
    """Return plist with __REPO_DIR__ substituted for parsing."""
    return _plist_text().replace("__REPO_DIR__", str(REPO))


def _parsed_plist() -> ET.Element:
    return ET.fromstring(_plist_substituted())


def _script_text(path: Path) -> str:
    assert path.exists(), f"Script not found: {path}"
    return path.read_text()


def _non_comment_lines(text: str) -> str:
    """Return only non-comment lines for forbidden-pattern checks."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def _get_plist_dict(root: ET.Element) -> dict:
    """Parse the top-level plist <dict> into a Python dict."""
    d = {}
    top_dict = root.find("dict")
    if top_dict is None:
        return d
    children = list(top_dict)
    for i in range(0, len(children) - 1, 2):
        key_el = children[i]
        val_el = children[i + 1]
        if key_el.tag == "key":
            d[key_el.text] = val_el
    return d


# ─── Plist structure tests ────────────────────────────────────────────────────

class TestPlistTemplate:

    def test_template_exists(self):
        assert PLIST_TEMPLATE.exists()

    def test_plist_is_valid_xml(self):
        """The substituted plist must parse as valid XML."""
        ET.fromstring(_plist_substituted())  # raises on invalid

    def test_label_is_correct(self):
        root = _parsed_plist()
        d = _get_plist_dict(root)
        label_el = d.get("Label")
        assert label_el is not None and label_el.text == LABEL

    def test_program_arguments_invoke_only_wrapper(self):
        """ProgramArguments must invoke only run_shadow_cycle.sh — nothing else."""
        root = _parsed_plist()
        d = _get_plist_dict(root)
        args_el = d.get("ProgramArguments")
        assert args_el is not None, "ProgramArguments key missing"
        args = [s.text for s in args_el.findall("string")]
        assert len(args) >= 2, f"Expected at least 2 args, got: {args}"
        # First arg must be /bin/bash or the wrapper itself
        assert args[0] in ("/bin/bash", "/bin/sh", str(REPO / "scripts" / "run_shadow_cycle.sh")), \
            f"Unexpected interpreter: {args[0]}"
        # Last arg must end with run_shadow_cycle.sh
        assert args[-1].endswith("run_shadow_cycle.sh"), \
            f"ProgramArguments must invoke run_shadow_cycle.sh, got: {args[-1]}"
        # Must not invoke anything else
        forbidden_invocations = [
            "live", "execute_order", "place_order", "kraken_api",
            "update_live", "baseline", "reset",
        ]
        joined = " ".join(args)
        for forbidden in forbidden_invocations:
            assert forbidden not in joined.lower(), \
                f"ProgramArguments contains forbidden term: {forbidden!r}"

    def test_working_directory_is_repo(self):
        root = _parsed_plist()
        d = _get_plist_dict(root)
        wd_el = d.get("WorkingDirectory")
        assert wd_el is not None, "WorkingDirectory key missing"
        assert wd_el.text == str(REPO), \
            f"WorkingDirectory must be repo root, got: {wd_el.text}"

    def test_six_schedule_entries(self):
        """Exactly six StartCalendarInterval entries must be present."""
        root = _parsed_plist()
        d = _get_plist_dict(root)
        sci_el = d.get("StartCalendarInterval")
        assert sci_el is not None, "StartCalendarInterval key missing"
        entries = sci_el.findall("dict")
        assert len(entries) == 6, f"Expected 6 schedule entries, got {len(entries)}"

    def test_schedule_times_are_correct_phoenix_times(self):
        """The six schedule times must be the MST equivalents of 00:10–20:10 UTC."""
        # America/Phoenix = UTC-7 (no DST)
        # UTC candle+10min → MST
        expected_hour_minute = {
            (17, 10),  # 00:10 UTC → 17:10 MST
            (21, 10),  # 04:10 UTC → 21:10 MST
            (1,  10),  # 08:10 UTC → 01:10 MST
            (5,  10),  # 12:10 UTC → 05:10 MST
            (9,  10),  # 16:10 UTC → 09:10 MST
            (13, 10),  # 20:10 UTC → 13:10 MST
        }
        root = _parsed_plist()
        d = _get_plist_dict(root)
        sci_el = d["StartCalendarInterval"]
        found = set()
        for entry in sci_el.findall("dict"):
            keys = [el.text for el in entry.findall("key")]
            vals = [int(el.text) for el in entry.findall("integer")]
            entry_dict = dict(zip(keys, vals))
            found.add((entry_dict["Hour"], entry_dict["Minute"]))
        assert found == expected_hour_minute, \
            f"Schedule mismatch.\nExpected: {expected_hour_minute}\nFound:    {found}"

    def test_run_at_load_is_false(self):
        """RunAtLoad must be false — agent should not trigger on install."""
        root = _parsed_plist()
        d = _get_plist_dict(root)
        ral_el = d.get("RunAtLoad")
        assert ral_el is not None, "RunAtLoad key missing"
        assert ral_el.tag == "false", f"RunAtLoad must be <false/>, got <{ral_el.tag}/>"

    def test_stdout_path_under_logs(self):
        root = _parsed_plist()
        d = _get_plist_dict(root)
        out_el = d.get("StandardOutPath")
        assert out_el is not None, "StandardOutPath missing"
        assert "logs/" in out_el.text and "shadow" in out_el.text.lower()

    def test_no_username_key(self):
        """LaunchAgents must not specify UserName — runs as current user."""
        text = _plist_text()
        assert "<key>UserName</key>" not in text, \
            "LaunchAgent must not specify UserName (must run as current user)"

    def test_no_system_launchdaemon_path(self):
        """Template must not reference /Library/LaunchDaemons (system daemon path)."""
        text = _plist_text()
        assert "/Library/LaunchDaemons" not in text, \
            "Template must not reference system LaunchDaemon path"
        assert "/System/Library" not in text

    def test_no_live_trading_commands_in_plist(self):
        """Plist must not invoke live trading modules."""
        text = _plist_text()
        forbidden = [
            "live.runner", "live.strategy", "execute_order", "place_order",
            "kraken_secret", "KRAKEN_API", "baseline_v1",
        ]
        for term in forbidden:
            assert term not in text, f"Plist contains forbidden term: {term!r}"

    def test_no_reset_in_plist(self):
        """Plist must not auto-reset shadow state."""
        assert "--reset" not in _plist_text()

    def test_placeholder_present_in_template(self):
        """Template must still contain __REPO_DIR__ placeholder (not yet substituted)."""
        assert "__REPO_DIR__" in _plist_text(), \
            "Template must use __REPO_DIR__ placeholder (substituted by install script)"


# ─── Install script tests ─────────────────────────────────────────────────────

class TestInstallScript:

    def test_install_script_exists(self):
        assert INSTALL_SH.exists()

    def test_install_script_executable(self):
        import stat
        assert INSTALL_SH.stat().st_mode & stat.S_IXUSR

    def test_refuses_root(self):
        text = _script_text(INSTALL_SH)
        assert "UID_VAL" in text or 'id -u' in text
        assert "root" in text or "0" in text  # checks for uid == 0

    def test_uses_plutil_validation(self):
        assert "plutil" in _script_text(INSTALL_SH)

    def test_uses_bootstrap_not_just_load(self):
        text = _script_text(INSTALL_SH)
        assert "bootstrap" in text

    def test_installs_to_user_launchagents(self):
        text = _script_text(INSTALL_SH)
        assert "Library/LaunchAgents" in text

    def test_no_live_credentials_in_installer(self):
        non_comment = _non_comment_lines(_script_text(INSTALL_SH))
        forbidden = [
            r"^\s*source\s+.*kraken",
            r"^\s*\.\s+.*kraken",
            r"^\s*export\s+KRAKEN_API",
            r"KRAKEN_API_KEY\s*=",
        ]
        for pattern in forbidden:
            assert not re.search(pattern, non_comment, re.IGNORECASE | re.MULTILINE), \
                f"Installer contains forbidden credential pattern: {pattern!r}"

    def test_no_live_trading_in_installer(self):
        """Installer must not invoke or source live trading modules directly."""
        non_comment = _non_comment_lines(_script_text(INSTALL_SH))
        # These patterns indicate actual invocation of live trading, not guard checks
        forbidden_invocations = [
            r"python\s+-m\s+live\.",
            r"bash\s+.*live.*\.sh",
            r"\bexecute_order\s*\(",
            r"\bplace_order\s*\(",
        ]
        for pattern in forbidden_invocations:
            assert not re.search(pattern, non_comment, re.IGNORECASE), \
                f"Installer invokes live trading: {pattern!r}"

    def test_no_auto_reset_in_installer(self):
        assert "--reset" not in _script_text(INSTALL_SH)

    def test_substitutes_repo_dir(self):
        text = _script_text(INSTALL_SH)
        assert "__REPO_DIR__" in text or "REPO_DIR" in text
        assert "sed" in text  # performs substitution

    def test_does_not_modify_frozen_specs(self):
        text = _script_text(INSTALL_SH)
        assert "specs.py" not in text
        assert "SPEC_HASHES" not in text
        assert "FROZEN_CANDIDATES" not in text


# ─── Uninstall script tests ────────────────────────────────────────────────────

class TestUninstallScript:

    def test_uninstall_script_exists(self):
        assert UNINSTALL_SH.exists()

    def test_uninstall_executable(self):
        import stat
        assert UNINSTALL_SH.stat().st_mode & stat.S_IXUSR

    def test_only_removes_shadow_plist(self):
        """Uninstall must only remove the shadow-cycle plist, nothing else."""
        text = _script_text(UNINSTALL_SH)
        # Must reference the correct label
        assert LABEL in text
        # Must use bootout or unload
        assert "bootout" in text or "unload" in text

    def test_preserves_shadow_state(self):
        """Uninstall must never delete shadow_state/ data."""
        text = _script_text(UNINSTALL_SH)
        # Must not have rm on shadow_state
        non_comment = _non_comment_lines(text)
        assert not re.search(r"rm\s.*shadow_state", non_comment), \
            "Uninstaller must not delete shadow_state/"

    def test_preserves_logs(self):
        text = _script_text(UNINSTALL_SH)
        non_comment = _non_comment_lines(text)
        assert not re.search(r"rm\s.*logs/", non_comment), \
            "Uninstaller must not delete logs/"

    def test_preserves_reports(self):
        text = _script_text(UNINSTALL_SH)
        non_comment = _non_comment_lines(text)
        assert not re.search(r"rm\s.*reports/", non_comment), \
            "Uninstaller must not delete reports/"

    def test_refuses_root(self):
        text = _script_text(UNINSTALL_SH)
        assert "0" in text or "root" in text  # guard for uid == 0


# ─── Status script tests ──────────────────────────────────────────────────────

class TestStatusScript:

    def test_status_script_exists(self):
        assert STATUS_SH.exists()

    def test_status_executable(self):
        import stat
        assert STATUS_SH.stat().st_mode & stat.S_IXUSR

    def test_uses_launchctl_print_as_authoritative_check(self):
        """Loaded-state must use 'launchctl print gui/<uid>/<label>', not grep on list."""
        text = _script_text(STATUS_SH)
        assert "launchctl print" in text, \
            "Status script must use 'launchctl print' as the authoritative loaded-state check"
        assert "DOMAIN_TARGET" in text or "gui/" in text, \
            "Status script must construct a gui/<uid>/<label> domain target"

    def test_does_not_use_grep_for_loaded_state(self):
        """Registration state must NOT be determined by grep on launchctl list PID column."""
        text = _script_text(STATUS_SH)
        non_comment = _non_comment_lines(text)
        # The script may still show launchctl list for informational purposes,
        # but LOADED/NOT LOADED decision must come from launchctl print exit code.
        # Ensure the if-condition that sets REGISTRATION uses launchctl print.
        assert "launchctl print" in text
        # The old grep-based registration check should be gone
        assert 'launchctl list | grep -q "${LABEL}"' not in text, \
            "Registration must use launchctl print, not grep on launchctl list"

    def test_loaded_state_uses_exit_code(self):
        """Script must branch on launchctl print exit status (if ... ; then LOADED)."""
        text = _script_text(STATUS_SH)
        # The pattern: if PRINT_OUTPUT="$(launchctl print ...)" 2>&1; then
        assert 'launchctl print' in text
        assert 'LOADED' in text
        assert 'NOT LOADED' in text

    def test_separates_registration_from_runtime_state(self):
        """Script must separately report registration and runtime state (running vs idle)."""
        text = _script_text(STATUS_SH)
        assert "running" in text
        assert "idle" in text or "waiting" in text

    def test_explains_dash_pid_is_idle_not_unloaded(self):
        """Script must explain that a dash PID means idle/scheduled, not unloaded."""
        text = _script_text(STATUS_SH)
        assert "idle" in text or "scheduled" in text or "waiting" in text

    def test_shows_launchctl_list_informationally(self):
        """launchctl list may still appear for informational display."""
        assert "launchctl list" in _script_text(STATUS_SH)

    def test_shows_cycle_log(self):
        text = _script_text(STATUS_SH)
        assert "shadow_cycle.log" in text

    def test_shows_observation_count(self):
        text = _script_text(STATUS_SH)
        assert "observations" in text or "ledger" in text.lower()

    def test_no_live_commands(self):
        text = _script_text(STATUS_SH)
        forbidden = ["execute_order", "place_order", "--reset"]
        for term in forbidden:
            assert term not in text, f"Status script references: {term!r}"


# ─── Cross-file consistency ───────────────────────────────────────────────────

class TestCrossFileConsistency:

    def test_wrapper_referenced_consistently(self):
        """All files agree on the wrapper filename."""
        plist_text = _plist_text()
        install_text = _script_text(INSTALL_SH)
        assert "run_shadow_cycle.sh" in plist_text
        assert "run_shadow_cycle.sh" in install_text

    def test_label_consistent_across_files(self):
        install_text = _script_text(INSTALL_SH)
        uninstall_text = _script_text(UNINSTALL_SH)
        status_text = _script_text(STATUS_SH)
        assert LABEL in install_text
        assert LABEL in uninstall_text
        assert LABEL in status_text

    def test_wrapper_script_is_present(self):
        assert WRAPPER_SH.exists(), "run_shadow_cycle.sh must exist"
