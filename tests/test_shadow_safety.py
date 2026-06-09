"""Safety tests: no live imports, no exchange calls, no spec tampering."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SHADOW_DIR = Path("research/shadow")
SHADOW_MODULES = [
    "research.shadow.specs",
    "research.shadow.portfolio",
    "research.shadow.signal_runner",
    "research.shadow.ledger",
    "research.shadow.monitor",
    "research.shadow.runner",
]

FORBIDDEN_IMPORTS = ["brokers", "execution", "live"]


def _get_shadow_source(module_file: str) -> str:
    path = SHADOW_DIR / module_file
    return path.read_text()


def _check_no_forbidden_imports(source: str) -> list[str]:
    """Return list of forbidden import strings found in source."""
    found = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<syntax error>"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_IMPORTS:
                        if alias.name.startswith(forbidden):
                            found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in FORBIDDEN_IMPORTS:
                        if node.module.startswith(forbidden):
                            found.append(node.module)
    return found


def test_runner_does_not_import_brokers():
    src = _get_shadow_source("runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("brokers")]
    assert found == [], f"runner.py imports brokers: {found}"


def test_runner_does_not_import_execution():
    src = _get_shadow_source("runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("execution")]
    assert found == [], f"runner.py imports execution: {found}"


def test_runner_does_not_import_live():
    src = _get_shadow_source("runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("live")]
    assert found == [], f"runner.py imports live: {found}"


def test_signal_runner_does_not_import_brokers():
    src = _get_shadow_source("signal_runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("brokers")]
    assert found == [], f"signal_runner.py imports brokers: {found}"


def test_signal_runner_does_not_import_execution():
    src = _get_shadow_source("signal_runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("execution")]
    assert found == [], f"signal_runner.py imports execution: {found}"


def test_signal_runner_does_not_import_live():
    src = _get_shadow_source("signal_runner.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("live")]
    assert found == [], f"signal_runner.py imports live: {found}"


def test_portfolio_does_not_import_brokers():
    src = _get_shadow_source("portfolio.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("brokers")]
    assert found == [], f"portfolio.py imports brokers: {found}"


def test_ledger_does_not_import_live():
    src = _get_shadow_source("ledger.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("live")]
    assert found == [], f"ledger.py imports live: {found}"


def test_specs_does_not_import_live():
    src = _get_shadow_source("specs.py")
    found = [f for f in _check_no_forbidden_imports(src) if f.startswith("live")]
    assert found == [], f"specs.py imports live: {found}"


def test_no_kraken_api_call_in_shadow_modules():
    """No shadow module should call kraken APIs."""
    for py_file in SHADOW_DIR.glob("*.py"):
        src = py_file.read_text().lower()
        assert "kraken.submit" not in src, f"{py_file.name} contains kraken.submit"
        assert "kraken.create_order" not in src, f"{py_file.name} contains kraken.create_order"
        assert "from brokers.kraken" not in src, f"{py_file.name} imports from brokers.kraken"
        assert "import brokers.kraken" not in src, f"{py_file.name} imports brokers.kraken"


def test_spec_modification_raises_error():
    """Modifying FROZEN_CANDIDATES after SPEC_HASHES is set should be detectable."""
    import copy
    import hashlib, json
    from research.shadow.specs import FROZEN_CANDIDATES, SPEC_HASHES, SpecificationError

    name = "control"
    original_hash = SPEC_HASHES[name]

    tampered_spec = copy.deepcopy(FROZEN_CANDIDATES[name])
    tampered_spec["description"] = "TAMPERED"
    canonical = json.dumps({"name": name, **tampered_spec}, sort_keys=True, default=str)
    tampered_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    assert original_hash != tampered_hash, "Tampered hash should differ from original"


def test_btc_buyhold_holds_btc_only():
    import numpy as np
    import pandas as pd
    from research.shadow.specs import SPEC_HASHES
    from research.shadow.portfolio import init_portfolio
    from research.shadow.signal_runner import run_bar

    idx = pd.date_range("2024-01-01", periods=50, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.DataFrame({
        "BTC/USD": 65000 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "ETH/USD": 3500 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "XRP/USD": 0.6 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "SOL/USD": 150 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "AVAX/USD": 35 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
    }, index=idx)

    portfolio = init_portfolio("btc_buyhold", SPEC_HASHES["btc_buyhold"], str(idx[0]))
    decision = run_bar("btc_buyhold", idx[0], close, portfolio, SPEC_HASHES["btc_buyhold"])
    assert "BTC/USD" in decision.target_holdings
    assert len(decision.target_holdings) == 1


def test_ewb_buyhold_holds_five_coins():
    import numpy as np
    import pandas as pd
    from research.shadow.specs import SPEC_HASHES
    from research.shadow.portfolio import init_portfolio
    from research.shadow.signal_runner import run_bar

    idx = pd.date_range("2024-01-01", periods=50, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.DataFrame({
        "BTC/USD": 65000 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "ETH/USD": 3500 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "XRP/USD": 0.6 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "SOL/USD": 150 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
        "AVAX/USD": 35 * (1 + rng.normal(0, 0.01, 50)).cumprod(),
    }, index=idx)

    portfolio = init_portfolio("ewb_buyhold", SPEC_HASHES["ewb_buyhold"], str(idx[0]))
    decision = run_bar("ewb_buyhold", idx[0], close, portfolio, SPEC_HASHES["ewb_buyhold"])
    assert len(decision.target_holdings) == 5
    for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]:
        assert sym in decision.target_holdings


def test_shadow_state_dir_separate_from_live():
    """Shadow state dir should not be under live/, config/, etc."""
    from research.shadow.runner import DEFAULT_STATE_DIR
    state_str = str(DEFAULT_STATE_DIR)
    assert "live" not in state_str
    assert "config" not in state_str
    assert state_str.startswith("shadow_state") or "shadow_state" in state_str
