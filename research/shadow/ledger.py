"""Append-only evidence ledger — one JSONL file per candidate."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.shadow.signal_runner import BarDecision
from research.shadow.portfolio import ShadowPortfolio


def _ledger_path(candidate_name: str, state_dir: Path) -> Path:
    return state_dir / "ledger" / f"{candidate_name}.jsonl"


def _data_hash(close_row: dict | None) -> str:
    """SHA-256 of the last close row."""
    if close_row is None:
        return ""
    canonical = json.dumps(close_row, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def append_decision(
    candidate_name: str,
    decision: BarDecision,
    prev_portfolio: ShadowPortfolio,
    new_equity: float,
    actual_return: float,
    state_dir: Path,
    close_last_row: dict | None = None,
) -> None:
    """Append one JSON record to the ledger. Never overwrites. Idempotent."""
    if is_bar_already_processed(candidate_name, decision.decision_ts, state_dir):
        return

    path = _ledger_path(candidate_name, state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "run_ts": run_ts,
        "candidate_name": candidate_name,
        "spec_hash": decision.spec_hash,
        "decision_ts": decision.decision_ts,
        "execution_ts": decision.execution_ts,
        "data_cutoff_ts": decision.data_cutoff_ts,
        "prev_portfolio_hash": prev_portfolio.prev_state_hash,
        "new_equity": new_equity,
        "actual_return_pct": actual_return * 100.0,
        "btc_regime_state": decision.btc_regime_state,
        "should_rebalance": decision.should_rebalance,
        "target_holdings": decision.target_holdings,
        "hypothetical_trades": decision.hypothetical_trades,
        "data_hash": _data_hash(close_last_row),
        "is_prospective": decision.is_prospective,
    }

    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_ledger(candidate_name: str, state_dir: Path) -> list[dict]:
    """Read all records. Returns empty list if no file."""
    path = _ledger_path(candidate_name, state_dir)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def get_last_decision_ts(candidate_name: str, state_dir: Path) -> pd.Timestamp | None:
    """Find the most recent decision timestamp in the ledger."""
    records = load_ledger(candidate_name, state_dir)
    if not records:
        return None
    timestamps = [pd.Timestamp(r["decision_ts"]) for r in records]
    return max(timestamps)


def is_bar_already_processed(
    candidate_name: str,
    decision_ts: str,
    state_dir: Path,
) -> bool:
    """Check if a bar has already been processed (idempotency)."""
    records = load_ledger(candidate_name, state_dir)
    target = str(pd.Timestamp(decision_ts))
    for r in records:
        if str(pd.Timestamp(r["decision_ts"])) == target:
            return True
    return False
