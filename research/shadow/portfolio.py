"""Persistent shadow portfolio state — stored as JSON, never sent to exchange."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ShadowPortfolio:
    candidate_name: str
    spec_hash: str
    started_at: str
    last_updated_at: str
    equity: float
    holdings: dict[str, float]
    cash_weight: float
    bar_count: int
    rebalance_count: int
    total_fee_dollars: float
    total_slippage_dollars: float
    total_executed_notional: float
    rank_replacements: int
    regime_transitions: int
    hold_age: dict[str, int]
    challenger_streak: dict[str, int]
    cooldown: dict[str, int]
    last_gate: bool
    gate_entry_confirm_count: int
    equity_history: list[dict]
    prev_state_hash: str
    pending_decision: dict | None = field(default=None)


def compute_state_hash(portfolio: ShadowPortfolio) -> str:
    """Return SHA-256 hex digest (16 chars) of the portfolio state (excluding prev_state_hash)."""
    d = asdict(portfolio)
    d.pop("prev_state_hash", None)
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def init_portfolio(
    name: str,
    spec_hash: str,
    started_at: str,
    initial_capital: float = 10_000.0,
) -> ShadowPortfolio:
    """Create a fresh portfolio for a candidate."""
    p = ShadowPortfolio(
        candidate_name=name,
        spec_hash=spec_hash,
        started_at=started_at,
        last_updated_at=started_at,
        equity=initial_capital,
        holdings={},
        cash_weight=1.0,
        bar_count=0,
        rebalance_count=0,
        total_fee_dollars=0.0,
        total_slippage_dollars=0.0,
        total_executed_notional=0.0,
        rank_replacements=0,
        regime_transitions=0,
        hold_age={},
        challenger_streak={},
        cooldown={},
        last_gate=False,
        gate_entry_confirm_count=0,
        equity_history=[],
        prev_state_hash="",
        pending_decision=None,
    )
    p.prev_state_hash = compute_state_hash(p)
    return p


def _portfolio_path(name: str, state_dir: Path) -> Path:
    return state_dir / "portfolios" / f"{name}.json"


def load_portfolio(name: str, state_dir: Path) -> ShadowPortfolio | None:
    """Load portfolio from JSON. Returns None if file doesn't exist."""
    path = _portfolio_path(name, state_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ShadowPortfolio(**data)


def save_portfolio(portfolio: ShadowPortfolio, state_dir: Path) -> str:
    """Save portfolio to JSON, set prev_state_hash. Returns new state hash."""
    path = _portfolio_path(portfolio.candidate_name, state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_hash = compute_state_hash(portfolio)
    portfolio.prev_state_hash = new_hash
    path.write_text(json.dumps(asdict(portfolio), indent=2, default=str))
    return new_hash
