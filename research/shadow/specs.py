"""Frozen candidate specifications for prospective shadow validation."""
from __future__ import annotations

import hashlib
import json

PROSPECTIVE_START = "2026-06-09T00:00:00+00:00"
SHADOW_INITIAL_CAPITAL = 10_000.0

FROZEN_REGIME = {
    "btc_ma_bars": 240,
    "entry_confirm_bars": 2,
    "exit_confirm_bars": 1,
    "entry_buffer_pct": 0.5,
    "exit_buffer_pct": 0.0,
    "min_duration_bars": 0,
    "rebalance_bars": 12,
    "fee_bps": 10,
    "slippage_bps": 5,
    "top_n": 3,
    "short_lookback_bars": 12,
    "medium_lookback_bars": 36,
    "min_history_bars": 36,
    "universe": ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"],
}

FROZEN_CANDIDATES = {
    "control": {
        "description": "combo_entry2_imm_buf05 strict top-3 replacement",
        "regime": FROZEN_REGIME,
        "rank_stability": {
            "rank_buffer": 0,
            "challenger_confirm_rebs": 0,
            "score_hurdle": 0.0,
            "min_hold_rebs": 0,
            "cooldown_rebs": 0,
        },
    },
    "min_hold_6": {
        "description": "frozen regime + 6-rebalance minimum hold (~12 calendar days)",
        "regime": FROZEN_REGIME,
        "rank_stability": {
            "rank_buffer": 0,
            "challenger_confirm_rebs": 0,
            "score_hurdle": 0.0,
            "min_hold_rebs": 6,
            "cooldown_rebs": 0,
        },
    },
    "rank_buffer_4": {
        "description": "frozen regime + retain incumbent while rank <= 4",
        "regime": FROZEN_REGIME,
        "rank_stability": {
            "rank_buffer": 4,
            "challenger_confirm_rebs": 0,
            "score_hurdle": 0.0,
            "min_hold_rebs": 0,
            "cooldown_rebs": 0,
        },
    },
    "combo_buf4_conf2_hold2": {
        "description": "frozen regime + rank_buffer=4 + challenger_confirm=2 + min_hold=2",
        "regime": FROZEN_REGIME,
        "rank_stability": {
            "rank_buffer": 4,
            "challenger_confirm_rebs": 2,
            "score_hurdle": 0.0,
            "min_hold_rebs": 2,
            "cooldown_rebs": 0,
        },
    },
    "btc_buyhold": {
        "description": "BTC/USD buy-and-hold benchmark",
        "regime": None,
        "rank_stability": None,
    },
    "ewb_buyhold": {
        "description": "Equal-weight five-coin buy-and-hold benchmark",
        "regime": None,
        "rank_stability": None,
    },
}


class SpecificationError(Exception):
    """Raised when a candidate specification is invalid or has been tampered with."""


def compute_spec_hash(name: str) -> str:
    """Return SHA-256 hex digest (16 chars) of the frozen candidate specification."""
    if name not in FROZEN_CANDIDATES:
        raise SpecificationError(f"Unknown candidate: {name!r}")
    spec = FROZEN_CANDIDATES[name]
    canonical = json.dumps({"name": name, **spec}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def verify_spec(name: str, expected_hash: str) -> bool:
    """Return True if current spec hash matches expected_hash."""
    if name not in FROZEN_CANDIDATES:
        raise SpecificationError(f"Unknown candidate: {name!r}")
    return compute_spec_hash(name) == expected_hash


SPEC_HASHES: dict[str, str] = {
    name: compute_spec_hash(name) for name in FROZEN_CANDIDATES
}


def assert_spec_integrity(name: str) -> None:
    """Raise SpecificationError if spec has been modified after SPEC_HASHES was computed."""
    if name not in FROZEN_CANDIDATES:
        raise SpecificationError(f"Unknown candidate: {name!r}")
    current_hash = compute_spec_hash(name)
    expected_hash = SPEC_HASHES[name]
    if current_hash != expected_hash:
        raise SpecificationError(
            f"Specification tampered: {name!r} hash mismatch "
            f"(expected={expected_hash!r}, got={current_hash!r})"
        )


def assert_all_spec_integrity() -> None:
    """Raise SpecificationError if any spec has been modified."""
    for name in FROZEN_CANDIDATES:
        assert_spec_integrity(name)


if __name__ == "__main__":
    print("=== Frozen Candidate Specifications ===")
    print()
    for name, h in SPEC_HASHES.items():
        desc = FROZEN_CANDIDATES[name]["description"]
        print(f"  {name:<28} hash={h}  {desc}")
    print()
    print(f"Prospective start: {PROSPECTIVE_START}")
    print(f"Initial capital:   ${SHADOW_INITIAL_CAPITAL:,.0f}")
