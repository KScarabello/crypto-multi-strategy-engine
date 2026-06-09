"""Tests for frozen candidate specifications."""
from __future__ import annotations

import pytest
from research.shadow.specs import (
    FROZEN_CANDIDATES,
    FROZEN_REGIME,
    SPEC_HASHES,
    PROSPECTIVE_START,
    SpecificationError,
    compute_spec_hash,
    verify_spec,
)


def test_all_candidates_have_hashes():
    for name in FROZEN_CANDIDATES:
        assert name in SPEC_HASHES
        assert isinstance(SPEC_HASHES[name], str)
        assert len(SPEC_HASHES[name]) > 0


def test_spec_hash_is_deterministic():
    for name in FROZEN_CANDIDATES:
        h1 = compute_spec_hash(name)
        h2 = compute_spec_hash(name)
        assert h1 == h2


def test_spec_hash_changes_on_field_change():
    import hashlib, json
    name = "control"
    spec = FROZEN_CANDIDATES[name]
    original_hash = compute_spec_hash(name)

    # Compute hash with a modified spec
    modified_spec = {**spec, "description": "MODIFIED"}
    canonical = json.dumps({"name": name, **modified_spec}, sort_keys=True, default=str)
    modified_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert original_hash != modified_hash


def test_unknown_candidate_raises_error():
    with pytest.raises(SpecificationError):
        compute_spec_hash("nonexistent_candidate")


def test_frozen_regime_parameters_match_in_sample():
    assert FROZEN_REGIME["btc_ma_bars"] == 240
    assert FROZEN_REGIME["rebalance_bars"] == 12
    assert FROZEN_REGIME["entry_confirm_bars"] == 2
    assert FROZEN_REGIME["exit_confirm_bars"] == 1
    assert FROZEN_REGIME["entry_buffer_pct"] == 0.5
    assert FROZEN_REGIME["top_n"] == 3
    assert FROZEN_REGIME["short_lookback_bars"] == 12
    assert FROZEN_REGIME["medium_lookback_bars"] == 36
    assert FROZEN_REGIME["min_history_bars"] == 36


def test_control_has_no_stability_rules():
    rs = FROZEN_CANDIDATES["control"]["rank_stability"]
    assert rs["rank_buffer"] == 0
    assert rs["challenger_confirm_rebs"] == 0
    assert rs["min_hold_rebs"] == 0


def test_min_hold_6_has_min_hold_rebs_6():
    rs = FROZEN_CANDIDATES["min_hold_6"]["rank_stability"]
    assert rs["min_hold_rebs"] == 6


def test_rank_buffer_4_has_rank_buffer_4():
    rs = FROZEN_CANDIDATES["rank_buffer_4"]["rank_stability"]
    assert rs["rank_buffer"] == 4


def test_combo_has_all_three_rules():
    rs = FROZEN_CANDIDATES["combo_buf4_conf2_hold2"]["rank_stability"]
    assert rs["rank_buffer"] == 4
    assert rs["challenger_confirm_rebs"] == 2
    assert rs["min_hold_rebs"] == 2


def test_spec_hash_length_is_16_chars():
    for name in FROZEN_CANDIDATES:
        assert len(SPEC_HASHES[name]) == 16


def test_spec_hashes_are_all_unique():
    hashes = list(SPEC_HASHES.values())
    assert len(hashes) == len(set(hashes))


def test_prospective_start_is_timezone_aware():
    import pandas as pd
    ts = pd.Timestamp(PROSPECTIVE_START)
    assert ts.tzinfo is not None
