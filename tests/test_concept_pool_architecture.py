"""Unit tests for concept pool architecture (2026-05-13).

Tests the extension to build_biological_brain_regions that adds:
- noun_pool_X regions + lang_input -> noun_pool plastic pathways
- verb_pool_X regions + lang_input -> verb_pool plastic pathways
- Optional FS cross-inhibition WITHIN kind (not across kinds, per
  design to allow composition like "go north" -> both pools fire)
- Reciprocal pool -> language_output for A->W readout

These tests verify the architectural assembly is correct.
A bridge-initialization integration test is in test_concept_pool_bridge.py.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Use a small config for speed (CPU-only fixture)
SMALL_CFG = dict(
    n_lang_input=256,
    n_motor_per_action=50,
    enable_motor_fs=True,
    n_motor_fs_per_action=6,
    enable_language_output=True,
    n_lang_output=256,
    n_noun_per_pool=50,
    n_noun_fs_per_pool=6,
    n_verb_per_pool=50,
    n_verb_fs_per_pool=6,
)


def test_default_off_no_noun_pools():
    """Default behavior: enable_noun_pools=False produces no noun regions."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(**SMALL_CFG)
    noun_regions = [r for r in regions if r.name.startswith("noun_pool_")]
    assert noun_regions == []
    noun_pathways = [p for p in pathways if p.to_region.startswith("noun_pool_")]
    assert noun_pathways == []


def test_default_off_no_verb_pools():
    """Default: enable_verb_pools=False produces no verb regions."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(**SMALL_CFG)
    verb_regions = [r for r in regions if r.name.startswith("verb_pool_")]
    assert verb_regions == []


def test_noun_pools_enabled_default_names():
    """enable_noun_pools=True with default names creates 4 pools."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True, **SMALL_CFG
    )
    noun_regions = [r for r in regions
                    if r.name.startswith("noun_pool_")
                    and not r.name.endswith("_fs")]
    names = sorted(r.name for r in noun_regions)
    assert names == [
        "noun_pool_APPLE", "noun_pool_CAT",
        "noun_pool_DOG", "noun_pool_RIVER",
    ]


def test_verb_pools_enabled_default_names():
    """enable_verb_pools=True with default names creates 2 pools."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True, **SMALL_CFG
    )
    verb_regions = [r for r in regions
                    if r.name.startswith("verb_pool_")
                    and not r.name.endswith("_fs")]
    names = sorted(r.name for r in verb_regions)
    assert names == ["verb_pool_COME", "verb_pool_GO"]


def test_noun_pools_custom_names():
    """Custom noun_pool_names override defaults."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        noun_pool_names=["FOO", "BAR"],
        **SMALL_CFG,
    )
    noun_regions = [r for r in regions
                    if r.name.startswith("noun_pool_")
                    and not r.name.endswith("_fs")]
    names = sorted(r.name for r in noun_regions)
    assert names == ["noun_pool_BAR", "noun_pool_FOO"]


def test_lang_input_to_noun_pathway_exists():
    """language_input -> noun_pool_X pathways are created and plastic."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True, **SMALL_CFG
    )
    lang_to_noun = [
        p for p in pathways
        if p.from_region == "language_input"
        and p.to_region.startswith("noun_pool_")
        and not p.to_region.endswith("_fs")
    ]
    assert len(lang_to_noun) == 4  # one per default noun
    for p in lang_to_noun:
        assert p.plastic is True
        assert p.plasticity_gate == "language_input_to_noun_pool"


def test_lang_input_to_verb_pathway_exists():
    """language_input -> verb_pool_X pathways are created and plastic."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True, **SMALL_CFG
    )
    lang_to_verb = [
        p for p in pathways
        if p.from_region == "language_input"
        and p.to_region.startswith("verb_pool_")
        and not p.to_region.endswith("_fs")
    ]
    assert len(lang_to_verb) == 2  # one per default verb
    for p in lang_to_verb:
        assert p.plastic is True
        assert p.plasticity_gate == "language_input_to_verb_pool"


def test_fs_cross_inhibition_within_kind():
    """noun FS inhibits OTHER noun pools (not own, not motor, not verb)."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **SMALL_CFG,
    )
    # FS-originated inhibitory pathways
    noun_fs_paths = [
        p for p in pathways
        if p.from_region.startswith("noun_pool_")
        and p.from_region.endswith("_fs")
    ]
    # Each FS should target OTHER noun pools (not self, not motor, not verb)
    for p in noun_fs_paths:
        from_pool = p.from_region[:-3]  # strip "_fs"
        # Must NOT target own pool
        assert p.to_region != from_pool
        # Must target a noun pool
        assert p.to_region.startswith("noun_pool_")
        # Must NOT target motor or verb pool
        assert not p.to_region.startswith("motor_")
        assert not p.to_region.startswith("verb_pool_")

    # Count: 4 noun pools, each FS inhibits 3 others = 12 total
    assert len(noun_fs_paths) == 12


def test_fs_does_not_cross_kinds():
    """verb FS does NOT inhibit motor or noun pools (design choice for composition)."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **SMALL_CFG,
    )
    # verb FS pathways
    verb_fs_paths = [
        p for p in pathways
        if p.from_region.startswith("verb_pool_")
        and p.from_region.endswith("_fs")
    ]
    for p in verb_fs_paths:
        assert p.to_region.startswith("verb_pool_")  # within kind
        # Not motor, not noun
        assert not p.to_region.startswith("motor_")
        assert not p.to_region.startswith("noun_pool_")

    # 2 verb pools, each FS inhibits 1 other = 2 total
    assert len(verb_fs_paths) == 2


def test_reciprocal_pool_to_lang_output_pathways():
    """noun_pool_X -> language_output and verb_pool_X -> language_output exist."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **SMALL_CFG,
    )
    # noun -> language_output
    noun_to_out = [
        p for p in pathways
        if p.from_region.startswith("noun_pool_")
        and not p.from_region.endswith("_fs")
        and p.to_region == "language_output"
    ]
    assert len(noun_to_out) == 4
    for p in noun_to_out:
        assert p.plastic is True
        assert p.plasticity_gate == "noun_pool_to_language_output"

    # verb -> language_output
    verb_to_out = [
        p for p in pathways
        if p.from_region.startswith("verb_pool_")
        and not p.from_region.endswith("_fs")
        and p.to_region == "language_output"
    ]
    assert len(verb_to_out) == 2
    for p in verb_to_out:
        assert p.plastic is True
        assert p.plasticity_gate == "verb_pool_to_language_output"


def test_no_lang_output_no_reciprocal_pathway():
    """If enable_language_output=False, no pool -> lang_output pathway exists."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    cfg = dict(SMALL_CFG)
    cfg["enable_language_output"] = False
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True, **cfg
    )
    recip = [
        p for p in pathways
        if p.to_region == "language_output"
    ]
    assert recip == []


def test_concept_pools_no_fs_when_motor_fs_disabled():
    """enable_motor_fs=False disables FS for concept pools too."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    cfg = dict(SMALL_CFG)
    cfg["enable_motor_fs"] = False
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True, **cfg
    )
    noun_fs_regions = [
        r for r in regions
        if r.name.startswith("noun_pool_") and r.name.endswith("_fs")
    ]
    assert noun_fs_regions == []


def test_total_pool_count_10_with_defaults():
    """4 motor + 4 noun + 2 verb = 10 distinct output pools."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **SMALL_CFG,
    )
    # Exclude FS regions (those don't count as output pools)
    output_pools = [
        r for r in regions
        if (r.name.startswith("motor_")
            and not r.name.startswith("motor_FS_"))
        or (r.name.startswith("noun_pool_")
            and not r.name.endswith("_fs"))
        or (r.name.startswith("verb_pool_")
            and not r.name.endswith("_fs"))
    ]
    assert len(output_pools) == 10  # 4 motor + 4 noun + 2 verb


def test_n_per_pool_overrides():
    """Per-kind n_per_pool sizes can differ from motor."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    cfg = dict(SMALL_CFG)
    cfg["n_noun_per_pool"] = 30
    cfg["n_verb_per_pool"] = 70
    regions, _ = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **cfg,
    )
    noun_regions = [r for r in regions
                    if r.name.startswith("noun_pool_")
                    and not r.name.endswith("_fs")]
    verb_regions = [r for r in regions
                    if r.name.startswith("verb_pool_")
                    and not r.name.endswith("_fs")]
    for r in noun_regions:
        assert r.n_neurons == 30
    for r in verb_regions:
        assert r.n_neurons == 70


def test_pool_kind_plasticity_gates_distinct():
    """Each pool kind has its OWN plasticity gate (so they can be
    frozen independently for staged experiments)."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        **SMALL_CFG,
    )
    # Get all gate names used by lang_input -> pool pathways
    gates = set()
    for p in pathways:
        if p.from_region == "language_input" and p.plastic:
            if p.plasticity_gate:
                gates.add(p.plasticity_gate)
    assert "language_input_to_motor" in gates
    assert "language_input_to_noun_pool" in gates
    assert "language_input_to_verb_pool" in gates


def test_adjective_pools_3rd_kind():
    """enable_adjective_pools=True adds 4 dedicated pools + FS within-kind."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_adjective_pools=True,
        **SMALL_CFG,
    )
    adj_pools = [r for r in regions
                 if r.name.startswith("adjective_pool_")
                 and not r.name.endswith("_fs")]
    names = sorted(r.name for r in adj_pools)
    assert names == [
        "adjective_pool_BIG", "adjective_pool_COLD",
        "adjective_pool_HOT", "adjective_pool_SMALL",
    ]
    # FS within-kind: 4 pools * 3 others = 12 cross-edges
    adj_fs_paths = [
        p for p in pathways
        if p.from_region.startswith("adjective_pool_")
        and p.from_region.endswith("_fs")
    ]
    assert len(adj_fs_paths) == 12
    # FS doesn't cross to noun/motor/verb
    for p in adj_fs_paths:
        assert p.to_region.startswith("adjective_pool_")


def test_all_3_concept_kinds_distinct_gates():
    """Each of 3 kinds has its own plasticity gate."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        enable_adjective_pools=True,
        **SMALL_CFG,
    )
    gates = set()
    for p in pathways:
        if p.from_region == "language_input" and p.plastic:
            if p.plasticity_gate:
                gates.add(p.plasticity_gate)
    assert "language_input_to_motor" in gates
    assert "language_input_to_noun_pool" in gates
    assert "language_input_to_verb_pool" in gates
    assert "language_input_to_adjective_pool" in gates


def test_total_pool_count_14_with_3_kinds():
    """4 motor + 4 noun + 2 verb + 4 adjective = 14 distinct output pools."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_noun_pools=True,
        enable_verb_pools=True,
        enable_adjective_pools=True,
        **SMALL_CFG,
    )
    output_pools = [
        r for r in regions
        if (r.name.startswith("motor_")
            and not r.name.startswith("motor_FS_"))
        or (r.name.startswith("noun_pool_")
            and not r.name.endswith("_fs"))
        or (r.name.startswith("verb_pool_")
            and not r.name.endswith("_fs"))
        or (r.name.startswith("adjective_pool_")
            and not r.name.endswith("_fs"))
    ]
    assert len(output_pools) == 14  # 4 motor + 4 noun + 2 verb + 4 adj
