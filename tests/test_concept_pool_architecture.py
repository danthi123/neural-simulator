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
    """enable_verb_pools=True with default names creates 2 pools.

    Note: build_biological_brain_regions defaults to ["GO","COME"] but
    concept_pool_demo overrides to 4 pools (GO/COME/STOP/LOOK) for FS
    symmetry. This test exercises the architecture-level default.
    """
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
    """verb FS does NOT inhibit motor or noun pools (design choice for composition).

    Uses default 2 verb pools (build_biological_brain_regions default).
    """
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

    # 2 verb pools (default), each FS inhibits 1 other = 2 total
    assert len(verb_fs_paths) == 2


def test_4_verb_pools_for_fs_symmetry():
    """With 4 verb pool names, FS symmetry matches motor/noun (3 cross-edges per FS).

    Critical for concept_pool_demo v2 (2026-05-13): the 2-pool default
    has 1-cross-edge per verb_FS which causes structural firing bias
    (verb_pool_COME dominated all 10 words in seed 42 v1 run). Fix is
    to use 4 verb pools so each FS has 3 cross-edges like noun/motor.
    """
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        verb_pool_names=["GO", "COME", "STOP", "LOOK"],
        **SMALL_CFG,
    )
    verb_pools = [r for r in regions
                  if r.name.startswith("verb_pool_")
                  and not r.name.endswith("_fs")]
    assert len(verb_pools) == 4

    # 4 verb pools, each FS inhibits 3 others = 12 cross-edges
    verb_fs_paths = [
        p for p in pathways
        if p.from_region.startswith("verb_pool_")
        and p.from_region.endswith("_fs")
    ]
    assert len(verb_fs_paths) == 12


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


# ============================================================================
# v15 unidirectional dlpfc_verb -> motor gating tests (2026-05-13 night)
# ============================================================================
# v15 fixes the v12 bidirectional feedback leakage by making verb_pool ->
# dlpfc_verb FORWARD ONLY (no back-feedback) and adding new dlpfc_verb ->
# motor_X gating pathways. The biology: PFC receives concept content via
# feedforward pathways, maintains via internal NMDA bistability, and gates
# downstream motor selection — without back-broadcasting to upstream concept
# areas (catalog G.06/G.08).


def test_v15_default_off_no_unidirectional_pathways():
    """Default behavior: enable_dlpfc_verb_unidirectional=False produces
    no v15 wiring."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_dlpfc_verb=True,
        **SMALL_CFG,
    )
    # No verb_pool -> dlpfc_verb pathways without the flag
    fwd = [p for p in pathways
           if p.from_region.startswith("verb_pool_")
           and p.to_region == "dlpfc_verb"]
    assert fwd == []
    # No dlpfc_verb -> motor_X pathways without the flag
    gate = [p for p in pathways
            if p.from_region == "dlpfc_verb"
            and p.to_region.startswith("motor_")
            and not p.to_region.startswith("motor_FS_")]
    assert gate == []


def test_v15_unidirectional_adds_forward_verb_to_dlpfc():
    """v15: verb_pool_X -> dlpfc_verb pathways exist (one per verb)."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        verb_pool_names=["GO", "COME", "STOP", "LOOK"],
        enable_dlpfc_verb=True,
        enable_dlpfc_verb_unidirectional=True,
        **SMALL_CFG,
    )
    fwd = [p for p in pathways
           if p.from_region.startswith("verb_pool_")
           and p.to_region == "dlpfc_verb"]
    fwd_names = sorted(p.from_region for p in fwd)
    assert fwd_names == [
        "verb_pool_COME", "verb_pool_GO",
        "verb_pool_LOOK", "verb_pool_STOP",
    ]
    # All forward pathways are plastic
    assert all(p.plastic for p in fwd)
    # All forward pathways are gated for selective training
    assert all(p.plasticity_gate == "verb_pool_to_dlpfc_uni" for p in fwd)


def test_v15_unidirectional_adds_dlpfc_to_motor():
    """v15: dlpfc_verb -> motor_X pathways exist (one per direction)."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_dlpfc_verb=True,
        enable_dlpfc_verb_unidirectional=True,
        **SMALL_CFG,
    )
    gate = [p for p in pathways
            if p.from_region == "dlpfc_verb"
            and p.to_region.startswith("motor_")
            and not p.to_region.startswith("motor_FS_")]
    gate_names = sorted(p.to_region for p in gate)
    assert gate_names == ["motor_E", "motor_N", "motor_S", "motor_W"]
    # All gating pathways are plastic
    assert all(p.plastic for p in gate)
    # All gating pathways are gated for selective training (compose window)
    assert all(p.plasticity_gate == "dlpfc_verb_to_motor_uni" for p in gate)


def test_v15_unidirectional_has_no_back_feedback():
    """v15 critical invariant: NO dlpfc_verb -> verb_pool_X back-feedback.

    This is the v12 leakage source — v15 fixes it by making the
    integration strictly feedforward.
    """
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_dlpfc_verb=True,
        enable_dlpfc_verb_unidirectional=True,
        **SMALL_CFG,
    )
    back_feed = [p for p in pathways
                 if p.from_region == "dlpfc_verb"
                 and p.to_region.startswith("verb_pool_")]
    assert back_feed == [], (
        f"v15 must NOT have dlpfc_verb -> verb_pool back-feedback "
        f"(this was the v12 leakage source). Found: {back_feed}"
    )
    # Also: no motor_X -> dlpfc_verb (motor cortex doesn't drive PFC
    # in this direction at the conceptual level)
    rev = [p for p in pathways
           if p.from_region.startswith("motor_")
           and not p.from_region.startswith("motor_FS_")
           and p.to_region == "dlpfc_verb"]
    assert rev == [], (
        f"v15 must NOT have motor_X -> dlpfc_verb (PFC receives concepts "
        f"from concept pools, not motor execution). Found: {rev}"
    )


# ============================================================================
# v16 direct verb_pool -> motor pathway tests (2026-05-13 night, post-v15)
# ============================================================================
# v16 abandons the PFC-region approach (v12/v15 both NEGATIVE on v14's
# reciprocal binding due to dlpfc_verb's 200-neuron perturbation of
# eligibility-trace state). v16 takes the simplest possible compositional
# approach: direct verb_pool_X -> motor_Y plastic pathways. NO new region.
# 16 plastic pathways (4 verbs × 4 motors). Zero-init + zero-jitter so
# Phase 1 is unaffected. Compose training opens the shared gate and drives
# (verb_word, motor_word) co-firing; STDP grows weights from 0.


def test_v16_default_off_no_direct_pathways():
    """Default behavior: enable_direct_verb_to_motor=False produces no
    direct verb -> motor pathways."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True, **SMALL_CFG,
    )
    direct = [p for p in pathways
              if p.from_region.startswith("verb_pool_")
              and p.to_region.startswith("motor_")
              and not p.to_region.startswith("motor_FS_")]
    assert direct == []


def test_v16_flag_on_adds_16_pathways():
    """v16: 4 verbs × 4 motors = 16 direct verb -> motor pathways."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        verb_pool_names=["GO", "COME", "STOP", "LOOK"],
        enable_direct_verb_to_motor=True,
        **SMALL_CFG,
    )
    direct = [p for p in pathways
              if p.from_region.startswith("verb_pool_")
              and p.to_region.startswith("motor_")
              and not p.to_region.startswith("motor_FS_")]
    assert len(direct) == 16
    # Each verb -> each motor exactly once
    pairs = set((p.from_region, p.to_region) for p in direct)
    expected = {(f"verb_pool_{v}", f"motor_{m}")
                for v in ["GO", "COME", "STOP", "LOOK"]
                for m in ["N", "E", "S", "W"]}
    assert pairs == expected


def test_v16_all_zero_init():
    """v16 critical invariant: ALL 16 pathways have weight_mean=0.0 +
    weight_jitter=0.0. This ensures the pathways are STRUCTURALLY
    present but FUNCTIONALLY silent until compose training opens the
    gate and grows weights via co-firing STDP. Phase 1 W->A and
    Phase 3 A->W must be preserved."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_direct_verb_to_motor=True,
        **SMALL_CFG,
    )
    direct = [p for p in pathways
              if p.from_region.startswith("verb_pool_")
              and p.to_region.startswith("motor_")
              and not p.to_region.startswith("motor_FS_")]
    assert len(direct) > 0
    for p in direct:
        assert p.weight_mean == 0.0, (
            f"v16 pathway {p.from_region}->{p.to_region} must default to "
            f"weight_mean=0.0 (got {p.weight_mean}). v15a regression at "
            f"weight_mean=2.0 + jitter=0.2 produced 8/16 Phase 1."
        )
        assert p.weight_jitter == 0.0, (
            f"v16 pathway {p.from_region}->{p.to_region} must default to "
            f"weight_jitter=0.0 (got {p.weight_jitter}). Non-zero jitter "
            f"injects noise current during Phase 1 even with weight_mean=0."
        )


def test_v16_shared_gate_name():
    """All v16 pathways share one plasticity gate name so compose
    training can open them all atomically."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_direct_verb_to_motor=True,
        **SMALL_CFG,
    )
    direct = [p for p in pathways
              if p.from_region.startswith("verb_pool_")
              and p.to_region.startswith("motor_")
              and not p.to_region.startswith("motor_FS_")]
    gate_names = set(p.plasticity_gate for p in direct)
    assert gate_names == {"verb_to_motor_direct"}, (
        f"v16 pathways must share gate name 'verb_to_motor_direct' "
        f"so compose training can open atomically. Got: {gate_names}"
    )


def test_v16_independent_of_v15():
    """v16 direct pathway is independent of v15 dlpfc unidirectional.
    Either, both, or neither can be enabled; v16 adds verb -> motor,
    v15 adds verb -> dlpfc -> motor. No interaction."""
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    # v16 only — no v15
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_direct_verb_to_motor=True,
        enable_dlpfc_verb=False,
        enable_dlpfc_verb_unidirectional=False,
        **SMALL_CFG,
    )
    # v16 pathways present
    direct = [p for p in pathways
              if p.from_region.startswith("verb_pool_")
              and p.to_region.startswith("motor_")
              and not p.to_region.startswith("motor_FS_")]
    assert len(direct) > 0
    # No dlpfc_verb region
    has_dlpfc = any(r.name == "dlpfc_verb" for r in regions)
    assert not has_dlpfc
    # No v15 pathways either
    v15_pathways = [p for p in pathways
                    if p.to_region == "dlpfc_verb"
                    or p.from_region == "dlpfc_verb"]
    assert v15_pathways == []


def test_v15_independent_of_v12_bidirectional():
    """v15 unidirectional + v12 bidirectional are independent flags.

    Either can be enabled; if both, both wirings apply (not recommended,
    but architecturally legal).
    """
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        enable_verb_pools=True,
        enable_dlpfc_verb=True,
        enable_dlpfc_verb_concept_integration=False,
        enable_dlpfc_verb_unidirectional=True,
        **SMALL_CFG,
    )
    # v15 wiring present
    fwd_uni = [p for p in pathways
               if p.from_region.startswith("verb_pool_")
               and p.to_region == "dlpfc_verb"
               and p.plasticity_gate == "verb_pool_to_dlpfc_uni"]
    assert len(fwd_uni) >= 2  # at least GO + COME default
    # v12 wiring absent
    back_v12 = [p for p in pathways
                if p.from_region == "dlpfc_verb"
                and p.to_region.startswith("verb_pool_")]
    assert back_v12 == []
