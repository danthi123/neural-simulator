"""Tests for Tier 2.3 dlpfc_verb addition to build_biological_brain_regions.

Validates the opt-in PFC verb pool builder per design at
docs/plans/2026-05-06-Tier2.3-two-word-phrases-design.md.
"""
import pytest


def test_default_disabled():
    """Without enable_dlpfc_verb, no PFC region should be added."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, pathways = build_biological_brain_regions()
    region_names = [r.name for r in regions]
    assert "dlpfc_verb" not in region_names
    pathway_pairs = [(p.from_region, p.to_region) for p in pathways]
    assert ("language_input", "dlpfc_verb") not in pathway_pairs


def test_enabled_adds_region_and_pathway():
    """With enable_dlpfc_verb=True, exactly 1 region + 1 pathway added."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    base_r, base_p = build_biological_brain_regions()
    new_r, new_p = build_biological_brain_regions(enable_dlpfc_verb=True)
    assert len(new_r) == len(base_r) + 1
    assert len(new_p) == len(base_p) + 1


def test_dlpfc_verb_region_specs():
    """PFC verb pool should match design specs:
    200 neurons, exc_fraction 0.8, internal_density 0.15,
    plastic_internal=False (frozen recurrence)."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, _ = build_biological_brain_regions(enable_dlpfc_verb=True)
    dlpfc = next(r for r in regions if r.name == "dlpfc_verb")
    assert dlpfc.n_neurons == 200
    assert dlpfc.exc_fraction == 0.8
    assert dlpfc.internal_density == 0.15
    assert dlpfc.plastic_internal is False
    # Should use cortical pyramidal type (NMDA-bistable when
    # cfg.enable_nmda=True at the per-region scope)
    assert "PYRAMIDAL" in dlpfc.izh_neuron_type


def test_lang_to_dlpfc_verb_pathway_specs():
    """language_input -> dlpfc_verb pathway should be plastic with
    plasticity_gate='language_input_to_dlpfc_verb'."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    _, pathways = build_biological_brain_regions(enable_dlpfc_verb=True)
    p = next(p for p in pathways
             if p.from_region == "language_input"
             and p.to_region == "dlpfc_verb")
    assert p.plastic is True
    assert p.plasticity_gate == "language_input_to_dlpfc_verb"
    assert p.density == 0.30
    assert p.weight_mean == 2.0


def test_custom_dlpfc_verb_size():
    """Override default 200 neurons via parameter."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, _ = build_biological_brain_regions(
        enable_dlpfc_verb=True, n_dlpfc_verb=100,
    )
    dlpfc = next(r for r in regions if r.name == "dlpfc_verb")
    assert dlpfc.n_neurons == 100


def test_dlpfc_verb_composes_with_motor_fs():
    """Tier 2.3 PFC pool should compose cleanly with Tier 1 motor_FS."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, pathways = build_biological_brain_regions(
        enable_dlpfc_verb=True, enable_motor_fs=True,
    )
    region_names = [r.name for r in regions]
    assert "dlpfc_verb" in region_names
    # All 4 motor_FS regions present
    for action in ["N", "E", "S", "W"]:
        assert f"motor_FS_{action}" in region_names


def test_dlpfc_verb_composes_with_language_output():
    """Tier 2.3 should compose with Tier 1 embodied bidirectional."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, pathways = build_biological_brain_regions(
        enable_dlpfc_verb=True, enable_language_output=True,
    )
    region_names = [r.name for r in regions]
    assert "dlpfc_verb" in region_names
    assert "language_output" in region_names
    # All Tier 1 motor -> language_output pathways present
    motor_to_lang_out = [
        p for p in pathways
        if p.to_region == "language_output"
        and p.from_region.startswith("motor_")
    ]
    assert len(motor_to_lang_out) == 4


def test_action_gate_builder_default():
    """build_tier_2_3_action_gate returns a config with 4 motor targets
    and a from_region_firing rule on dlpfc_verb."""
    from research.runners.text_minimal_isolation import (
        build_tier_2_3_action_gate,
    )
    nm = build_tier_2_3_action_gate()
    assert nm.name == "action_gate"
    assert nm.decay_tau_ms == 300.0
    assert len(nm.targets) == 4
    target_scopes = {t.scope for t in nm.targets}
    assert target_scopes == {
        "group:motor_N", "group:motor_E",
        "group:motor_S", "group:motor_W",
    }
    for t in nm.targets:
        assert t.target_type == "excitability_drive"
    assert len(nm.production_rules) == 1
    rule = nm.production_rules[0]
    assert rule.rule_type == "from_region_firing"
    assert rule.source_regions == ["dlpfc_verb"]


def test_action_gate_custom_drive():
    """drive_pA parameter changes the per-target sensitivity."""
    from research.runners.text_minimal_isolation import (
        build_tier_2_3_action_gate,
    )
    nm = build_tier_2_3_action_gate(drive_pA=100.0)
    for t in nm.targets:
        assert t.sensitivity == 100.0


def test_action_gate_custom_decay():
    """decay_tau_ms parameter changes the working-memory timescale."""
    from research.runners.text_minimal_isolation import (
        build_tier_2_3_action_gate,
    )
    nm = build_tier_2_3_action_gate(decay_tau_ms=500.0)
    assert nm.decay_tau_ms == 500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
