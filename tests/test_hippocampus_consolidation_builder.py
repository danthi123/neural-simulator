"""Tests for Phase 1.3 hippocampus consolidation builder addition.

Validates the opt-in `enable_hippocampus_consolidation=True` flag
on `build_biological_brain_regions` per design at
docs/plans/2026-05-06-Phase-1.3-consolidation-design.md.
"""
import pytest


def test_default_disabled():
    """Without flag, no hippocampus regions or pathways."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, pathways = build_biological_brain_regions()
    region_names = [r.name for r in regions]
    assert "ec" not in region_names
    assert "dg" not in region_names
    assert "ca3" not in region_names
    assert "ca1" not in region_names
    pairs = [(p.from_region, p.to_region) for p in pathways]
    assert ("ca1", "motor_N") not in pairs


def test_enabled_adds_5_regions():
    """+hippocampus adds exactly 5 regions: ec, dg, dg_pv_basket, ca3, ca1."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    base_r, _ = build_biological_brain_regions()
    new_r, _ = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    assert len(new_r) == len(base_r) + 5
    base_names = set(r.name for r in base_r)
    new_names = [r.name for r in new_r if r.name not in base_names]
    assert set(new_names) == {"ec", "dg", "dg_pv_basket", "ca3", "ca1"}


def test_enabled_adds_12_pathways():
    """+hippocampus adds 12 pathways:
    1 lang->ec
    1 ec->dg
    1 ec->dg_pv_basket
    1 dg_pv_basket->dg
    1 dg->ca3
    1 ec->ca1
    1 ca3->ca3 (SWR-gated)
    1 ca3->ca1
    4 ca1->motor_X (per-action consolidation)
    Total: 12 pathways.
    """
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    base_r, base_p = build_biological_brain_regions()
    new_r, new_p = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    assert len(new_p) == len(base_p) + 12


def test_ca3_recurrent_pathway_swr_gated():
    """The ca3 -> ca3 recurrent pathway must have plasticity_gate
    'ca3_swr_burst' for sleep replay control."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    _, pathways = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    ca3_recurrent = [
        p for p in pathways
        if p.from_region == "ca3" and p.to_region == "ca3"
    ]
    assert len(ca3_recurrent) == 1
    assert ca3_recurrent[0].plasticity_gate == "ca3_swr_burst"
    assert ca3_recurrent[0].plastic is True


def test_ca1_to_motor_consolidation_pathways():
    """Each motor pool gets a ca1 -> motor pathway with
    plasticity_gate 'ca1_to_motor' (the consolidation gate)."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    _, pathways = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    consolidation_pathways = [
        p for p in pathways
        if p.from_region == "ca1" and p.to_region.startswith("motor_")
    ]
    assert len(consolidation_pathways) == 4
    motor_targets = {p.to_region for p in consolidation_pathways}
    assert motor_targets == {"motor_N", "motor_E", "motor_S", "motor_W"}
    for p in consolidation_pathways:
        assert p.plasticity_gate == "ca1_to_motor"
        assert p.plastic is True


def test_ca1_to_lang_output_pathway_when_enabled():
    """ca1 -> language_output appears only when both
    enable_hippocampus_consolidation and enable_language_output are True."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    # Without language_output: no ca1 -> language_output
    _, p1 = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    pairs = [(p.from_region, p.to_region) for p in p1]
    assert ("ca1", "language_output") not in pairs
    # With language_output: ca1 -> language_output present
    _, p2 = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
        enable_language_output=True,
    )
    pairs = [(p.from_region, p.to_region) for p in p2]
    assert ("ca1", "language_output") in pairs


def test_ca3_internal_density_zero():
    """CA3 region should have internal_density=0 (recurrent rewired
    as explicit pathway with SWR gate)."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, _ = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    ca3 = next(r for r in regions if r.name == "ca3")
    assert ca3.internal_density == 0.0


def test_dg_uses_hippo_pyramidal():
    """DG should use IZH2007_HIPPO_PYRAMIDAL (granule-cell-like)."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, _ = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
    )
    dg = next(r for r in regions if r.name == "dg")
    assert "HIPPO" in dg.izh_neuron_type


def test_composes_with_tier_2_3():
    """Phase 1.3 hippocampus + Tier 2.3 dlpfc_verb should compose."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, pathways = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
        enable_dlpfc_verb=True,
        enable_language_output=True,
        enable_motor_fs=True,
    )
    region_names = [r.name for r in regions]
    assert "dlpfc_verb" in region_names
    assert "ca1" in region_names
    assert "language_output" in region_names
    for action in ["N", "E", "S", "W"]:
        assert f"motor_FS_{action}" in region_names


def test_custom_hippo_sizes():
    """Region size parameters should be respected."""
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    regions, _ = build_biological_brain_regions(
        enable_hippocampus_consolidation=True,
        n_ec=100, n_dg=300, n_ca3=150, n_ca1=180,
    )
    for name, expected in [("ec", 100), ("dg", 300), ("ca3", 150),
                            ("ca1", 180)]:
        r = next(r for r in regions if r.name == name)
        assert r.n_neurons == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
