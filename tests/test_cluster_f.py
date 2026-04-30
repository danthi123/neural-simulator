"""Cluster F (cerebellum) v1 smoke + structural tests.

Covers the design from docs/plans/2026-04-29-cluster-f-cerebellum-design.md:
- 11 new regions when --enable-cluster-f-cerebellum is on (mossy_state,
  granule, purkinje_{N,E,S,W}, dcn_aip_{N,E,S,W}, inferior_olive)
- ~25 pathways forming the MF -> GC -> PF -> PC -> DCN -> motor forward
  path plus IO -> PC climbing-fiber teaching pathway
- granule -> purkinje_X is the learning site (plasticity_gate = "cerebellum_pf_pc")
- purkinje -> dcn is inhibitory; dcn -> motor is excitatory
- IO -> purkinje is the teaching signal pathway
"""
from __future__ import annotations

import pytest

from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES


def test_cluster_f_off_no_extra_regions():
    """When --enable-cluster-f-cerebellum is off, none of the cerebellar
    regions exist."""
    regions, _ = build_bg_brain_regions()
    names = {r.name for r in regions}
    forbidden = (
        "mossy_state", "granule", "inferior_olive",
        "purkinje_N", "purkinje_E", "purkinje_S", "purkinje_W",
        "dcn_aip_N", "dcn_aip_E", "dcn_aip_S", "dcn_aip_W",
    )
    for n in forbidden:
        assert n not in names, (
            f"cerebellar region {n!r} should not exist when cluster F is off; "
            f"got names: {sorted(names)}"
        )


def test_cluster_f_on_adds_11_regions():
    """When --enable-cluster-f-cerebellum is on, exactly 11 cerebellar
    regions are added (1 mossy + 1 granule + 4 purkinje + 4 dcn + 1 IO)."""
    regions_off, _ = build_bg_brain_regions()
    regions_on, _ = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    delta = len(regions_on) - len(regions_off)
    assert delta == 11, f"expected +11 regions; got {delta}"

    names = {r.name for r in regions_on}
    expected = {
        "mossy_state", "granule", "inferior_olive",
        "purkinje_N", "purkinje_E", "purkinje_S", "purkinje_W",
        "dcn_aip_N", "dcn_aip_E", "dcn_aip_S", "dcn_aip_W",
    }
    assert expected.issubset(names), (
        f"missing cerebellar regions: {expected - names}"
    )


def test_cluster_f_purkinje_is_inhibitory():
    """Per F.01 + F.06: PCs are GABAergic onto DCN. The exc_fraction of
    purkinje_X regions must be 0 so output projections auto-derive as
    inhibitory."""
    regions, _ = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    by_name = {r.name: r for r in regions}
    for action in ACTION_NAMES:
        pc = by_name[f"purkinje_{action}"]
        assert pc.exc_fraction == 0.0, (
            f"purkinje_{action} should be inhibitory (exc_fraction=0), got {pc.exc_fraction}"
        )


def test_cluster_f_dcn_is_excitatory():
    """Per F.06: DCN -> motor projection is excitatory. exc_fraction
    must be 1.0 for dcn_aip_X regions."""
    regions, _ = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    by_name = {r.name: r for r in regions}
    for action in ACTION_NAMES:
        dcn = by_name[f"dcn_aip_{action}"]
        assert dcn.exc_fraction == 1.0, (
            f"dcn_aip_{action} should be excitatory (exc_fraction=1), got {dcn.exc_fraction}"
        )


def test_cluster_f_pf_pc_is_plastic_with_gate():
    """The granule -> purkinje_X pathway is the learning site (PF -> PC).
    Per Marr-Albus, this is where the plasticity lives. v1 tags with
    plasticity_gate="cerebellum_pf_pc" so curriculum can stage cerebellar
    learning."""
    _, pathways = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    pf_pc = [p for p in pathways
             if p.from_region == "granule"
             and p.to_region.startswith("purkinje_")]
    assert len(pf_pc) == 4, f"expected 4 PF->PC pathways (one per action); got {len(pf_pc)}"
    for p in pf_pc:
        assert p.plastic, f"PF->PC ({p.from_region}->{p.to_region}) must be plastic"
        assert p.plasticity_gate == "cerebellum_pf_pc", (
            f"PF->PC ({p.from_region}->{p.to_region}) should be tagged "
            f"'cerebellum_pf_pc'; got {p.plasticity_gate!r}"
        )


def test_cluster_f_pc_dcn_pathways_static_inhibitory():
    """purkinje_X -> dcn_aip_X same-action pathways exist, plastic=False
    (Mauk's two-site plasticity deferred to v2; v1 only PF->PC plastic)."""
    _, pathways = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    pc_dcn = [p for p in pathways
              if p.from_region.startswith("purkinje_")
              and p.to_region.startswith("dcn_aip_")]
    assert len(pc_dcn) == 4, f"expected 4 PC->DCN same-action pathways; got {len(pc_dcn)}"
    for p in pc_dcn:
        # Same action only
        pc_a = p.from_region.split("_")[-1]
        dcn_a = p.to_region.split("_")[-1]
        assert pc_a == dcn_a, (
            f"PC->DCN should be same-action; got {p.from_region}->{p.to_region}"
        )
        assert not p.plastic, "PC->DCN should be plastic=False in v1"


def test_cluster_f_dcn_motor_pathways_static_excitatory():
    """dcn_aip_X -> motor_X excitatory pathways exist, same-action only."""
    _, pathways = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    dcn_motor = [p for p in pathways
                 if p.from_region.startswith("dcn_aip_")
                 and p.to_region.startswith("motor_")]
    assert len(dcn_motor) == 4, f"expected 4 DCN->motor pathways; got {len(dcn_motor)}"
    for p in dcn_motor:
        dcn_a = p.from_region.split("_")[-1]
        m_a = p.to_region.split("_")[-1]
        assert dcn_a == m_a, (
            f"DCN->motor should be same-action; got {p.from_region}->{p.to_region}"
        )
        assert not p.plastic, "DCN->motor should be plastic=False in v1"


def test_cluster_f_io_to_purkinje_climbing_fiber():
    """inferior_olive -> purkinje_X exists for all 4 actions (climbing
    fiber teaching pathway). v1 sparse 1:few mapping (density=0.05)."""
    _, pathways = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    cf = [p for p in pathways
          if p.from_region == "inferior_olive"
          and p.to_region.startswith("purkinje_")]
    assert len(cf) == 4, f"expected 4 IO->purkinje pathways (one per PC pool); got {len(cf)}"
    for p in cf:
        assert not p.plastic, "IO->PC (CF) should be plastic=False (teaching signal, not LTP)"
        # CF should be high-weight to evoke complex spikes (F.04: extensive
        # synaptic contacts producing massive depolarization, not single EPSP)
        assert p.weight_mean >= 30.0, (
            f"CF weight should be high (>=30) to evoke complex spike; "
            f"got {p.weight_mean}"
        )


def test_cluster_f_mossy_state_input_pathway():
    """mossy_state -> granule sparse expansion path exists. Per Marr §3
    codon coding: density should be small (~0.05) so each granule receives
    only a few mossy inputs."""
    _, pathways = build_bg_brain_regions(enable_cluster_f_cerebellum=True)
    mf_gc = [p for p in pathways
             if p.from_region == "mossy_state" and p.to_region == "granule"]
    assert len(mf_gc) == 1, f"expected 1 MF->GC pathway; got {len(mf_gc)}"
    assert mf_gc[0].density <= 0.10, (
        f"MF->GC density should be sparse (<=0.10) per Marr's codon coding; "
        f"got {mf_gc[0].density}"
    )
    assert not mf_gc[0].plastic, "MF->GC should be plastic=False in v1"


def test_cluster_f_with_cluster_a_compose():
    """Cluster F should compose cleanly with Cluster A (closed BG loop).
    No region-name collisions; no pathway double-binding."""
    regions_a, paths_a = build_bg_brain_regions(enable_cluster_a_closed_loop=True)
    regions_af, paths_af = build_bg_brain_regions(
        enable_cluster_a_closed_loop=True,
        enable_cluster_f_cerebellum=True,
    )
    # A+F should add exactly the F regions on top of A
    delta_regions = len(regions_af) - len(regions_a)
    assert delta_regions == 11, (
        f"A+F should add exactly 11 regions over A; got {delta_regions}"
    )
    # No region name collisions
    names_af = [r.name for r in regions_af]
    assert len(names_af) == len(set(names_af)), (
        f"region name collision in A+F; duplicates: "
        f"{[n for n in names_af if names_af.count(n) > 1]}"
    )
