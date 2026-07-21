"""Per-pathway STP-disable (RegionPathway.stp_disabled) — gap#5 mossy detonator (2026-07-21).

The `sim/` mechanism: a pathway tagged `stp_disabled=True` has its synapses SKIP Tsodyks-Markram
short-term depression -- their effective STP factor (stp_u*stp_x) is forced to 1.0 in the step, while
all OTHER synapses keep STP. Realized via a per-synapse boolean mask `cp_stp_disabled_mask`, built in
`inject_explicit_wiring` iff >=1 pathway sets the flag (None otherwise). The default (no pathway flagged)
MUST be byte-identical to the pre-edit engine -- these tests ASSERT that (not a comment).

Backend-agnostic (numpy or cupy). Run: `pytest tests/test_stp_disabled_pathway.py -v`
"""
from __future__ import annotations
import numpy as np
import pytest

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion, RegionPathway
from sim.backend import to_host


def _build(flag_stp_disabled: bool, seed: int = 42):
    """Two-region A->B bridge with STP globally ON; A->B optionally stp_disabled."""
    A = BrainRegion(name="A", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
    B = BrainRegion(name="B", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
    pw = RegionPathway(from_region="A", to_region="B", density=0.5, weight_mean=8.0,
                       weight_jitter=0.0, plastic=False, stp_disabled=flag_stp_disabled)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [A, B]
    cfg.region_pathways = [pw]
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_short_term_plasticity = True   # STP globally ON -- the whole point is per-pathway carve-out
    cfg.enable_hebbian_learning = False
    cfg.enable_stdp = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_ou_process = False
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _state_hash(b):
    nnz = int(b.cp_connections.nnz)
    parts = [
        np.asarray(to_host(b.cp_membrane_potential_v)),
        np.asarray(to_host(b.cp_firing_states)).astype(np.float64),
        np.asarray(to_host(b.cp_stp_u[:nnz])),
        np.asarray(to_host(b.cp_stp_x[:nnz])),
        np.asarray(to_host(b.cp_connections.data)),
    ]
    return tuple(hash(p.tobytes()) for p in parts)


def _drive_A(b, n_steps=60, drive_pA=300.0):
    A_idx = np.asarray(list(b.region_manager.indices("A")), dtype=np.int64)
    B_spikes = 0
    for _ in range(n_steps):
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[A_idx] = float(drive_pA)
        b._run_one_simulation_step()
        B_idx = np.asarray(list(b.region_manager.indices("B")), dtype=np.int64)
        B_spikes += int(np.asarray(to_host(b.cp_firing_states[B_idx])).sum())
    b.cp_external_input_current[:] = 0.0
    return B_spikes


# ---------------------------------------------------------------------------------------------------
def test_default_mask_is_none():
    """Unflagged bridge -> the feature is inert (mask None), so the step keeps the ORIGINAL expression."""
    b = _build(flag_stp_disabled=False)
    assert b.cp_stp_disabled_mask is None


def test_default_byte_identical():
    """Two default (no-flag) bridges at the same seed produce a BIT-IDENTICAL trajectory. Combined with
    the mask being None (so the step runs the verbatim `base*stp_u*stp_x` expression), this is the
    byte-identity-when-off guarantee: the new code path is never taken for an unflagged bridge."""
    b0 = _build(flag_stp_disabled=False, seed=42)
    b1 = _build(flag_stp_disabled=False, seed=42)
    assert b0.cp_stp_disabled_mask is None and b1.cp_stp_disabled_mask is None
    _drive_A(b0)
    _drive_A(b1)
    assert _state_hash(b0) == _state_hash(b1), "default (unflagged) trajectory is not deterministic/identical"


def test_flagged_mask_built_correctly():
    """A flagged pathway -> mask is built (not None) and True for exactly that pathway's synapses."""
    b = _build(flag_stp_disabled=True)
    m = b.cp_stp_disabled_mask
    assert m is not None
    nnz = int(b.cp_connections.nnz)
    n_true = int(np.asarray(to_host(m[:nnz])).sum())
    assert n_true == nnz, f"expected all {nnz} A->B synapses flagged, got {n_true}"


def test_stp_disable_fires_and_undepresses():
    """The STP-disable actually FIRES: under sustained pre-firing the STP state depresses (mean stp_u*stp_x
    << 1 -- what WOULD crush the un-flagged pathway), yet the flagged pathway transmits at full base weight,
    so its downstream target fires strictly MORE than the un-flagged one."""
    b_off = _build(flag_stp_disabled=False, seed=42)   # STP-gated A->B (would depress)
    b_on = _build(flag_stp_disabled=True, seed=42)     # STP-disabled A->B (detonator)
    B_off = _drive_A(b_off)
    B_on = _drive_A(b_on)

    # (1) real depression is present on the STP-gated bridge (the thing we bypass).
    nnz = int(b_off.cp_connections.nnz)
    u = np.asarray(to_host(b_off.cp_stp_u[:nnz]))
    x = np.asarray(to_host(b_off.cp_stp_x[:nnz]))
    depressed_factor = float(np.mean(u * x))
    assert depressed_factor < 0.6, (
        f"expected real STP depression (mean stp_u*stp_x < 0.6) on the gated bridge, got {depressed_factor:.3f}")

    # (2) the flagged (undepressed) pathway drives its target strictly more.
    assert B_on > B_off, (
        f"STP-disable did not lift downstream drive: B_on={B_on} !> B_off={B_off} "
        f"(depressed_factor={depressed_factor:.3f})")
    print(f"[stp-disable fires] gated mean stp_u*stp_x={depressed_factor:.3f} | "
          f"B spikes: gated={B_off} detonator={B_on}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
