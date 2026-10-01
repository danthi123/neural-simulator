"""Tests for cfg.enable_bounded_conductance_integration (sim/config.py), the flag-gated,
default-OFF fix for the 2026-09-29 explicit-step conductance overshoot defect.

THE DEFECT: sim/kernels.py computes the synaptic current from the PREVIOUS step's
membrane potential, I_syn = g_e*(E_e-v) + g_i*(E_i-v) (fused_conductance_decay_and_current),
then advances v by an explicit-Euler step. When the summed conductance is large relative
to C/dt, one step drives v far PAST the reversal potential instead of toward it. Measured
(docs/drafts/conductance-overshoot-guard/WORKLOG.md): a basal-ganglia build with four
weight-40 GABA-A projections converging on a 'commit' pool drove v to -377 mV within 60
steps; the pool then fired at the refractory ceiling with its excitatory drive silent (the
sign of the inhibitory driving force flips once v is past E_i).

THE FIX (flag ON): route the AMPA/GABA-A pair, GABA_B and NMDA/slow-NMDA conductance
currents through an exponential-Euler bounded sibling kernel (sim/kernels.py
fused_bounded_conductance_decay_and_current / fused_bounded_gabab_decay_and_current /
fused_bounded_nmda_update_and_current) that solves the local linear ODE for that
conductance term exactly over dt, so one step can only move v toward its own reversal
potential, never past it. SCOPED to NeuronModel.IZHIKEVICH (2007) and NeuronModel.ADEX.

Uses SIM_BACKEND=numpy (set below, before any sim import) and cfg.seed set explicitly on
every build (CLAUDE.md: cfg.seed, not the unseeded global RNG, actually seeds the run).

Reproducer: a minimal region-framework bridge (see research/runners/_vocal_action_selector_gate.py
for the pattern this borrows: BrainRegion/RegionPathway + SimulationBridge, no scratch-path
imports) with 4 single-neuron GABA-A source pools converging on a small Izhikevich-2007 RS
'commit' pool with NO excitatory pathway -- literally the defect report's "four GABA-A
projections converging on a commit pool", scaled to a fast unit-test budget.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    NeuronType,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import get_backend, to_host  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402


# Reference sha256 of (stacked per-step firing-state raster bytes + final membrane-potential-v
# bytes) for _build_reference_network_bridge(seed=42), 60 steps, captured by running THIS
# network builder (proto_hash_net.py in the dev scratchpad, byte-identical logic to
# _build_reference_network_bridge/_run_and_hash below) against sim/ on `main` at commit
# 3c2907b85 (the tip this fix's worktree branched from), BEFORE sim/kernels.py, sim/bridge.py
# or sim/config.py were touched -- i.e. the pre-change code path, run directly (not
# monkeypatched). test_byte_identical_when_off below reproduces this same hash from the
# MODIFIED code with the flag both explicitly False and left at its default, on the SAME
# network, so a match against this constant is a match against the actual pre-fix engine.
_REFERENCE_HASH_FROM_MAIN = (
    "c14523058fe9ba091f033be2c68a4600683fce94bee798d96e87c72419790462"
)

_RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
_FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON


def _region(name, n, exc_fraction, neuron_type, *, enable_nmda=False):
    return BrainRegion(
        name=name,
        n_neurons=int(n),
        exc_fraction=float(exc_fraction),
        internal_density=0.0,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=neuron_type.name,
        enable_homeostasis=False,
        enable_nmda=bool(enable_nmda),
    )


def _base_cfg(seed, dt_ms=1.0):
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = float(dt_ms)
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    for flag in (
        "enable_short_term_plasticity", "enable_hebbian_learning",
        "enable_homeostasis", "enable_structural_plasticity",
        "enable_reward_modulation", "enable_stdp",
    ):
        setattr(cfg, flag, False)
    return cfg


def _finish_build(cfg, regions, pathways, drive_regions, drive_pA):
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    for name in drive_regions:
        idx = np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
        bridge.cp_external_input_current[xp.asarray(idx)] = xp.float32(drive_pA)
    return bridge


def _build_convergent_inhibition_bridge(
    seed, *, bounded, inh_weight=400.0, n_sources=4, n_commit=1,
    exc_weight=0.0, ou_sigma_pA=0.0, drive_pA=3000.0, dt_ms=1.0,
):
    """The defect's literal reproducer: n_sources single-neuron GABA-A pools (weight
    inh_weight each), driven to fire by a strong direct external current, converging on a
    'commit' Izhikevich-2007 RS pool with NO excitatory pathway (unless exc_weight > 0)."""
    cfg = _base_cfg(seed, dt_ms=dt_ms)
    cfg.enable_ou_process = bool(ou_sigma_pA > 0)
    if cfg.enable_ou_process:
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_sigma_pA)
        cfg.ou_tau_ms = 15.0
    cfg.enable_bounded_conductance_integration = bool(bounded)

    regions = [_region("commit", n_commit, 1.0, _RS)]
    pathways = []
    drive_regions = []
    for i in range(n_sources):
        name = f"inh_src_{i}"
        drive_regions.append(name)
        regions.append(_region(name, 1, 0.0, _FS))
        pathways.append(RegionPathway(
            from_region=name, to_region="commit", density=1.0,
            weight_mean=float(inh_weight), weight_jitter=0.0, plastic=False,
        ))
    if exc_weight > 0:
        regions.append(_region("exc_src", 1, 1.0, _RS))
        drive_regions.append("exc_src")
        pathways.append(RegionPathway(
            from_region="exc_src", to_region="commit", density=1.0,
            weight_mean=float(exc_weight), weight_jitter=0.0, plastic=False,
        ))
    return _finish_build(cfg, regions, pathways, drive_regions, drive_pA)


def _build_reference_network_bridge(seed, *, set_flag_explicitly):
    """A richer network exercising ALL FOUR conductance-current call sites this fix touches:
    AMPA/GABA-A (fused_conductance_decay_and_current), GABA_B (fused_gabab_decay_and_current),
    global NMDA and slow-NMDA-recurrent-shaped current (both fused_nmda_update_and_current --
    the recurrent one isn't separately wired here, so this covers the global-NMDA call site;
    see the WORKLOG open questions), plus OU noise. set_flag_explicitly=True/False sets
    cfg.enable_bounded_conductance_integration explicitly; None leaves it at the dataclass
    default (also False) -- used by test_byte_identical_when_off to check BOTH spellings of
    "off" against each other and against _REFERENCE_HASH_FROM_MAIN."""
    cfg = _base_cfg(seed)
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = 40.0
    cfg.ou_tau_ms = 15.0
    cfg.enable_nmda = True
    cfg.enable_gabab = True
    if set_flag_explicitly is not None:
        cfg.enable_bounded_conductance_integration = bool(set_flag_explicitly)

    regions = [_region("commit", 6, 1.0, _RS, enable_nmda=True)]
    pathways = []
    drive_regions = []
    for i in range(4):
        name = f"inh_src_{i}"
        drive_regions.append(name)
        regions.append(_region(name, 2, 0.0, _FS))
        pathways.append(RegionPathway(
            from_region=name, to_region="commit", density=1.0,
            weight_mean=40.0, weight_jitter=0.1, plastic=False,
        ))
    regions.append(_region("gabab_src", 2, 0.0, _FS))
    drive_regions.append("gabab_src")
    pathways.append(RegionPathway(
        from_region="gabab_src", to_region="commit", density=1.0,
        weight_mean=5.0, weight_jitter=0.1, plastic=False, receptor="gaba_b",
    ))
    regions.append(_region("exc_src", 4, 1.0, _RS))
    drive_regions.append("exc_src")
    pathways.append(RegionPathway(
        from_region="exc_src", to_region="commit", density=1.0,
        weight_mean=50.0, weight_jitter=0.1, plastic=False,
    ))
    return _finish_build(cfg, regions, pathways, drive_regions, 1500.0)


def _run_and_hash(bridge, n_steps):
    raster = []
    for _ in range(int(n_steps)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster.append(np.asarray(to_host(bridge.cp_firing_states), dtype=np.uint8).copy())
    raster = np.stack(raster, axis=0)
    v_final = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=np.float32)
    h = hashlib.sha256()
    h.update(raster.tobytes())
    h.update(v_final.tobytes())
    return h.hexdigest(), raster, v_final


# --- (a) BYTE-IDENTICAL OFF ------------------------------------------------------------

def test_byte_identical_when_off():
    """Flag absent (default) and flag explicitly False must produce IDENTICAL spike rasters
    + final membrane arrays, both to each other and to _REFERENCE_HASH_FROM_MAIN (captured
    from `main`, pre-fix -- see that constant's docstring for provenance). This is the
    contract that DEFAULT-OFF means byte-identical to the engine before this change."""
    digest_default, raster_default, v_default = _run_and_hash(
        _build_reference_network_bridge(seed=42, set_flag_explicitly=None), 60)
    digest_false, raster_false, v_false = _run_and_hash(
        _build_reference_network_bridge(seed=42, set_flag_explicitly=False), 60)

    assert digest_default == digest_false
    np.testing.assert_array_equal(raster_default, raster_false)
    np.testing.assert_array_equal(v_default, v_false)
    assert digest_default == _REFERENCE_HASH_FROM_MAIN, (
        "byte-identical-off contract broken: the unbounded (flag-off) path no longer "
        "matches the pre-fix engine captured on main at 3c2907b85"
    )
    # Sanity: the network is not degenerately silent (a trivial all-zero raster would make
    # the hash comparison meaningless).
    assert int(raster_default.sum()) > 0


# --- (b) OVERSHOOT REPRODUCED (flag off) -----------------------------------------------

def test_overshoot_reproduced_flag_off():
    """The literal defect reproducer with the flag OFF: v crashes well past the GABA-A
    reversal potential (< -150 mV), matching the measured -377 mV / -175 mV order of
    magnitude in the defect report (docs/drafts/conductance-overshoot-guard/WORKLOG.md)."""
    bridge = _build_convergent_inhibition_bridge(seed=42, bounded=False)
    commit_idx = np.asarray(bridge.region_manager.indices("commit"), dtype=np.int64)
    v_min = float(bridge.cp_membrane_potential_v[commit_idx].min())
    for _ in range(80):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        v_min = min(v_min, float(bridge.cp_membrane_potential_v[commit_idx].min()))
    assert v_min < -150.0, f"expected the unbounded path to overshoot past -150 mV, got {v_min}"


# --- (c) BOUNDED (flag on) --------------------------------------------------------------

def test_bounded_flag_on():
    """The SAME convergent-inhibition build with the flag ON: v is bounded at every step to
    >= (min reversal potential - 1 mV) -- here just E_i, the only active conductance -- and
    the commit pool NEVER fires, because its only input is inhibitory (excitatory drive is
    silent: exc_weight=0). Contrast with the flag-off run, which DOES spuriously fire the
    pool once v overshoots past E_i and the driving force sign flips (see the module
    docstring)."""
    bridge = _build_convergent_inhibition_bridge(seed=42, bounded=True)
    commit_idx = np.asarray(bridge.region_manager.indices("commit"), dtype=np.int64)
    e_i = float(bridge.core_config.syn_reversal_potential_i)
    floor = e_i - 1.0
    v_min = float(bridge.cp_membrane_potential_v[commit_idx].min())
    fired_any = False
    for _ in range(80):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        v_min = min(v_min, float(bridge.cp_membrane_potential_v[commit_idx].min()))
        if bool(np.asarray(to_host(bridge.cp_firing_states))[commit_idx].any()):
            fired_any = True
    assert v_min >= floor, f"bounded path overshot past E_i-1={floor}: v_min={v_min}"
    assert not fired_any, "commit pool fired with its excitatory drive silent (flag on)"

    # Cross-check: the SAME build with the flag off DOES overshoot past floor at some point
    # during the run (the same assertion test_overshoot_reproduced_flag_off makes on its own
    # build), so this is a genuine fix, not a network that never overshoots regardless of the
    # flag. Tracks the running min like the flag-on loop above -- the FINAL v alone is not
    # enough evidence (a fired neuron resets to c_reset=-50, which would mask the overshoot
    # that happened on an earlier step).
    unbounded = _build_convergent_inhibition_bridge(seed=42, bounded=False)
    ub_idx = np.asarray(unbounded.region_manager.indices("commit"), dtype=np.int64)
    ub_v_min = float(unbounded.cp_membrane_potential_v[ub_idx].min())
    for _ in range(80):
        unbounded._run_one_simulation_step()
        unbounded.runtime_state.current_time_ms += unbounded.core_config.dt_ms
        ub_v_min = min(ub_v_min, float(unbounded.cp_membrane_potential_v[ub_idx].min()))
    assert ub_v_min < floor


# --- (d) SMALL-CONDUCTANCE EQUIVALENCE (flag on vs off) ---------------------------------

def test_small_conductance_equivalence():
    """With weak synapses (g_tot*dt/C << 1), the bounded and unbounded paths should agree
    almost everywhere: the exp-Euler correction is a second-order effect at small
    conductance, so spike rasters need not be bit-identical (a neuron exactly on the
    threshold can tip either way under the FP-level difference) but should match on the
    large majority of neuron-steps. Network: a small commit pool with BOTH a weak convergent
    GABA-A input (matching the overshoot reproducer's topology, at 1/200th the weight) and a
    real excitatory drive + OU noise, so there is actual spiking to compare."""
    def _raster(bounded):
        bridge = _build_convergent_inhibition_bridge(
            seed=42, bounded=bounded, inh_weight=2.0, n_sources=4, n_commit=8,
            exc_weight=60.0, ou_sigma_pA=25.0, drive_pA=1200.0,
        )
        commit_idx = np.asarray(bridge.region_manager.indices("commit"), dtype=np.int64)
        rows = []
        for _ in range(200):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            rows.append(np.asarray(to_host(bridge.cp_firing_states), dtype=np.uint8)[commit_idx].copy())
        return np.stack(rows, axis=0)

    raster_off = _raster(False)
    raster_on = _raster(True)
    total_spikes = int(raster_off.sum())
    match_fraction = float((raster_off == raster_on).mean())

    assert total_spikes > 0, "degenerate reproducer: no spikes to compare"
    assert match_fraction >= 0.90, (
        f"small-conductance rasters diverged too much: match_fraction={match_fraction:.4f} "
        f"(spikes_off={total_spikes}, spikes_on={int(raster_on.sum())})"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
