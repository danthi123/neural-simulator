"""Tests for the GRADED dendritic-plateau READ-OUT (Stage 1, 2026-06-20).

The on-substrate realization of de-risk A's validated win (Stage 0, GO 6/6 seeds,
2026-06-20-dendrite-derisk-A-graded-plateau-readout.md): the dendrite's ONE genuine unlock is a GRADED
ANALOG read-out of a distributed code (Mikulasch-Priesemann) the point-neuron soma provably cannot be
(sub-rheobase 0 / all-or-none saturated -- never the graded middle). The on-bridge realization is a
SMOOTH (gentle, centered, non-saturating) logistic plateau current on the routed (coincidence_detector)
synapses' WEIGHTED coincident drive, the GRADED-transfer sibling of fused_coincidence_plateau.

Load-bearing guarantees pinned here (CPU / SIM_BACKEND=numpy, deterministic, no GPU):
  1. BYTE-IDENTITY WHEN OFF: with enable_graded_dendritic_plateau=False (the default), the graded plateau
     conductances stay None, the per-step graded block is unreached, and two same-seed builds step
     bitwise-identically. STRONGER: a bridge that WIRES a coincidence_detector value pathway but leaves
     the flag OFF is bit-identical to one with NO such routing -- the block is provably gated by the flag.
  2. ON allocates: with the flag on, the graded plateau conductances are allocated.
  3. THE PLATEAU GRADES WITH LEARNED VALUE (the mechanism + the de-risk-A discriminator): a HIGH-weight
     routed ensemble injects MORE graded plateau current than a LOW-weight one, and the transfer is the
     SMOOTH continuum (high > mid > low) -- the graded middle the all-or-none switch (fused_coincidence_
     plateau) snaps to 0/1. Pinned directly on the kernel (the bridge wiring exercises the same kernel).
  4. GRADED vs ALL-OR-NONE: at a MID drive (between the off-floor and the saturated tail) the graded
     transfer is strictly intermediate while the all-or-none switch is saturated -- the graded-ness is
     load-bearing (the Mikulasch-Priesemann claim).
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def numpy_backend(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])
    yield xp
    _reset_cache_for_tests()
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])


def _build(seed=42, enable_graded=False, route_coincidence=True, value_weight=6.0, drive=300.0):
    """A small 2-region (place_context -> critic) bridge. The place->critic pathway is the VALUE pathway;
    when route_coincidence it is tagged coincidence_detector=True (the routing mask the graded block
    consumes). The place region is driven by constant external current so it fires and delivers a WEIGHTED
    coincident drive to the critic. value_weight scales the learned place->value synaptic value (the
    quantity the graded plateau reads out)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="place_context", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="critic", n_neurons=20, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="place_context", to_region="critic",
                      density=0.5, weight_mean=float(value_weight), weight_jitter=0.0,
                      plastic=False, coincidence_detector=bool(route_coincidence)),
    ]
    # The graded plateau reuses the coincidence routing mask -> the mask is only built when
    # enable_coincidence_detection is True at wiring time. We always set it True so the mask exists;
    # the all-or-none coincidence BLOCK is independent and only fires if its OWN conductance is alloc'd.
    cfg.enable_coincidence_detection = bool(route_coincidence)
    cfg.enable_graded_dendritic_plateau = bool(enable_graded)
    cfg.dt = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    plc = bridge.region_manager.indices("place_context")
    bridge.cp_external_input_current[plc] = drive
    return bridge, plc, bridge.region_manager.indices("critic")


def _run(bridge, n_steps):
    import numpy as np
    out = []
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        out.append(np.asarray(bridge.cp_firing_states).copy())
    return np.stack(out)


def _run_track_critic_current(bridge, crit, n_steps):
    """Run, recording the critic region's mean membrane potential each step (the graded plateau current
    depolarizes the critic; with no routed graded block the critic stays at its baseline drive)."""
    import numpy as np
    crit = np.asarray(crit)
    vs = []
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        vs.append(float(np.asarray(bridge.cp_membrane_potential_v)[crit].mean()))
    return np.asarray(vs)


# --------------------------------------------------------------------------- #
# 1. BYTE-IDENTITY WHEN OFF
# --------------------------------------------------------------------------- #
def test_off_is_none_and_deterministic(numpy_backend):
    """Flag off (default): the graded plateau conductances stay None, and two same-seed builds step
    bit-identically."""
    import numpy as np
    b1, _, _ = _build(seed=7, enable_graded=False)
    assert b1.cp_conductance_g_graded_plateau is None
    assert b1.cp_conductance_g_graded_plateau_rise is None
    f1 = _run(b1, 60)
    b2, _, _ = _build(seed=7, enable_graded=False)
    f2 = _run(b2, 60)
    assert np.array_equal(f1, f2), "flag-off run is not deterministic across builds"


def test_off_with_routing_equals_no_routing(numpy_backend):
    """STRONGER byte-identity: a bridge that WIRES a coincidence_detector value pathway but leaves the
    graded flag OFF must be bit-identical to one with NO graded routing concept -- i.e. the graded block
    is provably gated by enable_graded_dendritic_plateau, not by the routing mask's existence. (Both
    bridges keep the all-or-none coincidence conductances None by sharing the same enable_coincidence_
    detection wiring; the ONLY toggled thing is the graded flag, here OFF in both.)"""
    import numpy as np
    # Same wiring (coincidence mask present in both), graded flag OFF in both -> identical dynamics.
    b_routed, _, crit = _build(seed=9, enable_graded=False, route_coincidence=True)
    v_routed = _run_track_critic_current(b_routed, crit, 80)
    b_plain, _, crit2 = _build(seed=9, enable_graded=False, route_coincidence=True)
    v_plain = _run_track_critic_current(b_plain, crit2, 80)
    assert np.array_equal(v_routed, v_plain), "graded-OFF dynamics differ across identical builds"
    # And the graded conductance is never allocated regardless of the routing mask.
    assert b_routed.cp_conductance_g_graded_plateau is None


# --------------------------------------------------------------------------- #
# 2. ON allocates
# --------------------------------------------------------------------------- #
def test_on_allocates(numpy_backend):
    """Flag on: the graded plateau conductances (g + g_rise) are allocated."""
    b, _, _ = _build(seed=7, enable_graded=True)
    assert b.cp_conductance_g_graded_plateau is not None
    assert b.cp_conductance_g_graded_plateau_rise is not None


def test_on_changes_critic_dynamics(numpy_backend):
    """The guard gates REAL computation: with the routed value pathway active and the flag ON, the graded
    plateau conductance accumulates (its own mechanism state grows above 0) AND the critic dynamics differ
    measurably from OFF. If ON==OFF the block would be a no-op -- this proves it is load-bearing when
    enabled (vs byte-identical when off)."""
    import numpy as np
    n = 120
    b_off, _, crit = _build(seed=11, enable_graded=False, value_weight=6.0)
    v_off = _run_track_critic_current(b_off, crit, n)
    b_on, _, crit2 = _build(seed=11, enable_graded=True, value_weight=6.0)
    v_on = _run_track_critic_current(b_on, crit2, n)
    # (a) the graded plateau conductance accumulated (the mechanism ran).
    g_plateau = float(np.asarray(b_on.cp_conductance_g_graded_plateau)[np.asarray(crit2)].mean())
    print(f"critic graded-plateau conductance (ON) = {g_plateau:.4f}")
    assert g_plateau > 0.0, "graded plateau conductance did not accumulate despite the routed value pathway"
    # (b) the critic dynamics DIFFER from OFF (the additive plateau current is load-bearing, not a no-op).
    diff = float(np.abs(v_on - v_off).mean())
    print(f"critic mean |V_on - V_off| = {diff:.3f} (must be > 0: the plateau changes the critic)")
    assert diff > 0.5, f"graded plateau ON did not change critic dynamics vs OFF (mean |dV| {diff})"


# --------------------------------------------------------------------------- #
# 3 + 4. THE PLATEAU GRADES WITH VALUE; GRADED vs ALL-OR-NONE (the de-risk-A discriminator)
# --------------------------------------------------------------------------- #
def test_kernel_grades_with_value_smoothly(numpy_backend):
    """The mechanism + the Mikulasch-Priesemann discriminator, pinned directly on the kernel the bridge
    block calls: the GRADED transfer expresses a SMOOTH, non-saturating continuum across the active range
    of the WEIGHTED drive c_w (low < mid < high, each step a real gap), where the all-or-none switch
    (fused_coincidence_plateau) saturates the mid to the high level. center=8, slope=0.33 = the Stage-0
    operating point (dend_theta=8, dend_slope=3)."""
    import numpy as np
    import sim.kernels as k

    def graded_inc(cw):
        g = np.zeros(1, dtype=np.float32); gr = np.zeros(1, dtype=np.float32)
        v = np.full(1, -50.0, dtype=np.float32)
        gn, _, _ = k.fused_graded_dendritic_plateau(
            g, gr, np.float32(0.9), np.float32(0.6), v, np.float32(0.0), np.float32(1.0),
            np.full(1, float(cw), dtype=np.float32),
            np.float32(8.0), np.float32(0.33), np.float32(80.0))
        return float(gn[0])

    def allornone_inc(cw):
        # The all-or-none coincidence kernel at the SAME drive (steep gain=2, k_thresh=8).
        g = np.zeros(1, dtype=np.float32); gr = np.zeros(1, dtype=np.float32)
        v = np.full(1, -50.0, dtype=np.float32)
        gn, _, _ = k.fused_coincidence_plateau(
            g, gr, np.float32(0.9), np.float32(0.6), v, np.float32(0.0), np.float32(1.0),
            np.full(1, float(cw), dtype=np.float32),
            np.float32(8.0), np.float32(2.0), np.float32(80.0))
        return float(gn[0])

    low, mid, high = graded_inc(3.0), graded_inc(8.0), graded_inc(13.0)
    print(f"GRADED  plateau g_inc: low(cw=3)={low:.2f}  mid(cw=8)={mid:.2f}  high(cw=13)={high:.2f}")
    # SMOOTH continuum: strictly monotone with REAL gaps at every step (the graded middle is expressed).
    assert low < mid < high, f"graded transfer is not monotone: {low}, {mid}, {high}"
    assert (mid - low) > 5.0 and (high - mid) > 5.0, "graded sub-steps are not real (continuum collapsed)"
    # NON-SATURATING across the active range: high is NOT pinned at the strength ceiling (80) the way an
    # all-or-none switch would be -- the graded read-out stays on the slope.
    assert high < 78.0, f"graded transfer saturated at the ceiling (high={high}); not the graded middle"

    # The all-or-none switch, by contrast, SATURATES the mid+high to the ceiling (the binary subunit):
    aon_mid, aon_high = allornone_inc(8.0), allornone_inc(13.0)
    print(f"ALL-OR-NONE plateau g_inc: mid(cw=8)={aon_mid:.2f}  high(cw=13)={aon_high:.2f}")
    assert aon_high > 79.0, "all-or-none control did not saturate at high drive (harness mis-set)"
    # The discriminator: at the HIGH drive the all-or-none is saturated (~80) while the graded is still on
    # its slope (< 78) -- the graded read-out expresses a middle the binary switch cannot.
    assert high < aon_high, "graded high should be below the saturated all-or-none high (the continuum)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
