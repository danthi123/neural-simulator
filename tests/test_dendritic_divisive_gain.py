"""Tests for the DENDRITIC per-presynaptic-source DIVISIVE GAIN (D2 Phase 1).

The on-substrate realization of the de-risked (D1/D1.5/D1.6/D1.7) per-input divisive normalization: each
presynaptic source's spike is scaled by a bounded gain g_i = sigma/(sigma + a_i), where a_i is that
source's own firing-rate EMA (cp_dendritic_source_activity). A high-frequency source is suppressed toward
0; a rare source passes near 1.

Load-bearing guarantees pinned here (CPU / SIM_BACKEND=numpy, deterministic, no GPU):
  1. BYTE-IDENTITY WHEN OFF: with enable_dendritic_divisive_gain=False (the default),
     cp_dendritic_source_activity stays None, the gain + EMA-update blocks are unreached, and two
     same-seed builds step bitwise-identically. (The project-wide byte-identity-when-off proof is the
     existing conversational suite passing verbatim; this pins the local guard.)
  2. ON allocates + updates: with the flag on, the per-source EMA is allocated and grows above 0 once
     sources fire.
  3. THE GAIN SUPPRESSES high-activity sources: a strongly-driven (high-activity) source delivers LESS
     drive to its target with the gain on than off -> the target fires less. (The mechanism.)
  4. SIGMA-HUGE ~= OFF: with the flag on but sigma huge, g_i -> 1 -> the run matches flag-off (proves the
     effect is the gain, not the code path).
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


def _build(seed=42, enable_gain=False, sigma=0.05, alpha=0.05, drive=140.0):
    """A small 2-region (source->target) bridge. The source is driven by constant external current so it
    fires at a high rate (its EMA rises -> the divisive gain suppresses it)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="source", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="target", n_neurons=20, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="source", to_region="target",
                      density=0.4, weight_mean=1.0, weight_jitter=0.0),
    ]
    cfg.dt = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_dendritic_divisive_gain = enable_gain
    cfg.dendritic_divisive_sigma = sigma
    cfg.dendritic_gain_ema_alpha = alpha
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    # drive the source region with constant external current
    src = bridge.region_manager.indices("source")
    bridge.cp_external_input_current[src] = drive
    return bridge, src, bridge.region_manager.indices("target")


def _run(bridge, n_steps):
    import numpy as np
    fired = []
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        fired.append(np.asarray(bridge.cp_firing_states).copy())
    return np.stack(fired)


def test_off_is_none_and_deterministic(numpy_backend):
    """Flag off (default): the per-source EMA stays None, and two same-seed builds step bit-identically."""
    import numpy as np
    b1, _, _ = _build(seed=7, enable_gain=False)
    assert b1.cp_dendritic_source_activity is None
    f1 = _run(b1, 60)
    b2, _, _ = _build(seed=7, enable_gain=False)
    f2 = _run(b2, 60)
    assert np.array_equal(f1, f2), "flag-off run is not deterministic across builds"


def test_on_allocates_and_updates(numpy_backend):
    """Flag on: the per-source EMA is allocated and grows above 0 once sources fire."""
    import numpy as np
    b, src, _ = _build(seed=7, enable_gain=True)
    assert b.cp_dendritic_source_activity is not None
    _run(b, 80)
    ema_src = np.asarray(b.cp_dendritic_source_activity)[np.asarray(src)]
    assert float(ema_src.mean()) > 0.0, "per-source activity EMA did not grow despite driven sources"


def _run_track_target_ge(bridge, tgt, n_steps):
    """Run, recording the target region's mean excitatory conductance (the quantity the gain scales) each
    step. Conductance is the direct mechanism signal (not gated by the somatic firing threshold)."""
    import numpy as np
    tgt = np.asarray(tgt)
    ge = []
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        ge.append(float(np.asarray(bridge.cp_conductance_g_e)[tgt].mean()))
    return np.asarray(ge)


def test_gain_suppresses_high_activity_source(numpy_backend):
    """The mechanism: a high-activity source delivers LESS drive to its target with the gain on. Measured
    on the target's excitatory CONDUCTANCE (what the gain scales) -- in the second half, after the source
    EMA has risen and the gain bites, on << off (same seed, same drive)."""
    import numpy as np
    n = 200
    b_off, _, tgt = _build(seed=11, enable_gain=False)
    ge_off = _run_track_target_ge(b_off, tgt, n)
    b_on, _, tgt2 = _build(seed=11, enable_gain=True, sigma=0.05, alpha=0.05)
    ge_on = _run_track_target_ge(b_on, tgt2, n)
    half = n // 2
    off2 = float(ge_off[half:].mean())
    on2 = float(ge_on[half:].mean())
    print(f"target excitatory conductance (2nd half): off={off2:.4f}  on={on2:.4f}  ratio={on2/off2:.3f}")
    assert on2 < 0.9 * off2, f"divisive gain did not suppress the high-activity source (on {on2} vs off {off2})"


def test_sigma_huge_approximates_off(numpy_backend):
    """Flag on but sigma huge -> g_i = sigma/(sigma+a) -> 1 -> the run matches flag-off (the effect is the
    gain, not the code path)."""
    import numpy as np
    n = 120
    b_off, _, _ = _build(seed=13, enable_gain=False)
    f_off = _run(b_off, n)
    b_huge, _, _ = _build(seed=13, enable_gain=True, sigma=1e6, alpha=0.05)
    f_huge = _run(b_huge, n)
    # not necessarily bitwise (the float multiply by ~1.0 differs in the last bit), but the firing pattern
    # must match to a tiny fraction of spikes.
    mismatch = float(np.mean(f_off != f_huge))
    print(f"sigma-huge vs off spike mismatch fraction = {mismatch:.6f}")
    assert mismatch < 0.005, f"sigma-huge gain is not ~= off (mismatch {mismatch})"
