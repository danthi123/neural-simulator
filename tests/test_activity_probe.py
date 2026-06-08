"""Tests for sim.activity_probe.RegionActivityProbe (frontend-revamp Phase 1).

The load-bearing guarantees these tests pin:

  1. SCIENCE-SAFETY / NON-PERTURBATION: sampling the probe must NOT change the
     sim's numerical trajectory at all. Two identical toy bridges (same seed)
     run for N steps — one with the probe sampling every step, one without —
     end with BITWISE-identical cp_firing_states + cp_membrane_potential_v.
     This is the strongest form of "with --emit-activity off (or even on) the
     run is byte-identical": the probe only READS state, never writes it.

  2. PER-REGION REDUCTION: sample() returns sane per-region firing fractions in
     [0,1] and per-pathway flux, computed from the region slices.

  3. OVERHEAD: the per-sample cost is small (O(regions+pathways), not
     O(neurons)). We MEASURE it (off vs on) and print it so the no-bottleneck
     claim is grounded in a number; we also assert a loose ceiling.

These use a ~45-neuron 2-region toy bridge under SIM_BACKEND=numpy (the same
pattern as tests/test_numpy_backend_integration.py) so they run in CI without a
GPU and are deterministic.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def numpy_backend(monkeypatch):
    """Force SIM_BACKEND=numpy + reload backend-binding modules (mirrors the
    fixture in tests/test_numpy_backend_integration.py)."""
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge",
                    "sim.activity_probe"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])
    yield xp
    _reset_cache_for_tests()
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge",
                    "sim.activity_probe"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])


def _build_toy_region_bridge(seed=42):
    """A small 2-region (cortex->motor) brain-region bridge. ~45 neurons; runs
    fast under numpy and exercises region_manager.indices() + pathways()."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="cortex", n_neurons=30, exc_fraction=0.8,
                    internal_density=0.1),
        BrainRegion(name="motor", n_neurons=15, exc_fraction=0.8,
                    internal_density=0.1),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="cortex", to_region="motor",
                      density=0.2, weight_mean=1.0, weight_jitter=0.1),
    ]
    cfg.dt = 1.0
    # Pin ALL RNG seeds so two builds are deterministically identical (cfg.seed
    # defaults to -1 = time-based random, which would make same-"seed" bridges
    # diverge regardless of the probe). With these fixed, the ONLY difference
    # between a probed and unprobed run is the probe — which only reads state.
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=rt, gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    return bridge


@pytest.mark.slow
def test_probe_constructs_and_samples(numpy_backend):
    """RegionActivityProbe builds from the region slices and sample() returns
    per-region firing fractions in [0,1] + per-pathway flux."""
    from sim.activity_probe import RegionActivityProbe

    bridge = _build_toy_region_bridge(seed=42)
    probe = RegionActivityProbe(bridge)

    # Knows both regions and the one pathway.
    assert probe.n_regions == 2
    assert set(probe.region_names()) == {"cortex", "motor"}
    assert probe.n_pathways == 1

    # Drive a few steps so something fires, then sample.
    for _ in range(20):
        bridge._run_one_simulation_step()
    regions, flux = probe.sample(bridge)

    assert set(regions.keys()) == {"cortex", "motor"}
    for name, rate in regions.items():
        assert 0.0 <= rate <= 1.0, f"{name} rate {rate} out of [0,1]"
    # Flux keyed by "<from>_to_<to>" (matches extract_per_pathway_csrs naming).
    assert "cortex_to_motor" in flux
    assert 0.0 <= flux["cortex_to_motor"] <= 1.0


@pytest.mark.slow
def test_probe_does_not_perturb_simulation(numpy_backend):
    """SCIENCE-SAFETY: sampling the probe every step must NOT change the sim's
    numerical trajectory. Two same-seed bridges run N steps — one sampled every
    step, one untouched — must end BITWISE-identical. This is the core
    'byte-identical with the probe present' guarantee: the probe only reads."""
    import numpy as np
    from sim.activity_probe import RegionActivityProbe

    N = 40

    # Bridge A: no probe.
    bridge_a = _build_toy_region_bridge(seed=42)
    for _ in range(N):
        bridge_a._run_one_simulation_step()
    fired_a = np.asarray(bridge_a.cp_firing_states).copy()
    v_a = np.asarray(bridge_a.cp_membrane_potential_v).copy()

    # Bridge B: identical seed, probe.sample() called EVERY step.
    bridge_b = _build_toy_region_bridge(seed=42)
    probe = RegionActivityProbe(bridge_b)
    for _ in range(N):
        bridge_b._run_one_simulation_step()
        probe.sample(bridge_b)  # the viz read — must be side-effect-free
    fired_b = np.asarray(bridge_b.cp_firing_states).copy()
    v_b = np.asarray(bridge_b.cp_membrane_potential_v).copy()

    # Bitwise identity: the probe did not perturb dynamics at all.
    assert np.array_equal(fired_a, fired_b), \
        "probe perturbed firing states (NOT side-effect-free)"
    assert np.array_equal(v_a, v_b), \
        "probe perturbed membrane potential (NOT side-effect-free)"


@pytest.mark.slow
def test_probe_overhead_is_small(numpy_backend, capsys):
    """OVERHEAD: measure step-loop time without the probe vs with the probe
    sampled every step. Report the per-step overhead so the no-bottleneck claim
    is grounded. The reduction is O(regions+pathways); assert a loose ceiling
    (the per-sample cost must be well under a step's cost)."""
    from sim.activity_probe import RegionActivityProbe

    N = 60

    # Baseline: steps only.
    bridge_a = _build_toy_region_bridge(seed=7)
    for _ in range(5):  # warmup
        bridge_a._run_one_simulation_step()
    t0 = time.perf_counter()
    for _ in range(N):
        bridge_a._run_one_simulation_step()
    base_s = time.perf_counter() - t0

    # With probe sampled every step.
    bridge_b = _build_toy_region_bridge(seed=7)
    probe = RegionActivityProbe(bridge_b)
    for _ in range(5):
        bridge_b._run_one_simulation_step()
        probe.sample(bridge_b)
    t0 = time.perf_counter()
    for _ in range(N):
        bridge_b._run_one_simulation_step()
        probe.sample(bridge_b)
    probe_s = time.perf_counter() - t0

    # Isolate the probe-only cost (sample without stepping) for a clean number.
    t0 = time.perf_counter()
    for _ in range(N):
        probe.sample(bridge_b)
    sample_only_s = time.perf_counter() - t0

    per_step_base_ms = base_s / N * 1e3
    per_step_probe_ms = probe_s / N * 1e3
    per_sample_ms = sample_only_s / N * 1e3
    overhead_ms = per_step_probe_ms - per_step_base_ms

    # Print so the measured overhead shows up with `pytest -s`.
    with capsys.disabled():
        print(
            f"\n[activity-probe overhead] step-only={per_step_base_ms:.3f} ms  "
            f"step+sample(every step)={per_step_probe_ms:.3f} ms  "
            f"delta/step={overhead_ms:+.3f} ms  "
            f"sample-only={per_sample_ms:.3f} ms/sample  "
            f"(2 regions / 1 pathway, numpy)"
        )

    # A single sample (the per-region reduction + host transfer) is cheap in
    # absolute terms. Loose ceiling to catch an accidental O(neurons) blowup
    # without being flaky on a loaded CI box.
    assert per_sample_ms < 50.0, (
        f"probe sample took {per_sample_ms:.2f} ms — unexpectedly expensive "
        f"for a {probe.n_regions}-region reduction"
    )
    # NOTE: in production the probe is sampled every Nth step (throttled), so
    # the amortized per-step overhead is this number / N — even smaller.


@pytest.mark.slow
def test_probe_requires_region_manager(numpy_backend):
    """Without the brain-region framework the probe raises (it needs the
    per-region neuron slices) — a clear error, not a silent no-op."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.activity_probe import RegionActivityProbe

    cfg = CoreSimConfig()
    cfg.num_neurons = 30
    cfg.dt = 1.0
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()
    assert bridge.region_manager is None
    with pytest.raises(RuntimeError):
        RegionActivityProbe(bridge)


@pytest.mark.slow
def test_probe_integration_with_emit_activity(numpy_backend):
    """End-to-end: probe.sample() -> emit_activity() produces a parseable
    [ACTIVITY] line whose region rates round-trip."""
    import io
    from sim.activity_probe import RegionActivityProbe
    from sim.progress import emit_activity, parse_activity_line

    bridge = _build_toy_region_bridge(seed=42)
    probe = RegionActivityProbe(bridge)
    for _ in range(15):
        bridge._run_one_simulation_step()
    regions, flux = probe.sample(bridge)

    buf = io.StringIO()
    emit_activity(bridge.runtime_state.current_time_ms, regions, flux,
                  step=15, file=buf)
    line = buf.getvalue().rstrip("\n")
    parsed = parse_activity_line(line)
    assert parsed is not None
    assert set(parsed["regions"].keys()) == {"cortex", "motor"}
    assert parsed["step"] == 15
