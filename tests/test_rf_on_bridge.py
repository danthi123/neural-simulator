"""Resonate-and-fire ON the SimulationBridge — minimal de-risk (owner-funded FHRR Option A, step 1).

Design: docs/plans/2026-06-05-rf-on-bridge-derisk-design.md. Proves the bridge can natively host resonate-and-fire
(RF) phasor neurons: a RESONATE_AND_FIRE model branch rotates (re,im)=(v,u) by exp(lambda+i*omega) each step and
fires on the upward Im zero-crossing; the spike step encodes the kick's phase. The kick is set by rf_kick(); the
recovered phases are read by rf_read_phases(). Gates: phase readout -> bind/unbind/bundle -> the composer task.

Reuses resonate_fire_fhrr.py's constants/helpers by import (NOT modified). NO bolted-on numpy: the resonate + the
phase readout run IN the bridge's own step on the bridge's own neuron-state arrays.
"""
import numpy as np
import pytest

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge

# Mirror the reference's cycle length (one phasor cycle = T bridge steps).
from research.runners.spiking_phasor_fhrr import CYCLE_STEPS


def _build_rf_bridge(n_neurons, seed=42):
    """A minimal bridge: allocate as Izhikevich (so v/u exist + init runs), then switch the step dynamics to
    RESONATE_AND_FIRE. No wiring, no plasticity, no noise -> a clean RF resonate readout."""
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n_neurons)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem",
              "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0  # no noise -> deterministic phase
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # Switch the per-step dynamics to RF (v/u already allocated by the Izhikevich init).
    bridge.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return bridge


def _circ_dist(a, b):
    """Circular distance between two phases in [0,1)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5)


def test_rf_phase_readout_on_bridge():
    """Gate 1: kick RF neurons with known phases -> the bridge's RF step reads them back as spike timing."""
    phases = np.arange(0.0, 1.0, 0.1)  # 10 distinct phases
    bridge = _build_rf_bridge(len(phases))
    kick = np.exp(2j * np.pi * phases)

    bridge.rf_kick(kick)
    for _ in range(CYCLE_STEPS + 8):
        bridge._run_one_simulation_step()
    recovered = np.asarray(bridge.rf_read_phases())

    err = _circ_dist(recovered, phases)
    assert float(np.mean(err)) < 0.02, (
        f"RF phase readout error {float(np.mean(err)):.4f} too high; "
        f"recovered={np.round(recovered, 3)} expected={np.round(phases, 3)}"
    )


def test_rf_bind_unbind_bundle_on_bridge():
    """Gate 2: the FHRR ops (bind=phasor_a*phasor_b, unbind=phasor_c*conj(a), bundle=sum) computed by kicking the
    bridge's RF neurons with the right complex value and reading the resonated phase back."""
    D = 64
    rng = np.random.default_rng(7)
    pa, pb, pc = (rng.uniform(0.0, 1.0, D) for _ in range(3))
    za, zb, zc = (np.exp(2j * np.pi * p) for p in (pa, pb, pc))
    bridge = _build_rf_bridge(D)

    def _resonate(kick):
        bridge.rf_kick(kick)
        for _ in range(CYCLE_STEPS + 8):
            bridge._run_one_simulation_step()
        return np.asarray(bridge.rf_read_phases())

    # bind: phase(a)+phase(b)
    assert float(np.mean(_circ_dist(_resonate(za * zb), (pa + pb) % 1.0))) < 0.03
    # unbind: phase(c)-phase(a)
    assert float(np.mean(_circ_dist(_resonate(zc * np.conj(za)), (pc - pa) % 1.0))) < 0.03
    # bundle: phase of the complex sum
    expected_bundle = (np.angle(za + zb + zc) / (2.0 * np.pi)) % 1.0
    assert float(np.mean(_circ_dist(_resonate(za + zb + zc), expected_bundle))) < 0.03


def test_rf_composer_task_on_bridge():
    """Gate 3 (the de-risk verdict): the project's compositional task at smoke scale, with EVERY resonate-and-fire
    operation routed through the bridge's RF step -> accuracy clears the frozen 0.80 bar AND abstention separates,
    matching the numpy reference (research/runners/spiking_phasor_fhrr.py). Proves the FHRR composition runs on the
    bridge's own neurons."""
    from research.findings.raw._rf_on_bridge_probe import run
    results, verdict = run(D=128, n_cues=8, n_fillers=8, loads=[2, 3], trials=3, period=1000, seed=42)
    assert verdict == "GO", f"RF-on-bridge composer task NEGATIVE: {results}"
    for load in (2, 3):
        assert results[load]["accuracy"] >= 0.80, f"load {load} acc {results[load]['accuracy']}"
        assert results[load]["abstention_separates"], f"load {load} abstention failed: {results[load]}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
