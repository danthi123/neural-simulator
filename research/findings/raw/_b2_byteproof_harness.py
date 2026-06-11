"""Byte-identity harness for the B-2 conductance-derivative protected edit.

Runs a fixed-seed Izhikevich kernel smoke + a Stage-B critic warm-up with the
edit DEFAULT-OFF, and prints a per-step rolling hash of cp_membrane_potential_v
and cp_firing_states. The hash sequence must be BIT-IDENTICAL between the
pre-edit commit and the post-edit (flag default-False) build — proving the new
guarded block is unreached and total_input_current_pA is byte-identical.

Usage:
    SIM_BACKEND=numpy python research/findings/raw/_b2_byteproof_harness.py
"""
from __future__ import annotations
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "numpy")


def _arr_hash(*arrs):
    import numpy as np
    h = hashlib.sha256()
    for a in arrs:
        try:
            from sim.backend import to_host
            a = to_host(a)
        except Exception:
            pass
        a = np.ascontiguousarray(a)
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def izh_smoke():
    """A tiny Izhikevich bridge with constant drive, stepped for N steps; the
    classic kernel-smoke (no regions, no GABA_B, no TD) — proves the core
    neuron dynamics are byte-identical."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.backend import get_backend
    xp, _ = get_backend()
    cfg = CoreSimConfig()
    cfg.seed = 42
    cfg.heterogeneity_seed = 42
    cfg.ou_seed = 42
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.num_neurons = 200
    cfg.connections_per_neuron = 10
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = 42
    bridge._initialize_simulation_data(called_from_playback_init=False)
    hashes = []
    bridge.cp_external_input_current[:] = xp.float32(250.0)
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * cfg.dt_ms)
        hashes.append(_arr_hash(bridge.cp_membrane_potential_v, bridge.cp_firing_states))
    return hashes


def stageb_warmup():
    """A Stage-B critic bridge (the TD probe's build, NO td flags) warmed up for
    N steps under a CS+US-like drive. Exercises the GABA_B block (enable_gabab on)
    + the brain-region framework + three-factor learning — the same code path the
    B-2 edit guards sit beside, but with enable_td_value_derivative DEFAULT-OFF."""
    from research.runners.snc_stageb_critic_probe import _build_stageb_bridge, _idx
    from sim.backend import get_backend
    xp, _ = get_backend()
    # gabab=True so the GABA_B block runs (the B-2 guard sits INSIDE it); td OFF.
    bridge, cfg = _build_stageb_bridge(
        42, gabab=True, cue_to_strio_weight=20.0, strio_to_snc_weight=2.5)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("cue", "striosome_value", "snc")}
    hashes = []
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[idx_map["cue"]] = xp.float32(600.0)
    bridge.cp_external_input_current[idx_map["snc"]] = xp.float32(220.0 + 400.0)
    for _ in range(60):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * cfg.dt_ms)
        hashes.append(_arr_hash(bridge.cp_membrane_potential_v, bridge.cp_firing_states))
    return hashes


def main():
    izh = izh_smoke()
    sb = stageb_warmup()
    combo = hashlib.sha256(("".join(izh) + "|" + "".join(sb)).encode()).hexdigest()[:16]
    print("IZH_FIRST=%s IZH_LAST=%s" % (izh[0], izh[-1]))
    print("SB_FIRST=%s SB_LAST=%s" % (sb[0], sb[-1]))
    print("COMBO=%s" % combo)
    print("IZH_ALL=%s" % "".join(izh))
    print("SB_ALL=%s" % "".join(sb))


if __name__ == "__main__":
    main()
