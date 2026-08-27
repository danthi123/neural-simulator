"""One-brain INTEGRATION phase — feasibility SMOKE for a LEARNED cross-region edge (the F2 crux).

This is a DESIGN-support smoke, not a GO result. It demonstrates the single mechanism the whole integration
phase rests on, on the REAL SimulationBridge substrate (numpy, tiny net, CPU):

  a cross-region synapse A->B that starts near-ZERO, GROWS from Hebbian co-activity (the substrate's own
  plasticity — NOT a hand-set weight matrix), and is then LOAD-BEARING + LESIONABLE:
    - VARY: with the edge present, driving source A raises target B's read above its A-silent baseline.
    - LESION: zero the grown edge; the A-on-vs-off difference VANISHES (the coupling caused it).

This is the design's FUNCTIONAL-INTEGRATION gate F2 (vary-then-lesion) in miniature, and the emergence bar
(§2): the cross-edge is the ONLY plastic synapse (plastic=True; every structural edge plastic=False), so it
learns from experience via `_apply_branchless_hebbian`, gated by the substrate's own gain, not host-wired.

Abstract stand-in for the first real interaction (d6 WM referent -> comprehension role accumulator): region
`src` = a held-referent assembly; region `tgt` = a role/decision accumulator with weak (ambiguous) drive.

NO sim/ edit. Forces the numpy backend so it never touches the GPU.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU (a GPU job may be running)

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host
from tools.lab import attributable_to

SRC_N, TGT_N, TGT_FS_N = 60, 60, 20
CROSS = "src_to_tgt_learned"          # the ONE learned cross-region population
CROSS_GATE = "src_to_tgt_gate"
W0 = 0.05                             # near-zero seed weight (the edge must GROW, not be pre-wired)
DRIVE_PA = 2000.0                    # co-drive during experience (regions plateau ~20 Hz -> trace ~0.2)
WEAK_PA = 95.0                       # sub/near-threshold "ambiguous" drive into tgt at test
SETTLE, TRAIN, TEST = 40, 150, 60


def _dense(pre, post, w, gate, plastic):
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    P = np.repeat(pre, len(post)); Q = np.tile(post, len(pre))
    return {"pre_indices": P, "post_indices": Q,
            "initial_weights": np.full(P.size, float(w), np.float32),
            "plastic": bool(plastic), "plasticity_gate": gate, "conn_type": "E_TO_E", "count": int(P.size)}


def _build(seed):
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="src", n_neurons=SRC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="tgt", n_neurons=TGT_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="tgt_fs", n_neurons=TGT_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [   # tgt lateral inhibition (a baseline decision circuit); FIXED (plastic=False)
        RegionPathway(from_region="tgt", to_region="tgt_fs", density=0.5, weight_mean=6.0,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="tgt_fs", to_region="tgt", density=0.5, weight_mean=7.0,
                      weight_jitter=0.0, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_hebbian_learning = True            # the substrate's OWN plasticity grows the cross-edge
    # RATE-WINDOW (BCM) Hebbian: two neurons BOTH active over a ~10-step window potentiate regardless of
    # exact-step spike alignment — the associative rule for asynchronously-firing populations (the per-step
    # coincidence rule cannot form a cross-assembly link from sparse firing).
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_thresh = 0.02          # trace[pre]*trace[post] gate (~0.2^2=0.04 during co-drive)
    cfg.hebbian_learning_rate = 0.05
    cfg.hebbian_max_weight = 40.0                 # F3: bounded — no runaway
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_homeostasis = False                # synaptic-scaling clip foot-gun (CLAUDE.md) — OFF
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    src_idx = np.asarray(rm.indices("src"), np.int64)
    tgt_idx = np.asarray(rm.indices("tgt"), np.int64)

    plan = dict(rm.build_wiring_plan(seed=int(seed)))
    plan[CROSS] = _dense(src_idx, tgt_idx, W0, CROSS_GATE, plastic=True)   # near-zero, PLASTIC
    inh = []
    for r in rm.regions():
        inh.extend(rm.inhibitory_indices(r.name))
    bridge.inject_explicit_wiring(plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(CROSS_GATE, 1.0)   # ONLY the cross-edge learns; all structural edges plastic=False

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    return bridge, xp, src_idx, tgt_idx


_STATE = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_firing_states",
          "cp_prev_firing_states", "cp_hebb_coactivity_trace")


def _snapshot(bridge, xp):
    snap = {}
    for a in _STATE:
        v = getattr(bridge, a, None)
        if v is not None:
            snap[a] = xp.asarray(to_host(v)).copy()
    return snap


def _restore(bridge, xp, snap):
    for a, v in snap.items():
        getattr(bridge, a)[:] = v
    bridge.cp_external_input_current[:] = 0.0


def _cross_mask(bridge, src_idx, tgt_idx):
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    return np.isin(row, src_idx) & np.isin(col, tgt_idx)


def _cross_w(bridge, mask):
    return float(np.asarray(to_host(bridge.cp_connections.data))[mask].mean())


def _tgt_rate(bridge, xp, src_idx, tgt_idx, drive_src, steps, snap=None):
    if snap is not None:
        _restore(bridge, xp, snap)          # common quiescent start -> no cross-condition history confound
    sp = 0
    for _ in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[xp.asarray(tgt_idx)] = xp.float32(WEAK_PA)   # ambiguous drive
        if drive_src:
            bridge.cp_external_input_current[xp.asarray(src_idx)] = xp.float32(DRIVE_PA)
        bridge._run_one_simulation_step()
        sp += int(to_host(bridge.cp_firing_states[xp.asarray(tgt_idx)].astype(xp.float64).sum()))
    bridge.cp_external_input_current[:] = 0.0
    return sp / float(steps * tgt_idx.size)


def run(seed=42):
    bridge, xp, src_idx, tgt_idx = _build(seed)
    mask = _cross_mask(bridge, src_idx, tgt_idx)
    w_init = _cross_w(bridge, mask)

    # EXPERIENCE: co-drive src+tgt so the referent is "held" while its role fires -> Hebbian grows the edge.
    for _ in range(TRAIN):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[xp.asarray(src_idx)] = xp.float32(DRIVE_PA)
        bridge.cp_external_input_current[xp.asarray(tgt_idx)] = xp.float32(DRIVE_PA)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms   # advance time (direct-step loop)
    bridge.cp_external_input_current[:] = 0.0
    w_grown = _cross_w(bridge, mask)

    # FREEZE plasticity for the read (the grown edge is now fixed; the read must not keep learning).
    bridge.core_config.enable_hebbian_learning = False
    bridge.set_plasticity_gate(CROSS_GATE, 0.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE):
        bridge._run_one_simulation_step()
    snap = _snapshot(bridge, xp)             # common quiescent state restored before every read condition

    # VARY: A-on vs A-off, with the grown edge in place.
    rate_on = _tgt_rate(bridge, xp, src_idx, tgt_idx, True, TEST, snap)
    rate_off = _tgt_rate(bridge, xp, src_idx, tgt_idx, False, TEST, snap)
    delta = rate_on - rate_off

    # LESION: zero the grown cross-edge; the A-on-vs-off difference must vanish.
    data = np.asarray(to_host(bridge.cp_connections.data)).copy()
    data[mask] = 0.0
    bridge.cp_connections.data = xp.asarray(data, dtype=bridge.cp_connections.data.dtype)
    rate_on_les = _tgt_rate(bridge, xp, src_idx, tgt_idx, True, TEST, snap)
    rate_off_les = _tgt_rate(bridge, xp, src_idx, tgt_idx, False, TEST, snap)
    delta_les = rate_on_les - rate_off_les

    grew = w_grown > w_init + 1e-3
    load_bearing = delta > 0.02
    # ATTRIBUTION (tools.lab): what fraction of the src-on-vs-off effect is attributable to the cross-edge?
    # treatment = the effect with the edge INTACT (delta); control = the effect with it LESIONED (delta_les).
    # The vary-then-lesion F2 test PASSES only if ~all of the effect is the edge's (it vanishes on lesion).
    frac_attributable = attributable_to(f"cross-edge coupling (seed {seed})", delta, delta_les)
    lesion_removes = frac_attributable is not None and frac_attributable > 0.7
    return {
        "seed": int(seed), "w_init": w_init, "w_grown": w_grown, "edge_grew": bool(grew),
        "rate_on": rate_on, "rate_off": rate_off, "delta": delta,
        "rate_on_lesion": rate_on_les, "rate_off_lesion": rate_off_les, "delta_lesion": delta_les,
        "frac_attributable_to_edge": (None if frac_attributable is None else float(frac_attributable)),
        "load_bearing": bool(load_bearing), "lesion_removes_effect": bool(lesion_removes),
        "PASS": bool(grew and load_bearing and lesion_removes),
    }


if __name__ == "__main__":
    import contextlib
    import sys
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44"])]
    with contextlib.redirect_stdout(sys.stderr):        # keep attribution/log prints off the JSON stream
        out = [run(s) for s in seeds]
    allpass = all(r["PASS"] for r in out)
    print(json.dumps({
        "provenance": {"runner": "research.runners._onebrain_integration_crossedge_smoke",
                       "argv": sys.argv, "backend": os.environ.get("SIM_BACKEND", "numpy")},
        "seeds": seeds, "all_pass": allpass, "n_seeds": len(out), "runs": out,
    }, indent=2))
