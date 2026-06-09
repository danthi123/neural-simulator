"""Last real chance for the hypothesis: an INTERMEDIATE place-code intensity where the ensemble fires
enough (~3-8 spk/step) to (a) nudge the critic and (b) give STDP fast pairing, so DA-gated LTP can
SHARPEN NEAR-selectivity beyond the input overlap (the LTP carves a narrower value field than the place
code's raw overlap). Faithful: NO critic teacher -- the critic must reach the up-state from the LEARNED
place->critic synapses alone. Sweep intensity x init x training-length; report trained NEAR/FAR."""
from __future__ import annotations
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, bk = get_backend(); print("backend:", bk, flush=True)

from research.runners.n9_place_graded_critic_stage2_derisk import (
    _build, _idx, _host, _step, _tick, _place_ensemble, _critic_rate_at_location, _mean_w,
    _calibrate_da, landmark_sensor_act, default_landmarks)

SEED = 42; GRID = 32
landmarks = default_landmarks(GRID)
n_bearing, n_dist, bexp, dist_sigma = 12, 8, 4.0, 4.0
falloff = 0.03; dist_max = GRID * 1.42
n_sensors = len(landmarks) * (n_bearing + n_dist)
g = GRID - 1
LOCS = {"near": (g*0.75, g*0.25), "far_a": (g*0.25, g*0.75),
        "far_b": (g*0.50, g*0.50), "far_c": (g*0.25, g*0.25)}


def cosd(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b)/(na*nb)) if na and nb else 0.0


def run(max_int, p2v_init, n_train, reward_lr=0.2):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=400, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=28.0, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=p2v_init, place_to_value_density=0.6,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=reward_lr, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    cfg.stdp_w_max = 60.0
    si = xp.asarray(_idx(bridge,"landmark_sensors")); pi = xp.asarray(_idx(bridge,"place"))
    ci = xp.asarray(_idx(bridge,"striosome_value")); sni = xp.asarray(_idx(bridge,"snc"))
    ls = {n: landmark_sensor_act(*LOCS[n], landmarks, n_bearing, n_dist, max_int, falloff, dist_sigma, dist_max, bexp) for n in LOCS}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(10):
        order = list(LOCS); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(ls[name], dtype=xp.float32); _step(bridge, 120)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    ens = {n: _place_ensemble(bridge, xp, si, pi, ls[n]) for n in LOCS}
    diffcos = np.mean([cosd(ens[a], ens[b]) for a,b in itertools.combinations(LOCS,2)])
    _calibrate_da(bridge, cfg, sni, 180.0, xp)
    near_set = set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(ens["near"])>0)[0])
    far_set = set()
    for fn in ["far_a","far_b","far_c"]:
        far_set |= set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(ens[fn])>0)[0])
    far_set -= near_set
    crit_init = _critic_rate_at_location(bridge, xp, si, ci, ls["near"], n_steps=120, warmup=30)
    wn0 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    bridge.set_plasticity_gate("value_input", 1.0)
    v_curve = []
    for t in range(n_train):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sni] = xp.float32(180.0); _step(bridge, 40)
        if getattr(bridge,"cp_eligibility_trace",None) is not None: bridge.cp_eligibility_trace[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[si] = xp.asarray(ls["near"], dtype=xp.float32)
        bridge.cp_external_input_current[sni] = xp.float32(480.0)
        spk = 0
        for _ in range(40):
            _tick(bridge); spk += int(bridge.cp_firing_states[ci].sum())
        v_curve.append(spk/80/(40e-3))
    bridge.set_plasticity_gate("value_input", 0.0); bridge.cp_external_input_current[:] = 0.0
    wn1 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    wf1 = _mean_w(bridge,"place","striosome_value",pre_subset=far_set)
    near = _critic_rate_at_location(bridge, xp, si, ci, ls["near"], n_steps=120, warmup=30)
    far_each = {n: _critic_rate_at_location(bridge, xp, si, ci, ls[n], n_steps=120, warmup=30) for n in ["far_a","far_b","far_c"]}
    far_max = max(far_each.values())
    del bridge
    return dict(diffcos=diffcos, crit_init=crit_init, wn0=wn0, wn1=wn1, wf1=wf1, near=near,
                far_max=far_max, far_each=far_each, v_early=np.mean(v_curve[:8]), v_late=np.mean(v_curve[-8:]))


print("\n=== intermediate-intensity sweep (NO teacher; LTP must self-bootstrap + sharpen) ===", flush=True)
print("    target: NEAR>=5Hz, ratio>=3, ideally diff-cos<0.5 (LTP sharpens beyond input overlap)", flush=True)
for (mi, p2v, ntr) in [(750.0, 1.0, 80), (900.0, 1.0, 80), (1000.0, 1.0, 100),
                       (900.0, 2.0, 100), (1100.0, 1.0, 120)]:
    r = run(mi, p2v, ntr)
    ratio = r["near"]/max(r["far_max"],1e-3)
    print(f"  max_int={mi:5.0f} p2v={p2v} n={ntr:3d}: diff-cos={r['diffcos']:.3f} crit_init={r['crit_init']:.1f} "
          f"V {r['v_early']:.1f}->{r['v_late']:.1f} | w_near {r['wn0']:.2f}->{r['wn1']:.2f} w_far->{r['wf1']:.2f} "
          f"| NEAR={r['near']:.1f}Hz FAR_max={r['far_max']:.1f}Hz ratio={ratio:.2f}", flush=True)
    print(f"      far_each={ {k: round(v,1) for k,v in r['far_each'].items()} }", flush=True)
