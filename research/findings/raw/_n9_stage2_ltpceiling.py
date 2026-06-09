"""Can DA-gated LTP push w_near to the critic-firing band (~30) from the DISTINCT sparse code
(max_int=600, diff-cos=0.21) while w_far stays low? If w_near PLATEAUS well below firing despite
aggressive training, that's the definitive negative (the sparse-distinct code can't reach the MSN
rheobase via learnable synapses). Also test: concentrate the sparse ensemble onto a SMALLER critic
(more current/cell), and a higher reward_lr / w_max."""
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
MAX_INT = 600.0  # the DISTINCT regime (diff-cos ~0.21)


def cosd(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b)/(na*nb)) if na and nb else 0.0


def run(p2v_init, p2v_density, n_strio, reward_lr, w_max, teach_pa, n_train, lm_w=28.0):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=400, n_strio=n_strio, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=lm_w, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=p2v_init, place_to_value_density=p2v_density,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=reward_lr, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    cfg.stdp_w_max = w_max
    si = xp.asarray(_idx(bridge,"landmark_sensors")); pi = xp.asarray(_idx(bridge,"place"))
    ci = xp.asarray(_idx(bridge,"striosome_value")); sni = xp.asarray(_idx(bridge,"snc"))
    ls = {n: landmark_sensor_act(*LOCS[n], landmarks, n_bearing, n_dist, MAX_INT, falloff, dist_sigma, dist_max, bexp) for n in LOCS}
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
    wn0 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    bridge.set_plasticity_gate("value_input", 1.0)
    wtrace = []
    for t in range(n_train):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sni] = xp.float32(180.0); _step(bridge, 40)
        if getattr(bridge,"cp_eligibility_trace",None) is not None: bridge.cp_eligibility_trace[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[si] = xp.asarray(ls["near"], dtype=xp.float32)
        bridge.cp_external_input_current[sni] = xp.float32(480.0)
        if teach_pa > 0: bridge.cp_external_input_current[ci] = xp.float32(teach_pa)
        _step(bridge, 40)
        if t % 50 == 0 or t == n_train-1:
            wtrace.append((t, _mean_w(bridge,"place","striosome_value",pre_subset=near_set)))
    bridge.set_plasticity_gate("value_input", 0.0); bridge.cp_external_input_current[:] = 0.0
    wn1 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    wf1 = _mean_w(bridge,"place","striosome_value",pre_subset=far_set)
    near = _critic_rate_at_location(bridge, xp, si, ci, ls["near"], n_steps=120, warmup=30)
    far_max = max(_critic_rate_at_location(bridge, xp, si, ci, ls[n], n_steps=120, warmup=30) for n in ["far_a","far_b","far_c"])
    del bridge
    return dict(diffcos=diffcos, wn0=wn0, wn1=wn1, wf1=wf1, near=near, far_max=far_max, wtrace=wtrace,
                n_near=len(near_set), n_far=len(far_set))


print("\n=== LTP ceiling test (DISTINCT max_int=600). Can w_near reach the firing band ~30? ===", flush=True)
CONFIGS = [
    # (p2v_init, density, n_strio, reward_lr, w_max, teach, n_train)
    (1.0, 0.6, 80, 0.30, 40.0, 600.0, 300),   # aggressive LTP, standard critic
    (1.0, 0.6, 80, 0.60, 60.0, 600.0, 300),   # higher lr + w_max
    (2.0, 0.9, 30, 0.30, 60.0, 600.0, 300),   # dense onto SMALL critic (more current/cell)
    (4.0, 0.9, 20, 0.30, 60.0, 600.0, 200),   # very small critic, higher init
]
for (p2v, dens, ns, lr, wm, teach, ntr) in CONFIGS:
    r = run(p2v, dens, ns, lr, wm, teach, ntr)
    ratio = r["near"]/max(r["far_max"],1e-3)
    wt = " ".join(f"t{t}:{w:.1f}" for t,w in r["wtrace"])
    print(f"  p2v={p2v} dens={dens} n_strio={ns} lr={lr} wmax={wm} n={ntr}: diff-cos={r['diffcos']:.3f}", flush=True)
    print(f"      w_near {r['wn0']:.2f}->{r['wn1']:.2f} (trace {wt}) w_far->{r['wf1']:.2f} "
          f"| NEAR={r['near']:.2f}Hz FAR_max={r['far_max']:.2f}Hz ratio={ratio:.2f} (n_near={r['n_near']})", flush=True)
