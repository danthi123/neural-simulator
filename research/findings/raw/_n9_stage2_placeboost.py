"""Can the place ensemble fire ROBUSTLY (a real in-field place-cell rate, 5-30 Hz/cell) AND stay
distinct-per-location? The Stage-1 threshold-only WTA gives sparse codes but ~1 spk/step total (too
weak to drive an MSN critic). Test boosting sensory intensity / place excitability so the ensemble is
a real up-state driver, while checking position-specificity survives. Then test if THAT drives + grows
the critic NEAR>>FAR."""
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
# Use 4 well-separated locations with healthy ensembles (from placefire probe).
LOCS = {"near": (g*0.75, g*0.25), "far_a": (g*0.25, g*0.75),
        "far_b": (g*0.50, g*0.50), "far_c": (g*0.25, g*0.25)}


def cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b)/(na*nb)) if na and nb else 0.0


def build_selforg(lm_w, max_int, passes=10, steps=120, n_place=400):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=n_place, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=lm_w, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=0.5, place_to_value_density=0.6,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=0.12, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    si = xp.asarray(_idx(bridge, "landmark_sensors")); pi = xp.asarray(_idx(bridge, "place"))
    ci = xp.asarray(_idx(bridge, "striosome_value")); sni = xp.asarray(_idx(bridge, "snc"))
    ls = {n: landmark_sensor_act(*LOCS[n], landmarks, n_bearing, n_dist, max_int,
                                 falloff, dist_sigma, dist_max, bexp) for n in LOCS}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(passes):
        order = list(LOCS); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(ls[name], dtype=xp.float32)
            _step(bridge, steps)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg, si, pi, ci, sni, ls


def place_rate_and_ens(bridge, si, pi, sensor_act, n_steps=80, warmup=20):
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[si] = xp.asarray(sensor_act, dtype=xp.float32)
    n = int(pi.size); counts = xp.zeros(n, dtype=xp.float32); spk = 0; m = 0
    for t in range(n_steps):
        _tick(bridge)
        counts += bridge.cp_firing_states[pi].astype(xp.float32)
        if t >= warmup: spk += int(bridge.cp_firing_states[pi].sum()); m += 1
    bridge.cp_external_input_current[:] = 0.0
    return spk/max(m,1), _host(counts)


print("\n=== place ensemble rate + distinctness vs sensory intensity (lm_w=28) ===", flush=True)
for mi in [450.0, 900.0, 1800.0, 3000.0]:
    bridge, cfg, si, pi, ci, sni, ls = build_selforg(28.0, mi)
    ens = {}; rates = {}
    for n in LOCS:
        r, e = place_rate_and_ens(bridge, si, pi, ls[n]); rates[n] = r; ens[n] = e
    diffcos = np.mean([cos(ens[a], ens[b]) for a,b in itertools.combinations(LOCS,2)])
    print(f"  max_int={mi:6.0f}: place_spk/step " + " ".join(f"{n}={rates[n]:5.1f}" for n in LOCS) +
          f"  diff-cos={diffcos:.3f}", flush=True)
    del bridge

print("\n=== combine: stronger lm_w + stronger sensory; place rate at NEAR ===", flush=True)
for lm_w, mi in [(28.0, 1800.0), (50.0, 1800.0), (50.0, 3000.0), (80.0, 3000.0)]:
    bridge, cfg, si, pi, ci, sni, ls = build_selforg(lm_w, mi)
    ens = {}; rates = {}
    for n in LOCS:
        r, e = place_rate_and_ens(bridge, si, pi, ls[n]); rates[n] = r; ens[n] = e
    diffcos = np.mean([cos(ens[a], ens[b]) for a,b in itertools.combinations(LOCS,2)])
    print(f"  lm_w={lm_w:4.0f} max_int={mi:5.0f}: place_spk/step " +
          " ".join(f"{n}={rates[n]:5.1f}" for n in LOCS) + f"  diff-cos={diffcos:.3f}", flush=True)
    del bridge

# Best config: train the critic with place at strong rate. Does NEAR>>FAR open?
print("\n=== critic training with strong place drive (lm_w=50, max_int=1800, p2v_init sweep) ===", flush=True)
for p2v in [2.0, 6.0, 12.0]:
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=400, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=50.0, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=p2v, place_to_value_density=0.6,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=0.12, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    si = xp.asarray(_idx(bridge,"landmark_sensors")); pi = xp.asarray(_idx(bridge,"place"))
    ci = xp.asarray(_idx(bridge,"striosome_value")); sni = xp.asarray(_idx(bridge,"snc"))
    ls = {n: landmark_sensor_act(*LOCS[n], landmarks, n_bearing, n_dist, 1800.0, falloff, dist_sigma, dist_max, bexp) for n in LOCS}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(10):
        order = list(LOCS); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(ls[name], dtype=xp.float32); _step(bridge, 120)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    _calibrate_da(bridge, cfg, sni, 180.0, xp)
    near_set = set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(_place_ensemble(bridge,xp,si,pi,ls["near"]))>0)[0])
    far_set = set()
    for fn in ["far_a","far_b","far_c"]:
        far_set |= set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(_place_ensemble(bridge,xp,si,pi,ls[fn]))>0)[0])
    far_set -= near_set
    wn0 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    crit_init = _critic_rate_at_location(bridge, xp, si, ci, ls["near"], n_steps=120, warmup=30)
    bridge.set_plasticity_gate("value_input", 1.0)
    for t in range(40):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sni] = xp.float32(180.0); _step(bridge, 40)
        if getattr(bridge,"cp_eligibility_trace",None) is not None: bridge.cp_eligibility_trace[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[si] = xp.asarray(ls["near"], dtype=xp.float32)
        bridge.cp_external_input_current[sni] = xp.float32(480.0); _step(bridge, 40)
    bridge.set_plasticity_gate("value_input", 0.0); bridge.cp_external_input_current[:] = 0.0
    wn1 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    wf1 = _mean_w(bridge,"place","striosome_value",pre_subset=far_set)
    near = _critic_rate_at_location(bridge, xp, si, ci, ls["near"], n_steps=120, warmup=30)
    far_max = max(_critic_rate_at_location(bridge, xp, si, ci, ls[n], n_steps=120, warmup=30) for n in ["far_a","far_b","far_c"])
    print(f"  p2v_init={p2v:4.1f}: crit_init={crit_init:.2f} w_near {wn0:.2f}->{wn1:.2f} w_far->{wf1:.2f} "
          f"(near/far {wn1/max(wf1,1e-6):.2f}) | NEAR={near:.2f}Hz FAR_max={far_max:.2f}Hz ratio={near/max(far_max,1e-3):.2f}", flush=True)
    del bridge
