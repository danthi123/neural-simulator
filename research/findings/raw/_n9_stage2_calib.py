"""Calibrate the Stage-2 sparse place->critic arm: what init weight fires the critic from a sparse
place ensemble (~14-30 active cells), and does NEAR vs FAR differ at a fixed weight (the position-
blindness risk on the PLASTIC arm)? Also test a critic-teacher bootstrap (briefly depolarize the
critic during NEAR co-presentation so STDP grabs a foothold, then LTP grows the NEAR arm)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, bk = get_backend(); print("backend:", bk, flush=True)

from research.runners.n9_place_graded_critic_stage2_derisk import (
    _build, _idx, _host, _step, _tick, _drive_landmarks, _critic_rate_at_location,
    _place_ensemble, _mean_w, _calibrate_da, landmark_sensor_act,
    default_locations, default_landmarks)

SEED = 42; GRID = 32
locations = default_locations(GRID); landmarks = default_landmarks(GRID)
n_bearing, n_dist, bexp, dist_sigma = 12, 8, 4.0, 4.0
max_int, falloff = 450.0, 0.03
dist_max = GRID * 1.42
n_sensors = len(landmarks) * (n_bearing + n_dist)


def build_and_selforg(p2v_weight, p2v_density=0.6, passes=8, steps=100):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=400, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=28.0, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=p2v_weight, place_to_value_density=p2v_density,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=0.12, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    si = xp.asarray(_idx(bridge, "landmark_sensors")); pi = xp.asarray(_idx(bridge, "place"))
    ci = xp.asarray(_idx(bridge, "striosome_value"))
    loc_sensor = {n: landmark_sensor_act(*locations[n], landmarks, n_bearing, n_dist, max_int,
                                         falloff, dist_sigma, dist_max, bexp) for n in locations}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(passes):
        order = list(locations); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, steps)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg, si, pi, ci, loc_sensor


print("\n=== sparse place ensemble active counts per location ===", flush=True)
bridge, cfg, si, pi, ci, loc_sensor = build_and_selforg(0.5)
for n in locations:
    e = _place_ensemble(bridge, xp, si, pi, loc_sensor[n])
    print(f"  {n:11s} active={int((np.asarray(e)>0).sum()):3d}/400  meanrate~{float(np.asarray(e).mean()):.2f}", flush=True)
del bridge

print("\n=== critic INIT-fire (NO value training) at place->value weight sweep ===", flush=True)
print("    (the question: is there a weight where NEAR fires but FAR stays silent? = position-blind risk)", flush=True)
for w in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]:
    bridge, cfg, si, pi, ci, loc_sensor = build_and_selforg(w)
    rates = {}
    for n in locations:
        rates[n] = _critic_rate_at_location(bridge, xp, si, ci, loc_sensor[n], n_steps=120, warmup=30)
    near = rates["near"]; far_max = max(rates[n] for n in locations if n.startswith("far"))
    print(f"  w={w:5.1f}: NEAR={near:6.2f}Hz  FAR_max={far_max:6.2f}Hz  "
          f"ratio={near/max(far_max,1e-3):5.2f}  all={ {k: round(v,1) for k,v in rates.items()} }", flush=True)
    del bridge

# Critic-teacher bootstrap: at a low init weight (NEAR can't fire alone), inject a brief teacher current
# into the critic DURING the NEAR place drive + reward, so STDP forms eligibility on the co-firing NEAR
# synapses; then LTP should grow w_near. Test whether this opens NEAR>>FAR.
print("\n=== teacher-bootstrap: low init weight + critic teacher during NEAR-train ===", flush=True)
for (init_w, teach_pa, n_tr) in [(0.5, 350.0, 40), (1.0, 350.0, 40), (0.5, 400.0, 60)]:
    bridge, cfg, si, pi, ci, loc_sensor = build_and_selforg(init_w)
    sni = xp.asarray(_idx(bridge, "snc"))
    _calibrate_da(bridge, cfg, sni, 180.0, xp)
    near_set = set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(_place_ensemble(bridge,xp,si,pi,loc_sensor["near"]))>0)[0])
    far_set = set()
    for fn in [n for n in locations if n.startswith("far")]:
        far_set |= set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(_place_ensemble(bridge,xp,si,pi,loc_sensor[fn]))>0)[0])
    far_set -= near_set
    wn0 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    wf0 = _mean_w(bridge,"place","striosome_value",pre_subset=far_set)
    bridge.set_plasticity_gate("value_input", 1.0)
    for t in range(n_tr):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[sni] = xp.float32(180.0); _step(bridge, 40)
        if getattr(bridge,"cp_eligibility_trace",None) is not None: bridge.cp_eligibility_trace[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[si] = xp.asarray(loc_sensor["near"], dtype=xp.float32)
        bridge.cp_external_input_current[sni] = xp.float32(180.0 + 300.0)
        bridge.cp_external_input_current[ci] = xp.float32(teach_pa)   # CRITIC TEACHER
        _step(bridge, 40)
    bridge.set_plasticity_gate("value_input", 0.0); bridge.cp_external_input_current[:] = 0.0
    wn1 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
    wf1 = _mean_w(bridge,"place","striosome_value",pre_subset=far_set)
    near = _critic_rate_at_location(bridge, xp, si, ci, loc_sensor["near"], n_steps=120, warmup=30)
    far_max = max(_critic_rate_at_location(bridge, xp, si, ci, loc_sensor[n], n_steps=120, warmup=30)
                  for n in locations if n.startswith("far"))
    print(f"  init={init_w} teach={teach_pa} n={n_tr}: w_near {wn0:.2f}->{wn1:.2f} w_far {wf0:.2f}->{wf1:.2f} "
          f"(near/far {wn1/max(wf1,1e-6):.2f}) | NEAR={near:.2f}Hz FAR_max={far_max:.2f}Hz ratio={near/max(far_max,1e-3):.2f}", flush=True)
    del bridge
