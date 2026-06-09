"""Instrument the Stage-2 critic: (1) does a direct teacher current fire the MSN-D1 critic (rheobase
check)? (2) what synaptic current does the critic receive from the NEAR place ensemble at various
weights? (3) does the eligibility trace / STDP form on place->critic when BOTH fire (teacher + place)?
This isolates WHY the smoke showed 0 critic firing + 0 weight growth."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, bk = get_backend(); print("backend:", bk, flush=True)

from research.runners.n9_place_graded_critic_stage2_derisk import (
    _build, _idx, _host, _step, _tick, _mean_w, _calibrate_da, _place_ensemble,
    landmark_sensor_act, default_locations, default_landmarks)

SEED = 42; GRID = 32
locations = default_locations(GRID); landmarks = default_landmarks(GRID)
n_bearing, n_dist, bexp, dist_sigma = 12, 8, 4.0, 4.0
max_int, falloff = 450.0, 0.03; dist_max = GRID * 1.42
n_sensors = len(landmarks) * (n_bearing + n_dist)


def build(p2v_weight, p2v_density=0.6):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=400, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=28.0, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=p2v_weight, place_to_value_density=p2v_density,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=0.12, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    return bridge, cfg


# (1) RHEOBASE: direct teacher current into the critic. Find the pA that fires it.
print("\n=== (1) critic MSN-D1 rheobase: direct external current sweep ===", flush=True)
bridge, cfg = build(0.5)
ci = xp.asarray(_idx(bridge, "striosome_value")); n_crit = int(ci.size)
for pa in [300, 339, 350, 380, 420, 500, 600, 800]:
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[ci] = xp.float32(pa)
    spk = 0
    for t in range(120):
        _tick(bridge)
        if t >= 30: spk += int(bridge.cp_firing_states[ci].sum())
    rate = spk / n_crit / (90 * 1e-3)
    print(f"  teacher={pa:4d}pA -> critic {rate:7.2f} Hz", flush=True)
bridge.cp_external_input_current[:] = 0.0
del bridge

# (2) place-driven synaptic current onto the critic, at various weights. Self-organize the place code,
#     then drive NEAR and report the critic's total synaptic input + g_e + V.
print("\n=== (2) place->critic synaptic current from the NEAR ensemble (after self-org) ===", flush=True)
for w in [4.0, 16.0, 30.0, 60.0]:
    bridge, cfg = build(w)
    si = xp.asarray(_idx(bridge, "landmark_sensors")); pi = xp.asarray(_idx(bridge, "place"))
    ci = xp.asarray(_idx(bridge, "striosome_value"))
    loc_sensor = {n: landmark_sensor_act(*locations[n], landmarks, n_bearing, n_dist, max_int,
                                         falloff, dist_sigma, dist_max, bexp) for n in locations}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(8):
        order = list(locations); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, 100)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    # drive NEAR, sample the critic's g_e (excitatory conductance) and membrane V at steady state
    bridge.cp_external_input_current[si] = xp.asarray(loc_sensor["near"], dtype=xp.float32)
    ge_samples = []; v_samples = []; place_active = []
    pidx_h = _idx(bridge, "place")
    for t in range(80):
        _tick(bridge)
        if t >= 40:
            ge = getattr(bridge, "cp_conductance_g_e", None)
            v = getattr(bridge, "cp_membrane_potential_v", None)
            if ge is not None: ge_samples.append(float(_host(ge[ci]).mean()))
            if v is not None: v_samples.append(float(_host(v[ci]).max()))
            place_active.append(int(_host(bridge.cp_firing_states[pi]).sum()))
    n_near = int((np.asarray(_place_ensemble(bridge, xp, si, pi, loc_sensor["near"]))>0).sum())
    print(f"  w={w:5.1f}: NEAR place_active~{np.mean(place_active):.1f}/step (ensemble {n_near} cells) "
          f"critic g_e~{np.mean(ge_samples) if ge_samples else float('nan'):.2f} "
          f"Vmax~{np.max(v_samples) if v_samples else float('nan'):.1f}mV", flush=True)
    bridge.cp_external_input_current[:] = 0.0
    del bridge

# (3) STDP/eligibility on place->critic when BOTH fire (teacher fires critic + place ensemble fires).
print("\n=== (3) does place->critic STDP form when teacher co-fires the critic with NEAR place? ===", flush=True)
bridge, cfg = build(4.0)
si = xp.asarray(_idx(bridge, "landmark_sensors")); pi = xp.asarray(_idx(bridge, "place"))
ci = xp.asarray(_idx(bridge, "striosome_value")); sni = xp.asarray(_idx(bridge, "snc"))
loc_sensor = {n: landmark_sensor_act(*locations[n], landmarks, n_bearing, n_dist, max_int,
                                     falloff, dist_sigma, dist_max, bexp) for n in locations}
bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
rng = np.random.default_rng(SEED)
for _p in range(8):
    order = list(locations); rng.shuffle(order)
    for name in order:
        bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
        bridge.cp_external_input_current[si] = xp.asarray(loc_sensor[name], dtype=xp.float32)
        _step(bridge, 100)
bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
_calibrate_da(bridge, cfg, sni, 180.0, xp)
near_set = set(int(_idx(bridge,"place")[i]) for i in np.where(np.asarray(_place_ensemble(bridge,xp,si,pi,loc_sensor["near"]))>0)[0])
wn0 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
bridge.set_plasticity_gate("value_input", 1.0)
# 30 trials: NEAR place drive (place fires) + 600pA teacher (critic fires) + reward burst (DA high)
elig_max = 0.0
for t in range(30):
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[sni] = xp.float32(180.0); _step(bridge, 40)
    if getattr(bridge,"cp_eligibility_trace",None) is not None: bridge.cp_eligibility_trace[:] = 0.0
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[si] = xp.asarray(loc_sensor["near"], dtype=xp.float32)
    bridge.cp_external_input_current[sni] = xp.float32(180.0 + 300.0)
    bridge.cp_external_input_current[ci] = xp.float32(600.0)  # strong teacher -> critic surely fires
    crit_spk = 0
    for _ in range(40):
        _tick(bridge); crit_spk += int(bridge.cp_firing_states[ci].sum())
    if getattr(bridge,"cp_eligibility_trace",None) is not None:
        elig_max = max(elig_max, float(_host(bridge.cp_eligibility_trace).max()))
    if t in (0, 5, 15, 29):
        wn = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
        da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
        print(f"  t={t:2d} critic_spk={crit_spk:4d} w_near={wn:.3f} elig_max={elig_max:.4f} DA={da:.3f}", flush=True)
bridge.set_plasticity_gate("value_input", 0.0)
wn1 = _mean_w(bridge,"place","striosome_value",pre_subset=near_set)
print(f"  RESULT w_near {wn0:.3f}->{wn1:.3f} (teacher-driven critic + NEAR place + DA reward)", flush=True)
