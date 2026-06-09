"""Why is the NEAR place ensemble so weak (0.1 spk/step)? Map place-pool firing per location, and
find NEAR/FAR locations where the place ensemble fires ROBUSTLY (enough spikes to drive the critic +
enough presynaptic pairing for LTP). Also test: does a stronger lm->place weight / more place cells
firing help? The place ensemble must be a real up-state driver, not 6 cells flickering."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, bk = get_backend(); print("backend:", bk, flush=True)

from research.runners.n9_place_graded_critic_stage2_derisk import (
    _build, _idx, _host, _step, _tick, _place_ensemble, landmark_sensor_act, default_landmarks)

SEED = 42; GRID = 32
landmarks = default_landmarks(GRID)
n_bearing, n_dist, bexp, dist_sigma = 12, 8, 4.0, 4.0
max_int, falloff = 450.0, 0.03; dist_max = GRID * 1.42
n_sensors = len(landmarks) * (n_bearing + n_dist)

# A denser grid of candidate locations to find robust-ensemble pairs.
g = GRID - 1
CANDS = {
    "c_25_25": (g*0.25, g*0.25), "c_25_50": (g*0.25, g*0.50), "c_25_75": (g*0.25, g*0.75),
    "c_50_25": (g*0.50, g*0.25), "c_50_50": (g*0.50, g*0.50), "c_50_75": (g*0.50, g*0.75),
    "c_75_25": (g*0.75, g*0.25), "c_75_50": (g*0.75, g*0.50), "c_75_75": (g*0.75, g*0.75),
    "c_40_60": (g*0.40, g*0.60), "c_60_40": (g*0.60, g*0.40),
}


def build_selforg(lm_w, n_place, passes=10, steps=120, locs=None):
    bridge, cfg = _build(SEED, n_sensors=n_sensors, n_place=n_place, n_strio=80, n_snc=30, grid_size=GRID,
                         lm_to_place_weight=lm_w, lm_to_place_density=0.5, lm_to_place_jitter=0.6,
                         place_to_value_weight=0.5, place_to_value_density=0.6,
                         place_to_value_jitter=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
                         reward_learning_rate=0.12, gabab=True, gabab_tau_decay=150.0,
                         gabab_propagation_strength=0.02, include_actor=False)
    si = xp.asarray(_idx(bridge, "landmark_sensors")); pi = xp.asarray(_idx(bridge, "place"))
    loc_sensor = {n: landmark_sensor_act(*locs[n], landmarks, n_bearing, n_dist, max_int,
                                         falloff, dist_sigma, dist_max, bexp) for n in locs}
    bridge.set_plasticity_gate("landmark_to_place", 1.0); bridge.set_plasticity_gate("value_input", 0.0)
    rng = np.random.default_rng(SEED)
    for _p in range(passes):
        order = list(locs); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0; _step(bridge, 20)
            bridge.cp_external_input_current[si] = xp.asarray(loc_sensor[name], dtype=xp.float32)
            _step(bridge, steps)
    bridge.set_plasticity_gate("landmark_to_place", 0.0); bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg, si, pi, loc_sensor


def place_fire_rate(bridge, si, pi, sensor_act, n_steps=80, warmup=20):
    """Total place spikes/step (the presynaptic drive strength to the critic)."""
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[si] = xp.asarray(sensor_act, dtype=xp.float32)
    spk = 0; m = 0
    for t in range(n_steps):
        _tick(bridge)
        if t >= warmup: spk += int(bridge.cp_firing_states[pi].sum()); m += 1
    bridge.cp_external_input_current[:] = 0.0
    return spk / max(m, 1)  # total place spikes per step


print("\n=== place-pool firing per candidate location (lm_w=28, n_place=400) ===", flush=True)
bridge, cfg, si, pi, loc_sensor = build_selforg(28.0, 400, locs=CANDS)
rows = []
for n in CANDS:
    e = _place_ensemble(bridge, xp, si, pi, loc_sensor[n])
    active = int((np.asarray(e) > 0).sum())
    fr = place_fire_rate(bridge, si, pi, loc_sensor[n])
    rows.append((n, active, fr))
    print(f"  {n:9s} active={active:3d}/400  place_spk/step={fr:6.2f}", flush=True)
del bridge

# strongest-firing candidate = NEAR; pick 3 well-separated weaker ones as FAR
rows.sort(key=lambda r: -r[2])
print(f"\n  strongest-firing: {[r[0] for r in rows[:4]]}", flush=True)

print("\n=== effect of lm->place weight on place firing (use c_50_50 center as probe) ===", flush=True)
for lm_w in [28.0, 40.0, 60.0, 90.0]:
    bridge, cfg, si, pi, loc_sensor = build_selforg(lm_w, 400, locs=CANDS)
    frs = {n: place_fire_rate(bridge, si, pi, loc_sensor[n]) for n in ["c_25_75","c_50_50","c_75_25","c_25_25"]}
    print(f"  lm_w={lm_w:5.1f}: " + "  ".join(f"{n}={v:5.1f}" for n,v in frs.items()), flush=True)
    del bridge
