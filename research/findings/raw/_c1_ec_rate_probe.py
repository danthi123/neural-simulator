"""C1 sub-diagnostic: how hard do the landmark sensors and EC actually fire, and what does it take to
drive DG sparsely-but-reliably (the rate bottleneck). Isolate landmark_sensors -> ec only."""
from __future__ import annotations
import os, sys, numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_d))))
from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, default_locations, default_landmarks)
from research.findings.raw._c1_trisyn_conduction_diag import build_trisyn_from_landmarks, _step, measure_hops
from sim.backend import get_backend

xp, backend = get_backend(); assert backend == "cupy"
grid = 32
locs = default_locations(grid); lms = default_landmarks(grid); dist_max = grid * 1.42
n_bearing, n_dist = 12, 8
n_sensors = len(lms) * (n_bearing + n_dist)

# Report the raw sensor drive distribution at one location
act = landmark_sensor_act(*locs["near"], lms, n_bearing, n_dist, 900.0, 0.03, 4.0, dist_max, 4.0)
print(f"sensor drive @near: n={len(act)} min={act.min():.0f} max={act.max():.0f} mean={act.mean():.0f} "
      f"median={np.median(act):.0f} n>300pA={int((act>300).sum())}")

for lm_w, lm_d, ec_dg_w, pvb in [(24, 0.30, 20, 0.0), (40, 0.5, 20, 0.0), (60, 0.6, 30, 0.0),
                                  (60, 0.6, 30, 4.0), (60, 0.6, 40, 4.0)]:
    bridge, cfg = build_trisyn_from_landmarks(
        42, n_sensors=n_sensors, n_ec=200, n_dg=800, n_dg_pv_basket=240, n_ca3=400, n_ca1=200,
        lm_to_ec_weight=lm_w, lm_to_ec_density=lm_d, ec_to_dg_weight=ec_dg_w, ec_to_dg_density=0.40,
        ec_to_pvb_weight=5.0, pvb_to_dg_weight=pvb, dg_to_ca3_weight=40, dg_to_ca3_density=0.10,
        ca3_rec_weight=1.5, ca3_rec_density=0.30, ca3_to_ca1_weight=4.0, ca3_to_ca1_density=0.30,
        ec_to_ca1_weight=3.0, ec_to_ca1_density=0.30, enable_nmda=True)
    rm = bridge.region_manager
    sidx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    rmap = {n: np.asarray(rm.indices(n), dtype=np.int64) for n in ("landmark_sensors", "ec", "dg")}
    hc = {n: measure_hops(bridge, xp, sidx, rmap, landmark_sensor_act(*locs[n], lms, n_bearing, n_dist,
              900.0, 0.03, 4.0, dist_max, 4.0), 100) for n in ("near", "far_a")}
    def line(name):
        return " ".join(f"{r}:af{float(np.mean(hc[name][r]>0)):.2f}/spk{float(np.sum(hc[name][r]))/100:.1f}"
                        for r in ("landmark_sensors", "ec", "dg"))
    print(f"lm_w={lm_w} lm_d={lm_d} ec_dg_w={ec_dg_w} pvb={pvb}: near[{line('near')}]  far_a[{line('far_a')}]")
