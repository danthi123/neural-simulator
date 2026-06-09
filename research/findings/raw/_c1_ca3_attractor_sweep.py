"""C1 decisive sweep: can the CA3 recurrent autoassociator be HIGH-RATE *and* DISTINCT-per-location?

This is THE C1 thesis test. Builds the landmark->trisynaptic loop once per (rec_w, inh_w) combo, self-orgs
(opens ec_to_dg/dg_to_ca3/ca3_swr_burst/ca3_to_ca1), then measures CA3+CA1 per location:
  - CA3 spk/step (RATE: needs ~>=8 to drive an MSN later)
  - CA3 active-fraction (sparsity)
  - CA3 diff-location cosine (DISTINCT: needs < 0.3)
Looks for ANY (rec_w, inh_w) where CA3 is BOTH high-rate AND distinct.

Fixed front-end at the conducting operating point found by the conduction diag:
  lm_w=60 lm_d=0.10 int=900 ec_dg_w=30 pvb=2 mossy_w=40 mossy_d=0.10, n_ca3_inh=120.
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_d))))
from research.runners.placecode_selforg_stage1_derisk import (
    landmark_sensor_act, default_locations, default_landmarks, cosine_counts)
from research.findings.raw._c1_trisyn_conduction_diag import (
    build_trisyn_from_landmarks, _step, measure_hops)
from sim.backend import get_backend

xp, backend = get_backend(); assert backend == "cupy"
grid = 32
locs = default_locations(grid); lms = default_landmarks(grid); dist_max = grid * 1.42
nb, nd = 12, 8
n_sensors = len(lms) * (nb + nd)
loc_names = list(locs.keys())
loc_act = {n: landmark_sensor_act(*locs[n], lms, nb, nd, 900.0, 0.03, 4.0, dist_max, 4.0) for n in loc_names}
pairs = list(itertools.combinations(loc_names, 2))

REC_WS = [float(x) for x in (os.environ.get("REC_WS", "3,5,6,7,8").split(","))]
INH_WS = [float(x) for x in (os.environ.get("INH_WS", "10,16,24").split(","))]
SELFORG = int(os.environ.get("SELFORG", "16"))
REC_D = float(os.environ.get("REC_D", "0.30"))

print(f"CA3 attractor sweep: rec_w={REC_WS} inh_w={INH_WS} selforg={SELFORG} rec_d={REC_D}")
print(f"{'rec_w':>6}{'inh_w':>6} | {'ca3_spk(mean)':>14}{'ca3_af':>8}{'ca3_diffcos':>12}{'ca1_spk':>9}"
      f"{'ca1_diffcos':>12}  verdict")
for rec_w, inh_w in itertools.product(REC_WS, INH_WS):
    t0 = time.time()
    bridge, cfg = build_trisyn_from_landmarks(
        42, n_sensors=n_sensors, n_ec=200, n_dg=800, n_dg_pv_basket=240, n_ca3=400, n_ca1=200,
        lm_to_ec_weight=60, lm_to_ec_density=0.10, ec_to_dg_weight=30, ec_to_dg_density=0.40,
        ec_to_pvb_weight=5.0, pvb_to_dg_weight=2.0, dg_to_ca3_weight=40, dg_to_ca3_density=0.10,
        ca3_rec_weight=rec_w, ca3_rec_density=REC_D, ca3_to_ca1_weight=6.0, ca3_to_ca1_density=0.30,
        ec_to_ca1_weight=3.0, ec_to_ca1_density=0.30,
        n_ca3_inh=120, ca3_to_inh_weight=8.0, ca3_to_inh_density=0.30,
        inh_to_ca3_weight=inh_w, inh_to_ca3_density=0.60, enable_nmda=True)
    rm = bridge.region_manager
    sidx = np.asarray(rm.indices("landmark_sensors"), dtype=np.int64)
    rmap = {n: np.asarray(rm.indices(n), dtype=np.int64) for n in ("ca3", "ca1")}
    # self-org
    for g in ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst", "ec_to_ca1"):
        try: bridge.set_plasticity_gate(g, 1.0)
        except Exception: pass
    rng = np.random.default_rng(42)
    for _p in range(SELFORG):
        order = list(loc_names); rng.shuffle(order)
        for name in order:
            bridge.cp_external_input_current[:] = 0.0
            _step(bridge, 20)
            bridge.cp_external_input_current[xp.asarray(sidx, dtype=xp.int64)] = \
                xp.asarray(loc_act[name], dtype=xp.float32)
            _step(bridge, 120)
    for g in ("landmark_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst", "ec_to_ca1"):
        try: bridge.set_plasticity_gate(g, 0.0)
        except Exception: pass
    bridge.cp_external_input_current[:] = 0.0
    hc = {n: measure_hops(bridge, xp, sidx, rmap, loc_act[n], 100) for n in loc_names}
    ca3_spk = np.mean([np.sum(hc[n]["ca3"]) / 100 for n in loc_names])
    ca3_af = np.mean([np.mean(hc[n]["ca3"] > 0) for n in loc_names])
    ca3_dc = np.mean([cosine_counts(hc[a]["ca3"], hc[b]["ca3"]) for a, b in pairs])
    ca1_spk = np.mean([np.sum(hc[n]["ca1"]) / 100 for n in loc_names])
    ca1_dc = np.mean([cosine_counts(hc[a]["ca1"], hc[b]["ca1"]) for a, b in pairs])
    verdict = "HIGH+DISTINCT!" if (ca3_spk >= 8 and ca3_dc < 0.3) else \
              ("global-attractor" if ca3_spk >= 8 else ("distinct-but-weak" if ca3_dc < 0.3 else "fail"))
    print(f"{rec_w:>6.1f}{inh_w:>6.1f} | {ca3_spk:>14.1f}{ca3_af:>8.2f}{ca3_dc:>12.3f}{ca1_spk:>9.2f}"
          f"{ca1_dc:>12.3f}  {verdict}  ({time.time()-t0:.0f}s)")
