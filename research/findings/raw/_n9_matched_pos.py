"""Find NEAR/FAR positions with MATCHED n_active (so the A1 density-blob baseline
is equal and gate-2 cleanly tests A2's learned selectivity). Also re-check the A1
weight needed for the critic to fire >=5 Hz at those positions at INIT."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp,_bk=get_backend(); print("backend:",_bk,flush=True)
from research.findings.raw._n9_upstate_calib import build, _idx, _grid_prefs, place_code, critic_rate

SEED=42; GRID=32; SIGMA=4.0; DMAX=800.0
# candidate matched-radius pairs (same |pos - center|, center=(15.5,15.5))
CANDS = {
  "midedge_NS": ((15.5, 24.0),(15.5, 7.0)),
  "diag_off":   ((8.0, 24.0),(24.0, 8.0)),
  "moderate":   ((10.0, 22.0),(22.0, 10.0)),
  "near_center":((12.0, 19.0),(19.0, 12.0)),
}
def nactive(vec): return int((vec>1).sum())

bridge,cfg=build(SEED, a1_weight=24.0)
drive_idx=_idx(bridge,"vs_place_drive"); ctx_idx=_idx(bridge,"vs_place_context"); crit_idx=_idx(bridge,"striosome_value")
prefs=_grid_prefs(len(drive_idx),GRID)
print("=== n_active match (drive>1pA) per candidate ===",flush=True)
for name,(p1,p2) in CANDS.items():
    v1=place_code(p1,prefs,DMAX,SIGMA); v2=place_code(p2,prefs,DMAX,SIGMA)
    print(f"  {name:12s}: {p1} n={nactive(v1):3d}  vs  {p2} n={nactive(v2):3d}",flush=True)
del bridge

# For the best-matched candidate, sweep A1 weight for the critic init-fire rate at BOTH positions.
best="diag_off"
p1,p2=CANDS[best]
print(f"\n=== critic init-fire at A1 weights for '{best}' {p1} vs {p2} (A2 at init 0.2) ===",flush=True)
for w in [20.0,24.0,28.0,32.0]:
    bridge,cfg=build(SEED,a1_weight=w)
    drive_idx=_idx(bridge,"vs_place_drive"); ctx_idx=_idx(bridge,"vs_place_context"); crit_idx=_idx(bridge,"striosome_value")
    prefs=_grid_prefs(len(drive_idx),GRID)
    v1=place_code(p1,prefs,DMAX,SIGMA); v2=place_code(p2,prefs,DMAX,SIGMA)
    r1=critic_rate(bridge,[drive_idx,ctx_idx],[v1,v1],crit_idx,n_steps=100,warmup=30)
    r2=critic_rate(bridge,[drive_idx,ctx_idx],[v2,v2],crit_idx,n_steps=100,warmup=30)
    print(f"  A1_w={w:5.1f}: critic@{p1}={r1:6.2f}Hz  critic@{p2}={r2:6.2f}Hz  (baseline blob, pre-training)",flush=True)
    del bridge
