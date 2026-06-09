"""N9 up-state CALIBRATION round 2 (CuPy): sweep A1 weight 14-32, and try a
wider-sigma place code (more convergent firing cells) to reach the up-state at a
LOWER per-synapse weight. Gate: critic NEAR 5-20 Hz, FAR ~0 Hz."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk, flush=True)
from research.findings.raw._n9_upstate_calib import build, _idx, _grid_prefs, place_code, critic_rate

SEED=42; GRID=32
NEAR=(26.571,26.571); FAR=(4.429,4.429)

def run(a1_w, sigma, dmax, n_vs=200, label=""):
    bridge, cfg = build(SEED, a1_weight=a1_w, n_vs=n_vs)
    drive_idx=_idx(bridge,"vs_place_drive"); ctx_idx=_idx(bridge,"vs_place_context"); crit_idx=_idx(bridge,"striosome_value")
    prefs=_grid_prefs(len(drive_idx),GRID)
    near=place_code(NEAR,prefs,dmax,sigma); far=place_code(FAR,prefs,dmax,sigma)
    na=int((near>1).sum())
    r_near=critic_rate(bridge,[drive_idx,ctx_idx],[near,near],crit_idx)
    r_far =critic_rate(bridge,[drive_idx,ctx_idx],[far,far],crit_idx)
    ok = (5.0 <= r_near <= 25.0) and (r_far < 1.0)
    print(f"  [{label}] A1_w={a1_w:5.1f} sigma={sigma} dmax={dmax:6.0f} n_active_near={na:3d}: "
          f"critic NEAR={r_near:6.2f}Hz FAR={r_far:6.2f}Hz  {'<= GOAL BAND' if ok else ''}", flush=True)
    del bridge
    return r_near, r_far

if __name__=="__main__":
    print("\n=== sweep A1 weight (sigma=4.0, dmax=800) ===", flush=True)
    for w in [14.0,16.0,18.0,20.0,24.0,28.0,32.0]:
        run(w, 4.0, 800.0, label="s4d800")
    print("\n=== wider sigma 6.0 (more convergent cells), lower weights ===", flush=True)
    for w in [10.0,14.0,18.0,22.0]:
        run(w, 6.0, 800.0, label="s6d800")
    print("\n=== higher dmax 1500 (more cells fire hard), sigma 4.0 ===", flush=True)
    for w in [10.0,14.0,18.0]:
        run(w, 4.0, 1500.0, label="s4d1500")
