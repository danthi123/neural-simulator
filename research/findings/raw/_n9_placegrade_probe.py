"""N9: is the A1 up-state PLACE-GRADED? Measure critic rate as the place bump
sweeps across the grid. If A1 (dense non-plastic) fires the critic at EVERY
location's bump (~flat), it is a blob, not a value-of-location (design Option D)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk, flush=True)
from research.findings.raw._n9_upstate_calib import build, _idx, _grid_prefs, place_code, critic_rate

SEED=42; GRID=32; SIGMA=4.0; DMAX=800.0

# Pick an A1 weight where the critic fires (w=28 from calib2: NEAR=10, FAR=29).
for w in [24.0, 28.0]:
    bridge, cfg = build(SEED, a1_weight=w)
    drive_idx=_idx(bridge,"vs_place_drive"); ctx_idx=_idx(bridge,"vs_place_context"); crit_idx=_idx(bridge,"striosome_value")
    prefs=_grid_prefs(len(drive_idx),GRID)
    print(f"\n=== A1 w={w}: critic rate as bump sweeps the diagonal (0,0)->(31,31) ===", flush=True)
    for frac in [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]:
        pos=(frac*31.0, frac*31.0)
        vec=place_code(pos,prefs,DMAX,SIGMA)
        r=critic_rate(bridge,[drive_idx,ctx_idx],[vec,vec],crit_idx, n_steps=100, warmup=30)
        print(f"  pos=({pos[0]:5.1f},{pos[1]:5.1f}) n_active={int((vec>1).sum()):3d}: critic={r:6.2f}Hz", flush=True)
    del bridge
