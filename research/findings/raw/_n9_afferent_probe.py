"""N9: why does vs_place_drive only fire ~9 Hz at 800 pA? Probe the afferent's
firing-rate vs drive curve, and the per-cell drive distribution of the place code."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk, flush=True)
from research.findings.raw._n9_upstate_calib import build, _idx, _grid_prefs, place_code, _host

SEED=42; GRID=32; SIGMA=4.0
NEAR=(26.571,26.571)

bridge, cfg = build(SEED, a1_weight=6.0)
drive_idx = _idx(bridge, "vs_place_drive")
prefs = _grid_prefs(len(drive_idx), GRID)

# Per-cell drive distribution at NEAR for a few DRIVE_MAX values
for dmax in [800.0, 1500.0, 3000.0]:
    vec = place_code(NEAR, prefs, dmax, SIGMA)
    print(f"DRIVE_MAX={dmax}: cells>1pA={int((vec>1).sum())}, "
          f"cells>200pA={int((vec>200).sum())}, cells>400pA={int((vec>400).sum())}, max={vec.max():.0f}",
          flush=True)

# Afferent firing-rate vs uniform drive (a clean f-I curve for the RS pyramidal afferent)
print("\n=== RS pyramidal afferent f-I (uniform drive into all vs_place_drive cells) ===", flush=True)
drive_cp = xp.asarray(drive_idx); n=len(drive_idx)
for pa in [200.0, 400.0, 600.0, 800.0, 1200.0, 2000.0]:
    spk=0; m=0
    for t in range(120):
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[drive_cp] = xp.float32(pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step*bridge.core_config.dt_ms
        if t>=40:
            spk += int(bridge.cp_firing_states[drive_cp].sum()); m+=1
    print(f"  uniform {pa:6.0f} pA -> afferent rate {spk/max(n,1)/((120-40)*1e-3):7.1f} Hz", flush=True)
