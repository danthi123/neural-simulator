"""N9: the SNc gap is 0/0 (critic over-inhibits SNc in both states). Diagnose:
sweep the lead, critic->SNc weight, and reward gain to see if a NEAR(predicted,
V-high)<FAR(unpredicted,V-low) SNc-burst gap exists for the trained two-region critic.

This is the FUNCTIONAL N9 deliverable (the value SUBTRACTION at the SNc), distinct from
the gate-2 absolute near>>far critic-RATE grade (which fails structurally)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np
from sim.backend import get_backend
xp,bk=get_backend(); print("backend:",bk,flush=True)

import research.runners.n9_convergent_upstate_derisk as D

SEED=42
NEAR=(8.0,24.0); FAR=(24.0,8.0)

def train_and_probe(a1_w, strio_to_snc_w, snc_reward_gain, lead_ms, n_train=40, seed=SEED):
    """Build + train the two-region critic, then measure SNc burst NEAR vs FAR at the given lead."""
    bridge, cfg = D._build(seed, a1_weight=a1_w, strio_to_snc_weight=strio_to_snc_w,
                           snc_da_sensitivity=8.0, reward_learning_rate=0.12, gabab=True,
                           gabab_propagation_strength=0.02)
    regions=("vs_place_drive","vs_place_context","striosome_value","snc",
             "sensor_place_readout","cortex_N","cortex_E","cortex_S","cortex_W")
    idx_map={n:xp.asarray(D._idx(bridge,n)) for n in regions}
    n_vs=len(D._host(idx_map["vs_place_drive"]))
    prefs=D._grid_prefs(n_vs,32)
    near=D._place_code(NEAR,prefs,800.0,4.0); far=D._place_code(FAR,prefs,800.0,4.0)
    D._calibrate_da(bridge,cfg,idx_map,180.0,xp)
    # train at NEAR
    for t in range(n_train):
        D._drive(bridge,idx_map,{"vs_place_drive":None,"vs_place_context":None},{"snc":180.0},40,xp)
        if getattr(bridge,"cp_eligibility_trace",None) is not None:
            bridge.cp_eligibility_trace[:]=0.0
        D._drive(bridge,idx_map,{"vs_place_drive":near,"vs_place_context":near},
                 {"snc":180.0+snc_reward_gain},40,xp)
    lead=int(round(lead_ms))
    def test(a1v,a2v,snc_pa):
        D._drive(bridge,idx_map,{"vs_place_drive":None,"vs_place_context":None},{"snc":180.0},60,xp,freeze_lr=0.0,cfg=cfg)
        if lead>0:
            D._drive(bridge,idx_map,{"vs_place_drive":a1v,"vs_place_context":a2v},{"snc":180.0},lead,xp,freeze_lr=0.0,cfg=cfg)
        snc_r,_,_=D._drive(bridge,idx_map,{"vs_place_drive":a1v,"vs_place_context":a2v},{"snc":snc_pa},40,xp,freeze_lr=0.0,cfg=cfg)
        return snc_r
    pred=test(near,near,180.0+snc_reward_gain)   # NEAR predicted (V high -> GABA_B cancels -> small)
    unpred=test(far,far,180.0+snc_reward_gain)   # FAR unpredicted (V low -> big)
    base=test(None,None,180.0)
    del bridge
    return pred,unpred,base

if __name__=="__main__":
    print("\n=== SNc gap 3-SEED robustness at the one promising config (a1=24, snc_w=4, lead=150) ===",flush=True)
    print("  PASS = unpred(FAR) > 1.3x pred(NEAR) AND unpred >= 10Hz, on >=3 seeds",flush=True)
    n_gap=0
    for seed in [42,43,44]:
        pred,unpred,base=train_and_probe(24.0, 4.0, 300.0, 150.0, seed=seed)
        gap=unpred/max(pred,1e-3)
        ok=(unpred>1.3*max(pred,1e-3) and unpred>=10.0)
        n_gap += int(ok)
        print(f"  seed={seed}: pred(NEAR)={pred:6.2f}Hz unpred(FAR)={unpred:6.2f}Hz "
              f"base={base:6.2f}Hz gap={gap:5.2f} {'<= GAP' if ok else ''}",flush=True)
    print(f"\n  SNc value-subtraction gap (best config): {n_gap}/3 seeds",flush=True)
