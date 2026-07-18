"""Gap #5 — dendritic-plateau BISTABILITY latch-and-hold probe (single cell, numpy, uses the REAL kernel).

Drives the coincidence plateau with a cue volley for `cue_steps`, then REMOVES the input and runs `hold_steps`; reads
the apical voltage. A transient dAP (self_regen=0) decays back to rest; a bistable dendrite (self_regen>0 + KIR down-
state stabilizer) HOLDS the plateau. Also: no-cue -> silent. This validates the sim/ kernel change end-to-end at the
single-cell mechanism level (before wiring into the CA3 network). GO signature: HELD-high with regen+KIR, DECAYS
without, SILENT with no cue.
"""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import numpy as np
from sim.kernels import fused_coincidence_plateau  # numpy backend -> plain callable


def sim_cell(self_regen, kir_g, cue_c=8.0, cue_steps=40, hold_steps=250, k_thresh=6.0,
             plateau_strength=4.0, gain=2.0, mg=1.0, tau=15.0, R=0.15, Er=-65.0, dt=1.0,
             v_hold=-35.0, v_hold_k=0.2, kir_ek=-90.0, kir_vhalf=-50.0, kir_k=8.0, n_sub=8,
             cue_depol=40.0):
    """cue_depol = a direct depolarizing (AMPA-like) current during the cue only, kicking v into the plateau-ignition
    zone (biology: AMPA triggers, NMDA sustains). The HOLD test is whether the plateau stays up AFTER both the
    coincidence drive AND this kick are removed."""
    """n_sub sub-steps + a physiological v clamp [-90, 5] mV keep explicit Euler stable across the stiff plateau
    (the plateau self-limits at E_e=0; the clamp only bounds numerical overshoot, it does not create the up state)."""
    decay = float(np.exp(-dt / 80.0)); decay_rise = float(np.exp(-dt / 2.0))
    g = np.array([0.0]); g_rise = np.array([0.0]); v = np.array([Er])
    sdt = dt / n_sub
    trace = []
    for t in range(cue_steps + hold_steps):
        c = np.array([cue_c if t < cue_steps else 0.0])
        g, g_rise, I = fused_coincidence_plateau(g, g_rise, decay, decay_rise, v, 0.0, mg,
                                                 c, k_thresh, gain, plateau_strength,
                                                 self_regen, v_hold, v_hold_k)
        depol = cue_depol if t < cue_steps else 0.0  # AMPA-like kick during the cue only
        for _ in range(n_sub):                       # sub-step the membrane ODE for stability (I held over the step)
            dv = -(v - Er) + R * I + depol            # apical leak + plateau current + cue kick (isolated cell)
            if kir_g != 0.0:
                gkir = kir_g / (1.0 + np.exp((v - kir_vhalf) / kir_k))
                dv = dv + gkir * (kir_ek - v)
            v = np.clip(v + (sdt / tau) * dv, -90.0, 5.0)
        trace.append(float(v[0]))
    return np.array(trace), cue_steps


def summary(self_regen, kir_g, cue_c=20.0):
    tr, cs = sim_cell(self_regen, kir_g, cue_c=cue_c)
    v_cue = float(np.mean(tr[cs - 5:cs]))            # apical V at end of cue
    v_hold = float(np.mean(tr[-30:]))                # apical V long after cue removed
    return v_cue, v_hold


def summary2(self_regen, kir_g, cue_c=8.0, cue_depol=40.0):
    tr, cs = sim_cell(self_regen, kir_g, cue_c=cue_c, cue_depol=cue_depol)
    return float(np.mean(tr[cs - 5:cs])), float(np.mean(tr[-30:]))


if __name__ == "__main__":
    Er = -65.0
    print("single-cell latch-and-hold (apical V, mV): v_cue=end of volley, v_hold=250 steps after cue removed")
    print(f"  rest = {Er}  | GO = correct cue HELD (v_hold high), transient/no-cue NOT held")
    print("[A] self_regen sweep at kir_g=2 (find the HOLD threshold):")
    for sr in (0.0, 0.3, 0.8, 1.5, 3.0):
        vc, vh = summary2(sr, 2.0)
        held = "HELD" if vh > Er + 15 else ("silent/decayed" if vh < Er + 5 else "partial")
        print(f"  self_regen={sr:.1f} kir_g=2.0: v_cue={vc:6.1f} v_hold={vh:6.1f} -> {held}")
    print("[B] the bistability triad at a HOLD-regime self_regen (regen+KIR):")
    SR = 1.5
    vc, vh = summary2(SR, 2.0);                    print(f"  correct cue (regen+KIR):  v_cue={vc:6.1f} v_hold={vh:6.1f} -> {'HELD' if vh>Er+15 else 'no'}")
    vc, vh = summary2(0.0, 2.0);                   print(f"  transient (no regen):     v_cue={vc:6.1f} v_hold={vh:6.1f} -> {'decays' if vh<Er+5 else 'HELD(bad)'}")
    vc, vh = summary2(SR, 2.0, cue_c=0.0, cue_depol=0.0); print(f"  no cue (regen+KIR):       v_cue={vc:6.1f} v_hold={vh:6.1f} -> {'SILENT' if vh<Er+5 else 'self-ignited(bad)'}")
