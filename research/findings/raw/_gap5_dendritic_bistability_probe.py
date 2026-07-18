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
             v_hold=-35.0, v_hold_k=0.2, kir_ek=-90.0, kir_vhalf=-50.0, kir_k=8.0, n_sub=8):
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
        for _ in range(n_sub):                       # sub-step the membrane ODE for stability (I held over the step)
            dv = -(v - Er) + R * I                    # apical leak + plateau current (NO soma coupling: isolated cell)
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


if __name__ == "__main__":
    Er = -65.0
    print("single-cell latch-and-hold (apical V, mV): v_cue=end of volley, v_hold=250 steps after removal")
    print(f"  rest = {Er}")
    for sr, kg, label in [(0.0, 0.0, "transient (current default)"),
                          (0.02, 0.0, "regen, no KIR"),
                          (0.02, 3.0, "regen + KIR (bistable target)"),
                          (0.05, 3.0, "stronger regen + KIR"),
                          (0.02, 5.0, "regen + stronger KIR")]:
        vc, vh = summary(sr, kg)
        held = "HELD" if vh > Er + 15 else ("silent/decayed" if vh < Er + 5 else "partial")
        print(f"  self_regen={sr:.2f} kir_g={kg:.1f} [{label:30s}]: v_cue={vc:6.1f} v_hold={vh:6.1f} -> {held}")
    print("no-cue control (cue_c=0, should stay silent):")
    for sr, kg in [(0.02, 3.0), (0.05, 3.0)]:
        vc, vh = summary(sr, kg, cue_c=0.0)
        print(f"  self_regen={sr:.2f} kir_g={kg:.1f}: v_cue={vc:6.1f} v_hold={vh:6.1f} -> {'SILENT' if vh < Er+5 else 'SELF-IGNITED (bad)'}")
