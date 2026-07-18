"""CI guard — gap #5 (2026-07-18): the dendritic-plateau kernel change gives INTRINSIC BISTABILITY (latch-and-hold),
not just a transient dAP. A v-gated self-regenerating SUSTAIN term (`coincidence_plateau_self_regen`) + a KIR down-state
stabilizer make a single cell LATCH into a plateau on a coincident cue and HOLD it after the cue is removed, stay SILENT
with no cue, and DECAY without the sustain term (the sustain is load-bearing). Also pins byte-identity: with the new
knobs at default (self_regen=0), the kernel output is unchanged. CPU/numpy, self-contained (real kernel).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.kernels import fused_coincidence_plateau

ER = -65.0


def _run(self_regen, kir_g, cue_c, cue_depol, cue_steps=40, hold_steps=250, k_thresh=6.0,
         plateau_strength=4.0, gain=2.0, mg=1.0, tau=15.0, R=0.15, dt=1.0,
         v_hold=-35.0, v_hold_k=0.2, kir_ek=-90.0, kir_vhalf=-50.0, kir_k=8.0, n_sub=8):
    decay = float(np.exp(-dt / 80.0)); decay_rise = float(np.exp(-dt / 2.0))
    g = np.array([0.0]); g_rise = np.array([0.0]); v = np.array([ER]); sdt = dt / n_sub
    tr = []
    for t in range(cue_steps + hold_steps):
        c = np.array([cue_c if t < cue_steps else 0.0])
        g, g_rise, I = fused_coincidence_plateau(g, g_rise, decay, decay_rise, v, 0.0, mg,
                                                 c, k_thresh, gain, plateau_strength, self_regen, v_hold, v_hold_k)
        depol = cue_depol if t < cue_steps else 0.0
        for _ in range(n_sub):
            dv = -(v - ER) + R * I + depol
            if kir_g != 0.0:
                dv = dv + (kir_g / (1.0 + np.exp((v - kir_vhalf) / kir_k))) * (kir_ek - v)
            v = np.clip(v + (sdt / tau) * dv, -90.0, 5.0)
        tr.append(float(v[0]))
    return np.array(tr)


def _v_hold(tr):
    return float(np.mean(tr[-30:]))


def test_bistability_latch_and_hold_triad():
    SR, KIR = 1.5, 2.0
    held = _v_hold(_run(SR, KIR, cue_c=8.0, cue_depol=40.0))          # correct cue + regen+KIR
    transient = _v_hold(_run(0.0, KIR, cue_c=8.0, cue_depol=40.0))    # same cue, NO self-regen
    nocue = _v_hold(_run(SR, KIR, cue_c=0.0, cue_depol=0.0))          # no cue
    assert held > ER + 15, f"correct cue must HOLD the plateau, got v_hold={held}"
    assert transient < ER + 5, f"without self_regen the plateau must DECAY, got v_hold={transient}"
    assert nocue < ER + 5, f"no cue must stay SILENT (no self-ignition), got v_hold={nocue}"
    assert held - transient > 30, "the self_regen sustain term must be load-bearing for the hold"


def test_hold_threshold_is_bistable():
    # below a self_regen threshold the plateau decays; above it, it holds -> a genuine bifurcation (not graded)
    below = _v_hold(_run(0.3, 2.0, cue_c=8.0, cue_depol=40.0))
    above = _v_hold(_run(1.5, 2.0, cue_c=8.0, cue_depol=40.0))
    assert below < ER + 5 and above > ER + 15


def test_self_regen_default_off_byte_identical():
    # the new knobs at default (self_regen=0) reproduce the transient plateau exactly (byte-identity of the default path)
    decay = float(np.exp(-1.0 / 80.0)); dr = float(np.exp(-1.0 / 2.0))
    g0, gr0, v = np.array([1.0]), np.array([0.2]), np.array([-40.0])
    c = np.array([10.0])
    a = fused_coincidence_plateau(g0.copy(), gr0.copy(), decay, dr, v, 0.0, 1.0, c, 6.0, 2.0, 80.0)  # default sustain args
    b = fused_coincidence_plateau(g0.copy(), gr0.copy(), decay, dr, v, 0.0, 1.0, c, 6.0, 2.0, 80.0, 0.0, -35.0, 0.2)
    for x, y in zip(a, b):
        assert np.allclose(np.asarray(x), np.asarray(y)), "self_regen=0 must be byte-identical to the no-arg default"
