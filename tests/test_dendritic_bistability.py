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


# --- gap #4 (2026-07-18): the BDSP top-down apical integration made BISTABLE (bridge.py ~7258). These pin the MATH of
# that edit (self-contained numpy reproduction of the exact _dvb formula), the same way the kernel tests above pin the
# plateau kernel. The REAL on-bridge "measured burst rate B rises" behavior is the runner probe's job, not CI.

def _bdsp_apical_run(bistable, self_regen, kir_g, drive_val, drive_steps=40, hold_steps=250,
                     tau=15.0, dt=1.0, v_hold=-35.0, v_hold_k=0.2, e_e=0.0,
                     kir_ek=-90.0, kir_vhalf=-50.0, kir_k=8.0):
    """Reproduce the bridge.py:7258 BDSP-apical integration EXACTLY, for one cell, drive ON then removed.
    Sustain = self_regen*sigmoid(v_hold_k*(v-v_hold))*(E_e - v) (self-limiting), matching fused_coincidence_plateau."""
    v = np.array([ER])
    tr = []
    for t in range(drive_steps + hold_steps):
        drive = drive_val if t < drive_steps else 0.0
        dvb = -(v - ER) + drive                                         # the prior plain leaky expression
        if bistable:
            if self_regen != 0.0:
                dvb = dvb + (self_regen / (1.0 + np.exp(-v_hold_k * (v - v_hold)))) * (e_e - v)
            if kir_g != 0.0:
                dvb = dvb + (kir_g / (1.0 + np.exp((v - kir_vhalf) / kir_k))) * (kir_ek - v)
        v = v + (dt / tau) * dvb
        tr.append(float(v[0]))
    return np.array(tr)


def test_bdsp_apical_bistable_off_is_plain_leaky():
    # bdsp_apical_bistable=False => _dvb is exactly -(v-Er)+drive => identical to the pre-edit single line, drive on or off
    off = _bdsp_apical_run(bistable=False, self_regen=2.0, kir_g=1.0, drive_val=40.0)   # bistable knobs set but flag OFF
    plain = _bdsp_apical_run(bistable=False, self_regen=0.0, kir_g=0.0, drive_val=40.0)
    assert np.allclose(off, plain), "flag OFF must be byte-identical to the plain leaky integration regardless of knobs"
    # and the plain path is a pure transient: with the drive removed it relaxes back toward E_rest
    assert plain[-1] < ER + 3.0, f"plain BDSP apical must DECAY after the drive is removed, got {plain[-1]:.2f}"


def test_bdsp_apical_bistable_on_latches_and_holds():
    held = _bdsp_apical_run(bistable=True, self_regen=2.0, kir_g=1.0, drive_val=80.0)     # error latches
    silent = _bdsp_apical_run(bistable=True, self_regen=2.0, kir_g=1.0, drive_val=0.0)    # no error -> P0 moat
    transient = _bdsp_apical_run(bistable=False, self_regen=0.0, kir_g=0.0, drive_val=80.0)
    assert held[-1] > ER + 30.0, f"a real top-down error must LATCH a held apical plateau, got {held[-1]:.2f}"
    assert silent[-1] < ER + 3.0, f"no error must stay SILENT at rest (the P0 moat), got {silent[-1]:.2f}"
    assert held[-1] - transient[-1] > 30.0, "the bistability must be load-bearing vs the transient regime"


# --- gap #4 (2026-07-18): the bistable plateau is the BTSP enabler -- it converts ms-timescale plasticity into a
# seconds-long BEHAVIORAL-TIMESCALE one-shot credit window (Bittner-Magee 2017 / Milstein 2021). Pins the core contrast
# from the _gap4_btsp_plateau_gated de-risk (the GO is 6-seed; CI checks one seed's mechanism + moat).

def test_btsp_bistable_plateau_extends_credit_window_to_seconds():
    from research.runners._gap4_btsp_plateau_gated_derisk import run
    r = run(42)
    # HELD plateau: an input ~0.9 s after the trigger still potentiates (behavioral timescale)
    assert r["held_far"] >= 0.5 * r["held_near"], "held plateau must potentiate a far (~0.9s) input (behavioral timescale)"
    # TRANSIENT plateau: the co-active input potentiates (ms window) but the far input does NOT
    assert r["transient_near"] > 0.0, "transient plateau must potentiate a co-active input (the ms window exists)"
    assert r["transient_far"] < 0.05 * r["held_near"], "transient plateau must NOT reach a far input (no seconds window)"
    # the bistability is load-bearing: held reaches far, transient does not
    assert r["held_far"] > r["transient_far"] + 0.3 * r["held_near"], "the bistable latch is load-bearing for the window"
    # the moat: no plateau (gate lesion) and no eligibility (never-active) => no potentiation
    assert r["held_noplateau"] <= 0.05 * r["held_near"], "no-plateau (gate lesion) must not potentiate (the moat)"
    assert r["held_neveractive"] <= 0.05 * r["held_near"], "never-active (no eligibility) must not potentiate"
