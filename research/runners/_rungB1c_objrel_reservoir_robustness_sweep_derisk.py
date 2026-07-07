"""RUNG B-1c OBJREL RESERVOIR-ROBUSTNESS SWEEP (2026-07-06 diagnostic, per two converging research gates).

THE DIAGNOSIS (NOT a capacity wall; the read-out is a SEPARATE residual). The object-relative (objrel) thematic-role
signal is LINEARLY present in the on-bridge SPIKING reservoir feature on 8/10 random seeds (the ANALYTIC 3-way RIDGE
reads objrel-slot0(THEME) held-out = 1.00) but ABSENT on 2/10 (seeds 103, 104: ridge 0.00 / ~0.17). Two research gates
diagnosed this as a MILD operating-point / finite-instance-variance issue -- NOT the Mikulasch-Priesemann representation
wall -- and ranked the CHEAPEST fix as LOWERING THE INPUT SCALING. The reservoir's `RES_IN_SCALE = 320 pA` drives the
Izhikevich neurons deep into a SATURATED / nonlinear regime that spends the fixed reservoir capacity budget on
NONLINEARITY at the expense of the LINEAR MEMORY the objrel read needs (Dambre-2012 capacity-conservation law: total
processing capacity is conserved; a saturated reservoir trades linear-memory capacity for higher-order nonlinear
terms). Lowering the input scaling (+ raising the recurrent weight scale toward the edge-of-chaos spectral radius
rho->1 + a longer per-token integration window T_step for more temporal memory) should restore the linear objrel read
on the 2 failing draws WITHOUT regressing the canonical read.

THE REFRAME (2026-07-06, verified fanned-across-cores): the PURE LINEAR RIDGE reads objrel-slot0 = 1.00 on ALL 10
seeds -- so the objrel signal is linearly PRESENT everywhere; there is NO reservoir-encoding problem. The ENTIRE residual
is the SPIKING READ-OUT: the analytic-Dale GRADED spiking read fails to reproduce that linear discriminant on some seeds
(103 -> 0.00, 104 -> 0.33, traced directly on the SAME cached feature). So this runner now measures BOTH reads (select
with --read) and, for the spiking read, exposes the READ-OUT OP-POINT as swept knobs -- the spiking read-out is the
actual target.

THE METRIC (fast -- NO read-out training, NO plasticity). For a given (reservoir-config, read, seed): build the
byte-identical on-bridge SPIKING reservoir with the swept reservoir params, DRIVE it on the objrel TRAIN + TEST
sentences, cache the final-state per-neuron spike-rate feature, then:
  * --read ridge   (default): fit the ANALYTIC 3-way RIDGE (closed-form; the exact `_ridge_readout`/`_fit_Ws_spiking`
    linear read) on TRAIN, deploy the host argmax on held-out TEST -- the LINEAR "is objrel present in the feature?"
    read (reads 1.00 on all 10 seeds -> the reservoir encoding is fine).
  * --read spiking : the analytic-Dale GRADED SPIKING read -- fit the ridge, split it into a Dale-legal E path (positive
    rows) + inhibitory-population I path (negative rows) via `D._analytic_dale_readout`, deploy the spike-count argmax
    via `D._score` (the EXACT read that produced 103=0.00 / 104=0.33). Reports its held-out objrel-slot0(THEME). THIS is
    the actual residual: does a higher-resolution op-point let the spike-count read recover the discriminant the ridge
    holds?
ALSO reports the canonical-slot accuracy (a sanity guard: canonical must stay high while objrel lifts). Each (config,
read, seed) is fast (~15-60s): build reservoir -> drive -> feature -> read -> objrel accuracy. NO Dale-legal spiking
read-out TRAINING (BPTT), NO delta rule -- the spiking read is the ANALYTIC Dale reference (weights = the ridge split),
so the read-out op-point (not any learning) is what is swept.

THE SWEEP KNOBS (all CLEAN module-constant OVERRIDES -- NO sim/ edit; reversed on exit):
  RESERVOIR (patched on C before the build; read as C globals INSIDE C.wire_reservoir at BUILD time, lines 263/275):
    * --in-scale  <pA>   : overrides C.RES_IN_SCALE (reservoir input scaling; 320 baseline).
    * --rec-scale <float>: multiplies C.RES_EXC_W & C.RES_INH_W (rho->1 edge-of-chaos; 1.0 baseline; E/I ratio preserved).
    * --t-step    <int>  : the per-token integration window for the FEATURE read (12 baseline; passed EXPLICITLY to the
                            read via _feature_at, since RES_T_STEP is a def-time default arg of _drive_and_read).
  SPIKING READ-OUT OP-POINT (patched on D around the spiking read; read as D globals in D._lif_forward /
  D.DANNReadout._inputs / D._analytic_dale_readout at CALL time -> patching before the read cleanly overrides them):
    * --read-t        <int>  : overrides D.READ_T (LIF spike-count integration steps; 25 baseline; HIGHER = more
                                spike-count RESOLUTION -- the hypothesis is thin-margin seeds need more resolution).
    * --read-in-scale <flt>  : overrides D.IN_SCALE (raw feature -> input-current gain; 0.5 baseline graded op-point).
    * --read-thresh   <flt>  : overrides D.THRESH (LIF spike threshold; 1.0 baseline).
    (--read-* are INERT under --read ridge; they only affect the spiking read.)

HONESTY. --read ridge is a LINEAR read (isolates the reservoir encoding -- fine on all seeds). --read spiking is the
analytic-Dale GRADED spike-count read (the actual residual). This is the ANALYTIC Dale reference (weights = the ridge
E/I split), so ONLY the op-point is swept -- NO read-out learning. Held-out test (distinct rng, no leakage). A
DIAGNOSTIC sweep to see whether a higher-resolution spiking op-point recovers the failing seeds toward the ridge's 1.00;
NOT a GO/surpass claim. If READ_T alone does not move a thin-margin seed, that is the honest signal that the fix is a
2-stage/calibrated read (per-pool bias calibration), NOT just more integration steps.

Run (ONE config x the 10 seeds; the controller fans out per-seed x per-op-point-config across cores):
  SIM_BACKEND=numpy python -u -m research.runners._rungB1c_objrel_reservoir_robustness_sweep_derisk \
      --seeds 42 43 44 45 46 100 101 102 103 104 --read spiking --read-t 100 \
      --json research/findings/raw/_rungB1c_objrel_resv_sweep_spk_readt100.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C  # noqa: E402
import research.runners._rungB1c_objrel_per_role_readout_derisk as PR  # noqa: E402
import research.runners._rungB1c_objrel_dann_readout_derisk as D  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX,
)

# ── the data recipe (IDENTICAL to the per-role / DANN harness so the ridge reads the SAME feature) ────────────────
N_TRAIN = 60             # train sentences/construction for the ridge fit (== the c2/per-role/DANN documented baseline)
N_TEST = 12              # held-out test facts/construction (DISTINCT rng from train -- the no-leakage control)
N_ROLES3 = 3             # the 3-way canonical read: AGENT(0), PREDICATE(1), THEME(2)
RIDGE_LAM = 0.1          # ridge regularization (the `analytic_dale_reference` / _ridge_readout lambda; 1e-1 = separable)


# ── clean, reversible reservoir-param override on the imported C module (NO sim/ edit) ────────────────────────────
@contextmanager
def _override_reservoir_params(in_scale, rec_scale):
    """Patch C.RES_IN_SCALE / C.RES_EXC_W / C.RES_INH_W for the duration of a build, then restore. These are read as
    MODULE GLOBALS inside C.wire_reservoir at BUILD time (the W_in scale + the recurrent exc/inh weights), so patching
    them on C before wire_reservoir cleanly overrides the reservoir DRAW. rec_scale multiplies BOTH exc & inh so the
    E/I ratio is preserved while the spectral radius scales toward rho->1 (edge-of-chaos). Fully reversible; no sim/
    edit, no reservoir-internal edit."""
    old_in, old_exc, old_inh = C.RES_IN_SCALE, C.RES_EXC_W, C.RES_INH_W
    C.RES_IN_SCALE = float(in_scale)
    C.RES_EXC_W = float(old_exc) * float(rec_scale)
    C.RES_INH_W = float(old_inh) * float(rec_scale)
    try:
        yield
    finally:
        C.RES_IN_SCALE, C.RES_EXC_W, C.RES_INH_W = old_in, old_exc, old_inh


@contextmanager
def _override_readout_op(read_t, read_in_scale, read_thresh):
    """Patch the SPIKING read-out op-point (D.READ_T / D.IN_SCALE / D.THRESH) for the duration of the analytic-Dale
    spiking read, then restore. These are read as MODULE GLOBALS at CALL time inside D._lif_forward (LEAK/THRESH),
    D.DANNReadout._inputs (IN_SCALE/READ_T), and D._analytic_dale_readout (bakes IN_SCALE into W_e/W_fi) -- so patching
    them on D before the read cleanly overrides the op-point (the IN_SCALE baked into the weights + applied at read time
    both read the SAME patched global, so the graded op-point stays consistent). None => keep D's default. Fully
    reversible; NO sim/ edit, NO read-out learning (this is the ANALYTIC Dale reference; only the op-point moves)."""
    old_t, old_in, old_thr = D.READ_T, D.IN_SCALE, D.THRESH
    if read_t is not None:
        D.READ_T = int(read_t)
    if read_in_scale is not None:
        D.IN_SCALE = float(read_in_scale)
    if read_thresh is not None:
        D.THRESH = float(read_thresh)
    try:
        yield
    finally:
        D.READ_T, D.IN_SCALE, D.THRESH = old_t, old_in, old_thr


# ── the reservoir feature at a swept t_step (RES_T_STEP is a def-time default arg of _drive_and_read, so pass it) ──
def _feature_at(res, enc, toks, t_step):
    """The whole-sequence SPIKING reservoir feature + a +1 bias element -- IDENTICAL to PR._feature except the per-token
    integration window is `t_step` (PR._feature / res.final_state hardwire the def-time default RES_T_STEP=12). Calls
    the reservoir's own `_drive_and_read` (the exact read final_state uses) with an explicit t_step so --t-step sweeps
    the FEATURE read window cleanly (no module patch of the def-time default)."""
    feat, _ = res._drive_and_read(enc.encode(toks), silence=False, ens=None, t_step=int(t_step))
    return np.concatenate([feat, [1.0]])


def _cache_slot_features(res, enc, sentences, t_step):
    """Cache {slot k: (X[n_k, feat_dim], y[n_k])} restricted to the 3-way canonical roles (GOAL/LOCATION skipped) --
    the SAME feature the c2 ridge + per-role reads consume. Driving the spiking reservoir is the expensive part; the
    ridge fit reuses the cached X/y."""
    S = defaultdict(list); Y = defaultdict(list)
    for toks, roles in sentences:
        f = _feature_at(res, enc, toks, t_step)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:                 # GOAL/LOCATION are not in the 3-way canonical read
                continue
            S[k].append(f); Y[k].append(tgt)
    return {k: (np.asarray(S[k], dtype=np.float64), np.asarray(Y[k], dtype=np.int64)) for k in S}


def _ridge_readout(X, y, lam=RIDGE_LAM):
    """The 3-way one-hot closed-form ridge read-out matrix W (feat_dim x N_ROLES3) -- the ANALYTIC linear discriminant
    (== _rungB1c_objrel_dann_readout._ridge_readout / C._fit_Ws_spiking's ridge solve). Held-out objrel-slot0 = 1.00 at
    lam=0.1 on the encoding seeds; this is the LINEAR read that isolates 'is objrel present in the reservoir feature?'."""
    T = np.zeros((len(y), N_ROLES3), dtype=np.float64)
    T[np.arange(len(y)), y] = 1.0
    Xd = X.astype(np.float64)
    return np.linalg.solve(Xd.T @ Xd + lam * np.eye(Xd.shape[1]), Xd.T @ T)


def _fit_slot_ridges(slot_train):
    """Fit one 3-way ridge per content slot on the cached TRAIN features. Returns {slot k: W[feat_dim, 3]}."""
    return {k: _ridge_readout(X, y) for k, (X, y) in slot_train.items()}


def _score_ridge(Wk, res, enc, sentences, t_step):
    """Deploy the per-slot ridge argmax on the HELD-OUT sentences (the feature is the REAL spiking reservoir read).
    Returns (overall_acc, slot0_acc, per_slot_hits, per_slot_tot). slot0_acc on the OBJREL set = the objrel-slot0
    (THEME) metric (role != position; the thing the failing seeds miss); on the CANON set = the AGENT sanity slot."""
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = _feature_at(res, enc, toks, t_step)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:
                continue
            if k not in Wk:
                continue
            pred = int(np.argmax((f @ Wk[k])[:N_ROLES3]))
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot)


def _score_spiking(ros, res, enc, sentences):
    """Deploy the analytic-Dale GRADED SPIKING read (D._score: spike-count argmax over the E+I output LIF) on the
    HELD-OUT sentences. Returns (overall_acc, slot0_acc, per_slot_hits, per_slot_tot). `ros` = the per-slot analytic
    Dale-legal read-outs (D._analytic_dale_readout, weights = the ridge E/I split). This is the EXACT read that
    produced 103=0.00 / 104=0.33; only the read-out OP-POINT (D.READ_T/IN_SCALE/THRESH) is swept -- NO learning."""
    ov, s0, ps_hit, ps_tot, _spk, _inh = D._score(ros, res, enc, sentences)
    return ov, s0, ps_hit, ps_tot


# ── the 2-stage per-pool-CALIBRATED spiking read (EMERGE-77 pattern: Turrigiano per-unit homeostatic bias norm) ────
# THE MECHANISM (established by direct instrumentation of the failing seeds). The objrel-slot0 failure is a spike-COUNT
# TIE: the analytic-Dale membrane correctly favours THEME (net drive THEME 0.26 > AGENT 0.23), but READ_T=25 QUANTIZES
# both the AGENT and THEME output neurons to the SAME integer spike count -- on 103/104 the raw slot0 count is [4,0,4]
# (a tie -> argmax defaults to AGENT); on the clean seeds it is [3,0,4] (THEME wins by 1). So the residual is a per-pool
# COUNT-tie at the ambiguous THEME/AGENT slot0, and the EMERGE-77 fix -- a per-pool bias measured at a TASK-BLIND
# reference, applied to BREAK THE TIE toward the reference-disfavoured (minority THEME) neuron -- is the clean lever. It
# is applied CONDITIONALLY (only when the top-2 output spike counts TIE within `tie_margin`), so a CONFIDENT clean read
# (42 = [3,0,4], margin 1) is NEVER perturbed -- the calibration is non-destructive to the already-correct seeds AND to
# canonical (slots 1/2, never ambiguous, are left RAW). This is the honest slot0-targeted, tie-only bias-calibration.
def _pool_bias_vector(ro, ref_feature):
    """STAGE 1 (CALIBRATE) -- the per-OUTPUT-ROLE-neuron BIAS for ONE per-slot analytic-Dale read-out `ro`: drive the
    read-out's OUTPUT LIF with the TASK-BLIND reference feature, count each output role neuron's spikes, mean-centre.
    The reference is TASK-BLIND: the caller passes the CLASS-BALANCED mean TRAIN feature for the slot (the equal-weight
    average of the per-role TRAIN-feature means -- balanced with the KNOWN TRAIN labels, NEVER the test labels/answer;
    the load-bearing anti-cheat). Class-balancing removes the 7:1 majority-AGENT tilt so the bias captures the systematic
    per-role output-neuron OFFSET, not the majority signal. Returns a mean-centred (3,) per-role bias vector."""
    _pred, out_ref, _inh = ro.predict_spikes(ref_feature.astype(np.float32))   # per-role output spike count at the ref
    b = out_ref.astype(np.float64)
    return b - float(b.mean())                                                # mean-centred (a per-role SCALAR offset)


def _class_balanced_ref(X, y):
    """The TASK-BLIND, role-neutral reference feature for a slot: the equal-weight mean of the per-role TRAIN-feature
    means (balanced with the KNOWN TRAIN labels, so the 7:1 majority-AGENT tilt is removed). No test-label peeking."""
    cms = [X[y == c].mean(axis=0) for c in np.unique(y)]
    return np.mean(cms, axis=0)


def _score_calibrated(ros, slot_train, res, enc, sentences, tie_margin=0):
    """STAGE 2 (READ) -- deploy the analytic-Dale spiking read; at the AMBIGUOUS slot0, when the top-2 OUTPUT LIF spike
    counts TIE within `tie_margin`, break the tie with the STAGE-1 per-pool bias (`argmax(out - b_0)`); otherwise deploy
    the RAW spike-count argmax. slots 1/2 (never ambiguous) are always RAW -> canonical is never perturbed. The bias for
    slot0 is measured at the TASK-BLIND CLASS-BALANCED reference (no test-label peeking). The RAW (un-calibrated) read is
    `_score_spiking` (the causal control -- it must still fail on 103/104). GENUINELY spiking: `out` is the OUTPUT LIF
    spike count, and the argmax is over spike counts (a per-pool SCALAR bias tie-break, NOT a sign-flip -- Dale-legality
    of the weights is untouched). Returns (overall_acc, slot0_acc, per_slot_hits, per_slot_tot)."""
    bias0 = _pool_bias_vector(ros[0], _class_balanced_ref(*slot_train[0])) if 0 in ros else None   # STAGE 1 (slot0)
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = PR._feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]                              # the module-level import (== D's _ROLE_IDX)
            if tgt >= N_ROLES3:
                continue
            if k not in ros:
                continue
            _pred, out, _inh = ros[k].predict_spikes(f)               # the RAW per-role output spike count (spiking)
            o = out.astype(np.float64)
            if k == 0 and bias0 is not None:                          # STAGE 2: tie-only slot0 bias-calibration
                top2 = np.sort(o)[::-1]
                if (top2[0] - top2[1]) <= tie_margin:                 # only intervene on a (near-)count-TIE
                    o = o - bias0
            pred = int(np.argmax(o))
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot)


# ── the GENUINE answer-independent fixes (NOT a THEME prior): break the saturation TIE by the ACTUAL graded DRIVE ──────
# THE DIAGNOSIS (confirmed by direct instrumentation, per the calibrated-read retraction). objrel-slot0 fails on 103/104
# because at the graded op-point BOTH the AGENT and THEME output pools saturate to the SAME integer spike count
# ([4,0,4] -> argmax defaults to AGENT), even though the graded output DRIVE genuinely favours THEME. The refuted fix
# ("--read calibrated") was a per-pool bias SUBTRACTION that adversarial-verify showed is byte-identical to a hard
# "pick THEME on a slot0 tie" MINORITY PRIOR (zero measured content). The two fixes below break the tie by the ACTUAL
# graded drive the spike count quantizes away -- answer-independent (they give whichever role the DRIVE favours, AGENT
# or THEME), so the DISTINGUISHING anti-cheat separates them from the prior: on an AGENT-favouring synthetic tie, a
# genuine fix gives AGENT; the THEME prior gives THEME.


def _graded_output_drive(ro, f):
    """The ANALOG (pre-threshold) net OUTPUT drive per role for a per-slot analytic-Dale read-out `ro`, computed from
    the read-out's OWN weights (ANSWER-INDEPENDENT -- no test labels, no 'THEME'). This is the continuous membrane the
    spike-count read QUANTIZES AWAY: the E-path excitatory drive PLUS the I-path inhibitory drive, with the interneuron
    population replaced by its CONTINUOUS activation (so this is the pure analog quantity, not a re-quantized spike
    count). For the analytic Dale read-out (W_e = Wpos*IN_SCALE, W_fi = Wneg*IN_SCALE, W_io = -I) this equals
    IN_SCALE*(f_s @ Wpos - f_s @ Wneg) = IN_SCALE * f_s @ W_ridge -- the exact ridge discriminant the ridge reads 1.00
    on all 10 seeds -- but it is derived here PURELY from `ro`'s weights, so it holds for any Dale-legal read-out. The
    feature is scaled by D.IN_SCALE exactly as ro._inputs does, so the graded op-point matches the spike read."""
    f_s = (np.asarray(f, dtype=np.float64) * float(D.IN_SCALE))
    drive_e = f_s @ ro.W_e.astype(np.float64)                       # (3,) excitatory analog drive   (W_e >= 0)
    drive_ih = f_s @ ro.W_fi.astype(np.float64)                     # (H,) analog drive onto interneurons (W_fi >= 0)
    drive_i = drive_ih @ ro.W_io.astype(np.float64)                 # (3,) inhibitory analog drive    (W_io <= 0)
    return drive_e + drive_i                                        # (3,) net analog output membrane (role-signed)


def _score_gradedtie(ros, res, enc, sentences, tie_margin=0):
    """`--read gradedtie` -- deploy the analytic-Dale spiking read; on an EXACT slot0 spike-count TIE (top-2 counts equal
    within `tie_margin`), break it by the argmax of the ANSWER-INDEPENDENT GRADED output DRIVE (`_graded_output_drive`,
    a real neural sub-threshold quantity), NOT a THEME prior -- it gives whichever role the DRIVE favours. Off a tie the
    RAW spike-count argmax is deployed unchanged; slots 1/2 are always RAW (canonical never perturbed). Dale-legal (no
    sign flip -- only the read-out DECISION rule uses the analog membrane). Returns (overall, slot0, per_slot_hits,
    per_slot_tot). The RAW read (`_score_spiking`) is the causal control (still fails 103/104)."""
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = PR._feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:
                continue
            if k not in ros:
                continue
            _pred, out, _inh = ros[k].predict_spikes(f)              # RAW per-role output spike count (genuinely spiking)
            o = out.astype(np.float64)
            if k == 0:                                               # slot0: the ambiguous THEME/AGENT slot
                top2 = np.sort(o)[::-1]
                if (top2[0] - top2[1]) <= tie_margin:               # only intervene on a (near-)count TIE
                    g = _graded_output_drive(ros[0], f)             # break by the ACTUAL graded drive (answer-independent)
                    pred = int(np.argmax(g))
                else:
                    pred = int(np.argmax(o))
            else:
                pred = int(np.argmax(o))                            # slots 1/2 always RAW -> canonical untouched
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot)


# ── per-pool GAIN normalization (Turrigiano homeostatic scaling): keep each pool in its LINEAR range at a TASK-BLIND ──
# reference drive so the spike RATE tracks the graded drive (THEME's higher drive => more spikes => no tie). The gain is
# a per-role POSITIVE scalar on that role's E AND I output weights (Dale-legal: positive scaling never flips a sign),
# chosen so the reference drive gives a target sub-saturation spike count. The reference is TASK-BLIND: the equal-weight
# per-role mean TRAIN feature (balanced with the KNOWN TRAIN labels, NEVER the test labels/answer).
def _class_balanced_ref_feat(X, y):
    """The TASK-BLIND role-neutral reference feature for a slot: the equal-weight mean of the per-role TRAIN-feature
    means (balanced with the KNOWN TRAIN labels so the 7:1 majority-AGENT tilt is removed). NO test-label peeking."""
    cms = [X[y == c].mean(axis=0) for c in np.unique(y)]
    return np.mean(cms, axis=0).astype(np.float64)


def _gain_normalize_readout(ro, ref_feat, target_count=2.0):
    """STAGE 1 (per-pool GAIN NORM) -- return a per-role POSITIVE gain vector g (3,) that scales role r's E and I output
    weights so that, at the TASK-BLIND reference drive, role r's OUTPUT LIF fires ~`target_count` spikes over READ_T --
    i.e. every pool sits in its LINEAR (non-saturated) range at the reference (Turrigiano synaptic scaling; the reference
    is TASK-BLIND, never the answer). Concretely: measure each role's RAW reference output spike count; g_r =
    target_count / max(raw_r, eps) but CLIPPED to <= 1 (only DOWN-scale a saturated pool -- never amplify above unity, so
    a silent pool is not blown up). Positive-only scaling => Dale-legality of every weight is preserved (no sign flip).
    Applied to a COPY of ro so the original is untouched. Returns (ro_scaled, g)."""
    _pred, out_ref, _inh = ro.predict_spikes(ref_feat.astype(np.float32))   # raw per-role ref output spike count
    raw = np.asarray(out_ref, dtype=np.float64)
    g = np.clip(float(target_count) / np.maximum(raw, 1e-6), None, 1.0)     # only DOWN-scale saturated pools (Dale-legal)
    ro2 = D.DANNReadout(ro.feat_dim, h_inh=ro.h_inh, seed=0)
    ro2.h_inh = ro.h_inh
    ro2.W_e = (ro.W_e * g[None, :].astype(np.float32))                      # scale each role's excitatory column (>= 0)
    ro2.W_fi = ro.W_fi.copy()                                               # interneuron INPUT weights unchanged (>= 0)
    ro2.W_io = (ro.W_io * g[None, :].astype(np.float32))                    # scale each role's inhibitory column (<= 0)
    return ro2, g


def _score_gainnorm(ros, slot_train, res, enc, sentences, target_count=2.0):
    """`--read gainnorm` -- STAGE 1: per-pool GAIN-normalize slot0 at the TASK-BLIND class-balanced reference so both the
    AGENT and THEME pools sit in their LINEAR range (no saturation). STAGE 2 (READ): deploy the SPIKE-COUNT argmax on the
    gain-normalized read-out -- now the spike RATE tracks the graded drive, so THEME's higher drive fires MORE and the
    tie is broken by GENUINE spiking, NOT a prior. Answer-independent: the reference is TASK-BLIND; the gain is a
    per-pool positive scalar (Dale-legal). slots 1/2 use the RAW read-out (canonical untouched). Returns (overall,
    slot0, per_slot_hits, per_slot_tot)."""
    ro0 = ros.get(0)
    ro0_gn = None
    if ro0 is not None and 0 in slot_train:
        ref = _class_balanced_ref_feat(*slot_train[0])
        ro0_gn, _g = _gain_normalize_readout(ro0, ref, target_count=target_count)
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = PR._feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:
                continue
            if k not in ros:
                continue
            ro_use = ro0_gn if (k == 0 and ro0_gn is not None) else ros[k]
            _pred, out, _inh = ro_use.predict_spikes(f)             # spike-count read on the gain-normalized pool
            pred = int(np.argmax(out.astype(np.float64)))
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot)


def _fix_predict_slot0(ro, ro_gn, f, mechanism, bias0=None, tie_margin=0):
    """What does ONE fix predict for slot0 on feature `f`? Returns the predicted role index (0=AGENT,1=PRED,2=THEME).
      raw       : spike-count argmax (the RAW causal control).
      gradedtie : spike-count argmax, but on a spike-count TIE break by argmax of the graded output drive.
      gainnorm  : spike-count argmax on the per-pool GAIN-normalized read-out `ro_gn`.
      calibrated: spike-count argmax, but on a TIE subtract the STAGE-1 per-pool bias `bias0` (the REFUTED prior).
    All are GENUINELY spiking (argmax over an output-LIF spike count / a per-pool bias/gain of it); NONE reads the
    label. This is the harness for the DISTINGUISHING anti-cheat."""
    _p, out, _i = ro.predict_spikes(f)
    o = out.astype(np.float64)
    if mechanism == "gainnorm":
        _p2, out2, _i2 = ro_gn.predict_spikes(f)
        return int(np.argmax(out2.astype(np.float64)))
    top2 = np.sort(o)[::-1]
    tied = (top2[0] - top2[1]) <= tie_margin
    if mechanism == "raw" or not tied:
        return int(np.argmax(o))
    if mechanism == "gradedtie":
        return int(np.argmax(_graded_output_drive(ro, f)))
    if mechanism == "calibrated":
        return int(np.argmax(o - bias0)) if bias0 is not None else int(np.argmax(o))
    return int(np.argmax(o))


def run_distinguishing(seed, corpus):
    """THE DISTINGUISHING ANTI-CHEAT (the load-bearing test that separates a GENUINE answer-independent fix from the
    refuted THEME-on-tie PRIOR). Construct real slot0 spike-count TIES with a KNOWN graded-drive direction:
      * AGENT-favouring ties: drawn from the CANONICAL (transitive) slot0 population -- true role AGENT, and the graded
        output drive genuinely favours AGENT (drive_AGENT > drive_THEME).
      * THEME-favouring ties: drawn from the OBJREL slot0 population -- true role THEME, graded drive favours THEME.
    A tie = the top-2 output spike counts are EQUAL (the exact [4,0,4] failure). On EACH tie population, report what
    each mechanism (raw / gradedtie / gainnorm / calibrated) predicts. THE KEY RESULT:
      * A GENUINE answer-independent fix gives AGENT on the AGENT-favouring ties AND THEME on the THEME-favouring ties.
      * A THEME PRIOR (the refuted 'calibrated') gives THEME on BOTH -> FAILS the AGENT-favouring case.
    Returns a dict of per-mechanism accuracy on each tie population (frac giving the DRIVE-FAVOURED role)."""
    C.WS_BIAS_SCALE_C2 = 0.0
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    ub, ens, inh, res, res_idx = PR._build(seed, corpus, enc)
    slot_train = _cache_slot_features(res, enc, train, C.RES_T_STEP)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]
    ros = D._analytic_dale_readout(slot_train, feat_dim, seed)
    ro0 = ros[0]
    # gain-normalized slot0 read-out + the refuted per-pool bias (both at the TASK-BLIND class-balanced reference)
    ref = _class_balanced_ref_feat(*slot_train[0])
    ro0_gn, _g = _gain_normalize_readout(ro0, ref, target_count=2.0)
    bias0 = _pool_bias_vector(ro0, ref)          # the REFUTED prior's per-pool bias (task-blind, mean-centred)

    def _slot0_features(sentences):
        out = []
        for toks, roles in sentences:
            positions = sorted(roles)
            pos0 = positions[0]
            tgt = _ROLE_IDX[roles[pos0]]
            if tgt >= N_ROLES3:
                continue
            f = PR._feature(res, enc, toks)
            out.append((f, tgt))
        return out

    def _collect_ties(sentences, favour_role):
        """Real slot0 cases that (a) spike-count TIE (top-2 output counts equal, the [4,0,4] failure) AND (b) whose
        graded drive genuinely favours `favour_role` (drive[favour] is the graded argmax) -- i.e. a KNOWN-direction tie."""
        ties = []
        for f, tgt in _slot0_features(sentences):
            _p, out, _i = ro0.predict_spikes(f)
            o = out.astype(np.float64)
            top2 = np.sort(o)[::-1]
            if (top2[0] - top2[1]) > 0:            # not a tie -> skip
                continue
            g = _graded_output_drive(ro0, f)
            if int(np.argmax(g)) != favour_role:   # the graded drive must favour the intended role
                continue
            ties.append((f, tgt, o, g))
        return ties

    theme_ties = _collect_ties(objr, favour_role=_ROLE_IDX["THEME"])    # real THEME-favouring ties (from objrel slot0)

    # ── SYNTHESIZE AGENT-favouring ties (the load-bearing case -- REAL data has NONE, because canonical slot0 never ties
    #    and objrel slot0's drive always favours THEME). Take a real AGENT-strong feature (canonical slot0, drive argmax
    #    AGENT) and a real THEME-strong feature (objrel slot0, drive argmax THEME); form the convex blend
    #    f_lambda = (1-lambda)*f_theme + lambda*f_agent and sweep lambda: as lambda rises the drive crosses THEME->AGENT
    #    and the spike counts pass through a TIE. Any blend that (a) spike-count TIES AND (b) has AGENT-favouring graded
    #    drive is a genuine AGENT-favouring tie: a fix that FOLLOWS THE DRIVE gives AGENT; a THEME PRIOR gives THEME. This
    #    is answer-independent (no labels used -- the blend is constructed purely from the drive/count geometry). ───────
    def _feat_pool(sentences, want_role):
        out = []
        for f, tgt in _slot0_features(sentences):
            if int(np.argmax(_graded_output_drive(ro0, f))) == want_role:
                out.append(f)
        return out
    agent_strong = _feat_pool(canon, _ROLE_IDX["AGENT"])   # AGENT-favouring-drive features (canonical slot0)
    theme_strong = _feat_pool(objr, _ROLE_IDX["THEME"])    # THEME-favouring-drive features (objrel slot0)
    synth_agent_ties = []
    srng = np.random.default_rng(seed * 31 + 7)
    for fa in agent_strong:
        for ft in theme_strong:
            for lam in np.linspace(0.02, 0.98, 49):        # sweep the blend from THEME-strong (lam~0) to AGENT-strong
                fb = ((1.0 - lam) * ft + lam * fa).astype(np.float32)
                _p, out, _i = ro0.predict_spikes(fb)
                o = out.astype(np.float64)
                top2 = np.sort(o)[::-1]
                if (top2[0] - top2[1]) > 0:                # require an EXACT spike-count tie (the [4,0,4] failure shape)
                    continue
                g = _graded_output_drive(ro0, fb)
                if int(np.argmax(g)) != _ROLE_IDX["AGENT"]:  # require the graded drive to genuinely favour AGENT
                    continue
                synth_agent_ties.append((fb, _ROLE_IDX["AGENT"], o, g))
                break                                       # one crossover tie per (fa, ft) pair is enough
        if len(synth_agent_ties) >= 24:
            break

    def _eval(ties, favoured_role):
        """For each mechanism, the fraction of the ties on which it predicts the DRIVE-FAVOURED role."""
        res_m = {}
        for mech in ("raw", "gradedtie", "gainnorm", "calibrated"):
            hits = sum(int(_fix_predict_slot0(ro0, ro0_gn, f, mech, bias0=bias0) == favoured_role)
                       for (f, _t, _o, _g) in ties)
            res_m[mech] = round(hits / max(len(ties), 1), 3)
        return res_m

    agent_res = _eval(synth_agent_ties, _ROLE_IDX["AGENT"])   # AGENT-favouring ties: genuine->AGENT (~1), prior->THEME (~0)
    theme_res = _eval(theme_ties, _ROLE_IDX["THEME"])         # THEME-favouring ties: genuine + prior both ->THEME (~1)

    # a genuine answer-independent fix gives the DRIVE-favoured role on BOTH populations; the THEME prior fails AGENT.
    genuine = {m: bool(agent_res.get(m, 0.0) >= 0.99 and theme_res.get(m, 0.0) >= 0.99)
               for m in ("raw", "gradedtie", "gainnorm", "calibrated")}
    return {
        "seed": int(seed),
        "n_agent_favouring_ties": len(synth_agent_ties),   # SYNTHETIC (real data has none -- the load-bearing case)
        "n_theme_favouring_ties": len(theme_ties),         # REAL (objrel slot0)
        "agent_favouring_tie_gives_favoured": agent_res,   # AGENT expected -> genuine fix ~1.0; THEME prior ~0.0 (KEY)
        "theme_favouring_tie_gives_favoured": theme_res,   # THEME expected -> genuine fix ~1.0; prior ~1.0 (both give THEME)
        "answer_independent": genuine,                     # True iff gives DRIVE-favoured on BOTH (genuine, not a prior)
        "example_agent_tie": ([{"counts": [int(x) for x in o], "graded_drive": [round(float(x), 4) for x in g],
                                "true_role": int(t)} for (_f, t, o, g) in synth_agent_ties[:3]]),
        "example_theme_tie": ([{"counts": [int(x) for x in o], "graded_drive": [round(float(x), 4) for x in g],
                                "true_role": int(t)} for (_f, t, o, g) in theme_ties[:3]]),
    }


def run_seed(seed, corpus, in_scale, rec_scale, t_step, read, read_t, read_in_scale, read_thresh):
    """Build the byte-identical c2 reservoir WITH the overridden (in_scale, rec_scale) reservoir params + the swept
    t_step feature window, cache the spiking reservoir feature, then read objrel with EITHER the analytic 3-way RIDGE
    (`read='ridge'`, the LINEAR "is objrel present?" read) OR the analytic-Dale GRADED SPIKING read (`read='spiking'`,
    D._analytic_dale_readout + D._score at the overridden op-point -- the actual residual). Reports objrel-slot0(THEME)
    + the canonical sanity accuracy. NO read-out training/plasticity in EITHER path. Returns the per-seed row dict."""
    t0 = time.time()
    # match the per-role/c2 data recipe knobs (idempotent; these are read-out-side, harmless to the feature read).
    C.WS_BIAS_SCALE_C2 = 0.0
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)          # DISTINCT rng => test facts held out from train (no leakage)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    # ── build the reservoir with the SWEPT (in_scale, rec_scale) -- clean module override, restored on exit ──────────
    with _override_reservoir_params(in_scale, rec_scale):
        ub, ens, inh, res, res_idx = PR._build(seed, corpus, enc)

    print(f"[resv-sweep seed {seed}] read={read} in_scale={in_scale} rec_scale={rec_scale} t_step={t_step} "
          f"(read_t={read_t} read_in_scale={read_in_scale} read_thresh={read_thresh}) caching spiking features on "
          f"{len(train)} train sentences (slice {res_idx[0]}..{res_idx[-1]})...", flush=True)
    slot_train = _cache_slot_features(res, enc, train, t_step)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]

    if read in ("spiking", "calibrated", "gradedtie", "gainnorm"):
        # the analytic-Dale GRADED spiking read at the OVERRIDDEN op-point (the actual residual). NOTE: D._score /
        # D._cache_slot_features drive the reservoir feature via PR._feature (RES_T_STEP=12 default) -- the reservoir
        # feature statistics are UNCHANGED by --t-step for the spiking path (the spiking-read op-point is the target),
        # so --read spiking is swept over the READ-OUT op-point, not the feature window.
        with _override_readout_op(read_t, read_in_scale, read_thresh):
            ros = D._analytic_dale_readout(slot_train, feat_dim, seed)
            if read == "calibrated":
                # STAGE 1+2 (CALIBRATE + tie-only READ): the TASK-BLIND CLASS-BALANCED reference per slot from the CACHED
                # TRAIN features (equal-weight per-role mean; balanced with the KNOWN TRAIN labels, NEVER the test
                # labels/answer -- the load-bearing anti-cheat). Applied ONLY to break a slot0 count-TIE (non-destructive).
                # NOTE: adversarial-verify REFUTED this as a MINORITY-THEME PRIOR (retained here as the refuted control).
                canon_acc, canon_s0, canon_ps, canon_pt = _score_calibrated(ros, slot_train, res, enc, canon)
                objr_acc, objr_s0, objr_ps, objr_pt = _score_calibrated(ros, slot_train, res, enc, objr)
            elif read == "gradedtie":
                # ANSWER-INDEPENDENT: on a slot0 count-tie, break by the argmax of the ACTUAL graded output drive.
                canon_acc, canon_s0, canon_ps, canon_pt = _score_gradedtie(ros, res, enc, canon)
                objr_acc, objr_s0, objr_ps, objr_pt = _score_gradedtie(ros, res, enc, objr)
            elif read == "gainnorm":
                # ANSWER-INDEPENDENT: per-pool GAIN norm at a TASK-BLIND reference -> linear range -> spike RATE tracks drive.
                canon_acc, canon_s0, canon_ps, canon_pt = _score_gainnorm(ros, slot_train, res, enc, canon)
                objr_acc, objr_s0, objr_ps, objr_pt = _score_gainnorm(ros, slot_train, res, enc, objr)
            else:
                canon_acc, canon_s0, canon_ps, canon_pt = _score_spiking(ros, res, enc, canon)
                objr_acc, objr_s0, objr_ps, objr_pt = _score_spiking(ros, res, enc, objr)
    else:
        Wk = _fit_slot_ridges(slot_train)
        canon_acc, canon_s0, canon_ps, canon_pt = _score_ridge(Wk, res, enc, canon, t_step)
        objr_acc, objr_s0, objr_ps, objr_pt = _score_ridge(Wk, res, enc, objr, t_step)

    elapsed = round(time.time() - t0, 1)
    key = "ridge" if read == "ridge" else "spiking"     # calibrated + spiking both file under the "spiking_*" keys
    d = {
        "seed": int(seed), "read": read,
        "in_scale": float(in_scale), "rec_scale": float(rec_scale), "t_step": int(t_step),
        "read_t": (int(read_t) if read_t is not None else int(D.READ_T)),
        "read_in_scale": (float(read_in_scale) if read_in_scale is not None else float(D.IN_SCALE)),
        "read_thresh": (float(read_thresh) if read_thresh is not None else float(D.THRESH)),
        "ridge_lambda": RIDGE_LAM, "feat_dim": int(feat_dim),
        f"{key}_objrel": {
            "objrel_acc": round(objr_acc, 3),
            "objrel_slot0_THEME": round(objr_s0, 3),                 # THE diagnostic metric (role != position)
            "objrel_per_slot": [f"{h}/{t}" for h, t in zip(objr_ps, objr_pt)],
        },
        f"{key}_canonical": {
            "canonical_acc": round(canon_acc, 3),
            "canonical_slot0_AGENT": round(canon_s0, 3),             # sanity: role == position, must stay high
            "canonical_per_slot": [f"{h}/{t}" for h, t in zip(canon_ps, canon_pt)],
        },
        # flat aliases so the aggregator reads uniformly across --read modes
        "objrel_slot0_THEME": round(objr_s0, 3),
        "objrel_acc": round(objr_acc, 3),
        "canonical_acc": round(canon_acc, 3),
        "elapsed_s": elapsed,
        "objrel_present": bool(objr_s0 >= 0.90),                     # objrel READ at this config+read
        "canonical_ok": bool(canon_acc >= 0.90),                     # the canonical read is not regressed
    }
    return d


def _print_seed(s, d):
    key = "ridge" if d["read"] == "ridge" else "spiking"     # all non-ridge reads file under "spiking_*"
    ro = d[f"{key}_objrel"]; rc = d[f"{key}_canonical"]
    tag = {"ridge": "RIDGE", "calibrated": "CALIBRATED", "gradedtie": "GRADEDTIE",
           "gainnorm": "GAINNORM"}.get(d["read"], "SPIKING")
    print(f"[seed {s}] read={d['read']} in{d['in_scale']:.0f} rec{d['rec_scale']:.2f} T{d['t_step']} "
          f"readT{d['read_t']} readIn{d['read_in_scale']:.2f} readThr{d['read_thresh']:.2f} "
          f"{tag} objrel-slot0(THEME) {ro['objrel_slot0_THEME']:.2f} (objrel-acc {ro['objrel_acc']:.2f} "
          f"slots {ro['objrel_per_slot']}) | canon-acc {rc['canonical_acc']:.2f} slot0(AGENT) "
          f"{rc['canonical_slot0_AGENT']:.2f} (slots {rc['canonical_per_slot']}) "
          f"[objrel-present {d['objrel_present']} canon-ok {d['canonical_ok']}] ({d['elapsed_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 100, 101, 102, 103, 104])
    ap.add_argument("--in-scale", type=float, default=C.RES_IN_SCALE,
                    help="W_in input drive scale (pA); C.RES_IN_SCALE baseline=320. The swept lever (160/80/40).")
    ap.add_argument("--rec-scale", type=float, default=1.0,
                    help="multiplier on C.RES_EXC_W/RES_INH_W (rho->1 edge-of-chaos; 1.0 baseline).")
    ap.add_argument("--t-step", type=int, default=C.RES_T_STEP,
                    help="per-token feature integration window (C.RES_T_STEP baseline=12; longer = more memory). "
                         "Affects the RESERVOIR feature; inert for --read spiking (that path reads the RES_T_STEP=12 "
                         "feature and sweeps the READ-OUT op-point instead).")
    ap.add_argument("--read", choices=["ridge", "spiking", "calibrated", "gradedtie", "gainnorm"], default="ridge",
                    help="ridge = the LINEAR 'is objrel present in the feature?' read (1.00 all 10 seeds -> reservoir "
                         "is fine); spiking = the analytic-Dale GRADED spike-count read (the actual residual; 103=0.00 "
                         "104=0.33 at the baseline op-point -- the RAW causal control); calibrated = the REFUTED per-pool "
                         "bias-subtraction (adversarial-verify: a MINORITY-THEME PRIOR, retained as the refuted control); "
                         "gradedtie = ANSWER-INDEPENDENT: on a slot0 spike-count TIE, break by argmax of the ACTUAL "
                         "graded output DRIVE (the pre-threshold membrane the count quantizes away, from the read-out's "
                         "OWN weights -- gives whichever role the DRIVE favours, AGENT or THEME, NOT a prior); "
                         "gainnorm = ANSWER-INDEPENDENT: per-pool GAIN normalization (Turrigiano) at a TASK-BLIND "
                         "reference so each pool sits in its LINEAR range -> the spike RATE tracks the graded drive -> "
                         "THEME's higher drive fires MORE -> no tie (spike-COUNT argmax kept).")
    ap.add_argument("--read-t", type=int, default=None,
                    help="[--read spiking] overrides D.READ_T (LIF spike-count integration steps; 25 baseline). "
                         "HIGHER = more spike-count resolution -- the thin-margin hypothesis.")
    ap.add_argument("--read-in-scale", type=float, default=None,
                    help="[--read spiking] overrides D.IN_SCALE (feature->input-current gain; 0.5 baseline graded).")
    ap.add_argument("--read-thresh", type=float, default=None,
                    help="[--read spiking] overrides D.THRESH (LIF spike threshold; 1.0 baseline).")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_rungB1c_objrel_resv_sweep.json")
    ap.add_argument("--distinguish", action="store_true",
                    help="Run the DISTINGUISHING ANTI-CHEAT instead of the sweep: construct real slot0 spike-count TIES "
                         "with a KNOWN graded-drive direction (AGENT-favouring from canonical slot0 vs THEME-favouring "
                         "from objrel slot0) and report what each mechanism (raw/gradedtie/gainnorm/calibrated) predicts. "
                         "A GENUINE answer-independent fix gives the DRIVE-FAVOURED role on BOTH (AGENT on AGENT-ties, "
                         "THEME on THEME-ties); the refuted THEME PRIOR gives THEME on both -> FAILS the AGENT case.")
    args = ap.parse_args()

    t0 = time.time()
    corpus = C.setup_corpus(seed=42)

    if args.distinguish:
        print(f"[resv-sweep DISTINGUISH] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])} | "
              f"the DISTINGUISHING anti-cheat: on real slot0 spike-count TIES with a KNOWN graded-drive direction, does "
              f"the fix give the DRIVE-FAVOURED role (GENUINE) or always THEME (a PRIOR)? seeds {args.seeds}", flush=True)
        drows = []
        for s in args.seeds:
            dd = run_distinguishing(s, corpus)
            drows.append(dd)
            af = dd["agent_favouring_tie_gives_favoured"]; tf = dd["theme_favouring_tie_gives_favoured"]
            print(f"[seed {s} DISTINGUISH] agent-ties n={dd['n_agent_favouring_ties']} theme-ties "
                  f"n={dd['n_theme_favouring_ties']} | AGENT-favouring-tie gives-AGENT: "
                  f"raw {af['raw']:.2f} gradedtie {af['gradedtie']:.2f} gainnorm {af['gainnorm']:.2f} "
                  f"calibrated {af['calibrated']:.2f} | THEME-favouring-tie gives-THEME: "
                  f"raw {tf['raw']:.2f} gradedtie {tf['gradedtie']:.2f} gainnorm {tf['gainnorm']:.2f} "
                  f"calibrated {tf['calibrated']:.2f} | answer-independent {dd['answer_independent']}", flush=True)
        # aggregate: a mechanism is GENUINELY answer-independent iff it gives the DRIVE-favoured role on BOTH tie
        # populations across all seeds that HAVE an agent-favouring tie (the load-bearing case the prior fails).
        n_agent = sum(d["n_agent_favouring_ties"] for d in drows)
        n_theme = sum(d["n_theme_favouring_ties"] for d in drows)
        def _pooled(pop_key, favoured_key):
            m = {}
            for mech in ("raw", "gradedtie", "gainnorm", "calibrated"):
                num = sum(d[pop_key][mech] * d[favoured_key] for d in drows)
                den = sum(d[favoured_key] for d in drows)
                m[mech] = round(num / max(den, 1), 3)
            return m
        agent_pooled = _pooled("agent_favouring_tie_gives_favoured", "n_agent_favouring_ties")
        theme_pooled = _pooled("theme_favouring_tie_gives_favoured", "n_theme_favouring_ties")
        genuine = {m: bool(agent_pooled[m] >= 0.99 and theme_pooled[m] >= 0.99)
                   for m in ("raw", "gradedtie", "gainnorm", "calibrated")}
        dagg = {
            "mode": "distinguishing_anti_cheat",
            "n_agent_favouring_ties_total": n_agent, "n_theme_favouring_ties_total": n_theme,
            "agent_favouring_gives_AGENT_pooled": agent_pooled,   # genuine fix ~1.0; THEME prior ~0.0 (the key row)
            "theme_favouring_gives_THEME_pooled": theme_pooled,   # both genuine + prior ~1.0
            "answer_independent_genuine": genuine,                # True only for a fix that follows the DRIVE, not a prior
        }
        print(f"\n[resv-sweep DISTINGUISH] POOLED (n_agent_ties={n_agent}, n_theme_ties={n_theme}):\n"
              f"  AGENT-favouring tie -> gives AGENT (the load-bearing test; a THEME PRIOR fails this): {agent_pooled}\n"
              f"  THEME-favouring tie -> gives THEME (both genuine + prior pass this): {theme_pooled}\n"
              f"  ANSWER-INDEPENDENT (genuine, follows the DRIVE not a prior): {genuine}", flush=True)
        if args.json:
            os.makedirs(os.path.dirname(args.json), exist_ok=True)
            with open(args.json, "w") as fh:
                json.dump({"rows": drows, "agg": dagg}, fh, indent=2, default=str)
            print(f"[resv-sweep DISTINGUISH] wrote {args.json}", flush=True)
        return
    read_desc = ("ANALYTIC 3-way RIDGE (LINEAR 'is objrel present?')" if args.read == "ridge"
                 else "analytic-Dale GRADED SPIKING read (the actual residual; op-point swept)")
    print(f"[resv-sweep] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])} | read={args.read}: "
          f"{read_desc} of the SPIKING reservoir feature, held-out test (distinct rng, no leakage). "
          f"RESERVOIR: in_scale={args.in_scale} rec_scale={args.rec_scale} t_step={args.t_step} "
          f"(baseline in_scale={C.RES_IN_SCALE} rec_scale=1.0 t_step={C.RES_T_STEP}). "
          f"READ-OUT OP-POINT: read_t={args.read_t if args.read_t is not None else D.READ_T} "
          f"read_in_scale={args.read_in_scale if args.read_in_scale is not None else D.IN_SCALE} "
          f"read_thresh={args.read_thresh if args.read_thresh is not None else D.THRESH} "
          f"(baseline READ_T={D.READ_T} IN_SCALE={D.IN_SCALE} THRESH={D.THRESH}). "
          f"DIAGNOSTIC, NOT a GO/surpass claim. NO read-out training/plasticity, NO sim/ edit.", flush=True)

    rows = []
    for s in args.seeds:
        d = run_seed(s, corpus, args.in_scale, args.rec_scale, args.t_step,
                     args.read, args.read_t, args.read_in_scale, args.read_thresh)
        rows.append(d)
        _print_seed(s, d)

    n_present = sum(r["objrel_present"] for r in rows)
    canon_all_ok = all(r["canonical_ok"] for r in rows)
    mean_objr_s0 = round(float(np.mean([r["objrel_slot0_THEME"] for r in rows])), 3)
    mean_canon = round(float(np.mean([r["canonical_acc"] for r in rows])), 3)
    failing = [r["seed"] for r in rows if not r["objrel_present"]]

    agg = {
        "read": args.read,
        "reservoir_config": {"in_scale": float(args.in_scale), "rec_scale": float(args.rec_scale),
                             "t_step": int(args.t_step)},
        "readout_op": {"read_t": (int(args.read_t) if args.read_t is not None else int(D.READ_T)),
                       "read_in_scale": (float(args.read_in_scale) if args.read_in_scale is not None else float(D.IN_SCALE)),
                       "read_thresh": (float(args.read_thresh) if args.read_thresh is not None else float(D.THRESH))},
        "baseline_reservoir": {"in_scale": float(C.RES_IN_SCALE), "rec_scale": 1.0, "t_step": int(C.RES_T_STEP)},
        "baseline_readout_op": {"read_t": int(D.READ_T), "read_in_scale": float(D.IN_SCALE), "read_thresh": float(D.THRESH)},
        "n_seeds": len(rows), "n_objrel_present": int(n_present),
        "objrel_present_seeds": [r["seed"] for r in rows if r["objrel_present"]],
        "objrel_failing_seeds": failing,
        "canonical_all_ok": bool(canon_all_ok),
        "mean_objrel_slot0_THEME": mean_objr_s0,
        "mean_canonical_acc": mean_canon,
        "ridge_lambda": RIDGE_LAM, "n_train": N_TRAIN, "n_test": N_TEST,
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    op_str = (f"in{args.in_scale:.0f} rec{args.rec_scale:.2f} T{args.t_step}" if args.read == "ridge"
              else f"readT{agg['readout_op']['read_t']} readIn{agg['readout_op']['read_in_scale']:.2f} "
                   f"readThr{agg['readout_op']['read_thresh']:.2f}")
    print(f"\n[resv-sweep] DIAGNOSTIC SUMMARY (read={args.read} {op_str}): objrel-slot0 >= 0.90 on "
          f"{n_present}/{len(rows)} seeds (mean objrel-slot0 {mean_objr_s0:.2f}); canonical mean {mean_canon:.2f} "
          f"(all-ok {canon_all_ok}). objrel-FAILING seeds: {failing}", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[resv-sweep] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
