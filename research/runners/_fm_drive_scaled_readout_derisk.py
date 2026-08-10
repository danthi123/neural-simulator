"""DRIVE-SCALED (homeostatic intrinsic-excitability SET-POINT) world-model neural read-out WINNER -- surpass the
UNDER-DRIVE wall that the population-scaled divisive normalization could NOT (it is the wrong sign).

THE RESIDUAL, PRECISELY NAMED (banked; `_fm_neural_wta_readout_derisk` + `_fm_scaled_norm_readout_derisk` + the
capacity sweep `fm_readout_capacity_np{250,500,1000}.json`). The forward-model reservoir (OnBridge LSM) encodes (s,a);
a delta-trained TWO-PATHWAY (W+ excitatory / W- feedforward-inhibitory) read-out decodes s'. The host-decodable
EVIDENCE CEILING (ridge / two-pathway heldout) RISES with reservoir capacity (ridge 0.77 -> 0.96 at n_pool 250 ->
1000). But the NEURAL spiking read-out does the OPPOSITE: the FIXED lateral-inhibition WTA COLLAPSES (0.40 -> 0.17
mean) as the reservoir grows, EVEN AS the ceiling improves. The named cause is UNDER-DRIVE, not under-competition: the
ensemble mean firing DROPS with reservoir size (`ens_mean_spk` 0.0102 -> 0.0069) -- the ensembles are STARVED at scale,
so the differential reservoir drive lands in the FLAT low part of the ensemble F-I curve where a fixed input margin
produces almost no firing-rate difference and the winner reads as chance. Population-scaled divisive normalization
(`_fm_scaled_norm_readout_derisk`) RESCUES only partially (0.32 @1000) and at n_pool=250 HURTS (0.56 -> 0.24;
ens_mean 0.0102 -> 0.0050) because ADDING inhibition is the WRONG SIGN for an UNDER-driven margin -- it cannot recover
drive it can only remove it.

THE HYPOTHESIS (this runner -- the RIGHT sign). The companion process biology runs alongside the competition, which we
replaced with a CONSTANT tonic floor, is a GAIN / EXCITABILITY SET-POINT: HOMEOSTATIC INTRINSIC PLASTICITY (Desai,
Rutherford & Turrigiano 1999; LeMasson/Marder/Abbott 1993 -- a neuron regulates its intrinsic conductances to hold a
TARGET average firing rate). Give each read-out ensemble a per-ensemble intrinsic-excitability controller that reads
its OWN average spiking (from `cp_firing_states`, over TRAIN) and drives its tonic excitability UP/down until its mean
firing hits a TARGET set-point that is INDEPENDENT of n_pool. As the reservoir grows and the raw drive weakens, the
controller raises the excitability MORE, so the ensembles sit at the SAME responsive operating point at every n_pool
instead of starving -- the SAME differential reservoir evidence now lands on the STEEP part of the F-I curve and the
winner reads the capacity-improved evidence (ceiling 0.96 @1000) instead of collapsing. This is size-invariant BY
CONSTRUCTION (the set-point is a fixed target rate; the controller supplies whatever drive is needed to reach it).

WHY THIS IS NEURAL, NOT A HOST LOGIT RESCALE (the exact anti-cheat). The controller's OUTPUT is a per-ensemble TONIC
EXCITATORY CURRENT injected into the ensemble neurons via `cp_external_input_current` (a real intrinsic-excitability
bias -- exactly what intrinsic homeostatic plasticity modulates, the neuron's Na/K conductance set-point). The neurons
then SPIKE under the fixed lateral-inhibition competition, and the winner is the ensemble that FIRES MOST, read from
`cp_firing_states` (the accepted neural-WTA read). There is NO `np.divide`/rescale of the read-out LOGITS -- there are
no logits. The per-ensemble floor is a SINGLE SCALAR set over ALL calibration pairs (identical for every test (s,a)),
so it carries ZERO per-(s,a) information -- the per-trial discrimination MUST come from the reservoir through the W+
read-out synapses, PROVEN by the wp-lesion collapse below. The content path is the VERBATIM, imported `_neural_predict`
from `_fm_neural_wta_readout_derisk` (grep-clean of map-matmul / logit-argmax, verified via inspect.getsource). This is
the ACCEPTED "on-substrate homeostasis = read-out floor calibration" pattern of the two GO neural-WTA precedents, with
the biological addition that the floor controller now targets a RATE SET-POINT (Desai/Turrigiano) rather than merely
equalizing around whatever starved level the raw drive produced.

THE CLEAN CONTRAST. drive_scaled and fixed_wta share the IDENTICAL wiring (fixed lateral-inhibition WTA, norm pool
OFF); they differ ONLY in the floor controller -- fixed_wta equalizes around the current mean (`_calibrate_floors`),
drive_scaled drives the mean to the target set-point (`_calibrate_floors_setpoint`). So the ONLY changed variable is
the homeostatic set-point term. `ens_mean_spk` with vs without drive-scaling directly reports whether the mechanism
recovered the starved firing toward the target.

TEETH (all in ONE process, on the SAME substrate/feature at each n_pool; nothing imported as a number).
  (i)   the EVIDENCE CEILING (ridge / two-pathway heldout) MEASURED IN-RUN at each n_pool.
  (ii)  the FIXED-WTA baseline (prior mechanism, `wta_ie` TRAIN-swept, equalize-only floors) MEASURED IN-RUN -- the
        apples-to-apples 0.40/0.17 contrast on the identical substrate/feature.
  (iii) the SCALED-NORM partial (0.32 @1000, `norm_gain` TRAIN-swept) MEASURED IN-RUN (cheap: same build) -- the
        wrong-sign mechanism drive-scaling must beat.
  (iv)  drive_scaled: fixed-WTA competition + the homeostatic RATE SET-POINT floors (`target_rate` TRAIN-swept). The
        KEY test: does it lift n_pool=1000 toward the ceiling WITHOUT losing at n_pool=250, and does `ens_mean_spk`
        recover from 0.0069 toward the target?
  (v)   content path VERBATIM-imported + grep-clean (winner from cp_firing_states; no map-matmul / logit-argmax).
  (vi)  LOAD-BEARING drive: the wp read-out lesion collapses (proves the floors carry no content -- discrimination is
        the reservoir's); reservoir-silence collapses; untrained random weights -> chance; matched-sham decoy UNCHANGED.
  (vii) seeded BYTE-IDENTICAL substrate (cfg.seed); backend RECORDED (assert_backend).

GO bar (per seed, at n_pool=1000 -- the KEY test): preconditions [reservoir+ensembles active, seeded, content clean,
sham UNCHANGED |d|<=0.08, backend=numpy, two-path==ridge] AND drive_scaled_heldout - fixed_wta_heldout >= 0.20
(drive-scaling load-bearing at scale) AND drive_scaled_heldout - max(chance,prior) >= 0.20 AND drive_scaled_heldout >=
twopath_rate_heldout - 0.20 (tracks the ceiling) AND drive_scaled_ens_mean > fixed_wta_ens_mean (drive actually rose)
AND wp-lesion + silence collapse >= 0.20 AND untrained <= chance+0.08.  Run ALSO at n_pool=250: the no-loss guard is
drive_scaled_heldout >= fixed_wta_heldout - 0.10.
HONEST NEGATIVE WITH TEETH otherwise -- if driving the ensembles to the target rate does NOT lift the read-out (the
margin is a CODE limit -- the spike-count representational margin -- not a DRIVE limit), report the measured reason
(e.g. `ens_mean_spk` reached the target but held-out did not rise -> the differential is not recoverable by operating
point, it is a representational-capacity ceiling). A first-class deliverable. Do NOT force GO.

SMOKE (single seed, numpy, reduced grids):
  SIM_BACKEND=numpy python -u -m research.runners._fm_drive_scaled_readout_derisk --seeds 42 --n-pool 250 --smoke \
      --out research/findings/raw/_fm_drive_scaled_readout_smoke.json
KEY TEST (per-seed parallel, each n_pool):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -u -m research.runners._fm_drive_scaled_readout_derisk \
      --seeds $s --n-pool 1000 --out research/findings/raw/_fm_drive_scaled_np1000_s$s.json & done; wait
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -u -m research.runners._fm_drive_scaled_readout_derisk \
      --seeds $s --n-pool 250  --out research/findings/raw/_fm_drive_scaled_np250_s$s.json & done; wait
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── reuse-by-import: the world + local read-out rule (forward_model_reservoir) ───────────────────────────────────
from research.runners._forward_model_reservoir_derisk import (   # noqa: E402
    _ACTIONS, _all_pairs, _step, _encode_seq, _target, _train_delta,
)
# ── reuse-by-import: the wash-out snapshot/restore ──────────────────────────────────────────────────────────────
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (   # noqa: E402
    _snapshot_state, _restore_state,
)
# ── reuse-by-import: the VETTED neural-WTA read-out (content path + equalize-only floors + accs + constants) ─────
from research.runners._fm_neural_wta_readout_derisk import (   # noqa: E402
    _neural_predict, _calibrate_floors, _accs, _rate_twopath_acc,
    ENS_P, FFI_P, FLOOR_BASE, READ_T_STEP, READ_REPLAY,
    SYN_SCALE_GRID, FFI_IN_GRID, FFI_OUT_GRID, WTA_IE_GRID,
)
# ── reuse-by-import: the scaled-norm build (has BOTH wta + norm pools) + its wiring + the fixed/norm sweep ───────
from research.runners._fm_scaled_norm_readout_derisk import (   # noqa: E402
    _build_bridge, _wire, _norm_pool_activity, _sweep_and_score, NORM_GAIN_GRID, NORM_N,
)
# ── reuse-by-import: standardization fold + host-ridge reference + covered split + reservoir feature ─────────────
from research.runners._fm_spiking_synaptic_readout_derisk import (   # noqa: E402
    _fold_standardization, _host_decode, _covered_split, _reservoir_feature,
)
from sim.backend import to_host                  # noqa: E402
from tools.lab import lever, assert_backend      # noqa: E402
from tools.verdict import Verdict                # noqa: E402

# ── the HOMEOSTATIC INTRINSIC-EXCITABILITY SET-POINT controller (Desai/Turrigiano; replaces equalize-only floors) ─
#   The controller reads each ensemble's OWN mean firing rate (per-neuron-per-step, from cp_firing_states) and drives
#   its per-ensemble tonic excitability UP/down until the rate hits TARGET_RATE -- a set-point INDEPENDENT of n_pool.
TARGET_RATE_GRID = (0.010, 0.014, 0.020)   # per-neuron-per-step ensemble firing set-point, TRAIN-swept (the operating
#                                            point; the working n_pool=250 fixed-WTA rate is ~0.0102)
SETPOINT_KP = 1400.0        # controller gain: pA of tonic current per unit (spikes/neuron/step) rate error
SETPOINT_ITERS = 8          # homeostatic iterations to converge the excitability to the set-point
FLOOR_MIN = 0.0
FLOOR_MAX = 320.0           # cap the intrinsic drive (prevents runaway / F-I saturation)


def _read_steps(G):
    """Number of sim steps `_neural_predict` accumulates per (s,a) read -- READ_REPLAY x len(seq) x READ_T_STEP. Used
    to convert the accumulated per-ensemble spike COUNT (dbg["x_spk"]) into a per-neuron-per-step RATE. G-fixed and
    n_pool-INDEPENDENT (so a fixed rate set-point is a fixed target at every n_pool)."""
    seq_len = len(_encode_seq((0, 0), _ACTIONS[0], G))
    return int(READ_REPLAY) * int(seq_len) * int(READ_T_STEP)


def _calibrate_floors_setpoint(b, idx, snap, G, cal_pairs, target_rate,
                               kp=SETPOINT_KP, iters=SETPOINT_ITERS):
    """HOMEOSTATIC INTRINSIC-EXCITABILITY SET-POINT (Desai, Rutherford & Turrigiano 1999). Per-ensemble controller:
    read the ensemble's own average firing rate over the TRAIN calibration pairs (from `cp_firing_states` via the
    imported `_neural_predict`) and adjust its tonic excitability toward TARGET_RATE. Unlike `_calibrate_floors`
    (which subtracts the deviation from the CURRENT mean -- it equalizes around whatever starved level the raw drive
    produced), this DRIVES the mean to the target, so at large n_pool where the raw drive is weak the controller
    supplies MORE excitability and the ensembles reach the same responsive operating point instead of starving. The
    floor is a per-ensemble tonic CURRENT (intrinsic-excitability bias), NOT a rescale of the winner logits."""
    steps = _read_steps(G)
    denom = float(steps * int(idx["ens_p"]))
    fx = np.full(G, FLOOR_BASE, np.float64); fy = np.full(G, FLOOR_BASE, np.float64)
    for _it in range(int(iters)):
        xs = np.zeros(G); ys = np.zeros(G)
        for (s, a) in cal_pairs:
            _p, dbg = _neural_predict(b, idx, snap, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
            xs += np.asarray(dbg["x_spk"]); ys += np.asarray(dbg["y_spk"])
        xs /= max(1, len(cal_pairs)); ys /= max(1, len(cal_pairs))
        rate_x = xs / denom; rate_y = ys / denom          # per-neuron-per-step ensemble firing rate
        fx = np.clip(fx + kp * (float(target_rate) - rate_x), FLOOR_MIN, FLOOR_MAX)
        fy = np.clip(fy + kp * (float(target_rate) - rate_y), FLOOR_MIN, FLOOR_MAX)
    return fx, fy


def _sweep_and_score_drive(b, rm, idx, seed, snap_feat, Wp_x, Wm_x, Wp_y, Wm_y, G,
                           train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                           base_wta, target_grid, cal_pairs, train_probe, tr_probe_sp):
    """Wire the FIXED lateral-inhibition WTA at `base_wta` (norm pool OFF -- IDENTICAL wiring to the fixed_wta arm),
    then sweep the homeostatic SET-POINT `target_rate` over `target_grid`, calibrating floors with
    `_calibrate_floors_setpoint` and scoring the neural WTA on a TRAIN probe; select the best target, re-calibrate,
    and score TRAIN+HELD. The ONLY difference from the fixed_wta arm is the floor controller. Returns
    (best_target, edges, snap_w, floors, train_acc, held_acc, ens_mean)."""
    edges = _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in, ffi_out,
                  wta_ie=base_wta, norm_gain=0.0)
    snap_w = _snapshot_state(b)
    best = None
    for tr in target_grid:
        fx, fy = _calibrate_floors_setpoint(b, idx, snap_w, G, cal_pairs, tr)
        hit = 0
        for (s, a), sp in zip(train_probe, tr_probe_sp):
            pred, _dbg = _neural_predict(b, idx, snap_w, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
            hit += int(pred == sp)
        acc = hit / max(1, len(train_probe))
        if best is None or acc > best[0]:
            best = (acc, tr)
    best_target = best[1]
    floors_x, floors_y = _calibrate_floors_setpoint(b, idx, snap_w, G, cal_pairs, best_target)
    tr_acc, _, _, _, _ = _accs(b, idx, snap_w, train, tr_sp, G, floors_x=floors_x, floors_y=floors_y)
    ho_acc, _, _, _, ens_mean = _accs(b, idx, snap_w, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    return best_target, edges, snap_w, (floors_x, floors_y), tr_acc, ho_acc, ens_mean


def _content_path_clean():
    """Winner read from the VETTED, IMPORTED `_neural_predict` (from `_fm_neural_wta_readout_derisk`): grep-clean via
    inspect.getsource that its code (docstring stripped) reads the winner from `cp_firing_states` (argmax over ensemble
    SPIKE-COUNTS) with NO host map-matmul / logit-argmax, and that THIS runner's content path IS that imported
    function (not a fork). The set-point controller only injects tonic CURRENT; it never touches the winner read."""
    code = inspect.getsource(_neural_predict)
    q = code.find('"""'); q2 = code.find('"""', q + 3)
    body = (code[:q] + code[q2 + 3:]) if (q >= 0 and q2 > q) else code
    forbidden = ("@ Ws", "@ W_eff", "feat @", "W_eff @", "Wp_x @", "@ f", "argmax(dx", "argmax(pred")
    reads_spikes = ("np.argmax(x_spk)" in body) and ("np.argmax(y_spk)" in body) and ("cp_firing_states" in body)
    imported_ok = (_neural_predict.__module__ == "research.runners._fm_neural_wta_readout_derisk")
    return bool(imported_ok and reads_spikes and not any(f in body for f in forbidden))


def _drive_is_neural():
    """ANTI-CHEAT: the drive/gain mechanism is a per-ensemble tonic CURRENT injected into `cp_external_input_current`
    (intrinsic-excitability bias), driven by a controller that reads `cp_firing_states` -- NOT a host rescale of the
    read-out logits. Verify from source that the controller (i) reads dbg["x_spk"]/["y_spk"] (which come from the
    imported neural read's cp_firing_states counts) and (ii) writes only to the per-ensemble tonic `fx`/`fy` (which
    `_neural_predict` injects as `cp_external_input_current`), with no divide/rescale on a logit vector."""
    full = inspect.getsource(_calibrate_floors_setpoint)
    q = full.find('"""'); q2 = full.find('"""', q + 3)
    ctrl = (full[:q] + full[q2 + 3:]) if (q >= 0 and q2 > q) else full   # strip docstring (it names 'logit')
    reads_spk = ('dbg["x_spk"]' in ctrl) and ('dbg["y_spk"]' in ctrl)
    writes_floor = ("fx = np.clip(fx + kp" in ctrl) and ("fy = np.clip(fy + kp" in ctrl)
    # _neural_predict injects fx/fy as tonic external current (not a logit rescale); confirm the injection site exists
    npsrc = inspect.getsource(_neural_predict)
    injects_current = ("cp_external_input_current[x_dev[r]] = np.float32(fx[r])" in npsrc) and \
                      ("cp_external_input_current[y_dev[r]] = np.float32(fy[r])" in npsrc)
    no_logit_rescale = ("np.divide" not in ctrl) and ("logit" not in ctrl)
    return bool(reads_spk and writes_floor and injects_current and no_logit_rescale)


def _derisk_one(seed, G=5, n_pool=250, heldout_frac=0.25, smoke=False, with_scaled_norm=True):
    t0 = time.time()
    backend = assert_backend("numpy", note=f"drive-scaled read-out seed={seed} n_pool={n_pool}")
    pairs = _all_pairs(G)
    out_dim = 2 * G

    b, rm, idx, cfg = _build_bridge(seed, G, n_pool)
    b2, _, _, _ = _build_bridge(seed, G, n_pool)

    def _thash(bb):
        arr = getattr(bb, "cp_neuron_firing_thresholds", None)
        return None if arr is None else hashlib.sha1(np.asarray(to_host(arr)).astype(np.float64).tobytes()).hexdigest()
    seeded = bool(_thash(b) is not None and _thash(b) == _thash(b2))
    del b2

    # reservoir-recurrence-only wiring for the FEATURE extraction (the read-out synapses come after training)
    from research.runners._fm_scaled_norm_readout_derisk import _reservoir_internal
    union0 = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union0["reservoir_internal"] = ri
    inh0 = []
    for region in rm.regions():
        inh0.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union0, output_inhibitory_indices=inh0 or None)
    snap = _snapshot_state(b)

    feats = {}
    spike_acc = []
    for (s, a) in pairs:
        f = _reservoir_feature(b, idx, snap, _encode_seq(s, a, G))
        feats[(s, a)] = f
        spike_acc.append(float(f.mean()))
    mean_spikes = float(np.mean(spike_acc))

    train, held = _covered_split(pairs, G, seed, heldout_frac)
    Xtr_raw = np.stack([feats[p] for p in train])
    mu = Xtr_raw.mean(0); sd = Xtr_raw.std(0) + 1e-6
    Xtr = np.stack([(feats[p] - mu) / sd for p in train])
    Ttr = np.stack([_target(_step(s, a, G), G) for (s, a) in train])
    tr_sp = [_step(s, a, G) for (s, a) in train]
    ho_sp = [_step(s, a, G) for (s, a) in held]

    W, bvec = _train_delta(Xtr, Ttr, out_dim, seed)
    W_eff, b_eff = _fold_standardization(W, bvec, mu, sd)
    ridge_train = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(train, tr_sp)]))
    ridge_held = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(held, ho_sp)]))

    Wp = np.clip(W_eff, 0.0, None); Wm = np.clip(-W_eff, 0.0, None)
    Wp_x, Wm_x = Wp[:G, :], Wm[:G, :]
    Wp_y, Wm_y = Wp[G:2 * G, :], Wm[G:2 * G, :]

    twopath_rate_train = _rate_twopath_acc(Wp_x, Wm_x, Wp_y, Wm_y, b_eff, feats, train, tr_sp, G)
    twopath_rate_held = _rate_twopath_acc(Wp_x, Wm_x, Wp_y, Wm_y, b_eff, feats, held, ho_sp, G)

    # two-pathway gains fixed at the baseline-selected values (not what we test)
    syn_scale = SYN_SCALE_GRID[-1]        # 6.0
    ffi_out = FFI_OUT_GRID[0]             # 4.0
    ffi_in = FFI_IN_GRID[0] if smoke else (FFI_IN_GRID[-1] if n_pool >= 800 else FFI_IN_GRID[0])

    n_sweep = 16 if not smoke else 12
    train_probe = train[:n_sweep]
    tr_probe_sp = [_step(s, a, G) for (s, a) in train_probe]
    cal_pairs = train[:n_sweep]

    fixed_grid = WTA_IE_GRID if not smoke else (40.0,)
    norm_grid = NORM_GAIN_GRID if not smoke else (1.0,)
    target_grid = TARGET_RATE_GRID if not smoke else (0.014,)

    # ============ BASELINE 1: the FIXED lateral-inhibition WTA (prior mechanism), MEASURED IN-RUN ============
    (wta_ie_sel, edges_fx, snap_fx, floors_fx, fixed_train, fixed_held,
     fixed_ens_mean) = _sweep_and_score(b, rm, idx, seed, snap, Wp_x, Wm_x, Wp_y, Wm_y, G,
                                        train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                                        "fixed", fixed_grid, cal_pairs, train_probe, tr_probe_sp)

    # ============ BASELINE 2: the SCALED-NORM partial (wrong-sign mechanism, 0.32 @1000), MEASURED IN-RUN ============
    if with_scaled_norm:
        (norm_gain_sel, edges_sn, snap_sn, floors_sn, scaled_train, scaled_held,
         scaled_ens_mean) = _sweep_and_score(b, rm, idx, seed, snap, Wp_x, Wm_x, Wp_y, Wm_y, G,
                                             train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                                             "norm", norm_grid, cal_pairs, train_probe, tr_probe_sp,
                                             base_wta=wta_ie_sel)
        norm_pool_spk = _norm_pool_activity(b, idx, snap_sn, held, G, floors_x=floors_sn[0], floors_y=floors_sn[1])
    else:
        norm_gain_sel, scaled_train, scaled_held, scaled_ens_mean, norm_pool_spk = (-1.0, -1.0, -1.0, -1.0, -1.0)

    # ============ THE MECHANISM: FIXED WTA competition + HOMEOSTATIC RATE SET-POINT floors, MEASURED IN-RUN ============
    (target_sel, edges, snap_w, (floors_x, floors_y), drive_train, drive_held,
     drive_ens_mean) = _sweep_and_score_drive(b, rm, idx, seed, snap, Wp_x, Wm_x, Wp_y, Wm_y, G,
                                              train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                                              wta_ie_sel, target_grid, cal_pairs, train_probe, tr_probe_sp)

    # ---- LOAD-BEARING drive: the floors carry NO per-(s,a) content -- lesion the W+ read-out synapses -> collapse ----
    pxe, qxe, wxe = edges["wp_x"]; pye, qye, wye = edges["wp_y"]
    b.set_pathway_weights("les_wp_x", pxe, qxe, np.zeros(len(pxe), np.float32), add_missing=False)
    b.set_pathway_weights("les_wp_y", pye, qye, np.zeros(len(pye), np.float32), add_missing=False)
    snap_lw = _snapshot_state(b)
    # recalibrate the set-point floors WITHOUT the read-out (fair: the ensembles still get their homeostatic drive to
    # the target -- if the floors themselves resolved the answer, this would NOT collapse)
    fxl, fyl = _calibrate_floors_setpoint(b, idx, snap_lw, G, cal_pairs, target_sel)
    lesion_wp_held, _, _, _, _ = _accs(b, idx, snap_lw, held, ho_sp, G, floors_x=fxl, floors_y=fyl)
    b.set_pathway_weights("res_wp_x", pxe, qxe, wxe, add_missing=False)
    b.set_pathway_weights("res_wp_y", pye, qye, wye, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- LESION 2: silence the reservoir input -> collapse ----
    silence_held, _, _, _, _ = _accs(b, idx, snap_w, held, ho_sp, G, silence=True,
                                     floors_x=floors_x, floors_y=floors_y)

    # ---- MATCHED SHAM: count-matched lesion of the OFF-DECODE decoy read-out -> UNCHANGED ----
    pd, qd, wd = edges["wp_dec"]
    b.set_pathway_weights("sham_dec", pd, qd, np.zeros(len(pd), np.float32), add_missing=False)
    snap_sham = _snapshot_state(b)
    sham_held, _, _, _, _ = _accs(b, idx, snap_sham, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    b.set_pathway_weights("res_dec", pd, qd, wd, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- UNTRAINED control: random non-negative weights of matched magnitude + the SAME set-point drive -> chance ----
    rng = np.random.default_rng(seed * 4242 + 1)
    Wp_x_r = rng.random(Wp_x.shape) * float(Wp.mean()); Wm_x_r = rng.random(Wm_x.shape) * float(Wm.mean())
    Wp_y_r = rng.random(Wp_y.shape) * float(Wp.mean()); Wm_y_r = rng.random(Wm_y.shape) * float(Wm.mean())
    _wire(b, rm, idx, seed, Wp_x_r, Wm_x_r, Wp_y_r, Wm_y_r, syn_scale, ffi_in, ffi_out,
          wta_ie=wta_ie_sel, norm_gain=0.0)
    snap_ut = _snapshot_state(b)
    fxr, fyr = _calibrate_floors_setpoint(b, idx, snap_ut, G, cal_pairs, target_sel)
    untrained_held, _, _, _, _ = _accs(b, idx, snap_ut, held, ho_sp, G, floors_x=fxr, floors_y=fyr)

    lever("drive_scaled_vs_fixed_wta", before=round(fixed_held, 4), after=round(drive_held, 4), required=False)
    if with_scaled_norm:
        lever("drive_scaled_vs_scaled_norm", before=round(scaled_held, 4), after=round(drive_held, 4), required=False)
    lever("ens_mean_drive_vs_fixed(recovery)", before=round(fixed_ens_mean, 6), after=round(drive_ens_mean, 6),
          required=False)
    lever("wp_readout_lesion", before=round(drive_held, 4), after=round(lesion_wp_held, 4), required=False)
    lever("reservoir_silence_lesion", before=round(drive_held, 4), after=round(silence_held, 4), required=False)
    lever("matched_sham_decoy", before=round(drive_held, 4), after=round(sham_held, 4), required=False)

    from collections import Counter
    tr_counter = Counter(tr_sp)
    prior_sp = tr_counter.most_common(1)[0][0] if tr_counter else (0, 0)
    prior_held = float(np.mean([prior_sp == sp for sp in ho_sp]))
    chance = 1.0 / (G * G)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), G=int(G), n_pool=int(n_pool), ens_p=int(ENS_P), ffi_p=int(FFI_P), norm_n=int(NORM_N),
        backend=backend, heldout_n=len(held), train_n=len(train), chance=float(chance), chance_per_block=float(1.0 / G),
        mean_reservoir_spikes_feature=mean_spikes,
        syn_scale=float(syn_scale), ffi_in=float(ffi_in), ffi_out=float(ffi_out),
        wta_ie_selected=float(wta_ie_sel), norm_gain_selected=float(norm_gain_sel), target_rate_selected=float(target_sel),
        setpoint_kp=float(SETPOINT_KP), setpoint_iters=int(SETPOINT_ITERS), floor_max=float(FLOOR_MAX),
        ridge_train_acc=ridge_train, ridge_heldout_acc=ridge_held,
        twopath_rate_train=twopath_rate_train, twopath_rate_heldout=twopath_rate_held,
        # the three mechanisms, MEASURED IN-RUN on the SAME substrate/feature
        fixed_wta_train=float(fixed_train), fixed_wta_heldout=float(fixed_held), fixed_wta_ens_mean=float(fixed_ens_mean),
        scaled_norm_train=float(scaled_train), scaled_norm_heldout=float(scaled_held),
        scaled_norm_ens_mean=float(scaled_ens_mean), norm_pool_mean_spk=float(norm_pool_spk),
        drive_scaled_train=float(drive_train), drive_scaled_heldout=float(drive_held),
        drive_scaled_ens_mean=float(drive_ens_mean),
        # anti-cheats + lesions
        lesion_wp_heldout=float(lesion_wp_held), lesion_silence_heldout=float(silence_held),
        matched_sham_heldout=float(sham_held), untrained_control_heldout=float(untrained_held),
        prior_lookup_heldout=prior_held,
        content_path_clean=_content_path_clean(), drive_is_neural=_drive_is_neural(), seeded=seeded,
        elapsed_s=round(elapsed, 1),
    )


def _verdict(d):
    v = Verdict("fm drive-scaled world-model read-out (homeostatic intrinsic-excitability set-point; Desai/Turrigiano)",
                chance=d["chance"])
    v.disabled("STDP/Hebbian/STP/structural", "fixed reservoir + delta-trained two-pathway synapses + a fixed "
               "lateral-inhibition WTA; the DRIVE mechanism is a per-ensemble HOMEOSTATIC INTRINSIC-EXCITABILITY "
               "SET-POINT controller (reads cp_firing_states, injects tonic cp_external_input_current) -- the "
               "accepted on-substrate read-out floor calibration, targeting a RATE set-point")
    v.require("backend == numpy", d["backend"] == "numpy", expect=True)
    v.require("two-pathway rate == ridge (decomposition exact)",
              abs(d["twopath_rate_heldout"] - d["ridge_heldout_acc"]), expect=lambda x: x <= 1e-6)
    v.require("reservoir active (feature)", d["mean_reservoir_spikes_feature"], expect=lambda x: x > 0.0)
    v.require("ensembles active (drive read)", d["drive_scaled_ens_mean"], expect=lambda x: x > 0.0)
    v.require("seeded (byte-identical substrate)", d["seeded"], expect=True)
    v.require("content path clean (imported neural read; no host matmul/logit-argmax)",
              d["content_path_clean"], expect=True)
    v.require("drive mechanism NEURAL (set-point on cp_firing_states -> tonic current, not a logit rescale)",
              d["drive_is_neural"], expect=True)
    v.require("matched sham UNCHANGED (|d|<=0.08)", abs(d["drive_scaled_heldout"] - d["matched_sham_heldout"]),
              expect=lambda x: x <= 0.08)
    go = (d["drive_scaled_heldout"] - d["fixed_wta_heldout"] >= 0.20          # drive-scaling beats fixed WTA (the claim)
          and d["drive_scaled_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20
          and d["drive_scaled_heldout"] >= d["twopath_rate_heldout"] - 0.20   # tracks the ceiling
          and d["drive_scaled_ens_mean"] > d["fixed_wta_ens_mean"]            # drive actually rose (mechanism)
          and (d["drive_scaled_heldout"] - d["lesion_wp_heldout"]) >= 0.20
          and (d["drive_scaled_heldout"] - d["lesion_silence_heldout"]) >= 0.20
          and d["untrained_control_heldout"] <= d["chance"] + 0.08)
    dec = v.decide(go=go)
    dec["go_criteria"] = {
        "drive_beats_fixed(>=+0.20)": bool(d["drive_scaled_heldout"] - d["fixed_wta_heldout"] >= 0.20),
        "beats_chance/prior(>=+0.20)":
            bool(d["drive_scaled_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20),
        "tracks_ceiling(>=twopath-0.20)": bool(d["drive_scaled_heldout"] >= d["twopath_rate_heldout"] - 0.20),
        "drive_recovered(ens_mean up)": bool(d["drive_scaled_ens_mean"] > d["fixed_wta_ens_mean"]),
        "wp_lesion_collapses(>=0.20)": bool((d["drive_scaled_heldout"] - d["lesion_wp_heldout"]) >= 0.20),
        "silence_collapses(>=0.20)": bool((d["drive_scaled_heldout"] - d["lesion_silence_heldout"]) >= 0.20),
        "untrained<=chance+0.08": bool(d["untrained_control_heldout"] <= d["chance"] + 0.08),
    }
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--n-pool", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-scaled-norm", action="store_true", help="skip the in-run scaled-norm baseline arm")
    ap.add_argument("--out", type=str, default="research/findings/raw/_fm_drive_scaled_readout_smoke.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        try:
            d = _derisk_one(seed, G=args.G, n_pool=args.n_pool, smoke=args.smoke,
                            with_scaled_norm=not args.no_scaled_norm)
            dec = _verdict(d)
            d["verdict"] = dec
            results.append(d)
            print(f"\n=== seed {seed} (n_pool={args.n_pool}) ===")
            for k in ("mean_reservoir_spikes_feature", "wta_ie_selected", "norm_gain_selected", "target_rate_selected",
                      "ridge_heldout_acc", "twopath_rate_heldout", "fixed_wta_heldout", "scaled_norm_heldout",
                      "drive_scaled_heldout", "fixed_wta_ens_mean", "scaled_norm_ens_mean", "drive_scaled_ens_mean",
                      "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
                      "untrained_control_heldout", "prior_lookup_heldout", "chance",
                      "backend", "content_path_clean", "drive_is_neural", "seeded", "elapsed_s"):
                print(f"  {k:34s} {d[k]}")
            print(f"  VERDICT: {dec['status']}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append({"seed": int(seed), "error": repr(e)})

    payload = {"runner": "_fm_drive_scaled_readout_derisk", "argv": sys.argv, "seeds": list(args.seeds),
               "n_pool": args.n_pool, "results": results,
               "preconditions": (results[0].get("verdict", {}).get("preconditions") if results else None)}
    outp = _REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
