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

    if read == "spiking":
        # the analytic-Dale GRADED spiking read at the OVERRIDDEN op-point (the actual residual). NOTE: D._score /
        # D._cache_slot_features drive the reservoir feature via PR._feature (RES_T_STEP=12 default) -- the reservoir
        # feature statistics are UNCHANGED by --t-step for the spiking path (the spiking-read op-point is the target),
        # so --read spiking is swept over the READ-OUT op-point, not the feature window.
        with _override_readout_op(read_t, read_in_scale, read_thresh):
            ros = D._analytic_dale_readout(slot_train, feat_dim, seed)
            canon_acc, canon_s0, canon_ps, canon_pt = _score_spiking(ros, res, enc, canon)
            objr_acc, objr_s0, objr_ps, objr_pt = _score_spiking(ros, res, enc, objr)
    else:
        Wk = _fit_slot_ridges(slot_train)
        canon_acc, canon_s0, canon_ps, canon_pt = _score_ridge(Wk, res, enc, canon, t_step)
        objr_acc, objr_s0, objr_ps, objr_pt = _score_ridge(Wk, res, enc, objr, t_step)

    elapsed = round(time.time() - t0, 1)
    key = "spiking" if read == "spiking" else "ridge"
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
    key = "spiking" if d["read"] == "spiking" else "ridge"
    ro = d[f"{key}_objrel"]; rc = d[f"{key}_canonical"]
    tag = "SPIKING" if key == "spiking" else "RIDGE"
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
    ap.add_argument("--read", choices=["ridge", "spiking"], default="ridge",
                    help="ridge = the LINEAR 'is objrel present in the feature?' read (1.00 all 10 seeds -> reservoir "
                         "is fine); spiking = the analytic-Dale GRADED spike-count read (the actual residual; 103=0.00 "
                         "104=0.33 at the baseline op-point).")
    ap.add_argument("--read-t", type=int, default=None,
                    help="[--read spiking] overrides D.READ_T (LIF spike-count integration steps; 25 baseline). "
                         "HIGHER = more spike-count resolution -- the thin-margin hypothesis.")
    ap.add_argument("--read-in-scale", type=float, default=None,
                    help="[--read spiking] overrides D.IN_SCALE (feature->input-current gain; 0.5 baseline graded).")
    ap.add_argument("--read-thresh", type=float, default=None,
                    help="[--read spiking] overrides D.THRESH (LIF spike threshold; 1.0 baseline).")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_rungB1c_objrel_resv_sweep.json")
    args = ap.parse_args()

    t0 = time.time()
    corpus = C.setup_corpus(seed=42)
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
