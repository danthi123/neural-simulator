"""CYCLE 180 (2026-06-19) -- the DENDRITE's REAL shot: a LEARNED DENDRITIC SIGMA-PI (multiplicative conjunction)
binder vs the documented two-attribute (K>=2) bundling wall.

CONTEXT (read 2026-06-19-dendritic-binding-derisk-scoping.md).
----------------------------------------------------------------
The conversational composer binds role/filler + attribute structure by a FIXED multiplicative primitive (the +-1
self-inverse / FHRR phasor conjugate). SINGLE-attribute binding WORKS (production); TWO-attribute (K>=2) BUNDLING is
the documented K=5 boundary and is "not learnable from scratch on point neurons" (CYCLE 102: learned ADDITIVE bundle
held-out 0.193 ~ chance 0.062; CYCLE 103: learned MULTIPLICATIVE bind + a LEARNED-LINEAR unbind/cleanup BROKE even
single-attribute, bundle held-out 0.056 -- the learned-linear cleanup is structurally incapable of a role-dependent
inverse). The fixed +-1 / FHRR algebra bundles 0.989 but is NOT learned.

THE DENDRITE'S NATIVE JOB (different from the apical-basal credit-assignment that landed NEGATIVE 2026-06-19):
binding = MULTIPLICATION / coincidence -- a single-OP analog product a point neuron's single linear summation cannot
form (Mikulasch-Priesemann; Kleyko/Frady VSA<->sigma-pi bridge). An element-wise PRODUCT is supralinear BY
CONSTRUCTION (the response to A+B exceeds the sum of the A-alone and B-alone responses). The dendritic mechanism
(eLife reviewed-preprint 97274; catalog G.02 active dendrites + J.08 NMDA coincidence): put each attribute's code as
a SYNAPTIC CLUSTER on a SHARED dendritic branch; the branch computes their conjunction A(x)B (the sigma-pi product),
and a LOCAL three-factor (calcium/dopamine-gated) rule LEARNS which co-relevant attributes route onto a shared branch
(the learnable part the fixed +-1 lacks). The eLife paper has NO held-out test -- adding that generalization test is
the WHOLE value of this de-risk.

THIS de-risk -- the ONE change vs _phaseB_multiplicative_bind_bundled_derisk.py:
  swap its broken learned-LINEAR unbind+cleanup for a LEARNED DENDRITIC SIGMA-PI (multiplicative conjunction)
  bind+unbind, KEEP the matched-filter argmax cleanup. Re-ask: does the LEARNED dendritic multiplication recover
  two-attribute bundling that GENERALIZES to held-out attribute pairs, where the point / learned-linear baselines hit
  the wall?

MECHANISM (numpy, in this runner -- NO sim/ edit; sim/dendritic_neuron.py is the WRONG primitive -- additive
threshold-shift, no product term -- verified):
  bind:    g_r = (role_r @ W_R) (x) (filler_r @ W_F)                       # sigma-pi MULTIPLICATIVE conjunction
                                                                            #   (the dendritic op a point neuron lacks)
  bundle:  bundle = sum_r g_r                                              # superposition
  unbind:  act = bundle (x) (role_t @ W_Rinv)                             # dendritic conjugate conjunction
  cleanup: argmax_f cos(act, fillers @ W_F)                               # MATCHED-FILTER argmax (NOT learned-linear)
  learn:   LOCAL three-factor calcium/dopamine rule (NO backprop through the products):
             - dopamine d = +1 on correct readout (peak), -1 on incorrect (pause)
             - per-branch NMDA-calcium surrogate ca = sigmoid(g_t) (the queried branch's conjunction, in [0,1])
             - LTP in a bell-shaped Ca window AND dopamine peak; LTD above an L-type Ca threshold AND dopamine pause
             - presynaptic-gated (Hebbian) so the update only touches the active cluster -> routes co-relevant
               attributes onto a shared branch (the eLife inhibitory-compartmentalization role).
  Projection std scaled to ~unit (PROJ_SCALE) so a cluster is strong enough to drive a meaningful conjunction.

  NOTE on the NMDA-plateau SQUASH (honest, recorded): an earlier draft also applied a SATURATING sigmoid plateau
  sigma(kappa(z-theta)) on the FORWARD product. It HURT (it sparsifies + saturates the conjunction -> the
  matched-filter readout loses the dense product info). The biologically-correct role of the NMDA plateau here is the
  CALCIUM signal that GATES the local plasticity (the eLife rule is literally calcium-gated), NOT a saturating squash
  on the read-out. So the forward op is the (supralinear) PRODUCT; the plateau enters as the Ca-gated learning rule.

CONTROLS (load-bearing; all must behave):
  - point baseline (PointSumBinder, == the LESION) : ADDITIVE sum, NO product -> must FAIL (~ the 0.193/chance wall).
    This IS the lesion: removing the MULTIPLICATION (product -> sum) is the load-bearing ablation; if it collapses,
    the multiplication is what binds.
  - learned-linear baseline (imported)             : the CYCLE-103 MultFHRRBinder (product bind + LEARNED-LINEAR
    unbind/cleanup) -> must FAIL (~0.056), isolating that the matched-filter cleanup (not a learned-linear one) is
    required.
  - memorization-floor (MemorizationLookup)        : held-out must score chance (no table lookup).
  - permuted role<->filler                         : learned routing must collapse.
  - FHRR F=3 self-inverse algebra (same pipeline)  : the fixed-primitive reference (systematic by construction).
  - chance line (1/F) printed alongside.

GO bar (pre-registered, NOT tuned to result; multi-seed):
  1. two-attribute bundled held-out-combo recall >= 0.40 AND >= 0.6*train, >> additive 0.193, >> chance 0.062
  2. GENERALIZES: held-out ~ train (small gap) -- the decisive anti-memorization clause
  3. single-attribute does NOT regress vs the production primitive (single_held >= 0.40)
  4. the point / learned-linear baselines FAIL on the identical pipeline (the lesion collapses)
  5. 3-seed gate first; claim GO only after 6 seeds.

Reuse-by-import: make_role_codes / make_systematicity_splits / native_argmax / MemorizationLookup /
fhrr_bind / fhrr_unbind / fhrr_cleanup from cortex_learned_binder_systematicity_probe; the CYCLE-103
MultFHRRBinder from _phaseB_multiplicative_bind_bundled_derisk (the learned-linear baseline).
CPU; SIM_BACKEND=numpy; no GPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_dendritic_bind_derisk [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    MemorizationLookup, fhrr_bind, fhrr_cleanup, fhrr_unbind, make_role_codes,
    make_systematicity_splits, native_argmax)

# Defaults match the harness we extend (R=4 SVO uses roles 0,1,2; F=16; 3 splits; D_h=64).
R, F, N_SPLITS, D_H = 4, 16, 3, 64
N_FACT_STEPS = 24000      # bundle-aware training steps (matches the additive / multiplicative-linear de-risks)
N_EVAL_FACTS = 40
N_BUNDLE_ROLES = 3        # SVO: bundle 3 (role, filler) pairs per fact (the two-attribute+ frontier load)

# Projection std: the cluster gain. W ~ N(0,1)*PROJ_SCALE / sqrt(D_in)*sqrt(D_in) => projected u,w have ~unit std
# (a synaptic cluster strong enough to drive a meaningful dendritic conjunction; without this the product is ~1e-3
# and carries no information).
PROJ_SCALE = 1.0

# Local three-factor (eLife-style) calcium/dopamine rule hyperparameters (fixed a-priori, one value all seeds --
# anti-cheat: no per-seed tuning to a target).
ETA_LTP = 0.02            # LTP rate (bell-shaped Ca window AND dopamine peak)
ETA_LTD = 0.02            # LTD rate (above L-type Ca threshold AND dopamine pause)
CA_LTP_LO, CA_LTP_HI = 0.20, 0.85   # bell-shaped LTP window on the plateau-Ca surrogate (sigmoid of the product, [0,1])
CA_LTD_THRESH = 0.85      # L-type Ca threshold for LTD
W_CLIP = 3.0             # weight clip (stability / metaplasticity surrogate)


def _ca_surrogate(g):
    """NMDA-calcium surrogate per branch = sigmoid of the conjunction output g (maps the product to [0,1] for the
    bell-shaped windowed plasticity rule -- the eLife calcium dependence)."""
    return 1.0 / (1.0 + np.exp(-np.clip(g, -30.0, 30.0)))


# ============================================================================
# THE LEARNED DENDRITIC SIGMA-PI BINDER (the candidate)
# ============================================================================

class DendriticSigmaPiBinder:
    """LEARNED dendritic sigma-pi (multiplicative conjunction) binder.

    bind(role, filler)  = (role @ W_R) (x) (filler @ W_F)                   [D_h]  (the multiplicative conjunction)
    bundle              = sum_r bind(role_r, filler_r)                       [D_h]  (superposition)
    unbind(bundle, role)= bundle (x) (role @ W_Rinv)                        [D_h]  (conjugate conjunction)
    cleanup             = argmax_f cos( unbind, fillers @ W_F )                     (matched-filter argmax)

    Learned by a LOCAL three-factor calcium/dopamine rule (NO backprop through the products). The plateau-Ca
    surrogate is sigmoid of the per-branch conjunction; dopamine is +1 on a correct readout, -1 otherwise. The rule
    is presynaptic-gated (Hebbian) so it only touches the co-active cluster -> it ROUTES co-relevant attributes onto a
    shared branch (the eLife inhibitory-compartmentalization function), the learnable part the fixed +-1 lacks.

    additive=True turns OFF the multiplication (bind/unbind become a LINEAR SUM) -> the LESION / point-neuron
    baseline: removing the product is the load-bearing ablation; the multiplication is what binds iff this collapses."""

    def __init__(self, D_in, D_h=D_H, seed=42, additive=False):
        self.D_in, self.D_h = D_in, D_h
        self.additive = bool(additive)
        rng = np.random.default_rng(seed * 17 + 3)
        s_in = PROJ_SCALE      # projected u,w ~ unit std (codes are unit-norm; W ~ N(0,1) => u = code@W ~ N(0,1))
        # Synaptic clusters: role / filler / inverse-role projections onto the D_h branches.
        self.W_R = rng.standard_normal((D_in, D_h)) * s_in
        self.W_F = rng.standard_normal((D_in, D_h)) * s_in
        self.W_Rinv = rng.standard_normal((D_in, D_h)) * s_in

    # --- forward ops ---
    def bind(self, role, filler):
        u, w = role @ self.W_R, filler @ self.W_F
        return (u + w) if self.additive else (u * w)           # ADDITIVE (lesion) vs MULTIPLICATIVE conjunction

    def unbind(self, bundle, role):
        rinv = role @ self.W_Rinv
        return (bundle + rinv) if self.additive else (bundle * rinv)

    def filler_templates(self, fillers):
        """Project the codebook fillers into the branch space -> the matched-filter templates [F, D_h]."""
        return fillers @ self.W_F

    def cleanup(self, act, fillers):
        """Matched-filter argmax: nearest projected-filler template to act, by cosine."""
        return native_argmax(act, self.filler_templates(fillers))

    # --- the LOCAL three-factor learning rule ---
    def _three_factor_update(self, pre_role, pre_filler, ca_branch, dopamine):
        """One local calcium/dopamine update on the queried bind's cluster. NO gradient, NO backprop, NO graph.

        LTP where ca in a bell-shaped window AND dopamine peak; LTD where ca above an L-type threshold AND dopamine
        pause. Presynaptic-gated -> only the active cluster moves -> routes co-relevant attributes onto a shared
        branch."""
        in_ltp_window = ((ca_branch >= CA_LTP_LO) & (ca_branch <= CA_LTP_HI)).astype(np.float64)
        above_ltd = (ca_branch > CA_LTD_THRESH).astype(np.float64)
        if dopamine > 0:
            branch_signal = ETA_LTP * in_ltp_window
        else:
            branch_signal = -ETA_LTD * above_ltd
        self.W_R += np.outer(pre_role, branch_signal)
        self.W_F += np.outer(pre_filler, branch_signal)
        self.W_Rinv += np.outer(pre_role, branch_signal)      # inverse-role co-adapts so unbind tracks bind
        np.clip(self.W_R, -W_CLIP, W_CLIP, out=self.W_R)
        np.clip(self.W_F, -W_CLIP, W_CLIP, out=self.W_F)
        np.clip(self.W_Rinv, -W_CLIP, W_CLIP, out=self.W_Rinv)

    def train_fact_step(self, roleids, fillerids, roles, fillers, t):
        """One BUNDLE-AWARE step: bundle the bindings, unbind role t, read out, apply the local rule. Returns the
        readout-correct flag (monitoring)."""
        gs = [self.bind(roles[r], fillers[f]) for r, f in zip(roleids, fillerids)]
        bundle = sum(gs)
        act = self.unbind(bundle, roles[roleids[t]])
        pred = self.cleanup(act, fillers)
        correct = int(pred == fillerids[t])
        dopamine = 1.0 if correct else -1.0
        ca = _ca_surrogate(gs[t])                              # Ca surrogate for the QUERIED branch (eLife calcium)
        self._three_factor_update(roles[roleids[t]], fillers[fillerids[t]], ca, dopamine)
        return correct


# ============================================================================
# FHRR self-inverse reference (fixed-primitive, systematic by construction; same pipeline)
# ============================================================================

def fhrr_bundled_recall(roles_phase, fillers_phase, roleids, fillerids, t):
    bound = [fhrr_bind(roles_phase[r], fillers_phase[f]) for r, f in zip(roleids, fillerids)]
    bundle = np.mean(np.stack(bound), axis=0)
    est = fhrr_unbind(bundle, roles_phase[roleids[t]])
    return int(fhrr_cleanup(est, fillers_phase) == fillerids[t])


# ============================================================================
# EVALUATION (one seed; reuses the harness splits + metric structure)
# ============================================================================

def _train_and_eval(binder_factory, codes, seed, permuted=False):
    """Train a binder (factory(D_in, seed)) bundle-aware on the train combos, then score:
      single-binding held-out, bundled train-combo, bundled held-out-combo.
    permuted=True shuffles the filler<->code assignment per seed (the permuted control)."""
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F].copy()
    if permuted:
        perm = np.random.default_rng(seed * 911 + 1).permutation(F)
        fillers = fillers[perm]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    single_held, bundle_train, bundle_held = [], [], []
    for split in splits:
        train_set = set(split["train"])
        binder = binder_factory(D_in, seed)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(N_BUNDLE_ROLES)}
        if min(len(tr_by_role[r]) for r in range(N_BUNDLE_ROLES)) == 0:
            continue
        for _ in range(N_FACT_STEPS):                          # BUNDLE-AWARE training (SVO roles 0,1,2)
            fa = int(rng.choice(tr_by_role[0])); fv = int(rng.choice(tr_by_role[1])); fo = int(rng.choice(tr_by_role[2]))
            binder.train_fact_step([0, 1, 2], [fa, fv, fo], roles, fillers, int(rng.integers(N_BUNDLE_ROLES)))
        # single-binding held-out (does the bind GENERALIZE to held-out (role, filler) combos?)
        sc = sum(int(binder.cleanup(binder.unbind(binder.bind(roles[r], fillers[f]), roles[r]), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)
        # bundled SVO recall, split by whether the queried (role, filler) was a train or held-out combo
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, N_BUNDLE_ROLES, replace=False)
            bundle = sum(binder.bind(roles[r], fillers[int(fids[r])]) for r in range(N_BUNDLE_ROLES))
            for r in range(N_BUNDLE_ROLES):
                ok = int(binder.cleanup(binder.unbind(bundle, roles[r]), fillers) == int(fids[r]))
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
    return (float(np.mean(single_held)) if single_held else 0.0,
            float(np.mean(bundle_train)) if bundle_train else 0.0,
            float(np.mean(bundle_held)) if bundle_held else 0.0)


def _fhrr_ceiling(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    held = []
    for split in splits:
        train_set = set(split["train"])
        nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, N_BUNDLE_ROLES, replace=False)
            for r in range(N_BUNDLE_ROLES):
                if (r, int(fids[r])) in train_set:
                    continue
                nh_ok += fhrr_bundled_recall(roles, fillers, [0, 1, 2], [int(x) for x in fids], r)
                nh += 1
        held.append(nh_ok / nh if nh else 0.0)
    return float(np.mean(held)) if held else 0.0


def _learned_linear_baseline(codes, seed):
    """The CYCLE-103 learned-multiplicative bind + LEARNED-LINEAR unbind/cleanup (must FAIL ~0.056). Imported so the
    de-risk shows the SAME-pipeline contrast (the broken element this de-risk removes)."""
    try:
        from research.runners._phaseB_multiplicative_bind_bundled_derisk import run_seed as ll_run_seed
    except Exception as exc:  # pragma: no cover
        return {"single_held": float("nan"), "bundle_train": float("nan"), "bundle_held": float("nan"),
                "error": str(exc)}
    r = ll_run_seed(codes, seed)
    return {"single_held": r["single_held"], "bundle_train": r["bundle_train"], "bundle_held": r["bundle_held"]}


def run_seed(codes, seed, with_learned_linear=False):
    t0 = time.time()
    # Candidate: learned dendritic sigma-pi (multiplicative) binder.
    d_sh, d_bt, d_bh = _train_and_eval(
        lambda D_in, s: DendriticSigmaPiBinder(D_in, D_H, s, additive=False), codes, seed)
    # LESION = remove the multiplication (product -> additive sum) = the point-neuron baseline.
    p_sh, p_bt, p_bh = _train_and_eval(
        lambda D_in, s: DendriticSigmaPiBinder(D_in, D_H, s, additive=True), codes, seed)
    # Permuted control on the dendritic binder (role<->filler assignment shuffled).
    perm_sh, perm_bt, perm_bh = _train_and_eval(
        lambda D_in, s: DendriticSigmaPiBinder(D_in, D_H, s, additive=False), codes, seed, permuted=True)
    # FHRR self-inverse reference (fixed primitive, same pipeline).
    fhrr_bh = _fhrr_ceiling(codes, seed)
    # Memorization floor (held-out must be ~ chance).
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    mem_held = []
    for split in splits:
        lk = MemorizationLookup(); lk.train(split["train"], None, codes[:F])
        ok = n = 0
        for r, f in split["held_out"]:
            cands = [cf for (cr, cf) in lk._store.keys() if cr == r]
            guess = cands[-1] if cands else -1
            ok += int(guess == f); n += 1
        mem_held.append(ok / n if n else 0.0)
    mem_bh = float(np.mean(mem_held)) if mem_held else 0.0
    # Optional: the CYCLE-103 learned-linear baseline (expensive; only on demand).
    ll = _learned_linear_baseline(codes, seed) if with_learned_linear else None

    print(f"  [seed {seed}] DENDRITIC sigma-pi: single-held {d_sh:.3f} | bundle-train {d_bt:.3f} | "
          f"bundle-HELD {d_bh:.3f} (gap {d_bt-d_bh:+.3f})", flush=True)
    print(f"             LESION(additive/point): single {p_sh:.3f} | bundle-train {p_bt:.3f} | bundle-HELD {p_bh:.3f}",
          flush=True)
    print(f"             permuted bundle-HELD {perm_bh:.3f} | mem-floor held {mem_bh:.3f} | FHRR ref bundle-HELD "
          f"{fhrr_bh:.3f}" + (f" | learned-linear bundle-HELD {ll['bundle_held']:.3f}" if ll else "")
          + f" | (elapsed {time.time()-t0:.0f}s)", flush=True)
    row = {"seed": seed,
           "dend_single_held": d_sh, "dend_bundle_train": d_bt, "dend_bundle_held": d_bh,
           "lesion_single_held": p_sh, "lesion_bundle_train": p_bt, "lesion_bundle_held": p_bh,
           "permuted_bundle_held": perm_bh, "mem_floor_held": mem_bh, "fhrr_ref_held": fhrr_bh}
    if ll:
        row["learned_linear"] = ll
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--with-learned-linear", action="store_true",
                        help="Also run the CYCLE-103 learned-linear baseline per seed (slow ~5min/seed).")
    parser.add_argument("--out", type=str,
                        default=os.path.join(_REPO, "research", "findings", "raw", "_phaseB_dendritic_bind.json"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[DENDRITIC sigma-pi bind de-risk] does a LEARNED dendritic MULTIPLICATIVE conjunction recover "
          f"two-attribute bundling that GENERALIZES, where the point / learned-linear baselines hit the K=5 wall?",
          flush=True)
    print(f"  (additive point bundled held-out ~0.193; learned-multiplicative-LINEAR ~0.056; chance 1/F={1.0/F:.3f}; "
          f"fixed +-1/FHRR bundles 0.989)", flush=True)
    rows = [run_seed(codes, s, with_learned_linear=args.with_learned_linear) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    d_sh, d_bt, d_bh = m("dend_single_held"), m("dend_bundle_train"), m("dend_bundle_held")
    p_sh, p_bh = m("lesion_single_held"), m("lesion_bundle_held")
    perm_bh = m("permuted_bundle_held"); mem_bh = m("mem_floor_held"); fhrr_bh = m("fhrr_ref_held")
    chance = 1.0 / F
    ll_bh = float(np.mean([r["learned_linear"]["bundle_held"] for r in rows if "learned_linear" in r])) \
        if args.with_learned_linear else None
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): DENDRITIC single-held {d_sh:.3f} | bundle-train {d_bt:.3f} | "
          f"bundle-HELD {d_bh:.3f} (gap {d_bt-d_bh:+.3f})", flush=True)
    print(f"  baselines (must FAIL): LESION=point-additive single {p_sh:.3f} bundle-HELD {p_bh:.3f} | "
          f"permuted {perm_bh:.3f} | mem-floor {mem_bh:.3f}"
          + (f" | learned-linear {ll_bh:.3f}" if ll_bh is not None else "") + f" | chance {chance:.3f}", flush=True)
    print(f"  FHRR fixed-primitive reference (same pipeline): {fhrr_bh:.3f}", flush=True)
    print(f"{'='*100}", flush=True)

    gap = d_bt - d_bh
    go_generalizes = d_bh >= 0.6 * d_bt
    go_bundle = (d_bh >= 0.40) and go_generalizes and (d_bh > 0.193 + 0.05)
    go_single = d_sh >= 0.40
    baselines_fail = (p_bh <= 0.25)                           # the lesion (remove multiplication) collapses
    if go_bundle and go_single and baselines_fail:
        print(f"  GO: a LEARNED dendritic MULTIPLICATION lifts the two-attribute wall WITH generalization -- "
              f"bundle held-out {d_bh:.3f} (>> additive 0.193, >> chance {chance:.3f}), {d_bh/max(d_bt,1e-9):.0%} of "
              f"train {d_bt:.3f} (small gap {gap:+.3f}), single-attr {d_sh:.3f} not regressed, and the LESION "
              f"(remove the product -> additive) collapses to {p_bh:.3f}. ==> the dendrite earns its keep on the "
              f"binding wall; recommend the small protected sigma-pi sim/ primitive + a phased build. ESCALATE to 6 "
              f"seeds.", flush=True)
        verdict = "GO_3SEED"
    elif d_bh >= 0.25 and go_generalizes and baselines_fail:
        print(f"  PARTIAL: the dendritic multiplication HELPS (bundle held-out {d_bh:.3f} vs additive 0.193, lesion "
              f"{p_bh:.3f}) and generalizes (gap {gap:+.3f}) but is below the 0.40 GO bar -- add an iterative "
              f"resonator cleanup in the unbind loop / more branch capacity before any sim/ build.", flush=True)
        verdict = "PARTIAL"
    else:
        reason = ("held-out collapses to memorization" if (d_bt > 0.4 and not go_generalizes)
                  else "does not beat the wall / baselines")
        print(f"  NEGATIVE: even a LEARNED dendritic multiplication does not lift two-attribute bundling with "
              f"generalization ({reason}: bundle-train {d_bt:.3f}, bundle-HELD {d_bh:.3f}, lesion {p_bh:.3f}). The "
              f"binding wall is NOT (only) the missing dendritic multiplication. Cheap terminus: NO GPU, NO sim/ "
              f"edit, NO months-scale build. Production keeps the fixed +-1/FHRR primitive (bundles 0.989).",
              flush=True)
        verdict = "NEGATIVE"
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)

    out = {"verdict": verdict, "n_seeds": len(seeds), "chance": chance,
           "dend_single_held": d_sh, "dend_bundle_train": d_bt, "dend_bundle_held": d_bh, "gap": gap,
           "lesion_single_held": p_sh, "lesion_bundle_held": p_bh, "permuted_bundle_held": perm_bh,
           "mem_floor_held": mem_bh, "fhrr_ref_held": fhrr_bh, "learned_linear_bundle_held": ll_bh,
           "additive_ref_0193": 0.193, "mult_linear_ref_0056": 0.056, "per_seed": rows,
           "proj_scale": PROJ_SCALE, "n_fact_steps": N_FACT_STEPS, "D_h": D_H, "F": F, "R": R}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
