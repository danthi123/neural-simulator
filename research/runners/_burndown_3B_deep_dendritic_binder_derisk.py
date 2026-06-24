"""BURNDOWN Phase-3B (2026-06-24) -- the DEEP DENDRITIC LEARNED BINDER: the ONE surviving untested hypothesis
from the cortex/dendrite scoping (research/findings/2026-06-24-learned-cortex-dendrite-phase3-scoping.md).

QUESTION
--------
Does a DEEP (>=2 hidden-layer) learned binder with APICAL-BASAL CREDIT ASSIGNMENT learn GENERALIZABLE
MULTI-ATTRIBUTE composition (the genuine FHRR-idealization residual = inventory C-1 / H-3)?

The cheap SINGLE-layer learned-dendritic-bind already came back NEGATIVE (2026-06-19-dendritic-binding-toy-derisk:
memorizes two-attribute 0.422 / generalizes 0.168 held-out, BELOW the fixed FHRR primitive 0.261). The DEEP regime
-- where the apical-basal dendrite is *designed* to do credit assignment (Sacramento-Senn 2018; Payeur-Naud-Richards
2021: credit assignment needs HIDDEN layers) -- is UNTESTED. That is the cheapest decisive next de-risk and it is
CPU/numpy with NO sim/ edit (it reuses the project's OWN DendriticMLP + DendriticLayer + urbanczik_senn_update
machine + the existing rigorous binding harness).

WHY DEEP CHANGES THE QUESTION (the literature-faithful reframe)
--------------------------------------------------------------
The single-layer dendritic sigma-pi (2026-06-19) had "nothing to credit-assign" -- a single trainable layer, the
feedback-alignment apical machinery has no hidden representation to shape. The apical-basal dendrite's WHOLE
function (Guerguiev-Lillicrap-Richards 2017; Sacramento-Senn 2018) is to deliver the top-down teaching signal to
HIDDEN layers via a FIXED-RANDOM apical projection (feedback alignment, NO weight transport) so deep features
self-organize. A multi-attribute unbind needs a role-DEPENDENT nonlinear read-out of a superposition; a deep network
with hidden layers + dendritic credit assignment is exactly the regime that could learn that nonlinear map -- the
genuine untested hypothesis.

THE PROBE (the cheapest-first formulation the scoping pre-registered)
--------------------------------------------------------------------
Multi-attribute composition = an SVO fact is a SUPERPOSITION of 3 role-filler bindings (roles 0,1,2; F fillers).
  bundle = sum_r  bind(role_r, filler_r)                        # superposition (the multi-attribute load)
  unbind+cleanup: given (bundle, query-role t) -> recover filler_t  (1-of-F)
The unbind+cleanup IS a classification problem: input [bundle ; query_role_code] -> softmax over F fillers. The DEEP
DendriticMLP learns this read-out with hidden-layer credit assignment (feedback alignment, NO weight transport).

The BIND that forms the bundle is the MULTIPLICATIVE (sigma-pi / Hadamard) dendritic conjunction -- the supralinear
op a point neuron's single linear summation cannot form (Mikulasch-Priesemann; Kleyko/Frady VSA<->sigma-pi). The
LESION replaces the product with an ADDITIVE sum (point-neuron) -- the decisive dendrite anti-cheat: if held-out
generalization survives the lesion, it is NOT coming from the dendritic multiplication.

ARMS (all on the IDENTICAL codes/splits; multi-seed)
----------------------------------------------------
  deep_dendrite (TEST)        : MULTIPLICATIVE bind -> DEEP DendriticMLP unbind/cleanup (>=2 hidden, Urbanczik-Senn
                                feedback-alignment credit assignment, NO weight transport). The question: held-out
                                multi-attr >= 0.40, generalizes (small train->held gap), multi-seed, lesion-collapses.
  single_layer_dendrite (CTL) : the 2026-06-19 single-layer sigma-pi (imported) -- MUST fail held-out (~0.168).
  learned_linear (CTL)        : the CYCLE-103 multiplicative-bind + LEARNED-LINEAR unbind (imported) -- MUST fail (~0.056).
  fixed_FHRR (POS CTL/ceiling): the +-1 / FHRR self-inverse algebra, same pipeline -- bundles ~0.989 / 2-attr held 0.261.
  memorization_floor (CTL)    : lookup table -> held-out ~ chance (1/F).
  LESION (the decisive anti-cheat): deep_dendrite with the MULTIPLICATIVE bind -> ADDITIVE sum (point-neuron).
                                MUST collapse to the additive/point floor (else the dendrite isn't load-bearing).
  permuted (CTL)              : role<->filler assignment shuffled -> the learned read-out must collapse.

  Also a fenced backprop ORACLE arm (DendriticMLP mode='oracle') -- a HAND-DERIVED true-gradient ceiling, used ONLY
  to certify the DEEP architecture has the CAPACITY to fit/generalize the task (so a feedback-alignment NEGATIVE is
  "the local rule can't credit-assign it", not "the architecture can't represent it"). NOT the deliverable; never
  the brain-based claim.

GO BAR (pre-registered, NOT tuned to result; per the scoping)
-------------------------------------------------------------
  GO       = deep_dendrite (feedback-alignment) held-out multi-attr >= 0.40 AND > both point-neuron controls AND
             > the single-layer 0.168, train->held gap SMALL (generalizes), 3-seed (escalate 6 for any GO claim),
             AND the LESION (remove the product) collapses.
  BOUNDARY = beats the single-layer dendrite but stays below the fixed FHRR (a characterized partial).
  NEGATIVE = the deep dendrite ALSO memorizes-but-doesn't-generalize -> the dendrite is comprehensively ruled out
             for learnable generalizable composition; the fixed +-1/FHRR primitive STAYS (binding-by-coincidence is
             a STRUCTURAL neural primitive, NOT a host shortcut) and the multi-attribute residual is an honest
             CHARACTERIZED point-neuron BOUNDARY -- itself a months-saving deliverable (the strong prior).

Reuse-by-import: make_role_codes / make_systematicity_splits / native_argmax / MemorizationLookup / fhrr_* from
cortex_learned_binder_systematicity_probe; DendriticSigmaPiBinder + the FHRR ceiling from _phaseB_dendritic_bind_derisk;
MultFHRRBinder (the learned-linear control) from _phaseB_multiplicative_bind_bundled_derisk; the DEEP credit-assignment
machine sim.dendritic_mlp.DendriticMLP (feedback alignment, no weight transport).
STRICTLY CPU; SIM_BACKEND=numpy; NO GPU; NO sim/ edit.
Run:  SIM_BACKEND=numpy python -u -m research.runners._burndown_3B_deep_dendritic_binder_derisk [--seeds 42,43,44]
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

# STRICTLY CPU/numpy -- the GPU is busy with a parallel burndown item. Force it BEFORE any sim import so
# DendriticMLP's module-level get_backend() resolves to numpy (NOT cupy).
os.environ["SIM_BACKEND"] = "numpy"

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    MemorizationLookup, fhrr_bind, fhrr_cleanup, fhrr_unbind, make_role_codes,
    make_systematicity_splits, native_argmax)
from research.runners._phaseB_dendritic_bind_derisk import (  # noqa: E402
    DendriticSigmaPiBinder, _train_and_eval as _train_eval_sigmapi, fhrr_bundled_recall)
from sim.dendritic_mlp import DendriticMLP  # the DEEP feedback-alignment credit-assignment machine  # noqa: E402

# Defaults match the harness we extend (SVO uses roles 0,1,2; F=16; 3 splits).
R, F, N_SPLITS = 4, 16, 3
N_BUNDLE_ROLES = 3        # SVO: bundle 3 (role, filler) pairs per fact (the multi-attribute frontier load)
N_EVAL_FACTS = 60         # held-out eval facts per split

# DEEP binder hyperparameters (fixed a-priori, one value all seeds -- anti-cheat: no per-seed tuning to a target).
D_BIND = 64               # dendritic-conjunction (bind) dimension == the single-layer harness D_h
HIDDEN = (128, 128)       # >=2 HIDDEN layers -> the DEEP credit-assignment regime (Sacramento-Senn / Payeur)
N_EPOCHS = 400            # training epochs over the train-combo fact distribution
BATCH = 64                # minibatch of bundled facts per step
LR = 0.3                  # feedback-alignment SGD lr (DendriticMLP momentum-normalized; one value all seeds)
PROJ_SCALE = 1.0          # bind-projection cluster gain (matches the single-layer harness)


# ============================================================================
# THE DEEP DENDRITIC BINDER (the candidate)
# ============================================================================

class DeepDendriticBinder:
    """MULTIPLICATIVE (sigma-pi) dendritic bind -> DEEP DendriticMLP unbind/cleanup (>=2 hidden layers, Urbanczik-Senn
    feedback-alignment credit assignment, NO weight transport, NO backprop weight transport).

    bind(role, filler)  = (role @ W_R) (x) (filler @ W_F)            [D_bind]  (the multiplicative conjunction; the
                                                                                supralinear dendritic op a point
                                                                                neuron's linear sum cannot form)
    bundle              = sum_r bind(role_r, filler_r)               [D_bind]  (superposition = the multi-attr load)
    unbind+cleanup      = DendriticMLP([D_bind + R -> *HIDDEN -> F]) ( [bundle ; onehot(query_role)] )  -> argmax filler

    The DEEP MLP is where the credit assignment lives: hidden layers learn (via the FIXED-RANDOM apical feedback B,
    NO weight transport) the role-DEPENDENT nonlinear read-out that pulls filler_t out of the superposition. This is
    the regime the single-layer sigma-pi (2026-06-19, 0.168) could not test (nothing to credit-assign).

    additive=True  -> bind/bundle become a LINEAR SUM (no product) = the LESION / point-neuron baseline. The DEEP MLP
                      is UNCHANGED (same depth, same credit assignment); only the supralinear bind is removed. If
                      held-out generalization survives this lesion, it is NOT coming from the dendritic multiplication.
    mode='oracle'  -> the DendriticMLP uses a HAND-DERIVED true gradient (NO autodiff). Capacity ceiling ONLY (fenced):
                      certifies the deep architecture CAN represent/generalize the map, so a feedback-alignment
                      NEGATIVE is "the local rule can't credit-assign it", not "the form is too weak". NOT brain-based.
    """

    def __init__(self, D_in, F_count, seed=42, additive=False, mode="local_correct",
                 hidden=HIDDEN, d_bind=D_BIND):
        self.D_in, self.F = D_in, F_count
        self.additive = bool(additive)
        self.mode = mode
        self.d_bind = d_bind
        rng = np.random.default_rng(seed * 17 + 3)
        s_in = PROJ_SCALE
        # FIXED bind-projection clusters (role / filler), like the single-layer harness. NOT learned by the deep MLP
        # (they form the conjunction the MLP then reads); kept fixed so the test isolates the DEEP read-out's ability
        # to credit-assign a role-dependent unbind from the multiplicative superposition.
        self.W_R = rng.standard_normal((D_in, d_bind)) * s_in
        self.W_F = rng.standard_normal((D_in, d_bind)) * s_in
        # the DEEP credit-assignment read-out: [bundle (d_bind) ; onehot query-role (R)] -> hidden... -> F fillers
        sizes = [d_bind + R, *hidden, F_count]
        self.mlp = DendriticMLP(sizes, seed=seed)

    # --- forward ops ---
    def bind(self, role, filler):
        u, w = role @ self.W_R, filler @ self.W_F
        return (u + w) if self.additive else (u * w)

    def bundle_facts(self, roleids, fillerids, roles, fillers):
        return sum(self.bind(roles[r], fillers[f]) for r, f in zip(roleids, fillerids))

    def _mlp_input(self, bundle, query_role):
        """[bundle ; onehot(query_role)] -- the role one-hot tells the read-out WHICH slot to recover."""
        oh = np.zeros(R)
        oh[query_role] = 1.0
        return np.concatenate([np.atleast_1d(bundle), oh])

    def predict_filler(self, bundle, query_role):
        x = self._mlp_input(bundle, query_role)[None, :]
        _, lg = self.mlp._forward(x)
        return int(np.argmax(lg[0]))

    def train_epoch(self, train_by_role, roles, fillers, rng, batch=BATCH, lr=LR):
        """One epoch: build BATCH bundled facts from train combos, query a random role, fit the DEEP MLP via
        feedback-alignment credit assignment (NO weight transport). Returns mean batch loss (monitoring)."""
        Xs, ys = [], []
        for _ in range(batch):
            fa = int(rng.choice(train_by_role[0]))
            fv = int(rng.choice(train_by_role[1]))
            fo = int(rng.choice(train_by_role[2]))
            fids = [fa, fv, fo]
            bundle = self.bundle_facts([0, 1, 2], fids, roles, fillers)
            t = int(rng.integers(N_BUNDLE_ROLES))
            Xs.append(self._mlp_input(bundle, t))
            ys.append(fids[t])
        X = np.stack(Xs)
        y = np.array(ys)
        self.mlp.train_step(X, y, self.mode, lr)
        return self.mlp.loss(X, y)


# ============================================================================
# EVALUATION (one seed; reuses the harness splits + the exact held-out/train-combo metric structure)
# ============================================================================

def _deep_train_and_eval(codes, seed, additive=False, mode="local_correct", permuted=False):
    """Train a DEEP dendritic binder bundle-aware on the train combos, then score the SAME three numbers the
    single-layer harness reports: single-binding held-out, bundled train-combo, bundled held-out-combo.
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
        # leakage assert (anti-cheat 1): no held-out combo appears in train
        assert len(set(split["train"]) & set(split["held_out"])) == 0, "LEAKAGE"
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(N_BUNDLE_ROLES)}
        if min(len(tr_by_role[r]) for r in range(N_BUNDLE_ROLES)) == 0:
            continue
        binder = DeepDendriticBinder(D_in, F, seed=seed, additive=additive, mode=mode)
        for _ in range(N_EPOCHS):
            binder.train_epoch(tr_by_role, roles, fillers, rng)
        # single-binding held-out: bundle ONE (role, filler) (a degenerate single-attr bundle), query that role.
        sc_ok = sc_n = 0
        for r, f in split["held_out"]:
            if r >= N_BUNDLE_ROLES:
                continue
            bundle = binder.bind(roles[r], fillers[f])      # single binding
            sc_ok += int(binder.predict_filler(bundle, r) == f)
            sc_n += 1
        single_held.append(sc_ok / sc_n if sc_n else 0.0)
        # bundled SVO recall, split by whether the queried (role, filler) was a train or held-out combo
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, N_BUNDLE_ROLES, replace=False)
            bundle = binder.bundle_facts([0, 1, 2], [int(x) for x in fids], roles, fillers)
            for r in range(N_BUNDLE_ROLES):
                ok = int(binder.predict_filler(bundle, r) == int(fids[r]))
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
    """The fixed +-1 / FHRR self-inverse algebra on the IDENTICAL pipeline (the positive-control / harness-soundness
    gate -- must bundle the multi-attribute held-out well, else the harness can't detect working bundling)."""
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


def _mem_floor(codes, seed):
    """Memorization floor: a lookup table can only return a train-seen filler for a role -> held-out ~ chance."""
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
    return float(np.mean(mem_held)) if mem_held else 0.0


def _single_layer_control(codes, seed):
    """The 2026-06-19 single-layer dendritic sigma-pi (imported) on the IDENTICAL harness -- the prior NEGATIVE
    (~0.168 held-out). Run here so the contrast is SAME-RUN, not cited."""
    return _train_eval_sigmapi(
        lambda D_in, s: DendriticSigmaPiBinder(D_in, D_BIND, s, additive=False), codes, seed)


def _learned_linear_control(codes, seed):
    """The CYCLE-103 learned-multiplicative bind + LEARNED-LINEAR unbind (imported) -- must FAIL (~0.056)."""
    try:
        from research.runners._phaseB_multiplicative_bind_bundled_derisk import run_seed as ll_run_seed
    except Exception as exc:  # pragma: no cover
        return {"single_held": float("nan"), "bundle_train": float("nan"), "bundle_held": float("nan"),
                "error": str(exc)}
    r = ll_run_seed(codes, seed)
    return {"single_held": r["single_held"], "bundle_train": r["bundle_train"], "bundle_held": r["bundle_held"]}


def run_seed(codes, seed, with_oracle=False, with_learned_linear=False):
    t0 = time.time()
    # TEST: DEEP dendritic binder (feedback-alignment credit assignment, multiplicative bind).
    d_sh, d_bt, d_bh = _deep_train_and_eval(codes, seed, additive=False, mode="local_correct")
    # LESION (the decisive anti-cheat): remove the multiplication (product -> additive sum). DEEP MLP UNCHANGED.
    l_sh, l_bt, l_bh = _deep_train_and_eval(codes, seed, additive=True, mode="local_correct")
    # Permuted control (role<->filler assignment shuffled) on the DEEP multiplicative binder.
    p_sh, p_bt, p_bh = _deep_train_and_eval(codes, seed, additive=False, mode="local_correct", permuted=True)
    # FHRR fixed-primitive ceiling (same pipeline).
    fhrr_bh = _fhrr_ceiling(codes, seed)
    # Memorization floor.
    mem_bh = _mem_floor(codes, seed)
    # Single-layer dendrite (prior NEGATIVE) on the same harness.
    sl_sh, sl_bt, sl_bh = _single_layer_control(codes, seed)
    # Optional: capacity ceiling (hand-derived true-grad oracle on the SAME deep architecture; fenced, NOT brain-based).
    o_bh = None
    if with_oracle:
        _, _, o_bh = _deep_train_and_eval(codes, seed, additive=False, mode="oracle")
    # Optional: the CYCLE-103 learned-linear control (slow).
    ll = _learned_linear_control(codes, seed) if with_learned_linear else None

    print(f"  [seed {seed}] DEEP dendrite (feedback-align): single-held {d_sh:.3f} | bundle-train {d_bt:.3f} | "
          f"bundle-HELD {d_bh:.3f} (gap {d_bt-d_bh:+.3f})", flush=True)
    print(f"             LESION(additive bind, same deep MLP): single {l_sh:.3f} | bundle-train {l_bt:.3f} | "
          f"bundle-HELD {l_bh:.3f}", flush=True)
    print(f"             single-layer dendrite (prior NEG): bundle-HELD {sl_bh:.3f} | permuted {p_bh:.3f} | "
          f"mem-floor {mem_bh:.3f} | FHRR ref {fhrr_bh:.3f}"
          + (f" | ORACLE(cap) {o_bh:.3f}" if o_bh is not None else "")
          + (f" | learned-linear {ll['bundle_held']:.3f}" if ll else "")
          + f" | (elapsed {time.time()-t0:.0f}s)", flush=True)
    row = {"seed": seed,
           "deep_single_held": d_sh, "deep_bundle_train": d_bt, "deep_bundle_held": d_bh,
           "lesion_single_held": l_sh, "lesion_bundle_train": l_bt, "lesion_bundle_held": l_bh,
           "permuted_bundle_held": p_bh, "mem_floor_held": mem_bh, "fhrr_ref_held": fhrr_bh,
           "single_layer_bundle_held": sl_bh, "single_layer_bundle_train": sl_bt, "single_layer_single_held": sl_sh}
    if o_bh is not None:
        row["oracle_bundle_held"] = o_bh
    if ll:
        row["learned_linear"] = ll
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--with-oracle", action="store_true",
                        help="Also run the hand-derived true-grad capacity ceiling (fenced, NOT brain-based).")
    parser.add_argument("--with-learned-linear", action="store_true",
                        help="Also run the CYCLE-103 learned-linear baseline per seed (slow).")
    parser.add_argument("--out", type=str,
                        default=os.path.join(_REPO, "research", "findings", "raw",
                                             "_burndown_3B_deep_dendritic_binder.json"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.environ["SIM_BACKEND"] = "numpy"
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    # report the code-correlation regime (these are the production PPMI stream codes = correlated)
    fcods = codes[:F]
    cos_vals = [float(fcods[i] @ fcods[j]) for i in range(F) for j in range(i + 1, F)]
    cos_mean = float(np.mean(cos_vals)); cos_max = float(np.max(np.abs(cos_vals)))
    chance = 1.0 / F

    print(f"[BURNDOWN 3B -- DEEP dendritic binder] does a DEEP (>=2 hidden) learned binder with apical-basal credit "
          f"assignment learn GENERALIZABLE multi-attribute composition where the SINGLE-layer dendrite (0.168) and "
          f"the learned-linear inverse (0.056) provably cannot?", flush=True)
    print(f"  codes: PPMI stream (correlated) F={F} between-cos mean {cos_mean:.3f} max {cos_max:.3f} | chance 1/F="
          f"{chance:.3f} | fixed +-1/FHRR bundles ~0.989 (2-attr held ~0.261) | single-layer dendrite held ~0.168",
          flush=True)
    print(f"  DEEP arch: bind D={D_BIND} -> DendriticMLP[{D_BIND}+{R} -> {HIDDEN} -> {F}] feedback-alignment, "
          f"NO weight transport; {N_EPOCHS} epochs x batch {BATCH}", flush=True)
    rows = [run_seed(codes, s, with_oracle=args.with_oracle, with_learned_linear=args.with_learned_linear)
            for s in seeds]

    def m(k):
        vals = [r[k] for r in rows if k in r]
        return float(np.mean(vals)) if vals else float("nan")
    d_sh, d_bt, d_bh = m("deep_single_held"), m("deep_bundle_train"), m("deep_bundle_held")
    l_bh = m("lesion_bundle_held"); l_sh = m("lesion_single_held")
    perm_bh = m("permuted_bundle_held"); mem_bh = m("mem_floor_held"); fhrr_bh = m("fhrr_ref_held")
    sl_bh = m("single_layer_bundle_held")
    o_bh = m("oracle_bundle_held") if args.with_oracle else None
    ll_bh = float(np.mean([r["learned_linear"]["bundle_held"] for r in rows if "learned_linear" in r])) \
        if args.with_learned_linear else None
    gap = d_bt - d_bh

    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): DEEP dendrite single-held {d_sh:.3f} | bundle-train {d_bt:.3f} | "
          f"bundle-HELD {d_bh:.3f} (gap {gap:+.3f})", flush=True)
    print(f"  controls: LESION(additive) bundle-HELD {l_bh:.3f} | single-layer dendrite {sl_bh:.3f} | permuted "
          f"{perm_bh:.3f} | mem-floor {mem_bh:.3f}"
          + (f" | learned-linear {ll_bh:.3f}" if ll_bh is not None else "") + f" | chance {chance:.3f}", flush=True)
    print(f"  ceilings: FHRR fixed-primitive {fhrr_bh:.3f}" + (f" | ORACLE capacity {o_bh:.3f}" if o_bh is not None
          else ""), flush=True)
    print(f"{'='*104}", flush=True)

    # PRE-REGISTERED verdict logic (NOT tuned to result).
    go_generalizes = (d_bt < 1e-9) or (d_bh >= 0.6 * d_bt)               # held-out tracks train (small gap)
    beats_controls = (d_bh > l_bh + 0.05) and (d_bh > sl_bh + 0.05) and (d_bh > 0.193 + 0.05)
    lesion_collapses = (l_bh <= 0.25) or (l_bh < d_bh - 0.15)
    go_bundle = (d_bh >= 0.40) and go_generalizes and beats_controls and lesion_collapses
    if go_bundle:
        verdict = "GO"
        print(f"  GO: a DEEP dendritic binder with apical-basal credit assignment LEARNS generalizable "
              f"multi-attribute composition -- bundle held-out {d_bh:.3f} (>> single-layer {sl_bh:.3f}, >> additive "
              f"0.193, >> chance {chance:.3f}), {d_bh/max(d_bt,1e-9):.0%} of train {d_bt:.3f} (small gap {gap:+.3f}), "
              f"the LESION (product->sum, SAME deep MLP) collapses to {l_bh:.3f}. ==> the FHRR exact-inverse "
              f"idealization IS dendrite-replaceable; a Stage-1 protected two-compartment sim/ build follows.",
              flush=True)
    elif d_bh >= 0.25 and go_generalizes and (d_bh > sl_bh + 0.05) and lesion_collapses:
        verdict = "BOUNDARY_PARTIAL"
        print(f"  BOUNDARY/PARTIAL: depth HELPS (bundle held-out {d_bh:.3f} > single-layer {sl_bh:.3f}, generalizes "
              f"gap {gap:+.3f}, lesion {l_bh:.3f}) but stays below the 0.40 GO bar and the fixed FHRR {fhrr_bh:.3f}. "
              f"A characterized partial -- the dendrite is not (yet) the clean FHRR replacement; the fixed primitive "
              f"stays for production.", flush=True)
    else:
        verdict = "NEGATIVE"
        reason = ("held-out collapses to memorization" if (d_bt > 0.4 and not go_generalizes)
                  else "does not generalize / does not beat the controls / lesion does not isolate the product")
        print(f"  NEGATIVE: even a DEEP dendritic binder with apical-basal credit assignment does NOT learn "
              f"generalizable multi-attribute composition ({reason}: bundle-train {d_bt:.3f}, bundle-HELD {d_bh:.3f}, "
              f"single-layer {sl_bh:.3f}, lesion {l_bh:.3f}, FHRR ceiling {fhrr_bh:.3f}"
              + (f", ORACLE capacity {o_bh:.3f}" if o_bh is not None else "") + f"). ==> the DEEP regime -- the ONE "
              f"surviving untested hypothesis -- is ALSO ruled out. The dendrite is COMPREHENSIVELY ruled out for "
              f"learnable generalizable multi-attribute composition. The fixed +-1/FHRR primitive STAYS "
              f"(binding-by-coincidence = a STRUCTURAL neural primitive, NOT a host shortcut); the multi-attribute "
              f"residual is an honest CHARACTERIZED point-neuron BOUNDARY. Phase 3B = an honest boundary, NOT a "
              f"months-scale build. NO GPU, NO sim/ edit consumed.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)

    out = {"verdict": verdict, "n_seeds": len(seeds), "seeds": seeds, "chance": chance,
           "code_regime": "ppmi_stream_correlated", "code_between_cos_mean": cos_mean, "code_between_cos_max": cos_max,
           "deep_single_held": d_sh, "deep_bundle_train": d_bt, "deep_bundle_held": d_bh, "gap": gap,
           "lesion_single_held": l_sh, "lesion_bundle_held": l_bh,
           "single_layer_bundle_held": sl_bh, "permuted_bundle_held": perm_bh, "mem_floor_held": mem_bh,
           "fhrr_ref_held": fhrr_bh, "oracle_capacity_bundle_held": o_bh, "learned_linear_bundle_held": ll_bh,
           "single_layer_ref_0168": 0.168, "additive_ref_0193": 0.193, "mult_linear_ref_0056": 0.056,
           "fhrr_bundles_ref_0989": 0.989, "fhrr_2attr_held_ref_0261": 0.261,
           "go_generalizes": bool(go_generalizes), "beats_controls": bool(beats_controls),
           "lesion_collapses": bool(lesion_collapses),
           "arch": {"D_bind": D_BIND, "hidden": list(HIDDEN), "n_epochs": N_EPOCHS, "batch": BATCH, "lr": LR,
                    "F": F, "R": R, "n_bundle_roles": N_BUNDLE_ROLES, "n_splits": N_SPLITS},
           "per_seed": rows}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
