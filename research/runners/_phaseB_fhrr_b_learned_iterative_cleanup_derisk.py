"""FHRR-B Option 1 (CYCLE 330) -- the LEARNED-ITERATIVE-CLEANUP + capacity sweep on the bundle inverse.

THE SHORTCUT (FHRR-B), precisely. The production composer binds role (x) filler and BUNDLES a fact
(= superposes the R role-filler bindings of a sentence into one vector) and recovers a role's filler with a
FIXED, exactly-invertible Fourier-Holographic-Reduced-Representation (FHRR) algebra. The bind/bundle/unbind
OPERATIONS are already on-substrate spiking (resonate-and-fire + complex synapses) and the concept CODES are
learned; the residual genuinely-host-DESIGNED piece is the EXACT-INVERSE FORM of the multi-attribute bundle
inverse (recover one role's filler from a superposition of R bindings). "Bundle" throughout = superpose
multiple role-filler bindings into one fact vector; "unbind from a bundle" = recover one role's filler from
that superposition.

THE OWNER'S RULE (CYCLE 329, `286f8368`): a BOUNDARY is NOT an exit -- it is a prompt to research more + try
NEW mechanisms until past it. So this de-risk does not classify-and-stop; either outcome routes to the next
mechanism (Option 2 deep/hidden-layer learned binder; Option 3 orthogonal-role TPR).

WHAT IS NEW HERE (vs the FRLF base, which is GO at 0.639 with a SINGLE-PASS nearest-cosine cleanup, D_h=64).
The scoping (`2026-06-20-FHRR-B-learned-binder-scoping.md`, Option 1) ranks the highest-leverage cheap shot:
a bundle has NO one-shot inverse (a sum is not invertible by the algebra) but DOES have an ITERATIVE
decomposition (the Frady-Kent-Sommer resonator-network principle: alternate "unbind by the fixed structure ->
clean each factor toward the learned codebook -> explain away the other bindings -> re-estimate"). The cleanup
is a LEARNABLE read-out (NOT the impossible exact reciprocal), so the from-scratch learner can legitimately
re-enter on the one sub-op it can carry. Two never-swept levers:
  (a) CAPACITY: sweep bind-space D_h in {64, 128, 256, 512} (VSA bundling capacity is dimension-bounded -- the
      FRLF 0.639 at D_h=64 is plausibly capacity-starved, not mechanism-limited).
  (b) LEARNED ITERATIVE CLEANUP in the unbind loop: a resonator-style explaining-away decomposition that
      jointly estimates all R fillers and subtracts the re-bound estimates of the OTHER roles (matched-filter
      / interference-cancellation), re-cleaning each role's filler against the learned codebook each pass.

THE BIND STAYS A FIXED +-1 SELF-INVERSE STRUCTURE (a structural neural primitive: coincidence /
dendritic-product). Only the CLEANUP / DECOMPOSITION is learned + iterative -- which is exactly the realistic
cortex form the idealization lacks (lossy, redundant, learned read-out). A GO here closes FHRR-B's read-out
half on a measured signal and shrinks the residual to "the bind op is a fixed self-inverse structure."

ARMS (all on the IDENTICAL leakage-free systematicity harness; R=4, F=16 fillers; held-out holds out specific
(role,filler) PAIRINGS while every role+every filler still appears in training -- the Fodor-Pylyshyn 1988
never-seen-combination test):
  - fixed-role + learned-filler, SINGLE-PASS cleanup       (= FRLF base; 0.639 prior, the comparison floor)
  - fixed-role + learned-filler, LEARNED-ITERATIVE cleanup (THE NEW LEVER; resonator explaining-away)
    x D_h in {64, 128, 256, 512}
  - LESION (iterative): scramble the role inverse used by the cleanup -> the decomposition reads no role
    structure -> must collapse to ~chance (proves the lift rides the bind op + the structured cleanup, not a
    code-overlap artifact).
  - fixed-+-1 POSITIVE CONTROL on both: pure +-1 self-inverse bind + pure +-1 cleanup (no learned W) -- the
    harness must show working bundling here (the 0.989 ceiling) so a NEGATIVE is real, not a broken harness.

CITED NEGATIVE ARMS (established on the SAME harness, not re-run here -- the headline A/B the new lever must
beat): learned ADDITIVE 0.193; learned-LINEAR inverse 0.056; learned-DENDRITIC 0.168.

CODE AXIS (anti-cheat #7 -- run BOTH so a GO is not a clean-code artifact, the F=3 resonator's exact failure
mode): the CLEAN stream codes (`_phaseB_stream_codes_320_seed42.npy`, between-cos ~0.014) AND the CORRELATED
grounded/neural production codes (`_phaseB_stream_codes_320_neural_seed{42,43,44}.npy`).

PRE-REGISTERED VERDICT (fixed bars, never tuned to the result):
  GO: bundled held-out >= 0.90 AND >= 0.6x train, on >=5/6 seeds, WHILE additive (0.193) + learned-linear
      (0.056) stay NEGATIVE on the same harness AND the lesion collapses to ~chance -> the learned iterative
      cleanup + fixed bind reaches fixed-algebra parity; FHRR-B's read-out half closes; Option 4 (small
      guarded on-bridge wiring) is warranted (hand the controller the GPU command).
  BOUNDARY: held-out LIFTS materially over the 0.639 FRLF floor (e.g. -> 0.75-0.85) but short of 0.90 ->
      the cleanup learns + the capacity helps but the bind-FORM gap is partly fundamental -> the next mechanism
      is Option 2 (deep/hidden-layer learned binder, the dendrite re-entry) and/or Option 3 (orthogonal-role
      TPR). Record the characterized partial.
  NEGATIVE: held-out stays ~0.639 across the sweep -> the gap is fundamental at this representation -> the
      next mechanism is the structurally-different representation (Option 3 orthogonal-role TPR). A NEGATIVE
      here does NOT close the arc (per the owner's rule).

ANTI-CHEATS (the standard battery, all foregrounded): (1) point-neuron additive + learned-linear MUST fall
short on the identical corpus/splits/seeds (cited 0.193/0.056); (2) fixed-+-1 positive control CARRIES (0.989
ceiling, re-run here); (3) HELD-OUT systematicity, leakage-ASSERTED, vs the memorization floor (lookup -> 0.0)
+ chance (0.062) -- the bar is held-out generalization, never raw recall (the exact confound that retracted
the 2026-05-14 transitive-inference + 2026-05-03 permuted-label results); (4) PERMUTED-ROLE control (shuffle
role->filler -> collapse); (5) LESION (scramble the cleanup's role inverse -> collapse to the additive floor);
(6) COMPOSITION-NOT-COHERENCE -- the metric is unbind-recovers-the-right-filler, not a decorrelation proxy;
(7) DECORRELATED-vs-CORRELATED codes BOTH reported; (8) PROVENANCE -- this is a numpy CHARACTERIZATION of the
learnable cleanup, NOT an on-bridge claim; the on-bridge realization is the gated Option 4; (9) NEVER WEAKEN
THE NO-CONFAB MOAT -- a familiarity gap (known-filler vs novel-filler max-cosine) is reported and must stay
positive; the gate threshold is NOT tuned on the test; (10) >=6 seeds (42/43/44/100/101/102), fractional
>=5/6, CPU/numpy.

Reuse-by-import (make_role_codes / make_systematicity_splits / native_argmax). CPU; SIM_BACKEND=numpy; no
sim/ edits. NO GPU launched.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_learned_iterative_cleanup_derisk
"""
from __future__ import annotations

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
    make_role_codes, make_systematicity_splits, native_argmax)

# ---------------------------------------------------------------------------
# Config (the cheap-first numpy de-risk)
# ---------------------------------------------------------------------------
R, F, N_SPLITS, LR = 4, 16, 3, 0.005           # R roles, F fillers, splits/seed, learning rate
N_FACT_STEPS = 24000                            # training steps (matches the FRLF base)
N_EVAL_FACTS = 40                               # bundles drawn per split for the held-out estimate
N_ITERS = 6                                     # resonator explaining-away passes
D_H_SWEEP = (64, 128, 256, 512)                 # the capacity sweep (lever a)
SEEDS = (42, 43, 44, 100, 101, 102)             # >=6 seeds per the standing rule
# n_bundle = how many bindings are superposed per fact. The SVO production fact = 3 (agent/action/patient).
N_BUNDLE = 3

# Cited established NEGATIVE / POSITIVE arms on the SAME harness (the headline A/B; not re-run for the learned
# arms -- the fixed-+-1 positive IS re-run below as the live harness-sanity control).
CITED_ADDITIVE_BUNDLED = 0.193
CITED_LEARNED_LINEAR_BUNDLED = 0.056
CITED_DENDRITIC_BUNDLED = 0.168
FIXED_FHRR_CEILING = 0.989


# ===========================================================================
# THE BINDER: fixed +-1 self-inverse role + learned filler projection W_F + learned cleanup readout W_O.
# Bind stays FIXED/structural; only W_F (filler embedding) + W_O (cleanup readout) learn, bundle-aware.
# ===========================================================================
class FixedRoleLearnedFillerIterativeBinder:
    """bind g = role_proj[r] (x) (filler @ W_F)  [role_proj in {+-1}^D_h, FIXED self-inverse, the EXACT
    inverse -- NO learned W_Rinv]. bundle = sum of the N_BUNDLE binds. READ-OUT:

      single-pass (the FRLF base):  est_r = (bundle (x) role_proj[r]) @ W_O  -> nearest filler.

      LEARNED ITERATIVE (the new lever -- a resonator-style explaining-away decomposition): jointly hold an
      estimate filler-INDEX for every role in the bundle; each pass, for each role r, form a residual that
      SUBTRACTS the re-bound estimates of the OTHER roles (interference cancellation), unbind r by the fixed
      +-1 inverse, run the LEARNED cleanup readout W_O, and clean to the nearest codebook filler. Iterate.
      The cleanup is learnable (W_O) + iterative (resonator); the BIND is the fixed +-1 structure.

    `lesion` (control): scramble the role inverse the ITERATIVE cleanup uses -> the decomposition reads no role
    structure -> must collapse. Training is identical to the FRLF (single-pass loss through W_F, W_O); the
    iterative cleanup is a READ-OUT-time decomposition over the SAME learned W_F/W_O (no extra trained params),
    which is the honest test of whether iteration alone recovers the gap."""

    _PARAMS = ("W_F", "W_O")

    def __init__(self, D_in, roles, D_h, lr=LR, lam=1e-4, seed=42):
        self.D_in, self.D_h, self.lr, self.lam = D_in, D_h, lr, lam
        rng = np.random.default_rng(seed * 17 + 3)
        R_proj = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
        self.role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)          # [R, D_h] FIXED +-1 self-inverse
        s_in, s_h = 1.0 / np.sqrt(D_in), 1.0 / np.sqrt(D_h)
        self.W_F = rng.standard_normal((D_in, D_h)) * s_in                   # filler embedding (LEARNED)
        self.W_O = rng.standard_normal((D_h, D_in)) * s_h                    # cleanup readout  (LEARNED)
        # a scrambled role inverse for the lesion control (a different +-1 pattern, structure-destroying)
        self._scram = np.where(rng.standard_normal((roles.shape[0], D_h)) >= 0.0, 1.0, -1.0)
        self.t = 0
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._PARAMS}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._PARAMS}

    def _adam(self, name, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._m[name], self._v[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** self.t); vhat = v / (1 - b2 ** self.t)
        getattr(self, name)[:] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # ---- bind / encode ----
    def filler_embed(self, filler):
        return filler @ self.W_F                                            # [D_h]

    def bind(self, role_idx, filler):
        return self.role_proj[role_idx] * self.filler_embed(filler)         # g [D_h]

    def bind_from_embed(self, role_idx, w):
        return self.role_proj[role_idx] * w                                 # re-bind a stored embedding

    # ---- single-pass read-out (the FRLF base) ----
    def unbind_single(self, bundle, role_idx, lesion=False):
        inv = self._scram[role_idx] if lesion else self.role_proj[role_idx]
        return (bundle * inv) @ self.W_O                                    # cleanup readout [D_in]

    # ---- LEARNED ITERATIVE read-out (the new lever: resonator explaining-away) ----
    def decode_iterative(self, bundle, roles_present, fillers, n_iters=N_ITERS, lesion=False):
        """Jointly recover the filler index for each role in `roles_present` from the bundle.

        Resonator-network explaining-away: maintain a filler-index estimate per role; each pass, for role r,
        cancel the other roles' contributions (subtract their re-bound learned embeddings), unbind r by the
        fixed +-1 inverse, learned-clean via W_O, then snap to the nearest codebook filler. The
        interference-cancellation is what a single pass lacks; iteration sharpens it.

        Returns {role_idx: filler_idx}. `fillers` is the [F, D_in] codebook. `lesion` scrambles the inverse
        used in BOTH the unbind and the re-bind (structure-destroying)."""
        roles_present = list(roles_present)
        inv = (lambda r: self._scram[r]) if lesion else (lambda r: self.role_proj[r])
        # init: single-pass estimate per role (the FRLF read-out, then snap)
        est_idx = {}
        for r in roles_present:
            e = (bundle * inv(r)) @ self.W_O
            est_idx[r] = native_argmax(e, fillers)
        # learned-embed cache for the codebook (so re-bind uses the SAME learned W_F, no new params)
        embeds = fillers @ self.W_F                                         # [F, D_h]
        for _ in range(max(0, n_iters - 1)):
            new_idx = {}
            for r in roles_present:
                # residual = bundle minus the re-bound estimates of the OTHER roles (explain away)
                resid = bundle.copy()
                for r2 in roles_present:
                    if r2 == r:
                        continue
                    resid = resid - (inv(r2) * embeds[est_idx[r2]])
                e = (resid * inv(r)) @ self.W_O                             # learned cleanup on the residual
                new_idx[r] = native_argmax(e, fillers)
            if new_idx == est_idx:
                est_idx = new_idx
                break
            est_idx = new_idx
        return est_idx

    # ---- training (identical to the FRLF: single-pass loss through W_F, W_O; the role is fixed) ----
    def train_fact_step(self, roleids, fillerids, roles, fillers, t):
        self.t += 1
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [self.role_proj[r] * w for r, w in zip(roleids, ws)]
        bundle = sum(gs)
        inv = self.role_proj[roleids[t]]
        act = bundle * inv
        est = act @ self.W_O
        err = est - fillers[fillerids[t]]
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_W_O = np.outer(act, d_est)
        d_act = self.W_O @ d_est
        d_bundle = d_act * inv
        d_W_F = np.zeros_like(self.W_F)
        for (f_id, r_id) in zip(fillerids, roleids):
            d_w = d_bundle * self.role_proj[r_id]
            d_W_F += np.outer(fillers[f_id], d_w)
        self._adam("W_O", d_W_O + self.lam * self.W_O)
        self._adam("W_F", d_W_F + self.lam * self.W_F)
        return loss


# ===========================================================================
# FIXED-+-1 POSITIVE CONTROL: pure +-1 self-inverse bind + pure +-1 cleanup, NO learned W (the 0.989 ceiling).
# Proves the harness detects working bundling -> a NEGATIVE on the learned arm is real, not a broken harness.
# ===========================================================================
def fixed_pm1_positive_control(codes, seed, D_h):
    """Pure +-1 MAP/FHRR self-inverse: filler_code -> sign(filler @ P) in {+-1}^D_h (FIXED random projection,
    cosine-preserving-ish); role_proj likewise +-1; bind = role_proj (x) filler_pm1; bundle = sum; unbind =
    sign(bundle (x) role_proj) compared to each filler's +-1 code by Hamming/cosine. NO learning."""
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 71 + 5)
    P = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    filler_pm1 = np.where(fillers @ P >= 0.0, 1.0, -1.0)                    # [F, D_h] fixed +-1 codes
    R_proj = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)                   # [R, D_h]
    held = []
    for split in splits:
        ntr = nh = nh_ok = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, N_BUNDLE, replace=False)
            bundle = sum(role_pm1[r] * filler_pm1[fids[r]] for r in range(N_BUNDLE))
            for r in range(N_BUNDLE):
                est = bundle * role_pm1[r]                                  # +-1 self-inverse unbind
                # nearest +-1 filler code by dot (== max agreement)
                pred = int(np.argmax(filler_pm1 @ est))
                if (r, int(fids[r])) in set(split["train"]):
                    ntr += 1
                else:
                    nh += 1; nh_ok += int(pred == fids[r])
        held.append(nh_ok / nh if nh else 0.0)
    return float(np.mean(held)) if held else 0.0


# ===========================================================================
# Per-seed run: trains ONE binder per split per D_h, scores single-pass vs iterative (+ lesion) on held-out.
# ===========================================================================
def familiarity_gap(binder, train_combos, fillers):
    """Anti-cheat #9 (no-confab moat, free here): max-cosine of the single-pass estimate to any codebook
    filler, for KNOWN (trained) fillers vs NOVEL out-of-distribution fillers. Must stay positive."""
    rng = np.random.default_rng(54321)
    known = []
    for ri, fi in train_combos[:min(20, len(train_combos))]:
        e = binder.unbind_single(binder.bind(ri, fillers[fi]), ri)
        e = e / (np.linalg.norm(e) + 1e-12)
        known.append(float(np.max(fillers @ e)))
    novel = []
    for ri in range(min(R, 4)):
        for _ in range(5):
            nf = rng.standard_normal(fillers.shape[1]); nf = nf / (np.linalg.norm(nf) + 1e-12)
            e = binder.unbind_single(binder.bind(ri, nf), ri)
            e = e / (np.linalg.norm(e) + 1e-12)
            novel.append(float(np.max(fillers @ e)))
    mk = float(np.mean(known)) if known else 0.0
    mn = float(np.mean(novel)) if novel else 0.0
    return mk, mn, mk - mn


def run_seed(codes, seed, d_h):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    single_held, iter_held, bundle_train_single, bundle_train_iter = [], [], [], []
    iter_held_lesion, perm_role_held = [], []
    fam_gaps = []
    for split in splits:
        train_set = set(split["train"])
        binder = FixedRoleLearnedFillerIterativeBinder(D_in=D_in, roles=roles, D_h=d_h, lr=LR, seed=seed)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(N_BUNDLE)}
        if min(len(tr_by_role[r]) for r in range(N_BUNDLE)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            picks = [int(rng.choice(tr_by_role[r])) for r in range(N_BUNDLE)]
            binder.train_fact_step(list(range(N_BUNDLE)), picks, roles, fillers, int(rng.integers(N_BUNDLE)))
        # eval bundles
        ntr_s = ntr_i = nh = 0
        ntr_ok_s = ntr_ok_i = nh_ok_s = nh_ok_i = nh_ok_iL = nh_ok_perm = 0
        # permuted-role control: a fixed role->role derangement applied at READ time only
        perm = np.array([(r + 1) % N_BUNDLE for r in range(N_BUNDLE)])
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, N_BUNDLE, replace=False)
            bundle = sum(binder.bind(r, fillers[fids[r]]) for r in range(N_BUNDLE))
            it = binder.decode_iterative(bundle, range(N_BUNDLE), fillers, lesion=False)
            itL = binder.decode_iterative(bundle, range(N_BUNDLE), fillers, lesion=True)
            for r in range(N_BUNDLE):
                ok_s = int(native_argmax(binder.unbind_single(bundle, r), fillers) == fids[r])
                ok_i = int(it[r] == fids[r])
                ok_iL = int(itL[r] == fids[r])
                # permuted-role: read role r but score against the deranged filler (must collapse)
                ok_perm = int(native_argmax(binder.unbind_single(bundle, r), fillers) == fids[perm[r]])
                if (r, int(fids[r])) in train_set:
                    ntr_s += 1; ntr_ok_s += ok_s
                    ntr_i += 1; ntr_ok_i += ok_i
                else:
                    nh += 1
                    nh_ok_s += ok_s; nh_ok_i += ok_i; nh_ok_iL += ok_iL; nh_ok_perm += ok_perm
        single_held.append(nh_ok_s / nh if nh else 0.0)
        iter_held.append(nh_ok_i / nh if nh else 0.0)
        iter_held_lesion.append(nh_ok_iL / nh if nh else 0.0)
        perm_role_held.append(nh_ok_perm / nh if nh else 0.0)
        bundle_train_single.append(ntr_ok_s / ntr_s if ntr_s else 0.0)
        bundle_train_iter.append(ntr_ok_i / ntr_i if ntr_i else 0.0)
        mk, mn, gap = familiarity_gap(binder, split["train"], fillers)
        fam_gaps.append(gap)

    def mn(x):
        return float(np.mean(x)) if x else 0.0
    row = {
        "seed": seed, "D_h": d_h,
        "single_held": mn(single_held), "iter_held": mn(iter_held),
        "train_single": mn(bundle_train_single), "train_iter": mn(bundle_train_iter),
        "iter_held_lesion": mn(iter_held_lesion), "perm_role_held": mn(perm_role_held),
        "familiarity_gap": mn(fam_gaps),
    }
    print(f"  [seed {seed} D_h {d_h:>3}] single held {row['single_held']:.3f} | "
          f"ITER held {row['iter_held']:.3f} (train {row['train_iter']:.3f}) | "
          f"lesion {row['iter_held_lesion']:.3f} | perm-role {row['perm_role_held']:.3f} | "
          f"fam-gap {row['familiarity_gap']:+.3f}", flush=True)
    return row


def load_codes(path):
    codes = np.load(path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    cos = [float(codes[i] @ codes[j]) for i in range(min(F, len(codes)))
           for j in range(i + 1, min(F, len(codes)))]
    return codes, (float(np.mean(cos)) if cos else 0.0), (float(np.max(np.abs(cos))) if cos else 0.0)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    raw = os.path.join(_REPO, "research", "findings", "raw")
    clean_path = os.path.join(raw, "_phaseB_stream_codes_320_seed42.npy")
    corr_paths = {s: os.path.join(raw, f"_phaseB_stream_codes_320_neural_seed{s}.npy")
                  for s in (42, 43, 44)}
    if not os.path.exists(clean_path):
        print(f"  [missing] {clean_path}", flush=True); return

    print("[FHRR-B Option 1: LEARNED-ITERATIVE-CLEANUP + capacity sweep on the bundle inverse]", flush=True)
    print("  the bind stays a FIXED +-1 self-inverse structure; only the CLEANUP is learned + iterative "
          "(resonator explaining-away).", flush=True)
    print(f"  cited NEGATIVE arms (same harness): additive {CITED_ADDITIVE_BUNDLED}, learned-linear "
          f"{CITED_LEARNED_LINEAR_BUNDLED}, dendritic {CITED_DENDRITIC_BUNDLED}; fixed-+-1 ceiling "
          f"{FIXED_FHRR_CEILING}. FRLF single-pass base 0.639.\n", flush=True)

    clean_codes, clean_cos, clean_cosmax = load_codes(clean_path)
    print(f"  CLEAN stream codes: between-cos mean {clean_cos:.4f} (max {clean_cosmax:.3f}) -- the "
          f"decorrelated regime\n", flush=True)

    # ---- CLEAN codes: capacity sweep x 6 seeds ----
    print("  == CLEAN (decorrelated) stream codes: D_h capacity sweep x 6 seeds ==", flush=True)
    clean_rows = {}
    for d_h in D_H_SWEEP:
        clean_rows[d_h] = [run_seed(clean_codes, s, d_h) for s in SEEDS]

    # ---- fixed-+-1 positive control (CLEAN, top D_h) ----
    pc_dh = D_H_SWEEP[-1]
    pc_clean = [fixed_pm1_positive_control(clean_codes, s, pc_dh) for s in SEEDS]
    print(f"\n  fixed-+-1 POSITIVE control (CLEAN, D_h {pc_dh}): held-out per seed "
          f"{[round(x, 3) for x in pc_clean]} -> mean {np.mean(pc_clean):.3f} "
          f"(harness-sanity: must be high, ~{FIXED_FHRR_CEILING})", flush=True)

    # ---- CORRELATED codes (anti-cheat #7), top D_h, on the 3 neural seeds available ----
    print(f"\n  == CORRELATED (grounded/neural) production codes: D_h {pc_dh} (anti-cheat #7) ==", flush=True)
    corr_rows = []
    corr_cos_report = {}
    for s in (42, 43, 44):
        p = corr_paths[s]
        if not os.path.exists(p):
            print(f"  [missing correlated] {p}", flush=True); continue
        cc, ccos, ccosmax = load_codes(p)
        corr_cos_report[s] = {"mean": ccos, "max": ccosmax}
        corr_rows.append(run_seed(cc, s, pc_dh))
    pc_corr = [fixed_pm1_positive_control(load_codes(corr_paths[s])[0], s, pc_dh)
               for s in (42, 43, 44) if os.path.exists(corr_paths[s])]

    # ===================== AGGREGATE + VERDICT =====================
    chance = 1.0 / F

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    # best D_h on CLEAN by mean iterative held-out
    best_dh = max(D_H_SWEEP, key=lambda d: agg(clean_rows[d], "iter_held"))
    best = clean_rows[best_dh]
    iter_mean = agg(best, "iter_held")
    single_mean = agg(best, "single_held")
    train_iter_mean = agg(best, "train_iter")
    lesion_mean = agg(best, "iter_held_lesion")
    perm_mean = agg(best, "perm_role_held")
    fam_mean = agg(best, "familiarity_gap")
    n_seeds_ge_090 = sum(1 for r in best if r["iter_held"] >= 0.90)

    print(f"\n{'='*108}", flush=True)
    print("  CAPACITY SWEEP (CLEAN codes, 6-seed mean iterative held-out):", flush=True)
    for d_h in D_H_SWEEP:
        rr = clean_rows[d_h]
        print(f"    D_h {d_h:>3}: single {agg(rr,'single_held'):.3f} | ITER {agg(rr,'iter_held'):.3f} "
              f"(train {agg(rr,'train_iter'):.3f}) | lesion {agg(rr,'iter_held_lesion'):.3f}", flush=True)
    print(f"\n  BEST D_h on CLEAN = {best_dh}: ITER held-out {iter_mean:.3f} | single-pass {single_mean:.3f} "
          f"| FRLF-base 0.639 | train {train_iter_mean:.3f}", flush=True)
    print(f"  vs cited NEGATIVE: additive {CITED_ADDITIVE_BUNDLED} | learned-linear "
          f"{CITED_LEARNED_LINEAR_BUNDLED} | dendritic {CITED_DENDRITIC_BUNDLED} | chance {chance:.3f}", flush=True)
    print(f"  fixed-+-1 ceiling (re-run, CLEAN) {np.mean(pc_clean):.3f} | LESION {lesion_mean:.3f} | "
          f"perm-role {perm_mean:.3f} | fam-gap {fam_mean:+.3f} | seeds>=0.90: {n_seeds_ge_090}/{len(best)}",
          flush=True)
    if corr_rows:
        print(f"\n  CORRELATED codes (D_h {pc_dh}, 3 neural seeds): ITER held-out "
              f"{agg(corr_rows,'iter_held'):.3f} | single {agg(corr_rows,'single_held'):.3f} | "
              f"lesion {agg(corr_rows,'iter_held_lesion'):.3f} | between-cos "
              f"{ {s: round(v['mean'],3) for s,v in corr_cos_report.items()} }", flush=True)
        if pc_corr:
            print(f"  fixed-+-1 ceiling (CORRELATED) {np.mean(pc_corr):.3f}", flush=True)
    print(f"{'='*108}", flush=True)

    # ---- pre-registered verdict (fixed bars) ----
    go = (iter_mean >= 0.90 and iter_mean >= 0.6 * train_iter_mean and n_seeds_ge_090 >= 5
          and lesion_mean < 0.25)
    boundary = (not go) and (iter_mean >= 0.639 + 0.08)   # materially lifts the FRLF floor
    lift_over_frlf = iter_mean - 0.639
    if go:
        verdict = "GO"
        print(f"  VERDICT: GO -- the LEARNED ITERATIVE CLEANUP + fixed +-1 bind reaches fixed-algebra parity: "
              f"iterative held-out {iter_mean:.3f} >= 0.90 on {n_seeds_ge_090}/{len(best)} seeds (>> additive "
              f"{CITED_ADDITIVE_BUNDLED}, >> learned-linear {CITED_LEARNED_LINEAR_BUNDLED}, >> FRLF 0.639), "
              f"{iter_mean/max(train_iter_mean,1e-9):.0%} of train, LESION collapses to {lesion_mean:.3f}. "
              f"==> FHRR-B's read-out half CLOSES (learned + lossy + redundant cleanup); the residual shrinks "
              f"to 'the bind op is a fixed self-inverse STRUCTURE' (a structural neural primitive). "
              f"Option 4 (small guarded on-bridge wiring) is warranted -- hand the controller the GPU command.",
              flush=True)
    elif boundary:
        verdict = "BOUNDARY"
        print(f"  VERDICT: BOUNDARY -- the iterative cleanup LIFTS the bundle inverse "
              f"{0.639:.3f} -> {iter_mean:.3f} (+{lift_over_frlf:.3f}) materially over the FRLF single-pass "
              f"floor (>> additive {CITED_ADDITIVE_BUNDLED}, lesion collapses {lesion_mean:.3f}) but short of "
              f"the 0.90 parity bar. The cleanup LEARNS + capacity helps, but the bind-FORM gap is partly "
              f"fundamental at this representation. ==> NEXT MECHANISM (per the owner's rule, NOT a close): "
              f"Option 2 (deep/hidden-layer learned binder, the dendrite re-entry; gate on the BPTT ceiling "
              f"<30min/seed) AND/OR Option 3 (orthogonal-role TPR, dissolves the 2-attribute boundary). "
              f"Record this characterized partial.", flush=True)
    else:
        verdict = "NEGATIVE"
        print(f"  VERDICT: NEGATIVE -- the iterative cleanup does NOT materially beat the FRLF single-pass "
              f"floor (iterative {iter_mean:.3f} vs 0.639); the gap to the fixed ceiling is fundamental at "
              f"THIS representation (a commutative shared codebook). ==> NEXT MECHANISM (NOT a close, per the "
              f"owner's rule): Option 3 (orthogonal-role TPR -- remove the resonator's permutation symmetry; "
              f"bind each attribute under a DISTINCT named role, so unbind has no ambiguity), and/or Option 2 "
              f"(deep learned binder). The single-pass FRLF + fixed bind is the current resting point.",
              flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {
        "verdict": verdict, "best_D_h_clean": best_dh,
        "clean_iter_held_mean": iter_mean, "clean_single_held_mean": single_mean,
        "clean_train_iter_mean": train_iter_mean, "clean_lesion_mean": lesion_mean,
        "clean_perm_role_mean": perm_mean, "clean_familiarity_gap_mean": fam_mean,
        "n_seeds_ge_090": n_seeds_ge_090, "n_seeds": len(best),
        "frlf_single_pass_base": 0.639, "lift_over_frlf": lift_over_frlf,
        "fixed_pm1_ceiling_clean": float(np.mean(pc_clean)),
        "cited_additive": CITED_ADDITIVE_BUNDLED, "cited_learned_linear": CITED_LEARNED_LINEAR_BUNDLED,
        "cited_dendritic": CITED_DENDRITIC_BUNDLED, "chance": chance,
        "clean_between_cos": clean_cos,
        "capacity_sweep_clean": {str(d): {"iter_held": agg(clean_rows[d], "iter_held"),
                                          "single_held": agg(clean_rows[d], "single_held"),
                                          "train_iter": agg(clean_rows[d], "train_iter"),
                                          "lesion": agg(clean_rows[d], "iter_held_lesion"),
                                          "per_seed": clean_rows[d]} for d in D_H_SWEEP},
        "correlated": {"iter_held_mean": agg(corr_rows, "iter_held") if corr_rows else None,
                       "single_held_mean": agg(corr_rows, "single_held") if corr_rows else None,
                       "lesion_mean": agg(corr_rows, "iter_held_lesion") if corr_rows else None,
                       "fixed_pm1_ceiling": float(np.mean(pc_corr)) if pc_corr else None,
                       "between_cos": corr_cos_report, "per_seed": corr_rows},
        "config": {"R": R, "F": F, "N_BUNDLE": N_BUNDLE, "N_FACT_STEPS": N_FACT_STEPS,
                   "N_ITERS": N_ITERS, "D_H_SWEEP": list(D_H_SWEEP), "seeds": list(SEEDS),
                   "n_splits": N_SPLITS},
    }
    path = os.path.join(raw, "_phaseB_fhrr_b_learned_iterative_cleanup.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
