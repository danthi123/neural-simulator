"""CYCLE 144 cheap-first A/B — does a FIXED self-inverse role + LEARNED filler codes recover BUNDLED facts?

THE CONFOUND THIS RESOLVES (from `_phaseB_multiplicative_bind_bundled_derisk.py`, CYCLE 103):
that arm tried to LEARN a LINEAR map `W_Rinv` as the superposition-unbind inverse -- but a linear map provably
cannot be the reciprocal/self-inverse a superposition-unbind needs (the role-dependent 1/u multiplication a
summing soma lacks). It collapsed even single-attribute (0.056, BROKEN -- not "multiplication doesn't help").

THE ONE CHANGE (the genuinely-untested middle): use a FIXED self-inverse role (the +-1 / conjugate-phasor
self-inverse the production composer ALREADY uses) for the bind/unbind, but LEARN the filler codes (and the
read-out). This is the un-explored point between "everything learned-linear" (0.056, broken) and "everything
fixed +-1" (0.989, the composer's known operating point). QUESTION: does fixed-self-inverse-role + LEARNED-filler
recover bundled multi-attribute facts where the learned-linear inverse could not -- AND still generalize
systematically to held-out (role, filler) combos?

FOUR ARMS ON IDENTICAL DATA (same facts, loads, splits, metrics) -- a clean A/B vs the 0.056 / 0.193 / 0.989:
  (1) FIXED-ROLE + LEARNED-FILLER   [NEW]      -- bind = role_pm1 (x) (filler@W_F); unbind = bundle (x) role_pm1;
                                                  cleanup = nearest learned-filler-repr; W_F + W_O learned bundle-aware.
  (2) LEARNED-LINEAR (MultFHRRBinder)          -- must stay ~0.056 (learned linear inverse can't be a reciprocal).
  (3) ADDITIVE (OnOffRateBinder, bundled)      -- must stay ~0.193 (point-neuron additive bind, no inverse).
  (4) FIXED +-1 FHRR (no training)             -- ~0.989 (the ceiling; the harness DETECTS working bundling).
chance = 1/F = 0.062.

GO  = arm-1 bundled held-out >= 0.40 in >=5/6 seeds WHILE additive AND learned-linear stay NEGATIVE (< ~0.25) on
      identical data AND held-out systematicity holds AND permuted/lesion collapse.
      Honest framing: a fixed self-inverse role is what the composer ALREADY does at 0.989 -- so a GO lifts the
      LEARNED-CODES boundary (fixed role + LEARNED fillers works), it is NOT "multiplication-from-scratch is new".
BOUNDARY = beats learned-linear/additive but well below 0.989, or seed-fragile.
NEGATIVE = arm-1 does NOT beat learned-linear/additive -> even the fixed product can't carry LEARNED fillers in
      superposition (the wall is deeper than the linear-inverse confound).

Reuse-by-import (the systematicity protocol + the three reference binders). Cached 320 stream codes. CPU; no GPU;
no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_role_learned_filler_bundling_derisk
      [--seeds 42,43,44,100,101,102]
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

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)
from research.runners._phaseB_multiplicative_bind_bundled_derisk import MultFHRRBinder  # noqa: E402
from research.runners._phaseB_spiking_bind_onoff_derisk import OnOffRateBinder  # noqa: E402

# Match the reference harnesses VERBATIM so this is a clean A/B vs 0.056 / 0.193 / 0.989.
R, F, N_SPLITS, D_H, LR = 4, 16, 3, 64, 0.005
N_FACT_STEPS = 24000      # bundle-aware training steps (== the additive + learned-linear de-risks)
N_EVAL_FACTS = 40         # bundled eval facts per split (== the learned-linear de-risk)
ADDITIVE_N_EPOCHS = 600   # OnOffRateBinder training epochs (== its de-risk)
ADDITIVE_READ_NOISE = 0.20
N_ABSTAIN = 60            # no-confab / abstention probes


class FixedRoleLearnedFillerBinder:
    """FIXED self-inverse role (+-1 hypervector) + LEARNED filler codes (+ learned read-out cleanup).

    role_pm1[r] in {+-1}^D_h  : a FIXED random projection of the role, BINARIZED -> its OWN inverse under (x)
                                (exactly the production composer's self-inverse primitive; NOT trained).
    filler_repr = filler @ W_F : the LEARNED filler code in bind-space [D_h]  (W_F trained).
    bind   g = role_pm1 (x) (filler @ W_F)                            (x) = elementwise product = the dendritic op.
    bundle = sum of binds.
    unbind act = bundle (x) role_pm1                                  (the +-1 self-inverse -> filler_repr + crosstalk)
    read-out  est = act @ W_O                                         (LEARNED cleanup: maps [D_h] -> [D_in])
    cleanup: nearest ORIGINAL filler code by cosine (native_argmax over `fillers` -- the SAME codebook as the
             learned-linear arm, so this is a clean A/B). W_O is trained jointly bundle-aware to regress the unbind
             back to the queried filler's original [D_in] code despite the 3-way crosstalk.

    The ONLY difference vs the learned-linear arm (MultFHRRBinder): the bind/unbind product uses a FIXED
    self-inverse role (a reciprocal BY CONSTRUCTION), not a learned linear `W_Rinv` (which can't be a reciprocal).
    The fillers + read-out are LEARNED (the genuinely-untested middle).
    """

    _PARAMS = ("W_F", "W_O")

    def __init__(self, D_in, role_pm1, D_h=D_H, lr=LR, lam=1e-4, seed=42):
        self.D_in, self.D_h, self.lr, self.lam = D_in, D_h, lr, lam
        self.role_pm1 = role_pm1                                   # [R, D_h] in {+-1} -- FIXED, not learned
        rng = np.random.default_rng(seed * 17 + 3)
        s_in = 1.0 / np.sqrt(D_in)
        self.W_F = rng.standard_normal((D_in, D_h)) * s_in        # filler -> bind-space repr (LEARNED)
        self.W_O = rng.standard_normal((D_h, D_in)) * s_in        # unbind act -> output repr (LEARNED read-out)
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

    def bind(self, role_id, filler):
        return self.role_pm1[role_id] * (filler @ self.W_F)        # g [D_h] (fixed-role multiplicative)

    def unbind(self, bundle, role_id):
        act = bundle * self.role_pm1[role_id]                      # (x) +-1 self-inverse -> filler_repr + crosstalk
        return act @ self.W_O                                      # learned cleanup read-out -> [D_in]

    def train_fact_step(self, roleids, fillerids, fillers, t):
        """One bundle-aware step: build the 3-way bundle, unbind role t, regress to filler_t's ORIGINAL code.
        Mirrors MultFHRRBinder.train_fact_step EXACTLY (same cleanup codebook = the original `fillers`), but the
        role product is the FIXED +-1 self-inverse (no learned W_R/W_Rinv). W_O is the learned read-out cleanup
        that maps the unbind activation [D_h] back to the original [D_in] filler-code space."""
        self.t += 1
        ws = [fillers[f] @ self.W_F for f in fillerids]            # [D_h] each (learned)
        gs = [self.role_pm1[r] * w for r, w in zip(roleids, ws)]   # fixed-role bind
        bundle = sum(gs)                                           # superposition [D_h]
        act = bundle * self.role_pm1[roleids[t]]                   # unbind role t (fixed self-inverse)
        est = act @ self.W_O                                       # [D_in]
        err = est - fillers[fillerids[t]]                          # regress to the ORIGINAL filler code [D_in]
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_W_O = np.outer(act, d_est)
        d_act = self.W_O @ d_est                                   # [D_h]
        d_bundle = d_act * self.role_pm1[roleids[t]]               # d(bundle * pm1)/d bundle = pm1
        # bundle = sum_i gs_i ; gs_i = pm1_i (x) (f_i @ W_F)  -> d gs_i = d_bundle ; d w_i = d_bundle * pm1_i
        d_W_F = np.zeros_like(self.W_F)
        for (r_id, f_id) in zip(roleids, fillerids):
            d_w = d_bundle * self.role_pm1[r_id]                   # d gs_i / d w_i
            d_W_F += np.outer(fillers[f_id], d_w)
        self._adam("W_O", d_W_O + self.lam * self.W_O)
        self._adam("W_F", d_W_F + self.lam * self.W_F)
        return loss


# ---------------------------------------------------------------------------
# Arm 1 — fixed-role + learned-filler (the NEW de-risk arm), with all controls.
# ---------------------------------------------------------------------------

def run_arm1(codes, seed, role_pm1_perm=False, lesion_sum=False):
    """Fixed-role + learned-filler bundled recovery + systematicity + controls.

    role_pm1_perm : PERMUTED-ROLE control -- bind/train with the TRUE role hypervectors but UNBIND query role r
                    using a DIFFERENT role's hypervector (a derangement of the 3 SVO roles). This genuinely breaks
                    the self-inverse bind<->unbind correspondence (unbinding role 0's slot with role 1's +-1 vector
                    -> noise), so recall must collapse to ~chance. (Permuting the role consistently on BOTH bind
                    AND unbind self-cancels -- +-1 is its own inverse for ANY vector -- which is why the unbind-only
                    permutation is the correct control.)
    lesion_sum    : LESION -- replace the unbind product (bundle (x) pm1) with a plain SUM (drop the multiplicative
                    self-inverse). The bundling lift must collapse to ~the additive value.
    Returns per-split-aggregated dict.
    """
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]
    D_in = fillers.shape[1]
    rng = np.random.default_rng(seed * 53 + 9)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    # FIXED +-1 self-inverse role hypervectors (same construction as the fixed-FHRR control):
    roles = make_role_codes(R, D_in, seed)
    R_proj = rng_pm1.standard_normal((D_in, D_H)) / np.sqrt(D_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)          # [R, D_h] in {+-1}
    # A derangement of ALL R roles for the permuted-role control (unbind with a DIFFERENT role's vector):
    unbind_map = [(r + 1) % R for r in range(R)]                   # r -> r+1 mod R (no fixed point for R>1)

    single_held, bundle_train, bundle_held = [], [], []
    abst_known, abst_novel, abst_breach = [], [], []
    for split in splits:
        train_set = set(split["train"])
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        binder = FixedRoleLearnedFillerBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_H, lr=LR, lam=1e-4, seed=seed)
        for _ in range(N_FACT_STEPS):                              # BUNDLE-AWARE training (SVO roles 0,1,2)
            fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
            binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], fillers, int(rng.integers(3)))

        # Cleanup codebook = the ORIGINAL filler codes (identical to the learned-linear arm -> clean A/B).
        # bind ALWAYS uses the true role vector; the permuted control corrupts only the UNBIND role.
        def _bind(r_id, f):
            return role_pm1[r_id] * (fillers[f] @ binder.W_F)

        def _unbind(bundle, r_id):
            if lesion_sum:
                act = bundle                                      # LESION: drop the (x) pm1 self-inverse (sum only)
            else:
                ur = unbind_map[r_id] if role_pm1_perm else r_id  # PERMUTED-ROLE: unbind with a DIFFERENT role's +-1
                act = bundle * role_pm1[ur]
            return act @ binder.W_O

        # single-binding held-out (does fixed-role + learned-filler still GENERALIZE?)
        sc = sum(int(native_argmax(_unbind(_bind(r, f), r), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)

        # bundled SVO recall, split by whether the queried (role, filler) was a train or held-out combo
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(_bind(r, int(fids[r])) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(_unbind(bundle, r), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)

        # MOAT (no-confab): an ABSENT filler (random unit code not in the codebook) -> abstain (low confidence).
        # confidence = max cosine of the unbind estimate to the original filler codebook. A familiarity gate placed
        # at the NATURAL separation point (midway between the known and novel confidence means) should accept knowns
        # and reject novels; a "breach" = a novel query whose confidence exceeds that mid-gap threshold (it would be
        # wrongly accepted). This is the honest "can a familiarity gate separate seen from absent fillers?" test --
        # NOT tuned on the result (the threshold is the data's own midpoint).
        if not (role_pm1_perm or lesion_sum):
            kc, nc = [], []
            rng_ab = np.random.default_rng(seed * 911 + 7)
            fn = np.linalg.norm(fillers, axis=1) + 1e-12          # cleanup codebook = original fillers
            for r, f in list(split["train"])[:N_ABSTAIN]:
                est = _unbind(_bind(r, f), r); est /= (np.linalg.norm(est) + 1e-12)
                kc.append(float(np.max(fillers @ est / fn)))
            for _ in range(N_ABSTAIN):
                novel = rng_ab.standard_normal(D_in); novel /= (np.linalg.norm(novel) + 1e-12)
                r = int(rng_ab.integers(3))
                bundle = role_pm1[r] * (novel @ binder.W_F)       # bind the ABSENT filler
                est = _unbind(bundle, r); est /= (np.linalg.norm(est) + 1e-12)
                nc.append(float(np.max(fillers @ est / fn)))
            thr = 0.5 * (float(np.mean(kc)) + float(np.mean(nc))) if (kc and nc) else 0.0
            breach = int(np.sum([c >= thr for c in nc]))          # novels that clear the mid-gap familiarity gate
            abst_known.append(float(np.mean(kc)) if kc else 0.0)
            abst_novel.append(float(np.mean(nc)) if nc else 0.0)
            abst_breach.append(breach)

    out = {
        "single_held": float(np.mean(single_held)) if single_held else 0.0,
        "bundle_train": float(np.mean(bundle_train)) if bundle_train else 0.0,
        "bundle_held": float(np.mean(bundle_held)) if bundle_held else 0.0,
    }
    if abst_known:
        out["abst_known"] = float(np.mean(abst_known))
        out["abst_novel"] = float(np.mean(abst_novel))
        out["abst_breach"] = int(np.sum(abst_breach))
    return out


# ---------------------------------------------------------------------------
# Baseline arms (re-run VERBATIM on identical data) — the A/B against 0.056 / 0.193 / 0.989.
# ---------------------------------------------------------------------------

def run_learned_linear(codes, seed):
    """Arm 2 — MultFHRRBinder (learned linear inverse). Must stay ~0.056 (single + bundled)."""
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    single_held, bundle_held = [], []
    from research.runners.cortex_learned_binder_systematicity_probe import native_argmax
    for split in splits:
        train_set = set(split["train"])
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        b = MultFHRRBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=0.0)
        for _ in range(N_FACT_STEPS):
            fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
            b.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], roles, fillers, int(rng.integers(3)))
        sc = sum(int(native_argmax(b.unbind(b.bind(roles[r], fillers[f]), roles[r]), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)
        nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(b.bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(b.unbind(bundle, roles[r]), fillers) == fids[r])
                if (r, int(fids[r])) not in train_set:
                    nh_ok += ok; nh += 1
        bundle_held.append(nh_ok / nh if nh else 0.0)
    return {"single_held": float(np.mean(single_held)) if single_held else 0.0,
            "bundle_held": float(np.mean(bundle_held)) if bundle_held else 0.0}


def run_additive(codes, seed):
    """Arm 3 — OnOffRateBinder additive bind, bundled. Must stay ~0.193."""
    from research.runners.cortex_learned_binder_systematicity_probe import native_argmax
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 71 + 13)
    bundle_held = []
    for split in splits:
        train_set = set(split["train"])
        b = OnOffRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=ADDITIVE_READ_NOISE)
        b.train(split["train"], roles, fillers, n_epochs=ADDITIVE_N_EPOCHS,
                batch_size=max(1, len(split["train"]) // 4))
        nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bound = sum(b._bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(b._unbind(bound, roles[r]), fillers) == fids[r])
                if (r, int(fids[r])) not in train_set:
                    nh_ok += ok; nh += 1
        bundle_held.append(nh_ok / nh if nh else 0.0)
    return {"bundle_held": float(np.mean(bundle_held)) if bundle_held else 0.0}


def run_fixed_pm1(codes, seed):
    """Arm 4 — FIXED +-1 FHRR (no training). The ceiling, ~0.989."""
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 31 + 5)
    R_proj = rng.standard_normal((D_in, D_H)) / np.sqrt(D_in)
    F_proj = rng.standard_normal((D_in, D_H)) / np.sqrt(D_in)
    role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    filler_repr = fillers @ F_proj
    fr_unit = filler_repr / (np.linalg.norm(filler_repr, axis=1, keepdims=True) + 1e-12)

    def cleanup(vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return int(np.argmax(fr_unit @ v))

    bun_ok = bun_n = 0
    for _ in range(N_EVAL_FACTS):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(role_proj[r] * filler_repr[fids[r]] for r in range(3))
        for r in range(3):
            bun_ok += int(cleanup(bundle * role_proj[r]) == fids[r]); bun_n += 1
    return {"bundled": bun_ok / bun_n if bun_n else 0.0}


def run_seed(codes, seed):
    a1 = run_arm1(codes, seed)
    a1_perm = run_arm1(codes, seed, role_pm1_perm=True)
    a1_lesion = run_arm1(codes, seed, lesion_sum=True)
    ll = run_learned_linear(codes, seed)
    add = run_additive(codes, seed)
    fx = run_fixed_pm1(codes, seed)
    row = {
        "seed": seed,
        "frlf_single": a1["single_held"], "frlf_btrain": a1["bundle_train"], "frlf_bheld": a1["bundle_held"],
        "frlf_perm_bheld": a1_perm["bundle_held"], "frlf_lesion_bheld": a1_lesion["bundle_held"],
        "abst_known": a1.get("abst_known", 0.0), "abst_novel": a1.get("abst_novel", 0.0),
        "abst_breach": a1.get("abst_breach", 0),
        "learned_linear_single": ll["single_held"], "learned_linear_bheld": ll["bundle_held"],
        "additive_bheld": add["bundle_held"], "fixed_pm1_bheld": fx["bundled"],
    }
    print(f"  [seed {seed}] FR+LF bundled held-out {row['frlf_bheld']:.3f} (single {row['frlf_single']:.3f}, "
          f"train {row['frlf_btrain']:.3f}) | learned-linear {row['learned_linear_bheld']:.3f} | "
          f"additive {row['additive_bheld']:.3f} | fixed-pm1 {row['fixed_pm1_bheld']:.3f}", flush=True)
    print(f"           controls: PERMUTED-role {row['frlf_perm_bheld']:.3f} | LESION(sum) {row['frlf_lesion_bheld']:.3f} | "
          f"moat known {row['abst_known']:.3f}/novel {row['abst_novel']:.3f} breach {row['abst_breach']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw",
                                         "_phaseB_fixed_role_learned_filler_bundling.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    chance = 1.0 / F
    print(f"[fixed-role + learned-filler bundling A/B] does a FIXED self-inverse role + LEARNED filler codes "
          f"recover BUNDLED facts where the learned-LINEAR inverse (0.056) and additive (0.193) could not? "
          f"(fixed-pm1 ceiling 0.989, chance {chance:.3f})", flush=True)
    print(f"  seeds={seeds}  N_FACT_STEPS={N_FACT_STEPS}  N_EVAL_FACTS={N_EVAL_FACTS}  D_h={D_H}  F={F}", flush=True)
    rows = [run_seed(codes, s) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    frlf = m("frlf_bheld"); frlf_single = m("frlf_single"); frlf_train = m("frlf_btrain")
    ll = m("learned_linear_bheld"); ll_single = m("learned_linear_single")
    add = m("additive_bheld"); fx = m("fixed_pm1_bheld")
    perm = m("frlf_perm_bheld"); lesion = m("frlf_lesion_bheld")
    abst_known = m("abst_known"); abst_novel = m("abst_novel")
    breach_total = int(np.sum([r["abst_breach"] for r in rows]))

    n_pass = sum(int(r["frlf_bheld"] >= 0.40) for r in rows)
    n_beats_baselines = sum(int(r["frlf_bheld"] >= 0.40
                                and r["learned_linear_bheld"] < 0.25
                                and r["additive_bheld"] < 0.25) for r in rows)
    systematicity_ok = frlf_single >= 0.40 and frlf_single >= 0.6 * max(frlf_train, 1e-9)
    controls_collapse = perm <= 0.25 and lesion <= max(add + 0.10, 0.25)
    # Moat is a PLUS (not a hard gate per owner 2026-06-17): require a clear known>novel familiarity separation,
    # not zero breaches at the data-midpoint threshold.
    moat_ok = abst_known >= abst_novel + 0.10

    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds) -- BUNDLED held-out-combo:", flush=True)
    print(f"    FIXED-ROLE + LEARNED-FILLER : {frlf:.3f}   (single {frlf_single:.3f}, train-combo {frlf_train:.3f})", flush=True)
    print(f"    learned-linear (MultFHRR)   : {ll:.3f}   (single {ll_single:.3f})   [prior 0.056]", flush=True)
    print(f"    additive (OnOff)            : {add:.3f}                            [prior 0.193]", flush=True)
    print(f"    fixed +-1 FHRR (ceiling)    : {fx:.3f}                            [prior 0.989]", flush=True)
    print(f"    chance                      : {chance:.3f}", flush=True)
    print(f"  CONTROLS: permuted-role {perm:.3f} | lesion(sum) {lesion:.3f} | "
          f"systematicity {'OK' if systematicity_ok else 'FAIL'}", flush=True)
    print(f"  MOAT (PLUS): conf known {abst_known:.3f} vs novel {abst_novel:.3f} (gap {abst_known-abst_novel:+.3f}, "
          f"separation {'OK' if moat_ok else 'WEAK'}); breaches@mid-gap {breach_total}", flush=True)
    print(f"  pass>=0.40: {n_pass}/{len(seeds)} | pass AND baselines-stay-NEGATIVE: {n_beats_baselines}/{len(seeds)}", flush=True)
    print(f"{'='*104}", flush=True)

    # Verdict: fractional >=5/6 bar (or >=5/6 of however many seeds were run). Moat is a PLUS, not a GO blocker.
    bar = int(np.ceil(5 / 6 * len(seeds)))
    go = (n_beats_baselines >= bar and systematicity_ok and controls_collapse)
    boundary = (n_pass >= bar and frlf >= max(ll, add) + 0.10 and not go)
    if go:
        verdict = "GO"
        print(f"  GO ({n_beats_baselines}/{len(seeds)} >= {bar}): a FIXED self-inverse role + LEARNED filler codes "
              f"RECOVERS bundled superposition ({frlf:.3f}) where the learned-LINEAR inverse ({ll:.3f}) and "
              f"additive ({add:.3f}) could NOT, on identical data; systematicity holds (single held-out "
              f"{frlf_single:.3f}); permuted-role ({perm:.3f}) + lesion ({lesion:.3f}) collapse; moat separation "
              f"{abst_known:.3f}>{abst_novel:.3f}. ==> LIFTS THE LEARNED-CODES BOUNDARY: fixed role + LEARNED fillers "
              f"works. HONEST FRAMING: a fixed self-inverse role is what the production composer ALREADY does at "
              f"{fx:.3f} -- so this proves the LEARNED-FILLER version holds, it is NOT multiplication-from-scratch. "
              f"Justifies the weeks-scale on-bridge spiking realization (route the LEARNED filler codes through the "
              f"coincidence-plateau self-inverse bind).", flush=True)
    elif boundary:
        verdict = "BOUNDARY"
        print(f"  BOUNDARY: fixed-role + learned-filler beats learned-linear/additive ({frlf:.3f} vs {ll:.3f}/"
              f"{add:.3f}) but is below the fixed-pm1 ceiling ({fx:.3f}) or seed-fragile "
              f"({n_beats_baselines}/{len(seeds)} < {bar}). The lever is real but the LEARNED fillers cost "
              f"accuracy vs the fully-fixed algebra -- localize (more capacity / a multiplicative cleanup) before "
              f"committing the on-bridge build.", flush=True)
    else:
        verdict = "NEGATIVE"
        print(f"  NEGATIVE: fixed-role + learned-filler ({frlf:.3f}) does NOT decisively beat learned-linear "
              f"({ll:.3f}) / additive ({add:.3f}) at the >=5/6 bar -- even the fixed-product bind can't carry "
              f"LEARNED fillers in superposition. The wall is DEEPER than the linear-inverse confound; the fixed "
              f"FHRR algebra (fixed on BOTH sides, {fx:.3f}) stays load-bearing for bundling, and the learned-bind "
              f"frontier is closed for multi-attribute facts. Does NOT justify the dendritic on-bridge build for "
              f"this op.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)

    out = {
        "verdict": verdict,
        "seeds": seeds, "n_seeds": len(seeds), "pass_bar": bar,
        "n_pass": n_pass, "n_beats_baselines": n_beats_baselines,
        "frlf_bundled_held": frlf, "frlf_single_held": frlf_single, "frlf_bundle_train": frlf_train,
        "learned_linear_bundled_held": ll, "learned_linear_single_held": ll_single,
        "additive_bundled_held": add, "fixed_pm1_bundled": fx, "chance": chance,
        "permuted_role_bheld": perm, "lesion_sum_bheld": lesion,
        "moat_conf_known": abst_known, "moat_conf_novel": abst_novel,
        "moat_separation_gap": abst_known - abst_novel, "moat_ok": bool(moat_ok),
        "moat_breaches_at_midgap": breach_total, "systematicity_ok": bool(systematicity_ok),
        "controls_collapse": bool(controls_collapse),
        "ref_prior": {"learned_linear": 0.056, "additive": 0.193, "fixed_pm1": 0.989},
        "per_seed": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
