"""CYCLE 196 — the UNTESTED MIDDLE of the learned-bind capability map: a FIXED self-inverse role + LEARNED filler.

The settled map (identical harness): a LEARNED bind can't bundle a 3-way superposition on point neurons -- additive
0.193, and a learned-LINEAR-inverse multiplicative 0.056/0.000 (a linear map can't be a reciprocal). A FIXED +-1
self-inverse bind bundles (0.989) -- but with a FIXED-RANDOM filler projection and NO held-out-combo split. The one
UNTESTED cell (the scoping `2026-06-18-step3-dendritic-learned-bind-frontier-scoping.md`): does a FIXED self-inverse
role + a LEARNED filler projection (trained bundle-aware) STILL bundle AND generalize to held-out (role, filler)
combos? Learning the filler embedding could COLLAPSE the near-orthogonality the fixed bind's bundling needs (a
subtle but real failure mode), or it could be compatible -- which would validate the production design ("learned
codes flowing through a FIXED coincidence primitive") on a measured, leakage-controlled signal.

This binder: role_proj[r] = sign(role @ R_proj) in {+-1}^D_h (FIXED, self-inverse -- the EXACT inverse, no learned
W_Rinv); bind g = role_proj[r] (x) (filler @ W_F) [W_F LEARNED]; bundle = sum g; unbind act = bundle (x) role_proj[r]
(the +-1 self-inverse); est = act @ W_O [W_O LEARNED] -> nearest filler. Trained bundle-aware (backprop through W_F,
W_O only; the role is fixed). (x) = elementwise product.

GATE (3 seeds, stream codes, F=16; the EXACT systematicity harness): bundled held-out-combo >= 0.40 AND >=
0.6x train-combo AND single-binding held-out >= 0.40, while the additive (0.193) + learned-linear (0.056) NEGATIVEs
stand on the same harness and the fixed-+-1 control (0.989) carries. GO => a LEARNED filler embedding is COMPATIBLE
with the fixed self-inverse bind for bundling+generalization -- the production design ("learned codes + a FIXED
coincidence primitive") is the validated, principled resting point; the prior bundled NEGATIVEs were about LEARNING
the bind, not the codes. NEGATIVE => learning the filler embedding COLLAPSES the fixed bind's bundling (the bind
needs near-orthogonal fillers that bundle-aware learning destroys) -- a real, narrow boundary. Either is the
deliverable; it closes the learned-bind corner on a clean signal.

ANTI-CHEAT (mirrors the established 0.989-vs-0.193 contrast): the point-neuron additive/learned-linear arms MUST
fall short on the SAME corpus/splits/seeds (cited, not re-run here -- 0.193 / 0.056); held-out is a leakage-asserted
combo split (make_systematicity_splits) vs the memorization floor; a LESION control (replace the +-1 self-inverse
unbind with a SUM, i.e. drop the multiplicative inverse) must COLLAPSE the bundled recall (the product is
load-bearing).

Reuse-by-import (make_role_codes / make_systematicity_splits / native_argmax); cached 320 stream codes; CPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_role_learned_filler_bundled_derisk
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

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)

R, F, N_SPLITS, D_H, LR = 4, 16, 3, 64, 0.005
N_FACT_STEPS = 24000
N_EVAL_FACTS = 40


class FixedRoleLearnedFillerBinder:
    """A FIXED +-1 self-inverse role (the EXACT inverse) + a LEARNED filler projection W_F + learned readout W_O.
    bind g = role_proj[r] (x) (filler @ W_F); unbind act = bundle (x) role_proj[r]; est = act @ W_O. Trained
    bundle-aware -- only W_F, W_O learn; the role is fixed. `lesion_sum` (control) replaces the (x) self-inverse
    unbind with a plain SUM (drop the multiplicative inverse) -> the product is no longer load-bearing."""

    _PARAMS = ("W_F", "W_O")

    def __init__(self, D_in, roles, D_h=D_H, lr=LR, lam=1e-4, seed=42, lesion_sum=False):
        self.D_in, self.D_h, self.lr, self.lam = D_in, D_h, lr, lam
        self.lesion_sum = bool(lesion_sum)
        rng = np.random.default_rng(seed * 17 + 3)
        R_proj = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
        self.role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)         # [R, D_h] FIXED +-1 self-inverse
        s_in, s_h = 1.0 / np.sqrt(D_in), 1.0 / np.sqrt(D_h)
        self.W_F = rng.standard_normal((D_in, D_h)) * s_in
        self.W_O = rng.standard_normal((D_h, D_in)) * s_h
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

    def bind(self, role_idx, filler):
        return self.role_proj[role_idx] * (filler @ self.W_F)              # g [D_h] (fixed-role multiplicative)

    def unbind(self, bundle, role_idx):
        inv = np.ones(self.D_h) if self.lesion_sum else self.role_proj[role_idx]   # lesion: drop the +-1 inverse
        return (bundle * inv) @ self.W_O                                   # cleanup readout [D_in]

    def train_fact_step(self, roleids, fillerids, roles, fillers, t):
        self.t += 1
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [self.role_proj[r] * w for r, w in zip(roleids, ws)]
        bundle = sum(gs)
        inv = np.ones(self.D_h) if self.lesion_sum else self.role_proj[roleids[t]]
        act = bundle * inv
        est = act @ self.W_O
        err = est - fillers[fillerids[t]]
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_W_O = np.outer(act, d_est)
        d_act = self.W_O @ d_est
        d_bundle = d_act * inv                                             # act = bundle (x) inv (inv fixed)
        d_W_F = np.zeros_like(self.W_F)
        for (f_id, r_id) in zip(fillerids, roleids):
            d_w = d_bundle * self.role_proj[r_id]                          # g = role_proj (x) (filler@W_F)
            d_W_F += np.outer(fillers[f_id], d_w)
        self._adam("W_O", d_W_O + self.lam * self.W_O)
        self._adam("W_F", d_W_F + self.lam * self.W_F)
        return loss


def run_seed(codes, seed, lesion_sum=False):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    single_held, bundle_train, bundle_held = [], [], []
    for split in splits:
        train_set = set(split["train"])
        binder = FixedRoleLearnedFillerBinder(D_in=D_in, roles=roles, D_h=D_H, lr=LR, seed=seed, lesion_sum=lesion_sum)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
            binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], roles, fillers, int(rng.integers(3)))
        sc = sum(int(native_argmax(binder.unbind(binder.bind(r, fillers[f]), r), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(binder.bind(r, fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(binder.unbind(bundle, r), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
    sh = float(np.mean(single_held)) if single_held else 0.0
    bt = float(np.mean(bundle_train)) if bundle_train else 0.0
    bh = float(np.mean(bundle_held)) if bundle_held else 0.0
    tag = "LESION(sum)" if lesion_sum else "fixed-role+learned-filler"
    print(f"  [seed {seed}] {tag}: single held-out {sh:.3f} | BUNDLED train-combo {bt:.3f} | held-out-combo {bh:.3f}",
          flush=True)
    return {"seed": seed, "single_held": sh, "bundle_train": bt, "bundle_held": bh, "lesion": lesion_sum}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print("[fixed-role + learned-filler bundled de-risk] does a FIXED self-inverse role + a LEARNED filler embedding "
          "bundle + generalize? (the untested MIDDLE; additive 0.193, learned-linear 0.056, fixed-on-both 0.989)\n",
          flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]
    print("  -- LESION control (drop the +-1 self-inverse unbind -> a plain sum; the product must be load-bearing) --",
          flush=True)
    lesion = [run_seed(codes, s, lesion_sum=True) for s in (42, 43, 44)]

    def m(rs, k):
        return float(np.mean([r[k] for r in rs]))
    sh, bt, bh = m(rows, "single_held"), m(rows, "bundle_train"), m(rows, "bundle_held")
    lbh = m(lesion, "bundle_held")
    chance = 1.0 / F
    print(f"\n{'='*100}\n  MEAN (3 seeds): fixed-role+learned-filler single held-out {sh:.3f} | BUNDLED train {bt:.3f} | "
          f"held-out {bh:.3f} | LESION(sum) held-out {lbh:.3f} | chance {chance:.3f}", flush=True)
    print(f"{'='*100}", flush=True)
    go = bh >= 0.40 and bh >= 0.6 * bt and sh >= 0.40
    lesion_collapses = lbh < 0.25
    if go and lesion_collapses:
        print(f"  GO: a LEARNED filler embedding is COMPATIBLE with the FIXED self-inverse bind -- bundled held-out "
              f"{bh:.3f} (>> additive 0.193, >> learned-linear 0.056, >> chance {chance:.3f}), {bh/max(bt,1e-9):.0%} "
              f"of train-combo {bt:.3f}, single generalizes {sh:.3f}; the LESION (sum, no inverse) collapses to "
              f"{lbh:.3f} so the multiplicative self-inverse is load-bearing. ==> the production design 'LEARNED "
              f"codes flowing through a FIXED coincidence primitive' is the validated, principled resting point; the "
              f"prior bundled NEGATIVEs were about LEARNING THE BIND, not the codes. The learned-bind corner CLOSES.",
              flush=True)
    elif go and not lesion_collapses:
        print(f"  AMBIGUOUS: bundled held-out {bh:.3f} GO but the LESION(sum) did NOT collapse ({lbh:.3f}) -- the "
              f"multiplicative inverse may not be load-bearing on this harness; investigate the lesion before "
              f"claiming the product is the lever.", flush=True)
    else:
        print(f"  NEGATIVE: learning the filler embedding COLLAPSES the fixed bind's bundling (held-out {bh:.3f} < "
              f"0.40) -- bundle-aware learning of W_F destroys the near-orthogonality the +-1 bundling needs. A real, "
              f"narrow boundary: the fixed bind wants FIXED (or orthogonality-preserving) filler codes, not a freely "
              f"learned embedding. ==> the production composer's GIVEN stream codes (not a re-learned embedding) are "
              f"the right input to the fixed bind.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"single_held": sh, "bundle_train": bt, "bundle_held": bh, "lesion_sum_held": lbh, "chance": chance,
           "additive_bundled": 0.193, "learned_linear_bundled": 0.056, "fixed_both_bundled": 0.989,
           "go": bool(go and lesion_collapses), "per_seed": rows, "lesion": lesion}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_fixed_role_learned_filler_bundled.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
