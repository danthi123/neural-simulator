"""CYCLE 103 — the bundled NEGATIVE localizes here: does a LEARNED MULTIPLICATIVE bind recover superposition?

CYCLE 102 established (multi-seed) that a LEARNED ADDITIVE bind + LINEAR unbind generalizes single role-filler
bindings (held-out 0.806) but CANNOT separate a 3-way BUNDLED fact even trained bundle-aware (held-out 0.193 ~
chance 0.062). The reason is structural, not capacity: unbinding role t from a superposition needs the
role-specific INVERSE applied to the bundle -- a MULTIPLICATION (g -> g / u_t) -- which a point-neuron additive
bind + a shared LINEAR unbind provably cannot implement (the inverse is role-dependent; a fixed linear map can't
scale the bound by a role-dependent factor). MULTIPLICATION is a DENDRITIC operation (two-compartment neuron, on
the bridge) -- the same point-neuron limit the project keeps hitting (Mikulasch-Priesemann: whitening/decorrelation
is analog/dendritic; here, binding-superposition is multiplicative/dendritic).

THIS de-risk: a LEARNED FHRR-style bind -- MULTIPLICATIVE bind g = (role@W_R) (x) (filler@W_F), MULTIPLICATIVE
unbind r = bundle (x) (role@W_Rinv) [a learned approximate inverse role], + a linear cleanup readout est = r@W_O.
Trained bundle-aware (backprop through the two Hadamard products). (x) = elementwise product = the dendritic op.
Does multiplication give the LEARNED bind the superposition capacity the additive bind lacks, AND still
generalize systematically to held-out (role, filler) combos?

GATE (3 seeds, stream codes, F=16): bundled held-out-combo recall >> the additive NEGATIVE (0.193) and >> chance
(0.062), AND single-binding held-out generalizes (>> mem-floor). GO => a LEARNED multiplicative bind bundles +
generalizes => realize it on the two-compartment DENDRITIC substrate (the genuine superposition-capable learned
bind). PARTIAL => multiplicative bind helps but needs a multiplicative cleanup / resonator in the loop. NEGATIVE
=> multiplication alone is not enough; superposition needs an iterative cleanup (resonator network). Localize.

Reuse-by-import (the systematicity protocol: make_role_codes / make_systematicity_splits / native_argmax);
cached 320 stream codes; CPU; no GPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_multiplicative_bind_bundled_derisk
"""
from __future__ import annotations

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
N_FACT_STEPS = 24000      # bundle-aware training steps (matches the additive bundle-trained de-risk)
N_EVAL_FACTS = 40


class MultFHRRBinder:
    """LEARNED FHRR-style bind: MULTIPLICATIVE bind g = u (x) w (u = role@W_R, w = filler@W_F), MULTIPLICATIVE
    unbind act = bundle (x) uinv (uinv = role@W_Rinv = a learned approximate inverse role), + linear cleanup
    readout est = act@W_O. (x) = elementwise (Hadamard) product = the dendritic operation. Trained bundle-aware
    by backprop through the two products. Tests whether MULTIPLICATION gives the learned bind the superposition
    capacity the additive point-neuron bind provably lacks (CYCLE 102 NEGATIVE 0.193)."""

    _PARAMS = ("W_R", "W_F", "W_Rinv", "W_O")

    def __init__(self, D_in, D_h=D_H, lr=LR, lam=1e-4, seed=42, read_noise=0.0):
        self.D_in, self.D_h, self.lr, self.lam = D_in, D_h, lr, lam
        self.read_noise = read_noise
        rng = np.random.default_rng(seed * 17 + 3)
        s_in, s_h = 1.0 / np.sqrt(D_in), 1.0 / np.sqrt(D_h)
        self.W_R = rng.standard_normal((D_in, D_h)) * s_in
        self.W_F = rng.standard_normal((D_in, D_h)) * s_in
        self.W_Rinv = rng.standard_normal((D_in, D_h)) * s_in
        self.W_O = rng.standard_normal((D_h, D_in)) * s_h
        self._rng = np.random.default_rng(seed * 7 + 1)
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

    def _noisy(self, x):
        if self.read_noise <= 0:
            return x
        return x + self._rng.standard_normal(x.shape) * self.read_noise * (float(np.std(x)) + 1e-9)

    def bind(self, role, filler):
        return (role @ self.W_R) * (filler @ self.W_F)              # g [D_h] (multiplicative)

    def unbind(self, bundle, role):
        act = self._noisy(bundle * (role @ self.W_Rinv))           # multiplicative unbind
        return act @ self.W_O                                      # linear cleanup readout [D_in]

    def train_fact_step(self, roleids, fillerids, roles, fillers, t):
        self.t += 1
        us = [roles[r] @ self.W_R for r in roleids]
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [u * w for u, w in zip(us, ws)]
        bundle = sum(gs)                                           # superposition [D_h]
        uinv = roles[roleids[t]] @ self.W_Rinv
        act = self._noisy(bundle * uinv)                           # multiplicative unbind of role t
        est = act @ self.W_O
        err = est - fillers[fillerids[t]]
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_W_O = np.outer(act, d_est)
        d_act = self.W_O @ d_est
        d_bundle = d_act * uinv                                    # d(bundle*uinv)/d bundle
        d_uinv = d_act * bundle                                    # d(bundle*uinv)/d uinv
        d_W_Rinv = np.outer(roles[roleids[t]], d_uinv)
        d_W_R = np.zeros_like(self.W_R); d_W_F = np.zeros_like(self.W_F)
        for (r_id, f_id, u, w) in zip(roleids, fillerids, us, ws):
            d_g = d_bundle                                         # bundle = sum gs -> d_g_i = d_bundle (each)
            d_u = d_g * w; d_w = d_g * u                           # g = u (x) w
            d_W_R += np.outer(roles[r_id], d_u)
            d_W_F += np.outer(fillers[f_id], d_w)
        self._adam("W_O", d_W_O + self.lam * self.W_O)
        self._adam("W_Rinv", d_W_Rinv + self.lam * self.W_Rinv)
        self._adam("W_R", d_W_R + self.lam * self.W_R)
        self._adam("W_F", d_W_F + self.lam * self.W_F)
        return loss


def run_seed(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    single_held, bundle_train, bundle_held = [], [], []
    for split in splits:
        train_set = set(split["train"])
        binder = MultFHRRBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=0.0)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        for _ in range(N_FACT_STEPS):                              # BUNDLE-AWARE training (SVO roles 0,1,2)
            fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
            binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], roles, fillers, int(rng.integers(3)))
        # single-binding held-out (does the multiplicative bind still GENERALIZE?)
        sc = sum(int(native_argmax(binder.unbind(binder.bind(roles[r], fillers[f]), roles[r]), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)
        # bundled SVO recall, split by whether the queried (role, filler) was a train or held-out combo
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(binder.bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(binder.unbind(bundle, roles[r]), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
    sh = float(np.mean(single_held)) if single_held else 0.0
    bt = float(np.mean(bundle_train)) if bundle_train else 0.0
    bh = float(np.mean(bundle_held)) if bundle_held else 0.0
    print(f"  [seed {seed}] MULT-FHRR: single-binding held-out {sh:.3f} | BUNDLED train-combo {bt:.3f} | "
          f"held-out-combo {bh:.3f}", flush=True)
    return {"seed": seed, "single_held": sh, "bundle_train": bt, "bundle_held": bh}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[multiplicative-bind bundled de-risk] does a LEARNED MULTIPLICATIVE (FHRR-style) bind recover "
          f"superposition where the additive bind failed? (additive bundled held-out 0.193, chance 0.062)",
          flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    sh, bt, bh = m("single_held"), m("bundle_train"), m("bundle_held")
    chance = 1.0 / F
    print(f"\n{'='*98}\n  MEAN (3 seeds): MULT-FHRR single-binding held-out {sh:.3f} | BUNDLED train-combo {bt:.3f} | "
          f"held-out-combo {bh:.3f} | chance {chance:.3f} (additive bundled held-out was 0.193)", flush=True)
    print(f"{'='*98}", flush=True)
    if bh >= 0.40 and bh >= 0.6 * bt and sh >= 0.40:
        print(f"  GO: a LEARNED MULTIPLICATIVE bind RECOVERS superposition -- bundled held-out-combo {bh:.3f} "
              f"(>> additive 0.193, >> chance {chance:.3f}), {bh/max(bt,1e-9):.0%} of train-combo {bt:.3f}, AND "
              f"single-binding generalizes {sh:.3f}. Multiplication (the DENDRITIC op) is the lever the additive "
              f"point-neuron bind lacked. ==> realize it on the two-compartment dendritic substrate (the genuine "
              f"superposition-capable LEARNED bind).", flush=True)
    elif bh >= 0.25:
        print(f"  PARTIAL: multiplication HELPS (bundled held-out {bh:.3f} vs additive 0.193) but isn't decisive -- "
              f"add a multiplicative CLEANUP / resonator iteration in the unbind loop, or more capacity. "
              f"Superposition capacity is partly recovered by the dendritic op.", flush=True)
    else:
        print(f"  NEGATIVE: even a multiplicative bind doesn't separate the superposition ({bh:.3f}) -- multiplication "
              f"alone is insufficient; superposition needs an ITERATIVE cleanup (resonator network) in the loop. "
              f"Localize there, OR keep the fixed FHRR algebra for bundling (production default, V=320).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"single_held": sh, "bundle_train": bt, "bundle_held": bh, "chance": chance,
           "additive_bundled_held": 0.193, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_multiplicative_bind_bundled.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
