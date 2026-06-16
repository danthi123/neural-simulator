"""CYCLE 102 — step-3 de-risk: BUNDLE-AWARE training fixes the conversational fact structure?

The bundled-facts de-risk was NEGATIVE: a binder trained on SINGLE role-filler pairs collapses on 3-way
BUNDLED SVO facts (held-out-combo recall 0.206 ~ chance) -- the superposition crosstalk breaks the
single-pair-trained unbind. THE FIX (this de-risk): train the binder BUNDLE-AWARE -- present superposed facts
(agent+verb+object bound + summed) and train the unbind to recover EACH role's filler FROM THE BUNDLE. Does
that fix bundled recall AND still generalize to held-out (role, filler) combinations?

The bundle-aware train step: build a fact = bound(R0,f0) + bound(R1,f1) + bound(R2,f2) (ON/OFF, summed); pick a
random target role t; unbind(bundle, R_t) -> est; loss = MSE(est, f_t); backprop -- d_bundle flows EQUALLY to
all 3 binds (bundle = sum), so W_R/W_F learn to bind such that any role unbinds cleanly from the superposition.

GATE (3 seeds): bundled held-out-combo recall >> the single-pair NEGATIVE (0.206) and >> chance, AND ~ the
train-combo bundled recall (generalizes). GO => the learned bind supports conversation (with bundle-aware
training) => the on-bridge build is worthwhile. NEGATIVE => a genuine capacity limit on superposition.

Reuse-by-import (OnOffRateBinder bind/unbind + Adam + the systematicity splits); cached 320 stream codes; CPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundle_trained_derisk
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
from research.runners._phaseB_spiking_bind_onoff_derisk import OnOffRateBinder  # noqa: E402

R, F, N_SPLITS, D_H, LR, READ_NOISE = 4, 16, 3, 64, 0.005, 0.20
N_FACT_STEPS = 24000      # bundle-aware training steps (~ the single-pair epochs x combos)
N_EVAL_FACTS = 40


class BundleBinder(OnOffRateBinder):
    """OnOffRateBinder + a BUNDLE-AWARE train step: train the unbind to recover a role's filler from a 3-way
    superposed fact. The bind (W_R/W_F) is shared; d_bundle flows equally to all 3 binds (bundle = sum)."""

    def _bind_clean(self, role, filler):                      # ON/OFF bound (no read noise) + the pre-activation h
        h = role @ self.W_R + filler @ self.W_F + self.b_bind
        return np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)]), h

    def train_fact_step(self, roleids, fillerids, roles, fillers, t):
        self.t += 1
        binds, hs = [], []
        for r, f in zip(roleids, fillerids):
            b, h = self._bind_clean(roles[r], fillers[f]); binds.append(b); hs.append(h)
        bundle = self._noisy(sum(binds))                      # superposition + finite-population read
        role_h = roles[roleids[t]] @ self.W_RP
        concat = np.concatenate([bundle, role_h])
        est = self._noisy(concat @ self.W_U + self.b_unbind)
        err = est - fillers[fillerids[t]]
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_concat = self.W_U @ d_est
        d_W_U = np.outer(concat, d_est); d_b_unbind = d_est.copy()
        d_bundle = d_concat[:2 * self.D_h]; d_role_h = d_concat[2 * self.D_h:]
        d_W_RP = np.outer(roles[roleids[t]], d_role_h)
        d_W_R = np.zeros_like(self.W_R); d_W_F = np.zeros_like(self.W_F); d_b_bind = np.zeros_like(self.b_bind)
        for (r, f, h) in zip(roleids, fillerids, hs):         # d_bundle -> each bind (sum), accumulate
            d_on = d_bundle[:self.D_h]; d_off = d_bundle[self.D_h:]
            d_h = d_on * (h > 0).astype(np.float64) - d_off * (h < 0).astype(np.float64)
            d_W_R += np.outer(roles[r], d_h); d_W_F += np.outer(fillers[f], d_h); d_b_bind += d_h
        self.W_R, self.m_WR, self.v_WR = self._adam_update(self.W_R, d_W_R + self.lam * self.W_R, self.m_WR, self.v_WR)
        self.W_F, self.m_WF, self.v_WF = self._adam_update(self.W_F, d_W_F + self.lam * self.W_F, self.m_WF, self.v_WF)
        self.b_bind, self.m_bb, self.v_bb = self._adam_update(self.b_bind, d_b_bind, self.m_bb, self.v_bb)
        self.W_RP, self.m_WRP, self.v_WRP = self._adam_update(self.W_RP, d_W_RP + self.lam * self.W_RP, self.m_WRP, self.v_WRP)
        self.W_U, self.m_WU, self.v_WU = self._adam_update(self.W_U, d_W_U + self.lam * self.W_U, self.m_WU, self.v_WU)
        self.b_unbind, self.m_bu, self.v_bu = self._adam_update(self.b_unbind, d_b_unbind, self.m_bu, self.v_bu)
        return loss


def run_seed(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    bundle_train, bundle_held = [], []
    for split in splits:
        train_set = set(split["train"])
        binder = BundleBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)
        # BUNDLE-AWARE training: random SVO facts (roles 0,1,2) whose 3 bindings are all TRAIN combos
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            fa = rng.choice(tr_by_role[0]); fv = rng.choice(tr_by_role[1]); fo = rng.choice(tr_by_role[2])
            binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], roles, fillers, int(rng.integers(3)))
        # eval: bundled SVO facts; recall split by whether the queried (role, filler) was train or held-out
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(binder._bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(binder._unbind(bundle, roles[r]), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
    bt = float(np.mean(bundle_train)) if bundle_train else 0.0
    bh = float(np.mean(bundle_held)) if bundle_held else 0.0
    print(f"  [seed {seed}] BUNDLE-TRAINED recall: train-combo {bt:.3f} | held-out-combo {bh:.3f}", flush=True)
    return {"seed": seed, "bundle_train": bt, "bundle_held": bh}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[bundle-trained learned-bind de-risk] does BUNDLE-AWARE training fix bundled SVO recall + generalize? "
          f"(vs the single-pair-trained NEGATIVE held-out 0.206, chance 0.062)", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    bt, bh = m("bundle_train"), m("bundle_held")
    chance = 1.0 / F
    print(f"\n{'='*94}\n  MEAN (3 seeds): BUNDLE-TRAINED recall train-combo {bt:.3f} | held-out-combo {bh:.3f} | "
          f"chance {chance:.3f} (single-pair-trained held-out was 0.206)", flush=True)
    print(f"{'='*94}", flush=True)
    if bh >= 0.40 and bh >= 0.6 * bt:
        print(f"  GO: BUNDLE-AWARE training FIXES it -- bundled held-out-combo recall {bh:.3f} (>> single-pair 0.206, "
              f">> chance {chance:.3f}), {bh/max(bt,1e-9):.0%} of the train-combo {bt:.3f}. The learned bind, trained "
              f"on superpositions, recovers each role's filler from a 3-way bundle AND generalizes. ==> the learned "
              f"bind supports the conversational fact structure -> the on-bridge build is worthwhile.", flush=True)
    elif bh >= 0.20:
        print(f"  PARTIAL: bundle-aware training helps (held-out {bh:.3f} vs single-pair 0.206) but not decisive -- "
              f"more capacity (D_h) / cleanup / fewer simultaneous bindings; superposition capacity is the limit.",
              flush=True)
    else:
        print(f"  NEGATIVE: even bundle-aware training can't separate the superposition ({bh:.3f}) -- a genuine "
              f"capacity limit (D_h={D_H} too small for 3-way bundles at F={F}); needs more capacity or a cleanup "
              f"stage in the loop.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"bundle_train": bt, "bundle_held": bh, "chance": chance, "single_pair_held": 0.206, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_learned_bind_bundle_trained.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
