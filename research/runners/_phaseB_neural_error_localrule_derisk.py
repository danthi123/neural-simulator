"""On-substrate local read-out rule, biologization STEP 3c (cheap-first, numpy) — neuralise the per-output TEACHING
ERROR. The on-bridge read-out is now learned by real synaptic plasticity (6-seed GO, 2026-06-17-onsubstrate-readout-
rule-bridge-GO.md), but the per-output error `err_j = target_j - est_j` it learns from is still a HOST subtraction.
This de-risk asks: does a NEURAL error -- a predictive-coding error population (two error neurons per output firing
relu(target-est) and relu(est-target), rate-coded + Poisson-noisy), the signed error = ON_rate - OFF_rate -- drive
the SAME delta-rule learning as the exact host error? The subtraction is then done by the error neuron's excitatory
(target) minus inhibitory (prediction) inputs (the standard Rao-Ballard / Bastos predictive-coding error unit), not
a host formula. The target itself is an env/teacher scaffold (legitimate, like a supervised teaching signal); what
we neuralise here is the SUBTRACTION (target - prediction) that yields the per-output error.

This isolates the error-neuralisation (exact input act; only the error is rate-coded ON/OFF) from the input-rate-code
(already GO, CYCLE 157). Arms (6 seeds, systematicity protocol, bundled held-out):
  1. EXACT-ERR (reference)   -- the host delta rule (err = est - target exact). ~1.000.
  2. NEURAL-ERR (the de-risk) -- err from the ON/OFF rate-coded error population (Poisson + rectified).
  3. SCRAMBLED (anti-cheat)  -- neural err permuted across outputs -> must COLLAPSE (the per-output error is load-bearing).

GO = NEURAL-ERR >= 0.85x EXACT-ERR in >=5/6 seeds AND systematicity holds AND scrambled collapses (<0.5x neural).
Reuse-by-import (LocalRuleBinder + the systematicity harness). CPU/numpy.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_neural_error_localrule_derisk [--seeds ...] [--err-gain 20]
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
from research.runners._phaseB_localrule_readout_derisk import LocalRuleBinder  # noqa: E402

R, F, N_SPLITS = 4, 16, 3
N_FACT_STEPS = 24000
N_EVAL_FACTS = 40


class NeuralErrorBinder(LocalRuleBinder):
    """LocalRuleBinder but the per-output error is computed by a PREDICTIVE-CODING error population: for each output
    j, an ON error neuron fires ~relu(target_j - est_j) and an OFF error neuron ~relu(est_j - target_j) (rate-coded
    with Poisson spike-count noise over a window = err_gain); the signed teaching error fed to the delta rule is
    ON_rate - OFF_rate. The subtraction target-prediction is done by the error neuron (exc target - inh prediction),
    not a host formula. err_gain = spikes per unit error (the window / SNR knob)."""

    def __init__(self, D_in, role_pm1, D_h, seed, err_gain=20.0, scramble_err=False, **kw):
        super().__init__(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed, **kw)
        self.err_gain = float(err_gain)
        self.scramble_err = bool(scramble_err)
        self._rng = np.random.default_rng(seed * 131 + 9)
        self._perm = self._rng.permutation(D_in) if scramble_err else None

    def _neural_err(self, est, target):
        # predictive-coding error: ON = relu(target - est), OFF = relu(est - target); rate-coded + Poisson noise.
        diff = target - est
        on = np.maximum(diff, 0.0); off = np.maximum(-diff, 0.0)
        scale = np.mean(np.abs(diff)) + 1e-9
        on_r = self._rng.poisson(np.clip(on / scale, 0.0, None) * self.err_gain) / self.err_gain
        off_r = self._rng.poisson(np.clip(off / scale, 0.0, None) * self.err_gain) / self.err_gain
        return (on_r - off_r) * scale          # signed teaching error (target - est), neuralised

    def train_fact_step(self, roleids, fillerids, fillers, t):
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [self.role_pm1[r] * w for r, w in zip(roleids, ws)]
        bundle = sum(gs)
        act = bundle * self.role_pm1[roleids[t]]
        est = act @ self.W_O
        neg_err = self._neural_err(est, fillers[fillerids[t]])     # = target - est (neuralised); host LMS uses est-target
        err = -neg_err                                            # est - target, to match the host delta sign
        if self.scramble_err:
            err = err[self._perm]
        self.W_O -= self.lr * (np.outer(act, err) + self.lam * self.W_O)


def _eval(binder, split, fillers, train_set, rng):
    single = sum(int(native_argmax(binder.unbind(binder.bind(r, fillers[f]), r), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
    h_ok = h = 0
    for _ in range(N_EVAL_FACTS):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(binder.bind(r, fillers[int(fids[r])]) for r in range(3))
        for r in range(3):
            if (r, int(fids[r])) not in train_set:
                h_ok += int(native_argmax(binder.unbind(bundle, r), fillers) == fids[r]); h += 1
    return single, (h_ok / h if h else 0.0)


def run_seed(codes, seed, err_gain, n_steps):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; d_in = fillers.shape[1]
    roles = make_role_codes(R, d_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((d_in, 256)) / np.sqrt(d_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    exact_h, neural_h, scr_h, neural_s = [], [], [], []
    for split in splits:
        tr = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr[r]) for r in range(3)) == 0:
            continue
        train_set = set(split["train"])

        def _train(binder):
            rr = np.random.default_rng(seed * 53 + 9)
            for _ in range(n_steps):
                fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
                binder.train_fact_step([0, 1, 2], [int(fa), int(fv), int(fo)], fillers, int(rr.integers(3)))
            return binder

        exact = _train(LocalRuleBinder(D_in=d_in, role_pm1=role_pm1, D_h=256, seed=seed))
        neural = _train(NeuralErrorBinder(D_in=d_in, role_pm1=role_pm1, D_h=256, seed=seed, err_gain=err_gain))
        scr = _train(NeuralErrorBinder(D_in=d_in, role_pm1=role_pm1, D_h=256, seed=seed, err_gain=err_gain,
                                       scramble_err=True))
        _, e_h = _eval(exact, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        n_s, n_h = _eval(neural, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        _, s_h = _eval(scr, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        exact_h.append(e_h); neural_h.append(n_h); scr_h.append(s_h); neural_s.append(n_s)
    row = {"seed": seed, "exact": float(np.mean(exact_h)), "neural": float(np.mean(neural_h)),
           "scramble": float(np.mean(scr_h)), "neural_single": float(np.mean(neural_s))}
    print(f"  [seed {seed}] EXACT-err {row['exact']:.3f} | NEURAL-err {row['neural']:.3f} "
          f"(single {row['neural_single']:.3f}) | scramble {row['scramble']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--err-gain", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=N_FACT_STEPS)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_neural_error_localrule.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    t0 = time.time()
    print(f"[neural-error local-rule de-risk] does a predictive-coding ERROR population (ON/OFF rate-coded "
          f"target-prediction) drive the same delta-rule learning as the exact host error? gain={args.err_gain} "
          f"seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, args.err_gain, args.steps) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    exact, neural, scr, neural_s = m("exact"), m("neural"), m("scramble"), m("neural_single")
    n_par = sum(int(r["neural"] >= 0.85 * r["exact"]) for r in rows)
    bar = int(np.ceil(5 / 6 * len(seeds)))
    scramble_collapses = scr < 0.5 * max(neural, 1e-9)
    go = (n_par >= bar) and (neural_s >= 0.6) and scramble_collapses
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): EXACT-err {exact:.3f} | NEURAL-err {neural:.3f} (single {neural_s:.3f}) | "
          f"scramble {scr:.3f} | neural>=0.85x exact: {n_par}/{len(seeds)}", flush=True)
    if go:
        print(f"  GO: a NEURAL predictive-coding error population (ON/OFF rate-coded target-prediction) drives the "
              f"delta rule as well as the exact host error -- neural {neural:.3f} = {neural/max(exact,1e-9):.0%} of "
              f"exact in {n_par}/{len(seeds)} seeds, systematicity holds (single {neural_s:.3f}), scrambled-error "
              f"collapses ({scr:.3f}). ==> the per-output teaching SUBTRACTION is neuralisable (exc target - inh "
              f"prediction); the read-out learning's last host scaffold is removable. On-bridge realization next.",
              flush=True)
    elif not scramble_collapses:
        print(f"  INVALID: scrambled-error did NOT collapse ({scr:.3f} vs {neural:.3f}) -- re-examine.", flush=True)
    else:
        print(f"  BOUNDARY: the neural error degrades learning ({neural:.3f} vs exact {exact:.3f}) -- the error "
              f"population needs more spikes (window) or a cleaner ON/OFF read.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else ("INVALID" if not scramble_collapses else "BOUNDARY"),
           "seeds": seeds, "err_gain": args.err_gain, "exact": exact, "neural": neural, "neural_single": neural_s,
           "scramble": scr, "n_parity": n_par, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
