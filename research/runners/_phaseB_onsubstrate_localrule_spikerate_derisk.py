"""On-substrate local-rule read-out, STEP 3b cheap-first (numpy) — does the binder's LOCAL delta rule still reach
parity when the presynaptic activity it reads is a SPIKING-RATE code (noisy, rectified ON/OFF, quantized) instead
of an exact signed vector?

CONTEXT. CYCLE 153 (`2026-06-17-localrule-readout-NEF-GO.md`) proved a LOCAL delta rule (Widrow-Hoff/LMS:
dW_O = -lr * outer(act, err), a three-factor pre x post-error rule) learns the binder read-out decoder to Adam
parity (1.000) -- but with `act` an EXACT signed numpy vector. The residual is realizing that rule ON THE SPIKING
SUBSTRATE, where the unbind population's activity reaches the decoder as FIRING RATES: non-negative, split into
ON/OFF channels (the project's standard signed->rate trick, used by the on-bridge binder CYCLE 150), and corrupted
by per-presentation spike-count (Poisson) noise + quantization. This probe isolates the MECHANISM-AGNOSTIC part of
the on-substrate question (every per-output teaching mechanism feeds the SAME delta rule; what differs on-substrate
is the noisy ON/OFF rate code of `act`) BEFORE paying for a full LIF bridge. If the delta rule collapses under a
realistic spiking-rate code, that is a cheap NEGATIVE that saves the GPU build; if it holds, the on-bridge build
(deliver the per-output error as a climbing-fiber / supervised-clamp teaching signal -> the bridge three-factor
plasticity) is green-lit.

ARMS (identical data, 6 seeds, the systematicity protocol; held-out generalization):
  1. ADAM-both EXACT (reference upper bound)               -- the Step-1/2 binder, exact signed act.
  2. DELTA-Wo EXACT (NEF host local rule, CYCLE-153 1.000) -- exact signed act; the host local-rule reference.
  3. DELTA-Wo SPIKE-RATE (the substrate-faithful arm)      -- act -> ON/OFF Poisson spike-count rate code, then the
     SAME delta rule. Swept over spike_gain (spikes-per-unit-std = the window-length / SNR knob; the documented
     "more spikes = cleaner read-out" lever).
  4. ANTI-CHEAT scrambled-error                            -- arm-3 with the per-output error permuted across outputs
     before the update -> must COLLAPSE to floor (proves the per-output teaching error is load-bearing, not noise).

GO = arm-3 (spike-rate delta) reaches >= 0.85x arm-1 (Adam exact) in >= 5/6 seeds at the best spike_gain AND
single-binding systematicity holds (generalizes to held-out role-filler combos) AND the scrambled-error anti-cheat
collapses (< 0.5x arm-3). Reuse-by-import (LocalRuleBinder + the systematicity harness). CPU/numpy.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onsubstrate_localrule_spikerate_derisk \
          [--dh 256] [--spike-gains 5,20,80] [--seeds 42,43,44,100,101,102]
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
from research.runners._phaseB_fixed_role_learned_filler_bundling_derisk import FixedRoleLearnedFillerBinder  # noqa: E402
from research.runners._phaseB_localrule_readout_derisk import LocalRuleBinder  # noqa: E402

R, F, N_SPLITS = 4, 16, 3
N_FACT_STEPS = 24000
N_EVAL_FACTS = 40
LR_DELTA = 0.02      # plain-LMS step (exact-act reference arm)
NLMS_LR = 0.5        # normalized-LMS step (spike-rate arm; the normalizer divides out the input scale)


class SpikeRateLocalRuleBinder(LocalRuleBinder):
    """Fixed +-1 role + fixed-random encoder W_F + LOCAL delta-rule decoder W_O, but the unbind pre-activation `act`
    reaches the decoder as a SPIKING-RATE code: ON/OFF rectified (signed -> two non-negative channels), then a
    Poisson spike count over a window (spike_gain = spikes per unit of act-std = the SNR / window knob), divided
    back to a rate estimate. W_O is over the [2*D_h] ON/OFF channels. Per-presentation fresh spike noise (faithful
    to the bridge: each presentation is a fresh noisy read-out). scramble_err permutes the per-output error before
    the update (the anti-cheat)."""

    def __init__(self, D_in, role_pm1, D_h, seed, lr_delta=NLMS_LR, lam=1e-4, spike_gain=20.0, scramble_err=False):
        super().__init__(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed, lr_delta=lr_delta, lam=lam)
        self.W_O = np.zeros((2 * D_h, D_in))                 # decoder over ON/OFF rate channels
        self.spike_gain = float(spike_gain)
        self.scramble_err = bool(scramble_err)
        self._rng = np.random.default_rng(seed * 101 + 7)     # spike-count noise
        self._perm = self._rng.permutation(D_in) if scramble_err else None

    def _rate(self, act):
        """signed act [D_h] -> ON/OFF non-negative [2*D_h] -> Poisson spike-count rate code (noisy + quantized)."""
        scale = np.std(act) + 1e-9
        x = np.concatenate([np.maximum(act, 0.0), np.maximum(-act, 0.0)]) / scale   # ON/OFF, unit-std-ish
        counts = self._rng.poisson(np.clip(x, 0.0, None) * self.spike_gain)         # integer spikes (Poisson noise)
        return counts.astype(np.float64) / self.spike_gain                          # rate estimate [2*D_h]

    def unbind(self, bundle, role_id):
        # nan_to_num guard: the scramble (anti-cheat) arm's wrong-error updates can diverge W_O -> NaN; map to 0 so
        # it reads as chance (the intended collapse) instead of warning-spamming. The real arm is NLMS-stable (no NaN).
        return np.nan_to_num(self._rate(bundle * self.role_pm1[role_id]) @ self.W_O)

    def train_fact_step(self, roleids, fillerids, fillers, t):
        ws = [fillers[f] @ self.W_F for f in fillerids]
        gs = [self.role_pm1[r] * w for r, w in zip(roleids, ws)]
        bundle = sum(gs)
        rate = self._rate(bundle * self.role_pm1[roleids[t]])      # SPIKING-RATE pre-activation [2*D_h]
        est = rate @ self.W_O                                      # [D_in]
        err = est - fillers[fillerids[t]]                          # per-output post-error [D_in]
        if self.scramble_err:
            err = err[self._perm]                                  # anti-cheat: error no longer matches its output
        # NORMALIZED delta rule (NLMS): scale the pre x post-error update by the presynaptic input power. Plain LMS
        # diverges when the input scale changes (the ON/OFF rate code has 2x dims + Poisson variance vs the exact
        # signed act); normalizing by ||rate||^2 makes it stable for any input scale. Biologically = homeostatic
        # synaptic scaling (the bridge has it); the update is still LOCAL (pre x post-error / a local power estimate).
        norm = float(rate @ rate) + 1e-6
        self.W_O -= (self.lr / norm) * np.outer(rate, err) + self.lr * self.lam * self.W_O


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


def run_seed(codes, seed, D_h, spike_gains, n_steps=N_FACT_STEPS):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    adam_h, nef_h, nef_s = [], [], []
    sr_h = {g: [] for g in spike_gains}; sr_s = {g: [] for g in spike_gains}
    scr_h = []   # anti-cheat at the best (largest) gain
    best_g = max(spike_gains)
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

        adam = _train(FixedRoleLearnedFillerBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_h, lr=0.005, lam=1e-4, seed=seed))
        nef = _train(LocalRuleBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed))
        a_s, a_h = _eval(adam, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        n_s, n_h = _eval(nef, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        adam_h.append(a_h); nef_h.append(n_h); nef_s.append(n_s)
        for g in spike_gains:
            sb = _train(SpikeRateLocalRuleBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed, spike_gain=g))
            s_s, s_h = _eval(sb, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
            sr_h[g].append(s_h); sr_s[g].append(s_s)
        scr = _train(SpikeRateLocalRuleBinder(D_in=D_in, role_pm1=role_pm1, D_h=D_h, seed=seed,
                                              spike_gain=best_g, scramble_err=True))
        _, c_h = _eval(scr, split, fillers, train_set, np.random.default_rng(seed * 7 + 1))
        scr_h.append(c_h)
    row = {"seed": seed, "adam": float(np.mean(adam_h)), "nef_exact": float(np.mean(nef_h)),
           "nef_single": float(np.mean(nef_s)), "scramble": float(np.mean(scr_h)),
           "spike_rate": {str(g): float(np.mean(sr_h[g])) for g in spike_gains},
           "spike_rate_single": {str(g): float(np.mean(sr_s[g])) for g in spike_gains}}
    sr_str = " ".join(f"g{g}={row['spike_rate'][str(g)]:.3f}" for g in spike_gains)
    print(f"  [seed {seed} D_h={D_h}] ADAM {row['adam']:.3f} | DELTA-exact {row['nef_exact']:.3f} | "
          f"DELTA-spikerate[{sr_str}] | scramble {row['scramble']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--spike-gains", type=str, default="5,20,80")
    ap.add_argument("--steps", type=int, default=N_FACT_STEPS)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onsubstrate_localrule_spikerate.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    gains = [float(g) for g in args.spike_gains.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    t0 = time.time()
    print(f"[on-substrate local-rule spike-rate de-risk] does the delta rule survive an ON/OFF Poisson spike-rate "
          f"`act` at D_h={args.dh}? gains={gains} seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, args.dh, gains, n_steps=args.steps) for s in seeds]

    adam = float(np.mean([r["adam"] for r in rows]))
    nef = float(np.mean([r["nef_exact"] for r in rows]))
    scramble = float(np.mean([r["scramble"] for r in rows]))
    best_g = max(gains)
    sr_best = float(np.mean([r["spike_rate"][str(best_g)] for r in rows]))
    sr_best_single = float(np.mean([r["spike_rate_single"][str(best_g)] for r in rows]))
    n_par = sum(int(r["spike_rate"][str(best_g)] >= 0.85 * r["adam"]) for r in rows)
    bar = int(np.ceil(5 / 6 * len(seeds)))
    print(f"\n{'='*108}", flush=True)
    for g in gains:
        gm = float(np.mean([r["spike_rate"][str(g)] for r in rows]))
        print(f"  spike_gain {g:>5}: DELTA-spikerate {gm:.3f}  ({gm/max(adam,1e-9):.0%} of Adam)", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): ADAM {adam:.3f} | DELTA-exact {nef:.3f} | "
          f"DELTA-spikerate@g{best_g} {sr_best:.3f} (single {sr_best_single:.3f}) | scramble {scramble:.3f} | "
          f"spikerate>=0.85x Adam: {n_par}/{len(seeds)}", flush=True)
    scramble_collapses = scramble < 0.5 * sr_best
    go = (n_par >= bar) and (sr_best_single >= 0.6) and scramble_collapses
    if go:
        print(f"  GO: the LOCAL delta rule survives a realistic SPIKING-RATE pre-activation (ON/OFF Poisson "
              f"spike-count code) -- spike-rate read-out {sr_best:.3f} = {sr_best/max(adam,1e-9):.0%} of Adam in "
              f"{n_par}/{len(seeds)} seeds, systematicity holds (single {sr_best_single:.3f}), and the scrambled-"
              f"error anti-cheat collapses ({scramble:.3f} << {sr_best:.3f} = the per-output teaching error is "
              f"load-bearing). ==> the mechanism-agnostic part of the on-substrate question is GREEN; the on-bridge "
              f"build (deliver the per-output error as a teaching signal -> the bridge three-factor plasticity) is "
              f"green-lit.", flush=True)
    elif not scramble_collapses:
        print(f"  INVALID: the scrambled-error anti-cheat did NOT collapse ({scramble:.3f} vs {sr_best:.3f}) -- the "
              f"spike-rate read-out is succeeding without a correct per-output error; re-examine before any build.",
              flush=True)
    else:
        print(f"  BOUNDARY/NEGATIVE: the spike-rate code degrades the delta rule ({sr_best:.3f} vs Adam {adam:.3f}, "
              f"{n_par}/{len(seeds)} parity) -- the read-out needs more spikes (longer window), more decoder "
              f"capacity, or a noise-robust rule before the on-substrate build is justified.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": "GO" if go else ("INVALID" if not scramble_collapses else "BOUNDARY"),
           "D_h": args.dh, "seeds": seeds, "spike_gains": gains, "adam": adam, "delta_exact": nef,
           "spike_rate_best": sr_best, "spike_rate_best_single": sr_best_single, "best_gain": best_g,
           "scramble": scramble, "n_parity": n_par, "pass_bar": bar, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
