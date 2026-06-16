"""CYCLE 99 — step-3 cheap-first #1: does RATE-CODE READ NOISE break the learned binder's systematicity?

The deep-research scoping (2026-06-16) flagged the single biggest risk of the on-substrate (spiking) learned
binder: the rate-code SNR wall -- a value read from a FINITE neural population carries noise ~ 1/sqrt(n_per),
which could push the binder's held-out (systematic) accuracy toward the memorization floor. It also clarified
(verified) that the de-risked binder is ADDITIVE (tanh(role@W_R + filler@W_F)), so the spiking realization is
just synaptic projections + a saturating nonlinearity + POPULATION reads of the bound vector + the unbind
estimate. THIS de-risk isolates the population-read-noise risk BEFORE building the full spiking binder: train
the additive binder CLEAN, then score held-out under finite-population read noise (a PESSIMISTIC test -- a
binder trained WITH noise would be more robust). Sweep the noise <-> implied n_per (noise_frac ~ 1/sqrt(n_per)).

If held-out stays well above the memorization floor at REALISTIC population sizes (n_per ~ 12-32, the project's
NEF/population-code range) -> the rate-code wall does NOT break the learned binding -> the spiking realization
is viable (next: a spiking binder trained by surrogate-BPTT / three-factor). If it collapses -> the rate-code
noise is the wall, and the honest fallback is learned codes + the fixed FHRR op (validated to V=320).

Reuse-by-import (the validated systematicity protocol helpers + BilinearBinder); the cached 320 stream codes;
CPU; no GPU; no sim/ edits; no protected file changes. The ONLY change vs the de-risked binder is read noise.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_ratenoise_derisk
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
    BilinearBinder, make_role_codes, make_systematicity_splits, score_bilinear_unbind, score_memorization_floor)

R, F, N_SPLITS, N_EPOCHS, D_H, LR = 4, 16, 3, 500, 64, 0.005
NOISE_FRACS = (0.0, 0.10, 0.20, 0.30)     # ~ 1/sqrt(n_per): n_per = inf, 100, 25, 11
N_NOISE_TRIALS = 6                        # average the stochastic held-out scoring over trials


class NoisyReadBinder(BilinearBinder):
    """The de-risked additive binder, with optional finite-POPULATION read noise on the bound vector + the
    unbind estimate (set `read_noise` > 0 at SCORE time; train with it 0). Models the rate-code read of the
    bound/estimate from a population of ~1/read_noise^2 neurons. Nothing else changes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read_noise = 0.0
        self._rng = np.random.default_rng(0)

    def _add(self, v):
        if self.read_noise <= 0.0:
            return v
        return v + self._rng.standard_normal(v.shape) * self.read_noise * (float(np.std(v)) + 1e-9)

    def _bind(self, role, filler):
        return self._add(super()._bind(role, filler))

    def _unbind(self, bound, role):
        return self._add(super()._unbind(bound, role))


def run_seed(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    per_noise = {nf: [] for nf in NOISE_FRACS}
    mem_floors, cleans = [], []
    for split in splits:
        binder = NoisyReadBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed)
        binder.read_noise = 0.0
        binder.train(split["train"], roles, fillers, n_epochs=N_EPOCHS,
                     batch_size=max(1, len(split["train"]) // 4), verbose=False)
        cleans.append(score_bilinear_unbind(binder, split["held_out"], roles, fillers))
        mem_floors.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])
        for nf in NOISE_FRACS:
            binder.read_noise = nf
            binder._rng = np.random.default_rng(seed * 1000 + int(nf * 100))
            accs = [score_bilinear_unbind(binder, split["held_out"], roles, fillers) for _ in range(N_NOISE_TRIALS)]
            per_noise[nf].append(float(np.mean(accs)))
    out = {nf: float(np.mean(per_noise[nf])) for nf in NOISE_FRACS}
    memf = float(np.mean(mem_floors))
    print(f"  [seed {seed}] held-out vs read-noise: " +
          " | ".join(f"nf={nf:.2f} {out[nf]:.3f}" for nf in NOISE_FRACS) + f" | mem-floor {memf:.3f}", flush=True)
    return {"seed": seed, "held_by_noise": out, "mem_floor": memf}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[learned-bind rate-noise de-risk] stream codes {codes.shape} -- does finite-population READ NOISE "
          f"break the additive learned binder's systematicity? (noise_frac ~ 1/sqrt(n_per): "
          f"0.10->n_per~100, 0.20->~25, 0.30->~11)", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]
    memf = float(np.mean([r["mem_floor"] for r in rows]))
    mean_by_noise = {nf: float(np.mean([r["held_by_noise"][nf] for r in rows])) for nf in NOISE_FRACS}
    print(f"\n{'='*96}\n  MEAN (3 seeds) held-out vs read-noise: " +
          " | ".join(f"nf={nf:.2f} {mean_by_noise[nf]:.3f}" for nf in NOISE_FRACS) +
          f" | mem-floor {memf:.3f} | chance {1.0/F:.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    clean, n30 = mean_by_noise[0.0], mean_by_noise[0.30]
    n20 = mean_by_noise[0.20]
    if n20 >= memf + 0.25 and n20 >= 0.6 * clean:
        print(f"  GO: the additive learned binding SURVIVES realistic population read noise -- at n_per~25 "
              f"(noise 0.20) held-out {n20:.3f} stays well above mem-floor {memf:.3f} ({n20/max(clean,1e-9):.0%} of "
              f"the clean {clean:.3f}); even at n_per~11 (noise 0.30) {n30:.3f}. ==> the rate-code wall does NOT "
              f"break the learned bind -> the on-substrate spiking binder (surrogate-BPTT / three-factor) is "
              f"worth building.", flush=True)
    elif n20 >= memf + 0.10:
        print(f"  PARTIAL: read noise degrades but does not destroy systematicity (n_per~25 held-out {n20:.3f} vs "
              f"mem-floor {memf:.3f}) -- a larger population (n_per>=32) or noise-aware training is needed; the "
              f"spiking binder is viable with more averaging.", flush=True)
    else:
        print(f"  NEGATIVE: realistic read noise collapses systematicity to ~ the memorization floor (n_per~25 "
              f"{n20:.3f} vs floor {memf:.3f}) -- the rate-code wall breaks the learned bind; fallback = learned "
              f"codes + the fixed FHRR op. An honest substrate boundary.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"mean_by_noise": {str(k): v for k, v in mean_by_noise.items()}, "mem_floor": memf,
           "chance": 1.0 / F, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_learned_bind_ratenoise.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
