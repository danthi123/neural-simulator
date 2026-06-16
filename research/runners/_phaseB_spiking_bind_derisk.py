"""CYCLE 100 — step-3 cheap-first #2: a SPIKING-RATE additive binder via surrogate-gradient training. Does the
spiking representation (a NON-NEGATIVE saturating rate nonlinearity + finite-population read noise, trained
noise-aware with a surrogate gradient) generalize SYSTEMATICALLY like the numpy tanh binder?

CONTEXT. Cheap-first #1 retired the read-noise risk on the tanh binder (systematicity flat across n_per). But
the tanh bound is SIGNED [-1,1]; a real spiking RATE code is NON-NEGATIVE [0, r_max] (the LIF f-I curve), and
the spiking binder is trained by SURROGATE GRADIENT (smooth backward through a non-differentiable spike). This
de-risk makes both changes: the bind nonlinearity tanh -> SIGMOID (a non-negative, saturating rate transfer,
the mean-field of an LIF population), trained NOISE-AWARE (noisy forward read of bound + estimate, smooth
sigmoid-derivative backward = the surrogate gradient). Everything else (the additive bind = role@W_R +
filler@W_F, the linear unbind, Adam) is the validated BilinearBinder; ONLY the nonlinearity + the read model
change. This is the spiking analogue of the de-risked binder, the cheapest faithful form before the full LIF
surrogate-BPTT on `sim/bptt_snn.py`.

GATE (3 seeds, F=16, the systematicity protocol on the stream codes): the spiking-rate binder's held-out
generalization >> the memorization floor AND ~ the tanh baseline. SYSTEMATIC => the rate-code representation +
surrogate-gradient training carry the learned bind => the full on-bridge LIF spiking binder is worth building.
NEGATIVE => the non-negative rate code or the surrogate training breaks systematicity => localize honestly.

Reuse-by-import (the systematicity protocol helpers + BilinearBinder); the cached 320 stream codes; CPU; no
GPU; no sim/ edits. The ONLY change vs the de-risked binder = the spiking-rate nonlinearity + the read model.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_bind_derisk
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

R, F, N_SPLITS, N_EPOCHS, D_H, LR = 4, 16, 3, 600, 64, 0.005
READ_NOISE = 0.20      # finite-population read noise on bound + estimate (~ n_per 25; #1 showed it survivable)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SpikingRateBinder(BilinearBinder):
    """The de-risked ADDITIVE binder with the bind nonlinearity tanh -> SIGMOID (a non-negative saturating
    rate transfer, the mean-field of an LIF population), trained NOISE-AWARE: the forward READS bound + estimate
    through a finite population (additive zero-mean noise ~ read_noise*std), the backward uses the smooth
    sigmoid derivative (the surrogate gradient). bind = sigmoid(role@W_R + filler@W_F); unbind linear."""

    def __init__(self, *a, read_noise=READ_NOISE, **k):
        super().__init__(*a, **k)
        self.read_noise = read_noise
        self._rng = np.random.default_rng(self.seed * 7 + 1 if hasattr(self, "seed") else 1)

    def _noisy(self, v):
        if self.read_noise <= 0:
            return v
        return v + self._rng.standard_normal(v.shape) * self.read_noise * (float(np.std(v)) + 1e-9)

    def _bind(self, role, filler):                       # forward (scoring): non-negative rate + read noise
        return self._noisy(_sigmoid(role @ self.W_R + filler @ self.W_F + self.b_bind))

    def _unbind(self, bound, role):
        role_h = role @ self.W_RP
        concat = np.concatenate([bound, role_h])
        return self._noisy(concat @ self.W_U + self.b_unbind)

    def train_step(self, role, filler):                  # surrogate-gradient step (noisy forward, smooth backward)
        self.t += 1
        h_pre = role @ self.W_R + filler @ self.W_F + self.b_bind
        bound_clean = _sigmoid(h_pre)                    # the rate (non-negative)
        bound = self._noisy(bound_clean)                 # finite-population read
        role_h = role @ self.W_RP
        concat = np.concatenate([bound, role_h])
        est = self._noisy(concat @ self.W_U + self.b_unbind)
        err = est - filler
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_concat = self.W_U @ d_est
        d_W_U = np.outer(concat, d_est)
        d_b_unbind = d_est.copy()
        d_bound = d_concat[:self.D_h]
        d_role_h = d_concat[self.D_h:]
        d_W_RP = np.outer(role, d_role_h)
        d_h_pre = d_bound * (bound_clean * (1.0 - bound_clean))   # SURROGATE: smooth sigmoid derivative
        d_W_R = np.outer(role, d_h_pre)
        d_W_F = np.outer(filler, d_h_pre)
        d_b_bind = d_h_pre.copy()
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
    spk_held, tanh_held, memf = [], [], []
    for split in splits:
        # spiking-rate binder (sigmoid + read noise, surrogate-gradient)
        sb = SpikingRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)
        sb.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        spk_held.append(score_bilinear_unbind(sb, split["held_out"], roles, fillers))
        # tanh baseline (the de-risked binder, clean) for reference
        tb = BilinearBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed)
        tb.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        tanh_held.append(score_bilinear_unbind(tb, split["held_out"], roles, fillers))
        memf.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])
    sh, th, mf = float(np.mean(spk_held)), float(np.mean(tanh_held)), float(np.mean(memf))
    print(f"  [seed {seed}] SPIKING-rate held-out {sh:.3f} | tanh baseline {th:.3f} | mem-floor {mf:.3f}", flush=True)
    return {"seed": seed, "spiking": sh, "tanh": th, "mem_floor": mf}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[spiking-bind de-risk] stream codes {codes.shape} -- does a SPIKING-RATE additive binder (non-negative "
          f"sigmoid + read noise, surrogate-gradient) generalize systematically like the numpy tanh binder?",
          flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    spk, tanh, mf = m("spiking"), m("tanh"), m("mem_floor")
    print(f"\n{'='*94}\n  MEAN (3 seeds): SPIKING-rate held-out {spk:.3f} | tanh baseline {tanh:.3f} | mem-floor "
          f"{mf:.3f} | chance {1.0/F:.3f}", flush=True)
    print(f"{'='*94}", flush=True)
    if spk >= mf + 0.25 and spk >= 0.75 * tanh:
        print(f"  GO: the SPIKING-RATE binder GENERALIZES SYSTEMATICALLY -- held-out {spk:.3f} >> mem-floor {mf:.3f}, "
              f"{spk/max(tanh,1e-9):.0%} of the tanh baseline ({tanh:.3f}). The non-negative rate code + "
              f"surrogate-gradient training CARRY the learned bind. ==> build the full on-bridge LIF spiking "
              f"binder (the rate-approximation here is faithful; the LIF/surrogate-BPTT realization is next).",
              flush=True)
    elif spk >= mf + 0.10:
        print(f"  PARTIAL: the spiking-rate binder beats memorization ({spk:.3f} vs {mf:.3f}) but below the tanh "
              f"baseline ({tanh:.3f}) -- the non-negative rate code costs some systematicity; ON/OFF opponency "
              f"coding (signed via two rate channels) or more capacity may close it.", flush=True)
    else:
        print(f"  NEGATIVE: the spiking-rate representation collapses systematicity to ~ the floor ({spk:.3f} vs "
              f"{mf:.3f}) -- the non-negative rate code can't carry the signed bind; needs ON/OFF coding or the "
              f"signed tanh stays. An honest representational boundary.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"spiking": spk, "tanh": tanh, "mem_floor": mf, "chance": 1.0 / F, "read_noise": READ_NOISE,
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_spiking_bind.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
