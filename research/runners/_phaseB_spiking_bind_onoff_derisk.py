"""CYCLE 101 — step-3 cheap-first #2b: the ON/OFF opponency spiking binder (the fix #2 localized).

#2 showed a SINGLE non-negative rate code collapses the binder's systematicity (held-out 0.083 vs tanh 0.750)
because it loses the SIGN the additive bind needs. The project's established fix for signed values in spikes is
ON/OFF OPPONENCY: a signed value -> two non-negative rate channels (ON = positive part, OFF = negative part),
used throughout (NEF cleanup, FHRR, the biologization sweep). This de-risk realizes the bind that way:
  h = role@W_R + filler@W_F + b_bind         (signed pre-activation)
  ON = relu(h), OFF = relu(-h)               (two NON-NEGATIVE rate channels = the signed h)
  bound = concat(ON, OFF)  [2*D_h]           (a faithful spiking rate code that PRESERVES the sign)
  unbind: concat(ON, OFF, role_h) [3*D_h] @ W_U   (linear; recovers the sign from the two channels)
trained NOISE-AWARE (finite-population read noise on bound + estimate; surrogate backward d_h = d_ON*1[h>0] -
d_OFF*1[h<0]). Everything else is the validated BilinearBinder (additive bind, Adam, the systematicity protocol).

GATE (3 seeds, F=16, the stream codes): held-out >> mem-floor AND ~ the tanh baseline. GO => the signed bind IS
carried by ON/OFF rate coding + surrogate-gradient training => the full on-bridge LIF spiking binder (ON/OFF) is
worth building. PARTIAL/NEGATIVE => localize further.

Reuse-by-import (the systematicity protocol + BilinearBinder); cached 320 stream codes; CPU; no GPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_bind_onoff_derisk
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
READ_NOISE = 0.20


class OnOffRateBinder(BilinearBinder):
    """Additive bind with ON/OFF opponency rate coding of the bound vector (preserves the sign in two
    non-negative rate channels), trained noise-aware with a surrogate gradient. W_U is resized 2*D_h -> 3*D_h
    (ON + OFF + role_h)."""

    def __init__(self, *a, read_noise=READ_NOISE, **k):
        super().__init__(*a, **k)
        self.read_noise = read_noise
        self._seed = int(k.get("seed", 42))
        self._rng = np.random.default_rng(self._seed * 7 + 1)
        rng = np.random.default_rng(self._seed * 999 + 13)         # resize W_U: ON[D_h]+OFF[D_h]+role_h[D_h]
        scale = 1.0 / np.sqrt(self.D_in)
        self.W_U = rng.standard_normal((self.D_h * 3, self.D_in)) * scale
        self.m_WU = np.zeros_like(self.W_U)
        self.v_WU = np.zeros_like(self.W_U)

    def _noisy(self, v):
        if self.read_noise <= 0:
            return v
        return v + self._rng.standard_normal(v.shape) * self.read_noise * (float(np.std(v)) + 1e-9)

    def _bind(self, role, filler):
        h = role @ self.W_R + filler @ self.W_F + self.b_bind
        return self._noisy(np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)]))   # [2*D_h]

    def _unbind(self, bound, role):
        role_h = role @ self.W_RP
        return self._noisy(np.concatenate([bound, role_h]) @ self.W_U + self.b_unbind)

    def train_step(self, role, filler):
        self.t += 1
        h = role @ self.W_R + filler @ self.W_F + self.b_bind        # [D_h] signed
        on = np.maximum(h, 0.0); off = np.maximum(-h, 0.0)
        bound = self._noisy(np.concatenate([on, off]))              # [2*D_h]
        role_h = role @ self.W_RP
        concat = np.concatenate([bound, role_h])                    # [3*D_h]
        est = self._noisy(concat @ self.W_U + self.b_unbind)
        err = est - filler
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_concat = self.W_U @ d_est                                 # [3*D_h]
        d_W_U = np.outer(concat, d_est)
        d_b_unbind = d_est.copy()
        d_bound = d_concat[:2 * self.D_h]
        d_role_h = d_concat[2 * self.D_h:]
        d_W_RP = np.outer(role, d_role_h)
        d_on, d_off = d_bound[:self.D_h], d_bound[self.D_h:]
        d_h = d_on * (h > 0).astype(np.float64) - d_off * (h < 0).astype(np.float64)   # ON/OFF surrogate
        d_W_R = np.outer(role, d_h); d_W_F = np.outer(filler, d_h); d_b_bind = d_h.copy()
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
    onoff_held, tanh_held, memf = [], [], []
    for split in splits:
        ob = OnOffRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)
        ob.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        onoff_held.append(score_bilinear_unbind(ob, split["held_out"], roles, fillers))
        tb = BilinearBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed)
        tb.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        tanh_held.append(score_bilinear_unbind(tb, split["held_out"], roles, fillers))
        memf.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])
    oh, th, mf = float(np.mean(onoff_held)), float(np.mean(tanh_held)), float(np.mean(memf))
    print(f"  [seed {seed}] ON/OFF spiking held-out {oh:.3f} | tanh baseline {th:.3f} | mem-floor {mf:.3f}", flush=True)
    return {"seed": seed, "onoff": oh, "tanh": th, "mem_floor": mf}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[ON/OFF spiking-bind de-risk] stream codes {codes.shape} -- does ON/OFF opponency rate coding (signed "
          f"-> two non-negative channels) + surrogate-gradient carry the learned bind? (vs the #2 single-rate "
          f"collapse 0.083; tanh baseline ~0.75)", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    onoff, tanh, mf = m("onoff"), m("tanh"), m("mem_floor")
    print(f"\n{'='*94}\n  MEAN (3 seeds): ON/OFF spiking held-out {onoff:.3f} | tanh baseline {tanh:.3f} | mem-floor "
          f"{mf:.3f} | chance {1.0/F:.3f}", flush=True)
    print(f"{'='*94}", flush=True)
    if onoff >= mf + 0.25 and onoff >= 0.75 * tanh:
        print(f"  GO: ON/OFF opponency rate coding CARRIES the learned bind -- held-out {onoff:.3f} >> mem-floor "
              f"{mf:.3f}, {onoff/max(tanh,1e-9):.0%} of the tanh baseline ({tanh:.3f}). The single-rate collapse "
              f"(#2, 0.083) was the lost sign; ON/OFF restores it. ==> the spiking learned binder is VIABLE on the "
              f"substrate's standard signed-value coding -> build the full on-bridge LIF ON/OFF spiking binder "
              f"(then the local three-factor / e-prop version).", flush=True)
    elif onoff >= mf + 0.10:
        print(f"  PARTIAL: ON/OFF beats memorization + the single-rate collapse ({onoff:.3f} vs {mf:.3f}) but below "
              f"the tanh baseline ({tanh:.3f}) -- more capacity / epochs / noise tuning; the sign is partly "
              f"recovered.", flush=True)
    else:
        print(f"  NEGATIVE: even ON/OFF coding doesn't carry it ({onoff:.3f} vs floor {mf:.3f}) -- the issue is "
              f"not just the sign; inspect the surrogate training / read noise / capacity.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"onoff": onoff, "tanh": tanh, "mem_floor": mf, "chance": 1.0 / F, "read_noise": READ_NOISE,
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_spiking_bind_onoff.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
