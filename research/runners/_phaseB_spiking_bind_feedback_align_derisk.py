"""CYCLE 102 — step-3 cheap-first #2c: is the spiking learned binder BRAIN-FAITHFUL? (a LOCAL learning rule,
no weight transport.)

#2b showed the ON/OFF spiking binder works under SURROGATE-GRADIENT training -- but that is EXACT backprop:
the hidden-layer error uses W_U (the forward readout weights) transposed in the backward (`d_concat = W_U @
d_est`). That "weight transport" is biologically implausible (a neuron cannot read its downstream synapses'
weights). The brain-faithful fix (Lillicrap feedback alignment; the e-prop/three-factor local-rule family the
deep-research scoping identified) is to broadcast the readout error through a FIXED RANDOM feedback matrix B
instead of W_U.T -- the hidden layer then learns by a LOCAL rule (its own pre-activity x a broadcast learning
signal), no weight transport. The readout layer's own gradient (outer(concat, d_est)) is already local.

If the binder still generalizes systematically under feedback alignment, then a cortex can LEARN to bind via
LOCAL plasticity + a broadcast teaching signal (no backprop) -- the genuine brain-faithful learned bind.

GATE (3 seeds, F=16, stream codes): held-out >> mem-floor AND ~ the exact-backprop ON/OFF binder (0.806).
GO => brain-faithful learned bind viable -> build the full on-bridge LIF binder (ON/OFF + local three-factor).
PARTIAL/NEGATIVE => random feedback isn't enough; the binding needs e-prop eligibility traces (localize).

Reuse-by-import (OnOffRateBinder + the systematicity protocol); cached 320 stream codes; CPU; no GPU; no sim/.
The ONLY change vs #2b = the hidden-layer error path (W_U.T -> fixed random B).
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_bind_feedback_align_derisk
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
    make_role_codes, make_systematicity_splits, score_bilinear_unbind, score_memorization_floor)
from research.runners._phaseB_spiking_bind_onoff_derisk import OnOffRateBinder  # noqa: E402

R, F, N_SPLITS, N_EPOCHS, D_H, LR, READ_NOISE = 4, 16, 3, 600, 64, 0.005, 0.20


class FeedbackAlignmentBinder(OnOffRateBinder):
    """The ON/OFF spiking binder with FEEDBACK ALIGNMENT: the hidden-layer error is broadcast through a FIXED
    RANDOM matrix B (no weight transport), not W_U transposed. The readout W_U still learns by its own local
    gradient. The brain-faithful learning rule (local plasticity + a broadcast teaching signal)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rng = np.random.default_rng(self._seed * 1234 + 99)
        self.B = rng.standard_normal((self.D_h * 3, self.D_in)) / np.sqrt(self.D_in)   # FIXED random feedback

    def train_step(self, role, filler):
        self.t += 1
        h = role @ self.W_R + filler @ self.W_F + self.b_bind
        on = np.maximum(h, 0.0); off = np.maximum(-h, 0.0)
        bound = self._noisy(np.concatenate([on, off]))
        role_h = role @ self.W_RP
        concat = np.concatenate([bound, role_h])
        est = self._noisy(concat @ self.W_U + self.b_unbind)
        err = est - filler
        loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_concat = self.B @ d_est                       # FEEDBACK ALIGNMENT (was W_U @ d_est = weight transport)
        d_W_U = np.outer(concat, d_est)                 # the readout still learns by its OWN local gradient
        d_b_unbind = d_est.copy()
        d_bound = d_concat[:2 * self.D_h]
        d_role_h = d_concat[2 * self.D_h:]
        d_W_RP = np.outer(role, d_role_h)
        d_on, d_off = d_bound[:self.D_h], d_bound[self.D_h:]
        d_h = d_on * (h > 0).astype(np.float64) - d_off * (h < 0).astype(np.float64)
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
    fa_held, ex_held, memf = [], [], []
    for split in splits:
        fa = FeedbackAlignmentBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)
        fa.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        fa_held.append(score_bilinear_unbind(fa, split["held_out"], roles, fillers))
        ex = OnOffRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)  # exact-backprop ref
        ex.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        ex_held.append(score_bilinear_unbind(ex, split["held_out"], roles, fillers))
        memf.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])
    fh, eh, mf = float(np.mean(fa_held)), float(np.mean(ex_held)), float(np.mean(memf))
    print(f"  [seed {seed}] FEEDBACK-ALIGN held-out {fh:.3f} | exact-backprop ON/OFF {eh:.3f} | mem-floor {mf:.3f}",
          flush=True)
    return {"seed": seed, "feedback_align": fh, "exact": eh, "mem_floor": mf}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[feedback-align spiking-bind de-risk] stream codes {codes.shape} -- does the ON/OFF spiking binder "
          f"generalize under a BRAIN-FAITHFUL local rule (feedback alignment, NO weight transport)?", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    fa, ex, mf = m("feedback_align"), m("exact"), m("mem_floor")
    print(f"\n{'='*96}\n  MEAN (3 seeds): FEEDBACK-ALIGN held-out {fa:.3f} | exact-backprop ON/OFF {ex:.3f} | "
          f"mem-floor {mf:.3f} | chance {1.0/F:.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    if fa >= mf + 0.25 and fa >= 0.70 * ex:
        print(f"  GO (brain-faithful): the spiking learned binder generalizes under FEEDBACK ALIGNMENT -- held-out "
              f"{fa:.3f} >> mem-floor {mf:.3f}, {fa/max(ex,1e-9):.0%} of exact backprop ({ex:.3f}), with NO weight "
              f"transport. ==> a cortex can LEARN to bind via LOCAL plasticity + a broadcast teaching signal "
              f"(no backprop) -> the genuine brain-faithful learned bind. Build the full on-bridge LIF binder "
              f"(ON/OFF + local three-factor / e-prop).", flush=True)
    elif fa >= mf + 0.10:
        print(f"  PARTIAL: feedback alignment beats memorization ({fa:.3f} vs {mf:.3f}) but below exact backprop "
              f"({ex:.3f}) -- random feedback is partly sufficient; e-prop eligibility traces / more epochs may "
              f"close the gap. Brain-faithful learning is viable, just weaker than exact.", flush=True)
    else:
        print(f"  NEGATIVE: random feedback does NOT carry the binding ({fa:.3f} vs floor {mf:.3f}) -- the brain-"
              f"faithful local rule needs more than random feedback (e-prop eligibility); the exact-backprop "
              f"result stands but its biological plausibility is open. Localize.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"feedback_align": fa, "exact": ex, "mem_floor": mf, "chance": 1.0 / F, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_spiking_bind_feedback_align.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
