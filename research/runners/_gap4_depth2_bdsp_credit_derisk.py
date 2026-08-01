"""gap#4 DEPTH-2 credit: does the coincidence-gated BDSP rule GENERALIZE through depth where feedback-alignment did NOT?

THE REALIZATION that motivates this. Every gap#4 credit rule this session tested trained ONE hidden layer, but the
oracle is DEPTH-2 (`DendriticMLP([n_in, hidden, hidden, k])`). So `deep_credit_share` measured a SINGLE-layer rule
against a TWO-layer ceiling — "deep credit" (credit assignment THROUGH depth, the literal gap#4 question) was never
actually tested. The DEPTH regime has a KNOWN WALL: `2026-07-01-emerge1-deep-dendritic-representation-BOUNDARY` showed
a depth-2 dendritic net learning by FEEDBACK ALIGNMENT (fixed-random per-layer feedback + Urbanczik-Senn) MEMORIZES
but does NOT generalize on a depth-2 Boolean task (held-out 0.58, train->1.00, vs oracle backprop 0.95) — "the
emergence wall is the LOCAL RULE's depth-scaling." BUT that used FA (the exhausted family). This session produced a
GENUINELY DIFFERENT rule — the coincidence-gated + sigmoid-baseline BDSP that gave a weak-but-real positive at single
layer. THE DECISIVE UNTESTED QUESTION: does THAT rule generalize at depth-2 where FA didn't?

THE TEST (reuse emerge1's EXACT depth-2 task + baselines; rate DendriticMLP, CPU, minutes): the same depth-2 task
(pair-XORs -> threshold-over-XORs; a single hidden layer / memorizer provably can't generalize to held-out patterns),
the same arms (oracle backprop ~0.95, deep FA ~0.58, single-layer ~0.25, apical-lesion floor ~0.50, wrong-sign
~chance), PLUS the new arm: deep BDSP. The BDSP rule at EACH hidden layer replaces FA's graded `ap*sigmoid'` with a
COINCIDENCE gate (binary pre EVENT x binary post EVENT) times a SIGMOID-BASELINE bounded credit (sigmoid(beta*ap) -
Pbar), transport-free (per-layer fixed-random feedback B, never W^T). All the DFA plumbing + the momentum/mean-batch
optimizer are inherited byte-identical; the ONLY change is which credit each hidden layer computes.

GO GATE (does BDSP break the depth-scaling wall FA hit?): deep BDSP held-out >= deep FA held-out + 0.07 AND deep BDSP
held-out - train gap < 0.20 (GENERALIZES, doesn't just memorize) AND deep BDSP > apical-lesion floor + 0.07, on >=5/6
seeds. Anti-cheats: wrong-sign BDSP anti-learns (<= chance+0.05); apical-lesion (B=0) floors; oracle >= 0.80. NEGATIVE
(deep BDSP also memorizes ~0.58) = a strong CONSOLIDATED result: the depth-scaling wall is robust across FA AND BDSP.
NO `sim/` edit (subclass of DendriticMLP).
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from sim.dendritic_mlp import DendriticMLP, xp, _sig, _MOMENTUM
from research.runners._emerge1_deep_dendritic_representation_derisk import (
    make_task, _hidden_rep, _probe_latents, N_BITS, N_PAIRS)


class BDSPDendriticMLP(DendriticMLP):
    """Depth-2 coincidence-gated + sigmoid-baseline BDSP credit. Inherits the forward pass, the per-layer fixed-random
    feedback B (transport-free), and the momentum/mean-batch optimizer byte-identical; overrides ONLY the hidden-layer
    credit each step computes. `mode="bdsp"` / `mode="bdsp_wrongsign"`."""
    def init_bdsp(self, p0=0.30, beta=1.0):
        self.Pbar = [xp.full(self.W[li].shape[1], float(p0)) for li in range(len(self.W) - 1)]  # per hidden layer
        self.beta = float(beta)
        self._vel = None
        # ADJACENT-layer feedback for the CHAINED burstprop (Payeur): Q[l] maps the layer-ABOVE's credit into
        # layer l's space (transport-free: fixed random, INDEPENDENT of W, never W^T). For hidden layer l (its
        # activation is acts[l+1], size = W[l].shape[1]) the layer above has size W[l+1].shape[1] (or the output k).
        rngq = np.random.default_rng(1234567)
        self.Q = []
        nW = len(self.W)
        for l in range(nW - 1):
            above = self.W[l + 1].shape[1]                          # size of the layer above (next hidden or output)
            here = self.W[l].shape[1]                               # size of hidden layer l
            self.Q.append(xp.asarray(rngq.normal(0, 1.0 / np.sqrt(here), (above, here))))
        return self

    def train_step(self, X, y, mode, lr, rho=0.1):
        if mode not in ("bdsp", "bdsp_wrongsign", "bdsp_truegrad", "bdsp_soft", "burstprop"):
            return super().train_step(X, y, mode, lr)
        acts, e = self._debug_fwd_err(X, y)                         # acts (sigmoid), e = softmax(logits) - onehot(y)
        nW = len(self.W)
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ e)                                 # output-layer local delta (same as FA)
        if mode == "burstprop":
            # CHAINED top-down burst credit (Payeur): each layer's apical = the layer-ABOVE's credit projected through
            # the ADJACENT feedback Q[l] (transport-free). burst = event x sigmoid(apical); credit = burst - baseline;
            # the credit CHAINS DOWN (top_signal for the next deeper layer = THIS layer's credit) -> a genuine
            # multi-layer top-down target, unlike DFA's independent projection of the output error to each layer.
            top_signal = e                                          # into the top hidden layer = output error
            for li in range(nW - 2, -1, -1):                        # TOP hidden layer DOWN (chained)
                a_prev, a_l = acts[li], acts[li + 1]
                apical = top_signal @ self.Q[li]                    # (N, hidden_li) top-down from the layer above
                sig = 1.0 / (1.0 + xp.exp(-self.beta * apical))     # apical-modulated burst probability
                burst = a_l * sig                                   # event rate x burst prob = burst rate
                credit = burst - self.Pbar[li][None, :]             # burst DEVIATION from baseline (the credit)
                pre_ev = (a_prev > (0.0 if li == 0 else 0.5)).astype(a_prev.dtype)
                upd[li] = -(pre_ev.T @ credit)                      # dW ∝ pre event x burst deviation
                self.Pbar[li] = (1 - rho) * self.Pbar[li] + rho * burst.mean(0)
                top_signal = credit                                # CHAIN: next deeper layer's top-down = this credit
            m = max(1, X.shape[0])
            if self._vel is None:
                self._vel = [xp.zeros_like(w) for w in self.W]
            for li in range(nW):
                self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
                self.W[li] = self.W[li] + lr * self._vel[li]
            return
        sgn = +1.0 if mode == "bdsp_wrongsign" else -1.0          # descent (bdsp/truegrad) vs anti-learn (wrongsign)
        # DIAGNOSTIC ONLY: bdsp_truegrad feeds the coincidence-gated rule the TRUE backpropagated deep signal (uses
        # W^T -> NOT a shippable transport-free rule) to isolate whether the level-1-capture residual is the SIGNAL
        # (misaligned deep feedback) or the RULE (coincidence gate loses info). d = backprop error per layer.
        d_bp = e if mode == "bdsp_truegrad" else None
        for li in range(nW - 1):
            a_prev, a_l = acts[li], acts[li + 1]
            if mode == "bdsp_truegrad":
                for lj in range(nW - 1, li, -1):                  # backprop e down to the post-activation of layer li
                    aj = acts[lj]
                    d_bp = (d_bp @ self.W[lj].T) * aj * (1.0 - aj)
                ap = d_bp; d_bp = e                               # true error signal at layer li; reset for next li
            else:
                ap = e @ self.B[li]                               # (N, hidden_li) DFA-projected error, transport-free
            # BINARY EVENTS (bdsp/truegrad/wrongsign): input (li=0) is +/-1 -> >0; hidden sigmoid -> >0.5.
            # SOFT gate (bdsp_soft): the GRADED activation (isolates whether the regularization is the binary gate or
            # the bounded sigmoid-baseline credit -- a middle ground between graded-FA and binary-BDSP).
            if mode == "bdsp_soft":
                pre_ev = (a_prev + 1.0) * 0.5 if li == 0 else a_prev   # graded, in [0,1]
                post_ev = a_l
            else:
                pre_ev = (a_prev > (0.0 if li == 0 else 0.5)).astype(a_prev.dtype)
                post_ev = (a_l > 0.5).astype(a_l.dtype)
            sig = 1.0 / (1.0 + xp.exp(-self.beta * ap))            # P_post in [0,1]
            credit = sig - self.Pbar[li][None, :]                  # sigmoid-baseline BDSP credit
            upd[li] = sgn * (pre_ev.T @ (post_ev * credit))        # COINCIDENCE-gated (pre AND post events) x credit
            self.Pbar[li] = (1 - rho) * self.Pbar[li] + rho * sig.mean(0)   # EMA baseline
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train(net, X, y, mode, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            net.train_step(X[perm[i:i + batch]], y[perm[i:i + batch]], mode=mode, lr=lr)


def run_seed(seed, epochs, lr, batch, hidden, beta, p0, verbose=True):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]                             # >=2 hidden layers (the deep regime)
    shal = [N_BITS, hidden, 2]
    Xtr = xp.asarray(Xtr); Xte = xp.asarray(Xte)
    out = {"seed": seed}
    def acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    # ---- oracle backprop (ceiling / task-sanity) ----
    net = DendriticMLP(deep, seed=seed); _train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
    out["oracle_train"], out["oracle_heldout"] = acc(net)

    # ---- deep FA (the emerge1 baseline that MEMORIZED: ~0.58 held-out) ----
    net = DendriticMLP(deep, seed=seed); _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    out["deepFA_train"], out["deepFA_heldout"] = acc(net)
    out["deepFA_probe"] = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)

    # ---- deep BDSP (THE NEW ARM: does it generalize through depth?) ----
    net = BDSPDendriticMLP(deep, seed=seed).init_bdsp(p0=p0, beta=beta)
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    out["deepBDSP_train"], out["deepBDSP_heldout"] = acc(net)
    out["deepBDSP_probe"] = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    out["deepBDSP_gen_gap"] = round(out["deepBDSP_train"] - out["deepBDSP_heldout"], 4)

    # ---- DIAGNOSTIC: BDSP rule + TRUE backprop deep signal (uses W^T -- NOT shippable) -> is the residual the
    #      SIGNAL (misaligned deep feedback) or the RULE (coincidence gate)? If this captures level-1, it's the signal.
    net = BDSPDendriticMLP(deep, seed=seed).init_bdsp(p0=p0, beta=beta)
    _train(net, Xtr, ytr, "bdsp_truegrad", epochs, lr, batch, seed)
    out["deepBDSP_truegrad_heldout"] = acc(net)[1]
    out["deepBDSP_truegrad_probe"] = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)

    # ---- controls ----
    net = DendriticMLP(shal, seed=seed); _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    out["single_train"], out["single_heldout"] = acc(net)
    net = DendriticMLP(deep, seed=seed); net.B = [xp.zeros_like(b) for b in net.B]
    _train(net, Xtr, ytr, "local_correct", epochs, lr, batch, seed)
    out["apical_lesion_train"], out["apical_lesion_heldout"] = acc(net)
    # BDSP anti-cheat: wrong-sign BDSP must anti-learn (not generalize)
    net = BDSPDendriticMLP(deep, seed=seed).init_bdsp(p0=p0, beta=beta)
    _train(net, Xtr, ytr, "bdsp_wrongsign", epochs, lr, batch, seed)
    out["bdsp_wrongsign_heldout"] = acc(net)[1]
    out["chance"] = float(max(np.mean(np.asarray(yte) == c) for c in np.unique(yte)))
    # ATTRIBUTION: how much of deep BDSP's held-out is above the no-credit (apical-lesion) floor -- i.e. is the
    # generalization the CREDIT's, or already present in a frozen-random deep hidden? (attribution-required gate.)
    from tools.lab import attributable_to
    attributable_to("deep BDSP held-out vs the apical-lesion (no-credit) floor",
                    out["deepBDSP_heldout"], out["apical_lesion_heldout"])
    if verbose:
        print(f"  [seed {seed}] oracle {out['oracle_heldout']:.3f} | deep FA {out['deepFA_heldout']:.3f}"
              f"(tr {out['deepFA_train']:.3f}) | deep BDSP {out['deepBDSP_heldout']:.3f}(tr {out['deepBDSP_train']:.3f} "
              f"gap {out['deepBDSP_gen_gap']:+.3f}) | single {out['single_heldout']:.3f} | lesion "
              f"{out['apical_lesion_heldout']:.3f} | wrong-sign {out['bdsp_wrongsign_heldout']:.3f} | "
              f"chance {out['chance']:.3f}", flush=True)
        print(f"    probe latents: FA {out['deepFA_probe']:.3f}  BDSP {out['deepBDSP_probe']:.3f} (chance ~0.5)",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 DEPTH-2 coincidence-gated BDSP credit vs feedback-alignment.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--margin", type=float, default=0.07)
    ap.add_argument("--out", default="research/findings/raw/gap4/depth2_bdsp/depth2_bdsp.json")
    a = ap.parse_args()
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.epochs, a.lr, a.batch, a.hidden, a.beta, a.p0))
    except Exception as e:
        err = repr(e); traceback.print_exc()
    summary = {"probe": "gap4_depth2_bdsp_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden, "beta": a.beta,
                          "p0": a.p0, "task": "depth-2 pair-XOR->threshold (emerge1)", "margin": a.margin},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def m(k):
            return float(np.nanmean([p[k] for p in per]))
        n = len(per); need = int(np.ceil(0.834 * n))
        beats_fa = sum(1 for p in per if p["deepBDSP_heldout"] >= p["deepFA_heldout"] + a.margin)
        generalizes = sum(1 for p in per if p["deepBDSP_gen_gap"] < 0.20
                          and p["deepBDSP_heldout"] >= p["apical_lesion_heldout"] + a.margin)
        anti_ok = (all(p["bdsp_wrongsign_heldout"] <= p["chance"] + 0.05 for p in per)
                   and m("oracle_heldout") >= 0.80)
        go = bool(beats_fa >= need and generalizes >= need and anti_ok)
        agg = {"n_seeds": n, "mean_oracle": m("oracle_heldout"), "mean_deepFA": m("deepFA_heldout"),
               "mean_deepBDSP": m("deepBDSP_heldout"), "mean_deepBDSP_train": m("deepBDSP_train"),
               "mean_deepBDSP_gen_gap": m("deepBDSP_gen_gap"), "mean_apical_lesion": m("apical_lesion_heldout"),
               "mean_deepFA_probe": m("deepFA_probe"), "mean_deepBDSP_probe": m("deepBDSP_probe"),
               "bdsp_beats_fa_by_margin": beats_fa, "bdsp_generalizes": generalizes, "anti_cheats_clean": bool(anti_ok),
               "seeds_needed": need}
        summary["aggregate"] = agg; summary["GO"] = go
        if go:
            summary["verdict"] = (f"DEPTH-2 BDSP GO ({beats_fa}/{n} beat FA, generalizes {generalizes}/{n}) -- the "
                                  f"coincidence-gated BDSP rule BREAKS the depth-scaling wall FA hit: deep BDSP "
                                  f"{agg['mean_deepBDSP']:.3f} vs deep FA {agg['mean_deepFA']:.3f} (oracle "
                                  f"{agg['mean_oracle']:.3f}), gen-gap {agg['mean_deepBDSP_gen_gap']:+.3f} (generalizes, "
                                  f"not memorizes). A real advance on the gap#4 depth wall.")
        else:
            summary["verdict"] = (f"DEPTH-2 BDSP NEGATIVE (beats FA {beats_fa}/{n} need {need}, generalizes "
                                  f"{generalizes}/{n}, anti_ok {anti_ok}) -- deep BDSP {agg['mean_deepBDSP']:.3f} "
                                  f"(tr {agg['mean_deepBDSP_train']:.3f}, gap {agg['mean_deepBDSP_gen_gap']:+.3f}) vs "
                                  f"deep FA {agg['mean_deepFA']:.3f}, oracle {agg['mean_oracle']:.3f}. The depth-"
                                  f"scaling wall holds across FA AND BDSP -- a consolidated deep-credit-through-depth "
                                  f"boundary.")
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-depth2-bdsp] {summary['verdict']}", flush=True)
    print(f"[gap4-depth2-bdsp] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
