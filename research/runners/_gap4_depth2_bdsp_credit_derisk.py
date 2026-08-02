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
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from sim.backend import to_host
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

    # ------------------------------------------------------------------ ADDITIVE (default-OFF): the GRADED-credit
    # feedback-source ladder. σ′ is ON for ALL arms; the SINGLE VARIABLE is the descending FEEDBACK matrix. No binary
    # event/coincidence gate anywhere here -- this is plain real-valued error propagation, the recipe WF-Act-PC says
    # reaches backprop at depth and that the depth-2 arc never assembled (it always squeezed the signal through the
    # binary burst gate). NONE of the existing modes/methods below are touched -> every prior arm is byte-identical.
    def init_graded(self, kp_lr=0.1, kp_decay=1e-3, fb_seed=0):
        """Init the graded ladder. Y[k] maps the layer-ABOVE error (size sizes[k+2]) down to hidden layer k+1 (size
        sizes[k+1]); DRAWN FROM A SEPARATE SEED STREAM (no weight transport). SHARED init for the 'dfa' (fixed-random
        Y) and 'kp' (KP-LEARNED Y) arms so the ONLY difference is whether Y learns; 'truegrad' ignores Y (uses W^T)."""
        self._vel = None
        self.kp_lr = float(kp_lr); self.kp_decay = float(kp_decay)
        yrng = np.random.default_rng(int(fb_seed))                  # SEPARATE stream: never derived from any forward W
        s = self.sizes
        self.Y = [xp.asarray(yrng.normal(0, 1.0, (s[k + 2], s[k + 1]))) for k in range(len(s) - 2)]
        return self

    def _graded_train_step(self, X, y, mode, lr):
        """Chained GRADED-credit error-prop, σ′ ON at every hidden layer, NO binary gate:
              e_l = (e_{l+1} @ FEEDBACK_l) * a_l*(1-a_l) ;   W_l += lr * (a_{l-1}^T @ e_l)      (descent-signed e)
        SINGLE VARIABLE = FEEDBACK_l source: 'dfa' fixed-random Y | 'kp' KP-LEARNED transport-free Y | 'truegrad' W^T
        (the fenced backprop oracle anchor). '_wrong' negates the output teacher (anti-learn); permuted labels are done
        by the caller. TRANSPORT-FREE for dfa/kp: the credit path reads ONLY Y + local activity, NEVER a forward W."""
        rest = mode[len("graded_"):]
        wrong = rest.endswith("_wrong")
        fb = rest[:-len("_wrong")] if wrong else rest              # 'dfa' | 'kp' | 'truegrad'
        acts, e_pos = self._debug_fwd_err(X, y)                    # e_pos = softmax(logits) - onehot(y) (pos. gradient)
        nW = len(self.W)
        delta_out = -e_pos                                         # descent-signed output error
        if wrong:
            delta_out = -delta_out                                 # negate the teacher -> the WHOLE net anti-learns
        upd = [None] * nW
        upd[-1] = acts[-1].T @ delta_out                          # output-layer descent update (linear head; no σ′)
        b = delta_out                                             # descending error (descent-signed); starts at output
        for li in range(nW - 2, -1, -1):                          # TOP hidden layer -> bottom (recursive top->bottom)
            a_l = acts[li + 1]                                    # this hidden layer's sigmoid activation
            if fb == "kp":
                # Kolen-Pollack LEARNED feedback (Akrout 2019 weight-mirror), TRANSPORT-FREE: update Y[li] from the
                # LOCAL pre=acts[li+1] / post=b outer product ONLY -- never reads a forward W. +outer matches W[li+1]'s
                # descent increment (b is descent-signed) so Y[li]^T -> W[li+1] under the shared decay -> Y approximates
                # the transpose WITHOUT copying it.
                m0 = max(1, a_l.shape[0])
                outer = (b.T @ a_l) / m0                          # (sizes[li+2], sizes[li+1]) == Y[li].shape; LOCAL only
                self.Y[li] = self.Y[li] + lr * (self.kp_lr * outer - self.kp_decay * self.Y[li])
            FB = self.W[li + 1].T if fb == "truegrad" else self.Y[li]   # truegrad reads W^T (oracle); dfa/kp read Y
            e_l = (b @ FB) * (a_l * (1.0 - a_l))                  # GRADED credit: back-projected error * σ′(pre_act) ON
            upd[li] = acts[li].T @ e_l                            # plain graded error-prop update (NO binary gate)
            b = e_l                                              # CHAIN: the graded error descends to the next layer
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]

    def train_step(self, X, y, mode, lr, rho=0.1):
        if isinstance(mode, str) and mode.startswith("graded_"):    # ADDITIVE graded-credit feedback-source ladder
            return self._graded_train_step(X, y, mode, lr)
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


# ====================================================================================================================
# ADDITIVE (default-OFF) GRADED-CREDIT FEEDBACK-SOURCE LADDER (--feedback-ladder). σ′ ON for all arms, NO binary gate;
# the single variable is the descending feedback matrix: dfa (fixed-random) / kp (KP-learned, transport-free) / truegrad
# (W^T = backprop oracle anchor). Decides whether the "transport-free ceiling" (dfa ~0.63 plateau) is a real wall or an
# artifact of the binary event gate + fixed-random feedback: if kp ~ truegrad >> dfa, learned feedback closed the gap.
# ====================================================================================================================
def _wh(net):
    """SHA256 of the concatenated forward weights -> a byte-identity fingerprint (backend-agnostic via to_host)."""
    h = hashlib.sha256()
    for w in net.W:
        h.update(np.ascontiguousarray(np.asarray(to_host(w), dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def _fb_align_cos(net):
    """Per-layer cosine(Y[li], W[li+1]^T): does the LEARNED feedback approximate the transpose (KP's target)? ~0 for
    fixed-random dfa; climbs positive if KP mirrored W. Diagnostic ONLY (never fed to the credit)."""
    cs = []
    for li in range(len(net.W) - 1):
        Y = np.asarray(to_host(net.Y[li]), dtype=np.float64).ravel()
        WT = np.asarray(to_host(net.W[li + 1]), dtype=np.float64).T.ravel()
        cs.append(round(float(Y @ WT / (np.linalg.norm(Y) * np.linalg.norm(WT) + 1e-9)), 4))
    return cs


def run_feedback_ladder(seed, epochs, lr, batch, hidden, kp_lr, kp_decay, verbose=True):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]
    Xtr_x, Xte_x = xp.asarray(Xtr), xp.asarray(Xte)
    chance = float(max(np.mean(np.asarray(yte) == c) for c in np.unique(yte)))
    out = {"seed": seed, "chance": round(chance, 4)}

    def acc(net):
        return round(float(net.accuracy(Xtr_x, ytr)), 4), round(float(net.accuracy(Xte_x, yte)), 4)

    def probe(net):
        return round(_probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte), 4)

    # ---- fenced backprop oracle (task ceiling + the truegrad-identity anchor) ----
    orc = DendriticMLP(deep, seed=seed); _train(orc, Xtr_x, ytr, "oracle", epochs, lr, batch, seed)
    out["oracle_train"], out["oracle_heldout"] = acc(orc); out["oracle_probe"] = probe(orc)
    out["oracle_whash"] = _wh(orc)

    def run_arm(fb, wrong=False, permute=False):
        y = np.asarray(ytr).copy()
        if permute:
            np.random.default_rng(seed + 4242).shuffle(y)          # deterministic label permutation (leakage control)
        net = BDSPDendriticMLP(deep, seed=seed).init_graded(kp_lr=kp_lr, kp_decay=kp_decay, fb_seed=seed + 9973)
        mode = "graded_" + fb + ("_wrong" if wrong else "")
        _train(net, Xtr_x, y, mode, epochs, lr, batch, seed)
        return net

    # ---- the 3-arm single-variable ladder (feedback source is the ONLY difference; σ′ ON, graded, no gate) ----
    for fb in ("dfa", "kp", "truegrad"):
        net = run_arm(fb)
        out[fb + "_train"], out[fb + "_heldout"] = acc(net); out[fb + "_probe"] = probe(net)
        if fb == "kp":
            out["kp_Y_vs_WT_cos"] = _fb_align_cos(net)             # did the learned feedback mirror the transpose?
        if fb == "truegrad":
            out["truegrad_whash"] = _wh(net)

    # ---- anti-cheats on the transport-free arms (dfa is the headline, kp is the coordinator's test arm) ----
    out["dfa_permuted_heldout"] = acc(run_arm("dfa", permute=True))[1]   # -> ~chance (no leakage)
    out["dfa_wrongsign_heldout"] = acc(run_arm("dfa", wrong=True))[1]    # -> below/at chance (credit sign load-bearing)
    out["kp_permuted_heldout"] = acc(run_arm("kp", permute=True))[1]     # -> ~chance (no leakage)
    out["kp_wrongsign_heldout"] = acc(run_arm("kp", wrong=True))[1]      # -> below/at chance (credit sign load-bearing)

    # ---- correctness anchor: graded 'truegrad' recursion reproduces the backprop oracle (numeric; whash differs only
    #      by ~5e-13 FP-associativity of the sigma' multiply grouping, so compare accuracies/probe not raw bytes) ----
    out["truegrad_matches_oracle"] = bool(abs(out["truegrad_heldout"] - out["oracle_heldout"]) < 1e-6
                                          and abs(out["truegrad_train"] - out["oracle_train"]) < 1e-6
                                          and abs(out["truegrad_probe"] - out["oracle_probe"]) < 1e-6)

    if verbose:
        print(f"  [seed {seed}] chance {out['chance']:.3f} | ORACLE held {out['oracle_heldout']:.3f} "
              f"(tr {out['oracle_train']:.3f}, probe {out['oracle_probe']:.3f})", flush=True)
        print(f"    LADDER (graded credit, sigma' ON, NO binary gate; feedback-source = the single variable):",
              flush=True)
        print(f"      dfa      (fixed-random) held {out['dfa_heldout']:.3f} (tr {out['dfa_train']:.3f}, "
              f"probe {out['dfa_probe']:.3f})", flush=True)
        print(f"      kp       (KP-learned)   held {out['kp_heldout']:.3f} (tr {out['kp_train']:.3f}, "
              f"probe {out['kp_probe']:.3f})  Y-vs-W^T cos {out['kp_Y_vs_WT_cos']}", flush=True)
        print(f"      truegrad (W^T oracle)   held {out['truegrad_heldout']:.3f} (tr {out['truegrad_train']:.3f}, "
              f"probe {out['truegrad_probe']:.3f})  [reproduces oracle: {out['truegrad_matches_oracle']}]", flush=True)
        print(f"    anti-cheats: dfa permuted {out['dfa_permuted_heldout']:.3f} / wrong-sign "
              f"{out['dfa_wrongsign_heldout']:.3f} | kp permuted {out['kp_permuted_heldout']:.3f} / wrong-sign "
              f"{out['kp_wrongsign_heldout']:.3f}  (chance {out['chance']:.3f})", flush=True)
    return out


def _main_feedback_ladder(a):
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_feedback_ladder(s, a.epochs, a.lr, a.batch, a.hidden, a.kp_lr, a.kp_decay))
    except Exception as e:
        err = repr(e); traceback.print_exc()
    summary = {"probe": "gap4_graded_credit_feedback_source_ladder", "seeds": a.seeds,
               "backend": os.environ.get("SIM_BACKEND"),
               "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                          "kp_lr": a.kp_lr, "kp_decay": a.kp_decay,
                          "task": "depth-2 pair-XOR->threshold (emerge1)",
                          "mechanism": "graded credit + sigma' ON at every hidden layer, NO binary gate; feedback "
                                       "source (dfa/kp/truegrad) is the single variable"},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def m(k):
            return round(float(np.nanmean([p[k] for p in per])), 4)
        dfa_h, kp_h, tg_h = m("dfa_heldout"), m("kp_heldout"), m("truegrad_heldout")
        orc_h, ch = m("oracle_heldout"), m("chance")
        PLATEAU = 0.63                                              # the banked transport-free "ceiling" (FA memorizes)
        best_tf = max(dfa_h, kp_h)                                  # best TRANSPORT-FREE arm (dfa or kp)
        # decisive read: does ANY transport-free graded arm clear the banked ~0.63 plateau by a clear margin, with
        # clean anti-cheats? If yes, the plateau was NOT a hard transport-free wall.
        anti_ok = (all(p["dfa_permuted_heldout"] <= p["chance"] + 0.06 for p in per)
                   and all(p["dfa_wrongsign_heldout"] <= p["chance"] + 0.06 for p in per)
                   and all(p["kp_permuted_heldout"] <= p["chance"] + 0.06 for p in per)
                   and all(p["kp_wrongsign_heldout"] <= p["chance"] + 0.06 for p in per))
        tg_ok = all(p["truegrad_matches_oracle"] for p in per)
        clears_plateau = bool(best_tf >= PLATEAU + 0.10 and anti_ok)
        summary["aggregate"] = {"mean_dfa": dfa_h, "mean_kp": kp_h, "mean_truegrad": tg_h, "mean_oracle": orc_h,
                                "chance": ch, "banked_plateau": PLATEAU, "best_transport_free": best_tf,
                                "transport_free_clears_plateau": clears_plateau, "anti_cheats_clean": bool(anti_ok),
                                "truegrad_matches_oracle_all_seeds": bool(tg_ok)}
        summary["verdict"] = (
            f"GRADED-CREDIT FEEDBACK-SOURCE LADDER: dfa {dfa_h:.3f} | kp {kp_h:.3f} | truegrad {tg_h:.3f} "
            f"(oracle {orc_h:.3f}, chance {ch:.3f}, banked plateau ~{PLATEAU:.2f}); best transport-free {best_tf:.3f}; "
            f"anti-cheats {'clean' if anti_ok else 'DIRTY'}; truegrad reproduces oracle: {tg_ok}. "
            + ("=> a TRANSPORT-FREE graded rule CLEARS the ~0.63 plateau => the banked 'transport-free ceiling' was an "
               "artifact of the binary event/coincidence gate (removed here), NOT a hard wall."
               if clears_plateau else
               "=> no transport-free arm cleared the plateau at this config (needs a KP/hyperparam sweep, or the wall "
               "is real)."))
    else:
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-feedback-ladder] {summary['verdict']}", flush=True)
    print(f"[gap4-feedback-ladder] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0


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
    # ADDITIVE (default-OFF): the graded-credit feedback-source ladder (dfa / kp / truegrad).
    ap.add_argument("--feedback-ladder", action="store_true",
                    help="run the graded-credit feedback-source ladder (sigma' ON, no binary gate) instead of the "
                         "BDSP sweep: dfa (fixed-random) vs kp (KP-learned transport-free) vs truegrad (W^T oracle).")
    ap.add_argument("--kp-lr", type=float, default=0.1, help="Kolen-Pollack feedback learning rate (kp arm only).")
    ap.add_argument("--kp-decay", type=float, default=1e-3, help="Kolen-Pollack symmetric weight decay (kp arm only).")
    a = ap.parse_args()
    if a.feedback_ladder:
        return _main_feedback_ladder(a)
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
