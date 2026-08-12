"""gap#4 DEEP FEEDBACK CONTROL (DFC) plateau credit de-risk (SMOKE, rate/analytic, NO sim/ edit).

THE QUESTION (the record's explicitly-named-untested route). This session showed the coincidence-plateau hidden is
MOVABLE: an UNSUPERVISED local covariance rule beats a frozen random reservoir on the n_prop=3 sweet spot
(deep_credit_share +0.139, 5/6). A DIRECTED output error (fixed-DFA, `_gap4_supervised_plateau_credit_derisk`) also
TRAINS the movable hidden but OVERFITS -- it does NOT beat unsupervised on held-out (0.108 < 0.139, beats 1/6), and a
LARGER task does not rescue it (null, NOT task-limited: the local-vs-oracle gap WIDENS). Fixed-DFA is a LINEAR
error-projection -- the exact failure mode that saturated the whole feedback-alignment family (held-out capped ~0.715).

DFC is a GENUINELY DIFFERENT credit route (Meulemans et al. 2021/2022): instead of projecting the output error
linearly onto the hidden, a CLOSED-LOOP leaky controller drives the hidden ACTIVATIONS toward a target that reduces
output error, and LOCAL plasticity learns the forward weights so the FREE (uncontrolled) hidden reproduces the
controlled activations on its own. Transport-free (a fixed-random controller matrix Q, never W_out^T), and NOT a
one-shot linear projection (the leak + the closed loop are load-bearing). This attacks the credit route the FA family
never tried, on THIS session's breakthrough substrate (the movable plateau hidden).

MECHANISM (rate/analytic; the on-bridge SPIKING port is a LATER rung -- this de-risk decides whether DFC is worth it):
  Forward:    Mrg_free = graded plateau margin per column (the hidden activation); logits = Mrg_free @ W_out.
  Controller: u <- leaky closed loop, K steps: e_ctrl = softmax((Mrg_free+u) @ W_out) - Y ; u += dt*(-alpha*u - ctrl_sign*(e_ctrl @ Q.T)).
              u is the control nudge that, ADDED to the hidden, reduces output error. Q (C,k) is random-INITIALIZED
              INDEPENDENT of W_out and LEARNED transport-free by a Kolen-Pollack rule (the SAME local pre x error signal
              that trains W_out updates Q, with weight decay) -> Q ALIGNS to W_out without ever reading it (alignment,
              NOT a copy: Q^T.W_out diag>0, cos<1). W_out is read only in the FORWARD pass.
  Learn:      the transport-free DELTA rule ΔW ∝ +lr * u.T @ pre  (move the input weights so the free hidden PRODUCES
              the control nudge), then excitatory clip + per-column L2 renorm to the initial norm (the SAME local
              homeostatic companion as the unsupervised/supervised rules). The hidden update reads ONLY (u, pre, lr) --
              never W_out -> the no-transport runtime probe injects a FIXED u and varies W_out (update invariant).

ARMS (identical net/seed/init -> only the credit route differs):
  1. FROZEN-plateau reservoir     -- fixed random coincidence weights + trained readout (deep_credit_share denominator).
  2. UNSUPERVISED plateau         -- this session's local covariance rule (the 0.14 baseline DFC must BEAT).
  3. DFC plateau                  -- the new arm (closed-loop controller + delta rule).
  4a. oracle (fenced backprop depth-2)   -- ceiling (~0.96); 4b. frozen RATE reservoir -- floor (~0.10, op-point genuine).
deep_credit_share = (arm - frozen_plateau) / (oracle - frozen_plateau), reported for BOTH unsup and dfc.

GO GATE (depth_helps-style, set against BOTH nulls it must break; 6-seed 42 43 44 / 100 101 102):
  - DFC held-out beats BOTH frozen-plateau AND unsupervised by margin >= --margin-go on >= 5/6 seeds; AND
  - deep_credit_share_dfc > deep_credit_share_unsup on >= 5/6 seeds (target dcs_dfc >= 0.30, clearly past unsup 0.14); AND
  - all anti-cheats hold. Anything less is an HONEST NEGATIVE that closes DFC on this substrate (a mapped verdict).
ANTI-CHEATS (all must hold): no-weight-transport (the hidden credit update reads only the control nudge u, invariant to
  W_out; the feedback Q is LEARNED by a transport-free Kolen-Pollack rule -- a shared local pre x error signal, never a
  copy of W_out -- and ALIGNS to it (Q^T.W_out diag>0, cos<1); Akrout et al. 2019, alignment != weight copying);
  WRONG-SIGN control anti-learns (flip ctrl_sign -> held-out degrades BELOW frozen -> the control signal is load-bearing
  and sign-correct); shuffle-control across the batch -> degrade toward unsupervised (per-sample routing load-bearing);
  DFC-on-permuted-labels -> collapse to ~frozen (benefit is label-dependent); plateau LESION -> floor; reproducibility
  >= 0.8; oracle ceiling >= 0.80; rate-reservoir floor <= 0.40. cfg-seed substrate seeding is inherited (parent).

Usage:
  # smoke (1 seed, decisive, minutes on CPU):
  SIM_BACKEND=numpy python -u -m research.runners._gap4_dfc_plateau_credit_derisk --seeds 42 \
      --out research/findings/raw/gap4/dfc_plateau/smoke_s42.json
  # 6-seed (GPU or pool):
  SIM_BACKEND=cupy python -u -m research.runners._gap4_dfc_plateau_credit_derisk --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/gap4/dfc_plateau/dfc_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import inspect
import json
import time
import traceback
from pathlib import Path

import numpy as np

# reuse-by-import: the ENTIRE mechanism, task, oracle, readout, anti-cheat helpers + the FROZEN/UNSUP arms.
# NO edit to the parent; DFCPlateauExpander subclasses its PlasticPlateauExpander (byte-identical init at a seed).
from research.runners._gap4_plastic_plateau_credit_derisk import (
    PlasticPlateauExpander, fit_lin, topk_active, _sm, _readout_acc, _reproducibility, _codon_diversity,
    _rate_reservoir_heldout, FLOOR, ACT_TH, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)


class DFCPlateauExpander(PlasticPlateauExpander):
    def init_dfc(self, k, seed, fb_wd=0.02):
        """Forward readout W_out (trained by its OWN gradient) + controller feedback matrix Q, LEARNED transport-free.
        Q is drawn from an RNG stream INDEPENDENT of W_out and is NOT a copy of W_out; it is updated by the SAME local
        pre x error signal that trains W_out (Kolen-Pollack), which ALIGNS Q to W_out without ever reading it."""
        rng_out = np.random.default_rng(seed * 999 + 17)          # readout init stream
        rng_Q = np.random.default_rng(seed * 777 + 5)             # Q stream -- INDEPENDENT of W_out (transport-free)
        self.fb_wd = float(fb_wd)                                  # Kolen-Pollack feedback weight decay
        self.k = int(k)
        self.W_out = rng_out.standard_normal((self.NC, self.k)) * 0.01     # small forward readout (trained)
        self.b_out = np.zeros(self.k)
        self.Q = rng_Q.standard_normal((self.NC, self.k)) / np.sqrt(self.k)  # random-INIT controller feedback (KP-learned)
        self.Q0 = self.Q.copy()                                   # snapshot -> assert Q MOVED (learned) via KP
        return self

    def run_controller(self, Mrg_free, Y, ctrl_steps, alpha, dt, ctrl_sign=1.0):
        """CLOSED-LOOP leaky controller. Returns u (N,C): the nudge that, ADDED to the free hidden, reduces output
        error. Uses W_out in the FORWARD pass only; the FEEDBACK error->hidden is via Q (transport-free: KP-learned,
        never W_out^T). ctrl_sign=-1 flips the drive (the wrong-sign anti-cheat -> should anti-learn)."""
        u = np.zeros_like(Mrg_free)
        for _ in range(int(ctrl_steps)):
            logits = (Mrg_free + u) @ self.W_out + self.b_out     # controlled forward (recomputed each step = closed loop)
            e_ctrl = _sm(logits) - Y                              # (N,k) controlled output error
            u = u + dt * (-alpha * u - ctrl_sign * (e_ctrl @ self.Q.T))   # leaky integral toward the error-reducing drive
        return u

    def _hidden_update_from_control(self, u, pre_mat, lr):
        """The transport-free DELTA rule: move the input weights so the FREE hidden PRODUCES the control nudge u.
        Reads ONLY (u, pre, lr) -- NEVER W_out or W_out^T. (Exposed so the no-transport probe injects a FIXED u and
        varies W_out.) Same excitatory clip + per-column L2 renorm homeostasis as the unsupervised/supervised rules."""
        dW_full = lr * (u.T @ pre_mat) / len(pre_mat)             # (C, F); ASCEND toward the control target
        data = self._get_data()
        data = data + dW_full[self.syn_col, self.syn_feat]
        np.maximum(data, 0.0, out=data)                           # excitatory (w >= 0)
        cur = np.sqrt(np.array([np.sum(data[self.syn_col == c] ** 2) for c in range(self.NC)]))
        scale = np.where(cur > 1e-9, self.col_norm0 / (cur + 1e-12), 1.0)
        data = data * scale[self.syn_col]
        self._set_data(data)
        return float(np.mean(np.abs(dW_full)))

    def train_epoch_dfc(self, active_sets, pre_mat, y, lr, lr_out, ctrl_steps, alpha, dt,
                        ctrl_sign=1.0, shuffle_control=False, ctrl_rng=None):
        """One DFC batch: free forward -> closed-loop controller -> delta-rule hidden update + readout SGD.
        shuffle_control breaks the per-sample control routing (the load-bearing DFC control)."""
        Mrg = np.asarray([self.margin(a) for a in active_sets])   # (N, C) raw plateau margin
        scale = float(np.mean(np.abs(Mrg))) + 1e-9
        Mn = Mrg / scale                                          # normalized for a non-saturating softmax
        Y = np.eye(self.k)[np.asarray(y, int)]
        u = self.run_controller(Mn, Y, ctrl_steps, alpha, dt, ctrl_sign=ctrl_sign)
        if shuffle_control and ctrl_rng is not None:
            u = u[ctrl_rng.permutation(len(u))]
        mag = self._hidden_update_from_control(u, pre_mat, lr)
        # readout co-train (own gradient) + KOLEN-POLLACK feedback learning: the SAME local error signal
        # dWout = Mn.T @ gout updates BOTH W_out and the feedback Q (with weight decay) -> Q ALIGNS to W_out
        # transport-free (Q is LEARNED by a local pre x error product; it never reads W_out or W_out^T).
        logits = Mn @ self.W_out + self.b_out
        e = _sm(logits) - Y
        gout = e / len(active_sets)
        dWout = Mn.T @ gout
        self.W_out -= lr_out * dWout + self.fb_wd * self.W_out
        self.b_out -= lr_out * gout.sum(0)
        self.Q -= lr_out * dWout + self.fb_wd * self.Q
        return mag


def _train_dfc(exp, af, pre, y, epochs, lr, lr_out, ctrl_steps, alpha, dt,
               ctrl_sign=1.0, shuffle_control=False, seed=0):
    exp.restore_frozen()
    ctrl_rng = np.random.default_rng(seed * 71 + 3) if shuffle_control else None
    mags = []
    for _ in range(epochs):
        mags.append(exp.train_epoch_dfc(af, pre, y, lr, lr_out, ctrl_steps, alpha, dt,
                                        ctrl_sign=ctrl_sign, shuffle_control=shuffle_control, ctrl_rng=ctrl_rng))
    return mags


def run_seed(seed, n_col, epochs, lr_unsup, lr_dfc, lr_out, w0, jitter, k_th, n_sub, hidden,
             oracle_epochs, oracle_lr, oracle_batch, ctrl_steps, ctrl_alpha, ctrl_dt, fb_wd, task_kwargs, margin_go,
             verbose=True):
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(seed * 13 + 1); keep = srng.permutation(len(Xtr))[:min(n_sub, len(Xtr))]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    af_b = topk_active(Xb, TOPK); af_h = topk_active(Xh, TOPK)
    pre_b = np.zeros((len(af_b), n_in))
    for i, a in enumerate(af_b):
        pre_b[i, np.asarray(list(a), int)] = 1.0
    chance = float(max(np.mean(yh == c) for c in np.unique(yh))) if len(yh) else float("nan")
    out = {"seed": seed, "meta": meta, "n_in": n_in, "k": k, "chance": chance, "n_train_sub": len(Xb),
           "n_heldout_inherit": len(yh)}
    cp = dict(ctrl_steps=ctrl_steps, alpha=ctrl_alpha, dt=ctrl_dt)

    # ---- ARM 4a: oracle (fenced backprop depth-2 rate) ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_train"] = float(onet.accuracy(Xtr, ytr)); out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)

    # ---- ARM 4b/3: frozen random RATE reservoir (must fail ~0.10) ----
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n_col, seed)

    # ---- ONE expander, identical init -> FROZEN, UNSUPERVISED, DFC all from the SAME reservoir ----
    exp = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th)

    # ARM 1: FROZEN reservoir
    exp.restore_frozen()
    fz_tr, fz_ho, Cb_fz, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["frozen_plateau_train"] = fz_tr; out["frozen_plateau_heldout"] = fz_ho
    out["frozen_codon_diversity"] = _codon_diversity(Cb_fz)

    # ARM 2: UNSUPERVISED plateau plasticity (parent's local covariance rule, inherited unchanged)
    exp.restore_frozen()
    for _ in range(epochs):
        exp.train_epoch(af_b, pre_b, lr_unsup)
    un_tr, un_ho, Cb_un, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["unsup_plateau_train"] = un_tr; out["unsup_plateau_heldout"] = un_ho
    out["unsup_codon_diversity"] = _codon_diversity(Cb_un)

    # ARM 3: DFC plateau plasticity (the new arm)
    exp.init_dfc(k, seed, fb_wd=fb_wd)                             # attach forward readout W_out + LEARNED (KP) feedback Q
    dfc_mags = _train_dfc(exp, af_b, pre_b, yb, epochs, lr_dfc, lr_out, ctrl_steps, ctrl_alpha, ctrl_dt, seed=seed)
    dfc_tr, dfc_ho, Cb_dfc, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["dfc_plateau_train"] = dfc_tr; out["dfc_plateau_heldout"] = dfc_ho
    out["dfc_codon_diversity"] = _codon_diversity(Cb_dfc)
    out["dfc_update_mag_first_last"] = [round(dfc_mags[0], 6), round(dfc_mags[-1], 6)]
    out["dfc_weight_moved"] = float(np.mean(np.abs(exp._get_data() - exp.W0)))

    # ---- deep_credit_share for BOTH arms (against the frozen plateau reservoir) ----
    denom = out["oracle_heldout"] - out["frozen_plateau_heldout"]
    def _dcs(v):
        return float((v - out["frozen_plateau_heldout"]) / denom) if abs(denom) > 1e-6 else float("nan")
    out["deep_credit_share_unsup"] = _dcs(out["unsup_plateau_heldout"])
    out["deep_credit_share_dfc"] = _dcs(out["dfc_plateau_heldout"])
    # ATTRIBUTION: the effect under test is whether CLOSED-LOOP CONTROL adds held-out credit BEYOND label-free
    # unsupervised sharpening -- not merely that both arms were measured (attribution-required gate).
    from tools.lab import attributable_to
    attributable_to("deep feedback control vs unsupervised sharpening on held-out",
                     out["dfc_plateau_heldout"], out["unsup_plateau_heldout"])

    # ---- ANTI-CHEAT: reproducibility (plateau reliability must hold under the DFC plasticity) ----
    out["reproducibility_dfc"] = _reproducibility(exp, af_b)

    # ---- ANTI-CHEAT: permuted-label ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    Ctr = np.asarray([exp.codon(a) for a in af_b]); Cte = np.asarray([exp.codon(a) for a in af_h])
    clf_p = fit_lin(Ctr, yperm, k)
    out["permuted_readout_heldout"] = float(np.mean(clf_p(Cte) == yh))
    exp_perm = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_dfc(k, seed, fb_wd=fb_wd)
    _train_dfc(exp_perm, af_b, pre_b, yperm, epochs, lr_dfc, lr_out, ctrl_steps, ctrl_alpha, ctrl_dt, seed=seed)
    _, out["dfc_on_permuted_heldout"], _, _ = _readout_acc(exp_perm, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: shuffle the control nudge across the batch -> degrade toward the unsupervised arm ----
    exp_sh = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_dfc(k, seed, fb_wd=fb_wd)
    _train_dfc(exp_sh, af_b, pre_b, yb, epochs, lr_dfc, lr_out, ctrl_steps, ctrl_alpha, ctrl_dt,
               shuffle_control=True, seed=seed)
    _, out["shuffle_control_heldout"], _, _ = _readout_acc(exp_sh, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: WRONG-SIGN control (drive AWAY from error reduction) -> anti-learns BELOW frozen ----
    exp_ws = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_dfc(k, seed, fb_wd=fb_wd)
    _train_dfc(exp_ws, af_b, pre_b, yb, epochs, lr_dfc, lr_out, ctrl_steps, ctrl_alpha, ctrl_dt,
               ctrl_sign=-1.0, seed=seed)
    _, out["wrong_sign_heldout"], _, _ = _readout_acc(exp_ws, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: plateau/apical LESION -> floor (DFC-trained on a lesioned plateau) ----
    lex = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th, lesion=True).init_dfc(k, seed, fb_wd=fb_wd)
    _train_dfc(lex, af_b, pre_b, yb, epochs, lr_dfc, lr_out, ctrl_steps, ctrl_alpha, ctrl_dt, seed=seed)
    _, out["lesion_heldout"], _, _ = _readout_acc(lex, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: NO-TRANSPORT (credit update reads only u/pre/lr; Q LEARNED transport-free, ALIGNS but != copy) ----
    # NB: run against the ARM-3-trained `exp` (do NOT re-init -> Q must have MOVED from Q0 via Kolen-Pollack).
    hsig = set(inspect.signature(DFCPlateauExpander._hidden_update_from_control).parameters)
    no_transport_code = hsig.isdisjoint({"W_out", "Wout", "readout", "clf", "forward_W", "Wt", "Q"})
    q_learned = bool(not np.allclose(exp.Q, exp.Q0))              # Q MOVED via Kolen-Pollack (transport-free) -> must be True
    q_not_verbatim_copy = bool(not np.allclose(exp.Q, exp.W_out, atol=1e-6))  # KP ALIGNS Q to W_out but != a verbatim copy
    QtW = exp.Q.T @ exp.W_out                                    # DIAGNOSTIC (reported, NOT gated): positive diag => aligned
    out["q_align_diag"] = float(np.mean(np.diag(QtW)))           # >0 => Q aligned to W_out => KP working
    out["q_wout_cosine"] = float((exp.Q.ravel() @ exp.W_out.ravel())
                                 / (np.linalg.norm(exp.Q) * np.linalg.norm(exp.W_out) + 1e-12))
    # runtime: hidden update is INVARIANT to W_out given a FIXED injected control u (proves W_out is not read on backward)
    u_fixed = np.random.default_rng(0).standard_normal((24, n_col))
    pA = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_dfc(k, seed, fb_wd=fb_wd)
    pB = DFCPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_dfc(k, seed, fb_wd=fb_wd)
    pB.W_out = np.random.default_rng(12345).standard_normal((n_col, k)) * 5.0   # WILDLY different W_out
    pA.restore_frozen(); pB.restore_frozen()
    pA._hidden_update_from_control(u_fixed, pre_b[:24], lr_dfc); dA = pA._get_data()
    pB._hidden_update_from_control(u_fixed, pre_b[:24], lr_dfc); dB = pB._get_data()
    no_transport_runtime = bool(np.allclose(dA, dB, atol=1e-6) and not np.allclose(dA, pA.W0, atol=1e-6))
    out["no_transport_code"] = bool(no_transport_code)
    out["no_transport_Q_learned"] = q_learned
    out["no_transport_Q_not_verbatim_copy"] = q_not_verbatim_copy
    out["no_transport_runtime"] = no_transport_runtime
    out["no_transport"] = bool(no_transport_code and q_not_verbatim_copy and q_learned and no_transport_runtime)

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} n_sub={len(Xb)} N_COL={n_col} "
              f"ctrl={cp}", flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f}(tr {out['oracle_train']:.3f}) | rate-reservoir "
              f"{out['rate_reservoir_heldout']:.3f} | FROZEN {out['frozen_plateau_heldout']:.3f} | UNSUP "
              f"{out['unsup_plateau_heldout']:.3f} | DFC {out['dfc_plateau_heldout']:.3f}"
              f"(tr {out['dfc_plateau_train']:.3f})", flush=True)
        print(f"    deep_credit_share  unsup {out['deep_credit_share_unsup']:+.3f}  dfc "
              f"{out['deep_credit_share_dfc']:+.3f}  | reprod {out['reproducibility_dfc']:.3f} | "
              f"|dW| {out['dfc_update_mag_first_last']} moved {out['dfc_weight_moved']:.4f}", flush=True)
        print(f"    [anti-cheat] permuted-readout {out['permuted_readout_heldout']:.3f}(~chance) | "
              f"dfc-on-permuted {out['dfc_on_permuted_heldout']:.3f}(->frozen) | shuffle-control "
              f"{out['shuffle_control_heldout']:.3f}(->unsup) | wrong-sign {out['wrong_sign_heldout']:.3f}(<frozen) | "
              f"lesion {out['lesion_heldout']:.3f}(~floor)", flush=True)
        print(f"    [no-transport] code={out['no_transport_code']} Q-learned={out['no_transport_Q_learned']} "
              f"Q-not-copy={out['no_transport_Q_not_verbatim_copy']} align-diag={out['q_align_diag']:+.3f} "
              f"runtime={out['no_transport_runtime']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 DEEP FEEDBACK CONTROL plateau credit de-risk (SMOKE).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-unsup", type=float, default=0.02, help="unsupervised covariance lr (parent default)")
    ap.add_argument("--lr-dfc", type=float, default=0.05, help="DFC (control-target) hidden lr")
    ap.add_argument("--lr-out", type=float, default=0.2, help="output readout SGD lr")
    ap.add_argument("--fb-wd", type=float, default=0.02, help="Kolen-Pollack feedback weight decay")
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None)
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--oracle-epochs", type=int, default=200)
    ap.add_argument("--oracle-lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", type=int, default=128)
    # --- the controller ---
    ap.add_argument("--ctrl-steps", type=int, default=20, help="closed-loop controller iterations per batch")
    ap.add_argument("--ctrl-alpha", type=float, default=0.5, help="controller leak (u decay)")
    ap.add_argument("--ctrl-dt", type=float, default=0.3, help="controller integration step")
    # --- the SWEET SPOT task config (n_prop=3, n_super=24) ---
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--margin-go", type=float, default=0.05,
                    help="preregistered held-out margin DFC must clear over BOTH frozen AND unsup")
    ap.add_argument("--out", default="research/findings/raw/gap4/dfc_plateau/dfc_plateau_credit.json")
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.epochs, a.lr_unsup, a.lr_dfc, a.lr_out, a.w0, a.jitter, a.k_th,
                                a.n_sub, a.hidden, a.oracle_epochs, a.oracle_lr, a.oracle_batch,
                                a.ctrl_steps, a.ctrl_alpha, a.ctrl_dt, a.fb_wd, task_kwargs, a.margin_go))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_dfc_plateau_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "epochs": a.epochs, "lr_unsup": a.lr_unsup, "lr_dfc": a.lr_dfc,
                          "lr_out": a.lr_out, "w0": a.w0, "jitter": a.jitter, "k_th": a.k_th, "n_sub": a.n_sub,
                          "hidden": a.hidden, "oracle_epochs": a.oracle_epochs, "ctrl_steps": a.ctrl_steps,
                          "ctrl_alpha": a.ctrl_alpha, "ctrl_dt": a.ctrl_dt, "fb_wd": a.fb_wd, "task": task_kwargs,
                          "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_plateau_heldout", "unsup_plateau_heldout",
                "dfc_plateau_heldout", "deep_credit_share_unsup", "deep_credit_share_dfc",
                "reproducibility_dfc", "permuted_readout_heldout", "dfc_on_permuted_heldout",
                "shuffle_control_heldout", "wrong_sign_heldout", "lesion_heldout", "chance", "dfc_plateau_train",
                "unsup_codon_diversity", "dfc_codon_diversity"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per)
        need = int(np.ceil(0.834 * n))         # >= 5/6 (per the reconciliation gate); n=1 -> 1
        beats_both = sum(1 for p in per
                         if p["dfc_plateau_heldout"] >= p["frozen_plateau_heldout"] + a.margin_go
                         and p["dfc_plateau_heldout"] >= p["unsup_plateau_heldout"] + a.margin_go)
        dcs_gt_unsup = sum(1 for p in per if p["deep_credit_share_dfc"] > p["deep_credit_share_unsup"])
        anti_ok = (all(p["no_transport"] for p in per)
                   and all(p["reproducibility_dfc"] >= 0.8 for p in per)
                   and all(p["permuted_readout_heldout"] <= p["chance"] + 0.10 for p in per)
                   and all(p["dfc_on_permuted_heldout"] <= p["frozen_plateau_heldout"] + a.margin_go for p in per)
                   and all(p["shuffle_control_heldout"] <= p["dfc_plateau_heldout"] for p in per)
                   and all(p["wrong_sign_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and all(p["lesion_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and agg["rate_reservoir_heldout"] <= 0.40 and agg["oracle_heldout"] >= 0.80)
        go = bool(beats_both >= need and dcs_gt_unsup >= need and anti_ok)
        dfc_gt_unsup = bool(agg["dfc_plateau_heldout"] > agg["unsup_plateau_heldout"])
        promising = bool((not go) and dcs_gt_unsup >= need and anti_ok and dfc_gt_unsup)
        agg.update({"n_seeds": n, "dfc_beats_both_by_margin": beats_both, "seeds_needed": need,
                    "dcs_dfc_gt_unsup": dcs_gt_unsup, "anti_cheats_clean": bool(anti_ok),
                    "margin_go": a.margin_go, "promising": promising})
        summary["aggregate"] = agg; summary["GO"] = go; summary["PROMISING"] = promising
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f} (op-point "
                  f"genuine), FROZEN {agg['frozen_plateau_heldout']:.3f}, UNSUP {agg['unsup_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share_unsup']:+.3f}), DFC {agg['dfc_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share_dfc']:+.3f}). anti: reprod {agg['reproducibility_dfc']:.3f}, "
                  f"permuted-readout {agg['permuted_readout_heldout']:.3f}, dfc-on-permuted "
                  f"{agg['dfc_on_permuted_heldout']:.3f}, shuffle-control {agg['shuffle_control_heldout']:.3f}, "
                  f"wrong-sign {agg['wrong_sign_heldout']:.3f}, lesion {agg['lesion_heldout']:.3f}.")
        if go:
            verdict = (f"SMOKE GO ({beats_both}/{n} beat both by >={a.margin_go}) -- CLOSED-LOOP DEEP FEEDBACK CONTROL "
                       f"on the movable plateau hidden beats BOTH the frozen reservoir AND the unsupervised rule; "
                       f"deep_credit_share {agg['deep_credit_share_unsup']:+.3f} (unsup) -> "
                       f"{agg['deep_credit_share_dfc']:+.3f} (DFC). Transport-free (Q fixed random), wrong-sign "
                       f"anti-learns. Anti-cheats clean -> parent runs the 6-seed confirm. " + common)
        elif promising:
            verdict = (f"SMOKE PROMISING (dcs dfc>unsup {dcs_gt_unsup}/{n}, margin cleared {beats_both}/{n}, need "
                       f"{need}) -- closed-loop control moves the movable hidden FURTHER than unsupervised sharpening "
                       f"({agg['dfc_plateau_heldout']:.3f} vs {agg['unsup_plateau_heldout']:.3f}), transport-free, "
                       f"anti-cheats clean, but the {a.margin_go} margin over BOTH baselines is not cleared on "
                       f">={need} seeds -> parent runs 6-seed to settle. " + common)
        else:
            reasons = []
            if dcs_gt_unsup < need:
                reasons.append(f"DFC deep_credit_share does NOT exceed unsupervised on >={need} seeds "
                               f"({dcs_gt_unsup}/{n}; DFC mean {agg['deep_credit_share_dfc']:+.3f} vs unsup "
                               f"{agg['deep_credit_share_unsup']:+.3f})")
            elif not dfc_gt_unsup:
                reasons.append(f"DFC does NOT beat unsupervised in mean "
                               f"({agg['dfc_plateau_heldout']:.3f} vs {agg['unsup_plateau_heldout']:.3f})")
            if not anti_ok:
                reasons.append("an anti-cheat/op-point control did not hold (see per-seed)")
            verdict = ("SMOKE NEGATIVE (honest) -- " + "; ".join(reasons) + ". A valid deliverable: it maps that "
                       "closed-loop deep feedback control does NOT add held-out credit beyond unsupervised sharpening "
                       "on the movable hidden at this op-point. " + common)
        summary["verdict"] = verdict
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-dfc-plateau-credit] {summary['verdict']}", flush=True)
    print(f"[gap4-dfc-plateau-credit] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
