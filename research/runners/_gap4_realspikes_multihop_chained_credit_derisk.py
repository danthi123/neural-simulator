"""gap#4 REAL-SPIKES MULTI-HOP CHAINED credit de-risk -- the DEPTH version of the single-layer spiking port.

THE HYPOTHESIS. The committed single-layer negative
(`2026-08-01-gap4-spiking-port-sigmaprime-KP-single-layer-does-NOT-beat-frozen-6seed-NEGATIVE`; learned dcs
+0.012, 3/6) does NOT refute the depth mechanism -- it lacks the multi-hop CHAIN where KP's depth-rescue lives.
The rate result (`2026-08-01-gap4-transport-free-ceiling-FALSIFIED-...`) shows the power is at DEPTH: on MNIST,
KP-learned transport-free feedback rescues depth-4 (FA 0.531 -> KP 0.876, 6/6) while at depth-2 fixed FA already
suffices. This runner ports that DEPTH crux to the real-spikes substrate: >=2 plastic layers, chained descending
credit e_l = (e_{l+1} @ Y_l) . sigma'(v-theta)_l at each plastic layer, each Y_l KP-LEARNED transport-free.

THE HONEST QUESTION. Does multi-hop chaining BEAT the single-layer result (dcs goes clearly positive), or does
the credit DEGRADE through the spiking depth (the FA-depth-degradation the rate result showed, that KP is meant
to fix)? If multi-hop is no better than single-layer, that is a real finding (the depth-rescue does not survive
the spiking read regime), naming the next mechanism -- not a failure to hide.

ARCHITECTURE (2 plastic real-spikes coincidence-plateau layers, stacked):
  u0 = features (NF, input current-driven, they SPIKE) --W1--> u1 = cols0 (NC0 plateau) --W2--> u2 = cols1 (NC1
  plateau) --W_out--> logits(k). W1 = layer-0 coincidence weights (plastic); W2 = layer-1 coincidence weights
  (plastic); W_out = internal readout head (target access = legit). Each layer's forward pass + sigma'(v-theta)
  read is REAL SPIKES (features spike, coincidence detection fires, membrane read is graded).
  INTER-LAYER COUPLING (the NAMED architectural shortcut): the columns never somatically SPIKE
  (documented on this substrate), so cols0's graded margin M0 cannot drive cols1 via spikes. M0 is re-encoded as
  bounded input CURRENT for layer-1's spiking feature neurons (a fixed forward gain, calibrated once on the
  frozen reservoir, shared across ALL arms). The FORWARD coupling uses layer-1's own forward weights (legit); the
  CREDIT path is transport-free per layer (Y_l never reads any W).

THE CHAINED TRANSPORT-FREE CREDIT (mode credit; permuted/wrong_sign corrupt the target):
  e_out = P - onehot                                  # readout error (target access = legit)
  e_u2  = (e_out @ Y_out) . sigma'_1                  # error at cols1; reads Y_out (k,NC1), NOT W_out
  dW2   = -lr * (e_u2^T @ M0) / N                     # layer-1 weight update; input activity = M0 (cols0 margins)
  e_u1  = (e_u2  @ Y2   ) . sigma'_0                  # error at cols0; reads Y2 (NC1,NC0), NOT W2
  dW1   = -lr * (e_u1^T @ pre0) / N                   # layer-0 weight update; input activity = feature spikes
  each layer: add dW, max(0) (excitatory), per-column L2-renorm to init (the parent's local homeostasis).
  KP (learned, transport-free): Y_out <- kp(Y_out, pre=M1, post=e_out); Y2 <- kp(Y2, pre=M0, post=e_u2).
  The SAME parent `_kp_update` generalizes: pre = the forward weight's INPUT activation, post = its OUTPUT error.
  sigma'(v-theta) is read per layer from the SOMATIC membrane (atan-surrogate), normalized to a RELATIVE gate
  (mean 1.0) so it selects WHICH columns receive credit without its ~0.004 absolute magnitude vanishing the update.

THE SINGLE VARIABLE vs the FROZEN MULTI-LAYER reservoir: are BOTH layers' weights shaped by the chained credit?
ARMS: FROZEN 2-layer reservoir (no plasticity, only readout learns); CREDIT-multihop (both layers plastic via the
chained rule). Plus the SINGLE-LAYER arm (the parent's `SigmaPrimeCreditExpander.train_credit`, features->cols
only) for the head-to-head the finding calls for. deep_credit_share = (credit - frozen)/(oracle - frozen).

ANTI-CHEATS (all on the multihop credit arm): permuted-label -> chance; WRONG-SIGN teacher -> at/below floor;
apical/layer LESION (both layers) -> floor; NO-TRANSPORT per layer (code: the chained-update method + `_kp_update`
expose no readout/forward weight; Y's move only via `_kp_update`); reproducibility >= 0.8 through 2 layers;
oracle >= 0.80, rate-reservoir fails. Per-layer credit-alignment cos(Y_l, W_l) recorded (KP working <=> Y aligns
with the forward weight). Backend stamped. NO sim/ edit (subclass of the parent expander; additive runner).

Run (numpy CPU; ~ (NF+NC0)+(NC0+NC1) neurons):
    SIM_BACKEND=numpy python -m research.runners._gap4_realspikes_multihop_chained_credit_derisk --seeds 42 --epochs 40
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import inspect
import json
import time
import traceback
from pathlib import Path

import numpy as np

from research.runners._gap4_realspikes_sigmaprime_credit_derisk import SigmaPrimeCreditExpander, _softmax
from research.runners._gap4_plastic_plateau_credit_derisk import (
    fit_lin, topk_active, _reproducibility, _codon_diversity, _rate_reservoir_heldout, FLOOR, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)
from research.runners._emerge12_stageB2_bridge_tm_derisk import reset_soma, _clear_apical
from sim.backend import to_host as _host


# ============================================================================================================
# ChainLayer -- a real-spikes coincidence-plateau expander that also accepts a GRADED per-feature input CURRENT
# (so an upstream layer's graded margin can drive it) and can apply a supplied credit weight-delta in place.
# ============================================================================================================
class ChainLayer(SigmaPrimeCreditExpander):
    def forward_drive(self, drive_vec):
        """ONE real-spikes forward pass driven by a per-FEATURE current vector -> (margin[NC], sigmap[NC]).
        Generalizes the parent forward_credit (which drives a binary active-set with self.drive_pa): here the NF
        feature neurons receive `drive_vec` pA each. margin = max(0, mean_window(v_apical) - FLOOR); sigmap =
        mean_window sigma'(v_soma - vt) (atan-surrogate) -- the graded low-CV somatic credit gate."""
        b = self.b; xp = b.xp if hasattr(b, "xp") else np
        n = int(b.core_config.num_neurons)
        reset_soma(b); _clear_apical(b)
        inp = np.zeros(n, np.float32)
        inp[self.ci[:self.NF]] = np.asarray(drive_vec, np.float32)
        inp_x = xp.asarray(inp)
        cidx = self.ci[self.NF:self.NF + self.NC]
        sp = np.zeros(self.NC); ap = np.zeros(self.NC)
        for _ in range(self.n_steps):
            b.cp_external_input_current[:] = inp_x
            b._run_one_simulation_step()
            v = np.asarray(_host(b.cp_membrane_potential_v))[cidx].astype(np.float64)
            sp += 1.0 / (1.0 + (self.beta * (v - self.col_vt)) ** 2)
            vap = getattr(b, "cp_v_apical", None)
            if vap is not None:
                ap += np.asarray(_host(vap))[cidx].astype(np.float64)
        margin = np.maximum(0.0, ap / self.n_steps - FLOOR)
        return margin, sp / self.n_steps

    def apply_dW(self, dW):
        """Add credit delta dW (shape (NC, NF)) to the coincidence weights, then max(0) + per-column L2-renorm to
        the initial norm (the parent's local homeostasis). Mirrors train_credit's write block."""
        data = self._get_data()
        data = data + dW[self.syn_col, self.syn_feat]
        np.maximum(data, 0.0, out=data)
        cur = np.sqrt(np.array([np.sum(data[self.syn_col == c] ** 2) for c in range(self.NC)]))
        scale = np.where(cur > 1e-9, self.col_norm0 / (cur + 1e-12), 1.0)
        data = data * scale[self.syn_col]
        self._set_data(data)

    def dense_forward_weight(self):
        """Reconstruct the dense forward weight matrix (NC, NF) from the substrate connections (for Y-vs-W
        alignment ONLY -- never read by any credit path)."""
        data = self._get_data()
        W = np.zeros((self.NC, self.NF))
        W[self.syn_col, self.syn_feat] = data
        return W


def _mk_layer(n_feat, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, beta, feedback, kp_lr, kp_decay, lesion=False):
    e = ChainLayer(n_feat, n_col, seed, w0=w0, jitter=jitter, k_th=k_th, lesion=lesion)
    return e.configure_read(drive_pa, n_steps).configure_credit(beta, feedback, kp_lr, kp_decay, seed)


class MultiHopChain:
    """A stack of 2 real-spikes ChainLayers with a fixed forward coupling gain and a chained transport-free
    credit rule. `train` shapes BOTH layers; the FROZEN control is the SAME stack with lr_ff=0."""

    def __init__(self, layers, feedback, kp_lr, kp_decay, couple_topk=8):
        assert len(layers) == 2, "smoke build is the minimal 2-hop chain (the depth crux needs >=2 plastic layers)"
        self.layers = layers
        self.L0, self.L1 = layers
        self.NF = self.L0.NF
        self.feedback = str(feedback); self.kp_lr = float(kp_lr); self.kp_decay = float(kp_decay)
        self.couple_topk = int(couple_topk)
        self.couple_scale = 1.0; self.couple_cap = float(self.L1.drive_pa) * 1.5

    def restore_frozen(self):
        for L in self.layers:
            L.restore_frozen()

    # ---- forward ----
    def _couple(self, m0):
        """Inter-layer forward coupling: cols0's graded margin -> layer-1 input CURRENT, but SPARSIFIED to the
        top-`couple_topk` columns (k-winners). The coincidence-plateau expander is designed for SPARSE input (the
        single layer is driven by TOPK=4 active features); an all-columns-active dense drive pushes layer-1 out of
        its plateau regime (measured: dense coupling -> layer-1 never plateaus). Sparse competitive selection between
        cortical layers is the biological form. Graded WITHIN the winners (magnitude preserved)."""
        d = np.zeros_like(m0)
        if self.couple_topk >= len(m0):
            idx = np.where(m0 > 0)[0]
        else:
            idx = np.argpartition(m0, -self.couple_topk)[-self.couple_topk:]
            idx = idx[m0[idx] > 0]
        d[idx] = self.couple_scale * m0[idx]
        np.clip(d, 0.0, self.couple_cap, out=d)
        return d

    def calibrate_coupling(self, drive0_batch, pct=60.0):
        """Fix the inter-layer forward gain ONCE on the frozen reservoir: scale the per-input top-k cols0 margins so
        their `pct` percentile maps to layer-1's drive_pa (the current that makes a binary active feature spike).
        Shared across ALL arms -> the only variable is plasticity, not the coupling gain."""
        M0 = np.array([self.L0.forward_drive(dv)[0] for dv in drive0_batch])
        k = min(self.couple_topk, M0.shape[1])
        topvals = np.sort(M0, axis=1)[:, -k:]                       # per-input top-k cols0 margins (the winners)
        pos = topvals[topvals > 0]
        ref = np.percentile(pos, pct) if pos.size else 1.0
        self.couple_scale = float(self.L1.drive_pa / (ref + 1e-9))
        return {"couple_scale": self.couple_scale, "M0_topk_ref_pct": float(ref), "M0_max": float(M0.max()),
                "M0_mean": float(M0.mean()), "M0_frac_active": float(np.mean(M0 > 0)),
                "couple_topk": self.couple_topk}

    def forward(self, drive0_batch):
        """Real-spikes forward through both layers. Returns ([M0,M1],[S0,S1]) with M0/M1 graded margins and
        S0/S1 the per-layer graded sigma'(v-theta) gates."""
        N = len(drive0_batch)
        M0 = np.zeros((N, self.L0.NC)); S0 = np.zeros((N, self.L0.NC))
        M1 = np.zeros((N, self.L1.NC)); S1 = np.zeros((N, self.L1.NC))
        for i, dv in enumerate(drive0_batch):
            m0, s0 = self.L0.forward_drive(dv)
            m1, s1 = self.L1.forward_drive(self._couple(m0))
            M0[i] = m0; S0[i] = s0; M1[i] = m1; S1[i] = s1
        return [M0, M1], [S0, S1]

    def top_reps(self, drive0_batch):
        """Top-layer graded margins M1 (N, NC1) with the CURRENT weights (codon = M1 > 0)."""
        return self.forward(drive0_batch)[0][1]

    # ---- chained transport-free credit ----
    def train(self, drive0_batch, pre0_mat, y, k, epochs, lr_ff, lr_out, seed, mode="credit"):
        """Chained sigma'-gated credit over 2 plastic layers. Single variable = lr_ff (0 = frozen).
        modes: credit|permuted|wrong_sign = TRANSPORT-FREE (e_u2 reads Y_out not W_out; e_u1 reads Y2 not W2; Y's
        move only via `_kp_update`). mode='oracle' = the W^T TRANSPORT CEILING diagnostic (NOT shippable, clearly
        labeled): each epoch Y_out<-W_out^T and Y2<-the TRUE layer-1 forward weight, so the descending learning
        signal is the EXACT loss gradient routed by the forward weights (Bellec e-prop's L_j = dE/dz_j with perfect
        transport) x the same sigma'(v-theta) eligibility. It bounds how much DIRECTED credit this real-spikes read
        regime can carry AT ALL; the KP arm is the shippable transport-free candidate measured against it."""
        rng = np.random.default_rng(seed * 7 + 3)
        W_out = 0.01 * rng.standard_normal((self.L1.NC, k)); b_out = np.zeros(k)
        y_rng = np.random.default_rng(seed + 9973)
        Y_out = y_rng.standard_normal((k, self.L1.NC)) / np.sqrt(self.L1.NC)           # feedback for the readout
        Y2 = y_rng.standard_normal((self.L1.NC, self.L0.NC)) / np.sqrt(self.L0.NC)     # feedback for W2 (layer-1 fwd)
        Y_out0, Y20 = Y_out.copy(), Y2.copy()
        onehot = np.eye(k)[y]; N = len(drive0_batch)
        mags = []
        for _ in range(epochs):
            (M0, M1), (S0, S1) = self.forward(drive0_batch)               # both layers, CURRENT weights
            self._mu = M1.mean(0); self._sd = M1.std(0) + 1e-6
            M1n = (M1 - self._mu) / self._sd
            for _ in range(20):                                          # inner head fit (well-conditioned error)
                P = _softmax(M1n @ W_out + b_out); g = (P - onehot) / N
                W_out = W_out - lr_out * (M1n.T @ g + 3e-3 * W_out); b_out = b_out - lr_out * g.sum(0)
            P = _softmax(M1n @ W_out + b_out); e_out = P - onehot
            if mode == "wrong_sign":
                e_out = -e_out
            if lr_ff > 0.0:
                S1n = S1 / (S1.mean() + 1e-9); S0n = S0 / (S0.mean() + 1e-9)
                if mode == "oracle":
                    # ORACLE-TRANSPORT-CEILING (begin) -- W^T routing = the EXACT-backprop ceiling (NOT shippable)
                    Y_out = W_out.T.copy(); Y2 = self.L1.dense_forward_weight()   # TRUE forward weights, re-read each epoch
                    # ORACLE-TRANSPORT-CEILING (end)
                e_u2 = (e_out @ Y_out) * S1n                             # error at cols1  (reads Y_out, NOT W_out)
                dW2 = -lr_ff * (e_u2.T @ M0) / N                         # layer-1 update; input activity = M0
                self.L1.apply_dW(dW2)
                e_u1 = (e_u2 @ Y2) * S0n                                 # error at cols0  (reads Y2, NOT W2)
                dW1 = -lr_ff * (e_u1.T @ pre0_mat) / N                   # layer-0 update; input activity = spikes
                self.L0.apply_dW(dW1)
                mags.append([float(np.mean(np.abs(dW1))), float(np.mean(np.abs(dW2)))])
                if self.feedback == "learned" and mode == "credit":
                    Y_out = self.L1._kp_update(Y_out, M1, e_out, lr_out)  # transport-free (parent's verified kp)
                    Y2 = self.L1._kp_update(Y2, M0, e_u2, lr_out)
            else:
                mags.append([0.0, 0.0])
            W_out = W_out - lr_out * (M1.T @ e_out) / N; b_out = b_out - lr_out * e_out.mean(0)
        self._Y_moved = bool((not np.allclose(Y_out, Y_out0)) or (not np.allclose(Y2, Y20)))
        # per-layer credit-alignment cos(Y_l, W_l): KP working <=> feedback aligns with the forward weight
        W2 = self.L1.dense_forward_weight()                             # (NC1, NC0)  -- diagnostic read only
        self._align = {
            "Y_out_vs_Wout": _cos(Y_out.ravel(), W_out.T.ravel()),      # Y_out (k,NC1) mirrors W_out^T (k,NC1)
            "Y2_vs_W2": _cos(Y2.ravel(), W2.ravel()),                   # Y2 (NC1,NC0) mirrors W2 (NC1,NC0)
        }
        return mags, (W_out, b_out)


def _cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else float("nan")


def _chain_eval(chain, drive0_tr, ytr, drive0_te, yte, k):
    """Primary (binary top-codon, like the parent) + graded (standardized top-margin) refits, both fit_lin.
    Returns dict of train/heldout for both reads + the train codon matrix (for diversity/sparsity)."""
    Mtr = chain.top_reps(drive0_tr); Mte = chain.top_reps(drive0_te)
    Ctr = (Mtr > 0).astype(np.float64); Cte = (Mte > 0).astype(np.float64)
    clf_c = fit_lin(Ctr, ytr, k)
    codon_tr = float(np.mean(clf_c(Ctr) == ytr)); codon_ho = float(np.mean(clf_c(Cte) == yte))
    mu = Mtr.mean(0); sd = Mtr.std(0) + 1e-6
    clf_g = fit_lin((Mtr - mu) / sd, ytr, k)
    grad_tr = float(np.mean(clf_g((Mtr - mu) / sd) == ytr)); grad_ho = float(np.mean(clf_g((Mte - mu) / sd) == yte))
    return {"codon_train": codon_tr, "codon_heldout": codon_ho, "graded_train": grad_tr,
            "graded_heldout": grad_ho}, Ctr


def _single_eval(exp, af_tr, ytr, af_te, yte, k):
    """Single-layer arm eval, SAME two reads as the chain (codon via exp.codon = parent's; graded via forward_batch)."""
    Ctr = np.asarray([exp.codon(a) for a in af_tr]); Cte = np.asarray([exp.codon(a) for a in af_te])
    clf_c = fit_lin(Ctr, ytr, k)
    codon_tr = float(np.mean(clf_c(Ctr) == ytr)); codon_ho = float(np.mean(clf_c(Cte) == yte))
    Mtr, _ = exp.forward_batch(af_tr); Mte, _ = exp.forward_batch(af_te)
    mu = Mtr.mean(0); sd = Mtr.std(0) + 1e-6
    clf_g = fit_lin((Mtr - mu) / sd, ytr, k)
    grad_tr = float(np.mean(clf_g((Mtr - mu) / sd) == ytr)); grad_ho = float(np.mean(clf_g((Mte - mu) / sd) == yte))
    return {"codon_train": codon_tr, "codon_heldout": codon_ho, "graded_train": grad_tr, "graded_heldout": grad_ho}


def _chain_reproducibility(chain, drive0_batch, n=8):
    M1 = chain.top_reps(drive0_batch[:n]); M2 = chain.top_reps(drive0_batch[:n])
    C1 = (M1 > 0).astype(float); C2 = (M2 > 0).astype(float)
    return float(np.mean([np.corrcoef(C1[i], C2[i])[0, 1] if C1[i].std() > 0 and C2[i].std() > 0 else 1.0
                          for i in range(len(C1))]))


def _dcs(credit_ho, frozen_ho, oracle_ho):
    d = oracle_ho - frozen_ho
    return float((credit_ho - frozen_ho) / d) if abs(d) > 1e-6 else float("nan")


def run_seed(seed, n_col, n_col2, epochs, lr_ff, lr_out, w0, jitter, k_th, n_sub, hidden, oracle_epochs, oracle_lr,
             oracle_batch, drive_pa, drive_pa2, n_steps, beta, feedback, kp_lr, kp_decay, couple_topk, task_kwargs,
             margin_go, verbose=True):
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(seed * 13 + 1); keep = srng.permutation(len(Xtr))[:min(n_sub, len(Xtr))]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    af_b = topk_active(Xb, TOPK); af_h = topk_active(Xh, TOPK)
    chance = float(max(np.mean(yh == c) for c in np.unique(yh))) if len(yh) else float("nan")
    out = {"seed": seed, "n_in": n_in, "k": k, "chance": chance, "n_train_sub": len(Xb), "n_heldout_inherit": len(yh),
           "n_col": n_col, "n_col2": n_col2, "drive_pa": drive_pa, "drive_pa2": drive_pa2, "n_steps": n_steps,
           "beta": beta, "feedback": feedback, "epochs": epochs}

    # ---- op-point controls: oracle (backprop depth-2) + frozen random RATE reservoir ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n_col, seed)

    # ---- layer-0 input drive vectors (binary active feats x drive_pa) + layer-0 pre-activity (real feat spikes) ----
    drive0_b = np.zeros((len(af_b), n_in)); drive0_h = np.zeros((len(af_h), n_in))
    for i, a in enumerate(af_b):
        drive0_b[i, np.asarray(list(a), int)] = drive_pa
    for i, a in enumerate(af_h):
        drive0_h[i, np.asarray(list(a), int)] = drive_pa

    def mk_chain(lesion=False):
        L0 = _mk_layer(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, beta, feedback, kp_lr, kp_decay, lesion)
        L1 = _mk_layer(n_col, n_col2, seed * 100003 + 7, w0, jitter, k_th, drive_pa2, n_steps, beta, feedback,
                       kp_lr, kp_decay, lesion)
        return MultiHopChain([L0, L1], feedback, kp_lr, kp_decay, couple_topk=couple_topk)

    ch = mk_chain()
    pre0_b = np.asarray([ch.L0.feat_spike_counts(a) for a in af_b])       # weight-independent -> precompute once
    ch.restore_frozen()
    out["coupling"] = ch.calibrate_coupling(drive0_b)                     # fix the inter-layer gain (frozen, shared)

    # ---- ARM 1: FROZEN 2-layer reservoir (both layers fixed; only the readout refit learns) ----
    ch.restore_frozen()
    fz, Cfz = _chain_eval(ch, drive0_b, yb, drive0_h, yh, k)
    out["frozen_codon_heldout"] = fz["codon_heldout"]; out["frozen_graded_heldout"] = fz["graded_heldout"]
    out["frozen_codon_diversity"] = _codon_diversity(Cfz); out["frozen_codon_sparsity"] = float(Cfz.mean())

    # ---- ARM 2: CREDIT-multihop (BOTH layers plastic via the chained transport-free rule) ----
    ch.restore_frozen()
    mags, _ = ch.train(drive0_b, pre0_b, yb, k, epochs, lr_ff, lr_out, seed, mode="credit")
    cr, Ccr = _chain_eval(ch, drive0_b, yb, drive0_h, yh, k)
    out["credit_codon_heldout"] = cr["codon_heldout"]; out["credit_graded_heldout"] = cr["graded_heldout"]
    out["credit_codon_train"] = cr["codon_train"]; out["credit_graded_train"] = cr["graded_train"]
    out["credit_codon_diversity"] = _codon_diversity(Ccr); out["credit_codon_sparsity"] = float(Ccr.mean())
    out["update_mag_first_last"] = [mags[0], mags[-1]]
    out["Y_moved"] = bool(getattr(ch, "_Y_moved", False))
    out["per_layer_alignment"] = getattr(ch, "_align", {})

    out["dcs_multihop_codon"] = _dcs(cr["codon_heldout"], fz["codon_heldout"], out["oracle_heldout"])
    out["dcs_multihop_graded"] = _dcs(cr["graded_heldout"], fz["graded_heldout"], out["oracle_heldout"])
    out["reproducibility"] = _chain_reproducibility(ch, drive0_b)

    # ---- SINGLE-LAYER arm (the parent's rule, features->cols only) for the head-to-head the finding calls for ----
    sl = _mk_layer(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, beta, feedback, kp_lr, kp_decay)
    pre_sl = np.asarray([sl.feat_spike_counts(a) for a in af_b])
    sl.restore_frozen()
    sfz = _single_eval(sl, af_b, yb, af_h, yh, k)
    sl.restore_frozen()
    sl.train_credit(af_b, pre_sl, yb, k, epochs, lr_ff, lr_out, seed, mode="credit")
    scr = _single_eval(sl, af_b, yb, af_h, yh, k)
    out["single_frozen_codon_heldout"] = sfz["codon_heldout"]; out["single_credit_codon_heldout"] = scr["codon_heldout"]
    out["single_frozen_graded_heldout"] = sfz["graded_heldout"]; out["single_credit_graded_heldout"] = scr["graded_heldout"]
    out["dcs_single_codon"] = _dcs(scr["codon_heldout"], sfz["codon_heldout"], out["oracle_heldout"])
    out["dcs_single_graded"] = _dcs(scr["graded_heldout"], sfz["graded_heldout"], out["oracle_heldout"])

    # ---- ANTI-CHEAT: permuted-label multihop -> chance (target corrupted; real credit collapses) ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    ch.restore_frozen()
    ch.train(drive0_b, pre0_b, yperm, k, epochs, lr_ff, lr_out, seed, mode="credit")
    pe, _ = _chain_eval(ch, drive0_b, yb, drive0_h, yh, k)
    out["permuted_codon_heldout"] = pe["codon_heldout"]; out["permuted_graded_heldout"] = pe["graded_heldout"]

    # ---- DIAGNOSTIC: wrong-sign teacher (ill-posed under a fresh true-label refit for sign-symmetric tasks;
    #      reported, not hard-gated -- same caveat the parent documents) ----
    ch.restore_frozen()
    ch.train(drive0_b, pre0_b, yb, k, epochs, lr_ff, lr_out, seed, mode="wrong_sign")
    ws, _ = _chain_eval(ch, drive0_b, yb, drive0_h, yh, k)
    out["wrong_sign_codon_heldout"] = ws["codon_heldout"]; out["wrong_sign_graded_heldout"] = ws["graded_heldout"]

    # ---- THE DIRECTED-CREDIT quantity: the lift ATTRIBUTABLE TO CORRECT LABELS = credit - permuted (a label-shuffle
    #      that keeps the SAME plasticity dynamics). A stacked plastic reservoir lifts held-out label-AGNOSTICALLY
    #      (renorm + sigma'-gated perturbation makes the top codon more separable); only credit>permuted is credit. ----
    out["credit_vs_permuted_graded"] = float(cr["graded_heldout"] - pe["graded_heldout"])
    out["credit_vs_wrongsign_graded"] = float(cr["graded_heldout"] - ws["graded_heldout"])
    _dd = out["oracle_heldout"] - out["frozen_graded_heldout"]
    out["dcs_directed_graded"] = float((cr["graded_heldout"] - pe["graded_heldout"]) / _dd) if abs(_dd) > 1e-6 else float("nan")

    # ---- ANTI-CHEAT: apical/plateau LESION (both layers) -> floor ----
    lex = mk_chain(lesion=True)
    pre0_l = np.asarray([lex.L0.feat_spike_counts(a) for a in af_b])
    lex.restore_frozen(); lex.calibrate_coupling(drive0_b)
    lex.train(drive0_b, pre0_l, yb, k, epochs, lr_ff, lr_out, seed, mode="credit")
    le, _ = _chain_eval(lex, drive0_b, yb, drive0_h, yh, k)
    out["lesion_codon_heldout"] = le["codon_heldout"]; out["lesion_graded_heldout"] = le["graded_heldout"]

    # ---- ORACLE-DIRECTED arm (the W^T TRANSPORT CEILING for directed credit on THIS real-spikes read regime; NOT
    #      shippable, clearly labeled): each epoch the descending learning signal is the EXACT loss gradient routed by
    #      the TRUE forward weights (Y_out=W_out^T, Y2=layer-1 forward weight) x the SAME sigma'(v-theta) eligibility as
    #      the KP arm. This ISOLATES the wall: on the CURRENT task, oracle==permuted => the read regime carries no
    #      DIRECTED signal (or none is NEEDED = generic plasticity already solves it); if oracle-permuted OPENS on the
    #      harder task where frozen fails, directed credit IS measurable there; where KP sits (==oracle vs ==permuted)
    #      then says whether transport-free feedback works or the alignment is the wall. Uses the same `ch` + coupling. ----
    ch.restore_frozen()
    ch.train(drive0_b, pre0_b, yb, k, epochs, lr_ff, lr_out, seed, mode="oracle")
    orc, _ = _chain_eval(ch, drive0_b, yb, drive0_h, yh, k)
    out["oracle_directed_codon_heldout"] = orc["codon_heldout"]
    out["oracle_directed_graded_heldout"] = orc["graded_heldout"]
    out["oracle_directed_alignment"] = getattr(ch, "_align", {})           # Y==W by construction => ~1.0 (sanity)
    # DIRECTED-CREDIT isolation (both reads): the lift ATTRIBUTABLE TO CORRECT LABELS = arm - permuted (a label shuffle
    # keeping the SAME plasticity dynamics). oracle-permuted = the CEILING for directed credit; kp-permuted = the
    # shippable candidate's directed lift. > 0 => that arm routes CORRECT-label error beyond label-agnostic plasticity.
    out["directed_oracle_graded"] = float(orc["graded_heldout"] - pe["graded_heldout"])
    out["directed_oracle_codon"] = float(orc["codon_heldout"] - pe["codon_heldout"])
    out["directed_kp_graded"] = float(cr["graded_heldout"] - pe["graded_heldout"])
    out["directed_kp_codon"] = float(cr["codon_heldout"] - pe["codon_heldout"])

    # ---- ANTI-CHEAT: NO-TRANSPORT (code, docstrings stripped) -- per layer / per feedback ----
    tsig = set(inspect.signature(MultiHopChain.train).parameters)
    kp_fn = SigmaPrimeCreditExpander._kp_update
    src_kp = inspect.getsource(kp_fn).replace(kp_fn.__doc__ or "", "")
    src_train = inspect.getsource(MultiHopChain.train).replace(MultiHopChain.train.__doc__ or "", "")
    # The training LOOP (everything BEFORE the post-hoc alignment diagnostic) is the credit path. The forward weight
    # is read ONLY in the diagnostic (dense_forward_weight, after the loop) -- exactly like reading W_out for eval.
    # The mode=='oracle' branch DELIBERATELY routes error by W^T transport as the labeled CEILING; it is gated by
    # `if mode == "oracle"` and NEVER runs for the shippable credit/KP path. Excise that guarded, sentinel-delimited
    # block before the transport scan (as the post-loop alignment diagnostic is already excised), and PROVE the guard
    # is present -> then scan the transport-FREE remainder for any forward-weight read.
    ob0 = src_train.find("# ORACLE-TRANSPORT-CEILING (begin)"); ob1 = src_train.find("# ORACLE-TRANSPORT-CEILING (end)")
    oracle_guarded = bool(ob0 > 0 and ob1 > ob0 and 'if mode == "oracle":' in src_train[max(0, ob0 - 160):ob0])
    src_tf = (src_train[:ob0] + src_train[ob1:]) if (ob0 > 0 and ob1 > ob0) else src_train   # transport-free path only
    train_body = src_tf.split("# per-layer credit-alignment")[0]
    # (a) the feedbacks Y_out/Y2 are assigned ONLY by _kp_update (init lines carry `y_rng`; snapshots carry `0`).
    y_assign_ok = all(("_kp_update" in ln) for ln in train_body.splitlines()
                      if ("Y_out =" in ln or "Y2 =" in ln) and "y_rng" not in ln and "Y_out0" not in ln)
    # (b) the transport-free path never reads a forward/coincidence weight (no dense_forward_weight / .W0 / W_out.T in
    #     the loop), and (c) the update kernel _kp_update holds no self.W / W_out. W_out in the loop = the readout head
    #     fit (legitimate target access, exactly as the parent single-layer runner permits).
    out["no_transport_code"] = bool(tsig.isdisjoint({"W_out", "readout", "clf", "Wout", "Wt"})
                                    and "W_out" not in src_kp and "self.W" not in src_kp
                                    and "dense_forward_weight" not in train_body and ".W0" not in train_body
                                    and "W_out.T" not in train_body and oracle_guarded and y_assign_ok)

    if verbose:
        al = out["per_layer_alignment"]
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} NC0={n_col} NC1={n_col2} "
              f"fb={feedback} couple_scale={out['coupling']['couple_scale']:.3g} "
              f"M0_active={out['coupling']['M0_frac_active']:.2f}", flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f} | rate-res {out['rate_reservoir_heldout']:.3f} | "
              f"[MULTIHOP codon] FROZEN {out['frozen_codon_heldout']:.3f} CREDIT {out['credit_codon_heldout']:.3f} "
              f"dcs {out['dcs_multihop_codon']:+.3f} | [graded] FZ {out['frozen_graded_heldout']:.3f} "
              f"CR {out['credit_graded_heldout']:.3f} dcs {out['dcs_multihop_graded']:+.3f}", flush=True)
        print(f"    [SINGLE codon] FZ {out['single_frozen_codon_heldout']:.3f} CR {out['single_credit_codon_heldout']:.3f} "
              f"dcs {out['dcs_single_codon']:+.3f} | [graded] dcs {out['dcs_single_graded']:+.3f}", flush=True)
        print(f"    [align] Y_out.Wout {al.get('Y_out_vs_Wout', float('nan')):+.3f} Y2.W2 "
              f"{al.get('Y2_vs_W2', float('nan')):+.3f} | Y_moved {out['Y_moved']} | |dW1,dW2| last {mags[-1]}", flush=True)
        print(f"    [anti-cheat] reprod {out['reproducibility']:.3f} | permuted codon {out['permuted_codon_heldout']:.3f} "
              f"| wrong_sign {out['wrong_sign_codon_heldout']:.3f} | lesion {out['lesion_codon_heldout']:.3f} | "
              f"no-transport {out['no_transport_code']}", flush=True)
        print(f"    [ISOLATION graded] frozen {out['frozen_graded_heldout']:.3f} permuted {out['permuted_graded_heldout']:.3f} "
              f"KP {out['credit_graded_heldout']:.3f} ORACLE {out['oracle_directed_graded_heldout']:.3f} || "
              f"directed: oracle-perm {out['directed_oracle_graded']:+.3f} KP-perm {out['directed_kp_graded']:+.3f}", flush=True)
        print(f"    [ISOLATION codon ] frozen {out['frozen_codon_heldout']:.3f} permuted {out['permuted_codon_heldout']:.3f} "
              f"KP {out['credit_codon_heldout']:.3f} ORACLE {out['oracle_directed_codon_heldout']:.3f} || "
              f"directed: oracle-perm {out['directed_oracle_codon']:+.3f} KP-perm {out['directed_kp_codon']:+.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 REAL-SPIKES MULTI-HOP CHAINED transport-free credit de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--n-col2", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr-ff", type=float, default=0.05)
    ap.add_argument("--lr-out", type=float, default=0.1)
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None)
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--oracle-epochs", type=int, default=200)
    ap.add_argument("--oracle-lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", type=int, default=128)
    ap.add_argument("--drive-pa", type=float, default=1200.0)
    ap.add_argument("--drive-pa2", type=float, default=1200.0)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--feedback", choices=["fixed", "learned"], default="learned")
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--couple-topk", type=int, default=8, help="k-winner sparsity of the inter-layer coupling")
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--margin-go", type=float, default=0.05)
    ap.add_argument("--task-hard", action="store_true",
                    help="HARDER task where the frozen reservoir FAILS (n_super=48, n_prop=4 => k=17; per b65e2cb3). "
                         "Overrides --n-super/--n-prop. The isolation reads only OPEN UP if directed credit becomes "
                         "measurable when the reservoir has room; the easy default (n_prop=3, k=9) is where frozen ~ oracle.")
    ap.add_argument("--out", default="research/findings/raw/gap4/realspikes/multihop_chained_credit.json")
    a = ap.parse_args()
    if a.task_hard:
        a.n_super = 48; a.n_prop = 4                          # k = 2^4 + 1 = 17, the reservoir-fails regime
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.n_col2, a.epochs, a.lr_ff, a.lr_out, a.w0, a.jitter, a.k_th, a.n_sub,
                                a.hidden, a.oracle_epochs, a.oracle_lr, a.oracle_batch, a.drive_pa, a.drive_pa2,
                                a.n_steps, a.beta, a.feedback, a.kp_lr, a.kp_decay, a.couple_topk, task_kwargs,
                                a.margin_go))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_realspikes_multihop_chained_credit", "seeds": a.seeds,
               "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "n_col2": a.n_col2, "epochs": a.epochs, "lr_ff": a.lr_ff,
                          "lr_out": a.lr_out, "beta": a.beta, "feedback": a.feedback, "kp_lr": a.kp_lr,
                          "kp_decay": a.kp_decay, "couple_topk": a.couple_topk, "drive_pa": a.drive_pa,
                          "drive_pa2": a.drive_pa2, "n_steps": a.n_steps, "task": task_kwargs,
                          "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_codon_heldout", "credit_codon_heldout",
                "frozen_graded_heldout", "credit_graded_heldout", "dcs_multihop_codon", "dcs_multihop_graded",
                "dcs_single_codon", "dcs_single_graded", "dcs_directed_graded", "reproducibility",
                "permuted_graded_heldout", "wrong_sign_graded_heldout", "lesion_graded_heldout",
                "credit_vs_permuted_graded", "credit_vs_wrongsign_graded", "permuted_codon_heldout",
                "wrong_sign_codon_heldout", "lesion_codon_heldout", "chance",
                "oracle_directed_graded_heldout", "oracle_directed_codon_heldout",
                "directed_oracle_graded", "directed_oracle_codon", "directed_kp_graded", "directed_kp_codon"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per); need = int(np.ceil(0.834 * n))
        # PRIMARY read for GO = the graded top-margin refit (the faithful somatic read; the columns never spike, so
        # the binary codon can be degenerate). The codon read is reported alongside for parent-comparability.
        beats = sum(1 for p in per
                    if p["credit_graded_heldout"] >= p["frozen_graded_heldout"] + a.margin_go)
        dcs_pos = sum(1 for p in per if p["dcs_multihop_graded"] > 0)
        # THE BINDING ANTI-CHEAT (this smoke earned it): a stacked plastic reservoir lifts held-out LABEL-AGNOSTICALLY
        # (permuted-label + wrong-sign teachers lift it as much as credit). So "beats frozen" is NOT credit -- the GO
        # REQUIRES the DIRECTED lift credit>permuted by the same margin (credit routes CORRECT-label error), on top
        # of the classic permuted<=frozen and lesion->floor controls.
        directed = sum(1 for p in per if p["credit_graded_heldout"] >= p["permuted_graded_heldout"] + a.margin_go)
        beats_wrongsign = sum(1 for p in per if p["credit_graded_heldout"] >= p["wrong_sign_graded_heldout"])
        anti_ok = (all(p["no_transport_code"] for p in per) and all(p["reproducibility"] >= 0.8 for p in per)
                   and all(p["credit_graded_heldout"] >= p["permuted_graded_heldout"] + a.margin_go for p in per)
                   and all(p["lesion_graded_heldout"] <= p["frozen_graded_heldout"] + 0.05 for p in per)
                   and agg["oracle_heldout"] >= 0.80 and agg["rate_reservoir_heldout"] <= 0.45)
        go = bool(beats >= need and dcs_pos == n and directed >= need and anti_ok)
        beats_single = bool(agg["dcs_multihop_graded"] > agg["dcs_single_graded"] + 0.02)
        promising = bool((not go) and directed >= need and dcs_pos == n
                         and agg["credit_graded_heldout"] > agg["frozen_graded_heldout"])
        # ---- ORACLE-DIRECTED ISOLATION (the diagnostic this runner adds): per read, per seed, does the CEILING
        #      (oracle) route directed credit above generic plasticity (permuted)? and does the shippable KP arm? ----
        orc_dir_g = sum(1 for p in per if p["directed_oracle_graded"] > 0)
        kp_dir_g = sum(1 for p in per if p["directed_kp_graded"] > 0)
        orc_dir_c = sum(1 for p in per if p["directed_oracle_codon"] > 0)
        # WHERE is the wall? oracle~=permuted (dir<=margin on the graded read) => TASK/READ-REGIME carries no directed
        # signal; oracle clearly>permuted while KP~=permuted => FEEDBACK-alignment wall (transport is the missing piece).
        oracle_directed_positive = bool(agg["directed_oracle_graded"] > a.margin_go)
        kp_directed_positive = bool(agg["directed_kp_graded"] > a.margin_go)
        if not oracle_directed_positive:
            isolation = ("READ-REGIME/TASK -- even the W^T oracle ceiling does NOT beat permuted by the margin "
                         "(directed signal is absent or not needed on this task)")
        elif not kp_directed_positive:
            isolation = ("FEEDBACK -- the oracle ceiling DOES carry directed credit (oracle>permuted) but the "
                         "transport-free KP arm does NOT (KP~=permuted): the feedback alignment is the wall, not the read regime")
        else:
            isolation = ("NONE OF THE WALLS BIND HERE -- both the oracle ceiling AND the transport-free KP arm carry "
                         "directed credit above permuted (transport-free credit works in this regime)")
        agg.update({"n_seeds": n, "credit_beats_frozen_by_margin_graded": beats, "seeds_needed": need,
                    "dcs_positive_graded": dcs_pos, "directed_credit_beats_permuted": directed,
                    "credit_beats_wrongsign": beats_wrongsign, "anti_cheats_clean": bool(anti_ok),
                    "margin_go": a.margin_go, "multihop_beats_single_graded": beats_single, "promising": promising,
                    "oracle_directed_positive_graded": orc_dir_g, "kp_directed_positive_graded": kp_dir_g,
                    "oracle_directed_positive_codon": orc_dir_c, "task_hard": bool(a.task_hard),
                    "isolation_verdict": isolation})
        summary["aggregate"] = agg; summary["GO"] = go; summary["PROMISING"] = promising
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-res {agg['rate_reservoir_heldout']:.3f}. "
                  f"MULTIHOP graded FROZEN {agg['frozen_graded_heldout']:.3f} CREDIT {agg['credit_graded_heldout']:.3f} "
                  f"(dcs {agg['dcs_multihop_graded']:+.3f}) vs SINGLE dcs {agg['dcs_single_graded']:+.3f}. "
                  f"DIRECTED credit-permuted {agg['credit_vs_permuted_graded']:+.3f} (dcs_directed "
                  f"{agg['dcs_directed_graded']:+.3f}); wrong_sign {agg['wrong_sign_graded_heldout']:.3f}, "
                  f"permuted {agg['permuted_graded_heldout']:.3f}, lesion {agg['lesion_graded_heldout']:.3f}.")
        if go:
            summary["verdict"] = (f"MULTIHOP GO ({beats}/{n} beat frozen, DIRECTED {directed}/{n}, dcs>0 {dcs_pos}/{n}) "
                                  f"-- chained transport-free credit routes CORRECT-label error through both "
                                  f"real-spikes layers, beyond label-agnostic plasticity. " + common)
        elif promising:
            summary["verdict"] = (f"MULTIHOP PROMISING (DIRECTED {directed}/{n}, dcs>0 {dcs_pos}/{n}, margin {beats}/{n}) "
                                  f"-- credit > permuted + frozen, {a.margin_go} margin not on all seeds. " + common)
        else:
            summary["verdict"] = (f"MULTIHOP NEGATIVE (frozen-beat {beats}/{n}, DIRECTED credit>permuted {directed}/{n} "
                                  f"need {need}, dcs>0 {dcs_pos}/{n}, anti_ok {anti_ok}) -- the multihop lift over frozen "
                                  f"is LABEL-AGNOSTIC (permuted/wrong-sign lift it as much); NOT directed credit. "
                                  + common)
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-realspikes-multihop] {summary['verdict']}", flush=True)
    if err is None and per and "aggregate" in summary:
        ag = summary["aggregate"]
        task_lbl = "HARD (n_prop=4,k=17)" if ag.get("task_hard") else "EASY (n_prop=3,k=9)"
        print(f"[ISOLATION | {task_lbl}] graded heldout means -- frozen {ag['frozen_graded_heldout']:.3f} | "
              f"permuted {ag['permuted_graded_heldout']:.3f} | KP {ag['credit_graded_heldout']:.3f} | "
              f"ORACLE(W^T) {ag['oracle_directed_graded_heldout']:.3f} | oracle_host {ag['oracle_heldout']:.3f} | "
              f"rate-res {ag['rate_reservoir_heldout']:.3f}", flush=True)
        print(f"[ISOLATION | {task_lbl}] directed credit (arm-permuted) -- oracle-perm {ag['directed_oracle_graded']:+.3f} "
              f"({ag['oracle_directed_positive_graded']}/{ag['n_seeds']}>0) | KP-perm {ag['directed_kp_graded']:+.3f} "
              f"({ag['kp_directed_positive_graded']}/{ag['n_seeds']}>0)", flush=True)
        print(f"[ISOLATION | {task_lbl}] WALL => {ag['isolation_verdict']}", flush=True)
    print(f"[gap4-realspikes-multihop] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
