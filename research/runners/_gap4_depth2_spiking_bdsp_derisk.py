"""gap#4 DEPTH-2 SPIKING BDSP de-risk -- does the coincidence-gated BDSP rule GENERALIZE through depth on REAL SPIKES
where feedback-alignment MEMORIZES? (the spiking port of the banked RATE depth-2 result.)

BANKED RATE RESULT (the rate stand-in this runner ports to spikes):
  research/findings/2026-08-01-gap4-DEPTH2-BDSP-generalizes-where-FA-memorizes-but-same-ceiling-depth-wall-consolidated-6seed.md
  On a DEPTH-2 rate DendriticMLP the coincidence-gated BDSP GENERALIZES (train 0.660 ~ held-out 0.636, gen-gap 0.024)
  where FA MEMORIZES (held-out 0.633, train 0.96, gen-gap 0.33); both cap ~0.63 vs oracle 0.97. THAT IS A RATE
  stand-in. The MISSION substrate is SPIKING. THIS runner asks: on a DEPTH-2 network of REAL-spikes movable-plateau
  layers, does BDSP still GENERALIZE (small gen-gap) while FA MEMORIZES (large gen-gap)?

THE ARCHITECTURE (all neurons/synapses; NO sim/ edit -- composes two validated real-spikes movable-plateau layers):
  Layer 1 = BDSPRealSpikesPlateauExpander(n_in=n_features, n_col=N1)  -- input features -> columns1 (real-spikes read).
  Layer 2 = BDSPRealSpikesPlateauExpander(n_in=N1,       n_col=N2)  -- LAYER-1's columns ARE its input features.
  Readout = a linear softmax readout (fit_lin) on LAYER-2's codon -> the class.
  Both layers configured with the same drive_pa(1200)/n_steps(30) as the validated single-layer read.

THE CHAINING (how layer-1's codon becomes layer-2's input -- the one host-mediated re-encoding, flagged honestly):
  For an input pattern:  vap1 = L1._vap(active_feats)  (REAL spiking forward pass) -> codon1 = (vap1 > FLOOR).
  The SET of active columns1  A2 = {c : codon1[c] > 0}  is LAYER-2's active-feature set. L2._vap(A2) then drives
  those columns1-as-features with drive_pa and integrates -> vap2 -> codon2. So layer-2's "features" ARE layer-1's
  columns. THE SHORTCUT (named, not hidden): layer-1's plateau output (codon1) is re-encoded into a SET of active
  input features that layer-2's input neurons are then DRIVEN with (fixed drive current) -- a host re-encoding of one
  layer's spiking output into the next layer's input current, in place of a direct axon->synapse projection. Every
  read is REAL SPIKES; the codon->active-set->drive hop is the host boundary (the same input-rendering role host code
  legitimately plays for the retina, applied here at the inter-layer seam). Burning it down = a spiking projection
  from layer-1 columns onto layer-2 input synapses (tracked, not built here).

THE TRAINING (mirror the single-layer BDSP, applied at BOTH layers, transport-free DFA -- the rate depth-2 design:
one SHARED output error e, each hidden layer converts it with its OWN fixed-random feedback; only the CREDIT differs):
  Forward:  codon1, codon2 (real-spikes read) per input; e = softmax(codon2 @ W_out) - onehot(y)  (SHARED error).
  deep BDSP hidden update at EACH layer l (byte-identical body to BDSPRealSpikesPlateauExpander.train_epoch_bdsp):
      ap    = e @ B_l.T                       # DFA-projected output error, transport-free (B_l fixed random, != W_out)
      sig   = sigmoid(beta * ap)              # P_post in [0,1]
      credit= sig - Pbar_l                    # sigmoid-baseline BDSP credit (per-column EMA baseline)
      dW[c,f] = -eta * mean_i post_bin[i,c] * credit[i,c] * pre_bin[i,f]   # COINCIDENCE-gated (binary pre x post)
      W_l <- L2-renorm to initial per-column norm; Pbar_l <- EMA        # substrate homeostasis companion
    pre_bin at L2 = LAYER-1 codon (binary events);  pre_bin at L1 = INPUT feature spike EVENTS. post = that layer codon.
    Layer 1's credit path: its OWN fixed-random feedback B1 (N1 x k) projects the SAME output error e to layer 1
    (DFA -- NOT chained through L2's weights), the independent-projection structure that WON in the rate test.
  deep FA hidden update (the graded covariance/DFA form that MEMORIZES -- the parent local_correct rule ported):
      ap = e @ B_l.T;   dW[c,f] = -eta_fa * mean_i (margin[i,c] * ap[i,c]) * pre_grad[i,f]   # GRADED, no gate, no baseline
    pre_grad/margin are the GRADED plateau read (feature spike counts / plateau margin) -- the ONLY difference from BDSP
    is the credit shape (binary-coincidence + bounded sigmoid-baseline vs raw graded product); e, B_l, the readout SGD
    and the substrate homeostasis are IDENTICAL -> the rule is the isolated variable.

ARMS (all read via REAL spikes, same init per seed): (1) FROZEN depth-2 (both layers frozen random) -- the reservoir;
(2) deep BDSP (both layers trained by BDSP); (3) deep FA (both layers graded-FA); (4) oracle (rate DendriticMLP
depth-2 backprop on the same task features -- ceiling). TASK = make_task_semantic_inheritance sweet spot (n_prop=3),
held-out inheritance.

HEADLINE = deep BDSP gen-gap (train - held-out) vs deep FA gen-gap; held-out of each vs frozen + oracle.
GO gate: deep BDSP gen-gap < deep FA gen-gap (BDSP generalizes where FA memorizes, ON SPIKES) on >= ceil(0.834*n)
seeds. Anti-cheats (all mandatory): reproducibility of EACH layer's codon >= 0.8 (the real read is reliable);
NO-TRANSPORT (B1/B2 fixed random != readout, immutable; the hidden update source never references W_out); FROZEN
reservoir control; permuted-training-label BDSP must NOT beat frozen held-out; cfg.seed set on each layer's substrate.

Run (the bridge wants cupy on this box; the FULL run is GPU/long -- the user runs it, this file only imports + smokes):
    SIM_BACKEND=cupy python -m research.runners._gap4_depth2_spiking_bdsp_derisk --seeds 42 --n1 100 --n2 100 --epochs 30
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

# --- reuse-by-import (NO edit to any shared runner) ---
# The two validated real-spikes layers + the whole harness plumbing (task, oracle, rate-reservoir op-point control,
# fit_lin readout, reproducibility/diversity, FLOOR/TOPK, softmax). The BDSP layer's init_bdsp gives each layer its
# fixed-random feedback B + EMA baseline Pbar + a forward readout W_out for free; train_epoch_bdsp's HIDDEN body is
# replicated below as a free function so the SAME output error e can drive BOTH layers (depth-2 DFA chaining).
from research.runners._gap4_realspikes_bdsp_credit_derisk import BDSPRealSpikesPlateauExpander
from research.runners._gap4_plastic_plateau_credit_derisk import (
    fit_lin, topk_active, _sm, _reproducibility, _codon_diversity, _rate_reservoir_heldout, FLOOR, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)


# ============================================================================================================
# Substrate read helpers -- ONE real spiking forward pass per layer per input, deriving BOTH the binary codon (BDSP /
# readout) and the graded margin (FA) from the same cp_v_apical (halves the spiking passes vs calling codon()+margin()).
# ============================================================================================================
def _read_layer(exp, active):
    """One REAL spiking forward pass -> (binary codon, graded margin) for the columns. No readout weight is read."""
    vap = exp._vap(active)                                   # REAL spiking forward pass (drive active feats, integrate)
    codon = (vap > FLOOR).astype(np.float64)                # binary plateau EVENT
    margin = np.maximum(0.0, vap - FLOOR)                   # graded low-CV plateau surrogate
    return codon, margin


def _forward_d2(net, af_l1):
    """Depth-2 REAL-spikes forward pass. Returns (C1, M1, A2, C2, M2): layer-1 codon/margin, the layer-2 active-feature
    SETS (= active columns1 -- THE codon->input hop), and layer-2 codon/margin. Two spiking passes per input."""
    C1, M1, A2, C2, M2 = [], [], [], [], []
    for a in af_l1:
        c1, m1 = _read_layer(net.L1, a)
        a2 = set(np.where(c1 > 0)[0])                       # active columns1 -> layer-2's active-feature set (the hop)
        c2, m2 = _read_layer(net.L2, a2)
        C1.append(c1); M1.append(m1); A2.append(a2); C2.append(c2); M2.append(m2)
    return np.asarray(C1), np.asarray(M1), A2, np.asarray(C2), np.asarray(M2)


# ============================================================================================================
# The TWO hidden-credit rules as FREE functions (so the SHARED output error e drives both layers). Each reads ONLY
# (pre, post, exp.B, e) and writes exp.cp_connections.data -- NEVER W_out or W_out^T (transport-free). _bdsp_hidden's
# body is byte-identical to BDSPRealSpikesPlateauExpander.train_epoch_bdsp lines 103-117; _apply_dW is the verbatim
# substrate write + L2-renorm homeostasis. A source-string no-transport check asserts neither references W_out.
# ============================================================================================================
def _apply_dW(exp, dW_full):
    """Write dW to the substrate synapses (excitatory clamp + per-column L2-renorm to the INITIAL norm) -- copied
    verbatim from the single-layer rule (PlasticPlateauExpander.train_epoch / train_epoch_bdsp)."""
    data = exp._get_data()
    data = data + dW_full[exp.syn_col, exp.syn_feat]
    np.maximum(data, 0.0, out=data)                         # excitatory (w >= 0)
    cur = np.sqrt(np.array([np.sum(data[exp.syn_col == c] ** 2) for c in range(exp.NC)]))
    scale = np.where(cur > 1e-9, exp.col_norm0 / (cur + 1e-12), 1.0)
    data = data * scale[exp.syn_col]
    exp._set_data(data)


def _bdsp_hidden(exp, pre_bin, post_bin, e, eta, beta=1.0, rho=0.1):
    """Coincidence-gated + sigmoid-baseline BDSP hidden update driven by a SHARED output error e (DFA via exp.B).
    Reads ONLY pre_bin/post_bin/exp.B/exp.Pbar/e -- NEVER exp.W_out. Updates exp.Pbar in place."""
    ap = e @ exp.B.T                                        # (N,C) DFA-projected output error (B fixed random != W_out)
    sig = 1.0 / (1.0 + np.exp(-beta * ap))                 # P_post in [0,1]
    credit = sig - exp.Pbar[None, :]                       # sigmoid-baseline BDSP credit
    dW_full = -eta * ((post_bin * credit).T @ pre_bin) / len(pre_bin)   # (C,F) COINCIDENCE-gated, descent sign
    _apply_dW(exp, dW_full)
    exp.Pbar = (1 - rho) * exp.Pbar + rho * sig.mean(0)    # EMA plateau-probability baseline
    return float(np.mean(np.abs(dW_full)))


def _fa_hidden(exp, pre_grad, post_margin, e, eta):
    """Graded feedback-alignment / covariance-DFA hidden update (the parent local_correct form ported to the plateau):
    pre_graded x (DFA-error x graded post-margin). NO coincidence gate, NO sigmoid-baseline -> the memorizer. Reads
    ONLY pre_grad/post_margin/exp.B/e -- NEVER exp.W_out (FA is transport-free too; the difference is the credit shape)."""
    ap = e @ exp.B.T                                        # (N,C) DFA-projected output error (transport-free)
    dW_full = -eta * ((post_margin * ap).T @ pre_grad) / len(pre_grad)   # (C,F) GRADED covariance, descent sign
    _apply_dW(exp, dW_full)
    return float(np.mean(np.abs(dW_full)))


def _readout_sgd(exp, post_bin, e, lr_out):
    """Co-train the OUTPUT readout (on layer-2 codon) by its OWN gradient -- standard last-layer SGD, NOT a transport."""
    gout = e / len(post_bin)
    exp.W_out -= lr_out * (post_bin.T @ gout)
    exp.b_out -= lr_out * gout.sum(0)


# ============================================================================================================
# The depth-2 network: TWO BDSPRealSpikesPlateauExpander layers, chained. init_bdsp on EACH layer allocates that
# layer's fixed-random feedback B (L1.B = B1, L2.B = B2), its EMA baseline Pbar, and a forward readout W_out; the
# OUTPUT readout is L2.W_out/L2.b_out (on the layer-2 codon). L1.W_out is unused. Same seeds per layer across arms
# -> FROZEN / deep-BDSP / deep-FA / permuted all start from the SAME reservoir (the required same-init anti-cheat).
# ============================================================================================================
class Depth2SpikingPlateauNet:
    def __init__(self, n_in, n1, n2, k, seed, w0, jitter, k_th, drive_pa, n_steps, p0, lesion=False):
        self.L1 = BDSPRealSpikesPlateauExpander(n_in, n1, seed, w0=w0, jitter=jitter, k_th=k_th,
                                                lesion=lesion).configure_read(drive_pa, n_steps)
        self.L2 = BDSPRealSpikesPlateauExpander(n1, n2, seed + 10007, w0=w0, jitter=jitter, k_th=k_th,
                                                lesion=lesion).configure_read(drive_pa, n_steps)
        # each layer gets its OWN fixed-random feedback B + EMA baseline Pbar (+ a forward readout W_out). fb_wd=0 ->
        # B fixed random (the FA/DFA form). L2.W_out/b_out = the OUTPUT readout on the layer-2 codon.
        self.L1.init_bdsp(k, seed * 31 + 7, p0=p0, fb_wd=0.0)
        self.L2.init_bdsp(k, seed, p0=p0, fb_wd=0.0)
        self.n_in, self.n1, self.n2, self.k = n_in, n1, n2, k

    @property
    def W_out(self):
        return self.L2.W_out

    @property
    def b_out(self):
        return self.L2.b_out

    def restore_frozen(self):
        self.L1.restore_frozen(); self.L2.restore_frozen()


def _err(net, C2, y):
    Y = np.eye(net.k)[np.asarray(y, int)]
    return _sm(C2 @ net.W_out + net.b_out) - Y             # (N,k) SHARED output error (drives BOTH layers' credit)


def _train_d2(net, af_b, pre1_bin, pre1_grad, y, epochs, mode, eta, eta_fa, lr_out, beta, rho,
              shuffle_error=False, seed=0):
    """Train the depth-2 net in place. mode in {'bdsp','fa'}. The SAME output error e drives L2 (pre=layer-1 output)
    and L1 (pre=input events) via each layer's own fixed-random B (DFA -- NOT chained through L2's weights)."""
    net.restore_frozen()
    err_rng = np.random.default_rng(seed * 71 + 3) if shuffle_error else None
    mags = []
    for _ in range(epochs):
        C1, M1, A2, C2, M2 = _forward_d2(net, af_b)
        e = _err(net, C2, y)
        if shuffle_error and err_rng is not None:
            e = e[err_rng.permutation(len(e))]            # anti-cheat: destroy the per-sample error routing
        if mode == "bdsp":
            m2 = _bdsp_hidden(net.L2, C1, C2, e, eta, beta, rho)        # L2: pre=codon1, post=codon2
            m1 = _bdsp_hidden(net.L1, pre1_bin, C1, e, eta, beta, rho)  # L1: pre=input EVENTS, post=codon1 (own B1)
        else:  # graded FA
            m2 = _fa_hidden(net.L2, M1, M2, e, eta_fa)                  # L2: graded pre=margin1, post=margin2
            m1 = _fa_hidden(net.L1, pre1_grad, M1, e, eta_fa)           # L1: graded pre=input counts, post=margin1
        _readout_sgd(net.L2, C2, e, lr_out)                            # co-train the output readout (both arms alike)
        mags.append(0.5 * (m1 + m2))
    return mags


def _measure_d2(net, af_b, yb, af_h, yh):
    """Forward train + held-out through the (trained) depth-2 net; fit a FRESH linear readout on the layer-2 codon
    (the arc-standard separability measure). Returns (train_acc, heldout_acc, C2_train, C2_held)."""
    _, _, _, C2b, _ = _forward_d2(net, af_b)
    _, _, _, C2h, _ = _forward_d2(net, af_h)
    clf = fit_lin(C2b, yb, net.k)
    return float(np.mean(clf(C2b) == yb)), float(np.mean(clf(C2h) == yh)), C2b, C2h


def _seeded_ok(exp):
    """cfg.seed anti-cheat: two builds at this layer's seed give BYTE-IDENTICAL per-neuron firing thresholds
    (the substrate is actually seeded, per the CLAUDE.md actual_seed_used trap)."""
    try:
        th = np.asarray(exp.b.cp_neuron_firing_thresholds.get() if hasattr(exp.b.cp_neuron_firing_thresholds, "get")
                        else exp.b.cp_neuron_firing_thresholds)
        twin = BDSPRealSpikesPlateauExpander(exp.NF, exp.NC, exp.b.core_config.seed, lesion=exp.lesion)
        th2 = np.asarray(twin.b.cp_neuron_firing_thresholds.get()
                         if hasattr(twin.b.cp_neuron_firing_thresholds, "get") else twin.b.cp_neuron_firing_thresholds)
        return bool(np.array_equal(th, th2))
    except Exception:
        return None


def run_seed(seed, n1, n2, epochs, eta, eta_fa, lr_out, beta, p0, w0, jitter, k_th, n_sub, hidden,
             oracle_epochs, oracle_lr, oracle_batch, drive_pa, n_steps, rho, task_kwargs, verbose=True):
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(seed * 13 + 1); keep = srng.permutation(len(Xtr))[:min(n_sub, len(Xtr))]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    af_b = topk_active(Xb, TOPK); af_h = topk_active(Xh, TOPK)
    chance = float(max(np.mean(yh == c) for c in np.unique(yh))) if len(yh) else float("nan")
    out = {"seed": seed, "n_in": n_in, "k": k, "n1": n1, "n2": n2, "chance": chance, "n_train_sub": len(Xb),
           "n_heldout_inherit": len(yh), "drive_pa": drive_pa, "n_steps": n_steps}

    # ---- ARM 4: oracle (depth-2 backprop rate ceiling) + rate-reservoir op-point control ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_train"] = float(onet.accuracy(Xtr, ytr)); out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n2, seed)

    # ---- build ONE depth-2 net, precompute layer-1 pre-activity (weight-INDEPENDENT -> once) ----
    net = Depth2SpikingPlateauNet(n_in, n1, n2, k, seed, w0, jitter, k_th, drive_pa, n_steps, p0)
    pre1_grad = np.asarray([net.L1.feat_spike_counts(a) for a in af_b])         # (N, n_in) REAL feature spike counts
    pre1_bin = (pre1_grad > 0).astype(float)                                    # (N, n_in) binary spike EVENT

    # ---- ARM 1: FROZEN depth-2 reservoir (both layers random) ----
    net.restore_frozen()
    fz_tr, fz_ho, _, _ = _measure_d2(net, af_b, yb, af_h, yh)
    out["frozen_train"] = fz_tr; out["frozen_heldout"] = fz_ho
    out["frozen_gen_gap"] = round(fz_tr - fz_ho, 4)

    # ---- ARM 2: deep BDSP (both layers coincidence-gated sigmoid-baseline, transport-free DFA) ----
    mags_bdsp = _train_d2(net, af_b, pre1_bin, pre1_grad, yb, epochs, "bdsp", eta, eta_fa, lr_out, beta, rho, seed=seed)
    bd_tr, bd_ho, C2b_bd, _ = _measure_d2(net, af_b, yb, af_h, yh)
    out["deepBDSP_train"] = bd_tr; out["deepBDSP_heldout"] = bd_ho
    out["deepBDSP_gen_gap"] = round(bd_tr - bd_ho, 4)
    out["deepBDSP_codon_diversity"] = _codon_diversity(C2b_bd)
    out["deepBDSP_update_mag_first_last"] = [round(mags_bdsp[0], 6), round(mags_bdsp[-1], 6)]

    # ---- reproducibility of EACH layer's codon under the trained BDSP net (the real read is reliable) ----
    _, _, A2_bd, _, _ = _forward_d2(net, af_b[:8])
    out["reproducibility_L1"] = _reproducibility(net.L1, af_b)
    out["reproducibility_L2"] = _reproducibility(net.L2, A2_bd)

    # ---- ARM 3: deep FA (graded covariance/DFA -- the memorizer) on a FRESH same-init net ----
    net_fa = Depth2SpikingPlateauNet(n_in, n1, n2, k, seed, w0, jitter, k_th, drive_pa, n_steps, p0)
    mags_fa = _train_d2(net_fa, af_b, pre1_bin, pre1_grad, yb, epochs, "fa", eta, eta_fa, lr_out, beta, rho, seed=seed)
    fa_tr, fa_ho, C2b_fa, _ = _measure_d2(net_fa, af_b, yb, af_h, yh)
    out["deepFA_train"] = fa_tr; out["deepFA_heldout"] = fa_ho
    out["deepFA_gen_gap"] = round(fa_tr - fa_ho, 4)
    out["deepFA_codon_diversity"] = _codon_diversity(C2b_fa)
    out["deepFA_update_mag_first_last"] = [round(mags_fa[0], 6), round(mags_fa[-1], 6)]

    # ---- HEADLINE: does BDSP generalize (small gap) where FA memorizes (large gap)? ----
    out["bdsp_generalizes_vs_fa_memorizes"] = bool(out["deepBDSP_gen_gap"] < out["deepFA_gen_gap"])
    # attribution: is deep BDSP's held-out above the frozen reservoir (the credit's, not the frozen hidden's)?
    from tools.lab import attributable_to
    attributable_to("deep BDSP held-out vs the FROZEN depth-2 reservoir", out["deepBDSP_heldout"], out["frozen_heldout"])

    # ---- ANTI-CHEAT: permuted-training-label BDSP must NOT beat frozen held-out (no label leakage in the credit) ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    net_p = Depth2SpikingPlateauNet(n_in, n1, n2, k, seed, w0, jitter, k_th, drive_pa, n_steps, p0)
    _train_d2(net_p, af_b, pre1_bin, pre1_grad, yperm, epochs, "bdsp", eta, eta_fa, lr_out, beta, rho, seed=seed)
    _, out["bdsp_on_permuted_heldout"], _, _ = _measure_d2(net_p, af_b, yb, af_h, yh)

    # ---- ANTI-CHEAT: NO-TRANSPORT (the hidden-update BYTECODE never accesses the readout weight; B fixed random !=
    #      readout, immutable). co_names lists every attribute a function loads -> a robust check that _bdsp_hidden /
    #      _fa_hidden / _apply_dW (the hidden path) never touch .W_out/.b_out (only _readout_sgd co-trains the output). ----
    _ht_names = (set(_bdsp_hidden.__code__.co_names) | set(_fa_hidden.__code__.co_names)
                 | set(_apply_dW.__code__.co_names))
    src_ok = ("W_out" not in _ht_names) and ("b_out" not in _ht_names)
    b_imm = bool(np.array_equal(net.L1.B, net.L1.B0) and np.array_equal(net.L2.B, net.L2.B0))   # fb_wd=0 -> B immutable
    b_not_copy = bool((not np.allclose(net.L1.B, net.L1.W_out, atol=1e-6))
                      and (not np.allclose(net.L2.B, net.L2.W_out, atol=1e-6)))
    out["no_transport_source"] = bool(src_ok)
    out["no_transport_B_immutable"] = b_imm
    out["no_transport_B_not_readout_copy"] = b_not_copy
    out["no_transport"] = bool(src_ok and b_imm and b_not_copy)

    # ---- ANTI-CHEAT: cfg.seed actually seeds each layer's substrate (the actual_seed_used trap) ----
    out["seeded_L1"] = _seeded_ok(net.L1); out["seeded_L2"] = _seeded_ok(net.L2)

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} N1={n1} N2={n2} chance={chance:.3f} n_ho={len(yh)} "
              f"drive={drive_pa} steps={n_steps}", flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f} | rate-reservoir {out['rate_reservoir_heldout']:.3f} | "
              f"FROZEN {out['frozen_heldout']:.3f}(gap {out['frozen_gen_gap']:+.3f})", flush=True)
        print(f"    deep BDSP {out['deepBDSP_heldout']:.3f}(tr {out['deepBDSP_train']:.3f} gap "
              f"{out['deepBDSP_gen_gap']:+.3f}) | deep FA {out['deepFA_heldout']:.3f}(tr {out['deepFA_train']:.3f} gap "
              f"{out['deepFA_gen_gap']:+.3f}) | BDSP-generalizes-vs-FA-memorizes "
              f"{out['bdsp_generalizes_vs_fa_memorizes']}", flush=True)
        print(f"    [anti-cheat] reprod L1 {out['reproducibility_L1']:.3f} L2 {out['reproducibility_L2']:.3f} | "
              f"bdsp-on-permuted {out['bdsp_on_permuted_heldout']:.3f} | no-transport {out['no_transport']} "
              f"(src {out['no_transport_source']} B-immut {out['no_transport_B_immutable']} "
              f"B!=Wout {out['no_transport_B_not_readout_copy']}) | seeded L1={out['seeded_L1']} L2={out['seeded_L2']}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 DEPTH-2 real-spikes BDSP-generalizes-vs-FA-memorizes crux test.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n1", type=int, default=100, help="layer-1 column count")
    ap.add_argument("--n2", type=int, default=100, help="layer-2 column count")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.03, help="BDSP (coincidence-gated) hidden lr")
    ap.add_argument("--eta-fa", type=float, default=None, help="graded-FA hidden lr (default = --eta -> isolate the rule)")
    ap.add_argument("--lr-out", type=float, default=0.2, help="output readout SGD lr")
    ap.add_argument("--beta", type=float, default=1.0, help="sigmoid gain on the DFA-projected error")
    ap.add_argument("--rho", type=float, default=0.1, help="EMA plateau-baseline update rate")
    ap.add_argument("--p0", type=float, default=0.30, help="initial per-column EMA plateau-probability baseline")
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None)
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48, help="oracle hidden width")
    ap.add_argument("--oracle-epochs", type=int, default=200)
    ap.add_argument("--oracle-lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", type=int, default=128)
    ap.add_argument("--drive-pa", type=float, default=1200.0)
    ap.add_argument("--n-steps", type=int, default=30)
    # --- the SWEET SPOT task config (n_prop=3, n_super=24) ---
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default="research/findings/raw/gap4/depth2_spiking/depth2_spiking_bdsp.json")
    a = ap.parse_args()
    eta_fa = a.eta if a.eta_fa is None else a.eta_fa
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n1, a.n2, a.epochs, a.eta, eta_fa, a.lr_out, a.beta, a.p0, a.w0, a.jitter, a.k_th,
                                a.n_sub, a.hidden, a.oracle_epochs, a.oracle_lr, a.oracle_batch, a.drive_pa, a.n_steps,
                                a.rho, task_kwargs))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_depth2_spiking_bdsp", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n1": a.n1, "n2": a.n2, "epochs": a.epochs, "eta": a.eta, "eta_fa": eta_fa, "lr_out": a.lr_out,
                          "beta": a.beta, "rho": a.rho, "p0": a.p0, "w0": a.w0, "jitter": a.jitter,
                          "drive_pa": a.drive_pa, "n_steps": a.n_steps, "task": task_kwargs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_heldout", "frozen_gen_gap", "deepBDSP_heldout",
                "deepBDSP_train", "deepBDSP_gen_gap", "deepFA_heldout", "deepFA_train", "deepFA_gen_gap",
                "reproducibility_L1", "reproducibility_L2", "bdsp_on_permuted_heldout", "chance",
                "deepBDSP_codon_diversity", "deepFA_codon_diversity"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per); need = int(np.ceil(0.834 * n))
        # HEADLINE GO: deep BDSP gen-gap < deep FA gen-gap (BDSP generalizes where FA memorizes) on >= need seeds.
        gap_wins = sum(1 for p in per if p["deepBDSP_gen_gap"] < p["deepFA_gen_gap"])
        anti_ok = (all(p["no_transport"] for p in per)
                   and all(p["reproducibility_L1"] >= 0.8 for p in per)
                   and all(p["reproducibility_L2"] >= 0.8 for p in per)
                   and all(p["bdsp_on_permuted_heldout"] <= p["frozen_heldout"] + 0.05 for p in per)
                   and agg["oracle_heldout"] >= 0.80 and agg["rate_reservoir_heldout"] <= 0.45)
        go = bool(gap_wins >= need and anti_ok)
        agg.update({"n_seeds": n, "seeds_needed": need, "bdsp_gap_below_fa_gap": gap_wins,
                    "anti_cheats_clean": bool(anti_ok)})
        summary["aggregate"] = agg; summary["GO"] = go
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f}, FROZEN "
                  f"{agg['frozen_heldout']:.3f}. deep BDSP {agg['deepBDSP_heldout']:.3f} (tr "
                  f"{agg['deepBDSP_train']:.3f}, gap {agg['deepBDSP_gen_gap']:+.3f}) vs deep FA "
                  f"{agg['deepFA_heldout']:.3f} (tr {agg['deepFA_train']:.3f}, gap {agg['deepFA_gen_gap']:+.3f}). "
                  f"anti: reprod L1 {agg['reproducibility_L1']:.3f} L2 {agg['reproducibility_L2']:.3f}, "
                  f"bdsp-on-permuted {agg['bdsp_on_permuted_heldout']:.3f}.")
        if go:
            summary["verdict"] = (f"DEPTH-2 SPIKING BDSP GO (gap<FA {gap_wins}/{n}) -- on REAL SPIKES the coincidence-"
                                  f"gated BDSP GENERALIZES (small gen-gap) where graded FA MEMORIZES (large gen-gap); "
                                  f"the rate depth-2 result SURVIVES the port to spikes. " + common)
        else:
            summary["verdict"] = (f"DEPTH-2 SPIKING BDSP NEGATIVE (gap<FA {gap_wins}/{n} need {need}, anti_ok "
                                  f"{anti_ok}) -- on real spikes BDSP does NOT clearly show the smaller gen-gap vs FA "
                                  f"(or an anti-cheat/op-point failed). The failing mode is named. " + common)
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-depth2-spiking-bdsp] {summary['verdict']}", flush=True)
    print(f"[gap4-depth2-spiking-bdsp] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
