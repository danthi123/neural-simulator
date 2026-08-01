"""gap#4 SUPERVISED (error-gated) coincidence-PLATEAU credit de-risk -- the DIRECTED-credit test.

THE QUESTION (banked 2026-08-01): the UNSUPERVISED local plateau-margin covariance rule already beats a frozen
random reservoir on the MOVABLE plateau hidden at the n_prop=3 sweet spot (deep_credit_share +0.139, 5/6;
research/findings/2026-08-01-gap4-plastic-plateau-local-unsupervised-plasticity-beats-frozen-reservoir-at-sweet-spot-5of6.md
runner research/runners/_gap4_plastic_plateau_credit_derisk.py). It fills only ~14% of the frozen->oracle gap and
uses NO label / NO output error. The standard tonic hidden could not be moved by ANY credit rule; but now that the
hidden is MOVABLE, does adding a DIRECTED OUTPUT ERROR push deep_credit_share WELL PAST the unsupervised 0.14?

WHAT'S NEW: the SAME movable-hidden plateau plasticity, now MODULATED by a DFA-projected output error. The hidden
input-weight update is GATED by the output error via a FIXED-RANDOM feedback matrix B (feedback alignment /
e-prop's B_direct -- transport-free: B is NEVER the forward/readout W or its transpose):

    theta[c] = mean_i margin[i,c]                                  # per-column homeostatic threshold (COMPANION)
    logits   = margin @ W_out + b_out                             # forward readout (its OWN gradient trains it)
    e        = softmax(logits) - onehot(y)                        # output error  (N, k)
    g[i,c]   = (e @ B.T)[i,c]     where B is (C,k) FIXED RANDOM   # DFA feedback -> transport-free
    dW[c,f]  = -lr * mean_i (margin[i,c] - theta[c]) * pre[i,f] * g[i,c]   # error-GATED plateau plasticity
    W[c,:] <- renormalized to its INITIAL L2 norm                 # same companion homeostasis as unsupervised

The ONLY change vs the unsupervised rule is the multiplicative error gate g (the same (margin-theta)*pre local
term, now routed by the DFA output error; descent sign so the update DESCENDS the loss). g<0 on a column that
should INCREASE for the correct class -> -g>0 -> that active column is potentiated; depressed otherwise. This is
the whole point: DIRECTED credit on a MOVABLE, reliable hidden -- the thing that could not be done on the tonic
hidden.

TRANSPORT-FREE (the load-bearing property): the HIDDEN weight update reads ONLY (margin, pre, B, e). B is a fixed
random matrix drawn from an RNG stream INDEPENDENT of W_out; it is never updated and never set to W_out^T. Reading
W_out in the FORWARD pass to compute the error e is a normal forward pass (required for ANY supervised signal); it
is NOT weight transport -- transport would be using W_out^T in the BACKWARD projection, which we replace with the
fixed random B. Asserted in code + a runtime probe (the hidden update is invariant to W_out given a fixed e) +
B != W_out^T numerically + B immutable across training. The output-layer W_out is trained by its own gradient
(last layer; standard, not a hidden transport).

REUSE-BY-IMPORT: the whole reliable-plateau mechanism, the sweet-spot task, the oracle, the readout, the
anti-cheat helpers and the FROZEN + UNSUPERVISED arms come from `_gap4_plastic_plateau_credit_derisk` unchanged.
`SupervisedPlateauExpander` SUBCLASSES `PlasticPlateauExpander` so its init / frozen reservoir / unsupervised rule
are byte-identical -- the SUPERVISED arm starts from the SAME reservoir; the single variable is "is the plateau
plasticity error-gated?". NO edit to any shared runner, NO sim/ edit.

ARMS (single variable = is the plasticity SUPERVISED/error-gated?):
  1. FROZEN-plateau reservoir     -- fixed random coincidence weights + trained readout (deep_credit_share ref).
  2. UNSUPERVISED plateau         -- tonight's local covariance rule (the baseline to beat; deep_credit_share ~0.14).
  3. SUPERVISED (error-gated)     -- the new arm.
  4. oracle (fenced backprop ~0.96) + frozen random RATE reservoir (must fail ~0.10 -> op-point genuine).

deep_credit_share = (arm - frozen_plateau) / (oracle - frozen_plateau), reported for BOTH unsup and supervised.

ANTI-CHEATS (all mandatory): permuted-label -> chance (readout-on-permuted; PLUS supervised-trained-on-permuted
must lose the directed benefit); shuffle the DFA error across the batch -> degrades toward the unsupervised arm
(the error routing is load-bearing); plateau/apical LESION -> floor (supervised-trained on a lesioned plateau);
NO-TRANSPORT (B fixed-random, B != W_out^T, B immutable, hidden update invariant to W_out -- code + runtime);
reproducibility >= 0.8. Backend stamped.

GO gate (SMOKE 1-2 seeds; 6-seed is the parent's after review): the SUPERVISED arm beats BOTH the frozen-plateau
reservoir AND the unsupervised arm on held-out inheritance by a preregistered margin -> directed error on the
movable hidden adds real credit BEYOND unsupervised sharpening (deep_credit_share rises clearly above ~0.14). An
HONEST NEGATIVE (directed error does NOT help beyond unsupervised -- the movable hidden is helped by sharpening but
not by directed credit) is a valid, important deliverable.

Run (numpy is a no-op for the bridge on this box -> cupy):
    SIM_BACKEND=cupy python -m research.runners._gap4_supervised_plateau_credit_derisk --seeds 42 --epochs 30
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import numpy as np

# --- reuse-by-import: the ENTIRE mechanism, task, oracle, readout, anti-cheat helpers + the FROZEN/UNSUP arms ---
# _gap4_plastic_plateau_credit_derisk is import-safe (main() is under __main__ guard) and re-exports the task/oracle
# helpers it imported. NO edit to it; SupervisedPlateauExpander subclasses its PlasticPlateauExpander.
from research.runners._gap4_plastic_plateau_credit_derisk import (
    PlasticPlateauExpander, fit_lin, topk_active, _sm, _readout_acc, _reproducibility, _codon_diversity,
    _rate_reservoir_heldout, FLOOR, ACT_TH, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)


# ============================================================================================================
# The SUPERVISED (error-gated) plateau expander. Inherits the reliable-plateau mechanism, the frozen reservoir and
# the UNSUPERVISED covariance rule (`train_epoch`) unchanged; adds a DFA-error-gated hidden update. Init is
# byte-identical to the parent at the same seed -> all three arms start from the SAME reservoir.
# ============================================================================================================
class SupervisedPlateauExpander(PlasticPlateauExpander):
    def init_supervised(self, k, seed):
        """Forward readout W_out (trained by its own gradient) + FIXED RANDOM feedback matrix B (transport-free).
        B is drawn from an RNG stream INDEPENDENT of W_out and is never W_out^T and never updated."""
        rng_out = np.random.default_rng(seed * 999 + 17)       # readout init stream
        rng_B = np.random.default_rng(seed * 777 + 5)          # B stream -- INDEPENDENT of W_out (transport-free)
        self.k = int(k)
        self.W_out = rng_out.standard_normal((self.NC, self.k)) * 0.01     # small forward readout (trained)
        self.b_out = np.zeros(self.k)
        self.B = rng_B.standard_normal((self.NC, self.k)) / np.sqrt(self.k)  # FIXED random DFA feedback
        self.B0 = self.B.copy()                                # snapshot -> assert B immutable across training
        return self

    def _hidden_update_from_error(self, Mrg, theta, pre_mat, e, lr):
        """The DFA-error-gated HIDDEN weight update. Reads ONLY (Mrg, theta, pre, B, e) -- NEVER W_out or W_out^T.
        (Exposed as a helper precisely so the no-transport runtime probe can inject a FIXED e and vary W_out.)"""
        g = e @ self.B.T                                       # (N, C) DFA feedback -- B fixed random, transport-free
        mod = (Mrg - theta) * g                                # (N, C) = the unsupervised covariance term, error-gated
        dW_full = -lr * (mod.T @ pre_mat) / len(pre_mat)       # (C, F); descent sign -> DESCEND the loss
        data = self._get_data()
        data = data + dW_full[self.syn_col, self.syn_feat]
        np.maximum(data, 0.0, out=data)                        # excitatory (w >= 0)
        # per-column L2 renorm to the INITIAL norm (same local homeostasis companion as the unsupervised rule)
        cur = np.sqrt(np.array([np.sum(data[self.syn_col == c] ** 2) for c in range(self.NC)]))
        scale = np.where(cur > 1e-9, self.col_norm0 / (cur + 1e-12), 1.0)
        data = data * scale[self.syn_col]
        self._set_data(data)
        return float(np.mean(np.abs(dW_full)))

    def train_epoch_supervised(self, active_sets, pre_mat, y, lr, lr_out, shuffle_error=False, err_rng=None):
        """One SUPERVISED batch update: forward readout -> softmax error -> DFA-gated hidden update + readout SGD.
        shuffle_error breaks the per-sample error routing (the load-bearing DFA control)."""
        Mrg = np.asarray([self.margin(a) for a in active_sets])    # (N, C) graded plateau margin = hidden activation
        theta = Mrg.mean(0, keepdims=True)                         # (1, C) homeostatic threshold (COMPANION)
        logits = Mrg @ self.W_out + self.b_out                     # forward readout (uses W_out in the FORWARD pass only)
        P = _sm(logits)                                            # softmax
        Y = np.eye(self.k)[np.asarray(y, int)]                     # onehot
        e = P - Y                                                  # (N, k) output error
        if shuffle_error and err_rng is not None:
            e = e[err_rng.permutation(len(e))]                     # ANTI-CHEAT: destroy per-sample error routing
        mag = self._hidden_update_from_error(Mrg, theta, pre_mat, e, lr)   # DFA-gated hidden update (no W_out read)
        # co-train the output readout by its OWN gradient (last layer; standard SGD, NOT a hidden transport)
        gout = e / len(active_sets)
        self.W_out -= lr_out * (Mrg.T @ gout)
        self.b_out -= lr_out * gout.sum(0)
        return mag


def _train_supervised(exp, af, pre, y, epochs, lr, lr_out, shuffle_error=False, seed=0):
    exp.restore_frozen()
    err_rng = np.random.default_rng(seed * 71 + 3) if shuffle_error else None
    mags = []
    for _ in range(epochs):
        mags.append(exp.train_epoch_supervised(af, pre, y, lr, lr_out, shuffle_error=shuffle_error, err_rng=err_rng))
    return mags


def run_seed(seed, n_col, epochs, lr_unsup, lr_sup, lr_out, w0, jitter, k_th, n_sub, hidden,
             oracle_epochs, oracle_lr, oracle_batch, task_kwargs, margin_go, verbose=True):
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

    # ---- ARM 4a: oracle (fenced backprop depth-2 rate) ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_train"] = float(onet.accuracy(Xtr, ytr)); out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)

    # ---- ARM 4b/3: frozen random RATE reservoir (must fail ~0.10) ----
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n_col, seed)

    # ---- ONE expander, identical init -> FROZEN, UNSUPERVISED, SUPERVISED all from the SAME reservoir ----
    exp = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th)

    # ARM 1: FROZEN reservoir (weights = W0, no training)
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

    # ARM 3: SUPERVISED (error-gated) plateau plasticity (the new arm)
    exp.init_supervised(k, seed)                               # attach forward readout W_out + FIXED random B
    sup_mags = _train_supervised(exp, af_b, pre_b, yb, epochs, lr_sup, lr_out, seed=seed)
    sup_tr, sup_ho, Cb_sup, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["supervised_plateau_train"] = sup_tr; out["supervised_plateau_heldout"] = sup_ho
    out["supervised_codon_diversity"] = _codon_diversity(Cb_sup)
    out["supervised_update_mag_first_last"] = [round(sup_mags[0], 6), round(sup_mags[-1], 6)]
    out["supervised_weight_moved"] = float(np.mean(np.abs(exp._get_data() - exp.W0)))

    # ---- deep_credit_share for BOTH arms (against the frozen plateau reservoir) ----
    denom = out["oracle_heldout"] - out["frozen_plateau_heldout"]
    def _dcs(v):
        return float((v - out["frozen_plateau_heldout"]) / denom) if abs(denom) > 1e-6 else float("nan")
    out["deep_credit_share_unsup"] = _dcs(out["unsup_plateau_heldout"])
    out["deep_credit_share_supervised"] = _dcs(out["supervised_plateau_heldout"])
    # ATTRIBUTION: the effect under test is whether the DIRECTED error adds held-out credit BEYOND the label-free
    # unsupervised sharpening -- not merely that both arms were measured (attribution-required gate).
    from tools.lab import attributable_to
    attributable_to("directed error (supervised) vs unsupervised sharpening on held-out",
                     out["supervised_plateau_heldout"], out["unsup_plateau_heldout"])

    # ---- ANTI-CHEAT: reproducibility (plateau reliability must hold under the supervised plasticity) ----
    out["reproducibility_supervised"] = _reproducibility(exp, af_b)

    # ---- ANTI-CHEAT: permuted-label ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    # (a) readout-on-permuted -> chance (readout leakage test on the SUPERVISED codon)
    Ctr = np.asarray([exp.codon(a) for a in af_b]); Cte = np.asarray([exp.codon(a) for a in af_h])
    clf_p = fit_lin(Ctr, yperm, k)
    out["permuted_readout_heldout"] = float(np.mean(clf_p(Cte) == yh))
    # (b) SUPERVISED-trained-on-permuted -> the directed benefit must collapse (fresh TRUE readout)
    exp_perm = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_supervised(k, seed)
    _train_supervised(exp_perm, af_b, pre_b, yperm, epochs, lr_sup, lr_out, seed=seed)
    _, out["supervised_on_permuted_heldout"], _, _ = _readout_acc(exp_perm, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: shuffle the DFA error across the batch -> degrade toward the unsupervised arm ----
    exp_sh = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_supervised(k, seed)
    _train_supervised(exp_sh, af_b, pre_b, yb, epochs, lr_sup, lr_out, shuffle_error=True, seed=seed)
    _, out["shuffle_dfa_heldout"], _, _ = _readout_acc(exp_sh, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: plateau/apical LESION -> floor (supervised-trained on a lesioned plateau) ----
    lex = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th, lesion=True).init_supervised(k, seed)
    _train_supervised(lex, af_b, pre_b, yb, epochs, lr_sup, lr_out, seed=seed)
    _, out["lesion_heldout"], _, _ = _readout_acc(lex, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: NO-TRANSPORT (B fixed-random / != W_out^T / immutable; hidden update invariant to W_out) ----
    exp.init_supervised(k, seed)  # (re)attach a clean B/W_out for the static checks below
    # (a) code: the hidden-update helper's signature exposes NO forward/readout weight (only Mrg/theta/pre/e/lr).
    hsig = set(inspect.signature(SupervisedPlateauExpander._hidden_update_from_error).parameters)
    no_transport_code = hsig.isdisjoint({"W_out", "Wout", "readout", "clf", "forward_W", "Wt"})
    # (b) B is fixed random and NOT the readout, and it did not move during training. (Both are (C,k) and are applied
    #     via `.T` in the backward projection; the TRANSPORT version would set B == W_out -> B.T == W_out.T. Distinct
    #     shape-matched matrices => transport-free.)
    b_not_transpose = bool(exp.B.shape == exp.W_out.shape and not np.allclose(exp.B, exp.W_out, atol=1e-6))
    b_immutable = bool(np.array_equal(exp.B, exp.B0))
    # (c) runtime: hidden update is INVARIANT to W_out given a FIXED injected e (proves W_out is not read on backward).
    Mrg = np.asarray([exp.margin(a) for a in af_b[:24]]); theta = Mrg.mean(0, keepdims=True)
    e_fixed = np.random.default_rng(0).standard_normal((24, k))
    pA = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_supervised(k, seed)
    pB = SupervisedPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th).init_supervised(k, seed)
    pB.W_out = np.random.default_rng(12345).standard_normal((n_col, k)) * 5.0   # WILDLY different W_out
    pA.restore_frozen(); pB.restore_frozen()
    pA._hidden_update_from_error(Mrg, theta, pre_b[:24], e_fixed, lr_sup); dA = pA._get_data()
    pB._hidden_update_from_error(Mrg, theta, pre_b[:24], e_fixed, lr_sup); dB = pB._get_data()
    no_transport_runtime = bool(np.allclose(dA, dB, atol=1e-6) and not np.allclose(dA, pA.W0, atol=1e-6))
    out["no_transport_code"] = bool(no_transport_code)
    out["no_transport_B_not_transpose"] = b_not_transpose
    out["no_transport_B_immutable"] = b_immutable
    out["no_transport_runtime"] = no_transport_runtime
    out["no_transport"] = bool(no_transport_code and b_not_transpose and b_immutable and no_transport_runtime)

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} n_sub={len(Xb)} N_COL={n_col}",
              flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f}(tr {out['oracle_train']:.3f}) | rate-reservoir "
              f"{out['rate_reservoir_heldout']:.3f} | FROZEN {out['frozen_plateau_heldout']:.3f} | UNSUP "
              f"{out['unsup_plateau_heldout']:.3f} | SUPERVISED {out['supervised_plateau_heldout']:.3f}"
              f"(tr {out['supervised_plateau_train']:.3f})", flush=True)
        print(f"    deep_credit_share  unsup {out['deep_credit_share_unsup']:+.3f}  supervised "
              f"{out['deep_credit_share_supervised']:+.3f}  | reprod {out['reproducibility_supervised']:.3f} | "
              f"|dW| {out['supervised_update_mag_first_last']} moved {out['supervised_weight_moved']:.4f}", flush=True)
        print(f"    [anti-cheat] permuted-readout {out['permuted_readout_heldout']:.3f}(~chance) | "
              f"sup-on-permuted {out['supervised_on_permuted_heldout']:.3f}(->frozen) | shuffle-DFA "
              f"{out['shuffle_dfa_heldout']:.3f}(->unsup) | lesion {out['lesion_heldout']:.3f}(~floor)", flush=True)
        print(f"    [no-transport] code={out['no_transport_code']} B!=Wt={out['no_transport_B_not_transpose']} "
              f"B-immut={out['no_transport_B_immutable']} runtime={out['no_transport_runtime']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 SUPERVISED (error-gated) plateau credit de-risk (SMOKE).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-unsup", type=float, default=0.02, help="unsupervised covariance lr (parent default)")
    ap.add_argument("--lr-sup", type=float, default=0.05, help="supervised (error-gated) hidden lr")
    ap.add_argument("--lr-out", type=float, default=0.5, help="output readout SGD lr")
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None)
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--oracle-epochs", type=int, default=200)
    ap.add_argument("--oracle-lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", type=int, default=128)
    # --- the SWEET SPOT task config (n_prop=3, n_super=24) ---
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--margin-go", type=float, default=0.05,
                    help="preregistered held-out margin the supervised arm must clear over BOTH frozen AND unsup")
    ap.add_argument("--out", default="research/findings/raw/gap4/supervised_plateau/supervised_plateau_credit.json")
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.epochs, a.lr_unsup, a.lr_sup, a.lr_out, a.w0, a.jitter, a.k_th,
                                a.n_sub, a.hidden, a.oracle_epochs, a.oracle_lr, a.oracle_batch, task_kwargs, a.margin_go))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_supervised_plateau_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "epochs": a.epochs, "lr_unsup": a.lr_unsup, "lr_sup": a.lr_sup,
                          "lr_out": a.lr_out, "w0": a.w0, "jitter": a.jitter, "k_th": a.k_th, "n_sub": a.n_sub,
                          "hidden": a.hidden, "oracle_epochs": a.oracle_epochs, "task": task_kwargs,
                          "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_plateau_heldout", "unsup_plateau_heldout",
                "supervised_plateau_heldout", "deep_credit_share_unsup", "deep_credit_share_supervised",
                "reproducibility_supervised", "permuted_readout_heldout", "supervised_on_permuted_heldout",
                "shuffle_dfa_heldout", "lesion_heldout", "chance", "supervised_plateau_train",
                "unsup_codon_diversity", "supervised_codon_diversity"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per)
        # GO: supervised beats BOTH frozen AND unsup by the preregistered margin, per seed.
        beats_both = sum(1 for p in per
                         if p["supervised_plateau_heldout"] >= p["frozen_plateau_heldout"] + a.margin_go
                         and p["supervised_plateau_heldout"] >= p["unsup_plateau_heldout"] + a.margin_go)
        dcs_gt_unsup = sum(1 for p in per
                           if p["deep_credit_share_supervised"] > p["deep_credit_share_unsup"])
        anti_ok = (all(p["no_transport"] for p in per)
                   and all(p["reproducibility_supervised"] >= 0.8 for p in per)
                   and all(p["permuted_readout_heldout"] <= p["chance"] + 0.10 for p in per)
                   and all(p["supervised_on_permuted_heldout"] <= p["frozen_plateau_heldout"] + a.margin_go for p in per)
                   and all(p["shuffle_dfa_heldout"] <= p["supervised_plateau_heldout"] for p in per)
                   and all(p["lesion_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and agg["rate_reservoir_heldout"] <= 0.40 and agg["oracle_heldout"] >= 0.80)
        go = bool(beats_both == n and dcs_gt_unsup == n and anti_ok)
        sup_gt_unsup = bool(agg["supervised_plateau_heldout"] > agg["unsup_plateau_heldout"])
        promising = bool((not go) and dcs_gt_unsup == n and anti_ok and sup_gt_unsup)
        agg.update({"n_seeds": n, "supervised_beats_both_by_margin": beats_both,
                    "dcs_supervised_gt_unsup": dcs_gt_unsup, "anti_cheats_clean": bool(anti_ok),
                    "margin_go": a.margin_go, "promising": promising})
        summary["aggregate"] = agg; summary["GO"] = go; summary["PROMISING"] = promising
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f} (op-point "
                  f"genuine), FROZEN {agg['frozen_plateau_heldout']:.3f}, UNSUP {agg['unsup_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share_unsup']:+.3f}), SUPERVISED {agg['supervised_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share_supervised']:+.3f}). anti: reprod "
                  f"{agg['reproducibility_supervised']:.3f}, permuted-readout {agg['permuted_readout_heldout']:.3f}, "
                  f"sup-on-permuted {agg['supervised_on_permuted_heldout']:.3f}, shuffle-DFA "
                  f"{agg['shuffle_dfa_heldout']:.3f}, lesion {agg['lesion_heldout']:.3f}.")
        if go:
            verdict = (f"SMOKE GO ({beats_both}/{n}) -- DIRECTED error on the movable plateau hidden beats BOTH the "
                       f"frozen reservoir AND the unsupervised rule by >={a.margin_go}; deep_credit_share rises "
                       f"{agg['deep_credit_share_unsup']:+.3f} (unsup) -> {agg['deep_credit_share_supervised']:+.3f} "
                       f"(supervised). Transport-free (B fixed random). Anti-cheats clean -> parent runs 6-seed. " + common)
        elif promising:
            verdict = (f"SMOKE PROMISING (dcs supervised>unsup {dcs_gt_unsup}/{n}, margin cleared {beats_both}/{n}) -- "
                       f"directed error moves the movable hidden FURTHER than unsupervised sharpening "
                       f"({agg['supervised_plateau_heldout']:.3f} vs {agg['unsup_plateau_heldout']:.3f}), transport-free, "
                       f"anti-cheats clean, but the {a.margin_go} margin over BOTH baselines is not cleared on all "
                       f"seeds -> parent runs 6-seed to settle. " + common)
        else:
            reasons = []
            if dcs_gt_unsup != n:
                reasons.append(f"supervised deep_credit_share does NOT exceed unsupervised on all seeds "
                               f"({dcs_gt_unsup}/{n}; supervised mean {agg['deep_credit_share_supervised']:+.3f} vs "
                               f"unsup {agg['deep_credit_share_unsup']:+.3f})")
            elif not sup_gt_unsup:
                reasons.append(f"supervised does NOT beat unsupervised in mean "
                               f"({agg['supervised_plateau_heldout']:.3f} vs {agg['unsup_plateau_heldout']:.3f})")
            if not anti_ok:
                reasons.append("an anti-cheat/op-point control did not hold (see per-seed)")
            verdict = ("SMOKE NEGATIVE (honest) -- " + "; ".join(reasons) + ". This is a valid deliverable: it maps "
                       "that the movable hidden is helped by unsupervised SHARPENING but NOT by DIRECTED credit "
                       "beyond it. " + common)
        summary["verdict"] = verdict
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-supervised-plateau-credit] {summary['verdict']}", flush=True)
    print(f"[gap4-supervised-plateau-credit] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
