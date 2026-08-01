"""gap#4 REAL-SPIKES BDSP credit de-risk -- the DECISIVE gap#4 crux test.

THE QUESTION: the DFA-error-gated supervised covariance rule (`_gap4_supervised_plateau_credit_derisk`) and the
unsupervised movable-plateau rule (`_gap4_realspikes_credit_derisk`) were each tested on the plateau hidden. This
runner combines the two DECISIVE ingredients that were still untested TOGETHER on the REAL-spikes read: (1) a
COINCIDENCE gate -- the hidden weight is updated only where a real feature SPIKE (pre event) coincides with a real
plateau EVENT (post codon), and (2) a SIGMOID-BASELINE BDSP credit -- the DFA-projected output error is squashed
through a sigmoid and compared against a per-column EMA baseline (behavioural-timescale synaptic plasticity: the
plateau probability relative to its running average sets the sign of the change). Trained and read entirely on REAL
SPIKES (the codon = cp_v_apical > FLOOR after a real spiking forward pass).

THE RULE (all local + transport-free -- the hidden update reads ONLY pre_bin, post_bin, B, e; NEVER W_out^T):

    post_bin[i,c] = codon(a_i)                      # (N,C) binary plateau EVENT via the REAL-spikes read
    logits        = post_bin @ W_out + b_out        # forward readout (its OWN gradient trains it)
    e             = softmax(logits) - onehot(y)     # (N,k) output error
    ap[i,c]       = (e @ B.T)[i,c]  (B fixed random) # DFA-projected error -> transport-free (B != W_out)
    sig[i,c]      = sigmoid(beta * ap[i,c])          # P_post in [0,1]
    credit[i,c]   = sig[i,c] - Pbar[c]               # sigmoid-baseline BDSP credit (EMA baseline Pbar)
    dW[c,f]       = -eta * mean_i post_bin[i,c] * credit[i,c] * pre_bin[i,f]   # COINCIDENCE-gated, descent sign
    W[c,:] <- renormalized to its INITIAL L2 norm    # per-column homeostasis companion (same as unsupervised)
    Pbar <- (1-rho) Pbar + rho mean_i sig            # EMA baseline update

The coincidence gate is the whole point: a synapse moves only where BOTH the presynaptic feature fired AND the
postsynaptic column plateaued, scaled by how much the DFA error says that column's plateau should rise/fall relative
to its own recent average. This is the BDSP-flavoured directed-credit test on a MOVABLE, RELIABLE, REAL-spikes hidden.

REUSE: the REAL-spikes read (`RealSpikesPlateauExpander._vap` + `feat_spike_counts` + `configure_read`), the frozen /
oracle / rate-reservoir op-point arms, the deep_credit_share, and the anti-cheat helpers come UNCHANGED from
`_gap4_realspikes_credit_derisk`; the DFA plumbing (fixed random B / W_out / readout SGD / shuffle-error control) is
lifted from `_gap4_supervised_plateau_credit_derisk`. NO edit to any shared runner, NO sim/ edit.

ANTI-CHEATS (all mandatory): reproducibility >= 0.8 under the REAL read; permuted-READOUT -> chance (readout leakage);
permuted-TRAINING-label (NEW, load-bearing) -- a fresh BDSP trained on shuffled y must NOT beat frozen; shuffle-DFA-
error -> degrades toward the frozen/unsupervised level (the error routing is load-bearing); plateau/apical LESION ->
floor; NO-TRANSPORT (train_epoch_bdsp signature exposes no readout weight, B fixed random, B != W_out, B immutable).

GO GATE (mirror the realspikes harness, 6-seed): credit (real-spikes BDSP) beats the FROZEN on-bridge reservoir on
held-out inheritance by >= --margin-go on >= ceil(0.834*n) seeds AND deep_credit_share > 0 on all n seeds, with the
op-point genuine (oracle >= 0.80, rate-reservoir <= 0.45) and all anti-cheats holding.

Run (mirror realspikes; SIM_BACKEND override is the user's call):
    SIM_BACKEND=cupy python -m research.runners._gap4_realspikes_bdsp_credit_derisk --seeds 42 --epochs 30
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

# --- reuse-by-import: the REAL-spikes read/harness + the covariance-rule plumbing + the DFA softmax ---
from research.runners._gap4_realspikes_credit_derisk import RealSpikesPlateauExpander
from research.runners._gap4_plastic_plateau_credit_derisk import (
    fit_lin, topk_active, _sm, _readout_acc, _reproducibility, _codon_diversity,
    _rate_reservoir_heldout, FLOOR, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)


# ============================================================================================================
# The BDSP (coincidence-gated + sigmoid-baseline) real-spikes plateau expander. Inherits the REAL-spikes read
# (_vap / feat_spike_counts / configure_read) and the whole bridge/reservoir plumbing UNCHANGED; adds a fixed
# random DFA feedback B, a forward readout W_out, and the BDSP credit rule. Init is byte-identical to the parent
# at the same seed -> the FROZEN and CREDIT arms start from the SAME reservoir.
# ============================================================================================================
class BDSPRealSpikesPlateauExpander(RealSpikesPlateauExpander):
    def init_bdsp(self, k, seed, p0=0.30):
        """Forward readout W_out (trained by its own gradient) + FIXED RANDOM DFA feedback B (INDEPENDENT stream,
        never W_out, never updated) + per-column EMA plateau-probability baseline Pbar (BDSP baseline)."""
        rng_out = np.random.default_rng(seed * 999 + 17)                    # readout init stream
        rng_B = np.random.default_rng(seed * 777 + 5)                       # B stream -- INDEPENDENT (transport-free)
        self.k = int(k)
        self.W_out = rng_out.standard_normal((self.NC, k)) * 0.01           # small forward readout (trained)
        self.b_out = np.zeros(k)
        self.B = rng_B.standard_normal((self.NC, k)) / np.sqrt(k)           # FIXED random DFA feedback
        self.B0 = self.B.copy()                                             # snapshot -> assert B immutable
        self.Pbar = np.full(self.NC, p0)                                    # EMA plateau-probability baseline
        return self

    def train_epoch_bdsp(self, active_sets, pre_bin, y, eta, lr_out, beta=1.0, rho=0.1,
                         shuffle_error=False, err_rng=None):
        """One SUPERVISED BDSP batch on the REAL-spikes read. The HIDDEN weight update reads ONLY
        (pre_bin, post_bin, B, e) -- NEVER W_out or W_out^T (reading W_out in the FORWARD pass to form the error is a
        normal forward pass, not weight transport; the BACKWARD projection uses the fixed random B). pre_bin[i,f] is a
        BINARY real feature-spike EVENT; post_bin[i,c] is the BINARY real plateau EVENT (codon)."""
        post_bin = np.asarray([self.codon(a) for a in active_sets])        # (N,C) binary plateau EVENT (real spikes)
        Y = np.eye(self.k)[np.asarray(y, int)]                             # (N,k) onehot
        logits = post_bin @ self.W_out + self.b_out                        # forward readout (W_out FORWARD only)
        P = _sm(logits)                                                    # softmax
        e = P - Y                                                          # (N,k) output error
        if shuffle_error and err_rng is not None:
            e = e[err_rng.permutation(len(e))]                             # ANTI-CHEAT: destroy per-sample routing
        ap = e @ self.B.T                                                  # (N,C) DFA-projected error (B fixed random)
        sig = 1.0 / (1.0 + np.exp(-beta * ap))                            # P_post in [0,1]
        credit = sig - self.Pbar[None, :]                                  # (N,C) sigmoid-baseline BDSP credit
        # COINCIDENCE-gated update: pre event x post event x credit; descent sign -> DESCEND the loss
        dW_full = -eta * ((post_bin * credit).T @ pre_bin) / len(active_sets)   # (C,F)
        data = self._get_data()
        data = data + dW_full[self.syn_col, self.syn_feat]
        np.maximum(data, 0.0, out=data)                                    # excitatory (w >= 0)
        # per-column L2 renorm to the INITIAL norm (COPIED verbatim from PlasticPlateauExpander.train_epoch)
        cur = np.sqrt(np.array([np.sum(data[self.syn_col == c] ** 2) for c in range(self.NC)]))
        scale = np.where(cur > 1e-9, self.col_norm0 / (cur + 1e-12), 1.0)
        data = data * scale[self.syn_col]
        self._set_data(data)
        # update the EMA plateau-probability baseline (BDSP behavioural-timescale baseline)
        self.Pbar = (1 - rho) * self.Pbar + rho * sig.mean(0)
        # co-train the output readout by its OWN gradient (last layer; standard SGD, NOT a hidden transport)
        gout = e / len(active_sets)
        self.W_out -= lr_out * (post_bin.T @ gout)
        self.b_out -= lr_out * gout.sum(0)
        return float(np.mean(np.abs(dW_full)))


def _mk(n_feat, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, lesion=False):
    return BDSPRealSpikesPlateauExpander(n_feat, n_col, seed, w0=w0, jitter=jitter, k_th=k_th,
                                         lesion=lesion).configure_read(drive_pa, n_steps)


def _train_bdsp(exp, af, pre_bin, y, epochs, eta, lr_out, beta, shuffle_error=False, seed=0):
    exp.restore_frozen()
    err_rng = np.random.default_rng(seed * 71 + 3) if shuffle_error else None
    mags = []
    for _ in range(epochs):
        mags.append(exp.train_epoch_bdsp(af, pre_bin, y, eta, lr_out, beta, shuffle_error=shuffle_error,
                                         err_rng=err_rng))
    return mags


def run_seed(seed, n_col, epochs, eta, lr_out, beta, p0, w0, jitter, k_th, n_sub, hidden, oracle_epochs, oracle_lr,
             oracle_batch, drive_pa, n_steps, task_kwargs, margin_go, verbose=True):
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(seed * 13 + 1); keep = srng.permutation(len(Xtr))[:min(n_sub, len(Xtr))]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    af_b = topk_active(Xb, TOPK); af_h = topk_active(Xh, TOPK)
    chance = float(max(np.mean(yh == c) for c in np.unique(yh))) if len(yh) else float("nan")
    out = {"seed": seed, "n_in": n_in, "k": k, "chance": chance, "n_train_sub": len(Xb), "n_heldout_inherit": len(yh),
           "drive_pa": drive_pa, "n_steps": n_steps}

    # ---- ARM 4a: oracle (fenced backprop depth-2) + ARM 4b: frozen random RATE reservoir (op-point controls) ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_train"] = float(onet.accuracy(Xtr, ytr)); out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n_col, seed)

    # ---- ONE expander, identical init -> FROZEN and CREDIT both from the SAME reservoir, read via REAL SPIKES ----
    exp = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps)
    # PRECOMPUTE the BINARY real feature-spike EVENT (weight-independent -> once) = pre-activity for the coincidence gate
    pre_bin = (np.asarray([exp.feat_spike_counts(a) for a in af_b]) > 0).astype(float)   # (N, F) binary spike event

    # ARM 1: FROZEN reservoir (real-spikes read)
    exp.restore_frozen()
    fz_tr, fz_ho, Cb_fz, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["frozen_plateau_train"] = fz_tr; out["frozen_plateau_heldout"] = fz_ho
    out["frozen_codon_diversity"] = _codon_diversity(Cb_fz)

    # ARM 2: CREDIT -- coincidence-gated sigmoid-baseline BDSP, trained + read on REAL SPIKES
    exp.init_bdsp(k, seed, p0)                                             # attach W_out + fixed random B + Pbar
    mags = _train_bdsp(exp, af_b, pre_bin, yb, epochs, eta, lr_out, beta, seed=seed)
    cr_tr, cr_ho, Cb_cr, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["credit_plateau_train"] = cr_tr; out["credit_plateau_heldout"] = cr_ho
    out["credit_codon_diversity"] = _codon_diversity(Cb_cr)
    out["credit_update_mag_first_last"] = [round(mags[0], 6), round(mags[-1], 6)]

    # ---- deep_credit_share = (credit - frozen) / (oracle - frozen) ----
    denom = out["oracle_heldout"] - out["frozen_plateau_heldout"]
    out["deep_credit_share"] = float((out["credit_plateau_heldout"] - out["frozen_plateau_heldout"]) / denom) \
        if abs(denom) > 1e-6 else float("nan")
    # ATTRIBUTION: does the coincidence-gated BDSP credit add held-out over the FROZEN reservoir, or is the reservoir
    # already carrying it? (attribution-required gate + the honest "whose is the difference".)
    from tools.lab import attributable_to
    attributable_to("coincidence-gated BDSP credit vs frozen on-bridge reservoir on held-out",
                    out["credit_plateau_heldout"], out["frozen_plateau_heldout"])

    # ---- ANTI-CHEAT: reproducibility under the REAL read (LOAD-BEARING) ----
    out["reproducibility"] = _reproducibility(exp, af_b)

    # ---- ANTI-CHEAT: permuted-READOUT -> chance (readout leakage on the trained codon) ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    Ctr = np.asarray([exp.codon(a) for a in af_b]); Cte = np.asarray([exp.codon(a) for a in af_h])
    clf_p = fit_lin(Ctr, yperm, k)
    out["permuted_readout_heldout"] = float(np.mean(clf_p(Cte) == yh))

    # ---- ANTI-CHEAT: permuted-TRAINING-label (NEW, load-bearing) -> a BDSP trained on shuffled y must NOT beat frozen ----
    exp_perm = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps)
    exp_perm.init_bdsp(k, seed, p0)
    _train_bdsp(exp_perm, af_b, pre_bin, yperm, epochs, eta, lr_out, beta, seed=seed)
    _, out["bdsp_on_permuted_heldout"], _, _ = _readout_acc(exp_perm, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: shuffle the DFA error across the batch -> degrade (error routing is load-bearing) ----
    exp_sh = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps)
    exp_sh.init_bdsp(k, seed, p0)
    _train_bdsp(exp_sh, af_b, pre_bin, yb, epochs, eta, lr_out, beta, shuffle_error=True, seed=seed)
    _, out["shuffle_error_heldout"], _, _ = _readout_acc(exp_sh, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: plateau/apical LESION -> floor (BDSP-trained on a lesioned real-spikes plateau) ----
    lex = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, lesion=True)
    pre_l = (np.asarray([lex.feat_spike_counts(a) for a in af_b]) > 0).astype(float)
    lex.init_bdsp(k, seed, p0)
    _train_bdsp(lex, af_b, pre_l, yb, epochs, eta, lr_out, beta, seed=seed)
    _, out["lesion_heldout"], _, _ = _readout_acc(lex, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: NO-TRANSPORT (the BDSP hidden update exposes no readout weight; B fixed random / != W_out / immutable) ----
    bsig = set(inspect.signature(BDSPRealSpikesPlateauExpander.train_epoch_bdsp).parameters)
    out["no_transport_code"] = bool(bsig.isdisjoint({"W_out", "readout", "clf", "Wout"}))
    out["no_transport_B_immutable"] = bool(np.array_equal(exp.B, exp.B0))          # B never updated during training
    out["no_transport_B_not_transpose"] = bool(not np.allclose(exp.B, exp.W_out))  # B is not the readout weight
    out["no_transport"] = bool(out["no_transport_code"] and out["no_transport_B_immutable"]
                               and out["no_transport_B_not_transpose"])

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} drive={drive_pa} steps={n_steps}",
              flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f} | rate-reservoir {out['rate_reservoir_heldout']:.3f} | "
              f"FROZEN {out['frozen_plateau_heldout']:.3f} | CREDIT(BDSP) {out['credit_plateau_heldout']:.3f}"
              f"(tr {out['credit_plateau_train']:.3f}) | deep_credit_share {out['deep_credit_share']:+.3f}", flush=True)
        print(f"    [anti-cheat] reprod {out['reproducibility']:.3f} | permuted-readout "
              f"{out['permuted_readout_heldout']:.3f} | bdsp-on-permuted {out['bdsp_on_permuted_heldout']:.3f} | "
              f"shuffle-DFA {out['shuffle_error_heldout']:.3f} | lesion {out['lesion_heldout']:.3f} | "
              f"no-transport code={out['no_transport_code']} B-immut={out['no_transport_B_immutable']} "
              f"B!=Wout={out['no_transport_B_not_transpose']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 REAL-SPIKES coincidence-gated sigmoid-baseline BDSP credit crux test.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.03, help="BDSP (coincidence-gated) hidden lr")
    ap.add_argument("--lr-out", type=float, default=0.2, help="output readout SGD lr")
    ap.add_argument("--beta", type=float, default=1.0, help="sigmoid gain on the DFA-projected error")
    ap.add_argument("--p0", type=float, default=0.30, help="initial per-column EMA plateau-probability baseline")
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None)
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48)
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
    ap.add_argument("--margin-go", type=float, default=0.05)
    ap.add_argument("--out", default="research/findings/raw/gap4/realspikes_bdsp/realspikes_bdsp_credit.json")
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.epochs, a.eta, a.lr_out, a.beta, a.p0, a.w0, a.jitter, a.k_th, a.n_sub,
                                a.hidden, a.oracle_epochs, a.oracle_lr, a.oracle_batch, a.drive_pa, a.n_steps,
                                task_kwargs, a.margin_go))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_realspikes_bdsp_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "epochs": a.epochs, "eta": a.eta, "lr_out": a.lr_out, "beta": a.beta,
                          "p0": a.p0, "w0": a.w0, "jitter": a.jitter, "drive_pa": a.drive_pa, "n_steps": a.n_steps,
                          "task": task_kwargs, "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_plateau_heldout", "credit_plateau_heldout",
                "deep_credit_share", "reproducibility", "permuted_readout_heldout", "bdsp_on_permuted_heldout",
                "shuffle_error_heldout", "lesion_heldout", "chance", "credit_plateau_train"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per); need = int(np.ceil(0.834 * n))
        beats = sum(1 for p in per if p["credit_plateau_heldout"] >= p["frozen_plateau_heldout"] + a.margin_go)
        dcs_pos = sum(1 for p in per if p["deep_credit_share"] > 0)
        anti_ok = (all(p["no_transport"] for p in per)
                   and all(p["reproducibility"] >= 0.8 for p in per)
                   and all(p["permuted_readout_heldout"] <= p["chance"] + 0.10 for p in per)
                   and all(p["bdsp_on_permuted_heldout"] <= p["frozen_plateau_heldout"] + a.margin_go for p in per)
                   and all(p["shuffle_error_heldout"] <= p["credit_plateau_heldout"] for p in per)
                   and all(p["lesion_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and agg["oracle_heldout"] >= 0.80 and agg["rate_reservoir_heldout"] <= 0.45)
        go = bool(beats >= need and dcs_pos == n and anti_ok)
        agg.update({"n_seeds": n, "credit_beats_frozen_by_margin": beats, "seeds_needed": need,
                    "dcs_positive": dcs_pos, "anti_cheats_clean": bool(anti_ok), "margin_go": a.margin_go})
        summary["aggregate"] = agg; summary["GO"] = go
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f}, FROZEN "
                  f"{agg['frozen_plateau_heldout']:.3f}, CREDIT(BDSP) {agg['credit_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share']:+.3f}). anti: reprod {agg['reproducibility']:.3f}, permuted-readout "
                  f"{agg['permuted_readout_heldout']:.3f}, bdsp-on-permuted {agg['bdsp_on_permuted_heldout']:.3f}, "
                  f"shuffle-DFA {agg['shuffle_error_heldout']:.3f}, lesion {agg['lesion_heldout']:.3f}.")
        if go:
            summary["verdict"] = (f"REAL-SPIKES BDSP GO ({beats}/{n} beat frozen, dcs>0 {dcs_pos}/{n}) -- the "
                                  f"coincidence-gated sigmoid-baseline BDSP directed-credit rule SURVIVES the port to "
                                  f"real spikes and beats the frozen on-bridge reservoir. " + common)
        else:
            summary["verdict"] = (f"REAL-SPIKES BDSP NEGATIVE (beats frozen {beats}/{n} need {need}, dcs>0 {dcs_pos}/{n}, "
                                  f"anti_ok {anti_ok}) -- the coincidence-gated BDSP rule does NOT clearly beat the "
                                  f"frozen on-bridge reservoir on real-spikes held-out. " + common)
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-realspikes-bdsp-credit] {summary['verdict']}", flush=True)
    print(f"[gap4-realspikes-bdsp-credit] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
