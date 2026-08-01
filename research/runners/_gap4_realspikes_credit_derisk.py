"""gap#4 REAL-SPIKES credit comparison (the DECISIVE test of the on-bridge spiking port).

The pre-gate (`_gap4_realspikes_pregate_probe`) CLEARED the degenerate-forward-pass boundary: a real spiking forward
pass on the movable-plateau bridge gives input-dependent, reproducible column codons (3/3 seeds). But those codons are
ANTI-correlated with the boolean-hold reset-read (the rate 5/6 stand-in), so the rate result does NOT transfer by
assumption. THIS runner answers the decisive question: does the UNSUPERVISED movable-plateau covariance rule, trained
and read on REAL SPIKES, still beat a FROZEN on-bridge reservoir on held-out inheritance (the real-spikes analogue of
the rate 5/6, deep_credit_share > 0)?

THE ONLY CHANGE from the rate 5/6 runner is the READ/pre-activity:
  - `_vap` -> a REAL spiking forward pass (drive features via cp_external_input_current, integrate n_steps, features
    SPIKE, propagate through the coincidence pathway to the columns' real plateau; return raw cp_v_apical per column).
    margin()/codon() (inherited) then use real-spikes plateaus.
  - pre-activity in the covariance rule = REAL feature SPIKE COUNTS (not a 0/1 indicator). Features are the INPUT layer,
    driven by input current -> their counts are INDEPENDENT of the trainable feature->column weights -> precompute once.
Everything else -- the per-column plateau-gated covariance rule, the L2-renorm homeostasis, the frozen reservoir, the
oracle/rate-reservoir op-point controls, deep_credit_share, the anti-cheats -- is inherited UNCHANGED. NO sim/ edit.

GO GATE (mirrors the rate 5/6, 6-seed): credit-trained (real-spikes read) beats the FROZEN on-bridge reservoir
(real-spikes read) on held-out by margin >= --margin-go on >= 5/6 seeds AND deep_credit_share > 0 on 6/6, with the
op-point genuine (oracle >= 0.80, rate-reservoir fails) and all anti-cheats holding (no-transport; permuted-label ->
chance; lesion -> floor; reproducibility >= 0.8 under the REAL read -- LOAD-BEARING).
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

from research.runners._gap4_plastic_plateau_credit_derisk import (
    PlasticPlateauExpander, fit_lin, topk_active, _readout_acc, _reproducibility, _codon_diversity,
    _rate_reservoir_heldout, FLOOR, ACT_TH, TOPK,
    make_task_semantic_inheritance, _train_oracle, _acc_on, DendriticMLP)
from research.runners._emerge12_stageB2_bridge_tm_derisk import reset_soma, _clear_apical
from sim.backend import to_host as _host


class RealSpikesPlateauExpander(PlasticPlateauExpander):
    """Identical bridge/rule/anti-cheats to PlasticPlateauExpander; the ONLY override is the READ: a real spiking
    forward pass replaces the boolean-hold reset-read. margin()/codon() inherited (they call the overridden _vap)."""
    def configure_read(self, drive_pa=1200.0, n_steps=30):
        self.drive_pa = float(drive_pa); self.n_steps = int(n_steps)
        return self

    def _vap(self, active_feats):
        """REAL spiking forward pass -> raw cp_v_apical per column. Drive the active FEATURE neurons with input current
        for n_steps; features integrate + SPIKE; their real firings drive the coincidence pathway to the columns'
        plateau. (Reads NO readout weight -> the no-transport property is preserved: the read is a forward pass only.)"""
        b = self.b; xp = b.xp if hasattr(b, "xp") else np
        n = int(b.core_config.num_neurons)
        reset_soma(b); _clear_apical(b)
        inp = np.zeros(n, np.float32)
        if len(active_feats):
            inp[self.ci[np.asarray(list(active_feats), int)]] = self.drive_pa
        inp_x = xp.asarray(inp)
        for _ in range(self.n_steps):
            b.cp_external_input_current[:] = inp_x
            b._run_one_simulation_step()
        vap = getattr(b, "cp_v_apical", None)
        if vap is None:
            return np.zeros(self.NC)
        return np.asarray(_host(vap))[self.ci][self.NF:self.NF + self.NC]

    def feat_spike_counts(self, active_feats):
        """REAL feature spike counts over the window (the pre-activity for the covariance rule). Weight-INDEPENDENT
        (features are input-driven), so this is precomputed once per input."""
        b = self.b; xp = b.xp if hasattr(b, "xp") else np
        n = int(b.core_config.num_neurons)
        reset_soma(b); _clear_apical(b)
        inp = np.zeros(n, np.float32)
        if len(active_feats):
            inp[self.ci[np.asarray(list(active_feats), int)]] = self.drive_pa
        inp_x = xp.asarray(inp); fs = np.zeros(self.NF)
        for _ in range(self.n_steps):
            b.cp_external_input_current[:] = inp_x
            b._run_one_simulation_step()
            fs += np.asarray(_host(b.cp_firing_states)).astype(np.float64)[self.ci[:self.NF]]
        return fs


def _mk(n_feat, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, lesion=False):
    return RealSpikesPlateauExpander(n_feat, n_col, seed, w0=w0, jitter=jitter, k_th=k_th,
                                     lesion=lesion).configure_read(drive_pa, n_steps)


def run_seed(seed, n_col, epochs, lr, w0, jitter, k_th, n_sub, hidden, oracle_epochs, oracle_lr, oracle_batch,
             drive_pa, n_steps, task_kwargs, margin_go, verbose=True):
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

    # ---- ONE expander, identical init -> FROZEN and CREDIT both from the SAME reservoir, both read via REAL SPIKES ----
    exp = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps)
    # PRECOMPUTE real feature spike counts for pre-activity (weight-independent -> once)
    pre_b = np.asarray([exp.feat_spike_counts(a) for a in af_b])           # (N, F) real feature spike counts

    # ARM 1: FROZEN reservoir (real-spikes read)
    exp.restore_frozen()
    fz_tr, fz_ho, Cb_fz, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["frozen_plateau_train"] = fz_tr; out["frozen_plateau_heldout"] = fz_ho
    out["frozen_codon_diversity"] = _codon_diversity(Cb_fz)

    # ARM 2: CREDIT -- unsupervised plateau covariance rule, trained + read on REAL SPIKES
    exp.restore_frozen()
    mags = []
    for _ in range(epochs):
        mags.append(exp.train_epoch(af_b, pre_b, lr))
    cr_tr, cr_ho, Cb_cr, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["credit_plateau_train"] = cr_tr; out["credit_plateau_heldout"] = cr_ho
    out["credit_codon_diversity"] = _codon_diversity(Cb_cr)
    out["credit_update_mag_first_last"] = [round(mags[0], 6), round(mags[-1], 6)]

    # ---- deep_credit_share = (credit - frozen) / (oracle - frozen) ----
    denom = out["oracle_heldout"] - out["frozen_plateau_heldout"]
    out["deep_credit_share"] = float((out["credit_plateau_heldout"] - out["frozen_plateau_heldout"]) / denom) \
        if abs(denom) > 1e-6 else float("nan")

    # ---- ANTI-CHEAT: reproducibility under the REAL read (LOAD-BEARING) ----
    out["reproducibility"] = _reproducibility(exp, af_b)

    # ---- ANTI-CHEAT: permuted-label readout -> chance ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    Ctr = np.asarray([exp.codon(a) for a in af_b]); Cte = np.asarray([exp.codon(a) for a in af_h])
    clf_p = fit_lin(Ctr, yperm, k)
    out["permuted_readout_heldout"] = float(np.mean(clf_p(Cte) == yh))

    # ---- ANTI-CHEAT: plateau/apical LESION -> floor (credit-trained on a lesioned real-spikes plateau) ----
    lex = _mk(n_in, n_col, seed, w0, jitter, k_th, drive_pa, n_steps, lesion=True)
    pre_l = np.asarray([lex.feat_spike_counts(a) for a in af_b])
    lex.restore_frozen()
    for _ in range(epochs):
        lex.train_epoch(af_b, pre_l, lr)
    _, out["lesion_heldout"], _, _ = _readout_acc(lex, af_b, yb, af_h, yh, k)

    # ---- ANTI-CHEAT: NO-TRANSPORT (the rule + read expose no readout weight) ----
    rsig = set(inspect.signature(RealSpikesPlateauExpander.train_epoch).parameters)
    vsig = set(inspect.signature(RealSpikesPlateauExpander._vap).parameters)
    out["no_transport_code"] = bool(rsig.isdisjoint({"W_out", "readout", "clf", "y", "labels"})
                                    and vsig.isdisjoint({"W_out", "readout", "clf"}))

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} drive={drive_pa} steps={n_steps}",
              flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f} | rate-reservoir {out['rate_reservoir_heldout']:.3f} | "
              f"FROZEN {out['frozen_plateau_heldout']:.3f} | CREDIT {out['credit_plateau_heldout']:.3f}"
              f"(tr {out['credit_plateau_train']:.3f}) | deep_credit_share {out['deep_credit_share']:+.3f}", flush=True)
        print(f"    [anti-cheat] reprod {out['reproducibility']:.3f} | permuted {out['permuted_readout_heldout']:.3f} | "
              f"lesion {out['lesion_heldout']:.3f} | no-transport {out['no_transport_code']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 REAL-SPIKES unsupervised movable-plateau credit comparison.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.02)
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
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--margin-go", type=float, default=0.05)
    ap.add_argument("--out", default="research/findings/raw/gap4/realspikes/realspikes_credit.json")
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.epochs, a.lr, a.w0, a.jitter, a.k_th, a.n_sub, a.hidden,
                                a.oracle_epochs, a.oracle_lr, a.oracle_batch, a.drive_pa, a.n_steps, task_kwargs,
                                a.margin_go))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_realspikes_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "epochs": a.epochs, "lr": a.lr, "w0": a.w0, "jitter": a.jitter,
                          "drive_pa": a.drive_pa, "n_steps": a.n_steps, "task": task_kwargs, "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        keys = ["oracle_heldout", "rate_reservoir_heldout", "frozen_plateau_heldout", "credit_plateau_heldout",
                "deep_credit_share", "reproducibility", "permuted_readout_heldout", "lesion_heldout", "chance",
                "credit_plateau_train"]
        agg = {kk: _m(kk) for kk in keys}
        n = len(per); need = int(np.ceil(0.834 * n))
        beats = sum(1 for p in per if p["credit_plateau_heldout"] >= p["frozen_plateau_heldout"] + a.margin_go)
        dcs_pos = sum(1 for p in per if p["deep_credit_share"] > 0)
        anti_ok = (all(p["no_transport_code"] for p in per) and all(p["reproducibility"] >= 0.8 for p in per)
                   and all(p["permuted_readout_heldout"] <= p["chance"] + 0.10 for p in per)
                   and all(p["lesion_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and agg["oracle_heldout"] >= 0.80 and agg["rate_reservoir_heldout"] <= 0.45)
        go = bool(beats >= need and dcs_pos == n and anti_ok)
        agg.update({"n_seeds": n, "credit_beats_frozen_by_margin": beats, "seeds_needed": need,
                    "dcs_positive": dcs_pos, "anti_cheats_clean": bool(anti_ok), "margin_go": a.margin_go})
        summary["aggregate"] = agg; summary["GO"] = go
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f}, FROZEN "
                  f"{agg['frozen_plateau_heldout']:.3f}, CREDIT {agg['credit_plateau_heldout']:.3f} "
                  f"(dcs {agg['deep_credit_share']:+.3f}). anti: reprod {agg['reproducibility']:.3f}, permuted "
                  f"{agg['permuted_readout_heldout']:.3f}, lesion {agg['lesion_heldout']:.3f}.")
        if go:
            summary["verdict"] = (f"REAL-SPIKES GO ({beats}/{n} beat frozen, dcs>0 {dcs_pos}/{n}) -- the unsupervised "
                                  f"movable-plateau rule SURVIVES the port to real spikes. " + common)
        else:
            summary["verdict"] = (f"REAL-SPIKES NEGATIVE (beats frozen {beats}/{n} need {need}, dcs>0 {dcs_pos}/{n}, "
                                  f"anti_ok {anti_ok}) -- the unsupervised rule does NOT clearly beat the frozen "
                                  f"on-bridge reservoir on real-spikes held-out. " + common)
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-realspikes-credit] {summary['verdict']}", flush=True)
    print(f"[gap4-realspikes-credit] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
