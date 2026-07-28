"""gap#4 DE-CONFOUNDED deep-credit de-risk: does the faithful BDSP credit rule TRAIN THE HIDDEN to accuracy that
BEATS a frozen-reservoir readout -- with the two load-bearing controls the prior arc kept MISSING?

Reuses `BdspNet` (the faithful numpy replica of `sim/kernels.fused_bdsp_update`, Payeur-Naud 2021 M1.2) by import.
The 2026-07-23 GO showed `bdsp > reservoir` 6-seed on MNIST, but WITHOUT (a) confirming the forward is input-selective
at the operating point and (b) a directed-credit lesion. The 2026-07-19 root-cause (rank-1 collapse + SATURATED hidden
gate) is exactly the confound this closes: if the hidden fires the SAME units for every input (not input-differential),
the credit verdict cannot be trusted. So this runner:

ARMS (accuracy on held-out MNIST):
  - reservoir            : hidden FROZEN random, only readout trained  = the credit-INDEPENDENT baseline (default-ON)
  - fa_linear            : the working FA/DFA rate rule (reference credit)
  - bdsp                 : the FAITHFUL on-bridge rule (coincidence gate + sigmoid-baseline credit) -- UNDER TEST
  - bdsp_shufE           : the DIRECTED-CREDIT LESION -- bdsp with the output error `e` SHUFFLED across the batch in
                           the HIDDEN-credit path ONLY (readout still trained on the true e). This destroys the
                           per-sample input<->error covariance that IS the hidden learning signal while preserving
                           the credit MAGNITUDE + the readout. If `bdsp > bdsp_shufE ~ reservoir`, the win comes from
                           the hidden receiving its OWN task error, not from any hidden movement / a mere reservoir.
  - bdsp_permB           : METHODOLOGY control (NOT a lesion) -- bdsp with the credit routed through a FIXED random
                           permutation of hidden units. Feedback alignment is routing-INVARIANT (a permuted fixed-
                           random feedback is just ANOTHER valid FA instance), so this is EXPECTED ~= bdsp; it
                           documents WHY permuting the feedback is not a valid directed-credit lesion for FA.
  - bdsp_shuffled_target : bdsp trained on PERMUTED training labels, evaluated on TRUE test labels = pipeline-honesty
                           / no-leak control. Must be ~chance.

CONTROLS / MEASUREMENTS (the load-bearing de-confound):
  - INIT input-selectivity of the sparse hidden binary code (the post_gate the credit uses AND the readout input),
    measured BEFORE any training, per hidden layer:
       active_rate (should ~= frac) ; frac_units_input_differential (0<p_j<1) ; code_diversity (#unique rows / n) ;
       mean_pairwise_hamming ; and the DECISIVE one -- ncc_acc: nearest-class-centroid accuracy of the RAW hidden code
       (class means from a train batch, classify test by nearest centroid; a learned-readout-FREE selectivity number).
    ncc_acc >> chance == the hidden fires input-differentially in a class-relevant way == an input-selective op-point.
  - POST-training input-selectivity for the bdsp arm (did directed credit change the representation).

GO gate (6-seed, per frac): forward IS input-selective at INIT (ncc_acc > 2x chance)  AND  bdsp > reservoir + 0.01
AND  bdsp > bdsp_shufE + 0.01 (directed credit load-bearing)  AND  bdsp_shuffled_target < 2x chance. Seed bug N/A
(numpy default_rng + explicit model seed; no SimulationBridge -- see test_determinism TestSubstrateActuallySeeded).
numpy CPU, local, $0. NO sim/ edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from sim.dendritic_mlp import _MOMENTUM, _sig  # noqa: E402
from research.runners._gap4_bdsp_faithful_credit_derisk import BdspNet, _load_mnist, _sparsify  # noqa: E402

xp, _ = get_backend()


class DeconfBdspNet(BdspNet):
    """BdspNet + a directed-credit-scramble lesion (fixed per-net permutation of the credit's hidden-unit routing) +
    an input-selectivity probe on the sparse hidden binary code. Everything else inherited unchanged."""

    def __init__(self, sizes, seed=0, frac=1.0, p0=0.30, beta=1.0, credit_lesion=None):
        super().__init__(sizes, seed=seed, frac=frac, p0=p0, beta=beta)
        # credit_lesion in {None, "permB", "shufE"}:
        #   permB = route the credit through a FIXED per-net permutation of hidden units (methodology control; FA is
        #           routing-invariant so this is EXPECTED == bdsp -- documents that permuting feedback is not a lesion).
        #   shufE = shuffle the output error `e` across the batch in the HIDDEN-credit path only (readout keeps true e)
        #           -- the GENUINE directed-credit lesion: destroys per-sample input<->error covariance.
        self.credit_lesion = credit_lesion
        prng = np.random.default_rng(seed * 977 + 3)
        self._perm = [xp.asarray(prng.permutation(sizes[li + 1])) for li in range(len(sizes) - 2)]
        self._lrng = np.random.default_rng(seed * 4099 + 11)   # for per-step error shuffle

    def train_step(self, X, y, mode, lr):
        # Copy of BdspNet.train_step, faithful to the parent (same coincidence gate + sigmoid-baseline credit +
        # momentum), with an optional credit_lesion applied ONLY to the hidden-credit path (readout uses the true e).
        dense, sparse, lg = self._forward(X)
        y = xp.asarray(y); e = self._softmax(lg); e[xp.arange(len(y)), y] -= 1.0
        nW = len(self.W); upd = [None] * nW
        upd[-1] = -(sparse[-1].T @ e)                          # readout ALWAYS trained on the true error
        # hidden-credit error: shuffle across the batch (shufE lesion) or keep true
        if self.credit_lesion == "shufE":
            perm = xp.asarray(self._lrng.permutation(e.shape[0]))
            e_hid = e[perm]
        else:
            e_hid = e
        for li in range(nW - 1):
            if mode == "reservoir":
                upd[li] = xp.zeros_like(self.W[li]); continue
            a_prev = sparse[li]
            ap = e_hid @ self.B[li]
            d_dense = dense[li + 1]
            spk_post = sparse[li + 1]
            post_gate = spk_post if mode in ("fa_coinc", "bdsp") else d_dense * (1.0 - d_dense)
            if mode in ("bdsp", "bdsp_nocoinc"):
                P = _sig(self.beta * ap)
                if self.Pbar[li] is None:
                    self.Pbar[li] = xp.full(P.shape[1], self.p0)
                self.Pbar[li] = 0.99 * self.Pbar[li] + 0.01 * P.mean(0)
                credit = P - self.Pbar[li][None, :]
            else:
                credit = ap
            if self.credit_lesion == "permB":
                credit = credit[:, self._perm[li]]             # routing permuted (methodology control, not a lesion)
            upd[li] = -(a_prev.T @ (credit * post_gate))
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]

    def hidden_binary_codes(self, X):
        """Return the list of sparse hidden binary codes (one per hidden layer) for input X, at current weights."""
        _, sparse, _ = self._forward(X)
        return sparse[1:]   # drop the input; keep each hidden layer's binary spike code

    def selectivity(self, Xtr, ytr, Xte, yte, n_pairs=400):
        """Input-selectivity of the sparse hidden binary code (the post_gate the credit uses + the readout input),
        measured at CURRENT weights. The decisive number is ncc_acc: a learned-readout-FREE nearest-class-centroid
        accuracy of the RAW hidden code -- if >> chance, the hidden fires input-differentially in a class-relevant
        way = an input-selective operating point. Returns per-hidden-layer metrics (list) + the last-layer headline."""
        Htr = self.hidden_binary_codes(Xtr)
        Hte = self.hidden_binary_codes(Xte)
        ytr = xp.asarray(ytr); yte = xp.asarray(yte)
        n_out = self.n_out
        rng = np.random.default_rng(12345)
        out = []
        for H_tr, H_te in zip(Htr, Hte):
            p = H_te.mean(0)                                        # per-unit activation rate on test
            active_rate = float(to_host(H_te.mean()))
            frac_diff = float(to_host(xp.mean((p > 1e-6) & (p < 1.0 - 1e-6))))   # input-differential units
            # code diversity: fraction of distinct rows
            H_host = np.asarray(to_host(H_te)).astype(np.int8)
            n = H_host.shape[0]
            code_div = float(np.unique(H_host, axis=0).shape[0]) / max(1, n)
            # mean pairwise normalized Hamming over random pairs
            ii = rng.integers(0, n, n_pairs); jj = rng.integers(0, n, n_pairs)
            ham = float(np.mean(np.abs(H_host[ii] - H_host[jj]).mean(1)))
            # nearest-class-centroid accuracy (learned-readout-FREE): class means on train, classify test by nearest
            cents = xp.stack([H_tr[ytr == c].mean(0) if int(to_host(xp.sum(ytr == c))) > 0
                              else xp.zeros(H_tr.shape[1]) for c in range(n_out)])   # (n_out, hid)
            # euclidean nearest centroid
            d = ((H_te[:, None, :] - cents[None, :, :]) ** 2).sum(-1)                # (n_te, n_out)
            pred = xp.argmin(d, 1)
            ncc = float(to_host(xp.mean(pred == yte)))
            out.append(dict(active_rate=round(active_rate, 4), frac_units_input_differential=round(frac_diff, 4),
                            code_diversity=round(code_div, 4), mean_pairwise_hamming=round(ham, 4),
                            ncc_acc=round(ncc, 4)))
        return out


def _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, frac, a, credit_lesion=None, shuffle_target=False):
    net = DeconfBdspNet(sizes, seed=seed, frac=frac, p0=a.p0, beta=a.beta, credit_lesion=credit_lesion)
    sel_init = net.selectivity(Xtr, ytr, Xte, yte)          # BEFORE any training = the operating-point control
    ytr_use = ytr
    if shuffle_target:
        prng = np.random.default_rng(seed * 71 + 5)
        ytr_use = ytr[prng.permutation(len(ytr))]           # destroy X->y; eval stays on TRUE test labels
    rng = np.random.default_rng(seed * 131 + 7)
    for _ in range(a.epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), a.batch):
            idx = order[s:s + a.batch]
            net.train_step(Xtr[idx], ytr_use[idx], mode, a.lr)
    acc = net.accuracy(Xte, yte)
    clean = credit_lesion is None and not shuffle_target
    sel_post = net.selectivity(Xtr, ytr, Xte, yte) if mode == "bdsp" and clean else None
    return acc, sel_init, sel_post


def one_seed(seed, a):
    Xtr, ytr, Xte, yte = _load_mnist(a.n_train, a.n_test, seed)
    sizes = [784] + [a.hidden] * a.depth + [10]
    chance = 1.0 / 10
    rows = []
    for frac in a.fracs:
        res, sel_init, _ = _train_eval("reservoir", Xtr, ytr, Xte, yte, sizes, seed, frac, a)
        fal, _, _ = _train_eval("fa_linear", Xtr, ytr, Xte, yte, sizes, seed, frac, a)
        bd, _, sel_post = _train_eval("bdsp", Xtr, ytr, Xte, yte, sizes, seed, frac, a)
        bd_shufE, _, _ = _train_eval("bdsp", Xtr, ytr, Xte, yte, sizes, seed, frac, a, credit_lesion="shufE")
        bd_permB, _, _ = _train_eval("bdsp", Xtr, ytr, Xte, yte, sizes, seed, frac, a, credit_lesion="permB")
        bd_sht, _, _ = _train_eval("bdsp", Xtr, ytr, Xte, yte, sizes, seed, frac, a, shuffle_target=True)
        # decisive selectivity = worst hidden layer's ncc_acc at INIT
        ncc_init = min(h["ncc_acc"] for h in sel_init)
        gate_sel = ncc_init > 2 * chance
        gate_res = bd > res + 0.01
        gate_shufE = bd > bd_shufE + 0.01               # directed (task-error) credit is load-bearing
        gate_sht = bd_sht < 2 * chance
        go = gate_sel and gate_res and gate_shufE and gate_sht
        row = dict(frac=frac, reservoir=round(res, 4), fa_linear=round(fal, 4), bdsp=round(bd, 4),
                   bdsp_shufE=round(bd_shufE, 4), bdsp_permB=round(bd_permB, 4),
                   bdsp_shuffled_target=round(bd_sht, 4), ncc_init=round(ncc_init, 4),
                   sel_init=sel_init, sel_post_bdsp=sel_post,
                   gate_input_selective=gate_sel, gate_bdsp_gt_reservoir=gate_res,
                   gate_bdsp_gt_shufE=gate_shufE, gate_shuffled_target_chance=gate_sht, GO=go)
        rows.append(row)
        print(f"  [seed {seed}] frac={frac:.2f}: RES={res:.3f} fa_lin={fal:.3f} bdsp={bd:.3f} "
              f"shufE={bd_shufE:.3f} permB={bd_permB:.3f} shufY={bd_sht:.3f} | ncc_init={ncc_init:.3f} "
              f"[sel>{2*chance:.2f}:{gate_sel} bdsp>res:{gate_res} bdsp>shufE:{gate_shufE} shufY<ch:{gate_sht}] GO={go}")
    return dict(seed=seed, sizes=sizes, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.1, 0.05])
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.03)          # the validated op-point (0.3 = dense->chance artifact)
    ap.add_argument("--out", default="research/findings/raw/gap4/deconfounded_credit.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap4-deconf] DE-CONFOUNDED credit-vs-reservoir (input-selectivity + directed-credit-scramble + "
          f"shuffled-target). hidden={a.hidden} depth={a.depth} fracs={a.fracs} lr={a.lr} p0={a.p0} beta={a.beta} "
          f"seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    print("[gap4-deconf] SUMMARY (mean over seeds):")
    for i, frac in enumerate(a.fracs):
        keys = ["reservoir", "fa_linear", "bdsp", "bdsp_shufE", "bdsp_permB", "bdsp_shuffled_target", "ncc_init"]
        agg = {k: float(np.mean([p["rows"][i][k] for p in per])) for k in keys}
        n_go = sum(p["rows"][i]["GO"] for p in per)
        print(f"  frac={frac:.2f}: RES={agg['reservoir']:.3f} fa_lin={agg['fa_linear']:.3f} bdsp={agg['bdsp']:.3f} "
              f"shufE={agg['bdsp_shufE']:.3f} permB={agg['bdsp_permB']:.3f} shufY={agg['bdsp_shuffled_target']:.3f} "
              f"ncc_init={agg['ncc_init']:.3f} | GO {n_go}/{len(per)}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, args=vars(a), per=per), f, indent=2)
    print(f"[gap4-deconf] wrote {a.out}")


if __name__ == "__main__":
    main()
