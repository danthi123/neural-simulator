"""ON-BRIDGE NODE PERTURBATION (the mission-critical escalation of the numpy fresh-class GO): does the zeroth-order,
three-factor deep-credit rule train a depth-2 HIDDEN region on the REAL spiking substrate, where the whole
feedback-alignment/burst family (BDSP/Burstprop/Urbanczik-Senn/pool-k/e-prop) sat at-or-below chance (incl. coupled)?

WHY THIS IS THE RIGHT ON-SUBSTRATE TEST: the numpy GO (0.944, k=8 antithetic, `_nodepert_deep_credit_derisk`) shows the
MECHANISM trains a depth-2 hidden layer to near-oracle. Node perturbation maps CLEANLY onto the sim's three-factor
machinery WITHOUT any backward channel: the perturbation IS an intrinsic-noise current injected into the HIDDEN region
(cp_external_input_current on idx_hid); the "credit" is a GLOBAL scalar loss-difference dL the world computes on the
spike-count readout (a legitimate reward-like signal from the body/environment, NOT host cognition); the weight update
is the local three-factor Hebbian dW_ij = -eta*(dL/sigma^2) * xi_j (post hidden-noise) * x_i (pre input-rate). Every
quantity is read from the bridge's real spikes; the outer-product+scale is the synaptic plasticity computation the sim
already runs shallowly (reward-modulated STDP). The genuine on-substrate risk: the spike-count readout dL is DISCRETE +
noisy, so NP's estimator variance (its own wall) is larger here -> more settle steps / larger k / more epochs may be
needed. That is exactly what this de-risk measures.

REUSE-BY-IMPORT: subclass `OnBridgeBDSPNet` (the committed 3-region input->hidden->output bridge scaffold) for the bridge
build + drive + settle + readout + pathway masks; ADD only (1) a hidden-region xi perturbation in the settle, (2) a
weight-WRITE into cp_connections.data[mask_in2hid], (3) the NP training loop. NO sim/ edit.

DE-RISK (single-variable vs the numpy GO): keep the tasks (emerge1 XOR-inheritance) + oracle/floor references.
GO = the NP-trained on-bridge HIDDEN region trains held-out acc ABOVE chance AND above the hidden-frozen floor (depth
helps) on the real spiking readout; the shuffle-dL anti-cheat (dL from a DIFFERENT example) MUST collapse to chance.

Run (numpy-CPU smoke): python -m research.runners._nodepert_onbridge_derisk --seed 42 --task emerge1 --smoke
     (GPU real):        SIM_BACKEND=cupy python -m research.runners._nodepert_onbridge_derisk --seeds 42 43 44 100 101 102 --task emerge1
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet


def _softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / max(e.sum(), 1e-12)


def _ce(readout, y):
    """readout = per-class spike counts -> softmax logits (scaled) -> cross-entropy. The scale keeps the softmax in a
    responsive range on small spike counts (a monotone read; the loss-DIFFERENCE is what NP uses, so the scale cancels
    to first order)."""
    p = _softmax(np.asarray(readout, float) * 0.5)
    return float(-np.log(max(p[int(y)], 1e-12))), p


class OnBridgeNPNet(OnBridgeBDSPNet):
    """Node-perturbation training on the 3-region bridge. Reuses the parent's bridge/drive/settle/readout; the apical
    BDSP credit path is never used (apical stays zero); credit is delivered as the three-factor NP weight update."""

    def _input_rate(self, x_bits, settle_steps):
        """presynaptic factor x_i: mean per-step firing fraction of each INPUT neuron during a clean settle."""
        from sim.backend import to_host
        self._reset_membrane(); self._set_apical(None); self._set_input_drive(x_bits)
        acc = np.zeros(len(self.idx_in))
        for _ in range(settle_steps):
            self.sb.cp_external_input_current[:] = self._drive_dev
            self.sb._run_one_simulation_step()
            acc += np.asarray(to_host(self.sb.cp_firing_states[self.idx_in])).astype(float)
        return acc / max(1, settle_steps)

    def _readout_perturbed(self, x_bits, settle_steps, xi):
        """SETTLE with a per-hidden-neuron current perturbation xi added to the standing hidden drive -> readout."""
        from sim.backend import to_host, from_host
        self._reset_membrane(); self._set_apical(None); self._set_input_drive(x_bits)
        drive = np.asarray(to_host(self._drive_dev)).astype(np.float32).copy()
        drive[self.idx_hid] += xi.astype(np.float32)            # the node perturbation = intrinsic-noise current on hidden
        pert = from_host(drive)
        acc = np.zeros(len(self.idx_out))
        for _ in range(settle_steps):
            self.sb.cp_external_input_current[:] = pert
            self.sb._run_one_simulation_step()
            acc += np.asarray(to_host(self.sb.cp_firing_states[self.idx_out])).astype(float)
        return np.array([acc[c * self.pool_out:(c + 1) * self.pool_out].sum() for c in range(self.n_classes)])

    def _write_in2hid_delta(self, delta_in_hid):
        """three-factor NP update: ADD delta_in_hid[i,j] (i in idx_in, j in idx_hid) into cp_connections.data over the
        input->hidden mask. Scatter by the masked COO (row=pre input i, col=post hidden j)."""
        from sim.backend import to_host, from_host
        data = np.asarray(to_host(self.sb.cp_connections.data)).astype(float)
        rpos = {v: i for i, v in enumerate(self.idx_in.tolist())}
        cpos = {v: i for i, v in enumerate(self.idx_hid.tolist())}
        rows = self._coo_row[self.mask_in2hid]; cols = self._coo_col[self.mask_in2hid]
        add = np.array([delta_in_hid[rpos[r], cpos[c]] for r, c in zip(rows, cols)])
        w_max = float(getattr(self.cfg, "bdsp_w_max", 200.0))
        newvals = np.clip(data[self.mask_in2hid] + add, -w_max, w_max)
        data[self.mask_in2hid] = newvals
        self.sb.cp_connections.data[:] = from_host(data)

    def train_np(self, Xtr, ytr, epochs, lr_hid, lr_out, sigma, settle_steps, seed, mode="np", k=1):
        """OUTPUT layer: clean delta rule on hidden->output (reuse the parent's readout for the pre=hidden factor).
        HIDDEN (input->hidden): NODE PERTURBATION. mode in {np, shuffle_dl, wrong_sign, hidden_frozen}."""
        rng = np.random.default_rng(seed + 5)
        idx = np.arange(len(Xtr))
        nH = len(self.idx_hid)
        for _ep in range(epochs):
            rng.shuffle(idx); perm = rng.permutation(len(Xtr))
            for t, j in enumerate(idx):
                x, yj = Xtr[j], int(ytr[j])
                r_clean = self._readout(x, settle_steps)            # unperturbed readout (parent freezes bdsp lr)
                L_clean, p = _ce(r_clean, yj)
                x_pre = self._input_rate(x, settle_steps)           # presynaptic factor
                grad = np.zeros((len(self.idx_in), nH))
                for _r in range(k):
                    xi = rng.standard_normal(nH) * sigma
                    Lp = _ce(self._readout_perturbed(x, settle_steps, xi), yj)[0]
                    Lm = _ce(self._readout_perturbed(x, settle_steps, -xi), yj)[0]
                    dL = 0.5 * (Lp - Lm)                            # antithetic central estimate
                    if mode == "shuffle_dl":                        # anti-cheat: credit from a DIFFERENT example
                        jk = perm[t]; xk, yk = Xtr[jk], int(ytr[jk])
                        Lpk = _ce(self._readout_perturbed(xk, settle_steps, xi), yk)[0]
                        Lmk = _ce(self._readout_perturbed(xk, settle_steps, -xi), yk)[0]
                        dL = 0.5 * (Lpk - Lmk)
                    if mode == "wrong_sign":
                        dL = -dL
                    grad += (dL / (sigma * sigma)) * np.outer(x_pre, xi)
                if mode != "hidden_frozen":
                    self._write_in2hid_delta(-lr_hid * grad / k)
                # OUTPUT: clean delta rule -- reuse the parent's apical output error injection as the learning signal.
                # (hidden->output plasticity via bdsp on the OUTPUT error only; keeps the output trainable while the
                #  hidden layer is trained purely by NP.)
                e = np.zeros(self.n_classes); e[yj] += 1.0; e -= p     # top error (target - p)
                self._apply_output_delta(e, x, settle_steps, lr_out)

    def _apply_output_delta(self, e_class, x_bits, settle_steps, lr_out):
        """clean delta rule on hidden->output: dW_hj_oc = lr_out * e_c * hid_rate_j. Read hidden rate during a settle,
        scatter into cp_connections.data over mask_hid2out."""
        from sim.backend import to_host, from_host
        self._reset_membrane(); self._set_apical(None); self._set_input_drive(x_bits)
        hacc = np.zeros(len(self.idx_hid))
        for _ in range(settle_steps):
            self.sb.cp_external_input_current[:] = self._drive_dev
            self.sb._run_one_simulation_step()
            hacc += np.asarray(to_host(self.sb.cp_firing_states[self.idx_hid])).astype(float)
        hid_rate = hacc / max(1, settle_steps)
        # per-output-neuron target error = its class's e_c
        data = np.asarray(to_host(self.sb.cp_connections.data)).astype(float)
        rpos = {v: i for i, v in enumerate(self.idx_hid.tolist())}
        cpos = {v: c for c in range(self.n_classes) for v in self.class_idx[c].tolist()}   # out neuron -> class
        rows = self._coo_row[self.mask_hid2out]; cols = self._coo_col[self.mask_hid2out]
        add = np.array([lr_out * e_class[cpos[c]] * hid_rate[rpos[r]] for r, c in zip(rows, cols)])
        w_max = float(getattr(self.cfg, "bdsp_w_max", 200.0))
        data[self.mask_hid2out] = np.clip(data[self.mask_hid2out] + add, -w_max, w_max)
        self.sb.cp_connections.data[:] = from_host(data)


def run(seed, task, parity_bits, epochs, lr_hid, lr_out, sigma, hidden, settle_steps, k):
    (Xtr, ytr), (Xte, yte), n_bits = _load_task(task, seed, parity_bits)
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    n_cls = int(max(ytr.max(), yte.max())) + 1
    chance = float(np.bincount(yte, minlength=n_cls).max() / len(yte))
    out = {"seed": seed, "task": task, "chance": round(chance, 3), "n_bits": int(n_bits)}
    for mode in ("np", "shuffle_dl", "hidden_frozen"):
        net = OnBridgeNPNet(seed, n_bits, hidden=hidden, pool_out=6)
        net.train_np(Xtr, ytr, epochs, lr_hid, lr_out, sigma, settle_steps, seed, mode=mode, k=k)
        out[mode] = round(net.accuracy(Xte, yte, settle_steps), 3)
        del net
    out["np_beats_chance"] = bool(out["np"] > chance + 0.05)
    out["shuffle_collapses"] = bool(out["shuffle_dl"] <= chance + 0.05)
    out["depth_helps"] = bool(out["np"] > out["hidden_frozen"] + 0.05)
    out["GO"] = bool(out["np_beats_chance"] and out["shuffle_collapses"] and out["depth_helps"])
    print(f"[np-onbridge seed={seed} {task}] chance={chance:.3f} | NP={out['np']:.3f} "
          f"shuffle_dl={out['shuffle_dl']:.3f} hidden_frozen={out['hidden_frozen']:.3f} "
          f"-> {'GO' if out['GO'] else 'no'} (beats={out['np_beats_chance']} shuffle_collapses={out['shuffle_collapses']} depth_helps={out['depth_helps']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", choices=["emerge1", "parity", "dense"], default="emerge1")
    ap.add_argument("--parity-bits", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr-hid", type=float, default=0.5); ap.add_argument("--lr-out", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=40.0)      # perturbation current (pA-scale on the hidden drive)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 4; a.settle_steps = 20
    res = [run(s, a.task, a.parity_bits, a.epochs, a.lr_hid, a.lr_out, a.sigma, a.hidden, a.settle_steps, a.k)
           for s in a.seeds]
    if len(res) > 1:
        ng = sum(1 for r in res if r["GO"])
        print(f"[np-onbridge] {ng}/{len(res)} seeds GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"))


if __name__ == "__main__":
    main()
