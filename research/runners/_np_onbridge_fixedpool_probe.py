"""ON-BRIDGE NP with a FIXED-POOLING readout (sidesteps the spiking-readout-TRAINING blocker): the earlier on-bridge NP
attempts trained a ridge/delta readout on the noisy spike-count output and it was noise-limited (delta-rule + population
averaging both ~chance). This probe removes the trained readout entirely: the readout is a FIXED class-pooling (class c =
output pool c; the "logit" for c = the summed spike count of pool c), so there is NOTHING to train on the read side. Node
perturbation then trains BOTH spiking layers (input->hidden AND hidden->output) to make the correct pool fire, via the
rate-coincidence three-factor (perturb a region's pre-activation with intrinsic-noise current xi, read the GLOBAL
fixed-pool CE loss-difference dL, update the region's incoming weights by dW += (dL/sigma^2) * outer(xi_t-node-credit, pre-rate)).

THE QUESTION: does NP's credit train the on-bridge spiking net to CLASSIFY (fixed pooling), where the trained-readout
attempts failed? GO = NP beats the frozen (untrained) floor + the shuffle anti-cheat collapses. If yes -> NP's deep credit
works on the real spiking substrate (the mission-critical on-spike claim, in the cleanest readout-free form).

Reuse-by-import: OnBridgeBDSPNet's bridge scaffold (regions/drive/settle/pooling/pathway masks) via OnBridgeNPNet. NO sim/ edit.
Run: SIM_BACKEND=numpy python -m research.runners._np_onbridge_fixedpool_probe --seeds 42 43 44
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet
from sim.backend import to_host, from_host


class FixedPoolNP(OnBridgeBDSPNet):
    """Fixed class-pooling readout (no trained readout) + node perturbation on BOTH layers."""

    def _pooled_logits(self, x_bits, settle):
        """settle (no learning, no apical) -> per-class summed output spike counts = the FIXED-pooling logits."""
        acc = self._readout(x_bits, settle)                        # parent: per-class summed pool counts (the fixed pooling)
        return np.asarray(acc, float)

    def _pooled_logits_pert(self, x_bits, settle, xi_hid, xi_out):
        """settle with per-region pre-activation perturbations (node perturbation) -> fixed-pool logits."""
        self._reset_membrane(); self._set_apical(None); self._set_input_drive(x_bits)
        drive = np.asarray(to_host(self._drive_dev)).astype(np.float32).copy()
        if xi_hid is not None:
            drive[self.idx_hid] += xi_hid.astype(np.float32)
        if xi_out is not None:
            drive[self.idx_out] += xi_out.astype(np.float32)
        dev = from_host(drive)
        acc = np.zeros(len(self.idx_out))
        for _ in range(settle):
            self.sb.cp_external_input_current[:] = dev
            self.sb._run_one_simulation_step()
            acc += np.asarray(to_host(self.sb.cp_firing_states[self.idx_out])).astype(float)
        return np.array([acc[c * self.pool_out:(c + 1) * self.pool_out].sum() for c in range(self.n_classes)])

    def _ce(self, logits, y):
        z = np.clip(np.asarray(logits, float) * 0.5, -30, 30); z = z - z.max()
        e = np.exp(z); p = e / e.sum()
        return float(-np.log(max(p[y], 1e-12)))

    def _rates(self, x_bits, settle, idx):
        self._reset_membrane(); self._set_apical(None); self._set_input_drive(x_bits)
        acc = np.zeros(len(idx))
        for _ in range(settle):
            self.sb.cp_external_input_current[:] = self._drive_dev
            self.sb._run_one_simulation_step()
            acc += np.asarray(to_host(self.sb.cp_firing_states[idx])).astype(float)
        return acc / max(1, settle)

    def _write_delta(self, mask, rows_idx, cols_idx, delta):
        data = np.asarray(to_host(self.sb.cp_connections.data)).astype(float)
        rpos = {v: i for i, v in enumerate(rows_idx.tolist())}
        cpos = {v: i for i, v in enumerate(cols_idx.tolist())}
        rows = self._coo_row[mask]; cols = self._coo_col[mask]
        add = np.array([delta[rpos[r], cpos[c]] for r, c in zip(rows, cols)])
        wmax = float(getattr(self.cfg, "bdsp_w_max", 200.0))
        data[mask] = np.clip(data[mask] + add, -wmax, wmax)
        self.sb.cp_connections.data[:] = from_host(data)

    def train(self, Xtr, ytr, epochs, lr, sigma, settle, k, seed, mode="np"):
        rng = np.random.RandomState(seed * 13 + 2)
        nH = len(self.idx_hid); nO = len(self.idx_out)
        for _ep in range(epochs):
            order = rng.permutation(len(Xtr)); perm = rng.permutation(len(Xtr))
            for t, j in enumerate(order):
                x, y = Xtr[j], int(ytr[j])
                hid_rate = self._rates(x, settle, self.idx_hid)
                in_rate = self._rates(x, settle, self.idx_in)
                gW = np.zeros((len(self.idx_in), nH)); gH = np.zeros((nH, nO))
                for _r in range(k):
                    xi_h = sigma * rng.standard_normal(nH); xi_o = sigma * rng.standard_normal(nO)
                    Lp = self._ce(self._pooled_logits_pert(x, settle, xi_h, xi_o), y)
                    Lm = self._ce(self._pooled_logits_pert(x, settle, -xi_h, -xi_o), y)
                    dL = 0.5 * (Lp - Lm)
                    if mode == "shuffle":
                        jk = perm[t]; xk, yk = Xtr[jk], int(ytr[jk])
                        dL = 0.5 * (self._ce(self._pooled_logits_pert(xk, settle, xi_h, xi_o), yk)
                                    - self._ce(self._pooled_logits_pert(xk, settle, -xi_h, -xi_o), yk))
                    coef = dL / (sigma * sigma)
                    gW += coef * np.outer(in_rate, xi_h)           # input->hidden credit (node xi_h x presynaptic in_rate)
                    gH += coef * np.outer(hid_rate, xi_o)          # hidden->output credit (node xi_o x presynaptic hid_rate)
                if mode != "frozen":
                    self._write_delta(self.mask_in2hid, self.idx_in, self.idx_hid, -lr * gW / k)
                    self._write_delta(self.mask_hid2out, self.idx_hid, self.idx_out, -lr * gH / k)

    def accuracy(self, X, y, settle):
        return float(np.mean([int(np.argmax(self._pooled_logits(X[i], settle)) == int(y[i])) for i in range(len(X))]))


def run(seed, task, epochs, lr, sigma, settle, k, hidden, pool_out, max_train, max_test, mode):
    (Xtr, ytr), (Xte, yte), n_bits = _load_task(task, seed, 24)
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    if max_train and len(Xtr) > max_train:
        s = np.random.RandomState(seed).permutation(len(Xtr))[:max_train]; Xtr, ytr = Xtr[s], ytr[s]
    if max_test and len(Xte) > max_test:
        s = np.random.RandomState(seed + 1).permutation(len(Xte))[:max_test]; Xte, yte = Xte[s], yte[s]
    net = FixedPoolNP(seed, n_bits, hidden=hidden, pool_out=pool_out, hidden_bias=0.0, output_bias=0.0,
                      fwd_wmean=90.0, fwd_wjit=2.0)
    net.train(Xtr, ytr, epochs, lr, sigma, settle, k, seed, mode=mode)
    return round(net.accuracy(Xte, yte, settle), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", default="emerge1")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=40.0)           # pre-activation perturbation current (pA)
    ap.add_argument("--settle", type=int, default=80); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--pool-out", type=int, default=12)
    ap.add_argument("--max-train", type=int, default=64); ap.add_argument("--max-test", type=int, default=64)
    a = ap.parse_args()
    for s in a.seeds:
        chance = None
        accs = {}
        for mode in ("np", "shuffle", "frozen"):
            accs[mode] = run(s, a.task, a.epochs, a.lr, a.sigma, a.settle, a.k, a.hidden, a.pool_out, a.max_train, a.max_test, mode)
        beats = accs["np"] > accs["frozen"] + 0.05
        shuf = accs["shuffle"] <= accs["frozen"] + 0.05
        print(f"[fixedpool-NP seed={s} {a.task}] NP={accs['np']:.3f} frozen={accs['frozen']:.3f} shuffle={accs['shuffle']:.3f} "
              f"-> {'GO' if (beats and shuf) else 'no'} (beats_frozen={beats} shuffle_collapses={shuf})", flush=True)


if __name__ == "__main__":
    main()
