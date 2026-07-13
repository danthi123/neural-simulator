"""Root-cause diagnostic for the on-bridge NP cold-start: is the spiking readout DISCRIMINATIVE (does input reach
output), or degenerate (flat -> argmax fixed -> exactly chance)? Print readout vectors for 2 distinct inputs + region
firing rates, before and after a short output-only delta train."""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task
from research.runners._nodepert_onbridge_derisk import OnBridgeNPNet, _ce

seed = 42
(Xtr, ytr), (Xte, yte), n_bits = _load_task("emerge1", seed, 24)
Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
settle = 80
# WORKING forward config (from _np_onbridge_fwdsweep): low bias + strong forward => input reaches output
net = OnBridgeNPNet(seed, n_bits, hidden=12, pool_out=6, hidden_bias=0.0, output_bias=0.0,
                    fwd_wmean=90.0, fwd_wjit=2.0)

# two inputs with DIFFERENT labels
i0 = int(np.where(ytr == 0)[0][0]); i1 = int(np.where(ytr == 1)[0][0])
print(f"n_bits={n_bits} n_classes={net.n_classes} idx_in={len(net.idx_in)} idx_hid={len(net.idx_hid)} idx_out={len(net.idx_out)}")
print("region rates x0:", {k: round(v, 3) for k, v in net.region_rates(Xtr[i0], settle).items()})
print("region rates x1:", {k: round(v, 3) for k, v in net.region_rates(Xtr[i1], settle).items()})
r0 = net._readout(Xtr[i0], settle); r1 = net._readout(Xtr[i1], settle)
print(f"PRE-train readout x0(y={ytr[i0]}): {np.round(r0,1)}  argmax={int(np.argmax(r0))}")
print(f"PRE-train readout x1(y={ytr[i1]}): {np.round(r1,1)}  argmax={int(np.argmax(r1))}")

# perturbation sanity: does a hidden xi measurably change the readout?
rng = np.random.default_rng(1)
xi = rng.standard_normal(len(net.idx_hid)) * 40.0
rp = net._readout_perturbed(Xtr[i0], settle, xi); rm = net._readout_perturbed(Xtr[i0], settle, -xi)
print(f"perturb +xi readout: {np.round(rp,1)}  -xi: {np.round(rm,1)}  |dL|={abs(_ce(rp,int(ytr[i0]))[0]-_ce(rm,int(ytr[i0]))[0]):.4f}")

# short OUTPUT-ONLY delta train (hidden frozen) -> does the readout become discriminative?
for ep in range(15):
    idx = rng.permutation(len(Xtr))
    for j in idx:
        x, yj = Xtr[j], int(ytr[j])
        _, p = _ce(net._readout(x, settle), yj)
        e = np.zeros(net.n_classes); e[yj] += 1.0; e -= p
        net._apply_output_delta(e, x, settle, 0.5)
r0 = net._readout(Xtr[i0], settle); r1 = net._readout(Xtr[i1], settle)
print(f"POST output-only readout x0(y={ytr[i0]}): {np.round(r0,1)}  argmax={int(np.argmax(r0))}")
print(f"POST output-only readout x1(y={ytr[i1]}): {np.round(r1,1)}  argmax={int(np.argmax(r1))}")
acc = net.accuracy(Xte[:60], yte[:60], settle)
print(f"POST output-only held-out acc(60) = {acc:.3f}  (chance {float(np.bincount(yte,minlength=net.n_classes).max()/len(yte)):.3f})")
