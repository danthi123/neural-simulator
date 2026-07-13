"""Operating-point probe: find an on-bridge config where the OUTPUT layer TRAINS above chance (the prerequisite for
testing NP's hidden credit). The tension: fire ENOUGH for a clean spike-count readout (more settle / moderate bias) yet
stay INPUT-DISCRIMINATIVE (strong forward). For each op-point: short output-only delta train -> held-out acc. The
op-point where output-only-lift > chance is the working substrate. (Same bottleneck that underlay D1's on-bridge BDSP.)"""
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
ntr = min(80, len(Xtr)); nte = min(60, len(Xte))
chance = float(np.bincount(yte, minlength=2).max() / len(yte))
rng = np.random.default_rng(0)

configs = [
    dict(hidden_bias=0.0,  output_bias=0.0,  fwd_wmean=90.0, settle=160, lr=0.2),
    dict(hidden_bias=80.0, output_bias=80.0, fwd_wmean=90.0, settle=120, lr=0.2),
    dict(hidden_bias=120.0,output_bias=120.0,fwd_wmean=120.0,settle=120, lr=0.15),
    dict(hidden_bias=60.0, output_bias=40.0, fwd_wmean=120.0,settle=200, lr=0.15),
]
for c in configs:
    settle = c.pop("settle"); lr = c.pop("lr")
    net = OnBridgeNPNet(seed, n_bits, hidden=12, pool_out=6, fwd_wjit=2.0, **c)
    rates = net.region_rates(Xtr[0], settle)
    # short OUTPUT-ONLY delta train (hidden frozen)
    for ep in range(10):
        idx = rng.permutation(ntr)
        for j in idx:
            x, yj = Xtr[j], int(ytr[j])
            _, p = _ce(net._readout(x, settle), yj)
            e = np.zeros(2); e[yj] += 1.0; e -= p
            net._apply_output_delta(e, x, settle, lr)
    acc = net.accuracy(Xte[:nte], yte[:nte], settle)
    print(f"cfg={c} settle={settle} lr={lr} | rates in/hid/out="
          f"{rates['input']:.3f}/{rates['hidden']:.3f}/{rates['output']:.3f} | output-only acc={acc:.3f} (chance {chance:.3f})"
          f" {'LIFTS' if acc > chance + 0.06 else '--'}", flush=True)
    del net
