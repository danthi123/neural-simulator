"""Fast forward-path sweep: does input reach output (readout DIFFERS by input) under lower bias + stronger forward
weights? Pre-train only (2 readouts per config) -> instant. The prerequisite for any on-bridge credit rule."""
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
from research.runners._nodepert_onbridge_derisk import OnBridgeNPNet

seed = 42
(Xtr, ytr), _, n_bits = _load_task("emerge1", seed, 24)
Xtr = np.asarray(Xtr, float)
settle = 40
# several distinct-label input PAIRS to check input->output discrimination broadly
pairs = []
for _ in range(4):
    i0 = int(np.random.RandomState(len(pairs)+1).choice(np.where(ytr == 0)[0]))
    i1 = int(np.random.RandomState(len(pairs)+7).choice(np.where(ytr == 1)[0]))
    pairs.append((i0, i1))

configs = [
    dict(hidden_bias=520, output_bias=520, fwd_wmean=6.0),    # the D1 default (bias-dominated baseline)
    dict(hidden_bias=120, output_bias=60,  fwd_wmean=30.0),
    dict(hidden_bias=60,  output_bias=0,   fwd_wmean=60.0),
    dict(hidden_bias=0,   output_bias=0,   fwd_wmean=90.0),
    dict(hidden_bias=40,  output_bias=20,  fwd_wmean=45.0, in_hi=1200.0),
]
for cfg in configs:
    net = OnBridgeNPNet(seed, n_bits, hidden=12, pool_out=6, fwd_wjit=2.0, **cfg)
    diffs = []; rates = net.region_rates(Xtr[pairs[0][0]], settle)
    for (i0, i1) in pairs:
        r0 = net._readout(Xtr[i0], settle); r1 = net._readout(Xtr[i1], settle)
        diffs.append(float(np.abs(r0 - r1).sum()))
    r0 = net._readout(Xtr[pairs[0][0]], settle)
    print(f"cfg={cfg} | rates in/hid/out={rates['input']:.3f}/{rates['hidden']:.3f}/{rates['output']:.3f} "
          f"| mean|r0-r1|={np.mean(diffs):.1f} sample_r0={np.round(r0,1)}", flush=True)
    del net
