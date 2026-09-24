"""A9 instrument diagnostic (dev seed 7, no gate reads it): does the untrained spiking forward pass TRANSMIT the
input? For each operating point (feedforward weight scale ff_w_init, synaptic propagation_strength, tonic background
scale) build the frozen net, read 90 training examples (spike read, S40 W30, plasticity off), and report per layer:
mean rate, input-dependence (between-example SD of the pooled read / its mean), and a ridge decode of the class from
each layer's read (train-on-60 / test-on-30, instrument only). Also the ff-scale check at the output (x0 vs x1)."""
import itertools, json, sys
import numpy as np
sys.path.insert(0, ".")
from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet, _ridge_decode
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
from sim.backend import to_host

(Xtr, ytr, _), _te, meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"])
sel = np.random.default_rng(3).permutation(len(Xtr))[:90]
X, y = Xtr[sel], ytr[sel]
FF = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4.0, 12.0, 40.0]
PS = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.05, 0.2]
TS = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.0, 0.5, 0.25]
tag = sys.argv[4] if len(sys.argv) > 4 else "a"
STP = [bool(int(v)) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [True]
rows = []
for ff, ps, ts, stp in itertools.product(FF, PS, TS, STP):
    net = Gap4ReadoutNet(X.shape[1], 32, k, seed=7, feedback="reservoir", n_hidden_layers=2, pool_k=4,
                         settle_steps=40, credit_steps=25, graded_credit=True, read_quantity="spikes", read_window=30,
                         eval_frozen=True, ff_w_init=ff, tonic_h_pA=450.0 * ts, tonic_o_pA=500.0 * ts)
    net.cfg.propagation_strength = ps
    net.cfg.enable_short_term_plasticity = stp
    net.cfg.bdsp_learning_rate = 0.0
    net.cfg.enable_structural_plasticity = False
    acts = net._forward_batch(X)
    r = {"ff_w_init": ff, "propagation_strength": ps, "tonic_scale": ts, "stp": stp}
    for li, name in enumerate(["in", "h1", "h2", "out"]):
        A = np.asarray(acts[li])
        r[f"{name}_rate"] = float(A.mean())
        r[f"{name}_input_dep"] = float(A.std(0).mean() / (A.mean() + 1e-9))
        if name != "in":
            r[f"{name}_decode"] = _ridge_decode(A[:60], y[:60], A[60:], y[60:], k, 1.0)[1]
    r["out_argmax_acc"] = float(np.mean(np.argmax(np.asarray(acts[-1]), 1) == y))
    rows.append(r)
    print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}), flush=True)
json.dump({"probe": "a9_forward_transmission_scan_dev_s7", "seed": 7, "rows": rows,
           "note": "dev diagnostic; decode = host ridge instrument; no pathway"},
          open(f"research/findings/raw/gap4/transport_ceiling_readout/diag_transmit_scan_{tag}_s7.json", "w"), indent=2)
