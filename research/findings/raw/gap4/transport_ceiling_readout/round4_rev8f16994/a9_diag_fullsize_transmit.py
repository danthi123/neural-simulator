"""A9 instrument diagnostic at the 2026-09-15 FULL size (H64 / pool_k 16, legacy config, dev seed 7; no gate reads it):
does the feedforward pathway change the reads at all? Zero the input->H1 and H2->out weights in turn and compare the
pooled H1 / output reads over 24 training examples (legacy event read, plasticity frozen for the reads). Then the same
with short-term depression bypassed on the feedforward synapses."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
from sim.backend import to_host

(Xtr, ytr, _), _te, meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"])
X = Xtr[:24]
out = {"probe": "a9_fullsize_ff_transmission_dev_s7", "seed": 7, "net": "hidden 64, pool_k 16 (the 2026-09-15 net)",
       "variants": {}}
for label, kw in (("legacy_stp_on", {}),
                  ("ff_stp_bypassed", dict(no_ff_stp=True, no_structural=True))):
    t0 = time.time()
    net = Gap4ReadoutNet(X.shape[1], 64, k, seed=7, feedback="reservoir", n_hidden_layers=2, pool_k=16,
                         settle_steps=40, credit_steps=25, graded_credit=True, eval_frozen=True, **kw)
    coo = net.br._get_cached_coo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    w0 = np.asarray(to_host(net.br.cp_connections.data)).copy()

    def mask(li):
        pre, post = net._ff_edges[li]
        return (row >= pre[0]) & (row <= pre[-1]) & (col >= post[0]) & (col <= post[-1])
    res = {}
    for tag, zero in (("intact", None), ("in_to_h1_zeroed", 0), ("h2_to_out_zeroed", 2)):
        w = w0.copy()
        if zero is not None:
            w[mask(zero)] = 0.0
        net.br.cp_connections.data[...] = net._xp.asarray(w)
        with net.frozen_reads():
            acts = net._forward_batch(X)
        res[tag] = {"h1_mean": float(np.mean(acts[1])), "out_mean": float(np.mean(acts[-1])),
                    "h1": np.asarray(acts[1]).tolist(), "out": np.asarray(acts[-1]).tolist()}
    net.br.cp_connections.data[...] = net._xp.asarray(w0)
    d_h1 = float(np.abs(np.asarray(res["intact"]["h1"]) - np.asarray(res["in_to_h1_zeroed"]["h1"])).mean())
    d_out = float(np.abs(np.asarray(res["intact"]["out"]) - np.asarray(res["h2_to_out_zeroed"]["out"])).mean())
    summ = {"h1_mean_intact": res["intact"]["h1_mean"], "h1_mean_input_zeroed": res["in_to_h1_zeroed"]["h1_mean"],
            "mean_abs_h1_change_when_input_weights_zeroed": d_h1,
            "out_mean_intact": res["intact"]["out_mean"], "out_mean_h2out_zeroed": res["h2_to_out_zeroed"]["out_mean"],
            "mean_abs_out_change_when_h2out_weights_zeroed": d_out, "seconds": round(time.time() - t0)}
    out["variants"][label] = {"summary": summ}
    print(label, json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in summ.items()}), flush=True)
json.dump(out, open("research/findings/raw/gap4/transport_ceiling_readout/diag_fullsize_ff_transmission_s7.json", "w"),
          indent=2)
