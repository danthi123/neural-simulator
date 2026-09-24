"""A9 instrument diagnostic (dev seed 7, no gate reads it).
(a) How much does the H2->out feedforward pathway drive the output read? Scale those weights by s and read.
(b) Per-arm learning-rate scan (arc meta-lesson #1): frozen arm (only the readout learns), spike read S40 W30 g20,
    3 epochs on the 400-example subsample, lr in LRS; train accuracy + mean |dw| on the output pathway."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
from sim.backend import to_host

arm = sys.argv[1] if len(sys.argv) > 1 else "reservoir"
LRS = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.05, 0.5, 2.0, 8.0]
(Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"])
keep = np.random.default_rng(7 * 13 + 1).permutation(len(Xtr))[:400]
Xb, yb = Xtr[keep], ytr[keep]
inh = np.asarray(idx["inh_idx"])
KW = dict(n_hidden_layers=2, pool_k=4, credit_steps=25, graded_credit=True, eval_frozen=True, spi_silence=True,
          read_quantity="spikes", settle_steps=40, read_window=30, read_gain=20.0)
out = {"probe": "a9_lr_scan_and_ff_drive_dev_s7", "seed": 7, "arm": arm, "kw": KW}


def out_mask(net):
    coo = net.br._get_cached_coo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    s2, so = net.slices[-2], net.slices[-1]
    return (row >= s2.start) & (row < s2.stop) & (col >= so.start) & (col < so.stop)


def reads(net, X):
    with net.frozen_reads():
        return np.asarray(net._forward_batch(X)[-1])


if arm == "reservoir":
    net = Gap4ReadoutNet(Xtr.shape[1], 32, k, seed=7, feedback="reservoir", **KW)
    m = out_mask(net)
    w0 = np.asarray(to_host(net.br.cp_connections.data)).copy()
    scan = {}
    for s in [0.0, 1.0, 3.0, 10.0]:
        w = w0.copy(); w[m] = w0[m] * s
        net.br.cp_connections.data[...] = net._xp.asarray(w.astype(w0.dtype))
        R = reads(net, Xb[:40])
        scan[str(s)] = {"mean_out_read": float(R.mean()), "between_example_sd": float(R.std(0).mean()),
                        "train_acc_40": float(np.mean(np.argmax(R, 1) == yb[:40]))}
        print("ff-scale", s, scan[str(s)], flush=True)
    net.br.cp_connections.data[...] = net._xp.asarray(w0)
    out["ff_out_scale"] = scan

rows = {}
for lr in LRS:
    t0 = time.time()
    net = Gap4ReadoutNet(Xtr.shape[1], 32, k, seed=7, feedback=arm, lr=lr, **KW)
    m = out_mask(net)
    w0 = np.asarray(to_host(net.br.cp_connections.data)).copy()
    rng = np.random.default_rng(7 + 777)
    accs = []
    for ep in range(3):
        for i in rng.permutation(len(Xb)):
            net._train_one(Xb[i], int(yb[i]), "bdsp")
        R = reads(net, Xb)
        accs.append(float(np.mean(np.argmax(R, 1) == yb)))
    w1 = np.asarray(to_host(net.br.cp_connections.data))
    Rte = reads(net, Xte[inh])
    rows[str(lr)] = {"train_acc_by_epoch": accs, "heldout": float(np.mean(np.argmax(Rte, 1) == yte[inh])),
                     "mean_abs_dw_out": float(np.abs(w1[m] - w0[m]).mean()), "mean_abs_w_out": float(np.abs(w0[m]).mean()),
                     "mean_abs_dw_other": float(np.abs(w1[~m] - w0[~m]).mean()),
                     "pred_hist": np.bincount(np.argmax(R, 1), minlength=k).tolist(), "sec": round(time.time() - t0)}
    print("lr", lr, rows[str(lr)], flush=True)
out["lr_scan"] = rows
json.dump(out, open(f"research/findings/raw/gap4/transport_ceiling_readout/diag_lr_scan_{arm}_s7.json", "w"), indent=2)
