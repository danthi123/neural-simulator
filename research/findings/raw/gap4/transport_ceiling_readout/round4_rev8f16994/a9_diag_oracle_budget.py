"""A9 instrument diagnostic (dev seed 7, no gate reads it): how many ONLINE example-updates does the RATE backprop
oracle (sim.dendritic_mlp.DendriticMLP, mode='oracle', hidden 32 and 64) need before held-out inheritance clears
chance? The spiking arms get epochs x 400 single-example updates; this bounds what the transport ceiling can reach at
a given budget, independent of spikes."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from sim.dendritic_mlp import DendriticMLP
from sim.backend import to_host
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance

(Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"]); inh = np.asarray(idx["inh_idx"])
chance = float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh])))
keep = np.random.default_rng(7 * 13 + 1).permutation(len(Xtr))[:400]
rows = []
for n_train, (Xa, ya) in (("400", (Xtr[keep], ytr[keep])), ("1260", (Xtr, ytr))):
    for H in (32, 64):
        for lr in (0.05, 0.3):
            net = DendriticMLP([Xa.shape[1], H, H, k], seed=7)
            rng = np.random.default_rng(7 + 777)
            curve = {}
            done = 0
            for ep in range(1, 81):
                for i in rng.permutation(len(ya)):
                    net.train_step(Xa[i:i + 1], ya[i:i + 1], mode="oracle", lr=lr)
                done += len(ya)
                if ep in (1, 2, 5, 10, 20, 40, 80):
                    _, lg = net._forward(np.asarray(Xte[inh], float))
                    curve[done] = float(np.mean(np.argmax(np.asarray(to_host(lg)), 1) == yte[inh]))
            r = {"n_train": n_train, "hidden": H, "lr": lr, "heldout_by_updates": curve, "chance": chance}
            rows.append(r)
            print(json.dumps(r), flush=True)
json.dump({"probe": "a9_oracle_online_budget_dev_s7", "seed": 7, "rows": rows,
           "note": "rate oracle, batch 1 (online), the spiking arms' update budget is epochs x 400"},
          open("research/findings/raw/gap4/transport_ceiling_readout/diag_oracle_online_budget_s7.json", "w"), indent=2)
