"""A9 instrument diagnostic (dev seed 7, no gate reads it): does ONE credit presentation move the output read the
right way? Frozen arm (only the output layer learns), spike read. For each variant: take 6 training examples; for each,
read the output (spike read, frozen reads), present it for credit 3 times, read again. Report the change of
(true-class read - mean other-class read). Variants test the Pbar aliasing (isi, pbar_alpha 0), the rule's step size
(lr) and the settle/window."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance

(Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"])
VARIANTS = {
    "base_spikes_S40W30_g20": dict(settle_steps=40, read_window=30, read_gain=20.0),
    "isi60": dict(settle_steps=40, read_window=30, read_gain=20.0, isi_steps=60),
    "pbar0": dict(settle_steps=40, read_window=30, read_gain=20.0, pbar_alpha=0.0),
    "pbar0_isi60": dict(settle_steps=40, read_window=30, read_gain=20.0, pbar_alpha=0.0, isi_steps=60),
    "lr0.25": dict(settle_steps=40, read_window=30, read_gain=20.0, lr=0.25),
    "event_read_S40W30": dict(settle_steps=40, read_window=30, read_gain=20.0, read_quantity="event"),
}
out = {"probe": "a9_onestep_output_learning_dev_s7", "seed": 7, "variants": {}}
for name, kw in VARIANTS.items():
    base = dict(n_hidden_layers=2, pool_k=4, credit_steps=25, graded_credit=True, eval_frozen=True,
                spi_silence=True, read_quantity="spikes")
    base.update(kw)
    net = Gap4ReadoutNet(Xtr.shape[1], 32, k, seed=7, feedback="reservoir", **base)
    deltas = []; rows = []
    for i in range(6):
        x, y = Xtr[i], int(ytr[i])
        def margin():
            with net.frozen_reads():
                r = net._forward_spiking(x)[-1]
            return float(r[y] - np.mean(np.delete(r, y))), r.tolist()
        m0, r0 = margin()
        traj = [m0]
        for rep in range(3):
            net._train_one(x, y, "bdsp")
            traj.append(margin()[0])
        deltas.append(traj[-1] - traj[0]); rows.append({"i": i, "y": y, "margin_traj": traj})
    res = {"mean_margin_change_after_3_presentations": float(np.mean(deltas)),
           "n_positive": int(np.sum(np.asarray(deltas) > 0)), "n": len(deltas), "rows": rows, "kw": kw}
    out["variants"][name] = res
    print(name, round(res["mean_margin_change_after_3_presentations"], 5), f"{res['n_positive']}/6 positive",
          [ [round(v, 4) for v in r["margin_traj"]] for r in rows[:3]], flush=True)
json.dump(out, open("research/findings/raw/gap4/transport_ceiling_readout/diag_onestep_output_learning_s7.json", "w"),
          indent=2)
