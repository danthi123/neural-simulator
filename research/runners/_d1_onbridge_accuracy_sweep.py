"""One config of the on-bridge BDSP learning-to-accuracy sweep: boundary (couple off) vs surpass (couple on + sparse
regime) vs the numpy oracle, on a NON-degenerate task. Gate: surpass held-out > chance AND > boundary; wrong-sign
anti-learns; apical-lesion collapses (the moat is clean in the sparse regime). Fan configs across cores; aggregate JSON.

Run one config:  SIM_BACKEND=numpy python -m research.runners._d1_onbridge_accuracy_sweep \
                    --task emerge1 --soma-g 80 --bias 350 --epochs 30 --json <out.json>
"""
import argparse, json
import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import (
    _load_task, OnBridgeBDSPNet, _numpy_oracle_heldout, _numpy_singlelayer_floor)


def _train_eval(couple, g, bias, task_data, n_bits, epochs, lr, settle, teach, mode="bdsp", seed=42):
    (Xtr, ytr), (Xte, yte) = task_data
    net = OnBridgeBDSPNet(seed=seed, n_bits=n_bits, hidden=12, couple_soma=couple, soma_g=g,
                          hidden_bias=bias, output_bias=bias, bdsp_lr=lr)
    diag = net.apical_coupling_diag(steps=250)
    for ep in range(epochs):
        net.train_epoch(Xtr, ytr, mode, settle_steps=settle, teach_steps=teach, shuffle_seed=seed + ep)
    return float(net.accuracy(Xte, yte, settle_steps=settle)), bool(diag["B_rises"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="emerge1"); ap.add_argument("--pbits", type=int, default=4)
    ap.add_argument("--soma-g", type=float, default=80.0); ap.add_argument("--bias", type=float, default=350.0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--settle", type=int, default=60); ap.add_argument("--teach", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--json", default=None)
    a = ap.parse_args()

    (Xtr, ytr), (Xte, yte), n_bits = _load_task(a.task, a.seed, a.pbits)
    chance = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    oracle = _numpy_oracle_heldout(n_bits, 12, Xtr, ytr, Xte, yte, epochs=300, lr=0.3, batch=8, seed=a.seed)
    floor = _numpy_singlelayer_floor(n_bits, Xtr, ytr, Xte, yte, epochs=300, lr=0.3, batch=8, seed=a.seed)
    td = ((Xtr, ytr), (Xte, yte))
    print(f"[sweep] task={a.task} n_bits={n_bits} n_tr={len(Xtr)} n_te={len(Xte)} chance={chance:.3f} "
          f"oracle={oracle:.3f} single-layer-floor={floor:.3f}  soma_g={a.soma_g} bias={a.bias} epochs={a.epochs}", flush=True)

    task_valid = oracle >= 0.80 and floor <= chance + 0.12
    kw = dict(g=a.soma_g, bias=a.bias, task_data=td, n_bits=n_bits, epochs=a.epochs,
              lr=a.lr, settle=a.settle, teach=a.teach, seed=a.seed)
    bnd, bnd_br = _train_eval(False, **kw)
    surp, surp_br = _train_eval(True, **kw)
    wrong, _ = _train_eval(True, **{**kw, "mode": "wrong_sign"})
    lesion, _ = _train_eval(True, **{**kw, "mode": "lesion"})

    go = bool(task_valid and (surp > chance + 0.05) and (surp > bnd + 0.05)
              and (wrong < chance) and (lesion <= chance + 0.05) and surp_br and (not bnd_br))
    row = {"task": a.task, "n_bits": n_bits, "soma_g": a.soma_g, "bias": a.bias, "epochs": a.epochs,
           "chance": round(chance, 3), "oracle": round(oracle, 3), "floor": round(floor, 3), "task_valid": task_valid,
           "boundary_heldout": round(bnd, 3), "boundary_B_rises": bnd_br,
           "surpass_heldout": round(surp, 3), "surpass_B_rises": surp_br,
           "wrong_sign_heldout": round(wrong, 3), "apical_lesion_heldout": round(lesion, 3), "GO": go}
    print(f"  boundary={row['boundary_heldout']} (B_rises {bnd_br}) | surpass={row['surpass_heldout']} (B_rises {surp_br}) "
          f"| wrong-sign={row['wrong_sign_heldout']} | apical-lesion={row['apical_lesion_heldout']} | chance={row['chance']} oracle={row['oracle']}", flush=True)
    print(f"  GATE: {'GO' if go else 'NOT-YET'} -- surpass clears chance+beats boundary while wrong-sign anti-learns + apical-lesion collapses + coupling B_rises only when on", flush=True)
    if a.json:
        json.dump(row, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
