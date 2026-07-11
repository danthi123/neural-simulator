"""Decisive learning-substrate diagnostic: can the committed on-bridge BDSP rule LEARN at all?

The full-data sweep showed TEST at/below chance for both couple-on (surpass) and couple-off (boundary). This isolates
the load-bearing question: with the apical->soma coupling ON (directed credit confirmed by the B_rises probe), can the
rule reduce TRAINING error -- i.e. OVERFIT a small train set? A learning rule that cannot overfit 64 samples with
hidden=60 genuinely is not learning on-bridge (the honest learning-substrate negative). Couple-off is the control.

Run one arm:  SIM_BACKEND=numpy python -m research.runners._d1_reconcile_diag --couple 1 --soma-g 80 --n-train 64 --epochs 80
"""
import argparse
import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet, _numpy_oracle_heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--couple", type=int, default=0)
    ap.add_argument("--soma-g", type=float, default=80.0)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle", type=int, default=50)
    ap.add_argument("--output-bias", type=float, default=20.0)
    a = ap.parse_args()

    (Xtr, ytr), (Xte, yte), n_bits = _load_task("emerge1", a.seed, 4)
    rng = np.random.default_rng(a.seed + 7)
    sel = rng.permutation(len(Xtr))[: a.n_train]
    Xs, ys = np.asarray(Xtr)[sel], np.asarray(ytr)[sel]
    chance = float(max(np.mean(ys == 0), np.mean(ys == 1)))
    # can a 2-layer numpy net overfit THIS small set? (ceiling that the rule should approach if it learns)
    oc = _numpy_oracle_heldout(n_bits, 32, Xs, ys, Xs, ys, epochs=400, lr=0.3, batch=8, seed=a.seed)
    tag = f"couple={a.couple} soma_g={a.soma_g if a.couple else 0}"
    print(f"[overfit] {tag}  n_train={len(Xs)} chance={chance:.3f} numpy-2layer-overfit-ceiling={oc:.3f}", flush=True)

    net = OnBridgeBDSPNet(seed=a.seed, n_bits=n_bits, hidden=60, couple_soma=bool(a.couple), soma_g=a.soma_g,
                          hidden_bias=a.output_bias, output_bias=a.output_bias, bdsp_lr=0.03, fwd_wmean=40.0, bdsp_w_max=200.0)
    diag = net.apical_coupling_diag(steps=200)
    tag = f"{tag} obias={a.output_bias}"
    print(f"[overfit] {tag}  apical B_rises={diag['B_rises']} (directed credit requires True): "
          f"B_rest={diag['B_rest']:.3f} B_apical={diag['B_apical']:.3f} (want B_apical>=~0.3, B_rest~0)", flush=True)

    done = 0
    for milestone in (10, 20, 40, a.epochs):
        while done < milestone:
            net.train_epoch(Xs, ys, "bdsp", settle_steps=a.settle, teach_steps=a.settle, shuffle_seed=a.seed + done)
            done += 1
        tr = float(net.accuracy(Xs, ys, settle_steps=a.settle))
        print(f"[overfit] {tag}  ep={done}: TRAIN-fit={tr:.3f} (chance {chance:.3f}, ceiling {oc:.3f})  "
              f"{'<<LEARNS>>' if tr > chance + 0.10 else '<<flat = not learning>>'}", flush=True)


if __name__ == "__main__":
    main()
