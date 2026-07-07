"""Control probe for the D1 microcircuit surpass GO (adversarial-verify lens 2, mechanism genuineness).

CLAIM under test: the microcircuit clears the depth-2 held-out bar (0.96) where raw Burstprop is NOISE-LIMITED
(~0.79), because Burstprop must estimate the credit from a NOISY per-unit burst fraction while the microcircuit
carries the SAME feedback-alignment error CLEANLY (the interneuron-cancelled apical = e_k = phi'*(Y^T @ e_{k+1}),
a low-variance weighted average).

THE DECISIVE CONTROL: if the Burstprop gap is genuinely finite-sample ESTIMATE noise, then AVERAGING MORE SAMPLES
per update (a BIGGER BATCH) should climb Burstprop's held-out toward the microcircuit's ~0.96. If bigger-batch
Burstprop closes the gap -> "noise-limited, closable by averaging" is CONFIRMED (and the microcircuit's clean error
just does that averaging analytically via Y^T@e). If it plateaus far below -> Burstprop has a residual limit beyond
finite-sample noise (a different, stronger claim). Either way the finding's framing is empirically pinned, NOT asserted.

Reuse-by-import of the D1 runner classes (NO edit to that runner). numpy/CPU smoke.
Run per-seed (fan across cores):  python -m research.runners._gnw_d1_microcircuit_control_probe --seed 42 --json raw/_ctl_42.json
"""
import argparse
import json
import numpy as np

from research.runners._gnw_d1_spiking_bdsp_derisk import (
    BDSPNet, MicrocircuitBDSPNet, make_task, _train, N_BITS,
)


def _heldout(Net, sizes, Xtr, ytr, Xte, yte, seed, epochs, lr, batch):
    net = Net(sizes, seed=seed, beta=1.0, p0=0.30)
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    return float(net.accuracy(Xte, yte)), float(net.accuracy(Xtr, ytr))


def run(seed, epochs=600, lr=0.3, hidden=128):
    (Xtr, ytr, _), (Xte, yte, _) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]
    n_tr = len(Xtr)
    out = {"seed": seed, "epochs": epochs, "lr": lr, "hidden": hidden, "n_train": int(n_tr)}

    # Burstprop across batch sizes: does more per-update averaging climb toward the microcircuit?
    bp = {}
    for batch in [32, 128, 512, n_tr]:
        te, tr = _heldout(BDSPNet, deep, Xtr, ytr, Xte, yte, seed, epochs, lr, min(batch, n_tr))
        bp[str(min(batch, n_tr))] = {"heldout": te, "train": tr}
    out["burstprop_by_batch"] = bp

    # Microcircuit across the SAME batch sweep -- is it BATCH-ROBUST (the genuine advantage) where Burstprop is fragile?
    mc = {}
    for batch in [32, 128, 512, n_tr]:
        te, tr = _heldout(MicrocircuitBDSPNet, deep, Xtr, ytr, Xte, yte, seed, epochs, lr, min(batch, n_tr))
        mc[str(min(batch, n_tr))] = {"heldout": te, "train": tr}
    out["microcircuit_by_batch"] = mc
    out["microcircuit_batch32"] = mc["32"]  # back-compat

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    r = run(a.seed, a.epochs, a.lr, a.hidden)
    bp = r["burstprop_by_batch"]
    print(f"[seed {a.seed}] microcircuit(b32)={r['microcircuit_batch32']['heldout']:.3f}  "
          f"burstprop b32={bp['32']['heldout']:.3f} "
          f"b128={bp['128']['heldout']:.3f} "
          f"b512={bp['512']['heldout']:.3f} "
          f"bFULL({r['n_train']})={bp[str(r['n_train'])]['heldout']:.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
