"""SECONDARY read for the gap#4 depth-rescue alignment probe: the SAME credit-path alignment measurement, but on the
OBLIGATORY-DEPTH-3 + backprop-OPTIMIZABLE `hier3` task (member->MID->SUPER->property) instead of the depth-2 XOR.

WHY. On XOR (depth-2) the N>=3 arms DO NOT TRAIN (accuracy collapses to chance / majority class), so the credit-path
alignment there is measured on a net whose forward NEVER organized -> the depth trend is unreadable (a training-failure
confound, NOT a clean attenuation-vs-rescue read). Alignment is a LEARNED quantity (FA theory, Lillicrap 2016: the
forward weights adapt so W^T aligns with the fixed feedback), so it is only defined where the net actually learned. The
hier3 task genuinely REQUIRES depth-3 AND is backprop-optimizable, so a deeper net CAN enter the learning regime there
-- giving a VALID >=2-point depth trend IF the local transport-free rule trains it.

Reuses `measure_arm` from fa_kp_alignment_probe.py (task-agnostic: it takes Xtr/ytr/Xte/yte directly) + make_task_hier3
from the shared runner. Subsample mirrors run_seed (default_rng(seed+13) permutation, first `subsample`). NO runner edit.
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import sys
import time
from pathlib import Path

_REPO = "/home/dant123/Projects/sim"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import make_task_hier3  # noqa: E402
import fa_kp_alignment_probe as P  # noqa: E402  -- reuse the task-agnostic measure_arm


def run_hier3(N, seed, hidden=32, T=24, epochs=200, lr=0.05, in_gain=1.0, batch_eval=256, subsample=2000):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_hier3(seed)
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    inh = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh]))) if len(inh) else float("nan")
    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]
    # alignment is evaluated on the INHERIT held-out set (the composition-generalization rows) when present
    Xte_e, yte_e = (Xte[inh], yte[inh]) if len(inh) else (Xte, yte)
    sizes = [n_in] + [hidden] * N + [k]
    fa = P.measure_arm("chained_fa", Xtr, ytr, Xte_e, yte_e, sizes, N, T, epochs, lr, lr, in_gain, seed, batch_eval)
    kp = P.measure_arm("chained_fa_kp", Xtr, ytr, Xte_e, yte_e, sizes, N, T, epochs, lr, lr, in_gain, seed, batch_eval)
    return {"N": N, "seed": seed, "sizes": sizes, "k": k, "chance": chance, "fa": fa, "kp": kp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for N in args.n_list:
        for sd in args.seeds:
            r = run_hier3(N, sd, epochs=args.epochs, subsample=args.subsample)
            rows.append(r)
            fa, kp = r["fa"], r["kp"]
            trained_fa = fa["held_acc"] > r["chance"] + 0.03
            trained_kp = kp["held_acc"] > r["chance"] + 0.03
            print(f"[hier3 N={N} seed={sd}] chance={r['chance']:.3f} k={r['k']} | "
                  f"FA held={fa['held_acc']:.3f}(train {fa['train_acc']:.3f}, learned={trained_fa}) | "
                  f"KP held={kp['held_acc']:.3f}(train {kp['train_acc']:.3f}, learned={trained_kp})", flush=True)
            print(f"    FA align hidden(deep->top) {['%+.3f' % a for a in fa['align_hidden']]} out {fa['align_output']:+.3f}")
            print(f"    KP align hidden(deep->top) {['%+.3f' % a for a in kp['align_hidden']]} out {kp['align_output']:+.3f}",
                  flush=True)

    print("\n" + "=" * 88)
    print("hier3 DEEPEST-hidden (li=0) alignment vs depth (only valid where the arm LEARNED > chance+0.03):")
    for N in args.n_list:
        for arm, lab in (("fa", "fixed-FA"), ("kp", "KP")):
            vals, learned = [], []
            for r in rows:
                if r["N"] != N:
                    continue
                vals.append(r[arm]["align_hidden"][0])
                learned.append(r[arm]["held_acc"] > r["chance"] + 0.03)
            m = float(np.mean([v for v in vals if v == v])) if vals else float("nan")
            print(f"  N={N} {lab:>9}: deepest={m:+.3f}  learned={learned}")
    print("=" * 88)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
