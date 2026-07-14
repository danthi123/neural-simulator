"""DECISIVE de-risk: does GRADED clean-error credit (enable_bdsp_graded_credit) TRAIN the depth-2 compositional-
inheritance task ON SPIKES, where the default BINARY-burst credit does NOT (0/6, the 2026-07-07 does-not-train wall)?

THE RESEARCH-GATE #1 SURPASS (2026-07-14 deep-research gate + adversarial verify). The on-bridge spiking deep-credit
wall was precisely diagnosed as a SINGLE-NEURON readout-VARIANCE wall: the postsynaptic credit factor is read as the
MEASURED burst rate cp_bdsp_B -- a finite-spike stochastic SAMPLE with CV~1.0-1.4 at the operating event rate -- so the
descending credit is noise-dominated and the net cannot fit the task (weights move in noise directions). The reframe:
every working deep-spiking trainer reads the credit factor from a GRADED, low-variance signal, not a 1-bit count.

The correctly-located, ALREADY-IMPLEMENTED fix (the adversarial skeptic's code-confirmed catch): the committed
`enable_bdsp_graded_credit` flag (bridge.py:7275, additive/default-off, 2026-07-12) swaps the credit factor from the
sampled burst B to the graded EXPECTATION cp_bdsp_E*cp_bdsp_P (event-rate x burst-probability) via the kernel identity
B-Pbar*E == E*(P-Pbar). This is the on-bridge M2.6 clean-error realization -- the low-variance credit that can carry
the fine per-synapse sign the sampled B cannot. It does NOT touch the presynaptic eligibility (cp_bdsp_E stays), so it
is correctly located (the skeptic refuted the mislocated "replace cp_bdsp_E with sigma'(v)" framing).

THE TEST (single variable): at the EXACT config where the BINARY-burst plain-FA net fails to train (0/3 in the K-trend:
tonic_h 560 / tonic_o 620 / apical 2000 / ff_w_init 4.5, H=24, ep=50, K=1, n_hidden_layers=2, the semantic-inheritance
task n_super=12/n_members=8/held_per_super=3/n_prop=2/member_id_dim=3/n_obs=16/noise=0.02), run TWO arms differing ONLY
in graded_credit: binary (False, the baseline) vs graded (True). 6-seed.

GO GATE: graded TRAINS (graded inherit_heldout > floor+0.03 AND > chance+0.03; ideally deep_train >> chance) where
binary does NOT (binary trains_at_all False). ANTI-CHEATS (on the graded arm, the one that should train): permuted-label
-> ~chance (no leakage); apical-lesion -> collapses to floor (the apical credit is load-bearing, not a drive artifact);
oracle 1.0 (task learnable). NO sim/ edit (reuse-by-import; the graded flag is the committed additive/default-off sim/
mechanism; the runner-side wiring is a default-False param threaded through _run_arm/OnBridgeBDSPNet)."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._semantic_inheritance_onbridge_spiking_derisk import _run_arm  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_onbridge_deep_credit_graded.json"
TRAIN_MARGIN = 0.03


def _trains(inh, floor_inh, chance):
    return bool((not np.isnan(inh)) and inh > floor_inh + TRAIN_MARGIN and inh > chance + TRAIN_MARGIN)


def ab_seed(seed, hidden, epochs, batch, settle_steps, credit_steps, lr, subsample, hp, task_kwargs,
            n_hidden_layers=2):
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]
    n_in = Xtr.shape[1]
    inh_idx = idx["inh_idx"]

    # depth gate + oracle ceiling (K/graded-independent; validates the TASK requires depth + is learnable)
    full_task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    s0 = stage0_depth_genuineness(full_task, idx, k, hidden=96, epochs=250, lr=0.3, batch=128, seed=seed)
    if len(inh_idx):
        yv = yte[inh_idx]
        chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle = {"inherit_heldout": _acc_on(onet, Xte, yte, inh_idx),
              "train": float(onet.accuracy(Xtr, ytr))}

    # subsample train for the on-bridge arms (held-out never subsampled) -- match the population runner
    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr_b, ytr_b, Ltr_b = Xtr[keep], ytr[keep], Ltr[keep]
    else:
        Xtr_b, ytr_b, Ltr_b = Xtr, ytr, Ltr
    task_b = ((Xtr_b, ytr_b, Ltr_b), (Xte, yte, Lte))

    common = dict(n_hidden_layers=n_hidden_layers, settle_steps=settle_steps, credit_steps=credit_steps,
                  lr=lr, hp=hp)

    # 1-layer floor (graded-independent; the fair shallow baseline)
    floor = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed, pool_k=1,
                     **{**common, "n_hidden_layers": 1})
    f_inh = floor["inherit_heldout"]

    # THE A/B (single variable = graded_credit) --------------------------------
    binary = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed, pool_k=1,
                      graded_credit=False, **common)
    graded = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed, pool_k=1,
                      graded_credit=True, **common)

    # anti-cheats on the GRADED arm (the one that should train if graded is the fix)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    permuted = _run_arm("plain_fa", ((Xtr_b, yperm, Ltr_b), (Xte, yte, Lte)), idx, n_in, hidden, k,
                        epochs, batch, seed, pool_k=1, graded_credit=True, mode="bdsp", **common)
    lesion = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed, pool_k=1,
                      graded_credit=True, mode="apical_lesion", **common)

    return {
        "seed": seed, "chance": chance, "meta": meta,
        "stage0_depth_separating": bool(s0.get("depth_separating")),
        "oracle": oracle, "floor_inherit_heldout": f_inh,
        "binary": {"inherit_heldout": binary["inherit_heldout"], "train": binary["train"],
                   "ff_weight_moved": binary["ff_weight_moved"],
                   "trains_at_all": _trains(binary["inherit_heldout"], f_inh, chance)},
        "graded": {"inherit_heldout": graded["inherit_heldout"], "train": graded["train"],
                   "ff_weight_moved": graded["ff_weight_moved"],
                   "trains_at_all": _trains(graded["inherit_heldout"], f_inh, chance)},
        "graded_permuted_inherit": permuted["inherit_heldout"],
        "graded_lesion_inherit": lesion["inherit_heldout"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--credit-steps", type=int, default=15)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--train-subsample", type=int, default=80)
    # favorable config (where binary K=1 fails 0/3)
    ap.add_argument("--tonic-h-pA", type=float, default=560.0)
    ap.add_argument("--tonic-o-pA", type=float, default=620.0)
    ap.add_argument("--apical-gain-pA", type=float, default=2000.0)
    ap.add_argument("--ff-w-init", type=float, default=4.5)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    # task (matches the population runner's does-not-train config)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    hp = {"tonic_h_pA": args.tonic_h_pA, "tonic_o_pA": args.tonic_o_pA,
          "apical_gain_pA": args.apical_gain_pA, "ff_w_init": args.ff_w_init, "pbar_alpha": args.pbar_alpha}
    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}

    t0 = time.time()
    try:
        r = ab_seed(args.seed, args.hidden, args.epochs, args.batch, args.settle_steps,
                    args.credit_steps, args.lr, args.train_subsample, hp, task_kwargs,
                    n_hidden_layers=args.n_hidden_layers)
    except Exception as e:
        r = {"seed": args.seed, "error": repr(e), "traceback": traceback.format_exc()}

    out = {"probe": "onbridge_deep_credit_graded_vs_binary_AB", "seed": args.seed,
           "config": {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
                      "settle_steps": args.settle_steps, "credit_steps": args.credit_steps,
                      "n_hidden_layers": args.n_hidden_layers, "train_subsample": args.train_subsample,
                      "hp": hp, "task": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "result": r}

    b = r.get("binary", {}); g = r.get("graded", {})
    if b and g:
        ch = r.get("chance", 0.333)
        go = bool(g.get("trains_at_all") and not b.get("trains_at_all")
                  and r.get("graded_permuted_inherit", 1.0) < ch + TRAIN_MARGIN
                  and r.get("graded_lesion_inherit", 1.0) < r.get("floor_inherit_heldout", 0) + 0.10)
        out["GO_graded_rescues"] = go
        out["verdict"] = (
            f"GRADED {'TRAINS' if g.get('trains_at_all') else 'fails'} "
            f"(inh {g.get('inherit_heldout'):.3f}, train {g.get('train'):.3f}) vs "
            f"BINARY {'trains' if b.get('trains_at_all') else 'fails'} "
            f"(inh {b.get('inherit_heldout'):.3f}, train {b.get('train'):.3f}); "
            f"chance {ch:.3f}, floor {r.get('floor_inherit_heldout'):.3f}, oracle "
            f"{r.get('oracle', {}).get('inherit_heldout'):.3f}; "
            f"graded permuted {r.get('graded_permuted_inherit'):.3f}, lesion {r.get('graded_lesion_inherit'):.3f}. "
            f"{'GO: graded credit RESCUES on-spike training (the readout-variance wall is surpassed by the low-variance graded credit)' if go else 'NO rescue at this config (points deeper: credit-STRUCTURE or scale, not read-variance)'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(out.get("verdict", json.dumps(r)[:300]))


if __name__ == "__main__":
    main()
