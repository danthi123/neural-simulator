"""CREDIT-STRUCTURE de-risk: does DECOLLE (per-layer LOCAL fixed-random credit) TRAIN the depth-2 compositional-
inheritance task ON SPIKES, where the multi-hop cross-layer feedback-alignment (FA) chain does NOT (0/6, the
does-not-train wall), and where reducing the read variance (graded credit) also does not rescue?

WHY (the credit-STRUCTURE reframe, 2026-07-14). The graded-vs-binary A/B REFUTED read-variance: lowering the credit-
factor variance (graded E*P vs sampled B) leaves training at chance, so the wall is the CREDIT DIRECTION / STRUCTURE of
the multi-hop FA descent at depth, not the readout noise. The research gate's #3 mechanism for a credit-STRUCTURE wall
(adversarially sharpened): DECOLLE (Kaiser-Mostafa-Neftci 2020, Deep Continuous Local Learning) DELETES the fragile
multi-hop cross-layer FA chain entirely -- each hidden layer trains from its OWN fixed-random LOCAL readout + local
target (no e_upper descent through the deep chain). If per-layer local credit TRAINS where the deep FA chain fails,
the wall IS the depth of the FA credit chain (and DECOLLE surpasses it); if it also fails, the wall is deeper still.

THE TEST (single variable = credit STRUCTURE, IDENTICAL read/config, the skeptic's sharpened form): at the exact config
where the deep plain-FA net fails (tonic_h 560/tonic_o 620/apical 2000/ff_w_init 4.5, H=24, ep=50, K=1, n_hidden=2, the
semantic-inheritance task), run TWO arms differing ONLY in the credit STRUCTURE: plain_fa (deep multi-hop FA descent,
the does-not-train baseline) vs decolle (per-layer local). 6-seed. GO GATE: decolle TRAINS (inherit > floor+0.03 AND >
chance+0.03) where plain_fa does NOT; permuted -> ~chance (no leakage); apical-lesion -> collapse (local credit load-
bearing); oracle 1.0 (task learnable). NO sim/ edit (the decolle credit is host-computed per the runner's existing
credit-injection design, exactly like plain_fa/burstprop; the committed enable_bdsp kernel moves the FF weights)."""
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

OUT = _REPO / "research" / "findings" / "raw" / "_onbridge_deep_credit_decolle.json"
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

    full_task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    s0 = stage0_depth_genuineness(full_task, idx, k, hidden=96, epochs=250, lr=0.3, batch=128, seed=seed)
    if len(inh_idx):
        yv = yte[inh_idx]
        chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle = {"inherit_heldout": _acc_on(onet, Xte, yte, inh_idx), "train": float(onet.accuracy(Xtr, ytr))}

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr_b, ytr_b, Ltr_b = Xtr[keep], ytr[keep], Ltr[keep]
    else:
        Xtr_b, ytr_b, Ltr_b = Xtr, ytr, Ltr
    task_b = ((Xtr_b, ytr_b, Ltr_b), (Xte, yte, Lte))

    common = dict(n_hidden_layers=n_hidden_layers, settle_steps=settle_steps, credit_steps=credit_steps,
                  lr=lr, hp=hp, pool_k=1)

    floor = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed,
                     **{**common, "n_hidden_layers": 1})
    f_inh = floor["inherit_heldout"]

    # THE A/B (single variable = credit STRUCTURE) ---------------------------
    deepfa = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed, **common)
    decolle = _run_arm("decolle", task_b, idx, n_in, hidden, k, epochs, batch, seed, **common)

    # anti-cheats on the DECOLLE arm
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    permuted = _run_arm("decolle", ((Xtr_b, yperm, Ltr_b), (Xte, yte, Lte)), idx, n_in, hidden, k,
                        epochs, batch, seed, mode="bdsp", **common)
    lesion = _run_arm("decolle", task_b, idx, n_in, hidden, k, epochs, batch, seed,
                      mode="apical_lesion", **common)

    return {
        "seed": seed, "chance": chance, "meta": meta,
        "stage0_depth_separating": bool(s0.get("depth_separating")),
        "oracle": oracle, "floor_inherit_heldout": f_inh,
        "deep_fa": {"inherit_heldout": deepfa["inherit_heldout"], "train": deepfa["train"],
                    "ff_weight_moved": deepfa["ff_weight_moved"],
                    "trains_at_all": _trains(deepfa["inherit_heldout"], f_inh, chance)},
        "decolle": {"inherit_heldout": decolle["inherit_heldout"], "train": decolle["train"],
                    "ff_weight_moved": decolle["ff_weight_moved"],
                    "trains_at_all": _trains(decolle["inherit_heldout"], f_inh, chance)},
        "decolle_permuted_inherit": permuted["inherit_heldout"],
        "decolle_lesion_inherit": lesion["inherit_heldout"],
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
    ap.add_argument("--tonic-h-pA", type=float, default=560.0)
    ap.add_argument("--tonic-o-pA", type=float, default=620.0)
    ap.add_argument("--apical-gain-pA", type=float, default=2000.0)
    ap.add_argument("--ff-w-init", type=float, default=4.5)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
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

    out = {"probe": "onbridge_deep_credit_decolle_vs_deepfa_AB", "seed": args.seed,
           "config": {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
                      "settle_steps": args.settle_steps, "credit_steps": args.credit_steps,
                      "n_hidden_layers": args.n_hidden_layers, "train_subsample": args.train_subsample,
                      "hp": hp, "task": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "result": r}

    d = r.get("deep_fa", {}); c = r.get("decolle", {})
    if d and c:
        ch = r.get("chance", 0.333)
        go = bool(c.get("trains_at_all") and not d.get("trains_at_all")
                  and r.get("decolle_permuted_inherit", 1.0) < ch + TRAIN_MARGIN
                  and r.get("decolle_lesion_inherit", 1.0) < r.get("floor_inherit_heldout", 0) + 0.10)
        out["GO_decolle_rescues"] = go
        out["verdict"] = (
            f"DECOLLE {'TRAINS' if c.get('trains_at_all') else 'fails'} "
            f"(inh {c.get('inherit_heldout'):.3f}, train {c.get('train'):.3f}) vs "
            f"deep-FA {'trains' if d.get('trains_at_all') else 'fails'} "
            f"(inh {d.get('inherit_heldout'):.3f}, train {d.get('train'):.3f}); "
            f"chance {ch:.3f}, floor {r.get('floor_inherit_heldout'):.3f}, oracle "
            f"{r.get('oracle', {}).get('inherit_heldout'):.3f}; decolle permuted "
            f"{r.get('decolle_permuted_inherit'):.3f}, lesion {r.get('decolle_lesion_inherit'):.3f}. "
            f"{'GO: per-layer LOCAL credit RESCUES on-spike training (the wall IS the multi-hop FA credit chain; DECOLLE surpasses it)' if go else 'NO rescue (the wall survives deletion of the deep chain -> deeper: on-spike credit fundamentally, or scale)'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(out.get("verdict", json.dumps(r)[:300]))


if __name__ == "__main__":
    main()
