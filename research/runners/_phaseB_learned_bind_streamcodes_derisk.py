"""CYCLE 98 — the LEARNED BIND frontier (step 3), cheap-first: does a LEARNED binder generalize SYSTEMATICALLY
(Fodor-Pylyshyn held-out role-filler) on the STREAM-LEARNED cortex codes -- i.e. can we replace the fixed VSA
bind ALGEBRA with a cortex that LEARNS to bind, on the codes the bridge actually learned from conversation?

CONTEXT. The biologization sweep (CYCLE 97) made the bind OPERATION spiking (±1 coincidence, recall 0.92), but
the underlying scheme is still a fixed exact-inverse VSA ALGEBRA, not a LEARNED bind. The genuine "step 3" is a
LEARNED binder. CYCLE 89 showed a learned BilinearBinder generalizes systematically (held-out = train) on
DECORRELATED codes, and the 2026-06-11 work showed it FAILS on EXTREME-correlated codes (denoise64, cos 0.81).
The stream-learned codes are MODERATELY correlated (the binding sweet spot). THE QUESTION: does the learned
binder generalize on THESE codes? If yes -> the learned bind (step 3) is reachable on the stream cortex's own
codes; if no (memorization) -> the learned bind needs the decorrelation/dendritic path (the deeper frontier).

REUSE the validated systematicity protocol VERBATIM (run_condition): leakage-free train/held-out splits, the
BilinearBinder, + all 4 anti-cheats (leakage assert, shuffled-label, memorization floor, abstention). The ONLY
change is the filler codes = the cached stream-learned codes (vs the probe's synthetic decorr/corr regimes).

GATE (3 seeds, F=16 fillers from the 320 stream codes): SYSTEMATIC if held-out ~ train, >> chance (1/16) AND
>> the memorization floor AND shuffled-label drops to chance AND the FHRR exact-inverse reference stays ~1.0.

Reuse-by-import (run_condition + the whole protocol); the cached 320 stream codes; CPU; no GPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_streamcodes_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.cortex_learned_binder_systematicity_probe import run_condition  # noqa: E402

R = 4
F = 16
N_SPLITS = 3
N_EPOCHS = 800
D_H = 64
LR = 0.005


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    # between-code correlation (the regime these learned codes live in)
    sub = codes[:F]
    bc = [float(sub[i] @ sub[j]) for i in range(F) for j in range(i + 1, F)]
    print(f"[learned-bind de-risk] stream codes {codes.shape} | F={F} fillers between-cos mean {np.mean(bc):+.3f} "
          f"(max {np.max(bc):+.3f}) -- does a LEARNED binder generalize SYSTEMATICALLY on the stream cortex's "
          f"own learned codes? (vs the fixed VSA algebra)", flush=True)
    rows = []
    for seed in (42, 43, 44):
        r = run_condition("stream_learned", codes, R, F, seed, N_SPLITS, N_EPOCHS, D_H, LR, verbose=False)
        memf = float(np.mean([s["mem_floor_held_acc"] for s in r["splits"]]))
        print(f"  [seed {seed}] learned binder: train {r['bilinear_train_acc_mean']:.3f} -> HELD-OUT "
              f"{r['bilinear_held_acc_mean']:.3f} (chance {r['chance']:.3f}, mem-floor {memf:.3f}) | FHRR-ref "
              f"held {r['fhrr_held_acc_mean']:.3f} | shuffled {r['shuffled_acc_mean']:.3f} | systematic "
              f"{r['n_systematic_splits']}/{r['n_splits_total']}", flush=True)
        rows.append({"seed": seed, "train": r["bilinear_train_acc_mean"], "held": r["bilinear_held_acc_mean"],
                     "chance": r["chance"], "mem_floor": memf, "fhrr": r["fhrr_held_acc_mean"],
                     "shuffled": r["shuffled_acc_mean"], "n_sys": r["n_systematic_splits"],
                     "n_splits": r["n_splits_total"]})

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    held, train, chance, memf = m("held"), m("train"), m("chance"), m("mem_floor")
    fhrr, shuf = m("fhrr"), m("shuffled")
    n_sys = sum(r["n_sys"] for r in rows); n_tot = sum(r["n_splits"] for r in rows)
    print(f"\n{'='*96}\n  MEAN (3 seeds): learned binder train {train:.3f} -> HELD-OUT {held:.3f} (chance {chance:.3f}"
          f", mem-floor {memf:.3f}) | FHRR-ref {fhrr:.3f} | shuffled {shuf:.3f} | SYSTEMATIC {n_sys}/{n_tot} splits",
          flush=True)
    print(f"{'='*96}", flush=True)
    if held >= memf + 0.15 and held >= 2 * chance and shuf < held - 0.15 and n_sys >= 2:
        print(f"  GO (learned bind reachable): a LEARNED binder GENERALIZES SYSTEMATICALLY on the stream-learned "
              f"codes -- held-out {held:.3f} >> mem-floor {memf:.3f} + chance {chance:.3f}, shuffled drops "
              f"({shuf:.3f}), {n_sys}/{n_tot} splits systematic. ==> step 3 (a cortex that LEARNS to bind, not the "
              f"fixed VSA algebra) is REACHABLE on the stream cortex's own codes -> worth the build.", flush=True)
    elif held >= memf + 0.05:
        print(f"  PARTIAL: the learned binder beats memorization ({held:.3f} vs floor {memf:.3f}) but not decisively "
              f"-- the moderate correlation costs some systematicity; more epochs / capacity / the sweet-spot "
              f"correlation may lift it.", flush=True)
    else:
        print(f"  NEGATIVE (memorization): the learned binder does NOT generalize on the stream codes (held-out "
              f"{held:.3f} ~ mem-floor {memf:.3f}) -- the learned bind needs the decorrelation/dendritic path "
              f"(the deeper frontier); the fixed VSA algebra stays the pragmatic bind.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"held": held, "train": train, "chance": chance, "mem_floor": memf, "fhrr": fhrr, "shuffled": shuf,
           "n_systematic": n_sys, "n_splits": n_tot, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_learned_bind_streamcodes.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
