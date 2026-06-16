"""CYCLE 102 — step-3 de-risk: does the LEARNED binder handle BUNDLED SVO facts (the conversational structure)
AND still generalize to held-out combinations?

The systematicity de-risks (#0..#2c) tested SINGLE role-filler binding. But a conversational FACT is a BUNDLE
(superposition) of three bindings: agent+verb+object. The unbind of one role from a 3-way bundle has crosstalk
from the other two -- and the binder was trained on SINGLE pairs, never bundles. THE QUESTION: can the learned
(ON/OFF spiking) binder, trained only on single role-filler pairs, recover each role's filler from a 3-way
bundle (who/what recall) AND generalize when the queried (role, filler) combination was HELD OUT of training?
If yes, the learned bind supports real conversation; if it collapses under bundling, the binder needs
bundle-aware training (localize) before any on-bridge build.

GATE (3 seeds, the stream codes): who/what recall on bundled facts >> chance, and HELD-OUT-combo recall ~
TRAIN-combo recall (generalizes under bundling). Compare to the single-binding held-out (the no-bundle ceiling).

Reuse-by-import (the validated ON/OFF spiking binder + the systematicity splits); cached 320 stream codes; CPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundled_facts_derisk
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

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)
from research.runners._phaseB_spiking_bind_onoff_derisk import OnOffRateBinder  # noqa: E402

R, F, N_SPLITS, N_EPOCHS, D_H, LR, READ_NOISE = 4, 16, 3, 600, 64, 0.005, 0.20
N_FACTS = 24       # SVO facts per split (each = 3 of the R roles bundled; use roles 0,1,2 = agent,verb,object)


def run_seed(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 71 + 13)
    single_held, bundle_train, bundle_held = [], [], []
    for split in splits:
        binder = OnOffRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=READ_NOISE)
        binder.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        train_set = set(split["train"])
        # single-binding held-out ceiling (no bundle)
        sc = sum(int(native_argmax(binder._unbind(binder._bind(roles[r], fillers[f]), roles[r]), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)
        # BUNDLED SVO facts: pick 3 roles (0=agent,1=verb,2=object); each fact = 3 distinct fillers bundled
        ntr_ok = ntr = nh_ok = nh = 0
        for _ in range(N_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bound = sum(binder._bind(roles[r], fillers[fids[r]]) for r in range(3))   # superposition [2*D_h]
            for r in range(3):                                  # query each role -> recover its filler
                pred = native_argmax(binder._unbind(bound, roles[r]), fillers)
                ok = int(pred == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
    sh, bt, bh = float(np.mean(single_held)), float(np.mean(bundle_train)), float(np.mean(bundle_held))
    print(f"  [seed {seed}] single-binding held-out {sh:.3f} | BUNDLED recall: train-combo {bt:.3f}, "
          f"held-out-combo {bh:.3f}", flush=True)
    return {"seed": seed, "single_held": sh, "bundle_train": bt, "bundle_held": bh}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[bundled-facts learned-bind de-risk] stream codes {codes.shape} -- does the ON/OFF spiking binder "
          f"(trained on SINGLE pairs) recover fillers from 3-way BUNDLED SVO facts + generalize?", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    sh, bt, bh = m("single_held"), m("bundle_train"), m("bundle_held")
    chance = 1.0 / F
    print(f"\n{'='*98}\n  MEAN (3 seeds): single-binding held-out {sh:.3f} | BUNDLED recall train-combo {bt:.3f}, "
          f"held-out-combo {bh:.3f} | chance {chance:.3f}", flush=True)
    print(f"{'='*98}", flush=True)
    if bh >= 0.40 and bh >= 0.6 * bt and bt >= 0.50:
        print(f"  GO: the learned binder HANDLES BUNDLED SVO facts AND generalizes -- bundled held-out-combo recall "
              f"{bh:.3f} (>> chance {chance:.3f}), {bh/max(bt,1e-9):.0%} of the train-combo bundled recall {bt:.3f}; "
              f"single-binding ceiling {sh:.3f}. The 3-way superposition crosstalk is tolerable + generalization "
              f"survives bundling. ==> the learned bind supports the conversational fact structure -> the on-bridge "
              f"build is worthwhile.", flush=True)
    elif bh >= 0.25:
        print(f"  PARTIAL: bundling degrades recall (held-out-combo {bh:.3f} vs single {sh:.3f}) but stays above "
              f"chance -- the 3-way crosstalk costs accuracy; bundle-AWARE training (train the unbind on bundles) "
              f"or capacity-cleanup should recover it. Generalization under bundling is partial.", flush=True)
    else:
        print(f"  NEGATIVE: bundling collapses recall to ~chance (held-out-combo {bh:.3f} vs chance {chance:.3f}) -- "
              f"the single-pair-trained unbind can't separate the 3-way superposition; the binder needs "
              f"bundle-aware training before any on-bridge build. Localize.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"single_held": sh, "bundle_train": bt, "bundle_held": bh, "chance": chance, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_learned_bind_bundled_facts.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
