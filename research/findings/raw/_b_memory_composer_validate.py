"""(B store integration validation) Production no-regression confirm: the opt-in spiking NEF cleanup composer
(`CoreSimComposer(enable_spiking_memory=True)`) must answer the capability matrix IDENTICALLY to the numpy
composer at the production dimension D=2048, multi-seed. The de-risk validated the cleanup MECHANISM on the
real est at proj_dim=800 (the harder/noisier regime, GO 0.978/0.993); this confirms the END-TO-END composer
(store -> unbind -> cleanup) at production D=2048 with the integrated flag.

Run AFTER the integration lands (it uses the enable_spiking_memory flag):
  python -u -m research.findings.raw._b_memory_composer_validate --proj-dim 2048 --seeds 42 43 44 \
      --out research/findings/raw/_b_memory_composer_validate.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer
from research.findings.raw._core_composer_grounded320_probe import production_codes


def run_matrix(comp):
    """A representative capability check on a built composer. Returns a dict of (query -> answer)."""
    comp.kb = []
    words = comp.words
    # pick distinct concept words as roles/fillers (deterministic)
    A, AC, P = words[3], words[7], words[11]
    A2, AC2, P2 = words[20], words[25], words[33]
    ADJ, NOUN = words[40], words[44]
    out = {}
    # flat facts
    comp.store(A, AC, P)
    comp.store(A2, AC2, P2)
    out["what_A_AC"] = comp.query_patient(A, AC)            # -> P
    out["who_AC_P"] = comp.query_agent(AC, P)               # -> A
    out["what_A2_AC2"] = comp.query_patient(A2, AC2)        # -> P2
    out["who_AC2_P2"] = comp.query_agent(AC2, P2)           # -> A2
    out["abstain_what"] = comp.query_patient(A, AC2)        # no such fact -> None
    out["render_A"] = comp.render_fact(A)                   # -> "A AC P"
    # one-attribute fact
    comp.kb = []
    comp.store(A, AC, (ADJ, NOUN))
    out["attr_what"] = comp.query_patient(A, AC)            # -> "ADJ NOUN"
    # negation / yes-no
    comp.kb = []
    comp.store(A, AC, P, polarity="AFFIRM")
    out["yesno_true"] = comp.ask_yes_no(A, AC, P)           # -> yes
    out["yesno_wrong"] = comp.ask_yes_no(A, AC, P2)         # patient mismatch -> no/unknown
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        print("SKIP: GPU backend required.")
        return

    rows = []
    total_match = total = 0
    for seed in args.seeds:
        codes_in = production_codes(args.vocab, 2000, 100, args.proj_dim, seed)
        words = [f"c{i:03d}" for i in range(args.vocab)]
        concepts = {w: codes_in[i] for i, w in enumerate(words)}
        comp_np = CoreSimComposer(seed=seed, proj_dim=args.proj_dim, concepts=concepts,
                                  enable_spiking_memory=False)
        ans_np = run_matrix(comp_np)
        comp_sp = CoreSimComposer(seed=seed, proj_dim=args.proj_dim, concepts=concepts,
                                  enable_spiking_memory=True)
        ans_sp = run_matrix(comp_sp)
        keys = sorted(ans_np)
        match = {k: (ans_np[k] == ans_sp[k]) for k in keys}
        n_match = sum(match.values())
        rows.append({"seed": seed, "numpy": ans_np, "spiking": ans_sp, "match": match,
                     "n_match": n_match, "n_total": len(keys)})
        total_match += n_match; total += len(keys)
        print(f"[validate] seed {seed}: {n_match}/{len(keys)} spiking==numpy", flush=True)
        for k in keys:
            if not match[k]:
                print(f"    MISMATCH {k}: numpy={ans_np[k]!r} spiking={ans_sp[k]!r}", flush=True)

    verdict = "GO" if total_match == total else "REGRESSION"
    print(f"\n[VERDICT] spiking-memory composer vs numpy: {total_match}/{total} match across seeds -> {verdict}")
    if args.out:
        json.dump({"rows": rows, "total_match": total_match, "total": total, "verdict": verdict},
                  open(args.out, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
