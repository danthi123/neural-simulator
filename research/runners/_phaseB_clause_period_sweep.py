"""Clause-safe resonate-period sweep: the flat-query period sweep (2026-06-17-resonate-period-free-speedup.md)
found who/what work at period>=32, but the agent adoption at period=48 BROKE embedded clauses
(test_embedded_clause). A recursive clause -- `Clause(cat, go, south)` bound as the patient of 'dog look ...' --
is a DOUBLE unbind (outer clause composite, then the 3 inner SVO roles), so it needs a longer resonate window for
a faithful phase read than a flat query. This sweep finds the smallest period that keeps BOTH flat who/what AND
the recursive clause render correct -> the clause-safe adoption period (still a real ~1.6-2x win over 208).

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_clause_period_sweep [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.core_sim_composition import Clause  # noqa: E402

VOCAB = ["dog", "cat", "bird", "fish", "elephant", "horse", "lion", "wolf",
         "go", "run", "fly", "swim", "eat", "see", "chase", "hunt", "look",
         "north", "south", "east", "west", "river", "tree", "mouse", "deer"]
FLAT = [("dog", "go", "north"), ("cat", "run", "south"), ("bird", "fly", "east"), ("fish", "swim", "west")]
CLAUSES = [("dog", "look", Clause("cat", "go", "south")),
           ("horse", "see", Clause("wolf", "hunt", "mouse"))]
PERIODS = [48, 64, 80, 100, 128, 160, 200]


def run_seed(seed, period):
    c = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB, period=period)
    for a, ac, p in FLAT:
        c.store(a, ac, p)
    for a, ac, cl in CLAUSES:
        c.store(a, ac, cl)
    flat = sum(int(c.query_patient(a, ac) == p) for a, ac, p in FLAT) / len(FLAT)
    clause = sum(int(c.query_patient(a, ac) == f"{cl.agent} {cl.action} {cl.patient}")
                 for a, ac, cl in CLAUSES) / len(CLAUSES)
    moat = int(c.query_patient("lion", "fly") is None)
    return flat, clause, moat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[clause-safe period sweep] smallest period keeping FLAT + recursive CLAUSE correct? seeds={seeds}\n",
          flush=True)
    rows = {}
    for period in PERIODS:
        fl, cl, mo = [], [], []
        for s in seeds:
            f, c, m = run_seed(s, period)
            fl.append(f); cl.append(c); mo.append(m)
        rows[period] = {"flat": float(np.mean(fl)), "clause": float(np.mean(cl)), "moat": int(min(mo))}
        print(f"  period={period:>3} (steps={period+8:>3}): flat {np.mean(fl):.3f} | clause {np.mean(cl):.3f} | "
              f"moat {min(mo)}/1", flush=True)

    ok = [p for p in PERIODS if rows[p]["flat"] >= 0.999 and rows[p]["clause"] >= 0.999 and rows[p]["moat"] == 1]
    print(f"\n{'='*84}", flush=True)
    if ok:
        pmin = min(ok)
        print(f"  CLAUSE-SAFE period = {pmin} (steps {pmin+8}): keeps flat + recursive clause + moat at full "
              f"accuracy -> {200/pmin:.2f}x fewer steps than 200. Adopt at the agent level gated on the FULL "
              f"conversational suite (incl. test_embedded_clause).", flush=True)
    else:
        print(f"  BOUNDARY: even period 200 is needed for the recursive clause (no shorter period clears it at "
              f"all seeds) -> the clause unbind is the binding constraint; keep 200.", flush=True)
    print(f"{'='*84}", flush=True)
    out = {"seeds": seeds, "periods": {str(p): rows[p] for p in PERIODS},
           "clause_safe_period": (min(ok) if ok else None)}
    path = os.path.join(_REPO, "research", "findings", "raw", "_clause_period_sweep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
