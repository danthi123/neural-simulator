"""One cell of the C4 scrambled-control sweep: the pass rate of the REAL scorer's C4 verdict for the realistic perfect
brain, under the unconstrained derangement (--max old) or the constrained draw at SCRAMBLE_R_MAX = --max.
    python docs/drafts/stage1-exit/bar_sanity_c4_sweep.py --max 0.05 --rho 0.93 --n 1000 --out cell.json
No brain, no GPU, pure numpy; ~100 s per 1000 draws."""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_c4_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max", default="0.10", help="'old' or a SCRAMBLE_R_MAX value")
ap.add_argument("--rho", default="realistic", help="'realistic' (U(0.93,0.96)), 'exact', or a float r(read, held value)")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out")
a = ap.parse_args()
rho = {"realistic": "realistic", "exact": None}.get(a.rho) if a.rho in ("realistic", "exact") else float(a.rho)
if a.max == "old":
    draw = C.derangement_old
else:
    C.G.SCRAMBLE_R_MAX = float(a.max)
    draw = C.derangement_constrained
rng = np.random.default_rng([a.seed, zlib.crc32(("%s|%s" % (a.max, a.rho)).encode())])
k = sum(1 for _ in range(a.n) if C.decide(C.make_data(rng, draw, rho=rho)))
rec = {"control": a.max, "rho": a.rho, "n": a.n, "seed": a.seed, "pass_rate": round(k / a.n, 4), "n_pass": k,
       "scorer": "research/runners/stage1_exit_gate.py aggregate (val_heldout verdict)", "scramble_r_max_used": getattr(C.G, "SCRAMBLE_R_MAX", None) if a.max != "old" else None}
print(json.dumps(rec))
if a.out:
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
C.cleanup()
