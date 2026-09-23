"""D3 affect->one-brain-pool LOCAL SMOKE (1 seed, minutes): the affect organ's own answer-preservation on the REAL
12-organ pool, without the other 11 organs' on-pool training (the full gate is `_onebrain_affect_pool_verify`,
staged on the mini-PC pool).

Checks (a subset of the pre-registered M1/M3/M4/M7, plus the X5 affect-side byte-off):
  * affect battery on the 12-organ pool == affect alone on the 12-organ superset config (max |delta| == 0.0)
  * graded tone levels on the pool == the standalone production ladder's, at every production appraisal
  * signed / neutral / lesion-collapse alive checks; determinism (read twice)
  * the same battery on the 12-organ pool WITH the arousal->surprise synapse == without it (the edge is outgoing
    only: it cannot change the ladder's own read)

  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_smoke --seed 42 \
      --json research/findings/raw/_onebrain_affect_pool/smoke_seed42.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def run(seed):
    from research.runners.onebrain_merge_framework import merge_organs
    from research.runners.onebrain_affect_pool import (
        AFFECT_DESCRIPTOR, affect_descriptors, PoolAffectLadder, PROD_SWEEP, build_affect_pool)
    from research.runners._onebrain_wave1_organread_verify import _maxdelta
    from research.runners.affect_production_organ import tone_level
    from research.runners._stageA_full_integration_derisk import LADDER_NEUTRAL_TOL
    t0 = time.time()
    merged = build_affect_pool(seed, xedge=False)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    lm = PoolAffectLadder(seed, shared=merged)
    r_m, a_m = lm.reads(), lm.answer()
    again = PoolAffectLadder(seed, shared=merged).reads()
    core = merge_organs([AFFECT_DESCRIPTOR], seed, config_descriptors=affect_descriptors(), wire=True)
    lc = PoolAffectLadder(seed, shared=core)
    r_c, a_c = lc.reads(), lc.answer()
    xpool = build_affect_pool(seed, xedge=True)
    lx = PoolAffectLadder(seed, shared=xpool)
    r_x = lx.reads()
    std = PoolAffectLadder(seed, shared=None)
    std_d = [std.read_differential(a)["differential"] for a in PROD_SWEEP]
    std_lv = tuple(int(tone_level(d)) for d in std_d)
    d_c = _maxdelta(r_m, r_c)[0]
    d_x = _maxdelta(r_m, r_x)[0]
    d_again = _maxdelta(r_m, again)[0]
    signs = all((r_m[f"diff[{a:+.1f}]"] > 0) == (a > 0) and r_m[f"diff[{a:+.1f}]"] != 0.0
                for a in PROD_SWEEP if abs(a) >= 0.5)
    checks = {
        "coresidence_byte_identical(12-pool vs alone-on-superset)": bool(d_c == 0.0 and a_m == a_c),
        "tone_levels_equal_standalone_production_ladder": bool(tuple(a_m) == std_lv),
        "signed_at_|a|>=0.5": bool(signs),
        "neutral_below_tol": bool(abs(r_m["diff[+0.0]"]) < LADDER_NEUTRAL_TOL),
        "readout_lesion_zero": bool(r_m["diff_lesion[+0.7]"] == 0.0),
        "deterministic": bool(d_again == 0.0),
        "xedge_does_not_touch_ladder_read": bool(d_x == 0.0),
    }
    from tools.lab import attributable_to
    intero_owns = attributable_to("relay->ladder synapse owns the +1 differential on the pool",
                                  r_m["diff[+1.0]"], r_m["diff_intero_lesion[+1.0]"])
    from tools.verdict import Verdict
    v = Verdict(f"D3 affect ladder on the 12-organ pool — 1-seed smoke (seed {seed})")
    for k, ok in checks.items():
        v.require(k, bool(ok), expect=True)
    v.disabled("the other 11 organs' on-pool training + reads (M2/M5/X arms)",
               why="smoke scope: the full gate is _onebrain_affect_pool_verify on the mini-PC pool")
    decided = v.decide(go=all(checks.values()))
    out = {"seed": int(seed), "n_all_neurons": n_all, "checks": checks, **decided,
           "pool_levels": list(a_m), "standalone_levels": list(std_lv),
           "pool_diffs": [r_m[f"diff[{a:+.1f}]"] for a in PROD_SWEEP], "standalone_diffs": std_d,
           "intero_synapse_attributable_frac": intero_owns, "maxdelta_coresidence": d_c, "maxdelta_xedge": d_x, "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps({k: out[k] for k in ("seed", "n_all_neurons", "checks", "status", "pool_levels",
                                          "standalone_levels", "pool_diffs", "standalone_diffs")}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = run(a.seed)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
