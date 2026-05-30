"""Task 6 CONTROLLER-ONLY decisive-run orchestrator for the phase-factored
integrated loop. Pure KILL-SAFE orchestration around the already-reviewed
gate (research/runners/phase_factored_loop_gate.py) + the frozen verdict
(research/runners/integrated_loop_core.py). Adds NO science logic and NO
bars -- it only sequences the full-scale per-load runs with per-load
crash-resilient caching, then aggregates + scores via the inherited frozen
verdict.

Why per-load caching: a full-scale multi-seed two-phase run is heavy
(the D7 production run was killed by a client crash mid-flight; KILL-SAFE
caches there saved ~46 hr). Here each ladder load N in {2,4,8} runs in its
own subprocess writing its own JSON; a crash loses at most the in-flight
load, and re-running skips any load whose JSON already exists. The gate's
--only-load flag is a pure run-scope filter (changes NO rng draw or scored
quantity), so per-load runs are byte-identical to a single combined run.

Run:  python -m research.runners.phase_factored_decisive
      [--seeds 42 43 44] [--out <decisive.json>]

NOT a subagent task. The controller runs this, then performs the mandatory
smell-test (scrutinize a PASS harder than a FAIL) on the recorded JSON and
propagates the honest outcome to both remotes.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.integrated_loop_core import (
    integrated_loop_verdict, _IL_LADDER,
)

_CACHE = os.path.join(_REPO, "research/findings/raw",
                      "phase_factored_decisive_cache")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=os.path.join(
        _REPO, "research/findings/raw/phase_factored_decisive.json"))
    a = ap.parse_args(argv)
    os.makedirs(_CACHE, exist_ok=True)

    loads = list(_IL_LADDER)  # (2, 4, 8)
    seed_args = [str(s) for s in a.seeds]

    # --- KILL-SAFE per-load full-scale runs ---
    for N in loads:
        load_json = os.path.join(_CACHE, "load_%d.json" % N)
        if os.path.exists(load_json):
            print("[decisive] load N=%d cached, skipping" % N, flush=True)
            continue
        print("[decisive] running full-scale load N=%d seeds=%s ..."
              % (N, seed_args), flush=True)
        cmd = [sys.executable, "-m",
               "research.runners.phase_factored_loop_gate",
               "--only-load", str(N), "--seeds", *seed_args,
               "--out", load_json]
        # Inherit stdout/stderr so progress streams to the controller log.
        res = subprocess.run(cmd, cwd=_REPO)
        if res.returncode != 0 or not os.path.exists(load_json):
            raise SystemExit(
                "[decisive] load N=%d FAILED (returncode %d); its cache was "
                "not written. Re-run to resume from the completed loads."
                % (N, res.returncode))

    # --- aggregate the per-load 3-seed rungs + score via frozen verdict ---
    rungs = []
    for N in loads:
        d = json.load(open(os.path.join(_CACHE, "load_%d.json" % N)))
        # each per-load run wrote result["rungs"] == [the 3-seed rung for N]
        rung = d["rungs"][0]
        if int(rung["N"]) != N:
            raise SystemExit(
                "[decisive] cache load_%d.json holds N=%s (corrupt cache)"
                % (N, rung["N"]))
        rungs.append(rung)

    result = integrated_loop_verdict(rungs)
    result["rungs"] = rungs
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("[decisive] GATE=%s  classification=%s"
          % (result.get("GATE"), result.get("classification")), flush=True)
    print("[decisive] wrote %s" % a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
