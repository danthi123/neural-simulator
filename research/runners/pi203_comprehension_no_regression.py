#!/usr/bin/env python
"""Vikunja #203 (lane E · Language): does honouring cp_synapse_plastic_mask at the six residual write sites
change the REPLY of the production comprehension organ?

The flag under test is cfg.enforce_plastic_mask_beyond_hebbian, armed through the umbrella env var
BRAIN_ENFORCE_PLASTIC_MASK, DEFAULT-OFF. A freeze that is real but inert on the reply is a no-regression; a
freeze that changes the reply is load-bearing and must be reported before any flip. Neither is assumed here:
the two arms are run and compared, seed by seed.

Each arm is a FRESH process (the organ + the learned lexicons are process-global singletons keyed by the first
seed/build, exactly as the #190 joint flip-soak requires), CPU/numpy, one seed.

Run:  PYTHONPATH=. CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy OMP_NUM_THREADS=1 \
        python -m research.runners.pi203_comprehension_no_regression \
        --out research/findings/raw/_pi203_frozen_synapse_residual/comprehension_no_regression.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys

SEEDS = (42, 43, 44, 100, 101, 102)
ARM = "research.runners._comprehension_learned_cues_joint_arm"
SEMANTIC = ("competent", "judge.comprehended", "judge.svo", "judge.margin", "repair_target.loadbearing")


def _leaves(obj, prefix=""):
    """Yield (path, value) for every scalar leaf of a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, prefix + "." + k if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, "%s[%d]" % (prefix, i))
    else:
        yield prefix, obj


def _run_one(seed: int, enforce: bool) -> dict:
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = "."
    if enforce:
        env["BRAIN_ENFORCE_PLASTIC_MASK"] = "1"
    else:
        env.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)
    out = "/tmp/pi203_comp_s%d_%s.json" % (seed, "on" if enforce else "off")
    p = subprocess.run([sys.executable, "-m", ARM, "--seed", str(seed), "--animacy", "on",
                        "--verb", "on", "--out", out], env=env, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or not os.path.exists(out):
        return {"seed": seed, "enforce": enforce, "error": p.stderr[-300:], "returncode": p.returncode}
    battery = json.load(open(out)).get("battery", {})
    items = {}
    for name, item in battery.items():
        digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        leaves = dict(_leaves(item))
        semantic = {k: leaves.get(k) for k in SEMANTIC}
        items[name] = {"digest": digest, "semantic": semantic}
    return {"seed": seed, "enforce": enforce, "items": items}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_pi203_frozen_synapse_residual/"
                                    "comprehension_no_regression.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()

    jobs = [(s, e) for s in a.seeds for e in (False, True)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(lambda j: _run_one(*j), jobs))

    by_seed = {}
    for r in runs:
        by_seed.setdefault(r["seed"], {})["on" if r["enforce"] else "off"] = r

    comparison = []
    for seed, arms in sorted(by_seed.items()):
        off, on = arms.get("off"), arms.get("on")
        if "items" not in (off or {}) or "items" not in (on or {}):
            comparison.append({"seed": seed, "status": "ARM_FAILED",
                              "detail": ((off or {}).get("error") or (on or {}).get("error"))[:300]})
            continue
        names = sorted(set(off["items"]) | set(on["items"]))
        differing = [n for n in names if off["items"].get(n, {}).get("digest") != on["items"].get(n, {}).get("digest")]
        semantic_changes = []
        for n in differing:
            for k in SEMANTIC:
                if off["items"].get(n, {}).get("semantic", {}).get(k) != on["items"].get(n, {}).get("semantic", {}).get(k):
                    semantic_changes.append("%s.%s: %r -> %r" % (n, k,
                                                                 off["items"][n]["semantic"].get(k),
                                                                 on["items"][n]["semantic"].get(k)))
        comparison.append({"seed": seed, "battery_items": len(names),
                          "items_differing": differing,
                          "semantic_changes": semantic_changes,
                          "identical": not differing})

    record = {
        "board": "Vikunja #203 -- comprehension-organ (lane E) reply-level arm of the frozen-synapse fix",
        "flag": "cfg.enforce_plastic_mask_beyond_hebbian, armed via env BRAIN_ENFORCE_PLASTIC_MASK (default OFF)",
        "backend": "numpy", "device": "CPU (no CUDA on this host; CUDA_VISIBLE_DEVICES empty)",
        "seeds": list(a.seeds), "runs": len(runs), "elapsed_note": "4 concurrent arms, ~2.5 min each",
        "comparison": comparison,
        "n_identical_seeds": sum(1 for c in comparison if c.get("identical")),
        "n_seeds_with_reply_change": sum(1 for c in comparison if c.get("identical") is False),
    }
    with open(a.out, "w") as f:
        json.dump(record, f, indent=2)
    print("[pi203-comprehension] %d runs -> %s" % (len(runs), a.out))
    for c in comparison:
        if c.get("identical"):
            print("  seed %s: replies IDENTICAL across arms (%d items)" % (c["seed"], c["battery_items"]))
        else:
            print("  seed %s: REPLY CHANGES in %d items: %s" % (c["seed"], len(c["items_differing"]),
                                                                c["items_differing"][:6]))


if __name__ == "__main__":
    main()
