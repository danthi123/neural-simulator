"""Vikunja #203 -- the pre-fix byte-identity comparison, as an artifact (not just a pytest assertion).

Claim under test: with cfg.enforce_plastic_mask_beyond_hebbian OFF (the default), the scenario in
tests/_plastic_mask_beyond_hebbian_scenario.py produces EXACTLY the same full-state SHA as the genuine
pre-fix source, recovered with `git archive` of the commit that parented the flag (710883be5). Six rules x six
seeds = 36 comparisons. The flag-OFF arm is additionally re-run twice in the CURRENT tree to show the
comparison is not a lucky single draw (determinism control).

CPU/numpy only. No GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENARIO_REL = "tests/_plastic_mask_beyond_hebbian_scenario.py"
RULES = ["scaling", "structural", "reward", "rebuild", "compaction", "combined"]
SEEDS = [42, 43, 44, 100, 101, 102]
OUT = "research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json"
PREFIX_REF = "710883be56ab65bfb2d77fc74861742fab053fe2"


def _extract(ref: str) -> str:
    wt = tempfile.mkdtemp(prefix="prefix_bh_")
    arc = subprocess.check_output(["git", "archive", ref, "sim", "tests", "experiment", "config", "ui"], cwd=REPO)
    p = subprocess.Popen(["tar", "-x", "-C", wt], stdin=subprocess.PIPE)
    p.stdin.write(arc)
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("tar extraction failed rc=%d" % p.returncode)
    shutil.copy2(os.path.join(REPO, SCENARIO_REL), os.path.join(wt, SCENARIO_REL))
    return wt


def _run(tree: str, rule: str, seed: int, enforce: bool) -> dict:
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["PYTHONPATH"] = tree + os.pathsep + env.get("PYTHONPATH", "")
    if enforce:
        env["BRAIN_ENFORCE_PLASTIC_MASK"] = "1"
    else:
        env.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)
    out = subprocess.check_output([sys.executable, "-c",
                                   "import json,sys;"
                                   "from tests._plastic_mask_beyond_hebbian_scenario import run;"
                                   "print(json.dumps(run(sys.argv[1], sys.argv[2]=='on', int(sys.argv[3]))))",
                                   rule, "on" if enforce else "off", str(seed)], cwd=tree, env=env, text=True)
    line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    runs, mismatches = [], []
    prefix_tree = _extract(PREFIX_REF)
    try:
        for rule in RULES:
            for seed in SEEDS:
                off = _run(REPO, rule, seed, False)
                off2 = _run(REPO, rule, seed, False)
                pre = _run(prefix_tree, rule, seed, False)
                ok = (off["sha"] == pre["sha"]) and (off["sha"] == off2["sha"])
                runs.append({"rule": rule, "seed": seed, "off_sha": off["sha"], "prefix_sha": pre["sha"],
                            "off_repeat_sha": off2["sha"], "identical": ok,
                            "frozen_max_delta_off": off["frozen_max_delta"],
                            "control_max_delta_off": off["control_max_delta"],
                            "fired_any_off": off["fired_any"]})
                if not ok:
                    mismatches.append({"rule": rule, "seed": seed, "off": off["sha"], "prefix": pre["sha"]})
    finally:
        shutil.rmtree(prefix_tree, ignore_errors=True)
    rec = {
        "generated_utc": t0.isoformat(),
        "elapsed_seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 2),
        "backend": "numpy", "device": "cpu", "seeds": SEEDS, "rules": RULES,
        "prefix_ref": PREFIX_REF,
        "note": "prefix_ref is the parent of the commit that introduced enforce_plastic_mask_beyond_hebbian; "
                "recovered with `git archive` (git worktree fails on this box, FAILURE_LOG 2026-10-31).",
        "comparisons": len(runs),
        "identical": sum(1 for r in runs if r["identical"]),
        "mismatches": mismatches,
        "runs": runs,
        "cost_projection": "36x3 scenario runs (6 rules x 6 seeds x {off, off-repeat, pre-fix}), numpy/CPU, "
                           "6-neuron bridge, 30 reads x 20 steps; ~1 s each, no GPU, no API cost",
    }
    path = os.path.join(REPO, OUT)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print("[byte-identity] %d/%d identical -> %s" % (rec["identical"], rec["comparisons"], OUT))
    if mismatches:
        print(json.dumps(mismatches, indent=2))
    np.set_printoptions(precision=4)


if __name__ == "__main__":
    main()
