"""Diagnostic: why does seed 44 fail to recall 'berkeley_county_virginia' -> 'culture_of_west_virginia'?

Tests the specific failed recall in isolation, comparing against other seeds to determine
if this is a genuine capability limit, a bug, or seed-specific behavior.

Run:  SIM_BACKEND=cupy .venv/bin/python -m research.runners._knowledge_scale_100k_cacheon_s44_recall_diag \
        --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \
        --json research/findings/raw/_knowledge_scale_100k_cacheon_s44_recall_diag.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from research.runners._knowledge_scale_100k_production_verify import main as verify_main


def _run_single(seed: int, bundle: str, out_path: str) -> dict:
    """Run the single-seed verify and return its output dict."""
    import sys
    argv = [
        "_knowledge_scale_100k_production_verify",
        "--bundle", bundle,
        "--enable-codebook-cache",
        "--seed", str(seed),
        "--json", out_path,
    ]
    sys.argv = argv
    ret = verify_main()
    with open(out_path) as f:
        return json.load(f)


def _extract_mismatch_details(res: dict) -> list:
    """Extract the oracle_byte_identity mismatches from a verify result."""
    obc = res.get("oracle_byte_identity", {})
    mismatches = obc.get("mismatches", [])
    return mismatches


def main():
    ap = argparse.ArgumentParser(description="Diagnostic: seed 44 recall failure")
    ap.add_argument("--bundle", default="/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k")
    ap.add_argument("--json", default="research/findings/raw/_knowledge_scale_100k_cacheon_s44_recall_diag.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = os.path.dirname(a.json)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    per_seed = {}
    errors = []

    for seed in seeds:
        print(f"[{seed}/{len(seeds)}] running verify, seed={seed} ...", flush=True)
        out_path = os.path.join(out_dir, f"_s{seed}_diag.json")
        try:
            res = _run_single(seed, a.bundle, out_path)
            per_seed[seed] = res
            obc = res.get("oracle_byte_identity", {})
            n_mismatches = obc.get("n_mismatches", 0)
            print(f"       status={res.get('status')} recall={res.get('scale_battery_flag_unset', {}).get('recall_rate')} "
                  f"n_mismatches={n_mismatches}", flush=True)
        except Exception as e:
            errors.append({"seed": seed, "error": str(e)})
            print(f"       ERROR: {e}", flush=True)

    # Now analyze the mismatches across seeds
    mismatch_analysis = {}
    for seed, res in per_seed.items():
        mismatches = _extract_mismatch_details(res)
        mismatch_analysis[seed] = {
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
            "recall_rate": res.get("scale_battery_flag_unset", {}).get("recall_rate"),
            "status": res.get("status"),
        }

    out = {
        "bundle": a.bundle,
        "seeds": seeds,
        "per_seed": {s: {k: v for k, v in per_seed[s].items() if k in ["status", "go", "oracle_byte_identity", "scale_battery_flag_unset"]} for s in per_seed},
        "mismatch_analysis": mismatch_analysis,
        "errors": errors,
        "elapsed_s": round(time.time() - t_start, 2),
    }

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {a.json}")

    # Summarize
    print(f"\n===== SUMMARY =====")
    for seed in seeds:
        ma = mismatch_analysis.get(seed, {})
        if ma:
            print(f"  seed {seed}: status={ma['status']}, recall={ma['recall_rate']}, n_mismatches={ma['n_mismatches']}")
            for m in ma.get("mismatches", []):
                cue = m.get("cue", [])
                oracle = m.get("oracle", "")
                live = m.get("live", "")
                print(f"    mismatch: cue={cue}, oracle={oracle}, live={live}")

    n_ok = sum(1 for s in per_seed.values() if s.get("go") is True)
    print(f"\nGO: {n_ok}/{len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())