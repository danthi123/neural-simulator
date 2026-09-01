"""6-seed codebook-cache production-load soak for the 79k bundle (board #192/#66).

Runs the end-to-end scale verify (with codebook-cache ON) for all 6 seeds (42/43/44/100/101/102)
on cupy, then aggregates the results. The 6-seed bar: all 6 must pass the latency bar (median < 1000 ms)
and recall >= 0.99 and moat = 0.

Run:  SIM_BACKEND=cupy .venv/bin/python -m research.runners._knowledge_scale_100k_cacheon_6seed \
        --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \
        --json research/findings/raw/_knowledge_scale_100k_cacheon_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def _run_single(seed: int, bundle: str, out_path: str, enable_decode_escalation: bool = False) -> dict:
    """Run the single-seed verify and return its output dict."""
    from research.runners._knowledge_scale_100k_production_verify import main as verify_main
    import sys
    # Reproduce the exact argv that verify_main expects
    argv = [
        "_knowledge_scale_100k_production_verify",  # fake program name
        "--bundle", bundle,
        "--enable-codebook-cache",
        "--seed", str(seed),
        "--json", out_path,
    ]
    if enable_decode_escalation:
        argv.append("--enable-decode-escalation")   # #66 seed-44 recall-hole fix
    sys.argv = argv
    ret = verify_main()
    # read the JSON back
    with open(out_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="6-seed codebook-cache production-load soak (#66/#192)")
    ap.add_argument("--bundle", default="/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k")
    ap.add_argument("--json", default="research/findings/raw/_knowledge_scale_100k_cacheon_6seed.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--enable-decode-escalation", action="store_true",
                   help="ON: pass --enable-decode-escalation to each per-seed verify (#66 seed-44 recall-hole fix)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = os.path.dirname(a.json)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    per_seed = {}
    errors = []
    for seed in seeds:
        print(f"[{seed}/{len(seeds)}] running 79k scale verify with codebook-cache ON, seed={seed} ...", flush=True)
        out_path = os.path.join(out_dir, f"_knowledge_scale_100k_cacheon_s{seed}.json")
        try:
            res = _run_single(seed, a.bundle, out_path, enable_decode_escalation=a.enable_decode_escalation)
            per_seed[seed] = res
            print(f"       status={res.get('status')} recall={res.get('scale_battery_flag_unset', {}).get('recall_rate')} "
                  f"lat_med={res.get('scale_battery_flag_unset', {}).get('latency_ms_median')}ms", flush=True)
        except Exception as e:
            errors.append({"seed": seed, "error": str(e)})
            print(f"       ERROR: {e}", flush=True)

    # aggregate
    n_ok = 0
    n_fail = 0
    for seed, res in per_seed.items():
        r = res.get("scale_battery_flag_unset", {})
        if (res.get("go") is True and
                (r.get("recall_rate", 0) or 0) >= 0.99 and
                r.get("moat_confab", 1) == 0 and
                (r.get("latency_ms_median", 1e9) or 1e9) < 1000.0):
            n_ok += 1
        else:
            n_fail += 1

    out = {
        "bundle": a.bundle,
        "seeds": seeds,
        "per_seed": per_seed,
        "errors": errors,
        "n_seeds_ok": n_ok,
        "n_seeds_fail": n_fail,
        "n_seeds_total": len(seeds),
        "all_6_GO": n_ok == len(seeds) and n_fail == 0 and len(errors) == 0,
        "status": "GO" if n_ok == len(seeds) else ("PARTIAL" if n_ok > 0 else "NO-GO"),
        "elapsed_s": round(time.time() - t_start, 2),
    }

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {a.json}")
    print(f"\n===== VERDICT: {out['status']} ({n_ok}/{len(seeds)} GO) elapsed={out['elapsed_s']}s =====")
    return 0 if out["all_6_GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())