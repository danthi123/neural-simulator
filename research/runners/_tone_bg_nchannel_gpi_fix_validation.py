"""6-seed validation of the N=5 GPi-tonic fix: does the N-channel race commit cleanly + track the host argmin
across the six seeds at the fixed tonic operating point (default gpi=500, thal=45 -- the mid-plateau of the
dev-seed-7 fine sweep)?

The dev-seed-7 fine sweep (`_tone_bg_nchannel_gpi_tonic_sweep`) found a robust clean window at gpi=300-650
(thal=45) where the N=5 race commits to the expected winner. This runner runs the FULL de-risk (intact +
arousal-lesion + direct-path-lesion, 5 v_held x `trials`) at the fixed tonic across the six seeds, so the
"wall closed" claim is a six-seed generalization (non-negotiable #6), not a dev-seed fluke. The lesion arms
also re-verify the anti-cheat: the commit is attributable to the spiking race (arousal) and the direct pathway,
not the host (the 2026-09-25 "who owns the computation" question).

GPU run (one brain-loading process, via tools/gpu_queue.sh). NO sim//webapp/ edit.

Usage:
  python -m research.runners._tone_bg_nchannel_gpi_fix_validation --gpi 500 --thal 45 \
      --seeds 42,43,44,100,101,102 --trials 20 --out research/findings/raw/_gpi_fix_6seed.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from research.runners._tone_bg_nchannel_selector import run_seed, selector_config

SIX_SEEDS = "42,43,44,100,101,102"


def main():
    ap = argparse.ArgumentParser(description="6-seed validation of the N=5 GPi-tonic fix (full de-risk at a fixed tonic)")
    ap.add_argument("--gpi", type=float, default=500.0, help="GPi tonic (pA) -- the fixed operating point")
    ap.add_argument("--thal", type=float, default=45.0, help="thalamic tonic (pA) -- fixed at the sub-threshold value")
    ap.add_argument("--seeds", type=str, default=SIX_SEEDS, help="comma-separated seeds")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = dataclasses.replace(selector_config("v2"), gpi_tonic_pA=args.gpi, thalamus_tonic_pA=args.thal)

    results = []
    for seed in seeds:
        res = run_seed(seed, trials=args.trials, config=cfg)
        results.append(res)
        print(json.dumps({
            "seed": seed, "go": res["go"],
            "intact_commit_rate": res["intact_commit_rate"],
            "no_arousal_commit_rate": res["no_arousal_commit_rate"],
            "no_direct_path_commit_rate": res["no_direct_path_commit_rate"],
            "intact_match_rate_vs_host_argmin": res["intact_match_rate_vs_host_argmin"],
            "checks": res["checks"],
        }, indent=2), flush=True)

    n_go = sum(1 for r in results if r["go"])
    summary = {
        "gpi_tonic_pA": args.gpi,
        "thalamus_tonic_pA": args.thal,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_go": n_go,
        "all_seeds_go": n_go == len(seeds),
        "mean_intact_commit_rate": round(sum(r["intact_commit_rate"] for r in results) / len(results), 4),
        "mean_match_rate": round(sum(r["intact_match_rate_vs_host_argmin"] for r in results) / len(results), 4),
        "per_seed": [{
            "seed": r["seed"], "go": r["go"],
            "intact_commit_rate": r["intact_commit_rate"],
            "no_arousal_commit_rate": r["no_arousal_commit_rate"],
            "no_direct_path_commit_rate": r["no_direct_path_commit_rate"],
            "intact_match_rate_vs_host_argmin": r["intact_match_rate_vs_host_argmin"],
            "modal_winner_by_v_held": r["modal_winner_by_v_held"],
            "checks": r["checks"],
        } for r in results],
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print("\n=== 6-seed summary ===", flush=True)
    print(json.dumps({k: summary[k] for k in (
        "gpi_tonic_pA", "thalamus_tonic_pA", "seeds", "n_go", "all_seeds_go",
        "mean_intact_commit_rate", "mean_match_rate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
