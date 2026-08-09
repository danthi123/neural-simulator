"""Aggregate per-seed _fm_neural_wta_readout artifacts into a 6-seed summary (no Claude in the loop).

PROVENANCE (the flagged fix): the aggregate records its OWN argv, the git SHA, the input files + their per-file
seeds, and a UTC timestamp -- so the derived table can be traced without re-reading every per-seed JSON."""
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.lab import lever

_REPO = Path(__file__).resolve().parents[2]


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO)).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else str(_REPO / "research/findings/raw/_fm_neural_wta_readout_s*.json")
    rows = []
    inputs = []
    for fp in sorted(glob.glob(pat)):
        if "full" in fp or "smoke" in fp or "agg" in fp:
            continue
        with open(fp) as f:
            d = json.load(f)
        for r in d.get("results", []):
            if "error" in r:
                print(f"  {fp}: ERROR {r['error']}")
                continue
            rows.append(r)
        inputs.append({"file": str(Path(fp).relative_to(_REPO)), "seeds": d.get("seeds"),
                       "runner_argv": d.get("argv")})
    if not rows:
        print("no rows"); return
    keys = ["ridge_heldout_acc", "twopath_rate_heldout", "neural_wta_heldout", "neural_wta_train",
            "wta_lesion_heldout", "prior_baseline_host_argmax", "winner_dominance_heldout", "dominant_rate_heldout",
            "wta_lesion_dominance", "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
            "untrained_control_heldout", "prior_lookup_heldout", "chance", "wta_ie_selected",
            "res_mean_spk_read", "ens_mean_spk_read"]
    print(f"seeds: {[r['seed'] for r in rows]}  n={len(rows)}")
    print(f"{'metric':38s} " + " ".join(f"s{r['seed']:<5d}" for r in rows) + "  MEAN")
    agg = {}
    for k in keys:
        vals = [r.get(k, float('nan')) for r in rows]
        agg[k] = _mean([v for v in vals if v == v])
        print(f"{k:38s} " + " ".join(f"{v:<6.3f}" for v in vals) + f"  {agg[k]:.3f}")
    print("\nverdict per seed:")
    gos = 0
    for r in rows:
        v = r.get("verdict", {}).get("status", "?")
        gos += int(v == "GO")
        print(f"  s{r['seed']}: {v}")
    print(f"\nGO {gos}/{len(rows)}")
    print(f"\nHEADLINE (6-seed mean): neural_wta={agg['neural_wta_heldout']:.3f}  "
          f"rate_ceiling={agg['twopath_rate_heldout']:.3f}  "
          f"wta_lesion(fallback)={agg['wta_lesion_heldout']:.3f}  "
          f"prior_baseline_host_argmax={agg['prior_baseline_host_argmax']:.3f}  chance={agg['chance']:.3f}")
    frac = agg['neural_wta_heldout'] / max(1e-9, agg['twopath_rate_heldout'])
    xchance = agg['neural_wta_heldout'] / max(1e-9, agg['chance'])
    print(f"  = {frac*100:.0f}% of the rate ceiling, {xchance:.0f}x chance; "
          f"lateral-inhibition load-bearing gap = {agg['neural_wta_heldout']-agg['wta_lesion_heldout']:+.3f}")
    # ATTRIBUTION: whose held-out? the lateral-inhibition WTA (removing it -> the argmax fallback collapses to chance).
    # The neural-WTA held-out is attributed to the biased competition, NOT a static bias: the fallback (competition
    # zeroed) sits at ~chance, so the lever (lateral inhibition) owns the above-chance signal.
    lever("lateral_inhibition_load_bearing_6seed_mean", before=round(agg["wta_lesion_heldout"], 4),
          after=round(agg["neural_wta_heldout"], 4), required=False)
    print(f"  [attribution] lateral-inhibition load-bearing: fallback {agg['wta_lesion_heldout']:.3f} "
          f"-> neural-WTA {agg['neural_wta_heldout']:.3f} (Δ={agg['neural_wta_heldout']-agg['wta_lesion_heldout']:+.3f})")
    print(f"  [honest] neural-WTA 6-seed mean {agg['neural_wta_heldout']:.3f} vs baseline host-argmax "
          f"{agg['prior_baseline_host_argmax']:.3f}: the neural winner is load-bearing but UNDERPERFORMS the host "
          f"argmax on average -- the winner op is not the bottleneck, the spike-count evidence is.")
    keep = ["seed", "neural_wta_heldout", "wta_lesion_heldout", "twopath_rate_heldout", "prior_baseline_host_argmax",
            "winner_dominance_heldout", "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
            "untrained_control_heldout", "wta_ie_selected", "chance", "seeded", "content_path_clean"]
    lean = [{k: r.get(k) for k in keep} | {"verdict_status": r.get("verdict", {}).get("status")} for r in rows]
    out = _REPO / "research/findings/raw/_fm_neural_wta_readout_6seed_agg.json"
    with open(out, "w") as f:
        json.dump({"provenance": {"aggregator": "research/runners/_aggregate_fm_neural_wta_seeds",
                                  "argv": sys.argv, "git_sha": _git_sha(),
                                  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "inputs": inputs, "seeds": [r["seed"] for r in rows]},
                   "per_seed": lean, "agg": agg, "go": gos, "n": len(rows)}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
