"""Aggregate per-seed _fm_learned_twopath_readout artifacts into a 6-seed summary (no Claude in the loop)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from tools.lab import before_after

_REPO = Path(__file__).resolve().parents[2]


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else str(_REPO / "research/findings/raw/_fm_learned_twopath_readout_s*.json")
    rows = []
    for fp in sorted(glob.glob(pat)):
        if "full" in fp or "smoke" in fp:
            continue
        with open(fp) as f:
            d = json.load(f)
        for r in d.get("results", []):
            if "error" in r:
                print(f"  {fp}: ERROR {r['error']}")
                continue
            rows.append(r)
    if not rows:
        print("no rows"); return
    keys = ["ridge_heldout_acc", "twopath_rate_heldout", "syn_heldout_acc", "syn_train_acc",
            "singlepath_spk_heldout", "prior_banked_singlepath_spk_heldout",
            "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
            "untrained_control_heldout", "prior_lookup_heldout", "chance",
            "res_mean_spk_read", "ens_mean_spk_read", "train_agree_neural_probe"]
    print(f"seeds: {[r['seed'] for r in rows]}  n={len(rows)}")
    print(f"{'metric':38s} " + " ".join(f"s{r['seed']:<5d}" for r in rows) + "  MEAN")
    agg = {}
    for k in keys:
        vals = [r.get(k, float('nan')) for r in rows]
        agg[k] = _mean([v for v in vals if v == v])
        print(f"{k:38s} " + " ".join(f"{v:<6.3f}" for v in vals) + f"  {agg[k]:.3f}")
    # go/no-go per seed
    print("\nverdict per seed:")
    gos = 0
    for r in rows:
        v = r.get("verdict", {}).get("status", "?")
        gos += int(v == "GO")
        print(f"  s{r['seed']}: {v}")
    print(f"\nGO {gos}/{len(rows)}")
    # headline honesty: syn vs ceiling, vs chance, vs prior banked
    print(f"\nHEADLINE (6-seed mean): syn_heldout={agg['syn_heldout_acc']:.3f}  "
          f"rate_ceiling={agg['twopath_rate_heldout']:.3f}  chance={agg['chance']:.3f}  "
          f"prior_banked_spiking={agg['prior_banked_singlepath_spk_heldout']:.3f}  "
          f"singlepath_spk(in-runner)={agg['singlepath_spk_heldout']:.3f}")
    frac = agg['syn_heldout_acc'] / max(1e-9, agg['twopath_rate_heldout'])
    xchance = agg['syn_heldout_acc'] / max(1e-9, agg['chance'])
    print(f"  = {frac*100:.0f}% of the rate ceiling, {xchance:.0f}x chance")
    # ATTRIBUTION at the aggregate level: whose is the 6-seed held-out? the W+ read-out synapses (mean lesion moves
    # it toward chance) -- not a static bias. before/after over the seed means.
    before_after("wp_readout_lesion_6seed_mean", before=round(agg["syn_heldout_acc"], 4),
                 after=round(agg["lesion_wp_heldout"], 4))
    # keep only a LEAN per-seed summary (decision-relevant fields); drop ceiling-constant train fits + verdict objects
    keep = ["seed", "syn_heldout_acc", "twopath_rate_heldout", "singlepath_spk_heldout",
            "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
            "untrained_control_heldout", "chance", "seeded", "content_path_clean"]
    lean = [{k: r.get(k) for k in keep} | {"verdict_status": r.get("verdict", {}).get("status")} for r in rows]
    out = _REPO / "research/findings/raw/_fm_learned_twopath_readout_6seed_agg.json"
    with open(out, "w") as f:
        json.dump({"per_seed": lean, "agg": agg, "go": gos, "n": len(rows)}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
