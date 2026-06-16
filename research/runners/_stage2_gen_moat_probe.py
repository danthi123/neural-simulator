"""Stage-2 gen-moat cheap-first probe: which familiarity statistic separates a KNOWN held-out cue from a
NO-CATEGORY novel cue on the UNIFIED bridge (so the no-confab moat abstains WITHOUT loosening the gate)?

Stage-2 (seed 42) breached the GENERALIZATION moat: the absolute win-fire (max per-category concept firing) of a
no-category cue (0.74) was ~91% of a known held-out cue (0.805), so the 0.6x familiarity gate (0.483) accepted it ->
CONFAB. Stage-1 passed the SAME moat because its magnitudes were larger (1.94 vs 1.00, ratio 0.52). The fix must be a
LEGITIMATE STRENGTHENING, never a looser gate. Hypothesis: a KNOWN cue produces a PEAKED category response (one
category clearly wins — h5_margin +0.40), while a NO-CATEGORY cue produces a DIFFUSE one (no clear winner); so a
MARGIN-based familiarity (winner - mean-of-others) / peakedness / z-score should separate them where the absolute
win-fire does not.

This probe builds ONE unified bridge (build_compose_bridge co_resident_generalization=True), reads the gen_concept
per-category firing (catmean) for each HELD-OUT cue and for several NO-CATEGORY novel cues, and reports — for each
candidate familiarity statistic — whether the WORST held-out value still exceeds EVERY novel value (a gate between
them would abstain all novel + accept all known). NO gate loosening; this measures separability of brain-read
statistics. Reuse-by-import; no sim/ edit.

Run: SIM_BACKEND=cupy python -m research.runners._stage2_gen_moat_probe --seed 42
"""
import argparse
import json
import time

import numpy as np

from sim.backend import get_backend
from research.runners.navigate_to_compose_then_answer import build_compose_bridge
from research.runners._unified_stage1_merged import _read_gen_spikes, _category_of_concept_spikes
from research.runners._genfrontier_capstone_vision_to_concept_derisk import novel_no_category_perc_set

N_NOVEL = 6   # several independent no-category cues (one could be lucky)


def _fam_stats(catmean):
    """Candidate familiarity statistics from the per-category concept firing vector. A KNOWN cue is PEAKED (one
    category wins); a NO-CATEGORY cue is DIFFUSE. Higher = more 'familiar' for every statistic."""
    cm = np.asarray(catmean, dtype=np.float64)
    winner = int(cm.argmax())
    mx = float(cm[winner])
    others = np.delete(cm, winner)
    mean_others = float(others.mean()) if others.size else 0.0
    mean_all = float(cm.mean())
    std_all = float(cm.std())
    return {
        "winfire": mx,                                   # the current (breaching) statistic
        "margin": mx - mean_others,                      # winner lead over the mean of the rest
        "margin_ratio": mx / (mean_others + 1e-9),       # winner / mean-of-others (peakedness, ratio form)
        "peakedness": mx / (mean_all + 1e-9),            # winner / overall mean
        "zscore": (mx - mean_all) / (std_all + 1e-9),    # winner standardized within the cue
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_stage2_gen_moat_probe.json")
    args = ap.parse_args()
    xp, backend = get_backend()
    print(f"[genmoat] backend={backend} seed={args.seed}", flush=True)

    t0 = time.time()
    bridge, composer, h, proj = build_compose_bridge(args.seed, with_body=True, co_resident_generalization=True)
    gen = h["gen"]
    n_cat = int(gen["N_CAT"])
    cat_ids = gen["gen_cat_ids"]
    held_out = list(gen["gen_held_out"])
    vis_sets = gen["vis_sets"]
    print(f"[genmoat] unified bridge built in {time.time()-t0:.0f}s | n_cat={n_cat} held_out={held_out}", flush=True)

    # KNOWN held-out cues -> per-category catmean -> familiarity statistics.
    ho = []
    for j in held_out:
        cpb, _f, _ct, _ft = _read_gen_spikes(bridge, gen, vis_sets[j], xp)
        keyed, catmean = _category_of_concept_spikes(cpb, cat_ids, n_cat)
        ho.append({"j": int(j), "true_cat": int(cat_ids[j]), "keyed": int(keyed),
                   "catmean": [round(float(c), 4) for c in catmean], "stats": _fam_stats(catmean)})

    # NO-CATEGORY novel cues (independent random sets).
    nov = []
    for k in range(N_NOVEL):
        rngm = np.random.default_rng(args.seed * 41 + 9 + k * 101)
        novel_set = novel_no_category_perc_set(gen["gen_W"], gen["gen_top_k"], n_cat, rngm)
        ncpb, _nf, _nct, _nft = _read_gen_spikes(bridge, gen, novel_set, xp)
        nkeyed, ncatmean = _category_of_concept_spikes(ncpb, cat_ids, n_cat)
        nov.append({"k": k, "keyed": int(nkeyed), "catmean": [round(float(c), 4) for c in ncatmean],
                    "stats": _fam_stats(ncatmean)})

    # For each statistic: does the WORST held-out exceed the BEST novel? (a gate between them abstains all novel.)
    summary = {}
    for stat in ("winfire", "margin", "margin_ratio", "peakedness", "zscore"):
        ho_vals = [d["stats"][stat] for d in ho]
        nov_vals = [d["stats"][stat] for d in nov]
        ho_min, nov_max = float(min(ho_vals)), float(max(nov_vals))
        separates = bool(ho_min > nov_max)
        gap = ho_min - nov_max
        summary[stat] = {"held_out_min": round(ho_min, 4), "held_out_mean": round(float(np.mean(ho_vals)), 4),
                         "novel_max": round(nov_max, 4), "novel_mean": round(float(np.mean(nov_vals)), 4),
                         "separates": separates, "gap_worstknown_minus_bestnovel": round(gap, 4)}
        print(f"[genmoat] {stat:12s}: held-out min {ho_min:.3f} (mean {np.mean(ho_vals):.3f}) | "
              f"novel max {nov_max:.3f} (mean {np.mean(nov_vals):.3f}) | SEPARATES={separates} (gap {gap:+.3f})",
              flush=True)

    best = max(summary, key=lambda s: (summary[s]["separates"], summary[s]["gap_worstknown_minus_bestnovel"]))
    verdict = {
        "best_statistic": best, "best_separates": summary[best]["separates"],
        "winfire_separates": summary["winfire"]["separates"],
        "note": ("a MARGIN/peakedness/z-score statistic separates known from no-category where the absolute "
                 "win-fire does not -> the legitimate gen-moat strengthening (no gate loosening)"
                 if summary[best]["separates"] and not summary["winfire"]["separates"]
                 else "absolute win-fire already separates here (the breach may be seed/gate-fraction specific)"
                 if summary["winfire"]["separates"]
                 else "NO statistic separates with N=%d novel cues -> need the population-code lever "
                      "(more gen_n_concept_per) or a deeper read fix" % N_NOVEL),
    }
    print(f"\n[genmoat] VERDICT {json.dumps(verdict, indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "summary": summary, "held_out": ho, "novel": nov}, f, indent=2)
    print(f"[genmoat] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
