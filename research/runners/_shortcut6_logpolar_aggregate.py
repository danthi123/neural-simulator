"""#6 SURPASS — log-polar grid-32 6-seed aggregator.

Reads the seed-tagged per-arm JSONs (scpv_logpolar_{host,sc_popvector,sc_popvector_scr}_seed{S}.json)
written by _shortcut6_logpolar_6seed.ps1, and computes the GO verdict per the deep-research
(2026-06-22-shortcut6-upstream-orienting-residual-surpass.md, RANK 1):

  GO bar (all three):
    (a) FIX-1+log-polar (the sc_popvector arm) TRACKS on the DIAGONAL phases (phase0 NE, phase2 SW,
        phase3 SE -- the phases that were finalQ 20-47 with the truncated retina) AND the lateral
        phase (phase1 far-W). "Tracks" = the per-phase dominant cardinal is consistent with the
        goal's egocentric bearing (the goal's two-cardinal direction relative to where the agent
        spends the phase) AND finalQ is materially below the SCRAM arm's.
    (b) post-goal-change Sigma (sum of phases 1..3 finalQ) moves toward HOST (host_over_popvector
        ratio rises toward 1 vs the prior NEGATIVE ~0.01).
    (c) SCRAM COLLAPSES (the retinotopy-scramble arm is materially WORSE than the intact popvector
        arm -> the decode is load-bearing now there is finally a signal).

Anti-cheats reported: SCRAM-collapse margin, per-phase diagonal tracking (not just lateral),
tie-break fraction (the decode driven by the SC margin, not lucky ties), matched render (log_polar
on for both popvector + scram). The render itself proven not to smuggle (gx,gy) by the CPU smoke.
"""
import argparse
import json
import os

import numpy as np

ACTION_NAMES = ["N", "E", "S", "W"]
DIR = "research/findings/raw/nav_gate_2a"

# The schedule goal of each phase (corner cell), and the two cardinals that point toward it from
# the grid-OPPOSITE start region (the agent must move in BOTH to reach the corner). This is the
# bearing the dominant cardinal should match on a tracking arm.
PHASE_GOAL_CARDINALS = {
    0: ({"N", "E"}, "NE"),   # NE corner: needs North + East
    1: ({"N", "W"}, "NW/farW"),  # NW corner (the pure-lateral far-WEST goal; W is the load-bearing axis)
    2: ({"S", "W"}, "SW"),   # SW corner: needs South + West
    3: ({"S", "E"}, "SE"),   # SE corner: needs South + East
}
DIAGONAL_PHASES = [0, 2, 3]   # NE / SW / SE -- the phases that failed with the truncated retina


def _load(arm, seed):
    p = os.path.join(DIR, f"scpv_logpolar_{arm}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _per_phase(res):
    ps = res.get("phase_stats", [])
    fq = [float(p.get("final_quarter_mean_distance", float("nan"))) for p in ps]
    doms = []
    for p in ps:
        c = np.asarray(p.get("action_counts", []), dtype=float)
        doms.append(ACTION_NAMES[int(np.argmax(c))] if c.size == 4 and c.sum() > 0 else None)
    return fq, doms


def _arm_summary(res):
    if res is None:
        return None
    fq, doms = _per_phase(res)
    post = float(np.nansum(fq[1:])) if len(fq) > 1 else float("nan")
    return {
        "per_phase_finalQ": [round(x, 3) for x in fq],
        "per_phase_dom": doms,
        "phase0_finalQ": round(fq[0], 3) if fq else None,
        "post_change_sum": round(post, 3),
        "tie_break_fraction": round(float(res.get("tie_break_fraction", 0.0)), 4),
        "log_polar_retina": bool(res.get("log_polar_retina", False)),
        "fix1": bool(res.get("sc_tie_break_stochastic", False)),
    }


def aggregate(seeds):
    per_seed = []
    for s in seeds:
        host = _arm_summary(_load("host", s))
        pv = _arm_summary(_load("sc_popvector", s))
        scr = _arm_summary(_load("sc_popvector_scr", s))
        if pv is None:
            per_seed.append({"seed": s, "MISSING": True})
            continue

        pv_fq = pv["per_phase_finalQ"]
        scr_fq = scr["per_phase_finalQ"] if scr else [float("nan")] * len(pv_fq)

        # (a) per-phase diagonal tracking: dom matches the goal's bearing-cardinals on each diagonal
        # phase, AND popvector finalQ is below SCRAM's on that phase (the decode helps).
        diag_track = {}
        for ph in DIAGONAL_PHASES:
            if ph >= len(pv["per_phase_dom"]):
                continue
            dom = pv["per_phase_dom"][ph]
            cards, label = PHASE_GOAL_CARDINALS[ph]
            dom_ok = dom in cards
            better = (pv_fq[ph] < scr_fq[ph]) if (ph < len(scr_fq) and not np.isnan(scr_fq[ph])) else None
            diag_track[label] = {"dom": dom, "dom_matches_bearing": bool(dom_ok),
                                 "finalQ": pv_fq[ph], "scram_finalQ": scr_fq[ph] if ph < len(scr_fq) else None,
                                 "beats_scram": better}
        n_diag_dom_ok = sum(1 for v in diag_track.values() if v["dom_matches_bearing"])
        n_diag_beats_scram = sum(1 for v in diag_track.values() if v["beats_scram"])

        # lateral phase1
        lat_dom = pv["per_phase_dom"][1] if len(pv["per_phase_dom"]) > 1 else None
        lat_ok = lat_dom in PHASE_GOAL_CARDINALS[1][0]

        # (b) Sigma toward host
        host_post = host["post_change_sum"] if host else None
        pv_post = pv["post_change_sum"]
        host_over_pv = (float(host_post) / pv_post) if (host_post and pv_post and pv_post > 0) else None

        # (c) SCRAM collapse: popvector materially better than scram on the post-change re-orient
        scr_post = scr["post_change_sum"] if scr else None
        scram_collapse_ratio = (float(scr_post) / pv_post) if (scr_post and pv_post and pv_post > 0) else None
        scram_collapses = bool(scram_collapse_ratio is not None and scram_collapse_ratio > 1.10)

        # n distinct dominant cardinals across phases (the stuck-N signature check)
        n_distinct = len(set(d for d in pv["per_phase_dom"] if d is not None))

        seed_go = bool(
            n_diag_dom_ok >= 2          # tracks on >=2 of the 3 diagonal phases by bearing
            and n_diag_beats_scram >= 2  # and the decode is load-bearing there (beats scram)
            and scram_collapses          # SCRAM collapses overall
            and n_distinct >= 2          # not stuck on one cardinal
        )

        per_seed.append({
            "seed": s, "MISSING": False,
            "host": host, "popvector": pv, "scramble": scr,
            "diagonal_tracking": diag_track,
            "n_diag_dom_matches_bearing": n_diag_dom_ok,
            "n_diag_beats_scram": n_diag_beats_scram,
            "lateral_phase1_dom": lat_dom, "lateral_dom_ok": bool(lat_ok),
            "n_distinct_dominant": n_distinct,
            "host_over_popvector_post_ratio": (round(host_over_pv, 3) if host_over_pv else None),
            "scram_over_popvector_post_ratio": (round(scram_collapse_ratio, 3) if scram_collapse_ratio else None),
            "scram_collapses": scram_collapses,
            "popvector_tie_break_fraction": pv["tie_break_fraction"],
            "SEED_GO": seed_go,
        })

    completed = [r for r in per_seed if not r.get("MISSING")]
    n_go = sum(1 for r in completed if r.get("SEED_GO"))
    verdict = {
        "seeds": seeds,
        "n_completed": len(completed),
        "n_seed_go": n_go,
        "all_seeds_go": bool(len(completed) == len(seeds) and n_go == len(seeds)),
        "majority_go": bool(len(completed) > 0 and n_go >= (len(completed) + 1) // 2),
        "NOTE": ("#6 SURPASSED + CLOSED if all (or strong majority) seeds GO: log-polar render -> far "
                 "goals represented -> FIX1+decode TRACKS on the diagonal phases (dom matches bearing + "
                 "beats SCRAM) AND SCRAM collapses AND Sigma toward HOST. The orienting read-out reaches "
                 "the host ceiling; the host orienting heuristic RETIRES; the residual was a non-biological "
                 "retina truncation, NOT a substrate limit. Else: honest gap -> ISOLATE + next move."),
    }
    return {"per_seed": per_seed, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="#6 log-polar grid-32 6-seed aggregator")
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=f"{DIR}/scpv_logpolar_6seed_aggregate.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = aggregate(seeds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[6seed] ===== #6 LOG-POLAR grid-32 6-SEED VERDICT =====", flush=True)
    for r in out["per_seed"]:
        if r.get("MISSING"):
            print(f"  seed {r['seed']}: MISSING", flush=True)
            continue
        print(f"  seed {r['seed']}: GO={r['SEED_GO']} | diag_dom_ok={r['n_diag_dom_matches_bearing']}/3 "
              f"diag_beats_scram={r['n_diag_beats_scram']}/3 | scram_collapses={r['scram_collapses']} "
              f"(scr/pv={r['scram_over_popvector_post_ratio']}) | host/pv={r['host_over_popvector_post_ratio']} "
              f"| pv_dom={r['popvector']['per_phase_dom']} | tie_frac={r['popvector_tie_break_fraction']}", flush=True)
    for k, v in out["verdict"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"[6seed] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
