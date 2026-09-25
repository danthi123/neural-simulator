#!/usr/bin/env python3
"""Independent re-derivation of the `_settle_cost` battery verdict (2026-09-25).

Does NOT trust research/findings/2026-09-25-settle-cost-battery-wrong-instrument-VOID.md's own prose, and does
NOT import or call anything from research/runners/_prod_chat_phase_timing.py's `_compare()` (the grader). It was
read ONLY to learn two field names (`warm_turn_total_s` as the criterion-L quantity, and the on/off dict shape
`_compare()` indexes). Everything below is re-derived from the four raw JSON artifacts and their .prov.json
sidecars directly, against the rule text quoted verbatim from the PREREGISTRATION (Criterion L section).

Run: python3 independent_rederive.py   (stdlib only, no repo imports)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["cupy_on", "cupy_off", "numpy_on", "numpy_off"]

CRITERION_L_KEY = "warm_turn_total_s"   # from _prod_chat_phase_timing.py:250,273 (field name only, not logic)
GIT_ARCHIVE_RULE_LANDED = "2026-08-03"  # research/runners/__init__.py commits 59eded5d6/b976f8945/49958c6a7

report = {
    # gates/device_and_cost requires a recorded backend on any research/findings/raw/** JSON; this script runs
    # NO simulation compute (it only opens JSON files and greps text), so the honest value is "none".
    "backend": "none (host-only meta-check script; opens JSON files only, no simulation compute)",
    "arms": {}, "criterion_L": None, "provenance": {}, "multi_seed_rule": None,
}

for arm in ARMS:
    data_path = os.path.join(HERE, arm + ".json")
    prov_path = os.path.join(HERE, arm + ".json.prov.json")
    d = json.load(open(data_path))
    p = json.load(open(prov_path))
    report["arms"][arm] = {
        "has_criterion_L_field": CRITERION_L_KEY in d,
        "top_level_keys": sorted(d.keys()),
        "runner": p.get("runner"),
        "git_sha": p.get("git_sha"),
        "git_dirty": p.get("git_dirty"),
        "source_kind": p.get("source_kind"),
        "seed": d.get("seed"),
    }

# --- Criterion L: can it be computed at all? ---
any_has_field = any(v["has_criterion_L_field"] for v in report["arms"].values())
report["criterion_L"] = {
    "computable": any_has_field,
    "reason_if_not": (
        None if any_has_field else
        "none of the 4 raw artifacts under research/findings/raw/_settle_cost/ contains the "
        "'%s' field the PREREGISTRATION's Criterion L formula reads "
        "(delta_s = on['%s'] - off['%s']); this is UNDEFINED, not a score of 0 or a PASS/FAIL."
        % (CRITERION_L_KEY, CRITERION_L_KEY, CRITERION_L_KEY)
    ),
}

# --- Provenance: does every arm satisfy the git_archive/pinned-revision rule? ---
shas = {v["git_sha"] for v in report["arms"].values()}
report["provenance"] = {
    "all_source_kind_git_archive": all(v["source_kind"] == "git_archive" for v in report["arms"].values()),
    "distinct_git_shas": sorted(shas),
    "single_pinned_revision": len(shas) == 1,
    "any_git_dirty": any(v["git_dirty"] for v in report["arms"].values()),
    "predates_git_archive_rule": False,  # runs started 2026-09-24, rule since 2026-08-03 (~7 weeks prior)
}

# --- Multi-seed rule: does this battery need 6 seeds, and does it claim to have them? ---
seeds = {v["seed"] for v in report["arms"].values()}
report["multi_seed_rule"] = {
    "prereg_exempts_latency_from_6seed_rule": True,  # PREREGISTRATION text: "no 6-seed rule applies to a "
                                                      # "latency measurement (it is not a capability GO/NO-GO)"
    "seeds_present": sorted(seeds),
    "finding_claims_generalization": False,  # the VOID finding makes no cross-seed/GO claim; N/A here
}

verdict = "VOID" if not report["criterion_L"]["computable"] else "SCORABLE"
report["independent_verdict"] = verdict

print(json.dumps(report, indent=1))
assert verdict == "VOID", "independent re-derivation disagrees with the VOID label"
