"""Before/after numbers for the 2026-10-18 slug_norm predicate-position fix (`_noun_before_verb`): the verifier-level bars on S1/S2/S3 and the
one-fact stub-mouth population, read from committed measurement files, so the report quotes nothing typed by hand.
  before: results_strata_s1_s2.json, results_strata_s3.json, stub_population_exhaustive.json, composer_stub_sweep_3000.json
  after : results_strata_s1_s2_after.json, results_strata_s3_after.json, stub_population_exhaustive_after.json, composer_stub_sweep_3000_after.json
usage (cwd = worktree, as a module): python -m research.findings.raw._moat_slug_norm_heldout.followup_numbers <out.json>"""
import json, os, socket, sys

import research.runners  # noqa: F401  (provenance door)

H = "research/findings/raw/_moat_slug_norm_heldout/"


def L(n):
    p = H + n
    return json.load(open(p)) if os.path.exists(p) else None


def bars(before, after, names):
    out = {}
    for n in names:
        row = {}
        identical = True
        for sd, pa in before["strata"][n]["per_seed"].items():
            pb = after["strata"][n]["per_seed"][sd]
            for sub in pa:
                if not isinstance(pa[sub], dict) or "b2h" not in pa[sub]:
                    continue
                for arm in ("stock", "b2h"):
                    for k in ("by_label", "stage_by_label", "kept_lenient_leaks", "kept_strict_extra_B"):
                        if pa[sub][arm][k] != pb[sub][arm][k]:
                            identical = False
        for tag, R in (("before", before), ("after", after)):
            s = R["strata"][n]["summary"]["all"]["b2h"]
            row[tag] = {"lenient_leaks": s["lenient_leaks_any_seed"], "lenient_n": s["lenient_unsupported_n"],
                        "strict_leaks": s["strict_leaks_any_seed"], "strict_n": s["strict_unsupported_n"],
                        "faithful_kept_seed42": s["answered_F_kept_seed42"]}
        row["every_verdict_identical_all_seeds_subsets_arms"] = identical
        out[n] = row
    return out


def pop(d):
    if not d:
        return None
    return {k: d[k] for k in ("n_facts", "cells(stock_ok,flag_on_ok)", "HURTS", "HELPS", "flag_on_answer_rate", "flag_on_fallback_to_single_triple", "hurts_reasons")
            if k in d} | {"hurts": [h.get("surface") or h.get("stock_surface") for h in d["hurts_rows"]]}


def main(out):
    r = {"_doc": "before/after numbers for the slug_norm predicate-position fix (module docstring)", "backend": "numpy", "device": "cpu",
         "host": socket.gethostname(), "script": H + "followup_numbers.py", "argv": sys.argv,
         "verifier_bars": {**bars(L("results_strata_s1_s2.json"), L("results_strata_s1_s2_after.json"), ("S1", "S2")),
                           **bars(L("results_strata_s3.json"), L("results_strata_s3_after.json"), ("S3",))},
         "stub_population_before": pop(L("stub_population_exhaustive.json")), "stub_population_after": pop(L("stub_population_exhaustive_after.json")),
         "composer_sweep_3000_before": pop(L("composer_stub_sweep_3000.json")), "composer_sweep_3000_after": pop(L("composer_stub_sweep_3000_after.json")),
         "population_vs_composer_sample_after": (L("stub_population_exhaustive_after.json") or {}).get("composer_sample_agreement")}
    json.dump(r, open(out, "w"), indent=1)
    for k, v in r.items():
        if k not in ("_doc", "argv"):
            print(k, json.dumps(v, default=str)[:900])


if __name__ == "__main__":
    main(sys.argv[1])
