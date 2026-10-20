"""One committed numeric record of the held-out BRAIN_MOAT_SLUG_NORM validation: reads the per-stratum measurement files and prints/writes
the numbers the report quotes, so nothing in the report is typed by hand.  Reads (all committed):
  results_strata_s1_s2.json, results_strata_s3.json            claim-gate verdicts, stock vs B2h, per stratum and subset
  composer_records_s1.json, composer_records_s3.json            RichAnswerComposer reply records (Qwen replay), flag on vs stock
  composer_records_s3_stub.json (optional)                      the same turns with the template stub mouth
  composer_stub_sweep_3000.json, stub_population_exhaustive.json  one-fact stub-mouth HURTS
  corpus.json / s3_corpus.json                                  raw-row counts of the unbuildable turns
ALLOWLIST MODE (--allowlist, 2026-10-20, prereg 2026-10-20-moat-allowlist-claim-checker): reads the *_allowlist.json files instead
(results_strata_s1_s2_allowlist.json, results_strata_s3_allowlist.json, composer_records_s1_allowlist.json,
composer_records_s3_allowlist.json, stub_population_exhaustive_allowlist.json; a missing file is skipped) and reports the three arms
stock / b2h / allowlist side by side, the prereg's bar-4 rows and the allowlist stub-population HURTS.  Without the flag the output
is exactly as before.
usage (cwd = worktree, as a module): python -m research.findings.raw._moat_slug_norm_heldout.heldout_numbers <out.json> [--allowlist]
"""
import collections, json, os, socket, sys

import research.runners  # noqa: F401  (provenance door)

H = "research/findings/raw/_moat_slug_norm_heldout/"


def load(name):
    p = H + name
    return json.load(open(p)) if os.path.exists(p) else None


ARMS = ("stock", "b2h")


def strat(R, name, sub):
    s = R["strata"][name]["summary"][sub]
    row = {}
    for arm in ARMS:
        x = s[arm]
        row[arm] = {"buildable_unique": x["buildable_unique_sentences"], "labels_in_buildable": x["labels_in_buildable"],
                    "lenient_unsupported_n": x["lenient_unsupported_n"], "lenient_leaks": x["lenient_leaks_any_seed"],
                    "lenient_upper95": x["lenient_upper95"], "strict_unsupported_n": x["strict_unsupported_n"],
                    "strict_leaks": x["strict_leaks_any_seed"], "strict_upper95": x["strict_upper95"],
                    "faithful_kept_seed42": x["answered_F_kept_seed42"],
                    "strict_leak_sentences": x["strict_leak_sentences"]}
        if arm == "allowlist":
            row[arm]["lenient_leak_sentences"] = x["lenient_leak_sentences"]
            row[arm]["like_for_like_b2h_buildable"] = {k: x["like_for_like_b2h_buildable"][k] for k in (
                "buildable_unique_sentences", "lenient_unsupported_n", "lenient_leaks_any_seed", "lenient_upper95",
                "strict_unsupported_n", "strict_leaks_any_seed", "strict_upper95")}
            row[arm]["kept_per_seed"] = {sd: {"lenient": len(v[sub]["allowlist"]["kept_lenient_leaks"]),
                                              "strict_extra_B": len(v[sub]["allowlist"]["kept_strict_extra_B"])}
                                         for sd, v in R["strata"][name]["per_seed"].items()}
    p = R["strata"][name]["per_seed"]["42"]
    row["unbuildable_turns"] = p["unbuildable_turns"]
    row["unbuildable_unique_sentences"] = p["unbuildable_unique_sentences"]
    row["unbuildable_stock_equals_b2h"] = p["unbuildable_stock_equals_b2h"]
    row["b2h_stage_by_label_seed42"] = p[sub]["b2h"]["stage_by_label"]
    row["b2h_by_label_seed42"] = p[sub]["b2h"]["by_label"]
    row["b2h_kept_per_seed"] = {sd: {"lenient": len(v[sub]["b2h"]["kept_lenient_leaks"]), "strict_extra_B": len(v[sub]["b2h"]["kept_strict_extra_B"])}
                                for sd, v in R["strata"][name]["per_seed"].items()}
    return row


def raw_rows_in_turns(corpus, turns):
    return sum(len(t.get("calls", [])) for t in corpus["turns"] if t["label"] in turns)


def main(out, allowlist=False):
    global ARMS
    sfx = "_allowlist" if allowlist else ""
    if allowlist:
        ARMS = ("stock", "b2h", "allowlist")
    r12 = load("results_strata_s1_s2%s.json" % sfx); r3 = load("results_strata_s3%s.json" % sfx)
    out_d = {"_doc": "numbers behind the held-out BRAIN_MOAT_SLUG_NORM report (module docstring)", "backend": "numpy", "device": "cpu",
             "host": socket.gethostname(), "script": H + "heldout_numbers.py", "argv": sys.argv,
             "seeds_note": "parser seeds 42,43,44,100,101,102,1 of one deterministic brain build: a single replicate per sentence, not independent draws",
             "verifier_level": {}, "composer_path": {}, "unbuildable": {}}
    if allowlist:
        out_d["_doc"] = "numbers behind the allowlist claim-checker prereg (stock / b2h / allowlist; module docstring)"
        out_d["arms"] = list(ARMS)
        out_d["bar4_f_single_triple"] = {}
    for nm, R in (("S1", r12), ("S2", r12), ("S3", r3)):
        if R is None or nm not in R["strata"]:
            continue
        if allowlist and "bar4_f_single_triple" in R["strata"][nm]:
            b4 = R["strata"][nm]["bar4_f_single_triple"]
            out_d["bar4_f_single_triple"][nm] = {k: b4[k] for k in ("n", "kept_by_allowlist_seed42", "rejected", "n_f_kept_by_b2h_multi_triple")}
        subs = ["all"] + (["one_fact", "S3_member_1to3", "four_plus"] if nm == "S3" else [])
        out_d["verifier_level"][nm] = {sub: strat(R, nm, sub) for sub in subs}
        out_d["verifier_level"][nm]["label_totals"] = R["strata"][nm]["label_totals"]
        out_d["verifier_level"][nm]["turns_by_subset"] = R["strata"][nm]["turns_by_subset"]
    if r12 is not None and "S2" in r12["strata"]:
        S2 = r12["strata"]["S2"]["per_seed"]["42"]["all"]["b2h"]
        out_d["verifier_level"]["S2"]["stage_by_variant_b2h_seed42"] = S2["stage_by_variant"]
        out_d["verifier_level"]["S2"]["stage_by_variant_stock_seed42"] = r12["strata"]["S2"]["per_seed"]["42"]["all"]["stock"]["stage_by_variant"]
        if allowlist:
            out_d["verifier_level"]["S2"]["stage_by_variant_allowlist_seed42"] = (
                r12["strata"]["S2"]["per_seed"]["42"]["all"]["allowlist"]["stage_by_variant"])
    # unbuildable rows
    c1 = load("corpus.json"); c3 = load("s3_corpus.json")
    for nm, R, c in (("S1", r12, c1), ("S2", r12, load("s2_corpus.json")), ("S3", r3, c3)):
        if R is None or nm not in R["strata"]:
            continue
        p = R["strata"][nm]["per_seed"]["42"]
        t = p["unbuildable_turns"]
        out_d["unbuildable"][nm] = {"turns": t, "unique_sentences": p["unbuildable_unique_sentences"],
                                    "raw_rows": raw_rows_in_turns(c, t), "labels": {l: v.get("UNBUILDABLE", 0) for l, v in p["all"]["b2h"]["by_label"].items()},
                                    "same_set_stock_and_b2h": p["unbuildable_stock_equals_b2h"]}
    for nm, fn in (("S1", "composer_records_s1.json"), ("S3", "composer_records_s3.json"), ("S3_stub", "composer_records_s3_stub.json"),
                   ("S1_stub", "composer_records_s1_stub.json")):
        if allowlist:
            fn = fn.replace(".json", "_allowlist.json")
        d = load(fn)
        if d:
            out_d["composer_path"][nm] = {k: d[k] for k in ("n_turns", "status_counts", "status_by_gathered_size", "stock_answered", "flag_on_answered",
                                                             "replay_validation", "replay_misses_flag_on", "replay_misses_flag_off")}
            out_d["composer_path"][nm]["HURTS_turns"] = [{"label": t["label"], "entity": t["entity"], "n_gathered": t["n_gathered"],
                                                          "stock_answer": t["stock_live"]["answer"], "flag_on_answer": t["flag_on_replay"]["answer"]}
                                                         for t in d["HURTS_turns"]]
            out_d["composer_path"][nm]["fell_back_to_single_triple_candidates"] = sum(t["flag_on_replay"]["fell_back_to_single_triple"] for t in d["turns"])
            out_d["composer_path"][nm]["unbuildable_turns_answered_flag_on"] = [t["label"] for t in d["turns"]
                                                                                if t["flag_on_replay"]["fell_back_to_single_triple"] and not t["flag_on_replay"]["abstained"]]
            if allowlist:
                out_d["composer_path"][nm]["arm"] = d.get("arm")
                out_d["composer_path"][nm]["b2h_answered"] = d.get("b2h_answered")
                out_d["composer_path"][nm]["LOSES_vs_b2h_turns"] = [t["label"] for t in d.get("LOSES_vs_b2h_turns", [])]
    sw = None if allowlist else load("composer_stub_sweep_3000.json")
    pop = load("stub_population_exhaustive_allowlist.json" if allowlist else "stub_population_exhaustive.json")
    out_d["stub_one_fact"] = {}
    for nm, d in (("composer_sweep_3000", sw), ("population_exhaustive", pop)):
        if d:
            out_d["stub_one_fact"][nm] = {k: d[k] for k in ("n_facts", "cells(stock_ok,flag_on_ok)", "stock_answer_rate", "flag_on_answer_rate", "HURTS", "HELPS",
                                                            "flag_on_fallback_to_single_triple", "hurts_reasons") if k in d}
            out_d["stub_one_fact"][nm]["hurts_surfaces"] = [h.get("stock_surface") or h.get("surface") for h in d["hurts_rows"]]
            if "hurts_by_relation" in d:
                out_d["stub_one_fact"][nm]["hurts_by_relation"] = d["hurts_by_relation"]
            if d.get("composer_sample_agreement"):
                out_d["stub_one_fact"][nm]["composer_sample_agreement"] = d["composer_sample_agreement"]
            if allowlist:
                out_d["stub_one_fact"][nm].update({k: d.get(k) for k in (
                    "arms", "limit", "n_population", "cells(stock_ok,allowlist_ok)", "allowlist_answer_rate", "HURTS_allowlist",
                    "HELPS_allowlist", "allowlist_unbuildable_rejected", "allowlist_hurts_reasons", "allowlist_hurts_by_relation")})
                out_d["stub_one_fact"][nm]["allowlist_hurts_surfaces"] = [h["surface"] for h in d.get("allowlist_hurts_rows", [])]
    json.dump(out_d, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out_d.items() if k not in ("verifier_level",)}, indent=1, default=str)[:6000])
    for nm, d in out_d["verifier_level"].items():
        for sub, x in d.items():
            if allowlist and isinstance(x, dict) and "allowlist" in x:
                a = x["allowlist"]
                print("%s/%s ALLOWLIST: sentences %d | lenient %d/%d (95%% upper %.4f) | strict %d/%d (95%% upper %.4f) | F kept %d of %d F" % (
                    nm, sub, a["buildable_unique"], a["lenient_leaks"], a["lenient_unsupported_n"], a["lenient_upper95"] or -1,
                    a["strict_leaks"], a["strict_unsupported_n"], a["strict_upper95"] or -1, a["faithful_kept_seed42"],
                    a["labels_in_buildable"].get("F", 0)))
            if isinstance(x, dict) and "b2h" in x:
                b = x["b2h"]
                print("%s/%s: buildable %d | lenient %d/%d (95%% upper %.4f) | strict %d/%d (95%% upper %.4f) | F kept %d of %d F | stock lenient %d" % (
                    nm, sub, b["buildable_unique"], b["lenient_leaks"], b["lenient_unsupported_n"], b["lenient_upper95"] or -1,
                    b["strict_leaks"], b["strict_unsupported_n"], b["strict_upper95"] or -1, b["faithful_kept_seed42"],
                    b["labels_in_buildable"].get("F", 0), x["stock"]["lenient_leaks"]))


if __name__ == "__main__":
    main(sys.argv[1], allowlist="--allowlist" in sys.argv[2:])
