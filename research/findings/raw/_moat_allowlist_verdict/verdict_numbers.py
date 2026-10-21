"""Numbers behind the VERDICT sections of the moat allowlist claim-checker prereg (builds 1-5).

Reads only committed JSON in this tree, plus two records passed by path: the build-5 mutation-round summary (written by
the mutation workflow's own builder script, copied here as `mutation_build5_summary.json` with its source sha256) and the
pytest junit record of the moat suite at the build-5 code. Writes one JSON (`--out`) that every number in the verdict
sections can be checked against. No number here is typed by hand.

Run: python -m research.findings.raw._moat_allowlist_verdict.verdict_numbers --out <json> --junit <xml> \
         --mutation <summary_b5.json>
"""
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import research.runners as _door  # provenance door: run record + sidecar for --out and declared outputs

RAW = "research/findings/raw"
HELD = RAW + "/_moat_slug_norm_heldout"
H3 = RAW + "/_moat_allowlist_hunt3"
H4 = RAW + "/_moat_allowlist_hunt4"
HERE = RAW + "/_moat_allowlist_verdict"

# build -> its registered blind round, its code commit (the matcher measured), and the file suffix of its measurements
BUILDS = {
    1: {"round": "r3", "hunt": H3 + "/results.json", "suffix": "_allowlist", "code": "9b8ade5a2"},
    2: {"round": "r4", "hunt": H4 + "/r4/results_build2.json", "suffix": "_allowlist2", "code": "98f3d59eb"},
    3: {"round": "r5", "hunt": H4 + "/r5/results_build3.json", "suffix": "_allowlist3", "code": "95de4db59"},
    4: {"round": "r6", "hunt": H4 + "/r6/results_build4.json", "suffix": "_allowlist4", "code": "43e111c48"},
    5: {"round": "r7", "hunt": H4 + "/r7/results_build5.json", "suffix": "_allowlist5", "code": "ef51d6243"},
}
# non-blind re-runs: the earlier rounds' cases shaped the later builds, so these are reported, never a bar
RERUNS = {
    2: {"r3": H3 + "/results_allowlist2.json"},
    3: {"r3": H4 + "/r3_rerun_build3.json", "r4": H4 + "/r4/results_build3.json"},
    4: {"r3": H4 + "/r3_rerun_build4.json", "r4": H4 + "/r4/results_build4.json", "r5": H4 + "/r5/results_build4.json"},
    5: {"r3": H4 + "/r3_rerun_build5.json", "r4": H4 + "/r4/results_build5.json", "r5": H4 + "/r5/results_build5.json",
        "r6": H4 + "/r6/results_build5.json"},
}
CASES = {"r3": H3, "r4": H4 + "/r4", "r5": H4 + "/r5", "r6": H4 + "/r6", "r7": H4 + "/r7"}


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _at(p):
    """The commit its provenance sidecar recorded (the code that produced it), or None without a sidecar."""
    try:
        return _load(p + ".prov.json").get("git_sha")
    except (OSError, ValueError):
        return None


def upper95(k, n):
    """One-sided 95% upper bound: rule of three at k == 0 (the prereg's section 2.1), Clopper-Pearson otherwise."""
    if n <= 0:
        return None
    if k == 0:
        return 3.0 / n
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):  # bisect P(X <= k | p) = 0.05
        mid = (lo + hi) / 2
        cdf, term = 0.0, (1 - mid) ** n
        for i in range(k + 1):
            if i:
                term *= (n - i + 1) / i * mid / (1 - mid)
            cdf += term
        lo, hi = (mid, hi) if cdf > 0.05 else (lo, mid)
    return hi


def hunt_summary(p):
    d = _load(p)
    s = d["summary"]
    out = {"file": p, "sha256": _sha(p), "measured_at": _at(p), "host": d.get("host")}
    if "entries" in s["allowlist"]:  # round-4+ runner: per entry (claim gate, composer), VALID, controls
        out["exit_code"] = d.get("exit_code")
        out["controls_all_ok"] = d.get("controls", {}).get("ALL_OK")
        for arm in ("stock", "b2h", "allowlist"):
            a = s[arm]
            r = {"VALID": a.get("VALID"), "n_undefined_case_entries": a.get("n_undefined_case_entries")}
            for lab in ("U", "B", "F"):
                g, c = a["entries"]["gate"][lab], a["entries"]["composer"][lab]
                either = sorted(set(g["kept_ids"]) | set(c["kept_ids"]))
                r[lab] = {"n": g["n"], "kept_gate": g["kept"], "kept_composer": c["kept"], "kept_either_entry": len(either),
                          "kept_ids_either_entry": either, "undefined": [g["UNDEFINED"], c["UNDEFINED"]]}
            out[arm] = r
    else:  # round-3 runner (run_hunt3.py): kept at any seed per entry, an error count per arm
        for arm in ("stock", "b2h", "allowlist"):
            a = s[arm]
            r = {"errors": a.get("errors"), "VALID": a.get("errors") == 0}
            for lab in ("U", "B", "F"):
                x = a[lab]
                r[lab] = {"n": x["n"], "kept_gate": x["kept_gate_any_seed"], "kept_composer": x["kept_composer_any_seed"],
                          "kept_either_entry": max(len(x["kept_ids"]), x["kept_gate_any_seed"], x["kept_composer_any_seed"]),
                          "kept_ids_either_entry": sorted(x["kept_ids"])}
            out[arm] = r
    al = out["allowlist"]
    out["allowlist_U_kept"] = al["U"]["kept_either_entry"]
    out["allowlist_U_n"] = al["U"]["n"]
    out["allowlist_U_upper95"] = upper95(al["U"]["kept_either_entry"], al["U"]["n"])
    out["allowlist_strict_kept"] = al["U"]["kept_either_entry"] + al["B"]["kept_either_entry"]
    out["allowlist_strict_n"] = al["U"]["n"] + al["B"]["n"]
    out["allowlist_strict_upper95"] = upper95(out["allowlist_strict_kept"], out["allowlist_strict_n"])
    out["allowlist_F_kept"] = al["F"]["kept_either_entry"]
    out["allowlist_F_n"] = al["F"]["n"]
    return out


def case_counts(rnd):
    cases = _load(CASES[rnd] + "/cases.json")
    adj = _load(CASES[rnd] + "/adjudication.json")
    exp = {}
    for c in cases:
        lab = c.get("label") or c.get("expect")
        exp[lab] = exp.get(lab, 0) + 1
    return {"cases": len(cases), "skeptic_labels": exp, "adjudicated": adj.get("final_counts"),
            "relabelled_to_B": len(adj.get("relabel_to_B", []))}


def strata(suffix):
    hn = HELD + "/heldout_numbers%s.json" % suffix
    if not os.path.exists(hn):
        return {"heldout_numbers": None, "note": "heldout_numbers%s.json not in this tree yet" % suffix}
    d = _load(hn)
    v = d["verifier_level"]
    out = {"heldout_numbers": hn, "sha256": _sha(hn), "measured_at": _at(hn)}
    for s in ("S1", "S2", "S3"):
        a = v[s]["all"]["allowlist"]
        out[s] = {k: a[k] for k in ("buildable_unique", "lenient_unsupported_n", "lenient_leaks", "lenient_upper95",
                                    "strict_unsupported_n", "strict_leaks", "strict_upper95", "faithful_kept_seed42")}
        out[s]["stock_faithful_kept_seed42"] = v[s]["all"]["stock"]["faithful_kept_seed42"]
        out[s]["b2h_faithful_kept_seed42"] = v[s]["all"]["b2h"]["faithful_kept_seed42"]
    cp = d.get("composer_path", {})
    out["composer"] = {}
    for s in ("S1", "S3"):
        x = cp.get(s)
        if x:
            out["composer"][s] = {"n_turns": x["n_turns"], "stock_answered": x["stock_answered"],
                                  "b2h_answered": x["b2h_answered"], "allowlist_answered": x["flag_on_answered"],
                                  "HURTS": len(x["HURTS_turns"]), "LOSES_vs_b2h": len(x["LOSES_vs_b2h_turns"]),
                                  "replay_misses": len(x["replay_misses_flag_on"]) + len(x["replay_misses_flag_off"]),
                                  "status_counts": x["status_counts"]}
    b4 = d.get("bar4_f_single_triple", {})
    out["bar4"] = {s: {"n": b4[s]["n"], "kept": b4[s]["kept_by_allowlist_seed42"]} for s in b4 if "n" in b4[s]}
    return out


def stub(suffix):
    p = HELD + "/stub_population_exhaustive%s.json" % suffix
    d = _load(p)
    keys = ("n_population", "stock_answer_rate", "allowlist_answer_rate", "HURTS_allowlist", "HELPS_allowlist",
            "allowlist_role_reversed_stock_leaks", "allowlist_meaning_drift_stock_leaks", "relation_table_sha256")
    out = {"file": p, "sha256": _sha(p), "measured_at": _at(p)}
    out.update({k: d.get(k) for k in keys})
    cells = d.get("cells(stock_ok,allowlist_ok)") or {}
    out["unvoiceable_under_flag"] = cells.get("(True, False)")
    out["role_reversed_by_relation"] = d.get("allowlist_role_reversed_by_relation")
    out["meaning_drift_by_relation"] = d.get("allowlist_meaning_drift_by_relation")
    return out


def junit(p):
    root = ET.parse(p).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    files = {}
    for case in root.iter("testcase"):
        f = (case.get("classname") or "").split(".")[1] if "." in (case.get("classname") or "") else case.get("classname")
        bad = any(ch.tag in ("failure", "error") for ch in case)
        r = files.setdefault(f, {"tests": 0, "not_passed": 0})
        r["tests"] += 1
        r["not_passed"] += int(bad)
    named = {}
    for case in root.iter("testcase"):
        n = case.get("name") or ""
        if any(t in n for t in ("bank", "board_265", "board_266", "golden", "flag_off", "must_keep")):
            named["%s::%s" % (case.get("classname"), n)] = not any(ch.tag in ("failure", "error") for ch in case)
    return {"file": p, "sha256": _sha(p), "totals": tot, "by_file": files, "bar_tests": named}


def bank():
    p = "tests/fixtures/moat_allowlist_bank.json"
    d = _load(p)
    exp_u = {src: c.get("U", 0) for src, c in d["counts_by_source"].items()}
    mk = _load("tests/fixtures/moat_allowlist_must_keep.json")
    return {"file": p, "sha256": _sha(p), "expect_U_by_source": exp_u, "expect_U_total": sum(exp_u.values()),
            "must_keep_single_triple": mk.get("n_single_triple"), "must_keep_allowlist_losses": len(mk.get("allowlist_losses", [])),
            "must_keep_kept": mk.get("n_single_triple", 0) - len(mk.get("allowlist_losses", []))}


def mutation(src, dst):
    d = _load(src)
    keep = {k: d[k] for k in ("commit", "n_mutants", "killed_by_corpus", "killed_by_pytest_after_corpus", "killed_total",
                              "survived_total", "pytest_command", "baseline", "witness_corpus_items") if k in d}
    for k in ("survived_equivalent", "survived_not_equivalent", "survived_unsure"):
        keep["n_" + k] = len(d.get(k, []))
    rec = {"_doc": "copy of the build-5 mutation round's summary (workflow wf_7282d984-21c, builder build_results5.py "
                   "in the record directory beside this file), made by verdict_numbers.py", "runner": __name__,
           "source_file": src, "source_sha256": _sha(src), "summary": d}
    with open(dst, "w") as fh:
        json.dump(rec, fh, indent=1)
    _door.declare_output(dst)
    keep["record"] = dst
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--junit", required=True)
    ap.add_argument("--mutation", required=True)
    a = ap.parse_args()
    res = {"_doc": __doc__.strip().splitlines()[0], "script": __name__, "argv": vars(a),
           "bound_rule": "one-sided 95% upper bound: rule of three at 0 kept, Clopper-Pearson otherwise (prereg 2.1)",
           "kept_rule": "a hunt case counts as kept when the allowlist keeps it at ANY of the seven seeds through EITHER entry "
                        "(claim gate or composer), the stricter of the two readings; round 3's runner records kept ids at "
                        "any seed",
           "builds": {}}
    for b, cfg in BUILDS.items():
        r = {"code": cfg["code"], "blind_round": cfg["round"], "hunt": hunt_summary(cfg["hunt"]),
             "hunt_cases": case_counts(cfg["round"]), "strata": strata(cfg["suffix"]), "stub": stub(cfg["suffix"]),
             "nonblind_reruns": {k: hunt_summary(p) for k, p in RERUNS.get(b, {}).items()}}
        r["bar3_pass"] = (r["hunt"]["allowlist_U_kept"] == 0 and r["hunt"]["allowlist"]["VALID"] is True
                          and r["hunt"].get("controls_all_ok", True) is not False)
        res["builds"][str(b)] = r
    res["bank_and_must_keep"] = bank()
    res["build5_moat_suite"] = junit(a.junit)
    res["build5_mutation"] = mutation(a.mutation, HERE + "/mutation_build5_summary.json")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for b, r in res["builds"].items():
        h = r["hunt"]
        print("build %s %s: U kept %d/%d (upper95 %.4f) strict %d/%d F %d/%d bar3_pass=%s | stub HURTS %s answer %.4f"
              % (b, r["blind_round"], h["allowlist_U_kept"], h["allowlist_U_n"], h["allowlist_U_upper95"],
                 h["allowlist_strict_kept"], h["allowlist_strict_n"], h["allowlist_F_kept"], h["allowlist_F_n"],
                 r["bar3_pass"], r["stub"]["HURTS_allowlist"], r["stub"]["allowlist_answer_rate"]))


if __name__ == "__main__":
    main()
