"""Numbers behind the VERDICT sections of the moat allowlist claim-checker prereg (builds 1-7).

Reads only committed JSON in this tree, plus records passed by path: the build-5 mutation-round summary (written by
the mutation workflow's own builder script, copied here as `mutation_build5_summary.json` with its source sha256) and the
pytest junit record of the moat suite at each build's code (`--junit` is build 5's, `--junit-build N=<xml>` any build's).
Writes one JSON (`--out`) that every number in the verdict sections can be checked against. No number here is typed by
hand.

Builds 6 and 7 (rounds 8 and 9) add the reply entry (a third entry besides the claim gate and the composer candidate):
a case is kept when ANY of the three entries keeps it, a reply case counting through the label of the selection that
reaches the user. Each round's own count rule comes from its adjudication.json (round 9: `known_residue` is reported,
not counted) and is cross-checked against that file's `bar3_count_note`. Build 7 was never measured on bars 1, 4 and 5:
those cells read "not measured", never zero. For the kept-B list of round 9 the displayed name spans are recomputed by the
pure matcher in this tree, only when the tree's matcher equals the declared code version (diff recorded in the output).

Run: python -m research.findings.raw._moat_allowlist_verdict.verdict_numbers --out <json> [--junit <xml>] \
         [--junit-build N=<xml> ...] [--mutation <summary_b5.json>] [--builds 1,2,3,4,5]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
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
    # build 6: the stub population is the code-version re-run (`_tip`); the older stub_population_exhaustive_allowlist6.json is a
    # pre-amendment diagnostic (code b7938ef42, 5 files differ from 2b7ab8c1d) and is never read here
    6: {"round": "r8", "hunt": H4 + "/r8/results_build6.json", "suffix": "_allowlist6", "stub_suffix": "_allowlist6_tip",
        "code": "2b7ab8c1d", "not_counted_classes": []},
    # build 7 failed bar 3 (round 9) and was never measured on bars 1, 4 and 5: no strata, stub or replay suffix
    7: {"round": "r9", "hunt": H4 + "/r9/results_build7.json", "suffix": None, "code": "423f25050",
        "not_counted_classes": ["known_residue"]},
}
# non-blind re-runs: the earlier rounds' cases shaped the later builds, so these are reported, never a bar
RERUNS = {
    2: {"r3": H3 + "/results_allowlist2.json"},
    3: {"r3": H4 + "/r3_rerun_build3.json", "r4": H4 + "/r4/results_build3.json"},
    4: {"r3": H4 + "/r3_rerun_build4.json", "r4": H4 + "/r4/results_build4.json", "r5": H4 + "/r5/results_build4.json"},
    5: {"r3": H4 + "/r3_rerun_build5.json", "r4": H4 + "/r4/results_build5.json", "r5": H4 + "/r5/results_build5.json",
        "r6": H4 + "/r6/results_build5.json"},
    6: {"r3": H4 + "/r3_rerun_build6.json", "r4": H4 + "/r4/results_build6.json", "r5": H4 + "/r5/results_build6.json",
        "r6": H4 + "/r6/results_build6.json", "r7": H4 + "/r7/results_build6.json"},
}
CASES = {"r3": H3, "r4": H4 + "/r4", "r5": H4 + "/r5", "r6": H4 + "/r6", "r7": H4 + "/r7", "r8": H4 + "/r8", "r9": H4 + "/r9"}
CODE_PATHS = ("research/runners", "webapp", "sim")   # the code a build's measurements must share (prereg Amendment 6)


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


def code_diff(declared, sha, tree=False):
    """'empty' when the code under research/runners, webapp and sim is identical between the commit `declared` and `sha` (or, with
    tree=True, between `declared` and the working tree, uncommitted edits included); 'differs: <stat>' or 'unavailable: <why>'
    otherwise; None without both ends."""
    if not declared or not sha:
        return None
    if sha == declared and not tree:
        return "empty"
    args = ["git", "diff", "--stat", declared] + ([] if tree else [sha]) + ["--"] + list(CODE_PATHS)
    try:
        r = subprocess.run(args, capture_output=True, text=True)
    except OSError as exc:
        return "unavailable: %s" % exc
    if r.returncode != 0:
        return "unavailable: " + r.stderr.strip()[:120]
    return "empty" if not r.stdout.strip() else "differs: " + r.stdout.strip().splitlines()[-1].strip()


def code_check(p, declared):
    """The sidecar's commit for artifact `p`, whether that tree was dirty, and whether the committed code at that commit equals the
    build's declared code version (a dirty tree cannot be checked from git: the flag is recorded beside the diff)."""
    try:
        side = _load(p + ".prov.json")
    except (OSError, ValueError):
        return {"measured_at": None, "git_dirty": None, "declared_code": declared, "code_diff_vs_declared": None}
    sha = side.get("git_sha")
    return {"measured_at": sha, "git_dirty": side.get("git_dirty"), "declared_code": declared,
            "code_diff_vs_declared": code_diff(declared, sha)}


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


def _is_reply_dict(x):
    return isinstance(x, dict)


def _sel_key(subset):
    """The subset_labels key of a 1-based selection ('1+3'; a single sentence '2')."""
    return "+".join(str(i) for i in subset)


def _reply_rows_kept(rows, arm, subset_labels):
    """{label: set of case ids} the reply entry kept in `arm`, recomputed from the per-record rows: a case counts for a label when,
    at any seed on either path, the text that reached the user is an in-order selection whose adjudicated label is that label
    (a record that emitted nothing, j = 0, counts for no label). Cross-checks the harness's own summary."""
    out = {"U": set(), "B": set(), "F": set()}
    for r in rows:
        recs = (r.get("reply") or {}).get(arm) if _is_reply_dict(r.get("reply")) else None
        for x in recs or []:
            sub = x.get("subset") or []
            if x.get("status") not in ("prefix", "subsequence") or not sub:
                continue
            lab = (subset_labels.get(r["id"]) or {}).get(_sel_key(sub))
            if lab in out:
                out[lab].add(r["id"])
    return out


def _reply_selections(row, arm, sl, lab):
    """The distinct texts that reached the user in `arm` through the reply entry whose selection label is `lab`, with how many
    (seed, path) records emitted each: [{selection, label, text, n_records, paths, seeds}]."""
    found = {}
    recs = (row.get("reply") or {}).get(arm) if _is_reply_dict(row.get("reply")) else None
    for x in recs or []:
        sub = x.get("subset") or []
        if x.get("status") not in ("prefix", "subsequence") or not sub or sl.get(_sel_key(sub)) != lab:
            continue
        e = found.setdefault((_sel_key(sub), x["text"]), {"selection": _sel_key(sub), "label": lab, "text": x["text"],
                                                           "n_records": 0, "paths": set(), "seeds": set()})
        e["n_records"] += 1
        e["paths"].add(x["path"])
        e["seeds"].add(x["seed"])
    return [dict(e, paths=sorted(e["paths"]), seeds=sorted(e["seeds"])) for e in found.values()]


_MATCHER = {}


def matcher(declared):
    """The pure matcher of this tree, only when the tree (uncommitted edits included) equals the declared code version; else None."""
    if declared not in _MATCHER:
        diff = code_diff(declared, "tree", tree=True)
        mod = None
        if diff == "empty":
            import research.runners._moat_allowlist as mod   # noqa: E402  (imported late: only build 7's spans need it)
        _MATCHER[declared] = (mod, diff)
    return _MATCHER[declared]


def displayed_spans(text, gathered, declared):
    """The name spans the matcher reads in `text` against this turn's gathered triples, one entry per sentence:
    {sentence, keep, reason, spans: [{slug, displayed}] (the entity spans in their displayed spelling, in order),
    ent_span_words (the matcher's own lower-case word split)}; plus whether the whole text is accepted by `allowlist_verify` (the
    claim gate's function), as a reproduction check. {"available": False, ...} when the tree's matcher is not the declared code."""
    mod, diff = matcher(declared)
    if mod is None:
        return {"available": False, "why": "matcher in this tree vs %s: %s" % (declared, diff)}
    g = [tuple(x) for x in gathered]
    out = {"available": True, "matcher_vs_declared": diff, "sentences": [],
           "whole_text_accepted_by_allowlist_verify": bool(mod.allowlist_verify(text, g)["accepted"])}
    for sent in mod.split_sentences(text):
        r = mod.allowlist_check(sent, g)
        row = {"sentence": sent, "keep": bool(r["keep"]), "reason": r["reason"] if not r["keep"] else "ok",
               "ent_span_words": r.get("ent_spans")}
        try:   # the displayed (original-case) spelling of each entity span: the matcher's own parse, read back through its token map
            lex = mod._lexicon(mod._key_of(g))
            toks = mod._TOKEN.findall(lex.normalise(sent))
            if toks and toks[-1] == ".":
                toks = toks[:-1]
            parses = mod._parses(toks, lex, mod._Fail())
            origs = lex.normalise_tracked(sent)[2]
            row["spans"] = [{"slug": slug, "displayed": origs[i]} for kind, i, slug in parses[0][2] if kind == "ent"]
        except Exception as exc:   # pragma: no cover - diagnostic only; the lower-case words above stay
            row["spans"] = None
            row["spans_error"] = "%s: %s" % (type(exc).__name__, exc)
        out["sentences"].append(row)
    return out


def _case_detail(i, cases, adj, rows, arm, lab, hit, span_code=None):
    """One kept case, for the verdict text: its class, labels, the text that reached the user through each entry that kept it,
    the judges' words, and (span_code given: the declared code the tree's matcher equals) the displayed spans of that text."""
    c, row = cases[i], rows[i]
    sl = (adj.get("subset_labels") or {}).get(i) or {}
    det = {"id": i, "class": c["class"], "skeptic_label": c["expect"], "case_label": adj["final_labels"][i],
           "sentence": c["sentence"], "gathered": c["gathered"], "kept_through": sorted(k for k, v in hit.items() if v),
           "judges": adj.get("judge_labels", {}).get(i), "relabel_reason": (adj.get("reasons") or {}).get(i),
           "skeptic_why": c.get("why")}
    if c.get("reply"):
        det["reply_sentences"] = c.get("sentences")
    det["reply_selections"] = _reply_selections(row, arm, sl, lab) if hit.get("reply") else []
    if span_code:
        texts = []
        if hit.get("gate") or hit.get("composer"):
            texts.append(("case sentence", c["sentence"]))
        texts += [("reply selection %s" % e["selection"], e["text"]) for e in det["reply_selections"]]
        det["spans"] = [{"of": what, "text": t, **displayed_spans(t, c["gathered"], span_code)} for what, t in texts]
    return det


def _note_counts(note):
    """The U and B counts an adjudication's bar3_count_note states ('114 cases are U', '86 counted cases are U', '88 are B',
    '23 counted B'), or None for one it does not state."""
    mu = re.search(r"(\d+) (?:counted )?cases are U", note or "")
    mb = re.search(r"(\d+) (?:counted )?(?:cases are B|are B|B\b)", (note or "").split("cases are U", 1)[-1])
    return {"U": int(mu.group(1)) if mu else None, "B": int(mb.group(1)) if mb else None}


def _reply_round(out, d, rnd, not_counted, declared):
    """Round 8 on: the reply entry. Per arm, a case is kept when ANY of the claim gate, the composer candidate or the reply entry
    keeps it (a reply case through the label of the selection that reached the user): the strict union. The count rule is the
    round's own (`not_counted` classes are reported, not counted; a B counts only when the skeptic wrote it expect-U), cross-checked
    against its adjudication.json bar3_count_note."""
    cases = {c["id"]: c for c in _load(CASES[rnd] + "/cases.json")}
    adj = _load(CASES[rnd] + "/adjudication.json")
    final, sub = adj["final_labels"], adj.get("subset_labels") or {}
    s = d["summary"]
    rows = {r["id"]: r for r in d["rows"]}
    counted = {lab: {i for i, l in final.items() if l == lab and cases[i]["expect"] == "U" and cases[i]["class"] not in not_counted}
               for lab in ("U", "B")}
    note = _note_counts(adj.get("bar3_count_note"))
    out["count_rule"] = {"not_counted_classes": list(not_counted), "counted_U_n": len(counted["U"]), "counted_B_n": len(counted["B"]),
                         "adjudication_note_counts": note, "agrees_with_note": all(note[k] in (None, len(counted[k])) for k in note),
                         "note": adj.get("bar3_count_note"),
                         "U_not_counted_n": sum(1 for i, l in final.items() if l == "U" and i not in counted["U"]),
                         "B_not_counted_n": sum(1 for i, l in final.items() if l == "B" and i not in counted["B"])}
    mod, mdiff = matcher(declared) if declared else (None, "no declared code")
    span_code = declared if mod is not None else None
    out["displayed_spans"] = {"declared_code": declared, "tree_matcher_vs_declared": mdiff, "available": mod is not None,
                              "note": ("recomputed by the pure matcher of this tree (allowlist_check / allowlist_verify), only when the "
                                       "tree's research/runners, webapp and sim equal the declared code; otherwise omitted")}
    out["reply_controls_strict_ok"] = d.get("reply_controls", {}).get("STRICT_OK")
    out["reply_cases"] = sum(1 for c in cases.values() if c.get("reply"))
    for arm in ("stock", "b2h", "allowlist"):
        a, r = s[arm], out[arm]
        kept_rows = _reply_rows_kept(d["rows"], arm, sub)
        ent = {}
        for lab in ("U", "B", "F"):
            hit = {"gate": set(a["entries"]["gate"][lab]["kept_ids"]), "composer": set(a["entries"]["composer"][lab]["kept_ids"]),
                   "reply": set(a["reply"]["kept"][lab]["ids"])}
            ent[lab] = hit
            u = hit["gate"] | hit["composer"] | hit["reply"]
            r[lab].update({"kept_reply_entry": len(hit["reply"]), "kept_ids_reply_entry": sorted(hit["reply"]),
                           "kept_union_all_entries": len(u), "kept_ids_union_all_entries": sorted(u),
                           "reply_entry_recomputed_from_rows_agrees": hit["reply"] == kept_rows[lab]})
        uu, bu = ent["U"], ent["B"]
        U_union, B_union = uu["gate"] | uu["composer"] | uu["reply"], bu["gate"] | bu["composer"] | bu["reply"]
        r["bar3"] = {"U_counted_n": len(counted["U"]), "U_counted_kept": len(U_union & counted["U"]),
                     "U_counted_kept_gate_and_composer_only": len((uu["gate"] | uu["composer"]) & counted["U"]),
                     "U_counted_kept_reply_entry_only_extra": len((uu["reply"] - uu["gate"] - uu["composer"]) & counted["U"]),
                     "U_kept_incl_not_counted": len(U_union), "U_n_incl_not_counted": sum(1 for l in final.values() if l == "U"),
                     "U_kept_not_counted_ids": sorted(U_union - counted["U"]),
                     "U_kept_not_counted_by_class": {c: sum(1 for i in U_union - counted["U"] if cases[i]["class"] == c)
                                                     for c in sorted({cases[i]["class"] for i in U_union - counted["U"]})},
                     "U_kept_counted_by_class": {c: sum(1 for i in U_union & counted["U"] if cases[i]["class"] == c)
                                                 for c in sorted({cases[i]["class"] for i in U_union & counted["U"]})},
                     "U_counted_by_class": {c: sum(1 for i in counted["U"] if cases[i]["class"] == c)
                                            for c in sorted({cases[i]["class"] for i in counted["U"]})},
                     "B_counted_n": len(counted["B"]), "B_counted_kept": len(B_union & counted["B"]),
                     "B_kept_total_any_case": len(B_union),
                     "B_kept_ids_not_in_counted_B": sorted(B_union - counted["B"]),
                     "strict_n": len(counted["U"]) + len(counted["B"]),
                     "strict_kept": len((U_union | B_union) & (counted["U"] | counted["B"]))}
        r["bar3"]["U_counted_upper95"] = upper95(r["bar3"]["U_counted_kept"], r["bar3"]["U_counted_n"])
        r["bar3"]["strict_upper95"] = upper95(r["bar3"]["strict_kept"], r["bar3"]["strict_n"])
        rp = a["reply"]
        r["reply_entry"] = {"n_reply_cases": rp["n"], "n_by_case_label": rp["n_by_case_label"], "UNDEFINED": rp["UNDEFINED"],
                            "kept_U_cases": rp["kept"]["U"]["n"], "kept_B_cases": rp["kept"]["B"]["n"], "kept_F_cases": rp["kept"]["F"]["n"],
                            "kept_U_ids": rp["kept"]["U"]["ids"], "kept_B_ids": rp["kept"]["B"]["ids"],
                            "full_reply_cases": len(rp["full_reply_ids"]), "cut_cases": len(rp["cut_ids"]),
                            "nothing_emitted_cases": len(rp["nothing_emitted_ids"]), "subsequence_cases": len(rp["subsequence_ids"]),
                            "j_distribution": rp["j_distribution"]}
        if arm == "allowlist":
            for lab, tag, S in (("U", "kept_U_cases", U_union), ("B", "kept_B_cases", B_union)):
                hits = ent[lab]
                r[tag] = [_case_detail(i, cases, adj, rows, "allowlist", lab,
                                       {k: i in hits[k] for k in ("gate", "composer", "reply")}, span_code) for i in sorted(S)]
                for det in r[tag]:
                    det["counted_by_case_label"] = det["id"] in counted[lab]
                    det["counted_for_bar3" + ("" if lab == "U" else "b")] = det["skeptic_label"] == "U" and det["class"] not in not_counted
            r["kept_B_note"] = ("kept B = every case the allowlist keeps through any entry whose own label is B, plus every reply case "
                                "whose emitted selection is labelled B (it can be a case whose own label is U: R9-RP02's sentence 2 "
                                "alone); counted_for_bar3b marks the skeptic's expect-U cases in counted classes (the bar-3b "
                                "population, known_residue excluded); an expect-F twin relabelled B by the judges is listed, not counted")
    al = out["allowlist"]["bar3"]
    out["allowlist_U_kept_gate_composer_only"] = al["U_counted_kept_gate_and_composer_only"]
    out["allowlist_U_kept_gate_composer_only_incl_not_counted"] = out["allowlist"]["U"]["kept_either_entry"]
    out["allowlist_U_kept"] = al["U_counted_kept"]
    out["allowlist_U_n"] = al["U_counted_n"]
    out["allowlist_U_upper95"] = al["U_counted_upper95"]
    out["allowlist_strict_kept"] = al["strict_kept"]
    out["allowlist_strict_n"] = al["strict_n"]
    out["allowlist_strict_upper95"] = al["strict_upper95"]


def hunt_summary(p, code=None, rnd=None, not_counted=()):
    d = _load(p)
    s = d["summary"]
    out = {"file": p, "sha256": _sha(p), "measured_at": _at(p), "host": d.get("host")}
    if code:
        out["code_check"] = code_check(p, code)
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
    if rnd and (s["allowlist"].get("reply") or {}).get("n"):   # the reply entry: a third way a case can be kept
        out["reply_entry_present"] = True
        _reply_round(out, d, rnd, list(not_counted), code)
        out["allowlist_all_entries_VALID"] = all(out[a].get("VALID") for a in ("stock", "b2h", "allowlist"))
    return out


def case_counts(rnd):
    cases = _load(CASES[rnd] + "/cases.json")
    adj = _load(CASES[rnd] + "/adjudication.json")
    exp = {}
    for c in cases:
        lab = c.get("label") or c.get("expect")
        exp[lab] = exp.get(lab, 0) + 1
    out = {"cases": len(cases), "skeptic_labels": exp, "adjudicated": adj.get("final_counts"),
           "relabelled_to_B": len(adj.get("relabel_to_B", []))}
    if adj.get("final_labels") and all("class" in c for c in cases):   # round 8 on: the class table, final label by class
        by = {}
        for c in cases:
            x = by.setdefault(c["class"], {})
            x[adj["final_labels"][c["id"]]] = x.get(adj["final_labels"][c["id"]], 0) + 1
        out["final_label_by_class"] = {k: dict(sorted(v.items())) for k, v in sorted(by.items())}
        out["reply_cases"] = sum(1 for c in cases if c.get("reply"))
        out["reply_selection_counts"] = adj.get("subset_counts")
    return out


def _bar4_must_keep(d):
    """Bar 4 (the 38 must-keep sentences = the F sentences B2h kept at seed 42 that encode one gathered triple, on S1 and S3), as
    this build's heldout_numbers recorded it: kept / n and each lost sentence with the shape it failed; compared with the fixture's
    static list of losses (captured at build 2, never regenerated), so a loss that is new at this build shows."""
    b4 = d.get("bar4_f_single_triple") or {}
    fx = _load("tests/fixtures/moat_allowlist_must_keep.json")
    fx_loss = {(x["stratum"], x["turn"], x["surface"]) for x in fx.get("allowlist_losses", [])}
    fx_all = {(x["stratum"], x["turn"], x["surface"]) for x in fx.get("single_triple", [])}
    lost = []
    for s in ("S1", "S3"):
        for r in (b4.get(s) or {}).get("rejected", []):
            key = (s, r["turn"], r["surface"])
            lost.append({"stratum": s, "turn": r["turn"], "svo": r["svo"], "surface": r["surface"],
                         "reason": r["allowlist_reject_reason"], "kept_at_any_seed": r.get("allowlist_kept_all_seeds"),
                         "in_fixture_loss_list_of_build_2": key in fx_loss, "in_fixture_38": key in fx_all})
    n = sum((b4.get(s) or {}).get("n", 0) for s in ("S1", "S3"))
    kept = sum((b4.get(s) or {}).get("kept_by_allowlist_seed42", 0) for s in ("S1", "S3"))
    return {"n": n, "kept": kept, "by_stratum": {s: {"n": b4[s]["n"], "kept": b4[s]["kept_by_allowlist_seed42"]} for s in ("S1", "S3") if s in b4},
            "lost": lost, "new_vs_fixture_loss_list": [x for x in lost if not x["in_fixture_loss_list_of_build_2"]],
            "fixture_losses_now_kept": [{"stratum": a, "turn": b, "surface": c} for a, b, c in sorted(fx_loss)
                                        if (a, b, c) not in {(x["stratum"], x["turn"], x["surface"]) for x in lost}],
            "fixture_n_single_triple": fx.get("n_single_triple"), "n_agrees_with_fixture": n == fx.get("n_single_triple"),
            "S2_not_part_of_the_38": ({"n": b4["S2"]["n"], "kept": b4["S2"]["kept_by_allowlist_seed42"]}
                                      if "S2" in b4 and "n" in b4["S2"] else None)}


def _composer_accepted(path):
    """Per answered turn of a composer-replay record file: the facts the composer accepted into the reply, as it recorded them
    (`n_accepted`): the distribution and total, for the 'sentences kept per reply' comparison between builds."""
    if not os.path.exists(path):
        return None
    ts = _load(path).get("turns", [])
    ans = [t for t in ts if not t["flag_on_replay"]["abstained"]]
    dist = {}
    for t in ans:
        k = str(t["flag_on_replay"]["n_accepted"])
        dist[k] = dist.get(k, 0) + 1
    return {"file": path, "answered": len(ans), "n_accepted_distribution": dict(sorted(dist.items())),
            "n_accepted_total": sum(t["flag_on_replay"]["n_accepted"] for t in ans)}


def strata(suffix, code=None):
    if suffix is None:
        return {"heldout_numbers": None, "measured": False,
                "note": "not measured: no strata, composer-replay or heldout_numbers run is registered for this build (it failed bar 3)",
                "files_matching_this_build_in_the_heldout_dir": sorted(glob.glob(HELD + "/*allowlist7*") + glob.glob(HELD + "/*build7*"))}
    hn = HELD + "/heldout_numbers%s.json" % suffix
    if not os.path.exists(hn):
        return {"heldout_numbers": None, "note": "heldout_numbers%s.json not in this tree yet" % suffix}
    d = _load(hn)
    v = d["verifier_level"]
    out = {"heldout_numbers": hn, "sha256": _sha(hn), "measured_at": _at(hn)}
    if code:
        files = {"heldout_numbers": hn}
        for tag, name in (("results_strata_s1_s2", "results_strata_s1_s2%s.json"), ("results_strata_s3", "results_strata_s3%s.json"),
                          ("composer_records_s1", "composer_records_s1%s.json"), ("composer_records_s3", "composer_records_s3%s.json")):
            files[tag] = HELD + "/" + (name % suffix)
        out["code_check"] = {k: dict(file=f, sha256=_sha(f), **code_check(f, code)) for k, f in files.items() if os.path.exists(f)}
    for s in ("S1", "S2", "S3"):
        a = v[s]["all"]["allowlist"]
        out[s] = {k: a[k] for k in ("buildable_unique", "lenient_unsupported_n", "lenient_leaks", "lenient_upper95",
                                    "strict_unsupported_n", "strict_leaks", "strict_upper95", "faithful_kept_seed42")}
        out[s]["stock_faithful_kept_seed42"] = v[s]["all"]["stock"]["faithful_kept_seed42"]
        out[s]["b2h_faithful_kept_seed42"] = v[s]["all"]["b2h"]["faithful_kept_seed42"]
        out[s]["strict_leak_sentences"] = a.get("strict_leak_sentences")
    cp = d.get("composer_path", {})
    out["composer"] = {}
    for s in ("S1", "S3"):
        x = cp.get(s)
        if x:
            out["composer"][s] = {"n_turns": x["n_turns"], "stock_answered": x["stock_answered"],
                                  "b2h_answered": x["b2h_answered"], "allowlist_answered": x["flag_on_answered"],
                                  "HURTS": len(x["HURTS_turns"]), "LOSES_vs_b2h": len(x["LOSES_vs_b2h_turns"]),
                                  "LOSES_vs_b2h_turns": sorted(t if isinstance(t, str) else t.get("label") for t in x["LOSES_vs_b2h_turns"]),
                                  "replay_misses": len(x["replay_misses_flag_on"]) + len(x["replay_misses_flag_off"]),
                                  "status_counts": x["status_counts"],
                                  "facts_accepted_per_answered_turn": _composer_accepted(HELD + "/composer_records_%s%s.json" % (s.lower(), suffix))}
    b4 = d.get("bar4_f_single_triple", {})
    out["bar4"] = {s: {"n": b4[s]["n"], "kept": b4[s]["kept_by_allowlist_seed42"]} for s in b4 if "n" in b4[s]}
    out["bar4_must_keep_38"] = _bar4_must_keep(d)
    return out


def stub(suffix, code=None):
    if suffix is None:
        return {"file": None, "measured": False,
                "note": "not measured: no stub-population run is registered for this build (it failed bar 3)",
                "files_matching_this_build_in_the_heldout_dir": sorted(glob.glob(HELD + "/stub_population*allowlist7*")
                                                                       + glob.glob(HELD + "/stub_population*build7*"))}
    p = HELD + "/stub_population_exhaustive%s.json" % suffix
    d = _load(p)
    keys = ("n_population", "stock_answer_rate", "allowlist_answer_rate", "HURTS_allowlist", "HELPS_allowlist",
            "allowlist_role_reversed_stock_leaks", "allowlist_meaning_drift_stock_leaks", "relation_table_sha256")
    out = {"file": p, "sha256": _sha(p), "measured_at": _at(p)}
    if code:
        out["code_check"] = code_check(p, code)
    out.update({k: d.get(k) for k in keys})
    cells = d.get("cells(stock_ok,allowlist_ok)") or {}
    out["unvoiceable_under_flag"] = cells.get("(True, False)")
    out["role_reversed_by_relation"] = d.get("allowlist_role_reversed_by_relation")
    out["meaning_drift_by_relation"] = d.get("allowlist_meaning_drift_by_relation")
    out["cells(stock_ok,allowlist_ok)"] = d.get("cells(stock_ok,allowlist_ok)")
    out["n_flag_cannot_voice_what_stock_voices"] = cells.get("(True, False)", 0) if cells else None
    out["allowlist_surface_changed"] = d.get("allowlist_surface_changed")
    out["allowlist_kept_by_no_reading_relation"] = d.get("allowlist_kept_by_no_reading_relation")
    return out


ROUTED_PATHS = {"spiking_recall_surface": ("spiking_recall",), "touchpoint_a": ("touchpoint_a",), "chatbrain_render": ("chatbrain_render",),
                "hypothesis_body": ("hypothesis",), "one_brain_substitution": ("one_brain",)}   # Amendment 6, bar 6: name match only
FLAG_OFF_WORDS = ("flag_off", "unchanged", "base_golden", "golden")


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
    routed = {}   # name match only: whether a test "calls the allowlist check and rejects a known-U input" is read from the test, not here
    for path, words in ROUTED_PATHS.items():
        hits = {}
        for case in root.iter("testcase"):
            n = case.get("name") or ""
            if any(w in n for w in words):
                hits["%s::%s" % (case.get("classname"), n)] = not any(ch.tag in ("failure", "error") for ch in case)
        routed[path] = {"n_tests": len(hits), "all_passed": all(hits.values()) if hits else None, "tests": hits}
    flag_off = {}   # the build-6 test file's flag-off / unchanged tests, by name
    for case in root.iter("testcase"):
        cn, n = case.get("classname") or "", case.get("name") or ""
        if cn.endswith("test_moat_allowlist_build6") and any(w in n for w in FLAG_OFF_WORDS):
            flag_off["%s::%s" % (cn, n)] = not any(ch.tag in ("failure", "error") for ch in case)
    return {"file": p, "sha256": _sha(p), "totals": tot, "by_file": files, "bar_tests": named,
            "routed_path_tests_by_name": routed, "flag_off_tests_by_name_in_build6_file": flag_off}


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


def _vs_build5(r6):
    """What build 6 changes against build 5 on the cells Amendment 6 asks to be reported beside build 5's: the stub answer rate and
    how many of build 5's unvoiceable one-fact facts are now voiced, the composer answered turns, and the must-keep set."""
    s5, t5 = strata("_allowlist5"), stub("_allowlist5")
    out = {"build5_stub_answer_rate": t5["allowlist_answer_rate"], "build6_stub_answer_rate": r6["stub"]["allowlist_answer_rate"],
           "build5_unvoiceable_under_flag": t5["unvoiceable_under_flag"],
           "build6_unvoiceable_under_flag": r6["stub"]["n_flag_cannot_voice_what_stock_voices"],
           "became_voiceable": (t5["unvoiceable_under_flag"] or 0) - (r6["stub"]["n_flag_cannot_voice_what_stock_voices"] or 0)}
    out["composer_answered"] = {s: {"build5": s5["composer"][s]["allowlist_answered"], "build6": r6["strata"]["composer"][s]["allowlist_answered"]}
                                for s in ("S1", "S3")}
    out["facts_accepted_per_answered_turn"] = {s: {"build5": s5["composer"][s]["facts_accepted_per_answered_turn"],
                                                   "build6": r6["strata"]["composer"][s]["facts_accepted_per_answered_turn"]} for s in ("S1", "S3")}
    out["bar4_must_keep"] = {"build5": [s5["bar4_must_keep_38"]["kept"], s5["bar4_must_keep_38"]["n"]],
                             "build6": [r6["strata"]["bar4_must_keep_38"]["kept"], r6["strata"]["bar4_must_keep_38"]["n"]]}
    out["faithful_kept_seed42"] = {s: {"build5": s5[s]["faithful_kept_seed42"], "build6": r6["strata"][s]["faithful_kept_seed42"]}
                                   for s in ("S1", "S2", "S3")}
    return out


def _parse_junit_builds(items):
    out = {}
    for it in items or []:
        b, _, path = it.partition("=")
        if not b.isdigit() or not path:
            raise SystemExit("--junit-build wants N=<xml path>, got %r" % it)
        out[int(b)] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--junit", help="the build-5 junit record (kept for the original invocation)")
    ap.add_argument("--junit-build", action="append", metavar="N=XML", help="a build's junit record, e.g. 6=/path/junit_build6.xml")
    ap.add_argument("--mutation", help="the build-5 mutation-round summary (copied beside this script)")
    ap.add_argument("--builds", default="1,2,3,4,5", help="comma-separated builds to summarise (default: the original 1-5)")
    a = ap.parse_args()
    builds = [int(x) for x in a.builds.split(",") if x.strip()]
    junit_paths = _parse_junit_builds(a.junit_build)
    res = {"_doc": __doc__.strip().splitlines()[0], "script": __name__, "argv": vars(a),
           "bound_rule": "one-sided 95% upper bound: rule of three at 0 kept, Clopper-Pearson otherwise (prereg 2.1)",
           "kept_rule": "a hunt case counts as kept when the allowlist keeps it at ANY of the seven seeds through EITHER entry "
                        "(claim gate or composer), the stricter of the two readings; round 3's runner records kept ids at "
                        "any seed",
           "kept_rule_reply_rounds": "rounds 8 and 9 add the reply entry: a case counts as kept when ANY of the claim gate, the composer "
                                     "candidate or the reply entry keeps it at any seed (the strict union); through the reply entry a "
                                     "case counts for the label of the selection that reached the user. `allowlist_U_kept` is the count "
                                     "under the round's own count rule (see each build's hunt.count_rule); "
                                     "`allowlist_U_kept_gate_composer_only` is the original two-entry reading",
           "builds": {}}
    for b in builds:
        cfg = BUILDS[b]
        r = {"code": cfg["code"], "blind_round": cfg["round"],
             "hunt": hunt_summary(cfg["hunt"], cfg["code"], cfg["round"], cfg.get("not_counted_classes", ())),
             "hunt_cases": case_counts(cfg["round"]), "strata": strata(cfg["suffix"], cfg["code"]),
             "stub": stub(cfg.get("stub_suffix", cfg["suffix"]), cfg["code"]),
             "nonblind_reruns": {k: hunt_summary(p, cfg["code"]) for k, p in RERUNS.get(b, {}).items()}}
        r["bar3_pass"] = (r["hunt"]["allowlist_U_kept"] == 0 and r["hunt"]["allowlist"]["VALID"] is True
                          and r["hunt"].get("controls_all_ok", True) is not False
                          and r["hunt"].get("reply_controls_strict_ok", True) is not False
                          and r["hunt"].get("allowlist_all_entries_VALID", True) is not False)
        if cfg["suffix"] is None:
            r["bars_1_4_5"] = "not measured (build %d failed bar 3 and was never measured on bars 1, 4 and 5)" % b
        if b == 6:
            r["vs_build5"] = _vs_build5(r)
        if b in junit_paths:
            r["moat_suite_junit"] = junit(junit_paths[b])
        res["builds"][str(b)] = r
    res["bank_and_must_keep"] = bank()
    res["bank_and_must_keep"]["note"] = ("must_keep_* is the fixture's static record, captured at build 2 and never regenerated: the "
                                         "per-build bar 4 is builds.N.strata.bar4_must_keep_38, read from that build's heldout_numbers")
    if a.junit:
        res["build5_moat_suite"] = junit(a.junit)
    if a.mutation:
        res["build5_mutation"] = mutation(a.mutation, HERE + "/mutation_build5_summary.json")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for b, r in res["builds"].items():
        h = r["hunt"]
        tail = ""
        if r["stub"].get("measured") is False:
            tail = " | stub/strata/composer: not measured"
        else:
            tail = " | stub HURTS %s answer %.4f" % (r["stub"]["HURTS_allowlist"], r["stub"]["allowlist_answer_rate"])
        print("build %s %s: U kept %d/%d (upper95 %.4f) strict %d/%d F %d/%d bar3_pass=%s%s"
              % (b, r["blind_round"], h["allowlist_U_kept"], h["allowlist_U_n"], h["allowlist_U_upper95"],
                 h["allowlist_strict_kept"], h["allowlist_strict_n"], h["allowlist_F_kept"], h["allowlist_F_n"],
                 r["bar3_pass"], tail))


if __name__ == "__main__":
    main()
