#!/usr/bin/env python3
"""merge corpus + pytest + witness search + manual equivalence notes -> results_b5.jsonl, summary_b5.json"""
import json, sys, collections
sys.path.insert(0, "/home/dant123/Projects/sim-rev/moat-allow5/mut")
import memmut5
from equiv_notes5 import NOTES, no_why, NO_NOTES
import check_witnesses5 as CW

MUT = "/home/dant123/Projects/sim-rev/moat-allow5/mut"
M = memmut5.mutants()
corpus = {json.loads(l)["id"]: json.loads(l) for l in open(MUT + "/corpus_b5.jsonl")}
pyt = {json.loads(l)["id"]: json.loads(l) for l in open(MUT + "/pytest_b5.jsonl")}
search = {}
import glob
for f in glob.glob(MUT + "/search_*.jsonl"):
    for l in open(f):
        r = json.loads(l); search[r["id"]] = r
rows = []
for m in M:
    mid = m["id"]
    c = corpus[mid]
    rec = {"id": mid, "cat": m["cat"], "desc": m["desc"], "edits": [{"file": f, "old": o, "new": n} for f, o, n in m["edits"]]}
    rec["corpus"] = {"killed": c["corpus_killed"], "killed_by": c["killed_by"], "keep_diff": c["keep_diff"], "first_witness": c["first_witness"]}
    if c["corpus_killed"]:
        rec["status"] = "KILLED_CORPUS"
        rec["pytest"] = None
    else:
        p = pyt.get(mid)
        rec["pytest"] = {k: p.get(k) for k in ("pytest_killed", "failed", "errors", "summary", "seconds", "rc", "timeout", "worktree")} if p else None
        if p is None:
            rec["status"] = "PYTEST_MISSING"
        elif p["pytest_killed"]:
            rec["status"] = "KILLED_PYTEST"
        else:
            rec["status"] = "SURVIVED"
    if rec["status"] == "SURVIVED":
        # witness: hand witnesses first (verified by running both), then the search sources
        w = CW.check(mid)
        src = "hand"
        if w is None and mid in search:
            for name in ("table", "fuzz", "soup", "ambfuzz"):
                f = search[mid][name]["first"]
                if f:
                    w = f; src = name; break
        note = NOTES.get(mid)
        if w:
            import re as _re
            from research.runners._moat_allowlist_relations import RELATIONS as _REL
            ents = {t for g in w["gathered"] for t in (g[0], g[2])}
            rels = {g[1] for g in w["gathered"]}
            shaped = all(_re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", e) for e in ents) and all(r in _REL for r in rels)
            w = dict(w, store_shaped_slugs=bool(shaped))
            rec["survivor"] = {"equivalent": "no", "witness_source": src, "witness": w, "why": (NO_NOTES.get(mid) or no_why(mid) or (note[1] if note and note[0] == "no" else "witness found: original and mutant differ on this input")),
                               "reason_only_diffs_elsewhere": any(search.get(mid, {}).get(n, {}).get("reason_only") for n in ("table", "fuzz", "soup", "ambfuzz"))}
        elif note:
            rec["survivor"] = {"equivalent": note[0], "witness_source": None, "witness": None, "why": note[1]}
        else:
            rec["survivor"] = {"equivalent": "unsure", "witness_source": None, "witness": None, "why": "no witness found and no note written"}
    rows.append(rec)
with open(MUT + "/results_b5.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(collections.Counter(r["status"] for r in rows))
print(collections.Counter(r["survivor"]["equivalent"] for r in rows if r["status"] == "SURVIVED"))

# ----------------------------------------------------------------------------------------------------------------- summary
n = len(rows)
kc = [r for r in rows if r["status"] == "KILLED_CORPUS"]
kp = [r for r in rows if r["status"] == "KILLED_PYTEST"]
sv = [r for r in rows if r["status"] == "SURVIVED"]
by_corpus = collections.Counter()
for r in kc:
    for c in r["corpus"]["killed_by"]:
        by_corpus[c] += 1
only_corpus = collections.Counter(r["corpus"]["killed_by"][0] for r in kc if len(r["corpus"]["killed_by"]) == 1)
cat = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    cat[r["cat"].split("/")[0]][r["status"]] += 1
summ = {
    "commit": "ef51d6243 (branch claude-build/moat-allowlist-1020e)",
    "n_mutants": n,
    "killed_by_corpus": len(kc),
    "killed_by_pytest_after_corpus": len(kp),
    "killed_total": len(kc) + len(kp),
    "survived_total": len(sv),
    "survived_equivalent": [r["id"] for r in sv if r["survivor"]["equivalent"] == "yes"],
    "survived_not_equivalent": [r["id"] for r in sv if r["survivor"]["equivalent"] == "no"],
    "survived_unsure": [r["id"] for r in sv if r["survivor"]["equivalent"] == "unsure"],
    "corpus_kill_counts_by_corpus": dict(by_corpus),
    "corpus_killed_by_only_this_corpus": dict(only_corpus),
    "by_category": {k: dict(v) for k, v in sorted(cat.items())},
    "survivors_not_equivalent_with_non_store_shaped_witness_slugs": [r["id"] for r in sv if r["survivor"]["equivalent"] == "no" and not r["survivor"]["witness"].get("store_shaped_slugs")],
    "survivors_detail": {r["id"]: {"equivalent": r["survivor"]["equivalent"], "why": r["survivor"]["why"],
                                   "witness": ({"surface": r["survivor"]["witness"]["surface"], "gathered": r["survivor"]["witness"]["gathered"],
                                                "original": r["survivor"]["witness"]["orig"], "mutant": r["survivor"]["witness"]["mutant"]}
                                               if r["survivor"]["witness"] else None)} for r in sv},
    "pytest_command": "pytest tests/test_moat_allowlist_build5.py tests/test_moat_allowlist_exact.py tests/test_moat_allowlist_witnesses.py tests/test_moat_allowlist_caps.py tests/test_moat_allowlist.py -q -p no:cacheprovider -x (same five files as specified, ordered so the likeliest killers run first; any failure kills, so order does not change a verdict)",
    "baseline": "824 passed on the unmutated worktree (wt-mut5 and wt-mut5-15), ~130 s",
    "witness_corpus_items": 10614,
}
json.dump(summ, open(MUT + "/summary_b5.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summ.items() if k != "by_category"}, indent=1))
