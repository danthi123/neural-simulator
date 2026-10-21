"""Build the two committed fixtures of tests/test_moat_allowlist.py (the strict allowlist claim checker, BRAIN_MOAT_ALLOWLIST).

1. tests/fixtures/moat_allowlist_bank.json -- the ADVERSARIAL BANK, ported from the verify-go leak hunts that refuted the
   slug-aware checker (B2h). Every case keeps its `source` file, its id/name, its gathered set and the expect label it was
   written with (U = unsupported, must be rejected; S = supported; "S?" = ambiguous, not scored). Sources (outside the repo, their
   sha256 recorded so the port can be re-checked):
     round 1  /home/dant123/Projects/sim-rev/moat-verify/flagoff-leak/leak_hunt.py, leak_hunt2.py   (CASES, read by AST: the
              files run a verifier at import, so only their literal top-level assignments are evaluated)
     round 2  /home/dant123/Projects/sim-rev/moat-verify2/leak-hunt/cases_a.py, cases_b.py           (pure data + a loop; runpy)
     sweep    /home/dant123/Projects/sim-rev/moat-verify2/leak-hunt/sweep_real.py                     (its template functions,
              read by AST, over the 90 held-out gathered sets of corpus.json (S1) + s3_corpus.json (S3) in this repo; an instance
              is expect U iff one of its intended claims is not a gathered triple -- the sweep's own leak rule)
2. tests/fixtures/moat_allowlist_must_keep.json -- every unique S1 and S3 sentence labelled F that the B2h arm KEPT at parser
   seed 42 (BRAIN_MOAT_SLUG_NORM=1 through ChatBrain._verify_claim_set, exactly as measure_strata.py computes the b2h arm), split
   by how many distinct gathered triples B2h grounded in it. The single-triple ones are the prereg's bar-4 set. `allowlist_losses`
   lists each of those the allowlist rejects with the reason (the shape it failed): written from the matcher's own verdict so
   the test can pin it, never typed by hand.

usage (cwd = worktree):  .venv/bin/python -m tests.fixtures.build_moat_allowlist_fixtures
"""
from __future__ import annotations

import ast
import collections
import hashlib
import itertools
import json
import os
import runpy
import sys

os.environ["SIM_BACKEND"] = "numpy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("BRAIN_MOAT_ALLOWLIST", None)
os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
H = os.path.join(ROOT, "research", "findings", "raw", "_moat_slug_norm_heldout")
sys.path.insert(0, H)
FIX = os.path.join(ROOT, "tests", "fixtures")

R1 = "/home/dant123/Projects/sim-rev/moat-verify/flagoff-leak/"
R2 = "/home/dant123/Projects/sim-rev/moat-verify2/leak-hunt/"
FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
SOURCES = {"leak_hunt.py": R1 + "leak_hunt.py", "leak_hunt2.py": R1 + "leak_hunt2.py", "cases_a.py": R2 + "cases_a.py",
           "cases_b.py": R2 + "cases_b.py", "sweep_real.py": R2 + "sweep_real.py"}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _literal_assignments(path):
    """Evaluate only the top-level `NAME = <literal / names / lists / tuples>` assignments of a script that runs work at import."""
    tree = ast.parse(open(path).read())
    allowed = (ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Load, ast.Store, ast.Assign, ast.UnaryOp, ast.USub)
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and all(isinstance(x, allowed) for x in ast.walk(n))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def _sweep_functions(path):
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("title", "rtxt", "sent", "instances")]
    ns = {"itertools": itertools, "collections": collections}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def heldout_sets():
    sets = []
    for fn, tag in (("corpus.json", "S1"), ("s3_corpus.json", "S3")):
        for t in json.load(open(os.path.join(H, fn)))["turns"]:
            sets.append((tag + ":" + t["label"], sorted({tuple(f) for f in t["gated"]})))
    return sets


class _Sets:
    """Gathered sets stored once, referenced by id."""

    def __init__(self):
        self.by_key, self.sets = {}, {}

    def ref(self, gathered, hint):
        key = tuple(sorted({tuple(f) for f in gathered}))
        if key not in self.by_key:
            gid = hint if hint not in self.sets else "%s#%d" % (hint, len(self.sets))
            self.by_key[key] = gid
            self.sets[gid] = [list(f) for f in key]
        return self.by_key[key]


def build_bank():
    S = _Sets()
    cases = []
    for src in ("leak_hunt.py", "leak_hunt2.py"):
        ns = _literal_assignments(SOURCES[src])
        names = {id(v): k for k, v in ns.items() if isinstance(v, list) and k != "CASES" and k.isupper()}
        for name, gated, prose, exp in ns["CASES"]:
            cases.append({"source": src, "id": name, "gathered": S.ref(gated, names.get(id(gated), src)), "prose": prose,
                          "expect": exp})
    for src in ("cases_a.py", "cases_b.py"):
        ns = runpy.run_path(SOURCES[src])
        names = {id(v): k for k, v in ns.items() if isinstance(v, list) and k.startswith("F_")}
        for cid, cls, facts, prose, exp in ns["CASES"]:
            cases.append({"source": src, "id": cid, "class": cls, "gathered": S.ref(facts, names.get(id(facts), cid)),
                          "prose": prose, "expect": exp})
    fn = _sweep_functions(SOURCES["sweep_real.py"])
    n_sup = 0
    for label, G in heldout_sets():
        gid = S.ref(G, label)
        Gs = set(G)
        seen = set()
        for tmpl, prose, intended, _note in fn["instances"](G):
            if (tmpl, prose) in seen:
                continue
            seen.add((tmpl, prose))
            unsupported = [list(c) for c in intended if tuple(c) not in Gs]
            if not unsupported:
                n_sup += 1                      # every intended claim gathered: not an expect-U input, not ported
                continue
            cases.append({"source": "sweep_real.py", "id": "%s|%s" % (label, tmpl), "class": tmpl, "gathered": gid,
                          "prose": prose, "expect": "U", "unsupported_intended": unsupported})
    per = collections.defaultdict(collections.Counter)
    for c in cases:
        per[c["source"]][c["expect"]] += 1
    return {"_doc": "adversarial bank for the allowlist claim checker, ported from the B2h verify-go leak hunts (rule: "
                    "tests/fixtures/build_moat_allowlist_fixtures.py docstring). expect U = must be rejected.",
            "generated_by": "tests/fixtures/build_moat_allowlist_fixtures.py",
            "backend": "none (host-only text port, no substrate)", "device": "cpu",
            "sources": {k: {"path": v, "sha256": _sha(v)} for k, v in SOURCES.items()},
            "heldout_corpora_sha256": {f: _sha(os.path.join(H, f)) for f in ("corpus.json", "s3_corpus.json")},
            "sweep_instances_all_intended_gathered_not_ported": n_sup,
            "counts_by_source": {k: dict(v) for k, v in per.items()},
            "gathered_sets": S.sets, "cases": cases}


def build_must_keep():
    from unique_sentences import unique
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.brain_conversational_agent import BridgeParser
    from research.runners._moat_allowlist import allowlist_check
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=42)
    S = _Sets()
    single, multi = [], []
    totals = {}
    os.environ["BRAIN_MOAT_SLUG_NORM"] = "1"
    try:
        for name, cfile, lfile in (("S1", "corpus.json", "labels.json"), ("S3", "s3_corpus.json", "s3_labels.json")):
            corpus = json.load(open(os.path.join(H, cfile)))
            lab = {(e["turn"], tuple(e["svo"]), e["surface"]): e["label"] for e in json.load(open(os.path.join(H, lfile)))["labels"]}
            turns = {t["label"]: t for t in corpus["turns"] if t.get("calls")}
            n_f = n_kept = 0
            for u in unique(corpus):
                k = (u["turn"], tuple(u["svo"]), u["surface"])
                if lab[k] != "F":
                    continue
                n_f += 1
                gated = turns[u["turn"]]["gated"]
                acc, res = chat._verify_claim_set(u["surface"], gated)
                if not acc:
                    continue
                n_kept += 1
                triples = sorted({tuple(g[:3]) for g in res["grounded"]})
                row = {"stratum": name, "turn": u["turn"], "svo": u["svo"], "surface": u["surface"],
                       "gathered": S.ref(gated, "%s:%s" % (name, u["turn"])), "b2h_grounded": [list(t) for t in triples]}
                (single if len(triples) == 1 else multi).append(row)
            totals[name] = {"F_unique": n_f, "F_kept_by_b2h_seed42": n_kept}
    finally:
        os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
    losses = []
    for r in single:
        a = allowlist_check(r["surface"], S.sets[r["gathered"]])
        if not a["keep"]:
            losses.append({"stratum": r["stratum"], "turn": r["turn"], "surface": r["surface"], "reason": a["reason"]})
    for r in multi:
        a = allowlist_check(r["surface"], S.sets[r["gathered"]])
        r["allowlist_keep"], r["allowlist_reason"] = bool(a["keep"]), a["reason"]
    first_per_relation = {}
    if os.path.exists(FACTS):
        for r in json.load(open(FACTS)):
            f = r["fact"]
            if f.get("polarity") == "AFFIRM" and f["action"] not in first_per_relation:
                first_per_relation[f["action"]] = [f["agent"], f["action"], f["patient"]]
    return {"_doc": "F-labelled S1/S3 sentences the B2h arm kept at parser seed 42 (BRAIN_MOAT_SLUG_NORM=1 via "
                    "ChatBrain._verify_claim_set); single_triple = B2h grounded exactly one distinct gathered triple (prereg bar 4). "
                    "allowlist_losses = single-triple sentences the allowlist matcher rejects, with the shape it failed (bar-4 "
                    "misses, reported, never loosened for). Rule: tests/fixtures/build_moat_allowlist_fixtures.py docstring.",
            "generated_by": "tests/fixtures/build_moat_allowlist_fixtures.py",
            "backend": os.environ.get("SIM_BACKEND"), "device": "cpu (BridgeParser on numpy, CUDA hidden)",
            "heldout_sha256": {f: _sha(os.path.join(H, f)) for f in ("corpus.json", "s3_corpus.json", "labels.json", "s3_labels.json")},
            "parser_seed": 42, "totals": totals, "n_single_triple": len(single), "n_multi_triple": len(multi),
            "gathered_sets": S.sets, "single_triple": single, "multi_triple": multi, "allowlist_losses": losses,
            "stub_one_per_relation_doc": "the first AFFIRM fact of each relation of the production LTM store (%s), for the "
                                         "per-relation stub-mouth pins" % FACTS,
            "stub_one_per_relation": [first_per_relation[k] for k in sorted(first_per_relation)]}


def main():
    bank = build_bank()
    with open(os.path.join(FIX, "moat_allowlist_bank.json"), "w") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("bank:", bank["counts_by_source"], "sweep all-gathered (not ported):", bank["sweep_instances_all_intended_gathered_not_ported"])
    mk = build_must_keep()
    with open(os.path.join(FIX, "moat_allowlist_must_keep.json"), "w") as f:
        json.dump(mk, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("must-keep:", mk["totals"], "single", mk["n_single_triple"], "multi", mk["n_multi_triple"],
          "allowlist losses", len(mk["allowlist_losses"]))
    for x in mk["allowlist_losses"]:
        print("  LOSS", x["stratum"], x["turn"], "|", x["surface"], "|", x["reason"])
    for x in mk["multi_triple"]:
        print("  MULTI", x["stratum"], x["turn"], "|", x["surface"], "| allowlist", x["allowlist_keep"], x["allowlist_reason"])


if __name__ == "__main__":
    main()
