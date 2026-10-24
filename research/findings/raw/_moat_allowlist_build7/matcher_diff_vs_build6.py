"""Build-7 REVIEW DIAGNOSTIC (non-blind; report only, not a pass): the pure matcher's verdict (`allowlist_verify`, no substrate, no parser seed)
at build 7 (this worktree's `research/runners/_moat_allowlist.py`) against build 6 (the same file at commit 2b7ab8c1d, read with `git show`
and executed as a separate module object, with `__file__` set to this worktree's path so the committed caps / names tables resolve the same
way). Over four populations:
  1. the exhaustive one-fact stub population: every AFFIRM fact of the production LTM store (wikidata_100k, 78,857), rendered by the stock
     `StubRenderer.render_svo` under BRAIN_MOAT_ALLOWLIST=1 exactly as `_moat_slug_norm_heldout/stub_population_exhaustive.py` renders its
     allowlist arm (that module's FACTS / ALLOW constants and the renderer are imported, not re-implemented), one fact per turn;
  2. the held-out corpora sentences (`_moat_slug_norm_heldout/{corpus,s2_corpus,s3_corpus}.json`, labels `labels.json` / `s2_labels.json` /
     `s3_labels.json`; the unique (turn, svo, surface) sentences against the turn's gathered set);
  3. the hunt cases of rounds 3-8 (labels = the committed build-6 results files' adjudicated labels; round 8 from the wt-hunt8 tree, path
     given on the command line, read only);
  4. EXTRA (not asked for): a natural-spelling cost scan of the store entities whose slug has a non-initial "i" word, rendered three ways
     as the SUBJECT of a store fact (slug verbatim, de-slugged lower case, str.title() with a lone "i" -> "I"), build 7 against build 6.
Cross-checks of the build-6 loader (each reported, none asserted): the number of stub facts build 6 keeps here vs the committed
stub_population_exhaustive_allowlist6.json; and, per hunt round, how many cases' build-6 verdict here differs from the committed
results_build6.json seed-42 allowlist verdict. A monotonicity cell (build 6 rejected, build 7 kept) is reported if it ever occurs.
Writes no case and reads no held-out stratum beyond the corpora above.

usage (cwd = worktree, as a module; refuses to overwrite):
  python -m research.findings.raw._moat_allowlist_build7.matcher_diff_vs_build6 <out.json> [<r8 dir>]
"""
import research.runners  # noqa: F401  (provenance door)
import collections
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())

from research.runners import _moat_allowlist as B7                                           # noqa: E402
from research.runners.brain_chat_tui import StubRenderer                                      # noqa: E402
from research.findings.raw._moat_slug_norm_heldout import stub_population_exhaustive as SP    # noqa: E402  (FACTS, ALLOW)

BUILD6_SHA = "2b7ab8c1d"
HELD = "research/findings/raw/_moat_slug_norm_heldout"
H3 = "research/findings/raw/_moat_allowlist_hunt3"
H4 = "research/findings/raw/_moat_allowlist_hunt4"
R8_DEFAULT = "/home/dant123/Projects/sim-rev/wt-hunt8/research/findings/raw/_moat_allowlist_hunt4/r8"


def load_build6():
    """The build-6 module object: `research/runners/_moat_allowlist.py` at BUILD6_SHA, executed in a fresh namespace."""
    src = subprocess.check_output(["git", "show", "%s:research/runners/_moat_allowlist.py" % BUILD6_SHA]).decode("utf-8")
    spec = importlib.util.spec_from_loader("research.runners._moat_allowlist_build6_snapshot", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(os.getcwd(), "research", "runners", "_moat_allowlist.py")
    exec(compile(src, "<_moat_allowlist.py at %s>" % BUILD6_SHA, "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode("utf-8")).hexdigest()


def rule_of(reason):
    """Which build-7 rule a build-7 reject reason names (by the texts of `_pronoun_name_reading` / `_inner_first_person_reading`)."""
    r = reason or ""
    if "reads as the pronoun" in r and "(written in lower case" in r:
        return "R7a one-word pronoun name, lower case"
    if "reads as the pronoun" in r and "(the first word of the sentence" in r:
        return "R7a one-word pronoun name, sentence-initial"
    if "a lower-case 'i' (word" in r:
        return "R7b(1) lower-case i, no capital in the name"
    if "holds 'I' (word" in r:
        return "R7b(2) I then a lower-case word"
    if "a conjunction before 'I'" in r:
        return "R7b(3) conjunction then I"
    return "other: " + r[:80]


class Pair:
    """Both versions' verdict on (sentence, gathered); counts and the changed rows."""

    def __init__(self, b6, b7):
        self.b6, self.b7 = b6, b7
        self.cells = collections.Counter()
        self.changed = []

    def run(self, sentence, gathered, meta):
        r6 = self.b6.allowlist_verify(sentence, gathered)
        r7 = self.b7.allowlist_verify(sentence, gathered)
        k6, k7 = bool(r6["accepted"]), bool(r7["accepted"])
        self.cells[(k6, k7)] += 1
        if k6 != k7:
            row = dict(meta, sentence=sentence, build6_kept=k6, build7_kept=k7, build7_rule=rule_of(r7["reject_reason"]),
                       build7_reason=r7["reject_reason"], build6_reason=r6["reject_reason"])
            self.changed.append(row)
        return k6, k7

    def summary(self):
        return {"n": sum(self.cells.values()), "cells(build6_kept,build7_kept)": {str(k): v for k, v in sorted(self.cells.items())},
                "n_changed": len(self.changed),
                "n_rejected_by_build7_not_build6": sum(1 for x in self.changed if x["build6_kept"] and not x["build7_kept"]),
                "n_kept_by_build7_not_build6": sum(1 for x in self.changed if x["build7_kept"] and not x["build6_kept"]),
                "changed_by_rule": dict(collections.Counter(x["build7_rule"] for x in self.changed))}


def stub_population(b6, b7):
    aff = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(SP.FACTS))
           if r["fact"].get("polarity") == "AFFIRM"]
    pair, stub, t0 = Pair(b6, b7), StubRenderer(), time.time()
    os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
    os.environ[SP.ALLOW] = "1"                 # the stub writes the shape-4 phrase for the nine no-reading relations under the flag only
    try:
        for i, svo in enumerate(aff):
            surf, _asserted = stub.render_svo(*svo)
            pair.run(surf, [list(svo)], {"svo": list(svo)})
            if (i + 1) % 10000 == 0:
                print("[stub] %d/%d t=%.0fs changed=%d" % (i + 1, len(aff), time.time() - t0, len(pair.changed)), flush=True)
    finally:
        os.environ.pop(SP.ALLOW, None)
    out = pair.summary()
    out["n_population"] = len(aff)
    out["changed_rows"] = pair.changed
    try:
        com = json.load(open(HELD + "/stub_population_exhaustive_allowlist6.json"))
        out["cross_check_committed_build6"] = {
            "file": HELD + "/stub_population_exhaustive_allowlist6.json", "committed_n_facts": com["n_facts"],
            "committed_allowlist_answer_rate": com["allowlist_answer_rate"],
            "committed_kept_count": round(com["allowlist_answer_rate"] * com["n_facts"]),
            "this_run_build6_kept_count": sum(v for (k6, _k7), v in pair.cells.items() if k6),
            "this_run_build7_kept_count": sum(v for (_k6, k7), v in pair.cells.items() if k7)}
    except Exception as exc:                   # report, do not hide
        out["cross_check_committed_build6"] = "unavailable: %s: %s" % (type(exc).__name__, exc)
    return out


def heldout(b6, b7):
    sys.path.insert(0, os.path.join(os.getcwd(), HELD))
    from unique_sentences import unique       # noqa: E402  (the strata measurer's own numbering)
    out = {}
    for name, cpath, lpath in (("S1 corpus.json", "corpus.json", "labels.json"), ("S2 s2_corpus.json", "s2_corpus.json", "s2_labels.json"),
                               ("S3 s3_corpus.json", "s3_corpus.json", "s3_labels.json")):
        corpus = json.load(open(os.path.join(HELD, cpath)))
        lab_doc = json.load(open(os.path.join(HELD, lpath)))
        lab = {(e["turn"], tuple(e["svo"]), e["surface"]): (e["label"], e.get("variant")) for e in lab_doc["labels"]}
        turns = {t["label"]: t for t in corpus["turns"] if t.get("calls")}
        pair = Pair(b6, b7)
        by_label = collections.Counter()
        for u in unique(corpus):
            k = (u["turn"], tuple(u["svo"]), u["surface"])
            label, variant = lab[k]
            k6, k7 = pair.run(u["surface"], turns[u["turn"]]["gated"], {"turn": u["turn"], "svo": u["svo"], "label": label, "variant": variant})
            by_label[(label, k6, k7)] += 1
        s = pair.summary()
        s["cells(label,build6_kept,build7_kept)"] = {str(k): v for k, v in sorted(by_label.items())}
        s["changed_rows"] = pair.changed
        out[name] = s
        print("[heldout]", name, s["n"], "changed", s["n_changed"], flush=True)
    return out


def hunt_rounds(b6, b7, r8_dir):
    rounds = [("r3", H3 + "/cases.json", H4 + "/r3_rerun_build6.json", None)] + \
             [("r%d" % n, "%s/r%d/cases.json" % (H4, n), "%s/r%d/results_build6.json" % (H4, n), None) for n in (4, 5, 6, 7)] + \
             [("r8", r8_dir + "/cases.json", r8_dir + "/results_build6.json", r8_dir + "/adjudication.json")]
    out = {}
    for rnd, cpath, rpath, apath in rounds:
        cases = {c["id"]: c for c in json.load(open(cpath))}
        rows = json.load(open(rpath))["rows"]
        final = json.load(open(apath))["final_labels"] if apath else None
        pair, by_label, loader_disagrees = Pair(b6, b7), collections.Counter(), []
        for r in rows:
            c = cases[r["id"]]
            label = final[r["id"]] if final else r["label"]
            k6, k7 = pair.run(c["sentence"], c["gathered"], {"id": r["id"], "label": label, "class": c.get("class")})
            by_label[(label, k6, k7)] += 1
            committed = r["gate"]["allowlist"][0]["v"]            # seed 42; the matcher is pure, so one seed stands for all
            if bool(committed) != k6:
                loader_disagrees.append(r["id"])
        s = pair.summary()
        s["cells(label,build6_kept,build7_kept)"] = {str(k): v for k, v in sorted(by_label.items())}
        s["files"] = {"cases": cpath, "results_build6": rpath, "labels_from": apath or rpath,
                      "cases_sha256": hashlib.sha256(open(cpath, "rb").read()).hexdigest()}
        s["cross_check_committed_build6_verdict_disagreements"] = loader_disagrees
        s["changed_rows"] = pair.changed
        out[rnd] = s
        print("[hunt]", rnd, s["n"], "changed", s["n_changed"], "loader disagreements", len(loader_disagrees), flush=True)
    return out


def natural_spelling_scan(b6, b7):
    """EXTRA: every store entity whose slug has a non-initial word "i", as the subject of one store fact in three spellings."""
    facts = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(SP.FACTS))
             if r["fact"].get("polarity") == "AFFIRM"]
    first = {}
    for svo in facts:
        for pos in (0, 2):
            s = svo[pos]
            if "i" in s.split("_")[1:] and s not in first:
                first[s] = (svo, pos)
    stub = StubRenderer()

    def spell(slug, how):
        words = [w for w in slug.split("_") if w]
        if how == "slug":
            return slug
        if how == "deslugged_lower":
            return " ".join(words)
        return " ".join("I" if w == "i" else w.title() for w in words)           # title(): "Charles I Of England"

    out = {"n_slugs": len(first), "spellings": {}}
    os.environ[SP.ALLOW] = "1"
    try:
        for how in ("slug", "deslugged_lower", "title_with_I"):
            pair = Pair(b6, b7)
            for slug, (svo, pos) in sorted(first.items()):
                # the stub's own sentence for the fact with this slug's spelling swapped in (subject or object position)
                surf, _ = stub.render_svo(*svo)
                shown = spell(slug, how)
                if how != "slug":
                    surf = surf.replace(slug, shown, 1)
                pair.run(surf, [list(svo)], {"slug": slug, "svo": list(svo), "spelling": how})
            s = pair.summary()
            s["changed_rows"] = pair.changed
            out["spellings"][how] = s
            print("[natural]", how, s["n"], "changed", s["n_changed"], flush=True)
    finally:
        os.environ.pop(SP.ALLOW, None)
    return out


def main(out_path, r8_dir=R8_DEFAULT):
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s" % out_path)
    b6, b6_sha = load_build6()
    out = {"_doc": __doc__.split("\n\n")[0], "backend": os.environ.get("SIM_BACKEND"), "device": "cpu", "host": socket.gethostname(),
           "argv": sys.argv, "build6_module": "research/runners/_moat_allowlist.py at git %s" % BUILD6_SHA, "build6_source_sha256": b6_sha,
           "build7_module_sha256": hashlib.sha256(open(B7.__file__, "rb").read()).hexdigest(), "r8_dir": r8_dir}
    t0 = time.time()
    out["stub_population"] = stub_population(b6, B7)
    out["heldout_corpora"] = heldout(b6, B7)
    out["hunt_rounds"] = hunt_rounds(b6, B7, r8_dir)
    out["extra_natural_spelling_cost_scan"] = natural_spelling_scan(b6, B7)
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({"stub": {k: v for k, v in out["stub_population"].items() if k != "changed_rows"}}, indent=1)[:2500])


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
