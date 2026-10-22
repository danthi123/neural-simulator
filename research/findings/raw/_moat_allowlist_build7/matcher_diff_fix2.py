"""Build-7 SECOND REVIEW-FIX DIAGNOSTIC (non-blind; report only, not a pass): a copy of matcher_diff_fix.py with one change of reference.
The pure matcher's verdict (`allowlist_verify`, no substrate, no parser seed) at THREE code versions, each read from git (or this worktree)
and executed as its own module object with `__file__` set to this worktree's path so the committed caps / names tables resolve the same way:
build 6 (commit 2b7ab8c1d), build 7 after its FIRST fix round (commit b76421436, the version the second review examined) and this worktree's
`research/runners/_moat_allowlist.py` (build 7 after the second fix round: `_abbrev` all-caps or a Roman ordinal, R7c(0) mixed case, R7c(1)
digits / person sides / the title-case function-word exemption / the no-capital u-ya-ye opener, R7c(2) "X ur Y", R7c(4) a class word after a
conjunction, R7c(5) a pointing word after the first word, weve / shes / thats, the object-side person relations of R7a, ye / ur / thine /
mine as pointing leads, and build-6 antecedent words in `_points_back`). The key "first_build7" in rows and summaries below names build 7
after its FIRST fix round (b76421436), kept so the row format matches matcher_diff_fix.json. Populations as in matcher_diff_widened.py ((1) stub, (2) held-out corpora, (3) hunt rounds 3-8,
(4) the six-spelling scan of the 64 store slugs with a non-initial "i" word), plus (5) a NATURAL-CASED spelling scan of EVERY store slug (the
subject or object of one store fact, in the stub's sentence with the slug's spelling swapped in): English title case (every word capitalised
except the English small words a/an/the/and/but/or/nor/for/so/yet/of/in/on/at/to/by/up/as/via/per/vs after the first word), the same with a
lone "i" word kept lower case, full title case (every word capitalised), and every names-table surface of the slug in English title case; and,
for the cost record only (not natural casing for a name, but the mouth writes store names so: held-out S2), sentence case and all lower case.
Identity of the build-7 module: its sha256 (the git_dirty flag of a provenance sidecar only says the tree had other uncommitted files).

usage (cwd = worktree, as a module; refuses to overwrite):
  python -m research.findings.raw._moat_allowlist_build7.matcher_diff_fix2 <out.json> [<r8 dir>]
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
BUILD7A_SHA = "b76421436"      # build 7 after its FIRST fix round (the second review's subject); rows call it "first_build7"
HELD = "research/findings/raw/_moat_slug_norm_heldout"
H3 = "research/findings/raw/_moat_allowlist_hunt3"
H4 = "research/findings/raw/_moat_allowlist_hunt4"
R8_DEFAULT = "/home/dant123/Projects/sim-rev/wt-hunt8/research/findings/raw/_moat_allowlist_hunt4/r8"


def load_module(sha, tag):
    """`research/runners/_moat_allowlist.py` at git `sha`, executed in a fresh module namespace -> (module, sha256 of its source)."""
    src = subprocess.check_output(["git", "show", "%s:research/runners/_moat_allowlist.py" % sha]).decode("utf-8")
    spec = importlib.util.spec_from_loader("research.runners._moat_allowlist_%s_snapshot" % tag, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(os.getcwd(), "research", "runners", "_moat_allowlist.py")
    exec(compile(src, "<_moat_allowlist.py at %s>" % sha, "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode("utf-8")).hexdigest()


def rule_of(reason):
    """Which build-7 rule a reject reason names (by the texts of `_pronoun_name_reading` / `_inner_first_person_reading`)."""
    r = reason or ""
    if "reads as the pronoun" in r and "(written in lower case" in r:
        return "R7a one-word pronoun name, lower case"
    if "reads as the pronoun" in r and "(the first word of the sentence" in r:
        return "R7a one-word pronoun name, sentence-initial"
    if "a lower-case 'i' (word" in r:
        return "R7b(1) lower-case i, no capital in the name"
    if "'i' (word" in r and "before the lower-case word" in r:
        return "R7b(2) lower-case i then a lower-case word"
    if "'I' (word" in r and "before the lower-case word" in r:
        return "R7b(2) upper-case I then a lower-case word"
    if "a conjunction before 'I'" in r:
        return "R7b(3) conjunction then I"
    if "written in mixed case, neither lower case, title case nor all-caps: not a name" in r:
        return "R7c(0) mixed case"
    if "in the person relation" in r and "opens with the pronoun / determiner" in r:
        return "R7c(1) opener in a person relation"
    if "opens with the pronoun / determiner" in r:
        return "R7c(1) opener before an uncapitalised word"
    if "in a name with no capital letter: a pronoun phrase, not a store-name display" in r:
        return "R7c(1c) u/us/ya/ye/ur opener, no capital"
    if "is made of pronouns and conjunctions only" in r:
        return "R7c(1b) pronouns and conjunctions only"
    if "written as prose" in r and "the pronoun '" in r:
        return "R7c(2) archaic / colloquial form"
    if "without its apostrophe" in r:
        return "R7c(3) contraction"
    if "directly after the conjunction" in r:
        return "R7c(4) class word after a conjunction"
    if "written in lower case inside the name" in r:
        return "R7c(5) lower-case pointing word inside the name"
    if "before the lower-case word" in r and "pronoun / determiner" in r:
        return "R7c(5) pointing word before a lower-case word"
    if "led by 'the' or a pointing word" in r:
        return "_points_back"
    if "reads as the pronoun" in r and "mixed case" in r:
        return "R7a mixed case"
    if "reads as the pronoun" in r and "where no title reading stands" in r:
        return "R7a person / place / object relation"
    return "other: " + r[:80]


class Trio:
    """The verdict of build 6 (`m6`), the first build 7 (`m7a`) and the widened build 7 (`m7`) on (sentence, gathered); counts and the changed
    rows, against build 6 and against the first build 7."""

    def __init__(self, m6, m7a, m7):
        self.m6, self.m7a, self.m7 = m6, m7a, m7
        self.c6, self.c7a = collections.Counter(), collections.Counter()
        self.ch6, self.ch7a = [], []

    def run(self, sentence, gathered, meta):
        r6, r7a, r7 = (m.allowlist_verify(sentence, gathered) for m in (self.m6, self.m7a, self.m7))
        k6, k7a, k7 = bool(r6["accepted"]), bool(r7a["accepted"]), bool(r7["accepted"])
        self.c6[(k6, k7)] += 1
        self.c7a[(k7a, k7)] += 1
        row = dict(meta, sentence=sentence, build6_kept=k6, first_build7_kept=k7a, build7_kept=k7, build7_rule=rule_of(r7["reject_reason"]),
                   build7_reason=r7["reject_reason"])
        if k6 != k7:
            self.ch6.append(row)
        if k7a != k7:
            self.ch7a.append(row)
        return k6, k7a, k7

    @staticmethod
    def _sum(cells, changed, lab):
        return {"cells(%s_kept,build7_kept)" % lab: {str(k): v for k, v in sorted(cells.items())}, "n_changed": len(changed),
                "n_rejected_by_build7": sum(1 for x in changed if not x["build7_kept"]),
                "n_kept_by_build7": sum(1 for x in changed if x["build7_kept"]),
                "changed_by_rule": dict(collections.Counter(x["build7_rule"] for x in changed)), "changed_rows": changed}

    def summary(self):
        return {"n": sum(self.c6.values()), "vs_build6": self._sum(self.c6, self.ch6, "build6"),
                "vs_first_build7": self._sum(self.c7a, self.ch7a, "first_build7")}


def stub_population(m6, m7a, m7):
    aff = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(SP.FACTS))
           if r["fact"].get("polarity") == "AFFIRM"]
    trio, stub, t0 = Trio(m6, m7a, m7), StubRenderer(), time.time()
    os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
    os.environ[SP.ALLOW] = "1"                 # the stub writes the shape-4 phrase for the nine no-reading relations under the flag only
    try:
        for i, svo in enumerate(aff):
            surf, _asserted = stub.render_svo(*svo)
            trio.run(surf, [list(svo)], {"svo": list(svo)})
            if (i + 1) % 20000 == 0:
                print("[stub] %d/%d t=%.0fs changed vs build6=%d vs first build7=%d" % (i + 1, len(aff), time.time() - t0, len(trio.ch6),
                                                                                       len(trio.ch7a)), flush=True)
    finally:
        os.environ.pop(SP.ALLOW, None)
    out = trio.summary()
    out["n_population"] = len(aff)
    try:
        com = json.load(open(HELD + "/stub_population_exhaustive_allowlist6.json"))
        out["cross_check_committed_build6"] = {
            "file": HELD + "/stub_population_exhaustive_allowlist6.json", "committed_n_facts": com["n_facts"],
            "committed_kept_count": round(com["allowlist_answer_rate"] * com["n_facts"]),
            "this_run_build6_kept_count": sum(v for (k6, _k7), v in trio.c6.items() if k6),
            "this_run_build7_kept_count": sum(v for (_k6, k7), v in trio.c6.items() if k7)}
    except Exception as exc:                   # report, do not hide
        out["cross_check_committed_build6"] = "unavailable: %s: %s" % (type(exc).__name__, exc)
    return out


def heldout(m6, m7a, m7):
    sys.path.insert(0, os.path.join(os.getcwd(), HELD))
    from unique_sentences import unique       # noqa: E402  (the strata measurer's own numbering)
    out = {}
    for name, cpath, lpath in (("S1 corpus.json", "corpus.json", "labels.json"), ("S2 s2_corpus.json", "s2_corpus.json", "s2_labels.json"),
                               ("S3 s3_corpus.json", "s3_corpus.json", "s3_labels.json")):
        corpus = json.load(open(os.path.join(HELD, cpath)))
        lab_doc = json.load(open(os.path.join(HELD, lpath)))
        lab = {(e["turn"], tuple(e["svo"]), e["surface"]): (e["label"], e.get("variant")) for e in lab_doc["labels"]}
        turns = {t["label"]: t for t in corpus["turns"] if t.get("calls")}
        trio = Trio(m6, m7a, m7)
        by_label = collections.Counter()
        for u in unique(corpus):
            k = (u["turn"], tuple(u["svo"]), u["surface"])
            label, variant = lab[k]
            k6, _k7a, k7 = trio.run(u["surface"], turns[u["turn"]]["gated"], {"turn": u["turn"], "svo": u["svo"], "label": label,
                                                                             "variant": variant})
            by_label[(label, k6, k7)] += 1
        s = trio.summary()
        s["cells(label,build6_kept,build7_kept)"] = {str(k): v for k, v in sorted(by_label.items())}
        out[name] = s
        print("[heldout]", name, s["n"], "changed vs build6", len(trio.ch6), "vs first build7", len(trio.ch7a), flush=True)
    return out


def hunt_rounds(m6, m7a, m7, r8_dir):
    rounds = [("r3", H3 + "/cases.json", H4 + "/r3_rerun_build6.json", None)] + \
             [("r%d" % n, "%s/r%d/cases.json" % (H4, n), "%s/r%d/results_build6.json" % (H4, n), None) for n in (4, 5, 6, 7)] + \
             [("r8", r8_dir + "/cases.json", r8_dir + "/results_build6.json", r8_dir + "/adjudication.json")]
    out = {}
    for rnd, cpath, rpath, apath in rounds:
        cases = {c["id"]: c for c in json.load(open(cpath))}
        rows = json.load(open(rpath))["rows"]
        final = json.load(open(apath))["final_labels"] if apath else None
        trio, by_label, loader_disagrees = Trio(m6, m7a, m7), collections.Counter(), []
        for r in rows:
            c = cases[r["id"]]
            label = final[r["id"]] if final else r["label"]
            k6, _k7a, k7 = trio.run(c["sentence"], c["gathered"], {"id": r["id"], "label": label, "class": c.get("class")})
            by_label[(label, k6, k7)] += 1
            committed = r["gate"]["allowlist"][0]["v"]            # seed 42; the matcher is pure, so one seed stands for all
            if bool(committed) != k6:
                loader_disagrees.append(r["id"])
        s = trio.summary()
        s["cells(label,build6_kept,build7_kept)"] = {str(k): v for k, v in sorted(by_label.items())}
        s["files"] = {"cases": cpath, "results_build6": rpath, "labels_from": apath or rpath,
                      "cases_sha256": hashlib.sha256(open(cpath, "rb").read()).hexdigest()}
        s["cross_check_committed_build6_verdict_disagreements"] = loader_disagrees
        out[rnd] = s
        print("[hunt]", rnd, s["n"], "changed vs build6", len(trio.ch6), "vs first build7", len(trio.ch7a), "loader disagreements",
              len(loader_disagrees), flush=True)
    return out


SPELLINGS = ("slug", "deslugged_lower", "title_with_I", "title_with_i", "sentence_with_I", "sentence_with_i")


def spell(slug, how):
    words = [w for w in slug.split("_") if w]
    if how == "slug":
        return slug
    if how == "deslugged_lower":
        return " ".join(words)
    if how.startswith("title"):                         # str.title() of each word; the lone "i" word is "I" or stays "i"
        lone = "I" if how == "title_with_I" else "i"
        return " ".join(lone if w == "i" else w.title() for w in words)
    lone = "I" if how == "sentence_with_I" else "i"    # sentence case: first word capitalised, the rest lower case
    return " ".join(w.capitalize() if k == 0 else (lone if w == "i" else w) for k, w in enumerate(words))


def natural_spelling_scan(m6, m7a, m7):
    """EXTRA: every store entity whose slug has a non-initial word "i", as the subject or object of one store fact, in six spellings."""
    facts = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(SP.FACTS))
             if r["fact"].get("polarity") == "AFFIRM"]
    first = {}
    for svo in facts:
        for s in (svo[0], svo[2]):
            if "i" in s.split("_")[1:] and s not in first:
                first[s] = svo
    stub = StubRenderer()
    out = {"n_slugs": len(first), "spellings": {}, "verdict_by_slug": {}}
    os.environ[SP.ALLOW] = "1"
    try:
        for how in SPELLINGS:
            trio = Trio(m6, m7a, m7)
            for slug, svo in sorted(first.items()):
                surf, _ = stub.render_svo(*svo)             # the stub's own sentence for the fact, then this slug's spelling swapped in
                if how != "slug":
                    surf = surf.replace(slug, spell(slug, how), 1)
                k6, k7a, k7 = trio.run(surf, [list(svo)], {"slug": slug, "svo": list(svo), "spelling": how})
                out["verdict_by_slug"].setdefault(slug, {})[how] = {"build6": k6, "first_build7": k7a, "build7": k7}
            s = trio.summary()
            out["spellings"][how] = s
            print("[natural]", how, s["n"], "changed vs build6", len(trio.ch6), "vs first build7", len(trio.ch7a), flush=True)
    finally:
        os.environ.pop(SP.ALLOW, None)
    return out


SMALL = frozenset("a an the and but or nor for so yet of in on at to by up as via per vs".split())


def english_title(text, lone_i="I"):
    """English title case of a space-separated display: every word capitalised except the small words after the first; a lone "i" -> `lone_i`."""
    out = []
    for k, w in enumerate(text.split(" ")):
        if w == "i":
            out.append(lone_i)
        elif k > 0 and w in SMALL:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def store_natural_scan(m6, m7a, m7):
    """EXTRA (build-7 fix): every store slug, as the subject or object of its first store fact, in natural cased spellings."""
    facts = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(SP.FACTS))
             if r["fact"].get("polarity") == "AFFIRM"]
    first = {}
    for svo in facts:
        for s in (svo[0], svo[2]):
            first.setdefault(s, svo)
    table = B7.names_table()
    stub = StubRenderer()
    hows = {"english_title_I": lambda sl: [english_title(" ".join(w for w in sl.split("_") if w))],
            "english_title_i": lambda sl: [english_title(" ".join(w for w in sl.split("_") if w), "i")],
            "full_title": lambda sl: [" ".join(w[:1].upper() + w[1:] for w in sl.split("_") if w)],
            "names_table_english_title": lambda sl: sorted(english_title(x) for x in (table.entities.get(sl) or ())),
            # not natural casing for a name, reported for the cost record: the mouth does write store names this way (held-out S2)
            "sentence_case": lambda sl: [" ".join(w.capitalize() if k == 0 else w for k, w in enumerate(x for x in sl.split("_") if x))],
            "deslugged_lower": lambda sl: [" ".join(w for w in sl.split("_") if w)]}
    out = {"n_slugs": len(first), "spellings": {}}
    touch = collections.Counter()              # store facts that touch each slug (as subject or object)
    for a, _v, p in facts:
        touch[a] += 1
        if p != a:
            touch[p] += 1
    os.environ[SP.ALLOW] = "1"
    try:
        for how, fn in hows.items():
            trio = Trio(m6, m7a, m7)
            for slug, svo in sorted(first.items()):
                base, _ = stub.render_svo(*svo)
                for sp in fn(slug):
                    if not sp or slug not in base:
                        continue
                    trio.run(base.replace(slug, sp, 1), [list(svo)], {"slug": slug, "svo": list(svo), "spelling": how, "spelled": sp})
            s = trio.summary()
            for lab, changed in (("vs_build6", trio.ch6), ("vs_first_build7", trio.ch7a)):
                newly = sorted({x["slug"] for x in changed if not x["build7_kept"]})
                s[lab]["newly_rejected_slugs"] = newly
                s[lab]["store_facts_touching_newly_rejected"] = sum(touch[x] for x in newly)
            s["vs_build6"].pop("changed_rows")          # the changes against build 6 hold build 7's accepted costs; keep the counts only
            s["vs_build6"]["changed_rows_note"] = "rows omitted (counts above); the rows that changed against the reviewed build 7 are listed"
            out["spellings"][how] = s
            print("[store-natural]", how, s["n"], "changed vs build6", s["vs_build6"]["n_changed"], "vs build7 fix 1",
                  len(trio.ch7a), flush=True)
    finally:
        os.environ.pop(SP.ALLOW, None)
    return out


def main(out_path, r8_dir=R8_DEFAULT):
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s" % out_path)
    m6, m6_sha = load_module(BUILD6_SHA, "build6")
    m7a, m7a_sha = load_module(BUILD7A_SHA, "build7fix1")
    out = {"_doc": __doc__.split("\n\n")[0], "backend": os.environ.get("SIM_BACKEND"), "device": "cpu", "host": socket.gethostname(),
           "argv": sys.argv, "build6_module": "research/runners/_moat_allowlist.py at git %s" % BUILD6_SHA, "build6_source_sha256": m6_sha,
           "first_build7_module": "build 7 after its FIRST fix round: research/runners/_moat_allowlist.py at git %s" % BUILD7A_SHA,
           "first_build7_source_sha256": m7a_sha,
           "build7_module_sha256": hashlib.sha256(open(B7.__file__, "rb").read()).hexdigest(), "r8_dir": r8_dir}
    t0 = time.time()
    out["stub_population"] = stub_population(m6, m7a, B7)
    out["heldout_corpora"] = heldout(m6, m7a, B7)
    out["hunt_rounds"] = hunt_rounds(m6, m7a, B7, r8_dir)
    out["extra_natural_spelling_cost_scan"] = natural_spelling_scan(m6, m7a, B7)
    out["extra_store_natural_cased_scan"] = store_natural_scan(m6, m7a, B7)
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(out_path, "w"), indent=1)
    print("done", out_path, out["elapsed_s"], "s")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
