"""Build the AI teacher's VETTED curriculum fixture (research/fixtures/ai_teacher_curriculum_v1.json).

Two tiers, both fixed on disk so every run (local or pool) teaches byte-identical sentences:
  * wikidata -- `shares_border_with` triples from a curated wikidata knowledge bundle (the same data lake the brain's
    optional LTM is built from; the AI-teacher experiment runs with the LTM OFF, so the brain does not hold them).
    Selected mechanically: both entity names a single lowercase alphabetic token (the brain's 3-content-word SVO
    parser), every object distinct and never a subject. Each record keeps the bundle name, the fact's index in
    facts.json and the file's sha256, so it can be re-checked against the source.
  * novel -- invented nouns (blicket, dax, wug ...), which no source and no prior can supply. Ground truth is the
    curriculum itself. These are the facts the brain "cannot already know".
One template renders every fact: statement "the <s> <verb-3sg> the <o>", question "what does the <s> <verb>".

Usage (needs the data lake; pool nodes have ~/Projects/sim-data):
  python -m tools.ai_teacher_build_curriculum --bundle ~/Projects/sim-data/knowledge_bundles/wikidata_100k \
      --out research/fixtures/ai_teacher_curriculum_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

NOVEL = [  # (id, subject, verb, object) -- invented nouns + verbs the brain ALREADY KNOWS (the ZPD rule below)
    ("n01", "blicket", "eat", "dax"),
    ("n02", "wug", "chase", "toma"),
    ("n03", "fep", "carry", "zorb"),
    ("n04", "kiki", "find", "modi"),
    ("n05", "tulver", "follow", "pilk"),
    ("n06", "snerg", "hold", "vatch"),
    ("n07", "quenk", "pull", "lorp"),
    ("n08", "mib", "watch", "yarb"),
    ("n09", "gazzer", "reach", "bouba"),
    ("n10", "plonk", "visit", "frell"),
]
DISTRACTORS = ["narf", "quib", "tesk", "vorn", "jubb", "glim"]   # used ONLY by a corrupted-teacher belief
# ZPD RULE (seen in the dev-seed-3 plumbing run, 2026-09-24): the brain's D4 comprehension monitor judges a sentence
# whose verb AND both nouns are unfamiliar to its cue lexicons as noise and asks "what do they refer to?" instead of
# learning it ("the selva borders the osona" was refused; "the blicket eats the dax" was learned). A teacher speaks in
# words the learner knows, so every template verb is one the brain's verb lexicon covers ("touch" paraphrases
# shares_border_with). The refused-sentence case is reported, not hidden: see the pre-registration.
RELATION_TEMPLATES = {"shares_border_with": "touch"}
N_WIKIDATA = 4
# Interleaved teaching order: K takes a prefix, so every K mixes tiers.
ORDER_PATTERN = ["n", "w", "n", "n", "w", "n", "n", "w", "n", "n", "w", "n", "n", "n"]

_TOK = re.compile(r"^[a-z]{3,}$")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_wikidata(facts_json):
    rows = json.load(open(facts_json))
    novel_words = {w for (_i, s, _v, o) in NOVEL for w in (s, o)} | set(DISTRACTORS)
    picked, subjects, objects = [], set(), set()
    for idx, r in enumerate(rows):
        f = r.get("fact", r)
        rel = f.get("action")
        if rel not in RELATION_TEMPLATES or f.get("polarity", "AFFIRM") != "AFFIRM":
            continue
        s, o = str(f.get("agent", "")), str(f.get("patient", ""))
        if not (_TOK.match(s) and _TOK.match(o)) or s == o:
            continue
        if s in subjects or o in objects or o in subjects or s in objects or {s, o} & novel_words:
            continue
        picked.append({"subject": s, "verb": RELATION_TEMPLATES[rel], "object": o, "relation": rel, "index": idx})
        subjects.add(s)
        objects.add(o)
        if len(picked) == N_WIKIDATA:
            break
    return picked


def build(bundle_dir, out):
    facts_json = os.path.join(bundle_dir, "facts.json")
    digest = _sha256(facts_json)
    wd = select_wikidata(facts_json)
    if len(wd) < N_WIKIDATA:
        raise SystemExit("only %d usable wikidata facts found" % len(wd))
    facts = []
    for i, (fid, s, v, o) in enumerate(NOVEL):
        facts.append({"id": fid, "subject": s, "verb": v, "object": o, "tier": "novel",
                      "provenance": {"source": "invented nouns (curriculum is the ground truth)"}})
    for j, w in enumerate(wd):
        facts.append({"id": "w%02d" % (j + 1), "subject": w["subject"], "verb": w["verb"], "object": w["object"],
                      "tier": "wikidata",
                      "provenance": {"source": "wikidata knowledge bundle", "bundle": os.path.basename(bundle_dir),
                                     "facts_json_sha256": digest, "index": w["index"], "relation": w["relation"],
                                     "template": "the <agent> %s the <patient>" % {"touch": "touches"}.get(
                                         w["verb"], w["verb"] + "s")}})
    n_ids = [f["id"] for f in facts if f["tier"] == "novel"]
    w_ids = [f["id"] for f in facts if f["tier"] == "wikidata"]
    order = []
    for t in ORDER_PATTERN:
        pool = n_ids if t == "n" else w_ids
        if pool:
            order.append(pool.pop(0))
    order += n_ids + w_ids
    cur = {"version": 1, "device": "none (static curriculum data; no simulation ran)",
           "facts": facts, "order": order, "distractors": DISTRACTORS,
           "templates": {"tell": "the {subject} {verb_3sg} the {object}", "ask": "what does the {subject} {verb}"},
           "notes": "Built by tools/ai_teacher_build_curriculum.py. Template-only teacher; an LLM may later "
                    "paraphrase a vetted sentence, never originate a fact."}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cur, fh, indent=2)
        fh.write("\n")
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="research/fixtures/ai_teacher_curriculum_v1.json")
    a = ap.parse_args()
    cur = build(os.path.expanduser(a.bundle), a.out)
    print(json.dumps({"n_facts": len(cur["facts"]), "order": cur["order"]}))


if __name__ == "__main__":
    main()
