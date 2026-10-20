"""Generator of the allowlist claim checker's EXACT-SURFACE table (build 4, `BRAIN_MOAT_ALLOWLIST`, default OFF).

Build 4 rule, FOLD TO FIND, EXACT TO ACCEPT: the store's lossy slug fold may still FIND which gathered slug a token run could be, but
a SUBJ/OBJ span is ACCEPTED only if its displayed text, compared by `surface_form` (NFC, lowercase, whitespace runs -> one space,
stripped), equals one of that slug's canonical surfaces: the de-slugged form (underscores -> spaces), or one of that entity's own store
names (canonical label or alias in wikidata5m, the source the store was built from) whose `sanitize()` equals the slug. Round 5 leaked
24 sentences through acceptance by fold-equality ("George V" + U+013A folds to george_v while it reads as George VI).

THE SOURCE AND THE ENTITY IDS are pinned exactly as the build-3 case table pins them: this module reuses
research/runners/_moat_allowlist_caps_build.py `reproduce`, which re-runs the curate steps of research/runners/_knowledge_core_curate.py
and must rebuild the store's 78,857 facts identically, so the ids behind every slug are the ones the store used.

OUTPUT (default research/findings/raw/_moat_allowlist_names/names_table.json; refuses to overwrite). ONLY THE EXCEPTIONS are stored: for
each store entity slug, the `surface_form` of every store name with sanitize(name) == slug whose form differs from the de-slugged form
(an accented, hyphenated or otherwise punctuated name). A slug with no row accepts only its de-slugged form. The matcher loads the
table once per process (`_moat_allowlist.names_table`); a missing or unreadable table accepts only de-slugged forms (fail closed).

usage (cwd = worktree, as a module so the provenance door stamps the table):
  bash tools/memcap.sh 4 -- .venv/bin/python -m research.runners._moat_allowlist_names_build \
      --out research/findings/raw/_moat_allowlist_names/names_table.json
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)

import argparse
import json
import os
import sys
import time

from research.runners._knowledge_core_curate import sanitize
from research.runners._moat_allowlist import deslug, surface_form
from research.runners._moat_allowlist_caps_build import DEFAULT_DATA, DEFAULT_STORE, _sha256, names_of, reproduce, store_vocab

DEFAULT_OUT = "research/findings/raw/_moat_allowlist_names/names_table.json"


def exceptions_for(slug, names):
    """-> sorted surface forms of `names` that spell `slug` by the store's rule and differ from its de-slugged form."""
    plain = deslug(slug)
    return sorted({surface_form(n) for n in names if sanitize(n) == slug} - {plain})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=DEFAULT_STORE)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-facts", type=int, default=100000,
                    help="the curate cap; wikidata_100k had fewer candidates than this, so every candidate was kept")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("refusing to overwrite %s" % a.out, file=sys.stderr)
        return 2
    t0 = time.time()
    with open(os.path.join(a.store_dir, "curation_report.json"), "r", encoding="utf-8") as fh:
        report = json.load(fh)
    ents, _rels, store_facts, n_rows = store_vocab(a.store_dir)
    ent_tok, _rel_tok, rep_facts, n_candidates = reproduce(a.data_dir, report["top_entities"], report["top_relations"], a.n_facts)
    identical = rep_facts == store_facts
    print("[names] reproduced %d facts (%d candidates) vs store %d: identical=%s (%.0fs)"
          % (len(rep_facts), n_candidates, len(store_facts), identical, time.time() - t0), flush=True)

    slug_ids = {}
    for eid, tok in ent_tok.items():
        if tok in ents:
            slug_ids.setdefault(tok, set()).add(eid)
    ent_names = names_of(os.path.join(a.data_dir, "wikidata5m_entity.txt"), {i for s in slug_ids.values() for i in s})

    entities, no_names, n_names, n_spelling, n_truncated = {}, [], 0, 0, 0
    for slug in sorted(ents):
        names = [n for eid in sorted(slug_ids.get(slug, ())) for n in ent_names.get(eid, ())]
        if not names:
            no_names.append(slug)
        n_names += len(names)
        spelling = [n for n in names if sanitize(n) == slug]
        n_spelling += len(spelling)
        n_truncated += sum(1 for n in spelling if len(sanitize(n, maxlen=10 ** 9)) > len(slug))
        rows = exceptions_for(slug, names)
        if rows:
            entities[slug] = rows

    doc = {
        "_doc": ("Exact-surface table of the allowlist claim checker (build 4, BRAIN_MOAT_ALLOWLIST: fold to find, exact to accept). "
                 "entity slug -> the surface forms (research/runners/_moat_allowlist.surface_form: NFC, lowercase, whitespace runs "
                 "collapsed) of that entity's store names with sanitize(name) == slug that differ from the de-slugged form. Only these "
                 "exceptions are stored: a slug with no row accepts only its de-slugged form."),
        "runner": "research/runners/_moat_allowlist_names_build.py",
        "source": {"store_dir": a.store_dir, "store_facts_sha256": _sha256(os.path.join(a.store_dir, "facts.json")),
                   "data_dir": a.data_dir, "top_entities": report["top_entities"], "top_relations": report["top_relations"],
                   "n_facts_cap": a.n_facts},
        "reproduction": {"store_fact_rows": n_rows, "store_facts": len(store_facts), "reproduced_facts": len(rep_facts),
                         "candidates": n_candidates, "identical": identical,
                         "only_in_store": len(store_facts - rep_facts), "only_reproduced": len(rep_facts - store_facts)},
        "n_store_entities": len(ents), "n_names_read": n_names, "n_names_spelling_their_slug": n_spelling,
        "n_names_spelling_their_slug_only_after_the_40_char_cut": n_truncated,
        "n_rows": len(entities), "n_surfaces": sum(len(v) for v in entities.values()),
        "no_names": no_names,
        "entities": entities,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    head = {k: v for k, v in doc.items() if k != "entities"}
    with open(a.out, "w", encoding="utf-8") as fh:      # one entity per line: small diffs, readable
        body = json.dumps(head, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write(body[:-2] + ',\n "entities": {\n')
        items = list(entities.items())
        for i, (k, v) in enumerate(items):
            fh.write("  %s: %s%s\n" % (json.dumps(k, ensure_ascii=False), json.dumps(v, ensure_ascii=False),
                                         "," if i < len(items) - 1 else ""))
        fh.write(" }\n}\n")
    with open(a.out, "r", encoding="utf-8") as fh:      # the file is valid JSON and round-trips
        assert json.load(fh)["entities"] == entities
    print("[names] %d store entities, %d rows, %d surfaces, %d without names; %d bytes -> %s (%.0fs)"
          % (len(ents), len(entities), doc["n_surfaces"], len(no_names), os.path.getsize(a.out), a.out, time.time() - t0), flush=True)
    return 0 if identical else 3


if __name__ == "__main__":
    sys.exit(main())
