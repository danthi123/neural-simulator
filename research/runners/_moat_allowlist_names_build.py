"""Generator of the allowlist claim checker's EXACT-SURFACE table (build 4, `BRAIN_MOAT_ALLOWLIST`, default OFF).

Build 4 rule, FOLD TO FIND, EXACT TO ACCEPT: the store's lossy slug fold may still FIND which gathered slug a token run could be, but
a SUBJ/OBJ span is ACCEPTED only if its displayed text, compared by `surface_form` (NFC, lowercase, runs of U+0020 and the four
display-identical spaces -> one space, stripped), equals one of that slug's canonical surfaces: the slug itself, the de-slugged form
(underscores -> spaces), or a row of this table. Round 5 leaked 24 sentences through acceptance by fold-equality ("George V" + U+013A
folds to george_v while it reads as George VI).

THE SOURCE AND THE ENTITY IDS are pinned exactly as the build-3 case table pins them: this module reuses
research/runners/_moat_allowlist_caps_build.py `reproduce`, which re-runs the curate steps of research/runners/_knowledge_core_curate.py
and must rebuild the store's 78,857 facts identically. `fact_carriers` then re-reads the training triples once and records, for every
store slug, the Wikidata ids that CARRY a store fact under it: the subject id or object id of any source line (h, r, t) whose slugs
are exactly a store fact (sh, sr, st). It also replays pass 2's best-score rule over those lines and checks that the winning lines
rebuild the store's facts (`winners_reproduce`).

THE ROWS (review rulings 2026-10-20):
  F1 a slug gets rows only when its carrier set is EXACTLY ONE id, and then only from that id's names (canonical label or alias in
     wikidata5m) whose `sanitize()` equals the slug. A slug whose store facts come from two or more ids gets no row (only its
     de-slugged form and the slug itself are accepted): taking every id's names re-imported the fold ("Jan" with an acute accent was
     accepted for j_n, whose facts are Jon's; the hyphenated "Saint-Cloud" of the French commune for saint_cloud facts of St. Cloud,
     Minnesota).
  F4 a name holding a Unicode format (Cf) or control (Cc) character (a U+200E / U+200F mark) is excluded.
  F2 for each kept name whose surface holds ", ", the same surface with every ", " written " " is also a row ("sète, france" ->
     "sète france"); it is compared exactly, nothing is normalised at run time.
  Only rows that differ from the de-slugged form are stored.

OUTPUT (default research/findings/raw/_moat_allowlist_names/names_table.json; refuses to overwrite): `entities` (slug -> sorted
surfaces), `row_ids` (slug -> the one carrier id its rows came from), `multi_id_slugs` (slug -> its carrier ids, every store slug with
two or more), and the per-step counts (`steps`). The matcher reads only `entities`, once per process (`_moat_allowlist.names_table`); a
missing or unreadable table accepts only the slug and its de-slugged form (fail closed).

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
import unicodedata
from collections import Counter

from research.runners._knowledge_core_curate import sanitize
from research.runners._moat_allowlist import deslug, surface_form
from research.runners._moat_allowlist_caps_build import DEFAULT_DATA, DEFAULT_STORE, _sha256, names_of, reproduce, store_vocab

DEFAULT_OUT = "research/findings/raw/_moat_allowlist_names/names_table.json"


def has_format_or_control(name):
    """F4: True when `name` holds a Unicode format (Cf) or control (Cc) character."""
    return any(unicodedata.category(c) in ("Cf", "Cc") for c in name)


def comma_derived(form):
    """F2: a surface with every ", " written " " (re-run through `surface_form`, so no double space can result)."""
    return surface_form(form.replace(", ", " "))


def exceptions_for(slug, names, f4=True, f2=True):
    """-> sorted surface forms of `names` (ONE id's names) that spell `slug` by the store's rule, minus Cf/Cc names (F4), plus the
    ", " -> " " form of each (F2), minus the de-slugged form."""
    forms = {surface_form(n) for n in names if sanitize(n) == slug and not (f4 and has_format_or_control(n))}
    if f2:
        forms |= {comma_derived(f) for f in forms if ", " in f}
    return sorted(forms - {deslug(slug)})


def fact_carriers(data_dir, ent_tok, rel_tok, facts):
    """One pass over the training triples -> (carriers, winners_reproduce, n_lines). carriers: slug -> the set of ids that stand in the
    subject or object place of ANY source line whose (subject, relation, object) slugs are a store fact. winners_reproduce: pass 2's
    rule (per (subject slug, relation slug), the first line with the highest ent_freq[h] + ent_freq[t]) replayed over those lines
    rebuilds exactly `facts`."""
    train = os.path.join(data_dir, "wikidata5m_transductive_train.txt")
    freq, lines = Counter(), []
    with open(train, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            h, r, t = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
            if not (h and r and t):
                continue
            if h in ent_tok:
                freq[h] += 1
            if t in ent_tok:
                freq[t] += 1
            sh, sr, st = ent_tok.get(h), rel_tok.get(r), ent_tok.get(t)
            if not (sh and sr and st) or sh == st or (sh, sr, st) not in facts:
                continue
            lines.append((h, t, sh, sr, st))
    carriers, best = {}, {}
    for h, t, sh, sr, st in lines:
        carriers.setdefault(sh, set()).add(h)
        carriers.setdefault(st, set()).add(t)
        score = freq[h] + freq[t]
        cur = best.get((sh, sr))
        if cur is None or score > cur[1]:
            best[(sh, sr)] = (st, score)
    winners_reproduce = {(s, r, o) for (s, r), (o, _sc) in best.items()} == facts
    return carriers, winners_reproduce, len(lines)


def build_rows(ents, carriers, ent_names, all_ids=None):
    """-> (entities, row_ids, multi_id_slugs, steps). `carriers`: slug -> ids carrying its store facts; `ent_names`: id -> names;
    `all_ids` (optional): slug -> every id the alias map gave that slug (the build-4 rule before F1, counted in `steps` only)."""
    entities, row_ids, multi = {}, {}, {}
    steps = {"before_f1": {"rows": 0, "surfaces": 0}, "f1": {"rows": 0, "surfaces": 0}, "f1_f4": {"rows": 0, "surfaces": 0},
             "f1_f4_f2": {"rows": 0, "surfaces": 0}}
    n_f4_names, f4_examples, n_f2_added, no_carrier = 0, [], 0, []

    def count(key, rows):
        if rows:
            steps[key]["rows"] += 1
            steps[key]["surfaces"] += len(rows)

    for slug in sorted(ents):
        if all_ids is not None:
            count("before_f1", exceptions_for(slug, [n for i in sorted(all_ids.get(slug, ())) for n in ent_names.get(i, ())],
                                              f4=False, f2=False))
        ids = carriers.get(slug, set())
        if len(ids) > 1:
            multi[slug] = sorted(ids)
            continue
        if not ids:
            no_carrier.append(slug)
            continue
        (eid,) = ids
        names = ent_names.get(eid, ())
        count("f1", exceptions_for(slug, names, f4=False, f2=False))
        bad = [n for n in names if sanitize(n) == slug and has_format_or_control(n)]
        n_f4_names += len(bad)
        f4_examples.extend([slug, eid, n] for n in bad)
        no_f2 = exceptions_for(slug, names, f2=False)
        count("f1_f4", no_f2)
        rows = exceptions_for(slug, names)
        n_f2_added += len(rows) - len(no_f2)
        count("f1_f4_f2", rows)
        if rows:
            entities[slug], row_ids[slug] = rows, eid
    steps.update(n_multi_id_slugs=len(multi), n_f4_names_excluded=n_f4_names, f4_names_excluded=f4_examples,
                 n_f2_surfaces_added=n_f2_added, no_carrier=no_carrier)
    return entities, row_ids, multi, steps


def _write(path, head, blocks):
    """JSON with `head` first, then each (key, dict) block one entry per line: small diffs, readable."""
    with open(path, "w", encoding="utf-8") as fh:
        body = json.dumps(head, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write(body[:-2])
        for key, d in blocks:
            fh.write(',\n %s: {\n' % json.dumps(key))
            items = list(d.items())
            for i, (k, v) in enumerate(items):
                fh.write("  %s: %s%s\n" % (json.dumps(k, ensure_ascii=False), json.dumps(v, ensure_ascii=False),
                                             "," if i < len(items) - 1 else ""))
            fh.write(" }")
        fh.write("\n}\n")


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
    ent_tok, rel_tok, rep_facts, n_candidates = reproduce(a.data_dir, report["top_entities"], report["top_relations"], a.n_facts)
    identical = rep_facts == store_facts
    print("[names] reproduced %d facts (%d candidates) vs store %d: identical=%s (%.0fs)"
          % (len(rep_facts), n_candidates, len(store_facts), identical, time.time() - t0), flush=True)
    carriers, winners_ok, n_lines = fact_carriers(a.data_dir, ent_tok, rel_tok, store_facts)
    covered = set(carriers) == ents
    print("[names] %d source lines carry store facts; carriers cover every store slug: %s; pass-2 winners rebuild the store: %s (%.0fs)"
          % (n_lines, covered, winners_ok, time.time() - t0), flush=True)

    all_ids = {}
    for eid, tok in ent_tok.items():
        if tok in ents:
            all_ids.setdefault(tok, set()).add(eid)
    ent_names = names_of(os.path.join(a.data_dir, "wikidata5m_entity.txt"), {i for s in all_ids.values() for i in s})
    assert all(i in ent_tok and ent_tok[i] == s for s, ids in carriers.items() for i in ids)
    entities, row_ids, multi, steps = build_rows(ents, carriers, ent_names, all_ids)

    doc = {
        "_doc": ("Exact-surface table of the allowlist claim checker (build 4, BRAIN_MOAT_ALLOWLIST: fold to find, exact to accept). "
                 "entities: slug -> the surface forms (research/runners/_moat_allowlist.surface_form) of the store names of the ONE "
                 "Wikidata id that carries every store fact under that slug (row_ids), whose sanitize() is the slug, without Unicode "
                 "Cf/Cc characters, plus each such form with ', ' written ' ', minus the de-slugged form. A slug with no row (multi_id_slugs, "
                 "or no exception) accepts only the slug and its de-slugged form."),
        "runner": "research/runners/_moat_allowlist_names_build.py",
        "source": {"store_dir": a.store_dir, "store_facts_sha256": _sha256(os.path.join(a.store_dir, "facts.json")),
                   "data_dir": a.data_dir, "top_entities": report["top_entities"], "top_relations": report["top_relations"],
                   "n_facts_cap": a.n_facts},
        "reproduction": {"store_fact_rows": n_rows, "store_facts": len(store_facts), "reproduced_facts": len(rep_facts),
                         "candidates": n_candidates, "identical": identical,
                         "only_in_store": len(store_facts - rep_facts), "only_reproduced": len(rep_facts - store_facts),
                         "carrier_lines": n_lines, "carriers_cover_every_store_slug": covered,
                         "carrier_winners_rebuild_the_store": winners_ok},
        "carrier_rule": ("an id carries a store fact under slug s when it is the subject (or object) of a training line whose "
                         "(subject, relation, object) slugs are a store fact with s in that place"),
        "n_store_entities": len(ents), "n_slugs_with_one_carrier": sum(1 for s in ents if len(carriers.get(s, ())) == 1),
        "n_rows": len(entities), "n_surfaces": sum(len(v) for v in entities.values()),
        "steps": steps,
    }
    ok = identical and covered and winners_ok and not steps["no_carrier"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    _write(a.out, doc, [("multi_id_slugs", multi), ("row_ids", row_ids), ("entities", entities)])
    with open(a.out, "r", encoding="utf-8") as fh:      # the file is valid JSON and round-trips
        back = json.load(fh)
    assert back["entities"] == entities and back["row_ids"] == row_ids and back["multi_id_slugs"] == multi
    print("[names] %d store entities, %d multi-id slugs, %d rows, %d surfaces; steps %s; %d bytes -> %s (%.0fs)"
          % (len(ents), len(multi), len(entities), doc["n_surfaces"],
             {k: v for k, v in steps.items() if k not in ("f4_names_excluded", "no_carrier")},
             os.path.getsize(a.out), a.out, time.time() - t0), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
