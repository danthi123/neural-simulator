"""Generator of the allowlist claim checker's CASE-FIDELITY table (build 3, `BRAIN_MOAT_ALLOWLIST`, default OFF).

Controller ruling 2026-10-20 ("no invented acronyms"): inside every SUBJ/OBJ span the matcher maps to a gathered slug, an all-caps token
of 2+ letters is kept only if that same token is written in all capitals in one of the entity's own store names (its canonical label or
aliases in the source the store was built from). The store's slug rule lowercases, so "the WHO" spells the band's slug the_who; the
band's names never write "WHO", the World Health Organization's do.

THE SOURCE. wikidata_100k (/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k, 78,857 facts, built 2026-08-22) was made by
research/runners/_knowledge_core_curate.py from wikidata5m with top_entities / top_relations from its curation_report.json (25,000 /
60): pass 1 ranks entities by degree in wikidata5m_transductive_train.txt, `_alias_map(..., clean=True)` names each top entity by
`pick_clean_alias` over its line of wikidata5m_entity.txt (sanitized), relations by `sanitize` of the first alias of their line of
wikidata5m_relation.txt, and pass 2 keeps the best-scored object per (subject, relation). This module re-runs exactly those steps with
the curate module's own functions, CHECKS that the reproduced fact set equals the store's facts.json (so the entity ids behind every
slug are the ones the store used), then reads every alias of each such id (its whole line: canonical label and aliases) and records the
all-caps tokens by the matcher's own `caps_tokens`. A slug made by several ids gets the union of their names' tokens.

OUTPUT (default research/findings/raw/_moat_allowlist_caps/caps_table.json; refuses to overwrite): every store entity slug -> its sorted
all-caps tokens ([] when none), every store relation slug -> the all-caps tokens of its label, the entities for which no id (so no
name) was found, and the reproduction check. The matcher loads it once per process (`_moat_allowlist.caps_table`); production never
reads wikidata5m.

usage (cwd = worktree, as a module so the provenance door stamps the table):
  bash tools/memcap.sh 4 -- .venv/bin/python -m research.runners._moat_allowlist_caps_build \
      --out research/findings/raw/_moat_allowlist_caps/caps_table.json
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

from research.runners._knowledge_core_curate import _CRUFT, _alias_map, pick_clean_alias, sanitize
from research.runners._moat_allowlist import caps_tokens

DEFAULT_STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"
DEFAULT_DATA = "/home/dant123/Projects/sim-data/wikidata5m"
DEFAULT_OUT = "research/findings/raw/_moat_allowlist_caps/caps_table.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_vocab(store_dir):
    """-> (entity slugs, relation slugs, fact set) of the store's facts.json."""
    with open(os.path.join(store_dir, "facts.json"), "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    facts = {(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in rows}
    ents = {f[0] for f in facts} | {f[2] for f in facts}
    rels = {f[1] for f in facts}
    return ents, rels, facts, len(rows)


def reproduce(data_dir, top_entities, top_relations, n_facts):
    """The curate steps of research/runners/_knowledge_core_curate.py `curate` up to the plain fact list (no alias facts: the
    wikidata_100k report predates them) -> (ent_tok, rel_tok, facts)."""
    train = os.path.join(data_dir, "wikidata5m_transductive_train.txt")
    ent_txt = os.path.join(data_dir, "wikidata5m_entity.txt")
    rel_txt = os.path.join(data_dir, "wikidata5m_relation.txt")
    ent_freq, rel_freq = Counter(), Counter()
    with open(train, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            h, r, t = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
            if not (h and r and t):
                continue
            ent_freq[h] += 1
            ent_freq[t] += 1
            rel_freq[r] += 1
    top_ent = {e for e, _ in ent_freq.most_common(top_entities)}
    top_rel = {r for r, _ in rel_freq.most_common(top_relations)}
    ent_tok = _alias_map(ent_txt, top_ent, clean=True)
    rel_tok = _alias_map(rel_txt, top_rel, clean=False)
    best = {}
    with open(train, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            h, r, t = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
            if h not in top_ent or t not in top_ent or r not in top_rel:
                continue
            sh, sr, st = ent_tok.get(h), rel_tok.get(r), ent_tok.get(t)
            if not (sh and sr and st) or sh == st:
                continue
            key = (sh, sr)
            score = ent_freq[h] + ent_freq[t]
            cur = best.get(key)
            if cur is None or score > cur[1]:
                best[key] = (st, score)
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:n_facts]
    facts = {(s, rel, obj) for (s, rel), (obj, _sc) in ranked}
    return ent_tok, rel_tok, facts, len(best)


def picked_label(aliases):
    """The RAW alias `pick_clean_alias` chose (it returns only its sanitized form): the same loop, returning `best` unsanitized. The
    caller checks sanitize(picked_label(a)) == pick_clean_alias(a), so a drift from the store's picker cannot pass silently."""
    best, best_score = None, -1e18
    for i, al in enumerate(aliases[:20]):
        a = al.strip()
        low = a.lower()
        if len(a) < 2 or any(c in a for c in "/()+|:0123456789") or any(k in low for k in _CRUFT):
            continue
        w = a.split()
        if len(w) > 5:
            continue
        score = 0.0
        if 2 <= len(w) <= 4:
            score += 2
        if a[:1].isupper():
            score += 1.5
        if a.islower():
            score -= 1
        score -= 0.15 * i
        if score > best_score:
            best_score, best = score, a
    if best is None:
        best = aliases[0] if aliases else ""
    return best


def names_of(path, ids):
    """-> {id: [name, ...]} (every alias on the id's line: canonical label and aliases), read like `_alias_map` reads the file."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] not in ids:
                continue
            out[parts[0]] = parts[1:]
    return out


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
    ents, rels, store_facts, n_rows = store_vocab(a.store_dir)
    ent_tok, rel_tok, rep_facts, n_candidates = reproduce(a.data_dir, report["top_entities"], report["top_relations"], a.n_facts)
    identical = rep_facts == store_facts
    print("[caps] reproduced %d facts (%d candidates) vs store %d: identical=%s (%.0fs)"
          % (len(rep_facts), n_candidates, len(store_facts), identical, time.time() - t0), flush=True)

    slug_ids = {}
    for eid, tok in ent_tok.items():
        if tok in ents:
            slug_ids.setdefault(tok, set()).add(eid)
    rel_ids = {}
    for rid, tok in rel_tok.items():
        if tok in rels:
            rel_ids.setdefault(tok, set()).add(rid)
    ent_names = names_of(os.path.join(a.data_dir, "wikidata5m_entity.txt"), {i for s in slug_ids.values() for i in s})
    rel_names = names_of(os.path.join(a.data_dir, "wikidata5m_relation.txt"), {i for s in rel_ids.values() for i in s})

    entities, labels, no_names, n_names, picker_drift = {}, {}, [], 0, []
    for slug in sorted(ents):
        toks = set()
        names = [n for eid in sorted(slug_ids.get(slug, ())) for n in ent_names.get(eid, ())]
        if not names:
            no_names.append(slug)
        n_names += len(names)
        for n in names:
            toks.update(caps_tokens(n))
        entities[slug] = sorted(toks)
        labs = set()                   # the canonical label(s) as written, kept only when a label itself holds an all-caps token
        for eid in sorted(slug_ids.get(slug, ())):
            al = ent_names.get(eid, [])
            lab = picked_label(al)
            if sanitize(lab) != pick_clean_alias(al) or sanitize(lab) != slug:
                picker_drift.append([slug, eid])
            if caps_tokens(lab):
                labs.add(lab)
        if labs:
            labels[slug] = sorted(labs)
    relations = {}
    for slug in sorted(rels):
        # the relation's LABEL (the first alias, the one `_alias_map(clean=False)` sanitized into the slug)
        rel_labels = [rel_names[rid][0] for rid in sorted(rel_ids.get(slug, ())) if rel_names.get(rid)]
        relations[slug] = sorted({t for lab in rel_labels for t in caps_tokens(lab)})
    no_label = [s for s in sorted(rels) if not rel_ids.get(s)]

    doc = {
        "_doc": ("Case-fidelity table of the allowlist claim checker (build 3, BRAIN_MOAT_ALLOWLIST): entity slug -> the all-caps tokens "
                 "(2+ letters, research/runners/_moat_allowlist.caps_tokens) written in any of that entity's names in the source the store "
                 "was built from; relation slug -> those of its label. A token absent here is an invented acronym in that span."),
        "runner": "research/runners/_moat_allowlist_caps_build.py",
        "source": {"store_dir": a.store_dir, "store_facts_sha256": _sha256(os.path.join(a.store_dir, "facts.json")),
                   "data_dir": a.data_dir, "top_entities": report["top_entities"], "top_relations": report["top_relations"],
                   "n_facts_cap": a.n_facts},
        "reproduction": {"store_fact_rows": n_rows, "store_facts": len(store_facts), "reproduced_facts": len(rep_facts),
                         "candidates": n_candidates, "identical": identical,
                         "only_in_store": len(store_facts - rep_facts), "only_reproduced": len(rep_facts - store_facts)},
        "n_entities": len(entities), "n_entities_with_caps": sum(1 for v in entities.values() if v),
        "n_caps_tokens": sum(len(v) for v in entities.values()), "n_names_read": n_names,
        "n_slugs_from_several_ids": sum(1 for s in ents if len(slug_ids.get(s, ())) > 1),
        "no_names": no_names, "relations_without_label": no_label, "picker_drift": picker_drift,
        "labels_doc": ("canonical label(s) as written (the raw alias pick_clean_alias chose, checked against it), only where the label "
                       "itself holds an all-caps token (the test sample of tests/test_moat_allowlist_caps.py reads them)"),
        "labels": labels,
        "relations": relations,
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
    print("[caps] %d entities (%d with all-caps tokens, %d tokens), %d without names, %d relations; %d bytes -> %s (%.0fs)"
          % (len(entities), doc["n_entities_with_caps"], doc["n_caps_tokens"], len(no_names), len(relations),
             os.path.getsize(a.out), a.out, time.time() - t0), flush=True)
    return 0 if identical and not picker_drift else 3


if __name__ == "__main__":
    sys.exit(main())
