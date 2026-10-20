"""Retention cost of the allowlist checker's build-3 case rule ("no invented acronyms") on the held-out strata, measured BEFORE any heavy
build-3 measurement (controller ruling 2026-10-20, item 5).

For every labelled unique sentence of S1 / S2 / S3 (research/findings/raw/_moat_slug_norm_heldout/{corpus,s2_corpus,s3_corpus}.json with
their committed labels) this records whether a SUBJ/OBJ span of the sentence (a run the matcher maps to a gathered subject/object slug of
its turn) holds an all-caps token (`caps_tokens`), and the matcher's verdict with the case rule (build 3) and without it (build 2's
matcher: the same code with `_case_violation` disabled). The matcher is called directly, as the production claim gate calls it under the
flag (`allowlist_verify(surface, gathered)`, every sentence of the surface must be kept; `allowlist_keep` per sentence is recorded too).
It is a pure host function: no parser, no seed.

usage (cwd = worktree; refuses to overwrite):
  .venv/bin/python -m research.findings.raw._moat_allowlist_caps.heldout_caps_cost research/findings/raw/_moat_allowlist_caps/heldout_caps_cost.json
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)

import collections
import json
import os
import sys

from research.runners import _moat_allowlist as A

H = "research/findings/raw/_moat_slug_norm_heldout"
STRATA = {"S1": ("corpus.json", "labels.json"), "S2": ("s2_corpus.json", "s2_labels.json"), "S3": ("s3_corpus.json", "s3_labels.json")}


def unique(corpus):
    """The (turn, svo, surface) sentences of a corpus in first-appearance order (as unique_sentences.unique numbers them)."""
    out, seen = [], set()
    for t in corpus["turns"]:
        for c in t.get("calls", []):
            k = (t["label"], tuple(c["svo"]), c["surface"])
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def span_caps(surface, gathered):
    """-> [(token, slug, span_text)] for every all-caps token inside a SUBJ/OBJ span of any sentence of `surface` (span = a token the
    matcher's normalisation made from a gathered subject/object slug; independent of whether the sentence parses)."""
    key = A._key_of(gathered)
    lex = A._lexicon(key)
    ents = {t for f in key for t in (f[0], f[2])}
    out = []
    for s in A.split_sentences(surface):
        tr = lex.normalise_tracked(s)
        if tr is None:
            out.append(("<untracked>", None, s))
            continue
        for tok, orig in zip(tr[1], tr[2]):
            if tok in ents:
                out.extend((c, tok, orig) for c in A.caps_tokens(orig))
    return out


def verdicts(surface, gathered):
    res = A.allowlist_verify(surface, gathered)
    keeps = [A.allowlist_keep(s, gathered) for s in A.split_sentences(surface)]
    return bool(res["accepted"]), res["reject_reason"], keeps


def main(out):
    if os.path.exists(out):
        print("refusing to overwrite %s" % out, file=sys.stderr)
        return 2
    table = A.caps_table()
    assert table.error is None, table.error
    real_case = A._case_violation
    doc = {"_doc": __doc__.strip().splitlines()[0], "script": "research/findings/raw/_moat_allowlist_caps/heldout_caps_cost.py",
           "caps_table": os.path.relpath(A.CAPS_TABLE_PATH), "strata": {}}
    for name, (cfile, lfile) in STRATA.items():
        corpus = json.load(open(os.path.join(H, cfile)))
        labs = {(e["turn"], tuple(e["svo"]), e["surface"]): e["label"] for e in json.load(open(os.path.join(H, lfile)))["labels"]}
        turns = {t["label"]: t for t in corpus["turns"] if t.get("calls")}
        rows, by_label = [], collections.defaultdict(collections.Counter)
        for k in unique(corpus):
            turn, svo, surface = k
            lab = labs[k]
            gathered = turns[turn]["gated"]
            caps = span_caps(surface, gathered)
            b3, b3_why, b3_keeps = verdicts(surface, gathered)
            A._case_violation = lambda parses, origs: None              # build 2's matcher (the case rule off)
            try:
                b2, _b2_why, _ = verdicts(surface, gathered)
            finally:
                A._case_violation = real_case
            c = by_label[lab]
            c["n"] += 1
            c["caps_in_entity_span"] += bool(caps)
            c["b2_kept"] += b2
            c["b3_kept"] += b3
            c["caps_and_b3_rejected"] += bool(caps) and not b3
            c["lost_b2_kept_b3_rejected"] += b2 and not b3
            c["gained_b3_kept_b2_rejected"] += b3 and not b2
            if caps or b2 != b3:
                rows.append({"turn": turn, "svo": list(svo), "surface": surface, "label": lab,
                             "caps_in_entity_span": [list(x) for x in caps], "b2_kept": b2, "b3_kept": b3,
                             "b3_reject_reason": b3_why, "b3_allowlist_keep_per_sentence": [list(x) for x in b3_keeps]})
        doc["strata"][name] = {"by_label": {k: dict(v) for k, v in sorted(by_label.items())}, "rows": rows}
        f = by_label.get("F", collections.Counter())
        print("%s: F %d, with an all-caps token in an entity span %d, of those build 3 rejects %d; F kept build2 %d -> build3 %d (lost %d)"
              % (name, f["n"], f["caps_in_entity_span"], f["caps_and_b3_rejected"], f["b2_kept"], f["b3_kept"],
                 f["lost_b2_kept_b3_rejected"]), flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
