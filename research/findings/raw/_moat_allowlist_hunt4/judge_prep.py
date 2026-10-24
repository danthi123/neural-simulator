"""Judge prep for a blind moat hunt round (round 8 on): anonymised, shuffled judge chunks and a private key. No judging, no checker.

usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt4.judge_prep <cases.json> <out_dir> --chunks N
                        [--seed S (default 8)] [--prefixes-only]
writes  <out_dir>/chunk_01.json .. chunk_NN.json   lists of judge items, each exactly {jid, gathered, sentence_as_written,
                                                    sentence_exact_escaped}: no id, expect, class, why, arguable or reply field
        <out_dir>/key.json                          PRIVATE (never shown to a judge): {"key": {jid: [case id, part]}, ...} where part is
                                                    "full" (the case's own sentence) or the key of a proper selection of a reply
                                                    case's sentences ("1", "1+2", "2", "1+3", ...: the subset_labels key family)

ITEMS. Every case gives one item: its `sentence` as written, with its `gathered` set. A reply case ("reply": true, 2 <= k <= 4
sentences, one per gathered triple) also gives one item per PROPER in-order non-empty selection of its sentences (Claude ruling
2026-10-21: the judges label every selection, so at most 15 items per case): every proper prefix ("1", "1+2", ...) and every other
in-order subsequence ("2", "1+3", "2+3", ...; the composer emits one when it drops a fact whose sentence fails alone before the
joined check). Each is the selected sentences joined as the composer joins them under the allowlist flag (`run_hunt.reply_join` with
the composer's `_terminate`: trailing U+0020/LF/CR stripped, "." added to a sentence with no terminal punctuation, one U+0020 between
sentences), judged against the case's FULL gathered set (the union the joined check reads). --prefixes-only writes the proper
prefixes only (the old behaviour; run_hunt then reads any other emitted selection as UNDEFINED). The items are shuffled together
(seeded) and split into N contiguous chunks, so a judge cannot tell a selection from a case. `sentence_exact_escaped` writes every
character outside printable ASCII as \\uXXXX (\\UXXXXXXXX above U+FFFF), LF as \\n, CR as \\r and a backslash as \\\\, so a judge
can see a U+00A0 or a stray control character.
jid = "j" + the first 10 hex digits of sha256(JSON [seed, case id, part]): random-looking, deterministic for a seed, unique (checked).
A malformed reply case (run_hunt.validate_reply_case) or a repeated case id stops the script (exit 2) before anything is written.
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)
import argparse
import hashlib
import json
import os
import random
import sys
from itertools import combinations

from research.findings.raw._moat_allowlist_hunt4 import run_hunt as H

ITEM_KEYS = ("jid", "gathered", "sentence_as_written", "sentence_exact_escaped")


def escape_exact(s):
    """Every character outside printable ASCII as \\uXXXX (\\UXXXXXXXX above U+FFFF), LF \\n, CR \\r, backslash \\\\."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append("\\u%04x" % o)
        else:
            out.append("\\U%08x" % o)
    return "".join(out)


def make_jid(seed, case_id, part):
    """The hash input is the JSON of [seed, case id, part]."""
    return "j" + hashlib.sha256(json.dumps([seed, case_id, part]).encode("utf-8")).hexdigest()[:10]


def proper_selections(k, prefixes_only=False):
    """The parts of a k-sentence reply case besides "full", as subset_labels keys: the proper prefixes "1", "1+2", ..., then (unless
    `prefixes_only`) every other in-order non-empty proper selection ("2", "1+3", ...). 2**k - 2 parts in all."""
    parts = [H.subset_key(range(1, j + 1)) for j in range(1, k)]
    if not prefixes_only:
        for n in range(1, k):
            for c in combinations(range(1, k + 1), n):
                if list(c) != list(range(1, n + 1)):
                    parts.append(H.subset_key(c))
    return parts


def part_text(case, part, terminate):
    """The text of one item of `case`: its `sentence` as written for "full", else the selection joined as the composer joins it."""
    if part == "full":
        return case["sentence"]
    return H.reply_join(case["sentences"], [int(x) for x in part.split("+")], terminate)


def build_items(cases, seed, prefixes_only=False, terminate=None):
    """(items, key): the anonymised items in seeded-shuffled order and the private key {jid: [case id, part]}. Raises ValueError on a
    repeated case id, a malformed reply case or a jid collision."""
    terminate = terminate or H.composer_terminate()
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("repeated case id(s): %s" % sorted({i for i in ids if ids.count(i) > 1}))
    items, key = [], {}
    for c in cases:
        parts = ["full"]
        if H.is_reply_case(c):
            why = H.validate_reply_case(c, terminate)
            if why:
                raise ValueError("%s: %s" % (c["id"], why))
            parts += proper_selections(len(c["sentences"]), prefixes_only)
        for part in parts:
            jid = make_jid(seed, c["id"], part)
            if jid in key:
                raise ValueError("jid collision %s (%s, %s)" % (jid, key[jid], [c["id"], part]))
            text = part_text(c, part, terminate)
            key[jid] = [c["id"], part]
            items.append({"jid": jid, "gathered": [list(f) for f in c["gathered"]], "sentence_as_written": text,
                          "sentence_exact_escaped": escape_exact(text)})
    random.Random(seed).shuffle(items)
    return items, key


def split_chunks(items, n):
    """n contiguous chunks whose sizes differ by at most one (every chunk non-empty when n <= len(items))."""
    if n < 1:
        raise ValueError("--chunks must be >= 1")
    q, r = divmod(len(items), n)
    out, i = [], 0
    for c in range(n):
        size = q + (1 if c < r else 0)
        out.append(items[i:i + size])
        i += size
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", help="the round's cases.json")
    ap.add_argument("out_dir")
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--prefixes-only", action="store_true",
                    help="write a reply case's proper prefixes only, not every in-order selection (the old behaviour)")
    a = ap.parse_args(argv)
    cases = json.load(open(a.cases))
    try:
        items, key = build_items(cases, a.seed, a.prefixes_only)
        chunks = split_chunks(items, a.chunks)
    except ValueError as e:
        print("judge_prep: %s" % e, file=sys.stderr)
        return 2
    os.makedirs(a.out_dir, exist_ok=True)
    for i, ch in enumerate(chunks, 1):
        p = os.path.join(a.out_dir, "chunk_%02d.json" % i)
        json.dump(ch, open(p, "w"), indent=1, ensure_ascii=False)
        research.runners.declare_output(p)
    sha = hashlib.sha256(open(a.cases, "rb").read()).hexdigest()
    n_reply = sum(1 for c in cases if H.is_reply_case(c))
    json.dump({"_doc": "PRIVATE judge key (judge_prep.py): jid -> [case id, part]; part 'full' = the case's sentence, else the key of "
                       "a proper in-order selection of a reply case's sentences ('1', '1+2', '2', '1+3', ...). Never shown to a judge.",
               "cases": a.cases, "cases_sha256": sha, "seed": a.seed, "chunks": a.chunks,
               "selections": "prefixes" if a.prefixes_only else "all",
               "n_cases": len(cases), "n_reply_cases": n_reply, "n_items": len(items), "key": key},
              open(os.path.join(a.out_dir, "key.json"), "w"), indent=1)
    research.runners.declare_output(os.path.join(a.out_dir, "key.json"))
    print("judge_prep: %d cases (%d reply) -> %d items in %d chunks, key %s" % (len(cases), n_reply, len(items), a.chunks,
                                                                                  os.path.join(a.out_dir, "key.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
