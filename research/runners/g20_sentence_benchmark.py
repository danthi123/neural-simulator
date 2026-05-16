"""Multi-bridge 3-way SENTENCE retrieval benchmark.

Pair benchmark (g20_xbridge_benchmark) measured `remember A is B`.
Real conversation needs N-word binding: `remember <subj> <verb> <obj>`
encodes a shared tag spanning 3 bridges; querying the subject should
surface BOTH the verb and the object. The 320 demo showed this for
ONE sentence (horse -> run 882 + fast 508). This quantifies it over
N random subject/verb/object triples with the same anti-cheat
pre/post discipline.

Substrate-dependent (sparse cross-bridge recall of a 3-way-shared
engram tag) -- NOT the pure tag-name role-query string match (that's
separately unit-tested in test_g20_sentence_roles.py).

Reuses the validated SharedPoolMember + _query_top (DRY).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from research.runners.g20_multibridge import (
    SharedPoolMember, read_vocab_file,
)
from research.runners.g20_xbridge_benchmark import _query_top


def sample_sentences(member_vocabs, names, n_sents: int, seed: int,
                      subj_bridge: str, verb_bridge: str,
                      obj_bridge: str, exclude_idx: int | None = 12,
                      ) -> List[Tuple[str, str, str]]:
    """Deterministically sample n (subj, verb, obj) word triples, one
    word from each of the named bridges. Pure / unit-testable."""
    rng = np.random.RandomState(seed * 53 + 11)
    bi = {n: i for i, n in enumerate(names)}
    sv, vv, ov = (member_vocabs[bi[subj_bridge]],
                  member_vocabs[bi[verb_bridge]],
                  member_vocabs[bi[obj_bridge]])

    def pick(vocab):
        while True:
            j = rng.randint(len(vocab))
            if exclude_idx is not None and j == exclude_idx:
                continue
            return vocab[j]

    out, seen = [], set()
    attempts = 0
    while len(out) < n_sents and attempts < n_sents * 50:
        attempts += 1
        s, v, o = pick(sv), pick(vv), pick(ov)
        if len({s, v, o}) != 3:
            continue
        key = (s, v, o)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridges", nargs="+", required=True)
    p.add_argument("--vocab-files", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=2000)
    p.add_argument("--sparsity", type=float, default=0.007)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--n-sents", type=int, default=20)
    p.add_argument("--subj-bridge", default="bridgeA_nouns")
    p.add_argument("--verb-bridge", default="bridgeB_verbs")
    p.add_argument("--obj-bridge", default="bridgeC_adj")
    p.add_argument("--top-k", type=int, default=5,
                    help="verb+obj must both be within top-K of subj query")
    p.add_argument("--exclude-idx", type=int, default=12)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    members = []
    for bp, vp, nm in zip(args.bridges, args.vocab_files, args.names):
        members.append(SharedPoolMember(
            bridge_path=bp, vocab=read_vocab_file(vp), name=nm,
            n_lang_input=args.n_lang_input,
            n_shared_pool=args.n_shared_pool, sparsity=args.sparsity,
            sparse=args.sparse, pattern_size=args.pattern_size))
    print(f"=== sentence benchmark: {args.n_sents} triples, "
          f"seed {args.seed}, top-{args.top_k} ===", flush=True)
    for m in members:
        m.load(args.seed)
    by_name = {m.name: m for m in members}

    exclude = None if args.exclude_idx < 0 else args.exclude_idx
    sents = sample_sentences(
        [m.vocab for m in members], args.names, args.n_sents,
        args.seed, args.subj_bridge, args.verb_bridge,
        args.obj_bridge, exclude)
    print(f"Sampled {len(sents)} sentences", flush=True)

    ms, mv, mo = (by_name[args.subj_bridge],
                  by_name[args.verb_bridge], by_name[args.obj_bridge])
    rows = []
    t0 = time.time()
    for i, (s, v, o) in enumerate(sents):
        pre = [w for w, _, _ in _query_top(members, s)[:args.top_k]]
        pre_has = (v in pre) or (o in pre)
        tag = f"{s}_{v}_{o}"
        ms.encode_partial(s, tag)
        mv.encode_partial(v, tag)
        mo.encode_partial(o, tag)
        for m in (ms, mv, mo):
            if tag not in m.encoded_tags:
                m.encoded_tags.append(tag)
        post = [w for w, _, _ in _query_top(members, s)[:args.top_k]]
        v_in, o_in = (v in post), (o in post)
        both = v_in and o_in
        genuine = both and not pre_has
        rows.append({"subj": s, "verb": v, "obj": o,
                      "v_in_topk": v_in, "o_in_topk": o_in,
                      "both": both, "genuine": genuine,
                      "pre_had_either": pre_has})
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(sents)} ({int(time.time()-t0)}s)",
                  flush=True)

    n = len(rows)
    nb = sum(r["both"] for r in rows)
    ng = sum(r["genuine"] for r in rows)
    nv = sum(r["v_in_topk"] for r in rows)
    no = sum(r["o_in_topk"] for r in rows)
    summary = {"n_sents": n, "top_k": args.top_k, "seed": args.seed,
               "both_rate": nb / max(n, 1),
               "genuine_rate": ng / max(n, 1),
               "verb_recall": nv / max(n, 1),
               "obj_recall": no / max(n, 1),
               "n_both": nb, "n_genuine": ng}
    print(f"\n=== RESULTS (3-way sentence retrieval) ===", flush=True)
    print(f"  verb in top-{args.top_k}:  {nv}/{n} = "
          f"{100*summary['verb_recall']:.1f}%", flush=True)
    print(f"  obj  in top-{args.top_k}:  {no}/{n} = "
          f"{100*summary['obj_recall']:.1f}%", flush=True)
    print(f"  BOTH:                {nb}/{n} = "
          f"{100*summary['both_rate']:.1f}%", flush=True)
    print(f"  GENUINE (anti-cheat): {ng}/{n} = "
          f"{100*summary['genuine_rate']:.1f}%", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary, "rows": rows},
                   open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
