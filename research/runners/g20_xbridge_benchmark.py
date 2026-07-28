"""Quantitative cross-bridge associative-retrieval benchmark for a
sparse-distributed G.20 ensemble.

The end-to-end demos prove cross-bridge memory works *anecdotally*
(apple->big #1). This turns that into a METRIC over N controlled
pairs, with an anti-cheat pre/post delta: a pair only counts as a
genuine learned association if B was NOT the top-1 associate of A
*before* `remember A is B` and IS top-1 *after*. That rules out
coincidental noise alignment (the project's permuted-label-control
discipline applied to retrieval).

Reuses the validated SharedPoolMember (load + recall_rates +
encode_partial) from g20_multibridge -- no new sim code.

Usage:
  python -m research.runners.g20_xbridge_benchmark \\
      --sparse --pattern-size 100 --n-shared-pool 2000 \\
      --n-lang-input 8192 --sparsity 0.007 --seed 42 \\
      --bridges <5 *.simstate.h5> --vocab-files <5 *.txt> \\
      --names bridgeA_nouns ... --n-pairs 30 \\
      --out research/findings/raw/g11_bg/g20_xbridge_bench_320.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from research.runners.g20_multibridge import (
    SharedPoolMember, read_vocab_file,
)


def sample_xbridge_pairs(member_vocabs: List[List[str]],
                          n_pairs: int, seed: int,
                          exclude_idx: int | None = 12,
                          ) -> List[Tuple[int, str, int, str]]:
    """Deterministically sample n_pairs cross-bridge (A, B) pairs.

    Returns list of (bridge_a_idx, word_a, bridge_b_idx, word_b) with
    bridge_a_idx != bridge_b_idx. `exclude_idx` drops that concept
    position from every bridge (the characterized idx-12 sparse-pattern
    failure) so the benchmark measures the healthy substrate; pairs
    touching it are reported separately by the caller if desired.

    Pure / no bridge -> unit-testable.
    """
    rng = np.random.RandomState(seed * 31 + 7)
    nb = len(member_vocabs)
    assert nb >= 2, "need >= 2 bridges for cross-bridge pairs"
    # eligible (bridge, word_idx) excluding the bad position
    elig = []
    for bi, vocab in enumerate(member_vocabs):
        for wi in range(len(vocab)):
            if exclude_idx is not None and wi == exclude_idx:
                continue
            elig.append((bi, wi))
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        a = elig[rng.randint(len(elig))]
        b = elig[rng.randint(len(elig))]
        if a[0] == b[0]:
            continue  # must be cross-bridge
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        wa = member_vocabs[a[0]][a[1]]
        wb = member_vocabs[b[0]][b[1]]
        if wa == wb:
            continue
        pairs.append((a[0], wa, b[0], wb))
    return pairs


def _query_top(members: List[SharedPoolMember], word: str,
                aggregation: str = "max",
                samebridge_factor: float = 0.4
                ) -> List[Tuple[str, float, str]]:
    """Replicate g20_multibridge.query_concept aggregation: search all
    bridges' tags containing `word`, recall, return ranked (assoc,
    rate, tag) excluding `word` itself.

    `aggregation` (artifact-safe, query-time only — characterized in
    2026-05-16-G20-distinct-submechanism-same-bridge-crosstalk):
      - "max": raw max rate across bridges (the current/baseline).
      - "perbridge_norm": scale each bridge's candidate rates by that
        bridge's own max in this query (0-1) before aggregating, so a
        high-baseline home bridge cannot win purely by magnitude.
      - "samebridge_downweight": multiply rate by `samebridge_factor`
        for candidates from the query word's OWN home bridge.
    """
    home = next((m.name for m in members
                 if word in m.vocab_set), None)
    results = []  # (cand, rate, tag, src_bridge)
    for m in members:
        matches = [t for t in m.encoded_tags if word in t.split("_")]
        for tag in matches:
            rates = m.recall_rates(tag)
            mmax = float(np.max(rates)) if len(rates) else 0.0
            order = np.argsort(-rates)
            for j in order[:5]:
                cand = m.vocab[j]
                if cand == word:
                    continue
                r = float(rates[j])
                if aggregation == "perbridge_norm":
                    r = r / mmax if mmax > 0 else 0.0
                elif (aggregation == "samebridge_downweight"
                      and m.name == home):
                    r = r * samebridge_factor
                results.append((cand, r, tag, m.name))
    by_word: Dict[str, Tuple[str, float, str]] = {}
    for w, r, tag, _src in results:
        if w not in by_word or r > by_word[w][1]:
            by_word[w] = (w, r, tag)
    return sorted(by_word.values(), key=lambda x: -x[1])


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
    p.add_argument("--n-pairs", type=int, default=30)
    p.add_argument("--encode-repeats", type=int, default=1,
                    help="repeat the cross-bridge encode N times before "
                         "POST (controlled one-shot-vs-reinforced test; "
                         "same --seed -> same pairs for A/B comparison)")
    p.add_argument("--exclude-idx", type=int, default=12,
                    help="drop this concept position (known bad); -1 = keep all")
    p.add_argument("--aggregation", default="max",
                    choices=["max", "perbridge_norm",
                             "samebridge_downweight"],
                    help="cross-bridge aggregation (artifact-safe, "
                         "query-time only)")
    p.add_argument("--samebridge-factor", type=float, default=0.4)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    members = []
    for bp, vp, nm in zip(args.bridges, args.vocab_files, args.names):
        m = SharedPoolMember(
            bridge_path=bp, vocab=read_vocab_file(vp), name=nm,
            n_lang_input=args.n_lang_input,
            n_shared_pool=args.n_shared_pool,
            sparsity=args.sparsity, sparse=args.sparse,
            pattern_size=args.pattern_size)
        members.append(m)
    print(f"=== xbridge benchmark: {len(members)} bridges, "
          f"{args.n_pairs} pairs, seed {args.seed} ===", flush=True)
    for m in members:
        m.load(args.seed)
        print(f"  loaded {m.name} ({m.n_concepts()} concepts, "
              f"{len(m.encoded_tags)} tags)", flush=True)

    exclude = None if args.exclude_idx < 0 else args.exclude_idx
    pairs = sample_xbridge_pairs(
        [m.vocab for m in members], args.n_pairs, args.seed, exclude)
    print(f"\nSampled {len(pairs)} cross-bridge pairs", flush=True)

    rows = []
    t0 = time.time()
    for i, (ba, wa, bb, wb) in enumerate(pairs):
        ma, mb = members[ba], members[bb]
        # PRE: B should not already be top-1 associate of A
        pre = _query_top(members, wa, args.aggregation,
                          args.samebridge_factor)
        pre_top = pre[0][0] if pre else None
        pre_b_rank = next((k for k, x in enumerate(pre)
                            if x[0] == wb), -1)
        # ENCODE cross-bridge tag "wa_wb" (optionally reinforced N times)
        tag = f"{wa}_{wb}"
        for _ in range(max(1, args.encode_repeats)):
            ma.encode_partial(wa, tag)
            mb.encode_partial(wb, tag)
        if tag not in ma.encoded_tags:
            ma.encoded_tags.append(tag)
        if tag not in mb.encoded_tags:
            mb.encoded_tags.append(tag)
        # POST
        post = _query_top(members, wa, args.aggregation,
                           args.samebridge_factor)
        post_top = post[0][0] if post else None
        post_b_rank = next((k for k, x in enumerate(post)
                             if x[0] == wb), -1)
        b_top1_post = (post_top == wb)
        genuine = b_top1_post and (pre_top != wb)
        rows.append({
            "a": wa, "a_bridge": ma.name, "b": wb, "b_bridge": mb.name,
            "pre_top": pre_top, "pre_b_rank": pre_b_rank,
            "post_top": post_top, "post_b_rank": post_b_rank,
            "b_top1_post": b_top1_post, "genuine": genuine,
            "post_b_rate": (post[post_b_rank][1]
                             if post_b_rank >= 0 else 0.0),
        })
        if (i + 1) % 5 == 0:
            print(f"  pair {i+1}/{len(pairs)} "
                  f"({int(time.time()-t0)}s)", flush=True)

    n = len(rows)
    n_top1 = sum(r["b_top1_post"] for r in rows)
    n_genuine = sum(r["genuine"] for r in rows)
    summary = {
        "n_pairs": n,
        "b_top1_post_rate": n_top1 / max(n, 1),
        "genuine_assoc_rate": n_genuine / max(n, 1),
        "n_top1": n_top1, "n_genuine": n_genuine,
        "seed": args.seed, "exclude_idx": exclude,
        "mean_post_b_rate_when_top1": (
            float(np.mean([r["post_b_rate"] for r in rows
                            if r["b_top1_post"]]))
            if n_top1 else 0.0),
    }
    print(f"\n=== RESULTS ===", flush=True)
    print(f"  B top-1 after encode:   {n_top1}/{n} = "
          f"{100*summary['b_top1_post_rate']:.1f}%", flush=True)
    print(f"  GENUINE (not-top->top1): {n_genuine}/{n} = "
          f"{100*summary['genuine_assoc_rate']:.1f}%  "
          f"(anti-cheat: rules out coincidence)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary, "rows": rows},
                   open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
