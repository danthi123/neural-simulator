"""Conversational interference / retention benchmark for a sparse
G.20 ensemble.

Every prior benchmark encodes a fact then immediately queries it.
Real conversation is LONG: you state fact 1 ... fact N, then refer
back to fact 1. The project's foundational thesis is continuous
learning WITHOUT catastrophic forgetting. This tests it at the
conversational-ensemble level: encode N sequential cross-bridge
facts, THEN re-query every one — does the first survive to the end?

Measures retention as a function of recency (early vs late facts)
and reports whether accumulating conversational load degrades earlier
bindings. Anti-cheat: a fact only counts retained if its associate
is top-1 at final query AND was not top-1 before any encoding.

Reuses validated SharedPoolMember + _query_top (DRY).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from research.runners.g20_multibridge import (
    SharedPoolMember, read_vocab_file,
)
from research.runners.g20_xbridge_benchmark import (
    _query_top, sample_xbridge_pairs,
)


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
    p.add_argument("--n-facts", type=int, default=30,
                    help="sequential cross-bridge facts to accumulate")
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
    print(f"=== interference benchmark: {args.n_facts} sequential "
          f"facts, seed {args.seed} ===", flush=True)
    for m in members:
        m.load(args.seed)

    exclude = None if args.exclude_idx < 0 else args.exclude_idx
    # distinct subjects -> dedupe by (a_bridge, a_word)
    raw = sample_xbridge_pairs(
        [m.vocab for m in members], args.n_facts * 3, args.seed, exclude)
    facts, seen_a = [], set()
    for ba, wa, bb, wb in raw:
        if wa in seen_a or wb == wa:
            continue
        seen_a.add(wa)
        facts.append((ba, wa, bb, wb))
        if len(facts) >= args.n_facts:
            break
    print(f"  {len(facts)} distinct-subject facts", flush=True)

    # PRE: none of the B's should already be top-1 of its A
    pre_top1 = []
    for ba, wa, bb, wb in facts:
        t = _query_top(members, wa)
        pre_top1.append(bool(t) and t[0][0] == wb)

    # Encode ALL facts sequentially (accumulating load)
    t0 = time.time()
    for i, (ba, wa, bb, wb) in enumerate(facts):
        ma, mb = members[ba], members[bb]
        tag = f"{wa}_{wb}"
        ma.encode_partial(wa, tag)
        mb.encode_partial(wb, tag)
        for m in (ma, mb):
            if tag not in m.encoded_tags:
                m.encoded_tags.append(tag)
        if (i + 1) % 10 == 0:
            print(f"  encoded {i+1}/{len(facts)} "
                  f"({int(time.time()-t0)}s)", flush=True)

    # POST (after ALL encodes): re-query every fact's subject
    rows = []
    for idx, (ba, wa, bb, wb) in enumerate(facts):
        t = _query_top(members, wa)
        top = t[0][0] if t else None
        retained = (top == wb)
        genuine = retained and not pre_top1[idx]
        rows.append({"pos": idx, "a": wa, "b": wb,
                      "final_top": top, "retained": retained,
                      "genuine": genuine})

    n = len(rows)
    nret = sum(r["genuine"] for r in rows)
    # retention by recency third
    th = max(1, n // 3)
    early = rows[:th]
    late = rows[-th:]
    er = sum(r["genuine"] for r in early) / max(len(early), 1)
    lr = sum(r["genuine"] for r in late) / max(len(late), 1)
    summary = {
        "n_facts": n,
        "overall_retention": nret / max(n, 1),
        "n_retained": nret,
        "early_third_retention": er,
        "late_third_retention": lr,
        "recency_gap": lr - er,
        "seed": args.seed,
    }
    print(f"\n=== RESULTS (conversational retention under load) ===",
          flush=True)
    print(f"  overall retention (genuine): {nret}/{n} = "
          f"{100*summary['overall_retention']:.1f}%", flush=True)
    print(f"  early-third: {100*er:.1f}%  late-third: {100*lr:.1f}%  "
          f"recency gap: {100*(lr-er):+.1f}pp", flush=True)
    print(f"  -> {'NO catastrophic forgetting' if er >= 0.5 else 'EARLY FACTS DEGRADED'} "
          f"(early-third retention {'>=' if er>=0.5 else '<'} 50%)",
          flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary, "rows": rows},
                   open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
