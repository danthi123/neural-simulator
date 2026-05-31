"""THROWAWAY (raw/): validate-FIRST for the cross-bridge teacher lever. The
intra-bridge encode_pair teacher fix (100->500) is shipped + confirmed; the
residual multi-hop hop-1 variance is CROSS-bridge (noun->hub), which uses
encode_partial -> encode_partial_pair_engram_sparse (still teacher=100, its
docstring-'validated' recipe). Does teacher=500 IMPROVE (or at least not regress)
cross-bridge multitag retrieval? If yes -> ship the encode_partial fix; if no ->
keep cross-bridge at 100.

Encode K cross-bridge noun->adj pairs at teacher=100 vs 500 (fresh 2-bridge member
set per teacher so no accumulation), query the noun via 2-member multitag, check
if the adj target is in top-3. Seed 42 smoke first.

Reuses SharedPoolMember + encode_partial_pair_engram_sparse (byte-unchanged imports).
"""
from __future__ import annotations
import os
import time
import numpy as np

from research.runners.g20_multibridge import SharedPoolMember, read_vocab_file
from research.runners.shared_pool_chat import encode_partial_pair_engram_sparse

VD = "research/findings/raw/g11_bg"
BD = f"{VD}/g20_sparse_bridges"  # seed 42
NOUN_BR = f"{BD}/bridgeA_nouns_sparse.simstate.h5"
ADJ_BR = f"{BD}/bridgeC_adj_sparse.simstate.h5"
NOUN_VOCAB = f"{VD}/g20_bridgeA_nouns_vocab.txt"
ADJ_VOCAB = f"{VD}/g20_bridgeC_adj_vocab.txt"

TEACHERS = [100.0, 500.0]
PAIRS = [("apple", "big"), ("dog", "small"), ("river", "cold"),
         ("cat", "hot"), ("tree", "tall"), ("bird", "fast")]
N_LANG_INPUT = 8192
N_SHARED_POOL = 2000
SPARSITY = 0.02
PATTERN_SIZE = 100
SEED = 42
TOP_K = 3


def make_members():
    members = []
    for path, vp, nm in [(NOUN_BR, NOUN_VOCAB, "bridgeA_nouns"),
                         (ADJ_BR, ADJ_VOCAB, "bridgeC_adj")]:
        m = SharedPoolMember(bridge_path=path, vocab=read_vocab_file(vp), name=nm,
                             n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
                             sparsity=SPARSITY, drive_pA=1500.0, drive_steps=100,
                             sparse=True, pattern_size=PATTERN_SIZE)
        m.load(SEED)
        members.append(m)
    return members


def query_top(members, word, top_n=TOP_K):
    """2-member multitag: search both bridges' tags for `word`, recall, aggregate
    max-rate-per-word, return top_n words (excluding the query word)."""
    by_word = {}
    for m in members:
        for tag in m.encoded_tags:
            if word not in tag.split("_"):
                continue
            rates = m.recall_rates(tag)
            for j in np.argsort(-rates)[:5]:
                cand = m.vocab[j]
                if cand == word:
                    continue
                if cand not in by_word or float(rates[j]) > by_word[cand]:
                    by_word[cand] = float(rates[j])
    return [w for w, _ in sorted(by_word.items(), key=lambda kv: -kv[1])[:top_n]]


def encode_xbridge(members, a, b, teacher):
    """Cross-bridge partial encode under tag a_b at the given teacher strength."""
    tag = f"{a}_{b}"
    for m in members:
        if a in m.vocab_set:
            encode_partial_pair_engram_sparse(
                m.bridge, a, tag, vocab=m.vocab, sparse_patterns=m.sparse_patterns,
                n_lang_input=m.n_lang_input, sparsity=m.sparsity, teacher_pA=teacher)
            m.encoded_tags.append(tag)
        elif b in m.vocab_set:
            encode_partial_pair_engram_sparse(
                m.bridge, b, tag, vocab=m.vocab, sparse_patterns=m.sparse_patterns,
                n_lang_input=m.n_lang_input, sparsity=m.sparsity, teacher_pA=teacher)
            m.encoded_tags.append(tag)
    return tag


def main():
    print("=== CROSS-BRIDGE TEACHER PROBE (encode_partial; seed 42 smoke) ===", flush=True)
    print(f"teachers={TEACHERS}; pairs={PAIRS}; criterion: adj target in noun-query top-{TOP_K}", flush=True)
    results = {}
    for T in TEACHERS:
        t0 = time.time()
        members = make_members()  # fresh per teacher (no accumulation)
        hits = 0
        detail = []
        for (a, b) in PAIRS:
            if a not in members[0].vocab_set or b not in members[1].vocab_set:
                detail.append(f"{a}->{b}:SKIP"); continue
            encode_xbridge(members, a, b, T)
            top = query_top(members, a)
            ok = b in top
            hits += int(ok)
            detail.append(f"{a}->{b}:{'OK' if ok else 'miss'}({top})")
        results[T] = hits
        print(f"  teacher={T:6.0f}: {hits}/{len(PAIRS)} cross-bridge in top-{TOP_K} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for d in detail:
            print(f"      {d}", flush=True)
        del members

    print(f"\nSUMMARY (seed 42): teacher=100 -> {results.get(100.0)}/{len(PAIRS)} | "
          f"teacher=500 -> {results.get(500.0)}/{len(PAIRS)}", flush=True)
    a100, a500 = results.get(100.0, 0), results.get(500.0, 0)
    if a500 > a100:
        print("VERDICT: teacher=500 IMPROVES cross-bridge -> worth shipping the encode_partial fix "
              "(multi-seed confirm first).", flush=True)
    elif a500 == a100:
        print("VERDICT: teacher=500 EQUAL to 100 cross-bridge (seed 42) -> neutral; ship only if "
              "multi-seed shows improvement, else keep 100.", flush=True)
    else:
        print("VERDICT: teacher=500 WORSE cross-bridge -> do NOT ship the encode_partial fix; keep 100.",
              flush=True)


if __name__ == "__main__":
    main()
