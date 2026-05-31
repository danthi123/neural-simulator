"""THROWAWAY probe: does DIRECTIONAL tag filtering rescue 2-hop transitive
reasoning under hub crowding? DO NOT COMMIT (raw/ throwaway).

The hub-reuse decisive test (2026-05-31-P4-multihop-hub-reuse-DECISIVE-
DEGRADES-WITH-FANIN) located the bottleneck precisely: hop-1 (find the hub)
is fine at all fan-in; the entire loss is at hop-2 -- querying a crowded hub
returns its many INCOMING nouns and buries the one OUTGOING edge, because
multitag retrieval is UNDIRECTED (matches ANY tag where the word is a token).

But the tags are NAME-ORDERED: "remember a is b" -> tag "a_b" (cue first,
associate second). So at hop-2, when we want the hub's OUTGOING edge, we can
FILTER to tags where the hub is the FIRST token (hub_*) and ignore the
incoming X_hub tags. This is a principled use of stored direction; it does
NOT create any A->C edge (anti-cheat preserved -- no a_c tag exists).

This probe reuses the EXACT shipped load/encode/recall (imported from
g20_multibridge) + the SAME hub graph as the hub-reuse test, and runs BOTH:
  - direction='any'  (UNDIRECTED, reproduces the DEGRADES baseline)
  - direction='out'  (DIRECTIONAL fix: word must be the FIRST tag token)
at hop-1 AND hop-2, so the comparison is one controlled run.

PRE-REGISTERED: the directional fix RESCUES multi-hop if direction='out'
full-2hop at fan-in 8 is >= 0.50 multi-seed (vs undirected 0.000). If it does
NOT reach 0.50, the encode-order is not reliable / the substrate aggregates
regardless -> multi-hop is honestly bounded to low-fan-in chains.

SMOKE mode (--smoke): seed 42 only, to verify the directional filter surfaces
C at fan-in 8 BEFORE the full multi-seed run (grounding-probe-before-decisive).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

from research.runners.g20_multibridge import (
    SharedPoolMember,
    read_vocab_file,
    find_member_for_pair,
    find_member_for_word,
)

N_LANG_INPUT = 8192
N_SHARED_POOL = 2000
SPARSITY = 0.02
PATTERN_SIZE = 100
DRIVE_PA = 1500.0
DRIVE_STEPS = 100
TOP_K = 3

VD = "research/findings/raw/g11_bg"
SEED_BRIDGE_DIRS = {
    42: f"{VD}/g20_sparse_bridges",
    43: f"{VD}/g20_sparse_bridges_s43",
    44: f"{VD}/g20_sparse_bridges_s44",
}
BRIDGE_BASENAMES = [
    "bridgeA_nouns_sparse.simstate.h5",
    "bridgeB_verbs_sparse.simstate.h5",
    "bridgeC_adj_sparse.simstate.h5",
    "bridgeD_spatial_sparse.simstate.h5",
    "bridgeE_functional_sparse.simstate.h5",
]
VOCABS = [
    f"{VD}/g20_bridgeA_nouns_vocab.txt",
    f"{VD}/g20_bridgeB_verbs_vocab.txt",
    f"{VD}/g20_bridgeC_adj_vocab.txt",
    f"{VD}/g20_bridgeD_spatial_vocab.txt",
    f"{VD}/g20_bridgeE_functional_vocab.txt",
]
NAMES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
         "bridgeD_spatial", "bridgeE_functional"]

HUB_GRAPH = {
    "fast": {"C": "small", "nouns": ["ball", "key"]},                       # fan-in 2
    "hot":  {"C": "dry",   "nouns": ["bird", "flower", "leaf", "fruit"]},   # fan-in 4
    "big":  {"C": "red",   "nouns": ["apple", "river", "dog", "cat",
                                     "tree", "fish", "mouse", "frog"]},     # fan-in 8
}


def build_members(seed):
    bdir = SEED_BRIDGE_DIRS[seed]
    members = []
    for base, vp, nm in zip(BRIDGE_BASENAMES, VOCABS, NAMES):
        vocab = read_vocab_file(vp)
        members.append(SharedPoolMember(
            bridge_path=f"{bdir}/{base}", vocab=vocab, name=nm,
            n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
            sparsity=SPARSITY, drive_pA=DRIVE_PA, drive_steps=DRIVE_STEPS,
            sparse=True, pattern_size=PATTERN_SIZE,
        ))
    return members


def query_concept_ranked(members, word, top_n=10, direction="any"):
    """Clone of g20_multibridge.query_concept + optional DIRECTION filter.
    direction='any' = shipped undirected behaviour (word anywhere in tag).
    direction='out' = word must be the FIRST tag token (its OUTGOING edges).
    direction='in'  = word must be the LAST tag token (its INCOMING edges)."""
    all_results = []
    for m in members:
        for tag in m.encoded_tags:
            toks = tag.split("_")
            if word not in toks:
                continue
            if direction == "out" and toks[0] != word:
                continue
            if direction == "in" and toks[-1] != word:
                continue
            rates = m.recall_rates(tag)
            for j in np.argsort(-rates)[:5]:
                cand = m.vocab[j]
                if cand == word:
                    continue
                all_results.append({"word": cand, "rate": float(rates[j]),
                                    "tag": tag, "bridge": m.name})
    if not all_results:
        return []
    by_word = {}
    for r in all_results:
        if r["word"] not in by_word or r["rate"] > by_word[r["word"]]["rate"]:
            by_word[r["word"]] = r
    return sorted(by_word.values(), key=lambda r: -r["rate"])[:top_n]


def encode_pair(members, a, b):
    m_both = find_member_for_pair(members, a, b)
    if m_both is not None:
        tag = m_both.encode_pair(a, b)
        m_both.encoded_tags.append(tag)
        return ("intra", m_both.name, tag)
    tag_name = f"{a}_{b}"
    for m in members:
        if a in m.vocab_set:
            m.encode_partial(a, tag_name); m.encoded_tags.append(tag_name)
        elif b in m.vocab_set:
            m.encode_partial(b, tag_name); m.encoded_tags.append(tag_name)
    return ("cross", tag_name)


def run_seed(seed):
    t0 = time.time()
    print(f"\n############ SEED {seed} ############", flush=True)
    members = build_members(seed)
    for m in members:
        m.load(seed)
    print(f"  [bridges loaded @ {time.time()-t0:.0f}s]", flush=True)

    # encode A->HUB then HUB->C (same as hub-reuse test)
    for hub, spec in HUB_GRAPH.items():
        for a in spec["nouns"]:
            encode_pair(members, a, hub)
    for hub, spec in HUB_GRAPH.items():
        info = encode_pair(members, hub, spec["C"])
        print(f"    HUB->C tag: {hub}->{spec['C']} = {info[-1]}", flush=True)

    by_fanin = {}
    for hub, spec in HUB_GRAPH.items():
        fi = len(spec["nouns"])
        for a in spec["nouns"]:
            by_fanin.setdefault(fi, []).append((a, hub, spec["C"]))

    rows = {}  # fanin -> {mode -> counts}
    for fi in sorted(by_fanin):
        chains = by_fanin[fi]
        n = len(chains)
        rows[fi] = {"n": n}
        for mode in ("any", "out"):
            two_hop = 0
            hop2_marg = 0
            for (a, hub, c) in chains:
                r1 = query_concept_ranked(members, a, 10, direction=mode)
                src = r1[0]["word"] if r1 else None
                r2 = query_concept_ranked(members, src, 10, direction=mode) if src else []
                if c in [x["word"] for x in r2][:TOP_K]:
                    two_hop += 1
                r2t = query_concept_ranked(members, hub, 10, direction=mode)
                if c in [x["word"] for x in r2t][:TOP_K]:
                    hop2_marg += 1
                if mode == "out":  # show the directional hop-2 result
                    print(f"    [out] fan{fi} {a}->{hub}->{c}: "
                          f"hub top3={[x['word'] for x in r2t][:TOP_K]} "
                          f"(C={c in [x['word'] for x in r2t][:TOP_K]})", flush=True)
            rows[fi][mode] = {"two_hop": two_hop, "hop2_marg": hop2_marg}
        print(f"  >> fan-in {fi} (n={n}): "
              f"ANY full_2hop={rows[fi]['any']['two_hop']}/{n} | "
              f"OUT full_2hop={rows[fi]['out']['two_hop']}/{n}", flush=True)
    print(f"  [seed {seed} wall {time.time()-t0:.0f}s]", flush=True)
    return rows


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    print("=== DIRECTIONAL MULTI-HOP RESCUE PROBE (any vs out) ===", flush=True)
    avail = [s for s in (42, 43, 44)
             if all(os.path.exists(f"{SEED_BRIDGE_DIRS[s]}/{b}")
                    for b in BRIDGE_BASENAMES)]
    seeds = avail[:1] if smoke else avail
    print(f"seeds={seeds} (smoke={smoke})  chance(top3/32)={TOP_K/32.0:.3f}", flush=True)
    if not seeds:
        print("NO bridges -- abort.", flush=True); sys.exit(1)

    res = {s: run_seed(s) for s in seeds}

    print("\n================ AGGREGATE (full_2hop multi-seed mean) ================",
          flush=True)
    fanins = sorted({fi for s in seeds for fi in res[s]})
    means = {}
    print(f"{'fan-in':>7} | {'ANY 2hop':>9} | {'OUT 2hop':>9}", flush=True)
    for fi in fanins:
        anys, outs = [], []
        for s in seeds:
            r = res[s].get(fi)
            if not r:
                continue
            anys.append(r["any"]["two_hop"] / r["n"])
            outs.append(r["out"]["two_hop"] / r["n"])
        ma = float(np.mean(anys)); mo = float(np.mean(outs))
        means[fi] = (ma, mo)
        print(f"{fi:>7} | {ma:>9.3f} | {mo:>9.3f}", flush=True)

    max_fi = max(fanins)
    out_at_max = means[max_fi][1]
    any_at_max = means[max_fi][0]
    print(f"\n  PRE-REGISTERED: directional RESCUES if OUT full_2hop at fan-in "
          f"{max_fi} >= 0.50 (undirected ANY = {any_at_max:.3f})", flush=True)
    if smoke:
        print(f"  SMOKE (seed 42): OUT@fan-in{max_fi}={out_at_max:.3f} -- "
              f"{'PROMISING -> run multi-seed' if out_at_max >= 0.50 else 'NOT promising'}",
              flush=True)
    else:
        verdict = ("RESCUED" if out_at_max >= 0.50 else "NOT-RESCUED")
        print(f"  MULTI-SEED VERDICT: {verdict} "
              f"(OUT@fan-in{max_fi}={out_at_max:.3f} vs ANY={any_at_max:.3f})", flush=True)
    print(f"\n[TOTAL wall {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
