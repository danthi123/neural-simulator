"""THROWAWAY probe: HUB-REUSE 2-hop transitive reasoning on the validated
G.20 sparse multitag conversational stack. DO NOT COMMIT.

This attacks the LOAD-BEARING harder condition that the prior all-distinct
probe (_multihop_reasoning_test.py) flagged: HUB REUSE. The prior probe got
8/8 2-hop ONLY because every chain's middle term B appeared in exactly one
chain (no competition at hop-2). Here the middle term is a HUB associated
with MANY nouns, so querying the hub returns many competing incoming-noun
associates and the 2-hop must still surface the correct C.

Reuses (by import) the validated building blocks from
research/runners/g20_multibridge.py so load/encode/recall are byte-identical
to the shipped stack. The ONLY additions are a RETURN-VALUE clone of
query_concept (the shipped one only prints) and the hub-graph scoring.

Design (clean degradation curve -- one hub per fan-in level, each with its
OWN distinct nouns and its OWN distinct C, so there is NO cross-hub
pollution; the only thing that varies between levels is how many nouns
point INTO the hub):

  fan-in 2  hub 'fast' : ball, key                          -> fast ; fast -> small
  fan-in 4  hub 'hot'  : bird, flower, leaf, fruit          -> hot  ; hot  -> dry
  fan-in 8  hub 'big'  : apple,river,dog,cat,tree,fish,mouse,frog -> big ; big -> red

A = noun (bridgeA_nouns), HUB + C = adjective (bridgeC_adj).
  A -> HUB  is cross-bridge (tag 'noun_hub' on bridgeA + bridgeC)
  HUB -> C  is intra-bridge (tag 'hub_C' on bridgeC, both adj)

2-hop chain for a noun A in a hub H with target C:
  hop1: query_concept(A) -> top-1 should be H
  hop2: query_concept(H) -> is C in top-3, DESPITE the many competing
        incoming-noun associates (and the other adj on bridge C)?
  full 2-hop transitive: chain hop1's ACTUAL top-1 into hop2; is C in top-3?

Anti-cheat: A->C is NEVER encoded. Confirmed empirically (before any
HUB->C edge is added, query A and check C not in top-3) AND by construction.

Runs seeds 42, 43, 44 (uses whichever of the per-seed bridge dirs exist).
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

# ---- canonical sparse invocation (from g20_sparse_ensemble_demo.ps1 / CLAUDE.md)
N_LANG_INPUT = 8192
N_SHARED_POOL = 2000
SPARSITY = 0.02
PATTERN_SIZE = 100
DRIVE_PA = 1500.0
DRIVE_STEPS = 100
TOP_K = 3  # "in top-3" criterion per the spec

VD = "research/findings/raw/g11_bg"
# per-seed bridge directories. seed 42 = the base dir; 43/44 = _sNN dirs.
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

# ---- HUB GRAPH: hub -> (target C, [incoming nouns ...])
# fan-in level keyed by len(nouns). All nouns are in bridgeA_nouns vocab,
# all hubs+Cs are in bridgeC_adj vocab. Sets are pairwise disjoint so a
# noun belongs to exactly one hub and a C comes from exactly one hub.
HUB_GRAPH = {
    "fast": {  # fan-in 2
        "C": "small",
        "nouns": ["ball", "key"],
    },
    "hot": {   # fan-in 4
        "C": "dry",
        "nouns": ["bird", "flower", "leaf", "fruit"],
    },
    "big": {   # fan-in 8
        "C": "red",
        "nouns": ["apple", "river", "dog", "cat",
                  "tree", "fish", "mouse", "frog"],
    },
}


def build_members(seed: int):
    bdir = SEED_BRIDGE_DIRS[seed]
    members = []
    for base, vp, nm in zip(BRIDGE_BASENAMES, VOCABS, NAMES):
        vocab = read_vocab_file(vp)
        m = SharedPoolMember(
            bridge_path=f"{bdir}/{base}", vocab=vocab, name=nm,
            n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
            sparsity=SPARSITY, drive_pA=DRIVE_PA, drive_steps=DRIVE_STEPS,
            sparse=True, pattern_size=PATTERN_SIZE,
        )
        members.append(m)
    return members


def query_concept_ranked(members, word, top_n=10):
    """RETURN-VALUE clone of g20_multibridge.query_concept. Identical
    max-rate-per-word aggregation; returns ranked [(word,rate,tag,bridge)]."""
    all_results = []
    for m in members:
        matches = [t for t in m.encoded_tags if word in t.split("_")]
        for tag in matches:
            rates = m.recall_rates(tag)
            sorted_idx = np.argsort(-rates)
            for j in sorted_idx[:5]:
                candidate = m.vocab[j]
                if candidate == word:
                    continue
                all_results.append({
                    "word": candidate,
                    "rate": float(rates[j]),
                    "tag": tag,
                    "bridge": m.name,
                })
    if not all_results:
        return []
    by_word = {}
    for r in all_results:
        if (r["word"] not in by_word
                or r["rate"] > by_word[r["word"]]["rate"]):
            by_word[r["word"]] = r
    ranked = sorted(by_word.values(), key=lambda r: -r["rate"])[:top_n]
    return ranked


def encode_pair(members, a, b):
    """Encode association (a, b) using the EXACT shipped routing (mirrors
    g20_multibridge 'remember a is b'): prefer a single bridge with both
    (intra-bridge encode_pair), else cross-bridge partial encode under tag
    a_b on each bridge that knows a or b."""
    m_both = find_member_for_pair(members, a, b)
    if m_both is not None:
        tag = m_both.encode_pair(a, b)
        m_both.encoded_tags.append(tag)
        return ("intra", m_both.name, tag)
    tag_name = f"{a}_{b}"
    encoded_in = []
    for m in members:
        if a in m.vocab_set:
            m.encode_partial(a, tag_name)
            m.encoded_tags.append(tag_name)
            encoded_in.append((m.name, a))
        elif b in m.vocab_set:
            m.encode_partial(b, tag_name)
            m.encoded_tags.append(tag_name)
            encoded_in.append((m.name, b))
    return ("cross", [n for n, _ in encoded_in], tag_name)


def run_seed(seed: int):
    t0 = time.time()
    print(f"\n############ SEED {seed} ############", flush=True)
    members = build_members(seed)
    for m in members:
        m.load(seed)
        print(f"  [{m.name}: {m.n_concepts()} concepts, "
              f"{len(m.encoded_tags)} pre-existing tags]", flush=True)
    print(f"  [all bridges loaded @ {time.time()-t0:.0f}s]", flush=True)

    # validate vocab membership
    for hub, spec in HUB_GRAPH.items():
        assert find_member_for_word(members, hub) is not None, \
            f"hub '{hub}' not in any vocab"
        assert find_member_for_word(members, spec["C"]) is not None, \
            f"C '{spec['C']}' not in any vocab"
        for a in spec["nouns"]:
            assert find_member_for_word(members, a) is not None, \
                f"noun '{a}' not in any vocab"

    # ---------------------------------------------------------------
    # STEP 1: encode all A -> HUB edges (fan-in built here)
    # ---------------------------------------------------------------
    print("  --- STEP 1: encode A->HUB edges (build fan-in) ---", flush=True)
    for hub, spec in HUB_GRAPH.items():
        for a in spec["nouns"]:
            info = encode_pair(members, a, hub)
            print(f"    {a:8} -> {hub:5} : {info[0]} {info[2]}", flush=True)

    # ---------------------------------------------------------------
    # STEP 2: ANTI-CHEAT pre-check -- with ONLY A->HUB encoded, is C
    # already in A's top-3? (should be NO; A->C never encoded)
    # ---------------------------------------------------------------
    print("  --- STEP 2: ANTI-CHEAT pre-check (A->C retrievable? should be NO) ---",
          flush=True)
    anticheat_total = 0
    anticheat_pass = 0
    for hub, spec in HUB_GRAPH.items():
        c = spec["C"]
        for a in spec["nouns"]:
            ranked = query_concept_ranked(members, a, top_n=TOP_K)
            topk = [r["word"] for r in ranked]
            c_in = c in topk
            anticheat_total += 1
            if not c_in:
                anticheat_pass += 1
            print(f"    fan{len(spec['nouns'])} {a:8} top3={topk}  "
                  f"C='{c}' present? {c_in}", flush=True)
    print(f"    ANTI-CHEAT: {anticheat_pass}/{anticheat_total} have C NOT "
          f"directly retrievable from A", flush=True)

    # ---------------------------------------------------------------
    # STEP 3: encode all HUB -> C edges (intra-bridge adj-adj)
    # ---------------------------------------------------------------
    print("  --- STEP 3: encode HUB->C edges ---", flush=True)
    for hub, spec in HUB_GRAPH.items():
        info = encode_pair(members, hub, spec["C"])
        print(f"    {hub:5} -> {spec['C']:6} : {info[0]} {info[2]}", flush=True)

    # ---------------------------------------------------------------
    # STEP 4: per-fan-in hop-1, hop-2 (hub crowded), full 2-hop
    # ---------------------------------------------------------------
    print("  --- STEP 4: hop-1 / hop-2(crowded hub) / full 2-hop ---",
          flush=True)
    # group by fan-in level
    by_fanin = {}  # fanin -> list of (a, hub, c)
    for hub, spec in HUB_GRAPH.items():
        fi = len(spec["nouns"])
        for a in spec["nouns"]:
            by_fanin.setdefault(fi, []).append((a, hub, spec["C"]))

    seed_rows = {}  # fanin -> dict of counts
    for fi in sorted(by_fanin):
        chains = by_fanin[fi]
        n = len(chains)
        hop1_top1 = 0
        hop2_marg = 0   # query TRUE hub H -> C in top3 (crowding test)
        two_hop = 0     # chain hop1 actual top1 -> C in top3
        for (a, hub, c) in chains:
            r1 = query_concept_ranked(members, a, top_n=10)
            r1w = [r["word"] for r in r1]
            h1 = bool(r1) and r1[0]["word"] == hub
            if h1:
                hop1_top1 += 1
            # hop-2 marginal: query the TRUE hub (this is the crowding test --
            # hub H has fan-in incoming nouns competing with C)
            r2t = query_concept_ranked(members, hub, top_n=10)
            r2tw = [r["word"] for r in r2t]
            c_marg = c in r2tw[:TOP_K]
            if c_marg:
                hop2_marg += 1
            # full 2-hop transitive: chain hop1's ACTUAL top1
            src = r1[0]["word"] if r1 else None
            r2 = query_concept_ranked(members, src, top_n=10) if src else []
            r2w = [r["word"] for r in r2]
            c_2hop = c in r2w[:TOP_K]
            if c_2hop:
                two_hop += 1
            print(f"    fan{fi} {a:8}->{hub:5}->{c:6} | "
                  f"hop1 top3={r1w[:TOP_K]} (H top1={h1}) | "
                  f"hub '{hub}' top3={r2tw[:TOP_K]} (C marg={c_marg}) | "
                  f"2hop src='{src}' top3={r2w[:TOP_K]} (C={c_2hop})",
                  flush=True)
        seed_rows[fi] = {
            "n": n,
            "hop1_top1": hop1_top1,
            "hop2_marg": hop2_marg,
            "two_hop": two_hop,
        }
        print(f"    >> fan-in {fi} (n={n}): hop1_top1={hop1_top1}/{n}  "
              f"hop2_marg(C in crowded-hub top3)={hop2_marg}/{n}  "
              f"full_2hop={two_hop}/{n}", flush=True)

    print(f"  [seed {seed} wall {time.time()-t0:.0f}s]", flush=True)
    return {
        "anticheat_pass": anticheat_pass,
        "anticheat_total": anticheat_total,
        "by_fanin": seed_rows,
    }


def main():
    t0 = time.time()
    print("=== HUB-REUSE MULTI-HOP REASONING PROBE (G.20 sparse multitag) ===",
          flush=True)
    seeds = [s for s in (42, 43, 44)
             if all(os.path.exists(f"{SEED_BRIDGE_DIRS[s]}/{b}")
                    for b in BRIDGE_BASENAMES)]
    print(f"Seeds with bridges on disk: {seeds}", flush=True)
    if not seeds:
        print("NO bridges found -- abort.", flush=True)
        sys.exit(1)

    results = {}
    for s in seeds:
        results[s] = run_seed(s)

    # ---------------- aggregate ----------------
    print("\n\n================ AGGREGATE ================", flush=True)
    fanins = sorted({fi for s in seeds for fi in results[s]["by_fanin"]})

    # chance baseline for "C in top-3": C is always an adjective; the adj
    # bridge has 32 concepts. Top-3 of 32 = 3/32. (This is the chance that a
    # specific target adj lands in a top-3 by luck.)
    chance_top3 = TOP_K / 32.0

    print(f"\nChance baseline (specific C in top-3 of 32 adj): "
          f"{chance_top3:.3f}", flush=True)
    print(f"\nANTI-CHEAT (A->C NOT directly retrievable before HUB->C edge):",
          flush=True)
    for s in seeds:
        ac = results[s]
        print(f"  seed {s}: {ac['anticheat_pass']}/{ac['anticheat_total']}",
              flush=True)

    print("\n--- DECISIVE TABLE: 2-hop accuracy vs HUB FAN-IN ---", flush=True)
    header = (f"{'fan-in':>7} | {'seed':>4} | {'hop1_top1':>10} | "
              f"{'hop2_marg':>10} | {'full_2hop':>10}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    fanin_means = {}  # fanin -> (mean hop1, mean hop2marg, mean 2hop)
    for fi in fanins:
        h1s, h2s, t2s = [], [], []
        for s in seeds:
            row = results[s]["by_fanin"].get(fi)
            if row is None:
                continue
            n = row["n"]
            h1 = row["hop1_top1"] / n
            h2 = row["hop2_marg"] / n
            t2 = row["two_hop"] / n
            h1s.append(h1); h2s.append(h2); t2s.append(t2)
            print(f"{fi:>7} | {s:>4} | {row['hop1_top1']:>2}/{n} ={h1:>5.2f} | "
                  f"{row['hop2_marg']:>2}/{n} ={h2:>5.2f} | "
                  f"{row['two_hop']:>2}/{n} ={t2:>5.2f}", flush=True)
        mh1 = float(np.mean(h1s)) if h1s else float("nan")
        mh2 = float(np.mean(h2s)) if h2s else float("nan")
        mt2 = float(np.mean(t2s)) if t2s else float("nan")
        fanin_means[fi] = (mh1, mh2, mt2)
        print(f"{fi:>7} | {'MEAN':>4} | {'':6}={mh1:>5.2f} | "
              f"{'':6}={mh2:>5.2f} | {'':6}={mt2:>5.2f}", flush=True)
        print("-" * len(header), flush=True)

    # ---------------- pre-registered verdict ----------------
    # ROBUST: full 2-hop >= 0.50 at fan-in 8 (multi-seed mean)
    # DEGRADES-WITH-FANIN: high at fan-in 2, drops below 0.50 by fan-in 8
    # NEGATIVE: below 0.50 even at fan-in 2 multi-seed
    print("\n=== 2-hop-vs-fanin curve (multi-seed mean full_2hop) ===",
          flush=True)
    for fi in fanins:
        print(f"  fan-in {fi}: full_2hop mean = {fanin_means[fi][2]:.3f} "
              f"(hop1={fanin_means[fi][0]:.3f}, "
              f"hop2_marg={fanin_means[fi][1]:.3f})", flush=True)

    max_fanin = max(fanins)
    min_fanin = min(fanins)
    t2_at_max = fanin_means[max_fanin][2]
    t2_at_min = fanin_means[min_fanin][2]

    if t2_at_max >= 0.50:
        verdict = "ROBUST"
    elif t2_at_min >= 0.50 and t2_at_max < 0.50:
        verdict = "DEGRADES-WITH-FANIN"
    elif t2_at_min < 0.50:
        verdict = "NEGATIVE"
    else:
        verdict = "UNCLASSIFIED (see numbers)"

    print(f"\n  PRE-REGISTERED VERDICT: {verdict}", flush=True)
    print(f"    deciding: full_2hop at fan-in {min_fanin} = {t2_at_min:.3f}, "
          f"at fan-in {max_fanin} = {t2_at_max:.3f}; "
          f"ROBUST>=0.50@max, chance={chance_top3:.3f}", flush=True)
    print(f"\n[TOTAL wall {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
