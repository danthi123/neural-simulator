"""THROWAWAY diagnostic (raw/): localize WHY the directional multi-hop "trace"
is bimodal -- the rescue was RESCUED-but-BIMODAL (seed 42 big_red 8/8, seed 43
0/8, seed 44 6/8; finding 2026-05-31-P4-multihop-directional-fix-...). Is seed
43's big_red failure (a) weak RECALL strength of the big_red engram (the tag
exists but stim'ing it barely activates 'red' -- ENCODING/RECALL-strength,
actionable via stronger encode), or (b) 'red' is structurally OUT-COMPETED by
other adjectives at recall (structural per-seed variance, harder)?

All HUB->C pairs (big->red fan-in8, hot->dry fan-in4, fast->small fan-in2) are
INTRA bridgeC_adj, so we only load bridgeC_adj per seed (cheap, ~30s each).
For each seed + pair: encode_pair -> stim the tag -> report the TARGET adj's
recall rate AND its RANK among all 32 adjectives. Low rate + bad rank on seed 43
big_red while seed 42 is high = the binding is genuinely weak on that bridge.

Reuses (by import) SharedPoolMember from g20_multibridge (byte-unchanged).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

from research.runners.g20_multibridge import SharedPoolMember, read_vocab_file

N_LANG_INPUT = 8192
N_SHARED_POOL = 2000
SPARSITY = 0.02
PATTERN_SIZE = 100
DRIVE_PA = 1500.0
DRIVE_STEPS = 100

VD = "research/findings/raw/g11_bg"
SEED_DIRS = {42: f"{VD}/g20_sparse_bridges",
             43: f"{VD}/g20_sparse_bridges_s43",
             44: f"{VD}/g20_sparse_bridges_s44"}
ADJ_BASENAME = "bridgeC_adj_sparse.simstate.h5"
ADJ_VOCAB = f"{VD}/g20_bridgeC_adj_vocab.txt"

PAIRS = [("big", "red"), ("hot", "dry"), ("fast", "small"), ("cold", "wet")]


def run_seed(seed):
    bdir = SEED_DIRS[seed]
    path = f"{bdir}/{ADJ_BASENAME}"
    if not os.path.exists(path):
        print(f"  [seed {seed}: no bridgeC_adj at {path}]", flush=True)
        return None
    vocab = read_vocab_file(ADJ_VOCAB)
    m = SharedPoolMember(bridge_path=path, vocab=vocab, name="bridgeC_adj",
                         n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
                         sparsity=SPARSITY, drive_pA=DRIVE_PA, drive_steps=DRIVE_STEPS,
                         sparse=True, pattern_size=PATTERN_SIZE)
    t0 = time.time()
    m.load(seed)
    print(f"\n### SEED {seed} (bridgeC_adj: {m.n_concepts()} adjectives, loaded {time.time()-t0:.0f}s) ###",
          flush=True)
    rows = []
    for (a, b) in PAIRS:
        if a not in m.vocab_set or b not in m.vocab_set:
            print(f"  [{a}->{b}: not both in adj vocab; skip]", flush=True)
            continue
        tag = m.encode_pair(a, b)
        m.encoded_tags.append(tag)
        rates = m.recall_rates(tag)
        order = np.argsort(-rates)
        b_idx = m.vocab.index(b)
        b_rate = float(rates[b_idx])
        b_rank = int(np.where(order == b_idx)[0][0]) + 1  # 1 = top
        top3 = [(m.vocab[j], float(rates[j])) for j in order[:3]]
        a_idx = m.vocab.index(a)
        a_rate = float(rates[a_idx])
        rows.append((a, b, b_rate, b_rank, a_rate))
        print(f"  {a:6}->{b:6}: target '{b}' rate={b_rate:7.1f} rank={b_rank:2d}/32 | "
              f"cue '{a}' rate={a_rate:7.1f} | top3={[(w, round(r)) for w, r in top3]}",
              flush=True)
    return rows


def main():
    seeds = [s for s in (42, 43, 44) if os.path.exists(f"{SEED_DIRS[s]}/{ADJ_BASENAME}")]
    print("=== PER-SEED BINDING-STRENGTH DIAGNOSTIC (bridgeC_adj engram recall) ===", flush=True)
    print(f"seeds={seeds}; for each pair, stim the encoded tag, report TARGET recall rate + rank/32",
          flush=True)
    allrows = {}
    for s in seeds:
        allrows[s] = run_seed(s)

    print("\n================ SUMMARY: target recall rate + rank by seed ================", flush=True)
    print(f"{'pair':14} | " + " | ".join(f"seed{s} (rate/rank)" for s in seeds), flush=True)
    for (a, b) in PAIRS:
        cells = []
        for s in seeds:
            r = allrows.get(s)
            hit = next((row for row in (r or []) if row[0] == a and row[1] == b), None)
            cells.append(f"{hit[2]:6.0f}/{hit[3]:2d}" if hit else "   --   ")
        print(f"{a+'->'+b:14} | " + " | ".join(cells), flush=True)
    print("\nREAD: big->red rank 1-3 on seed 42 but >3 (buried) on seed 43 => the big_red binding is "
          "genuinely WEAK on seed 43's bridge (recall-strength), so the bimodality is encoding/recall "
          "strength (actionable: stronger encode teacher) NOT a directional-filter issue. If big->red "
          "rank is fine (1-3) on ALL seeds, the multi-hop bimodality is elsewhere (chaining).", flush=True)


if __name__ == "__main__":
    main()
