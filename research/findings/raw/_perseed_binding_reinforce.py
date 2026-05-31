"""THROWAWAY (raw/): does REINFORCEMENT (re-encoding a weak pair multiple times)
raise the target's recall rank -> a simple no-code-change fix for the multi-hop
"trace" bimodality (diagnosed as per-pair-per-seed engram recall-strength lottery,
finding 2026-05-31-P4-multihop-trace-bimodality-DIAGNOSED-...).

Two weak cases from the diagnostic: seed 43 big->red (rank 8) and seed 44
cold->wet (rank 8). Encode the pair, read target rank; re-encode up to 5x,
re-reading target rank each time. If rank improves to <=3 -> reinforcement is
the actionable fix. If flat -> the per-seed sparse-pattern structure caps the
binding (deeper change or accept the lottery).

Reuses SharedPoolMember from g20_multibridge (byte-unchanged).
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
N_REINFORCE = 5

VD = "research/findings/raw/g11_bg"
SEED_DIRS = {42: f"{VD}/g20_sparse_bridges",
             43: f"{VD}/g20_sparse_bridges_s43",
             44: f"{VD}/g20_sparse_bridges_s44"}
ADJ_BASENAME = "bridgeC_adj_sparse.simstate.h5"
ADJ_VOCAB = f"{VD}/g20_bridgeC_adj_vocab.txt"

# (seed, cue, target) weak cases
CASES = [(43, "big", "red"), (44, "cold", "wet"), (42, "big", "red")]  # 42 = strong control


def target_rank(m, tag, target):
    rates = m.recall_rates(tag)
    order = np.argsort(-rates)
    t_idx = m.vocab.index(target)
    rank = int(np.where(order == t_idx)[0][0]) + 1
    return float(rates[t_idx]), rank


def run_case(seed, cue, target):
    path = f"{SEED_DIRS[seed]}/{ADJ_BASENAME}"
    if not os.path.exists(path):
        print(f"[seed {seed}: no bridge]", flush=True)
        return
    vocab = read_vocab_file(ADJ_VOCAB)
    m = SharedPoolMember(bridge_path=path, vocab=vocab, name="bridgeC_adj",
                         n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
                         sparsity=SPARSITY, drive_pA=DRIVE_PA, drive_steps=DRIVE_STEPS,
                         sparse=True, pattern_size=PATTERN_SIZE)
    t0 = time.time()
    m.load(seed)
    print(f"\n### seed {seed} {cue}->{target} (loaded {time.time()-t0:.0f}s) ###", flush=True)
    tag = None
    for i in range(1, N_REINFORCE + 1):
        tag = m.encode_pair(cue, target)      # re-encode (reinforce) each iteration
        if tag not in m.encoded_tags:
            m.encoded_tags.append(tag)
        rate, rank = target_rank(m, tag, target)
        flag = "<= top3 OK" if rank <= 3 else ""
        print(f"  encode x{i}: target '{target}' rate={rate:7.1f} rank={rank:2d}/32  {flag}",
              flush=True)


def main():
    print("=== REINFORCEMENT TEST: does re-encoding raise weak-pair target rank? ===", flush=True)
    print(f"N_REINFORCE={N_REINFORCE}; cases={CASES}", flush=True)
    for (s, c, t) in CASES:
        run_case(s, c, t)
    print("\nREAD: if rank falls to <=3 with more encodes on the weak cases (seed43 big->red, "
          "seed44 cold->wet) -> reinforcement is the simple actionable fix. If flat at rank>3 -> "
          "per-seed sparse-pattern structure caps it (needs deeper balanced-teacher encode or "
          "accept the lottery).", flush=True)


if __name__ == "__main__":
    main()
