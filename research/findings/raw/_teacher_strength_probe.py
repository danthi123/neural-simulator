"""THROWAWAY (raw/): does a STRONGER encode teacher (teacher_pA) reliably raise
a weak pair's target recall rank -> the balanced-teacher fix for the multi-hop
"trace" bimodality (recall-strength lottery, finding 2026-05-31-P4-multihop-trace-
bimodality-DIAGNOSED-...)? encode_pair_engram_sparse has teacher_pA=100.0 default;
SharedPoolMember.encode_pair uses the default. Test stronger teachers.

FRESH bridge per (case, teacher) so each is an INDEPENDENT single encode at that
strength (not accumulating like the unstable reinforcement re-encode test).
Cases: the two weak pairs (seed 43 big->red rank 8, seed 44 cold->wet rank 8).
If a stronger teacher reliably lifts target to rank <=3 -> balanced-teacher is a
simple 1-line fix (pass higher teacher_pA in encode_pair). If flat/unstable ->
the sparse-pattern overlap caps it; accept the lottery.
"""
from __future__ import annotations
import os
import time
import numpy as np

from research.runners.g20_multibridge import SharedPoolMember, read_vocab_file
from research.runners.shared_pool_chat import encode_pair_engram_sparse

VD = "research/findings/raw/g11_bg"
SEED_DIRS = {43: f"{VD}/g20_sparse_bridges_s43", 44: f"{VD}/g20_sparse_bridges_s44",
             42: f"{VD}/g20_sparse_bridges"}
ADJ = "bridgeC_adj_sparse.simstate.h5"
ADJ_VOCAB = f"{VD}/g20_bridgeC_adj_vocab.txt"

TEACHERS = [100.0, 500.0, 1500.0, 3000.0]
CASES = [(43, "big", "red"), (44, "cold", "wet")]
N_LANG_INPUT = 8192
N_SHARED_POOL = 2000
SPARSITY = 0.02
PATTERN_SIZE = 100


def fresh_member(seed):
    m = SharedPoolMember(bridge_path=f"{SEED_DIRS[seed]}/{ADJ}",
                         vocab=read_vocab_file(ADJ_VOCAB), name="bridgeC_adj",
                         n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
                         sparsity=SPARSITY, drive_pA=1500.0, drive_steps=100,
                         sparse=True, pattern_size=PATTERN_SIZE)
    m.load(seed)
    return m


def main():
    print("=== TEACHER-STRENGTH PROBE: does stronger teacher_pA lift weak-pair target rank? ===", flush=True)
    print(f"teachers={TEACHERS}; cases={CASES} (fresh bridge per (case,teacher))", flush=True)
    results = {}
    for (seed, a, b) in CASES:
        row = []
        for T in TEACHERS:
            if not os.path.exists(f"{SEED_DIRS[seed]}/{ADJ}"):
                print(f"  [seed {seed}: no bridge]", flush=True); break
            t0 = time.time()
            m = fresh_member(seed)
            tag = encode_pair_engram_sparse(
                m.bridge, a, b, vocab=m.vocab, sparse_patterns=m.sparse_patterns,
                n_lang_input=m.n_lang_input, sparsity=m.sparsity, teacher_pA=T)
            rates = m.recall_rates(tag)
            order = np.argsort(-rates)
            b_idx = m.vocab.index(b)
            rank = int(np.where(order == b_idx)[0][0]) + 1
            rate = float(rates[b_idx])
            row.append((T, rate, rank))
            print(f"  seed {seed} {a}->{b}  teacher={T:6.0f}: target '{b}' rate={rate:7.1f} rank={rank:2d}/32 "
                  f"{'<=3 OK' if rank <= 3 else ''}  ({time.time()-t0:.0f}s)", flush=True)
            del m
        results[(seed, a, b)] = row

    print("\n=== SUMMARY (rank by teacher_pA) ===", flush=True)
    for (seed, a, b), row in results.items():
        cells = " ".join(f"T{int(T)}=r{rank}" for (T, _, rank) in row)
        improved = len(row) >= 2 and row[-1][2] <= 3 and row[0][2] > 3
        print(f"  seed {seed} {a}->{b}: {cells}  {'-> STRONGER TEACHER LIFTS IT' if improved else ''}", flush=True)
    print("\nREAD: if target reaches rank <=3 at higher teacher on BOTH weak cases monotonically -> "
          "balanced-teacher is the simple fix (pass higher teacher_pA in SharedPoolMember.encode_pair). "
          "If non-monotonic/flat -> sparse-pattern overlap caps it; accept the per-pair lottery.", flush=True)


if __name__ == "__main__":
    main()
