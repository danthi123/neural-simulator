"""Scratch de-risk (verification-independent) for the Fork-2 rung-3 reframe.

CLAIM UNDER TEST: a two-(or-more)-overlapping-sequences next-symbol task can be built so that a k-th-order Markov
(count) predictor PROVABLY fails at the disambiguating branch point for any fixed k < the shared-middle length --
i.e. success there REQUIRES carrying long-range context. If this holds, the `context_lesion` anti-cheat (beat the
Markov floor) is meaningful and the reframe has a real high-order target. If a low-order Markov already wins, the
reframe is a bigram-lookup triviality and must be redesigned.

Design (Bouhadjar-Diesmann style): each sequence = [cue] + [SHARED middle of length L] + [branch].
  seq_i:  cue_i,  s_0, s_1, ..., s_{L-1},  branch_i
The middle s_0..s_{L-1} is IDENTICAL across all sequences; only the first (cue) and last (branch) symbols differ.
Predicting branch_i at the branch step requires remembering cue_i across L shared symbols -> order-(L+1) context.
A k-th-order Markov predictor at the branch step conditions on the last k symbols = (s_{L-k}..s_{L-1}) which are
identical across sequences -> it must predict the SAME distribution for every sequence -> chance among the branches.
No CPU-heavy work; pure logic + counting. numpy only.
"""
import numpy as np
from collections import defaultdict, Counter


def make_overlap_sequences(n_seq=4, middle_len=8, seed=42):
    """n_seq sequences sharing an identical middle of length `middle_len`; distinct cue + branch per sequence.
    Returns (sequences, vocab). Symbols are ints. cues=[0..n_seq-1], middle=[n_seq..n_seq+middle_len-1],
    branches=[n_seq+middle_len .. n_seq+middle_len+n_seq-1]. All distinct so the structure is unambiguous."""
    rng = np.random.default_rng(seed)
    cues = list(range(n_seq))
    middle = list(range(n_seq, n_seq + middle_len))
    branches = list(range(n_seq + middle_len, n_seq + middle_len + n_seq))
    seqs = [[cues[i]] + middle + [branches[i]] for i in range(n_seq)]
    vocab = n_seq + middle_len + n_seq
    return seqs, vocab, {"cues": cues, "middle": middle, "branches": branches}


def kth_order_markov_branch_acc(seqs, k, info):
    """Train a k-th-order count Markov model on all sequences, then evaluate its accuracy at the BRANCH step
    (predicting the final symbol) using only the preceding k symbols as context. Ties -> expected accuracy = 1/ties."""
    # build counts: context (last k symbols) -> Counter(next symbol)
    counts = defaultdict(Counter)
    for s in seqs:
        for t in range(len(s) - 1):
            ctx = tuple(s[max(0, t - k + 1): t + 1])   # up to k preceding symbols (inclusive of current)
            counts[ctx][s[t + 1]] += 1
    # evaluate at the branch step of each sequence
    correct = 0.0
    for s in seqs:
        t = len(s) - 2                                  # step whose next-symbol is the branch
        ctx = tuple(s[max(0, t - k + 1): t + 1])
        dist = counts[ctx]
        true_next = s[t + 1]
        if not dist:
            correct += 1.0 / len(info["branches"])      # unseen ctx -> chance
            continue
        top = max(dist.values())
        winners = [sym for sym, c in dist.items() if c == top]
        # expected accuracy under uniform tie-break, credited only if the true branch is among the argmax winners
        correct += (1.0 / len(winners)) if true_next in winners else 0.0
    return correct / len(seqs)


def full_context_branch_acc(seqs, info):
    """An oracle that conditions on the FULL prefix (=knows the cue) -> should be 1.0 (task is solvable with context)."""
    counts = defaultdict(Counter)
    for s in seqs:
        for t in range(len(s) - 1):
            ctx = tuple(s[: t + 1])                      # full prefix
            counts[ctx][s[t + 1]] += 1
    correct = 0.0
    for s in seqs:
        t = len(s) - 2
        ctx = tuple(s[: t + 1])
        dist = counts[ctx]
        true_next = s[t + 1]
        top = max(dist.values()); winners = [sym for sym, c in dist.items() if c == top]
        correct += (1.0 / len(winners)) if true_next in winners else 0.0
    return correct / len(seqs)


if __name__ == "__main__":
    for (n_seq, L) in [(2, 4), (4, 8), (4, 16), (8, 12)]:
        seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L)
        chance = 1.0 / n_seq
        full = full_context_branch_acc(seqs, info)
        print(f"\n=== n_seq={n_seq}, middle_len={L}, vocab={vocab}, chance={chance:.3f} ===")
        print(f"  full-context oracle branch-acc: {full:.3f}  (task solvable WITH context: {'YES' if full > 0.99 else 'NO'})")
        for k in [1, 2, 3, 5, L, L + 1, L + 2]:
            acc = kth_order_markov_branch_acc(seqs, k, info)
            note = ""
            if k <= L:
                note = "  <- should be ~chance (ctx inside shared middle)"
            elif k >= L + 1:
                note = "  <- k reaches the cue -> can disambiguate"
            print(f"  {k:>2}-order Markov branch-acc: {acc:.3f}{note}")
