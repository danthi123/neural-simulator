"""G1 pure scoring / control / reward logic (CPU-testable).

The permuted-ORDER control is the load-bearing anti-cheat: it has the
SAME concept multiset, only the ORDER scrambled. A system that merely
ignites the right concepts (no learned order) scores the true order
~equal to permuted; only genuine order-learning beats it >=10%.
"""
from __future__ import annotations
from itertools import permutations
import numpy as np


def score_order(decoded: list, intended: list) -> float:
    """1.0 iff decoded == intended; partial credit = fraction of
    positions whose concept matches the intended position. Pure,
    deterministic, in [0, 1]."""
    if not intended:
        return 0.0
    n = len(intended)
    d = list(decoded)[:n] + [None] * max(0, n - len(decoded))
    hits = sum(1 for i in range(n) if d[i] == intended[i])
    return hits / float(n)


def permuted_order_controls(intended: list, rng, n: int) -> list:
    """Up to n distinct non-identity orderings of the SAME multiset.
    Deterministic given rng. Exhaustive when n! is small."""
    base = list(intended)
    perms = [list(p) for p in set(permutations(base))
             if list(p) != base]
    perms.sort()
    if not perms:
        return []
    idx = rng.permutation(len(perms))[:n]
    return [perms[i] for i in sorted(idx.tolist())]


def compose_reward(decoded: list, intended: list,
                   gate_cleared: bool) -> float:
    """Self-comprehension agreement -> DA reward. Gate not cleared
    (any produced slot below the abstention gate) -> 0.0 (the
    no-confabulation moat: never reward a confabulated/low-confidence
    production). Else = ordered match score."""
    if not gate_cleared:
        return 0.0
    return score_order(decoded, intended)
