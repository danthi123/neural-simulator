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
    """Fraction of intended positions whose concept matches, divided
    by max(len(intended), len(produced)) so TRAILING CONFABULATION is
    penalized (a no-confabulation moat must see over-production). A
    clean terminal stop is NOT confabulation: trailing -1 sentinels
    (the SongHVC chain-end marker) are stripped before length is
    measured. 1.0 iff produced == intended exactly. In [0, 1]."""
    if not intended:
        return 0.0
    n = len(intended)
    produced = list(decoded)
    while produced and produced[-1] == -1:   # clean stop, not confab
        produced.pop()
    denom = max(n, len(produced))
    hits = sum(1 for i in range(n)
               if i < len(produced) and produced[i] == intended[i])
    return hits / float(denom)


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


_G1_MARGIN = 0.10  # FIXED pre-registered bar; never tuned post-hoc
_G1_ABS_FLOOR = 0.5   # FIXED pre-registered: a generative claim needs the
                      # MAJORITY of the proposition correctly ordered, not
                      # a tiny relative edge over a near-zero permuted score

def g1_verdict(true_score: float, best_perm_score: float,
               gate_cleared: bool) -> dict:
    """Pre-registered FIXED bar. PASS requires ALL of:
      (a) the produced proposition cleared the abstention gate,
      (b) a real permuted-ORDER contrast exists (best_perm_score > 0;
          if every permuted control decodes to nothing there is NO
          evidence of ORDER-learning, only weak concept ignition -> FAIL),
      (c) true-order score clears an absolute floor (>= _G1_ABS_FLOOR:
          the majority of the proposition correctly ordered), and
      (d) true-order beats the best permuted-ORDER control by >= 10%
          (relative; documented ">=", honored with a float epsilon).
    Bars (_G1_MARGIN, _G1_ABS_FLOOR) are module constants, never
    per-call tunable."""
    ts, ps = float(true_score), float(best_perm_score)
    pct = (100.0 * (ts - ps) / ps) if ps > 0.0 else 0.0
    gate = bool(
        gate_cleared
        and ps > 0.0
        and ts >= _G1_ABS_FLOOR
        and ts >= ps * (1.0 + _G1_MARGIN) - 1e-9
    )
    return {
        "true_score": ts, "best_perm_score": ps,
        "pct_over_permuted": pct, "gate_cleared": bool(gate_cleared),
        "margin_required_pct": 100.0 * _G1_MARGIN,
        "abs_floor": _G1_ABS_FLOOR,
        "gate": gate, "GATE": "PASS" if gate else "FAIL",
    }
