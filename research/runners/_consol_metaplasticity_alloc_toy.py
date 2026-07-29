#!/usr/bin/env python3
"""Does WEIGHT-HISTORY METAPLASTICITY produce a FREE-vs-TAKEN signal? (seconds, CPU)

THE PROBLEM, in one sentence (2026-07-28 findings): a slot's own accumulated afferent weight must set
its availability for a NEW binding -- uncued, and without a host read.

WHY THIS FAMILY, AND WHY IT IS NEW. Four mechanisms are REFUTED at 6 seeds, and all four are
FAIRNESS/COMPETITION mechanisms -- they equalise *who wins*: more replay (1/6, WORSE - rich-get-richer),
winner-inactive depression, duty-cycle threshold boost (wrong variable), Turrigiano synaptic scaling.
Fairness is not allocation: boosting equalises how OFTEN a slot wins but cannot stop two facts claiming
the SAME slot. Metaplasticity is a different quantity entirely -- not "who is winning" but "is this cell
ALREADY CLAIMED" -- which is exactly free-vs-taken. VERIFIED absent from the engine (no bcm /
sliding_threshold / metaplast / theta_m in sim/config.py, bridge.py, kernels.py).

THE RULE UNDER TEST (BCM sliding threshold, Bienenstock-Cooper-Munro 1982, applied to weight history):
    theta_i = theta_0 + beta * sum_j w_ij        # LOCAL to the postsynaptic cell: its OWN afferents
    potentiate slot i only if drive_i > theta_i  # an already-claimed cell resists a NEW binding
The metaplastic variable is a sum over cell i's own incoming synapses -- the same quantity
`_committed_count` already reads (_emerge14_stageC_onbridge_learning_derisk.py:154-165), and the same
one occupied-slot suppression already used on-substrate (_stp_binder_onbridge_derisk.py:88). It is
postsynaptically local, so it is realisable as an intrinsic property rather than a host lookup.

ANTI-CHEATS (each earned by a real failure this arc):
  * ENGAGEMENT COUNTERS -- an occupancy arm scored 0/6 purely because its threshold was never reachable
    and it silently degenerated into the control. Both branches must be shown to FIRE.
  * STABILITY -- a valid permutation is worthless if re-presenting a fact retrieves a different slot.
  * PLAIN CONTROL -- must still collapse, or the toy regime is too easy to be informative.
  * theta SWEEP -- report the whole beta range, not the one value that works, so a knife-edge is visible
    as a knife-edge.
"""
from __future__ import annotations

import numpy as np

N_SLOTS = 3
N_FEAT = 24
SEEDS = (42, 43, 44, 100, 101, 102)


def run(seed, beta, n_epochs=6, lr=0.02, w_max=1.0, theta0=0.35, noise=0.02):
    """Present N_SLOTS distinct facts repeatedly; each competes for a slot and binds Hebbian.

    Returns the fact->slot map, engagement counters, and stability.
    """
    rng = np.random.default_rng(seed)
    facts = rng.normal(size=(N_SLOTS, N_FEAT))
    facts /= np.linalg.norm(facts, axis=1, keepdims=True)
    W = rng.uniform(0, 0.01, size=(N_SLOTS, N_FEAT))  # slot x feature

    n_potentiated = 0   # the ALLOCATE branch fired (a slot was newly claimed)
    n_blocked = 0       # metaplasticity actually BLOCKED a claimed slot (the mechanism engaged)

    for _ in range(n_epochs):
        for f in rng.permutation(N_SLOTS):          # random order: no presentation-order artifact
            c = facts[f] + rng.normal(0, noise, N_FEAT)
            drive = W @ c                            # each slot's drive from this fact
            claimed = W.sum(axis=1)                  # LOCAL: cell i's own afferent total
            # BCM sliding threshold, subtractive form: an already-claimed cell needs MORE drive to win.
            # beta=0 reduces to plain argmax = the winner-take-all runaway (the honest control).
            score = drive - beta * claimed
            i = int(np.argmax(score))
            if i != int(np.argmax(drive)):
                n_blocked += 1                       # metaplasticity OVERRODE the runaway winner
            W[i] += lr * c * (w_max - W[i].mean())   # one-shot Hebbian bind, soft-bounded
            W[i] = np.clip(W[i], 0, w_max)
            n_potentiated += 1

    # READ-OUT: which slot does each fact retrieve? (pure argmax of drive, no metaplasticity at read)
    mapping = [int(np.argmax(W @ facts[f])) for f in range(N_SLOTS)]
    # STABILITY: re-present each fact with fresh noise; must retrieve the SAME slot
    stable = all(int(np.argmax(W @ (facts[f] + rng.normal(0, noise, N_FEAT)))) == mapping[f]
                 for f in range(N_SLOTS))
    valid = sorted(mapping) == list(range(N_SLOTS))  # a genuine permutation: distinct facts, distinct slots
    return dict(mapping=mapping, valid=bool(valid), stable=bool(stable),
                n_potentiated=n_potentiated, n_blocked=n_blocked)


def main():
    print("=" * 78)
    print("METAPLASTICITY (BCM sliding threshold on WEIGHT HISTORY) as a free-vs-taken signal")
    print("=" * 78)
    print("gate: permutation_valid AND stable, on 6/6 seeds. beta=0 is the PLAIN control.\n")
    print("%-8s %-10s %-10s %-14s %s" % ("beta", "valid", "stable", "engaged", "maps"))
    best = None
    for beta in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        rs = [run(s, beta) for s in SEEDS]
        nv = sum(r["valid"] for r in rs)
        ns = sum(r["stable"] for r in rs)
        blocked = sum(r["n_blocked"] for r in rs)
        both = sum(r["valid"] and r["stable"] for r in rs)
        # ENGAGEMENT: for beta>0 the mechanism must actually have blocked something, else this arm
        # silently degenerated into the control and its numbers mean nothing.
        eng = "n/a (control)" if beta == 0 else ("%d blocks" % blocked if blocked else "NEVER ENGAGED")
        flag = ""
        if beta > 0 and not blocked:
            flag = "   <- VOID ARM: threshold never reached, this is the control in disguise"
        print("%-8.2f %-10s %-10s %-14s %s%s"
              % (beta, "%d/6" % nv, "%d/6" % ns, eng, [r["mapping"] for r in rs], flag))
        if beta > 0 and blocked and (best is None or both > best[1]):
            best = (beta, both)

    print()
    plain = sum(run(s, 0.0)["valid"] and run(s, 0.0)["stable"] for s in SEEDS)
    print("PLAIN control (beta=0): %d/6 valid+stable" % plain)
    if plain >= 5:
        print("  WARNING: the control already passes -> this toy regime is TOO EASY to be informative.")
    if best:
        print("BEST metaplastic beta=%.2f: %d/6 valid+stable" % (best[0], best[1]))
        print()
        print("VERDICT: %s" % ("GO -- metaplasticity produces the free-vs-taken signal; take it to the "
                               "substrate." if best[1] == 6 and plain <= 2 else
                               "NOT a clean GO -- read the sweep above before building anything."))
    else:
        print("VERDICT: every metaplastic arm VOID or inert. The rule as written does not engage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
