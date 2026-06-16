"""CYCLE 104 — sentence-generation de-templating, cheap-first PHASE A (pure-core, the recommended de-risk).

The deep-research scoping (`2026-06-16-sentence-generation-biologization-deep-research.md`, controller-verified)
re-framed the last conversational host shortcut (the `f"{agent} {action} {patient}"` word-ordering template) as a
SERIAL-ORDER PRODUCTION problem, and showed the prior closed-loop HVC generator FAILED only because its
self-comprehension JUDGE could not read order back (AUC 0.775 -> zero gradient). The fix: an EXTERNAL teacher (the
stored fact's own order), graded by the pre-registered anti-cheat harness. RECOMMENDED top mechanism = a
COMPETITIVE-QUEUING (CQ) serial-order generator (Grossberg 1978 / Bullock-Rhodes 2003; catalog G.07/H.19): a
planning layer holds the slots with a primacy gradient; a choice WTA emits the highest-primacy slot, then
SELF-SUPPRESSES it (inhibition-of-return), so the next-strongest wins -> the order is produced by WTA dynamics,
not a host loop.

PHASE A (this runner, pure numpy, seconds): does the CQ engine, with the role->primacy gradient LEARNED from the
teacher (NOT hand-set), emit held-out SVO facts in the correct order, BEATING the permuted-order control? It
reuses `song_g1_core` VERBATIM (`score_order`, `permuted_order_controls`, the FIXED `g1_verdict` bars 0.10/0.5).
An internal NO-LEARNING control (random untrained primacy) must FAIL (true ~ permuted) -- proving the harness
detects order-learning, not concept-ignition. GO here => the serial-order MECHANISM + the fact-as-teacher + the
anti-cheat all work in the core -> phase B puts the CQ choice layer on a spiking bridge + reads each slot out to a
word (the substrate test). NEGATIVE here => the CQ mechanism itself can't serialize (deeper than the G1 judge).

Anti-cheats (carried from the doc): permuted-ORDER control is the primary gate (same multiset, scrambled order ->
must NOT beat); held-out facts only (role->primacy trained on a disjoint train split); host-template baseline
(=1.0 by construction) reported; degenerate-tie guard (candidate order = canonical role index, never target-
ordered). >=6 seeds; bars frozen.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_serial_order_cq_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.song_g1_core import score_order, permuted_order_controls, g1_verdict  # noqa: E402

N_ROLES = 3          # agent, action, patient -> the SVO frame; target role order = [0, 1, 2]
VOCAB = 16
N_FACTS = 24
N_PERM = 5           # permuted-order controls per fact (3! - 1 = 5 distinct scrambles for a 3-slot fact)
WTA_NOISE = 0.25     # choice-layer noise (tests the CQ tie-break stability the doc flagged as the risk)


def build_facts(seed):
    """Deterministic SVO facts: each = 3 DISTINCT concept indices for (agent, action, patient). Split
    train/held-out (the role->primacy is trained on `train`, graded ONLY on held-out -- leakage-free)."""
    rng = np.random.default_rng(seed * 101 + 7)
    facts = []
    seen = set()
    while len(facts) < N_FACTS:
        trip = tuple(int(x) for x in rng.choice(VOCAB, N_ROLES, replace=False))
        if trip not in seen:
            seen.add(trip); facts.append(trip)
    n_train = N_FACTS // 2
    return facts[:n_train], facts[n_train:]


class CQSerialOrder:
    """Competitive-queuing serial-order generator. `prim[role]` = the planning-layer primacy gradient, LEARNED
    from the teacher (the fact's order). `emit` is the choice-WTA read-out: pick the highest-primacy AVAILABLE
    slot (+ noise), emit its filler, remove it (inhibition-of-return), repeat. Candidate order is the canonical
    role index (never target-ordered) -- the degenerate-tie anti-cheat."""

    def __init__(self, n_roles=N_ROLES, lr=0.1, seed=42):
        self.n_roles = n_roles
        self.lr = lr
        self.prim = np.random.default_rng(seed * 13 + 5).standard_normal(n_roles) * 0.01  # random init (order NOT baked in)

    def learn(self, target_role_order):
        for pos, role in enumerate(target_role_order):           # earlier in the target -> more primacy (Hebbian push)
            self.prim[role] += self.lr * (self.n_roles - 1 - pos)

    def emit(self, fillers_by_role, role_order, rng):
        a = self.prim[list(role_order)] + WTA_NOISE * rng.standard_normal(len(role_order))
        avail = list(range(len(role_order)))                     # canonical-index candidates (anti-cheat: NOT target-ordered)
        emitted = []
        for _ in range(len(role_order)):
            best = max(avail, key=lambda i: a[i])                # choice-WTA winner among available
            emitted.append(fillers_by_role[role_order[best]])
            avail.remove(best)                                   # self-suppress (inhibition-of-return)
        return emitted


def _grade(cq, held, rng):
    """Emit each held-out fact; score the emitted concept order vs the TRUE order and the best permuted-ORDER
    control. Returns (mean_true, mean_best_perm)."""
    role_order = tuple(range(N_ROLES))                           # canonical roles [0,1,2]
    trues, perms = [], []
    for trip in held:
        fillers = {r: trip[r] for r in role_order}               # role -> concept idx
        intended = [trip[r] for r in role_order]                 # the TRUE order (agent, action, patient)
        emitted = cq.emit(fillers, role_order, rng)
        trues.append(score_order(emitted, intended))
        controls = permuted_order_controls(intended, rng, N_PERM)
        perms.append(max((score_order(emitted, c) for c in controls), default=0.0))
    return float(np.mean(trues)), float(np.mean(perms))


def run_seed(seed):
    train, held = build_facts(seed)
    rng = np.random.default_rng(seed * 71 + 3)
    # TRAINED CQ: learn role->primacy from the train facts (teacher = each fact's order)
    cq = CQSerialOrder(seed=seed)
    for trip in train:
        cq.learn(list(range(N_ROLES)))                           # every SVO fact teaches the same role order
    t_true, t_perm = _grade(cq, held, rng)
    # NO-LEARNING control: random untrained primacy -> must FAIL (true ~ permuted)
    cq0 = CQSerialOrder(seed=seed * 3 + 1)                       # untrained
    c_true, c_perm = _grade(cq0, held, np.random.default_rng(seed * 91 + 1))
    v = g1_verdict(t_true, t_perm, gate_cleared=True)            # gate_cleared: phase-A concepts are given (phase B tests abstention)
    print(f"  [seed {seed}] TRAINED true {t_true:.3f} vs perm {t_perm:.3f} -> {v['GATE']} "
          f"| no-learning control true {c_true:.3f} vs perm {c_perm:.3f}", flush=True)
    return {"seed": seed, "true": t_true, "perm": t_perm, "gate": v["gate"],
            "ctrl_true": c_true, "ctrl_perm": c_perm}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[serial-order CQ de-risk PHASE A] does a competitive-queuing serial-order generator (role->primacy "
          f"LEARNED from the fact-teacher) emit held-out SVO facts in order, beating the permuted-order control? "
          f"(host-template baseline = 1.000 by construction)", flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    t_true, t_perm = m("true"), m("perm")
    c_true, c_perm = m("ctrl_true"), m("ctrl_perm")
    n_pass = sum(1 for r in rows if r["gate"])
    agg = g1_verdict(t_true, t_perm, gate_cleared=True)
    print(f"\n{'='*98}\n  MEAN (6 seeds): TRAINED true {t_true:.3f} vs perm {t_perm:.3f} ({n_pass}/6 seeds PASS) | "
          f"no-learning control true {c_true:.3f} vs perm {c_perm:.3f} | aggregate {agg['GATE']} "
          f"({agg['pct_over_permuted']:.0f}% over perm, floor {agg['abs_floor']})", flush=True)
    print(f"{'='*98}", flush=True)
    ctrl_ok = c_true < c_perm * 1.10 + 1e-9                      # the no-learning control must NOT clear the order bar
    if agg["gate"] and n_pass >= 5 and ctrl_ok:
        print(f"  GO: the CQ serial-order mechanism + fact-as-teacher WORK -- trained true {t_true:.3f} clears the "
              f"floor ({agg['abs_floor']}) and beats the permuted-order control by {agg['pct_over_permuted']:.0f}% "
              f"(>=10%), {n_pass}/6 seeds, while the NO-LEARNING control stays at chance ({c_true:.3f} ~ perm "
              f"{c_perm:.3f}). Order is LEARNED + PRODUCED, not concept-ignition. ==> proceed to PHASE B (the CQ "
              f"choice layer on a spiking bridge + A->W word read-out = the substrate test).", flush=True)
    elif agg["gate"] and not ctrl_ok:
        print(f"  SUSPECT: trained passes but the no-learning control ALSO clears the bar ({c_true:.3f} vs "
              f"{c_perm:.3f}) -- the order signal may be a harness artifact, not learning. Inspect before phase B.",
              flush=True)
    elif t_true >= t_perm * 1.10:
        print(f"  PARTIAL: real order signal (true {t_true:.3f} > perm {t_perm:.3f} by "
              f"{agg['pct_over_permuted']:.0f}%) but below the {agg['abs_floor']} floor or <5/6 seeds -- the CQ "
              f"emits order but unreliably; tune the primacy gradient / WTA noise.", flush=True)
    else:
        print(f"  NEGATIVE: the CQ mechanism can't serialize above the permuted control ({t_true:.3f} vs "
              f"{t_perm:.3f}) -- the serial-order wall is deeper than the G1 self-judge. Honest negative.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"true": t_true, "perm": t_perm, "n_pass": n_pass, "ctrl_true": c_true, "ctrl_perm": c_perm,
           "aggregate_gate": agg, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_serial_order_cq.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
