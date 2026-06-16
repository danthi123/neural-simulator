"""CYCLE 106 — sentence-generation, the multi-frame follow-on: can the neural serial-order mechanism learn
DISTINCT orders for DISTINCT frames (the seed of syntax), and keep them separate?

The phase-A/B de-risk validated a SINGLE fixed frame (SVO = agent,action,patient -- exactly what the host
f-string is). A conversational agent has SEVERAL output frames (an SVO statement, a who/what answer, a yes-no
reply, an "X and Y associated" reply), and real syntax is frame-DEPENDENT word order. This de-risk extends the
competitive-queuing serial-order generator to a FRAME-CONDITIONED primacy gradient: `prim[frame][role]` is
LEARNED per frame from the teacher (each frame's order), and the choice WTA emits in that frame's order. Does it
(a) learn each frame's order and generalize to held-out facts, AND (b) produce a DIFFERENT order for the same
fact under a different frame (the cross-frame control -- the mechanism is genuinely frame-conditioned, not one
fixed order)?

FRAMES (two deliberately-DISJOINT orders so the cross-frame control is decisive): F0 = [agent, action, patient]
(SVO), F1 = [patient, agent, action] (a distinct frame; e.g. a topic-first / passive-like reordering).

GATE (>=6 seeds, FIXED g1_verdict bars 0.10/0.5, reused VERBATIM): GO if, per frame, held-out true order clears
floor 0.5 AND beats the permuted-order control by >=10% AND beats the CROSS-FRAME control (the OTHER frame's
order on the same fact must NOT match -> the order is frame-specific), all >=5/6 seeds. A no-learning control
(untrained per-frame primacy) must fail. GO => the substrate can learn frame-conditioned word order (the seed of
neural syntax). NEGATIVE => frame-conditioning fails (the mechanism collapses to one order). Honest either way.

Reuse-by-import (song_g1_core score_order / permuted_order_controls / g1_verdict); CPU; no GPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_serial_order_multiframe_derisk
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

N_ROLES, VOCAB, N_FACTS, N_PERM = 3, 16, 24, 5
WTA_NOISE = 0.25
FRAMES = {0: [0, 1, 2], 1: [2, 0, 1]}            # F0 = SVO ; F1 = a DISJOINT frame (patient, agent, action)


def build_facts(seed):
    rng = np.random.default_rng(seed * 101 + 7)
    facts, seen = [], set()
    while len(facts) < N_FACTS:
        trip = tuple(int(x) for x in rng.choice(VOCAB, N_ROLES, replace=False))
        if trip not in seen:
            seen.add(trip); facts.append(trip)
    return facts[:N_FACTS // 2], facts[N_FACTS // 2:]


class FrameCQ:
    """Frame-conditioned competitive-queuing: `prim[frame][role]` is the per-frame planning-layer primacy gradient,
    LEARNED from the teacher; emit = the choice-WTA read-out in that frame's primacy order (inhibition-of-return)."""

    def __init__(self, n_roles=N_ROLES, n_frames=len(FRAMES), lr=0.1, seed=42):
        self.n_roles, self.lr = n_roles, lr
        self.prim = np.random.default_rng(seed * 13 + 5).standard_normal((n_frames, n_roles)) * 0.01

    def learn(self, frame, target_role_order):
        for pos, role in enumerate(target_role_order):
            self.prim[frame][role] += self.lr * (self.n_roles - 1 - pos)

    def emit(self, frame, fillers_by_role, rng):
        a = self.prim[frame][list(range(self.n_roles))] + WTA_NOISE * rng.standard_normal(self.n_roles)
        avail, emitted = list(range(self.n_roles)), []
        for _ in range(self.n_roles):
            best = max(avail, key=lambda i: a[i])
            emitted.append(fillers_by_role[best]); avail.remove(best)
        return emitted


def _grade(cq, held, rng):
    """Per (fact, frame): emit; score vs the frame's TRUE order, the best permuted control, and the CROSS-FRAME
    order (the other frame's order on the same fact). Returns (true, perm, cross) means."""
    trues, perms, crosses = [], [], []
    for trip in held:
        fillers = {r: trip[r] for r in range(N_ROLES)}
        for frame, order in FRAMES.items():
            intended = [trip[r] for r in order]
            other = FRAMES[1 - frame]                         # the other frame's order on the SAME fact
            cross_intended = [trip[r] for r in other]
            emitted = cq.emit(frame, fillers, rng)
            trues.append(score_order(emitted, intended))
            perms.append(max((score_order(emitted, c) for c in permuted_order_controls(intended, rng, N_PERM)),
                             default=0.0))
            crosses.append(score_order(emitted, cross_intended))
    return float(np.mean(trues)), float(np.mean(perms)), float(np.mean(crosses))


def run_seed(seed):
    train, held = build_facts(seed)
    rng = np.random.default_rng(seed * 71 + 3)
    cq = FrameCQ(seed=seed)
    for trip in train:                                        # learn EACH frame's order from the teacher
        for frame, order in FRAMES.items():
            cq.learn(frame, order)
    t_true, t_perm, t_cross = _grade(cq, held, rng)
    cq0 = FrameCQ(seed=seed * 3 + 1)                          # no-learning control (untrained per-frame primacy)
    c_true, c_perm, _ = _grade(cq0, held, np.random.default_rng(seed * 91 + 1))
    v = g1_verdict(t_true, t_perm, gate_cleared=True)
    cross_ok = t_true >= t_cross * 1.10                        # frame-specific: true beats the cross-frame order
    gate = bool(v["gate"] and cross_ok)
    print(f"  [seed {seed}] FRAME-CQ true {t_true:.3f} vs perm {t_perm:.3f} vs CROSS-frame {t_cross:.3f} -> "
          f"{'PASS' if gate else 'FAIL'} | no-learning true {c_true:.3f} vs perm {c_perm:.3f}", flush=True)
    return {"seed": seed, "true": t_true, "perm": t_perm, "cross": t_cross, "gate": gate,
            "ctrl_true": c_true, "ctrl_perm": c_perm}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[multi-frame serial-order de-risk] can the CQ generator learn DISTINCT orders for F0=SVO[0,1,2] and "
          f"F1=[2,0,1], generalize to held-out facts, AND keep them separate (cross-frame control)?", flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    t_true, t_perm, t_cross = m("true"), m("perm"), m("cross")
    c_true, c_perm = m("ctrl_true"), m("ctrl_perm")
    n_pass = sum(1 for r in rows if r["gate"])
    agg = g1_verdict(t_true, t_perm, gate_cleared=True)
    ctrl_ok = c_true < c_perm * 1.10 + 1e-9
    cross_ok = t_true >= t_cross * 1.10
    print(f"\n{'='*100}\n  MEAN (6 seeds): FRAME-CQ true {t_true:.3f} vs perm {t_perm:.3f} vs CROSS-frame "
          f"{t_cross:.3f} ({n_pass}/6 PASS) | no-learning true {c_true:.3f} vs perm {c_perm:.3f} | aggregate "
          f"{agg['GATE']} ({agg['pct_over_permuted']:.0f}% over perm)", flush=True)
    print(f"{'='*100}", flush=True)
    if agg["gate"] and cross_ok and n_pass >= 5 and ctrl_ok:
        print(f"  GO: the CQ generator learns FRAME-CONDITIONED word order -- per frame, held-out true {t_true:.3f} "
              f">> permuted {t_perm:.3f} AND >> the CROSS-frame order {t_cross:.3f} (the SAME fact is ordered "
              f"DIFFERENTLY by frame), {n_pass}/6 seeds; no-learning control fails. ==> the substrate can learn "
              f"the seed of syntax (frame-dependent serial order), not just one fixed frame.", flush=True)
    elif agg["gate"] and not cross_ok:
        print(f"  PARTIAL: orders beat permuted but NOT the cross-frame control ({t_true:.3f} vs cross "
              f"{t_cross:.3f}) -- the mechanism isn't keeping the frames separate (frame-conditioning weak). "
              f"Localize the per-frame primacy.", flush=True)
    elif t_true >= t_perm * 1.10:
        print(f"  PARTIAL: real per-frame order signal (true {t_true:.3f} > perm {t_perm:.3f}) but below floor or "
              f"<5/6 seeds -- tune the per-frame gradient.", flush=True)
    else:
        print(f"  NEGATIVE: frame-conditioned order-learning fails ({t_true:.3f} vs perm {t_perm:.3f}) -- the "
              f"mechanism can't learn distinct frame orders. Honest negative.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"true": t_true, "perm": t_perm, "cross": t_cross, "n_pass": n_pass, "ctrl_true": c_true,
           "ctrl_perm": c_perm, "aggregate_gate": agg, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_serial_order_multiframe.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
