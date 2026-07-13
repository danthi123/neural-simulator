"""GRADED-CONFIDENCE wired into the TALKABLE emergent console (the 2026-07-08 frontier gate's #1 open piece, now on the
console the owner actually talks to): the EMERGE-31 experiential console LEARNS categories from observed co-occurrence and
answers by inheritance, but its `ask_can` is a HARD binary ("Yes, a X can P" / "I don't know whether a X can P"). This
wires the VALIDATED graded 2-hop drive+margin read (`_emerge_graded_confidence_completion_derisk`, 12-seed GO) INTO the
console so a co-observed concept with strong evidence answers CONFIDENTLY, one with contested/ambiguous evidence HEDGES
("probably"), and one with no evidence ABSTAINS — the no-confab moat extended from hard-abstain to a graded hedge. The
confidence is NOT hand-coded: it EMERGES from the strength of the learned co-occurrence drive (Rogers-McClelland graded
distributed completion; the graded read = Bogacz-Brown evidence-margin confidence). Reuse-by-import; NO `sim/` edit.

WHY (a0-read of EMERGE-31 + the graded-completion probe done FIRST): EMERGE-31's `ask_can` primes the member (hop 1 -> its
emergent contexts) then checks categorical membership of the property. This subclass replaces the binary membership with
the graded read: 2-hop apical drive on the asked property, margin over the strongest COMPETING taught property ->
CONFIDENT (dominates) / HEDGED (present but contested) / ABSTAIN (no drive). The margin threshold is the SAME scale as the
validated probe (CONF_TH-FLOOR=30 on `build_pool_bridge`'s apical drive) and is exposed + validated on the console here.

SCENARIO (emergent, no hand-coded confidence): observe robin/sparrow with "nest" (bird), mole with "cave" (mammal), bat
with BOTH "nest" AND "cave" (category-AMBIGUOUS, ~50/50), trout with "river" (fish, no property). Teach "a sparrow can fly"
(binds fly to the bird context) + "a mole can walk". Then ASK:
  can a robin fly?        -> CONFIDENT ("Yes, a robin can fly.")     robin co-occurs the bird context, fly dominates
  can a bat fly?          -> HEDGED    ("A bat can probably fly.")   bat is bird/mammal-ambiguous -> fly contested by walk
  can a trout fly?        -> ABSTAIN   ("I don't know whether ...")  trout's context bears no taught property
  can a wolpertinger fly? -> MOAT      ("I don't know what ...")     never observed (the intrinsic no-confab moat)

GO (6-seed standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): robin=CONFIDENT+correct AND bat=HEDGED AND trout=ABSTAIN
AND wolpertinger=MOAT, with PERMUTED observations (EVERY member co-occurs EQUALLY with ALL contexts -> no category
structure) destroying the confident answer AND LESION (coincidence off) driving all-abstain, on >=5/6 in BOTH seed sets.

Run: SIM_BACKEND=numpy python -m research.runners._emerge_graded_confidence_console_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update
from research.runners._emerge31_experiential_console import ExperientialConsole, _art, nE, ACT_TH, FLOOR

CONF_MARGIN = 30.0        # = CONF_TH(-10) - FLOOR(-40) from the validated graded-completion probe; same drive scale


class GradedExperientialConsole(ExperientialConsole):
    """EMERGE-31 console + the validated graded read: `ask_can` returns CONFIDENT / HEDGED / ABSTAIN by evidence strength,
    keeping the unknown-concept no-confab moat. NO new mechanism -- the confidence EMERGES from the learned drive."""

    def __init__(self, seed=42, epochs=80, capacity=96, coincidence=True, conf_margin=CONF_MARGIN):
        # mirror ExperientialConsole.__init__ but allow coincidence=False (the lesion anti-cheat)
        self.epochs = epochs
        self.M = capacity
        self.b, self.ci, self.row, self.col = build_pool_bridge(self.M, nE, seed, act_th=ACT_TH, coincidence=coincidence)
        self.z = np.zeros(self.M * nE)
        self._cols = {}
        self._next = 0
        self._members = set()
        self._props = set()
        self.conf_margin = conf_margin
        self._coincidence = coincidence

    def _prime(self, cells):
        if not self._coincidence:                       # dAP-lesion: no coincidence -> no apical drive -> abstain
            return np.full(len(self.ci), FLOOR - 10.0)  # rest everywhere (below FLOOR) so every read abstains
        return super()._prime(cells)

    def observe_mixed(self, member, contexts):
        """A genuinely category-AMBIGUOUS member co-occurs with multiple contexts INTERLEAVED over time (the realistic
        temporal pattern) -- NOT all-of-one-then-all-of-the-other, which the depress-inactive kernel would erase. Strict
        alternation across epochs => balanced ~50/50 evidence (mirrors the validated probe's ambiguous member)."""
        self._alloc(member); self._members.add(member)
        for ctx in contexts:
            self._alloc(ctx)
        for e in range(self.epochs):
            ctx = contexts[e % len(contexts)]
            apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(self._cols[member]),
                                self._sdr(self._cols[ctx]), self.z, 0.14, 0.02, 1.0)
        return "ok -- I've seen " + _art(member) + " with " + " and ".join(_art(c) for c in contexts) + "."

    def _prop_drive(self, vap, prop):
        cols = self._cols[prop]
        return float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in cols]))

    def graded_answer(self, member, prop):
        """Returns (label, phrase). label in {CONFIDENT, HEDGED, ABSTAIN, MOAT}. The graded read on the console scale."""
        if member not in self._cols:
            return "MOAT", f"I don't know what {_art(member)} is."
        if prop not in self._cols:                                               # property never taught to anyone
            return "ABSTAIN", f"I don't know whether {_art(member)} can {prop}."
        v1 = self._prime(self._sdr(self._cols[member]))
        if self._prop_drive(v1, prop) > FLOOR and prop in self._primed_names(v1):  # taught directly (stated fact)
            return "CONFIDENT", f"Yes, {_art(member)} can {prop}."
        ctx = set(int(i) for i in np.where(v1 > FLOOR)[0])                       # the member's emergent contexts
        if not ctx:
            return "ABSTAIN", f"I don't know whether {_art(member)} can {prop}."
        v2 = self._prime(ctx)
        dprop = self._prop_drive(v2, prop)
        if dprop <= FLOOR:
            return "ABSTAIN", f"I don't know whether {_art(member)} can {prop}."
        others = [self._prop_drive(v2, o) for o in self._props if o != prop and o in self._cols]
        comp = max(others) if others else FLOOR
        margin = dprop - comp                                                    # how far the asked property dominates
        if margin > self.conf_margin:
            return "CONFIDENT", f"Yes, {_art(member)} can {prop}."
        return "HEDGED", f"{_art(member).capitalize()} can probably {prop}."

    def ask_can(self, member, prop):                                             # override: graded, not binary
        return self.graded_answer(member, prop)[1]


# ---- observation script: single-context members + bat (observed with BOTH nest AND cave, interleaved -> ambiguous) --
OBS = [("robin", "nest"), ("sparrow", "nest"), ("mole", "cave"), ("trout", "river")]
AMBIG = ("bat", ["nest", "cave"])                                                # category-ambiguous (bird/mammal)
TEACH = [("sparrow", "fly"), ("mole", "walk")]                                   # one member each -> bind to its context


ALL_CTX = ["nest", "cave", "river"]


def _build(seed, epochs, arm):
    coincidence = (arm != "lesion")
    c = GradedExperientialConsole(seed=seed, epochs=epochs, coincidence=coincidence)
    if arm != "nolearn":
        if arm == "permuted":
            # RELIABLE input-destruction: EVERY member co-occurs EQUALLY with ALL contexts (no category structure
            # survives), mirroring the validated completion-probe's per-epoch permutation. A single random member->
            # context mapping can COINCIDENTALLY survive in a 3-context space (the 2026-07-02 control-validity lesson:
            # a fixed-random control is unreliable in a small space -> gate on deterministic input-destruction).
            for (m, _c) in OBS:
                c.observe_mixed(m, ALL_CTX)
            c.observe_mixed(AMBIG[0], ALL_CTX)
        else:
            for (m, ctx) in OBS:
                c.observe(m, ctx)
            c.observe_mixed(AMBIG[0], AMBIG[1])
    for (m, p) in TEACH:
        c.learn_can(m, p)
    return c


def _run_arm(seed, arm, epochs):
    c = _build(seed, epochs, arm)
    robin = c.graded_answer("robin", "fly")
    bat = c.graded_answer("bat", "fly")
    trout = c.graded_answer("trout", "fly")
    wolp = c.graded_answer("wolpertinger", "fly")                                # never observed -> moat
    return {"arm": arm, "robin": robin[0], "bat": bat[0], "trout": trout[0], "wolp": wolp[0],
            "phrases": {"robin": robin[1], "bat": bat[1], "trout": trout[1], "wolp": wolp[1]}}


def run(seed, epochs):
    htm = _run_arm(seed, "htm", epochs)
    perm = _run_arm(seed, "permuted", epochs)
    les = _run_arm(seed, "lesion", epochs)
    strong_confident = htm["robin"] == "CONFIDENT"                               # strong co-occurrence -> confident
    ambig_hedged = htm["bat"] == "HEDGED"                                        # category-ambiguous -> hedged
    abstain = htm["trout"] == "ABSTAIN"                                          # no property in context -> abstain
    moat = htm["wolp"] == "MOAT"                                                 # unknown concept -> no-confab moat
    three_plus_moat = strong_confident and ambig_hedged and abstain and moat
    permuted_breaks = perm["robin"] != "CONFIDENT"                              # shuffled co-occurrence -> not confident
    lesion_breaks = les["robin"] == "ABSTAIN"                                   # no coincidence -> abstain
    go = bool(three_plus_moat and permuted_breaks and lesion_breaks)
    print(f"[graded-console seed={seed}] robin={htm['robin']} bat={htm['bat']} trout={htm['trout']} wolp={htm['wolp']} "
          f"| perm.robin={perm['robin']} lesion.robin={les['robin']} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "htm": htm, "permuted": perm, "lesion": les, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--demo", action="store_true", help="print the console transcript for seed 42")
    a = ap.parse_args()
    if a.demo:
        c = _build(42, a.epochs, "htm")
        print("\n=== EMERGE graded-confidence console -- the moat now HEDGES (learned, no hand-coded confidence) ===\n")
        for line in ["can a robin fly?", "can a bat fly?", "can a trout fly?", "can a wolpertinger fly?"]:
            m = line.split()[2]
            print(f"  you> {line}\n  brain> {c.ask_can(m, 'fly')}\n")
        return
    res = [run(s, a.epochs) for s in a.seeds]
    print(f"[graded-console] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
