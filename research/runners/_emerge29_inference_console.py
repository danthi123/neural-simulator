"""EMERGE-29 / toward-language+semantics — the CONVERSATIONAL INFERENCE CONSOLE: the owner TEACHES the emergent spiking
brain an is-a taxonomy + class properties, then ASKS questions whose answers were NEVER told — and the brain INFERS
them by inheritance up the is-a chain, with an honest no-confab moat. It unifies the EMERGE-25 talk-to-and-teach
console with the EMERGE-26/27 emergent Collins-Quillian inheritance, on the real spiking `SimulationBridge`:
  - TEACH is-a:      "a robin is a bird",  "a bird is an animal"   (build the taxonomy)
  - TEACH property:  "a bird can fly",     "an animal breathes"    (attach a property to a category)
  - ASK (inference): "can a robin fly?"    -> "Yes, a robin can fly."          (inherited 1 level up, never told)
                     "does a robin breathe?"-> "Yes, a robin breathes."         (inherited 2 levels up)
                     "can a robin swim?"   -> "I don't know whether a robin can swim."  (honest -- not inherited)
                     "can a zzz fly?"      -> "I don't know what a zzz is."      (moat -- unknown concept)

MECHANISM (emergent, no inference engine): each concept has a CONTENT block; "x is a y" gives x the shared code of y
(x's ancestors' content blocks), so a member's code overlaps its whole taxonomy. "a y can P" potentiates y's content
block -> P via the committed `sim/` three-term kernel. Asking "can x P?" presents x's content + all its ancestor
blocks; the ancestor that owns P primes P through the learned pathway -> x inherits it, though x's own code was never
bound to P. A concept with no taxonomy drives nothing -> the moat abstains. NO `sim/` edit.

`--demo` for a transcript; `--script "a robin is a bird;a bird can fly;can a robin fly?"`; no args = interactive.
CPU numpy-backend; reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, re
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

nE = 8
ACT_TH = 2
FLOOR = -40.0


def _art(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w                    # article agreement (host echo polish)


class SemanticConsole:
    """Teach an is-a taxonomy + class properties; ask inference questions answered by inheritance + a no-confab moat."""

    def __init__(self, seed=42, epochs=80, capacity=64):
        self.epochs = epochs
        self.M = capacity
        self.b, self.ci, self.row, self.col = build_pool_bridge(self.M, nE, seed, act_th=ACT_TH, coincidence=True)
        self.z = np.zeros(self.M * nE)
        self._cols = {}                                                         # name -> content-block cols
        self._next = 0
        self.parent = {}                                                        # concept -> immediate superordinate

    def _alloc(self, name, k=3):
        if name not in self._cols:
            if self._next + k > self.M:
                raise RuntimeError("out of column capacity -- raise capacity")
            self._cols[name] = list(range(self._next, self._next + k)); self._next += k
        return self._cols[name]

    def _sdr(self, cols):
        return set(c * nE + 0 for c in cols)

    def _ancestors(self, x):
        chain, seen = [], set()
        while x is not None and x not in seen:
            seen.add(x); chain.append(x); x = self.parent.get(x)
        return chain

    # ---- teaching -----------------------------------------------------------------------------------------------
    def learn_isa(self, member, category):
        self._alloc(member); self._alloc(category)
        self.parent[member] = category
        return f"ok -- {_art(member)} is {_art(category)}."

    def learn_property(self, concept, prop):
        self._alloc(concept); self._alloc(prop)
        for _ in range(self.epochs):
            apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(self._cols[concept]),
                                self._sdr(self._cols[prop]), self.z, 0.14, 0.02, 1.0)
        return f"ok -- {_art(concept)} can {prop}."

    # ---- inference ----------------------------------------------------------------------------------------------
    def _primed_props(self, concept):
        cols = [c for a in self._ancestors(concept) for c in self._cols[a]]
        ab = np.zeros(len(self.ci), bool)
        for i in self._sdr(cols):
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = _host(getattr(self.b, "cp_v_apical"))[self.ci]
        primed = set()
        for name, ccols in self._cols.items():
            if name in self._ancestors(concept):
                continue
            if float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in ccols])) > FLOOR:
                primed.add(name)
        return primed

    def does(self, concept, prop):
        """Answer 'can a <concept> <prop>?' by inheritance, honestly abstaining on the unknown."""
        if concept not in self._cols:
            return f"I don't know what {_art(concept)} is."
        if prop in self._primed_props(concept):
            return f"Yes, {_art(concept)} can {prop}."
        return f"I don't know whether {_art(concept)} can {prop}."

    # ---- transitive ordering (the other inference form) --------------------------------------------------------
    def learn_beats(self, x, y):
        """An ordering premise 'x beats y' -> x's code -> y's code (the same coincidence primitive)."""
        self._alloc(x); self._alloc(y)
        for _ in range(self.epochs):
            apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(self._cols[x]),
                                self._sdr(self._cols[y]), self.z, 0.14, 0.02, 1.0)
        return f"ok -- {x} beats {y}."

    def _reachable(self, start, depth=8):
        reached, active = set(), self._sdr(self._cols[start])
        for _ in range(depth):
            ab = np.zeros(len(self.ci), bool)
            for i in active:
                ab[i] = True
            _prime_from_winners(self.b, self.ci, ab)
            vap = _host(getattr(self.b, "cp_v_apical"))[self.ci]
            nxt = None
            for nm, cols in self._cols.items():
                if nm in reached or nm == start:
                    continue
                dr = float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in cols]))
                if dr > FLOOR and (nxt is None or dr > nxt[1]):
                    nxt = (nm, dr)
            if nxt is None:
                break
            reached.add(nxt[0]); active = self._sdr(self._cols[nxt[0]])
        return reached

    def beats(self, x, z):
        """Answer 'does x beat z?' by transitive chaining over the learned ordering, honestly abstaining."""
        if x not in self._cols or z not in self._cols:
            return f"I don't know who {x if x not in self._cols else z} is."
        if z in self._reachable(x) and x not in self._reachable(z):
            return f"Yes, {x} beats {z}."
        if x in self._reachable(z):
            return f"No, {z} beats {x}."
        return f"I don't know whether {x} beats {z}."


# ---- a tiny natural-language front end (host parsing = the world/keyboard interface) -----------------------------
_ISA = re.compile(r"(?:a|an)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)", re.I)
_PROP = re.compile(r"(?:a|an)\s+(\w+)\s+can\s+(\w+)", re.I)
_ASK = re.compile(r"can\s+(?:a|an)\s+(\w+)\s+(\w+)\??", re.I)
_ASKBEAT = re.compile(r"does\s+(\w+)\s+beat\s+(\w+)\??", re.I)                   # transitive query
_BEATS = re.compile(r"(\w+)\s+beats\s+(\w+)", re.I)                             # ordering premise


def handle(console, line):
    line = line.strip()
    if not line:
        return None
    m = _ASKBEAT.search(line)                                                  # match queries before premises
    if m:
        return console.beats(m.group(1).lower(), m.group(2).lower())
    m = _ASK.search(line)
    if m:
        return console.does(m.group(1).lower(), m.group(2).lower())
    m = _ISA.search(line)
    if m:
        return console.learn_isa(m.group(1).lower(), m.group(2).lower())
    m = _PROP.search(line)
    if m:
        return console.learn_property(m.group(1).lower(), m.group(2).lower())
    m = _BEATS.search(line)
    if m:
        return console.learn_beats(m.group(1).lower(), m.group(2).lower())
    return "(say 'a X is a Y', 'a Y can P', 'can a X P?', 'X beats Y', or 'does X beat Z?')"


def _demo(seed=42, epochs=80):
    c = SemanticConsole(seed=seed, epochs=epochs)
    print("\n=== EMERGE-29 conversational inference console (teach a taxonomy, ask inferred questions; no transformer) ===\n")
    for line in ["a robin is a bird", "a bird is an animal", "a trout is a fish", "a fish is an animal",
                 "a bird can fly", "an animal can breathe", "a fish can swim"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- ASK by INHERITANCE (answers were NEVER told; inferred up the is-a chain) ---")
    for line in ["can a robin fly?", "can a robin breathe?", "can a trout swim?", "can a trout breathe?",
                 "can a robin swim?", "can a zzz fly?"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- teach an ORDERING, ASK by TRANSITIVE inference (non-adjacent never told) ---")
    for line in ["alice beats bob", "bob beats carol", "carol beats dave"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    for line in ["does alice beat dave?", "does dave beat alice?", "does bob beat dave?", "does alice beat zoe?"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--script", default=None)
    a = ap.parse_args()
    if a.demo:
        _demo(a.seed, a.epochs); return 0
    c = SemanticConsole(seed=a.seed, epochs=a.epochs)
    print("inference console -- teach: 'a X is a Y' / 'a Y can P'; ask: 'can a X P?'  (Ctrl-D to exit)")
    if a.script:
        for line in a.script.split(";"):
            r = handle(c, line)
            if r is not None:
                print(f"  you> {line.strip()}\n  brain> {r}")
        return 0
    try:
        while True:
            r = handle(c, input("you> "))
            if r is not None:
                print(f"brain> {r}")
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
