"""EMERGE-31 / the CAPSTONE console — LEARN CATEGORIES FROM EXPERIENCE, then CONVERSE + INFER: the owner does not TELL
the brain a taxonomy (EMERGE-29 did that); instead the owner OBSERVES members co-occurring with contexts, the brain
DISCOVERS the category grouping unsupervised (EMERGE-30), and then a property taught via ONE member is INHERITED by a
DIFFERENT member the brain only ever OBSERVED in the same contexts — the full observe → learn → infer → converse
loop, on the real spiking `SimulationBridge`, NO `sim/` edit.

  you> a robin lives-with a nest        (OBSERVE co-occurrence -- no category named)
  you> a sparrow lives-with a nest
  you> a robin lives-with a treetop
  you> a sparrow lives-with a treetop
  you> a robin can fly                  (TEACH a property via ONE member)
  you> can a sparrow fly?               brain> Yes, a sparrow can fly.      (INFERRED -- never told; sparrow was only
                                                                             OBSERVED in the same contexts as robin)
  you> can a shark fly?                 brain> I don't know what a shark is.  (moat)

MECHANISM (emergent, no inference engine, no transformer): "X lives-with Y" learns X-content -> Y-context (on-bridge
Hebbian co-occurrence, the committed `sim/` three-term kernel). Members sharing contexts learn to activate the SAME
context cells -> the emergent category. "a X can P" is taught by presenting X, priming its learned contexts, and
binding P to (X + its contexts) -> the property attaches to the shared context. Asking "can a Y P?" reads P directly
(if Y was the taught member) or via the shared context (Y -> emergent context -> P) -> a co-observed member inherits.
A member never observed drives no context -> the moat abstains.

`--demo` / `--script "a robin lives-with a nest;...;can a sparrow fly?"` / interactive. CPU numpy-backend; reuse-by-
import (`_emerge14` + `_emerge12`); NO `sim/` edit.
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
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


class ExperientialConsole:
    """Learn categories from observed co-occurrence; teach a property via one member; a co-observed member infers it."""

    def __init__(self, seed=42, epochs=60, capacity=96):
        self.epochs = epochs
        self.M = capacity
        self.b, self.ci, self.row, self.col = build_pool_bridge(self.M, nE, seed, act_th=ACT_TH, coincidence=True)
        self.z = np.zeros(self.M * nE)
        self._cols = {}
        self._next = 0
        self._members = set()
        self._props = set()

    def _alloc(self, name, k=3):
        if name not in self._cols:
            if self._next + k > self.M:
                raise RuntimeError("out of column capacity")
            self._cols[name] = list(range(self._next, self._next + k)); self._next += k
        return self._cols[name]

    def _sdr(self, cols):
        return set(c * nE + 0 for c in cols)

    def _prime(self, cells):
        ab = np.zeros(len(self.ci), bool)
        for i in cells:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        return _host(getattr(self.b, "cp_v_apical"))[self.ci]

    def _primed_names(self, vap, exclude=()):
        out = set()
        for nm, cols in self._cols.items():
            if nm in exclude:
                continue
            if float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in cols])) > FLOOR:
                out.add(nm)
        return out

    # ---- teaching from experience -------------------------------------------------------------------------------
    def observe(self, member, context):
        """'a member lives-with a context' -> learn member-content -> context (co-occurrence)."""
        self._alloc(member); self._alloc(context); self._members.add(member)
        for _ in range(self.epochs):
            apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(self._cols[member]),
                                self._sdr(self._cols[context]), self.z, 0.14, 0.02, 1.0)
        return f"ok -- I've seen {_art(member)} with {_art(context)}."

    def learn_can(self, member, prop):
        """'a member can P' -> bind P to (member + its emergent contexts) so co-observed members inherit."""
        self._alloc(member); self._alloc(prop); self._members.add(member); self._props.add(prop)
        for _ in range(self.epochs):
            v = self._prime(self._sdr(self._cols[member]))
            ctx = set(int(i) for i in np.where(v > FLOOR)[0])                  # the member's emergent contexts
            pre = self._sdr(self._cols[member]) | ctx
            apply_kernel_update(self.b, self.row, self.col, self.ci, pre, self._sdr(self._cols[prop]),
                                self.z, 0.14, 0.02, 1.0)
        return f"ok -- {_art(member)} can {prop}."

    # ---- inference ----------------------------------------------------------------------------------------------
    def ask_can(self, member, prop):
        if member not in self._cols:
            return f"I don't know what {_art(member)} is."
        v1 = self._prime(self._sdr(self._cols[member]))
        if prop in self._primed_names(v1):                                    # taught directly (the exemplar)
            return f"Yes, {_art(member)} can {prop}."
        ctx = set(int(i) for i in np.where(v1 > FLOOR)[0])                    # emergent context cells
        if ctx and prop in self._primed_names(self._prime(ctx)):             # inherited via the shared context
            return f"Yes, {_art(member)} can {prop}."
        return f"I don't know whether {_art(member)} can {prop}."


_OBS = re.compile(r"(?:a|an)\s+(\w+)\s+lives-with\s+(?:a|an)\s+(\w+)", re.I)
_CAN = re.compile(r"(?:a|an)\s+(\w+)\s+can\s+(\w+)", re.I)
_ASK = re.compile(r"can\s+(?:a|an)\s+(\w+)\s+(\w+)\??", re.I)


def handle(console, line):
    line = line.strip()
    if not line:
        return None
    m = _ASK.search(line)
    if m:
        return console.ask_can(m.group(1).lower(), m.group(2).lower())
    m = _OBS.search(line)
    if m:
        return console.observe(m.group(1).lower(), m.group(2).lower())
    m = _CAN.search(line)
    if m:
        return console.learn_can(m.group(1).lower(), m.group(2).lower())
    return "(say 'a X lives-with a Y', 'a X can P', or 'can a X P?')"


def _demo(seed=42, epochs=60):
    c = ExperientialConsole(seed=seed, epochs=epochs)
    print("\n=== EMERGE-31 experiential console -- learn categories from EXPERIENCE, then infer (no transformer) ===\n")
    obs = ["a robin lives-with a nest", "a sparrow lives-with a nest", "a robin lives-with a treetop",
           "a sparrow lives-with a treetop", "a trout lives-with a river", "a pike lives-with a river",
           "a trout lives-with a reef", "a pike lives-with a reef"]
    for line in obs:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- teach a property via ONE member of each emergent group ---")
    for line in ["a robin can fly", "a trout can swim"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- ASK about the co-observed members (never told; inferred via the DISCOVERED grouping) ---")
    for line in ["can a robin fly?", "can a sparrow fly?", "can a pike swim?", "can a sparrow swim?", "can a shark fly?"]:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--script", default=None)
    a = ap.parse_args()
    if a.demo:
        _demo(a.seed, a.epochs); return 0
    c = ExperientialConsole(seed=a.seed, epochs=a.epochs)
    print("experiential console -- observe: 'a X lives-with a Y'; teach: 'a X can P'; ask: 'can a X P?'  (Ctrl-D to exit)")
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
