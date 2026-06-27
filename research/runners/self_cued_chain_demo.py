"""Tier 2.2 console probe — SELF-CUED associative chain-of-thought ("starting from X, what follows?").

The agent is given a START concept and NOTHING ELSE -- no caller-supplied hop/relation list. It SELECTS each next
relation to chase by LEARNED association strength over its OWN stored facts, then chases it via the validated
single hop (RFPhasorComposer.query_patient), re-cleaning between hops. It emits the self-generated chain, or
ABSTAINS honestly at a dead end / an unknown start (the no-confab moat at every hop). This is the difference 2.2
adds over the production reason_chain (which takes the relation list from the caller): the agent THINKS the chain.

De-risked GO numpy 3 seeds x 3 D (self-cued 2-hop 1.00 vs spreading floor 0.08; lesion-the-association/permuted/
re-cue all collapse; moat at every hop; no compounding to 4 hops) -- 2026-06-27-tier2.2-chain-of-thought-GO.md.

Run (scripted demo, CPU):
  SIM_BACKEND=numpy python -m research.runners.self_cued_chain_demo
Interactive REPL ("> dog" -> the brain's self-cued chain):
  SIM_BACKEND=numpy python -m research.runners.self_cued_chain_demo --repl
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer

# A small relational world. Two food-chains (followed by the `eat` relation, reinforced so the agent PREFERS it),
# plus distractor `see` facts (a different relation, weaker) so the chain is genuine relational following, not
# co-occurrence smearing. Dead-end concepts (the chain tails) abstain rather than fabricate.
CHAINS = [
    ["dog", "cat", "mouse", "bug", "leaf"],
    ["lion", "deer", "grass", "seed", "soil"],
]
DISTRACTOR = "ball"
EAT, SEE = "eat", "see"


def build_brain(seed=42, D=128):
    vocab = sorted({EAT, SEE, DISTRACTOR} | {c for ch in CHAINS for c in ch})
    c = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    for ch in CHAINS:
        for a, p in zip(ch[:-1], ch[1:]):
            for _ in range(3):                 # reinforce the chain relation -> the selector PICKS it
                c.store(a, EAT, p)
            c.store(a, SEE, DISTRACTOR)         # a weaker distractor relation
    return c


def think(composer, start, max_hops=5):
    """Return a human-readable line for the self-cued chain from `start`, or an honest abstain."""
    if start not in composer.words:
        return f"  I don't know '{start}'."                          # unknown concept -> honest "I don't know"
    term, path = composer.chain_of_thought(start, max_hops=max_hops, return_path=True)
    if len(path) == 1:
        return f"  starting from '{start}': (nothing follows -- I have no association to chase from there)"
    return f"  starting from '{start}': " + " -> ".join(path) + f"   [reached '{term}' in {len(path)-1} self-cued hops]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--repl", action="store_true", help="interactive: type a start concept, see the self-cued chain")
    a = ap.parse_args()

    print("[Tier 2.2 self-cued chain-of-thought] building the brain (a small relational world) ...", flush=True)
    brain = build_brain(seed=a.seed, D=a.dim)
    print(f"  known concepts: {', '.join(brain.words)}\n"
          "  the agent SELECTS each next hop by its own learned association (no plan supplied); the moat abstains "
          "at a dead end / unknown.\n", flush=True)

    if a.repl:
        print("Type a start concept (e.g. 'dog' or 'lion'); 'quit' to exit.", flush=True)
        while True:
            try:
                s = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if s.lower() in ("quit", "exit", ":q"):
                break
            if s:
                print(think(brain, s), flush=True)
        return

    # scripted demo: each chain head -> a self-generated chain; an unknown + a dead-end -> honest abstain.
    print("=== self-generated chains (hops chosen by the agent, not supplied) ===", flush=True)
    for ch in CHAINS:
        print(think(brain, ch[0]), flush=True)
    print("\n=== honest abstention (the no-confab moat) ===", flush=True)
    print(think(brain, "leaf"), flush=True)        # a chain TAIL: nothing follows -> abstain (no fabricated hop)
    print(think(brain, "unicorn"), flush=True)     # an unknown concept -> "I don't know"
    print(think(brain, DISTRACTOR), flush=True)    # ball is only ever a PATIENT (never an agent) -> nothing follows
    print("\n(Per the 2026-05-14 retraction precedent, the chain is genuine self-cued relational following: "
          "lesioning the learned association collapses it to chance -- see the de-risk + CI guard.)", flush=True)


if __name__ == "__main__":
    main()
