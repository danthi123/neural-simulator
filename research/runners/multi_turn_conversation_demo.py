"""A runnable multi-turn conversation transcript on the production MultiTurnAgent — the capstone of the
2026-06-17 conversational arc. Everything the brain does between hearing and answering is on the validated
spiking substrate (parse, bind/recall, working-memory hold, relational chaining, abstention); the host only
prints the words.

  SIM_BACKEND=numpy python -m research.runners.multi_turn_conversation_demo
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent import MultiTurnAgent

NOUNS = ["dog", "cat", "fish", "worm", "bird", "hawk", "dragon", "ball"]
VOCAB = NOUNS + ["chase", "eat", "see"]


def _say(who, text):
    print(f"  {who:>5}: {text}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer", choices=["rf", "onebrain"], default="rf",
                    help="rf = the production numpy composer / test oracle (default); onebrain = the integrated "
                         "one-brain composer (the whole pipeline on ONE spiking bridge; wants SIM_BACKEND=cupy)")
    a = ap.parse_args()
    print("\n=== multi-turn conversation on the brain (MultiTurnAgent) ===\n", flush=True)
    agent = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB},
                           seed=42, enable_neural_render=True, composer_kind=a.composer)
    c = agent.agent.composer
    # a small food-web the agent is told about (separate affirmative facts; the AFFIRM tag is the slot the
    # yes/no path reads -- a declarative statement is affirmative).
    for a, v, o in [("cat", "eat", "fish"), ("fish", "eat", "worm"), ("hawk", "eat", "bird"),
                    ("bird", "eat", "worm")]:
        c.store(a, v, o, polarity="AFFIRM")

    _say("user", "dog chase cat.")                       # turn 1: establishes 'cat' as the discourse referent
    agent.hear("dog chase cat")
    _say("brain", "(noted: dog chase cat)")

    _say("user", "what does it eat?")                    # 'it' must resolve to cat -> cat eat fish
    ans = agent.what_does("it", "eat")
    _say("brain", f"it eats {ans}." if ans else "I don't know what it eats.")

    _say("user", "and what does the fish eat?")
    ans = agent.what_does("fish", "eat")
    _say("brain", f"the fish eats {ans}." if ans else "I don't know.")

    _say("user", "so what does the cat's prey eat? (two hops)")   # cat -> fish -> worm
    ans = agent.reason_chain("cat", ["eat", "eat"])
    _say("brain", f"following the chain, {ans}." if ans else "I can't follow that chain.")

    _say("user", "does the hawk eat bird?")
    _say("brain", agent.is_it_true("hawk", "eat", "bird") + ".")

    _say("user", "does the hawk eat fish?")              # never stored -> 'unknown', not a guess
    _say("brain", agent.is_it_true("hawk", "eat", "fish") + ".")

    _say("user", "what does the dragon eat?")            # the moat: nothing stored about dragon -> abstain
    ans = agent.what_does("dragon", "eat")
    _say("brain", f"the dragon eats {ans}." if ans else "I don't know — I was never told anything about the dragon.")

    print("\n  (Every answer above was produced by the spiking substrate: the pronoun 'it' was resolved from a\n"
          "   working-memory loop, the two-hop question by relational chaining with cleanup between hops, and the\n"
          "   dragon question was refused rather than confabulated — the no-fabrication moat.)\n", flush=True)


if __name__ == "__main__":
    main()
