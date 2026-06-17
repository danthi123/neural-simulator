"""A runnable transcript on MultiTurnAgentV2 -- multi-referent disambiguation with an ORDER-ENCODED discourse
buffer on the spiking resonate-and-fire phasor substrate. This is the production version of the CYCLE-135 de-risk
(2026-06-17-ordered-wm-position-binding-derisk.md): a turn-2 bare pronoun resolves to the FOREGROUNDED
(most-recent gamma-slot) referent among SEVERAL held -- the case the rate-attractor buffer could not (the
order-control), now solved by ADDRESSING referents by slot rather than competing them in rate.

  SIM_BACKEND=numpy python -m research.runners.multi_turn_ordered_wm_demo
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "see", "eat"]


def _say(who, text):
    print(f"  {who:>5}: {text}", flush=True)


def main():
    print("\n=== multi-turn disambiguation on the brain (MultiTurnAgentV2, order-encoded discourse) ===\n",
          flush=True)
    agent = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB},
                             seed=42, enable_neural_render=True)
    c = agent.agent.composer
    print(f"  (discourse buffer: order-encoded gamma-slot WM, D={c.D}, familiarity threshold "
          f"{agent.wm.match_threshold:.3f} [principled, not the de-risk's frozen 0.15])\n", flush=True)
    # a small food-web (who eats what).
    for ag, ob in [("cat", "fish"), ("dog", "worm"), ("fish", "worm"), ("bird", "ball")]:
        c.store(ag, "eat", ob, polarity="AFFIRM")

    print("  -- Scene A: two referents, the pronoun foregrounds the most-recent --", flush=True)
    _say("user", "the dog saw the cat.")                  # introduces dog (slot0), cat (slot1=most-recent)
    agent.hear("dog see cat")
    _say("brain", f"(holding referents {agent.held_referents()}; foregrounded = "
                  f"{agent.most_recent_referent()})")
    _say("user", "what does it eat?")                     # 'it' -> cat (most-recent) -> cat eat fish
    ans = agent.what_does("it", "eat")
    _say("brain", f"it eats {ans}." if ans else "I don't know what it eats.")

    print("\n  -- Scene B: SWAP the order -> the SAME pronoun foregrounds the OTHER referent (the wall) --",
          flush=True)
    agent2 = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    c2 = agent2.agent.composer
    for ag, ob in [("cat", "fish"), ("dog", "worm"), ("fish", "worm"), ("bird", "ball")]:
        c2.store(ag, "eat", ob, polarity="AFFIRM")
    _say("user", "the cat saw the dog.")                  # dog now most-recent (order swapped)
    agent2.hear("cat see dog")
    _say("brain", f"(holding referents {agent2.held_referents()}; foregrounded = "
                  f"{agent2.most_recent_referent()})")
    _say("user", "what does it eat?")                     # 'it' -> dog (most-recent) -> dog eat worm
    ans = agent2.what_does("it", "eat")
    _say("brain", f"it eats {ans}." if ans else "I don't know what it eats.")
    print("        ^ the resolution FLIPPED (cat->fish vs dog->worm) purely because the discourse ORDER changed\n"
          "          -- the order-control the rate-attractor buffer failed (0/6); here it is by-slot, 6/6.",
          flush=True)

    print("\n  -- Scene C: the no-confab moat (no referent held -> abstain, not a guess) --", flush=True)
    agent3 = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=42)
    _say("user", "what does it eat?")                     # empty discourse -> abstain
    ans = agent3.what_does("it", "eat")
    _say("brain", f"it eats {ans}." if ans else "I have no idea what 'it' refers to -- nothing has been mentioned.")

    print("\n  (Every resolution above is on the spiking RF phasor substrate: each referent is bound to a gamma-slot\n"
          "   POSITION phasor; a pronoun reads the most-recent slot via spiking unbind; the winner is WHICH SLOT,\n"
          "   so it flips with the discourse order -- and an empty buffer grounds nothing, so the pronoun abstains.)\n",
          flush=True)


if __name__ == "__main__":
    main()
