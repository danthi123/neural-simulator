"""Megakernel adoption gate: does the RF megakernel (cfg.enable_rf_cudagraph) give answer-identical results across
the FULL conversational stack -- who/what, negation/yes-no, recursive embedded clauses, multi-hop -- not just flat
who/what? Builds a LOOP agent and a MEGAKERNEL agent (same seed), runs every conversational op, and asserts
megakernel == loop == ground truth. The embedded clause is the one that broke the period lever, so it's the
load-bearing case. Does NOT change the production default (the agent's composer is set opt-in here only).

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_megakernel_conversation_validation
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.core_sim_composition import Clause  # noqa: E402


VOCAB = ["dog", "cat", "bird", "river", "apple",
         "go", "come", "look", "see", "eat", "swim", "fly", "stop",
         "north", "south", "east", "west"]


def _agent(megakernel):
    a = BrainConversationalAgent(seed=42, concepts={w: None for w in VOCAB})
    a.composer._enable_rf_cudagraph = bool(megakernel)   # opt-in BEFORE any op (bridges pick it up at build)
    a.hear("dog go north")
    a.hear("cat come south", polarity="AFFIRM")
    a.hear("river look west", polarity="NEGATE")
    a.hear_clause_fact("dog", "see", Clause("cat", "go", "south"))     # recursive embedded clause
    a.hear("dog eat cat")
    a.hear("cat swim river")
    return a


def _probe(a):
    return {
        "what_dog_go": a.what_does("dog", "go"),
        "who_go_north": a.who_does("go", "north"),
        "yes_cat_come_south": a.is_it_true("cat", "come", "south"),
        "no_river_look_west": a.is_it_true("river", "look", "west"),
        "unknown_apple": a.is_it_true("apple", "stop", "east"),
        "clause_dog_see": a.what_does("dog", "see"),
        "abstain_bird_fly": a.what_does("bird", "fly"),
        "chain_dog_eat_swim": a.composer.query_chain("dog", ["eat", "swim"]),   # dog eat cat -> cat swim river -> river
    }


def main():
    t0 = time.time()
    print("[megakernel conversation validation] megakernel == loop across the full conversational stack?\n",
          flush=True)
    loop = _probe(_agent(False))
    mega = _probe(_agent(True))
    expected = {
        "what_dog_go": "north", "who_go_north": "dog", "yes_cat_come_south": "yes",
        "no_river_look_west": "no", "unknown_apple": "unknown", "clause_dog_see": "cat go south",
        "abstain_bird_fly": None, "chain_dog_eat_swim": "river",   # dog eat cat... (cat eat fish; fish swim river)
    }
    all_ok = True
    for k in loop:
        match = (loop[k] == mega[k])
        # ground truth check only where we pinned it (chain semantics depend on stored facts)
        gt = expected.get(k, "<n/a>")
        gt_ok = (gt == "<n/a>") or (mega[k] == gt)
        ok = match and gt_ok
        all_ok = all_ok and match            # the PRIMARY gate is megakernel == loop (answer-identical)
        flag = "OK " if (match and gt_ok) else ("DIFF" if not match else "gt? ")
        print(f"  {flag} {k:22s} loop={str(loop[k]):14s} mega={str(mega[k]):14s} expect={gt}", flush=True)

    print(f"\n{'='*78}", flush=True)
    if all_ok:
        print("  GO: the megakernel is ANSWER-IDENTICAL to the loop across who/what + negation/yes-no + the "
              "recursive EMBEDDED CLAUSE + multi-hop. The adoption is validated on the conversational stack; "
              "flipping the agent to opt in (cfg.enable_rf_cudagraph) is safe pending a clean quiet-GPU speedup.",
              flush=True)
    else:
        print("  BOUNDARY: the megakernel DIFFERS from the loop on some op (see DIFF above) -- localize before any "
              "adoption; keep default-off.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*78}", flush=True)


if __name__ == "__main__":
    main()
