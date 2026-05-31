"""Owner-facing demo: conversation where the text->role PARSING is LEARNED (Hebbian), not a
positional template -- closing the one hardcoded step in the conversational agent.

Shows: the Hebbian-learned parser (conjunctive position x voice -> role, validated multi-seed in
_insubstrate_parser_stdp_probe.py) assigns roles for BOTH active and passive sentence structures;
the spiking bind stores the fact; a relational query answers VOICE-INVARIANTLY. Because the parser
LEARNED the active<->passive role flip, "dog go north" (active) and "north is go by dog" (passive,
structurally) are understood as the SAME fact (dog is the agent) -- something a positional template
cannot do.

Honest scope: the 16-word vocab has no natural passive morphology, so passive sentences here are
structural ("N2 is V by N1") with the concept words as fillers; the parser handles the STRUCTURE
(content-position + function-word 'by' presence), which is the learned part. Reuses the validated
e2e machinery (train the parser + parse + bind + query) by import.

Run:  python -m research.runners.compose_learned_parse_demo
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw._insubstrate_parser_bind_e2e_probe as E   # validated learned-parse->bind
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    a = ap.parse_args()
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    print(f"=== Conversation with a LEARNED parser (no positional template) (backend={backend}, "
          f"seed={a.seed}) ===")
    print("training the Hebbian parser (conjunctive position x voice -> role)...")
    learned = E.train_parser_and_extract(a.seed, xp)     # the LEARNED conjunction-index -> role map
    parse_ok = sum(int(learned[k] == E.PA.GT[k]) for k in range(6))
    print(f"  learned parser map {parse_ok}/6: {[learned[k] for k in range(6)]}")
    print("  (this map was ACQUIRED by Hebbian co-firing, including the active<->passive flip -- "
          "NOT hardcoded position->role)\n")

    words, concepts = E.load_concepts(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.proj_dim and a.proj_dim > 0:
        Pm = rng.standard_normal((concepts[words[0]].shape[0], a.proj_dim)) / np.sqrt(concepts[words[0]].shape[0])
        concepts = {w: E._center(concepts[w] @ Pm) for w in words}
    D = concepts[words[0]].shape[0]
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bridge, idx = P.build(a.seed, D, xp)

    def pick(*c):
        for w in c:
            if w in concepts:
                return w
        return words[0]
    dog, go, north = pick("dog"), pick("go", "come"), pick("north", "river")

    print("--- The user states the SAME fact two ways; the LEARNED parser assigns roles for each ---")
    # active "dog go north": content order [dog, go, north], voice=active
    act = E.parse_sentence([dog, go, north], False, learned)
    # passive "north is go by dog": content order [north, go, dog], voice=passive (the parser flips it)
    pas = E.parse_sentence([north, go, dog], True, learned)
    print(f"  active  '{dog} {go} {north}'         -> parsed {act}")
    print(f"  passive '{north} is {go} by {dog}'   -> parsed {pas}")
    S_act = RM.bind_fact_spiking(bridge, idx, act, concepts, roles, D, xp)
    S_pas = RM.bind_fact_spiking(bridge, idx, pas, concepts, roles, D, xp)

    print("\n--- Ask 'who is the agent?' of each (answered by spiking unbind + cleanup) ---")
    a_act = RM.unbind_spiking(bridge, idx, S_act, "agent", roles, concepts, words, D, xp)
    a_pas = RM.unbind_spiking(bridge, idx, S_pas, "agent", roles, concepts, words, D, xp)
    print(f"  agent of the active  sentence -> {a_act}")
    print(f"  agent of the passive sentence -> {a_pas}")
    ok = (a_act == dog) and (a_pas == dog)
    print(f"\n  Both -> '{dog}'? {ok}  -- the LEARNED parser + spiking bind understand active and passive "
          f"as the same meaning (voice-invariant). A positional template would call '{north}' the agent of "
          f"the passive sentence.")
    print("\nThe text->role step is LEARNED (Hebbian), not a positional template; the bind/unbind that "
          "form and query the fact are spiking composition. Honest scope: 16-word vocab, SVO frames.")


if __name__ == "__main__":
    main()
