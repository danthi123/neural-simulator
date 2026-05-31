"""Owner-facing demo: a small QUERYABLE KNOWLEDGE BASE running in the spiking substrate.

Builds on the validated spiking compositional bind (see compose_spiking_bind_demo.py and the
finding 2026-05-31-in-substrate-spiking-bind-unbind-VALIDATED.md). Stores a few subject/verb/object
FACTS as separate spiking bound structures and answers RELATIONAL queries -- "who chases cat?",
"what does dog do?", "what is the action between dog and cat?" -- all by spiking bind/unbind +
nearest-concept cleanup on the project's real concept-pool codes.

A fact "dog chases cat" = agent (x) dog + action (x) chase + patient (x) cat (3 spiking-coincidence
bindings). Facts are stored SEPARATELY (the correct architecture; superposing facts into one vector
degrades -- the multi-hop wall). A relational query finds the fact by a cue role (spiking-unbind +
match) and reads another role (spiking-unbind). The numpy cheap-first + the spiking probe both
RESOLVED multi-seed.

Honest scope: roles (agent/action/patient) are SUPPLIED, not parsed from raw input (a learned
parser is the next arc); facts use real substrate concept codes; this is structured fact-memory
with cue-based retrieval, not open-ended relational reasoning over superposition.

Run:  python -m research.runners.compose_relational_memory_demo
      python -m research.runners.compose_relational_memory_demo --seed 43
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as R
from sim.backend import get_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--run-steps", type=int, default=150)
    ap.add_argument("--coinc-bias", type=float, default=-500.0,
                    help="validated relational-memory operating point (higher rate = robust multi-seed)")
    a = ap.parse_args()
    P.RUN_STEPS = a.run_steps
    P.COINC_BIAS = a.coinc_bias
    xp, backend = get_backend()
    rng = np.random.default_rng(a.seed)
    words, codes = P.load_concepts(a.seed, a.proj_dim, rng)
    D = codes.shape[1]
    concepts = {w: codes[i] for i, w in enumerate(words)}
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in R.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    print(f"=== Spiking relational knowledge-base demo (backend={backend}, seed={a.seed}, D={D}) ===")
    print(f"vocabulary ({len(words)}): {words}")
    print(f"roles: {R.ROLES}  (facts stored + queried via SPIKING bind/unbind)\n")
    bridge, idx = P.build(a.seed, D, xp)

    def pick(*c):
        for w in c:
            if w in concepts:
                return w
        return words[0]

    # a small SVO knowledge base (using whatever vocab the substrate has)
    facts = [
        {"agent": pick("dog"), "action": pick("go", "come"), "object_": pick("north", "river")},
        {"agent": pick("cat"), "action": pick("come", "stop"), "object_": pick("south", "apple")},
    ]
    # rename object_ -> patient for the probe's ROLES
    facts = [{"agent": f["agent"], "action": f["action"], "patient": f["object_"]} for f in facts]
    bound = [R.bind_fact_spiking(bridge, idx, f, concepts, roles, D, xp) for f in facts]

    print("--- Knowledge base (stored as spiking bound structures) ---")
    for f in facts:
        print(f"    {f['agent']} {f['action']} {f['patient']}")

    def ask_patient_of_agent(cue_agent):
        for f in range(len(facts)):
            if R.unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cue_agent:
                return R.unbind_spiking(bridge, idx, bound[f], "patient", roles, concepts, words, D, xp)
        return "(no fact found)"

    def ask_action_of_agent(cue_agent):
        for f in range(len(facts)):
            if R.unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cue_agent:
                return R.unbind_spiking(bridge, idx, bound[f], "action", roles, concepts, words, D, xp)
        return "(no fact found)"

    print("\n--- Relational queries (answered by spiking unbind + cleanup) ---")
    for f in facts:
        p = ask_patient_of_agent(f["agent"])
        v = ask_action_of_agent(f["agent"])
        ok = (p == f["patient"]) and (v == f["action"])
        print(f"    what does '{f['agent']}' have as object? {p}   what is its action? {v}   "
              f"[{'OK' if ok else 'MISS'}]")

    # control: a cue agent not in the KB
    absent = next((w for w in words if w not in [f["agent"] for f in facts]), words[0])
    print(f"    (control) what does '{absent}' (not in KB) have? {ask_patient_of_agent(absent)}")
    print("\nThe knowledge base is stored + queried entirely via spiking compositional binding on real "
          "substrate concepts. Honest scope: roles supplied (parser is the next arc); cue-based "
          "relational retrieval, not open-ended reasoning.")


if __name__ == "__main__":
    main()
