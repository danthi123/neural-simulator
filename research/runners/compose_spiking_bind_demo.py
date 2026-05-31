"""Owner-facing demo: compositional role-binding running IN the spiking substrate.

The in-substrate analogue of compose_vsa_demo.py (which used numpy algebra on captured
codes). Here the bind (role (x) filler) and unbind are computed BY SPIKING NEURONS via
threshold coincidence detection -- the biologically-sound realization the owner asked for.

Mechanism (validated 2026-05-31; see research/findings/2026-05-31-in-substrate-spiking-
bind-unbind-VALIDATED.md): a custom bridge with role_ON/OFF + fill_ON/OFF driven source
populations synapsing into 4 coincidence banks A/B/C/D that realize the +-1 Hadamard
(bound_ON = AND(role_ON,fill_ON)+AND(role_OFF,fill_OFF), etc.). The SAME coincidence layer
reused for unbind. Concepts = the substrate's real concept-pool codes (overlapping,
between-cos ~0.70); cleanup = nearest concept (ID-separable, no near-orthogonality needed).

Capability shown:
  - bind a 'sentence' = {subject: X, verb: Y, object: Z} in spiking dynamics (no training);
  - answer role queries ("who? what action? what?") by spiking unbind + cleanup;
  - GENERALIZE: any never-bound (subject, verb, object) works -- VSA composes by construction.

Run:  python -m research.runners.compose_spiking_bind_demo
      python -m research.runners.compose_spiking_bind_demo --seed 43 --proj-dim 800
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
from sim.backend import get_backend

ROLES = ["subject", "verb", "object"]


def bind_sentence_spiking(bridge, idx, slots, concepts, role_codes, role_idx, D, xp):
    """slots: {role_name: word}. Bind each (role, concept) in spiking; superpose (sum) the
    bound rates; ON/OFF opponency (common-mode removal). Returns canonical bound ON/OFF."""
    bound_on = np.zeros(D); bound_off = np.zeros(D)
    for rname, word in slots.items():
        c_on, c_off = P.onoff(concepts[word])
        fon, foff = P._scale_to_current(c_on, c_off, P.FILL_DRIVE)
        b_on, b_off = P.hadamard_spiking(bridge, idx, role_codes[role_idx[rname]], fon, foff, D, xp)
        bound_on += b_on; bound_off += b_off
    bsig = bound_on - bound_off                 # ON/OFF opponency (signed bound)
    return P.onoff(bsig)


def query_spiking(bridge, idx, bound_onoff, rname, concepts, role_codes, role_idx, D, xp, top=3):
    """Unbind a role from the bound (spiking) + clean up to nearest concept(s)."""
    fon, foff = P._scale_to_current(bound_onoff[0], bound_onoff[1], P.FILL_DRIVE)
    e_on, e_off = P.hadamard_spiking(bridge, idx, role_codes[role_idx[rname]], fon, foff, D, xp)
    est = e_on - e_off
    words = list(concepts.keys())
    sims = np.array([concepts[w] @ est for w in words])
    order = np.argsort(-sims)
    return [words[i] for i in order[:top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800, help="<=0 uses full raw D=3200")
    ap.add_argument("--run-steps", type=int, default=120)
    ap.add_argument("--n-generalize", type=int, default=12)
    a = ap.parse_args()
    P.RUN_STEPS = a.run_steps
    xp, backend = get_backend()
    rng = np.random.default_rng(a.seed)
    words, codes = P.load_concepts(a.seed, a.proj_dim, rng)
    D = codes.shape[1]
    concepts = {w: codes[i] for i, w in enumerate(words)}
    role_codes = P.make_roles(8, D, rng)
    role_idx = {ROLES[i]: i for i in range(len(ROLES))}
    print(f"=== Spiking compositional bind demo (backend={backend}, seed={a.seed}, D={D}, "
          f"8D={8*D} neurons) ===")
    print(f"concepts ({len(words)}): {words}")
    print(f"roles: {ROLES}  (bind + unbind computed BY SPIKING coincidence neurons)\n")
    bridge, idx = P.build(a.seed, D, xp)

    def pick(*cands):
        for c in cands:
            if c in concepts:
                return c
        return words[0]

    sents = [
        {"subject": pick("dog", "cat"), "verb": pick("go", "come"), "object": pick("north", "river")},
        {"subject": pick("cat", "dog"), "verb": pick("come", "stop"), "object": pick("south", "apple")},
        {"subject": pick("apple", "dog"), "verb": pick("stop", "look"), "object": pick("big", "east")},
    ]
    print("--- Single-sentence binding + role queries (in spiking) ---")
    for s in sents:
        bound = bind_sentence_spiking(bridge, idx, s, concepts, role_codes, role_idx, D, xp)
        who = query_spiking(bridge, idx, bound, "subject", concepts, role_codes, role_idx, D, xp, top=1)[0]
        did = query_spiking(bridge, idx, bound, "verb", concepts, role_codes, role_idx, D, xp, top=1)[0]
        what = query_spiking(bridge, idx, bound, "object", concepts, role_codes, role_idx, D, xp, top=1)[0]
        ok = (who == s["subject"]) and (did == s["verb"]) and (what == s["object"])
        print(f"  bind {s}")
        print(f"    who(subject)? {who}   action(verb)? {did}   what(object)? {what}   "
              f"[{'OK' if ok else 'MISS'}]")

    print("\n--- Generalization (novel combinations, no training) ---")
    gc = gt = 0
    for _ in range(a.n_generalize):
        s = {r: str(rng.choice(words)) for r in ROLES}
        bound = bind_sentence_spiking(bridge, idx, s, concepts, role_codes, role_idx, D, xp)
        ok = all(query_spiking(bridge, idx, bound, r, concepts, role_codes, role_idx, D, xp, top=1)[0] == s[r]
                 for r in ROLES)
        gc += int(ok); gt += 1
    print(f"  {gt} random novel sentences: {gc}/{gt} fully correct (all 3 roles recovered in spiking)")
    print(f"\nThe bind/unbind are computed by spiking coincidence neurons on real substrate concept "
          f"codes. Biologically-sound compositional binding, realized IN the substrate.")


if __name__ == "__main__":
    main()
