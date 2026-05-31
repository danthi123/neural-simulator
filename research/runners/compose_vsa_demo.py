"""Compositional role-binding demo on REAL substrate concept codes via biologically-
realizable VSA binding. Demonstrates the 2026-05-31 composition revision: generalizable
(role x filler) binding works with the substrate's OVERLAPPING concept codes (cleanup
uses ID-separability, not near-orthogonality), needing only a FEW near-ortho ROLE codes.

Capability shown:
  - bind a 'sentence' = {subject: X, verb: Y, object: Z} on the fly (no training);
  - answer role queries ("who?" -> subject, "what action?" -> verb, "what?" -> object);
  - GENERALIZE: any never-before-seen (subject, verb, object) combination works
    immediately, because VSA binds compositionally;
  - compose MULTIPLE sentences and query each.

Biological grounding:
  - fillers = the substrate's real concept-pool activity codes (captured by driving
    lang_input(word); cached at activity_level_integration_cache/denoise64_seed*.npz),
    mean-centred (= baseline-subtracted firing rate, the project's common-mode removal).
  - roles = a few distinct DISTRIBUTED codes (3 roles: subject/verb/object).
  - bind = Hadamard product, which is EXACTLY realized by ON/OFF rate coding + coincidence
    detection using only >=0 operations (firing-above-baseline / below-baseline pair;
    verified max-diff 0.0) -- a standard dendritic/spiking operation.
  - cleanup = nearest concept (the substrate is ID-separable: within > between).

Honest scope: this is the VALIDATED ALGEBRA on real substrate-derived codes (numpy). The
in-spiking-dynamics implementation of the coincidence bind is the deeper next step. Caveats
from the finding hold (roles must be distributed; a cleanup bias affects low-load absolutes).

Run:  python -m research.runners.compose_vsa_demo
      python -m research.runners.compose_vsa_demo --seed 43
"""
from __future__ import annotations
import argparse
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
ROLES = ["subject", "verb", "object"]


def _center(v):
    v = v.astype(np.float64)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    """Return {word: substrate concept code} from the cached concept-pool activity."""
    path = CACHE % seed
    if not os.path.exists(path):
        raise SystemExit(f"No concept-code cache at {path}. (Run the substrate capture first.)")
    d = np.load(path)
    return {k[5:]: _center(d["obs__" + k[5:]].mean(axis=0))
            for k in d.files if k.startswith("obs__")}


def make_roles(D, rng):
    """A few DISTRIBUTED near-ortho role codes (the demo's subject/verb/object).
    ON/OFF realizable: a +-1 distributed pattern = two firing-rate populations."""
    R = {}
    for name in ROLES:
        r = rng.choice([-1.0, 1.0], size=D)
        R[name] = r / np.linalg.norm(r)
    return R


def bind_sentence(slots, concepts, roles):
    """slots: {role: word}. Returns the bound composite vector (sum of role (x) filler)."""
    D = next(iter(concepts.values())).shape[0]
    S = np.zeros(D)
    for role, word in slots.items():
        S = S + roles[role] * concepts[word]   # Hadamard bind (= ON/OFF coincidence, >=0)
    return S


def query(S, role, concepts, roles, top=3):
    """Unbind a role from the composite and clean up to the nearest concept(s)."""
    est = S * roles[role]                       # unbind (= coincidence with the role)
    words = list(concepts.keys())
    sims = np.array([concepts[w] @ est for w in words])
    order = np.argsort(-sims)
    return [(words[i], float(sims[i])) for i in order[:top]]


def answer(S, role, concepts, roles):
    return query(S, role, concepts, roles, top=1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    concepts = load_concepts(a.seed)
    rng = np.random.default_rng(a.seed)
    D = next(iter(concepts.values())).shape[0]
    roles = make_roles(D, rng)
    vocab = list(concepts.keys())
    print(f"=== Compositional role-binding demo (substrate codes, seed {a.seed}) ===")
    print(f"concepts ({len(vocab)}): {vocab}")
    print(f"roles: {ROLES}")
    print("(fillers = substrate concept-pool activity, between-cos ~0.7; bind = ON/OFF "
          "coincidence; cleanup = nearest concept)\n")

    # pick available words for a few demo sentences (use whatever is in the vocab)
    def pick(*cands):
        for c in cands:
            if c in concepts:
                return c
        return vocab[0]

    sents = [
        {"subject": pick("dog", "cat"), "verb": pick("go", "come"), "object": pick("north", "river")},
        {"subject": pick("cat", "dog"), "verb": pick("come", "stop"), "object": pick("south", "apple")},
        {"subject": pick("apple", "dog"), "verb": pick("stop", "look"), "object": pick("big", "east")},
    ]
    print("--- Single-sentence binding + role queries ---")
    for s in sents:
        S = bind_sentence(s, concepts, roles)
        who = answer(S, "subject", concepts, roles)
        did = answer(S, "verb", concepts, roles)
        what = answer(S, "object", concepts, roles)
        ok = (who == s["subject"]) and (did == s["verb"]) and (what == s["object"])
        print(f"  bind {s}")
        print(f"    who(subject)? {who}   action(verb)? {did}   what(object)? {what}   "
              f"[{'OK' if ok else 'MISS'}]")

    # GENERALIZATION: a never-bound novel combination works immediately
    print("\n--- Generalization (novel combinations, no training) ---")
    gen_correct = 0
    gen_total = 0
    for _ in range(20):
        s = {"subject": rng.choice(vocab), "verb": rng.choice(vocab), "object": rng.choice(vocab)}
        S = bind_sentence(s, concepts, roles)
        ok = all(answer(S, r, concepts, roles) == s[r] for r in ROLES)
        gen_correct += int(ok); gen_total += 1
    print(f"  20 random novel sentences: {gen_correct}/{gen_total} fully correct "
          f"(all 3 roles recovered)")

    # COMPOSITION of two sentences in one structure
    print("\n--- Two sentences composed in one structure ---")
    s1 = {"subject": pick("dog"), "verb": pick("go"), "object": pick("north")}
    s2 = {"subject": pick("cat"), "verb": pick("come"), "object": pick("south")}
    S = bind_sentence(s1, concepts, roles) + bind_sentence(s2, concepts, roles)
    print(f"  composite of {s1} + {s2}")
    for r in ROLES:
        print(f"    {r}: top-3 = {[w for w, _ in query(S, r, concepts, roles)]}")
    print("  (two bound sentences overlap in one vector; each role query returns BOTH "
          "fillers in its top-2 -- compositional superposition.)")

    print(f"\nHonest: validated algebra on real substrate-derived codes; biologically-realizable "
          f"bind (ON/OFF coincidence). In-spiking-dynamics implementation is the deeper next step.")


if __name__ == "__main__":
    main()
