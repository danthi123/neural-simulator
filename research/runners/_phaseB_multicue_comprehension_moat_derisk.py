"""GAP-1 cheap-first de-risk -- the COMPREHENSION-level no-confab moat for the multi-cue role parser.

THE HOLE (verified, brain_conversational_agent.py:342/319): the production `hear_multicue`/`hear_case` call
`parse()` (ALWAYS commits a role), NOT `parse_decisive()`. So a genuinely ambiguous degraded sentence -- two
animate nouns + a symmetric verb, scrambled order (no content cue breaks the tie, position unreliable) --
CONFABULATES + STORES a wrong fact at comprehension, which the query-time moat cannot un-store.

THE FIX (de-risked here): route `hear()` through `parse_decisive`; on `decisive=False`, ABSTAIN (store nothing).
`MultiCueRoleParser.parse_decisive` already exists + is validated (the content gate, multicue_role_parser.py:122);
GAP-1 just wires it into the production hear path. Scoping: research/findings/2026-06-22-robust-multicue-parser-scoping.md.

This de-risk drives the proposed fix EXTERNALLY (no agent edit yet) on fresh per-sentence agents, and validates the
GO bar + anti-cheats, multi-seed. GO -> wire the gate into `hear_multicue`/`hear_case` + a CI test.

  SIM_BACKEND=numpy python -m research.runners._phaseB_multicue_comprehension_moat_derisk --seeds 42
  SIM_BACKEND=numpy python -m research.runners._phaseB_multicue_comprehension_moat_derisk --seeds 42,43,44,100,101,102
"""
import argparse
import json
import os
import random

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

ANIM = ["dog", "cat", "fox", "bird", "wolf", "owl"]
INAN = ["apple", "ball", "rock", "book", "stick", "bone"]
SYMM = ["chase", "push", "meet", "pass"]            # symmetric: BOTH nouns are plausible agents (content-ambiguous)
ASYM = ["eat", "carry", "throw", "hold"]            # asymmetric: animate agent acts on inanimate patient (decisive)
VERBS = SYMM + ASYM
VOCAB = {w: None for w in ANIM + INAN + VERBS}
N_PER = 5


def _agent(seed, abstain_margin=None):
    a = BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=VOCAB,
                                 enable_multicue_competition=True, multicue_verbs=VERBS)
    if abstain_margin is not None:
        a._ensure_multicue_parser()._abstain_margin = float(abstain_margin)
    return a


def _hear_decisive(agent, words):
    """The PROPOSED GAP-1 fix: gate on parse_decisive -> ABSTAIN (store nothing) when content is non-decisive."""
    _roles, decisive = agent._ensure_multicue_parser().parse_decisive(words)
    if not decisive:
        return False
    agent.hear(" ".join(words))
    return True


def _ambiguous(rng):
    """Two animate nouns + a symmetric verb, object-fronted/scrambled -> genuinely undecidable. Must ABSTAIN."""
    out = []
    for _ in range(N_PER):
        a, b = rng.sample(ANIM, 2)
        out.append([b, rng.choice(SYMM), a])        # [nounB, symm-verb, nounA] -- no content/position decider
    return out


def _decisive(rng):
    """Animate + inanimate + asymmetric verb, OBJECT-FRONTED -> content decides (inanimate=patient). Must RESOLVE."""
    out = []
    for _ in range(N_PER):
        out.append([rng.choice(INAN), rng.choice(ASYM), rng.choice(ANIM)])   # [patient, verb, agent] fronted
    return out


def _canonical(rng):
    out = []
    for _ in range(N_PER):
        out.append([rng.choice(ANIM), rng.choice(ASYM), rng.choice(INAN)])   # [agent, verb, patient] canonical SVO
    return out


def _q_nonnull(agent, w):
    """A who/what query that returns non-None iff a fact for this sentence's nouns/verb was stored."""
    v = w[1]
    return agent.who_does(v, w[0]) is not None or agent.who_does(v, w[2]) is not None


def run_seed(seed):
    r = {"seed": int(seed)}
    rng = random.Random(seed)
    amb, dec, can = _ambiguous(rng), _decisive(rng), _canonical(rng)

    # FIX on ambiguous: abstain (store nothing) -> queries None
    fix_abstain = fix_confab = 0
    for w in amb:
        a = _agent(seed)
        stored = _hear_decisive(a, w)
        if not stored and not _q_nonnull(a, w):
            fix_abstain += 1
        elif stored and _q_nonnull(a, w):
            fix_confab += 1
    r["fix_ambiguous_abstain"] = fix_abstain / len(amb)
    r["fix_ambiguous_confab"] = fix_confab

    # CURRENT (the bug): agent.hear() always parses+stores -> confabulates on ambiguous
    cur_confab = 0
    for w in amb:
        a = _agent(seed)
        a.hear(" ".join(w))
        if _q_nonnull(a, w):
            cur_confab += 1
    r["current_confab"] = cur_confab / len(amb)

    # ANTI-CHEAT 1 (margin-LESION, the decisive control): abstain_margin=0 -> the gate can NEVER fire -> it must
    # reproduce the confabulation (proves the abstention is CAUSED by the gate, not the parser silently failing).
    lesion_abstain = 0
    for w in amb:
        a = _agent(seed, abstain_margin=0.0)
        if not _hear_decisive(a, w):
            lesion_abstain += 1
    r["lesion_ambiguous_abstain"] = lesion_abstain / len(amb)

    # FIX on decisive (object-fronted): must RESOLVE (the object-fronted win is not lost)
    dec_res = 0
    for w in dec:
        p, v, ag = w
        a = _agent(seed)
        if _hear_decisive(a, w) and a.who_does(v, p) == ag and a.what_does(ag, v) == p:
            dec_res += 1
    r["fix_decisive_resolve"] = dec_res / len(dec)

    # FIX on canonical: unregressed
    can_res = 0
    for w in can:
        ag, v, p = w
        a = _agent(seed)
        if _hear_decisive(a, w) and a.who_does(v, p) == ag and a.what_does(ag, v) == p:
            can_res += 1
    r["fix_canonical_resolve"] = can_res / len(can)

    # ANTI-CHEAT 2 (moat-never-weakened): a stored decisive fact + an UNSTORED query -> None (query-time moat intact)
    a = _agent(seed)
    _hear_decisive(a, ["apple", "eat", "dog"])
    r["moat_unstored_none"] = (a.who_does("carry", "ball") is None and a.what_does("owl", "push") is None)

    r["GO"] = bool(
        r["fix_ambiguous_abstain"] >= 0.99 and r["fix_ambiguous_confab"] == 0      # the fix abstains, 0 confab
        and r["current_confab"] >= 0.50                                            # the bug genuinely exists
        and r["lesion_ambiguous_abstain"] <= 0.10                                  # the gate CAUSES the abstention
        and r["fix_decisive_resolve"] >= 0.80 and r["fix_canonical_resolve"] >= 0.80
        and r["moat_unstored_none"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="research/findings/raw/_multicue_comprehension_moat_derisk.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = [run_seed(s) for s in seeds]
    for r in rows:
        print(f"seed {r['seed']}: fix_abstain={r['fix_ambiguous_abstain']:.2f} confab={r['fix_ambiguous_confab']} | "
              f"current_confab={r['current_confab']:.2f} | lesion_abstain={r['lesion_ambiguous_abstain']:.2f} | "
              f"dec_resolve={r['fix_decisive_resolve']:.2f} can_resolve={r['fix_canonical_resolve']:.2f} | "
              f"moat={r['moat_unstored_none']} -> {'GO' if r['GO'] else 'NO-GO'}", flush=True)
    n_go = sum(r["GO"] for r in rows)
    verdict = "GO" if n_go >= max(1, len(seeds) - (1 if len(seeds) >= 6 else 0)) else "NEGATIVE"
    print(f"\nOVERALL: {n_go}/{len(seeds)} GO -> {verdict}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"seeds": seeds, "rows": rows, "n_go": n_go, "verdict": verdict}, f, indent=2)


if __name__ == "__main__":
    main()
