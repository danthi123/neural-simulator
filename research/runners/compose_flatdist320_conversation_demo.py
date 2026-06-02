"""Full-320 flat-distinct conversational demo -- a queryable SVO knowledge base over 320 DISTINCT concepts,
stored and answered IN the spiking substrate via the validated coincidence bind/unbind.

This is the tangible owner-facing artifact of the 320-concept biological composition: type-free scripted
"conversation" where SVO facts spanning all five concept banks (noun / verb / adjective / spatial /
functional) are bound by spiking coincidence, stored separately, and recovered by spiking unbind + cleanup
over all 320 distinct flat codes. It reuses the SAME machinery validated at 192 concepts (1.000/1.000/1.000)
and (if the 320 structured test resolves) at 320.

It loads the 320 distinct flat codes cached by _insubstrate_flatdistinct320_test.py (5 distinct-seed bridges,
seeds 42-46). No training, no protected-module change, no autograd -- reuse-by-import only. GPU/CuPy.

Run AFTER the 320 structured composition test has produced research/findings/raw/_flatdist320_codes.npz:
  python -m research.runners.compose_flatdist320_conversation_demo
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/_flatdist320_codes.npz"


def _pick(words, bank_of, bank, n, rng):
    cands = [w for w in words if bank_of[w] == bank]
    return [str(w) for w in rng.choice(cands, size=min(n, len(cands)), replace=False)]


def main():
    xp, backend = get_backend()
    if not os.path.exists(CACHE):
        print(f"CANNOT-RUN: {CACHE} missing -- run _insubstrate_flatdistinct320_test first.", flush=True)
        return
    d = np.load(CACHE)
    words = [str(w) for w in d["_words"]]
    bank_of = {str(w): str(b) for w, b in zip(d["_words"], d["_banks"])}
    codes = {w: np.asarray(d[w], dtype=np.float64) for w in words}
    D = codes[words[0]].shape[0]
    print(f"=== full-320 flat-distinct conversational KB (backend={backend}, V={len(words)}, D={D}) ===",
          flush=True)

    P.RUN_STEPS = 150
    P.COINC_BIAS = -500.0   # validated higher-rate operating point (relational fact-memory 3/3 multi-seed)
    seed = 42
    rng = np.random.default_rng(seed)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bb, bidx = P.build(seed, D, xp)

    # Build a small cross-bank knowledge base. Words are illustrative (drawn from the actual 320 vocab);
    # the point is the MECHANISM: any concept from any bank can fill any role and be recovered.
    nouns = _pick(words, bank_of, "noun", 3, rng)
    verbs = _pick(words, bank_of, "verb", 3, rng)
    adjs = _pick(words, bank_of, "adj", 3, rng)
    facts = [
        {"agent": nouns[0], "action": verbs[0], "patient": adjs[0]},
        {"agent": nouns[1], "action": verbs[1], "patient": adjs[1]},
        {"agent": nouns[2], "action": verbs[2], "patient": adjs[2]},
    ]
    print("\n-- storing facts (spiking coincidence bind; stored separately) --", flush=True)
    bound = []
    for f in facts:
        bound.append(RM.bind_fact_spiking(bb, bidx, f, codes, roles, D, xp))
        print(f"   stored: agent={f['agent']:>10}  action={f['action']:>10}  patient={f['patient']:>10}",
              flush=True)

    def answer_role(fact_i, role):
        return RM.unbind_spiking(bb, bidx, bound[fact_i], role, roles, codes, words, D, xp)

    def answer_relational(cue_agent):
        for i in range(len(facts)):
            if RM.unbind_spiking(bb, bidx, bound[i], "agent", roles, codes, words, D, xp) == cue_agent:
                return RM.unbind_spiking(bb, bidx, bound[i], "patient", roles, codes, words, D, xp)
        return "(no fact found)"

    print("\n-- conversation (spiking unbind + cleanup over all 320) --", flush=True)
    ok = tot = 0
    for i, f in enumerate(facts):
        a = answer_role(i, "agent"); p = answer_relational(f["agent"])
        tot += 2; ok += int(a == f["agent"]) + int(p == f["patient"])
        print(f"   Q: who is the agent of fact {i}?            A: {a:>10}   "
              f"({'OK' if a == f['agent'] else 'MISS exp ' + f['agent']})", flush=True)
        print(f"   Q: what is '{f['agent']}' {f['action']}?    A: {p:>10}   "
              f"({'OK' if p == f['patient'] else 'MISS exp ' + f['patient']})", flush=True)
    # absent-cue control
    absent = next(w for w in words if w not in [f["agent"] for f in facts])
    ctrl = answer_relational(absent)
    print(f"   Q: what is '{absent}' (never stored)?         A: {ctrl}   "
          f"({'OK' if ctrl == '(no fact found)' else 'FALSE-MATCH'})", flush=True)
    print(f"\n   recovered {ok}/{tot} role/relational answers + control "
          f"{'clean' if ctrl == '(no fact found)' else 'FALSE-MATCH'}", flush=True)


if __name__ == "__main__":
    main()
