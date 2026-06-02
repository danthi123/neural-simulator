"""Negation + yes/no + who-questions on the full-320 flat-distinct substrate -- the richer conversational
stack (K=4: agent/action/patient/polarity) on the validated 320-concept biological composition.

The negation mechanism (a bound POLARITY tag AFFIRM/NEGATE, _insubstrate_negation_probe.py) and who-QA
(unbind agent) are validated on a small-vocab substrate. This ports them to the real 320 distinct-code
substrate -- a genuine K=4 question: the structured/any-bank composition is K=3 (agent/action/patient) and
held (1.000 / 0.992 6-seed); does adding the 4th polarity role still answer yes/no correctly on the noisier
real 320 codes? Tests, multi-seed: affirmed fact -> "yes"; negated fact -> "no"; who-question -> the agent;
unknown fact -> "unknown" (the anti-artifact abstention control).

FROZEN PASS: yes/no accuracy >= 0.80 AND who accuracy >= 0.80 AND unknown-control >= 0.80, all multi-seed.
Reuse-by-import (NEG.bind_fact4 K=4, RM.unbind_spiking, the cached 320 codes); no protected-module change;
no autograd. GPU/CuPy. Run after _insubstrate_flatdistinct320_test has cached the codes:
  python -m research.findings.raw._insubstrate_flatdist320_negation_qa_test
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.findings.raw._insubstrate_negation_probe as NEG
from sim.backend import get_backend

CACHE = "research/findings/raw/_flatdist320_codes.npz"
SEEDS = [42, 43, 44]
TRIALS_PER = 10


def _center(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    xp, backend = get_backend()
    if not os.path.exists(CACHE):
        print(f"CANNOT-CONCLUDE: {CACHE} missing -- run _insubstrate_flatdistinct320_test first.", flush=True)
        return
    d = np.load(CACHE)
    words = [str(w) for w in d["_words"]]
    base_codes = {w: _center(d[w]) for w in words}
    D = base_codes[words[0]].shape[0]
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    pol_words = ["AFFIRM", "NEGATE"]
    print(f"=== full-320 NEGATION / yes-no / who-QA (K=4) (backend={backend}, V={len(words)}, D={D}) ===",
          flush=True)

    yn_res, who_res, ctrl_res = [], [], []
    transcript_seed = SEEDS[0]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        concepts = dict(base_codes)
        for tag in pol_words:                       # distinct polarity-filler codes (seeded)
            concepts[tag] = _center(rng.standard_normal(D))
        roles = {r: rng.choice([-1.0, 1.0], size=D) for r in NEG.ROLES4}
        roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
        bb, bidx = P.build(seed, D, xp)

        def yesno(bounds, facts, agent, action, patient):
            for b, fc in zip(bounds, facts):
                if (RM.unbind_spiking(bb, bidx, b, "agent", roles, concepts, words, D, xp) == agent and
                        RM.unbind_spiking(bb, bidx, b, "action", roles, concepts, words, D, xp) == action and
                        RM.unbind_spiking(bb, bidx, b, "patient", roles, concepts, words, D, xp) == patient):
                    pol = RM.unbind_spiking(bb, bidx, b, "polarity", roles, concepts, pol_words, D, xp)
                    return "yes" if pol == "AFFIRM" else "no"
            return "unknown"

        yn_ok = who_ok = ctrl_ok = tot = 0
        sample = None
        for t in range(TRIALS_PER):
            pk = rng.choice(len(words), 6, replace=False)
            aff = {"agent": words[pk[0]], "action": words[pk[1]], "patient": words[pk[2]], "polarity": "AFFIRM"}
            neg = {"agent": words[pk[3]], "action": words[pk[4]], "patient": words[pk[5]], "polarity": "NEGATE"}
            bounds = [NEG.bind_fact4(bb, bidx, aff, concepts, roles, D, xp),
                      NEG.bind_fact4(bb, bidx, neg, concepts, roles, D, xp)]
            facts = [aff, neg]
            a_yn = yesno(bounds, facts, aff["agent"], aff["action"], aff["patient"])    # -> yes
            n_yn = yesno(bounds, facts, neg["agent"], neg["action"], neg["patient"])    # -> no
            yn_ok += int(a_yn == "yes" and n_yn == "no")
            # who-question on the affirmed fact
            who = RM.unbind_spiking(bb, bidx, bounds[0], "agent", roles, concepts, words, D, xp)
            who_ok += int(who == aff["agent"])
            # unknown-fact abstention control
            spare = [w for w in words if w not in [aff[r] for r in ("agent", "action", "patient")] +
                     [neg[r] for r in ("agent", "action", "patient")]]
            ctrl_ok += int(yesno(bounds, facts, spare[0], spare[1], spare[2]) == "unknown")
            tot += 1
            if seed == transcript_seed and t == 0:
                sample = (aff, neg, a_yn, n_yn, who, yesno(bounds, facts, spare[0], spare[1], spare[2]),
                          (spare[0], spare[1], spare[2]))
        yn_res.append(yn_ok / tot); who_res.append(who_ok / tot); ctrl_res.append(ctrl_ok / tot)
        print(f"  seed {seed}: yes/no={yn_ok/tot:.3f}  who={who_ok/tot:.3f}  unknown-control={ctrl_ok/tot:.3f}",
              flush=True)
        if sample is not None:
            aff, neg, a_yn, n_yn, who, c_yn, sp = sample
            print(f"    [transcript seed {seed}] taught: '{aff['agent']} {aff['action']} {aff['patient']}' (affirm) "
                  f"+ '{neg['agent']} not {neg['action']} {neg['patient']}' (negate)", flush=True)
            print(f"      does {aff['agent']} {aff['action']} {aff['patient']}? -> {a_yn}   "
                  f"does {neg['agent']} {neg['action']} {neg['patient']}? -> {n_yn}   "
                  f"who {aff['action']} {aff['patient']}? -> {who}   "
                  f"does {sp[0]} {sp[1]} {sp[2]} (never taught)? -> {c_yn}", flush=True)

    myn, mwho, mc = float(np.mean(yn_res)), float(np.mean(who_res)), float(np.mean(ctrl_res))
    print(f"\nRESULT: yes/no {yn_res} (mean {myn:.3f}) | who {who_res} (mean {mwho:.3f}) | "
          f"unknown-control {ctrl_res} (mean {mc:.3f})", flush=True)
    if min(yn_res) >= 0.80 and min(who_res) >= 0.80 and min(ctrl_res) >= 0.80:
        print("VERDICT: RESOLVES -- the K=4 negation + yes/no + who-QA conversational stack runs on the full-320 "
              "biological substrate, multi-seed. Negation = an explicit bound polarity tag (not absence); "
              "the substrate abstains on never-taught facts.", flush=True)
    else:
        print(f"VERDICT: a metric dips below 0.80 multi-seed (yes/no {min(yn_res):.2f}, who {min(who_res):.2f}, "
              f"ctrl {min(ctrl_res):.2f}) -- the K=4 polarity load is the limit on the noisier real 320 codes; "
              "characterise (raise rate / window, or K=3 + separate polarity store).", flush=True)


if __name__ == "__main__":
    main()
