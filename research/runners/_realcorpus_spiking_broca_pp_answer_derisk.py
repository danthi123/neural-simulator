"""FULLY-SPIKING GENERATION of the real-corpus PP (spatial) answer: "the owl runs TO the pond" (C_PPGOAL) and
"the owl runs ON the rock" (C_PPLOC) -- the 6-slot construction order produced ON SPIKES by the EMERGE-72/74
registry producer, and every word (incl. the 3sg verb via PRODUCTIVE inflection + the prepositions to/on)
spelled ON SPIKES by the productive multi-bridge A->W. Extends the spiking generation from property (F_MODAL)
+ transitive (C_TRANS) to the SPATIAL constructions. Regular verb (run->runs) sidesteps the -ies allomorphy.
Gate-first moat. NO `sim/` edit. Requires SIM_BACKEND=numpy + BRIDGE-1/2/3/4 + the affix A->W checkpoints.
"""
from __future__ import annotations
import argparse
from research.runners._emerge72_construction_registry_derisk import RegistryBrocaProducer, decision
from research.runners._emerge74_transitive_ditransitive_derisk import SVOConstructionRegistry, build_stream_svo, emerge_v3
from research.runners._realcorpus_productive_multi_speaker import ProductiveMultiSpeaker


def run(seed=42, aw_seed=42):
    speaker = ProductiveMultiSpeaker(seed=seed, aw_seed=aw_seed)
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    fits = reg.registered_fits()
    have = [c for c in ("C_PPGOAL", "C_PPLOC") if c in fits]
    if not have:
        return {"n_ok": 0, "n_ans": 0, "moat_ok": False, "have": have}
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq, spell=speaker.spell)
    # WARM-UP: the FIRST A->W spell after build reads from an un-settled substrate (a known first-production
    # transient); a throwaway render settles it so the scored renders are history-independent (EMERGE-61 motivation).
    prod.speak(decision("ANSWER", construction=have[0], subject="cat", verb="run", obj="rock"))
    # PP facts (regular verb 'run' -> 'runs' via productive inflection); prepositions rendered by construction
    facts = [("C_PPGOAL", "owl", "run", "pond"), ("C_PPLOC", "owl", "run", "rock"),
             ("C_PPGOAL", "cat", "walk", "hill"), ("C_PPLOC", "dog", "walk", "nest")]
    facts = [f for f in facts if f[0] in have]
    prep = {"C_PPGOAL": "to", "C_PPLOC": "on"}
    n_ok = n_ans = 0
    prod0 = prod.production_count                            # AFTER the warm-up (so the moat count matches n_ans)
    for (c, s, v, o) in facts:
        expect = f"the {s} {emerge_v3(v, already_3sg=None)} {prep[c]} the {o}"
        n_ans += 1
        out = prod.speak(decision("ANSWER", construction=c, subject=s, verb=v, obj=o))
        ok = (out.get("surface") == expect)
        n_ok += int(ok)
        print(f"  {c}({s},{v},{o}) -> SPIKING-BROCA -> \"{out.get('surface')}\"  "
              f"{'[exact, order+words(incl productive 3sg) ON SPIKES]' if ok else '[got != '+expect+']'}", flush=True)
    prodA = prod.production_count
    prod.speak(decision("ABSTAIN"))
    moat_ok = (prod.production_count == prodA == prod0 + n_ans)
    print(f"  ABSTAIN -> producer NOT invoked (moat: {prod.production_count == prodA})", flush=True)
    return {"n_ok": n_ok, "n_ans": n_ans, "moat_ok": moat_ok, "have": have}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    print(f"[spiking-Broca PP answer] the spatial answer 'the SUBJ VERB:3sg to/on the OBJ' ON SPIKES | seed={a.seed}", flush=True)
    r = run(a.seed)
    go = r["n_ans"] > 0 and r["n_ok"] == r["n_ans"] and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the PP spatial answer is produced FULLY ON SPIKES "
          f"({r['n_ok']}/{r['n_ans']} exact: the C_PPGOAL/C_PPLOC order via the registry producer + every word incl. the "
          f"3sg verb (PRODUCTIVE inflection) + the prepositions to/on via the A->W); gate-first moat holds "
          f"({r['moat_ok']}). Spiking generation broadened to SPATIAL relations.", flush=True)


if __name__ == "__main__":
    main()
