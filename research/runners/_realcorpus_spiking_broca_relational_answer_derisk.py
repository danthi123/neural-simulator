"""FULLY-SPIKING GENERATION of the real-corpus RELATIONAL answer ("the dog eats the cat"): the transitive
C_TRANS slot ORDER ("the SUBJ VERB:3sg the OBJ") is produced ON SPIKES by the EMERGE-72/74 signature-keyed
registry producer (competitive queuing + wash-out), and EVERY word -- including the 3sg verb surface --
spelled ON SPIKES by the 3-bridge A->W (BRIDGE-1 animals + BRIDGE-3 3sg verbs). Extends the property-answer
generation (F_MODAL, EMERGE frames) to the transitive SVO construction. "Simulate Broca" for the relational
answer.

The C_TRANS construction is MINED from the corpus stream (EMERGE-74's `build_stream_svo`); the FILLERS are
the console's relational facts (animal VERB animal). `RegistryBrocaProducer(cq, spell=multi_bridge_speaker
.spell)` GATE-FIRST (abstain -> the producer is NEVER invoked -> moat by construction). The 3sg inflection
follows EMERGE-74's standard (emerge_v3 produces the surface; the SURFACE is then spelled on spikes). NO
`sim/` edit. Requires SIM_BACKEND=numpy + the BRIDGE-1/3 A->W checkpoints.
"""
from __future__ import annotations
import argparse

from research.runners._emerge72_construction_registry_derisk import (
    RegistryBrocaProducer, decision,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (
    SVOConstructionRegistry, build_stream_svo, emerge_v3,
)
from research.runners._realcorpus_multi_bridge_speaker import MultiBridgeFrameSpeaker


def run(seed=42, aw_seed=42):
    speaker = MultiBridgeFrameSpeaker(seed=aw_seed)                 # the 3-bridge A->W (animals + 3sg verbs + objects)
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    if "C_TRANS" not in reg.registered_fits():
        return {"n_ok": 0, "n_ans": 0, "moat_ok": False, "trans_mined": False}
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq, spell=speaker.spell)          # gate-first; every slot spelled on spikes

    # the console's relational facts (animal VERB animal), all fillers covered by the 3-bridge A->W
    facts = [("dog", "eat", "cat"), ("wolf", "chase", "rabbit"), ("fox", "see", "bird"), ("bear", "like", "fish")]
    n_ok = n_ans = 0
    prod0 = prod.production_count
    for (s, v, o) in facts:
        expect = f"the {s} {emerge_v3(v, already_3sg=None)} the {o}"     # C_TRANS surface, e.g. "the dog eats the cat"
        n_ans += 1
        out = prod.speak(decision("ANSWER", construction="C_TRANS", subject=s, verb=v, obj=o))
        ok = (out.get("surface") == expect)
        n_ok += int(ok)
        print(f"  reason({s},{v},{o}) -> SPIKING-BROCA -> \"{out.get('surface')}\"  "
              f"{'[exact, order+words ON SPIKES]' if ok else '[got != '+expect+']'}", flush=True)
    # gate-first moat: ABSTAIN never invokes the producer
    prodA = prod.production_count
    prod.speak(decision("ABSTAIN"))
    moat_ok = (prod.production_count == prodA == prod0 + n_ans)
    print(f"  ABSTAIN -> producer NOT invoked (0 productions on abstain: {prod.production_count == prodA})  [moat]", flush=True)
    return {"n_ok": n_ok, "n_ans": n_ans, "moat_ok": moat_ok, "trans_mined": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[spiking-Broca RELATIONAL answer] the real-corpus transitive answer 'the SUBJ VERB:3sg the OBJ' "
          f"produced ON SPIKES (C_TRANS order via competitive queuing) + words via the 3-bridge A->W | seed={a.seed}", flush=True)
    r = run(a.seed)
    go = r["trans_mined"] and r["n_ok"] == r["n_ans"] and r["n_ans"] > 0 and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the real-corpus RELATIONAL answer 'the <subj> <verb>s the <obj>' "
          f"is produced FULLY ON SPIKES ({r['n_ok']}/{r['n_ans']} exact: the C_TRANS SLOT ORDER via the spiking-Broca "
          f"registry producer + every WORD incl. the 3sg verb via the 3-bridge A->W read-out); gate-first moat holds "
          f"(0 productions on abstain: {r['moat_ok']}). Fully-spiking transitive sentence generation.", flush=True)


if __name__ == "__main__":
    main()
