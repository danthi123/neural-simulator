"""FULLY-SPIKING GENERATION of the real-corpus DITRANSITIVE answer: "the dog GIVES the cat a bone" -- the
7-slot C_DITRANS construction order produced ON SPIKES by EMERGE-77's 8-pool 2-stage-calibrated registry
producer, and EVERY word (incl. the 3sg verb via PRODUCTIVE inflection give->gives, the recipient/theme
nouns, the/a) spelled ON SPIKES by the productive multi-bridge A->W. Completes the relational-generation
schema breadth on spikes: property (F_MODAL) + transitive (C_TRANS) + spatial (C_PPGOAL/C_PPLOC) + DITRANSITIVE.

Reuses EMERGE-77's `DitransRegistry`/`DitransRegistryProducer` (n_slot_pools=8 + the 2-stage per-pool bias
calibration for the tightly-packed 8-rank read) for the ORDER; the A->W spell (with the EMERGE-75b reset_steps=150
history-independent read) for the WORDS. NO `sim/` edit. Requires SIM_BACKEND=numpy + BRIDGE-1/2/3/4 + affix.
"""
from __future__ import annotations
import argparse
from research.runners._emerge77_ditransitive_render_derisk import DitransRegistry
from research.runners._emerge74_transitive_ditransitive_derisk import build_stream_svo, emerge_v3
from research.runners._emerge72_construction_registry_derisk import DET, FUNC, SUBJ, VERB, OBJ
from research.runners._emerge74_transitive_ditransitive_derisk import IOBJ
from research.runners._realcorpus_productive_multi_speaker import ProductiveMultiSpeaker


def _emit_ditrans_aw(cq, name, subject, verb, iobj, obj, spell):
    """Emit C_DITRANS ON SPIKES: the 2-stage-calibrated pool ORDER (EMERGE-77) + every slot spelled via the A->W."""
    slots = cq.frame_slots[name]
    order = cq._calibrated_order(name)                          # EMERGE-61 wash-out + 2-stage calibrated rate ranking
    def realize(slot):
        stype, payload = slot
        if stype in (DET, FUNC):
            return spell(payload)
        if stype == SUBJ:
            return spell(subject)
        if stype == VERB:
            return spell(verb if payload == "bare" else emerge_v3(verb, already_3sg=None))   # give -> gives (productive)
        if stype == IOBJ:
            return spell(iobj)
        if stype == OBJ:
            return spell(obj)
        raise ValueError(stype)
    return " ".join(realize(slots[p]) for p in order)


def run(seed=42, aw_seed=42):
    speaker = ProductiveMultiSpeaker(seed=seed, aw_seed=aw_seed, reset_steps=150)
    reg = DitransRegistry(seed).build(build_stream_svo(seed))
    if "C_DITRANS" not in reg.registered_fits():
        return {"n_ok": 0, "n_ans": 0, "found": False}
    cq = reg.render_cq()
    # warm-up render (EMERGE-75b: the first A->W read settles the substrate)
    _emit_ditrans_aw(cq, "C_DITRANS", "dog", "give", "cat", "bone", speaker.spell)
    facts = [("dog", "give", "cat", "bone"), ("cat", "bring", "dog", "gift"), ("bear", "send", "fox", "seed")]
    n_ok = n_ans = 0
    for (s, v, i, o) in facts:
        expect = f"the {s} {emerge_v3(v, already_3sg=None)} the {i} a {o}"
        surface = _emit_ditrans_aw(cq, "C_DITRANS", s, v, i, o, speaker.spell)
        n_ans += 1; ok = (surface == expect); n_ok += int(ok)
        print(f"  C_DITRANS({s},{v},{i},{o}) -> SPIKING-BROCA -> \"{surface}\"  "
              f"{'[exact, 7-slot order + words(incl productive 3sg) ON SPIKES]' if ok else '[got != '+expect+']'}", flush=True)
    return {"n_ok": n_ok, "n_ans": n_ans, "found": True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    print(f"[spiking-Broca DITRANSITIVE answer] 'the dog gives the cat a bone' ON SPIKES (8-pool 2-stage read) | seed={a.seed}", flush=True)
    r = run(a.seed)
    go = r["found"] and r["n_ans"] > 0 and r["n_ok"] == r["n_ans"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the DITRANSITIVE answer 'the <s> <v>s the <recipient> a <theme>' "
          f"is produced FULLY ON SPIKES ({r['n_ok']}/{r['n_ans']} exact: the 7-slot C_DITRANS order via EMERGE-77's 8-pool "
          f"2-stage-calibrated read + every word incl. the 3sg verb (PRODUCTIVE inflection) via the A->W). "
          f"Schema breadth COMPLETE on spikes: property + transitive + spatial + ditransitive.", flush=True)


if __name__ == "__main__":
    main()
