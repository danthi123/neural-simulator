"""FULLY-SPIKING GENERATION of the real-corpus property answer: the SENTENCE STRUCTURE ('the X can Y') is
produced ON SPIKES by the EMERGE-59 spiking-Broca producer (slot order via competitive queuing on a slot
bridge), and every WORD by the breadth concept-pool A->W -- replacing the host f-string template. "Simulate
Broca": the order is a spiking read-out, not a host literal.

The real-corpus reasoner decides (subject, verb) [inherit -> class verb; cancel -> exception verb]; that
decision is handed to `BrocaProducer` GATE-FIRST (abstain -> the producer is NEVER invoked -> moat by
construction), which renders the F_MODAL frame ("the <subj> can <verb>") with the slot ORDER on spikes and
each slot spelled by the A->W read-out. Both the slot bridge and the A->W bridge run on the numpy backend
in ONE process. Reuse-by-import. NO `sim/` edit. Requires SIM_BACKEND=numpy.
"""
from __future__ import annotations
import argparse

from research.runners._emerge65_self_organized_producer_derisk import SelfOrganizedProducer
from research.runners._emerge62_discover_function_words_derisk import build_stream
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL


def run(seed=42, aw_seed=42):
    aw = ConceptFrameSpeaker("bridges/breadth_aw/seed42.simstate.h5", seed=aw_seed,
                             vocab=VOCAB, word_to_pool=WORD_TO_POOL)   # the A->W spell callback (on spikes)
    # the EMERGE-65 SELF-ORGANIZED producer (exact order via MinedInventoryFrameSlotCQ + the EMERGE-61 wash-out,
    # position-independent by construction -- fixes the base-CQ order tail); built from the corpus stream.
    sop = SelfOrganizedProducer(seed).build_from_corpus(build_stream(seed))
    producer = sop.producer(spell=aw.spell)                          # gate-first; every slot spelled by the A->W

    # decisions the real-corpus reasoner would hand over (inherit -> class verb; cancel -> exception verb) + a moat
    decisions = [
        {"gate": "ANSWER", "frame": "F_MODAL", "subject": "bird", "verb": "sleep", "expect": "the bird can sleep"},
        {"gate": "ANSWER", "frame": "F_MODAL", "subject": "frog", "verb": "run",   "expect": "the frog can run"},
        {"gate": "ANSWER", "frame": "F_MODAL", "subject": "dog",  "verb": "eat",   "expect": "the dog can eat"},
        {"gate": "ABSTAIN", "expect": None},
    ]
    n_ok = n_ans = 0
    prod_before = producer.production_count
    for d in decisions:
        out = producer.speak(d)
        if d["gate"] == "ABSTAIN":
            moat = (out["produced"] is False and producer.production_count == prod_before)
            print(f"  ABSTAIN -> produced={out['produced']} (producer NOT invoked: {moat})  [moat]", flush=True)
            continue
        prod_before = producer.production_count
        n_ans += 1
        ok = (out["surface"] == d["expect"])
        n_ok += int(ok)
        print(f"  reason({d['subject']},{d['verb']}) -> SPIKING-BROCA -> \"{out['surface']}\"  "
              f"{'[exact, order+words ON SPIKES]' if ok else '[got != '+d['expect']+']'}", flush=True)
    moat_ok = (producer.production_count == n_ans)                    # exactly n_ans productions (0 on the abstain)
    return {"n_ok": n_ok, "n_ans": n_ans, "moat_ok": moat_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[spiking-Broca property answer] the real-corpus property answer STRUCTURE produced ON SPIKES "
          f"(slot order via competitive queuing) + words via the A->W | seed={a.seed}", flush=True)
    r = run(a.seed)
    go = r["n_ok"] == r["n_ans"] and r["n_ans"] > 0 and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the real-corpus property answer 'the <subj> can <verb>' is "
          f"produced FULLY ON SPIKES ({r['n_ok']}/{r['n_ans']} exact: the SLOT ORDER via the spiking-Broca competitive "
          f"queuing + every WORD via the A->W read-out), replacing the host template; gate-first moat holds "
          f"(0 productions on abstain: {r['moat_ok']}). Fully-spiking sentence generation.", flush=True)


if __name__ == "__main__":
    main()
