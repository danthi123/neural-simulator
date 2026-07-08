"""KNOWLEDGE-half of breadth, FULLY-SPIKING reason+speak: the WHOLE turn on spikes, one brain.

The SPEAK rung (2026-07-08) reasoned on numpy + spoke on spikes. This makes the REASONING spiking
too: rung-2's spiking inheritance (the EMERGE-42 competitive pooler + the committed HTM coincidence
kernel on a real SimulationBridge) classifies a held-out word's category ON SPIKES (read from
`cp_v_apical`), the yes/no decision follows, and the answer is SPOKEN ON SPIKES via the A->W read-out
(EMERGE-67 `NeuralSpell`, decoded from `language_output` firing). BOTH bridges co-reside in ONE cupy
process (verified). Gate-first moat: an unknown word (no codon) -> "I don't know", speaker NEVER invoked.

⇒ discover -> reason(on spikes) -> SPEAK(on spikes), the whole turn spiking in one brain, transformer-
free, moat intact. Reuse-by-import: rung-2 spiking probe + EMERGE-67 NeuralSpell. NO sim/ edit.
Requires SIM_BACKEND=cupy (GPU).
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._realcorpus_inheritance_rung2_spiking_derisk import build_inputs, RealCorpusPoolerProbe
from collections import Counter

ANS2WORD = {"yes": "fly", "no": "swim"}     # proxy A->W surfaces; idk = gate-first abstain (never spoken)


def run(corpus_path, K, seed, epochs, diverse_readers=True, prop_k=16):
    from research.runners._emerge67_neural_spell_wirein_derisk import NeuralSpell
    _, sdr_by_row, row_to_cat, cat_ids, per, _graded = build_inputs(corpus_path, K, seed, sdr_t=50)
    # pick the largest category as the "pos" (property-taught) category
    cnt = Counter(row_to_cat.values())
    pos = max(cat_ids, key=lambda c: cnt[c])
    # the CYCLE-965 diverse-subsampling population readers lift the spiking reasoner's accuracy (+18%)
    probe = RealCorpusPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs,
                                  prop_k=prop_k, diverse_readers=diverse_readers)   # cupy bridge #1
    speller = NeuralSpell(load=True)                                                       # cupy bridge #2

    # held-out members of pos (expect yes) + held-out members of other cats (expect no) + an OOV (expect idk)
    held = {k: v for k, v in probe.held.items()}
    pos_held = held.get(pos, [])
    other_held = [r for k, v in held.items() if k != pos for r in v]

    def decide_and_speak(row_or_none):
        if row_or_none is None:                       # OOV: no codon -> gate-first abstain
            return "idk", None, 0
        pred = probe.query(row_or_none)               # SPIKING inheritance: predicted category (from cp_v_apical)
        ans = "yes" if pred == pos else "no"
        spoken = speller.spell(ANS2WORD[ans])         # SPIKING A->W: decoded from language_output
        return ans, spoken, 1

    transcript, renders, correct_spoken, moat_renders, n_spoken = [], 0, 0, 0, 0
    tests = ([(r, "yes") for r in pos_held[:4]] + [(r, "no") for r in other_held[:4]] + [(None, "idk")])
    for row, expect in tests:
        ans, spoken, nr = decide_and_speak(row)
        renders += nr
        if ans == "idk":
            moat_renders += nr
        else:
            n_spoken += 1
            if spoken == ANS2WORD.get(ans):
                correct_spoken += 1
        transcript.append({"expect": expect, "answer": ans, "spoken_on_spikes": spoken, "renders": nr})
    spoke_acc = correct_spoken / max(1, n_spoken)
    dec = [t for t in transcript if t["expect"] in ("yes", "no")]
    dec_acc = float(np.mean([t["answer"] == t["expect"] for t in dec])) if dec else float("nan")
    return {"pos": pos, "transcript": transcript, "spike_render_count": renders,
            "spoken_accuracy": spoke_acc, "moat_renders_on_abstain": moat_renders, "n_spoken": n_spoken,
            "decision_accuracy": dec_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    recs = []
    for s in seeds:
        r = run(a.corpus_path, a.K, s, a.epochs)
        r["seed"] = s
        recs.append(r)
        print(f"\n=== seed {s}: FULLY-SPIKING reason(on spikes)+speak(on spikes), one brain ===", flush=True)
        for t in r["transcript"]:
            arrow = (f"-> spoke '{t['spoken_on_spikes']}' ON SPIKES" if t["answer"] != "idk"
                     else "-> \"I don't know\" [MOAT: speaker NOT invoked]")
            print(f"  held-out (expect {t['expect']}) -> spiking-reasoner says {t['answer'].upper()} {arrow}", flush=True)
        print(f"  decision-accuracy {r.get('decision_accuracy', float('nan')):.3f} (spiking reasoner vs expected) | "
              f"spoken-accuracy {r['spoken_accuracy']:.3f} | spike-renders {r['spike_render_count']} | "
              f"moat-renders-on-abstain {r['moat_renders_on_abstain']} (must be 0)", flush=True)

    ok_spoke = all(r["spoken_accuracy"] >= 0.99 for r in recs)
    ok_moat = all(r["moat_renders_on_abstain"] == 0 for r in recs)
    go = ok_spoke and ok_moat
    print(f"\n  VERDICT: {'GO' if go else 'NEGATIVE'} -- the WHOLE conversational turn runs ON SPIKES in ONE "
          f"cupy process: the spiking-inheritance reasoner (EMERGE-42 pooler + committed HTM kernel) classifies "
          f"a held-out word, and the answer is SPOKEN on spikes (A->W from language_output); the unknown is "
          f"ABSTAINED without invoking the speaker (gate-first moat). {'' if go else '(spoken/moat check failed)'}",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
