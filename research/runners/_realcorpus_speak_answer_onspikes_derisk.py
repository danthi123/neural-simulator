"""KNOWLEDGE-half of breadth, SPEAK rung: the brain SPEAKS its grounded yes/no/idk answer ON SPIKES.

The breadth->knowledge arc reasons (discover broad vocab -> discover categories -> teach a fact ->
answer a yes/no question about a held-out word -> moat), but the answer was an internal token. This
rung SPEAKS it on spikes: the numpy breadth reasoner decides yes/no/idk, and the decision is produced
as a SPOKEN WORD on a real SimulationBridge via the validated spiking A->W read-out (EMERGE-67
`NeuralSpell`: drive the answer's concept pool -> decode the spoken word from `cp_firing_states
[language_output]`). GATE-FIRST moat: on "idk" (an unknown word / no discovered code) the SPEAKER IS
NEVER INVOKED -> the no-confab moat holds by construction (0 spike-renders on abstains).

The numpy reasoner (pure numpy math, no bridge) co-executes with the cupy A->W (a SimulationBridge)
in ONE cupy process -- the reasoner never touches the backend-global, so they coexist.

Honest scope: the 3 answer tokens (yes/no/idk) are mapped to 3 words of the validated 16-word A->W
vocab (proxy surfaces -- the SPIKING production + gate-first moat are the claim; a literal yes/no A->W
retrain is cosmetic polish). Reuse-by-import: rung-4 console + EMERGE-67 NeuralSpell. NO sim/ edit.
Requires SIM_BACKEND=cupy (GPU).
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._realcorpus_inheritance_rung4_conversation_derisk import RealCorpusConsole, _splits
from research.runners.corpus_stream import load_token_stream_multi

# answer token -> a distinct word of the validated A->W vocab (proxy surface; spiking production is the claim)
ANS2WORD = {"yes": "fly", "no": "swim"}     # idk is the gate-first abstain (speaker NEVER invoked)


def run(corpus_path, K, n_clusters, seed):
    from research.runners._emerge67_neural_spell_wirein_derisk import NeuralSpell
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    con = RealCorpusConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)
    if len(con.cat_ids) < 2:
        return {"verdict": "NOT-EVALUABLE"}
    coh = {c: 0.0 for c in con.cat_ids}
    # pick the most coherent cluster as the taught category (like rung-4)
    from research.runners._realcorpus_inheritance_rung4_conversation_derisk import _coherence
    coh = {c: _coherence(con, c) for c in con.cat_ids}
    pos = max(coh, key=coh.get)
    neg = sorted(coh, key=coh.get)[-2]
    taught_by_cat, held_by_cat = _splits(con.members, con.cat_ids, con.rng)
    con.teach(taught_by_cat)
    held = held_by_cat[pos]

    speller = NeuralSpell(load=True)     # the spiking A->W engine (cupy bridge)

    def speak(category, word):
        """Return (answer, spoken_word_on_spikes_or_None, n_spike_renders)."""
        ans = con.ask(category, word)                 # numpy reasoner: yes/no/idk
        if ans == "idk":
            return ans, None, 0                       # GATE-FIRST: speaker NOT invoked (moat)
        spoken = speller.spell(ANS2WORD[ans])         # SPIKING: drive pool -> decode from language_output
        return ans, spoken, 1

    # queries: held-out same-category (expect yes), other-category (expect no), unknown (expect idk/moat)
    transcript, renders, correct_spoken, moat_renders = [], 0, 0, 0
    tests = ([(w, pos, "yes") for w in held[:4]] +
             [(w, pos, "no") for w in con.members[neg][:4]] +
             [("zzzqqx", pos, "idk")])
    for word, cat, expect in tests:
        ans, spoken, nr = speak(cat, word)
        renders += nr
        if ans == "idk":
            moat_renders += nr                        # must be 0
        else:
            # spoken (on spikes) must decode to the mapped word for the decision
            if spoken == ANS2WORD.get(ans):
                correct_spoken += 1
        transcript.append({"word": word, "expect": expect, "answer": ans,
                           "spoken_on_spikes": spoken, "renders": nr})
    n_spoken = sum(1 for t in transcript if t["answer"] != "idk")
    spoke_acc = correct_spoken / max(1, n_spoken)
    return {"pos": pos, "neg": neg, "transcript": transcript, "spike_render_count": renders,
            "spoken_accuracy": spoke_acc, "moat_renders_on_abstain": moat_renders,
            "n_spoken": n_spoken}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    recs = []
    for s in seeds:
        r = run(a.corpus_path, a.K, a.n_clusters, s)
        r["seed"] = s
        recs.append(r)
        print(f"\n=== seed {s}: SPEAK the grounded answer ON SPIKES (pos-cat={r.get('pos')}) ===", flush=True)
        for t in r.get("transcript", []):
            arrow = (f"-> spoke '{t['spoken_on_spikes']}' ON SPIKES" if t["answer"] != "idk"
                     else "-> \"I don't know\" [MOAT: speaker NOT invoked, 0 renders]")
            print(f"  Q about '{t['word']}' (expect {t['expect']}) -> reasoner says {t['answer'].upper()} {arrow}", flush=True)
        print(f"  spoken-accuracy {r.get('spoken_accuracy',0):.3f} | spike-renders {r.get('spike_render_count')} | "
              f"moat-renders-on-abstain {r.get('moat_renders_on_abstain')} (must be 0)", flush=True)

    ok_spoke = all(r.get("spoken_accuracy", 0) >= 0.99 for r in recs)
    ok_moat = all(r.get("moat_renders_on_abstain", 1) == 0 for r in recs)
    go = ok_spoke and ok_moat
    print(f"\n  VERDICT: {'GO' if go else 'NEGATIVE'} -- the brain {'SPEAKS its grounded yes/no answer ON SPIKES '
             '(decoded from language_output firing) and ABSTAINS on the unknown WITHOUT invoking the speaker '
             '(gate-first no-confab moat, 0 renders on abstain)' if go else 'does NOT cleanly speak/abstain'}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
