"""KNOWLEDGE-half of breadth, the COMPLETE talkable loop: discover -> reason -> speak a fluent FRAME.

Ties the whole arc into one pipeline: the emergent reasoner (rung-4) DISCOVERS categories from a real
corpus, is TAUGHT a property, decides a HELD-OUT word inherits it, and the brain SPEAKS a fluent
grounded frame ("the cat can go") with the content word ON SPIKES (the v16 concept-pool A->W), or
ABSTAINS on an unknown word (gate-first moat). The complete discover -> reason -> fluent-speak loop.

Honest scope: the reasoner + the v16 A->W overlap at the spellable words (dog/cat/apple/river + the
verbs), so the loop is demonstrated over that overlap; the broad-vocab version = an A->W retrain on the
reasoner's full discovered vocab. Content word on spikes; closed-class the/can host-rendered
(EMERGE-68 spiking function words = follow-on). Reuse-by-import. NO sim/ edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_inheritance_rung4_conversation_derisk import RealCorpusConsole, _splits
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners.corpus_stream import load_token_stream_multi


def run(corpus_path, K, n_clusters, bridge_path, verb, seed):
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    con = RealCorpusConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)  # numpy reasoner
    speaker = ConceptFrameSpeaker(bridge_path, seed=seed)                            # cupy A->W frame speaker
    spellable = set(speaker.vocab)

    # find a DISCOVERED cluster that contains >=2 spellable words (so we can teach one, hold out another)
    target_cat = None
    for c in con.cat_ids:
        sp = [w for w in con.members[c] if w in spellable]
        if len(sp) >= 2:
            target_cat = c; break
    if target_cat is None:
        print("  no discovered cluster with >=2 spellable members -- try a different K/n_clusters"); return None

    taught_by_cat, held_by_cat = _splits(con.members, con.cat_ids, con.rng)
    con.teach(taught_by_cat)                          # teach every cluster its distinct property
    print(f"  DISCOVERED a cluster with spellable members {[w for w in con.members[target_cat] if w in spellable]}; "
          f"TAUGHT it the property (spoken as the verb '{verb}')", flush=True)

    def answer(word):
        """The complete loop: reason (does word belong to the taught cluster?) -> speak a fluent frame / abstain."""
        if word not in con.row_of:
            return "I don't know", "moat"             # gate-first: not in the discovered vocab
        pred = con._predict_cat(word)                 # the emergent reasoner's decision
        if pred != target_cat:
            return f"(reasoner: {word} is not in that category)", "no"
        if word not in spellable:
            return f"(reasoner says yes, but '{word}' is outside the A->W vocab)", "unspellable"
        frame, _ = speaker.speak_frame(word, verb)    # SPEAK the fluent frame, content ON SPIKES
        return frame, "spoke"

    # demo: held-out spellable member of the taught cluster (expect a spoken frame) + an unknown (moat)
    held_spellable = [w for w in held_by_cat[target_cat] if w in spellable]
    queries = held_spellable[:2] + ["zzzqqx"]
    transcript = []
    for q in queries:
        out, kind = answer(q)
        tag = ("-> SPOKE ON SPIKES" if kind == "spoke" else
               "[MOAT: unknown -> I don't know]" if kind == "moat" else f"[{kind}]")
        print(f"  ask about '{q}' -> \"{out}\"  {tag}", flush=True)
        transcript.append({"q": q, "out": out, "kind": kind})
    spoke = [t for t in transcript if t["kind"] == "spoke"]
    moat_ok = all(t["kind"] == "moat" for t in transcript if t["q"] == "zzzqqx")
    return {"n_spoke": len(spoke), "moat_ok": moat_ok, "target_cat": target_cat, "transcript": transcript}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--bridge", default="bridges/v16/seed42.simstate.h5")
    ap.add_argument("--verb", default="go")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[full pipeline: discover -> reason -> fluent frame] corpus={a.corpus_path} K={a.K}", flush=True)
    r = run(a.corpus_path, a.K, a.n_clusters, a.bridge, a.verb, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE"); return
    go = r["n_spoke"] >= 1 and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the COMPLETE loop: the brain DISCOVERS categories from a "
          f"real corpus, REASONS (a held-out word inherits its category's property), and SPEAKS a fluent grounded "
          f"FRAME with the content word ON SPIKES ({r['n_spoke']} spoken), or ABSTAINS on the unknown (gate-first "
          f"moat {r['moat_ok']}). Content on spikes; the/can host-rendered; overlap vocab (broad = A->W retrain).",
          flush=True)


if __name__ == "__main__":
    main()
