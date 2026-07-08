"""SPOKEN CANCELLATION: the talkable brain SPEAKS the override for an exception member.

Composes two validated pieces over the emergent real-corpus reasoner:
  * CANCELLATION (6-seed GO, `_realcorpus_cancellation_derisk`): a member's OWN property overrides its
    category's inherited one, via a regulated graded apical drive.
  * FRAME SPEECH (GO, `_realcorpus_full_frame_speech_derisk`): the brain speaks "the <subject> can <verb>"
    with the content words produced ON SPIKES (the breadth concept-pool A->W).

Flow: discover the ANIMAL cluster from TinyStories -> teach the class a property spoken as verb V1
("animals can run") -> teach ONE spellable-animal member an EXCEPTION spoken as verb V2 ("but the fish
sleeps") -> ASK about each held-out spellable animal:
  * the EXCEPTION member  -> the reasoner cancels -> SPEAK its OWN property "the fish can sleep" (V2, ON SPIKES)
  * an inheriting member  -> SPEAK the inherited property "the frog can run" (V1, ON SPIKES)
  * an unknown word       -> "I don't know" (gate-first moat)

So the brain distinguishes an exception from inheriting members and SPEAKS the correct property. Reuses
the breadth A->W (`bridges/breadth_aw/seed42.simstate.h5`, 8 animals + 6 verbs + the/can). NO sim/ edit.
Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_cancellation_derisk import CancellingConsole, _pick_pos, _ANIMALS
from research.runners._realcorpus_inheritance_rung4_conversation_derisk import _splits, _coherence
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners.corpus_stream import load_token_stream_multi


def run(corpus_path, K, n_clusters, bridge_path, v1, v2, seed):
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    con = CancellingConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)   # numpy reasoner
    from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL
    speaker = ConceptFrameSpeaker(bridge_path, seed=seed, vocab=VOCAB, word_to_pool=WORD_TO_POOL)  # cupy A->W
    spellable_animals = set(speaker.vocab) & _ANIMALS
    if v1 not in speaker.vocab or v2 not in speaker.vocab:
        print(f"  verbs {v1}/{v2} must be in the A->W vocab {sorted(speaker.vocab)}"); return None

    if len(con.cat_ids) < 2:
        print("  <2 categories discovered"); return None
    coh = {c: _coherence(con, c) for c in con.cat_ids}
    pos = _pick_pos(con, coh)
    taught_by_cat, held_by_cat = _splits(con.members, con.cat_ids, con.rng)
    con.teach(taught_by_cat)                                            # class property (spoken as v1)
    held = held_by_cat[pos]

    # the EXCEPTION = a HELD-OUT, SPELLABLE-ANIMAL member of pos that inherits before the exception.
    exc = next((w for w in held if w in spellable_animals and con.ask_class(pos, w) == "yes"), None)
    if exc is None:
        print(f"  seed {seed}: no held-out spellable-animal member inherits in cluster {pos} "
              f"(members {[w for w in con.members[pos] if w in spellable_animals]}) -- try another seed/K"); return None
    con.teach_exception_adaptive(exc, "own", margin=2.0)               # its OWN property (spoken as v2)
    print(f"  DISCOVERED animal cluster {pos}: spellable {[w for w in con.members[pos] if w in spellable_animals]}; "
          f"TAUGHT class->'{v1}', EXCEPTION '{exc}'->'{v2}'", flush=True)

    def answer(word):
        """reason (inherit / exception / other / unknown) -> SPEAK the right property on spikes / abstain."""
        if word not in con.row_of:
            return "I don't know", "moat"
        pred = con._predict_all(word)
        if pred == ("exc", "own"):
            frame, _ = speaker.speak_frame(word, v2)                   # OVERRIDE: its own property, ON SPIKES
            return frame, "override"
        if pred == ("cat", pos):
            frame, _ = speaker.speak_frame(word, v1)                   # INHERITED class property, ON SPIKES
            return frame, "inherit"
        return f"(reasoner: {word} not in the taught category)", "other"

    # demo: the exception member + up to 2 inheriting spellable-animal members + a moat probe
    inheriting = [w for w in held if w in spellable_animals and w != exc and con.ask_class(pos, w) == "yes"][:2]
    queries = [exc] + inheriting + ["zzzqqx"]
    transcript, spoke_override, spoke_inherit = [], 0, 0
    for q in queries:
        out, kind = answer(q)
        tag = {"override": "-> SPOKE OVERRIDE ON SPIKES", "inherit": "-> SPOKE INHERITED ON SPIKES",
               "moat": "[MOAT: unknown -> I don't know]"}.get(kind, f"[{kind}]")
        print(f"  ask 'does the {q} {v1}?' -> \"{out}\"  {tag}", flush=True)
        transcript.append({"q": q, "out": out, "kind": kind})
        spoke_override += int(kind == "override" and v2 in out)
        spoke_inherit += int(kind == "inherit" and v1 in out)
    moat_ok = all(t["kind"] == "moat" for t in transcript if t["q"] == "zzzqqx")
    # the override frame must speak V2 (its own) and NOT V1 (the inherited) -> a genuine spoken cancellation
    override_frame = next((t["out"] for t in transcript if t["kind"] == "override"), "")
    cancel_spoken = (v2 in override_frame) and (v1 not in override_frame)
    return {"seed": seed, "pos": pos, "exc": exc, "spoke_override": spoke_override,
            "spoke_inherit": spoke_inherit, "cancel_spoken": bool(cancel_spoken), "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--bridge", default="bridges/breadth_aw/seed42.simstate.h5")
    ap.add_argument("--v1", default="run", help="the inherited class property (spoken verb)")
    ap.add_argument("--v2", default="sleep", help="the exception member's OWN property (spoken verb)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[spoken cancellation] corpus={a.corpus_path} K={a.K} class-verb='{a.v1}' exception-verb='{a.v2}'", flush=True)
    r = run(a.corpus_path, a.K, a.n_clusters, a.bridge, a.v1, a.v2, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE"); return
    go = r["spoke_override"] >= 1 and r["cancel_spoken"] and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the brain SPEAKS the OVERRIDE for the exception member "
          f"('{r['exc']}' can {a.v2}, its own property ON SPIKES, NOT the inherited {a.v1}), speaks the inherited "
          f"property for others ({r['spoke_inherit']} spoken), and ABSTAINS on the unknown (moat {r['moat_ok']}). "
          f"Composes the 6-seed cancellation GO + the frame-speech GO.", flush=True)


if __name__ == "__main__":
    main()
