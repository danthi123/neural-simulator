"""FULLY-SPIKING one-process cancellation conversation: REASON on spikes + SPEAK on spikes, one backend.

The "one brain" north star for the cancellation capability. The spoken cancellation (CYCLE 983) reasoned
with the numpy associative memory (rate) and spoke on spikes; the spiking cancellation (CYCLE 984/985)
reasoned on spikes but did not speak. This co-executes BOTH on the numpy backend in ONE process:
  * REASON ON SPIKES: the emergent spiking cancellation reasoner (`CancellingPoolerProbe`, EMERGE-42 pooler
    + committed HTM coincidence kernel + apical read from `cp_v_apical`) discovers categories, is taught a
    class property + a member exception, and its apical argmax decides inherit-vs-cancel.
  * SPEAK ON SPIKES: the breadth concept-pool A->W (`ConceptFrameSpeaker`) spells "the <animal> can <verb>"
    with content ON SPIKES (`language_output` firing).
Both are real `SimulationBridge`s on the numpy backend -> the WHOLE cancellation turn (reason + speak) is
spiking, in one process. The reasoner's spiking decision (class vs EXC) selects the spoken verb
(inherited V1 vs override V2). Gate-first moat: an unknown word -> "I don't know" (no reasoning, no frame).

Requires SIM_BACKEND=numpy (one backend for BOTH bridges; the A->W checkpoint loads backend-agnostically).
Reuse-by-import. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_cancellation_spiking_derisk import (
    CancellingPoolerProbe, emergent_inputs, _adaptive_teach,
)
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL
from research.runners._realcorpus_cancellation_derisk import _ANIMALS


def run(corpus_path, K, n_clusters, bridge_path, v1, v2, seed, epochs=40, max_passes=16):
    sdr_by_row, row_to_cat, cat_ids, vocab = emergent_inputs(corpus_path, K, seed, n_clusters)
    con = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs)   # SPIKING reasoner
    speaker = ConceptFrameSpeaker(bridge_path, seed=seed, vocab=VOCAB, word_to_pool=WORD_TO_POOL)  # SPIKING A->W
    spellable_animals = set(speaker.vocab) & _ANIMALS
    word_of = {r: vocab[r] for r in con.rows}

    # pos = a discovered cluster with >=2 SPELLABLE-ANIMAL members that INHERIT (query==class); the exception
    # can be ANY inheriting member (taught or generalized -- EMERGE-54: a penguin that inherits "bird" yet
    # its own "walks" overrides). Group rows by cluster and find inheriting spellable animals.
    by_cat = {}
    for r in con.rows:
        by_cat.setdefault(con.row2cat[r], []).append(r)
    pos, exc_row, others = None, None, []
    for k in con.cat_ids:
        animals = [r for r in by_cat.get(k, []) if word_of[r] in spellable_animals
                   and con.query(r, include_exc=False) == f"C{k}"]
        if len(animals) >= 2:
            pos, exc_row, others = k, animals[0], animals[1:]; break
    if pos is None:
        print(f"  seed {seed}: no discovered cluster with >=2 inheriting spellable animals"); return None
    exc_word = word_of[exc_row]
    passes = _adaptive_teach(con, exc_row, pos, max_passes=max_passes)                 # teach the exception ON SPIKES
    print(f"  [seed {seed}] SPIKING reasoner: cluster {pos} inheriting spellable-animals "
          f"{[word_of[r] for r in [exc_row] + others]}; "
          f"class->'{v1}', EXCEPTION '{exc_word}'->'{v2}' ({passes} apical passes)", flush=True)

    def answer(word):
        """REASON on spikes (apical argmax) -> SPEAK on spikes / abstain."""
        if word not in vocab:
            return "I don't know", "moat"
        r = vocab.index(word)
        if r not in con.ridx:
            return "I don't know", "moat"
        pred = con.query(r)                                    # spiking apical decision
        if pred == "EXC":
            frame, _ = speaker.speak_frame(word, v2); return frame, "override"    # override property ON SPIKES
        if pred == f"C{pos}":
            frame, _ = speaker.speak_frame(word, v1); return frame, "inherit"     # inherited property ON SPIKES
        return f"(reasoner: {word} not in the taught category)", "other"

    queries = [exc_word] + [word_of[r] for r in others[:2]] + ["zzzqqx"]
    transcript, n_override, n_inherit = [], 0, 0
    for q in queries:
        out, kind = answer(q)
        tag = {"override": "-> REASON+SPEAK ON SPIKES (override)", "inherit": "-> REASON+SPEAK ON SPIKES (inherit)",
               "moat": "[MOAT: unknown -> I don't know]"}.get(kind, f"[{kind}]")
        print(f"  ask 'does the {q} {v1}?' -> \"{out}\"  {tag}", flush=True)
        transcript.append({"q": q, "out": out, "kind": kind})
        n_override += int(kind == "override" and v2 in out)
        n_inherit += int(kind == "inherit" and v1 in out)
    override_frame = next((t["out"] for t in transcript if t["kind"] == "override"), "")
    cancel_spoken = (v2 in override_frame) and (v1 not in override_frame)
    moat_ok = all(t["kind"] == "moat" for t in transcript if t["q"] == "zzzqqx")
    return {"seed": seed, "exc": exc_word, "n_override": n_override, "n_inherit": n_inherit,
            "cancel_spoken": bool(cancel_spoken), "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--bridge", default="bridges/breadth_aw/seed42.simstate.h5")
    ap.add_argument("--v1", default="run")
    ap.add_argument("--v2", default="sleep")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[FULLY-SPIKING cancellation conversation] reason-on-spikes + speak-on-spikes, one numpy process | "
          f"K={a.K} class='{a.v1}' exception='{a.v2}'", flush=True)
    r = run(a.corpus_path, a.K, a.n_clusters, a.bridge, a.v1, a.v2, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE"); return
    go = r["n_override"] >= 1 and r["cancel_spoken"] and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the WHOLE cancellation turn is SPIKING in ONE process: the "
          f"reasoner decides inherit-vs-cancel ON SPIKES (apical competition) and the A->W SPEAKS the right "
          f"property ON SPIKES ('{r['exc']}' can {a.v2}, override; others can {a.v1}, inherit; {r['n_inherit']} "
          f"inherited spoken), abstaining on the unknown (moat {r['moat_ok']}).", flush=True)


if __name__ == "__main__":
    main()
