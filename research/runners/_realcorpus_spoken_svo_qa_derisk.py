"""SPOKEN relational SVO Q&A: the talkable brain SPEAKS the answer to "what does the dog eat? → fish".

Ties the relational SVO capability (CYCLE 988, 6-seed GO: store/answer SVO facts over the brain's OWN
real-corpus codes + moat) to SPEECH (the breadth concept-pool A->W): store speakable facts (animal-eat-
animal), ask a relational question, RECOVER the object by FHRR unbind, and SPEAK it ON SPIKES; abstain
on an unstored relation (gate-first moat).

Flow: pick spellable-animal facts present in the discovered vocab (dog eat fish, cat eat frog, ...) ->
store them (FHRR bind over real-corpus codes) -> ask "what does <subj> eat?" -> unbind the patient +
cleanup -> if it recovers a spellable object, SPEAK it ON SPIKES (A->W); ask an unstored relation ->
"I don't know" (moat). numpy FHRR reasoner + numpy A->W co-execute in one process. Requires
SIM_BACKEND=numpy. Reuse-by-import. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_svo_qa_derisk import SVOStore
from research.runners._realcorpus_svo_compose_probe import _phasors, _role
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL
from research.runners._realcorpus_cancellation_derisk import _ANIMALS
from research.runners.corpus_stream import load_token_stream_multi


def run(corpus_path, K, bridge_path, verb, seed, aw_seed=42):
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    vocab, gfreq = discover_vocab(stories, K)
    row_of = {w: i for i, w in enumerate(vocab)}
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    rows = list(range(len(vocab)))
    Z = _phasors(codes, rows, seed)
    store = SVOStore(Z, rows, (_role(rng), _role(rng), _role(rng)))

    # build the A->W speaker at the CHECKPOINT's seed (42), NOT the reasoner's seed -- save_checkpoint does
    # not persist firing thresholds (CLAUDE.md gotcha), so a mismatched build seed perturbs the decode.
    speaker = ConceptFrameSpeaker(bridge_path, seed=aw_seed, vocab=VOCAB, word_to_pool=WORD_TO_POOL)
    spellable_animals = [a for a in _ANIMALS if a in row_of and a in speaker.vocab]
    if verb not in row_of or verb not in speaker.vocab:
        print(f"  verb '{verb}' not in discovered vocab / A->W"); return None
    if len(spellable_animals) < 4:
        print(f"  need >=4 spellable animals in the discovered vocab (have {spellable_animals})"); return None

    # store animal-VERB-animal facts over spellable animals: (a0 verb a1), (a2 verb a3), ...
    rng.shuffle(spellable_animals)
    pairs = [(spellable_animals[i], spellable_animals[i + 1]) for i in range(0, len(spellable_animals) - 1, 2)]
    facts = [(row_of[s], row_of[verb], row_of[o]) for (s, o) in pairs]
    for f in facts:
        store.store(*f)
    print(f"  STORED facts: " + ", ".join(f"'the {s} {verb}s {o}'" for (s, o) in pairs), flush=True)

    def ask(subj):
        """'what does <subj> <verb>?' -> recover the object (FHRR) -> SPEAK it on spikes / abstain (moat)."""
        if subj not in row_of:
            return "I don't know", "moat"
        o_row = store.answer_patient(row_of[subj], row_of[verb])
        if o_row is None:
            return "I don't know", "moat"
        obj = vocab[o_row]
        if obj not in speaker.vocab:
            return f"({obj}, not spellable)", "unspellable"
        spoken = speaker.spell(obj)                        # SPEAK the object ON SPIKES
        return spoken, "spoke"

    # demo: ask about each stored subject (expect the spoken object) + an unstored subject (moat)
    stored_subjects = [s for (s, o) in pairs]
    unstored = next((a for a in _ANIMALS if a in row_of and a not in stored_subjects), "zzzqqx")
    queries = stored_subjects[:2] + [unstored, "zzzqqx"]
    n_spoke, n_correct = 0, 0
    true_obj = {s: o for (s, o) in pairs}
    transcript = []
    for q in queries:
        out, kind = ask(q)
        exp = true_obj.get(q)
        ok = (kind == "spoke" and out == exp)
        tag = ("-> SPOKE OBJECT ON SPIKES" + (" [correct]" if ok else f" [expected {exp}]")) if kind == "spoke" \
              else "[MOAT: unknown/unstored -> I don't know]" if kind == "moat" else f"[{kind}]"
        print(f"  ask 'what does the {q} {verb}?' -> \"{out}\"  {tag}", flush=True)
        transcript.append({"q": q, "out": out, "kind": kind, "ok": ok})
        n_spoke += int(kind == "spoke"); n_correct += int(ok)
    moat_ok = all(t["kind"] == "moat" for t in transcript if t["q"] in (unstored, "zzzqqx"))
    return {"seed": seed, "n_spoke": n_spoke, "n_correct": n_correct, "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--bridge", default="bridges/breadth_aw/seed42.simstate.h5")
    ap.add_argument("--verb", default="eat")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[SPOKEN relational SVO Q&A] reason (FHRR) + speak (A->W), one numpy process | K={a.K} verb='{a.verb}'",
          flush=True)
    r = run(a.corpus_path, a.K, a.bridge, a.verb, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE"); return
    go = r["n_correct"] >= 1 and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- the brain ANSWERS a relational question by RECOVERING the "
          f"object (FHRR unbind over its own codes) and SPEAKING it ON SPIKES ({r['n_correct']}/{r['n_spoke']} correct), "
          f"and ABSTAINS on an unstored relation (moat {r['moat_ok']}).", flush=True)


if __name__ == "__main__":
    main()
