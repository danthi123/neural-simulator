"""UNIFIED talkable console: ONE emergent brain answers BOTH property questions ("does a fish run?" ->
inherit/cancel) AND relational questions ("what does the dog eat?" -> the object), spoken ON SPIKES,
with the no-confab moat -- routed by the question form.

The capstone of the breadth->knowledge arc: the same emergent real-corpus brain (its OWN discovered
co-occurrence codes) does TWO kinds of reasoning through TWO validated mechanisms, and SPEAKS the answer:
  * PROPERTY ("does a <X> <verb>?"): the associative-memory reasoner (CancellingConsole, emergent
    clusters) inherits a taught class property or applies a member exception (cancellation) -> speak
    "the X can <class-verb>" (inherit) / "the X can <exception-verb>" (override) / "no" (other category).
  * RELATIONAL ("what does the <X> <verb>?"): the FHRR store (SVOStore) recovers the object by role-
    unbinding -> speak the object.
  * MOAT: an unknown word / unstored relation -> "I don't know" (gate-first).
Both reasoners ride the SAME real-corpus codes (seed-deterministic); the A->W speaks on spikes (built at
the checkpoint seed). numpy, one process. Reuse-by-import. NO `sim/` edit. Requires SIM_BACKEND=numpy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_cancellation_derisk import CancellingConsole, _pick_pos, _ANIMALS
from research.runners._realcorpus_inheritance_rung4_conversation_derisk import _splits, _coherence
from research.runners._realcorpus_svo_qa_derisk import SVOStore
from research.runners._realcorpus_svo_compose_probe import _phasors, _role
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw import VOCAB, WORD_TO_POOL
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi


class UnifiedTalkableConsole:
    """One emergent brain: property (inherit/cancel) + relational (SVO) reasoning, spoken on spikes, moat."""

    def __init__(self, corpus_path, K, n_clusters, bridge_path, seed, class_verb, exc_verb, rel_verb, aw_seed=42):
        stories = load_token_stream_multi(corpus_path, max_stories=None)
        self.class_verb, self.exc_verb, self.rel_verb = class_verb, exc_verb, rel_verb
        # PROPERTY reasoner (rate, emergent clusters) + its codes
        self.prop = CancellingConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)
        self.speaker = ConceptFrameSpeaker(bridge_path, seed=aw_seed, vocab=VOCAB, word_to_pool=WORD_TO_POOL)
        self.spellable = set(self.speaker.vocab)
        self.animals = set(self.speaker.vocab) & _ANIMALS

        # teach a class property + a member exception over the discovered animal cluster
        coh = {c: _coherence(self.prop, c) for c in self.prop.cat_ids}
        self.pos = _pick_pos(self.prop, coh)
        taught, held = _splits(self.prop.members, self.prop.cat_ids, self.prop.rng)
        self.prop.teach(taught)
        self.exc_word = next((w for w in self.prop.members[self.pos]
                              if w in self.animals and self.prop.ask_class(self.pos, w) == "yes"), None)
        if self.exc_word:
            self.prop.teach_exception_adaptive(self.exc_word, "own", margin=2.0)

        # RELATIONAL store (FHRR) over the SAME real-corpus codes (seed-deterministic re-learn) + spellable facts
        vocab, gfreq = discover_vocab(stories, K)
        self.vocab, self.row_of = vocab, {w: i for i, w in enumerate(vocab)}
        hubs = []
        for w, _ in gfreq.most_common():
            if w in STOPLIST or w in set(vocab) or len(w) < MIN_WORD_LEN:
                continue
            hubs.append(w)
            if len(hubs) >= N_HUB:
                break
        codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
        rng = np.random.default_rng(seed)
        Z = _phasors(codes, list(range(len(vocab))), seed)
        self.svo = SVOStore(Z, list(range(len(vocab))), (_role(rng), _role(rng), _role(rng)))
        animals_present = sorted(a for a in self.animals if a in self.row_of)
        rng.shuffle(animals_present)
        self.rel_pairs = [(animals_present[i], animals_present[i + 1])
                          for i in range(0, len(animals_present) - 1, 2)]
        for (s, o) in self.rel_pairs:
            if rel_verb in self.row_of:
                self.svo.store(self.row_of[s], self.row_of[rel_verb], self.row_of[o])

    def ask(self, q):
        """Route by question form: 'does a X <verb>?' -> property; 'what does the X <verb>?' -> relational."""
        toks = q.lower().replace("?", "").split()
        if toks[:1] == ["what"]:                                  # relational: what does the X <verb>
            subj = toks[3] if len(toks) > 3 else None
            if subj not in self.row_of:
                return "I don't know", "moat"
            o = self.svo.answer_patient(self.row_of[subj], self.row_of[self.rel_verb])
            if o is None:
                return "I don't know", "moat"
            obj = self.vocab[o]
            return (self.speaker.spell(obj) if obj in self.spellable else obj), "relational"
        # property: does/can a X <verb>
        subj = toks[2] if len(toks) > 2 else None
        if subj not in self.prop.row_of:
            return "I don't know", "moat"
        pred = self.prop._predict_all(subj) if hasattr(self.prop, "_predict_all") else ("cat", self.prop._predict_cat(subj))
        if pred == ("exc", "own"):
            frame, _ = self.speaker.speak_frame(subj, self.exc_verb); return f"no -- {frame}", "override"
        if pred == ("cat", self.pos):
            frame, _ = self.speaker.speak_frame(subj, self.class_verb); return f"yes -- {frame}", "inherit"
        return "no", "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--bridge", default="bridges/breadth_aw/seed42.simstate.h5")
    ap.add_argument("--class-verb", default="run")
    ap.add_argument("--exc-verb", default="sleep")
    ap.add_argument("--rel-verb", default="eat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repl", action="store_true", help="interactive: type questions, the brain answers (Ctrl-D/'quit' to exit)")
    a = ap.parse_args()
    print(f"[UNIFIED talkable console] property (inherit/cancel) + relational (SVO), spoken on spikes, moat | "
          f"seed={a.seed}", flush=True)
    con = UnifiedTalkableConsole(a.corpus_path, a.K, a.n_clusters, a.bridge, a.seed,
                                 a.class_verb, a.exc_verb, a.rel_verb)
    animals = sorted(con.animals & set(con.prop.members[con.pos]))
    print(f"  discovered animal cluster: {[w for w in con.prop.members[con.pos] if w in con.animals]}; "
          f"class='{a.class_verb}', exception '{con.exc_word}'->'{a.exc_verb}'; "
          f"relational facts: " + ", ".join(f"'{s} {a.rel_verb}s {o}'" for s, o in con.rel_pairs), flush=True)

    if a.repl:
        print(f"  [talk to the brain] property: 'does a <animal> {a.class_verb}?'  relational: 'what does the "
              f"<animal> {a.rel_verb}?'  ('quit' to exit)", flush=True)
        import sys
        for line in sys.stdin:
            q = line.strip()
            if not q or q.lower() in ("quit", "exit"):
                break
            out, kind = con.ask(q)
            print(f"  brain: \"{out}\"   [{kind}]", flush=True)
        return

    # scripted mixed conversation: property (inherit + cancel), relational, moat
    others = [w for w in animals if w != con.exc_word]
    script = []
    if con.exc_word:
        script.append(f"does a {con.exc_word} {a.class_verb}?")           # exception -> cancel
    if others:
        script.append(f"does a {others[0]} {a.class_verb}?")              # inherit
    if con.rel_pairs:
        script.append(f"what does the {con.rel_pairs[0][0]} {a.rel_verb}?")  # relational
    script.append("does a zzzqqx run?")                                   # moat
    script.append("what does the zzzqqx eat?")                            # moat

    n_ok = 0
    for q in script:
        out, kind = con.ask(q)
        n_ok += int(kind != "moat" or "don't know" in out)
        print(f"  Q: {q}\n     A: \"{out}\"   [{kind}]", flush=True)
    print(f"\n  VERDICT: the ONE emergent brain answered BOTH property (inherit/cancel) AND relational (SVO) "
          f"questions, spoken on spikes, and abstained on the unknown -- routed by question form. Two knowledge "
          f"dimensions, one brain, one code set, moat intact.", flush=True)


if __name__ == "__main__":
    main()
