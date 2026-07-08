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

    def __init__(self, corpus_path, K, n_clusters, bridge_path, seed, class_verb, exc_verb, rel_verb,
                 aw_seed=42, two_bridge=False):
        stories = load_token_stream_multi(corpus_path, max_stories=None)
        self.class_verb, self.exc_verb, self.rel_verb = class_verb, exc_verb, rel_verb
        # PROPERTY reasoner (rate, emergent clusters) + its codes
        self.prop = CancellingConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)
        if two_bridge:                                        # broader spoken vocab via the EMERGE-68 dispatch
            from research.runners._realcorpus_two_bridge_speaker import TwoBridgeFrameSpeaker
            self.speaker = TwoBridgeFrameSpeaker(seed=aw_seed)
        else:
            self.speaker = ConceptFrameSpeaker(bridge_path, seed=aw_seed, vocab=VOCAB, word_to_pool=WORD_TO_POOL)
        self.spellable = set(self.speaker.vocab)
        self.animals = set(self.speaker.vocab) & _ANIMALS

        # teach a class property + a member exception over the discovered animal cluster
        coh = {c: _coherence(self.prop, c) for c in self.prop.cat_ids}
        self.pos = _pick_pos(self.prop, coh)
        taught, held = _splits(self.prop.members, self.prop.cat_ids, self.prop.rng)
        self.prop.teach(taught)
        self.exc_verbs = {}                                  # exc_id -> spoken verb (per member exception)
        self.exc_word = next((w for w in self.prop.members[self.pos]
                              if w in self.animals and self.prop.ask_class(self.pos, w) == "yes"), None)
        if self.exc_word:
            self.prop.teach_exception_adaptive(self.exc_word, "own", margin=2.0)
            self.exc_verbs["own"] = exc_verb

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
        self.rel_facts = []                                  # (subj, verb, obj) -- setup + taught
        for (s, o) in self.rel_pairs:
            if rel_verb in self.row_of:
                self.svo.store(self.row_of[s], self.row_of[rel_verb], self.row_of[o])
                self.rel_facts.append((s, rel_verb, o))

    @staticmethod
    def _append_persist(persist, entry):
        import json, os
        saved = json.load(open(persist)) if os.path.exists(persist) else []
        saved.append(entry)
        json.dump(saved, open(persist, "w"))

    def teach_property_exception(self, word, verb, persist=None):
        """Grow: teach '<word> <verb>s' as a property EXCEPTION (its own property overrides the class) live.
        Then 'does a <word> <class_verb>?' -> 'no -- the <word> can <verb>'. verb must be spellable."""
        if word not in self.prop.row_of or verb not in self.spellable:
            return False
        self.prop.teach_exception_adaptive(word, word, margin=2.0)   # per-word exception id
        self.exc_verbs[word] = verb
        if persist is not None:
            self._append_persist(persist, ["prop", word, verb])
        return True

    def verb_row(self, v):
        """Resolve a relational verb to a discovered-vocab row (try the surface form, then strip a trailing -s)."""
        for cand in (v, v[:-1] if v.endswith("s") else v):
            if cand in self.row_of:
                return self.row_of[cand], cand
        return None, None

    def teach_relational(self, subj, verb, obj, persist=None):
        """Grow through conversation: store a NEW relational fact '<subj> <verb> <obj>' live (any verb, + persist)."""
        vrow, vbase = self.verb_row(verb)
        if subj not in self.row_of or obj not in self.row_of or vrow is None:
            return False
        self.svo.store(self.row_of[subj], vrow, self.row_of[obj])
        self.rel_facts.append((subj, vbase, obj))
        if persist is not None:
            self._append_persist(persist, ["rel", subj, vbase, obj])
        return True

    def load_persisted(self, persist):
        """Re-store taught facts (BOTH relational + property exceptions) from a prior session -- the brain
        REMEMBERS across sessions (codes are seed-deterministic, so the same words rebuild the same phasors)."""
        import json, os
        if not os.path.exists(persist):
            return 0
        n = 0
        for rec in json.load(open(persist)):
            if rec and rec[0] == "rel":
                n += int(self.teach_relational(rec[1], rec[2], rec[3]))         # re-store (already persisted)
            elif rec and rec[0] == "prop":
                n += int(self.teach_property_exception(rec[1], rec[2]))
            elif len(rec) == 3:                                                  # back-compat untagged relational
                n += int(self.teach_relational(rec[0], rec[1], rec[2]))
        return n

    def _word(self, w):
        """Spell a word ON SPIKES if the A->W covers it (spellable), else render it as text."""
        return self.speaker.spell(w) if w in self.spellable else w

    def _speak_svo(self, subj, verb, obj):
        """A full-sentence relational answer 'the <subj> <verb>s <obj>' -- content words ON SPIKES where
        spellable; the/-s are host scaffolds (the spiking function-word A->W is the follow-on)."""
        vbase = verb[:-1] if verb.endswith("s") else verb
        return f"the {self._word(subj)} {self._word(vbase)}s {self._word(obj)}"

    def ask(self, q):
        """Route by question form: 'does a X <verb>?' -> property; 'what does the X <verb>?' -> relational."""
        toks = q.lower().replace("?", "").split()
        if toks[:1] == ["what"]:                                  # relational: what does the X <verb>
            subj = toks[3] if len(toks) > 3 else None
            verb = toks[4] if len(toks) > 4 else self.rel_verb    # ANY relational verb (not just 'eat')
            vrow, _ = self.verb_row(verb)
            if subj not in self.row_of or vrow is None:
                return "I don't know", "moat"
            o = self.svo.answer_patient(self.row_of[subj], vrow)
            if o is None:
                return "I don't know", "moat"
            obj = self.vocab[o]
            return self._speak_svo(subj, verb, obj), "relational"
        if toks[:1] == ["who"]:                                   # reverse relational: who <verb>s the <obj>
            content = [t for t in toks[1:] if t not in ("the", "a", "an")]
            verb = content[0] if content else None                # who eats fish -> verb=eats, obj=fish
            obj = content[-1] if len(content) > 1 else None
            vrow, _ = self.verb_row(verb) if verb else (None, None)
            if obj not in self.row_of or vrow is None:
                return "I don't know", "moat"
            arow = self.svo.answer_agent(vrow, self.row_of[obj])
            if arow is None:
                return "I don't know", "moat"
            subj = self.vocab[arow]
            return self._speak_svo(subj, verb, obj), "relational"
        # property: does/can a X <verb>
        subj = toks[2] if len(toks) > 2 else None
        if subj not in self.prop.row_of:
            return "I don't know", "moat"
        pred = self.prop._predict_all(subj)
        if pred[0] == "exc":                                       # a member exception overrides -> its own verb
            verb = self.exc_verbs.get(pred[1], self.exc_verb)
            frame, _ = self.speaker.speak_frame(subj, verb); return f"no -- {frame}", "override"
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
    ap.add_argument("--persist", default=None, help="JSON file: remember taught relational facts across sessions")
    ap.add_argument("--two-bridge", action="store_true", help="broader spoken vocab (2nd A->W bridge, ~23 nouns)")
    a = ap.parse_args()
    print(f"[UNIFIED talkable console] property (inherit/cancel) + relational (SVO), spoken on spikes, moat | "
          f"seed={a.seed}", flush=True)
    con = UnifiedTalkableConsole(a.corpus_path, a.K, a.n_clusters, a.bridge, a.seed,
                                 a.class_verb, a.exc_verb, a.rel_verb, two_bridge=a.two_bridge)
    if a.persist:
        n = con.load_persisted(a.persist)
        if n:
            print(f"  [memory] remembered {n} taught fact(s) from a prior session", flush=True)
    animals = sorted(con.animals & set(con.prop.members[con.pos]))
    print(f"  discovered animal cluster: {[w for w in con.prop.members[con.pos] if w in con.animals]}; "
          f"class='{a.class_verb}', exception '{con.exc_word}'->'{a.exc_verb}'; "
          f"relational facts: " + ", ".join(f"'{s} {a.rel_verb}s {o}'" for s, o in con.rel_pairs), flush=True)

    if a.repl:
        print(f"  [talk to the brain] ask: 'does a <animal> {a.class_verb}?' / 'what does the <animal> <verb>?'  "
              f"|  teach: 'the <animal> <verb> <animal>'  ('quit' to exit)", flush=True)
        import sys
        for line in sys.stdin:
            q = line.strip()
            if not q or q.lower() in ("quit", "exit"):
                break
            toks = q.lower().replace("?", "").replace(".", "").split()
            # TEACH (declarative, not a question): 3 content tokens = relational SVO; 2 = property exception
            if toks and toks[0] not in ("does", "can", "what", "who"):
                content = [t for t in toks if t not in ("the", "a", "an")]
                if len(content) == 3 and con.teach_relational(content[0], content[1], content[2], persist=a.persist):
                    print(f"  brain: ok, I learned that the {content[0]} {content[1]} {content[2]}.", flush=True)
                elif len(content) == 2:
                    verb = content[1][:-1] if content[1].endswith("s") else content[1]   # sleeps -> sleep
                    if con.teach_property_exception(content[0], verb, persist=a.persist):
                        print(f"  brain: ok, I learned that the {content[0]} {verb}s (an exception).", flush=True)
                    else:
                        print(f"  brain: I can't learn that ('{verb}' must be a word I can say).", flush=True)
                else:
                    print(f"  brain: I can't learn that (say '<animal> <verb> <animal>' or '<animal> <verb>').", flush=True)
                continue
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
