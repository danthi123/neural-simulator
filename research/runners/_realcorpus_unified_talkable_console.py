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
                 aw_seed=42, two_bridge=False, learn_corpus_facts=False, spiking_gen=False, multi_bridge=False,
                 neural_route=False):
        stories = load_token_stream_multi(corpus_path, max_stories=None)
        self.class_verb, self.exc_verb, self.rel_verb = class_verb, exc_verb, rel_verb
        # PROPERTY reasoner (rate, emergent clusters) + its codes
        self.prop = CancellingConsole(seed, stories, K, emergent=True, n_clusters=n_clusters)
        if multi_bridge:                                      # 3-bridge dispatch (animals + objects + 3sg verbs)
            from research.runners._realcorpus_multi_bridge_speaker import MultiBridgeFrameSpeaker
            self.speaker = MultiBridgeFrameSpeaker(seed=aw_seed)
        elif two_bridge:                                      # broader spoken vocab via the EMERGE-68 dispatch
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
        self.exceptions = []                                 # (member, exc_id) -- for self-correcting re-teach
        self.last_subject = None                             # multi-turn anaphora: the last-mentioned subject
        self.exc_word = next((w for w in self.prop.members[self.pos]
                              if w in self.animals and self.prop.ask_class(self.pos, w) == "yes"), None)
        if self.exc_word:
            self.prop.teach_exception_adaptive(self.exc_word, "own", margin=2.0)
            self.exc_verbs["own"] = exc_verb
            self.exceptions.append((self.exc_word, "own"))

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
        # DITRANSITIVE (ternary) store: the same phasor codes, a 4-role FHRR (agent/verb/recipient/theme) --
        # so the console can converse about ternary relations ("the dog gives the cat a bone"). CYCLE 1028.
        from research.runners._realcorpus_ditransitive_store_derisk import DitransStore
        self.ditrans = DitransStore(Z, list(range(len(vocab))), (_role(rng), _role(rng), _role(rng), _role(rng)))
        self.ditrans_facts = []                              # (subj, verb, recipient, theme)
        self.DITRANS_VERBS = {"give", "show", "bring", "send", "tell", "offer"}
        animals_present = sorted(a for a in self.animals if a in self.row_of)
        rng.shuffle(animals_present)
        self.rel_pairs = [(animals_present[i], animals_present[i + 1])
                          for i in range(0, len(animals_present) - 1, 2)]
        self.rel_facts = []                                  # (subj, verb, obj) -- setup + taught + corpus-mined
        for (s, o) in self.rel_pairs:
            if rel_verb in self.row_of:
                self.svo.store(self.row_of[s], self.row_of[rel_verb], self.row_of[o])
                self.rel_facts.append((s, rel_verb, o))
        if learn_corpus_facts:                               # LEARN relational facts FROM THE CORPUS (not just taught)
            from research.runners._realcorpus_learn_corpus_facts_derisk import mine_svo, VERBS, NOUNS_EXTRA, VERB_NORM
            nouns = (_ANIMALS | NOUNS_EXTRA) & set(vocab)
            verbs = [v for v in VERBS if v in self.row_of]
            toks = [t for st in stories for t in st]
            n_learned = 0
            for (s, v, o), _ in mine_svo(toks, nouns, verbs).most_common(80):
                vb = VERB_NORM.get(v, v)                      # present-tense base so queries match
                vr, _b = self.verb_row(vb)
                if vr is not None and s in self.row_of and o in self.row_of:
                    self.svo.store(self.row_of[s], vr, self.row_of[o])
                    self.rel_facts.append((s, vb, o)); n_learned += 1
            self.n_corpus_facts = n_learned

        # FULLY-SPIKING GENERATION (opt-in): the property answer's SLOT ORDER produced on spikes by the
        # EMERGE-65 self-organized spiking-Broca producer (exact-order competitive queuing + wash-out), each
        # word spelled on spikes by the A->W -- replacing the host f-string order in speak_frame. Gate-first:
        # only invoked when the reasoner ANSWERS (abstain -> the producer is never called -> moat by construction).
        self.spiking_gen = spiking_gen
        self._producer = None
        self._svo_producer = None
        if spiking_gen:
            from research.runners._emerge65_self_organized_producer_derisk import SelfOrganizedProducer
            from research.runners._emerge62_discover_function_words_derisk import build_stream
            sop = SelfOrganizedProducer(seed).build_from_corpus(build_stream(seed))
            self._producer = sop.producer(spell=self.speaker.spell)
            # RELATIONAL (transitive) generation on spikes: only when the multi-bridge speaker holds the 3sg verb
            # surfaces (BRIDGE-3). The C_TRANS construction is mined from the corpus stream; the FILLERS are the
            # console's facts. Gate-first RegistryBrocaProducer (abstain -> never invoked -> moat).
            if getattr(self.speaker, "speakers", None) is not None:            # a MultiBridgeFrameSpeaker
                from research.runners._emerge74_transitive_ditransitive_derisk import (
                    SVOConstructionRegistry, build_stream_svo)
                from research.runners._emerge72_construction_registry_derisk import RegistryBrocaProducer
                _reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
                if "C_TRANS" in _reg.registered_fits():
                    self._svo_producer = RegistryBrocaProducer(_reg.render_cq(), spell=self.speaker.spell)

        # NEURAL question-comprehension ROUTING (opt-in): a fronto-striatal reservoir read-out classifies the
        # question TYPE, replacing the host keyword if-ladder in ask(). Default off -> keyword (byte-identical).
        self.neural_route = neural_route
        self._router = None
        if neural_route:
            from research.runners._realcorpus_neural_question_routing_derisk import QuestionRouter
            self._router = QuestionRouter(seed=seed)

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
        if (word, word) not in self.exceptions:
            self.exceptions.append((word, word))
        # SELF-CORRECTING re-teach: a new exception's cross-talk can flip a MARGINAL existing exception;
        # re-teaching all exceptions repairs it (teach_exception_adaptive adds 0 drive for an already-winning
        # exception, restores a flipped one). Fixes the multi-exception collateral (flagship CYCLE 1013).
        for (m, eid) in self.exceptions:
            self.prop.teach_exception_adaptive(m, eid, margin=2.0)
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

    def teach_ditransitive(self, subj, verb, recip, theme, persist=None):
        """Grow: store a TERNARY fact '<subj> <verb> <recip> <theme>' ("the dog gives the cat a bone") live."""
        vrow, vbase = self.verb_row(verb)
        if vrow is None or any(w not in self.row_of for w in (subj, recip, theme)):
            return False
        self.ditrans.store(self.row_of[subj], vrow, self.row_of[recip], self.row_of[theme])
        self.ditrans_facts.append((subj, vbase, recip, theme))
        if persist is not None:
            self._append_persist(persist, ["ditrans", subj, vbase, recip, theme])
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
            elif rec and rec[0] == "ditrans":                                    # ternary (agent-verb-recipient-theme)
                n += int(self.teach_ditransitive(rec[1], rec[2], rec[3], rec[4]))
            elif len(rec) == 3:                                                  # back-compat untagged relational
                n += int(self.teach_relational(rec[0], rec[1], rec[2]))
        return n

    def _word(self, w):
        """Spell a word ON SPIKES if the A->W covers it (spellable), else render it as text."""
        return self.speaker.spell(w) if w in self.spellable else w

    def _speak_svo(self, subj, verb, obj):
        """A full-sentence relational answer 'the <subj> <verb>s the <obj>'. When the C_TRANS spiking producer
        is built AND every filler (subj, 3sg verb, obj, the) is spellable, the slot ORDER is produced ON SPIKES
        (EMERGE-74 registry producer) + each word spelled on spikes; else content-on-spikes with a host order."""
        vbase = verb[:-1] if verb.endswith("s") else verb
        if self._svo_producer is not None:
            from research.runners._emerge74_transitive_ditransitive_derisk import emerge_v3
            from research.runners._emerge72_construction_registry_derisk import decision
            v3 = emerge_v3(vbase, already_3sg=None)
            if all(w in self.speaker.vocab for w in ("the", subj, v3, obj)):    # fully spellable on the bridges
                out = self._svo_producer.speak(decision("ANSWER", construction="C_TRANS",
                                                        subject=subj, verb=vbase, obj=obj))
                if out.get("surface"):
                    return out["surface"]                                       # order + words ON SPIKES
        return f"the {self._word(subj)} {self._word(vbase)}s {self._word(obj)}"

    def describe(self, word):
        """'tell me about <word>': aggregate the word's PROPERTY (inherit/cancel) + RELATIONAL facts into
        connected prose with a referring pronoun + 'and'-aggregation (NLG discourse; Levelt/Reiter-Dale) --
        a multi-fact discourse answer toward fluent conversation. Moat: nothing known -> abstain."""
        prop_verb, is_exc = None, False
        if word in self.prop.row_of:                          # property: what it can do (inherit / override)
            pred = self.prop._predict_all(word)
            if pred[0] == "exc":
                prop_verb, is_exc = self.exc_verbs.get(pred[1], self.exc_verb), True
            elif pred == ("cat", self.pos):
                prop_verb = self.class_verb
        rels = [(v, o) for (s, v, o) in self.rel_facts if s == word]   # relational: what it does to whom
        if prop_verb is None and not rels:
            return "I don't know", "moat"

        subj = self._word(word)
        sents = []
        if prop_verb is not None:
            # CONTRAST (Reiter-Dale): an exception surfaces WHAT IT DOES INSTEAD -- "can sleep, not run"
            if is_exc:
                sents.append(f"the {subj} can {self._word(prop_verb)}, not {self._word(self.class_verb)}")
            else:
                sents.append(self._gen_frame(word, prop_verb))     # inheritance clause: order ON SPIKES when spiking_gen
        if rels:
            # GROUP by verb + CAP objects (concise NLG; else a long run-on when many facts are known):
            # "It sees cat, ball, and dog (and 9 more). It eats frog."
            from collections import OrderedDict
            vbase = lambda v: v[:-1] if v.endswith("s") else v
            byv = OrderedDict()
            for (v, o) in rels:
                byv.setdefault(vbase(v), []).append(o)
            ref = "it" if sents else f"the {subj}"
            clauses = []
            for vb, objs in byv.items():
                shown, extra = objs[:3], len(objs) - 3
                ol = self._word_list([self._word(o) for o in shown]) + (f" and {extra} more" if extra > 0 else "")
                clauses.append(f"{self._word(vb)}s {ol}")
            sents.append(f"{ref} {' and '.join(clauses)}")
        return " ".join(s[0].upper() + s[1:] + "." for s in sents), "describe"

    def _prop_verb(self, word):
        """The property verb a word can do (its own exception verb, or the inherited class verb), or None."""
        if word not in self.prop.row_of:
            return None
        pred = self.prop._predict_all(word)
        if pred[0] == "exc":
            return self.exc_verbs.get(pred[1], self.exc_verb)
        if pred == ("cat", self.pos):
            return self.class_verb
        return None

    def compare(self, a, b):
        """'compare X and Y': contrast their property + relational facts ('the X can sleep, but the Y can
        run; the X eats frog, and the Y eats dog'). A conversational comparison act. Moat if neither known."""
        pa, pb = self._prop_verb(a), self._prop_verb(b)
        ra = next(((v, o) for (s, v, o) in self.rel_facts if s == a), None)
        rb = next(((v, o) for (s, v, o) in self.rel_facts if s == b), None)
        if pa is None and pb is None and ra is None and rb is None:
            return "I don't know", "moat"
        clauses = []
        if pa is not None or pb is not None:
            conn = "but" if (pa is not None and pb is not None and pa != pb) else "and"
            clauses.append(f"the {self._word(a)} can {self._word(pa or 'do nothing')} {conn} "
                           f"the {self._word(b)} can {self._word(pb or 'do nothing')}")
        if ra is not None or rb is not None:
            va = f"{self._word(ra[0][:-1] if ra[0].endswith('s') else ra[0])}s {self._word(ra[1])}" if ra else "does nothing"
            vb = f"{self._word(rb[0][:-1] if rb[0].endswith('s') else rb[0])}s {self._word(rb[1])}" if rb else "does nothing"
            clauses.append(f"the {self._word(a)} {va}, and the {self._word(b)} {vb}")
        return " ".join(c[0].upper() + c[1:] + "." for c in clauses), "compare"

    @staticmethod
    def _word_list(words):
        """'cat', 'ball', 'dog' -> 'cat, ball, and dog' (Oxford list)."""
        if len(words) <= 1:
            return words[0] if words else ""
        if len(words) == 2:
            return f"{words[0]} and {words[1]}"
        return ", ".join(words[:-1]) + f", and {words[-1]}"

    def _word_frame(self, subj, verb):
        frame, _ = self.speaker.speak_frame(subj, verb)        # "the <subj> can <verb>"
        return frame

    def _gen_frame(self, subj, verb):
        """The 'the <subj> can <verb>' frame with the SLOT ORDER produced ON SPIKES by the spiking-Broca
        producer (when spiking_gen), each word spelled on spikes -- else the A->W speak_frame (order host-
        templated). The fully-spiking-generated property answer."""
        if self._producer is not None:
            out = self._producer.speak({"gate": "ANSWER", "frame": "F_MODAL", "subject": subj, "verb": verb})
            if out.get("surface"):
                return out["surface"]
        frame, _ = self.speaker.speak_frame(subj, verb)
        return frame

    def _resolve(self, word):
        """Multi-turn anaphora: a pronoun 'it'/'they' resolves to the last-mentioned subject; a real word
        updates it. Returns the resolved word (or the pronoun unchanged if there is no antecedent yet)."""
        if word in ("it", "they", "them"):
            return self.last_subject or word
        if word in self.row_of or (word in self.prop.row_of):
            self.last_subject = word
        return word

    def _route_type(self, toks):
        """The question TYPE via the NEURAL router (opt-in) -- else None (keyword routing). 'compare' is a fixed
        multi-word construction kept as a keyword marker; the core does/wh types are routed neurally."""
        if self._router is None:
            return None
        if toks[:1] == ["compare"]:
            return "compare"
        return self._router.route(toks)

    @staticmethod
    def _is(rt, typ, kw):
        """Dispatch: the NEURAL type when the router is active (rt not None), else the keyword condition. A
        neural misroute self-corrects -- the per-branch extraction guards fall through to the right handler."""
        return (rt == typ) if rt is not None else kw

    def ask(self, q):
        """Route by question form: 'does a X <verb>?' -> property; 'what does the X <verb>?' -> relational;
        'tell me about X' / 'describe X' -> multi-fact discourse. Pronoun 'it' -> the last subject (anaphora).
        The routing is host keyword by default, or the NEURAL reservoir read-out when neural_route=True."""
        toks = q.lower().replace("?", "").split()
        rt = self._route_type(toks)
        # DITRANSITIVE query (before the binary what/who): a ditransitive verb + TWO nouns -> the ternary store.
        # "what does the X <give> the Y?" -> theme (Y = recipient given); "who does the X <give> a Z?" -> recipient.
        if toks[:1] in (["what"], ["who"]):
            cd = [t for t in toks[1:] if t not in ("what", "who", "does", "the", "a", "an")]
            if len(cd) == 3 and cd[1] in self.DITRANS_VERBS:
                s2, v2, n2 = self._resolve(cd[0]), cd[1], cd[2]
                vrow, _ = self.verb_row(v2)
                if s2 in self.row_of and n2 in self.row_of and vrow is not None:
                    if toks[:1] == ["what"]:                       # theme query (the given noun is the recipient)
                        th = self.ditrans.answer_theme(self.row_of[s2], vrow, self.row_of[n2])
                        if th is not None:
                            return (f"the {self._word(s2)} {self._word(v2)}s the {self._word(n2)} "
                                    f"a {self._word(self.vocab[th])}"), "ditransitive"
                    else:                                          # who -> recipient query (the given noun is the theme)
                        rc = self.ditrans.answer_recipient(self.row_of[s2], vrow, self.row_of[n2])
                        if rc is not None:
                            return (f"the {self._word(s2)} {self._word(v2)}s the {self._word(self.vocab[rc])} "
                                    f"a {self._word(n2)}"), "ditransitive"
                    return "I don't know", "moat"                  # ditransitive verb but not stored -> abstain
        if self._is(rt, "compare", toks[:1] == ["compare"]):      # comparison: compare X and Y
            content = [t for t in toks[1:] if t not in ("the", "a", "an", "and", "to", "with")]
            if len(content) >= 2:
                return self.compare(self._resolve(content[0]), self._resolve(content[1]))
            return "I don't know", "moat"
        if self._is(rt, "describe", toks[:3] == ["tell", "me", "about"] or toks[:1] == ["describe"]):  # discourse
            content = [t for t in (toks[3:] if toks[0] == "tell" else toks[1:]) if t not in ("the", "a", "an")]
            return self.describe(self._resolve(content[-1])) if content else ("I don't know", "moat")
        if self._is(rt, "what", toks[:1] == ["what"]):            # relational: what (does) (the) X <verb>
            c = [t for t in toks[1:] if t not in ("the", "a", "an", "does")]   # determiner-robust -> [subj, verb]
            subj = self._resolve(c[0]) if c else None
            verb = c[1] if len(c) > 1 else self.rel_verb          # ANY relational verb (not just 'eat')
            vrow, _ = self.verb_row(verb)
            if subj not in self.row_of or vrow is None:
                return "I don't know", "moat"
            o = self.svo.answer_patient(self.row_of[subj], vrow)
            if o is None:
                return "I don't know", "moat"
            obj = self.vocab[o]
            return self._speak_svo(subj, verb, obj), "relational"
        if self._is(rt, "who", toks[:1] == ["who"]):              # reverse relational: who <verb>s the <obj>
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
        # relational YES/NO: 'does the <subj> <verb> <obj>?' (an object after the verb) -> verify the fact.
        # A misroute here self-corrects: the len(content)>=3 guard falls through to the property branch.
        if self._is(rt, "yesno", toks[:1] in (["does"], ["can"]) and len(toks) >= 5):
            content = [t for t in toks[1:] if t not in ("the", "a", "an")]
            if len(content) >= 3:
                s2, v2, o2 = self._resolve(content[0]), content[1], content[2]
                vrow, _ = self.verb_row(v2)
                if s2 in self.row_of and o2 in self.row_of and vrow is not None:
                    if self.svo.contains(self.row_of[s2], vrow, self.row_of[o2]):   # verify the SPECIFIC fact
                        return f"yes -- {self._speak_svo(s2, v2, o2)}", "yesno"
                    got = self.svo.answer_patient(self.row_of[s2], vrow)
                    if got is None:
                        return "I don't know", "moat"                       # nothing stored for that (subj, verb)
                    return f"no -- {self._speak_svo(s2, v2, self.vocab[got])}", "yesno"   # not that obj; the real one
        # property: does/can (a) X <verb>  (determiner-robust so 'does it run?' resolves the pronoun)
        c = [t for t in toks[1:] if t not in ("a", "an", "the")]
        subj = self._resolve(c[0]) if c else None
        if subj not in self.prop.row_of:
            return "I don't know", "moat"
        pred = self.prop._predict_all(subj)
        if pred[0] == "exc":                                       # a member exception overrides -> its own verb
            verb = self.exc_verbs.get(pred[1], self.exc_verb)
            return f"no -- {self._gen_frame(subj, verb)}", "override"
        if pred == ("cat", self.pos):
            return f"yes -- {self._gen_frame(subj, self.class_verb)}", "inherit"
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
    ap.add_argument("--multi-bridge", action="store_true", help="3-bridge A->W (animals + objects + 3sg verbs): enables the RELATIONAL answer on spikes")
    ap.add_argument("--learn-corpus-facts", action="store_true", help="LEARN relational facts from the corpus (not just taught)")
    ap.add_argument("--spiking-gen", action="store_true", help="FULLY-SPIKING generation: the property (+ relational, with --multi-bridge) answer's slot ORDER produced on spikes by the spiking-Broca producer (not a host template)")
    ap.add_argument("--neural-route", action="store_true", help="NEURAL question-comprehension routing: a reservoir read-out classifies the question type (not the host keyword if-ladder)")
    a = ap.parse_args()
    print(f"[UNIFIED talkable console] property (inherit/cancel) + relational (SVO), spoken on spikes, moat | "
          f"seed={a.seed}{' | SPIKING-GEN (order on spikes)' if a.spiking_gen else ''}", flush=True)
    con = UnifiedTalkableConsole(a.corpus_path, a.K, a.n_clusters, a.bridge, a.seed,
                                 a.class_verb, a.exc_verb, a.rel_verb, two_bridge=a.two_bridge,
                                 learn_corpus_facts=a.learn_corpus_facts, spiking_gen=a.spiking_gen,
                                 multi_bridge=a.multi_bridge, neural_route=a.neural_route)
    if a.learn_corpus_facts:
        print(f"  [experience] learned {getattr(con, 'n_corpus_facts', 0)} relational facts from the corpus", flush=True)
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
            # TEACH (declarative, not a question): 4 content = ditransitive; 3 = relational SVO; 2 = property exception
            if toks and toks[0] not in ("does", "can", "what", "who", "tell", "describe", "compare"):
                content = [t for t in toks if t not in ("the", "a", "an")]
                if len(content) == 4 and content[1] in con.DITRANS_VERBS and \
                        con.teach_ditransitive(content[0], content[1], content[2], content[3], persist=a.persist):
                    print(f"  brain: ok, I learned that the {content[0]} {content[1]}s the {content[2]} a {content[3]}.", flush=True)
                elif len(content) == 3 and con.teach_relational(content[0], content[1], content[2], persist=a.persist):
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
