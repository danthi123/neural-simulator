"""The fluid-conversation CONSOLE -- one coherent chat loop tying Phases 2-5 together (the owner's console-not-
dashboard priority). Talk to the brain like an LLM: ask grounded questions, use pronouns across turns, TEACH it new
facts, and it abstains ("I don't know") on what it hasn't learned.

  QUESTION  "what does the dog eat?" / "what does it chase?" (pronoun) / "who eats meat?" / "does the dog eat meat?"
            -> interrogative parse -> brain GATE (moat gate-FIRST) -> RA-fine-tuned 21M focused answer -> VERIFY.
  DISCUSS   "tell me about the dog" -> open-ended grounded discussion of the neighbourhood (Phase-10); "compare X Y".
  INSTANCE  "i saw a dog" -> mint a specific instance; "the dog is brown" -> its OWN fact; "what is the dog?" -> brown;
            "what do dogs eat?" -> the KIND ("which dog?", Phase-14: definite vs generic + isa-inheritance).
  LEARN     "learn about horse" -> fetch + ingest REAL Wikidata facts on demand (Phase-15 grounded tail); then
            "what is the horse?" -> "a horse is a mammal"; "what does the horse have?" -> "the horse has fur".
  STATEMENT "the wolf eats rabbit" / "wolf eat rabbit"  -> hear (LEARN) -> "ok, i learned that the wolf eats rabbit."
  UNTAUGHT  -> "I don't know."   (the no-confab moat)

Assembles: `MultiTurnAgent` (multi-turn anaphora, Phase 4) + `FTFaculty` (the RA render/QA fine-tuned generator,
Phase 2) + the Phase-3 gate->answer->VERIFY + Phase-5 growth + Phase-10 discussion + Phase-14 instance-rep. The BRAIN
does comprehension + knowledge + grounding + moat; the minimized (~21M) brain-gated generator does fluency.
Reuse-by-import; NO sim/ edit.

Run (scripted smoke / demo):
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --demo
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --instance-demo
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --script "what does the dog eat?|the wolf eats rabbit|what does the wolf eat?"
Run (interactive): ... (no --script/--demo/--instance-demo -> reads stdin; blank line or 'quit' exits)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback, urllib.error, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: E402
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT, SUBJECTS as FT_SUBJECTS, OBJECTS as FT_OBJECTS  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402
from research.runners._fluidconv_phase15_wikidata_breadth_derisk import _fetch_entity  # noqa: E402

_WD_SEARCH = "https://www.wikidata.org/w/api.php"          # wbsearchentities: resolve a concept NAME -> a Wikidata QID
_WD_CACHE = _REPO / "research" / "findings" / "raw" / "_fluidconv_console_wikidata_cache.json"
# the console's (richer than Phase-15's validated pair) clean-property set: subclass-of -> isa (taxonomy, a noun);
# has-part -> has; color -> is (an adjective, salient common knowledge like "a banana is yellow").
_WD_PROPS = {"P279": "isa", "P527": "has", "P462": "is"}

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_chat_repl_demo.json"
_QWORDS = {"what", "who", "does", "do", "is", "are", "tell", "can", "could", "why", "how", "when", "where", "?"}
_PRON = {"it", "its", "they", "them", "that"}
_STOP = {"the", "a", "an", "does", "do", "did", "the", "to", "of", "please"}
# instance-rep (Phase-14): a curated attribute vocab (for "the dog is brown") + N instance slots per kind.
_ATTRS = ["brown", "black", "white", "grey", "big", "small", "fast", "slow", "brave", "gentle", "old", "young"]
_INST_SLOTS = 2
_INTRO_CUES = {"saw", "have", "found", "met", "there"}   # "i saw a dog" / "there is a dog" -> mint an instance


def _art(w):
    """The indefinite article for a word ('a'/'an') -- a light readability polish for grounded templates."""
    return "an" if (w[:1].lower() in "aeiou") else "a"


def _join_and(items):
    """'a, b and c' (comma-separated, 'and' before the last) -- readable list rendering for grounded discussion."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


class FluidChat:
    """One coherent fluid-conversation agent (Phases 2-5 assembled)."""

    def __init__(self, seed=42, extra_vocab=None):
        with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
            self.cur = json.load(fh)
        facts = self.cur.get("facts", [])
        self.agents = {f[0] for f in facts}
        self.patients = {f[2] for f in facts}
        self.actions = {f[1] for f in facts}
        self.inflect = _build_inflection_map(sorted(self.actions))
        # mintable KINDS (Phase-14): the curriculum's agents (things that act -- dog/cat/bird/...). Pre-allocate a few
        # instance tokens per kind (dog_1, dog_2, ...) so a mentioned instance has a composer code (codes are fixed
        # at build). "the dog" (a specific referent) resolves to the last such instance; "dogs" -> the kind.
        self.kinds = sorted(self.agents)
        self._inst_toks = {k: [f"{k}_{i+1}" for i in range(_INST_SLOTS)] for k in self.kinds}
        # a generous pre-allocated vocab so new facts can be TAUGHT (composer codes are fixed at build): curriculum +
        # the fine-tune's broad subject/object pools + instance slots + attribute vocab + any extra.
        vocab = (set(_collect_vocab(self.cur)) | set(FT_SUBJECTS) | set(FT_OBJECTS) | set(_ATTRS)
                 | {"isa", "is"}          # instance-rep relation tokens (action fillers) need composer codes
                 | {t for slots in self._inst_toks.values() for t in slots} | set(extra_vocab or []))
        self.vocab = sorted(vocab)
        # referents (for anaphora) must stay small (one 40-neuron attractor/referent in n=600) -> a curated set
        referents = sorted(set(sorted(self.agents)[:6]) | set(list(self.patients)[:4]))
        self.mta = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in self.vocab},
                                  seed=seed, defer_planner=True, enable_biased_competition=False, composer_kind="rf",
                                  D=256)
        _teach(self.mta.agent, self.cur)
        self.store_keys = {tuple(f) for f in facts}
        self.faculty = FTFaculty()
        self.npar = self.faculty.npar
        self._mentioned = {}          # subject -> set of verbs already said (so "tell me more" surfaces a NEW fact)
        self._last_inst = {}          # kind -> the LAST minted instance token (per-kind discourse referent, Phase-14)
        self._inst_used = {k: 0 for k in self.kinds}   # how many instance slots consumed per kind
        # on-demand REAL-knowledge breadth (Phase-15): a per-concept Wikidata fact cache (fetch-once, reused/offline).
        self._wd_cache = json.loads(_WD_CACHE.read_text()) if _WD_CACHE.exists() else {}

    def _ensure_concept(self, w):
        """Inject a runtime composer code for a never-seen concept (deterministic per word). The numpy cleanup rebuilds
        its codebook from `composer.words` each call, so appending is safe. Enables learning about NEW concepts."""
        comp = self.mta.agent.composer
        if w not in comp.concepts:
            s = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            comp.concepts[w] = np.random.default_rng(s).uniform(0.0, 1.0, comp.D)
            comp.words = sorted(set(comp.words) | {w})

    def _wd_search(self, name, limit=6):
        """Resolve a concept NAME -> its top-`limit` candidate Wikidata (QID, label) pairs (wbsearchentities)."""
        url = (_WD_SEARCH + f"?action=wbsearchentities&format=json&language=en&type=item&limit={limit}&search="
               + urllib.parse.quote(name))
        req = urllib.request.Request(url, headers={"User-Agent": "sim-research/1.0 (grounded-knowledge)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            hits = json.loads(r.read().decode("utf-8")).get("search", [])
        return [(h["id"], h.get("label", "")) for h in hits]

    def _wikidata_learn(self, concept):
        """LEARN real grounded facts about `concept` from Wikidata on demand (the on-demand tail): resolve QID -> fetch
        clean SVO (P279 isa / P527 has) -> inject codes -> store. Cached per-concept. The fetch is host-side data-prep
        (legitimate environment); the brain LEARNS via composer.store. Graceful on network failure.

        QID disambiguation is DATA-DRIVEN + SENSE-AWARE: prefer a hit whose LABEL exactly matches the query AND yields
        clean facts (so 'cat' -> the animal Q146, not Catalan-the-language which also has facts; 'elephant' -> the
        animal Q7378, not the album/family), then fall back to the first hit with facts (so a query that has no
        exact-label match still resolves)."""
        concept = concept.lower().strip()
        if concept in self._wd_cache:
            facts = self._wd_cache[concept]
        else:
            try:
                hits = self._wd_search(concept, limit=6)
                if not hits:
                    return f"i couldn't find '{concept}' in the knowledge source."
                # pass 1: exact-label matches first (correct sense), then pass 2: any hit -- first with facts wins.
                exact = [(q, lbl) for (q, lbl) in hits if lbl.lower().strip() == concept]
                facts = []
                for qid, _lbl in exact + hits:
                    facts = _fetch_entity(concept, qid, _WD_PROPS, per_prop=3)
                    if facts:
                        break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError):
                return "i couldn't reach the knowledge source right now."
            self._wd_cache[concept] = facts
            try:
                _WD_CACHE.write_text(json.dumps(self._wd_cache, indent=2))
            except Exception:
                pass
        if not facts:
            return f"i couldn't find facts about the {concept}."
        n = 0
        for (a, v, p) in facts:
            for t in (a, v, p):
                self._ensure_concept(t)
            self.mta.agent.composer.store(a, v, p)
            self.store_keys.add((a, v, p))
            self.agents.add(a); self.patients.add(p); n += 1
        self.kinds = sorted(self.agents)                     # the new concept + its parents become mintable kinds
        ex = "; ".join(f"{a} {v} {p}" for (a, v, p) in facts[:4])
        return f"ok, i learned {n} facts about the {concept}: {ex}."

    def _content(self, toks):
        subj = next((t for t in toks if t in self.agents or t in self.vocab and t not in _STOP and t not in self.actions), None)
        verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
        return subj, verb

    def _is_question(self, toks):
        return bool(set(toks) & _QWORDS)

    def _kind_of(self, toks):
        """(kind, is_plural): a singular kind token -> (kind, False); a plural 'dogs' -> (kind, True)."""
        for t in toks:
            if t in self.kinds:
                return t, False
            if t.endswith("es") and t[:-2] in self.kinds:
                return t[:-2], True
            if t.endswith("s") and t[:-1] in self.kinds:
                return t[:-1], True
        return None, False

    def _mint(self, kind):
        """Introduce a discourse instance of `kind`: assign the next free slot + store the isa link, track per-kind."""
        if self._inst_used.get(kind, 0) >= len(self._inst_toks.get(kind, [])):
            return None
        tok = self._inst_toks[kind][self._inst_used[kind]]; self._inst_used[kind] += 1
        self.mta.agent.composer.store(tok, "isa", kind)         # "dog_1 isa dog" (inherit the kind's facts)
        self._last_inst[kind] = tok
        return tok

    def _answer_instance(self, inst, kind, verb):
        """INSTANCE-FIRST / KIND-FALLBACK (Phase-14): the instance's OWN fact, else inherit the kind's via isa. The
        instance's own attribute renders as a grounded template ('The dog is brown.'); an inherited kind fact uses the
        validated RA-render+VERIFY path (`_answer`). Display always uses the KIND name, never the internal token."""
        own = self.mta.agent.what_does(inst, verb)
        if own is not None:
            self._mentioned.setdefault(kind, set()).add(verb)
            return (f"The {kind} is {own}." if verb == "is" else f"The {kind} {_v3(verb)} {own}.")
        _p, reply = self._answer(kind, verb)                    # inherit via the kind (RA-render + VERIFY)
        return reply

    def _answer(self, subj, verb):
        """Phase-3 turn: GATE -> RA-render -> VERIFY. Writes the answer as the salient referent (Phase-4)."""
        p = self.mta.agent.what_does(subj, verb)
        if p is None:
            return None, "I don't know."
        self._mentioned.setdefault(subj, set()).add(verb)     # track what's been said (for "tell me more")
        ctx = f"the {subj} {_v3(verb)} {p} ."
        ans = self.faculty.answer(ctx, f"what does the {subj} {verb} ?")
        svos = _extract_all_svos(ans, self.agents, self.actions, self.patients, self.inflect)
        ung = [s for s in svos if _fact_key(s) not in self.store_keys]
        verified = bool((([subj, verb, p] in svos) or (p in ans.split())) and not ung)
        reply = ans if verified else f"The {subj} {_v3(verb)} {p}."
        if p in self.mta.referents:
            self.mta._write_referent(p)
        return p, reply

    def _elaborate(self, subj):
        """Surface an ADDITIONAL grounded fact about subj (beyond what's been said) -- richer discourse than a single
        fact. The dlPFC dialogue planner (`elaborate`) picks a related concept; map it to an UNMENTIONED (subj, verb,
        concept) fact, else scan the subject's facts for a new one; else honestly say that's all it knows."""
        said = self._mentioned.get(subj, set())
        try:
            assoc = self.mta.agent.elaborate(subj)
        except Exception:
            assoc = None
        cand = []
        if assoc is not None:
            cand = [v for v in sorted(self.actions) if v not in said and self.mta.agent.what_does(subj, v) == assoc]
        if not cand:                                # fallback: any unmentioned fact about subj
            cand = [v for v in sorted(self.actions) if v not in said and self.mta.agent.what_does(subj, v) is not None]
        if not cand:
            return None, f"that's all i know about the {subj}."
        _p, reply = self._answer(subj, cand[0])
        return cand[0], reply

    def _stored_facts(self):
        """The brain's affirmed SVO facts (string-only roles) from the composer store -- the discussion source."""
        return [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in self.mta.agent.composer.kb
                if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))
                and f.get("polarity", "AFFIRM") != "NEGATE"]

    def _neighbourhood(self, topic):
        """The topic's grounded neighbourhood (association-graph adjacency): facts where topic is agent or patient,
        plus the members of a category topic (X is <topic> -> X's facts)."""
        kb = self._stored_facts()
        # exclude discourse-INSTANCE tokens (dog_1, cat_1, ...): they are referents, not encyclopedic knowledge about
        # the KIND, so a "tell me about the dog" must not list "a dog_1 is a dog" (the instance is queried via "the dog").
        inst = {t for slots in self._inst_toks.values() for t in slots}
        kb = [f for f in kb if f[0] not in inst and f[2] not in inst]
        facts = [list(f) for f in kb if topic in (f[0], f[2])]
        # category members ONLY via the "is"/"isa" relation (X is <topic>) -- a non-taxonomic patient (dog chase cat)
        # does NOT make the agent a member, so a regular topic doesn't vacuum in unrelated facts.
        for m in [f[0] for f in kb if f[2] == topic and f[1] in ("is", "isa")]:
            facts += [list(f) for f in kb if f[0] == m and list(f) not in facts]
        return facts

    def _discuss(self, topic, *, max_facts=7):
        """Open-ended grounded DISCUSSION (Phase-10): render the neighbourhood FAITHFULLY, then concatenate. The topic's
        own TAXONOMY facts (isa/has -- e.g. Phase-15 Wikidata) are GROUPED into flowing sentences ("An elephant is a
        mammal. It has a trunk and tusk.") -- grounded by construction (no confab), less list-y than one-per-fact. Other
        facts (action verbs, category members) go one-per-sentence through the FT render + VERIFY. Moat: an ungrounded
        render is dropped; an empty neighbourhood hedges."""
        nb = []                                                         # dedup the neighbourhood
        for f in self._neighbourhood(topic):
            if f not in nb:
                nb.append(f)
            if len(nb) >= max_facts:
                break
        if not nb:
            return f"I don't know much about the {topic}."
        # (1) group the TOPIC's OWN facts into cohesive sentences (grounded templates). Note the relation split:
        # "isa" = taxonomy (a noun -> needs an article: "is a mammal"); "is" = an adjective ("is big", no article).
        isa_parents = [p for (a, v, p) in nb if a == topic and v == "isa"]
        is_attrs = [p for (a, v, p) in nb if a == topic and v == "is"]
        has_parts = [p for (a, v, p) in nb if a == topic and v == "has"]
        lead = []
        if isa_parents:
            lead.append(f"{_art(topic).capitalize()} {topic} is "
                        f"{_join_and([f'{_art(x)} {x}' for x in isa_parents[:3]])}.")
        if is_attrs:
            pron = "It" if isa_parents else f"{_art(topic).capitalize()} {topic}"
            lead.append(f"{pron} is {_join_and(is_attrs[:3])}.")
        if has_parts:
            pron = "It" if (isa_parents or is_attrs) else f"The {topic}"
            lead.append(f"{pron} has {_join_and(has_parts[:4])}.")
        # (2) the rest (action verbs + category-member facts) render one-per-sentence via FT + VERIFY.
        sentences = []
        for (a, v, p) in nb:
            if a == topic and v in ("isa", "is", "has"):
                continue                                                # already covered by the grouped lead
            if v == "isa":                                             # a member's taxonomy fact (noun -> article)
                sentences.append(f"{_art(a)} {a} is {_art(p)} {p}.")
                continue
            if v == "is":                                              # an adjective attribute (no article)
                sentences.append(f"{_art(a)} {a} is {p}.")
                continue
            if v == "has":
                sentences.append(f"{_art(a)} {a} has {p}.")
                continue
            # curriculum ACTION verbs (chase/eat/like) -> the FT render + VERIFY path (fluent + drift-guarded).
            q = f"what does the {a} {v} ?"
            one = self.faculty.answer(f"the {a} {_v3(v)} {p} .", q)
            svos = _extract_all_svos(one, self.agents, self.actions, self.patients, self.inflect)
            ungrounded = [s for s in svos if _fact_key(s) not in self.store_keys]
            # VERIFY: keep ONLY if the SPECIFIC fact is faithfully asserted (on-topic + no drift to another fact) and
            # nothing ungrounded -- this drops both confabulation AND grounded-but-off-topic render drift.
            if ([a, v, p] in svos) and not ungrounded:
                sentences.append(one.strip())
        out = lead + sentences                                          # grouped taxonomy lead, then the rest
        return (f"Here's what I know about the {topic}: " + " ".join(out)) if out \
            else f"I don't know much about the {topic}."

    def turn(self, text):
        """One conversation turn: statement -> learn; question -> gate->answer->verify OR discuss; untaught -> abstain."""
        raw = text.strip()
        toks = [t.strip("?.!,") for t in raw.lower().split()]
        toks = [t for t in toks if t]
        if not toks:
            return "?"
        # LEARN-ON-DEMAND (Phase-15): "learn about X" / "look up X" / "study X" -> fetch + ingest X's real Wikidata
        # facts (the on-demand grounded tail). Checked first so "about"/"learn" don't fall into DISCUSS/LEARN-SVO.
        if (("learn" in toks and "about" in toks) or "look" in toks and "up" in toks or toks[:1] == ["study"]):
            after = toks[toks.index("about") + 1:] if "about" in toks else toks[1:]
            concept = next((t for t in after if t not in _STOP and t.isalpha()), None)
            if concept is not None:
                return self._wikidata_learn(concept)
        # INSTANCE ATTRIBUTE (Phase-14), declarative -- "the dog is/was brown". Checked BEFORE the question split
        # because "is"/"was" are also question cues; a subject-first clause with an attribute + no wh-word/'?' is a
        # STATEMENT. Routes to the minted instance's OWN episodic fact.
        _k0, _pl0 = self._kind_of(toks)
        _ts0 = set(toks)
        if (_k0 is not None and not _pl0 and "the" in _ts0 and ("is" in _ts0 or "was" in _ts0)
                and not (_ts0 & {"what", "who", "how", "why"}) and "?" not in raw
                and any(t in _ATTRS for t in toks) and _k0 in self._last_inst):
            attr = next(t for t in toks if t in _ATTRS)
            self.mta.agent.composer.store(self._last_inst[_k0], "is", attr)
            return f"ok, the {_k0} is {attr}."
        if self._is_question(toks):
            tset = set(toks)
            has_pron = any(t in _PRON for t in toks)
            verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
            subj = next((t for t in toks if t in self.agents), None)
            obj = next((t for t in toks if t in self.patients and t != subj), None)
            if has_pron and subj is None:                       # resolve a pronoun agent via the held referent
                subj = self.mta._resolve("it", query_verb=verb)

            # ELABORATE ("tell me more about the dog" / "what else about the dog") -> a NEW grounded fact via the
            # dlPFC dialogue planner (checked BEFORE describe so 'more'/'else' don't fall into the first-fact describe)
            if ("more" in tset or "else" in tset) and subj is not None:
                _v, reply = self._elaborate(subj)
                return reply

            known = self.agents | self.patients
            def _norm(t):                                    # map a token to a known concept (handles plurals: dogs->dog)
                if t in known:
                    return t
                if t.endswith("es") and t[:-2] in known:
                    return t[:-2]
                if t.endswith("s") and t[:-1] in known:
                    return t[:-1]
                return None
            concepts_in = [c for c in (_norm(t) for t in toks) if c is not None]
            # COMPARE ("how are dogs and cats different?" / "compare X and Y") -> discuss BOTH neighbourhoods
            if ("different" in tset or "compare" in tset or "difference" in tset) and len(concepts_in) >= 2:
                x, y = concepts_in[0], concepts_in[1]
                dx, dy = self._discuss(x), self._discuss(y)
                return f"{dx} And {dy[0].lower()}{dy[1:]}" if dy else dx

            # DISCUSS ("tell me about the dog" / "what do you think about the dog" / "what about predators") ->
            # an open-ended grounded discussion of the topic's neighbourhood (Phase-10), not a one-fact lookup.
            if ("tell" in tset or "about" in tset or "think" in tset) and (subj is not None or concepts_in):
                topic = subj or concepts_in[0]
                return self._discuss(topic)

            # YES/NO ("does the dog eat meat?" / "is it true the dog eats meat?") -> is_it_true
            if ("does" in tset or "do" in tset or "is" in tset or "are" in tset) and subj and verb and obj:
                truth = self.mta.agent.is_it_true(subj, verb, obj)
                if truth == "yes":
                    self.store_keys.add((subj, verb, obj))
                    _p, sent = self._answer(subj, verb)         # RA-render the confirmed fact
                    return f"Yes, {sent[0].lower()}{sent[1:]}" if sent and sent[0].isupper() else f"Yes, {sent}"
                return "No." if truth == "no" else "I don't know."

            # WHO ("who eats meat?") -> agent query
            if "who" in tset and verb and obj:
                who = self.mta.agent.who_does(verb, obj)
                if who is None:
                    return "I don't know."
                self.agents.add(who)
                _p, reply = self._answer(who, verb)
                return reply

            # INSTANCE (Phase-14): a definite singular "the dog" that has a minted instance -> the instance's OWN fact
            # (or inherited via isa). A plural/generic "dogs" (or an un-minted kind) falls through to the kind path.
            kind_q, is_plural_q = self._kind_of(toks)
            verb_q = verb if verb is not None else ("is" if ("is" in tset or "was" in tset) else None)
            if (kind_q is not None and not is_plural_q and "the" in tset and kind_q in self._last_inst
                    and obj is None and verb_q is not None):
                return self._answer_instance(self._last_inst[kind_q], kind_q, verb_q)
            # KIND taxonomy (Phase-15 learned facts): "what is the elephant?" -> its isa parent; "what does the tree
            # have?" -> a has-part. Handles the is/have relations the curriculum action verbs (chase/eat/like) lack.
            if kind_q is not None and obj is None and verb is None:
                if "is" in tset or "was" in tset:
                    par = self.mta.agent.what_does(kind_q, "isa")       # taxonomy: a noun -> "is a mammal"
                    if par is not None:
                        return f"{_art(kind_q)} {kind_q} is {_art(par)} {par}."
                    attr = self.mta.agent.what_does(kind_q, "is")       # adjective -> "is big" (no article)
                    return f"the {kind_q} is {attr}." if attr is not None else "I don't know."
                if "has" in tset or "have" in tset:
                    part = self.mta.agent.what_does(kind_q, "has")
                    return f"the {kind_q} has {part}." if part is not None else "I don't know."
            # normalize a plural/bare kind mention to the kind concept for a GENERIC query ("what do dogs eat?")
            if subj is None and kind_q is not None:
                subj = kind_q

            # WHAT (default) -> patient query
            if subj is None or verb is None:
                return "I don't know."
            _p, reply = self._answer(subj, verb)
            return reply
        # INSTANCE-REP (Phase-14) statements, checked before the kind-fact SVO parse:
        tset_s = set(toks)
        kind_s, is_plural_s = self._kind_of(toks)
        #  MINT: "i saw a dog" / "there is a dog" -> introduce a discourse instance of the kind (dog_1 isa dog).
        if (kind_s is not None and not is_plural_s and "the" not in tset_s
                and ("a" in tset_s or "an" in tset_s) and (tset_s & _INTRO_CUES)):
            tok = self._mint(kind_s)
            return f"ok, a {kind_s}." if tok is not None else "ok."
        #  ATTRIBUTE: "the dog is/was brown" (a minted instance present) -> store the instance's OWN episodic fact.
        if (kind_s is not None and not is_plural_s and "the" in tset_s and ("is" in tset_s or "was" in tset_s)
                and kind_s in self._last_inst):
            attr = next((t for t in toks if t in _ATTRS), None)
            if attr is not None:
                self.mta.agent.composer.store(self._last_inst[kind_s], "is", attr)
                return f"ok, the {kind_s} is {attr}."
        # STATEMENT -> LEARN (growth). parse S V O over the vocab.
        subj = next((t for t in toks if t in self.vocab and t not in _STOP and self.inflect.get(t) not in self.actions), None)
        verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
        obj = None
        if verb is not None:
            after = toks[toks.index(next(t for t in toks if self.inflect.get(t) == verb)) + 1:]
            obj = next((t for t in after if t in self.vocab and t not in _STOP), None)
        if subj and verb and obj:
            self.mta.hear(f"{subj} {verb} {obj}")
            self.store_keys.add((subj, verb, obj))
            # the learned subject/object become known entities so LATER questions find them (growth)
            self.agents.add(subj); self.patients.add(obj)
            return f"ok, i learned that the {subj} {_v3(verb)} {obj}."
        return "sorry, i didn't understand that."


DEMO = [
    "what does the dog chase?",      # 0 -> the dog chases cat.   (writes 'cat')
    "what does it eat?",             # 1 -> it=cat -> the cat eats fish.  (anaphora, Phase 4)
    "the wolf eats rabbit",          # 2 -> ok, learned  (growth, Phase 5)
    "what does the wolf eat?",       # 3 -> the wolf eats rabbit.  (learned fact usable)
    "does the dog eat meat?",        # 4 -> Yes, the dog eats meat.  (yes/no)
    "does the cat eat grass?",       # 5 -> No.  (yes/no negative)
    "who eats meat?",                # 6 -> the dog eats meat.  (who -> agent)
    "tell me about the bird",        # 7 -> the bird eats seed.  (describe)
    "tell me more about the dog",    # 8 -> a NEW dog fact (chase+eat already said -> the dog likes bone)  (elaborate)
    "what does the lion eat?",       # 9 -> I don't know.  (moat)
]

INSTANCE_DEMO = [                    # Phase-14: "which dog?" -- a specific instance vs the generic kind
    "i saw a dog",                   # 0 -> ok, a dog.            (mint dog_1 isa dog)
    "the dog is brown",              # 1 -> ok, the dog is brown. (store the instance's OWN fact)
    "what is the dog?",              # 2 -> The dog is brown.     (the instance's own fact, not the kind's)
    "what does the dog eat?",        # 3 -> the dog eats meat.    (INHERITED from the kind via isa)
    "what do dogs eat?",             # 4 -> the dog eats meat.    (GENERIC "dogs" -> the kind)
    "i saw a cat",                   # 5 -> ok, a cat.            (mint a 2nd instance, different kind)
    "what is the dog?",              # 6 -> The dog is brown.     (distinct-persist: still the dog instance)
    "what does the wolf eat?",       # 7 -> I don't know.         (moat: "wolf" never introduced)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="run the canned demo transcript (Q&A + anaphora + growth + moat)")
    ap.add_argument("--instance-demo", action="store_true", help="run the instance-rep transcript (mint + definite/generic + inherit + distinct + moat)")
    ap.add_argument("--script", default=None, help="'|'-separated turns to run then exit")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    try:
        chat = FluidChat(seed=a.seed)
        print(f"[fluid-chat] ready -- brain (comprehension+knowledge+moat) + a ~{chat.npar:.0f}M brain-gated "
              f"generator (fluency). dev={chat.faculty.device}\n", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"ERROR: {e}"); return 1

    turns = (DEMO if a.demo else INSTANCE_DEMO if a.instance_demo
             else (a.script.split("|") if a.script else None))
    transcript = []
    if turns is not None:
        for t in turns:
            reply = chat.turn(t)
            transcript.append({"you": t.strip(), "brain": reply})
            print(f"  you>   {t.strip()}\n  brain> {reply}", flush=True)
        # a light self-check for the canned demo
        go = None
        if a.instance_demo:
            def _isaid(i, sub):
                return sub in transcript[i]["brain"].lower()
            go = bool("a dog" in transcript[0]["brain"].lower()          # mint
                      and _isaid(1, "brown")                             # attribute stored
                      and _isaid(2, "brown")                             # instance own fact (definite)
                      and _isaid(3, "meat")                              # inherited via isa
                      and _isaid(4, "meat")                              # generic "dogs" -> the kind
                      and _isaid(6, "brown")                             # distinct-persist after a 2nd mint
                      and "know" in transcript[7]["brain"].lower())      # moat
            print(f"\n  [instance-demo self-check] mint/attribute/own/inherit/generic/distinct-persist/moat "
                  f"all correct: {go}", flush=True)
        elif a.demo:
            def _said(i, sub):
                return sub in transcript[i]["brain"].lower()
            elab = transcript[8]["brain"].lower()                             # elaborate -> a NEW dog fact
            go = bool(_said(0, "cat") and _said(1, "fish")                     # what + anaphora
                      and "learned" in transcript[2]["brain"].lower() and _said(3, "rabbit")   # growth + usable
                      and (_said(4, "yes") and _said(4, "meat"))              # yes/no positive
                      and transcript[5]["brain"].lower().startswith(("no", "i don't"))          # yes/no negative
                      and _said(6, "dog")                                     # who -> dog
                      and _said(7, "seed")                                    # describe the bird
                      and ("dog" in elab and ("bone" in elab or "meat" in elab or "cat" in elab))  # elaborate: a dog fact
                      and "know" in transcript[9]["brain"].lower())            # moat
            print(f"\n  [demo self-check] what/anaphora/growth/yes-no/who/describe/elaborate/moat all correct: {go}",
                  flush=True)
        out = {"probe": "fluidconv_chat_repl", "seed": a.seed, "demo": bool(a.demo), "transcript": transcript,
               "demo_all_correct": go, "npar_M": round(chat.npar, 1),
               "elapsed_seconds": round(time.time() - t0, 1)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  [saved] {a.out}", flush=True)
        return 0 if (go is None or go) else 1
    # interactive
    print("  (interactive; blank line or 'quit' to exit)\n", flush=True)
    while True:
        try:
            line = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        print(f"  brain> {chat.turn(line)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
