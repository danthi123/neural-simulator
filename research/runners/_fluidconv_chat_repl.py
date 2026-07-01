"""The fluid-conversation CONSOLE -- one coherent chat loop tying Phases 2-5 together (the owner's console-not-
dashboard priority). Talk to the brain like an LLM: ask grounded questions, use pronouns across turns, TEACH it new
facts, and it abstains ("I don't know") on what it hasn't learned.

  QUESTION  "what does the dog eat?" / "what does it chase?" (pronoun) / "who eats meat?" / "does the dog eat meat?"
            -> interrogative parse -> brain GATE (moat gate-FIRST) -> RA-fine-tuned 21M focused answer -> VERIFY.
  STATEMENT "the wolf eats rabbit" / "wolf eat rabbit"  -> hear (LEARN) -> "ok, i learned that the wolf eats rabbit."
  UNTAUGHT  -> "I don't know."   (the no-confab moat)

Assembles: `MultiTurnAgent` (multi-turn anaphora, Phase 4) + `FTFaculty` (the RA render/QA fine-tuned generator,
Phase 2) + the Phase-3 gate->answer->VERIFY + Phase-5 growth. The BRAIN does comprehension + knowledge + grounding +
moat; the minimized (~21M) brain-trained brain-gated generator does fluency. Reuse-by-import; NO sim/ edit.

Run (scripted smoke / demo):
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --demo
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --script "what does the dog eat?|the wolf eats rabbit|what does the wolf eat?"
Run (interactive): ... (no --script/--demo -> reads stdin; blank line or 'quit' exits)
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

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

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_chat_repl_demo.json"
_QWORDS = {"what", "who", "does", "do", "is", "are", "tell", "can", "could", "why", "how", "when", "where", "?"}
_PRON = {"it", "its", "they", "them", "that"}
_STOP = {"the", "a", "an", "does", "do", "did", "the", "to", "of", "please"}


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
        # a generous pre-allocated vocab so new facts can be TAUGHT (composer codes are fixed at build): curriculum +
        # the fine-tune's broad subject/object pools + any extra.
        vocab = set(_collect_vocab(self.cur)) | set(FT_SUBJECTS) | set(FT_OBJECTS) | set(extra_vocab or [])
        self.vocab = sorted(vocab)
        # referents (for anaphora) must stay small (one 40-neuron attractor/referent in n=600) -> a curated set
        referents = sorted(set(sorted(self.agents)[:6]) | set(list(self.patients)[:4]))
        self.mta = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in self.vocab},
                                  seed=seed, defer_planner=True, enable_biased_competition=False, composer_kind="rf")
        _teach(self.mta.agent, self.cur)
        self.store_keys = {tuple(f) for f in facts}
        self.faculty = FTFaculty()
        self.npar = self.faculty.npar
        self._mentioned = {}          # subject -> set of verbs already said (so "tell me more" surfaces a NEW fact)

    def _content(self, toks):
        subj = next((t for t in toks if t in self.agents or t in self.vocab and t not in _STOP and t not in self.actions), None)
        verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
        return subj, verb

    def _is_question(self, toks):
        return bool(set(toks) & _QWORDS)

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
        facts = [list(f) for f in kb if topic in (f[0], f[2])]
        # category members ONLY via the "is" relation (X is <topic>) -- a non-"is" patient (dog chase cat) does NOT
        # make the agent a member, so a regular topic doesn't vacuum in unrelated facts.
        for m in [f[0] for f in kb if f[2] == topic and f[1] == "is"]:
            facts += [list(f) for f in kb if f[0] == m and list(f) not in facts]
        return facts

    def _discuss(self, topic, *, max_facts=7):
        """Open-ended grounded DISCUSSION (Phase-10): render each neighbourhood fact FAITHFULLY (single-fact; a
        multi-fact context makes the 21M confabulate) + per-sentence VERIFY + concatenate. Moat: an ungrounded render
        is dropped; an empty neighbourhood hedges."""
        nb = []                                                         # dedup the neighbourhood
        for f in self._neighbourhood(topic):
            if f not in nb:
                nb.append(f)
            if len(nb) >= max_facts:
                break
        if not nb:
            return f"I don't know much about the {topic}."
        sentences = []
        for (a, v, p) in nb:
            q = f"what is the {a} ?" if v == "is" else f"what does the {a} {v} ?"
            one = self.faculty.answer(f"the {a} {_v3(v)} {p} .", q)
            svos = _extract_all_svos(one, self.agents, self.actions, self.patients, self.inflect)
            ungrounded = [s for s in svos if _fact_key(s) not in self.store_keys]
            # VERIFY: keep ONLY if the SPECIFIC fact is faithfully asserted (on-topic + no drift to another fact) and
            # nothing ungrounded -- this drops both confabulation AND grounded-but-off-topic render drift.
            if ([a, v, p] in svos) and not ungrounded:
                sentences.append(one.strip())
        return (f"Here's what I know about the {topic}: " + " ".join(sentences)) if sentences \
            else f"I don't know much about the {topic}."

    def turn(self, text):
        """One conversation turn: statement -> learn; question -> gate->answer->verify OR discuss; untaught -> abstain."""
        raw = text.strip()
        toks = [t.strip("?.!,") for t in raw.lower().split()]
        toks = [t for t in toks if t]
        if not toks:
            return "?"
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

            # WHAT (default) -> patient query
            if subj is None or verb is None:
                return "I don't know."
            _p, reply = self._answer(subj, verb)
            return reply
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="run the canned demo transcript (Q&A + anaphora + growth + moat)")
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

    turns = DEMO if a.demo else (a.script.split("|") if a.script else None)
    transcript = []
    if turns is not None:
        for t in turns:
            reply = chat.turn(t)
            transcript.append({"you": t.strip(), "brain": reply})
            print(f"  you>   {t.strip()}\n  brain> {reply}", flush=True)
        # a light self-check for the canned demo
        go = None
        if a.demo:
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
