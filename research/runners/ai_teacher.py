"""AI-TEACHER — a separate conversation partner that teaches the brain through CHAT ONLY (roadmap P2.1).

WHY. Owner direction (2026-09-24): the brain's knowledge must be LEARNED by the brain and grow over time like a
human's, never "fancy RAG the LLM pulls from"; the mouth stays fact-free; "I don't know" should become a learning
moment. The master roadmap (docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md, P2.1) names a TEMPORARY AI-teacher
scaffold that accelerates early growth and later graduates to real-human interaction. Under the brain-based-only
standard the teacher is legitimate host code ONLY as the SOCIAL ENVIRONMENT: it produces sentences and hears replies,
exactly as a person talking to a child does. Everything between hearing a sentence and answering one is the brain's
job (comprehension, the decision to learn, the synaptic write, recall, the curiosity ask).

THE ONE CHANNEL. `AITeacher(channel=...)` receives a callable `channel(text) -> reply_text`. That is the teacher's
only access to the brain: it sends a sentence and reads back the reply STRING (never the reply's structured fields,
never the composer, never a store). This module imports nothing from `sim/`, `webapp/`, or any brain runner, and it
never calls a store writer (`store`, `store_fact`, `_write_block`, `build_ltm_from_facts`, `encode_fast`,
`hebbian_encode`, ...). Both properties are CHECKED, not asserted: tests/test_ai_teacher_isolation.py scans this
file's imports and runs a full teacher session with every store-writing entry point monkeypatched to raise; the
experiment runner also installs a stack-aware guard (research/runners/ai_teacher_guard.py) during real sessions.

WHAT THE TEACHER DOES (template-only today; an LLM may later PARAPHRASE a vetted sentence, never originate a fact):
  (a) CURIOSITY ANSWER — when the brain's reply contains an ask ("... what can you tell me about blicket?"), the
      teacher answers with simple declarative sentences for every fact it holds about that topic.
  (b) LESSON — a paced curriculum: for each fact it first asks the brain the question the fact answers (the brain
      usually abstains and asks back — the learning moment), then answers the ask, or tells the fact if the brain
      did not ask.
  (c) QUIZ — later it asks each question again, judges the reply TEXT, and restates the fact when the reply is wrong.
The teacher's facts come from a VETTED source (research/fixtures/ai_teacher_curriculum_v1.json: wikidata triples
rendered by one template, plus invented-noun facts the brain cannot already know). Its BELIEFS may be manipulated by
the experimenter (a permuted or partly-corrupted copy of the vetted facts) to test that the brain learns what it was
TOLD, not what the source says.

Default-OFF, two ways (amendment 2 of the pre-registration made the flag load-bearing; before it, the flag was a label
nothing read, and default-OFF held only because nothing imports this module): (1) `AITeacher(...)` raises
`AITeacherDisabled` unless BRAIN_AI_TEACHER is on (`ai_teacher_enabled`); the experiment runner sets it per arm,
production never does; (2) nothing in webapp/, sim/ or the production runners imports this module (only the
experiment runner and its tests do).

HONESTY BOUNDARY. The teacher's judgement of a reply (right/wrong) is a string match on the reply text, the same
information a human teacher has. It is an environment decision, not a brain read-out, and it is never credited to
the brain.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
CURRICULUM_PATH = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "research", "fixtures",
                               "ai_teacher_curriculum_v1.json")

_ON = ("1", "true", "on", "yes")


def ai_teacher_enabled() -> bool:
    """Harness flag (default OFF). `AITeacher` refuses to start unless it is on; production never sets it."""
    return os.environ.get("BRAIN_AI_TEACHER", "").strip().lower() in _ON


class AITeacherDisabled(RuntimeError):
    """Raised when an `AITeacher` is constructed with BRAIN_AI_TEACHER off (the default)."""


# ── the vetted fact source ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    verb: str          # base form ("eat"); rendered 3sg in a statement ("eats")
    obj: str
    tier: str          # "novel" (invented nouns) | "wikidata" (rendered wikidata triple)
    source: dict = field(default_factory=dict, compare=False, hash=False)

    def key(self):
        return (self.subject, self.verb)


def third_person(verb: str) -> str:
    """Template morphology for the statement form. Covers the curriculum's verbs; not a general English inflector."""
    v = verb.lower()
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    if len(v) > 1 and v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def render_tell(f: Fact) -> str:
    """The declarative sentence a teacher says: 'the blicket eats the dax'."""
    return "the %s %s the %s" % (f.subject, third_person(f.verb), f.obj)


def render_ask(f: Fact) -> str:
    """The question a teacher asks: 'what does the blicket eat' (same frame the D6 protocol validated)."""
    return "what does the %s %s" % (f.subject, f.verb)


def load_curriculum(path: str = CURRICULUM_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def curriculum_facts(cur: dict) -> List[Fact]:
    """Facts in the curriculum's teaching ORDER."""
    by_id = {r["id"]: r for r in cur["facts"]}
    out = []
    for fid in cur["order"]:
        r = by_id[fid]
        out.append(Fact(id=r["id"], subject=r["subject"], verb=r["verb"], obj=r["object"], tier=r["tier"],
                        source=r.get("provenance", {})))
    return out


class TeacherKnowledge:
    """What the teacher BELIEVES (a list of facts). Built from the vetted source; the experimenter may hand the
    teacher a permuted or corrupted copy to test that the brain learns what it is told."""

    def __init__(self, facts: List[Fact], label: str = "vetted"):
        self.facts = list(facts)
        self.label = label
        keys = [f.key() for f in self.facts]
        if len(set(keys)) != len(keys):
            raise ValueError("each (subject, verb) must be unique: one question -> one answer")

    def about(self, topic: str) -> List[Fact]:
        t = (topic or "").lower()
        return [f for f in self.facts if f.subject == t]

    def answer_for(self, f: Fact) -> Optional[Fact]:
        for g in self.facts:
            if g.key() == f.key():
                return g
        return None

    def permuted(self, seed: int) -> "TeacherKnowledge":
        """A DERANGEMENT of the objects (no fact keeps its own object), drawn per seed. The teacher then teaches
        'the blicket eats the <someone else's object>' and believes it."""
        objs = [f.obj for f in self.facts]
        n = len(objs)
        if n < 2:
            raise ValueError("a derangement needs >= 2 facts")
        rng = random.Random(1000003 * int(seed) + 17)
        while True:
            perm = list(range(n))
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)) and all(objs[perm[i]] != objs[i] for i in range(n)):
                break
        return TeacherKnowledge([Fact(f.id, f.subject, f.verb, objs[perm[i]], f.tier,
                                      dict(f.source, belief="permuted"))
                                 for i, f in enumerate(self.facts)], label="permuted(seed=%d)" % seed)

    def corrupted(self, positions, distractors) -> "TeacherKnowledge":
        """A teacher that is WRONG about the facts at `positions` (object replaced by a distractor noun that appears
        nowhere else in the curriculum) and right about the rest. Used to measure teacher-error propagation."""
        pos = list(positions)
        if len(distractors) < len(pos):
            raise ValueError("need one distractor per corrupted position")
        out = []
        for i, f in enumerate(self.facts):
            if i in pos:
                d = distractors[pos.index(i)]
                out.append(Fact(f.id, f.subject, f.verb, d, f.tier, dict(f.source, belief="corrupted",
                                                                         truth=f.obj)))
            else:
                out.append(f)
        return TeacherKnowledge(out, label="corrupted(%s)" % ",".join(str(p) for p in pos))


# ── reading a reply (text only, as a person would) ─────────────────────────────────────────────────────────────────
CURIOSITY_ASK_RE = re.compile(r"what can you tell me about (?:the |a |an )?([a-z][a-z'\-]*)", re.I)
_ABSTAIN_RE = re.compile(r"^\s*i don'?t know\b", re.I)


def curiosity_topics(reply: str) -> List[str]:
    """Topics the brain ASKED about in this reply ('... what can you tell me about blicket?')."""
    return [m.group(1).lower() for m in CURIOSITY_ASK_RE.finditer(reply or "")]


def mentions(reply: str, word: str) -> bool:
    return re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(word.lower()), (reply or "").lower()) is not None


def reply_answers(reply: str, f: Fact) -> bool:
    """A teacher's judgement of a quiz reply: it names the expected object and is not an 'I don't know'."""
    return mentions(reply, f.obj) and not _ABSTAIN_RE.match(reply or "")


# ── the teacher ────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Utterance:
    phase: str
    act: str            # ask | tell | curiosity_answer | correction
    fact_id: Optional[str]
    text: str
    reply: str


class AITeacher:
    """A template teacher. `channel(text) -> reply_text` is its ONLY access to the brain."""

    def __init__(self, channel: Callable[[str], str], knowledge: TeacherKnowledge,
                 answer_other_topics: bool = True):
        if not ai_teacher_enabled():
            raise AITeacherDisabled("BRAIN_AI_TEACHER is off (the default): the AI teacher does not start")
        if not callable(channel):
            raise TypeError("channel must be callable(text) -> reply text")
        self._channel = channel
        self.knowledge = knowledge
        self.answer_other_topics = answer_other_topics
        self.log: List[Utterance] = []
        self.told: Dict[str, int] = {}

    # the only I/O
    def _say(self, phase, act, fact_id, text) -> str:
        reply = self._channel(text)
        if not isinstance(reply, str):
            raise TypeError("the channel must hand the teacher reply TEXT only (got %s)" % type(reply).__name__)
        self.log.append(Utterance(phase, act, fact_id, text, reply))
        if act in ("tell", "curiosity_answer", "correction") and fact_id is not None:
            self.told[fact_id] = self.told.get(fact_id, 0) + 1
        return reply

    def answer_curiosity(self, reply: str, phase: str, restrict_to=None) -> List[str]:
        """(a) Answer every ask in `reply` about a topic the teacher knows, one declarative sentence per fact.
        Returns the fact ids told. `restrict_to` limits the answer to facts in that id set (the lesson's pacing)."""
        told = []
        for topic in curiosity_topics(reply):
            for f in self.knowledge.about(topic):
                if restrict_to is not None and f.id not in restrict_to:
                    continue
                if f.id in told:
                    continue
                self._say(phase, "curiosity_answer", f.id, render_tell(f))
                told.append(f.id)
        return told

    def lesson(self, facts: List[Fact]) -> List[dict]:
        """(b) A paced lesson. For each fact: ask first; if the brain asks back about the subject, answer the ask;
        otherwise tell the fact. Records how each fact was delivered and whether the brain already knew it."""
        records = []
        allowed = set()
        for f in facts:
            belief = self.knowledge.answer_for(f) or f
            allowed.add(belief.id)
            reply = self._say("lesson", "ask", belief.id, render_ask(belief))
            pre_known = reply_answers(reply, belief)
            delivered = []
            if not pre_known:
                if belief.subject in curiosity_topics(reply):
                    delivered = self.answer_curiosity(reply, "lesson", restrict_to=allowed)
                if belief.id not in delivered:
                    self._say("lesson", "tell", belief.id, render_tell(belief))
                    delivered.append(belief.id)
                    how = "tell"
                else:
                    how = "curiosity_answer"
            else:
                how = "already_known"
            records.append({"fact_id": belief.id, "pre_known": pre_known, "delivered_by": how,
                            "asked_back": belief.subject in curiosity_topics(reply)})
        return records

    def quiz(self, facts: List[Fact], correct: bool = True) -> List[dict]:
        """(c) Ask each question again; restate the fact when the reply is wrong (if `correct`)."""
        records = []
        for f in facts:
            belief = self.knowledge.answer_for(f) or f
            reply = self._say("quiz", "ask", belief.id, render_ask(belief))
            ok = reply_answers(reply, belief)
            corrected = False
            if not ok and correct:
                self._say("quiz", "correction", belief.id, render_tell(belief))
                corrected = True
            records.append({"fact_id": belief.id, "answered_right": ok, "corrected": corrected})
        return records

    def session(self, facts: List[Fact], quiz: bool = True) -> dict:
        lesson = self.lesson(facts)
        quiz_rec = self.quiz(facts) if quiz else []
        return {"knowledge": self.knowledge.label, "lesson": lesson, "quiz": quiz_rec,
                "utterances": [asdict(u) for u in self.log], "told": dict(self.told)}
