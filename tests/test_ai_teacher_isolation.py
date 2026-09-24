"""The AI teacher reaches the brain ONLY through the chat channel (research/runners/ai_teacher.py).

Checked three ways:
  1. static: the teacher module imports only the standard library and names no store-writing function;
  2. strict runtime: a full teacher session (lesson + curiosity answers + quiz + corrections) runs to completion with
     EVERY brain store-writing entry point monkeypatched to raise (research/runners/ai_teacher_guard.py, mode="raise"),
     against a text-only fake brain;
  3. attribution: the stack-aware guard used in real sessions flags a write reached from a teacher frame without the
     chat boundary, and does NOT flag the same write reached through a registered boundary (both directions tested).
"""
import ast
import os
import re
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.runners import ai_teacher as AT  # noqa: E402
from research.runners.ai_teacher_guard import (TeacherIsolationGuard, StoreWriteForbidden,  # noqa: E402
                                               ENTRY_POINTS)

TEACHER_SRC = os.path.join(ROOT, "research", "runners", "ai_teacher.py")
STDLIB_OK = {"__future__", "json", "os", "random", "re", "dataclasses", "typing"}
FORBIDDEN_NAMES = {name for _m, names in ENTRY_POINTS for name in names} | {"composer", "store_conns", "kb"}


class FakeTextBrain:
    """A text-only stand-in for the brain: learns 'the S Vs the O', answers 'what does the S V', and on an unknown
    topic says it does not know and asks back. Pure python: it touches no brain module."""

    def __init__(self, learns=True, wrong=None):
        self.facts = {}
        self.learns = learns
        self.wrong = wrong or {}
        self.heard = []

    def __call__(self, text):
        self.heard.append(text)
        m = re.match(r"^what does the (\w+) (\w+)$", text)
        if m:
            s, v = m.groups()
            if s in self.wrong:
                return "the %s %ss the %s" % (s, v, self.wrong[s])
            if (s, v) in self.facts:
                return "the %s %ss the %s" % (s, v, self.facts[(s, v)])
            return ("I don't know about that. My curiosity is piqued — I haven't learned about %s yet: what can you "
                    "tell me about %s?" % (s, s))
        m = re.match(r"^the (\w+) (\w+) the (\w+)$", text)
        if m and self.learns:
            s, w, o = m.groups()
            for v in {w[:-1], w[:-2], w[:-3] + "y"}:      # crude de-inflection: 'likes'->'like', 'carries'->'carry'
                self.facts[(s, v)] = o
        return text


def _facts(K=4):
    return AT.curriculum_facts(AT.load_curriculum())[:K]


# ── 1. static ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_teacher_imports_only_stdlib():
    tree = ast.parse(open(TEACHER_SRC).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= STDLIB_OK, "the teacher imports %s" % sorted(mods - STDLIB_OK)


def test_teacher_names_no_store_writer():
    tree = ast.parse(open(TEACHER_SRC).read())
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert not (used & FORBIDDEN_NAMES), "the teacher names %s" % sorted(used & FORBIDDEN_NAMES)


# ── 2. strict runtime: every store writer raises; the session still completes ──────────────────────────────────────
def test_full_session_with_every_store_writer_raising():
    brain = FakeTextBrain()
    with TeacherIsolationGuard(mode="raise") as g:
        assert g.n_patched_at_install >= 10, g.report()
        req = {"OneBrainComposer._write_block", "ChatBrain._maybe_acquire", "d6_hebbian_store.hebbian_encode"}
        names = set(g.patched_names)
        for r in req:
            assert any(n.endswith(r.split(".")[-1]) and r.split(".")[0] in n for n in names), (r, names)
        # the patch is live: a real writer now raises
        from research.runners.one_brain_composer import OneBrainComposer
        with pytest.raises(StoreWriteForbidden):
            OneBrainComposer.store(object(), "a", "b", "c")
        t = AT.AITeacher(brain, AT.TeacherKnowledge(_facts(4)))
        rec = t.session(_facts(4))
        assert sum(g.counts.values()) == 1          # only the deliberate sanity call above
    assert all(r["answered_right"] for r in rec["quiz"])
    assert {f.key() for f in _facts(4)} <= set(brain.facts)
    assert all(r["delivered_by"] == "curiosity_answer" for r in rec["lesson"])
    # uninstall restored the originals
    from research.runners.one_brain_composer import OneBrainComposer
    assert not getattr(OneBrainComposer.store, "_ai_teacher_guard", False)


# ── 3. attribution (the real-session guard) ────────────────────────────────────────────────────────────────────────
def _fake_store_module():
    mod = types.ModuleType("_fake_brain_store_for_teacher_test")

    class FakeStore:
        written = []

        def store(self, *a):
            FakeStore.written.append(a)
    FakeStore.__module__ = mod.__name__
    mod.FakeStore = FakeStore
    sys.modules[mod.__name__] = mod
    return mod


def legit_channel_factory(store):
    def legit_channel(text):          # registered as a boundary: the brain's own pipeline writes
        store.store("heard", text)
        return text
    return legit_channel


def test_guard_flags_teacher_write_without_the_channel():
    mod = _fake_store_module()
    store = mod.FakeStore()

    def sneaky_channel(text):         # NOT a boundary: a write reached straight from a teacher frame
        store.store("sneaky", text)
        return text
    g = TeacherIsolationGuard(mode="attribute", entry_points=[(mod.__name__, ["store"])],
                              boundaries={("tests/test_ai_teacher_isolation.py", "legit_channel")})
    with g:
        with pytest.raises(StoreWriteForbidden):
            AT.AITeacher(sneaky_channel, AT.TeacherKnowledge(_facts(2))).lesson(_facts(2))
    assert len(g.violations) == 1 and g.counts[("teacher", "%s.FakeStore.store" % mod.__name__)] == 1


def test_guard_allows_the_same_write_through_the_channel():
    mod = _fake_store_module()
    store = mod.FakeStore()
    g = TeacherIsolationGuard(mode="attribute", entry_points=[(mod.__name__, ["store"])],
                              boundaries={("tests/test_ai_teacher_isolation.py", "legit_channel")})
    with g:
        AT.AITeacher(legit_channel_factory(store), AT.TeacherKnowledge(_facts(2))).lesson(_facts(2))
    assert g.violations == [] and g.counts[("brain", "%s.FakeStore.store" % mod.__name__)] >= 2


# ── teacher behaviour ──────────────────────────────────────────────────────────────────────────────────────────────
def test_teacher_answers_curiosity_with_declaratives():
    brain = FakeTextBrain()
    t = AT.AITeacher(brain, AT.TeacherKnowledge(_facts(2)))
    t.lesson(_facts(2))
    tells = [u for u in t.log if u.act == "curiosity_answer"]
    assert len(tells) == 2 and all(re.match(r"^the \w+ \w+ the \w+$", u.text) for u in tells)
    assert all("?" not in u.text for u in tells)


def test_quiz_corrects_wrong_and_forgotten_answers():
    f = _facts(2)
    forgetful = FakeTextBrain(learns=False)
    t = AT.AITeacher(forgetful, AT.TeacherKnowledge(f))
    rec = t.session(f)
    assert [r["corrected"] for r in rec["quiz"]] == [True, True]
    wrong = FakeTextBrain(wrong={f[0].subject: "narf"})
    t2 = AT.AITeacher(wrong, AT.TeacherKnowledge(f))
    rec2 = t2.session(f)
    assert rec2["quiz"][0]["corrected"] and not rec2["quiz"][0]["answered_right"]
    assert AT.render_tell(f[0]) in [u.text for u in t2.log if u.act == "correction"]


def test_permuted_is_a_derangement_and_varies_by_seed():
    k = AT.TeacherKnowledge(_facts(4))
    for seed in (7, 42, 43, 44, 100, 101, 102):
        p = k.permuted(seed)
        assert all(a.obj != b.obj for a, b in zip(k.facts, p.facts))
        assert sorted(f.obj for f in p.facts) == sorted(f.obj for f in k.facts)
    assert len({tuple(f.obj for f in k.permuted(s).facts) for s in (7, 42, 43, 44, 100, 101, 102)}) > 1


def test_corrupted_teacher_is_wrong_only_where_told():
    cur = AT.load_curriculum()
    k = AT.TeacherKnowledge(_facts(4)).corrupted([1, 3], cur["distractors"])
    assert [f.obj == g.obj for f, g in zip(k.facts, _facts(4))] == [True, False, True, False]
    words = {w for r in cur["facts"] for w in (r["subject"], r["object"])}
    assert not ({f.obj for f in k.facts[1::2]} & words)


def test_curriculum_is_vetted_and_well_formed():
    cur = AT.load_curriculum()
    facts = AT.curriculum_facts(cur)
    assert len({f.key() for f in facts}) == len(facts)
    assert {f.tier for f in facts[:4]} == {"novel", "wikidata"}
    for f in facts:
        if f.tier == "wikidata":
            assert re.fullmatch(r"[0-9a-f]{64}", f.source["facts_json_sha256"]) and isinstance(f.source["index"], int)
        assert re.fullmatch(r"[a-z]+", f.subject) and re.fullmatch(r"[a-z]+", f.obj)
    assert AT.render_tell(facts[0]) == "the blicket eats the dax"
    assert AT.render_ask(facts[0]) == "what does the blicket eat"


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("BRAIN_AI_TEACHER", raising=False)
    assert AT.ai_teacher_enabled() is False
    monkeypatch.setenv("BRAIN_AI_TEACHER", "1")
    assert AT.ai_teacher_enabled() is True


def test_experiment_gate_selftest():
    from research.runners import ai_teacher_experiment as E
    assert E.selftest() == 0
