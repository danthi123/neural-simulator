"""CI guard for the unified talkable console (the breadth->knowledge capstone): ONE emergent brain answers
BOTH property (inherit/cancel) AND relational (SVO, any verb) questions, teaches BOTH dimensions live,
remembers across sessions, and abstains (no-confab moat) -- routed by question form.

Locks in the CYCLE 981-997 talkable-brain arc against regression. Skips gracefully if the corpus or the
breadth A->W bridge is absent (the same pattern as the project's other bridge-dependent tests). numpy-only.
"""
import os
import numpy as np
import pytest

CORPUS = "data/corpus/tinystories.txt"
BRIDGE = "bridges/breadth_aw/seed42.simstate.h5"
pytestmark = pytest.mark.skipif(not (os.path.exists(CORPUS) and os.path.exists(BRIDGE)),
                                reason="needs the TinyStories corpus + the breadth A->W bridge (regenerable)")


@pytest.fixture(scope="module")
def console():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
    return UnifiedTalkableConsole(CORPUS, 256, 10, BRIDGE, seed=42,
                                  class_verb="run", exc_verb="sleep", rel_verb="eat")


def test_property_inheritance_and_cancellation(console):
    """A cluster member inherits the class property; the taught exception overrides it (cancellation)."""
    # the setup exception overrides -> 'no'; another inheriting member -> 'yes'
    assert console.exc_word is not None
    out, kind = console.ask(f"does a {console.exc_word} run?")
    assert kind == "override" and out.startswith("no")
    inheritors = [w for (w, _v, _o) in [] ] or [w for w in console.prop.members[console.pos]
                                                if w in console.animals and console.ask(f"does a {w} run?")[1] == "inherit"]
    assert inheritors, "expected at least one inheriting animal in the cluster"
    out, kind = console.ask(f"does a {inheritors[0]} run?")
    assert kind == "inherit" and out.startswith("yes")


def test_relational_answer_and_moat(console):
    """A stored relational fact answers; an unstored relation and an unknown word abstain (moat)."""
    subj = console.rel_facts[0][0]
    out, kind = console.ask(f"what does the {subj} eat?")
    assert kind == "relational" and out and "don't know" not in out
    # moat: unknown word
    _, k1 = console.ask("what does the zzzqqx eat?")
    _, k2 = console.ask("does a zzzqqx run?")
    assert k1 == "moat" and k2 == "moat"


def test_teach_relational_grows_live(console):
    """Teach a NEW relational fact live -> the brain answers it (growth through conversation)."""
    # pick a spellable animal in vocab that is NOT already a subject, + a spellable object
    subjects = {s for (s, _v, _o) in console.rel_facts}
    cand = [a for a in sorted(console.animals) if a in console.row_of and a not in subjects]
    assert len(cand) >= 2, "need two teachable animals"
    subj, obj = cand[0], cand[1]
    assert console.ask(f"what does the {subj} eat?")[1] == "moat"    # not known yet
    assert console.teach_relational(subj, "eat", obj)
    out, kind = console.ask(f"what does the {subj} eat?")
    assert kind == "relational" and obj in out                      # now answered (full sentence contains the obj)


def test_teach_property_exception_grows_live(console):
    """Teach a property EXCEPTION live -> a previously-inheriting member now overrides."""
    inheritors = [w for w in console.prop.members[console.pos]
                  if w in console.animals and w != console.exc_word
                  and console.ask(f"does a {w} run?")[1] == "inherit"]
    if not inheritors:
        pytest.skip("no additional inheriting member to convert into an exception")
    w = inheritors[0]
    assert console.teach_property_exception(w, "sleep")
    out, kind = console.ask(f"does a {w} run?")
    assert kind == "override" and "sleep" in out


def test_relational_any_verb(console):
    """The relational Q&A handles an arbitrary discovered verb (not just 'eat')."""
    # find a spellable verb-like word? use a known common verb present in vocab
    for v in ("like", "see", "want"):
        if v in console.row_of:
            subj = "dog" if "dog" in console.row_of else console.rel_facts[0][0]
            obj = "fish" if "fish" in console.row_of else console.rel_facts[0][2]
            if console.teach_relational(subj, v, obj):
                out, kind = console.ask(f"what does the {subj} {v}?")
                assert kind == "relational" and "don't know" not in out
                return
    pytest.skip("no alternate relational verb present in the discovered vocab")


def test_who_question_subject_recovery(console):
    """A who-question recovers the subject of a stored relational fact; an unknown object abstains."""
    subj, verb, obj = console.rel_facts[0]
    out, kind = console.ask(f"who {verb} {obj}?")
    assert kind == "relational" and subj in out          # the full-sentence answer contains the subject
    _, k = console.ask("who eats zzzqqx?")
    assert k == "moat"


def test_describe_multifact_discourse(console):
    """'tell me about X' aggregates X's facts into connected prose; an unknown word abstains."""
    subj = console.rel_facts[0][0]
    out, kind = console.ask(f"tell me about the {subj}")
    assert kind == "describe" and "." in out and "don't know" not in out
    _, k = console.ask("tell me about the zzzqqx")
    assert k == "moat"
