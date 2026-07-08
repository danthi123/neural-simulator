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
BRIDGE2 = "bridges/breadth_aw2/seed42.simstate.h5"
BRIDGE3 = "bridges/breadth_aw3/seed42.simstate.h5"
pytestmark = pytest.mark.skipif(not (os.path.exists(CORPUS) and os.path.exists(BRIDGE)),
                                reason="needs the TinyStories corpus + the breadth A->W bridge (regenerable)")
_HAS_MULTIBRIDGE = os.path.exists(BRIDGE2) and os.path.exists(BRIDGE3)
BRIDGE4 = "bridges/breadth_aw4/seed42.simstate.h5"
AFFIX = "bridges/affix_aw/seed42.simstate.h5"
_HAS_RICH = _HAS_MULTIBRIDGE and os.path.exists(BRIDGE4) and os.path.exists(AFFIX)


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


def test_multiturn_anaphora(console):
    """A pronoun 'it' resolves to the last-mentioned subject across turns."""
    subj = console.rel_facts[0][0]
    console.ask(f"what does the {subj} eat?")                 # establishes last_subject = subj
    out, kind = console.ask("what does it eat?")              # 'it' -> subj (determiner-robust parse)
    assert kind == "relational" and "don't know" not in out
    # 'it' with the same referent gives the same answer as the explicit subject
    assert console.ask(f"what does the {subj} eat?")[0] == out


def test_multi_exception_isolation(console):
    """Teaching a SECOND property exception does not flip the FIRST (self-correcting re-teach)."""
    if console.exc_word is None:
        pytest.skip("no setup exception member")
    before, _ = console.ask(f"does a {console.exc_word} run?")   # the setup exception -> 'no -- ... can sleep'
    assert before.startswith("no")
    # teach a NEW exception on a different inheriting animal
    other = next((w for w in console.prop.members[console.pos]
                  if w in console.animals and w != console.exc_word
                  and console.ask(f"does a {w} run?")[1] == "inherit"), None)
    if other is None:
        pytest.skip("no other inheriting animal to make a 2nd exception")
    assert console.teach_property_exception(other, "sleep")
    after, kind = console.ask(f"does a {console.exc_word} run?")  # the FIRST exception must still hold
    assert kind == "override" and after.startswith("no")


def test_ditransitive_teach_query_moat(console):
    """The console stores + queries a TERNARY (ditransitive) relation: teach '<s> gives <r> <t>' -> query the
    theme ('what does the s give the r?') + the recipient ('who does the s give a t?'); the unstored abstains."""
    dv = next((v for v in console.DITRANS_VERBS if console.verb_row(v)[0] is not None), None)
    if dv is None:
        pytest.skip("no ditransitive verb in the discovered vocab")
    nouns = [w for w in console.row_of if w in console.spellable][:5]
    if len(nouns) < 4:
        pytest.skip("need >=4 spellable vocab words for the ternary fact + moat probe")
    s, r, t = nouns[0], nouns[1], nouns[2]
    assert console.teach_ditransitive(s, dv, r, t)
    out, kind = console.ask(f"what does the {s} {dv} the {r}?")
    assert kind == "ditransitive" and t in out                    # theme recovered
    out2, kind2 = console.ask(f"who does the {s} {dv} a {t}?")
    assert kind2 == "ditransitive" and r in out2                  # recipient recovered
    _, km = console.ask(f"what does the {nouns[3]} {dv} the {r}?")  # unstored subject
    assert km == "moat"


def test_pp_spatial_teach_query_moat(console):
    """The console stores + queries SPATIAL relations: teach '<s> <v> to <d1>' (goal) + '<s> <v> on <d2>' (loc)
    -> 'where does the s v to?' -> d1, 'where does the s v on?' -> d2 (goal/location kept distinct); unstored abstains."""
    verb = next((v for v in ("run", "walk", "go", "fly", "jump", "look") if console.verb_row(v)[0] is not None), None)
    if verb is None:
        pytest.skip("no spatial-capable verb in the discovered vocab")
    nouns = [w for w in console.row_of if w in console.spellable][:5]
    if len(nouns) < 3:
        pytest.skip("need >=3 spellable vocab words")
    s, d1, d2 = nouns[0], nouns[1], nouns[2]
    assert console.teach_pp(s, verb, d1, "goal")
    assert console.teach_pp(s, verb, d2, "loc")
    out, kind = console.ask(f"where does the {s} {verb} to?")
    assert kind == "spatial" and d1 in out                        # goal destination recovered
    out2, kind2 = console.ask(f"where does the {s} {verb} on?")
    assert kind2 == "spatial" and d2 in out2                      # location destination (distinct from goal)
    _, km = console.ask(f"where does the {nouns[3] if len(nouns) > 3 else 'zzzqqx'} {verb} to?")
    assert km == "moat"                                           # unstored subject -> abstain


@pytest.fixture(scope="module")
def spiking_console():
    """A console whose property answers are GENERATED ON SPIKES (spiking_gen): the slot ORDER is produced by
    the EMERGE-65 self-organized spiking-Broca producer (not a host template). Built once for the module."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
    return UnifiedTalkableConsole(CORPUS, 256, 10, BRIDGE, seed=42,
                                  class_verb="run", exc_verb="sleep", rel_verb="eat", spiking_gen=True)


def test_spiking_gen_property_answer_on_spikes(spiking_console):
    """The property answer's slot ORDER is produced on spikes (spiking-Broca producer) + the gate-first moat:
    an inheriting member -> a well-formed 'yes -- the <w> can run' with the producer invoked; an unknown word
    -> abstain with the producer NEVER invoked (moat by construction)."""
    con = spiking_console
    assert con._producer is not None
    w = next((x for x in con.prop.members[con.pos]
              if x in con.animals and x != con.exc_word and con.ask(f"does a {x} run?")[1] == "inherit"), None)
    if w is None:
        pytest.skip("no inheriting animal in the discovered cluster")
    p0 = con._producer.production_count
    out, kind = con.ask(f"does a {w} run?")
    assert kind == "inherit" and out.startswith("yes -- the ") and " can run" in out   # order ON SPIKES
    assert con._producer.production_count > p0                                          # the producer generated it
    # gate-first moat: an unknown word abstains WITHOUT invoking the producer
    p1 = con._producer.production_count
    _, k = con.ask("does a zzzqqx run?")
    assert k == "moat" and con._producer.production_count == p1                          # producer NOT invoked


@pytest.fixture(scope="module")
def relational_spiking_console():
    """A console whose RELATIONAL (transitive) answers are GENERATED ON SPIKES: the C_TRANS slot ORDER by the
    EMERGE-72/74 registry producer, every word (incl. the 3sg verb) by the 3-bridge A->W. Needs BRIDGE-2/3."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
    return UnifiedTalkableConsole(CORPUS, 256, 10, BRIDGE, seed=42, class_verb="run", exc_verb="sleep",
                                  rel_verb="eat", spiking_gen=True, multi_bridge=True)


@pytest.fixture(scope="module")
def rich_console():
    """A console whose RICHER-relation (ditransitive) answers are GENERATED ON SPIKES (rich_gen): the 8-pool
    EMERGE-77 producer for the order + the ProductiveMultiSpeaker (BRIDGE-1/2/3/4 + affix, productive 3sg) for
    the words. Needs BRIDGE-4 + the affix A->W."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
    return UnifiedTalkableConsole(CORPUS, 256, 10, BRIDGE, seed=42, class_verb="run", exc_verb="sleep",
                                  rel_verb="eat", rich_gen=True)


@pytest.mark.skipif(not _HAS_RICH, reason="needs BRIDGE-2/3/4 + the affix A->W checkpoints (regenerable)")
def test_rich_gen_ditransitive_answer_on_spikes(rich_console):
    """rich_gen: the console's DITRANSITIVE answer is produced ON SPIKES (8-pool producer + productive A->W) --
    teach '<s> gives <r> <t>' -> 'what does the s give the r?' -> the exact 'the s gives the r a t' surface."""
    con = rich_console
    assert con._ditrans_producer is not None
    dv = next((v for v in con.DITRANS_VERBS if con.verb_row(v)[0] is not None and v in con.speaker.vocab), None)
    usable = [w for w in con.row_of if w in con.speaker.vocab]
    if dv is None or len(usable) < 3:
        pytest.skip("no ditransitive verb + fillers in (discovered vocab ∩ speaker vocab)")
    s, r, t = usable[0], usable[1], usable[2]
    assert con.teach_ditransitive(s, dv, r, t)
    out, kind = con.ask(f"what does the {s} {dv} the {r}?")
    assert kind == "ditransitive"
    assert out == f"the {s} {dv}s the {r} a {t}"                  # the exact ditransitive surface, generated on spikes


@pytest.fixture(scope="module")
def neural_route_console():
    """A console whose question-TYPE routing is NEURAL (a fronto-striatal reservoir read-out) instead of the
    host keyword if-ladder. Built once for the module."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_unified_talkable_console import UnifiedTalkableConsole
    return UnifiedTalkableConsole(CORPUS, 256, 10, BRIDGE, seed=42, class_verb="run", exc_verb="sleep",
                                  rel_verb="eat", neural_route=True)


def test_neural_route_dispatches_every_question_type(neural_route_console):
    """The NEURAL router (reservoir read-out) routes every console question type to the correct handler --
    including the hard property-vs-yes-no split (both 'does'-initial), and the moat."""
    con = neural_route_console
    assert con._router is not None
    storable = [a for a in con.animals if a in con.row_of]
    assert len(storable) >= 2
    s, o = storable[0], storable[1]
    con.teach_relational(s, "eat", o)
    inher = next((w for w in con.prop.members[con.pos]
                  if w in con.animals and w != con.exc_word and con.ask(f"does a {w} run?")[1] == "inherit"), None)
    assert inher is not None
    assert con.ask(f"does a {inher} run?")[1] == "inherit"          # property (does a X verb)
    assert con.ask(f"does a {con.exc_word} run?")[1] == "override"  # property exception
    assert con.ask(f"what does the {s} eat?")[1] == "relational"    # relational what
    assert con.ask(f"who eats the {o}?")[1] == "relational"         # relational who
    assert con.ask(f"does the {s} eat {o}?")[1] == "yesno"          # relational yes/no (does the X verb Y)
    assert con.ask(f"tell me about the {s}")[1] == "describe"       # multi-fact discourse
    assert con.ask("does a zzzqqx run?")[1] == "moat"               # abstain (moat unaffected)


@pytest.mark.skipif(not _HAS_MULTIBRIDGE, reason="needs the BRIDGE-2 + BRIDGE-3 A->W checkpoints (regenerable)")
def test_relational_answer_generated_on_spikes(relational_spiking_console):
    """The transitive C_TRANS answer 'the <subj> <verb>s the <obj>' is produced ON SPIKES (registry producer +
    3-bridge A->W). Teach a fully-spellable animal-verb-animal fact, ask it back -> the 5-slot surface renders."""
    con = relational_spiking_console
    assert con._svo_producer is not None                          # C_TRANS mined + producer built
    # a fact whose subject/3sg-verb/object are all covered by the 3-bridge A->W (dog/eats/cat)
    assert con.teach_relational("dog", "eat", "cat")
    out, kind = con.ask("what does the dog eat?")
    assert kind == "relational"
    assert out == "the dog eats the cat"                          # slot order + every word ON SPIKES
    # the moat is unaffected: an unknown relation abstains
    _, k = con.ask("what does the zzzqqx eat?")
    assert k == "moat"
