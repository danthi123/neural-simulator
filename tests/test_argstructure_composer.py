"""CI guard for Tier 0.1 (verb-frame argument structure) + 0.2 (fixed-capacity WM).

Guards the production research/runners/argstructure_composer.py: typed oblique roles + the per-verb frame
lexicon + FrameCQ render + the no-confab moat + the agrammatism anti-cheat (0.1), and the vocabulary-independent
fixed-capacity working-memory neuron count (0.2). All CPU/numpy. See
research/findings/2026-06-27-tier0-argstructure-wm-GO.md.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, FixedCapacityDiscourseWM, reparse_to_fact, FUNCTION_WORDS)


@pytest.fixture(scope="module")
def comp():
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase",
             "park", "ball", "bone", "table", "river"]
    c = ArgStructureComposer(seed=42, D=64, vocab=vocab)
    c.store_fact({"agent": "boy", "action": "go", "GOAL": "park"})
    c.store_fact({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"})
    c.store_fact({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"})
    c.store_fact({"agent": "cat", "action": "chase", "patient": "river"})
    return c


def test_typed_role_recall(comp):
    """A typed oblique role recalls correctly (GOAL/THEME/RECIPIENT/LOCATION beyond the bare patient)."""
    assert comp.query_role("GOAL", agent="boy", action="go") == "park"
    assert comp.query_role("THEME", agent="girl", action="give") == "ball"
    assert comp.query_role("RECIPIENT", agent="girl", action="give") == "dog"
    assert comp.query_role("LOCATION", agent="dog", action="put") == "table"
    assert comp.query_role("patient", agent="cat", action="chase") == "river"


def test_render_boy_goes_to_the_park(comp):
    """The headline render: 'the boy goes to the park' -- preposition 'to' + determiner 'the' from the frame."""
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    assert comp.render(fact, comp._composite_for(fact)) == "the boy goes to the park"


def test_frame_lexicon_coverage(comp):
    """give->THEME+RECIPIENT, put->THEME+LOCATION, default transitive each render with their scaffold."""
    assert comp.render({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
                       comp._composite_for({"agent": "girl", "action": "give"})) \
        == "the girl gives the ball to the dog"
    assert comp.render({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
                       comp._composite_for({"agent": "dog", "action": "put"})) \
        == "the dog puts the bone on the table"
    assert comp.render({"agent": "cat", "action": "chase", "patient": "river"},
                       comp._composite_for({"agent": "cat", "action": "chase"})) \
        == "the cat chases the river"


def test_no_confab_moat(comp):
    """Unstored cues abstain (None); 0 false-accepts."""
    assert comp.query_role("GOAL", agent="boy", action="eat") is None
    assert comp.query_role("GOAL", agent="cat", action="go") is None
    assert comp.query_role("THEME", agent="dog", action="give") is None


def test_verify_reparse(comp):
    """The rendered prose re-parses to the stored typed fact (content-mismatch would reject)."""
    for fact in ({"agent": "boy", "action": "go", "GOAL": "park"},
                 {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
                 {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"}):
        rendered = comp.render(fact, comp._composite_for(fact))
        assert reparse_to_fact(rendered, fact)


def test_agrammatism_anti_cheat(comp):
    """Ablating the closed-class scaffold collapses to telegraphic 'boy go park' (reproduces Broca's): no function
    words, no tense morpheme, and DIFFERENT from the full render -- proving the scaffold does real work."""
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    full = comp.render(fact, comp._composite_for(fact))
    tele = comp.render(fact, comp._composite_for(fact), ablate_closed_class=True)
    assert tele != full
    assert all(w not in FUNCTION_WORDS for w in tele.split())   # no determiners / prepositions
    assert "goes" not in tele.split()                           # bare verb, no agreement morpheme
    assert tele == "boy go park"


def test_fixed_capacity_wm_constant_neuron_count():
    """Tier 0.2: the WM substrate neuron-count is CONSTANT across vocab sizes -- the balloon is gone."""
    counts = set()
    for V in (16, 320, 3000):
        wm = FixedCapacityDiscourseWM(seed=42, D=64, vocab=[f"w{i}" for i in range(V)], n_slots=4)
        wm.hold([f"w{i}" for i in range(3)])
        wm.read(0)
        counts.add(wm.wm_neuron_count())
    assert len(counts) == 1, f"WM neuron-count must be vocab-independent, got {counts}"


def test_fixed_wm_holds_and_reads_in_order():
    """The fixed WM holds an ordered sequence and reads each slot back (the storage/buffer split works)."""
    vocab = [f"w{i}" for i in range(16)]
    wm = FixedCapacityDiscourseWM(seed=42, D=128, vocab=vocab, n_slots=4)
    wm.hold(["w3", "w7", "w1"])
    assert wm.read(0) == "w3"
    assert wm.read(1) == "w7"
    assert wm.read(2) == "w1"
