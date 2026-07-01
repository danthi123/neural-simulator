"""CI guard for the fluid-conversation CONSOLE (`research/runners/_fluidconv_chat_repl.py`).

Pins the two OFFLINE self-checks the console ships with -- the base demo (what/anaphora/growth/yes-no/who/describe/
elaborate/moat) and the instance-rep demo (mint/attribute/own-fact/isa-inheritance/generic/distinct-persist/moat) --
so the multi-turn grounded loop can't silently regress. CPU (numpy); skips gracefully when the fine-tuned generator
checkpoint or the curriculum is absent (a build artifact, like the concept-cache-gated composer tests). NO network:
the Wikidata learn-on-demand path is exercised by the Phase-15 de-risk, not here.
"""
from __future__ import annotations
import os
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")   # must precede the sim.backend import; CPU-portable

from research.runners._fluidconv_phase2_ra_finetune import FT_CKPT  # noqa: E402
from research.runners._grounded_lang_p2_derisk import CURRICULUM  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.path.exists(FT_CKPT) and os.path.exists(os.path.abspath(CURRICULUM))),
    reason="fluid-conv console needs the RA fine-tune checkpoint + curriculum (build artifacts)",
)


@pytest.fixture(scope="module")
def chat():
    from research.runners._fluidconv_chat_repl import FluidChat
    return FluidChat(seed=42)


def _replies(chat, turns):
    return [chat.turn(t) for t in turns]


def test_base_demo_self_check(chat):
    """The base fluid-conversation demo: what + anaphora + growth + yes/no + who + describe + elaborate + moat."""
    from research.runners._fluidconv_chat_repl import DEMO
    r = _replies(chat, DEMO)

    def said(i, sub):
        return sub in r[i].lower()

    assert said(0, "cat")                                   # what does the dog chase -> cat
    assert said(1, "fish")                                  # anaphora: it (=cat) eats fish
    assert "learned" in r[2].lower() and said(3, "rabbit")  # growth: learn + reuse
    assert said(4, "yes") and said(4, "meat")               # yes/no positive
    assert r[5].lower().startswith(("no", "i don't"))       # yes/no negative
    assert said(6, "dog")                                   # who eats meat -> dog
    assert said(7, "seed")                                  # describe the bird
    elab = r[8].lower()
    assert "dog" in elab and ("bone" in elab or "meat" in elab or "cat" in elab)   # elaborate: a dog fact
    assert "know" in r[9].lower()                           # moat: untaught -> I don't know


def test_discourse_plan_synthesis(chat):
    """Phase-16: DISCUSS renders a topic's grounded facts as ONE connected prose (aggregation + connectives), and
    COMPARE fires a checkable Contrast -- grounded by construction, no free generation."""
    dog = chat.turn("tell me about the dog").lower()
    # connected prose: an aggregation/Joint connective is present (not one isolated sentence per fact)
    assert (" and " in dog) or ("; it " in dog)
    assert "dog" in dog and "meat" in dog                    # grounded (the dog's stored facts)
    cmp = chat.turn("compare dog and cat").lower()
    assert "but" in cmp                                       # checkable Contrast (the dog/cat patients differ)
    assert "dog" in cmp and "cat" in cmp
    # checkable GIST: teach a wolf that shares a fact with the dog, then ask what they share
    chat.turn("the wolf eats meat")
    shared = chat.turn("what do dogs and wolves share?").lower()
    assert "both" in shared and "eat meat" in shared          # entailment-only intersection (dog+wolf both eat meat)


def test_persistence_save_load(chat, tmp_path):
    """Phase-17: a learned fact is written to the persist file and survives a save/load round-trip (the brain
    remembers). Uses the cache-backed `learn about elephant` (offline) so real Wikidata facts populate the grown set.
    Cross-instance reload is covered by the fast bare-agent de-risk; this guards the save_state/load_state plumbing."""
    import json as _json
    reply = chat.turn("learn about elephant").lower()         # cache-backed (offline); populates the grown set
    assert "elephant" in reply and "mammal" in reply
    state = str(tmp_path / "state.json")
    n = chat.save_state(state)
    saved = [tuple(f) for f in _json.loads(open(state).read())["learned"]]
    assert ("elephant", "isa", "mammal") in saved             # persisted
    assert chat.load_state(state) == 0                        # idempotent (already known -> 0 new)
    assert chat.mta.agent.what_does("elephant", "isa") == "mammal"   # still recalled
    assert n >= 1
    # taxonomy chain (real 2-level Wikidata subclass, offline via the parent-extended cache: elephant->mammal->vertebrata)
    cls = chat.turn("how is the elephant classified?").lower()
    assert "mammal" in cls and "which is" in cls
    # grounded WHY: the isa-path explanation (offline) + the moat (no fabricated reason)
    why = chat.turn("why is an elephant a vertebrata?").lower()
    assert why.startswith("because") and "mammal" in why      # explained via the real path
    assert "know" in chat.turn("why is an elephant a fish?").lower()   # moat: fish not an ancestor -> abstain


def test_instance_demo_self_check():
    """Phase-14 instance-rep in the console: 'which dog?' -- a specific referent vs the generic kind, on a FRESH
    console (a distinct instance-mint state from the base-demo fixture)."""
    from research.runners._fluidconv_chat_repl import FluidChat, INSTANCE_DEMO
    c = FluidChat(seed=42)
    r = [c.turn(t) for t in INSTANCE_DEMO]

    def said(i, sub):
        return sub in r[i].lower()

    assert "a dog" in r[0].lower()          # mint
    assert said(1, "brown")                 # the dog is brown -> stored
    assert said(2, "brown")                 # what is the dog? -> the instance's OWN fact (definite)
    assert said(3, "meat")                  # what does the dog eat? -> inherited via isa
    assert said(4, "meat")                  # what do dogs eat? -> the generic kind
    assert said(6, "brown")                 # after minting a cat, "the dog" STILL -> brown (distinct-persist)
    assert "know" in r[7].lower()           # moat: "the wolf" never introduced
