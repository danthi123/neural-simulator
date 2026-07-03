"""CI guard for EMERGE-53 — the PERCEPTION-GROUNDED conversational console: the brain SEES an object through the real
Gabor/V1 front end, the competitive pooler DISCOVERS its category from the VISUAL similarity, and the user talks about
it in plain language (inherit / cancel / abstain). CPU/numpy, offline. Composes EMERGE-34 perception + EMERGE-42/51
inference/console; NO sim/ edit.

These tests pin the load-bearing behaviors on ONE seed (fast): (1) the full scripted transcript -- held-out PERCEIVED
objects inherit their visually-discovered category's property, the exception member cancels, the moat abstains on a
never-seen percept; (2) the PER-IMAGE PIXEL SCRAMBLE control collapses held-out inheritance (the load-bearing perception
control -- destroying the visual similarity makes the categories undiscoverable); (3) the NL front end + moat on a
never-seen object name.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest

from research.runners._emerge53_perception_grounded_conversation import (
    _run_and_check, PerceptionGroundedConsole, handle,
    _BIRD_HELDOUT, _FISH_HELDOUT, _BIRD_EXC, _FISH_EXC,
)

SEED = 42


@pytest.fixture(scope="module")
def checked():
    """Run the scripted perception-grounded transcript once (SEE Gabor/V1 objects -> discover -> teach -> ask)."""
    c, ch = _run_and_check(SEED, scramble=False)
    return c, ch


def test_heldout_perceived_inheritance_cancellation_moat(checked):
    """A NOVEL perceived object (SEEN through Gabor/V1, never named in a fact) inherits its VISUALLY-discovered
    category's property; the exception member answers ITS specific fact (cancellation); the moat abstains on a
    never-seen percept."""
    c, ch = checked
    # held-out perceived-object inheritance (owl/wren -> bird 'fly', minnow/gar -> fish 'swim'); >= 0.75 gate
    assert ch["inherit"] >= 0.75, f"held-out perceived inheritance {ch['inherit']} < 0.75"
    # cancellation: the exception members answer their OWN specific fact, not the class default
    assert ch["cancel"] >= 0.99, f"cancellation {ch['cancel']} < 0.99"
    # the no-confab moat abstains on a never-seen / visually-novel percept
    assert ch["moat_unknown"] is True
    # RSA pixel-provenance: the discovered structure tracks the raw-pixel visual similarity (label-free)
    assert ch["rsa"] >= 0.5, f"RSA pixel-provenance {ch['rsa']} < 0.5"


def test_natural_language_replies(checked):
    """The natural-language replies read correctly: a held-out perceived bird inherits fly; the penguin cancels to
    walks; a never-seen token abstains."""
    c, ch = checked
    assert handle(c, "can a %s fly?" % _BIRD_HELDOUT[0]).startswith("Yes")     # INHERIT via visually-discovered bird codon
    assert handle(c, "can a %s fly?" % _BIRD_EXC[0]).startswith("No")          # CANCEL -- penguin's own exception
    assert handle(c, "can a zzz fly?").startswith("I don't know")              # MOAT -- never seen
    # the moat is also 0 false-accepts across several never-seen tokens
    fa = sum(0 if c.moat_abstains(t, "fly") else 1 for t in ("zzz", "qqq", "wobble"))
    assert fa == 0, f"moat false-accepts {fa} != 0"


def test_per_image_scramble_collapses_inheritance():
    """The LOAD-BEARING perception control: per-image pixel scramble destroys the visual similarity -> the pooler can't
    discover the categories -> held-out PERCEIVED-object inheritance collapses (isolating real perception as the cause)."""
    _, ch = _run_and_check(SEED, scramble=False)
    _, chs = _run_and_check(SEED, scramble=True)
    assert ch["inherit"] >= 0.75
    assert chs["inherit"] <= ch["inherit"] - 0.30, (
        f"scramble did not collapse inheritance: intact {ch['inherit']} vs scrambled {chs['inherit']}")
    # the RSA pixel-provenance also collapses under scramble (structure was in the visual features)
    assert chs["rsa"] <= ch["rsa"] - 0.30, f"scramble RSA {chs['rsa']} vs intact {ch['rsa']}"


def test_moat_on_unseen_object_name():
    """A fresh console: an object the brain has never SEEN -> the perception front end returns nothing -> the console
    abstains (does not invent a category)."""
    c = PerceptionGroundedConsole(seed=SEED)
    # never call see(...) -> asking about any object abstains
    assert c.ask_can("robin", "fly").startswith("I don't know")
    # 'see' a novel/never-rendered name -> the perceptual world has no such object -> honest abstain
    assert handle(c, "see wobble").startswith("I don't know")
