"""CI guard for the EMERGE-29 conversational inference console: teach an is-a taxonomy + class properties, then ask
questions whose answers were NEVER told -- the brain infers them by inheritance up the is-a chain, with a no-confab
moat. CPU (numpy); skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def console():
    try:
        from research.runners._emerge29_inference_console import SemanticConsole, handle
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge29 deps unavailable: {e}")
    c = SemanticConsole(seed=42, epochs=80)
    for line in ["a robin is a bird", "a bird is an animal", "a trout is a fish", "a fish is an animal",
                 "a bird can fly", "an animal can breathe", "a fish can swim"]:
        handle(c, line)
    return c, handle


def test_inheritance_one_level(console):
    c, handle = console
    assert handle(c, "can a robin fly?") == "Yes, a robin can fly."           # inherited from bird (never told)


def test_inheritance_two_levels(console):
    c, handle = console
    assert handle(c, "can a robin breathe?") == "Yes, a robin can breathe."   # inherited from animal, 2 levels up
    assert handle(c, "can a trout breathe?") == "Yes, a trout can breathe."


def test_honest_abstention_not_inherited(console):
    c, handle = console
    assert handle(c, "can a robin swim?") == "I don't know whether a robin can swim."   # robin is not a fish


def test_moat_unknown_concept(console):
    c, handle = console
    assert handle(c, "can a zzz fly?") == "I don't know what a zzz is."       # unknown concept -> moat


if __name__ == "__main__":
    from research.runners._emerge29_inference_console import SemanticConsole, handle
    c = SemanticConsole(seed=42, epochs=80)
    for line in ["a robin is a bird", "a bird is an animal", "a trout is a fish", "a fish is an animal",
                 "a bird can fly", "an animal can breathe", "a fish can swim"]:
        handle(c, line)
    g = (c, handle)
    test_inheritance_one_level(g); test_inheritance_two_levels(g)
    test_honest_abstention_not_inherited(g); test_moat_unknown_concept(g)
    print("OK: emerge29 inference console -- inheritance (1+2 levels) + honest abstention + moat")
