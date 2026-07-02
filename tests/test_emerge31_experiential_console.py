"""CI guard for the EMERGE-31 experiential console: the owner OBSERVES members co-occurring with contexts (no labels),
the brain DISCOVERS the category grouping, and a property taught via one member is INFERRED for a co-observed member.
The full observe -> learn -> infer -> converse loop, with a no-confab moat. CPU (numpy); skips gracefully if the
substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest

_OBS = ["a robin lives-with a nest", "a sparrow lives-with a nest", "a robin lives-with a treetop",
        "a sparrow lives-with a treetop", "a trout lives-with a river", "a pike lives-with a river",
        "a trout lives-with a reef", "a pike lives-with a reef"]


@pytest.fixture(scope="module")
def console():
    try:
        from research.runners._emerge31_experiential_console import ExperientialConsole, handle
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge31 deps unavailable: {e}")
    c = ExperientialConsole(seed=42, epochs=60)
    for line in _OBS:
        handle(c, line)
    handle(c, "a robin can fly")                                 # teach via ONE member of each emergent group
    handle(c, "a trout can swim")
    return c, handle


def test_taught_member(console):
    c, handle = console
    assert handle(c, "can a robin fly?") == "Yes, a robin can fly."


def test_inferred_via_discovered_grouping(console):
    """A co-observed member inherits the property though it was NEVER told -- via the emergent (discovered) grouping."""
    c, handle = console
    assert handle(c, "can a sparrow fly?") == "Yes, a sparrow can fly."       # co-observed with robin, never told
    assert handle(c, "can a pike swim?") == "Yes, a pike can swim."           # co-observed with trout


def test_honest_abstention_across_groups(console):
    c, handle = console
    assert handle(c, "can a sparrow swim?") == "I don't know whether a sparrow can swim."   # sparrow not in the fish group


def test_moat_unobserved_member(console):
    c, handle = console
    assert handle(c, "can a shark fly?") == "I don't know what a shark is."   # never observed -> moat


if __name__ == "__main__":
    from research.runners._emerge31_experiential_console import ExperientialConsole, handle
    c = ExperientialConsole(seed=42, epochs=60)
    for line in _OBS:
        handle(c, line)
    handle(c, "a robin can fly"); handle(c, "a trout can swim")
    g = (c, handle)
    test_taught_member(g); test_inferred_via_discovered_grouping(g)
    test_honest_abstention_across_groups(g); test_moat_unobserved_member(g)
    print("OK: emerge31 experiential console -- observe -> learn categories -> infer for co-observed member + moat")
