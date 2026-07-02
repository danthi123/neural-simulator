"""CI guard for EMERGE-26 emergent inheritance: Collins-Quillian property inheritance (with cancellation) emerges on the
spiking HTM cortex with no inference engine. A never-taught member inherits its class property from a shared
superordinate code; a member-specific fact cancels the inherited default; a no-superordinate concept abstains. CPU
(numpy); skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge26_emergent_inheritance_derisk import InheritanceProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge26 deps unavailable: {e}")
    return InheritanceProbe(seed=42, epochs=80)


def test_inheritance_beyond_told_facts(probe):
    """Members inherit the class property though their own property was NEVER taught (only 'a BIRD flies'/'a FISH swims')."""
    for m in ("robin", "sparrow", "canary"):
        assert probe.query(m) == "flies"
    for m in ("trout", "salmon"):
        assert probe.query(m) == "swims"


def test_cancellation_specific_beats_inherited(probe):
    """A member-specific fact cancels the inherited default (penguin was told 'walks' -> WALKS, not the inherited flies)."""
    assert probe.query("penguin") == "walks"


def test_moat_no_superordinate_abstains(probe):
    """A concept with no superordinate drives no class pathway -> abstains, never confabulates a property."""
    assert probe.query("novel") == "ABSTAIN"


def test_deranged_superordinate_collapses():
    """Members sharing the WRONG superordinate no longer inherit the correct property -> isolates the is-a code."""
    from research.runners._emerge26_emergent_inheritance_derisk import InheritanceProbe, _deranged_super, MEMBERS_INHERIT
    p = InheritanceProbe(seed=42, epochs=80, super_map=_deranged_super())
    correct = sum(p.query(m) == prop for m, prop in MEMBERS_INHERIT.items())
    assert correct == 0                                          # every member now inherits the wrong class's property


if __name__ == "__main__":
    from research.runners._emerge26_emergent_inheritance_derisk import InheritanceProbe
    pr = InheritanceProbe(seed=42, epochs=80)
    test_inheritance_beyond_told_facts(pr); test_cancellation_specific_beats_inherited(pr)
    test_moat_no_superordinate_abstains(pr); test_deranged_superordinate_collapses()
    print("OK: emerge26 emergent inheritance -- inference beyond told facts + cancellation + moat + deranged-collapse")
