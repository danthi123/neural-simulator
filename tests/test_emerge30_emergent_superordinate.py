"""CI guard for EMERGE-30 emergent structure from experience: the shared superordinate is not host-designed -- it
emerges from a co-occurrence stream, and inheritance rides the LEARNED grouping. A member never told its category or
property inherits the class property; a never-observed member abstains. CPU (numpy); skips gracefully if the substrate
deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge30_emergent_superordinate_derisk import EmergentProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge30 deps unavailable: {e}")
    return EmergentProbe(seed=42, epochs=80)


def test_inheritance_over_emergent_grouping(probe):
    """Members inherit their latent-category property though the category was NEVER labeled -- only co-occurrence observed."""
    from research.runners._emerge30_emergent_superordinate_derisk import MEMBERS, CONTENT, CATPROP
    for m in MEMBERS:
        assert probe.inherit(CONTENT[m]) == CATPROP[MEMBERS[m]]


def test_held_out_members_inherit(probe):
    """The highlighted held-out members (canary, pike) inherit purely via the co-occurrence-learned grouping."""
    from research.runners._emerge30_emergent_superordinate_derisk import HELD_OUT, CONTENT, MEMBERS, CATPROP
    for m in HELD_OUT:
        assert probe.inherit(CONTENT[m]) == CATPROP[MEMBERS[m]]


def test_moat_never_observed_abstains(probe):
    from research.runners._emerge30_emergent_superordinate_derisk import NOVEL_CONTENT
    assert probe.inherit(NOVEL_CONTENT) == "ABSTAIN"


def test_permuted_context_collapses():
    """Scrambled co-occurrence (random context per member) -> no category emerges -> inheritance collapses to chance."""
    from research.runners._emerge30_emergent_superordinate_derisk import EmergentProbe, MEMBERS, CONTENT, CATPROP
    p = EmergentProbe(seed=42, epochs=80, permute=True)
    acc = sum(p.inherit(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in MEMBERS) / len(MEMBERS)
    assert acc <= 0.7                                            # at/below chance, well under the intact 1.00


if __name__ == "__main__":
    from research.runners._emerge30_emergent_superordinate_derisk import EmergentProbe
    pr = EmergentProbe(seed=42, epochs=80)
    test_inheritance_over_emergent_grouping(pr); test_held_out_members_inherit(pr)
    test_moat_never_observed_abstains(pr); test_permuted_context_collapses()
    print("OK: emerge30 emergent superordinate -- inheritance over LEARNED grouping + held-out + moat + permuted-collapse")
