"""CI guard for EMERGE-27 multi-level taxonomic inheritance: a concept inherits properties from MULTIPLE levels of its
is-a hierarchy (breathes from ANIMAL 2 levels up + flies from BIRD 1 level up), and a member-specific cancellation at
one dimension does not block inheritance at another (penguin walks but still breathes). CPU (numpy); skips gracefully
if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge27_multilevel_taxonomy_derisk import TaxonomyProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge27 deps unavailable: {e}")
    return TaxonomyProbe(seed=42, epochs=80)


def test_multilevel_inheritance(probe):
    """A concept inherits from BOTH hierarchy levels: breathes (ANIMAL, 2 up) + its mid-level locomotion (never taught)."""
    r = probe.query("robin");  assert r["RESP"] == "breathes" and r["LOCO"] == "flies"
    t = probe.query("trout");  assert t["RESP"] == "breathes" and t["LOCO"] == "swims"


def test_dimension_isolation_cancellation(probe):
    """penguin's specific 'walks' cancels the inherited flies at locomotion, while respiration 'breathes' survives."""
    p = probe.query("penguin")
    assert p["LOCO"] == "walks" and p["RESP"] == "breathes"


def test_moat_no_ancestors_abstains(probe):
    q = probe.query("novel")
    assert q["RESP"] == "ABSTAIN" and q["LOCO"] == "ABSTAIN"


def test_deranged_ancestors_collapse_midlevel():
    """Concepts sharing the WRONG mid-level no longer inherit the correct locomotion -> isolates the mid is-a code."""
    from research.runners._emerge27_multilevel_taxonomy_derisk import TaxonomyProbe, _deranged_ancestors, EXPECT
    p = TaxonomyProbe(seed=42, epochs=80, ancestors=_deranged_ancestors())
    correct = sum(p.query(c)["LOCO"] == EXPECT[c]["LOCO"] for c in ("robin", "trout"))
    assert correct == 0


if __name__ == "__main__":
    from research.runners._emerge27_multilevel_taxonomy_derisk import TaxonomyProbe
    pr = TaxonomyProbe(seed=42, epochs=80)
    test_multilevel_inheritance(pr); test_dimension_isolation_cancellation(pr)
    test_moat_no_ancestors_abstains(pr); test_deranged_ancestors_collapse_midlevel()
    print("OK: emerge27 multi-level taxonomy -- multi-level inheritance + dimension isolation + moat + deranged-collapse")
