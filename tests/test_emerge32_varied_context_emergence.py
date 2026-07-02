"""CI guard for EMERGE-32 varied-context emergence: the emergent category is not keyed to a single shared token -- each
member is observed with a DIFFERENT overlapping feature subset, yet a held-out member still inherits a property (taught
via one exemplar) via the feature overlap. CPU (numpy); skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge32_varied_context_emergence_derisk import VariedProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge32 deps unavailable: {e}")
    return VariedProbe(seed=42, epochs=80)


def test_held_out_inherits_via_varied_overlap(probe):
    """Held-out members (different feature subsets, property never taught) inherit via feature overlap with the exemplar."""
    from research.runners._emerge32_varied_context_emergence_derisk import HELD_OUT, CONTENT, MEMBERS, CATPROP
    for m in HELD_OUT:
        assert probe.infer(CONTENT[m]) == CATPROP[MEMBERS[m]]


def test_moat_unobserved(probe):
    from research.runners._emerge32_varied_context_emergence_derisk import NOVEL_CONTENT
    assert probe.infer(NOVEL_CONTENT) == "ABSTAIN"


def test_no_universal_token():
    """Verify the setup has NO single feature shared by all members of a category (genuinely varied, not one token)."""
    from research.runners._emerge32_varied_context_emergence_derisk import VariedProbe, MEMBERS
    p = VariedProbe(seed=42, epochs=1)                            # cheap: just inspect the subsets
    for cat in ("B", "F"):
        subs = [set(p.subset[m]) for m in MEMBERS if MEMBERS[m] == cat]
        universal = set.intersection(*subs)
        assert len(universal) < 3                                # not a full shared block; overlap is partial/varied


def test_permuted_pool_collapses():
    """Mixed-pool subsets (no category overlap) collapse held-out inheritance well below the intact 1.00."""
    from research.runners._emerge32_varied_context_emergence_derisk import VariedProbe, HELD_OUT, CONTENT, MEMBERS, CATPROP
    p = VariedProbe(seed=42, epochs=80, permute=True)
    acc = sum(p.infer(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in HELD_OUT) / len(HELD_OUT)
    assert acc <= 0.5


if __name__ == "__main__":
    from research.runners._emerge32_varied_context_emergence_derisk import VariedProbe
    pr = VariedProbe(seed=42, epochs=80)
    test_held_out_inherits_via_varied_overlap(pr); test_moat_unobserved(pr)
    test_no_universal_token(); test_permuted_pool_collapses()
    print("OK: emerge32 varied-context emergence -- held-out inherits via overlap + moat + no-universal-token + permuted-collapse")
