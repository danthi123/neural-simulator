"""CI guard for EMERGE-45: a three-level discovered taxonomy + transitivity. Stacking the competitive pooler 3 deep
(features -> sub-category -> genus -> order) discovers a 3-level hierarchy; inheritance chains through 2 learned levels so
a held-out sub-category inherits its order property; the sibling order stays false; permuted collapses. CPU (numpy)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge45_three_level_hierarchy_derisk import ThreeLevelProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge45 deps unavailable: {e}")
    return ThreeLevelProbe(seed=42, epochs=40)


def test_two_level_order_inheritance(probe):
    """A held-out sub-category inherits its ORDER property 2 discovered levels up."""
    assert probe.held_out_order_acc() >= 0.8


def test_transitivity_sibling_false(probe):
    """The sibling order's property is NOT inherited (transitive discrimination)."""
    assert probe.transitivity_ok() >= 0.8


def test_permuted_cooccurrence_collapses():
    """Random cross-order co-occurrence -> the hierarchy isn't discovered -> order inheritance collapses."""
    from research.runners._emerge45_three_level_hierarchy_derisk import ThreeLevelProbe
    assert ThreeLevelProbe(seed=42, epochs=40, permute=True).held_out_order_acc() <= 0.7


if __name__ == "__main__":
    from research.runners._emerge45_three_level_hierarchy_derisk import ThreeLevelProbe
    p = ThreeLevelProbe(seed=42, epochs=40)
    assert p.held_out_order_acc() >= 0.8 and p.transitivity_ok() >= 0.8
    print("OK: emerge45 three-level hierarchy -- 2-level order inheritance + transitivity")
