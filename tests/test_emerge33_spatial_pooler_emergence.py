"""CI guard for EMERGE-33 self-organized emergent superordinate: a competitive HTM Spatial Pooler develops a shared
column block for same-category members from varied inputs; the on-bridge inheritance rides the self-organized block --
a held-out member inherits, a disjoint code abstains. CPU (numpy); skips gracefully if the substrate deps are
unavailable. (Pooler training is ~heavier; kept to seed 42.)"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge33_spatial_pooler_emergence_derisk import PoolerInheritProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge33 deps unavailable: {e}")
    return PoolerInheritProbe(seed=42, epochs=80)


def test_held_out_inherits_via_self_organized_block(probe):
    """A held-out member (property never taught) inherits via the self-organized shared column block."""
    assert probe.held_out_acc() >= 0.9


def test_moat_disjoint_code_abstains(probe):
    assert probe.moat() == 1.0


def test_no_pooler_control_collapses():
    """Random codes (no pooler) form no shared block -> held-out inheritance collapses."""
    from research.runners._emerge33_spatial_pooler_emergence_derisk import PoolerInheritProbe
    p = PoolerInheritProbe(seed=42, epochs=80, pooler=False)
    assert p.held_out_acc() <= 0.5


if __name__ == "__main__":
    from research.runners._emerge33_spatial_pooler_emergence_derisk import PoolerInheritProbe
    pr = PoolerInheritProbe(seed=42, epochs=80)
    test_held_out_inherits_via_self_organized_block(pr); test_moat_disjoint_code_abstains(pr)
    test_no_pooler_control_collapses()
    print("OK: emerge33 spatial pooler -- self-organized block + held-out inheritance + moat + no-pooler collapse")
