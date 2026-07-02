"""CI guard for EMERGE-33 self-organized emergent superordinate: a competitive HTM Spatial Pooler develops a shared
column block for same-category members from varied inputs; the on-bridge inheritance rides the self-organized block --
held-out members inherit, a disjoint code abstains, and the dAP-lesion (mechanism removal) collapses it. CPU (numpy);
skips gracefully if deps unavailable. (Pooler training is heavy; seed 42 only. The multi-seed runner validates the
input-destruction permuted-features control.)"""
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
    """Held-out members (property never taught) inherit via the self-organized shared column block."""
    assert probe.held_out_acc() >= 0.83                          # >=5 of 6 held-out (HOLD=3 per category)


def test_moat_disjoint_code_abstains(probe):
    assert probe.moat() == 1.0


def test_dap_lesion_collapses():
    """dAP-LESION (coincidence off -> no priming) deterministically collapses the inheritance -- a clean mechanism-
    ablation control (unlike a fixed-random-code control, which is coincidental in a small column space)."""
    from research.runners._emerge33_spatial_pooler_emergence_derisk import PoolerInheritProbe
    p = PoolerInheritProbe(seed=42, epochs=80, lesion=True)
    assert p.held_out_acc() == 0.0


if __name__ == "__main__":
    from research.runners._emerge33_spatial_pooler_emergence_derisk import PoolerInheritProbe
    pr = PoolerInheritProbe(seed=42, epochs=80)
    assert pr.held_out_acc() >= 0.83 and pr.moat() == 1.0
    assert PoolerInheritProbe(seed=42, epochs=80, lesion=True).held_out_acc() == 0.0
    print("OK: emerge33 spatial pooler -- self-organized block + held-out inheritance + moat + dAP-lesion collapse")
