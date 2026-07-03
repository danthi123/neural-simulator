"""CI guard for EMERGE-44: the stacked pooler discovers a multi-level taxonomy. A second competitive pooler layer pools the
first layer's codons by co-occurrence into superordinates; inheritance chains L1->L2 so a held-out member inherits its
superordinate property; permuted co-occurrence + L1->L2 lesion collapse it. CPU (numpy); skips if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge44_stacked_pooler_derisk import StackedPoolerProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge44 deps unavailable: {e}")
    return StackedPoolerProbe(seed=42, epochs=40)


def test_l2_discovers_superordinates(probe):
    """The L2 pooler groups L1 codons by superordinate (within-super L2 overlap exceeds cross-super)."""
    assert probe.l2_grouping() >= 0.1


def test_superordinate_inheritance_chains(probe):
    """A held-out member inherits its superordinate property via the L1->L2 chain."""
    assert probe.held_out_super_acc() >= 0.8


def test_permuted_cooccurrence_collapses():
    """Random cross-super co-occurrence -> L2 can't discover superordinates -> inheritance collapses."""
    from research.runners._emerge44_stacked_pooler_derisk import StackedPoolerProbe
    assert StackedPoolerProbe(seed=42, epochs=40, permute=True).held_out_super_acc() <= 0.7


if __name__ == "__main__":
    from research.runners._emerge44_stacked_pooler_derisk import StackedPoolerProbe
    p = StackedPoolerProbe(seed=42, epochs=40)
    assert p.l2_grouping() >= 0.1 and p.held_out_super_acc() >= 0.8
    print("OK: emerge44 stacked pooler -- L2 discovers superordinates, inheritance chains L1->L2")
