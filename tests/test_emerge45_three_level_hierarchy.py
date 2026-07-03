"""CI guard for EMERGE-45: a three-level discovered taxonomy. Stacking the competitive pooler 3 deep (features ->
sub-category -> genus -> order) discovers a 3-level hierarchy; a held-out sub-category infers its order property with
~zero sibling-confusion; permuted collapses. HONEST FRAMING (per the 2026-07-02 control-completeness audit): the
old NORDER=2 'transitivity' metric was near-tautological with order_acc (not-sibling == correct-or-abstain), so it is
replaced by SIBLING-CONFUSION (fraction inferring the WRONG order, separate from abstentions -- should be ~0); and an
L2/genus-only readout already carries most of the order signal (genus-proximity floor), so most of the inference chains
through the L2/genus grouping with L3 a smaller, seed-variable increment. CPU (numpy)."""
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


def test_held_out_order_inheritance(probe):
    """A held-out sub-category infers its ORDER property from the discovered hierarchy (GO gate; order_acc stands)."""
    assert probe.held_out_order_acc() >= 0.8


def test_sibling_confusion_near_zero(probe):
    """No held-out member commits the WRONG (sibling) order -- the honest, non-tautological discrimination metric.
    (Separate from abstentions: unlike the old 'transitivity', an abstain does NOT count as a pass here.)"""
    assert probe.sibling_confusion_rate() <= 0.05


def test_l2genus_floor_is_honest_floor(probe):
    """HONEST-CONTROL disclosure: each held-out sub shares its genus with exactly one trained sub, so an L2/genus-only
    readout already carries most of the order signal (above chance 0.5), and the full L3 readout is >= that floor. This
    pins the audit's finding that L2/genus grouping -- not L3 alone -- carries most of the signal."""
    floor = probe.held_out_order_acc_l2only()
    assert floor >= 0.55                                          # genus-proximity floor is well above chance (0.5)
    assert probe.held_out_order_acc() >= floor - 1e-9            # L3 does not underperform the genus floor


def test_permuted_cooccurrence_collapses():
    """Random cross-genus+cross-order co-occurrence -> the hierarchy isn't discovered -> order inheritance collapses."""
    from research.runners._emerge45_three_level_hierarchy_derisk import ThreeLevelProbe
    assert ThreeLevelProbe(seed=42, epochs=40, permute=True).held_out_order_acc() <= 0.7


if __name__ == "__main__":
    from research.runners._emerge45_three_level_hierarchy_derisk import ThreeLevelProbe
    p = ThreeLevelProbe(seed=42, epochs=40)
    assert p.held_out_order_acc() >= 0.8 and p.sibling_confusion_rate() <= 0.05
    assert p.held_out_order_acc_l2only() >= 0.55
    print("OK: emerge45 three-level hierarchy -- order inheritance + sibling-confusion ~0 + L2/genus floor")
