"""CI guard for EMERGE-39: the FULLY-ON-SUBSTRATE competitive pooler. HTM-Spatial-Pooler feature->column permanences live
in the bridge's coincidence synapse weights and are learned by the committed sim/ kernel (potentiation) PLUS the added
winner-INACTIVE depression (selectivity); overlapping categories separate; the selectivity term is load-bearing; permuted
+ dAP-lesion collapse it. CPU (numpy); skips gracefully if deps unavailable.

These assertions cover exactly the VALID (GO-gated) controls: the on-substrate separation, the mechanism-ablation
(with-vs-without selectivity) margin, and the input-destruction (permuted) collapse. The FIXED (no-learn random-projection)
arm is intentionally NOT asserted -- it is a fixed-random-code control, unreliable in this small representation space
(per-seed spread ~0.28-0.83), reported-only, per 2026-07-02-anti-cheat-control-validity-methodology.md."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge39_onsubstrate_competitive_pooler_derisk import OnSubstrateCompetitivePoolerProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge39 deps unavailable: {e}")
    return OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40)


def test_onsubstrate_pooler_separates_overlapping_categories(probe):
    """The on-substrate learned pooler separates 6 OVERLAPPING categories: held-out inheritance is high."""
    assert probe.held_out_acc() >= 0.8


def test_winner_inactive_depression_is_load_bearing():
    """The added winner-inactive (selectivity) depression is load-bearing: without it, discrimination collapses."""
    from research.runners._emerge39_onsubstrate_competitive_pooler_derisk import OnSubstrateCompetitivePoolerProbe
    withsel = OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40, selectivity=True).held_out_acc()
    without = OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40, selectivity=False).held_out_acc()
    assert withsel >= without + 0.25


def test_permuted_features_collapse():
    """Scrambled feature-category structure -> the pooler can't tune -> inheritance collapses."""
    from research.runners._emerge39_onsubstrate_competitive_pooler_derisk import OnSubstrateCompetitivePoolerProbe
    assert OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40, permute=True).held_out_acc() <= 0.55


if __name__ == "__main__":
    from research.runners._emerge39_onsubstrate_competitive_pooler_derisk import OnSubstrateCompetitivePoolerProbe
    p = OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40)
    assert p.held_out_acc() >= 0.8
    assert OnSubstrateCompetitivePoolerProbe(seed=42, epochs=40, selectivity=False).held_out_acc() < p.held_out_acc()
    print("OK: emerge39 on-substrate competitive pooler -- learned separates overlapping cats, selectivity load-bearing")
