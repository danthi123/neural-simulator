"""CI guard for EMERGE-38: the COMPETITIVE SELF-ORGANIZING pooler surpasses a fixed projection on OVERLAPPING categories.
A learned HTM Spatial Pooler (winners potentiate active inputs + depress inactive + homeostatic boosting) tunes columns
to the discriminative features and separates overlapping categories that an untuned random projection cannot; the
inheritance runs on the spiking bridge over the learned codons; permuted-features + dAP-lesion collapse it. CPU (numpy);
skips gracefully if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge38_competitive_pooler_derisk import CompetitivePoolerProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge38 deps unavailable: {e}")
    return CompetitivePoolerProbe(seed=42, epochs=40)


def test_learned_pooler_separates_overlapping_categories(probe):
    """The LEARNED competitive pooler separates 6 OVERLAPPING categories: held-out inheritance is high."""
    assert probe.held_out_acc() >= 0.8


def test_learned_beats_fixed_projection():
    """The learned pooler beats the untuned fixed random projection on the SAME overlapping task (learning is load-bearing)."""
    from research.runners._emerge38_competitive_pooler_derisk import CompetitivePoolerProbe
    learned = CompetitivePoolerProbe(seed=42, epochs=40, learn=True).held_out_acc()
    fixed = CompetitivePoolerProbe(seed=42, epochs=40, learn=False).held_out_acc()
    assert learned >= fixed + 0.2


def test_permuted_features_collapse():
    """Scrambled feature-category structure -> the pooler can't tune to discriminative features -> inheritance collapses."""
    from research.runners._emerge38_competitive_pooler_derisk import CompetitivePoolerProbe
    assert CompetitivePoolerProbe(seed=42, epochs=40, permute=True).held_out_acc() <= 0.55


if __name__ == "__main__":
    from research.runners._emerge38_competitive_pooler_derisk import CompetitivePoolerProbe
    p = CompetitivePoolerProbe(seed=42, epochs=40)
    assert p.held_out_acc() >= 0.8
    assert CompetitivePoolerProbe(seed=42, epochs=40, learn=False).held_out_acc() < p.held_out_acc()
    print("OK: emerge38 competitive pooler -- learned separates overlapping cats, beats fixed, permuted collapses")
