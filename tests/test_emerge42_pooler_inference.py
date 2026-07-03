"""CI guard for EMERGE-42: the pooler-discovered categories REASON. The competitive self-organizing pooler discovers
overlapping categories from experience; the full Collins-Quillian inference (class inheritance + member-specific-override
cancellation) runs over the learned codons on the spiking bridge. Permuted-features collapses inheritance. CPU (numpy);
skips gracefully if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge42_pooler_inference_derisk import PoolerInferenceProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge42 deps unavailable: {e}")
    return PoolerInferenceProbe(seed=42, epochs=40)


def test_cancellation_overrides_inherited(probe):
    """The overridden member answers its SPECIFIC fact (OVR), not the class default -- via its member-identity ensemble."""
    from research.runners._emerge42_pooler_inference_derisk import OVERRIDE_MEMBER
    assert probe.query(OVERRIDE_MEMBER) == "OVR"


def test_inheritance_over_discovered_categories(probe):
    """GENUINE HOLD-OUT: the tested members are EXCLUDED from CLASS teaching (self.held), so this measures generalization
    via the pooler-discovered overlapping-category codons -- a never-directly-taught member inherits its category's class
    property via the shared codon, not by direct retrieval."""
    assert probe.inheritance_acc() >= 0.8


def test_permuted_features_collapse_inheritance():
    """Scrambled features -> the pooler can't discover categories -> held-out inheritance collapses (isolates the
    discovered structure). Permuted mean ~0.15 3-seed; the <=0.55 bound is a comfortable single-seed guard."""
    from research.runners._emerge42_pooler_inference_derisk import PoolerInferenceProbe
    assert PoolerInferenceProbe(seed=42, epochs=40, permute=True).inheritance_acc() <= 0.55


if __name__ == "__main__":
    from research.runners._emerge42_pooler_inference_derisk import PoolerInferenceProbe, OVERRIDE_MEMBER
    p = PoolerInferenceProbe(seed=42, epochs=40)
    assert p.query(OVERRIDE_MEMBER) == "OVR" and p.inheritance_acc() >= 0.8
    print("OK: emerge42 pooler-discovered categories -- cancellation + inheritance over learned codons")
