"""CI guard for EMERGE-35 fully-spiking sparse-expansion pooler: a spiking Marr-Albus codon column layer (features ->
a large decorrelated column layer, coincidence-driven, NO numpy kWTA) forms category-separating codes scaling to 4
categories + supports on-bridge inheritance; the dAP-lesion collapses it. CPU (numpy); seed 42 only (250-column pooler
is heavy). The multi-seed runner validates the input-destruction permuted-features control."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge35_spiking_pooler_derisk import SpikingPoolerProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge35 deps unavailable: {e}")
    return SpikingPoolerProbe(seed=42, epochs=40)


def test_four_category_held_out_inheritance(probe):
    """Held-out members across 4 latent categories inherit their category's property (chance 0.25) via the spiking codon."""
    assert probe.held_out_acc() >= 0.85


def test_dap_lesion_collapses():
    """dAP-LESION (coincidence off -> columns don't fire) deterministically collapses the inheritance."""
    from research.runners._emerge35_spiking_pooler_derisk import SpikingPoolerProbe
    p = SpikingPoolerProbe(seed=42, epochs=40, lesion=True)
    assert p.held_out_acc() <= 0.35                              # near/below chance 0.25 (no codons -> abstain/guess)


if __name__ == "__main__":
    from research.runners._emerge35_spiking_pooler_derisk import SpikingPoolerProbe
    pr = SpikingPoolerProbe(seed=42, epochs=40)
    assert pr.held_out_acc() >= 0.85
    assert SpikingPoolerProbe(seed=42, epochs=40, lesion=True).held_out_acc() <= 0.35
    print("OK: emerge35 fully-spiking sparse-expansion pooler -- 4-category held-out inheritance + dAP-lesion collapse")
