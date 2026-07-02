"""CI guard for EMERGE-36 fully-spiking perception->pooler->inference pipeline: objects seen through the real Gabor/V1
front end -> a spiking sparse-expansion codon (EMERGE-35, no numpy kWTA) -> a held-out perceived object inherits its
visual category's property; the dAP-lesion collapses it. CPU (numpy); seed 42 only (V1 encode + 250-column pooler are
heavy). The multi-seed runner validates the load-bearing per-image pixel-scramble control."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge36_spiking_perception_pipeline_derisk import SpikingPerceptionProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge36 deps unavailable: {e}")
    return SpikingPerceptionProbe(seed=42, epochs=40)


def test_held_out_perceived_object_inherits(probe):
    """A held-out PERCEIVED object inherits its visual category's property via the fully-spiking pipeline."""
    assert probe.held_out_acc() >= 0.83


def test_dap_lesion_collapses():
    """dAP-LESION (coincidence off -> no codon) deterministically collapses the inheritance."""
    from research.runners._emerge36_spiking_perception_pipeline_derisk import SpikingPerceptionProbe
    p = SpikingPerceptionProbe(seed=42, epochs=40, lesion=True)
    assert p.held_out_acc() <= 0.5


if __name__ == "__main__":
    from research.runners._emerge36_spiking_perception_pipeline_derisk import SpikingPerceptionProbe
    pr = SpikingPerceptionProbe(seed=42, epochs=40)
    assert pr.held_out_acc() >= 0.83
    assert SpikingPerceptionProbe(seed=42, epochs=40, lesion=True).held_out_acc() <= 0.5
    print("OK: emerge36 fully-spiking perception pipeline -- SEE -> spiking codon -> inheritance + dAP-lesion collapse")
