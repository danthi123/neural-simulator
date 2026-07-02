"""CI guard for EMERGE-34 perception-grounded emergence: the brain forms categories from REAL sensory experience --
objects seen through the Gabor/V1 front end, categories discovered by a pooler, held-out PERCEIVED objects inherit a
property on the spiking bridge; the dAP-lesion (mechanism removal) collapses it. Seed 42 only (V1 encode + pooler are
heavy). The multi-seed runner validates the LOAD-BEARING per-image pixel-scramble input-destruction control (which is
noisy at a single seed -- mean ~0.53 collapse vs held ~0.97)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge34_perception_grounded_emergence_derisk import PerceptionEmergeProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge34 deps unavailable: {e}")
    return PerceptionEmergeProbe(seed=42, epochs=80)


def test_held_out_perceived_objects_inherit(probe):
    """Held-out PERCEIVED objects inherit their visual category's property (learned from experience, not told)."""
    assert probe.held_out_acc() >= 0.83                          # >=5 of 6 held-out objects (HOLD=3 per category)


def test_moat_disjoint_code_abstains(probe):
    assert probe.moat() == 1.0


def test_dap_lesion_collapses():
    """dAP-LESION (bridge coincidence off -> no priming) deterministically collapses the inheritance -- a clean
    mechanism-ablation control. (The per-image pixel-scramble input-destruction control is the load-bearing perception
    control but is noisy at a single seed; the multi-seed runner validates it: mean ~0.53 vs held ~0.97.)"""
    from research.runners._emerge34_perception_grounded_emergence_derisk import PerceptionEmergeProbe
    p = PerceptionEmergeProbe(seed=42, epochs=80, lesion=True)
    assert p.held_out_acc() == 0.0


if __name__ == "__main__":
    from research.runners._emerge34_perception_grounded_emergence_derisk import PerceptionEmergeProbe
    pr = PerceptionEmergeProbe(seed=42, epochs=80)
    assert pr.held_out_acc() >= 0.83 and pr.moat() == 1.0
    assert PerceptionEmergeProbe(seed=42, epochs=80, lesion=True).held_out_acc() == 0.0
    print("OK: emerge34 perception-grounded emergence -- see -> discover categories -> infer + dAP-lesion collapse + moat")
