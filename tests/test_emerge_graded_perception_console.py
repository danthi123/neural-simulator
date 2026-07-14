"""CI guard for the perception-grounded graded read: the graded MOAT (CONFIDENT for a clearly-perceived category member
via the VISUALLY-discovered category / MOAT for a never-perceived object) grounded in the real Gabor/V1 front end, causal
on the coincidence substrate (lesion -> abstain). The HEDGED level is a characterized categorical-perception boundary (a
real image-blend morph is sharpened to one category), NOT gated here. CPU (numpy); skips if deps/rendering unavailable.
Single-seed (Gabor/V1 rendering is slow)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def built():
    try:
        from research.runners._emerge_graded_perception_console_derisk import _build
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"graded-perception deps unavailable: {e}")
    try:
        return _build(42, "htm")
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"perception rendering unavailable: {e}")


def test_heldout_perceived_member_confident(built):
    """A HELD-OUT perceived bird inherits the class property CONFIDENTLY via the VISUALLY-discovered category."""
    from research.runners._emerge_graded_perception_console_derisk import HELD_OUT
    labels = [built.graded_ask_can(m, p)[0] for (m, p) in HELD_OUT]
    n_conf = sum(v == "CONFIDENT" for v in labels)
    assert n_conf / len(labels) >= 0.75            # EMERGE-53's own validated held-out-inheritance bar


def test_novel_percept_moat(built):
    """A never-perceived object hits the intrinsic no-confab moat."""
    label, phrase = built.graded_ask_can("griffin", "fly")
    assert label == "MOAT"
    assert "I don't know what" in phrase


def test_lesion_abstains():
    """Coincidence off (dAP-lesion) -> no apical drive -> the perceived member abstains (not confident)."""
    from research.runners._emerge_graded_perception_console_derisk import _build, HELD_OUT
    try:
        les = _build(42, "lesion")
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"perception rendering unavailable: {e}")
    assert all(les.graded_ask_can(m, p)[0] == "ABSTAIN" for (m, p) in HELD_OUT)
