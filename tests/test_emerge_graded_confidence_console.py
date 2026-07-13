"""CI guard for the graded-confidence console wire-in: the EMERGE-31 experiential console's answer is now a graded
three-level response (CONFIDENT / HEDGED / ABSTAIN) plus the unknown-concept no-confab moat, driven by the strength of
the learned co-occurrence (not hand-coded). A strongly co-observed member answers confidently, a category-ambiguous
member hedges, a member whose context bears no property abstains, and a never-observed concept hits the moat. The
coincidence-lesion collapses the confident answer. CPU (numpy); skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def built():
    try:
        from research.runners._emerge_graded_confidence_console_derisk import _build
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"graded-console deps unavailable: {e}")
    return _build(42, 80, "htm")


def test_strong_member_confident(built):
    """A member strongly co-observed with the property-bearing context answers CONFIDENTLY."""
    assert built.graded_answer("robin", "fly") == ("CONFIDENT", "Yes, a robin can fly.")


def test_ambiguous_member_hedges(built):
    """A category-ambiguous member (co-observed with two competing contexts) HEDGES -- the graded moat."""
    label, phrase = built.graded_answer("bat", "fly")
    assert label == "HEDGED"
    assert "probably" in phrase


def test_no_property_context_abstains(built):
    """A member whose context bears no taught property ABSTAINS (not a false assertion)."""
    assert built.graded_answer("trout", "fly")[0] == "ABSTAIN"


def test_unknown_concept_moat(built):
    """A never-observed concept hits the intrinsic no-confab moat."""
    label, phrase = built.graded_answer("wolpertinger", "fly")
    assert label == "MOAT"
    assert "I don't know what" in phrase


def test_lesion_collapses_confident():
    """Coincidence off (dAP-lesion) -> no apical drive -> even the strong member abstains."""
    from research.runners._emerge_graded_confidence_console_derisk import _build
    les = _build(42, 80, "lesion")
    assert les.graded_answer("robin", "fly")[0] == "ABSTAIN"


def test_permuted_destroys_confident():
    """No category structure (every member co-occurs equally with all contexts) -> the strong member is no longer confident."""
    from research.runners._emerge_graded_confidence_console_derisk import _build
    perm = _build(42, 80, "permuted")
    assert perm.graded_answer("robin", "fly")[0] != "CONFIDENT"
