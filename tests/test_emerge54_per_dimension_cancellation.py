"""CI guard for EMERGE-54: PER-DIMENSION (Collins-Quillian) cancellation over the pooler-DISCOVERED conversational codes.
Verifies the fix for the EMERGE-52 wrinkle -- a member's exception overrides ONLY its own property DIMENSION and no longer
blocks inheriting UNRELATED class properties. The load-bearing checks: the overridden member (penguin) answers its exception
on the OVERRIDDEN dimension ('penguin flies' == No, walks) AND inherits the class default on a DIFFERENT dimension
('penguin breathes' == Yes) -- the OLD code failed the second. Also: non-overridden members inherit on all dimensions; the
no-confab moat abstains on an unknown token; the dAP-lesion collapses inheritance. CPU/numpy, offline; skips gracefully if
the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def checked():
    try:
        from research.runners._emerge54_per_dimension_cancellation_derisk import (
            _check, handle, _BIRD_EXC, _FISH_EXC, _BIRD_HELDOUT, _FISH_HELDOUT)
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge54 deps unavailable: {e}")
    c, ch = _check(seed=42)
    return {"c": c, "ch": ch, "handle": handle,
            "bird_exc": _BIRD_EXC[0], "fish_exc": _FISH_EXC[0],
            "bird_held": _BIRD_HELDOUT, "fish_held": _FISH_HELDOUT}


def test_per_dimension_cancellation_penguin_breathes_yes_flies_no(checked):
    """THE FIX (both must hold): the penguin answers its LOCOMOTION exception ('penguin flies' == No, walks) AND still
    INHERITS the RESPIRATION class default ('penguin breathes' == Yes). The old code answered 'No, a penguin walks' to
    BOTH -- leaking the locomotion exception across into respiration."""
    c, handle = checked["c"], checked["handle"]
    pen = checked["bird_exc"]
    fly = handle(c, "can a %s fly?" % pen)
    breathe = handle(c, "can a %s breathe?" % pen)
    assert fly.startswith("No,"), f"penguin flies should be No (locomotion overridden), got: {fly!r}"
    assert breathe.startswith("Yes,"), f"penguin breathes should be Yes (respiration INHERITED -- the fix), got: {breathe!r}"
    # the OTHER exception member behaves identically (LOCOMOTION overridden, RESPIRATION inherited)
    pike = checked["fish_exc"]
    assert handle(c, "can a %s swim?" % pike).startswith("No,"), "pike swims should be No (locomotion overridden)"
    assert handle(c, "can a %s breathe?" % pike).startswith("Yes,"), "pike breathes should be Yes (respiration INHERITED)"


def test_nonoverridden_members_inherit_all_dimensions_and_moat(checked):
    """Non-overridden members inherit on ALL dimensions (locomotion + respiration); sibling-branch is not inherited; the
    no-confab moat abstains on an unknown token."""
    c, handle = checked["c"], checked["handle"]
    owl, minnow = checked["bird_held"], checked["fish_held"]
    assert handle(c, "can a %s fly?" % owl).startswith("Yes,"), "owl inherits locomotion (fly)"
    assert handle(c, "can a %s breathe?" % owl).startswith("Yes,"), "owl inherits respiration (breathe)"
    assert handle(c, "can a %s swim?" % minnow).startswith("Yes,"), "minnow inherits locomotion (swim)"
    assert handle(c, "can a %s breathe?" % minnow).startswith("Yes,"), "minnow inherits respiration (breathe)"
    # sibling-discrimination: a held-out bird does NOT inherit the fish branch's 'swim'
    assert handle(c, "can a %s swim?" % owl).startswith("I don't know"), "owl (a bird) does not inherit fish 'swim'"
    # no-confab moat: an unknown/never-observed token abstains
    assert handle(c, "can a zzz breathe?").startswith("I don't know what"), "moat abstains on an unknown token"
    # aggregate gate values from the scripted check
    ch = checked["ch"]
    assert ch["per_dim_cancellation"] >= 0.99, ch
    assert ch["nonoverride_inherit"] >= 0.99, ch
    assert ch["sibling_confusion"] <= 0.01, ch
    assert ch["moat_unknown"] is True, ch


def test_dap_lesion_collapses_inheritance():
    """The dAP-LESION (no coincidence/two-compartment substrate) collapses the per-dimension inheritance read to abstain --
    the graded-apical substrate the fix reads through is load-bearing."""
    from research.runners._emerge54_per_dimension_cancellation_derisk import _check
    _, ch_lesion = _check(seed=42, lesion=True)
    assert ch_lesion["nonoverride_inherit"] <= 0.01, ch_lesion
    assert ch_lesion["per_dim_cancellation"] <= 0.01, ch_lesion


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
