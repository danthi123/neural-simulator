"""CI guard for EMERGE-55: EMERGENT DIMENSIONS -- the per-dimension (Collins-Quillian) cancellation grouping is LEARNED from
the statistics of experience (mutually-exclusive alternates = one dimension), replacing EMERGE-54's host `PROP_DIM` lexicon.
Load-bearing checks: (1) the learned grouping matches the true dimensions (fly/walk/swim/lurk together as locomotion
alternates, breathe separate); (2) per-dimension cancellation over the LEARNED dimensions still holds -- the overridden
member answers its exception on its dimension ('penguin flies' == No, walks) AND inherits the class default on a DIFFERENT
dimension ('penguin breathes' == Yes); (3) the no-confab moat abstains on an unknown token; (4) the DESTROYED-EXCLUSIVITY
control (destroy the mutual-exclusivity statistics -> every property a singleton dimension) BREAKS the per-dimension
cancellation, proving the LEARNED grouping is load-bearing (no host fallback). CPU/numpy, offline; skips gracefully if the
substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def deps():
    try:
        from research.runners._emerge55_emergent_dimensions_derisk import (
            _check, handle, learn_dimensions, TRUE_DIM, _dimension_discovery_score)
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge55 deps unavailable: {e}")
    return {"_check": _check, "handle": handle, "learn_dimensions": learn_dimensions,
            "TRUE_DIM": TRUE_DIM, "_dimension_discovery_score": _dimension_discovery_score}


def test_learn_dimensions_pure_statistics(deps):
    """The dimension learner (pure co-occurrence statistics, no bridge) groups mutually-exclusive alternates into ONE
    dimension and isolates the freely-co-occurring property. Members each have exactly one locomotion (fly/walk/swim/lurk)
    AND breathe -> the learner must put the four locomotions together and breathe separate."""
    learn_dimensions = deps["learn_dimensions"]
    # member -> {properties TRUE for it}: every member breathes + has exactly one locomotion (an exception substitutes)
    stats = {
        "robin": {"fly", "breathe"}, "sparrow": {"fly", "breathe"}, "eagle": {"fly", "breathe"},
        "penguin": {"walk", "breathe"},                              # locomotion exception (walk replaces fly)
        "trout": {"swim", "breathe"}, "salmon": {"swim", "breathe"},
        "pike": {"lurk", "breathe"},                                 # locomotion exception (lurk replaces swim)
    }
    dim_of, dims = learn_dimensions(stats)
    # fly/walk/swim/lurk share ONE dimension; breathe is in a DIFFERENT dimension
    loco = {dim_of["fly"], dim_of["walk"], dim_of["swim"], dim_of["lurk"]}
    assert len(loco) == 1, f"the four locomotions should share one learned dimension, got {loco}"
    assert dim_of["breathe"] not in loco, "breathe should be in a different learned dimension from the locomotions"
    assert deps["_dimension_discovery_score"](dim_of) == 1.0, "discovery should exactly match the true 2-way partition"


def test_per_dimension_cancellation_over_learned_dims_and_moat(deps):
    """End-to-end on the spiking bridge: with the LEARNED dimensions driving the read, the overridden member answers its
    exception on its learned dimension AND inherits on a different learned dimension; non-overridden members inherit on all
    learned dimensions; sibling-branch is not inherited; the no-confab moat abstains."""
    c, ch = deps["_check"](seed=42)
    handle = deps["handle"]
    # dimension-discovery matched the truth
    assert ch["discovery"] >= 0.90, ch["learned_dims"]
    # the FIX over LEARNED dimensions: penguin flies == No (locomotion overridden), breathes == Yes (respiration inherited)
    assert handle(c, "can a penguin fly?").startswith("No,"), "penguin flies should be No (learned-locomotion overridden)"
    assert handle(c, "can a penguin breathe?").startswith("Yes,"), "penguin breathes should be Yes (respiration inherited)"
    assert handle(c, "can a pike swim?").startswith("No,"), "pike swims should be No (learned-locomotion overridden)"
    assert handle(c, "can a pike breathe?").startswith("Yes,"), "pike breathes should be Yes (respiration inherited)"
    # non-overridden members inherit on all learned dimensions; sibling-branch not inherited; moat abstains
    assert handle(c, "can a owl fly?").startswith("Yes,"), "owl inherits locomotion (fly)"
    assert handle(c, "can a owl breathe?").startswith("Yes,"), "owl inherits respiration (breathe)"
    assert handle(c, "can a owl swim?").startswith("I don't know"), "owl (a bird) does not inherit fish 'swim'"
    assert handle(c, "can a zzz breathe?").startswith("I don't know what"), "moat abstains on an unknown token"
    assert ch["per_dim_cancellation"] >= 0.99 and ch["nonoverride_inherit"] >= 0.99, ch


def test_destroyed_exclusivity_control_breaks_the_read(deps):
    """LOAD-BEARING: destroying the mutual-exclusivity statistics (every member gets the full property vocab -> every
    property a singleton dimension) makes the LEARNED grouping wrong -> the exception's dimension no longer matches the
    asked property's -> per-dimension cancellation BREAKS (penguin wrongly answers Yes to flies). This proves the LEARNED
    grouping is doing the work, not a host fallback."""
    _check, handle = deps["_check"], deps["handle"]
    c_real, ch_real = _check(seed=42)
    c_ctrl, ch_ctrl = _check(seed=42, shuffle_labels=True)
    # real: penguin flies == No (cancellation works)
    assert ch_real["per_dim_cancellation"] >= 0.99
    # control: the learned dimensions are wrong (discovery collapses) and per-dim cancellation breaks
    assert ch_ctrl["discovery"] < 0.90, f"control should have wrong learned dimensions, got {ch_ctrl['learned_dims']}"
    assert ch_ctrl["per_dim_cancellation"] <= 0.5, "destroyed-exclusivity should break per-dimension cancellation"
    assert handle(c_ctrl, "can a penguin fly?").startswith("Yes,"), (
        "under destroyed exclusivity the exception fails to cancel -> penguin wrongly answers Yes to flies")
