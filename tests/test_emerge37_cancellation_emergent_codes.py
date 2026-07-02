"""CI guard for EMERGE-37 cancellation on emergent codes: the full Collins-Quillian inference (inheritance + specific-
override cancellation) works on codes learned from experience. A member-specific fact cancels the inherited class
default; non-overridden members inherit via the learned grouping; the dAP-lesion collapses it. CPU (numpy); skips
gracefully if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge37_cancellation_emergent_codes_derisk import CancellationProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge37 deps unavailable: {e}")
    return CancellationProbe(seed=42, epochs=80)


def test_cancellation_overrides_inherited(probe):
    """The overridden member answers its SPECIFIC fact, not the inherited class default (robin -> walks, not flies)."""
    from research.runners._emerge37_cancellation_emergent_codes_derisk import OVERRIDE
    assert probe.infer(OVERRIDE[0]) == OVERRIDE[1]


def test_non_overridden_inherit(probe):
    """Non-overridden members inherit the class property via the LEARNED (from co-occurrence) grouping."""
    from research.runners._emerge37_cancellation_emergent_codes_derisk import INHERIT_MEMBERS, CATPROP, MEMBERS
    for m in INHERIT_MEMBERS:
        assert probe.infer(m) == CATPROP[MEMBERS[m]]


def test_permuted_context_collapses_inheritance():
    """Scrambled co-occurrence -> no emergent grouping -> inheritance collapses (isolates the learned grouping)."""
    from research.runners._emerge37_cancellation_emergent_codes_derisk import CancellationProbe, INHERIT_MEMBERS, CATPROP, MEMBERS
    p = CancellationProbe(seed=42, epochs=80, permute=True)
    acc = sum(p.infer(m) == CATPROP[MEMBERS[m]] for m in INHERIT_MEMBERS) / len(INHERIT_MEMBERS)
    assert acc <= 0.6


if __name__ == "__main__":
    from research.runners._emerge37_cancellation_emergent_codes_derisk import CancellationProbe, OVERRIDE, INHERIT_MEMBERS, CATPROP, MEMBERS
    pr = CancellationProbe(seed=42, epochs=80)
    assert pr.infer(OVERRIDE[0]) == OVERRIDE[1]
    assert all(pr.infer(m) == CATPROP[MEMBERS[m]] for m in INHERIT_MEMBERS)
    print("OK: emerge37 cancellation on emergent codes -- override + inheritance-via-learned-grouping + permuted-collapse")
