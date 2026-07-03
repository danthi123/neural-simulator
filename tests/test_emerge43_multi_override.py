"""CI guard for EMERGE-43: multi-override cancellation over discovered categories. Many member-specific exceptions coexist
with class inheritance over the pooler-discovered overlapping categories -- each overridden member answers its own
exception (no cross-bleed); HELD-OUT non-taught members inherit via the shared codon (genuine generalization, not
direct retrieval); permuted collapses. CPU (numpy); skips if deps unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def probe():
    try:
        from research.runners._emerge43_multi_override_derisk import MultiOverrideProbe
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge43 deps unavailable: {e}")
    return MultiOverrideProbe(seed=42, epochs=40)


def test_each_override_answers_its_own_exception(probe):
    """Every overridden member answers ITS OWN exception (no cross-bleed to another member's fact)."""
    assert probe.override_acc() >= 0.85


def test_non_overridden_inherit(probe):
    """HELD-OUT members (never taught the class property) inherit via the shared codon -- genuine generalization, and the
    many overrides don't disrupt it."""
    assert probe.inheritance_acc() >= 0.8


def test_permuted_features_collapse():
    """Scrambled features -> the pooler can't discover categories -> inheritance collapses."""
    from research.runners._emerge43_multi_override_derisk import MultiOverrideProbe
    assert MultiOverrideProbe(seed=42, epochs=40, permute=True).inheritance_acc() <= 0.55


if __name__ == "__main__":
    from research.runners._emerge43_multi_override_derisk import MultiOverrideProbe
    p = MultiOverrideProbe(seed=42, epochs=40)
    assert p.override_acc() >= 0.85 and p.inheritance_acc() >= 0.8
    print("OK: emerge43 multi-override -- N exceptions coexist with inheritance, no cross-bleed")
