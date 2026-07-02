"""CI guard for the EMERGE-21 end-to-end language console: the emergent sequence cortex GENERATES grounded word
sequences, GENERALIZES to a similar untrained cue, and ABSTAINS for a novel/ungrounded cue (the intrinsic no-confab
moat). CPU (numpy backend); no GPU. Skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def cortex():
    try:
        from research.runners._emerge21_language_console import LanguageCortex
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge21 deps unavailable: {e}")
    return LanguageCortex(seed=42, epochs=80)


def test_grounded_production(cortex):
    """A trained (grounded) cue produces the learned fact end-to-end."""
    assert cortex.respond("dog") == ["dog", "chased", "ball", "home"]
    assert cortex.respond("cat") == ["cat", "chased", "ball", "away"]


def test_generalization(cortex):
    """An untrained but SIMILAR cue generalizes the family's continuation (never trained on wolf/fox/lion)."""
    assert cortex.respond("wolf")[-1] == "home"                  # canine -> home (generalized from dog)
    assert cortex.respond("fox")[-1] == "home"
    assert cortex.respond("lion")[-1] == "away"                  # feline -> away (generalized from cat)


def test_intrinsic_moat_abstains(cortex):
    """A truly-novel/ungrounded cue (disjoint code) drives no coincidence -> ABSTAINS, never confabulates a fact."""
    for novel in ("zzz", "qqq"):
        out = cortex.respond(novel)
        assert out[-1] == "<I don't know>"                       # abstains
        assert "home" not in out and "away" not in out          # no confabulated fact


if __name__ == "__main__":
    c = cortex.__wrapped__() if hasattr(cortex, "__wrapped__") else None
    from research.runners._emerge21_language_console import LanguageCortex
    c = LanguageCortex(seed=42, epochs=80)
    test_grounded_production(c); test_generalization(c); test_intrinsic_moat_abstains(c)
    print("OK: emerge21 language console -- grounded production + generalization + intrinsic moat")
