"""CI guard for the EMERGE-25 grounded growing console: talk to + teach the emergent spiking brain. Grounded grammatical
production + family generalization + online growth (teach live) + retention + the no-confab moat. CPU (numpy); skips
gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def console():
    try:
        from research.runners._emerge25_grounded_growing_console import LanguageConsole
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge25 deps unavailable: {e}")
    return LanguageConsole(seed=42, epochs=80)


def test_grounded_production(console):
    assert console.respond("dog") == "dog chased ball."
    assert console.respond("cat") == "cat ate fish."
    assert console.respond("owl") == "owl saw moon."


def test_family_generalization(console):
    assert console.respond("wolf") == "wolf chased ball."         # canine, like dog
    assert console.respond("lion") == "lion ate fish."           # feline, like cat
    assert console.respond("hawk") == "hawk saw moon."           # avian, like owl


def test_moat_abstains(console):
    assert console.respond("zzz") == "I don't know."             # unknown -> abstain


def test_online_growth_then_retention():
    """A fresh console: teach a new fact live -> it grows; old facts + moat survive."""
    from research.runners._emerge25_grounded_growing_console import LanguageConsole
    c = LanguageConsole(seed=42, epochs=80)
    assert c.respond("bear") == "I don't know."                  # unknown before teaching
    c.teach(["bear", "grabbed", "honey"])
    assert c.respond("bear") == "bear grabbed honey."           # learned live
    assert c.respond("dog") == "dog chased ball."               # retained (no forgetting)
    assert c.respond("zzz") == "I don't know."                  # moat still holds


if __name__ == "__main__":
    from research.runners._emerge25_grounded_growing_console import LanguageConsole
    c = LanguageConsole(seed=42, epochs=80)
    test_grounded_production(c); test_family_generalization(c); test_moat_abstains(c)
    test_online_growth_then_retention()
    print("OK: emerge25 grounded growing console -- grounded + generalize + teach-live + retention + moat")
