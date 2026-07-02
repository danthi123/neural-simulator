"""CI guard for EMERGE-24 online growth: the emergent producer learns a NEW grounded fact LIVE (mid-life, same spiking
bridge), produces it, retains the old facts (no catastrophic forgetting), and keeps the no-confab moat. CPU (numpy);
skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def grown():
    try:
        from research.runners._emerge24_online_growth_derisk import GrowthProducer, BASE, NEW
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge24 deps unavailable: {e}")
    p = GrowthProducer(seed=42)
    p.learn(BASE, 80)
    pre = p.generate("fox")                                       # BEFORE teaching
    p.learn([NEW], 80)                                            # teach live
    return p, pre


def test_new_subject_abstains_before_teaching(grown):
    _, pre = grown
    assert pre[0] == ["<abstain>"]                               # fox genuinely unknown before it is taught


def test_learned_new_fact(grown):
    p, _ = grown
    assert p.generate("fox") == (["fox", "saw", "seed"], ["NOUN", "VERB", "NOUN"])


def test_retention_no_forgetting(grown):
    p, _ = grown
    assert p.generate("dog") == (["dog", "chased", "ball"], ["NOUN", "VERB", "NOUN"])
    assert p.generate("cat") == (["cat", "ate", "fish"], ["NOUN", "VERB", "NOUN"])


def test_moat_holds_through_growth(grown):
    p, _ = grown
    assert p.generate("zzz")[0] == ["<abstain>"]


if __name__ == "__main__":
    from research.runners._emerge24_online_growth_derisk import GrowthProducer, BASE, NEW
    p = GrowthProducer(seed=42); p.learn(BASE, 80)
    pre = p.generate("fox"); p.learn([NEW], 80)
    g = (p, pre)
    test_new_subject_abstains_before_teaching(g); test_learned_new_fact(g)
    test_retention_no_forgetting(g); test_moat_holds_through_growth(g)
    print("OK: emerge24 online growth -- learns-live + retention + moat")
