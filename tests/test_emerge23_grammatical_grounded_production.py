"""CI guard for the EMERGE-23 capstone: the emergent sequence cortex GENERATES full grammatical, grounded sentences,
GENERALIZES them to a similar untrained cue (via the family block), and ABSTAINS for a novel/ungrounded cue (the
intrinsic no-confab moat). Grammar is read from the shared POS-class block; content from the distinguishing
content+family blocks. CPU (numpy backend); no GPU. Skips gracefully if the substrate deps are unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest


@pytest.fixture(scope="module")
def producer():
    try:
        from research.runners._emerge23_grammatical_grounded_production_derisk import Producer
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"emerge23 deps unavailable: {e}")
    return Producer(seed=42, epochs=80)


def test_grounded_grammatical(producer):
    """A grounded (trained) cue generates the correct grounded sentence, POS-grammatical (NOUN VERB NOUN)."""
    assert producer.generate("dog") == (["dog", "chased", "ball"], ["NOUN", "VERB", "NOUN"])
    assert producer.generate("cat") == (["cat", "ate", "fish"], ["NOUN", "VERB", "NOUN"])


def test_generalization_via_family(producer):
    """A similar untrained cue (wolf~dog canine, lion~cat feline) generalizes its family's grounded continuation."""
    assert producer.generate("wolf") == (["wolf", "chased", "ball"], ["NOUN", "VERB", "NOUN"])
    assert producer.generate("lion") == (["lion", "ate", "fish"], ["NOUN", "VERB", "NOUN"])


def test_intrinsic_moat_abstains(producer):
    """A truly-novel/ungrounded cue (fully-disjoint code, no family) drives no distinguishing coincidence -> ABSTAINS."""
    out, pos = producer.generate("zzz")
    assert out == ["<abstain>"] and pos == []


def test_dap_lesion_collapses():
    """dAP-LESION (coincidence off): no plateau -> nothing primed -> grounded production collapses to abstain."""
    from research.runners._emerge23_grammatical_grounded_production_derisk import Producer
    p = Producer(seed=42, epochs=80, lesion=True)
    assert p.generate("dog")[0] == ["<abstain>"]


if __name__ == "__main__":
    from research.runners._emerge23_grammatical_grounded_production_derisk import Producer
    pr = Producer(seed=42, epochs=80)
    test_grounded_grammatical(pr); test_generalization_via_family(pr); test_intrinsic_moat_abstains(pr)
    test_dap_lesion_collapses()
    print("OK: emerge23 grammatical grounded producer -- grounded + generalize + moat + dAP-lesion")
