"""KeyedHippocampus with key="S0" selects exactly what the default Hippocampus selects; other keys run end to end."""
import os

import pytest

from brain.cortex import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL), reason="cortex weights not on this machine")

FACTS = ["My cat is called Pixel.", "I was born in Lyon.", "My dog is named Biscuit.", "I drive a red Volvo."]
CUES = ["What is my cat called?", "Where was I born?", "What is my dog's name?", "What car do I drive?"]


@pytest.fixture(scope="module")
def setup():
    from brain.brain import BIRTH
    from brain.cortex import Cortex
    from brain.plastic_cortex import PlasticCortex
    c = Cortex(device="cpu")
    pc = PlasticCortex(c, rank=1, layers=[])
    pc.enabled = False
    _, base = c.read(BIRTH, None)
    return c, pc, [s.float() for s in base]


def test_s0_equals_default(setup):
    from brain.hippocampus import Hippocampus
    from brain.hippocampus_keys import KeyedHippocampus
    c, pc, base = setup
    h0, h1 = Hippocampus(), KeyedHippocampus(pc, key="S0")
    for f in FACTS:
        h0.store(c, base, f)
        h1.store(c, base, f)
    for q in CUES:
        e0, s0 = h0.recall(c, base, q)
        e1, s1 = h1.recall(c, base, q)
        assert (e0 is None) == (e1 is None) and (e0 is None or e0.text == e1.text)
        assert abs(s0 - s1) < 1e-3
    assert not hasattr(c, "key_of") or c.key_of is type(c).key_of  # the per-call swap is undone


@pytest.mark.parametrize("key", ["S_all", "pool_all", "last5", "learned"])
def test_other_keys_run(setup, key):
    from brain.hippocampus_keys import KeyedHippocampus, LearnedKey
    c, pc, base = setup
    lk = LearnedKey(pc.L * pc.C) if key == "learned" else None
    h = KeyedHippocampus(pc, key=key, learned=lk)
    for f in FACTS:
        h.store(c, base, f)
    h.rekey()
    ep, sim = h.recall(c, base, CUES[0])
    assert isinstance(sim, float)
