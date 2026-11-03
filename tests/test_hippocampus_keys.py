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


def test_s_all_from_delta_matches_explicit_key(setup):
    """S_all read from the stored deltas = centred cosine of the explicit concatenated key (each layer normalised)."""
    import torch
    import torch.nn.functional as F
    from brain.hippocampus_keys import KeyedHippocampus, centred_cos
    c, pc, base = setup
    h = KeyedHippocampus(key="S_all")
    for f in FACTS:
        h.store(c, base, f)
    assert all(e.key is None for e in h.episodes)

    def full_key(delta):
        return torch.cat([F.normalize(delta[3 * l + 1].flatten().float(), dim=0) for l in range(c.n_layers)])

    K = torch.stack([full_key(e.delta) for e in h.episodes])
    for q in CUES:
        _, st = c.read(f"User: {q}\n\n", base)
        want = centred_cos([K], [full_key(c.matrix_delta(st, base))[None]])[0]
        got = h.similarities(c, base, q)
        assert torch.allclose(got, want, atol=1e-4)


def test_s_all_episode_costs_only_its_delta(setup):
    from brain.hippocampus_keys import KeyedHippocampus, episode_bytes
    c, pc, base = setup
    h = KeyedHippocampus(key="S_all")
    ep = h.store(c, base, FACTS[0])
    assert episode_bytes(ep) == sum(d.numel() * 2 for d in ep.delta if d is not None)


def test_refit_bar_is_per_instance(setup):
    from brain.hippocampus import Hippocampus
    from brain.hippocampus_keys import KeyedHippocampus
    h = KeyedHippocampus(key="S_all", familiarity=0.3, margin=0.05)
    assert (h.FAMILIARITY, h.MARGIN) == (0.3, 0.05)
    assert (Hippocampus.FAMILIARITY, Hippocampus.MARGIN) == (0.12, 0.02)
