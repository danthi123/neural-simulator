"""UseHippocampus: re-telling raises one episode's strength; distinct tellings stay distinct; base class unchanged."""
import torch

from brain.hippocampus import Hippocampus
from brain.hippocampus_use import UseHippocampus


class StubCortex:
    """Deterministic fake: the text decides the matrix-memory delta (two layers), so no weights are needed."""

    def read(self, text, state):
        g = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
        base = state if state is not None else [torch.zeros(4)] * 6
        st = [torch.zeros(4), torch.randn(2, 3, 3, generator=g), torch.zeros(4), torch.zeros(4),
              torch.randn(2, 3, 3, generator=g), torch.zeros(4)]
        return None, st

    def matrix_delta(self, state, base):
        return [state[i].clone() if i % 3 == 1 else None for i in range(len(base))]

    @staticmethod
    def key_of(delta, layer=0):
        return delta[layer * 3 + 1].flatten().float()


BASE = [torch.zeros(4)] * 6


def test_retelling_merges_and_adds_strength():
    h, c = UseHippocampus(), StubCortex()
    for _ in range(3):
        h.store(c, BASE, "My cat is called Pixel.")
    h.store(c, BASE, "I was born in Lyon.")
    assert len(h.episodes) == 2
    assert [e.strength for e in h.episodes] == [3.0, 1.0]
    assert h.episodes[0].uses == 2


def test_base_hippocampus_still_duplicates():
    h, c = Hippocampus(), StubCortex()
    for _ in range(3):
        h.store(c, BASE, "My cat is called Pixel.")
    assert len(h.episodes) == 3 and all(e.strength == 1.0 for e in h.episodes)
