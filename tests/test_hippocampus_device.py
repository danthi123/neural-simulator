"""Stored memories remember the device they were stored on; recall/reinstate follow the live cortex's device.

The bug pinned here: `w = torch.tensor([...])` is CPU-born, so `score = sims * w` died with
"Expected all tensors to be on the same device" whenever the cortex ran on cuda. The same mix lurked in
`keys = torch.stack(...)` (episodes saved on one device, loaded against a cortex on another) and in
`reinstate` (add_delta moves dtype but not device). No model weights: a mock cortex on a chosen device.
"""
import sys
import types
import zlib

import pytest
import torch

try:  # brain.cortex imports the rwkv package at module level; the tests never load a model, so a stub is enough
    import rwkv.model  # noqa: F401
except ImportError:
    for _n in ("rwkv", "rwkv.model", "rwkv.utils"):
        sys.modules[_n] = types.ModuleType(_n)
    sys.modules["rwkv.model"].RWKV = object
    sys.modules["rwkv.utils"].PIPELINE = object

from brain.cortex import Cortex  # noqa: E402
from brain.hippocampus import Hippocampus  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda to mix with cpu-stored tensors")


class MockCortex:
    """Deterministic cortex on one device. The three slot roles follow Cortex.matrix_delta/key_of/add_delta."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def read(self, text, state=None):
        st = [torch.zeros(4, device=self.device) for _ in range(3)]
        st[1] = st[1] + float(zlib.crc32(text.encode()) % 997) / 997.0
        return torch.zeros(4, device=self.device), st

    matrix_delta = Cortex.matrix_delta

    @staticmethod
    def key_of(delta, layer=0):
        return Cortex.key_of(delta, layer)

    @staticmethod
    def add_delta(state, delta, alpha):
        return Cortex.add_delta(state, delta, alpha)


def _base(cortex):
    return cortex.read("base", None)[1]


def _stored(h, cortex):
    """Two episodes stored through `cortex` (their tensors live on that cortex's device)."""
    h.store(cortex, _base(cortex), "My cat is called Pixel.", speaker="User", salience=2.0)
    h.store(cortex, _base(cortex), "My sister's name is Marguerite.", speaker="User", salience=2.0)
    return h


@CUDA
def test_recall_completes_when_the_stored_keys_live_on_another_device():
    cpu, cuda = MockCortex("cpu"), MockCortex("cuda")
    h = _stored(Hippocampus(), cpu)               # keys born on cpu (as after torch.load of a cpu-saved brain)
    assert all(e.key.device.type == "cpu" for e in h.episodes)
    ep, sim = h.recall(cuda, _base(cuda), "My cat is called Pixel.", question=True)   # died before the fix
    assert ep is not None and "Pixel" in ep.text and isinstance(sim, float)
    assert all(e.key.device.type == "cpu" for e in h.episodes)                        # store-side untouched


@CUDA
def test_reinstate_brings_a_stored_delta_to_the_state_device():
    cpu, cuda = MockCortex("cpu"), MockCortex("cuda")
    h = _stored(Hippocampus(), cpu)
    state = cuda.read("live", None)[1]
    before = [x.clone() for x in state]
    out = h.reinstate(cuda, state, h.episodes[0])
    assert all(x.device.type == "cuda" for x in out)
    assert all(torch.equal(a, b) for a, b in zip(before, state))                     # add_delta must not mutate
    assert any(not torch.equal(a, b) for a, b in zip(out, before))                   # and must actually reinstate


@CUDA
def test_recall_weights_land_on_the_similarity_device():
    cpu, cuda = MockCortex("cpu"), MockCortex("cuda")
    h = _stored(Hippocampus(), cuda)              # everything on cuda: `w` must still be cuda-born, not cpu
    ep, _ = h.recall(cuda, _base(cuda), "My cat is called Pixel.", question=True)
    assert ep is not None


def test_recall_and_reinstate_still_work_on_one_device():
    cpu = MockCortex("cpu")
    h = _stored(Hippocampus(), cpu)
    ep, sim = h.recall(cpu, _base(cpu), "My cat is called Pixel.", question=True)
    assert ep is not None and "Pixel" in ep.text and sim > 0.5
    out = h.reinstate(cpu, cpu.read("live", None)[1], h.episodes[1])
    assert all(x.device.type == "cpu" for x in out)