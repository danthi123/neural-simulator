"""The plastic cortex is the pretrained cortex until sleep changes it, and sleep can change it."""
import os

import pytest
import torch

from brain.cortex import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL), reason="cortex weights not on this machine")


@pytest.fixture(scope="module")
def pc():
    from brain.cortex import Cortex
    from brain.plastic_cortex import PlasticCortex
    return PlasticCortex(Cortex(device="cpu"), rank=4)


def test_zero_adapters_equal_the_package_forward(pc):
    ids = pc.cortex.pipe.encode("User: My sister's name is Priya.\n\nAssistant:")
    ref, _ = pc.cortex.model.forward(ids, None)
    ours, _ = pc(ids)
    assert float((ref.float() - ours[-1].detach()).abs().max()) < 1e-3


def test_gradients_reach_only_the_adapters(pc):
    ids = pc.cortex.pipe.encode("User: Hello there.\n\nAssistant: Hi!")
    logits, _ = pc(ids)
    torch.nn.functional.cross_entropy(logits[:-1], torch.tensor(ids[1:])).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pc.parameters())
    assert all(not v.requires_grad for v in pc.z.values())
