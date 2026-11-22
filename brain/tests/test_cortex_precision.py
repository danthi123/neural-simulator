# Capability: Lifelong learning
# Record: owner question 2026-11-22 (full precision on GPU); K-P DEV read device confound (results/2026-11-22-kp-dev-read-1.5b.md)
# Expected-fail: none
"""Cortex(cuda_precision=...): the default keeps the old strategies; fp32 on cuda is opt-in; bad values refuse."""
import pytest
from brain import cortex as C


class _Fake:
    def __init__(self, model, strategy):
        self.strategy = strategy
        self.args = type("A", (), {"n_layer": 2})()


@pytest.fixture
def fake(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "RWKV", lambda model, strategy: seen.append(strategy) or _Fake(model, strategy))
    monkeypatch.setattr(C, "PIPELINE", lambda m, v: None)
    monkeypatch.setattr(C, "resolve_model", lambda p: "/x/m.pth")
    return seen


def test_default_strategies_unchanged(fake):
    C.Cortex("m", device="cpu"); C.Cortex("m", device="cuda")
    assert fake == ["cpu fp32", "cuda fp16"]


def test_cuda_fp32_opt_in(fake):
    c = C.Cortex("m", device="cuda", cuda_precision="fp32")
    assert fake == ["cuda fp32"] and c.strategy == "cuda fp32"
    C.Cortex("m", device="cpu", cuda_precision="fp32")
    assert fake[-1] == "cpu fp32"


def test_bad_precision_refused(fake):
    with pytest.raises(ValueError):
        C.Cortex("m", device="cuda", cuda_precision="bf16")
