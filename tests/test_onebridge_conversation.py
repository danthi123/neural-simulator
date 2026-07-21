"""CI guard: a REACHABLE grounded conversation on ONE bridge -- composer retrieval + gate-first moat + spiking WKV
render, all on one SimulationBridge (2026-07-20). GPU-only + needs the grounded-ft ckpt; skips otherwise."""
import os
import pytest

from sim.backend import is_gpu_backend

_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"

pytestmark = pytest.mark.skipif(
    not is_gpu_backend() or not os.path.exists(_CKPT),
    reason="one-bridge conversation demo is GPU-only and needs the grounded-ft ckpt")


def test_onebridge_grounded_conversation_and_moat():
    from research.runners._gap_onebridge_conversation_demo import OneBridgeChat
    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse")]
    chat = OneBridgeChat(facts, seed=42, ckpt=_CKPT)

    inv0 = chat.wkv.n_invocations
    # known -> grounded render contains the composer-retrieved answer word
    r1, a1 = chat.ask("dog", "chase")
    assert a1 == "cat" and "cat" in r1, f"grounded render wrong: {r1!r} (ans {a1})"
    r2, a2 = chat.ask("owl", "eat")
    assert a2 == "mouse" and "mouse" in r2, f"grounded render wrong: {r2!r} (ans {a2})"
    inv_known = chat.wkv.n_invocations

    # unknown -> abstain, WKV NOT invoked (gate-first moat)
    before = chat.wkv.n_invocations
    r3, a3 = chat.ask("lion", "roar")
    assert a3 is None and "don't know" in r3, f"moat broken: {r3!r}"
    assert chat.wkv.n_invocations == before, "WKV invoked on an abstain (gate-first moat broken)"
    assert inv_known - inv0 == 2, "WKV should be invoked exactly once per known question"
