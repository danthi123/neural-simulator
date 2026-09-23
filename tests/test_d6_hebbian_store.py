"""D6 learn-through-use: the in-conversation fact WRITE by a local Hebbian rule (research/runners/d6_hebbian_store.py).

Pins: (1) OFF (unset) is byte-identical to the direct composite copy; (2) the Hebbian-written block recalls with the
direct path's answers and phase (content = the rule's output, not a copy); (3) the FREEZE lesion blocks ONLY
in-conversation writes -- the taught fact is gone, the build-time fact still recalls, the frozen weight is 0;
(4) the freeze flag without the conversation context does not freeze; (5) the Hebbian write is deterministic;
(6) the D6 probe's verdict fails in every failing direction (runner selftest). numpy, tiny vocab (~1 min total).
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

VOCAB = sorted({"dog", "chase", "cat", "eat", "fish", "wolf", "hunt", "deer", "fox", "berry", "bird", "worm"})


def _build(store=None, freeze=None, seed=42):
    from research.runners.one_brain_composer import OneBrainComposer
    for k, v in (("BRAIN_D6_HEBBIAN_STORE", store), ("BRAIN_D6_HEBBIAN_FREEZE", freeze)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, vocab_headroom=2)


def _block(c, i):
    return np.array([complex(w) for (_p, _q, w) in c.store_conns[i * c.D:(i + 1) * c.D]])


@pytest.fixture(scope="module")
def arms():
    from research.runners import d6_hebbian_store as d6
    out = {}
    c = _build(None); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["unset"] = c
    c = _build("0"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["off"] = c
    c = _build("1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["hebb"] = c
    c = _build("1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["hebb_rep"] = c
    c = _build("1", "1"); c.hear("dog chase cat")
    with d6.conversation_write(c):
        c.hear("wolf hunt deer")
    out["frozen_conv"] = c
    c = _build("1", "1"); c.hear("dog chase cat"); c.hear("wolf hunt deer"); out["frozen_no_ctx"] = c
    for k in ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_HEBBIAN_FREEZE"):
        os.environ.pop(k, None)
    return out


def test_off_is_byte_identical(arms):
    assert arms["unset"].store_conns == arms["off"].store_conns


def test_hebbian_block_recalls_and_matches_direct_phase(arms):
    h, d = arms["hebb"], arms["unset"]
    assert h.query_patient("dog", "chase") == "cat"
    assert h.query_patient("wolf", "hunt") == "deer"
    assert h.query_patient("fox", "eat") is None                  # moat: nothing stored
    for i in range(2):
        dphi = np.angle(_block(h, i) * np.conj(_block(d, i)))
        assert float(np.mean(np.abs(dphi))) < 0.05                 # phase = the rule's output ~= the composite
    assert h._d6_last_encode["frozen"] is False and h._d6_last_encode["n_saturated"] == h.D


def test_freeze_blocks_only_the_in_conversation_write(arms):
    f = arms["frozen_conv"]
    assert f._d6_last_encode["frozen"] is True
    assert float(np.max(np.abs(_block(f, 1)))) == 0.0            # the lever moved: no weight change
    assert float(np.mean(np.abs(_block(f, 0)))) > 0.5             # the build-time block was written
    assert f.query_patient("wolf", "hunt") is None                # taught-in-conversation fact is gone
    assert f.query_patient("dog", "chase") == "cat"               # read path intact


def test_freeze_without_conversation_context_does_not_freeze(arms):
    f = arms["frozen_no_ctx"]
    assert f.query_patient("wolf", "hunt") == "deer"


def test_hebbian_write_is_deterministic(arms):
    assert arms["hebb"].store_conns == arms["hebb_rep"].store_conns


def test_d6_probe_verdict_selftest():
    from research.runners.d6_learn_through_use_lb import selftest
    assert selftest() == 0
