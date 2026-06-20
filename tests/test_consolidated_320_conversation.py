"""Regression guard for the consolidation: the PRODUCTION conversational agent talking with the 320-concept
codes it LEARNED FROM CONVERSATION (the stream cortex). Skips gracefully when the cached stream codes are
absent (they are large research artifacts, not committed), matching the project's on-brain test idiom.

The load-bearing assertions: the no-confab moat must hold (ZERO false-accepts) and recall must be perfect on the
small fact set. A false-accept here would be a moat breach.
"""
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
_RAW = os.path.join(_REPO, "research", "findings", "raw")
# host = the double-centring read-out (the escape / test-oracle path). neural = the on-bridge read-out
# normalization (burndown #5: per-hub spike-freq adaptation + per-concept feedforward inhibition) — the
# PRODUCTION DEFAULT (the demo's `--readout` default). Both guarded; each skips if its codes are absent.
_HOST_CODES = os.path.join(_RAW, "_phaseB_stream_codes_320_seed42.npy")
_NEURAL_CODES = os.path.join(_RAW, "_phaseB_stream_codes_320_neural_seed42.npy")


def _run_conversation(codes_path, readout):
    """Drive the production agent's full who/what+moat turn on the cached codes; return the run_seed dict."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.consolidated_320_conversation_demo import run_seed
    from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
    from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    codes = np.load(codes_path)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    return run_seed(42, codes, vocab, cat_ids, readout)


def _assert_go(r):
    """Load-bearing: the no-confab moat holds (0 false-accepts) and the full turn is correct == the baseline."""
    assert r["false_accept"] == 0, f"MOAT BREACH: {r['breaches']}"
    assert r["abstain"] == 1.0
    assert r["recall"] == 1.0
    assert r["yes_no_ok"]
    assert r["describe_ok"]
    assert r["elaborate_ok"]
    assert r["go"]


@pytest.mark.skipif(
    not os.path.exists(_HOST_CODES),
    reason="320 stream-learned host codes cache absent (large research artifact, not committed)",
)
def test_production_agent_on_stream_learned_codes_seed42():
    """The production agent on the host double-centring read-out codes (the escape path): perfect recall, the
    no-confab moat holds (0 false-accepts), yes/no + describe + elaborate all correct."""
    _assert_go(_run_conversation(_HOST_CODES, "host"))


@pytest.mark.skipif(
    not os.path.exists(_NEURAL_CODES),
    reason="320 stream-learned NEURAL codes cache absent (large research artifact, not committed)",
)
def test_production_agent_on_neural_readout_codes_seed42():
    """The PRODUCTION DEFAULT path (burndown #5 flip): the production agent on the fully-on-bridge read-out
    normalization codes (per-hub spike-freq adaptation + per-concept feedforward inhibition) == the host
    baseline — perfect recall AND the no-confab moat holds at 0 false-accepts (the lower margin does NOT leak
    a false-accept)."""
    _assert_go(_run_conversation(_NEURAL_CODES, "neural"))
