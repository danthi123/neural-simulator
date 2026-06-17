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
_CODES = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_CODES),
    reason="320 stream-learned codes cache absent (large research artifact, not committed)",
)


def test_production_agent_on_stream_learned_codes_seed42():
    """The production agent converses end-to-end on the cortex's learned-from-conversation codes: perfect recall,
    the no-confab moat holds (0 false-accepts), yes/no + describe + elaborate all correct."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.consolidated_320_conversation_demo import run_seed
    from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
    from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    codes = np.load(_CODES)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)

    r = run_seed(42, codes, vocab, cat_ids, "host")

    # the no-confab moat is load-bearing: a single false-accept is a breach.
    assert r["false_accept"] == 0, f"MOAT BREACH: {r['breaches']}"
    assert r["abstain"] == 1.0
    assert r["recall"] == 1.0
    assert r["yes_no_ok"]
    assert r["describe_ok"]
    assert r["elaborate_ok"]
    assert r["go"]
