"""CI GUARD (track-A closeout): the production conversational agent must keep running FULLY BRAIN-BASED when the
three neural flags are flipped ON -- enable_spiking_cleanup (spiking matched-filter + WTA cleanup), enable_substrate
_store (spiking weight-store fact memory), enable_neural_render (spiking competitive-queuing word order) -- at
parity with the numpy-default oracle, with the no-confab moat intact.

Why this test exists: numpy is the production DEFAULT for speed, and the spiking versions are opt-in. Without this
guard the validated spiking path silently BIT-ROTS as the production code evolves around it ("validated once" decays
to "validated last March"). This test runs the capability matrix with the flags ON every CI run, so the brain-based
claim stays earned. See `2026-06-18-conversational-brain-based-only-audit.md` + the numpy-vs-spiking discussion.

GPU-only (the spiking ops need CuPy); skips gracefully if the concept-code cache is absent (like the other on-brain
agent tests).
"""
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gpu_backend(),
                                reason="the production spiking flags need the CuPy/GPU substrate")

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "see", "eat", "swim", "stop",
         "north", "south", "east", "west"]


def _build(seed, spiking):
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    a = BrainConversationalAgent(seed=seed, concepts={w: None for w in VOCAB},
                                 enable_spiking_cleanup=spiking, enable_substrate_store=spiking,
                                 enable_neural_render=spiking)
    a.hear("dog go north")
    a.hear("cat come south", polarity="AFFIRM")
    a.hear("river look west", polarity="NEGATE")
    a.hear("dog eat cat")
    return a


def _matrix(a):
    return {
        "what_dog_go": a.what_does("dog", "go"),
        "who_go_north": a.who_does("go", "north"),
        "yes_cat_come_south": a.is_it_true("cat", "come", "south"),
        "no_river_look_west": a.is_it_true("river", "look", "west"),
        "unknown_apple": a.is_it_true("apple", "stop", "east"),
        "abstain_bird_see": a.what_does("bird", "see"),
    }


def test_production_spiking_flags_match_oracle_and_moat():
    try:
        oracle = _matrix(_build(42, spiking=False))
        spiking = _matrix(_build(42, spiking=True))
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")

    expected = {"what_dog_go": "north", "who_go_north": "dog", "yes_cat_come_south": "yes",
                "no_river_look_west": "no", "unknown_apple": "unknown", "abstain_bird_see": None}

    # the fully-spiking production path must equal the numpy oracle AND ground truth on every op
    for k in oracle:
        assert spiking[k] == oracle[k], f"spiking flag path diverged from numpy oracle on {k}: {spiking[k]} != {oracle[k]}"
        assert spiking[k] == expected[k], f"spiking flag path wrong on {k}: {spiking[k]} != {expected[k]}"

    # the no-confab moat must hold under the spiking store + cleanup
    assert spiking["unknown_apple"] == "unknown", "moat breach: unstored fact not abstained (spiking path)"
    assert spiking["abstain_bird_see"] is None, "moat breach: unstored agent not abstained (spiking path)"
