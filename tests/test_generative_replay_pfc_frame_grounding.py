"""Grounding pin for the generative-replay + PFC-frame arc (6th
architecture). RED until Tasks 1 + 2 land. Verifies the new frozen
verdict module + runner module are importable + frozen-bar constants
are pinned to the documented values + REQUIRED_KEYS tuple has the
correct shape + main() is callable.

After Task 1: test_frozen_verdict_module_importable passes.
After Task 2: test_runner_main_importable passes.
"""

def test_frozen_verdict_module_importable():
    from research.runners.generative_replay_pfc_frame_core import (
        generative_replay_pfc_frame_verdict,
        REQUIRED_KEYS,
        _GR_FULL_MIN, _GR_UNIFORM_CTRL_MAX, _GR_DIRECT_RETAIN_MIN,
        _GR_ABSTAIN_CORRECT_MIN, _GR_SCALE_TOL, _GR_LADDER, _GR_MIN_SEEDS,
    )
    assert _GR_FULL_MIN == 0.80
    assert _GR_UNIFORM_CTRL_MAX == 0.10
    assert _GR_DIRECT_RETAIN_MIN == 0.80
    assert _GR_ABSTAIN_CORRECT_MIN == 0.90
    assert _GR_SCALE_TOL == 0.10
    assert _GR_LADDER == (2, 3, 5)
    assert _GR_MIN_SEEDS == 3
    # REQUIRED_KEYS shape verified separately via the frozen verdict
    # tests; here we just confirm the tuple exists and has 6 entries.
    assert isinstance(REQUIRED_KEYS, tuple)
    assert len(REQUIRED_KEYS) == 6
    assert callable(generative_replay_pfc_frame_verdict)


def test_runner_main_importable():
    from research.runners.generative_replay_pfc_frame_runner import main
    assert callable(main)
