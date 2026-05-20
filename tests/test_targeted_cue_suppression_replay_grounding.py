"""Grounding pin for the 7th arc (targeted cue-suppression-during-replay
+ amplified tag stim + persistent PFC-frame + higher n_replays_per_tag).
RED until Tasks 1+2 land.

After Task 1: test_frozen_verdict_module_importable passes.
After Task 2: test_runner_main_importable passes.
"""

def test_frozen_verdict_module_importable():
    from research.runners.targeted_cue_suppression_replay_core import (
        targeted_cue_suppression_replay_verdict,
        REQUIRED_KEYS,
        _TC_FULL_MIN, _TC_UNIFORM_CTRL_MAX, _TC_DIRECT_RETAIN_MIN,
        _TC_ABSTAIN_CORRECT_MIN, _TC_SCALE_TOL, _TC_LADDER, _TC_MIN_SEEDS,
    )
    assert _TC_FULL_MIN == 0.80
    assert _TC_UNIFORM_CTRL_MAX == 0.10
    assert _TC_DIRECT_RETAIN_MIN == 0.80
    assert _TC_ABSTAIN_CORRECT_MIN == 0.90
    assert _TC_SCALE_TOL == 0.10
    assert _TC_LADDER == (2, 3, 5)
    assert _TC_MIN_SEEDS == 3
    assert isinstance(REQUIRED_KEYS, tuple)
    assert len(REQUIRED_KEYS) == 6
    assert callable(targeted_cue_suppression_replay_verdict)


def test_runner_main_importable():
    from research.runners.targeted_cue_suppression_replay_runner import main
    assert callable(main)
