"""Grounding pin for the 8th arc (pool-readout substitution; SUPERSEDES
the dedicated-region design at be78d14; empirically motivated by
multi-seed signal at 4d6a3a6). RED until Tasks 1+2 land.

After Task 1: test_frozen_verdict_module_importable passes.
After Task 2: test_runner_main_importable passes.
"""

def test_frozen_verdict_module_importable():
    from research.runners.pool_readout_8th_arc_core import (
        pool_readout_8th_arc_verdict,
        REQUIRED_KEYS,
        _CP_FULL_MIN, _CP_UNIFORM_CTRL_MAX, _CP_DIRECT_RETAIN_MIN,
        _CP_ABSTAIN_CORRECT_MIN, _CP_SCALE_TOL, _CP_LADDER, _CP_MIN_SEEDS,
    )
    assert _CP_FULL_MIN == 0.80
    assert _CP_UNIFORM_CTRL_MAX == 0.10
    assert _CP_DIRECT_RETAIN_MIN == 0.80
    assert _CP_ABSTAIN_CORRECT_MIN == 0.90
    assert _CP_SCALE_TOL == 0.10
    assert _CP_LADDER == (2, 3, 5)
    assert _CP_MIN_SEEDS == 3
    assert isinstance(REQUIRED_KEYS, tuple)
    assert len(REQUIRED_KEYS) == 6
    assert callable(pool_readout_8th_arc_verdict)


def test_runner_main_importable():
    from research.runners.pool_readout_8th_arc_runner import main
    assert callable(main)
