"""Grounding pin for theta-gamma mode-unification arc.

RED until Task 1 + Task 2 land. Verifies that the new frozen verdict
module + runner module are importable + the runner exposes a main()
entry point + the frozen-bars constants are pinned.

This is Task 0 of the theta-gamma mode-unification implementation plan
(docs/plans/2026-05-20-theta-gamma-mode-unification-implementation.md).
The two test functions intentionally fail RED until:
  - Task 1 creates research/runners/theta_gamma_mode_unification_core.py
    (frozen capability-verdict module with the _TG_* bars).
  - Task 2 creates research/runners/theta_gamma_mode_unification_runner.py
    (net-new runner exposing a main() entry point).

Frozen bars (must be set ONCE in the core module and NEVER re-tuned):
  _TG_FULL_MIN              = 0.80
  _TG_UNIFORM_CTRL_MAX      = 0.10
  _TG_DIRECT_RETAIN_MIN     = 0.80
  _TG_ABSTAIN_CORRECT_MIN   = 0.90
  _TG_SCALE_TOL             = 0.10
  _TG_LADDER                = (2, 3, 5)
  _TG_MIN_SEEDS             = 3
"""


def test_frozen_verdict_module_importable():
    from research.runners.theta_gamma_mode_unification_core import (
        theta_gamma_mode_unification_verdict,
        REQUIRED_KEYS,
        _TG_FULL_MIN,
        _TG_UNIFORM_CTRL_MAX,
        _TG_DIRECT_RETAIN_MIN,
        _TG_ABSTAIN_CORRECT_MIN,
        _TG_SCALE_TOL,
        _TG_LADDER,
        _TG_MIN_SEEDS,
    )
    assert _TG_FULL_MIN == 0.80
    assert _TG_UNIFORM_CTRL_MAX == 0.10
    assert _TG_DIRECT_RETAIN_MIN == 0.80
    assert _TG_ABSTAIN_CORRECT_MIN == 0.90
    assert _TG_SCALE_TOL == 0.10
    assert _TG_LADDER == (2, 3, 5)
    assert _TG_MIN_SEEDS == 3
    assert REQUIRED_KEYS == (
        "N",
        "n_seeds",
        "full_acc",
        "uniform_ctrl_acc",
        "direct_retain_acc",
        "abstain_correct",
    )
    assert callable(theta_gamma_mode_unification_verdict)


def test_runner_main_importable():
    from research.runners.theta_gamma_mode_unification_runner import main
    assert callable(main)
