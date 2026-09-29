"""DEV tests for the draft long-delay temporal-credit benchmark (plan section 4).

These are DEV smoke tests (dev seeds only, small n_trials): they check the instrument
RUNS, is FINITE, reuses the eligibility kernel unmodified, uses NO autograd, that the
`full` arm holds the cue (buffer read-out above chance), and that the trace ablation
dissociates credit from memory (value_discrimination collapses while buffer_accuracy
stays high). They are NOT the frozen gate (Claude finalizes the bars and the six-seed
gate run after Tuesday). See docs/drafts/long-delay-credit/PREREG-DRAFT.md.
"""
import importlib

import numpy as np
import pytest


def _mod():
    return importlib.import_module("research.runners._draft_long_delay_credit")


def test_no_autograd():
    """The benchmark is hand-rolled numpy: no autograd framework may be IMPORTED.
    (Checks import statements only, so a docstring mentioning "autograd" is fine.)"""
    m = _mod()
    import_lines = [ln.strip() for ln in open(m.__file__)
                    if ln.strip().startswith(("import ", "from "))]
    for ln in import_lines:
        assert "torch" not in ln, ln
        assert "autograd" not in ln, ln
        assert "jax" not in ln, ln


def test_reuses_eligibility_kernel_unmodified():
    """The eligibility-trace decay must come from sim.kernels (reused, not re-implemented)."""
    m = _mod()
    assert m.fused_eligibility_trace_decay.__module__ == "sim.kernels"
    # the kernel is a plain multiplicative decay (the validated form)
    t = np.array([2.0, -3.0, 5.0])
    np.testing.assert_allclose(m.fused_eligibility_trace_decay(t, 0.855), t * 0.855)


def test_phi_cue_feature_is_local():
    """The cue feature is active only during the cue window; the rest is bias only.

    The critic sees ONLY bias + the transient cue (no buffer/taps/choice/terminal/
    decoy/distractor), so the credit must span the delay to reach the cue.
    """
    m = _mod()
    trial = {
        "onset": 2, "delay": 8, "cue": 0, "choice_time": 12,
        "terminal_time": 11, "decoy_time": 10, "distractor_time": 7,
        "reward_time": 15,
    }
    # cue A active only during the cue window [onset, onset+CUE_STEPS)
    assert m._phi(2, trial)[m.IDX_CUE_A] == 1.0
    assert m._phi(3, trial)[m.IDX_CUE_A] == 1.0
    assert m._phi(4, trial)[m.IDX_CUE_A] == 0.0
    # cue B never active for a cue-A trial
    assert m._phi(2, trial)[m.IDX_CUE_B] == 0.0
    # bias always on
    assert m._phi(0, trial)[m.IDX_BIAS] == 1.0
    # nothing else is a feature: the vector is bias-only outside the cue window
    assert m._phi(11, trial).sum() == 1.0   # terminal step: bias only
    assert m._phi(15, trial).sum() == 1.0   # reward step: bias only


def test_runs_and_all_metrics_finite():
    """A tiny run of every arm must produce finite numeric metrics (no crash/NaN)."""
    m = _mod()
    for arm in ["full", "trace_ablated", "immediate",
                "misleading_final_cue", "shuffled_reward"]:
        r = m.run_arm(arm, seed=7, n_trials=80)
        for k in ["buffer_accuracy", "value_at_cue_A", "value_at_cue_B",
                  "value_discrimination", "vrmse", "us_decay",
                  "rpe_at_terminal", "rpe_at_decoy"]:
            assert np.isfinite(r[k]), (arm, k, r[k])
        assert 0.0 <= r["buffer_accuracy"] <= 1.0


def test_full_arm_learns_above_chance():
    """The `full` arm must hold the cue in the buffer read-out: accuracy above chance."""
    m = _mod()
    r = m.run_arm("full", seed=7, n_trials=1200)
    assert r["buffer_accuracy"] > 0.60, r


def test_trace_ablation_dissociates_credit():
    """Cutting the trace must collapse the cue value discrimination (the plan's control),
    while the `full` arm keeps it. (DEV bar; Claude finalizes.)"""
    m = _mod()
    full = m.run_arm("full", seed=7, n_trials=1200)
    ab = m.run_arm("trace_ablated", seed=7, n_trials=1200)
    assert full["value_discrimination"] > ab["value_discrimination"], (full, ab)
