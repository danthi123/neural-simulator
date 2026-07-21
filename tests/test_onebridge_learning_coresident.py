"""CI guard: the render read-out LEARNING (delta-rule over cp_ssm_state) runs co-resident with the composer on ONE
bridge and GENERALIZES (teacher-student, held-out); frozen = load-bearing; the interleaved composer op does not
perturb it (2026-07-20, post adversarial-audit). GPU-only; skips on numpy."""
import pytest

from sim.backend import is_gpu_backend

pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="GPU-only (RF ops + ssm read-out)")


def test_learning_generalizes_coresident_with_composer():
    from research.runners._gap_onebridge_learning_coresident_derisk import _run
    # MAIN with the composer interleaved: held-out loss drops a lot (genuine generalization, n_train>>n_read)
    l0, l1, ans, abstain = _run(42, epochs=25, lr=2.0, D_cmp=64, frozen=False, interleave=True)
    assert l1 < 0.2 * l0, f"read-out did not generalize on shared bridge: {l0} -> {l1}"
    assert ans == ["cat", "mouse"], f"composer recall wrong while learning: {ans}"
    assert abstain is None, "no-confab moat broken while learning"
    # INTERLEAVE non-interference: identical training WITHOUT the composer op -> identical held-out loss
    _l0b, l1b, _a, _ab = _run(42, epochs=25, lr=2.0, D_cmp=64, frozen=False, interleave=False)
    assert abs(l1 - l1b) < 1e-6, f"composer op perturbed the learning: {l1} vs {l1b}"


def test_frozen_readout_does_not_generalize():
    from research.runners._gap_onebridge_learning_coresident_derisk import _run
    l0, l1, _, _ = _run(42, epochs=25, lr=2.0, D_cmp=64, frozen=True, interleave=True)
    assert l1 > 0.5 * l0, "frozen read-out's held-out loss dropped (delta update not load-bearing)"
