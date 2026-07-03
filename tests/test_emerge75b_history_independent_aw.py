"""CPU-safe tests for EMERGE-75b -- history-independent A->W read-out (EMERGE-61 wash-out on the A->W decode).

The on-spikes A->W read-out is GPU-only (SIM_BACKEND=cupy); these tests verify the CPU-testable parts: the wash-out
INSTALL + TOGGLE logic (with a fake bridge), the class wiring, and that the GPU de-risk skips gracefully on numpy.
The full 6-seed GPU de-risk is validated by `--derisk` on cupy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge75b_history_independent_aw_derisk as m75b
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _STATE_ARRAYS, _snapshot_state


class _FakeBridge:
    """A minimal stand-in exposing the EMERGE-61 _STATE_ARRAYS as numpy arrays, so the wash-out snapshot/restore can be
    unit-tested without a GPU bridge. `cp_recovery_variable_u` is the load-bearing Izhikevich slow-adaptation array."""

    def __init__(self):
        for name in _STATE_ARRAYS:
            setattr(self, name, np.zeros(8, dtype=np.float32))


class _FakeEngine:
    """A fake A->W engine whose `_decode` ACCUMULATES adaptation (mutates cp_recovery_variable_u) and returns the
    current accumulated value -- so we can assert the wash-out resets it before each decode."""

    def __init__(self):
        self.bridge = _FakeBridge()

    def _decode(self, word):
        # simulate the Izhikevich slow-adaptation accumulating across sequential decodes
        self.bridge.cp_recovery_variable_u += 1.0
        return (word, float(self.bridge.cp_recovery_variable_u[0]), 0.0, 0)


def test_install_history_independence_snapshots_and_wraps():
    eng = _FakeEngine()
    m75b._install_history_independence(eng)
    assert hasattr(eng, "_hi_snapshot")
    assert hasattr(eng, "_hi_enabled") and eng._hi_enabled is True
    # the snapshot captured the clean (all-zero) recovery-variable state
    assert "cp_recovery_variable_u" in eng._hi_snapshot
    assert float(eng._hi_snapshot["cp_recovery_variable_u"][0]) == 0.0


def test_hi_on_makes_decode_history_independent():
    """With the wash-out ON, every decode restores the clean state first -> the accumulated value is reset each call."""
    eng = _FakeEngine()
    m75b._install_history_independence(eng)   # hi ON by default
    # each decode restores u to 0 then adds 1 -> always returns 1.0 regardless of history
    vals = [eng._decode(f"w{i}")[1] for i in range(5)]
    assert vals == [1.0, 1.0, 1.0, 1.0, 1.0], vals


def test_hi_off_lets_adaptation_accumulate():
    """With the wash-out OFF (the EMERGE-75 baseline), adaptation accumulates across decodes -> the value climbs."""
    eng = _FakeEngine()
    m75b._install_history_independence(eng)
    eng._hi_enabled = False
    vals = [eng._decode(f"w{i}")[1] for i in range(5)]
    assert vals == [1.0, 2.0, 3.0, 4.0, 5.0], vals   # un-washed accumulation (reproduces the deep-history regression)


def test_toggle_restores_history_independence():
    """Toggling hi back ON after an un-washed burst re-establishes history-independence."""
    eng = _FakeEngine()
    m75b._install_history_independence(eng)
    eng._hi_enabled = False
    for i in range(4):
        eng._decode(f"w{i}")            # accumulate to 4
    eng._hi_enabled = True
    assert eng._decode("z")[1] == 1.0   # wash-out restores the clean snapshot -> 1.0 again


def test_unified_histindep_is_subclass():
    from research.runners._emerge75_aw_vocab_scaling_derisk import UnifiedNeuralSpell75
    assert issubclass(m75b.UnifiedHistIndepSpell75, UnifiedNeuralSpell75)
    # set_hi exists and is the toggle API
    assert hasattr(m75b.UnifiedHistIndepSpell75, "set_hi")


def test_snapshot_state_roundtrip_on_fake_bridge():
    b = _FakeBridge()
    b.cp_recovery_variable_u[:] = 7.0
    snap = _snapshot_state(b)
    b.cp_recovery_variable_u[:] = 99.0      # perturb
    from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
    _restore_state(b, snap)
    assert float(b.cp_recovery_variable_u[0]) == 7.0   # restored to the snapshot


def test_scope_and_overflow_vocab_reexported():
    # the de-risk reuses EMERGE-75's scope + overflow vocab verbatim (no vocab change in 75b)
    assert len(m75b._OVF_VOCAB16) == 16
    assert set(m75b._OVF_FUNC) == {"to", "on", "is"}
    assert len(m75b._OVF_OBJ) == 13
    assert "C_PPGOAL" in m75b._SCOPE_CONSTRUCTIONS and "F_MODAL" in m75b._SCOPE_CONSTRUCTIONS


@pytest.mark.skipif(os.environ.get("SIM_BACKEND", "numpy") != "cupy",
                    reason="the on-spikes A->W read-out needs SIM_BACKEND=cupy; validated by --derisk on GPU")
def test_gpu_derisk_smoke():
    rc = m75b._derisk([42])
    assert rc in (0, 1)
