"""CI guard -- comprehension-monitor's lesion-twin crash fix (2026-09-29, B2d reply-level rescore audit).

`research/runners/comprehension_production_organ.py`'s `_lesion_comp()` builds its lesion twin STANDALONE
(`_build_comp(self.seed)`, no `shared=`) precisely so it needs no pool guard -- but `_snapshot_rest()` branched
only on whether the ORGAN itself was pool-attached (`self._shared is not None`), not on WHICH comp bridge was
being snapshotted. When the organ runs pool-attached (production default, `BRAIN_ONEBRAIN_WAVE3_POOL`),
`_snapshot_rest(self.comp_lesion)` wrongly took the shared-pool branch and stored the POOL's N=7002 rest
snapshot for the twin's OWN N=1032-neuron bridge; `_hard_reset()` then raised
`ValueError: could not broadcast input array from shape (7002,) into shape (1032,)`. This crashed 88/88 retained
comprehension-monitor lesion arms since the 2026-09-21 adequate pilot (every battery through B2b): the row's
"6/6 seeds load-bearing" verdict was a harness crash caught by the server's except branch
(`webapp/server.py:6108-6109`, `comprehended` True -> None read as a regression), not a genuine lesion effect.

These tests exercise `_snapshot_rest` / `_hard_reset` directly on lightweight fake comp bridges (no real spiking
substrate needed -- the bug is a branch-selection defect in these two methods, independent of what runs on the
bridge). This reproduces the exact crash shape (a real N=7002 vs N=1032 size mismatch) without building the full
merged cortical pool. CPU/numpy, no brain build.
"""
import contextlib
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

import research.runners.comprehension_production_organ as CO

POOL_N = 7002     # the one-brain merged pool's bridge size (matches the real crash report)
TWIN_N = 1032     # the comprehension organ's own standalone bridge size


class _FakeBridge:
    """A minimal duck-typed stand-in for `SpikingRoleCompetition.bridge`: only the arrays
    `_snapshot_rest`/`_hard_reset` touch, each independently sized so a wrong-size assignment raises exactly
    the reported ValueError."""

    def __init__(self, n, fill=0.0):
        self.n = n
        self.cp_membrane_potential_v = np.full(n, fill, dtype=np.float32)
        self.cp_recovery_variable_u = np.full(n, fill, dtype=np.float32)
        self.cp_conductance_g_e = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_i = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_gabab = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_nmda = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_nmda_rise = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_nmda_recurrent = np.zeros(n, dtype=np.float32)
        self.cp_conductance_g_nmda_recurrent_rise = np.zeros(n, dtype=np.float32)
        self.cp_firing_states = np.zeros(n, dtype=bool)
        self.cp_refractory_timers = np.zeros(n, dtype=np.float32)
        self.cp_prev_firing_states = np.zeros(n, dtype=bool)
        self.cp_neuron_activity_ema = np.zeros(n, dtype=np.float32)
        self.cp_neuron_firing_thresholds = np.zeros(n, dtype=np.float32)
        self.cp_external_input_current = np.zeros(n, dtype=np.float32)
        self.n_steps_run = 0

    def _run_one_simulation_step(self):
        self.n_steps_run += 1


class _FakeComp:
    def __init__(self, n, fill=0.0):
        self.bridge = _FakeBridge(n, fill=fill)
        self.xp = np


class _FakeShared:
    """A minimal stand-in for the one-brain merged pool: only `.snap` (the pre-settled whole-bridge rest
    snapshot) and `.read_isolation()` (used only on the self.comp / intact path, never on the lesion twin)."""

    def __init__(self, n):
        self.snap = {
            "cp_membrane_potential_v": np.full(n, -65.0, dtype=np.float32),
            "cp_recovery_variable_u": np.full(n, -14.0, dtype=np.float32),
        }

    def read_isolation(self, name):
        return contextlib.nullcontext()


def _pool_attached_organ():
    """An organ whose `_shared` is set (pool-attached, the production-default configuration) and whose
    `self.comp` is a fake bridge SIZED LIKE THE POOL (matching what `_build_comp(seed, shared=pool)` would
    actually build)."""
    shared = _FakeShared(POOL_N)
    organ = CO.ComprehensionProductionOrgan(seed=42, shared=shared)
    organ.comp = _FakeComp(POOL_N)
    organ._built = True
    return organ, shared


def test_lesion_twin_snapshot_no_longer_takes_the_shared_pool_branch():
    """THE FIX: a standalone twin (comp_lesion-shaped, N=1032) snapshotted while the organ is pool-attached
    must take its OWN 40-step settle, not the pool's N=7002 snapshot -- so `_hard_reset` on it must not raise."""
    organ, shared = _pool_attached_organ()
    twin = _FakeComp(TWIN_N)
    organ._snapshot_rest(twin)          # must not raise, and must not borrow the pool's (7002,)-shaped rest
    rest_v, rest_u = organ._rest[id(twin)]
    assert rest_v.shape == (TWIN_N,)
    assert rest_u.shape == (TWIN_N,)
    # the manual settle path was actually exercised (not silently skipped)
    assert twin.bridge.n_steps_run == 40
    organ._hard_reset(twin)             # this is exactly where the reported ValueError fired pre-fix
    assert twin.bridge.cp_membrane_potential_v.shape == (TWIN_N,)


def test_lesion_twin_snapshot_pre_fix_shape_mismatch_would_have_raised():
    """Documents the exact pre-fix crash shape: if the twin's rest snapshot were (wrongly) borrowed from the
    pool's differently-sized snap, `_hard_reset`'s in-place assignment raises ValueError -- pinning that the
    fix is not vacuous (the two sizes genuinely conflict when the branch is wrong)."""
    twin = _FakeComp(TWIN_N)
    pool_shaped_rest = np.full(POOL_N, -65.0, dtype=np.float32)
    with pytest.raises(ValueError, match="broadcast"):
        twin.bridge.cp_membrane_potential_v[:] = pool_shaped_rest


def test_intact_path_still_uses_the_shared_pool_snapshot_byte_identical():
    """Intact-path behaviour is UNCHANGED by the fix: `comp is self.comp` is trivially true for the organ's own
    (pool-built) comp, so it still takes the shared-snapshot branch and copies the pool's snap values exactly,
    with no 40-step settle run on it."""
    organ, shared = _pool_attached_organ()
    organ._snapshot_rest(organ.comp)
    rest_v, rest_u = organ._rest[id(organ.comp)]
    assert np.array_equal(rest_v, shared.snap["cp_membrane_potential_v"])
    assert np.array_equal(rest_u, shared.snap["cp_recovery_variable_u"])
    assert organ.comp.bridge.n_steps_run == 0     # no manual settle -- the pool's own snapshot was used verbatim
    organ._hard_reset(organ.comp)
    assert np.array_equal(organ.comp.bridge.cp_membrane_potential_v, shared.snap["cp_membrane_potential_v"])


def test_standalone_organ_unaffected_shared_none():
    """`shared=None` (the pre-existing, non-pooled organ) always took the manual-settle branch for every comp,
    self.comp included -- unchanged by this fix (the added `comp is self.comp` clause is only reachable when
    `self._shared is not None`)."""
    organ = CO.ComprehensionProductionOrgan(seed=42, shared=None)
    organ.comp = _FakeComp(TWIN_N)
    organ._built = True
    organ._snapshot_rest(organ.comp)
    assert organ.comp.bridge.n_steps_run == 40
    organ._hard_reset(organ.comp)   # no raise
