"""Unit tests for the `from_novelty` production rule fill (DR-1 curiosity inversion,
2026-07-23) in sim/neuromodulators.py.

The rule was a RESERVED stub (`return 0.0`). The fill makes it read the brain's
epistemic-gap scalar `core_config.current_novelty_signal` (the exact sibling of
`current_reward_signal` for `from_reward`) and produce `sensitivity*(novelty-baseline)`
to drive a curiosity neuromodulator (typically an excitability_drive on an ASK pool).

These tests pin BOTH:
  (1) BYTE-IDENTICAL-WHEN-OFF: with no novelty signal written (default 0.0), the rule
      returns exactly 0.0 -> a config that does not write a novelty signal is unchanged.
  (2) the new behavior: a written novelty signal raises the curiosity concentration,
      monotone in novelty; a group-scoped excitability_drive target yields a per-neuron
      drive over the ASK pool.

Backend-portable (numpy): the from_novelty branch reads only scalar config fields, so
the manager can be initialized with numpy as the array module.
"""
from __future__ import annotations

import numpy as np

from sim.neuromodulators import (
    ModulatorTarget,
    NeuromodulatorConfig,
    NeuromodulatorManager,
    ProductionRule,
)


class _FakeCfg:
    current_novelty_signal = 0.0
    novelty_baseline = 0.0
    current_reward_signal = 0.0
    reward_baseline = 0.0


class _FakeBridge:
    def __init__(self):
        self.core_config = _FakeCfg()


def _curiosity_nm(sensitivity=1.0, target=None):
    return NeuromodulatorConfig(
        name="curiosity",
        baseline=0.0,
        decay_tau_ms=200.0,
        concentration_min=0.0,
        concentration_max=5.0,
        targets=[target] if target is not None else [],
        production_rules=[ProductionRule(rule_type="from_novelty", sensitivity=sensitivity)],
    )


def test_from_novelty_no_signal_is_byte_identical_to_reserved_stub():
    """With no novelty signal written, from_novelty returns exactly 0.0 (as the old stub
    did) -> the curiosity concentration stays pinned at baseline across many steps."""
    mgr = NeuromodulatorManager([_curiosity_nm()], dt_ms=1.0)
    mgr.initialize(n_neurons=8, cp_module=np)
    bridge = _FakeBridge()  # current_novelty_signal defaults to 0.0
    for _ in range(500):
        mgr.step(bridge)
    assert mgr.get_concentration("curiosity") == 0.0


def test_from_novelty_missing_config_is_noop():
    """A bridge without core_config (unit-test / partial bootstrap) -> 0, no crash."""
    mgr = NeuromodulatorManager([_curiosity_nm()], dt_ms=1.0)
    mgr.initialize(n_neurons=8, cp_module=np)

    class _NoCfg:
        pass

    for _ in range(10):
        mgr.step(_NoCfg())
    assert mgr.get_concentration("curiosity") == 0.0


def test_from_novelty_raises_curiosity_on_high_novelty():
    """A written novelty signal drives the curiosity concentration above baseline."""
    mgr = NeuromodulatorManager([_curiosity_nm(sensitivity=1.0)], dt_ms=1.0)
    mgr.initialize(n_neurons=8, cp_module=np)
    bridge = _FakeBridge()
    bridge.core_config.current_novelty_signal = 0.9
    mgr.step(bridge)
    assert mgr.get_concentration("curiosity") > 0.4
    # decays back to baseline when the concept is no longer novel (signal cleared)
    bridge.core_config.current_novelty_signal = 0.0
    for _ in range(2000):
        mgr.step(bridge)
    assert mgr.get_concentration("curiosity") < 0.05


def test_from_novelty_monotone_in_novelty():
    """Higher epistemic gap -> higher steady curiosity concentration (the wanting tracks
    the gap). Measured as the first-step production, which is sensitivity*(novelty-base)."""
    conc = {}
    for nov in (0.0, 0.3, 0.6, 0.9):
        mgr = NeuromodulatorManager([_curiosity_nm(sensitivity=1.0)], dt_ms=1.0)
        mgr.initialize(n_neurons=8, cp_module=np)
        bridge = _FakeBridge()
        bridge.core_config.current_novelty_signal = float(nov)
        mgr.step(bridge)
        conc[nov] = mgr.get_concentration("curiosity")
    assert conc[0.0] < conc[0.3] < conc[0.6] < conc[0.9]


def test_from_novelty_baseline_subtracts():
    """novelty_baseline is subtracted (a novelty at baseline produces ~0)."""
    mgr = NeuromodulatorManager([_curiosity_nm(sensitivity=1.0)], dt_ms=1.0)
    mgr.initialize(n_neurons=8, cp_module=np)
    bridge = _FakeBridge()
    bridge.core_config.novelty_baseline = 0.5
    bridge.core_config.current_novelty_signal = 0.5  # at baseline -> no production
    mgr.step(bridge)
    assert mgr.get_concentration("curiosity") == 0.0


def test_from_novelty_drives_group_scoped_excitability():
    """A group-scoped excitability_drive target driven by from_novelty yields a per-neuron
    ASK-pool drive that scales with novelty (this is how curiosity -> ASK-pool spiking)."""
    target = ModulatorTarget(target_type="excitability_drive", scope="group:ask",
                             sensitivity=300.0)
    mgr = NeuromodulatorManager([_curiosity_nm(sensitivity=1.0, target=target)], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=np)
    mgr.set_group_indices({"ask": [0, 1, 2, 3]})
    bridge = _FakeBridge()
    bridge.core_config.current_novelty_signal = 0.8
    mgr.step(bridge)
    drive = mgr.compute_excitability_drive_per_neuron(cp_traits=None)
    assert drive is not None
    drive = np.asarray(drive)
    # the ASK slice is driven; the rest is untouched
    assert float(drive[:4].mean()) > 10.0
    assert float(drive[4:].sum()) == 0.0
    # scope=all path is not used here
    assert mgr.compute_excitability_drive_pA() == 0.0


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
