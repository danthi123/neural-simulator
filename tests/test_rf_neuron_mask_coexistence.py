"""Regression tests for the sliced-RF-ops co-residence edit (sim/bridge.py rf_kick + _rf_advance_one).

The edit adds an optional `neuron_mask` to the RF ops so a resonate-and-fire (RF) composer slice can co-reside
on one bridge with an Izhikevich (navigation) slice without either corrupting the other. These pin:
  - mask=None recovers the kicked phases (the default RF path still works);
  - an RF op on a MASKED slice of an Izhikevich bridge reproduces a standalone RF bridge EXACTLY (correctness);
  - the co-resident Izhikevich slice's v/u are byte-identical across the RF op (isolation).
See research/findings/2026-06-10-unification-5b-rf-izh-KILL-and-minimal-edit-approach.md.
"""
import numpy as np
import pytest

from sim.backend import to_host
from research.runners.rf_phasor_composer import _build_rf_bridge
from research.runners.derisk_unification_5b_rf_izh_coexistence import build_izh_bridge

PERIOD = 200
SEED = 42


def _kicked_phases(r):
    return (np.arange(r) + 0.5) / r


def test_rf_mask_none_recovers_kicked_phases():
    """Default path (no mask): a free-resonate recovers the kicked phases -> the RF path is intact."""
    r = 8
    phases = _kicked_phases(r)
    b = _build_rf_bridge(r, seed=SEED)
    b.rf_kick(np.exp(2j * np.pi * phases), period=PERIOD, lam=0.0)
    b.rf_resonate_steps(PERIOD + 8)
    got = b.rf_read_phases()
    # phases recovered within one-step quantization (1/period)
    assert np.max(np.abs(got - phases)) <= 1.5 / PERIOD


def test_rf_mask_slice_matches_standalone_and_isolates_izhikevich():
    """An RF op on a masked slice of an Izhikevich bridge == a standalone RF bridge, and leaves the
    Izhikevich slice byte-identical."""
    n_izh, n_rf = 24, 8
    n = n_izh + n_rf
    phases = _kicked_phases(n_rf)
    kick_r = np.exp(2j * np.pi * phases)

    # standalone RF reference
    ref = _build_rf_bridge(n_rf, seed=SEED)
    ref.rf_kick(kick_r, period=PERIOD, lam=0.0)
    ref.rf_resonate_steps(PERIOD + 8)
    standalone = ref.rf_read_phases()

    # mixed Izhikevich bridge with an RF slice [n_izh : n_izh+n_rf]
    mix = build_izh_bridge(n, seed=SEED)
    try:
        import cupy as _cp  # noqa
        xp = _cp
    except Exception:
        xp = np
    izh_idx = xp.asarray(np.arange(0, n_izh))
    for _ in range(20):
        mix.cp_external_input_current[:] = 0.0
        mix.cp_external_input_current[izh_idx] = 600.0
        mix._run_one_simulation_step()
    mix.cp_external_input_current[:] = 0.0
    izh_v_before = to_host(mix.cp_membrane_potential_v[:n_izh]).copy()
    izh_u_before = to_host(mix.cp_recovery_variable_u[:n_izh]).copy()

    mask = np.zeros(n, dtype=bool)
    mask[n_izh:n_izh + n_rf] = True
    full_kick = np.zeros(n, dtype=np.complex128)
    full_kick[n_izh:n_izh + n_rf] = kick_r
    mix.rf_kick(full_kick, period=PERIOD, lam=0.0, neuron_mask=mask)
    mix.rf_resonate_steps(PERIOD + 8)
    mixed = mix.rf_read_phases()[n_izh:n_izh + n_rf]

    izh_v_after = to_host(mix.cp_membrane_potential_v[:n_izh])
    izh_u_after = to_host(mix.cp_recovery_variable_u[:n_izh])

    # (1) correctness: the masked slice reproduces the standalone bridge exactly
    assert np.array_equal(mixed, standalone), f"masked-slice phases != standalone (max d={np.max(np.abs(mixed-standalone)):.3e})"
    # (2) isolation: the Izhikevich slice is byte-identical across the RF op
    assert np.array_equal(izh_v_before, izh_v_after), "RF op moved the Izhikevich slice v"
    assert np.array_equal(izh_u_before, izh_u_after), "RF op moved the Izhikevich slice u"
    # the RF op actually ran (the RF slice is off-rest)
    assert float(np.max(np.abs(to_host(mix.cp_membrane_potential_v[n_izh:n_izh + n_rf])))) > 1e-6
