"""SIM_STABLE_INTEGRATION (sim/bridge.py STABLE_INTEGRATION_ENV) -- the whole-brain, DEFAULT-OFF stable-integration
switch. When "1", every SimulationBridge runs with cfg.enable_bounded_conductance_integration=True,
cfg.izh_membrane_substeps=4 and cfg.izh_substep_refractory_rule="model_reset" (Izhikevich; AdEx gets the bounded
integration only), whatever the organ configured, applied at construction, at _initialize_simulation_data and
re-asserted at the top of every step. Unset / "" / "0": nothing is touched and a short bridge run is byte-identical to
the pre-switch engine (the bounded-conductance reference network and its hash captured on main).
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sim.bridge as SB  # noqa: E402
from sim import CoreSimConfig, NeuronModel, SimulationBridge  # noqa: E402
from tests.test_bounded_conductance_integration import (  # noqa: E402
    _REFERENCE_HASHES_FROM_MAIN, _backend_name, _build_reference_network_bridge, _run_and_hash)

ENV = SB.STABLE_INTEGRATION_ENV


def _run_ref(monkeypatch, value, *, organ_bounded=None, n_steps=60):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    b = _build_reference_network_bridge(seed=42, set_flag_explicitly=organ_bounded)
    h, raster, v = _run_and_hash(b, n_steps)
    return b, h, raster, v


@pytest.mark.parametrize("value", [None, "", "0"])
def test_off_is_byte_identical_to_the_pre_switch_engine(monkeypatch, value):
    b, h, raster, v = _run_ref(monkeypatch, value)
    assert int(raster.sum()) > 0
    assert b.stable_integration_requested is False
    assert b.stable_integration_applied is False
    assert b.stable_integration_effective is None and b.stable_integration_reasserts == 0
    cfg = b.core_config
    assert cfg.enable_bounded_conductance_integration is False
    assert cfg.izh_membrane_substeps == 1 and cfg.izh_substep_refractory_rule == "model_reset"
    assert b.substep_calls == 0
    ref = _REFERENCE_HASHES_FROM_MAIN.get(_backend_name())
    if ref is not None:
        assert h == ref
    # and against an unset run in the same process (both spellings of off agree)
    _, h_unset, _, v_unset = _run_ref(monkeypatch, None)
    assert h == h_unset
    np.testing.assert_array_equal(v, v_unset)


def test_on_reads_back_and_overrides_the_organ(monkeypatch):
    """The organ explicitly sets bounded False: the switch still runs all three, and the step actually sub-steps."""
    b, h_on, raster, v = _run_ref(monkeypatch, "1", organ_bounded=False)
    cfg = b.core_config
    assert b.stable_integration_requested is True
    assert b.stable_integration_applied is True
    assert b.stable_integration_scope == "izhikevich_full"
    assert b.stable_integration_effective == {"enable_bounded_conductance_integration": True,
                                              "izh_membrane_substeps": 4,
                                              "izh_substep_refractory_rule": "model_reset"}
    assert (cfg.enable_bounded_conductance_integration, cfg.izh_membrane_substeps,
            cfg.izh_substep_refractory_rule) == (True, 4, "model_reset")
    assert b.stable_integration_organ_values["enable_bounded_conductance_integration"] is False
    assert b.stable_integration_organ_values["izh_membrane_substeps"] == 1
    assert b.substep_calls == 60                      # every step took the sub-step path
    assert int(raster.sum()) > 0 and np.all(np.isfinite(v))
    _, h_off, _, _ = _run_ref(monkeypatch, None)
    assert h_on != h_off                              # the switch reached the engine


def test_on_reasserts_after_the_organ_changes_fields_post_build(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    b = _build_reference_network_bridge(seed=42, set_flag_explicitly=None)
    cfg = b.core_config
    cfg.enable_bounded_conductance_integration = False      # an organ turning it off after building the bridge
    cfg.izh_membrane_substeps = 1
    b._run_one_simulation_step()
    assert (cfg.enable_bounded_conductance_integration, cfg.izh_membrane_substeps) == (True, 4)
    assert b.stable_integration_reasserts == 1
    assert b.substep_calls == 1
    b._run_one_simulation_step()
    assert b.stable_integration_reasserts == 1              # nothing to change on the next step


def test_on_megakernel_still_declines(monkeypatch):
    """With every other megakernel guard satisfied and the organ asking for neither option, the re-asserted switch
    makes the dispatch fall through to the Python path."""
    monkeypatch.setenv(ENV, "1")
    b = _build_reference_network_bridge(seed=42, set_flag_explicitly=None)
    cfg = b.core_config
    monkeypatch.setattr(SB, "is_gpu_backend", lambda: True)
    cfg.enable_step_megakernel = True
    cfg.fast_spike_reset = True
    cfg.read_only_fast_step = True
    cfg.enable_bounded_conductance_integration = False
    cfg.izh_membrane_substeps = 1
    b._apply_stable_integration(per_step=True)
    assert b._step_megakernel_can_dispatch() is False


def test_model_scoping(monkeypatch):
    """AdEx gets the bounded integration only; Hodgkin-Huxley is left exactly as configured."""
    monkeypatch.setenv(ENV, "1")
    cfg = CoreSimConfig()
    cfg.neuron_model_type = NeuronModel.ADEX.name
    b = SimulationBridge(core_config=cfg)
    assert b.stable_integration_scope == "adex_bounded_only" and b.stable_integration_applied is False
    assert (cfg.enable_bounded_conductance_integration, cfg.izh_membrane_substeps) == (True, 1)
    cfg2 = CoreSimConfig()
    cfg2.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
    b2 = SimulationBridge(core_config=cfg2)
    assert b2.stable_integration_scope == "not_applicable:HODGKIN_HUXLEY"
    assert (cfg2.enable_bounded_conductance_integration, cfg2.izh_membrane_substeps) == (False, 1)
    # a config built Izhikevich, then switched to HH before init: the sub-steps the constructor set are taken back
    cfg3 = CoreSimConfig()
    cfg3.neuron_model_type = NeuronModel.IZHIKEVICH.name
    b3 = SimulationBridge(core_config=cfg3)
    assert cfg3.izh_membrane_substeps == 4
    cfg3.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
    b3._apply_stable_integration()
    assert (cfg3.enable_bounded_conductance_integration, cfg3.izh_membrane_substeps) == (False, 1)
    assert b3.stable_integration_applied is False


@pytest.mark.parametrize("bad", ["yes", "2", "true"])
def test_bad_value_raises(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError):
        SimulationBridge(core_config=CoreSimConfig())
