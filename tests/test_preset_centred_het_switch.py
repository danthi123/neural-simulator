"""SIM_PRESET_CENTRED_HET (sim/bridge.py PRESET_CENTRED_HET_ENV) -- the whole-brain, DEFAULT-OFF switch for
`CoreSimConfig.preset_centred_parameter_heterogeneity`. When "1", every SimulationBridge's core config gets the flag
forced True (at construction and at every `_initialize_simulation_data`), whatever the organ configured, so each
region-preset neuron's a/b/d/C jitter is centred on its own preset. Unset / "" / "0": nothing is touched and the
build is byte-identical to the pre-switch engine (the flag's own pre-edit golden). The switch reaches ONLY neurons
that belong to a region with a preset; trait-typed neurons and untyped regions keep the global-centred draw by the
flag's own rule, and that limit is pinned here (it is why the switch cannot fix a trait-typed population such as the
LearnedAssocGraph pool / fs cells).
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sim.bridge as SB  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from tests._capture_preset_centred_het_golden import (  # noqa: E402
    CASES, GOLDEN_PATH, IZH_ARRAYS, build_selector, snapshot, untyped_region)

ENV = SB.PRESET_CENTRED_HET_ENV
FLAG = "preset_centred_parameter_heterogeneity"


def _set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


def _host(a):
    return np.asarray(to_host(a)).astype(np.float64)


@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
@pytest.mark.parametrize("case", sorted(CASES))
def test_off_spellings_are_byte_identical_to_the_pre_edit_golden(monkeypatch, value, case):
    _set_env(monkeypatch, value)
    golden = json.load(open(GOLDEN_PATH))
    _xp, backend = get_backend()
    if backend != golden["backend"]:
        pytest.skip(f"golden captured on {golden['backend']}, this run is {backend}")
    builder, arrays = CASES[case]
    assert snapshot(builder(), arrays) == golden["cases"][case]


@pytest.mark.parametrize("value", [None, "", "0"])
def test_off_reads_back_untouched(monkeypatch, value):
    _set_env(monkeypatch, value)
    b = build_selector(9)
    assert b.preset_centred_het_requested is False
    assert b.preset_centred_het_applied is False
    assert b.preset_centred_het_organ_value is None
    assert getattr(b.core_config, FLAG) is False
    rb = b.preset_centred_het_readback
    assert rb["heterogeneity_ran"] is True and rb["preset_centred"] is False
    assert rb["n_with_region_preset"] == 0 and rb["by_param"] == {}


def test_on_forces_the_flag_and_reads_back(monkeypatch):
    _set_env(monkeypatch, "1")
    b = build_selector(9)
    assert b.preset_centred_het_requested is True
    assert b.preset_centred_het_applied is True
    assert b.preset_centred_het_organ_value is False           # the organ never turned it on
    assert getattr(b.core_config, FLAG) is True
    rb = b.preset_centred_het_readback
    assert rb["preset_centred"] is True and rb["n_neurons"] == b.core_config.num_neurons
    assert 0 < rb["n_preset_off_centre"] <= rb["n_with_region_preset"] <= rb["n_neurons"]
    assert set(rb["by_param"]) == {"izh_a_val", "izh_b_val", "izh_d_val", "izh_C_val"}


def test_on_overrides_an_organ_that_set_the_flag_false(monkeypatch):
    _set_env(monkeypatch, "1")
    b = build_selector(9, updates={FLAG: False})
    assert getattr(b.core_config, FLAG) is True
    assert b.preset_centred_het_organ_value is False and b.preset_centred_het_readback["preset_centred"] is True


def test_on_equals_the_config_flag_and_differs_from_off(monkeypatch):
    """Env ON with the config at default builds exactly what the config flag builds with the env unset."""
    _set_env(monkeypatch, None)
    off = build_selector(9)
    cfg_on = build_selector(9, updates={FLAG: True})
    _set_env(monkeypatch, "1")
    env_on = build_selector(9)
    for name in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(getattr(env_on, name)), _host(getattr(cfg_on, name)))
    assert any(not np.array_equal(_host(getattr(env_on, n)), _host(getattr(off, n))) for n in IZH_ARRAYS)


def test_on_reasserts_at_re_initialization(monkeypatch):
    """An organ (or a config reload) that clears the flag between builds cannot undo the switch: the next
    _initialize_simulation_data sets it again and the draws are centred."""
    _set_env(monkeypatch, "1")
    b = build_selector(9)
    first = {n: _host(getattr(b, n)).copy() for n in IZH_ARRAYS}
    setattr(b.core_config, FLAG, False)
    b._initialize_simulation_data(called_from_playback_init=False)
    assert getattr(b.core_config, FLAG) is True
    assert b.preset_centred_het_readback["preset_centred"] is True
    # the re-initialised draw is centred again: region means stay on the presets (not on the RS centres)
    stn = np.asarray(sorted(int(i) for i in b.region_manager.indices("selector_stn")), dtype=np.int64)
    assert _host(b.cp_izh_b)[stn].mean() > -1.5, "STN b is centred on its preset -1, not on the RS -2"
    assert first["cp_izh_b"].shape == _host(b.cp_izh_b).shape


def test_a_bad_value_raises_at_construction(monkeypatch):
    _set_env(monkeypatch, "yes")
    with pytest.raises(ValueError, match=ENV):
        build_selector(9)


def test_untyped_region_is_not_reached(monkeypatch):
    """A region with no preset keeps the global-centred draw with the switch ON (identical to OFF)."""
    _set_env(monkeypatch, None)
    off = build_selector(12, extra_regions=[untyped_region()])
    _set_env(monkeypatch, "1")
    on = build_selector(12, extra_regions=[untyped_region()])
    idx = np.asarray(sorted(int(i) for i in on.region_manager.indices("untyped_pool")), dtype=np.int64)
    assert idx.size > 0
    for name in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(getattr(on, name))[idx], _host(getattr(off, name))[idx])


def test_trait_typed_population_is_not_reached(monkeypatch):
    """The limit that matters for the stable-integration offenders: LearnedAssocGraph builds its bridge with
    research.runners._D_sparse_heteroassoc.build (two regions, no izh_neuron_type, cfg.num_traits default > 1), so its
    cells are typed by trait and carry the global-centred a/b/d/C draw. The switch changes none of them."""
    from research.runners._D_sparse_heteroassoc import build
    _set_env(monkeypatch, None)
    off = build(7, n_pool=60, n_fs=24)
    _set_env(monkeypatch, "1")
    on = build(7, n_pool=60, n_fs=24)
    assert on.preset_centred_het_applied is True and on.preset_centred_het_readback["preset_centred"] is True
    assert on.preset_centred_het_readback["n_with_region_preset"] == 0
    for name in IZH_ARRAYS + ("cp_izh_k", "cp_izh_vr", "cp_izh_vt", "cp_izh_c_reset"):
        np.testing.assert_array_equal(_host(getattr(on, name)), _host(getattr(off, name)))
