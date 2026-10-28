"""CoreSimConfig.trait_centred_parameter_heterogeneity / SIM_TRAIT_CENTRED_HET (sim/bridge.py TRAIT_CENTRED_HET_ENV) --
parameter heterogeneity centred on the preset of each TRAIT-typed neuron, DEFAULT OFF.

With cfg.num_traits > 1 the Izhikevich init gives every neuron one of the defined IZH2007 presets by trait; a neuron in
no region with an izh_neuron_type keeps that trait preset, and the global draw then overwrites its a/b/d/C with
RS-centred values (docs/drafts/preset-centred-het/CENSUS.md, C2.1: the region flag leaves these neurons alone on
purpose). The variant draws the same samples in the same order (RNG consumption unchanged) and turns each into a
ratio against its distribution's centre times the neuron's OWN trait preset, for the neurons no region preset covers.

Pinned here: unset / "" / "0" / flag False is byte-identical to the pre-edit engine (golden captured on the unedited
bridge, tests/_trait_centred_het_golden.json); the variant equals the engine's own recentring rule applied to the OFF
bridge's draw; a trait on the distribution's centre (RS) is bit-identical; region-typed neurons are never touched by
it and trait neurons never by the region flag (disjoint sets, both on = union); single-type pools, HH pools and
re-initialisation behave as stated. Numpy CPU.
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
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import DefaultIzhikevichParamsManager, NeuronType  # noqa: E402
from tests._capture_preset_centred_het_golden import build_hh, rng_state_hash  # noqa: E402
from tests._capture_trait_centred_het_golden import (  # noqa: E402
    ARRAYS, CASES, FIXED_ARRAYS, GOLDEN_PATH, build_lag, build_pool, masked_untyped, mixed_regions)
from tests._capture_preset_centred_het_golden import IZH_ARRAYS, snapshot  # noqa: E402

ENV = SB.TRAIT_CENTRED_HET_ENV
FLAG = "trait_centred_parameter_heterogeneity"
REGION_FLAG = "preset_centred_parameter_heterogeneity"
KEYS = {"cp_izh_a": ("izh_a_val", "a"), "cp_izh_b": ("izh_b_val", "b"),
        "cp_izh_d_increment": ("izh_d_val", "d_increment"), "cp_izh_C": ("izh_C_val", "C")}
RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL


def _set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


def _host(bridge, name):
    return np.asarray(to_host(getattr(bridge, name))).astype(np.float64)


def _presets(bridge):
    """Per neuron, the a/b/d/C of the trait preset the init gave it (the viz label written at trait assignment; stale for
    a typed region, which is why callers only use it for neurons outside every typed region)."""
    labels = bridge.runtime_state.neuron_types_list_for_viz
    out = {k: np.zeros(len(labels)) for k in KEYS}
    for i, lab in enumerate(labels):
        prm = DefaultIzhikevichParamsManager.PARAMS[NeuronType["IZH2007_" + lab.replace("Izh2007_", "")]]
        for k, (_hk, f) in KEYS.items():
            out[k][i] = float(np.float32(prm[f]))       # the float32 value the init stored in the bridge's array
    return out, np.array(labels)


def _region_idx(bridge, name):
    return np.asarray(sorted(int(i) for i in bridge.region_manager.indices(name)), dtype=np.int64)


def _expected_from_off(off, trait_mask):
    """The engine's own recentring rule (`_recentre_het_draw_host`) applied to the OFF bridge's draw for the trait neurons
    whose trait preset is off the distribution centre; everything else is the OFF value. Same samples => same result."""
    presets, _labels = _presets(off)
    dists = off.core_config.heterogeneity_distributions
    out = {}
    for name, (hk, _f) in KEYS.items():
        draw = _host(off, name)
        centre = SB.SimulationBridge._het_distribution_centre(dists[hk])
        has = trait_mask & (np.abs(presets[name] - centre) > 1e-6 * abs(centre))
        out[name] = SB.SimulationBridge._recentre_het_draw_host(draw, dists[hk], has, presets[name]).astype(np.float64)
    return out


def test_flag_defaults_off():
    assert CoreSimConfig().trait_centred_parameter_heterogeneity is False


@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
@pytest.mark.parametrize("case", sorted(CASES))
def test_unset_is_byte_identical_to_the_pre_edit_golden(monkeypatch, value, case):
    """Env unset / "" / "0": every array, the thresholds and the global RNG state after the build hash to the golden
    captured on the unedited bridge, for trait-typed pools, mixed pools, the region-scoped and masked paths and a
    single-type pool."""
    _set_env(monkeypatch, value)
    golden = json.load(open(GOLDEN_PATH))
    _xp, backend = get_backend()
    if backend != golden["backend"]:
        pytest.skip(f"golden captured on {golden['backend']}, this run is {backend}")
    assert snapshot(CASES[case](), ARRAYS) == golden["cases"][case]


def test_config_flag_false_with_env_on_elsewhere_is_unchanged(monkeypatch):
    """The config flag at its default, env unset, is the golden path; setting the flag False explicitly is the same."""
    _set_env(monkeypatch, None)
    golden = json.load(open(GOLDEN_PATH))
    _xp, backend = get_backend()
    if backend != golden["backend"]:
        pytest.skip("golden backend differs")
    snap = snapshot(build_pool(5, mixed_regions(), updates={FLAG: False}), ARRAYS)
    assert snap == golden["cases"]["mixed_typed_untyped_s5"]


@pytest.mark.parametrize("value", [None, "", "0"])
def test_off_reads_back_untouched(monkeypatch, value):
    _set_env(monkeypatch, value)
    b = build_lag(7, 60, 24)
    assert b.trait_centred_het_requested is False and b.trait_centred_het_applied is False
    assert b.trait_centred_het_organ_value is None
    assert getattr(b.core_config, FLAG) is False
    assert b.trait_centred_het_readback == {"heterogeneity_ran": True, "trait_centred": False}


def test_on_forces_the_flag_and_reads_back(monkeypatch):
    _set_env(monkeypatch, "1")
    b = build_lag(7, 60, 24)
    assert b.trait_centred_het_requested is True and b.trait_centred_het_applied is True
    assert b.trait_centred_het_organ_value is False           # the organ never turned it on
    assert getattr(b.core_config, FLAG) is True
    rb = b.trait_centred_het_readback
    assert rb["trait_centred"] is True and rb["trait_typed"] is True and rb["n_neurons"] == 84
    assert rb["n_trait_neurons"] == 84                        # no region carries a preset
    assert 0 < rb["n_trait_off_centre"] < rb["n_trait_neurons"]   # the RS trait is on the centre and is not counted
    assert set(rb["by_param"]) == {"izh_a_val", "izh_b_val", "izh_d_val", "izh_C_val"}


def test_a_bad_value_raises_at_construction(monkeypatch):
    _set_env(monkeypatch, "yes")
    with pytest.raises(ValueError, match=ENV):
        build_lag(7, 60, 24)


def test_on_overrides_an_organ_that_set_the_flag_false(monkeypatch):
    _set_env(monkeypatch, "1")
    b = build_pool(5, mixed_regions(), updates={FLAG: False})
    assert getattr(b.core_config, FLAG) is True and b.trait_centred_het_organ_value is False


def test_env_on_equals_the_config_flag_and_differs_from_off(monkeypatch):
    _set_env(monkeypatch, None)
    off = build_lag(7, 120, 30)
    _set_env(monkeypatch, "1")
    env_on = build_lag(7, 120, 30)
    assert any(not np.array_equal(_host(env_on, n), _host(off, n)) for n in IZH_ARRAYS)
    # and the config flag with the env unset builds the same thing as the env switch
    _set_env(monkeypatch, None)
    b_off = build_pool(5, mixed_regions())
    b_cfg = build_pool(5, mixed_regions(), updates={FLAG: True})
    _set_env(monkeypatch, "1")
    b_env = build_pool(5, mixed_regions())
    for n in ARRAYS:
        np.testing.assert_array_equal(_host(b_env, n), _host(b_cfg, n))
    assert any(not np.array_equal(_host(b_env, n), _host(b_off, n)) for n in IZH_ARRAYS)


def test_same_samples_same_rng_and_only_the_four_jittered_arrays_move(monkeypatch):
    """Same draws in the same order: the global RNG state and the thresholds after the build, and every never-jittered
    array, are identical with the variant on and off; RS-trait neurons keep a, b, d, C bit for bit."""
    _set_env(monkeypatch, None)
    off = build_lag(7, 300, 60)
    off_rng = rng_state_hash()
    _set_env(monkeypatch, "1")
    on = build_lag(7, 300, 60)
    assert rng_state_hash() == off_rng
    np.testing.assert_array_equal(_host(on, "cp_neuron_firing_thresholds"), _host(off, "cp_neuron_firing_thresholds"))
    for n in FIXED_ARRAYS:
        np.testing.assert_array_equal(_host(on, n), _host(off, n))
    _p, labels = _presets(on)
    rs = labels == "Izh2007_RS_CORTICAL_PYRAMIDAL"
    assert rs.sum() > 20
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on, n)[rs], _host(off, n)[rs])
        assert np.any(_host(on, n)[~rs] != _host(off, n)[~rs])


def test_each_trait_is_centred_on_its_own_preset_by_the_engines_rule(monkeypatch):
    """ON equals the engine's own recentring rule applied to the OFF bridge's draw, neuron for neuron (to the float32
    cast), and each trait's mean lands on its own preset (the thalamic relay b is positive again, FS C is about 20)."""
    _set_env(monkeypatch, None)
    off = build_lag(7, 300, 60)
    _set_env(monkeypatch, "1")
    on = build_lag(7, 300, 60)
    expected = _expected_from_off(off, np.ones(on.core_config.num_neurons, dtype=bool))
    for n in IZH_ARRAYS:
        np.testing.assert_allclose(_host(on, n), expected[n].astype(np.float32).astype(np.float64), rtol=0, atol=0)
    presets, labels = _presets(on)
    for lab in sorted(set(labels)):
        m = labels == lab
        for n in IZH_ARRAYS:
            p = presets[n][m][0]
            if p != 0:
                assert abs(_host(on, n)[m].mean() / p - 1.0) < 0.15, (lab, n)
    relay = labels == "Izh2007_THALAMIC_RELAY"
    assert _host(off, "cp_izh_b")[relay].mean() < 0 < _host(on, "cp_izh_b")[relay].mean()


def test_region_typed_neurons_are_not_touched_by_the_trait_variant(monkeypatch):
    """Next to a typed STN region: the variant leaves the STN neurons exactly as OFF (RS-centred overwrite of the
    region preset, as today) and recentres only the untyped region's trait-typed neurons."""
    _set_env(monkeypatch, None)
    off = build_pool(5, mixed_regions())
    on_cfg = build_pool(5, mixed_regions(), updates={FLAG: True})
    stn, pool = _region_idx(off, "typed_stn"), _region_idx(off, "untyped_pool")
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on_cfg, n)[stn], _host(off, n)[stn])
    assert any(np.any(_host(on_cfg, n)[pool] != _host(off, n)[pool]) for n in IZH_ARRAYS)
    rb = on_cfg.trait_centred_het_readback
    assert rb["n_trait_neurons"] == pool.size and rb["n_neurons"] == stn.size + pool.size
    mask = np.zeros(on_cfg.core_config.num_neurons, dtype=bool)
    mask[pool] = True
    expected = _expected_from_off(off, mask)
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on_cfg, n), expected[n].astype(np.float32).astype(np.float64))


def test_the_two_flags_act_on_disjoint_neuron_sets(monkeypatch):
    """Region flag alone leaves trait neurons as OFF (the flag's own rule, untouched); trait flag alone leaves region
    neurons as OFF; both on = the region flag's result on the region neurons and the trait flag's on the trait neurons."""
    _set_env(monkeypatch, None)
    off = build_pool(5, mixed_regions())
    reg = build_pool(5, mixed_regions(), updates={REGION_FLAG: True})
    trt = build_pool(5, mixed_regions(), updates={FLAG: True})
    both = build_pool(5, mixed_regions(), updates={REGION_FLAG: True, FLAG: True})
    stn, pool = _region_idx(off, "typed_stn"), _region_idx(off, "untyped_pool")
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(reg, n)[pool], _host(off, n)[pool])      # region flag: trait neurons as OFF
        np.testing.assert_array_equal(_host(trt, n)[stn], _host(off, n)[stn])        # trait flag: region neurons as OFF
        np.testing.assert_array_equal(_host(both, n)[stn], _host(reg, n)[stn])
        np.testing.assert_array_equal(_host(both, n)[pool], _host(trt, n)[pool])
    assert any(np.any(_host(reg, n)[stn] != _host(off, n)[stn]) for n in IZH_ARRAYS)
    assert both.preset_centred_het_readback["preset_centred"] is True
    assert both.trait_centred_het_readback["trait_centred"] is True
    # the region flag's own read-back is unchanged by the variant (region neurons only)
    assert both.preset_centred_het_readback == reg.preset_centred_het_readback


def test_region_scoped_redraw_path(monkeypatch):
    """per_region_parameter_heterogeneity redraws every region's slice from a region-scoped stream; the variant recentres
    the trait neurons of that draw too (same samples: ON equals the rule applied to the OFF region-scoped draw)."""
    _set_env(monkeypatch, None)
    u = {"per_region_parameter_heterogeneity": True}
    off = build_pool(8, mixed_regions(), updates=u)
    on = build_pool(8, mixed_regions(), updates={**u, FLAG: True})
    stn, pool = _region_idx(off, "typed_stn"), _region_idx(off, "untyped_pool")
    mask = np.zeros(on.core_config.num_neurons, dtype=bool)
    mask[pool] = True
    expected = _expected_from_off(off, mask)
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on, n)[stn], _host(off, n)[stn])
        np.testing.assert_array_equal(_host(on, n), expected[n].astype(np.float32).astype(np.float64))
    assert any(np.any(_host(on, n)[pool] != _host(off, n)[pool]) for n in IZH_ARRAYS)


def test_global_off_per_region_mask_path(monkeypatch):
    """Global heterogeneity OFF, an untyped region opted in through the mask: its trait neurons are recentred, the typed
    STN region keeps its deterministic preset exactly as OFF."""
    _set_env(monkeypatch, None)
    u = {"enable_parameter_heterogeneity": False}
    off = build_pool(6, masked_untyped(), updates=u)
    on = build_pool(6, masked_untyped(), updates={**u, FLAG: True})
    stn, pool = _region_idx(off, "typed_stn"), _region_idx(off, "masked_untyped")
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on, n)[stn], _host(off, n)[stn])
    mask = np.zeros(on.core_config.num_neurons, dtype=bool)
    mask[pool] = True
    expected = _expected_from_off(off, mask)
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on, n)[pool], expected[n].astype(np.float32).astype(np.float64)[pool])
    assert any(np.any(_host(on, n)[pool] != _host(off, n)[pool]) for n in IZH_ARRAYS)


def test_single_type_pool_is_not_reached(monkeypatch):
    """cfg.num_traits == 1: every neuron is the default type (the CENSUS C2.2 case, separate): the variant does nothing."""
    _set_env(monkeypatch, None)
    off = build_pool(4, n=120, updates={"num_traits": 1})
    _set_env(monkeypatch, "1")
    on = build_pool(4, n=120, updates={"num_traits": 1})
    assert on.trait_centred_het_applied is True
    assert on.trait_centred_het_readback == {"heterogeneity_ran": True, "trait_centred": False}
    for n in ARRAYS:
        np.testing.assert_array_equal(_host(on, n), _host(off, n))


def test_plain_pool_without_a_region_framework_is_recentred(monkeypatch):
    _set_env(monkeypatch, None)
    off = build_pool(3, n=150)
    _set_env(monkeypatch, "1")
    on = build_pool(3, n=150)
    assert on.trait_centred_het_readback["trait_typed"] is True
    assert on.trait_centred_het_readback["n_trait_neurons"] == 150
    mask = np.ones(150, dtype=bool)
    expected = _expected_from_off(off, mask)
    for n in IZH_ARRAYS:
        np.testing.assert_array_equal(_host(on, n), expected[n].astype(np.float32).astype(np.float64))


def test_hh_population_is_not_reached(monkeypatch):
    """The variant is Izhikevich-only: an HH pool builds identically with the flag on."""
    _set_env(monkeypatch, None)
    off = build_hh(1741)
    on = build_hh(1741, updates={FLAG: True})
    assert on.trait_centred_het_readback == {"heterogeneity_ran": True, "trait_centred": False}
    for n in ("cp_hh_C_m", "cp_hh_g_Na_max", "cp_hh_g_K_max", "cp_hh_g_L"):
        np.testing.assert_array_equal(_host(on, n), _host(off, n))


def test_on_reasserts_at_re_initialization(monkeypatch):
    _set_env(monkeypatch, "1")
    b = build_lag(7, 120, 30)
    first = {n: _host(b, n).copy() for n in IZH_ARRAYS}
    setattr(b.core_config, FLAG, False)
    b._initialize_simulation_data(called_from_playback_init=False)
    assert getattr(b.core_config, FLAG) is True and b.trait_centred_het_readback["trait_centred"] is True
    assert first["cp_izh_b"].shape == _host(b, "cp_izh_b").shape
