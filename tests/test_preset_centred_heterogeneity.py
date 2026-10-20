"""`CoreSimConfig.preset_centred_parameter_heterogeneity` (default OFF) and the always-on overwrite warning.

The defect (docs/drafts/preset-centred-het/CENSUS.md): with parameter heterogeneity on, the a/b/d/C draws are centred
on the GLOBAL cfg.izh_*_val (RS defaults) and overwrite every region's Izhikevich preset (HH: the C_m/g_Na/g_K/g_L
overrides), so an STN / GPe / MSN / FS region runs as an RS-centred hybrid. The flag recentres each draw on the
neuron's own preset with the same samples (same RNG consumption) and the same relative spread.

What is pinned here:
  * flag OFF is byte-identical to the pre-edit bridge (hashes captured on the unedited code, committed before the
    edit: tests/_capture_preset_centred_het_golden.py -> tests/_preset_centred_het_golden.json);
  * flag ON: per neuron, value / preset == (flag-OFF value) / (distribution centre), so region means sit on the
    preset and region CVs equal today's; neurons without a region preset, the thresholds and the RNG state after the
    build are identical to flag OFF; the same holds on the region-scoped redraw, the per-region mask path and HH;
  * the warning fires once per build, naming exactly the overwritten regions, when the flag is OFF, and never when ON.
"""
from __future__ import annotations

import json
import math
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import CoreSimConfig, SimulationBridge  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from sim.enums import DefaultIzhikevichParamsManager, NeuronType  # noqa: E402
from tests._capture_preset_centred_het_golden import (  # noqa: E402
    CASES, GOLDEN_PATH, HH_ARRAYS, IZH_ARRAYS, build_hh, build_selector, masked_stn_region, rng_state_hash,
    snapshot, untyped_region,
)

FLAG = "preset_centred_parameter_heterogeneity"
ON = {FLAG: True}
WARN_MARKER = "overwrites the presets of"

# array -> (heterogeneity key, preset field)
IZH = {
    "cp_izh_a": ("izh_a_val", "a"),
    "cp_izh_b": ("izh_b_val", "b"),
    "cp_izh_d_increment": ("izh_d_val", "d_increment"),
    "cp_izh_C": ("izh_C_val", "C"),
}
HH = {
    "cp_hh_C_m": ("hh_C_m", "hh_C_m_override"),
    "cp_hh_g_Na_max": ("hh_g_Na_max", "hh_g_Na_max_override"),
    "cp_hh_g_K_max": ("hh_g_K_max", "hh_g_K_max_override"),
    "cp_hh_g_L": ("hh_g_L", "hh_g_L_override"),
}


def _host(a):
    return np.asarray(to_host(a)).astype(np.float64)


def _idx(bridge, region):
    return np.asarray(sorted(int(i) for i in bridge.region_manager.indices(region)), dtype=np.int64)


def _centre(bridge, key):
    return SimulationBridge._het_distribution_centre(bridge.core_config.heterogeneity_distributions[key])


def _izh_preset(type_name):
    return DefaultIzhikevichParamsManager.PARAMS[NeuronType[type_name]]


def _typed_regions(bridge):
    return [r for r in bridge.core_config.brain_regions if r.izh_neuron_type]


def _build_pair(builder):
    """(flag OFF bridge + RNG hash, flag ON bridge + RNG hash), each RNG hash taken right after its own build."""
    off = builder(None)
    off_rng = rng_state_hash()
    on = builder(ON)
    on_rng = rng_state_hash()
    return off, off_rng, on, on_rng


@pytest.fixture(scope="module")
def selector_pair():
    return _build_pair(lambda u: build_selector(9, updates=u))


@pytest.fixture(scope="module")
def selector_untyped_pair():
    return _build_pair(lambda u: build_selector(12, extra_regions=[untyped_region()], updates=u))


# ── flag OFF: byte-identical to the pre-edit bridge ──────────────────────────────────────────────────────────────

def test_flag_defaults_off():
    assert CoreSimConfig().preset_centred_parameter_heterogeneity is False


@pytest.mark.parametrize("case", sorted(CASES))
def test_flag_off_byte_identical_to_pre_edit_golden(case):
    golden = json.load(open(GOLDEN_PATH))
    _xp, backend = get_backend()
    if backend != golden["backend"]:
        pytest.skip(f"golden captured on {golden['backend']}, this run is {backend}")
    builder, arrays = CASES[case]
    assert snapshot(builder(), arrays) == golden["cases"][case]


# ── flag ON: centred on the preset, same draws, same spread ──────────────────────────────────────────────────────

def test_flag_on_centres_every_typed_region_on_its_preset(selector_pair):
    off, _r0, on, _r1 = selector_pair
    regions = _typed_regions(on)
    assert {r.izh_neuron_type for r in regions} >= {
        "IZH2007_STN_BURST", "IZH2007_GPE_PACEMAKER", "IZH2007_GPI_OUTPUT", "IZH2007_STRIATAL_MSN_D1",
        "IZH2007_STRIATAL_MSN_D2", "IZH2007_FS_CORTICAL_INTERNEURON", "IZH2007_THALAMIC_RELAY"}
    for array, (key, field) in IZH.items():
        centre = _centre(off, key)
        v_off, v_on = _host(getattr(off, array)), _host(getattr(on, array))
        for region in regions:
            idx = _idx(on, region.name)
            preset = float(_izh_preset(region.izh_neuron_type)[field])
            # Same sample, recentred: value / preset == today's value / today's centre, neuron by neuron.
            np.testing.assert_allclose(v_on[idx] / preset, v_off[idx] / centre, rtol=1e-5,
                                       err_msg=f"{region.name} {array}")
            # Region mean sits on the preset (lognormal a: the mean of exp(0.3 z) is ~1.046 x the median).
            assert abs(v_on[idx].mean() / preset - 1.0) < 0.25, (region.name, array, v_on[idx].mean(), preset)
            # Relative spread equals today's.
            cv_on = v_on[idx].std() / abs(v_on[idx].mean())
            cv_off = v_off[idx].std() / abs(v_off[idx].mean())
            assert abs(cv_on - cv_off) < 1e-3, (region.name, array, cv_on, cv_off)


def test_flag_on_stn_reads_its_own_preset_not_rs(selector_pair):
    off, _r0, on, _r1 = selector_pair
    idx = _idx(on, "selector_stn")
    a_on, a_off = _host(on.cp_izh_a)[idx].mean(), _host(off.cp_izh_a)[idx].mean()
    b_on, b_off = _host(on.cp_izh_b)[idx].mean(), _host(off.cp_izh_b)[idx].mean()
    assert 0.004 < a_on < 0.0065 and a_off > 0.02           # STN a 0.005 kept; flag OFF reads ~0.03 (RS)
    assert -1.3 < b_on < -0.7 and b_off < -1.5              # STN b -1 kept; flag OFF reads ~-2 (RS)
    gpe = _idx(on, "gpe_0")
    assert _host(on.cp_izh_b)[gpe].min() > 0.0               # GPe b +1: the sign survives (flag OFF flips it)
    assert _host(off.cp_izh_b)[gpe].max() < 0.0


def test_flag_on_leaves_neurons_without_a_region_preset_identical(selector_untyped_pair):
    off, _r0, on, _r1 = selector_untyped_pair
    idx = _idx(on, "untyped_pool")
    assert idx.size == 24
    for array in IZH:
        np.testing.assert_array_equal(to_host(getattr(on, array))[idx], to_host(getattr(off, array))[idx])


@pytest.mark.parametrize("pair", ["selector_pair", "selector_untyped_pair"])
def test_flag_on_draws_the_same_samples(pair, request):
    """Same RNG consumption: thresholds and the global RNG state after the build are identical to flag OFF."""
    off, off_rng, on, on_rng = request.getfixturevalue(pair)
    np.testing.assert_array_equal(to_host(on.cp_neuron_firing_thresholds), to_host(off.cp_neuron_firing_thresholds))
    assert on_rng == off_rng


def test_flag_on_region_scoped_redraw_is_centred_too():
    off = build_selector(14, updates={"per_region_parameter_heterogeneity": True})
    on = build_selector(14, updates={"per_region_parameter_heterogeneity": True, **ON})
    for array, (key, field) in IZH.items():
        centre = _centre(off, key)
        v_off, v_on = _host(getattr(off, array)), _host(getattr(on, array))
        for region in _typed_regions(on):
            idx = _idx(on, region.name)
            preset = float(_izh_preset(region.izh_neuron_type)[field])
            np.testing.assert_allclose(v_on[idx] / preset, v_off[idx] / centre, rtol=1e-5,
                                       err_msg=f"{region.name} {array}")


def test_flag_on_per_region_mask_path_is_centred_and_unmasked_presets_stay_exact():
    updates = {"enable_parameter_heterogeneity": False}
    off = build_selector(9, extra_regions=[masked_stn_region()], updates=updates)
    on = build_selector(9, extra_regions=[masked_stn_region()], updates={**updates, **ON})
    masked = _idx(on, "het_masked_stn")
    stn = _izh_preset("IZH2007_STN_BURST")
    for array, (key, field) in IZH.items():
        centre = _centre(off, key)
        v_off, v_on = _host(getattr(off, array)), _host(getattr(on, array))
        np.testing.assert_allclose(v_on[masked] / float(stn[field]), v_off[masked] / centre, rtol=1e-5)
        assert v_on[masked].std() > 0.0                                    # jittered
        gpe = _idx(on, "gpe_0")                                            # unmasked: deterministic preset
        expected = np.float32(_izh_preset("IZH2007_GPE_PACEMAKER")[field])
        assert (to_host(getattr(on, array))[gpe] == expected).all()
        assert (to_host(getattr(off, array))[gpe] == expected).all()


def test_flag_on_hh_overrides_are_centred_and_plain_region_identical():
    off = build_hh(1741)
    on = build_hh(1741, updates=ON)
    override = on.core_config.brain_regions[0]
    idx, plain = _idx(on, "hh_override"), _idx(on, "hh_plain")
    for array, (key, field) in HH.items():
        centre = _centre(off, key)
        preset = float(getattr(override, field))
        v_off, v_on = _host(getattr(off, array)), _host(getattr(on, array))
        np.testing.assert_allclose(v_on[idx] / preset, v_off[idx] / centre, rtol=1e-5, err_msg=array)
        np.testing.assert_array_equal(to_host(getattr(on, array))[plain], to_host(getattr(off, array))[plain])


def test_recentre_rule_edge_cases():
    recentre = SimulationBridge._recentre_het_draw_host
    has = np.array([True, True, False])
    # gaussian, negative mean: ratio = sample / mean, kept in [0.1, 3.0]; a zero preset gets no jitter.
    out = recentre(np.array([-3.0, -100.0, -7.0], dtype=np.float32), {"type": "gaussian", "mean": -2.0, "std": 0.5},
                   has, np.array([-1.0, 0.0, 99.0]))
    np.testing.assert_allclose(out, [-1.5, 0.0, -7.0])
    # gaussian with a zero mean has no ratio: the sample shifts the preset additively.
    out = recentre(np.array([0.5, -0.25, 3.0], dtype=np.float32), {"type": "gaussian", "mean": 0.0, "std": 1.0},
                   has, np.array([10.0, 2.0, 0.0]))
    np.testing.assert_allclose(out, [10.5, 1.75, 3.0])
    # lognormal: ratio = sample / exp(mean_log), no clip.
    out = recentre(np.array([0.06, 0.003, 0.03], dtype=np.float32),
                   {"type": "lognormal", "mean_log": math.log(0.03), "sigma_log": 0.3},
                   has, np.array([0.005, 0.2, 1.0]))
    np.testing.assert_allclose(out, [0.01, 0.02, 0.03], rtol=1e-6)


# ── the loud check ───────────────────────────────────────────────────────────────────────────────────────────────

def _warnings(text):
    return [line for line in text.splitlines() if WARN_MARKER in line]


def test_warning_fires_once_naming_the_overwritten_regions_when_off(capsys):
    capsys.readouterr()
    build_selector(9)
    lines = _warnings(capsys.readouterr().out)
    assert len(lines) == 1, lines
    line = lines[0]
    assert "[WARNING]" in line and f"{FLAG}=True" in line
    assert "selector_stn [IZH2007_STN_BURST] (a 0.005->0.03, b -1->-2, d 75->100, C 80->100)" in line
    for name in ("selector_reset", "str_fsi_0", "commit_fs_1", "str_d1_0", "str_d2_1", "gpe_0", "gpi_1", "thal_0"):
        assert name in line, name
    for rs_region in ("practice_arousal", "proposal_0", "motor_1", "commit_0,", "commit_0 "):
        assert rs_region not in line, rs_region          # RS preset == the global centre: not overwritten


def test_warning_names_only_the_masked_region_when_global_heterogeneity_is_off(capsys):
    capsys.readouterr()
    build_selector(9, extra_regions=[masked_stn_region()], updates={"enable_parameter_heterogeneity": False})
    lines = _warnings(capsys.readouterr().out)
    assert len(lines) == 1, lines
    assert "of 1 region(s): het_masked_stn [IZH2007_STN_BURST]" in lines[0]
    assert "gpe_0" not in lines[0]


def test_warning_covers_hh_overrides(capsys):
    capsys.readouterr()
    build_hh(1741)
    lines = _warnings(capsys.readouterr().out)
    assert len(lines) == 1 and "hh_override [HH overrides]" in lines[0] and "g_Na_max 88->50" in lines[0]
    assert "hh_plain" not in lines[0]


@pytest.mark.parametrize("updates", [ON, {"enable_parameter_heterogeneity": False}])
def test_warning_absent_when_flag_on_or_heterogeneity_off(updates, capsys):
    capsys.readouterr()
    build_selector(9, updates=updates)
    assert _warnings(capsys.readouterr().out) == []
