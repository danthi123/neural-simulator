"""Capture the flag-OFF byte-identity golden for `CoreSimConfig.preset_centred_parameter_heterogeneity`.

Run ONCE on the UNEDITED bridge (before the preset-centred heterogeneity edit). The regression test
`tests/test_preset_centred_heterogeneity.py::test_flag_off_byte_identical_to_pre_edit_golden` rebuilds every case
below on the edited bridge (flag at its default, OFF) and asserts each hash matches this golden EXACTLY: the four
jittered Izhikevich arrays (a, b, d, C) or the four jittered HH arrays (C_m, g_Na_max, g_K_max, g_L), the per-neuron
firing thresholds, and the global RNG state (numpy legacy + python `random`) after the build.

The cases cover every heterogeneity path the edit touches: the global draw (BG selector bridge, the affect-marker WTA
organ), a region with no preset, the opt-in region-scoped draw (per_region_parameter_heterogeneity), the global-OFF
per-region mask (BrainRegion.enable_heterogeneity=True) and the HH conductance draw with per-region HH overrides.

Usage:  SIM_BACKEND=numpy SIM_NO_PROVENANCE=1 python -u tests/_capture_preset_centred_het_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

GOLDEN_PATH = os.path.join(_HERE, "_preset_centred_het_golden.json")

IZH_ARRAYS = ("cp_izh_a", "cp_izh_b", "cp_izh_d_increment", "cp_izh_C")
HH_ARRAYS = ("cp_hh_C_m", "cp_hh_g_Na_max", "cp_hh_g_K_max", "cp_hh_g_L")


def _host(array):
    from sim.backend import to_host
    return np.ascontiguousarray(np.asarray(to_host(array)))


def _sha(array) -> str:
    return hashlib.sha256(_host(array).tobytes()).hexdigest()


def rng_state_hash() -> str:
    """Hash of the global RNG state a build leaves behind (numpy legacy MT19937 + python `random`)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(np.asarray(keys, dtype=np.uint32).tobytes())
    h.update(repr((int(pos), int(has_gauss), float(cached))).encode())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def snapshot(bridge, arrays=IZH_ARRAYS) -> dict:
    out = {name: _sha(getattr(bridge, name)) for name in arrays}
    thr = getattr(bridge, "cp_neuron_firing_thresholds", None)
    out["cp_neuron_firing_thresholds"] = None if thr is None else _sha(thr)
    out["rng_state"] = rng_state_hash()
    return out


def untyped_region(name="untyped_pool", n=24):
    """A region with NO per-region neuron type (izh_neuron_type None): its neurons keep the global centre."""
    from sim.regions import BrainRegion
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=1.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                       plastic_internal=False, izh_neuron_type=None, enable_homeostasis=False)


def masked_stn_region(name="het_masked_stn", n=24):
    """An STN-preset region that opts into heterogeneity while the global flag is OFF (the per-region mask path)."""
    from sim.regions import BrainRegion
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=1.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                       plastic_internal=False, izh_neuron_type="IZH2007_STN_BURST",
                       enable_homeostasis=False, enable_heterogeneity=True)


def build_selector(seed, *, extra_regions=(), updates=None):
    """The BG selector bridge (research/runners/_vocal_action_selector_gate.py), the production bg-action-selection
    organ's builder: STN / GPe / GPi / MSN-D1/D2 / FS / thalamic presets per region, global heterogeneity ON."""
    from research.runners._vocal_action_selector_gate import build_selector_bridge
    return build_selector_bridge(int(seed), extra_regions=tuple(extra_regions),
                                 core_config_updates=dict(updates or {}))


def build_affect_marker(seed=42):
    """The affect-marker WTA organ's private bridge (production affect-marker-spiking-wta): RS marker pools + FS
    cross-inhibition pools, global heterogeneity ON by default."""
    from research.runners import _affect_marker_wta_derisk as amw
    bridge, _m, _f = amw._build_bridge(int(seed), amw.N_VALENCE_POOLS, "aff_val")
    return bridge


def build_hh(seed=1741, *, updates=None):
    """A small HH bridge with global heterogeneity ON: one region with HH overrides, one without."""
    from sim import CoreSimConfig, GPUConfig, NeuronModel, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.regions import BrainRegion

    regions = [
        BrainRegion(name="hh_override", n_neurons=40, internal_density=0.0,
                    hh_C_m_override=1.4, hh_g_Na_max_override=88.0,
                    hh_g_K_max_override=24.0, hh_g_L_override=0.12),
        BrainRegion(name="hh_plain", n_neurons=30, internal_density=0.0),
    ]
    cfg = CoreSimConfig(
        num_neurons=sum(r.n_neurons for r in regions), connections_per_neuron=0, seed=int(seed),
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY", dt_ms=0.05,
        enable_brain_region_framework=True, brain_regions=regions, region_pathways=[],
        enable_parameter_heterogeneity=True, enable_ou_process=False, enable_hebbian_learning=False,
        enable_short_term_plasticity=False, enable_homeostasis=False, enable_stdp=False,
        enable_structural_plasticity=False, enable_reward_modulation=False,
    )
    for name, value in (updates or {}).items():
        if not hasattr(cfg, name):
            raise AttributeError(f"CoreSimConfig has no field {name!r}")
        setattr(cfg, name, value)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                              gpu_config=GPUConfig(enable_profiling=False))
    bridge._initialize_simulation_data()
    return bridge


# name -> (builder taking an `updates` dict, arrays hashed). Every case runs with the new flag at its default (OFF).
CASES = {
    "selector_s9": (lambda u=None: build_selector(9, updates=u), IZH_ARRAYS),
    "selector_s12_untyped": (lambda u=None: build_selector(12, extra_regions=[untyped_region()], updates=u),
                             IZH_ARRAYS),
    "selector_s14_region_scoped": (
        lambda u=None: build_selector(14, updates={"per_region_parameter_heterogeneity": True, **(u or {})}),
        IZH_ARRAYS),
    "selector_s9_global_off_mask": (
        lambda u=None: build_selector(9, extra_regions=[masked_stn_region()],
                                      updates={"enable_parameter_heterogeneity": False, **(u or {})}),
        IZH_ARRAYS),
    "affect_marker_val_s42": (lambda u=None: build_affect_marker(42), IZH_ARRAYS),
    "hh_override_s1741": (lambda u=None: build_hh(1741, updates=u), HH_ARRAYS),
}


def capture() -> dict:
    from sim.backend import get_backend
    _xp, backend = get_backend()
    cases = {}
    for name, (builder, arrays) in CASES.items():
        cases[name] = snapshot(builder(), arrays)
    return {"backend": backend, "device": "cpu" if backend == "numpy" else "gpu", "cases": cases}


if __name__ == "__main__":
    golden = capture()
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN_PATH} ({len(golden['cases'])} cases, backend {golden['backend']})")
