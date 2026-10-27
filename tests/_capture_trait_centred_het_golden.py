"""Capture the unset-byte-identity golden for `CoreSimConfig.trait_centred_parameter_heterogeneity` / SIM_TRAIT_CENTRED_HET.

Run ONCE on the UNEDITED bridge (research/stable-mechc2 at f58274479, before the trait-centred edit). The regression
test `tests/test_trait_centred_heterogeneity.py::test_unset_is_byte_identical_to_the_pre_edit_golden` rebuilds every
case below on the edited bridge (flag at its default OFF, env unset or "0") and asserts each hash matches this golden
EXACTLY: the four jittered Izhikevich arrays (a, b, d, C), the five never-jittered ones (k, vr, vt, vpeak, c_reset),
the neuron type ids, the per-neuron firing thresholds and the global RNG state (numpy legacy + python `random`) after
the build.

The cases are the trait-typed populations the variant is about (cfg.num_traits default 5, no region preset): the
LearnedAssocGraph bridge at two sizes and seeds (`_D_sparse_heteroassoc.build`), a plain pool with no region
framework, an untyped region next to a typed (STN) region, the same with the region-scoped redraw
(per_region_parameter_heterogeneity), and the global-OFF per-region mask path; plus a single-type pool
(num_traits = 1), which the variant must never touch.

Usage:  SIM_BACKEND=numpy SIM_NO_PROVENANCE=1 python -u tests/_capture_trait_centred_het_golden.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests._capture_preset_centred_het_golden import IZH_ARRAYS, snapshot  # noqa: E402

GOLDEN_PATH = os.path.join(_HERE, "_trait_centred_het_golden.json")
FIXED_ARRAYS = ("cp_izh_k", "cp_izh_vr", "cp_izh_vt", "cp_izh_vpeak", "cp_izh_c_reset", "cp_neuron_type_ids")
ARRAYS = IZH_ARRAYS + FIXED_ARRAYS


def _region(name, n, izh_type=None, **kw):
    from sim.regions import BrainRegion
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                       inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, izh_neuron_type=izh_type,
                       enable_homeostasis=False, **kw)


def build_pool(seed, regions=(), *, n=0, updates=None):
    """A small Izhikevich bridge, global heterogeneity ON, cfg.num_traits at its default (5) unless `updates` sets it.
    `regions` empty -> no region framework (a plain pool of n neurons); otherwise one cell population per region."""
    from sim import CoreSimConfig, GPUConfig, NeuronModel, RuntimeState, SimulationBridge, VisualizationConfig
    regions = list(regions)
    cfg = CoreSimConfig(
        num_neurons=(sum(r.n_neurons for r in regions) if regions else int(n)), connections_per_neuron=0,
        seed=int(seed), neuron_model_type=NeuronModel.IZHIKEVICH.name, dt_ms=0.5,
        enable_brain_region_framework=bool(regions), brain_regions=regions, region_pathways=[],
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


def build_lag(seed, n_pool, n_fs):
    """The LearnedAssocGraph bridge (research/runners/_D_sparse_heteroassoc.build): two regions, neither typed."""
    from research.runners._D_sparse_heteroassoc import build
    return build(int(seed), n_pool=int(n_pool), n_fs=int(n_fs))


def mixed_regions():
    """A typed STN region beside an untyped one: the untyped region's cells are typed by trait."""
    return [_region("typed_stn", 24, "IZH2007_STN_BURST"), _region("untyped_pool", 60)]


def masked_untyped():
    """Global heterogeneity OFF; an untyped region opts in through the per-region mask."""
    return [_region("typed_stn", 24, "IZH2007_STN_BURST"), _region("masked_untyped", 60, enable_heterogeneity=True)]


# name -> builder taking an `updates` dict (extra CoreSimConfig fields set before the build; ignored by the LAG cases).
CASES = {
    "lag_s7_small": lambda u=None: build_lag(7, 60, 24),
    "lag_s1_mid": lambda u=None: build_lag(1, 200, 40),
    "plain_traits_s3": lambda u=None: build_pool(3, n=150, updates=u),
    "mixed_typed_untyped_s5": lambda u=None: build_pool(5, mixed_regions(), updates=u),
    "mixed_region_scoped_s8": lambda u=None: build_pool(
        8, mixed_regions(), updates={"per_region_parameter_heterogeneity": True, **(u or {})}),
    "masked_untyped_global_off_s6": lambda u=None: build_pool(
        6, masked_untyped(), updates={"enable_parameter_heterogeneity": False, **(u or {})}),
    "single_type_s4": lambda u=None: build_pool(4, n=120, updates={"num_traits": 1, **(u or {})}),
}


def capture() -> dict:
    from sim.backend import get_backend
    _xp, backend = get_backend()
    cases = {}
    for name, builder in CASES.items():
        cases[name] = snapshot(builder(), ARRAYS)
    return {"backend": backend, "device": "cpu" if backend == "numpy" else "gpu", "cases": cases}


if __name__ == "__main__":
    golden = capture()
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN_PATH} ({len(golden['cases'])} cases, backend {golden['backend']})")
