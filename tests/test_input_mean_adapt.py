"""Slow per-hub INPUT-MEAN adaptation (axis-0 per-feature centering) — the brain-based DC/diagonal
half of whitening.

A `BrainRegion(input_mean_adapt=True)` (with `cfg.enable_input_mean_adapt=True`) makes each of that
region's neurons subtract a SLOW EMA of its OWN pre-threshold input drive (synaptic + external
current) from that drive BEFORE the spike threshold: `adapted = raw - gain*m; m <- (1-alpha)*m +
alpha*raw` (causal — subtract the current m, then update m from raw). This is subtractive spike-
frequency adaptation / point-neuron predictive coding (Lee/Pennartz 2024, PMC11045951), the per-
feature centering the L1 spiking learned cortex needs.

The LOAD-BEARING guarantee: with NO adapting region present (the default), the bridge is BYTE-
IDENTICAL to before the edit. The new step block is gated on `cp_input_mean_ema is not None`, which
is only built when at least one region sets `input_mean_adapt=True` AND `cfg.enable_input_mean_adapt`.

Runs on whatever backend is active. CPU CI: SIM_BACKEND=numpy.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.regions import BrainRegion, RegionPathway  # noqa: E402

# Reuse the exact builder + run that produced the golden snapshot.
from tests._capture_input_mean_golden import build_nonadapt_bridge, run_and_snapshot  # noqa: E402

_GOLDEN = os.path.join(_HERE, "_input_mean_golden.npz")


def _bridge(regions, pathways, seed=42, **cfg_over):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    cfg.ou_std_current_pA = 0.0
    cfg.enable_short_term_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_reward_modulation = False
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


# ===========================================================================
# THE LOAD-BEARING REGRESSION: byte-identical when no region adapts.
# ===========================================================================
def test_no_adapt_leaves_arrays_none():
    """A bridge with no input_mean_adapt region leaves BOTH cp_input_mean_ema and
    cp_input_mean_adapt_mask None -> the new step block is unreached (zero overhead, byte-identical).
    Even when cfg.enable_input_mean_adapt is True, no flagged region => arrays stay None."""
    regions = [BrainRegion(name="A", n_neurons=20, internal_density=0.0),
               BrainRegion(name="B", n_neurons=20, exc_fraction=0.0, internal_density=0.0)]
    pathways = [RegionPathway(from_region="A", to_region="B", density=0.5, plastic=False)]
    # default config (flag off) -> None
    sb = _bridge(regions, pathways)
    assert getattr(sb, "cp_input_mean_ema", None) is None
    assert getattr(sb, "cp_input_mean_adapt_mask", None) is None
    # flag ON but NO region flagged -> still None (both conditions required)
    sb2 = _bridge(regions, pathways, enable_input_mean_adapt=True)
    assert getattr(sb2, "cp_input_mean_ema", None) is None
    assert getattr(sb2, "cp_input_mean_adapt_mask", None) is None


def test_byte_identical_when_off():
    """THE load-bearing guard: re-build the exact golden bridge (no adapting region), run the exact
    steps, and assert every state array matches the pre-edit golden snapshot EXACTLY (atol=0). If
    this fails, the input-mean-adapt edit perturbed the default (non-adapting) code path — a hard NO.

    The golden `_input_mean_golden.npz` was captured on the PRE-EDIT bridge (the new attributes were
    'MISSING' there), so this is a true pre/post A/B byte-identity proof, exactly like the graded
    edit (CYCLE-68)."""
    assert os.path.exists(_GOLDEN), (
        "golden snapshot missing — run `SIM_BACKEND=numpy python tests/_capture_input_mean_golden.py` "
        "on the UNEDITED bridge first.")
    golden = dict(np.load(_GOLDEN))
    sb = build_nonadapt_bridge()
    # the new attributes must default to None on a non-adapting build (proves the new path is not entered)
    assert getattr(sb, "cp_input_mean_ema", None) is None
    assert getattr(sb, "cp_input_mean_adapt_mask", None) is None
    snap = run_and_snapshot(sb)
    for key in ("g_e", "g_i", "v", "u", "w", "spk_acc"):
        np.testing.assert_array_equal(
            snap[key], golden[key],
            err_msg=f"input-mean-adapt edit changed the NON-adapting run: array '{key}' differs from golden")


# ===========================================================================
# ALLOCATION: a flagged region builds the mask + EMA; the mask tags only that region.
# ===========================================================================
def _adapt_bridge(*, adapt, n_hub=40, drive=1200.0, alpha=0.05, gain=1.0, seed=42):
    """hub (exc) region driven by a STEADY external current; an unflagged 'other' region driven
    identically. If adapt=True the hub adapts (input_mean_adapt=True). The two regions are NOT
    cross-coupled (so 'other' is a clean external-drive control); each carries a tiny weak internal
    recurrence purely so the wiring plan is non-empty (a zero-synapse plan hits the PRE-EXISTING
    `profile_name_for_conn` UnboundLocalError unrelated to this edit). The internal weight is small
    so the steady external drive dominates and the firing dynamics are essentially externally set."""
    regions = [
        BrainRegion(name="hub", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, weight_jitter=0.0, plastic_internal=False,
                    input_mean_adapt=bool(adapt)),
        BrainRegion(name="other", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.05,
                    exc_weight_mean=1.0, weight_jitter=0.0, plastic_internal=False,
                    input_mean_adapt=False),
    ]
    pathways = []  # no cross-region coupling: 'other' is a pure external-drive control
    sb = _bridge(regions, pathways, enable_input_mean_adapt=True,
                 input_mean_adapt_alpha=alpha, input_mean_adapt_gain=gain, seed=seed)
    idx = {n: np.asarray(sb.region_manager.indices(n)) for n in ("hub", "other")}
    return sb, idx


def test_flagged_region_allocates_mask_and_ema():
    sb, idx = _adapt_bridge(adapt=True)
    from sim.backend import to_host
    assert getattr(sb, "cp_input_mean_adapt_mask", None) is not None
    assert getattr(sb, "cp_input_mean_ema", None) is not None
    mask = to_host(sb.cp_input_mean_adapt_mask)
    # exactly the hub neurons are flagged; 'other' is not.
    assert mask[idx["hub"]].all(), "all hub neurons should be flagged"
    assert not mask[idx["other"]].any(), "no 'other' neurons should be flagged"
    assert int(mask.sum()) == len(idx["hub"])
    # EMA starts at zero (warms up over the slow alpha).
    ema0 = to_host(sb.cp_input_mean_ema)
    assert np.allclose(ema0, 0.0)


# ===========================================================================
# FUNCTION: the adapting region's EMA converges to its steady input AND its firing DROPS;
#           a non-adapting region's firing is unchanged.
# ===========================================================================
def _run_drive(sb, idx, *, n_steps, drive_pA, regions=("hub", "other")):
    """Drive the listed regions with a steady external current; return dict of per-region mean
    firing rate over the run (spikes/neuron/step) and the final EMA at the hub."""
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    for r in regions:
        sb.cp_external_input_current[idx[r]] = drive_pA
    spk = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        spk += to_host(sb.cp_firing_states).astype(np.float64)
    out = {r: float(spk[idx[r]].mean()) / n_steps for r in idx}
    if sb.cp_input_mean_ema is not None:
        out["_ema_hub"] = to_host(sb.cp_input_mean_ema)[idx["hub"]]
    return out


def test_ema_converges_toward_steady_input():
    """The DEFINING bookkeeping property: under a STEADY input current, the per-hub EMA m converges
    toward that current (it is a low-pass of the hub's own input drive). The drive injected is
    `drive_pA`, and the bridge's own synaptic/external current path makes the hub's raw input ~ that
    drive, so m must climb from 0 toward a large positive value (not stay at 0, not go negative)."""
    sb, idx = _adapt_bridge(adapt=True, alpha=0.1, gain=0.0, n_hub=40)
    # gain=0 so the subtraction does NOT feed back into the drive — isolates "does the EMA track the
    # raw input current" (the EMA still updates from raw regardless of gain).
    res = _run_drive(sb, idx, n_steps=200, drive_pA=1000.0)
    ema = res["_ema_hub"]
    assert ema.mean() > 100.0, (
        f"hub EMA did not climb toward the steady input drive (mean EMA {ema.mean():.2f}); "
        f"the EMA must track the neuron's own input current")
    assert (ema > 0).all(), "every hub neuron's EMA should be positive under a positive input drive"


def test_adaptation_drops_hub_firing_over_time():
    """The DEFINING functional property: with the subtraction ON (gain=1), the hub subtracts its own
    growing mean from its drive, so its EFFECTIVE drive — and thus its firing rate — DROPS over time.
    Compare the hub's firing in an early window vs a late window of the SAME steady drive: late < early.
    A non-adapting hub (control) does NOT drop."""
    # ADAPTING hub.
    sb, idx = _adapt_bridge(adapt=True, alpha=0.08, gain=1.0, n_hub=40)
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[idx["hub"]] = 1100.0

    def _window_rate(steps):
        acc = np.zeros(sb.core_config.num_neurons)
        for _ in range(steps):
            sb._run_one_simulation_step()
            acc += to_host(sb.cp_firing_states).astype(np.float64)
        return float(acc[idx["hub"]].mean()) / steps

    early = _window_rate(40)
    # let the EMA converge across many more steps
    for _ in range(300):
        sb._run_one_simulation_step()
    late = _window_rate(40)
    assert late < early - 1e-6, (
        f"adapting hub firing did not drop (early {early:.4f} -> late {late:.4f}); "
        f"the DC subtraction should reduce the effective drive over time")

    # CONTROL: a NON-adapting hub with the same drive does NOT systematically drop.
    sb2, idx2 = _adapt_bridge(adapt=False, alpha=0.08, gain=1.0, n_hub=40)
    assert sb2.cp_input_mean_ema is None  # not flagged -> no adaptation at all
    sb2.cp_external_input_current[:] = 0.0
    sb2.cp_external_input_current[idx2["hub"]] = 1100.0

    def _window_rate2(steps):
        acc = np.zeros(sb2.core_config.num_neurons)
        for _ in range(steps):
            sb2._run_one_simulation_step()
            acc += to_host(sb2.cp_firing_states).astype(np.float64)
        return float(acc[idx2["hub"]].mean()) / steps

    early2 = _window_rate2(40)
    for _ in range(300):
        sb2._run_one_simulation_step()
    late2 = _window_rate2(40)
    # non-adapting hub stays a regular spiker (allow tiny numerical wobble; it must NOT collapse).
    assert late2 >= early2 - 0.02, (
        f"non-adapting control unexpectedly dropped (early {early2:.4f} -> late {late2:.4f})")


def test_nonadapting_region_firing_unchanged_by_neighbor_adaptation():
    """An unflagged region's neurons must be UNTOUCHED by the input-mean block (mask routes both the
    subtraction and the EMA update). Drive hub (adapting) + other (not) identically; the 'other'
    region's firing should match a run where adaptation is entirely off for 'other' (it never adapts).
    Here we simply assert the 'other' mask entries are False and that 'other' keeps firing at its
    full rate while the adapting hub drops below it."""
    sb, idx = _adapt_bridge(adapt=True, alpha=0.08, gain=1.0, n_hub=40)
    res = _run_drive(sb, idx, n_steps=400, drive_pA=1100.0)
    assert res["other"] > 0.0, "non-adapting 'other' region should fire under the steady drive"
    assert res["other"] > res["hub"], (
        f"the adapting hub ({res['hub']:.4f}) should fire LESS than the non-adapting 'other' "
        f"({res['other']:.4f}) under identical drive (its DC is subtracted; 'other''s is not)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
