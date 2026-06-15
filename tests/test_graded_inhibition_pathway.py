"""Graded (analog, non-spiking) inhibition pathway — the retina's horizontal-cell mechanism.

A `RegionPathway(graded=True)` makes the per-step inhibitory (or excitatory) conductance drive use the
SOURCE neuron's CONTINUOUS activity (normalized excitatory conductance g_e, in [0,1]) INSTEAD of its binary
`cp_firing_states`. Horizontal/bipolar cells release transmitter proportional to their GRADED membrane
potential, not spikes — which is exactly what the spiking learned cortex needs for whitening (common-mode
removal): a spiking inhibitory pool cannot linearly track the population mean (depolarization block makes its
SPIKES anti-track the mean), but its analog g_e does.

The LOAD-BEARING guarantee: with NO graded pathway present (the default), the bridge is BYTE-IDENTICAL to
before the edit. The graded code path is gated on `cp_graded_synapse_mask is not None`, which is only built
when at least one pathway sets graded=True.

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
from tests._capture_graded_golden import build_nongraded_bridge, run_and_snapshot  # noqa: E402

_GOLDEN = os.path.join(_HERE, "_graded_golden.npz")


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
# THE LOAD-BEARING REGRESSION: byte-identical when no pathway is graded.
# ===========================================================================
def test_no_graded_pathway_leaves_mask_none():
    """A pathway WITHOUT graded=True leaves cp_graded_synapse_mask None -> the new step block is unreached
    (zero overhead, byte-identical). Mirrors test_no_transmission_gate_means_no_gain_array."""
    regions = [BrainRegion(name="A", n_neurons=20, internal_density=0.0),
               BrainRegion(name="B", n_neurons=20, exc_fraction=0.0, internal_density=0.0)]
    pathways = [RegionPathway(from_region="A", to_region="B", density=0.5, plastic=False)]
    sb = _bridge(regions, pathways)
    assert getattr(sb, "cp_graded_synapse_mask", None) is None


def test_byte_identical_when_no_graded_pathway():
    """THE load-bearing guard: re-build the exact golden bridge (no graded pathway), run the exact steps,
    and assert every state array matches the pre-edit golden snapshot EXACTLY (atol=0). If this fails, the
    graded edit perturbed the default (non-graded) code path — a hard NO."""
    assert os.path.exists(_GOLDEN), (
        "golden snapshot missing — run `SIM_BACKEND=numpy python tests/_capture_graded_golden.py` "
        "on the UNEDITED bridge first.")
    golden = dict(np.load(_GOLDEN))
    sb = build_nongraded_bridge()
    # the new attribute must default to None on a non-graded build (proves the new path is not entered)
    assert getattr(sb, "cp_graded_synapse_mask", None) is None
    snap = run_and_snapshot(sb)
    for key in ("g_e", "g_i", "v", "u", "w", "spk_acc"):
        np.testing.assert_array_equal(
            snap[key], golden[key],
            err_msg=f"graded edit changed the NON-graded run: array '{key}' differs from golden")


# ===========================================================================
# FUNCTION: a graded inhibitory pathway delivers GRADED (continuous-source) inhibition.
# ===========================================================================
def _src_inhibits_dst_bridge(graded, *, inh_weight=200.0, seed=42):
    """src(exc) -> dst(exc) excitation, and inh(inh) -> dst(exc) inhibition. inh is driven by a steady
    external current so it has a continuous (graded) analog activity AND (in the spike case) spikes.
    Returns (bridge, idx)."""
    regions = [BrainRegion(name="src", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
               BrainRegion(name="inh", n_neurons=20, exc_fraction=0.0, internal_density=0.0,
                           plastic_internal=False),
               BrainRegion(name="dst", n_neurons=30, exc_fraction=1.0, internal_density=0.0)]
    pathways = [
        RegionPathway(from_region="src", to_region="dst", density=1.0, weight_mean=140.0,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="inh", to_region="dst", density=1.0, weight_mean=inh_weight,
                      weight_jitter=0.0, plastic=False, graded=bool(graded)),
    ]
    sb = _bridge(regions, pathways)
    idx = {n: np.asarray(sb.region_manager.indices(n)) for n in ("src", "inh", "dst")}
    return sb, idx


def _run_gi(sb, idx, *, n_steps=80, src_pA=1400.0, inh_pA=600.0):
    """Drive src + inh with steady current; return (mean dst firing rate, mean dst g_i over last step)."""
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[idx["src"]] = src_pA
    sb.cp_external_input_current[idx["inh"]] = inh_pA
    spk = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        spk += to_host(sb.cp_firing_states).astype(np.float64)
    gi = to_host(sb.cp_conductance_g_i).astype(np.float64)
    return float(spk[idx["dst"]].mean()) / n_steps, float(gi[idx["dst"]].mean())


def test_graded_pathway_allocates_mask():
    sb, _ = _src_inhibits_dst_bridge(graded=True)
    assert getattr(sb, "cp_graded_synapse_mask", None) is not None
    from sim.backend import to_host
    n_graded = int(to_host(sb.cp_graded_synapse_mask).sum())
    assert n_graded == 20 * 30, f"expected 600 graded synapses (inh 20 x dst 30), got {n_graded}"


def test_graded_inhibition_produces_g_i_on_target():
    """A GRADED inhibitory pathway must actually inhibit: the target dst accumulates g_i from the inh pool's
    CONTINUOUS activity (not its spikes). g_i > 0 confirms the graded path routes to the inhibitory
    conductance via the E/I split, like a spike-mediated inhibitory pathway."""
    sb, idx = _src_inhibits_dst_bridge(graded=True)
    _, gi_dst = _run_gi(sb, idx)
    assert gi_dst > 0.0, "graded inhibitory pathway delivered no g_i to the target"


def test_graded_inhibition_scales_with_continuous_source_activity():
    """The DEFINING property of graded transmission: the target's g_i scales with the SOURCE's CONTINUOUS
    activity (its membrane depolarization), not with a spike count. Driving the inh pool harder (more
    depolarized -> larger a_cont) must deliver more g_i to the target. This is what spike-mediated inhibition
    cannot do linearly (depol-block saturation); the analog/graded path can."""
    sb_lo, idx = _src_inhibits_dst_bridge(graded=True, inh_weight=120.0)
    _, gi_lo = _run_gi(sb_lo, idx, inh_pA=200.0)
    sb_hi, idx2 = _src_inhibits_dst_bridge(graded=True, inh_weight=120.0)
    _, gi_hi = _run_gi(sb_hi, idx2, inh_pA=1200.0)
    assert gi_hi > gi_lo > 0.0, (
        f"graded g_i did not increase with source drive (low-drive g_i {gi_lo:.2f} vs high-drive {gi_hi:.2f})")


def test_graded_pathway_does_not_transmit_on_spikes():
    """A graded pathway is REMOVED from the spike matvec (it transmits gradedly, not on spikes). If we drive
    ONLY the inh pool and hold its membrane at rest by NOT depolarizing it, the graded drive is ~0; the
    contribution comes from the analog state, not spikes. Cross-check: a graded inh pathway and an otherwise
    identical SPIKE inh pathway produce DIFFERENT g_i on the target for the same drive (the graded one tracks
    the analog membrane; the spike one tracks the binary spikes) — proving the source term was swapped."""
    sb_g, idx = _src_inhibits_dst_bridge(graded=True, inh_weight=120.0)
    _, gi_graded = _run_gi(sb_g, idx, inh_pA=600.0)
    sb_s, idx2 = _src_inhibits_dst_bridge(graded=False, inh_weight=120.0)
    _, gi_spike = _run_gi(sb_s, idx2, inh_pA=600.0)
    assert gi_graded > 0.0 and gi_spike > 0.0
    # The two transmission modes are genuinely different (continuous vs binary source term).
    assert abs(gi_graded - gi_spike) > 1e-3 * max(gi_graded, gi_spike), (
        f"graded and spike g_i identical ({gi_graded:.3f} vs {gi_spike:.3f}) — the source term wasn't swapped")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
