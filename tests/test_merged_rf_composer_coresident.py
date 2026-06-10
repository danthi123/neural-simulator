"""STEP 2b adapter correctness (CPU): the co-resident RF composer == the standalone composer, exactly.

This is the merged-`n` form of STEP-2b acceptance gate 1 (the 5b edited-version) at the composer-op level: a
``MergedRFComposer`` running a bind / unbind on a SLICE of an Izhikevich bridge must reproduce a standalone
``RFPhasorComposer``'s result EXACTLY, while a co-resident Izhikevich slice's membrane state stays byte-identical
across the RF op.

It runs on the numpy backend (small D, tiny bridge) so it is CI/CPU-safe and does not need (or contend) the GPU.
The full-scale conversational acceptance (gate b on the ~7k-neuron merged bridge) is the GPU test
``tests/test_nav_conv_step2b_coresident.py``.
"""
import numpy as np
import pytest

from sim.backend import get_backend


VOCAB = ["dog", "cat", "go", "north", "south"]
D = 8                      # small projection dim for a fast CPU test
RF_SIZE = 7 * D            # the rf region size the merged builder reserves (covers a 6-role bundle)
NAV_SLICE = 40             # a block of co-resident Izhikevich ("navigation") neurons before the rf slice


def _izh_bridge(n, seed=42):
    """A plain Izhikevich bridge of `n` neurons, no plasticity / framework / OU — the minimal co-resident host."""
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _merged_composer():
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.nav_conv_merged_bridge import MergedRFComposer
    standalone = RFPhasorComposer(seed=42, D=D, vocab=VOCAB, period=200)
    b = _izh_bridge(NAV_SLICE + RF_SIZE, seed=42)
    merged = MergedRFComposer(b, rf_base=NAV_SLICE, rf_size=RF_SIZE,
                              seed=42, D=D, vocab=VOCAB, period=200)
    return standalone, merged, b


def _seed_izh_slice(bridge, n):
    """Put some non-trivial Izhikevich state on the first `n` (navigation) neurons; return a snapshot of (v, u)."""
    xp, _ = get_backend()
    v = bridge.cp_membrane_potential_v
    u = bridge.cp_recovery_variable_u
    v[:n] = xp.asarray(np.linspace(-65.0, -50.0, n), dtype=v.dtype)
    u[:n] = xp.asarray(np.linspace(-13.0, -8.0, n), dtype=u.dtype)
    return v[:n].copy(), u[:n].copy()


def test_co_resident_bind_matches_standalone():
    """bind(role, filler) on the rf slice equals the standalone composer's bind, to floating-point."""
    standalone, merged, _ = _merged_composer()
    ref = standalone._bind(standalone.roles["agent"], standalone.concepts["dog"])
    got = merged._bind(merged.roles["agent"], merged.concepts["dog"])
    assert got.shape == ref.shape == (D,)
    assert np.allclose(got, ref, atol=1e-9), f"max|delta| = {np.max(np.abs(got - ref))}"


def test_co_resident_unbind_matches_standalone():
    """The full bind->unbind round-trip on the rf slice recovers the same phases as the standalone composer."""
    standalone, merged, _ = _merged_composer()
    comp_ref = standalone._bind(standalone.roles["agent"], standalone.concepts["dog"])
    rec_ref = standalone._unbind_phases(comp_ref, "agent")
    comp_got = merged._bind(merged.roles["agent"], merged.concepts["dog"])
    rec_got = merged._unbind_phases(comp_got, "agent")
    assert np.allclose(rec_got, rec_ref, atol=1e-9), f"max|delta| = {np.max(np.abs(rec_got - rec_ref))}"


def test_co_resident_op_leaves_izh_slice_byte_identical():
    """The masked RF op writes ONLY the rf slice; the co-resident Izhikevich (navigation) neurons' v/u are unchanged."""
    _, merged, bridge = _merged_composer()
    v0, u0 = _seed_izh_slice(bridge, NAV_SLICE)
    merged._bind(merged.roles["agent"], merged.concepts["dog"])          # a full RF op (kick + resonate loop)
    assert bool((bridge.cp_membrane_potential_v[:NAV_SLICE] == v0).all()), "navigation membrane v perturbed by RF op"
    assert bool((bridge.cp_recovery_variable_u[:NAV_SLICE] == u0).all()), "navigation recovery u perturbed by RF op"


def test_op_larger_than_rf_region_raises():
    """An op that needs more than the reserved rf slice fails loudly (not a silent index overrun)."""
    _, merged, _ = _merged_composer()
    conns = [(0, 0, 1.0)]
    kick = np.zeros(RF_SIZE + 1, dtype=np.complex128)
    with pytest.raises(ValueError):
        merged._resonate(RF_SIZE + 1, conns, kick)


def test_spiking_cleanup_co_residence_is_rejected():
    """STEP 2b scopes binding co-residence; the spiking-cleanup readout co-residence is explicitly out of scope."""
    from research.runners.nav_conv_merged_bridge import MergedRFComposer
    b = _izh_bridge(NAV_SLICE + RF_SIZE, seed=42)
    with pytest.raises(NotImplementedError):
        MergedRFComposer(b, rf_base=NAV_SLICE, rf_size=RF_SIZE,
                         seed=42, D=D, vocab=VOCAB, period=200, enable_spiking_cleanup=True)
