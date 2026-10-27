"""CI guard -- apical GABA_A routing (cs-race f1i section E, research/coordination/cs-race-f1i-PACKET.md).

cfg.enable_apical_gaba_routing + a plan pathway tagged receptor="gaba_a_apical": the routed synapses leave the somatic
g_i matvec and feed a per-neuron apical conductance (decay = GABA_A tau) that the BDSP apical update applies against
v_apical as a BOUNDED exponential-Euler term toward E_i. Pins:
  1. flag absent == flag False, and flag on with no tagged pathway, are byte-identical (state hash after N steps), and
     the flag-off path matches the engine BEFORE this edit (reference hash captured at 8e13b44cc on numpy);
  2. a routed spike raises g_ap only on its targets and gives NO somatic g_i; a closed transmission gate gives g_ap == 0;
  3. one step never carries v_apical past E_i, at g_ap in {0.1, 10, 1e4}, from above or below (the 2026-09-29 explicit-
     Euler overshoot family is not reintroduced); small g_ap matches the explicit form to first order;
  4. an untagged GABA pathway is unchanged by the routing (whole somatic state over N steps equals the plan without the
     tagged pathway);
  5. every misuse case raises; the routing mask survives a checkpoint round trip.
"""
import hashlib
import importlib
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sim.backend import get_backend, to_host  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.config import (CoreSimConfig, GPUConfig, NeuronModel, RuntimeState,  # noqa: E402
                        VisualizationConfig)

N = 40
EXC = np.arange(0, 24, dtype=np.int64)            # excitatory cells; 0-11 get somatic GABA, 12-23 apical GABA
EXC_SOMA = EXC[:12]
EXC_AP = EXC[12:]
INH_SOMA = np.arange(24, 32, dtype=np.int64)      # basket-like (somatic GABA_A)
INH_SOM = np.arange(32, 40, dtype=np.int64)       # SOM-like (apical GABA_A when tagged)
INH = np.concatenate([INH_SOMA, INH_SOM])
N_STEPS = 60
# Flag-off state hash captured from the engine BEFORE the apical-routing edit (base 8e13b44cc), numpy backend, by
# running _state_hash(_run(_build(42), N_STEPS)) with this file's builder against that tree's sim/ package (the only
# directory the edit touches); also the plan without the SOM pathway.
_REF_HASH_PRE_EDIT_NUMPY = "c4dcaa70edb7e2ee5fa9bce8cd4d75e155f9efcf5851d15fe08e428a524bf26f"
_REF_HASH_PRE_EDIT_NUMPY_NO_SOM = "81589e0416f0f86951de01e2018e4d00c641df3dd27c37b3536ffbf074d47350"


def _dense(pre, post, w, **extra):
    pre_r = np.repeat(np.asarray(pre, dtype=np.int64), len(post))
    post_r = np.tile(np.asarray(post, dtype=np.int64), len(pre))
    d = {"pre_indices": pre_r, "post_indices": post_r,
         "initial_weights": np.full(pre_r.shape[0], float(w), dtype=np.float32), "plastic": False}
    d.update(extra)
    return d


def _plan(*, tag=False, with_som_path=True, som_gate=None, som_w=4.0):
    plan = {
        "ee": _dense(EXC, EXC, 0.6),
        "ei": _dense(EXC, INH, 1.5),
        "ie_soma": _dense(INH_SOMA, EXC_SOMA, 3.0),         # untagged GABA_A: stays somatic
    }
    if with_som_path:
        extra = {}
        if tag:
            extra["receptor"] = "gaba_a_apical"
        if som_gate:
            extra["transmission_gate"] = som_gate
        plan["som_ap"] = _dense(INH_SOM, EXC_AP, som_w, **extra)
    return plan


def _build(seed=42, *, flag=None, plan=None, bdsp=True, two_comp=False, apical_drive=True, cfg_extra=None):
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = seed
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_structural_plasticity",
              "enable_watts_strogatz", "enable_ou_process", "enable_hebbian_learning"):
        setattr(cfg, f, False)
    cfg.enable_short_term_plasticity = True
    cfg.enable_bdsp = bool(bdsp)
    cfg.bdsp_apical_couples_soma = bool(bdsp)
    cfg.bdsp_apical_soma_g = 2.0
    cfg.enable_two_compartment_dap = bool(two_comp)
    if flag is not None:
        cfg.enable_apical_gaba_routing = bool(flag)
    for k, v in (cfg_extra or {}).items():
        setattr(cfg, k, v)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    b.inject_explicit_wiring(_plan() if plan is None else plan, output_inhibitory_indices=INH)
    xp = b.xp
    if bdsp and apical_drive:
        b.cp_bdsp_apical_drive = xp.zeros(N, dtype=xp.float32)
    drive = np.zeros(N, dtype=np.float32)
    drive[EXC] = 260.0
    drive[INH] = 120.0
    b.cp_external_input_current = xp.asarray(drive)
    return b


def _step(b, n=1):
    for _ in range(n):
        b._run_one_simulation_step()
        b.runtime_state.current_time_ms += b.core_config.dt_ms


def _run(b, n_steps=N_STEPS):
    raster = []
    for _ in range(n_steps):
        _step(b)
        raster.append(np.asarray(to_host(b.cp_firing_states)).astype(np.uint8))
    return b, np.stack(raster)


def _state_hash(run):
    b, raster = run
    h = hashlib.sha256()
    h.update(raster.tobytes())
    for name in ("cp_membrane_potential_v", "cp_conductance_g_e", "cp_conductance_g_i", "cp_v_apical"):
        arr = getattr(b, name)
        h.update(name.encode())
        h.update(b"None" if arr is None else np.asarray(to_host(arr)).tobytes())
    return h.hexdigest()


def _h(arr):
    return np.asarray(to_host(arr))


# ---------------------------------------------------------------- 1. byte identity when off
def test_flag_absent_equals_flag_false():
    r_abs = _run(_build(42, flag=None))
    r_off = _run(_build(42, flag=False))
    assert int(r_abs[1].sum()) > 0, "degenerate silent network makes the hash meaningless"
    assert _state_hash(r_abs) == _state_hash(r_off)
    assert r_off[0].cp_conductance_g_apical_gaba is None and r_off[0].cp_apical_gaba_synapse_mask is None


def test_flag_on_without_tagged_pathway_is_byte_identical():
    r_off = _run(_build(42, flag=False))
    b_on = _build(42, flag=True)
    assert b_on.cp_conductance_g_apical_gaba is None and b_on.cp_apical_gaba_synapse_mask is None
    assert _state_hash(_run(b_on)) == _state_hash(r_off)


def test_flag_off_matches_pre_edit_engine():
    _, name = get_backend()
    if str(name).lower() != "numpy":
        pytest.skip("the pre-edit reference hash was captured on numpy only")
    r = _run(_build(42, flag=None))
    assert _state_hash(r) == _REF_HASH_PRE_EDIT_NUMPY
    r2 = _run(_build(42, flag=None, plan=_plan(with_som_path=False)))
    assert _state_hash(r2) == _REF_HASH_PRE_EDIT_NUMPY_NO_SOM


# ---------------------------------------------------------------- 2. routing goes to the apical, not the soma
def _one_som_volley(b):
    """Zero every conductance, make only the SOM-like cells 'fired last step', run ONE step."""
    xp = b.xp
    b.cp_conductance_g_e[:] = 0.0
    b.cp_conductance_g_i[:] = 0.0
    if b.cp_conductance_g_apical_gaba is not None:
        b.cp_conductance_g_apical_gaba[:] = 0.0
    prev = np.zeros(N, dtype=bool)
    prev[INH_SOM] = True
    b.cp_prev_firing_states = xp.asarray(prev)
    b.cp_external_input_current = xp.zeros(N, dtype=xp.float32)
    _step(b)


def test_routed_spike_raises_apical_conductance_not_somatic_gi():
    b = _build(42, flag=True, plan=_plan(tag=True))
    assert b.cp_apical_gaba_synapse_mask is not None
    _one_som_volley(b)
    g_ap = _h(b.cp_conductance_g_apical_gaba)
    g_i = _h(b.cp_conductance_g_i)
    assert np.all(g_ap[EXC_AP] > 0.0)
    assert np.all(np.delete(g_ap, EXC_AP) == 0.0)          # only on its targets
    assert np.all(g_i[EXC_AP] == 0.0)                       # NO somatic g_i from the routed synapses
    # control: the same plan untagged, flag off -> the same volley lands in somatic g_i
    c = _build(42, flag=False, plan=_plan(tag=False))
    _one_som_volley(c)
    assert np.all(_h(c.cp_conductance_g_i)[EXC_AP] > 0.0)
    # the apical increment equals the somatic one it replaced, scaled by the two propagation strengths
    s_ap = float(b.core_config.apical_gaba_propagation_strength)
    s_i = float(c.core_config.inhibitory_propagation_strength)
    np.testing.assert_allclose(g_ap[EXC_AP] / s_ap, _h(c.cp_conductance_g_i)[EXC_AP] / s_i, rtol=1e-5)


def test_routed_spike_pulls_v_apical_down():
    b = _build(42, flag=True, plan=_plan(tag=True))
    _step(b, 3)
    v0 = _h(b.cp_v_apical).copy()
    _one_som_volley(b)
    _step(b)                                               # the conductance acts in the BDSP update
    v1 = _h(b.cp_v_apical)
    assert np.all(v1[EXC_AP] < v0[EXC_AP])


def test_closed_transmission_gate_gives_zero_apical_conductance():
    b = _build(42, flag=True, plan=_plan(tag=True, som_gate="cs_som"))
    b.set_transmission_gate("cs_som", 0.0)
    _one_som_volley(b)
    assert np.all(_h(b.cp_conductance_g_apical_gaba) == 0.0)
    assert np.all(_h(b.cp_conductance_g_i)[EXC_AP] == 0.0)


# ---------------------------------------------------------------- 3. the bounded apical term
@pytest.mark.parametrize("g", [0.1, 10.0, 1e4])
@pytest.mark.parametrize("v0", [-65.0, -40.0, -85.0])
def test_one_step_never_crosses_e_i(g, v0):
    b = _build(42, flag=True, plan=_plan(tag=True))
    _step(b)                                               # allocate cp_v_apical
    xp = b.xp
    e_i = float(b.core_config.syn_reversal_potential_i)
    tau = float(b.core_config.apical_tau_ms)
    dt = float(b.core_config.dt_ms)
    decay_i = float(np.exp(-dt / float(b.core_config.syn_tau_g_i)))
    b.cp_firing_states[:] = False
    b.cp_prev_firing_states = xp.zeros(N, dtype=bool)      # no spike: g_ap only decays this step
    b.cp_v_apical = xp.full(N, v0, dtype=xp.float32)
    b.cp_bdsp_apical_drive = xp.full(N, v0 - float(b.core_config.apical_E_rest), dtype=xp.float32)  # leak term == 0
    b.cp_conductance_g_apical_gaba = xp.full(N, g / decay_i, dtype=xp.float32)
    _step(b)
    v1 = _h(b.cp_v_apical)
    if v0 > e_i:
        assert np.all(v1 >= e_i) and np.all(v1 < v0)
    else:
        assert np.all(v1 <= e_i) and np.all(v1 > v0)
    expected = v0 + (e_i - v0) * (1.0 - np.exp(-g * dt / tau))
    np.testing.assert_allclose(v1, expected, rtol=0, atol=2e-3)
    if g == 1e4:
        assert np.all(v1 == np.float32(e_i))                # reaches E_i, exactly, never past it
        assert abs(g * (e_i - v0) * dt / tau) > 100.0       # the explicit form would have overshot by far
    if g == 0.1:                                            # first order == the explicit g*(E_i - v0)
        np.testing.assert_allclose(v1 - v0, g * (e_i - v0) * dt / tau, rtol=1e-2)


# ---------------------------------------------------------------- 4. untagged GABA unchanged
def test_untagged_gaba_pathway_unchanged_by_routing():
    """Flag on + tagged SOM path vs flag off + no SOM path at all: the somatic state is identical over N steps (the
    routed synapses add nothing to g_e/g_i, and the depressed apical gives no positive coupling), so the untagged
    somatic GABA_A pathway (ie_soma) behaves exactly as without the routing."""
    r_on = _run(_build(42, flag=True, plan=_plan(tag=True)))
    r_none = _run(_build(42, flag=False, plan=_plan(with_som_path=False)))
    assert int(r_none[1].sum()) > 0
    np.testing.assert_array_equal(r_on[1], r_none[1])
    for name in ("cp_membrane_potential_v", "cp_conductance_g_e", "cp_conductance_g_i"):
        np.testing.assert_array_equal(_h(getattr(r_on[0], name)), _h(getattr(r_none[0], name)), err_msg=name)
    assert float(_h(r_on[0].cp_conductance_g_apical_gaba)[EXC_AP].max()) >= 0.0


# ---------------------------------------------------------------- 5. fail loudly
def test_tagged_pathway_with_flag_off_raises():
    with pytest.raises(ValueError, match="enable_apical_gaba_routing is False"):
        _build(42, flag=False, plan=_plan(tag=True))
    with pytest.raises(ValueError, match="enable_apical_gaba_routing is False"):
        _build(42, flag=None, plan=_plan(tag=True))


def test_tagged_pathway_without_bdsp_raises():
    with pytest.raises(ValueError, match="enable_bdsp"):
        _build(42, flag=True, plan=_plan(tag=True), bdsp=False)


def test_tagged_pathway_with_two_compartment_dap_raises():
    with pytest.raises(ValueError, match="enable_two_compartment_dap"):
        _build(42, flag=True, plan=_plan(tag=True), two_comp=True)


def test_excitatory_source_raises():
    plan = _plan()
    plan.pop("ee")                                          # keep the bad edges' coordinates unique
    plan["bad"] = _dense(EXC_SOMA[:2], EXC_AP, 1.0, receptor="gaba_a_apical")
    with pytest.raises(ValueError, match="non-inhibitory source"):
        _build(42, flag=True, plan=plan)


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="negative weight"):
        _build(42, flag=True, plan=_plan(tag=True, som_w=-1.0))


def test_graded_tagged_raises():
    plan = _plan(tag=True)
    plan["som_ap"]["graded"] = True
    with pytest.raises(ValueError, match="coincidence_detector/graded"):
        _build(42, flag=True, plan=plan)


def test_shared_coordinate_raises():
    plan = _plan(tag=True)
    plan["dup"] = _dense(INH_SOM[:1], EXC_AP[:1], 1.0)        # same (pre,post) as a tagged edge, untagged
    with pytest.raises(ValueError, match="both receptor='gaba_a_apical' and"):
        _build(42, flag=True, plan=plan)


def test_missing_apical_drive_raises_at_step():
    b = _build(42, flag=True, plan=_plan(tag=True), apical_drive=False)
    b.strict_step_errors = True                             # the engine's loud-step opt-in
    with pytest.raises(RuntimeError, match="cp_bdsp_apical_drive is None"):
        _step(b)


def test_flag_turned_off_after_wiring_raises_at_step():
    b = _build(42, flag=True, plan=_plan(tag=True))
    b.core_config.enable_apical_gaba_routing = False
    b.strict_step_errors = True                             # the engine's loud-step opt-in
    with pytest.raises(RuntimeError, match="routing mask is present"):
        _step(b)


def test_rewire_without_tag_clears_routing():
    b = _build(42, flag=True, plan=_plan(tag=True))
    b.inject_explicit_wiring(_plan(tag=False), output_inhibitory_indices=INH)
    assert b.cp_apical_gaba_synapse_mask is None and b.cp_conductance_g_apical_gaba is None


def test_megakernel_refuses_routed_step():
    b = _build(42, flag=True, plan=_plan(tag=True))
    b.core_config.enable_step_megakernel = True
    assert b._step_megakernel_can_dispatch() is False


def test_routing_mask_survives_checkpoint(tmp_path):
    b = _build(42, flag=True, plan=_plan(tag=True))
    _step(b, 5)
    mask = _h(b.cp_apical_gaba_synapse_mask)[: int(b.cp_connections.nnz)].copy()
    g_ap = _h(b.cp_conductance_g_apical_gaba).copy()
    ck = tmp_path / "apgaba.simstate.h5"
    assert b.save_checkpoint(str(ck)) is True
    r = _build(42, flag=True, plan=_plan(tag=False))
    assert r.cp_apical_gaba_synapse_mask is None
    assert r.load_checkpoint(str(ck)) is True
    np.testing.assert_array_equal(_h(r.cp_apical_gaba_synapse_mask), mask)
    np.testing.assert_array_equal(_h(r.cp_conductance_g_apical_gaba), g_ap)
