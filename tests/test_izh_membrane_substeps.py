"""cfg.izh_membrane_substeps (sim/config.py) -- the default-OFF Izhikevich-2007 membrane sub-step integrator.

THE DEFECT (research/findings/2026-10-26-izh-membrane-substeps-derisk.md): the bridge advances
dv/dt = (k(v-vr)(v-vt) - u + I)/C by ONE explicit-Euler step of dt = 1 ms. The map's local multiplier is
1 + dt*k(2v-vr-vt)/C, so a striatal MSN preset (k 1, vr -80, vt -25) is unstable at rest below C = 27.5 pF.
In the BG neural-seam organ (dev seeds 72/75/77) three MSNs drawn at C 18-24 pF were kicked by a selector-reset
GABA_A volley into an absorbing spike / reset / deep-dip 2-step cycle; with b < 0, u ran away and the dip went
to -1918 / -1633 / -460 mV. Bounded conductance integration was on and does not cover the -u/C or quadratic term.

T0 here replays those three cells OPEN-LOOP through a one-neuron SimulationBridge (the real bounded conductance
kernel, Izhikevich update, threshold, reset and refractory timer): the recorded per-step conductances and other
currents of each cell (tests/fixtures/izh_substep_cells.npz, built by tests/fixtures/
build_izh_substep_cells_fixture.py from the diagnosis probe) are written into the bridge each step with the
conductance decay set to 1 so the recorded values are used as given. Flag off must reproduce the organ's own
trace (v_min and spike count exactly); C = 50 pF flag off and n = 4 flag on must keep v in [-150, vpeak].
T1 pins the default path: izh_membrane_substeps=1 set explicitly is byte-identical to the default, and the
bounded-conductance pre-fix engine hash still reproduces.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import get_backend, to_host  # noqa: E402
from sim.kernels import (  # noqa: E402
    izhikevich2007_dynamics_update,
    izhikevich2007_substep_update,
)
from sim.regions import BrainRegion, RegionPathway  # noqa: E402

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "izh_substep_cells.npz")
_F = np.load(_FIXTURE)
_META = json.loads(bytes(_F["meta_json"]).decode())
CELLS = sorted(_META["cells"])          # s72 (str_d2_1 idx 439), s75 (str_d1_0 idx 145), s77 (str_d1_1 idx 418)


def _host(x):
    return np.asarray(to_host(x))


def _region(name, preset):
    return BrainRegion(name=name, n_neurons=1, exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                       inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, izh_neuron_type=preset,
                       enable_homeostasis=False, enable_nmda=False)


def _cell_bridge(cell, *, n_sub=None, rule=None, C=None, fast_spike_reset=None):
    """One MSN (the cell's preset, then its drawn a/b/C/d) + one never-driven pad FS cell (a bridge with zero
    synapses does not initialise). Bounded conductance ON as in the organ; plasticity, OU, heterogeneity off.
    n_sub None leaves izh_membrane_substeps at its default."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = 7
    cfg.enable_brain_region_framework = True
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp", "enable_ou_process",
                 "enable_parameter_heterogeneity"):
        setattr(cfg, flag, False)
    cfg.enable_bounded_conductance_integration = True
    assert cfg.refractory_period_steps == 2       # the organ's value
    if n_sub is not None:
        cfg.izh_membrane_substeps = n_sub
    if rule is not None:
        cfg.izh_substep_refractory_rule = rule
    if fast_spike_reset is not None:
        cfg.fast_spike_reset = bool(fast_spike_reset)
    cfg.brain_regions = [_region("msn", cell["preset"]), _region("pad", "IZH2007_FS_CORTICAL_INTERNEURON")]
    cfg.region_pathways = [RegionPathway(from_region="pad", to_region="msn", density=1.0, weight_mean=1.0,
                                         weight_jitter=0.0, plastic=False, receptor="gaba_a")]
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    assert b.is_initialized
    j = int(b.region_manager.indices("msn")[0])
    params = dict(cell["params"])
    if C is not None:
        params["C"] = float(C)
    for name, value in params.items():
        getattr(b, "cp_izh_" + name)[j] = np.float32(value)
    b.cp_membrane_potential_v[j] = np.float32(cell["v0"])
    b.cp_recovery_variable_u[j] = np.float32(cell["u0"])
    # use the recorded (already decayed) conductances as given: the organ's bounded kernel saw exactly these
    b._cached_decay_e = 1.0
    b._cached_decay_i = 1.0
    return b, j


def _replay(key, **kw):
    cell = _META["cells"][key]
    b, j = _cell_bridge(cell, **kw)
    g_e, g_i, i_other = _F[key + "_g_e"], _F[key + "_g_i"], _F[key + "_I_other"]
    n = len(g_i)
    v = np.empty(n, np.float32)
    fired = np.zeros(n, bool)
    for t in range(n):
        b.cp_conductance_g_e[j] = g_e[t]
        b.cp_conductance_g_i[j] = g_i[t]
        b.cp_external_input_current[j] = i_other[t]
        b._run_one_simulation_step()
        v[t] = _host(b.cp_membrane_potential_v)[j]
        fired[t] = bool(_host(b.cp_firing_states)[j])
    return v, fired, b, cell


# --- T0: the breaching cells ------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", CELLS)
def test_flag_off_reproduces_organ_lock_in(key):
    """Flag off: the one-neuron replay IS the organ trace (exact v_min and spike count), and it is the lock-in:
    v below -150 mV and a spike on about every second step for the rest of the 2755-step run."""
    v, fired, b, cell = _replay(key)
    organ = cell["organ"]
    assert b.substep_calls == 0 and b.refractory_model_resets == 0
    assert float(v.min()) == organ["v_min"]
    assert int(fired.sum()) == organ["spikes"]
    assert int(np.argmax(v < -150.0)) == organ["first_step_lt_m150"]
    assert float(v.min()) < -150.0
    assert int(fired.sum()) >= 1000


@pytest.mark.parametrize("key", CELLS)
def test_null_preset_C_flag_off_no_breach(key):
    """Null: the same cell and input with C at the 50 pF preset and the flag off never leaves [-150, vpeak]."""
    v, fired, _, cell = _replay(key, C=50.0)
    assert float(v.min()) >= -150.0
    assert float(v.max()) <= cell["params"]["vpeak"]
    assert int(fired.sum()) < 100


@pytest.mark.parametrize("rule", ["model_reset", "no_engine_timer"])
@pytest.mark.parametrize("key", CELLS)
def test_substeps_4_bounded(key, rule):
    """Flag n = 4 (either refractory rule): v in [-150, vpeak] on EVERY step, few spikes, and the sub-step path
    ran on every step (engagement counter, not a config echo)."""
    v, fired, b, cell = _replay(key, n_sub=4, rule=rule)
    assert b.core_config.izh_membrane_substeps == 4
    assert b.core_config.izh_substep_refractory_rule == rule
    assert b.substep_calls == len(v)
    assert float(v.min()) >= -150.0
    assert float(v.max()) <= cell["params"]["vpeak"]
    assert int(fired.sum()) <= 50


@pytest.mark.parametrize("rule", ["model_reset", "no_engine_timer"])
def test_substeps_legacy_reset_path_matches_fast_path(rule):
    """The sub-step path feeds both reset paths (fast_spike_reset True / False) the same v/u; with the effective
    refractory steps routed through both, the two give the same trace (s75 replay, n = 4)."""
    v_fast, f_fast, b_fast, _ = _replay("s75", n_sub=4, rule=rule, fast_spike_reset=True)
    v_leg, f_leg, b_leg, _ = _replay("s75", n_sub=4, rule=rule, fast_spike_reset=False)
    np.testing.assert_array_equal(v_fast, v_leg)
    np.testing.assert_array_equal(f_fast, f_leg)
    assert b_fast.refractory_model_resets == b_leg.refractory_model_resets
    np.testing.assert_array_equal(_host(b_fast.cp_refractory_timers), _host(b_leg.cp_refractory_timers))


@pytest.mark.parametrize("n_sub,rule", [(1, "model_reset"), (4, "model_reset"), (4, "no_engine_timer")])
def test_refractory_rules_on_a_strongly_driven_fs_cell(n_sub, rule):
    """The pad FS cell driven at 1200 pA (the organ's selector_reset level) reaches vpeak within 1 ms of a spike.
    Off: one spike per 2 steps (engine timer). Rule (a): the same transmitted rate, the extra in-refractory
    crossings become counted model resets. Rule (b): no engine timer, so it fires on consecutive steps."""
    b, _ = _cell_bridge(_META["cells"]["s72"], n_sub=None if n_sub == 1 else n_sub, rule=rule)
    pad = int(b.region_manager.indices("pad")[0])
    b.cp_external_input_current[pad] = np.float32(1200.0)
    fired = []
    for _ in range(60):
        b._run_one_simulation_step()
        fired.append(bool(_host(b.cp_firing_states)[pad]))
    fired = np.asarray(fired[10:])
    consecutive = int((fired[1:] & fired[:-1]).sum())
    if n_sub == 1 or rule == "model_reset":
        assert consecutive == 0
        assert 20 <= int(fired.sum()) <= 26
    if n_sub == 1:
        assert b.substep_calls == 0 and b.refractory_model_resets == 0
    elif rule == "model_reset":
        assert b.substep_calls == 60 and b.refractory_model_resets >= 20
    else:
        assert b.substep_calls == 60 and b.refractory_model_resets == 0
        assert consecutive >= 40


# --- T0: the sub-step function itself ---------------------------------------------------------------------------

def _arrays(n, **over):
    xp, _ = get_backend()
    base = dict(v=-60.0, u=0.0, C=50.0, k=1.0, vr=-80.0, vt=-25.0, a=0.01, b=-20.0, I=0.0)
    base.update(over)
    return {k: xp.full(n, val, dtype=xp.float32) if np.isscalar(val) else xp.asarray(val, dtype=xp.float32)
            for k, val in base.items()}


def _call_sub(A, n_sub, thr, c, d, refr=None):
    return izhikevich2007_substep_update(A["v"], A["u"], A["C"], A["k"], A["vr"], A["vt"], A["a"], A["b"], A["I"],
                                         1.0, n_sub, thr, c, d, refr)


def test_one_substep_equals_single_step():
    A = _arrays(3, v=[-80.0, -60.0, -30.0], I=[0.0, 100.0, 400.0])
    xp, _ = get_backend()
    thr = xp.full(3, 1e9, dtype=xp.float32)
    c = xp.full(3, -55.0, dtype=xp.float32)
    d = xp.full(3, 150.0, dtype=xp.float32)
    v1, u1, n_res = _call_sub(A, 1, thr, c, d)
    v0, u0 = izhikevich2007_dynamics_update(A["v"], A["u"], A["C"], A["k"], A["vr"], A["vt"], A["a"], A["b"],
                                            A["I"], 1.0)
    np.testing.assert_array_equal(_host(v1), _host(v0))
    np.testing.assert_array_equal(_host(u1), _host(u0))
    assert int(_host(n_res)) == 0


def test_detection_uses_the_threshold_array_not_vpeak():
    """Two identical neurons, thresholds 0 and 1e9: the first stops advancing at the first sub-step whose v
    reaches 0 mV (its v is that sub-step's value), the second integrates all four sub-steps."""
    xp, _ = get_backend()
    A = _arrays(2, v=-10.0, I=2000.0)
    thr = xp.asarray([0.0, 1e9], dtype=xp.float32)
    c = xp.full(2, -55.0, dtype=xp.float32)
    d = xp.full(2, 150.0, dtype=xp.float32)
    v4, u4, _ = _call_sub(A, 4, thr, c, d)
    # manual sub-steps
    v, u = A["v"], A["u"]
    vals = []
    for _ in range(4):
        v, u = izhikevich2007_dynamics_update(v, u, A["C"], A["k"], A["vr"], A["vt"], A["a"], A["b"], A["I"], 0.25)
        vals.append((_host(v).copy(), _host(u).copy()))
    first = next(i for i, (vv, _) in enumerate(vals) if vv[0] >= 0.0)
    assert first < 3
    assert _host(v4)[0] == vals[first][0][0] and _host(u4)[0] == vals[first][1][0]
    assert _host(v4)[1] == vals[3][0][1] and _host(u4)[1] == vals[3][1][1]


def test_rule_a_refractory_crosser_gets_model_reset_and_is_counted():
    """refractory_mask (rule (a)): a refractory crosser is reset to c with u += d at the sub-step it reaches
    threshold and keeps integrating; a non-refractory crosser stops at/above threshold for the bridge to fire.
    Checked against a hand-rolled sub-step loop, value for value."""
    xp, _ = get_backend()
    A = _arrays(2, v=20.0, I=3000.0)
    thr = xp.full(2, 40.0, dtype=xp.float32)
    c = xp.full(2, -55.0, dtype=xp.float32)
    d = xp.full(2, 150.0, dtype=xp.float32)
    refr = xp.asarray([True, False])
    v4, u4, n_res = _call_sub(A, 4, thr, c, d, refr)
    v4, u4 = _host(v4), _host(u4)
    # hand-rolled reference with the same kernel
    v, u = A["v"], A["u"]
    resets, stopped, v_stop, u_stop = 0, False, None, None
    for _ in range(4):
        v, u = izhikevich2007_dynamics_update(v, u, A["C"], A["k"], A["vr"], A["vt"], A["a"], A["b"], A["I"], 0.25)
        hv, hu = _host(v).copy(), _host(u).copy()
        if not stopped and hv[1] >= 40.0:
            stopped, v_stop, u_stop = True, hv[1], hu[1]
        if hv[0] >= 40.0:
            hv[0], hu[0] = np.float32(-55.0), np.float32(hu[0] + np.float32(150.0))
            resets += 1
        v, u = xp.asarray(hv), xp.asarray(hu)
    assert resets >= 1
    assert int(_host(n_res)) == resets
    assert v4[0] == _host(v)[0] and u4[0] == _host(u)[0]
    assert v4[0] < 40.0
    assert stopped and v4[1] == v_stop and u4[1] == u_stop and v4[1] >= 40.0


# --- T1: default path unchanged -----------------------------------------------------------------------------------

def test_default_is_one_and_rule_model_reset():
    cfg = CoreSimConfig()
    assert cfg.izh_membrane_substeps == 1
    assert cfg.izh_substep_refractory_rule == "model_reset"
    d = cfg.to_dict()
    assert d["izh_membrane_substeps"] == 1 and d["izh_substep_refractory_rule"] == "model_reset"


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "4"])
def test_config_rejects_bad_substeps(bad):
    with pytest.raises(ValueError):
        CoreSimConfig(izh_membrane_substeps=bad)


def test_config_rejects_bad_rule():
    with pytest.raises(ValueError):
        CoreSimConfig(izh_substep_refractory_rule="silent")


def test_config_roundtrip_keeps_fields():
    cfg = CoreSimConfig(izh_membrane_substeps=4, izh_substep_refractory_rule="no_engine_timer")
    back = CoreSimConfig(**{k: v for k, v in cfg.to_dict().items()
                            if k in ("izh_membrane_substeps", "izh_substep_refractory_rule")})
    assert back.izh_membrane_substeps == 4 and back.izh_substep_refractory_rule == "no_engine_timer"


def test_substeps_refuse_non_izhikevich_model():
    """n > 1 with a non-Izhikevich model fails the build loudly (the bridge's initialisation catches the
    ValueError, logs it CRITICAL and leaves is_initialized False) and the settings helper raises directly."""
    from tests.test_bounded_conductance_integration import _base_cfg, _region as _breg
    from sim import NeuronType
    cfg = _base_cfg(42)
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.izh_membrane_substeps = 4
    with pytest.raises(ValueError, match="IZHIKEVICH"):
        SimulationBridge._izh_substep_settings(cfg)
    cfg.brain_regions = [_breg("a", 2, 1.0, NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL),
                         _breg("b", 2, 0.0, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON)]
    cfg.region_pathways = [RegionPathway(from_region="b", to_region="a", density=1.0, weight_mean=1.0,
                                         weight_jitter=0.0, plastic=False)]
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    assert not b.is_initialized
    cfg.izh_membrane_substeps = 1
    assert SimulationBridge._izh_substep_settings(cfg) == (1, "model_reset")


def test_megakernel_declines_substeps(monkeypatch):
    """With every other megakernel guard satisfied (GPU faked, read-only regime, bounded off) the sub-step guard
    alone flips dispatch: n = 1 dispatches, n = 4 falls through to the Python path that implements it."""
    import sim.bridge as SB
    b, _ = _cell_bridge(_META["cells"]["s72"])
    cfg = b.core_config
    monkeypatch.setattr(SB, "is_gpu_backend", lambda: True)
    cfg.enable_step_megakernel = True
    cfg.fast_spike_reset = True
    cfg.read_only_fast_step = True
    cfg.enable_bounded_conductance_integration = False
    cfg.izh_membrane_substeps = 1
    assert b._step_megakernel_can_dispatch() is True
    cfg.izh_membrane_substeps = 4
    assert b._step_megakernel_can_dispatch() is False


def test_explicit_one_is_byte_identical_to_default_and_prefix_engine():
    """A short full-bridge run (the bounded-conductance reference network: AMPA/GABA_A, GABA_B, NMDA, OU) with
    izh_membrane_substeps=1 set explicitly is byte-identical to the default build, and both still reproduce the
    pre-fix engine hash captured on main at 3c2907b85 (numpy)."""
    from tests.test_bounded_conductance_integration import (
        _REFERENCE_HASHES_FROM_MAIN, _backend_name, _build_reference_network_bridge, _run_and_hash)
    # build + run one bridge at a time: the build and the OU noise both draw on the seeded global stream
    b_default = _build_reference_network_bridge(seed=42, set_flag_explicitly=None)
    h_default, r_default, v_default = _run_and_hash(b_default, 60)
    b_one = _build_reference_network_bridge(seed=42, set_flag_explicitly=None)
    b_one.core_config.izh_membrane_substeps = 1
    b_one.core_config.izh_substep_refractory_rule = "no_engine_timer"   # ignored at 1
    h_one, r_one, v_one = _run_and_hash(b_one, 60)
    assert int(r_default.sum()) > 0
    assert h_default == h_one
    np.testing.assert_array_equal(v_default, v_one)
    assert b_default.substep_calls == 0 and b_one.substep_calls == 0
    ref = _REFERENCE_HASHES_FROM_MAIN.get(_backend_name())
    if ref is not None:
        assert h_default == ref


@pytest.mark.skip(reason="BG-organ env switch (BRAIN_BG_IZH_SUBSTEPS) lives on research/bg-tonic-loop, not main; "
                         "on main the whole-brain switch SIM_STABLE_INTEGRATION covers it "
                         "(tests/test_stable_integration_switch.py)")
def test_bg_switch_reaches_core_config(monkeypatch):
    """BRAIN_BG_IZH_SUBSTEPS / BRAIN_BG_IZH_SUBSTEP_RULE set the fields through build_selector_bridge; unset, the
    defaults stay."""
    from research.runners._vocal_action_selector_gate import build_selector_bridge
    for k in ("BRAIN_BG_IZH_SUBSTEPS", "BRAIN_BG_IZH_SUBSTEP_RULE", "BRAIN_BG_PRESET_CELLS", "BRAIN_BG_BOUNDED_COND"):
        monkeypatch.delenv(k, raising=False)
    off = build_selector_bridge(9)
    assert off.core_config.izh_membrane_substeps == 1
    assert off.core_config.izh_substep_refractory_rule == "model_reset"
    monkeypatch.setenv("BRAIN_BG_IZH_SUBSTEPS", "4")
    monkeypatch.setenv("BRAIN_BG_IZH_SUBSTEP_RULE", "no_engine_timer")
    on = build_selector_bridge(9)
    assert on.core_config.izh_membrane_substeps == 4
    assert on.core_config.izh_substep_refractory_rule == "no_engine_timer"
