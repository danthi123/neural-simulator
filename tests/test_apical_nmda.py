"""CI guard -- the apical NMDA conductance (cs-race f1j section E, research/coordination/cs-race-f1j-PACKET.md).

cfg.enable_apical_nmda: a per-cell gating state cp_apical_nmda_s on the BDSP apical rises from that cell's own apical
drive (cp_bdsp_apical_drive, the runner's write) and decays with apical_nmda_tau_ms; the conductance
g_N = apical_nmda_g0 * s / s_full is Mg-blocked by the engine's Jahr & Stevens expression at nmda_mg_concentration and
applied against (E_e - v_apical) as a BOUNDED exponential-Euler step. Pins (E5):
  1. flag absent == flag False, and both equal the engine BEFORE this edit (reference hash captured on numpy at
     origin/main 9f2b083c4 with this file's builder);
  2. flag on, g0 > 0, zero drive everywhere: byte-identical to flag off and s == 0.0 exactly (inert at rest);
  3. flag on, g0 == 0 (APV) with a drive: every state array byte-identical to flag off, while s rises to s_full on the
     driven cells and stays exactly 0.0 on the undriven ones (the receptor is blocked, not the input);
  4. g_N > 0 only on driven cells; after the drive ends g_N decays by exactly exp(-dt/tauN) per step; under sustained
     full drive g_N converges to g0;
  5. the Mg block equals the Jahr & Stevens formula (slope 0.062, 3.57 mM) and the block implied by the engine's
     fused_nmda_update_and_current at the same v and nmda_mg_concentration, so the two expressions cannot drift apart;
  6. one step never crosses E_e, at g_N*B in {0.1, 10, 1e4} (v == E_e exactly at 1e4 from below);
  7. every FAIL-LOUDLY case of E2 raises; the megakernel refuses the step; the gating survives a checkpoint round trip;
  8. with the f1i apical GABA routing on, the GABA and NMDA steps are both computed from v0 and each equals its formula
     (the order is pinned; no aggregate bound is claimed);
  9. replay match: one slot driven 120 for 150 ms then 0, no SOM, (g0, tauN) = (8, 50), against the open-loop replay's
     reference model ported below: max |dv| <= 0.01 mV over 800 ms.
"""
import hashlib
import math
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
EXC = np.arange(0, 24, dtype=np.int64)
INH = np.arange(24, 40, dtype=np.int64)
DRIVEN = EXC[4:12]                                 # the 8 cells whose apical arbor is driven (one "slot")
UNDRIVEN = np.setdiff1d(np.arange(N), DRIVEN)
PULSE = 120.0                                      # the f1h / f1i apical pulse (gap4 GO value)
N_STEPS = 80
PULSE_STEPS = 30
# The f1h circuit's frozen gap4 GO values for the bistable BDSP apical (research/runners/cs_race_f1.py GAP4_GO).
GAP4 = dict(coincidence_plateau_self_regen=2.0, coincidence_plateau_v_hold=-35.0, apical_kir_g=1.0,
            bdsp_learning_rate=0.0, coincidence_plateau_v_hold_k=0.2, apical_kir_E_K=-90.0, apical_kir_vhalf=-50.0,
            apical_kir_k=8.0, apical_tau_ms=15.0, apical_E_rest=-65.0, bdsp_v_apical_scale=0.05)
NMDA = dict(g0=8.0, tau_ms=50.0, drive_ref=PULSE)
# Flag-off state hash captured from the engine BEFORE the apical-NMDA edit (origin/main 9f2b083c4), numpy backend, by
# running _state_hash(_run(_build(42))) with this file's builder against that tree's sim/ package.
_REF_HASH_PRE_EDIT_NUMPY = "731c173dd7fc0ff0539b5164eed3633232b5669a7f25673e884aad085051d5ea"


def _dense(pre, post, w, **extra):
    pre_r = np.repeat(np.asarray(pre, dtype=np.int64), len(post))
    post_r = np.tile(np.asarray(post, dtype=np.int64), len(pre))
    d = {"pre_indices": pre_r, "post_indices": post_r,
         "initial_weights": np.full(pre_r.shape[0], float(w), dtype=np.float32), "plastic": False}
    d.update(extra)
    return d


def _plan(*, som=False):
    plan = {"ee": _dense(EXC, EXC, 0.6), "ei": _dense(EXC, INH, 1.5), "ie": _dense(INH[:8], EXC, 3.0)}
    if som:                                         # the f1i apical GABA_A routing, SOM-like cells -> every EXC apical
        plan["som_ap"] = _dense(INH[8:], EXC, 4.0, receptor="gaba_a_apical")
    else:
        plan["ie2"] = _dense(INH[8:], EXC, 4.0)
    return plan


def _build(seed=42, *, nmda=None, bdsp=True, bistable=True, two_comp=False, micro=False, som=False,
           apical_drive=True, cfg_extra=None):
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
    cfg.bdsp_apical_bistable = bool(bistable)
    for k, v in GAP4.items():
        setattr(cfg, k, v)
    cfg.enable_two_compartment_dap = bool(two_comp)
    if micro:
        cfg.enable_bdsp_microcircuit = True
    if som:
        cfg.enable_apical_gaba_routing = True
    if nmda is not None:
        cfg.enable_apical_nmda = True
        for k, v in nmda.items():
            setattr(cfg, "apical_nmda_" + k, v)
    for k, v in (cfg_extra or {}).items():
        setattr(cfg, k, v)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    b.inject_explicit_wiring(_plan(som=som), output_inhibitory_indices=INH)
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


def _set_drive(b, on, cells=DRIVEN, amp=PULSE):
    d = np.zeros(N, dtype=np.float32)
    if on:
        d[cells] = amp
    b.cp_bdsp_apical_drive = b.xp.asarray(d)


def _run(b, n_steps=N_STEPS, pulse=(0, PULSE_STEPS), amp=PULSE, record=None):
    raster = []
    for t in range(n_steps):
        _set_drive(b, pulse[0] <= t < pulse[1], amp=amp)
        _step(b)
        raster.append(np.asarray(to_host(b.cp_firing_states)).astype(np.uint8))
        if record is not None:
            record(b, t)
    return b, np.stack(raster)


_HASHED = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_conductance_g_e", "cp_conductance_g_i",
           "cp_v_apical", "cp_bdsp_P", "cp_bdsp_Pbar", "cp_bdsp_E", "cp_bdsp_B")


def _state_hash(run):
    b, raster = run
    h = hashlib.sha256()
    h.update(raster.tobytes())
    for name in _HASHED:
        arr = getattr(b, name)
        h.update(name.encode())
        h.update(b"None" if arr is None else np.asarray(to_host(arr)).tobytes())
    h.update(np.asarray(to_host(b.cp_connections.data)).tobytes())
    return h.hexdigest()


def _h(arr):
    return np.asarray(to_host(arr))


def _gN(b):
    from sim.bridge import apical_nmda_conductance
    return _h(apical_nmda_conductance(b.cp_apical_nmda_s, b.core_config, b.xp))


def _s_full(cfg):
    from sim.bridge import apical_nmda_s_full
    return apical_nmda_s_full(cfg)


# ---------------------------------------------------------------- 1. byte identity when off
def test_flag_absent_equals_flag_false():
    r_abs = _run(_build(42))
    r_off = _run(_build(42, cfg_extra={"enable_apical_nmda": False}))
    assert int(r_abs[1].sum()) > 0, "degenerate silent network makes the hash meaningless"
    assert _state_hash(r_abs) == _state_hash(r_off)
    assert r_abs[0].cp_apical_nmda_s is None and r_off[0].cp_apical_nmda_s is None


def test_flag_off_matches_pre_edit_engine():
    _, name = get_backend()
    if str(name).lower() != "numpy":
        pytest.skip("the pre-edit reference hash was captured on numpy only")
    assert _state_hash(_run(_build(42))) == _REF_HASH_PRE_EDIT_NUMPY


# ---------------------------------------------------------------- 2. inert at rest
def test_flag_on_zero_drive_is_inert():
    r_off = _run(_build(42), amp=0.0)
    b_on = _build(42, nmda=NMDA)
    assert b_on.cp_apical_nmda_s is not None and int(b_on.cp_apical_nmda_s.shape[0]) == N
    r_on = _run(b_on, amp=0.0)
    assert _state_hash(r_on) == _state_hash(r_off)
    s = _h(r_on[0].cp_apical_nmda_s)
    assert s.dtype == np.float32 and np.all(s == 0.0)
    assert np.all(_gN(r_on[0]) == 0.0)


# ---------------------------------------------------------------- 3. APV: the receptor is blocked, not the input
def test_apv_is_byte_identical_and_the_gating_still_runs():
    r_off = _run(_build(42))
    seen = {}

    def rec(b, t):
        if t == PULSE_STEPS - 1:
            seen["s_end"] = _h(b.cp_apical_nmda_s).copy()
        seen.setdefault("s_max", np.zeros(N, dtype=np.float32))
        seen["s_max"] = np.maximum(seen["s_max"], _h(b.cp_apical_nmda_s))
        assert np.all(_gN(b) == 0.0)

    b_apv = _build(42, nmda=dict(NMDA, g0=0.0))
    r_apv = _run(b_apv, record=rec)
    assert _state_hash(r_apv) == _state_hash(r_off)
    s_full = _s_full(b_apv.core_config)
    np.testing.assert_allclose(seen["s_end"][DRIVEN], s_full, rtol=1e-5)
    assert np.all(seen["s_max"][UNDRIVEN] == 0.0)


# ---------------------------------------------------------------- 4. conductance: where, decay, convergence
def test_conductance_only_on_driven_cells_decays_exactly_and_converges_to_g0():
    g_hist = []
    b = _build(42, nmda=NMDA)
    _run(b, record=lambda bb, t: g_hist.append(_gN(bb).copy()))
    G = np.stack(g_hist)                                    # (T, N)
    assert np.all(G[:, UNDRIVEN] == 0.0)
    assert np.all(G[:, DRIVEN] > 0.0)
    np.testing.assert_allclose(G[PULSE_STEPS - 1, DRIVEN], NMDA["g0"], rtol=1e-5)   # sustained full drive -> g0
    dN = math.exp(-b.core_config.dt_ms / NMDA["tau_ms"])
    ratio = G[PULSE_STEPS:, DRIVEN] / G[PULSE_STEPS - 1:-1, DRIVEN]
    np.testing.assert_allclose(ratio, dN, rtol=1e-6)       # after the drive ends: exactly exp(-dt/tauN) per step


def test_rise_is_bounded_at_any_input():
    b = _build(42, nmda=NMDA)
    _run(b, n_steps=5, pulse=(0, 5), amp=1e3)              # u = 1e3/120: the bounded rise saturates, never passes 1
    s = _h(b.cp_apical_nmda_s)
    assert np.all(s[DRIVEN] <= 1.0) and np.all(s[DRIVEN] > 0.99) and np.all(s[UNDRIVEN] == 0.0)


# ---------------------------------------------------------------- 5. the Mg block
def _jahr_stevens(v, mg):
    return 1.0 / (1.0 + (mg / 3.57) * math.exp(-0.062 * v))


def test_mg_block_formula_and_kernel_agreement():
    from sim.bridge import apical_nmda_mg_block
    from sim.kernels import fused_nmda_update_and_current
    xp, _ = get_backend()
    mg = float(CoreSimConfig().nmda_mg_concentration)
    assert mg == 1.0
    for v in (-80.0, -24.0, 0.0):
        b = float(_h(apical_nmda_mg_block(xp.asarray(np.float32([v])), mg, xp))[0])
        assert b == pytest.approx(_jahr_stevens(v, mg), rel=1e-6), v
    for v in (-80.0, -24.0, -5.0):
        vv = xp.asarray(np.float32([v]))
        one, zero = xp.asarray(np.float32([1.0])), xp.asarray(np.float32([0.0]))
        _, _, i_nmda = fused_nmda_update_and_current(one, zero, 1.0, 1.0, vv, 0.0, mg)
        b_kernel = float(_h(i_nmda)[0]) / (0.0 - v)
        b_engine = float(_h(apical_nmda_mg_block(vv, mg, xp))[0])
        assert b_engine == pytest.approx(b_kernel, rel=2e-7), v
    assert _jahr_stevens(-24.0, 1.0) != pytest.approx(1.0 / (1.0 + (1.0 / 3.57) * math.exp(0.08 * 24.0)))  # not 0.08


# ---------------------------------------------------------------- 6. the bounded term
def _one_step_setup(v0, gB, *, som=False, g_ap=0.0):
    """A bridge with the plain (non-bistable) BDSP apical, one step from v0 with the leak term exactly 0, a gating state
    chosen so g_N*B(v0) is about gB this step, and (optionally) a routed apical GABA conductance g_ap."""
    from sim.bridge import apical_nmda_mg_block
    b = _build(42, nmda=dict(NMDA, g0=1.0), bistable=False, som=som)
    _step(b)                                                # allocate cp_v_apical
    xp = b.xp
    cfg = b.core_config
    s_full = _s_full(cfg)
    B0 = float(_h(apical_nmda_mg_block(xp.asarray(np.float32([v0])), float(cfg.nmda_mg_concentration), xp))[0])
    cfg.apical_nmda_g0 = float(gB / B0)
    dN = math.exp(-cfg.dt_ms / cfg.apical_nmda_tau_ms)
    b.cp_firing_states[:] = False
    b.cp_prev_firing_states = xp.zeros(N, dtype=bool)      # no spike this step
    b.cp_v_apical = xp.full(N, v0, dtype=xp.float32)
    b.cp_bdsp_apical_drive = xp.full(N, v0 - float(cfg.apical_E_rest), dtype=xp.float32)   # leak term == 0
    b.cp_apical_nmda_s = xp.full(N, s_full / dN, dtype=xp.float32)
    if som:
        decay_i = float(np.exp(-cfg.dt_ms / float(cfg.syn_tau_g_i)))
        b.cp_conductance_g_apical_gaba = xp.full(N, g_ap / decay_i, dtype=xp.float32)
    return b


def _expected_steps(b, v0):
    """The NMDA and GABA steps as the packet's E4 / f1i E5 formulas, from v0 and the engine's post-step gating."""
    from sim.bridge import apical_nmda_mg_block
    xp = b.xp
    cfg = b.core_config
    v0a = xp.full(N, v0, dtype=xp.float32)
    tau = xp.float32(cfg.apical_tau_ms)
    dt = xp.float32(cfg.dt_ms)
    gN = apical_nmda_conductance_of(b)
    B = apical_nmda_mg_block(v0a, float(cfg.nmda_mg_concentration), xp)
    ee = xp.float32(cfg.syn_reversal_potential_e)
    ns = ((ee - v0a) * (1.0 - xp.exp(xp.clip(-((gN * B) * dt) / tau, -80.0, 0.0)))).astype(xp.float32)
    gs = None
    if b.cp_conductance_g_apical_gaba is not None:
        ei = xp.float32(cfg.syn_reversal_potential_i)
        gs = ((ei - v0a) * (1.0 - xp.exp(xp.clip(-(b.cp_conductance_g_apical_gaba * dt) / tau, -80.0, 0.0))))
        gs = gs.astype(xp.float32)
    return ns, gs


def apical_nmda_conductance_of(b):
    from sim.bridge import apical_nmda_conductance
    return apical_nmda_conductance(b.cp_apical_nmda_s, b.core_config, b.xp)


@pytest.mark.parametrize("gB", [0.1, 10.0, 1e4])
@pytest.mark.parametrize("v0", [-65.0, -40.0, -85.0, 15.0])
def test_one_step_never_crosses_e_e(gB, v0):
    b = _one_step_setup(v0, gB)
    _step(b)
    v1 = _h(b.cp_v_apical)
    e_e = float(b.core_config.syn_reversal_potential_e)
    if v0 < e_e:
        assert np.all(v1 <= e_e) and np.all(v1 > v0)
    else:
        assert np.all(v1 >= e_e) and np.all(v1 < v0)
    ns, _ = _expected_steps(b, v0)
    np.testing.assert_array_equal(v1, (np.float32(v0) + _h(ns)).astype(np.float32))
    if gB == 1e4 and v0 < e_e:
        assert np.all(v1 == np.float32(e_e))                # reaches E_e exactly, never past it
        assert abs(gB * (e_e - v0) / float(b.core_config.apical_tau_ms)) > 100.0   # explicit Euler would overshoot


def test_gaba_and_nmda_steps_both_from_v0_in_pinned_order():
    v0 = -30.0
    b = _one_step_setup(v0, 2.0, som=True, g_ap=3.0)
    assert b.cp_apical_gaba_synapse_mask is not None
    _step(b)
    v1 = _h(b.cp_v_apical)
    ns, gs = _expected_steps(b, v0)
    want = ((np.float32(v0) + _h(gs)).astype(np.float32) + _h(ns)).astype(np.float32)
    np.testing.assert_array_equal(v1, want)
    # each term is its own formula from v0: not the NMDA step from (v0 + GABA step)
    assert float(np.abs(_h(ns)).min()) > 0.0 and float(np.abs(_h(gs)).min()) > 0.0


# ---------------------------------------------------------------- 7. fail loudly, megakernel, checkpoint
@pytest.mark.parametrize("kw,match", [
    (dict(bdsp=False), "enable_bdsp"),
    (dict(two_comp=True), "enable_two_compartment_dap"),
    (dict(micro=True), "enable_bdsp_microcircuit"),
    (dict(nmda=dict(NMDA, g0=-1.0)), "apical_nmda_g0"),
    (dict(nmda=dict(NMDA, tau_ms=0.0)), "apical_nmda_tau_ms"),
    (dict(nmda=dict(NMDA, drive_ref=0.0)), "apical_nmda_drive_ref"),
    (dict(nmda=dict(NMDA, rise_per_ms=0.0)), "apical_nmda_rise_per_ms"),
    (dict(cfg_extra={"nmda_mg_concentration": -0.5}), "nmda_mg_concentration"),
])
def test_misuse_raises_at_build(kw, match):
    kw = dict(kw)
    kw.setdefault("nmda", NMDA)
    with pytest.raises(ValueError, match=match):
        _build(42, **kw)


def test_validation_also_at_wiring_time():
    b = _build(42, nmda=NMDA)
    b.core_config.enable_two_compartment_dap = True
    with pytest.raises(ValueError, match="enable_two_compartment_dap"):
        b.inject_explicit_wiring(_plan(), output_inhibitory_indices=INH)


def test_missing_apical_drive_raises_at_step():
    b = _build(42, nmda=NMDA, apical_drive=False)
    b.strict_step_errors = True
    with pytest.raises(RuntimeError, match="cp_bdsp_apical_drive is None"):
        _step(b)


@pytest.mark.parametrize("field", ["enable_apical_nmda", "enable_bdsp"])
def test_flag_turned_off_after_allocation_raises_at_step(field):
    b = _build(42, nmda=NMDA)
    _step(b)
    setattr(b.core_config, field, False)
    b.strict_step_errors = True
    with pytest.raises(RuntimeError, match="cp_apical_nmda_s is present"):
        _step(b)


def test_config_broken_after_build_raises_at_step():
    b = _build(42, nmda=NMDA)
    b.core_config.apical_nmda_tau_ms = -1.0
    b.strict_step_errors = True
    with pytest.raises(ValueError, match="apical_nmda_tau_ms"):
        _step(b)


def test_megakernel_refuses_the_step():
    b = _build(42, nmda=NMDA)
    b.core_config.enable_step_megakernel = True
    assert b._step_megakernel_can_dispatch() is False


def test_gating_survives_checkpoint(tmp_path):
    b = _build(42, nmda=NMDA)
    _run(b, n_steps=PULSE_STEPS + 5)
    s = _h(b.cp_apical_nmda_s).copy()
    assert float(s.max()) > 0.0
    ck = tmp_path / "apnmda.simstate.h5"
    assert b.save_checkpoint(str(ck)) is True
    r = _build(43, nmda=NMDA)
    assert float(_h(r.cp_apical_nmda_s).max()) == 0.0
    assert r.load_checkpoint(str(ck)) is True
    np.testing.assert_array_equal(_h(r.cp_apical_nmda_s), s)
    off = _build(43)
    assert off.load_checkpoint(str(ck)) is True             # the checkpoint carries its config and its gating ...
    assert off.core_config.enable_apical_nmda is True and off.core_config.apical_nmda_g0 == NMDA["g0"]
    np.testing.assert_array_equal(_h(off.cp_apical_nmda_s), s)
    off.core_config.enable_apical_nmda = False              # ... and a step with the flag turned off refuses it
    off.strict_step_errors = True
    with pytest.raises(RuntimeError, match="cp_apical_nmda_s is present"):
        _step(off)


# ---------------------------------------------------------------- 9. the replay match
def _replay_reference(v0, drive, u, g0, tau_n, T):
    """The f1j open-loop replay's reference model (scratch core.replay, kin 'sat', integ 'bounded', no SOM), one cell.

    BDSP apical: leak, drive, the regen latch and the KIR down state (the f1h gap4 values), plus the A-NMDA term:
    saturating gating (per-step rise 0.5 at full input), decay tauN, normalized so the conductance at the input's end
    is g0, Mg block 1 mM / slope 0.062, reversal 0 mV, bounded exponential-Euler step. float64.
    """
    TAU, ER, SR, VH, VHK, EE, KG, EK, KVH, KK = 15.0, -65.0, 2.0, -35.0, 0.2, 0.0, 1.0, -90.0, -50.0, 8.0
    dN = np.exp(-1.0 / tau_n)
    alpha = 0.5
    scale = g0 / (alpha / (1 - dN + alpha * dN))
    v, s = float(v0), 0.0
    out = np.empty(T)
    for t in range(T):
        dvb = (-(v - ER) + drive[t] + SR / (1 + np.exp(-VHK * (v - VH))) * (EE - v)
               + KG / (1 + np.exp((v - KVH) / KK)) * (EK - v))
        s = s * dN
        s = s + alpha * u[t] * (1 - s)
        geff = scale * s / (1.0 + (1.0 / 3.57) * np.exp(-0.062 * v))
        ns = (EE - v) * (1 - np.exp(np.clip(-geff / TAU, -80, 0)))
        v = v + (1.0 / TAU) * dvb + ns
        out[t] = v
    return out


def test_replay_match_single_slot():
    b = _build(42, nmda=dict(g0=8.0, tau_ms=50.0, drive_ref=PULSE))
    for _ in range(300):                                    # settle the down state with no drive (the runner's warm-up)
        _set_drive(b, False)
        _step(b)
    assert np.all(_h(b.cp_apical_nmda_s) == 0.0)
    v_start = _h(b.cp_v_apical)[DRIVEN].astype(np.float64)
    T = 800
    trace = np.empty((T, DRIVEN.size))
    for t in range(T):
        _set_drive(b, t < 150)
        _step(b)
        trace[t] = _h(b.cp_v_apical)[DRIVEN]
    drive = np.where(np.arange(T) < 150, PULSE, 0.0)
    u = (drive > 0).astype(float)
    for j in range(DRIVEN.size):
        ref = _replay_reference(v_start[j], drive, u, 8.0, 50.0, T)
        assert float(np.abs(trace[:, j] - ref).max()) <= 0.01
    assert trace[149].min() > -35.0 and trace[799].min() > -35.0      # the plateau latched and held (no SOM)
