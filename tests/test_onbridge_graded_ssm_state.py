"""CI guard for gap#1 M1 (2026-07-20): the on-bridge GRADED `cp_ssm_state` integrator realizes the WKV/SSM leaky
recurrence EXACTLY, which is the load-bearing mechanism behind "the on-bridge recurrent LM state beats the fair
interpolated trigram at deep context, 6/6 seeds".

The full LM result is the runner's own sweep (`_emerge_wkv_onbridge_derisk.py --ssm-state --use-ssm-readout`); this
fast CPU test pins the MECHANISM claims that result rests on, so a future `sim/` change cannot silently break it:

  1. the exact mapping  k_leak = 1-decay, shunt = 0  ->  lam_eff = decay, and injecting v/(1-decay) reproduces
     a_t = decay*a_{t-1} + v_t  to float32 precision (the verify-first corr=1.000 claim);
  2. the MEMORYLESS anti-cheat is real: k_leak = 1 (lam = 0) genuinely destroys the integration (state == current
     input only) -- so the control that collapses in the GO table is doing what it claims;
  3. the SHUNT freeze used by the M2 spiking-input path: shunt = -1 -> lam = 1 -> the state HOLDS and ignores inject
     (this is what lets an encode window run without decaying the state).

Guarded per the silent-failure discipline: an "inert / exact" claim belongs in an ASSERTION, not a comment.
"""
import numpy as np
import pytest


def _build(n, k_leak, seed=42):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_selective_ssm_state = True
    cfg.ssm_k_leak = float(k_leak)
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed        # NOTE: cfg.seed is what actually seeds the substrate
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_parameter_heterogeneity = False; cfg.enable_conductance_noise = False
    cfg.brain_regions = [BrainRegion(name="chan", n_neurons=n, exc_fraction=1.0, internal_density=0.05,
                                     exc_weight_mean=1.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                     plastic_internal=False)]
    cfg.region_pathways = []
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, np.asarray(b.region_manager.indices("chan"))


def _drive(b, idx, vals, shunt=0.0):
    from sim.backend import to_host, get_backend
    xp, _ = get_backend()
    n = int(b.core_config.num_neurons)
    inj = np.zeros(n, np.float32); inj[idx] = np.asarray(vals, dtype=np.float32)
    b.cp_ssm_inject[:] = (xp.asarray(inj) if xp is not None else inj)
    b.cp_ssm_shunt[:] = float(shunt)
    b._run_one_simulation_step()
    return np.asarray(to_host(b.cp_ssm_state)).astype(np.float64)[idx]


def test_graded_ssm_state_reproduces_wkv_recurrence_exactly():
    """k_leak = 1-decay, shunt = 0, inject = v/(1-decay)  =>  a_t = decay*a_{t-1} + v_t (float32-exact)."""
    pytest.importorskip("numpy")
    decay = 0.6546                                                # a representative trained uniform decay
    n = 8
    b, idx = _build(n, k_leak=1.0 - decay)
    rng = np.random.default_rng(0)
    vs = rng.normal(0.0, 1.0, size=(12, n))
    a_ref = np.zeros(n)
    for t in range(len(vs)):
        v = np.maximum(vs[t], 0.0)                                # dual-nonneg: a NON-NEGATIVE channel
        got = _drive(b, idx, v / (1.0 - decay), shunt=0.0)
        a_ref = decay * a_ref + v
        assert np.allclose(got, a_ref, atol=2e-3, rtol=2e-3), (
            f"step {t}: on-bridge cp_ssm_state {got[:3]} != WKV recurrence {a_ref[:3]} "
            "-- the exact-mapping claim behind the M1 GO is broken")


def test_memoryless_anticheat_really_destroys_integration():
    """k_leak = 1 => lam = 0 => the state is ONLY the current input (no memory). The GO table's control is real."""
    n = 6
    b, idx = _build(n, k_leak=1.0)
    rng = np.random.default_rng(1)
    prev = None
    for t in range(6):
        v = np.abs(rng.normal(0.0, 1.0, size=n))
        got = _drive(b, idx, v, shunt=0.0)
        assert np.allclose(got, v, atol=2e-3), f"step {t}: lam=0 must give state == inject, got {got[:3]} vs {v[:3]}"
        if prev is not None:
            assert not np.allclose(got, prev, atol=1e-6), "a memoryless state must not carry the previous value"
        prev = got


def test_shunt_minus_one_freezes_the_state():
    """shunt = -1 => lam = clip(1 - k*(1-1),0,1) = 1 => the state HOLDS and IGNORES inject (the M2 encode-window freeze)."""
    decay = 0.7
    n = 5
    b, idx = _build(n, k_leak=1.0 - decay)
    held = _drive(b, idx, np.full(n, 1.0) / (1.0 - decay), shunt=0.0)   # write something
    for _ in range(5):                                                  # now freeze and blast a different inject
        got = _drive(b, idx, np.full(n, 99.0), shunt=-1.0)
        assert np.allclose(got, held, atol=2e-3), (
            f"shunt=-1 must FREEZE the state (got {got[:3]}, expected held {held[:3]}) "
            "-- the M2 encode-window freeze depends on this")
