"""CI guard — gap#4 (2026-07-18): the on-bridge BTSP plateau-gated one-shot credit block in
SimulationBridge._run_one_simulation_step. Builds a real 16-neuron bridge (pre->post plastic pathway) and checks the
three load-bearing properties of the guarded `enable_btsp` block:
  1. POTENTIATION: a co-active presynaptic input under a dendritic PLATEAU (cp_v_apical above v_hold) potentiates the
     synapse ONE-SHOT (dw > 0).
  2. MOAT: a SILENT apical (cp_v_apical at rest, IS == 0) potentiates NOTHING (dw == 0).
  3. BYTE-IDENTICAL OFF: enable_btsp=False => the block is unreached, cp_btsp_pre_elig stays None, no weight moves.
CPU/numpy, self-contained (real bridge). Mirrors the scratch smoke that first validated the wiring.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge

xp, _ = get_backend()


def _build(enable_btsp, seed=42):
    regions = [
        BrainRegion(name="pre", n_neurons=8, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="post", n_neurons=8, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [RegionPathway(from_region="pre", to_region="post", density=1.0,
                              weight_mean=0.5, weight_jitter=0.0, plastic=True)]
    cfg = CoreSimConfig(seed=seed)
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_input_divisive_norm", "enable_nmda"):
        setattr(cfg, f, False)
    cfg.enable_btsp = bool(enable_btsp)
    cfg.btsp_learning_rate = 0.01
    cfg.btsp_elig_tau_ms = 1000.0
    cfg.btsp_w_max = 5.0
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _run(enable_btsp, plateau, seed=42, steps=60):
    sb = _build(enable_btsp, seed)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre"))); post_idx = np.asarray(list(rm.indices("post")))
    n = sb.cp_membrane_potential_v.size
    sb.cp_v_apical = xp.full(n, -65.0, dtype=xp.float32)          # allocate the apical at rest
    w0 = float(xp.asarray(sb.cp_connections.data).sum())
    drive = np.zeros(n, dtype=np.float32); drive[pre_idx] = 900.0  # make PRE fire -> pre-eligibility
    for _ in range(steps):
        sb.cp_external_input_current[:] = xp.asarray(drive)
        sb.cp_v_apical[post_idx] = xp.float32(-20.0 if plateau else -65.0)   # plateau (IS>0) vs silent (IS==0)
        sb._run_one_simulation_step()
    w1 = float(xp.asarray(sb.cp_connections.data).sum())
    elig_alloc = sb.cp_btsp_pre_elig is not None
    return w1 - w0, elig_alloc


def test_onbridge_btsp_potentiates_under_plateau():
    dw, alloc = _run(enable_btsp=True, plateau=True)
    assert dw > 0.1, f"a co-active synapse under a plateau must potentiate one-shot, got dw={dw:.4f}"
    assert alloc, "cp_btsp_pre_elig must be allocated when enable_btsp=True"


def test_onbridge_btsp_moat_silent_apical():
    dw, _ = _run(enable_btsp=True, plateau=False)
    assert abs(dw) < 1e-4, f"a silent apical (IS=0) must not potentiate anything (the moat), got dw={dw:.4f}"


def test_onbridge_btsp_byte_identical_when_off():
    dw, alloc = _run(enable_btsp=False, plateau=True)
    assert abs(dw) < 1e-9, f"enable_btsp=False must move no weight (byte-identical), got dw={dw:.6f}"
    assert not alloc, "cp_btsp_pre_elig must stay None when enable_btsp=False"


def test_onbridge_btsp_behavioral_timescale_via_real_bistable_plateau():
    """The REAL bistable plateau (bistable BDSP apical) makes on-bridge BTSP seconds-long: a HELD plateau potentiates
    ~8x more than a TRANSIENT one (the bistability is load-bearing on the substrate). One seed of the 6-seed GO de-risk."""
    from research.runners._gap4_btsp_onbridge_behavioral_timescale_derisk import run
    r = run(42)
    assert r["held_dw"] >= 0.3, f"held plateau must potentiate one-shot, got {r['held_dw']:.3f}"
    assert r["held_dw"] > 3.0 * max(r["transient_dw"], 1e-6), "the bistable latch must be load-bearing vs transient"
    assert abs(r["moat_dw"]) <= 0.02 * r["held_dw"], "silent apical must not potentiate (the moat)"
    assert abs(r["off_dw"]) < 1e-9, "enable_btsp=False must be byte-identical"
    assert r["held_v_apical_end"] > -35.0, "the held plateau must stay above v_hold (a real seconds-long latch)"
    assert r["transient_v_apical_end"] < -50.0, "the transient plateau must decay back toward rest"
