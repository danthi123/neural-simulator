"""CI guard — M1' (2026-07-25): the DENDRITIC SUSTAINED-SPIKE-COUNT gate on the BTSP write eligibility
(`btsp_win_gate_theta` / `btsp_win_gate_hill_n` + `bridge.reset_btsp_window()`).

The primitive: a per-presynaptic-source BOX-CAR WINDOWED spike COUNT (with an explicit reset) passed through an
ABSOLUTE (NOT network-normalized) Hill gate, applied to the eligibility BEFORE the synaptic summation. Biology:
Kandel 6e Ch13 pp296-298 (per-spine Ca2+ compartmentalisation), Polsky/Mel/Schiller J Neurosci 2009 (a basal
dendrite prefers 5-10 SUSTAINED afferents over 20+ TRANSIENT ones), Bradshaw PNAS 2003 (CaMKII Hill ~8 against an
ABSOLUTE set-point).

Four properties, all load-bearing:
  1. BYTE-IDENTICAL OFF — theta=0.0 (default) => cp_btsp_win_count stays None and the resulting weight vector is
     BIT-EQUAL to the pre-edit code (golden md5, measured by running the identical harness on the stashed pre-edit
     sim/bridge.py + sim/config.py; deterministic numpy backend).
  2. RESET — reset_btsp_window() zeroes the counter (this IS the box-car; without it the count is cumulative).
  3. GATE ENGAGES — with theta above the sustained count, the write is cut; below it, the write survives.
  4. SUSTAINED-vs-TRANSIENT — the discrimination the existing spatial coincidence count structurally cannot make:
     a source firing MANY times writes; an equally-numerous source firing FEW times does not.
CPU/numpy, self-contained (real bridge).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import hashlib
import numpy as np
import pytest
from sim.backend import get_backend, to_host
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge

xp, _BACKEND = get_backend()

# Golden: md5 of cp_connections.data after the 60-step plateau protocol below, measured on the PRE-EDIT
# sim/bridge.py + sim/config.py (git stash) with the identical harness. Any change to the default (gate-off)
# BTSP path breaks this.
_GOLDEN_OFF_WHASH = "b14caf505f444290a992e71af87a3457"
_GOLDEN_OFF_DW = 7.9346046447753906


def _build(theta=0.0, hill_n=8.0, n_pre=8, seed=42):
    regions = [
        BrainRegion(name="pre", n_neurons=n_pre, exc_fraction=1.0, internal_density=0.0,
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
    cfg.enable_btsp = True
    cfg.btsp_learning_rate = 0.01
    cfg.btsp_elig_tau_ms = 1000.0
    cfg.btsp_w_max = 5.0
    cfg.btsp_win_gate_theta = float(theta)
    cfg.btsp_win_gate_hill_n = float(hill_n)
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _run(theta=0.0, hill_n=8.0, steps=60, drive_steps=None, reset_at_start=False):
    """Drive `pre` for `drive_steps` of `steps` under a held plateau on `post`; return (dw, whash, bridge)."""
    sb = _build(theta, hill_n)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre")))
    post_idx = np.asarray(list(rm.indices("post")))
    n = sb.cp_membrane_potential_v.size
    sb.cp_v_apical = xp.full(n, -65.0, dtype=xp.float32)
    w0 = float(to_host(sb.cp_connections.data).sum())
    drive = np.zeros(n, dtype=np.float32)
    drive[pre_idx] = 900.0
    off = np.zeros(n, dtype=np.float32)
    if reset_at_start:
        sb.reset_btsp_window()
    for s in range(steps):
        sb.cp_external_input_current[:] = xp.asarray(drive if (drive_steps is None or s < drive_steps) else off)
        sb.cp_v_apical[post_idx] = xp.float32(-20.0)
        sb._run_one_simulation_step()
    w1 = float(to_host(sb.cp_connections.data).sum())
    whash = hashlib.md5(to_host(sb.cp_connections.data).tobytes()).hexdigest()
    return w1 - w0, whash, sb


def test_win_gate_off_is_byte_identical_to_pre_edit():
    """Property 1 — the default (theta=0.0) path is BIT-EQUAL to the pre-edit code, and allocates nothing."""
    dw, whash, sb = _run(theta=0.0)
    assert sb.cp_btsp_win_count is None, \
        "cp_btsp_win_count must NEVER be allocated when btsp_win_gate_theta=0.0 (the guard must not execute)"
    if _BACKEND == "numpy":     # numpy is deterministic; cupy reductions are not bit-reproducible
        assert whash == _GOLDEN_OFF_WHASH, \
            f"gate-off weights must be bit-equal to the pre-edit golden, got {whash} != {_GOLDEN_OFF_WHASH}"
        assert dw == pytest.approx(_GOLDEN_OFF_DW, abs=0.0), f"gate-off dw drifted: {dw!r}"


def test_win_gate_reset_zeroes_the_boxcar():
    """Property 2 — reset_btsp_window() is the box-car reset; it must zero an accumulated count (and be a
    no-op, allocating nothing, when the gate is off)."""
    _, _, sb = _run(theta=5.0)
    assert sb.cp_btsp_win_count is not None, "the counter must be allocated when the gate is on"
    assert float(to_host(sb.cp_btsp_win_count).max()) > 0.0, "the counter must accumulate spikes"
    sb.reset_btsp_window()
    assert float(to_host(sb.cp_btsp_win_count).max()) == 0.0, "reset_btsp_window() must zero the counter"
    off = _build(theta=0.0)
    off.reset_btsp_window()            # must not raise, must not allocate
    assert off.cp_btsp_win_count is None


def test_win_gate_engages_absolute_threshold():
    """Property 3 — an ABSOLUTE theta ABOVE the achievable window count cuts the write to ~0; a theta BELOW it
    leaves the write essentially intact. (An absolute threshold, NOT a fraction of a network max: raising theta
    past the count must kill the write even though the *relative* ranking of sources is unchanged.)"""
    dw_off, _, _ = _run(theta=0.0)
    dw_low, _, sb_low = _run(theta=3.0)          # count reaches ~30 over 60 steps => gate ~1
    dw_high, _, sb_high = _run(theta=10000.0)    # unreachable count => gate ~0
    cmax_low = float(to_host(sb_low.cp_btsp_win_count).max())
    assert cmax_low > 5.0, f"the window count must actually accumulate (max={cmax_low})"
    assert abs(dw_high) < 1e-4, f"an unreachable absolute theta must cut the write, got dw={dw_high:.6f}"
    assert dw_low > 0.5 * dw_off, f"a theta below the count must leave the write intact, got {dw_low:.4f} vs {dw_off:.4f}"


def test_win_gate_prefers_sustained_over_transient():
    """Property 4 — THE primitive. A source that fires MANY times over the window writes; the SAME number of
    sources firing FEW times does not. This is exactly what a per-step / spatial coincidence count cannot express
    (Polsky/Mel/Schiller 2009: 5-10 sustained afferents beat 20+ transient ones)."""
    steps = 60
    dw_sustained, _, sb_s = _run(theta=8.0, steps=steps, drive_steps=steps)
    dw_transient, _, sb_t = _run(theta=8.0, steps=steps, drive_steps=6)
    c_s = float(to_host(sb_s.cp_btsp_win_count).max())
    c_t = float(to_host(sb_t.cp_btsp_win_count).max())
    assert c_s > 8.0 >= c_t or c_s > c_t, f"the sustained window must accumulate more count ({c_s} vs {c_t})"
    assert dw_sustained > 0.1, f"a sustained source above theta must write, got {dw_sustained:.4f}"
    assert dw_transient < 0.25 * dw_sustained, \
        f"a transient source below theta must be gated OUT, got {dw_transient:.4f} vs {dw_sustained:.4f}"
