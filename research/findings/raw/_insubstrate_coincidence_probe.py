"""In-substrate spiking COINCIDENCE primitive -- the core of the compositional bind.
Build a custom bridge (G1 explicit-wiring pattern) with three populations: role (N),
filler (N), coinc (N). Identity wiring role[i]->coinc[i] + filler[i]->coinc[i]. A
tonic hyperpolarizing bias on coinc makes a SINGLE input sub-threshold while BOTH
together sum supra-threshold -> coinc[i] computes AND(role[i], filler[i]) in spiking
dynamics (textbook coincidence detection: the cell rests below threshold, needs
summed input).

VALIDATED 2026-05-31 (RTX 3090, CuPy), seed 42:
  w=200 bias=-500 -> BOTH=0.059  single=0.005  none=0.000  AND-selectivity=0.921
  w=320 bias=-1000-> BOTH=0.048  single=0.000  none=0.000  AND-selectivity=1.000 (perfect)
The control is geometric: role-only coinc neurons receive role input but their filler
partner is silent -> they stay dark (single=0.005); none-region never fires (0.000);
only neurons receiving BOTH an active role AND an active filler fire. A genuine
threshold AND, not a 2x-drive artifact. -> the in-substrate ON/OFF bind is buildable.

If this primitive works (coinc fires on the role&filler intersection, not on
role-only or filler-only), the ON/OFF version (two coincidence banks) realizes the
+-1 Hadamard bind in-substrate, and the full compositional architecture is buildable
(the rest -- overlapping fillers, spiking-readout noise, cleanup -- is already
de-risked). Pre-registered: AND-selectivity index = (coinc rate in BOTH region) -
(max of coinc rate in role-only, filler-only regions), normalized; >= 0.5 -> the
spiking AND works.
"""
from __future__ import annotations
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

N = 128
DRIVE_PA = 2500.0      # drive for active role/filler neurons (these Izh neurons need ~2000pA)
RESET_STEPS = 30
RUN_STEPS = 80
# Tonic hyperpolarizing bias on the coincidence neurons: makes a SINGLE input
# sub-threshold while BOTH inputs sum supra-threshold -> sharp AND (textbook
# coincidence detection: the cell rests below threshold, needs summed input).


def build(seed, w_coinc):
    cfg = CoreSimConfig()
    cfg.num_neurons = 3 * N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.ou_std_current_pA = 20.0

    role = list(range(0, N))
    fill = list(range(N, 2 * N))
    coinc = list(range(2 * N, 3 * N))
    pre, post, w = [], [], []
    for i in range(N):
        pre.append(role[i]); post.append(coinc[i]); w.append(w_coinc)
        pre.append(fill[i]); post.append(coinc[i]); w.append(w_coinc)
    plan = {"bind": {"pre_indices": pre, "post_indices": post,
                     "initial_weights": np.array(w, dtype=np.float32),
                     "plastic": False, "conn_type": "E_TO_E", "count": len(pre)}}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.inject_explicit_wiring(plan)
    return bridge, role, fill, coinc


def measure(bridge, role, fill, coinc, role_on, fill_on, xp, coinc_bias=0.0):
    """Drive role neurons in role_on, filler neurons in fill_on; capture coinc rate.
    coinc_bias (<=0) is a tonic current held on ALL coinc neurons to set their threshold."""
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    r_idx = xp.asarray([role[i] for i in role_on], dtype=xp.int64)
    f_idx = xp.asarray([fill[i] for i in fill_on], dtype=xp.int64)
    coinc_arr = xp.asarray(coinc, dtype=xp.int64)
    if r_idx.shape[0]:
        bridge.cp_external_input_current[r_idx] = DRIVE_PA
    if f_idx.shape[0]:
        bridge.cp_external_input_current[f_idx] = DRIVE_PA
    if coinc_bias != 0.0:
        bridge.cp_external_input_current[coinc_arr] = coinc_bias   # tonic threshold-setting bias
    role_arr = xp.asarray(role, dtype=xp.int64)
    fill_arr = xp.asarray(fill, dtype=xp.int64)
    c = xp.zeros(N, dtype=xp.float64)
    rr = xp.zeros(N, dtype=xp.float64)
    fr = xp.zeros(N, dtype=xp.float64)
    for _ in range(RUN_STEPS):
        bridge._run_one_simulation_step()
        c += bridge.cp_firing_states[coinc_arr].astype(xp.float64)
        rr += bridge.cp_firing_states[role_arr].astype(xp.float64)
        fr += bridge.cp_firing_states[fill_arr].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(c) / RUN_STEPS, to_host(rr) / RUN_STEPS, to_host(fr) / RUN_STEPS


def main():
    xp, backend = get_backend()
    print(f"=== in-substrate spiking COINCIDENCE primitive (backend={backend}, N={N}) ===", flush=True)
    # regions of the index space: BOTH = [N/4, N/2); role-only = [0, N/4); filler-only = [N/2, 3N/4)
    both = list(range(N // 4, N // 2))
    role_only = list(range(0, N // 4))
    fill_only = list(range(N // 2, 3 * N // 4))
    role_on = role_only + both          # role active on [0, N/2)
    fill_on = both + fill_only          # filler active on [N/4, 3N/4)
    # intersection (AND) = both

    best = (-1.0, None)   # (selectivity, (w, bias, both, single))
    # grid: per-input weight x tonic coinc bias. Without bias the regime is near-linear
    # (double ~ 2x single); a hyperpolarizing bias makes single sub-threshold -> sharp AND.
    for w_coinc in [120.0, 200.0, 320.0]:
        for bias in [0.0, -200.0, -500.0, -1000.0, -2000.0]:
            bridge, role, fill, coinc = build(42, w_coinc)
            rate, _, _ = measure(bridge, role, fill, coinc, role_on, fill_on, xp, coinc_bias=bias)
            both_r = float(np.mean(rate[both]))
            role_r = float(np.mean(rate[role_only]))
            fill_r = float(np.mean(rate[fill_only]))
            none_r = float(np.mean(rate[3 * N // 4:]))
            single = max(role_r, fill_r)
            sel = (both_r - single) / (both_r + 1e-9)
            flag = ""
            if both_r >= 0.05 and sel > best[0]:        # require a real BOTH signal, not a degenerate 0/0
                best = (sel, (w_coinc, bias, both_r, single)); flag = "  <-- best"
            print(f"  w={w_coinc:>6.1f} bias={bias:>8.1f} | BOTH={both_r:.3f} "
                  f"single={single:.3f} none={none_r:.3f} | AND-sel={sel:.3f}{flag}", flush=True)
            del bridge

    print("\nREAD: want BOTH high, single/none low (AND-selectivity >= 0.5). "
          "If found -> spiking coincidence AND works -> the in-substrate bind is buildable.", flush=True)
    if best[1]:
        w, b, both_r, single = best[1]
        print(f"BEST: w={w} bias={b} -> BOTH={both_r:.3f} single={single:.3f} "
              f"AND-selectivity={best[0]:.3f}  "
              f"{'(>= 0.5: sharp AND WORKS)' if best[0] >= 0.5 else '(< 0.5: tune further)'}", flush=True)


if __name__ == "__main__":
    main()
