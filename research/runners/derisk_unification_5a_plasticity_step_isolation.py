"""De-risk 5a for the navigation+conversational single-instance unification (roadmap step 2).

Per `docs/plans/2026-06-10-nav-conv-single-instance-unification-design.md` §5a, this is the REQUIRED,
cheapest-first de-risk that the merge's load-bearing isolation claim depends on — and it must pass on BOTH
the looser (RF on its own bridge) and the strict (RF co-resident) paths, because the all-Izhikevich
nav+parser+dlPFC merge needs it either way.

THE QUESTION (two halves):
  1. PLASTICITY ISOLATION. On ONE shared bridge that runs the navigation brain's learning (reward-modulated
     STDP + a global dopamine neuromodulator whose `plasticity_rate` target has scope="all") AND the parser's
     Hebbian learning (both are GLOBAL config flags), does a per-synapse plasticity gate held at 0.0 keep a
     fixed conversational population BYTE-IDENTICAL, while the ungated populations still learn?
       - The 2026-06-04 conversational unification (`2026-06-04-unified-bridge-plasticity-isolation.md`) already
         proved the gate freezes a fixed population against global HEBBIAN. What it did NOT exercise is the
         navigation stressor: reward-modulated STDP + the dopamine scope="all" plasticity_rate multiplier.
         The design §4.3 trust-but-verified in code that the reward path is ALSO per-synapse gated
         (`sim/bridge.py:6456-6457`); 5a confirms that EMPIRICALLY.
  2. STEP COEXISTENCE. The two brains use DIFFERENT stepping disciplines on ONE step loop (nav steps a
     sensorimotor loop continuously; the conversational ops drive-then-read a slice). Does running a
     navigation-length burst CORRUPT a subsequent conversational read of the frozen slice?

THE TINY BRIDGE (all Izhikevich, dt=1.0, brain-region framework — exactly the merge's world):
  - nav-like group:    nav_ctx -> nav_d1     (plastic, UNGATED; learns under STDP+reward+dopamine)
  - conv-like FROZEN:   conv_a  -> conv_b     (plastic, plasticity_gate="conv_frozen" held 0.0)
  - parser-like control: parser_a -> parser_b (plastic, UNGATED; the Hebbian control)
  Every pathway is tagged with a plasticity gate so its synapse indices are individually addressable for the
  byte-comparison (nav_learn / parser_learn are left at the default gain 1.0 = no behavioural change; only
  conv_frozen is zeroed). The dopamine modulator is the flagship's own `_default_dopamine_config()`
  (from_reward -> dopamine; plasticity_rate scope="all").

PROCEDURE:
  read1  = conv functional read (drive conv_a from rest, measure conv_b firing)
  BURST  = drive nav_ctx + conv_a + parser_a, pulse current_reward_signal>0 each trial, ~1500 steps. conv is
           ACTIVE during the burst (the HARDEST freeze test: it accumulates STDP eligibility + Hebbian
           co-activity that the gate must zero), while nav stepping runs continuously around it.
  read2  = conv functional read (identical protocol, from rest)

PASS (all four):
  (a) conv_frozen weights BYTE-IDENTICAL before vs after the burst (np.array_equal).
  (b) nav_learn weights CHANGED   (control non-vacuous: reward-modulated STDP is live).
  (c) parser_learn weights CHANGED (control non-vacuous: Hebbian is live).
  (d) read1 == read2 within a tight tolerance (step coexistence: the nav burst did not corrupt the conv read).

A FAIL on (a) means the global dopamine/STDP path reaches the gated slice despite the gate -> a concrete
isolation bug/boundary to surface (a real finding). A FAIL on (d) means step-coexistence needs stronger
per-slice reset/quiescence before the merge (also a real finding). Honest negatives ARE the deliverable.

Run on GPU (CuPy) for the real verdict; the bridge is tiny (~210 neurons) so it is seconds either way.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.neuromodulators import _default_dopamine_config
from sim.backend import get_backend, to_host


# ── the tiny two-brain bridge ────────────────────────────────────────────────────────────────────────────
def build_regions_pathways(conv_weight_mean: float = 6.0):
    """Three disjoint pre->post groups; every pathway gate-tagged so its synapses are addressable."""
    regions = [
        BrainRegion(name="nav_ctx", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="nav_d1", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="conv_a", n_neurons=40, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="conv_b", n_neurons=50, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="parser_a", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="parser_b", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
    ]
    pathways = [
        # nav: the reward-modulated STDP learner (ungated; gain stays 1.0).
        RegionPathway(from_region="nav_ctx", to_region="nav_d1",
                      density=0.6, weight_mean=8.0, weight_jitter=0.2,
                      plastic=True, plasticity_gate="nav_learn"),
        # conv: the FROZEN conversational population (gate zeroed after init). weight_mean models a real
        # frozen conversational weight (the composer's role-routes are ~300, the bind population ~320); it
        # must sit WITHIN the clip bounds (clip_max) or the ungated global weight clip moves it (the 5a gap).
        RegionPathway(from_region="conv_a", to_region="conv_b",
                      density=0.5, weight_mean=float(conv_weight_mean), weight_jitter=0.2,
                      plastic=True, plasticity_gate="conv_frozen"),
        # parser: the Hebbian control (ungated; gain stays 1.0).
        RegionPathway(from_region="parser_a", to_region="parser_b",
                      density=0.5, weight_mean=6.0, weight_jitter=0.2,
                      plastic=True, plasticity_gate="parser_learn"),
    ]
    return regions, pathways


def build_bridge(seed: int, enable_stdp: bool = True, enable_reward: bool = True,
                 enable_hebbian: bool = True, clip_max: float = 20.0,
                 conv_weight_mean: float = 6.0) -> SimulationBridge:
    regions, pathways = build_regions_pathways(conv_weight_mean=conv_weight_mean)
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    # The merged-bridge learning condition: nav reward-STDP + parser Hebbian, BOTH global.
    cfg.enable_stdp = bool(enable_stdp)
    cfg.enable_reward_modulation = bool(enable_reward)
    cfg.reward_learning_rate = 0.05
    cfg.enable_hebbian_learning = bool(enable_hebbian)   # the parser needs it; on the shared bridge it is global
    cfg.hebbian_learning_rate = 0.01
    # THE 5a FINDING: the per-synapse plasticity gate freezes weight UPDATES (Hebbian potentiation/decay,
    # STDP delta, reward delta — all gated) but NOT the two ungated global weight CLIPS
    # (sim/bridge.py:6175 Hebbian, :6480 reward). A frozen weight OUTSIDE the active rule's clip bounds is
    # moved by the clip. The merge mitigation (modelled here): set the clip bounds ABOVE the frozen
    # conversational population's max weight (the composer's real-valued role-routes ~300 / bind ~320). With
    # clip_max >= the frozen weight, the gate isolates updates cleanly; with the defaults (hebbian_max=1.0,
    # stdp_w_max=2.0) below the frozen weight, the clip moves it (run with --clip-max 1.0 to reproduce).
    cfg.stdp_w_max = float(clip_max)
    cfg.hebbian_max_weight = float(clip_max)
    # match the nav: no homeostasis / STP / OU / heterogeneity / structural plasticity
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    # the flagship's own global dopamine: from_reward -> dopamine; plasticity_rate scope="all"
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [_default_dopamine_config()]

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────
def gate_syn_indices(bridge: SimulationBridge, gate_name: str) -> np.ndarray:
    """The cp_connections.data indices for a plasticity-gate-tagged pathway (host int array)."""
    key = bridge._canonicalize_gate_name(gate_name)
    if key not in bridge._plasticity_gate_indices_gpu:
        raise KeyError(
            f"gate '{gate_name}' (canon '{key}') not registered. "
            f"known: {list(bridge._plasticity_gate_indices_gpu.keys())}"
        )
    return to_host(bridge._plasticity_gate_indices_gpu[key]).astype(np.int64)


def free_run(bridge: SimulationBridge, xp, steps: int) -> None:
    """Zero ALL external input + reward, step quietly so Izhikevich state relaxes to rest before a read."""
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()


def conv_read(bridge: SimulationBridge, xp, conv_a_idx, conv_b_idx,
              drive_pa: float, settle: int, window: int) -> float:
    """Drive conv_a from rest, return mean conv_b firing fraction over the read window.

    A functional read of the frozen conversational slice. With conv_a->conv_b weights frozen and the slice at
    rest beforehand, this is deterministic and must be identical before vs after the nav burst.
    """
    free_run(bridge, xp, settle)
    spikes = 0.0
    n = int(conv_b_idx.size)
    total_steps = window
    for t in range(window):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[conv_a_idx] = drive_pa
        bridge._run_one_simulation_step()
        spikes += float(to_host(bridge.cp_firing_states[conv_b_idx]).sum())
    # leave the bridge quiet afterward
    bridge.cp_external_input_current[:] = 0.0
    return spikes / (n * total_steps)


def run_burst(bridge: SimulationBridge, xp, driven_idx, reward_pa_regions,
              n_trials: int, steps_per_trial: int, reward_steps: int,
              drive_pa: float, reward_value: float) -> None:
    """Navigation-length burst: drive the pre regions, pulse reward each trial.

    `driven_idx` = concatenated global indices of the pre regions driven (nav_ctx + conv_a + parser_a). conv is
    ACTIVE during the burst (hardest freeze test). Reward is pulsed for `reward_steps` of each trial so the
    STDP eligibility converts to weight AND the dopamine concentration rises (plasticity_rate scope="all").
    """
    for trial in range(n_trials):
        for s in range(steps_per_trial):
            bridge.cp_external_input_current[:] = 0.0          # per-step fresh drive (= per-slice control)
            bridge.cp_external_input_current[driven_idx] = drive_pa
            # reward pulse near the end of the integration window (after co-firing builds eligibility)
            if s >= steps_per_trial - reward_steps:
                bridge.core_config.current_reward_signal = float(reward_value)
            else:
                bridge.core_config.current_reward_signal = 0.0
            bridge._run_one_simulation_step()
        # inter-trial: drop drive + reward
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
    bridge.core_config.current_reward_signal = 0.0
    bridge.cp_external_input_current[:] = 0.0


def main():
    ap = argparse.ArgumentParser(description="Unification de-risk 5a: plasticity isolation + step coexistence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--steps-per-trial", type=int, default=50)
    ap.add_argument("--reward-steps", type=int, default=15)
    ap.add_argument("--drive-pa", type=float, default=800.0)
    ap.add_argument("--reward-value", type=float, default=1.0)
    ap.add_argument("--read-drive-pa", type=float, default=800.0)
    ap.add_argument("--read-settle", type=int, default=120)
    ap.add_argument("--read-window", type=int, default=60)
    ap.add_argument("--read-tol", type=float, default=1e-6)
    ap.add_argument("--clip-max", type=float, default=20.0,
                    help="hebbian_max_weight + stdp_w_max. >= the frozen conv weight => clip does not bite "
                         "(gate isolates updates, PASS); < it => the ungated clip moves the frozen slice "
                         "(the 5a gap; try --clip-max 1.0).")
    ap.add_argument("--conv-weight-mean", type=float, default=6.0,
                    help="frozen conv pathway weight (models the composer's real-valued role-routes ~300).")
    ap.add_argument("--out", type=str, default="research/findings/raw/derisk_5a_isolation.json")
    args = ap.parse_args()

    xp, backend = get_backend()
    print(f"[5a] backend={backend} seed={args.seed} clip_max={args.clip_max} conv_weight={args.conv_weight_mean}")

    bridge = build_bridge(args.seed, clip_max=args.clip_max, conv_weight_mean=args.conv_weight_mean)
    rm = bridge.region_manager
    n_neurons = int(bridge.core_config.num_neurons)
    nnz = int(bridge.cp_connections.nnz)
    print(f"[5a] bridge: {len(bridge.core_config.brain_regions)} regions, {n_neurons} neurons, {nnz} synapses")

    # global index arrays for driving
    nav_ctx_idx = xp.asarray(rm.indices("nav_ctx"), dtype=xp.int64)
    conv_a_idx = xp.asarray(rm.indices("conv_a"), dtype=xp.int64)
    conv_b_idx = xp.asarray(rm.indices("conv_b"), dtype=xp.int64)
    parser_a_idx = xp.asarray(rm.indices("parser_a"), dtype=xp.int64)
    driven_idx = xp.concatenate([nav_ctx_idx, conv_a_idx, parser_a_idx])

    # FREEZE the conversational slice (the load-bearing gate); leave nav_learn / parser_learn at 1.0
    bridge.set_plasticity_gate("conv_frozen", 0.0)

    # synapse-index handles for the byte comparison
    conv_syn = gate_syn_indices(bridge, "conv_frozen")
    nav_syn = gate_syn_indices(bridge, "nav_learn")
    parser_syn = gate_syn_indices(bridge, "parser_learn")
    print(f"[5a] synapses — conv_frozen={conv_syn.size}  nav_learn={nav_syn.size}  parser_learn={parser_syn.size}")

    # weights BEFORE
    w_before = to_host(bridge.cp_connections.data).copy()

    # conv functional read BEFORE the nav burst
    read1 = conv_read(bridge, xp, conv_a_idx, conv_b_idx,
                      args.read_drive_pa, args.read_settle, args.read_window)

    # the navigation-length reward-STDP + dopamine burst (conv active)
    run_burst(bridge, xp, driven_idx, [conv_a_idx],
              n_trials=args.n_trials, steps_per_trial=args.steps_per_trial,
              reward_steps=args.reward_steps, drive_pa=args.drive_pa, reward_value=args.reward_value)

    # conv functional read AFTER the nav burst
    read2 = conv_read(bridge, xp, conv_a_idx, conv_b_idx,
                      args.read_drive_pa, args.read_settle, args.read_window)

    # weights AFTER
    w_after = to_host(bridge.cp_connections.data)

    # ── verdict ──────────────────────────────────────────────────────────────────────────────────────────
    conv_max_abs = float(np.max(np.abs(w_after[conv_syn] - w_before[conv_syn]))) if conv_syn.size else 0.0
    conv_identical = bool(np.array_equal(w_before[conv_syn], w_after[conv_syn]))
    nav_delta = float(np.max(np.abs(w_after[nav_syn] - w_before[nav_syn]))) if nav_syn.size else 0.0
    parser_delta = float(np.max(np.abs(w_after[parser_syn] - w_before[parser_syn]))) if parser_syn.size else 0.0
    nav_changed = nav_delta > 0.0
    parser_changed = parser_delta > 0.0
    read_diff = abs(read2 - read1)
    read_match = read_diff <= args.read_tol

    da_conc = None
    try:
        da_conc = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    except Exception:
        pass

    passed = conv_identical and nav_changed and parser_changed and read_match

    print("\n=== De-risk 5a verdict ===")
    print(f"(a) conv_frozen BYTE-IDENTICAL : {conv_identical}  (max|dw| over gated synapses = {conv_max_abs:.3e})")
    print(f"(b) nav_learn CHANGED          : {nav_changed}  (max|dw| = {nav_delta:.4f})")
    print(f"(c) parser_learn CHANGED       : {parser_changed}  (max|dw| = {parser_delta:.4f})")
    print(f"(d) conv read coexistence      : {read_match}  (read1={read1:.6f} read2={read2:.6f} |d|={read_diff:.3e} tol={args.read_tol})")
    print(f"    dopamine end-concentration : {da_conc}  (baseline 0.5; >0.5 confirms the scope='all' stressor was active)")
    print(f"\n[5a] {'PASS — isolation + step coexistence hold under the nav stressor' if passed else 'FAIL — see which gate(s) above'}")

    result = {
        "derisk": "5a_plasticity_step_isolation",
        "backend": backend,
        "seed": args.seed,
        "n_neurons": n_neurons,
        "nnz": nnz,
        "n_syn": {"conv_frozen": int(conv_syn.size), "nav_learn": int(nav_syn.size),
                  "parser_learn": int(parser_syn.size)},
        "conv_identical": conv_identical,
        "conv_max_abs_delta": conv_max_abs,
        "nav_changed": nav_changed,
        "nav_max_delta": nav_delta,
        "parser_changed": parser_changed,
        "parser_max_delta": parser_delta,
        "read1": read1,
        "read2": read2,
        "read_diff": read_diff,
        "read_match": read_match,
        "dopamine_end_concentration": da_conc,
        "pass": passed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[5a] wrote {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
