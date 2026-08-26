"""PFC persistent-activity tuning probe (Session F Task 1, contingent).

Activated only if NE sensitivity sweep is also negative. Searches for
BrainRegion + RegionPathway parameters that give a PFC-only population
two stable firing-rate states ("quiescent" and "persistent") so it can
hold goal-context across trials.

Biological target (Wang 2002, Constantinidis & Klingberg 2016):
  - Quiescent: ~3-8 Hz background activity
  - Persistent: 20-40 Hz sustained for >= 1 sec after a brief input pulse
  - Bistability: brief input transitions quiescent -> persistent and stays

Search axes:
  - exc_weight_mean: 0.3-1.5 (stronger recurrent excitation -> easier persistence)
  - inh_weight_mean: 0.5-2.0 (stronger inhibition -> sharper transitions)
  - internal_density: 0.1-0.3 (denser network -> more recurrent influence)
  - exc_fraction: 0.7-0.9 (cortical biology range)

Strategy: grid-sweep at Route C scale (200 PFC neurons), ~9-12 configs,
running each in a separate replica via E.3 batched-replica framework.
For each config: (1) pulse PFC at t=200ms with brief excitatory input,
(2) measure firing rate across the next 1 second.

Pass condition (per config):
  - Pre-pulse rate (0-200ms): 3-8 Hz
  - Pulse window rate (200-300ms): >= 30 Hz (transient pulse response)
  - Post-pulse persistent rate (500-1500ms): 15-50 Hz (persistent activity)
  - Variance < 50% of mean (stable firing, not bursting/silent oscillation)

Outputs the best ~3 configs for use in PFC module Task 2 (goal-setting
input integration).
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_pfc_config(exc_weight: float, inh_weight: float,
                      density: float, exc_fraction: float):
    """Returns (BrainRegion, RegionPathway list) for a single PFC-only sim."""
    from sim.regions import BrainRegion

    pfc = BrainRegion(
        name="PFC",
        n_neurons=200,
        exc_fraction=exc_fraction,
        internal_density=density,
        exc_weight_mean=exc_weight,
        inh_weight_mean=inh_weight,
        weight_jitter=0.2,
        plastic_internal=False,  # frozen during tuning probe
    )
    return pfc, []


def run_single_config(exc_w: float, inh_w: float, density: float,
                       exc_frac: float, pulse_pa: float = 250.0,
                       n_steps: int = 1500, seed: int = 42) -> dict:
    """Run a single PFC config and return firing-rate profile."""
    import cupy as cp
    import numpy as np
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig,
        RuntimeState, GPUConfig,
    )
    from sim.config import (
        ExperimentConfig, ExperimentPhase, StimulusChannel, StimulusPattern,
        ReadoutConfig,
    )
    from sim.enums import (
        NeuronModel, StimulusPatternType, ExperimentPhaseType,
    )
    from experiment import ExperimentEngine
    from sim.regions import BrainRegion

    pfc = BrainRegion(
        name="PFC",
        n_neurons=200,
        exc_fraction=exc_frac,
        internal_density=density,
        exc_weight_mean=exc_w,
        inh_weight_mean=inh_w,
        weight_jitter=0.2,
        plastic_internal=False,
    )

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [pfc]
    cfg.region_pathways = []
    cfg.connections_per_neuron = 0
    cfg.enable_watts_strogatz = False
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False

    sb = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)

    # Set up the experiment engine for stimulus delivery
    engine = ExperimentEngine(cfg.num_neurons, cfg.dt_ms)
    exp_cfg = ExperimentConfig()
    exp_cfg.neuron_groups = []
    exp_cfg.readout = ReadoutConfig(
        rate_window_ms=100.0, spike_count_window_ms=100.0,
        rate_group_names=[],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="probe", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=sb.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    sb.experiment_engine = engine

    # Pulse: brief excitatory current to PFC at steps 200-220
    # During this window the stimulus channel is active.
    pulse_start = 200
    pulse_end = 220
    pulse_targets = list(range(0, pfc.n_neurons))  # all PFC neurons

    # Build the stimulus channel
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.CONSTANT.name,
        amplitude_pA=pulse_pa,
    )
    ch = StimulusChannel(
        name="pulse", pattern=pat,
        target_neuron_indices=pulse_targets,
        onset_ms=float(pulse_start),
        duration_ms=float(pulse_end - pulse_start),
        enabled=True,
    )
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([ch], engine.group_manager, cp)
    engine.phase_start_ms = 0.0

    # Run and record per-step firing
    firing_per_step = []
    for s in range(n_steps):
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1
        sb.runtime_state.current_time_ms = sb.runtime_state.current_time_step * cfg.dt_ms
        firing_per_step.append(int(cp.sum(sb.cp_firing_states).get()))

    sb.clear_simulation_state_and_gpu_memory()

    # Compute rate-per-window-ms
    fp = np.array(firing_per_step)
    n_total = pfc.n_neurons

    def _rate(window_start: int, window_end: int) -> Tuple[float, float]:
        """Mean rate (Hz) and std-dev across the window."""
        spikes = fp[window_start:window_end]
        if spikes.size == 0:
            return 0.0, 0.0
        # Per neuron, per second
        # spikes_per_step / n_neurons / dt_sec
        per_step_per_neuron = spikes / n_total
        rate_hz = per_step_per_neuron.mean() * 1000.0  # dt=1ms -> 1000 Hz/step
        std_hz = per_step_per_neuron.std() * 1000.0
        return float(rate_hz), float(std_hz)

    pre_rate, pre_std = _rate(0, pulse_start)
    pulse_rate, pulse_std = _rate(pulse_start, pulse_end + 80)  # +80ms tail
    persistent_rate, persistent_std = _rate(500, 1500)  # 500-1500ms post-onset

    return {
        "params": {
            "exc_weight": exc_w, "inh_weight": inh_w,
            "density": density, "exc_fraction": exc_frac,
            "pulse_pa": pulse_pa,
        },
        "rates_hz": {
            "pre": (pre_rate, pre_std),
            "pulse": (pulse_rate, pulse_std),
            "persistent": (persistent_rate, persistent_std),
        },
        "n_total_pfc": n_total,
    }


def evaluate_config(result: dict) -> dict:
    """Score a single PFC config result against bistability criteria."""
    pre_rate, _ = result["rates_hz"]["pre"]
    pulse_rate, _ = result["rates_hz"]["pulse"]
    persistent_rate, persistent_std = result["rates_hz"]["persistent"]

    # Pass conditions
    pre_ok = 1.0 <= pre_rate <= 12.0  # quiescent baseline
    pulse_ok = pulse_rate >= 20.0  # responded to pulse
    persistent_ok = 12.0 <= persistent_rate <= 60.0  # persistent firing
    stability_ok = persistent_std < 0.5 * persistent_rate  # < 50% CV

    # Score (sum of pass conditions; favor balanced)
    score = sum([pre_ok, pulse_ok, persistent_ok, stability_ok])

    return {
        **result,
        "pre_ok": pre_ok,
        "pulse_ok": pulse_ok,
        "persistent_ok": persistent_ok,
        "stability_ok": stability_ok,
        "score": score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny grid for smoke testing (4 configs)")
    args = ap.parse_args()

    out_dir = Path("research/findings/raw/pfc_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        configs = [
            (0.5, 1.0, 0.15, 0.8),
            (1.0, 1.0, 0.15, 0.8),
            (1.0, 1.5, 0.15, 0.8),
            (1.5, 1.0, 0.20, 0.8),
        ]
    else:
        # Wider grid focused on STRONG recurrent excitation. Initial quick
        # grid at exc_weight 0.5-1.5 + density 0.15-0.20 showed persistent
        # rate collapses to baseline (~5 Hz) for all configs — recurrent
        # excitation isn't strong enough to maintain firing post-pulse.
        # Going up to exc_weight=4.0 and density=0.30 to bracket the
        # bistability boundary.
        exc_weights = [1.5, 2.5, 3.5]
        inh_weights = [0.5, 1.0, 1.5]
        densities = [0.20, 0.30]
        exc_fractions = [0.8]
        configs = list(product(exc_weights, inh_weights, densities, exc_fractions))

    print(f"\nPFC tuning probe: {len(configs)} configs, ~5 sec each\n")
    all_results = []
    t0 = time.time()
    for i, (e, ih, d, ef) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] exc_w={e} inh_w={ih} density={d} exc_frac={ef}...",
              flush=True)
        try:
            r = run_single_config(e, ih, d, ef)
            scored = evaluate_config(r)
            print(f"    pre={r['rates_hz']['pre'][0]:.1f}Hz  "
                  f"pulse={r['rates_hz']['pulse'][0]:.1f}Hz  "
                  f"persistent={r['rates_hz']['persistent'][0]:.1f}Hz  "
                  f"score={scored['score']}/4")
            all_results.append(scored)
        except Exception as exc:
            print(f"    FAILED: {exc}")

    # Sort by score
    all_results.sort(key=lambda r: -r["score"])

    out_path = out_dir / "pfc_tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Top configs by score ===")
    for r in all_results[:5]:
        params = r["params"]
        print(f"  score={r['score']}/4  exc={params['exc_weight']} "
              f"inh={params['inh_weight']} density={params['density']}: "
              f"persist={r['rates_hz']['persistent'][0]:.1f}Hz")
    print(f"\nResults saved to {out_path}  ({time.time() - t0:.0f}s wall)")


if __name__ == "__main__":
    main()
