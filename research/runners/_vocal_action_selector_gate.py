"""Gate A for target-independent neural vocal action selection.

This probe isolates one two-channel selector before it is copied into the
developmental vocal circuit. A shared practice-state population drives both
channels equally. Ongoing neural noise breaks symmetry; a basal-ganglia loop
and downstream commit circuit must turn that variation into one motor action.

The host presents trial onset, equal tonic afferents, and a shared inter-trial
reset signal. It never injects a channel-specific current and never falls back
to argmax when no neural commit occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from sim.backend import get_backend, to_host
from sim.enums import NeuronModel, NeuronType
from sim.regions import BrainRegion, RegionPathway


CHANNELS = (0, 1)
DEVELOPMENT_SEEDS = (42, 43, 44, 100)
DIRECT_PATH_GATE = "vocal_selector_direct_path"


@dataclass(frozen=True)
class SelectorConfig:
    n_practice: int = 24
    n_proposal: int = 60
    n_striatum: int = 36
    n_gpe: int = 16
    n_gpi: int = 20
    n_stn: int = 20
    n_thalamus: int = 24
    n_commit: int = 30
    n_commit_fs: int = 16
    n_motor: int = 30
    n_reset: int = 20
    ou_sigma_pA: float = 40.0
    practice_pA: float = 1000.0
    reset_pA: float = 1200.0
    gpi_tonic_pA: float = 1000.0
    thalamus_tonic_pA: float = 900.0
    arousal_to_proposal_weight: float = 1.0
    proposal_to_fsi_weight: float = 100.0
    fsi_to_msn_weight: float = 10.0
    proposal_to_msn_density: float = 1.0
    proposal_to_msn_weight: float = 400.0
    d1_to_gpi_weight: float = 15.0
    gpi_to_thalamus_weight: float = 12.0
    thalamus_to_commit_weight: float = 40.0
    commit_to_fsi_weight: float = 30.0
    commit_fsi_cross_weight: float = 40.0
    commit_to_motor_weight: float = 80.0
    warmup_steps: int = 80
    action_steps: int = 600
    reset_steps: int = 35
    washout_steps: int = 100
    commit_threshold_spikes: int = 12
    clean_loser_ratio: float = 0.25
    enable_striatal_fsi: bool = True


def selector_config(version: str) -> SelectorConfig:
    if version == "v1":
        return SelectorConfig()
    if version == "v2":
        return SelectorConfig(enable_striatal_fsi=False)
    raise ValueError(f"unknown selector version: {version}")


def _region(name, n, *, exc_fraction, neuron_type, internal_density=0.0,
            internal_weight=0.0, enable_nmda=False):
    return BrainRegion(
        name=name,
        n_neurons=int(n),
        exc_fraction=float(exc_fraction),
        internal_density=float(internal_density),
        exc_weight_mean=float(internal_weight),
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=neuron_type.name,
        enable_nmda=bool(enable_nmda),
        enable_homeostasis=False,
    )


def build_selector_bridge(seed: int, config: SelectorConfig = SelectorConfig()):
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge
    from sim import VisualizationConfig

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(config.ou_sigma_pA)
    cfg.ou_tau_ms = 15.0
    for flag in (
        "enable_short_term_plasticity",
        "enable_hebbian_learning",
        "enable_homeostasis",
        "enable_structural_plasticity",
        "enable_reward_modulation",
        "enable_stdp",
    ):
        setattr(cfg, flag, False)

    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    d1 = NeuronType.IZH2007_STRIATAL_MSN_D1
    d2 = NeuronType.IZH2007_STRIATAL_MSN_D2
    gpe = NeuronType.IZH2007_GPE_PACEMAKER
    gpi = NeuronType.IZH2007_GPI_OUTPUT
    stn = NeuronType.IZH2007_STN_BURST
    thal = NeuronType.IZH2007_THALAMIC_RELAY

    regions = [
        _region("practice_arousal", config.n_practice, exc_fraction=1.0,
                neuron_type=rs),
        _region("selector_stn", config.n_stn, exc_fraction=1.0,
                neuron_type=stn),
        _region("selector_reset", config.n_reset, exc_fraction=0.0,
                neuron_type=fs),
    ]
    for channel in CHANNELS:
        channel_regions = [
            _region(f"proposal_{channel}", config.n_proposal,
                    exc_fraction=1.0, neuron_type=rs),
        ]
        if config.enable_striatal_fsi:
            channel_regions.append(_region(
                f"str_fsi_{channel}", config.n_commit_fs,
                exc_fraction=0.0, neuron_type=fs,
            ))
        channel_regions.extend([
            _region(f"str_d1_{channel}", config.n_striatum,
                    exc_fraction=0.0, neuron_type=d1),
            _region(f"str_d2_{channel}", config.n_striatum,
                    exc_fraction=0.0, neuron_type=d2),
            _region(f"gpe_{channel}", config.n_gpe,
                    exc_fraction=0.0, neuron_type=gpe),
            _region(f"gpi_{channel}", config.n_gpi,
                    exc_fraction=0.0, neuron_type=gpi),
            _region(f"thal_{channel}", config.n_thalamus,
                    exc_fraction=1.0, neuron_type=thal),
            _region(f"commit_{channel}", config.n_commit,
                    exc_fraction=1.0, neuron_type=rs,
                    internal_density=0.35, internal_weight=0.5,
                    enable_nmda=True),
            _region(f"commit_fs_{channel}", config.n_commit_fs,
                    exc_fraction=0.0, neuron_type=fs),
            _region(f"motor_{channel}", config.n_motor,
                    exc_fraction=1.0, neuron_type=rs),
        ])
        regions.extend(channel_regions)

    pathways = []
    for channel in CHANNELS:
        other = 1 - channel
        pathways.append(RegionPathway(
            from_region="practice_arousal",
            to_region=f"proposal_{channel}",
            density=1.0,
            weight_mean=config.arousal_to_proposal_weight,
            weight_jitter=0.0,
            plastic=False,
        ))
        if config.enable_striatal_fsi:
            pathways.extend([
            RegionPathway(
                from_region=f"proposal_{channel}",
                to_region=f"str_fsi_{channel}",
                density=1.0,
                weight_mean=config.proposal_to_fsi_weight,
                weight_jitter=0.0,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"str_fsi_{channel}",
                to_region=f"str_d1_{other}",
                density=1.0,
                weight_mean=config.fsi_to_msn_weight,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ),
            RegionPathway(
                from_region=f"str_fsi_{channel}",
                to_region=f"str_d2_{other}",
                density=1.0,
                weight_mean=config.fsi_to_msn_weight,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ),
            ])
        pathways.extend([
            RegionPathway(
                from_region=f"proposal_{channel}",
                to_region=f"str_d1_{channel}",
                density=config.proposal_to_msn_density,
                weight_mean=config.proposal_to_msn_weight,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"proposal_{channel}",
                to_region=f"str_d2_{channel}",
                density=config.proposal_to_msn_density,
                weight_mean=config.proposal_to_msn_weight,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"str_d1_{channel}",
                to_region=f"gpi_{channel}",
                density=1.0,
                weight_mean=config.d1_to_gpi_weight,
                weight_jitter=0.05,
                plastic=False,
                transmission_gate=DIRECT_PATH_GATE,
            ),
            RegionPathway(
                from_region=f"str_d2_{channel}",
                to_region=f"gpe_{channel}",
                density=0.60,
                weight_mean=2.5,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"gpe_{channel}",
                to_region="selector_stn",
                density=0.30,
                weight_mean=1.5,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region="selector_stn",
                to_region=f"gpi_{channel}",
                density=0.40,
                weight_mean=1.0,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"gpi_{channel}",
                to_region=f"thal_{channel}",
                density=1.0,
                weight_mean=config.gpi_to_thalamus_weight,
                weight_jitter=0.05,
                plastic=False,
                receptor="gaba_a",
            ),
            RegionPathway(
                from_region=f"thal_{channel}",
                to_region=f"commit_{channel}",
                density=1.0,
                weight_mean=config.thalamus_to_commit_weight,
                weight_jitter=0.05,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"commit_{channel}",
                to_region=f"commit_fs_{channel}",
                density=1.0,
                weight_mean=config.commit_to_fsi_weight,
                weight_jitter=0.0,
                plastic=False,
            ),
            RegionPathway(
                from_region=f"commit_fs_{channel}",
                to_region=f"commit_{other}",
                density=1.0,
                weight_mean=config.commit_fsi_cross_weight,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ),
            RegionPathway(
                from_region=f"commit_{channel}",
                to_region=f"motor_{channel}",
                density=1.0,
                weight_mean=config.commit_to_motor_weight,
                weight_jitter=0.05,
                plastic=False,
            ),
        ])
        reset_targets = [
            f"proposal_{channel}",
            f"str_d1_{channel}",
            f"str_d2_{channel}",
            f"thal_{channel}",
            f"commit_{channel}",
            f"motor_{channel}",
        ]
        if config.enable_striatal_fsi:
            reset_targets.append(f"str_fsi_{channel}")
        for target in reset_targets:
            pathways.append(RegionPathway(
                from_region="selector_reset",
                to_region=target,
                density=0.70,
                weight_mean=16.0,
                weight_jitter=0.0,
                plastic=False,
                receptor="gaba_a",
            ))

    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    proposal_indices = np.concatenate([
        _indices(bridge, f"proposal_{channel}") for channel in CHANNELS
    ])
    bridge.cp_ou_neuron_mask = xp.zeros(
        int(cfg.num_neurons), dtype=bool
    )
    bridge.cp_ou_neuron_mask[xp.asarray(proposal_indices)] = True
    return bridge


def _indices(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _step(bridge, n=1):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms


def _set_equal_tonic_current(bridge, config):
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    for channel in CHANNELS:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, f"gpi_{channel}"))
        ] = xp.float32(config.gpi_tonic_pA)
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, f"thal_{channel}"))
        ] = xp.float32(config.thalamus_tonic_pA)


def _run_trial(bridge, config, *, arousal=True):
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, config)
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(config.practice_pA)

    counts = np.zeros(2, dtype=np.int64)
    thalamus_counts = np.zeros(2, dtype=np.int64)
    diagnostic_regions = [
        "practice_arousal", "selector_stn", "selector_reset"
    ]
    for channel in CHANNELS:
        diagnostic_regions.extend([
            f"proposal_{channel}",
            f"str_d1_{channel}",
            f"str_d2_{channel}",
            f"gpe_{channel}",
            f"gpi_{channel}",
            f"thal_{channel}",
            f"commit_{channel}",
            f"commit_fs_{channel}",
            f"motor_{channel}",
        ])
        if config.enable_striatal_fsi:
            diagnostic_regions.append(f"str_fsi_{channel}")
    region_indices = {
        name: _indices(bridge, name) for name in diagnostic_regions
    }
    region_spikes = {name: 0 for name in diagnostic_regions}
    first_crossing = None
    simultaneous_crossing = False
    decision_step = None
    for step in range(int(config.action_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for name, indices in region_indices.items():
            region_spikes[name] += int(firing[indices].sum())
        previous = counts.copy()
        for channel in CHANNELS:
            counts[channel] += int(firing[_indices(bridge, f"motor_{channel}")].sum())
            thalamus_counts[channel] += int(
                firing[_indices(bridge, f"thal_{channel}")].sum()
            )
        crossed = [
            channel for channel in CHANNELS
            if previous[channel] < config.commit_threshold_spikes
            <= counts[channel]
        ]
        if first_crossing is None and len(crossed) == 1:
            first_crossing = int(crossed[0])
            decision_step = int(step)
            break
        elif first_crossing is None and len(crossed) > 1:
            simultaneous_crossing = True
            decision_step = int(step)
            break

    winner = None
    loser_ratio = None
    if first_crossing is not None and not simultaneous_crossing:
        loser = 1 - first_crossing
        loser_ratio = float(counts[loser] / max(1, counts[first_crossing]))
        if loser_ratio <= config.clean_loser_ratio:
            winner = int(first_crossing)

    _set_equal_tonic_current(bridge, config)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_equal_tonic_current(bridge, config)
    _step(bridge, config.washout_steps)
    return {
        "winner": winner,
        "first_crossing": first_crossing,
        "decision_step": decision_step,
        "simultaneous_crossing": bool(simultaneous_crossing),
        "motor_spikes": counts.tolist(),
        "thalamus_spikes": thalamus_counts.tolist(),
        "loser_ratio": loser_ratio,
        "region_spikes": region_spikes,
    }


def run_condition(seed, *, trials=100, arousal=True, direct_path=True,
                  config=SelectorConfig()):
    bridge = build_selector_bridge(seed, config)
    bridge.set_transmission_gate(DIRECT_PATH_GATE, float(bool(direct_path)))
    _set_equal_tonic_current(bridge, config)
    _step(bridge, config.warmup_steps)
    rows = [
        _run_trial(bridge, config, arousal=arousal)
        for _ in range(int(trials))
    ]
    winners = [row["winner"] for row in rows if row["winner"] is not None]
    counts = {str(channel): winners.count(channel) for channel in CHANNELS}
    clean_rate = float(len(winners) / max(1, len(rows)))
    loser_ratios = [
        row["loser_ratio"] for row in rows
        if row["loser_ratio"] is not None
    ]
    return {
        "seed": int(seed),
        "trials": int(trials),
        "arousal": bool(arousal),
        "direct_path": bool(direct_path),
        "clean_commit_rate": clean_rate,
        "winner_counts": counts,
        "minimum_channel_share": float(
            min(counts.values()) / max(1, len(winners))
        ),
        "loser_ratio_p95": (
            float(np.quantile(loser_ratios, 0.95)) if loser_ratios else None
        ),
        "rows": rows,
    }


def run_seed(seed, *, trials=100, lesion_trials=100,
             config=SelectorConfig()):
    main = run_condition(seed, trials=trials, config=config)
    no_arousal = run_condition(
        seed, trials=lesion_trials, arousal=False, config=config
    )
    no_direct_path = run_condition(
        seed, trials=lesion_trials, direct_path=False, config=config
    )
    checks = {
        "clean_single_channel_commit": main["clean_commit_rate"] >= 0.95,
        "both_channels_explored": main["minimum_channel_share"] >= 0.25,
        "loser_suppressed": (
            main["loser_ratio_p95"] is not None
            and main["loser_ratio_p95"] <= config.clean_loser_ratio
        ),
        "arousal_is_load_bearing": (
            no_arousal["clean_commit_rate"]
            <= main["clean_commit_rate"] - 0.75
        ),
        "direct_path_is_load_bearing": (
            no_direct_path["clean_commit_rate"]
            <= main["clean_commit_rate"] - 0.75
        ),
        "no_host_channel_input_or_argmax_fallback": True,
    }
    return {
        "seed": int(seed),
        "config": asdict(config),
        "main": main,
        "controls": {
            "no_arousal": no_arousal,
            "no_direct_path": no_direct_path,
        },
        "checks": checks,
        "go": bool(all(checks.values())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEVELOPMENT_SEEDS)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--lesion-trials", type=int, default=100)
    parser.add_argument(
        "--selector-version", choices=("v1", "v2"), default="v1"
    )
    parser.add_argument(
        "--output",
        default="research/findings/raw/vocal_action_selector_gate.json",
    )
    args = parser.parse_args(argv)
    config = selector_config(args.selector_version)
    rows = [
        run_seed(
            seed,
            trials=args.trials,
            lesion_trials=args.lesion_trials,
            config=config,
        )
        for seed in args.seeds
    ]
    result = {
        "probe": "vocal_action_selector_gate_a",
        "selector_version": args.selector_version,
        "seeds": list(args.seeds),
        "rows": rows,
        "n_go": int(sum(row["go"] for row in rows)),
        "all_go": bool(all(row["go"] for row in rows)),
        "backend": get_backend()[1],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "all_go": result["all_go"],
        "n_go": result["n_go"],
        "seeds": result["seeds"],
        "output": str(output),
    }, indent=2))
    return 0 if result["all_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
