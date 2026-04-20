"""Headless G1 runner: train + test loop that writes a results JSON.

Architecture per docs/plans/2026-04-20-g1-encoder-decoder-loss-design.md:
- Build the 68-neuron G1 network (64 input + 4 output Izhikevich RS)
- Inject explicit connectivity via `SimulationBridge.inject_explicit_wiring`
- Per example: 200 ms stimulus + 100 ms gap
    - Stimulus: RATE_VECTOR_POISSON on inputs with per-neuron rate from dataset
    - Training: add 400 pA teacher current to correct-class output neuron
    - Readout window: spikes in [100, 200] ms of the stimulus period
- Decoder: argmax over per-output-neuron spike counts
- Supervision: STDP updates driven by teacher-forced firing of correct class
  + fixed lateral inhibition (output trait=1) silencing competitors

Usage (programmatic):
    from research.runners.g1_runner import run_g1
    result = run_g1(
        dataset_path='research/datasets/tiny_patterns.npz',
        out_path='research/findings/raw/g1-seed42.json',
        seed=42, n_epochs=10,
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import (ExperimentConfig, ExperimentPhase,
                        StimulusChannel, StimulusPattern, NeuronGroup,
                        ReadoutConfig)
from sim.enums import (StimulusPatternType, ExperimentPhaseType,
                       NeuronGroupRole)
from experiment import ExperimentEngine

from research.datasets.tiny_patterns import TinyPatternDataset
from research.runners.g1_network import build_g1_network_config, G1NetworkSpec
from research.runners.g1_decoder import compute_metrics


STIMULUS_MS = 200.0
GAP_MS = 100.0
READOUT_START_MS = 100.0
READOUT_END_MS = 200.0
TEACHER_CURRENT_PA = 400.0


def run_g1(
    dataset_path,
    out_path,
    seed,
    n_epochs=10,
    max_train_per_epoch=None,
    max_test_per_epoch=None,
    verbose=True,
):
    import cupy as cp

    ds = TinyPatternDataset.load(dataset_path)
    K = int(ds.metadata["K"])
    n_features = int(ds.metadata["n_features"])
    assert n_features == 64 and K == 4, "G1 network spec assumes 64 features, 4 classes."

    spec = G1NetworkSpec()
    core_cfg, wiring_plan = build_g1_network_config(seed=seed, spec=spec)
    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=viz_cfg,
        runtime_state=runtime_state, gpu_config=gpu_cfg,
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized, "Bridge init failed"

    # Replace auto-generated connectivity with our explicit plan, and mark
    # output neurons as inhibitory so lateral synapses flow through the
    # inhibitory conductance channel.
    bridge.inject_explicit_wiring(
        wiring_plan,
        output_inhibitory_indices=spec.output_indices,
    )

    # Zero out the baseline DC drive — we want only the Poisson input and
    # teacher current to drive the network during G1.
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

    engine = ExperimentEngine(core_cfg.num_neurons, core_cfg.dt_ms)
    exp_cfg = ExperimentConfig()
    exp_cfg.neuron_groups = [
        NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=spec.input_indices),
        NeuronGroup(name="output", role=NeuronGroupRole.OUTPUT.name,
                    neuron_indices=spec.output_indices),
    ]
    exp_cfg.readout = ReadoutConfig(
        rate_window_ms=100.0, spike_count_window_ms=100.0,
        rate_group_names=["input", "output"],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="g1_training", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    # Mark the engine as running so the bridge injects its stimulus current.
    engine.is_experiment_running = True

    # Wire the engine into the bridge so the sim step picks up the stimulus.
    bridge.experiment_engine = engine

    all_results = {
        "seed": seed, "n_epochs": n_epochs,
        "dataset": str(Path(dataset_path).name),
        "dataset_metadata": ds.metadata,
        "spec": {
            "n_input": spec.n_input, "n_output": spec.n_output,
            "weight_max_cap": spec.weight_max_cap,
            "lateral_inhibition_weight": spec.lateral_inhibition_weight,
            "init_weight_range": [spec.init_weight_min, spec.init_weight_max],
        },
        "epochs": [],
    }

    rng = np.random.default_rng(seed)
    train_N = min(len(ds.X_train), max_train_per_epoch or len(ds.X_train))
    test_N = min(len(ds.X_test), max_test_per_epoch or len(ds.X_test))

    # i2o_count is the number of plastic input->output synapses — the first
    # `i2o_count` entries in cp_connections.data (in CSR order may differ;
    # safer: slice the weight array directly from the CSR).
    i2o_count = wiring_plan["input_to_output"]["count"]

    for epoch in range(n_epochs):
        t_epoch = time.time()
        order = rng.permutation(len(ds.X_train))[:train_N]

        train_spike_counts = np.zeros((train_N, K), dtype=np.int32)
        train_labels = ds.y_train[order]
        for i, idx in enumerate(order):
            counts = _present_example(
                bridge, engine, ds.X_train[idx],
                teacher_class=int(ds.y_train[idx]),
                spec=spec, cp=cp,
            )
            train_spike_counts[i] = counts
        train_metrics = compute_metrics(train_spike_counts, train_labels)

        test_spike_counts = np.zeros((test_N, K), dtype=np.int32)
        test_labels = ds.y_test[:test_N]
        for i in range(test_N):
            counts = _present_example(
                bridge, engine, ds.X_test[i],
                teacher_class=None,
                spec=spec, cp=cp,
            )
            test_spike_counts[i] = counts
        test_metrics = compute_metrics(test_spike_counts, test_labels)

        # Extract input->output weights. They live in cp_connections.data;
        # for the CSR built by inject_explicit_wiring, entries with pre in
        # [0..63] and post in [64..67] are the plastic ones.
        w_all = bridge.cp_connections
        # Convert to COO to get (pre, post, weight) triples.
        coo = w_all.tocoo(copy=False)
        pre_h = cp.asnumpy(coo.row)
        post_h = cp.asnumpy(coo.col)
        wvals = cp.asnumpy(coo.data)
        i2o_mask = (pre_h < spec.n_input) & (post_h >= spec.n_input)
        w_i2o = wvals[i2o_mask]

        epoch_record = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "mean_margin_train": train_metrics["mean_margin"],
            "mean_margin_test": test_metrics["mean_margin"],
            "mean_weight": float(w_i2o.mean()) if w_i2o.size else 0.0,
            "weight_std": float(w_i2o.std()) if w_i2o.size else 0.0,
            "weight_min": float(w_i2o.min()) if w_i2o.size else 0.0,
            "weight_max": float(w_i2o.max()) if w_i2o.size else 0.0,
            "train_confusion": train_metrics["confusion"].tolist(),
            "test_confusion": test_metrics["confusion"].tolist(),
            "time_seconds": time.time() - t_epoch,
        }
        all_results["epochs"].append(epoch_record)
        if verbose:
            print(
                f"[seed={seed}] Epoch {epoch}: "
                f"train_acc={epoch_record['train_accuracy']:.3f}  "
                f"test_acc={epoch_record['test_accuracy']:.3f}  "
                f"margin_test={epoch_record['mean_margin_test']:+.2f}  "
                f"W in [{epoch_record['weight_min']:.3f}, {epoch_record['weight_max']:.3f}]  "
                f"{epoch_record['time_seconds']:.1f}s"
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    return all_results


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _present_example(bridge, engine, rate_vector, teacher_class, spec, cp):
    """Step the sim through one example; return (n_output,) spike counts in the readout window."""
    dt = bridge.core_config.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    n_gap_steps = int(GAP_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    # Stimulus channel: per-neuron Poisson rate from this example's rate vector.
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=250.0,
        spike_duration_ms=1.0,
        rate_vector_hz=[float(r) for r in rate_vector],
    )
    ch_input = StimulusChannel(
        name="input_pattern", pattern=pat,
        target_neuron_indices=spec.input_indices,
        onset_ms=0.0, duration_ms=STIMULUS_MS,
        enabled=True,
    )
    channels = [ch_input]

    # Teacher channel: constant 400 pA to the correct-class output neuron,
    # injected only during the stimulus window and only during training.
    if teacher_class is not None:
        teacher_idx = spec.output_indices[teacher_class]
        teacher_pat = StimulusPattern(
            pattern_type=StimulusPatternType.CONSTANT.name,
            amplitude_pA=TEACHER_CURRENT_PA,
        )
        ch_teacher = StimulusChannel(
            name="teacher", pattern=teacher_pat,
            target_neuron_indices=[teacher_idx],
            onset_ms=0.0, duration_ms=STIMULUS_MS,
            enabled=True,
        )
        channels.append(ch_teacher)

    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize(channels, engine.group_manager, cp)
    engine.phase_start_ms = bridge.runtime_state.current_time_ms

    counts = np.zeros(spec.n_output, dtype=np.int32)
    out_idx_cp = cp.asarray(spec.output_indices, dtype=cp.int32)

    for step in range(n_stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

        if readout_start_step <= step < readout_end_step:
            fired = bridge.cp_firing_states[out_idx_cp].get()
            counts += fired.astype(np.int32)

    # GAP: stimulus off
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([], engine.group_manager, cp)
    for step in range(n_gap_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

    return counts
