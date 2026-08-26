"""G1.v2: Perceptron-style reward-modulated co-activity learning.

Why v2: v1 used teacher-forced STDP and failed — the LTP bias drove uniform
saturation instead of class-specific selectivity. See
`research/findings/2026-04-20-g1.md` for the post-mortem.

Design:
- Same 64-input + 4-output topology as v1.
- SimulationBridge runs as a forward pass only. All sim plasticity disabled
  (STDP off, Hebbian off, structural off, homeostasis optional).
- Per example:
    1. Present RATE_VECTOR_POISSON stimulus for 200 ms + 100 ms gap.
    2. Record per-input spike counts (input neurons 0..63) in the readout
       window [100, 200] ms — call this `input_activity` (dim 64).
    3. Record per-output spike counts (neurons 64..67) in the readout window
       — call this `output_activity` (dim 4).
    4. Predict: argmax(output_activity). Tie-break lowest index.
    5. Apply perceptron delta:
         target = one_hot(correct_class)        # shape (4,)
         predicted = softmax(output_activity)   # shape (4,), smooth gradient
         error = target - predicted             # shape (4,)
         dW[k, i] = lr * error[k] * input_activity[i]
       Apply dW to cp_connections.data for input->output synapses only.
       Clip W to [0, stdp_w_max].
- Key: this is a *local* rule in the sense that each weight update depends
  only on pre and post spiking and an external scalar error signal. No
  gradient descent through the sim.

Success criterion: test_accuracy > 55% by epoch 9, monotone-ish learning
curve, each seed >= 45%.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import (ExperimentConfig, ExperimentPhase, StimulusChannel,
                        StimulusPattern, NeuronGroup, ReadoutConfig)
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


def _softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x / max(temp, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_g1_v2(
    dataset_path,
    out_path,
    seed,
    n_epochs=10,
    max_train_per_epoch=None,
    max_test_per_epoch=None,
    learning_rate=0.001,
    initial_weight_mean=0.3,
    initial_weight_std=0.05,
    lateral_inhibition_weight=1.0,
    verbose=True,
):
    import cupy as cp

    ds = TinyPatternDataset.load(dataset_path)
    K = int(ds.metadata["K"])
    n_features = int(ds.metadata["n_features"])
    assert n_features == 64 and K == 4

    # Build the network using v1's wiring helper, then override weights to
    # higher random init + disable ALL sim plasticity.
    spec = G1NetworkSpec()
    spec.lateral_inhibition_weight = lateral_inhibition_weight
    core_cfg, wiring_plan = build_g1_network_config(seed=seed, spec=spec)
    # Disable sim plasticity — v2 learning is external.
    core_cfg.enable_stdp = False
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_short_term_plasticity = False
    # Sim defaults propagation_strength=0.05 assume thousands of converging synapses.
    # This G1 network has only 64 inputs per output — boost propagation so
    # pre-synaptic spikes can actually drive outputs. Izhikevich RS rheobase
    # ~60 pA, and the quadratic leak pulls voltage back below threshold unless
    # input current > ~100 pA. 3.0 hits that window.
    core_cfg.propagation_strength = 3.0
    core_cfg.inhibitory_propagation_strength = 3.0

    # Higher initial weights — we want outputs to fire early so we get
    # signal for the perceptron update.
    rng = np.random.default_rng(seed)
    w_i2o = np.clip(
        rng.normal(initial_weight_mean, initial_weight_std,
                   size=wiring_plan["input_to_output"]["count"]),
        0.01, spec.weight_max_cap,
    ).astype(np.float32)
    wiring_plan["input_to_output"]["initial_weights"] = w_i2o

    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=viz_cfg,
        runtime_state=runtime_state, gpu_config=gpu_cfg,
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized
    bridge.inject_explicit_wiring(
        wiring_plan, output_inhibitory_indices=spec.output_indices
    )
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
        name="g1v2", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    all_results = {
        "seed": seed, "n_epochs": n_epochs,
        "dataset": str(Path(dataset_path).name),
        "dataset_metadata": ds.metadata,
        "algo": "g1_v2_perceptron_co_activity",
        "learning_rate": learning_rate,
        "initial_weight_mean": initial_weight_mean,
        "lateral_inhibition_weight": lateral_inhibition_weight,
        "spec": {
            "n_input": spec.n_input, "n_output": spec.n_output,
            "weight_max_cap": spec.weight_max_cap,
        },
        "epochs": [],
    }

    # Precompute the ordering of input->output synapses in cp_connections.data
    # so we know where to apply weight updates. The CSR was built via coo->csr;
    # rebuild a (pre, post) -> flat_idx map once.
    coo = bridge.cp_connections.tocoo(copy=False)
    pre_h = cp.asnumpy(coo.row).astype(np.int32)
    post_h = cp.asnumpy(coo.col).astype(np.int32)
    # cp_connections.data indices correspond to CSR-sorted (pre, post) pairs.
    # Use tocoo(copy=False) ordering, which matches the internal data array.
    i2o_mask = (pre_h < spec.n_input) & (post_h >= spec.n_input)
    i2o_flat_indices = np.where(i2o_mask)[0]  # indices into cp_connections.data
    i2o_pre = pre_h[i2o_flat_indices]
    i2o_post = post_h[i2o_flat_indices] - spec.n_input  # 0..3

    rng_train = np.random.default_rng(seed)
    train_N = min(len(ds.X_train), max_train_per_epoch or len(ds.X_train))
    test_N = min(len(ds.X_test), max_test_per_epoch or len(ds.X_test))

    for epoch in range(n_epochs):
        t_epoch = time.time()
        order = rng_train.permutation(len(ds.X_train))[:train_N]

        train_spike_counts = np.zeros((train_N, K), dtype=np.int32)
        train_labels = ds.y_train[order]

        for i, idx in enumerate(order):
            out_counts, in_counts = _present_example(
                bridge, engine, ds.X_train[idx], spec=spec, cp=cp,
            )
            train_spike_counts[i] = out_counts

            # Perceptron co-activity update — outside the sim.
            target = np.zeros(K, dtype=np.float64)
            target[int(ds.y_train[idx])] = 1.0
            pred = _softmax(out_counts.astype(np.float64), temp=max(1.0, out_counts.max()))
            error = target - pred  # shape (K,)

            # Normalize input activity to [0, 1] so lr magnitude is stable.
            in_norm = in_counts.astype(np.float64) / max(in_counts.max(), 1.0)

            # Apply dW[k, i] = lr * error[k] * in_norm[i] to cp_connections.data.
            # Build a flat delta vector aligned with i2o_flat_indices.
            dW = learning_rate * error[i2o_post] * in_norm[i2o_pre]  # shape (256,)
            # Apply on GPU.
            dW_cp = cp.asarray(dW.astype(np.float32))
            flat_cp = cp.asarray(i2o_flat_indices)
            bridge.cp_connections.data[flat_cp] += dW_cp
            # Clip to valid range.
            lo = cp.float32(0.0)
            hi = cp.float32(spec.weight_max_cap)
            cp.clip(bridge.cp_connections.data, lo, hi, out=bridge.cp_connections.data)

        train_metrics = compute_metrics(train_spike_counts, train_labels)

        # Test pass — no updates.
        test_spike_counts = np.zeros((test_N, K), dtype=np.int32)
        test_labels = ds.y_test[:test_N]
        for i in range(test_N):
            out_counts, _ = _present_example(
                bridge, engine, ds.X_test[i], spec=spec, cp=cp,
            )
            test_spike_counts[i] = out_counts
        test_metrics = compute_metrics(test_spike_counts, test_labels)

        # Weight stats.
        w_vals = cp.asnumpy(bridge.cp_connections.data[cp.asarray(i2o_flat_indices)])

        epoch_record = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "mean_margin_train": train_metrics["mean_margin"],
            "mean_margin_test": test_metrics["mean_margin"],
            "mean_weight": float(w_vals.mean()),
            "weight_std": float(w_vals.std()),
            "weight_min": float(w_vals.min()),
            "weight_max": float(w_vals.max()),
            "train_confusion": train_metrics["confusion"].tolist(),
            "test_confusion": test_metrics["confusion"].tolist(),
            "time_seconds": time.time() - t_epoch,
        }
        all_results["epochs"].append(epoch_record)
        if verbose:
            print(
                f"[v2 seed={seed}] Epoch {epoch}: "
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


def _present_example(bridge, engine, rate_vector, spec, cp):
    """Run one example; return (output_counts [4], input_counts [64]) over readout window."""
    dt = bridge.core_config.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    n_gap_steps = int(GAP_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=1000.0,
        spike_duration_ms=2.0,
        rate_vector_hz=[float(r) for r in rate_vector],
    )
    ch_input = StimulusChannel(
        name="input_pattern", pattern=pat,
        target_neuron_indices=spec.input_indices,
        onset_ms=0.0, duration_ms=STIMULUS_MS,
        enabled=True,
    )
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([ch_input], engine.group_manager, cp)
    engine.phase_start_ms = bridge.runtime_state.current_time_ms

    out_counts = np.zeros(spec.n_output, dtype=np.int32)
    in_counts = np.zeros(spec.n_input, dtype=np.int32)
    out_idx_cp = cp.asarray(spec.output_indices, dtype=cp.int32)
    in_idx_cp = cp.asarray(spec.input_indices, dtype=cp.int32)

    for step in range(n_stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

        if readout_start_step <= step < readout_end_step:
            fired_out = bridge.cp_firing_states[out_idx_cp].get().astype(np.int32)
            out_counts += fired_out
            fired_in = bridge.cp_firing_states[in_idx_cp].get().astype(np.int32)
            in_counts += fired_in

    # GAP — stimulus off
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([], engine.group_manager, cp)
    for step in range(n_gap_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

    return out_counts, in_counts
