"""G1.v3: Reservoir + external linear readout.

v1 (teacher-forced STDP) and v2 (perceptron co-activity) both NO-GO because
the sim's forward pass is too weak in a 64-input → 4-output direct mapping.
See `research/findings/2026-04-20-g1.md` for the post-mortem.

v3 uses the sim in its calibrated regime:
- 64 input neurons (Poisson-driven).
- 200 hidden neurons (160 excitatory + 40 inhibitory), sparse recurrent
  connectivity (~15% density).
- Input → hidden projection, sparse (~25% density).
- NO output neurons in the sim. The readout is external: an
  `sklearn.linear_model.LogisticRegression` trained on hidden-neuron spike
  counts over the readout window.

Why this should work:
- The dataset is linearly separable (logreg on raw rates = 100% test).
- Hidden spike counts are a noisy, higher-dim feature representation of the
  rate vector. A linear classifier can pick up the signal.
- The sim runs in its default parameter regime (propagation_strength=0.05,
  ~thousands of converging connections per neuron per sim design), so
  synaptic dynamics work as intended.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import (ExperimentConfig, ExperimentPhase,
                        StimulusChannel, StimulusPattern, NeuronGroup,
                        ReadoutConfig, CoreSimConfig)
from sim.enums import (StimulusPatternType, ExperimentPhaseType,
                       NeuronGroupRole, NeuronModel)
from experiment import ExperimentEngine

from research.datasets.tiny_patterns import TinyPatternDataset


STIMULUS_MS = 200.0
GAP_MS = 100.0
READOUT_START_MS = 50.0  # Skip the first 50 ms transient.
READOUT_END_MS = 200.0


def _build_reservoir_plan(
    seed,
    n_input=64,
    n_hidden_exc=160,
    n_hidden_inh=40,
    input_to_hidden_density=0.5,
    hidden_to_hidden_density=0.1,
    input_to_hidden_weight=1.5,
    hidden_exc_weight=0.3,
    hidden_inh_weight=0.8,
):
    """Build a (CoreSimConfig, wiring_plan) pair for the reservoir network.

    Traits: 0 = excitatory, 1 = inhibitory.
    Input indices: 0..63 (trait 0).
    Hidden excitatory: 64..223 (trait 0).
    Hidden inhibitory: 224..263 (trait 1).
    """
    n_total = n_input + n_hidden_exc + n_hidden_inh
    input_idx = list(range(0, n_input))
    hidden_exc_idx = list(range(n_input, n_input + n_hidden_exc))
    hidden_inh_idx = list(range(n_input + n_hidden_exc, n_total))
    hidden_idx = hidden_exc_idx + hidden_inh_idx

    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = n_total
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core_cfg.seed = int(seed)
    core_cfg.dt_ms = 1.0
    core_cfg.num_traits = 2
    core_cfg.inhibitory_trait_indices = [1]
    core_cfg.connections_per_neuron = 0  # Overridden by explicit wiring

    # Disable all plasticity — reservoir is fixed, readout is external.
    core_cfg.enable_stdp = False
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_watts_strogatz = False
    # Boost propagation — even with 200 hidden neurons, each one only gets ~32
    # incoming input connections (64 inputs × 0.5 density). The sim default
    # propagation_strength=0.05 assumes ~1000 converging connections per neuron.
    core_cfg.propagation_strength = 1.0
    core_cfg.inhibitory_propagation_strength = 1.0

    # Keep modest OU noise so the reservoir has some background dynamics.
    core_cfg.ou_std_current_pA = 60.0

    rng = np.random.default_rng(seed)

    # Input → hidden (sparse): each input connects to input_to_hidden_density
    # of hidden neurons with excitatory weight.
    pre_ih, post_ih = [], []
    for i in input_idx:
        n_conn = max(1, int(len(hidden_idx) * input_to_hidden_density))
        targets = rng.choice(hidden_idx, size=n_conn, replace=False)
        for t in targets:
            pre_ih.append(i)
            post_ih.append(int(t))
    w_ih = rng.normal(input_to_hidden_weight, input_to_hidden_weight * 0.2,
                      size=len(pre_ih)).astype(np.float32)
    w_ih = np.clip(w_ih, 0.01, None)

    # Hidden → hidden (recurrent, sparse): each hidden connects to
    # hidden_to_hidden_density of other hidden neurons. Excitatory pre-neurons
    # get hidden_exc_weight, inhibitory pre-neurons get hidden_inh_weight.
    pre_hh, post_hh = [], []
    weights_hh = []
    for i in hidden_idx:
        is_inh = i in hidden_inh_idx
        candidates = [j for j in hidden_idx if j != i]
        n_conn = max(1, int(len(candidates) * hidden_to_hidden_density))
        targets = rng.choice(candidates, size=n_conn, replace=False)
        base_w = hidden_inh_weight if is_inh else hidden_exc_weight
        for t in targets:
            pre_hh.append(i)
            post_hh.append(int(t))
            weights_hh.append(base_w + rng.normal(0, base_w * 0.2))
    w_hh = np.clip(np.asarray(weights_hh, dtype=np.float32), 0.01, None)

    wiring_plan = {
        "input_to_hidden": {
            "pre_indices": pre_ih,
            "post_indices": post_ih,
            "initial_weights": w_ih,
            "plastic": False,
            "conn_type": "E_TO_E_AND_E_TO_I",
            "count": len(pre_ih),
        },
        "hidden_recurrent": {
            "pre_indices": pre_hh,
            "post_indices": post_hh,
            "initial_weights": w_hh,
            "plastic": False,
            "conn_type": "MIXED",
            "count": len(pre_hh),
        },
        "layout": {
            "input_idx": input_idx,
            "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx,
            "hidden_idx": hidden_idx,
        },
    }
    return core_cfg, wiring_plan


def run_g1_v3(
    dataset_path,
    out_path,
    seed,
    n_epochs=1,          # Linear readout — one pass is enough. Multi-epoch
                         # is only useful if we re-sample hidden states.
    max_train_per_epoch=None,
    max_test_per_epoch=None,
    verbose=True,
):
    import cupy as cp
    from sklearn.linear_model import LogisticRegression

    ds = TinyPatternDataset.load(dataset_path)
    K = int(ds.metadata["K"])

    core_cfg, wiring_plan = _build_reservoir_plan(seed=seed)
    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=viz_cfg,
        runtime_state=runtime_state, gpu_config=gpu_cfg,
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized

    # Set traits explicitly before injecting wiring so the bridge's inhibitory
    # routing cache gets the right mask.
    layout = wiring_plan["layout"]
    n_total = core_cfg.num_neurons
    new_traits = np.zeros(n_total, dtype=np.int32)
    for i in layout["hidden_inh_idx"]:
        new_traits[i] = 1
    bridge.cp_traits = cp.asarray(new_traits)
    bridge._cached_inhibitory_mask = None

    bridge.inject_explicit_wiring(wiring_plan, output_inhibitory_indices=None)

    # Zero baseline DC drive — Poisson input is the only input to the network.
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

    # Wire up minimal experiment engine (just for stimulus_manager).
    engine = ExperimentEngine(core_cfg.num_neurons, core_cfg.dt_ms)
    exp_cfg = ExperimentConfig()
    exp_cfg.neuron_groups = [
        NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["input_idx"]),
        NeuronGroup(name="hidden", role=NeuronGroupRole.HIDDEN.name,
                    neuron_indices=layout["hidden_idx"]),
    ]
    exp_cfg.readout = ReadoutConfig(
        rate_window_ms=100.0, spike_count_window_ms=100.0,
        rate_group_names=["input", "hidden"],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="reservoir", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    # Feature extraction pass
    n_hidden = len(layout["hidden_idx"])
    hidden_idx_cp = cp.asarray(layout["hidden_idx"], dtype=cp.int32)

    def extract_features(X):
        n = len(X)
        F = np.zeros((n, n_hidden), dtype=np.float32)
        for i in range(n):
            F[i] = _present_example_and_count(
                bridge, engine, X[i], layout, hidden_idx_cp, cp,
            )
        return F

    t_total = time.time()
    train_N = min(len(ds.X_train), max_train_per_epoch or len(ds.X_train))
    test_N = min(len(ds.X_test), max_test_per_epoch or len(ds.X_test))

    t_ext = time.time()
    X_train_feat = extract_features(ds.X_train[:train_N])
    if verbose:
        print(f"[v3 seed={seed}] extracted {train_N} train features in "
              f"{time.time()-t_ext:.1f}s  "
              f"(mean hidden spikes per example: {X_train_feat.mean()*n_hidden:.1f})")

    t_ext = time.time()
    X_test_feat = extract_features(ds.X_test[:test_N])
    if verbose:
        print(f"[v3 seed={seed}] extracted {test_N} test features in "
              f"{time.time()-t_ext:.1f}s")

    y_train = ds.y_train[:train_N]
    y_test = ds.y_test[:test_N]

    # Train linear readout
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X_train_feat, y_train)
    train_acc = clf.score(X_train_feat, y_train)
    test_acc = clf.score(X_test_feat, y_test)

    # Confusion matrix
    y_pred_test = clf.predict(X_test_feat)
    confusion = np.zeros((K, K), dtype=np.int32)
    for y, p in zip(y_test, y_pred_test):
        confusion[int(y), int(p)] += 1

    all_results = {
        "seed": seed,
        "algo": "g1_v3_reservoir_logreg",
        "dataset": str(Path(dataset_path).name),
        "dataset_metadata": ds.metadata,
        "n_input": 64,
        "n_hidden_exc": 160,
        "n_hidden_inh": 40,
        "n_train_used": int(train_N),
        "n_test_used": int(test_N),
        "mean_hidden_rate_hz_train": float(X_train_feat.mean() * 1000.0 /
                                           (READOUT_END_MS - READOUT_START_MS)),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_confusion": confusion.tolist(),
        "total_time_seconds": time.time() - t_total,
    }
    if verbose:
        print(f"[v3 seed={seed}] train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
              f"mean hidden rate={all_results['mean_hidden_rate_hz_train']:.1f} Hz")
        print(f"  test_confusion:\n{confusion}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def _present_example_and_count(bridge, engine, rate_vector, layout, hidden_idx_cp, cp):
    """Stimulate input with this rate vector; return hidden-neuron spike counts
    in [READOUT_START_MS, READOUT_END_MS) of the stimulus period."""
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
    ch = StimulusChannel(
        name="input_pattern", pattern=pat,
        target_neuron_indices=layout["input_idx"],
        onset_ms=0.0, duration_ms=STIMULUS_MS,
        enabled=True,
    )
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([ch], engine.group_manager, cp)
    engine.phase_start_ms = bridge.runtime_state.current_time_ms

    n_hidden = len(layout["hidden_idx"])
    counts = np.zeros(n_hidden, dtype=np.int32)

    for step in range(n_stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

        if readout_start_step <= step < readout_end_step:
            fired = bridge.cp_firing_states[hidden_idx_cp].get().astype(np.int32)
            counts += fired

    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([], engine.group_manager, cp)
    for step in range(n_gap_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

    return counts
