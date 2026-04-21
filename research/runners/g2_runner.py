"""G2: Sim-local STDP bends the learning curve.

v3 (G1 GO) used a frozen reservoir + external LogReg. v3 proved the scaffolding
works (mean 71.3% test across 3 seeds). G2 asks: does turning on STDP inside
the sim — on the input->hidden projection, with the reservoir still fixed —
improve the features the external LogReg sees?

Success = test accuracy climbs across epochs.
NO-GO = flat or dropping curve.
PARTIAL = change < 3 pp.

Architecture:
- Same 264-neuron reservoir as v3 (64 input + 160 hidden exc + 40 hidden inh).
- input->hidden: STDP-plastic, symmetric A+/A- (0.008 each) to avoid the
  runaway LTP we saw in G1.v1.
- hidden->hidden: frozen via the per-synapse plastic mask.
- Per epoch: run all training examples through the sim (STDP updates weights);
  extract train+test features; train fresh LogReg on train features; score.
- 8 epochs per seed, 3 seeds.
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
READOUT_START_MS = 50.0
READOUT_END_MS = 200.0


def _build_g2_plan(
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
    """Same reservoir topology as v3, but input->hidden is plastic=True
    and hidden->hidden is plastic=False.
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
    core_cfg.connections_per_neuron = 0

    # STDP on. Symmetric rates — the G1.v1 post-mortem showed A+ > A-
    # caused uniform weight inflation; symmetric means net change tracks
    # co-activity asymmetry (pre-before-post) without drift.
    core_cfg.enable_stdp = True
    core_cfg.stdp_a_plus = 0.008
    core_cfg.stdp_a_minus = 0.008
    core_cfg.stdp_tau_plus_ms = 20.0
    core_cfg.stdp_tau_minus_ms = 20.0
    core_cfg.stdp_w_min = 0.0
    core_cfg.stdp_w_max = 3.0

    # All other plasticity off.
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_watts_strogatz = False

    # Boosted propagation (same as v3).
    core_cfg.propagation_strength = 1.0
    core_cfg.inhibitory_propagation_strength = 1.0
    core_cfg.ou_std_current_pA = 60.0

    rng = np.random.default_rng(seed)

    # Input -> hidden (PLASTIC).
    pre_ih, post_ih = [], []
    for i in input_idx:
        n_conn = max(1, int(len(hidden_idx) * input_to_hidden_density))
        targets = rng.choice(hidden_idx, size=n_conn, replace=False)
        for t in targets:
            pre_ih.append(i)
            post_ih.append(int(t))
    w_ih = np.clip(
        rng.normal(input_to_hidden_weight, input_to_hidden_weight * 0.2,
                   size=len(pre_ih)),
        0.01, None,
    ).astype(np.float32)

    # Hidden -> hidden (FIXED — the reservoir).
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

    plan = {
        "input_to_hidden": {
            "pre_indices": pre_ih,
            "post_indices": post_ih,
            "initial_weights": w_ih,
            "plastic": True,       # STDP updates these
            "conn_type": "E_TO_E_AND_E_TO_I",
            "count": len(pre_ih),
        },
        "hidden_recurrent": {
            "pre_indices": pre_hh,
            "post_indices": post_hh,
            "initial_weights": w_hh,
            "plastic": False,      # reservoir is frozen
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
    return core_cfg, plan


def run_g2(
    dataset_path,
    out_path,
    seed,
    n_epochs=8,
    max_train_per_epoch=None,
    max_test_per_epoch=None,
    verbose=True,
):
    import cupy as cp
    from sklearn.linear_model import LogisticRegression

    ds = TinyPatternDataset.load(dataset_path)
    K = int(ds.metadata["K"])

    core_cfg, plan = _build_g2_plan(seed=seed)
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized

    layout = plan["layout"]
    n_total = core_cfg.num_neurons
    new_traits = np.zeros(n_total, dtype=np.int32)
    for i in layout["hidden_inh_idx"]:
        new_traits[i] = 1
    bridge.cp_traits = cp.asarray(new_traits)
    bridge._cached_inhibitory_mask = None

    bridge.inject_explicit_wiring(plan, output_inhibitory_indices=None)

    # Sanity: plastic mask is built.
    assert bridge.cp_synapse_plastic_mask is not None, \
        "G2 requires the plastic mask (hidden->hidden should be frozen)."
    n_plastic = int(cp.asnumpy(bridge.cp_synapse_plastic_mask).sum())
    if verbose:
        print(f"[g2 seed={seed}] {n_plastic} plastic synapses, "
              f"{bridge.cp_connections.nnz - n_plastic} frozen")

    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

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
        name="g2", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    n_hidden = len(layout["hidden_idx"])
    hidden_idx_cp = cp.asarray(layout["hidden_idx"], dtype=cp.int32)

    def extract_features(X):
        F = np.zeros((len(X), n_hidden), dtype=np.float32)
        for i in range(len(X)):
            F[i] = _present_example_count(
                bridge, engine, X[i], layout, hidden_idx_cp, cp,
            )
        return F

    train_N = min(len(ds.X_train), max_train_per_epoch or len(ds.X_train))
    test_N = min(len(ds.X_test), max_test_per_epoch or len(ds.X_test))
    rng_train = np.random.default_rng(seed)

    results = {
        "seed": seed, "n_epochs": n_epochs,
        "algo": "g2_stdp_input_to_hidden_then_logreg",
        "dataset": str(Path(dataset_path).name),
        "dataset_metadata": ds.metadata,
        "n_plastic_synapses": n_plastic,
        "n_frozen_synapses": bridge.cp_connections.nnz - n_plastic,
        "n_train_used": int(train_N),
        "n_test_used": int(test_N),
        "epochs": [],
    }

    for epoch in range(n_epochs):
        t0 = time.time()
        # Train pass: shuffle, present, STDP updates weights.
        order = rng_train.permutation(len(ds.X_train))[:train_N]
        X_train_feat = extract_features(ds.X_train[order])
        y_train = ds.y_train[order]

        # Test pass.
        X_test_feat = extract_features(ds.X_test[:test_N])
        y_test = ds.y_test[:test_N]

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train_feat, y_train)
        train_acc = clf.score(X_train_feat, y_train)
        test_acc = clf.score(X_test_feat, y_test)

        # Plastic weight stats (only input->hidden).
        w_all = cp.asnumpy(bridge.cp_connections.data)
        mask = cp.asnumpy(bridge.cp_synapse_plastic_mask)
        w_plastic = w_all[mask]

        epoch_record = {
            "epoch": epoch,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "mean_hidden_rate_hz_train": float(
                X_train_feat.mean() * 1000.0 / (READOUT_END_MS - READOUT_START_MS)
            ),
            "mean_hidden_rate_hz_test": float(
                X_test_feat.mean() * 1000.0 / (READOUT_END_MS - READOUT_START_MS)
            ),
            "plastic_weight_mean": float(w_plastic.mean()),
            "plastic_weight_std": float(w_plastic.std()),
            "plastic_weight_min": float(w_plastic.min()),
            "plastic_weight_max": float(w_plastic.max()),
            "time_seconds": time.time() - t0,
        }
        results["epochs"].append(epoch_record)
        if verbose:
            print(
                f"[g2 seed={seed}] ep {epoch}: "
                f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
                f"W in [{epoch_record['plastic_weight_min']:.3f}, "
                f"{epoch_record['plastic_weight_max']:.3f}] "
                f"(mean {epoch_record['plastic_weight_mean']:.3f})  "
                f"hid rate={epoch_record['mean_hidden_rate_hz_train']:.1f} Hz  "
                f"{epoch_record['time_seconds']:.1f}s",
                flush=True,
            )

        # Write incrementally so a caller polling the JSON file sees progress.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def _present_example_count(bridge, engine, rate_vector, layout, hidden_idx_cp, cp):
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
