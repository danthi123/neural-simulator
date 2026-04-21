"""G3: Persistence across sessions.

Trains like G2 but supports `save_after` (drop a checkpoint mid-training)
and `start_from` (resume from a checkpoint). The gate asks whether resuming
produces a trajectory that matches a clean run at the same epoch.

Checkpoint bundle: `<name>.simstate.h5` (sim state, including
cp_synapse_plastic_mask) + `<name>.g3.json` (runner state: epoch index,
training-order RNG state, per-epoch history).
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
from research.runners.g2_runner import _build_g2_plan, _present_example_count


def _save_runner_state(path, state):
    """Serialize runner state (epoch, RNG state, history) to JSON."""
    obj = {
        "epoch": state["epoch"],
        "rng_state_json": json.dumps(state["rng_state"], default=_json_safe),
        "history": state["history"],
        "dataset_path": state["dataset_path"],
        "seed": state["seed"],
        "train_N": state["train_N"],
        "test_N": state["test_N"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_safe)


def _load_runner_state(path):
    with open(path) as f:
        obj = json.load(f)
    return {
        "epoch": int(obj["epoch"]),
        "rng_state": json.loads(obj["rng_state_json"]),
        "history": list(obj["history"]),
        "dataset_path": obj["dataset_path"],
        "seed": int(obj["seed"]),
        "train_N": int(obj["train_N"]),
        "test_N": int(obj["test_N"]),
    }


def _json_safe(obj):
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def _rng_from_state(state_dict):
    """Reconstruct a numpy default_rng from its bit_generator state dict."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state_dict
    return rng


def run_g3(
    dataset_path,
    out_path,
    seed,
    n_epochs=6,
    save_after=None,           # epoch index after which to checkpoint
    checkpoint_prefix=None,    # filesystem prefix for checkpoint bundle
    start_from=None,           # same prefix to resume from
    max_train_per_epoch=None,
    max_test_per_epoch=None,
    verbose=True,
):
    import cupy as cp
    from sklearn.linear_model import LogisticRegression

    ds = TinyPatternDataset.load(dataset_path)
    K = int(ds.metadata["K"])

    # ------------- Setup (fresh OR resumed) --------------
    if start_from is not None:
        runner_state = _load_runner_state(f"{start_from}.g3.json")
        assert runner_state["seed"] == seed, \
            f"Checkpoint seed mismatch: ckpt={runner_state['seed']} vs req={seed}"
        # Load the bridge from its simstate.h5
        bridge = SimulationBridge(
            core_config=None, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        bridge.load_checkpoint(f"{start_from}.simstate.h5")
        start_epoch = runner_state["epoch"] + 1
        rng_train = _rng_from_state(runner_state["rng_state"])
        history = list(runner_state["history"])
        train_N = runner_state["train_N"]
        test_N = runner_state["test_N"]

        # Rebuild layout from core config (same reservoir spec as G2).
        n_input = 64
        n_hidden_exc = 160
        n_hidden_inh = 40
        layout = {
            "input_idx": list(range(n_input)),
            "hidden_exc_idx": list(range(n_input, n_input + n_hidden_exc)),
            "hidden_inh_idx": list(range(n_input + n_hidden_exc,
                                         n_input + n_hidden_exc + n_hidden_inh)),
        }
        layout["hidden_idx"] = layout["hidden_exc_idx"] + layout["hidden_inh_idx"]
    else:
        core_cfg, plan = _build_g2_plan(seed=seed)
        bridge = SimulationBridge(
            core_config=core_cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        bridge._initialize_simulation_data(called_from_playback_init=False)
        layout = plan["layout"]
        new_traits = np.zeros(core_cfg.num_neurons, dtype=np.int32)
        for i in layout["hidden_inh_idx"]:
            new_traits[i] = 1
        bridge.cp_traits = cp.asarray(new_traits)
        bridge._cached_inhibitory_mask = None
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=None)
        if bridge.cp_external_input_current is not None:
            bridge.cp_external_input_current[:] = 0.0

        train_N = min(len(ds.X_train), max_train_per_epoch or len(ds.X_train))
        test_N = min(len(ds.X_test), max_test_per_epoch or len(ds.X_test))
        rng_train = np.random.default_rng(seed)
        history = []
        start_epoch = 0

    # ExperimentEngine (fresh for both fresh and resumed paths; it holds
    # only stimulus + group plumbing, nothing load-sensitive).
    engine = ExperimentEngine(bridge.core_config.num_neurons, bridge.core_config.dt_ms)
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
        name="g3", phase_type=ExperimentPhaseType.TRAINING.name,
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

    # ------------- Training loop --------------
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        order = rng_train.permutation(len(ds.X_train))[:train_N]
        X_train_feat = extract_features(ds.X_train[order])
        y_train = ds.y_train[order]
        X_test_feat = extract_features(ds.X_test[:test_N])
        y_test = ds.y_test[:test_N]

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train_feat, y_train)
        train_acc = float(clf.score(X_train_feat, y_train))
        test_acc = float(clf.score(X_test_feat, y_test))

        w_all = cp.asnumpy(bridge.cp_connections.data)
        if bridge.cp_synapse_plastic_mask is not None:
            mask = cp.asnumpy(bridge.cp_synapse_plastic_mask)
            w_plastic = w_all[mask]
        else:
            w_plastic = w_all

        record = {
            "epoch": epoch,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "plastic_weight_mean": float(w_plastic.mean()),
            "plastic_weight_std": float(w_plastic.std()),
            "plastic_weight_min": float(w_plastic.min()),
            "plastic_weight_max": float(w_plastic.max()),
            "time_seconds": time.time() - t0,
        }
        history.append(record)
        if verbose:
            print(f"[g3 seed={seed}] ep {epoch}: "
                  f"train={train_acc:.3f} test={test_acc:.3f} "
                  f"W in [{record['plastic_weight_min']:.3f}, {record['plastic_weight_max']:.3f}] "
                  f"{record['time_seconds']:.1f}s",
                  flush=True)

        # Incremental out-file so progress is visible under buffered stdout.
        results = {
            "seed": seed, "n_epochs": n_epochs,
            "algo": "g3_stdp_then_logreg_resumable",
            "dataset": str(Path(dataset_path).name),
            "dataset_metadata": ds.metadata,
            "resumed_from": start_from,
            "epochs": history,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_safe)

        # Checkpoint if requested.
        if save_after is not None and epoch == save_after and checkpoint_prefix:
            bridge.save_checkpoint(f"{checkpoint_prefix}.simstate.h5")
            _save_runner_state(
                f"{checkpoint_prefix}.g3.json",
                {
                    "epoch": epoch,
                    "rng_state": rng_train.bit_generator.state,
                    "history": history,
                    "dataset_path": str(Path(dataset_path).resolve()),
                    "seed": seed,
                    "train_N": train_N,
                    "test_N": test_N,
                },
            )
            if verbose:
                print(f"[g3 seed={seed}] checkpoint saved at epoch {epoch}: "
                      f"{checkpoint_prefix}.*", flush=True)

    return results
