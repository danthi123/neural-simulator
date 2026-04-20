# G1 Implementation Plan — Dataset → Encoder → Sim → Decoder → Loss

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the minimum viable pipeline from a Poisson-pattern dataset on disk, through the existing SimulationBridge, to a decoded class prediction and a measurable loss — then run a 3-seed × 10-epoch training sweep to establish the G1 baseline learning curve.

**Architecture:** Single-layer supervised-STDP classifier. 64 Poisson-driven input neurons → 4 plastic output neurons with lateral inhibition. Teacher current drives the correct class during training; at test time the decoder argmaxes over output spike counts. Details: `docs/plans/2026-04-20-g1-encoder-decoder-loss-design.md`.

**Tech stack:** Python 3.12, CuPy, NumPy, h5py, pytest. `SimulationBridge` (`sim/bridge.py`), `ExperimentEngine` (`experiment/engine.py`), `StimulusManager` (`experiment/stimulus.py`). Run tests with `python -m pytest` — the top-level `pytest` binary on this machine points at a different env that lacks `h5py`.

**Commit cadence:** Every step with a code or doc change lands as its own commit after its tests are green. Run `python -m pytest tests/` before each commit.

---

## Task 1 — Tiny-patterns dataset module

### 1.1 Failing test: stable dataset generation

**Files:**
- Create: `tests/test_tiny_patterns_dataset.py`

**Step 1: write the failing test.**

```python
# tests/test_tiny_patterns_dataset.py
import os
import hashlib
import numpy as np
import pytest

from research.datasets.tiny_patterns import TinyPatternDataset, build_dataset


def test_build_dataset_stable_with_same_seed(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)

    ds1 = TinyPatternDataset.load(out)
    ds2 = TinyPatternDataset.load(out)

    assert ds1.X_train.shape == (200, 64)
    assert ds1.y_train.shape == (200,)
    assert ds1.X_test.shape == (50, 64)
    assert ds1.y_test.shape == (50,)
    assert ds1.X_train.dtype == np.float32
    assert ds1.y_train.dtype == np.int32
    assert np.array_equal(ds1.X_train, ds2.X_train)
    assert np.array_equal(ds1.y_train, ds2.y_train)


def test_rates_clipped_to_range(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=500, n_test=100, noise_sigma=20.0,  # huge noise to stress clipping
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    assert ds.X_train.min() >= 1.0
    assert ds.X_train.max() <= 40.0
    assert ds.X_test.min() >= 1.0
    assert ds.X_test.max() <= 40.0


def test_classes_well_separated(tmp_path):
    """Class means should be distinguishable: min pairwise L2 distance > threshold."""
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=400, n_test=100, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)

    class_means = np.zeros((4, 64), dtype=np.float32)
    for k in range(4):
        class_means[k] = ds.X_train[ds.y_train == k].mean(axis=0)

    # Pairwise distances between class means
    min_dist = float('inf')
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(class_means[i] - class_means[j]))
            min_dist = min(min_dist, d)
    # Empirical threshold: with rate_range ~40 Hz and 64 features, any reasonable
    # class-mean-vector scheme should give >30 Hz pairwise L2 distance.
    assert min_dist > 30.0, f"Classes not separated: min pairwise L2 distance = {min_dist:.2f}"


def test_labels_balanced(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    for k in range(4):
        assert (ds.y_train == k).sum() >= 40
        assert (ds.y_test == k).sum() >= 8


def test_metadata_roundtrip(tmp_path):
    out = tmp_path / "tp.npz"
    build_dataset(out, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=200, n_test=50, noise_sigma=4.0,
                  rate_min=1.0, rate_max=40.0)
    ds = TinyPatternDataset.load(out)
    assert ds.metadata["seed"] == 0xD47A5E7
    assert ds.metadata["K"] == 4
    assert ds.metadata["n_features"] == 64
    assert ds.metadata["noise_sigma"] == pytest.approx(4.0)
```

**Step 2: run the test, confirm it fails.**

```bash
python -m pytest tests/test_tiny_patterns_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'research'` or similar.

### 1.2 Implement the module

**Files:**
- Create: `research/__init__.py` (empty)
- Create: `research/datasets/__init__.py` (empty)
- Create: `research/datasets/tiny_patterns.py`

```python
# research/datasets/tiny_patterns.py
"""TinyPatternDataset: K-class Poisson rate-vector synthetic dataset.

Each class has a fixed mean rate vector (drawn once from a class-mean RNG).
Examples are sampled as class_mean + Gaussian(0, noise_sigma), clipped to
[rate_min, rate_max]. Labels are balanced across classes.

Saved as a single .npz with X_train, y_train, X_test, y_test, class_means,
and a JSON metadata blob recording the generator seed and hyperparameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TinyPatternDataset:
    X_train: np.ndarray   # (n_train, n_features) float32, rates in Hz
    y_train: np.ndarray   # (n_train,) int32
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path):
        data = np.load(path)
        metadata = json.loads(str(data["metadata_json"]))
        return cls(
            X_train=data["X_train"].astype(np.float32),
            y_train=data["y_train"].astype(np.int32),
            X_test=data["X_test"].astype(np.float32),
            y_test=data["y_test"].astype(np.int32),
            metadata=metadata,
        )


def build_dataset(
    out_path,
    *,
    seed,
    K=4,
    n_features=64,
    n_train=200,
    n_test=50,
    noise_sigma=4.0,
    rate_min=1.0,
    rate_max=40.0,
):
    """Generate and save a TinyPatternDataset."""
    rng = np.random.default_rng(seed)

    # Draw class mean rate vectors with a margin so noise doesn't always saturate.
    margin = 5.0
    class_means = rng.uniform(rate_min + margin, rate_max - margin,
                              size=(K, n_features)).astype(np.float32)

    def _sample(n_per_split, split_seed_offset):
        split_rng = np.random.default_rng(seed + split_seed_offset)
        labels = np.tile(np.arange(K, dtype=np.int32), n_per_split // K + 1)[:n_per_split]
        split_rng.shuffle(labels)
        X = np.empty((n_per_split, n_features), dtype=np.float32)
        for i, y in enumerate(labels):
            noise = split_rng.normal(0.0, noise_sigma, size=n_features).astype(np.float32)
            X[i] = np.clip(class_means[y] + noise, rate_min, rate_max)
        return X, labels

    X_train, y_train = _sample(n_train, split_seed_offset=1)
    X_test, y_test = _sample(n_test, split_seed_offset=2)

    metadata = {
        "seed": int(seed),
        "K": int(K),
        "n_features": int(n_features),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "noise_sigma": float(noise_sigma),
        "rate_min": float(rate_min),
        "rate_max": float(rate_max),
        "class_means_shape": list(class_means.shape),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        class_means=class_means,
        metadata_json=np.array(json.dumps(metadata)),
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build"])
    p.add_argument("--out", default="research/datasets/tiny_patterns.npz")
    p.add_argument("--seed", type=lambda x: int(x, 0), default=0xD47A5E7)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--noise-sigma", type=float, default=4.0)
    args = p.parse_args()

    if args.command == "build":
        build_dataset(args.out, seed=args.seed, K=args.K, n_features=args.n_features,
                      n_train=args.n_train, n_test=args.n_test, noise_sigma=args.noise_sigma)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

**Step 3: run the tests, confirm pass.**

```bash
python -m pytest tests/test_tiny_patterns_dataset.py -v
```

Expected: all 5 tests PASS.

**Step 4: build the canonical dataset on disk.**

```bash
python -m research.datasets.tiny_patterns build --out research/datasets/tiny_patterns.npz
```

Expected: `Wrote research/datasets/tiny_patterns.npz`.

**Step 5: commit.**

```bash
git add research/__init__.py research/datasets/__init__.py \
        research/datasets/tiny_patterns.py \
        research/datasets/tiny_patterns.npz \
        tests/test_tiny_patterns_dataset.py
git commit -m "feat: add TinyPatternDataset for G1 — K-class Poisson rate-vector synthetic data"
```

---

## Task 2 — `RATE_VECTOR_POISSON` stimulus pattern

This is the encoder primitive — per-neuron Poisson rate, not the scalar `POISSON_SPIKE_TRAIN` that already exists.

### 2.1 Failing tests

**Files:**
- Create: `tests/test_rate_vector_poisson_stimulus.py`

The existing `tests/test_experiment_system.py` uses a MockCuPy with NumPy. We'll mirror that pattern but add a seedable RNG so Poisson tests are deterministic.

```python
# tests/test_rate_vector_poisson_stimulus.py
import numpy as np
import pytest


class MockRandom:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def random(self, n):
        return self.rng.random(n).astype(np.float32)
    def randn(self, n):
        return self.rng.standard_normal(n).astype(np.float32)


class MockCuPy:
    float32 = np.float32
    int32 = np.int32
    bool_ = np.bool_

    def __init__(self, seed=0):
        self.random = MockRandom(seed=seed)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)
    @staticmethod
    def array(data, dtype=None):
        return np.array(data, dtype=dtype)
    @staticmethod
    def sum(arr):
        class R:
            def __init__(self, v): self.v = v
            def get(self): return self.v
        return R(np.sum(arr))
    @staticmethod
    def where(cond, x, y):
        return np.where(cond, x, y)
    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)


from sim.enums import StimulusPatternType
from sim.config import StimulusPattern, StimulusChannel, NeuronGroup
from experiment.groups import NeuronGroupManager
from experiment.stimulus import StimulusManager


def _build_manager(mock_cp, n_neurons, target_indices, rate_vector_hz,
                   spike_current_pA=200.0, spike_duration_ms=1.0,
                   duration_ms=1000.0, dt_ms=1.0):
    gm = NeuronGroupManager(n_neurons)
    # We rely on target_neuron_indices path so the group manager is minimally used.
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=spike_current_pA,
        spike_duration_ms=spike_duration_ms,
        rate_vector_hz=list(rate_vector_hz),
    )
    ch = StimulusChannel(
        name="inp_ch",
        pattern=pat,
        target_neuron_indices=list(target_indices),
        onset_ms=0.0,
        duration_ms=duration_ms,
    )
    sm = StimulusManager(n_neurons, dt_ms)
    sm.initialize([ch], gm, mock_cp)
    return sm, ch


def test_zero_rate_produces_zero_spikes():
    cp = MockCuPy(seed=1)
    n_neurons = 10
    target = list(range(10))
    rates = [0.0] * 10
    sm, ch = _build_manager(cp, n_neurons, target, rates)
    total = np.zeros(n_neurons, dtype=np.float32)
    for step in range(2000):
        I = sm.compute_step_current(current_time_ms=step * 1.0, phase_start_ms=0.0, cp_module=cp)
        total += I
    assert total.sum() == 0.0, "Zero-rate neurons should never spike"


def test_uniform_rate_matches_expected_poisson():
    cp = MockCuPy(seed=2)
    n_neurons = 64
    target = list(range(n_neurons))
    target_rate = 20.0
    rates = [target_rate] * n_neurons
    spike_duration_ms = 1.0
    sm, ch = _build_manager(cp, n_neurons, target, rates,
                            spike_current_pA=200.0,
                            spike_duration_ms=spike_duration_ms)
    sim_duration_s = 5.0
    n_steps = int(sim_duration_s * 1000)
    spike_counts = np.zeros(n_neurons, dtype=np.int64)
    for step in range(n_steps):
        I = sm.compute_step_current(current_time_ms=step * 1.0, phase_start_ms=0.0, cp_module=cp)
        spike_counts += (I > 0).astype(np.int64)
    empirical_rates = spike_counts / sim_duration_s
    mean_emp = float(np.mean(empirical_rates))
    assert abs(mean_emp - target_rate) / target_rate < 0.25, \
        f"Empirical rate {mean_emp:.2f} Hz vs target {target_rate:.2f} Hz (>25% off)"


def test_per_neuron_rate_differentiation():
    cp = MockCuPy(seed=3)
    n = 8
    target = list(range(n))
    rates = [5.0] * 4 + [30.0] * 4
    sm, ch = _build_manager(cp, n, target, rates, spike_duration_ms=1.0)
    counts = np.zeros(n, dtype=np.int64)
    for step in range(3000):
        I = sm.compute_step_current(current_time_ms=step * 1.0, phase_start_ms=0.0, cp_module=cp)
        counts += (I > 0).astype(np.int64)
    low = counts[:4].mean()
    high = counts[4:].mean()
    assert high > 3 * low, f"High-rate neurons ({high}) should be >>3x low-rate ({low})"


def test_rate_vector_length_must_match_target_count():
    cp = MockCuPy(seed=4)
    n = 5
    target = list(range(n))
    rates = [10.0, 10.0]  # only 2 entries for 5 targets
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        rate_vector_hz=list(rates),
    )
    ch = StimulusChannel(
        name="ch", pattern=pat, target_neuron_indices=list(target),
        onset_ms=0.0, duration_ms=500.0,
    )
    gm = NeuronGroupManager(n)
    sm = StimulusManager(n, 1.0)
    with pytest.raises(ValueError, match="rate_vector_hz length"):
        sm.initialize([ch], gm, cp)


def test_disabled_channel_produces_no_current():
    cp = MockCuPy(seed=5)
    n = 8
    target = list(range(n))
    rates = [40.0] * n
    sm, ch = _build_manager(cp, n, target, rates)
    ch.enabled = False
    for step in range(500):
        I = sm.compute_step_current(current_time_ms=step * 1.0, phase_start_ms=0.0, cp_module=cp)
        assert np.all(I == 0), f"Disabled channel should produce zero current at step {step}"
```

**Step 2: run tests, confirm failures.**

```bash
python -m pytest tests/test_rate_vector_poisson_stimulus.py -v
```

Expected: all fail with `AttributeError: RATE_VECTOR_POISSON` or `StimulusPattern has no attribute 'rate_vector_hz'`.

### 2.2 Add enum + dataclass field

**Files:**
- Modify: `sim/enums.py:386-395` — add `RATE_VECTOR_POISSON` to `StimulusPatternType`.
- Modify: `sim/config.py:400-433` — add `rate_vector_hz: List[float] = field(default_factory=list)` to `StimulusPattern`.

```python
# sim/enums.py — inside StimulusPatternType enum
    RATE_VECTOR_POISSON = "RATE_VECTOR_POISSON"  # Per-neuron Poisson rate vector
```

```python
# sim/config.py — inside StimulusPattern dataclass
    # Per-neuron Poisson rate vector (for RATE_VECTOR_POISSON pattern).
    # Length must equal the number of target neurons in the channel.
    rate_vector_hz: List[float] = field(default_factory=list)
```

### 2.3 Implement the new pattern in StimulusManager

**Files:**
- Modify: `experiment/stimulus.py` — add a `_poisson_rate_vectors` dict to `__init__` and `cleanup`; handle the new pattern type in both `initialize` (length validation + dense rate array allocation) and `_compute_pattern` (Bernoulli draw per neuron per step).

Implementation sketch:

```python
# experiment/stimulus.py — in __init__
        self._poisson_rate_vectors = {}  # channel_name -> full-n rate array
```

```python
# experiment/stimulus.py — in initialize(), after the existing Poisson init:
            if ch.pattern.pattern_type == StimulusPatternType.RATE_VECTOR_POISSON.name:
                target_indices = self._resolve_targets(ch, group_manager)
                if len(ch.pattern.rate_vector_hz) != len(target_indices):
                    raise ValueError(
                        f"RATE_VECTOR_POISSON rate_vector_hz length "
                        f"({len(ch.pattern.rate_vector_hz)}) must equal number of "
                        f"target neurons ({len(target_indices)}) for channel '{ch.name}'"
                    )
                rate_full = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)
                if len(target_indices) > 0:
                    idx = cp_module.array(target_indices, dtype=cp_module.int32)
                    rate_full[idx] = cp_module.array(
                        ch.pattern.rate_vector_hz, dtype=cp_module.float32
                    )
                self._poisson_rate_vectors[ch.name] = rate_full
                self._poisson_active[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
                self._poisson_timers[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)
```

```python
# experiment/stimulus.py — in _compute_pattern, add a new elif after POISSON_SPIKE_TRAIN:
        elif p.pattern_type == StimulusPatternType.RATE_VECTOR_POISSON.name:
            rate_vec = self._poisson_rate_vectors.get(channel.name)
            if rate_vec is None:
                return cp_module.float32(0.0)
            p_spike = rate_vec * (self.dt_ms / 1000.0)
            timers = self._poisson_timers.get(channel.name)
            if timers is None:
                return cp_module.float32(0.0)
            timers = timers - self.dt_ms
            timers_clipped = cp_module.maximum(timers, cp_module.float32(0.0))
            draws = self._rng.random(self.n_neurons)
            new_spikes = (draws < p_spike) & mask & (timers_clipped <= 0)
            timers_next = cp_module.where(
                new_spikes, cp_module.float32(p.spike_duration_ms), timers_clipped
            )
            self._poisson_timers[channel.name] = timers_next
            is_active = timers_next > 0
            return cp_module.where(is_active, cp_module.float32(p.spike_current_pA), cp_module.float32(0.0))
```

```python
# experiment/stimulus.py — extend cleanup()
        self._poisson_rate_vectors.clear()
```

**Step 4: re-run the test.**

```bash
python -m pytest tests/test_rate_vector_poisson_stimulus.py -v
```

Expected: all 5 pass.

**Step 5: full regression sweep.**

```bash
python -m pytest tests/ --tb=short
```

Expected: 106 previously-passing tests still pass + 5 new.

**Step 6: commit.**

```bash
git add sim/enums.py sim/config.py experiment/stimulus.py \
        tests/test_rate_vector_poisson_stimulus.py
git commit -m "feat(stimulus): add RATE_VECTOR_POISSON pattern — per-neuron Poisson rate encoding for G1"
```

---

## Task 3 — G1 network wiring helper

Builds the specific 68-neuron topology: 64 input + 4 output + all-to-all plastic input→output + all-to-all (minus self) fixed lateral inhibition at the output layer.

### 3.1 Failing tests

**Files:**
- Create: `tests/test_g1_network.py`

```python
# tests/test_g1_network.py
import numpy as np
import pytest

from research.runners.g1_network import build_g1_network_config, G1NetworkSpec


def test_spec_defaults():
    spec = G1NetworkSpec()
    assert spec.n_input == 64
    assert spec.n_output == 4
    assert spec.n_total == 68
    assert spec.input_indices == list(range(0, 64))
    assert spec.output_indices == list(range(64, 68))


def test_build_g1_config_produces_wiring_plan():
    core_cfg, wiring_plan = build_g1_network_config(seed=42)
    assert core_cfg.num_neurons == 68
    assert core_cfg.neuron_model_type == "IZHIKEVICH"
    assert core_cfg.seed == 42
    assert core_cfg.dt_ms == 1.0
    assert core_cfg.enable_stdp is True
    assert core_cfg.enable_watts_strogatz is False
    assert "input_to_output" in wiring_plan
    assert wiring_plan["input_to_output"]["count"] == 64 * 4
    assert wiring_plan["input_to_output"]["plastic"] is True
    assert "output_lateral_inhibition" in wiring_plan
    assert wiring_plan["output_lateral_inhibition"]["count"] == 4 * 3
    assert wiring_plan["output_lateral_inhibition"]["plastic"] is False


def test_initial_weights_in_range():
    core_cfg, wiring_plan = build_g1_network_config(seed=7)
    w = np.asarray(wiring_plan["input_to_output"]["initial_weights"])
    assert w.shape == (64 * 4,)
    assert w.min() >= 0.05 - 1e-6
    assert w.max() <= 0.15 + 1e-6


def test_lateral_inhibition_weights_sign_and_magnitude():
    core_cfg, wiring_plan = build_g1_network_config(seed=0)
    w = np.asarray(wiring_plan["output_lateral_inhibition"]["initial_weights"])
    assert w.shape == (4 * 3,)
    assert np.all(w > 0), "Weights are positive magnitudes; sign handled by sim inhibitory trait"
    assert np.allclose(w, 1.0, atol=1e-6)


def test_edges_are_correct_pairs():
    core_cfg, wiring_plan = build_g1_network_config(seed=0)
    i2o = wiring_plan["input_to_output"]
    pre = list(i2o["pre_indices"])
    post = list(i2o["post_indices"])
    assert len(pre) == len(post) == 256
    assert min(pre) == 0 and max(pre) == 63
    assert min(post) == 64 and max(post) == 67
    pairs = set(zip(pre, post))
    assert len(pairs) == 256
    lat = wiring_plan["output_lateral_inhibition"]
    lpre, lpost = list(lat["pre_indices"]), list(lat["post_indices"])
    assert len(lpre) == 12
    for a, b in zip(lpre, lpost):
        assert 64 <= a <= 67 and 64 <= b <= 67
        assert a != b
```

**Step 2: run, confirm failure.**

### 3.2 Implement

**Files:**
- Create: `research/runners/__init__.py` (empty)
- Create: `research/runners/g1_network.py`

```python
# research/runners/g1_network.py
"""Wiring helper that builds the G1 network topology.

Returns (CoreSimConfig, wiring_plan). The runner injects the explicit
wiring into SimulationBridge after _initialize_simulation_data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np

from sim.config import CoreSimConfig
from sim.enums import NeuronModel


@dataclass
class G1NetworkSpec:
    n_input: int = 64
    n_output: int = 4

    init_weight_min: float = 0.05
    init_weight_max: float = 0.15
    weight_max_cap: float = 1.5
    lateral_inhibition_weight: float = 1.0

    @property
    def n_total(self):
        return self.n_input + self.n_output

    @property
    def input_indices(self):
        return list(range(0, self.n_input))

    @property
    def output_indices(self):
        return list(range(self.n_input, self.n_input + self.n_output))


def build_g1_network_config(seed, spec=None):
    spec = spec or G1NetworkSpec()

    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = spec.n_total
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core_cfg.seed = int(seed)
    core_cfg.dt_ms = 1.0
    core_cfg.connections_per_neuron = 0
    core_cfg.num_traits = 2
    core_cfg.inhibitory_trait_indices = [1]

    core_cfg.enable_stdp = True
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_homeostasis = True
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_watts_strogatz = False

    core_cfg.stdp_a_plus = 0.012
    core_cfg.stdp_a_minus = 0.010
    core_cfg.stdp_weight_max = spec.weight_max_cap
    core_cfg.stdp_weight_min = 0.0

    core_cfg.ou_std_current_pA = 30.0

    rng = np.random.default_rng(seed)

    # All-to-all input -> output
    pre_i2o, post_i2o = [], []
    for i in spec.input_indices:
        for o in spec.output_indices:
            pre_i2o.append(i)
            post_i2o.append(o)
    w_i2o = rng.uniform(spec.init_weight_min, spec.init_weight_max,
                        size=len(pre_i2o)).astype(np.float32)

    # Output -> Output, minus self-loops
    pre_lat, post_lat = [], []
    for a in spec.output_indices:
        for b in spec.output_indices:
            if a == b:
                continue
            pre_lat.append(a)
            post_lat.append(b)
    w_lat = np.full(len(pre_lat), spec.lateral_inhibition_weight, dtype=np.float32)

    wiring_plan = {
        "input_to_output": {
            "pre_indices": pre_i2o,
            "post_indices": post_i2o,
            "initial_weights": w_i2o,
            "plastic": True,
            "conn_type": "E_TO_E",
            "count": len(pre_i2o),
        },
        "output_lateral_inhibition": {
            "pre_indices": pre_lat,
            "post_indices": post_lat,
            "initial_weights": w_lat,
            "plastic": False,
            "conn_type": "I_TO_E",
            "count": len(pre_lat),
        },
        "spec": spec,
    }
    return core_cfg, wiring_plan
```

**Step 3: run tests, confirm pass.**

**Step 4: commit.**

```bash
git add research/runners/__init__.py research/runners/g1_network.py \
        tests/test_g1_network.py
git commit -m "feat(g1): add network wiring helper — 64-input + 4-output + plastic i->o + lateral inhibition"
```

---

## Task 4 — Decoder utility

Pure function: given output spike counts, return predicted class + margin.

### 4.1 Failing tests

**Files:**
- Create: `tests/test_g1_decoder.py`

```python
# tests/test_g1_decoder.py
import numpy as np
import pytest

from research.runners.g1_decoder import decode_prediction, compute_margin, compute_metrics


def test_decode_prediction_argmax():
    counts = np.array([3, 10, 5, 2], dtype=np.int32)
    assert decode_prediction(counts) == 1


def test_decode_prediction_ties_deterministic():
    counts = np.array([5, 5, 5, 5], dtype=np.int32)
    assert decode_prediction(counts) == 0


def test_margin_positive_when_correct():
    counts = np.array([1, 10, 2, 3], dtype=np.int32)
    m = compute_margin(counts, correct_class=1)
    # 10 - mean([1,2,3]) = 10 - 2 = 8
    assert m == pytest.approx(8.0)


def test_margin_negative_when_wrong():
    counts = np.array([10, 1, 2, 3], dtype=np.int32)
    m = compute_margin(counts, correct_class=1)
    # 1 - mean([10,2,3]) = 1 - 5 = -4
    assert m == pytest.approx(-4.0)


def test_metrics_batch_accuracy_and_margin():
    counts = np.array([
        [3, 10, 5, 2],
        [8, 2, 1, 0],
        [1, 1, 9, 2],
        [0, 0, 0, 5],
        [2, 5, 3, 4],
    ], dtype=np.int32)
    y = np.array([1, 0, 3, 3, 2], dtype=np.int32)
    m = compute_metrics(counts, y)
    assert m["accuracy"] == pytest.approx(3.0 / 5.0)
    assert m["n"] == 5
    assert "mean_margin" in m
    assert int(m["confusion"].sum()) == 5


def test_compute_margin_silent_network():
    counts = np.zeros(4, dtype=np.int32)
    assert decode_prediction(counts) == 0
    assert compute_margin(counts, correct_class=2) == 0.0
```

### 4.2 Implement

**Files:**
- Create: `research/runners/g1_decoder.py`

```python
# research/runners/g1_decoder.py
"""Decoder + loss helpers for G1: spike counts -> class prediction, margin, metrics."""

from __future__ import annotations

import numpy as np


def decode_prediction(spike_counts):
    if spike_counts.size == 0:
        raise ValueError("spike_counts must be non-empty")
    return int(np.argmax(spike_counts))


def compute_margin(spike_counts, correct_class):
    if spike_counts.size < 2:
        return 0.0
    correct = float(spike_counts[correct_class])
    mask = np.ones_like(spike_counts, dtype=bool)
    mask[correct_class] = False
    others = spike_counts[mask]
    if others.size == 0:
        return 0.0
    return correct - float(others.mean())


def compute_metrics(spike_counts, labels):
    assert spike_counts.ndim == 2
    n, K = spike_counts.shape
    preds = np.argmax(spike_counts, axis=1)
    acc = float((preds == labels).mean()) if n > 0 else 0.0
    margins = np.array([compute_margin(spike_counts[i], int(labels[i])) for i in range(n)])
    confusion = np.zeros((K, K), dtype=np.int32)
    for y_true, y_pred in zip(labels, preds):
        confusion[int(y_true), int(y_pred)] += 1
    return {
        "accuracy": acc,
        "mean_margin": float(margins.mean()) if n > 0 else 0.0,
        "margins": margins,
        "predictions": preds,
        "confusion": confusion,
        "n": n,
    }
```

**Step 3: run tests, confirm pass.**

**Step 4: commit.**

```bash
git add research/runners/g1_decoder.py tests/test_g1_decoder.py
git commit -m "feat(g1): add decoder + metric helpers — argmax prediction + margin + confusion"
```

---

## Task 5 — Headless G1 runner

The orchestrator. Builds the network, injects wiring, loads dataset, runs epochs, writes results.

**Required SimulationBridge change**: add `inject_explicit_wiring(wiring_plan)` method. It's only called by the G1 runner — zero impact on existing experiments.

### 5.1 Failing smoke test

**Files:**
- Create: `tests/test_g1_runner_smoke.py`

```python
# tests/test_g1_runner_smoke.py
"""Smoke test for the G1 runner: 1 epoch, 10 train + 5 test examples.
Verifies end-to-end pipeline runs and produces the expected JSON schema.
Does NOT assert convergence — that's for the findings doc.
"""
import json
from pathlib import Path

import pytest


def test_g1_runner_smoke(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g1_runner import run_g1
    from research.datasets.tiny_patterns import build_dataset

    ds_path = tmp_path / "ds.npz"
    build_dataset(ds_path, seed=0xD47A5E7, K=4, n_features=64,
                  n_train=20, n_test=8, noise_sigma=4.0)

    out = tmp_path / "result.json"
    result = run_g1(
        dataset_path=str(ds_path),
        out_path=str(out),
        seed=42,
        n_epochs=1,
        max_train_per_epoch=10,
        max_test_per_epoch=8,
        verbose=False,
    )

    assert out.exists()
    with open(out) as f:
        data = json.load(f)

    assert data["seed"] == 42
    assert data["n_epochs"] == 1
    assert "epochs" in data and len(data["epochs"]) == 1
    epoch = data["epochs"][0]
    for key in ("epoch", "train_accuracy", "test_accuracy", "mean_margin_test",
                "mean_weight", "weight_std", "time_seconds"):
        assert key in epoch, f"Missing key: {key}"
    assert 0.0 <= epoch["test_accuracy"] <= 1.0
    assert 0.0 <= epoch["train_accuracy"] <= 1.0
```

### 5.2 Implement SimulationBridge wiring injection

**Files:**
- Modify: `sim/bridge.py` — add `inject_explicit_wiring` method on `SimulationBridge`.

The method must:
1. Clear existing synapses.
2. Rebuild `cp_connections` (CSR sparse) from concatenated pre/post/weight arrays.
3. Reset the synapse-side arrays (`cp_synapse_weights`, `cp_synapse_pre_indices`, `cp_synapse_post_indices`, plastic mask or connection-type array) to match.

**Grep first** to find the exact field names:
```bash
grep -nE "cp_synapse_|cp_connections|cp_synapse_weights" sim/bridge.py | head -40
```

Then a minimal version along these lines:

```python
def inject_explicit_wiring(self, wiring_plan):
    """Replace auto-generated connectivity with an explicit wiring plan.

    Called AFTER _initialize_simulation_data. Used by research runners that
    need precise topology (G1 classifier and later gates).
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import numpy as np

    n = self.core_config.num_neurons
    all_pre, all_post, all_w = [], [], []
    for name, group in wiring_plan.items():
        if not isinstance(group, dict) or "pre_indices" not in group:
            continue
        all_pre.extend(group["pre_indices"])
        all_post.extend(group["post_indices"])
        all_w.extend([float(x) for x in group["initial_weights"]])

    pre = cp.asarray(np.array(all_pre, dtype=np.int32))
    post = cp.asarray(np.array(all_post, dtype=np.int32))
    w = cp.asarray(np.array(all_w, dtype=np.float32))

    coo = csp.coo_matrix((w, (pre, post)), shape=(n, n))
    self.cp_connections = coo.tocsr()
    self._invalidate_coo_cache()

    # Reset synapse-indexed arrays. Reuse _init_synapse_arrays_with_capacity +
    # _add_synapses_to_arrays if they exist, otherwise set fields directly.
    new_count = len(all_pre)
    if hasattr(self, "_init_synapse_arrays_with_capacity"):
        self._init_synapse_arrays_with_capacity(new_count, self.core_config)
    # Fill the synapse arrays. EXACT attribute names depend on bridge.py —
    # patch once grep confirms.
    self.cp_synapse_pre_indices[:new_count] = pre
    self.cp_synapse_post_indices[:new_count] = post
    self.cp_synapse_weights[:new_count] = w
    if hasattr(self, "cp_synapse_count"):
        self.cp_synapse_count = new_count
    elif hasattr(self.runtime_state, "current_synapse_count"):
        self.runtime_state.current_synapse_count = new_count
```

Expect the first run of the smoke test to fail on some exact attribute name. **Fix iteratively** — each failure points to the next missing detail. This is the main bridge-integration risk; budget a debug hour.

### 5.3 Implement `g1_runner.py`

**Files:**
- Create: `research/runners/g1_runner.py`

```python
# research/runners/g1_runner.py
"""Headless G1 runner: train + test loop that writes a results JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import (ExperimentConfig, ExperimentPhase, StimulusChannel,
                        StimulusPattern, NeuronGroup, ReadoutConfig)
from sim.enums import (StimulusPatternType, ExperimentPhaseType, NeuronGroupRole)
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
    bridge.inject_explicit_wiring(wiring_plan)

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
    engine.initialize(cp_traits=None, cp_module=cp)

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

    for epoch in range(n_epochs):
        t_epoch = time.time()
        order = rng.permutation(len(ds.X_train))[:train_N]

        train_spike_counts = np.zeros((train_N, K), dtype=np.int32)
        train_labels = ds.y_train[order]
        for i, idx in enumerate(order):
            counts = _present_example(
                bridge, engine, ds.X_train[idx], teacher_class=int(ds.y_train[idx]),
                spec=spec, cp=cp,
            )
            train_spike_counts[i] = counts
        train_metrics = compute_metrics(train_spike_counts, train_labels)

        test_spike_counts = np.zeros((test_N, K), dtype=np.int32)
        test_labels = ds.y_test[:test_N]
        for i in range(test_N):
            counts = _present_example(
                bridge, engine, ds.X_test[i], teacher_class=None,
                spec=spec, cp=cp,
            )
            test_spike_counts[i] = counts
        test_metrics = compute_metrics(test_spike_counts, test_labels)

        i2o_count = wiring_plan["input_to_output"]["count"]
        w = bridge.cp_synapse_weights[:i2o_count].get()
        epoch_record = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "mean_margin_train": train_metrics["mean_margin"],
            "mean_margin_test": test_metrics["mean_margin"],
            "mean_weight": float(w.mean()),
            "weight_std": float(w.std()),
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "train_confusion": train_metrics["confusion"].tolist(),
            "test_confusion": test_metrics["confusion"].tolist(),
            "time_seconds": time.time() - t_epoch,
        }
        all_results["epochs"].append(epoch_record)
        if verbose:
            print(f"[seed={seed}] Epoch {epoch}: "
                  f"train_acc={epoch_record['train_accuracy']:.3f}  "
                  f"test_acc={epoch_record['test_accuracy']:.3f}  "
                  f"margin_test={epoch_record['mean_margin_test']:+.2f}  "
                  f"W∈[{epoch_record['weight_min']:.3f}, {epoch_record['weight_max']:.3f}]  "
                  f"{epoch_record['time_seconds']:.1f}s")

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
    """Step the sim through one example. Returns (n_output,) spike count array."""
    dt = bridge.core_config.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    n_gap_steps = int(GAP_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=250.0,
        spike_duration_ms=1.0,
        rate_vector_hz=[float(r) for r in rate_vector],
    )
    ch = StimulusChannel(
        name="input_pattern", pattern=pat,
        target_neuron_indices=spec.input_indices,
        onset_ms=0.0, duration_ms=STIMULUS_MS,
        enabled=True,
    )
    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([ch], engine.group_manager, cp)

    counts = np.zeros(spec.n_output, dtype=np.int32)
    out_idx_cp = cp.asarray(spec.output_indices, dtype=cp.int32)

    teacher_amp = float(TEACHER_CURRENT_PA if teacher_class is not None else 0.0)
    teacher_target = spec.output_indices[teacher_class] if teacher_class is not None else None

    for step in range(n_stim_steps):
        # Teacher current injected via the bridge's external-current hook.
        # Exact attribute name TBD; grep for `cp_external_current` or similar in bridge.py.
        if teacher_target is not None and hasattr(bridge, "cp_external_current_pA"):
            bridge.cp_external_current_pA[teacher_target] = cp.float32(teacher_amp)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
        if teacher_target is not None and hasattr(bridge, "cp_external_current_pA"):
            bridge.cp_external_current_pA[teacher_target] = cp.float32(0.0)

        if readout_start_step <= step < readout_end_step:
            fired = bridge.cp_firing_states[out_idx_cp].get()
            counts += fired.astype(np.int32)

    engine.stimulus_manager.cleanup()
    engine.stimulus_manager.initialize([], engine.group_manager, cp)
    for step in range(n_gap_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt

    return counts
```

### 5.4 Debug & iterate

Run the smoke test:

```bash
python -m pytest tests/test_g1_runner_smoke.py -v -s
```

When it fails (likely on the first try), apply **systematic debugging**:

1. **Read the exact error.** Python will say exactly which attribute is missing.
2. **Reproduce with a minimal standalone script** — copy the runner setup into a scratch `.py` file, run directly, print intermediate state.
3. **Grep for the real attribute name** in `sim/bridge.py`. Look for `cp_synapse`, `cp_external`, `cp_stimulus`, `cp_input_current`.
4. **Patch the runner or `inject_explicit_wiring`.** One hypothesis, one fix.
5. **Re-run. Repeat.**

Three consecutive fixes without visible progress → STOP and re-read `_run_one_simulation_step` top-to-bottom. The mental model is likely wrong.

### 5.5 Full regression sweep + commit

```bash
python -m pytest tests/ --tb=short
```

Expected: all green.

```bash
git add research/runners/g1_runner.py sim/bridge.py tests/test_g1_runner_smoke.py
git commit -m "feat(g1): add headless runner — train+test loop over TinyPatterns with teacher-forced STDP"
```

---

## Task 6 — First full training run + findings doc

### 6.1 Run the sweep

```bash
python -c "
from research.runners.g1_runner import run_g1
from datetime import datetime
import os
ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
os.makedirs('research/findings/raw', exist_ok=True)
for seed in [42, 43, 44]:
    run_g1(
        dataset_path='research/datasets/tiny_patterns.npz',
        out_path=f'research/findings/raw/g1-seed{seed}-{ts}.json',
        seed=seed,
        n_epochs=10,
        verbose=True,
    )
"
```

Budget: ~5 min per seed × 3 seeds. If significantly slower, investigate (step time, example count).

### 6.2 Aggregate + write findings doc

**Files:**
- Create: `research/findings/2026-04-20-g1.md`
- Create: `research/findings/raw/g1-seed{42,43,44}-*.json` (generated by the sweep)

Findings doc template:

```markdown
# G1 Findings — Dataset → Encoder → Sim → Decoder → Loss

**Date:** 2026-04-20
**Gate:** G1 — minimum viable trainable pipeline
**Verdict:** GO | NO-GO | PARTIAL

## Protocol
- Dataset: `research/datasets/tiny_patterns.npz` (K=4, 64-dim Poisson rates, 200 train / 50 test, noise σ=4 Hz)
- Network: 64 input + 4 output Izhikevich RS, all-to-all STDP-plastic input→output, fixed lateral inhibition at output (weight 1.0)
- Training: supervised teacher current (400 pA) on correct class, 200 ms stimulus + 100 ms gap per example
- Decoder: argmax over output spike counts in [100, 200] ms window
- Seeds: 42, 43, 44 (training); dataset seed 0xD47A5E7 (fixed across seeds)
- Epochs: 10

## Raw numbers
| Seed | Epoch 0 test acc | Epoch 9 test acc | Mean margin test @ epoch 9 | Final mean W | W max |
|------|-----------------|------------------|---------------------------|--------------|-------|
| 42   | ...             | ...              | ...                       | ...          | ...   |
| 43   | ...             | ...              | ...                       | ...          | ...   |
| 44   | ...             | ...              | ...                       | ...          | ...   |
| Mean | ...             | ...              | ...                       | ...          | ...   |

## Verdict
- Gate threshold: mean test accuracy > 55%, each seed ≥ 45%, monotonic-ish learning curve.
- Observed: ...
- Verdict: GO / NO-GO / PARTIAL

## What was surprising
- ...

## Next step
- If GO: proceed to G2 design.
- If NO-GO: specific failure mode + proposed pivot.
- If PARTIAL: targeted tuning round (teacher magnitude, STDP rates, readout window), or scope-down to K=3.

## Raw results
See `research/findings/raw/g1-seed*.json`.
```

### 6.3 CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md` — add under `[Unreleased]` → `Added`:

```
- **G1 pipeline** — first minimum-viable dataset → encoder → sim → decoder → loss round-trip
  - `TinyPatternDataset`: K-class Poisson rate-vector synthetic dataset (`research/datasets/tiny_patterns.py`)
  - `RATE_VECTOR_POISSON` stimulus pattern for per-neuron Poisson rate encoding
  - `g1_network.py` + `g1_runner.py`: headless runner with teacher-forced supervised STDP
  - `g1_decoder.py`: argmax + margin + confusion matrix
  - First training findings in `research/findings/2026-04-20-g1.md`
```

### 6.4 Commit the findings + CHANGELOG

```bash
git add research/findings/ CHANGELOG.md
git commit -m "feat(g1): first training run — 3 seeds × 10 epochs on TinyPatterns (results: <GO|NO-GO>)"
```

### 6.5 Push to origin (only if gate fires GO)

```bash
git push origin main
```

---

## Global regression guard

Before every commit in this plan, run:

```bash
python -m pytest tests/ --tb=short
```

Must remain green. The design doc requires no regression on existing tests.

After Task 5, also run the biological benchmark suite to catch STDP/connectivity regressions:

```bash
python run_benchmarks.py --benchmark stdp-timing
python run_benchmarks.py --benchmark gamma-oscillations
```

---

## YAGNI call-outs

- No per-synapse plastic mask unless the trait-based route fails.
- No epoch-level model checkpointing in G1 (that's G3).
- No curve plotting in the runner — emit JSON, plot externally.
- No multi-config grid sweep — just the 3-seed run.
- No UI integration.
- No experiment-preset wrapper. G1 is headless-only.
