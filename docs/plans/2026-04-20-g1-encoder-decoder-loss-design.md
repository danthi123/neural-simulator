---
type: plan
status: live
date: 2026-04-20
---

# Design: G1 — Dataset → Encoder → Sim → Decoder → Loss Round-Trip

**Date:** 2026-04-20
**Status:** Approved (operator pre-authorized autonomous decisions)
**Scope:** Gate G1 of the artificial-life mission — build the minimum viable pipeline from a dataset on disk to a brain that has a measurable loss against ground truth, using only plasticity rules the simulator already ships with.

---

## 1. Context and gap

The existing simulator validates its mechanisms (STDP timing curve, E/I balance, STP fidelity, gamma oscillations, homeostasis — all PASS per `docs/SCIENCE_ROADMAP.md`) and runs hand-crafted experiment presets (stimulus-response, Pavlovian, R-STDP gridworld, frequency sweep — all PASS).

None of those demonstrate **dataset-driven learning**. There is no encoder from arbitrary data to stimulus, no decoder from readouts to a structured answer, and no loss function comparing answer to ground truth. Everything is synthetic current patterns and hand-tuned success criteria.

G1 closes that gap with the smallest possible end-to-end pipeline. Later gates extend it (G2 = training bends the loss curve; G3 = persistence across sessions; G4 = generalization; G5 = sensorimotor loop).

---

## 2. Alternatives considered

### A. MNIST classification (Diehl & Cook style)
784-input Poisson encoding, 10 output classes, STDP + homeostasis. Known-trainable (~95% accuracy with pure unsupervised STDP), high prestige.

**Rejected for G1.** Diehl & Cook ran ≥60K images per epoch, each image 350 ms, for multiple epochs — hours per training run. Too slow for the iterate-to-G2 cadence. Revisit in G4 or later when the pipeline works on a toy and we want a real benchmark.

### B. Tiny Poisson-pattern classification (CHOSEN)
K=4 synthetic classes, 64-dim rate-vector inputs, 4 direct output neurons, STDP + teacher-forced supervision + fixed lateral inhibition at the output layer. ~1 s per example, 200 train + 50 test per epoch. Known-trainable (Masquelier & Thorpe 2007, Mozafari et al. 2019 simplified single-layer).

**Selected** because it:
- Exercises the full encoder → sim → decoder → loss pipeline.
- Uses only primitives the simulator already has, plus one minimal stimulus extension (per-neuron Poisson rate vector).
- Completes a training run in minutes, not hours — seeds×epochs fit in a single session.
- Scales the difficulty by K, rate separability, and noise — clear knobs for G2 when we want to make it harder.
- Has clean chance baseline (25% for K=4), unambiguous success metric.

### C. XOR / parity
2 binary inputs, 1 output, trained via R-STDP.

**Rejected.** Too narrow. Binary inputs trivialize the encoder; 1-bit output trivializes the decoder. Doesn't exercise the "dataset on disk" requirement meaningfully. Also a poor staircase to G2 — there's no natural harder version.

---

## 3. Architecture

### 3.1 Dataset (`research/datasets/tiny_patterns.py`)

```python
class TinyPatternDataset:
    """K synthetic Poisson-rate patterns with Gaussian noise around class means.

    Stored on disk as a single .npz with:
        X_train:  (n_train,  n_features)  float32  rates in Hz, clipped to [1, 40]
        y_train:  (n_train,)               int32    class label in [0, K)
        X_test:   (n_test,   n_features)  float32
        y_test:   (n_test,)               int32
        metadata: JSON dict (seed, K, n_features, rate_range, noise_sigma, n_train, n_test)
    """
```

Defaults:
- `K = 4` classes
- `n_features = 64` input neurons
- `rate_range = (1.0, 40.0)` Hz
- `class_mean_rates` drawn once from Uniform(5, 35 Hz) per class with fixed generator seed → stable across runs
- `noise_sigma = 4.0` Hz additive Gaussian, then clipped to rate_range
- `n_train = 200`, `n_test = 50`
- Generator seed separate from training seed — the dataset is fixed across the 3 training seeds to isolate training variance from dataset variance.

CLI: `python -m research.datasets.tiny_patterns build --out research/datasets/tiny_patterns.npz`.

### 3.2 Encoder — new stimulus pattern type

Add `StimulusPatternType.RATE_VECTOR_POISSON`: per-neuron Poisson rate, one rate value per target neuron (not one scalar applied to all). Existing `POISSON_SPIKE_TRAIN` uses a single rate for the whole channel mask; that's insufficient for pattern encoding.

New pattern fields:
- `rate_vector_hz`: `np.ndarray [n_target]` — rates, one per target neuron in the channel's target order
- `spike_current_pA`: same as existing Poisson (amplitude of current delivered during a spike event)
- `spike_duration_ms`: same as existing Poisson

Implementation: `StimulusManager._compute_pattern()` gets a new branch that broadcasts `rate_vector_hz` across the target mask and evaluates Bernoulli `p = rate * dt / 1000` per neuron per step. The stimulus current at each step is `spike_current_pA` on neurons whose timer is active, 0 elsewhere — same pattern as existing Poisson, just with per-neuron rates.

### 3.3 Network topology

- 68 total neurons:
  - **Input group** (indices 0..63): 64 Izhikevich RS neurons. Receive only the encoder's stimulus current. No recurrent input. Spike output drives downstream synapses.
  - **Output group** (indices 64..67): 4 Izhikevich RS neurons. Receive input→output synapses (all-to-all, plastic) and output→output lateral inhibitory synapses (all-to-all except self-loop, fixed).

- **Synapses:**
  - Input→output: 64×4 = 256 plastic excitatory. Initial weights uniform in [0.05, 0.15]. W ∈ [0, 1.5]. STDP on.
  - Output→output: 4×3 = 12 fixed inhibitory. Weight = −1.0 (modulated by sim's inhibitory propagation scale). STDP off on this sub-population.

- **Neuron model:** Izhikevich RS for speed. No HH (too slow for G1 iteration). No AdEx initially; can swap later.

- **dt:** 1.0 ms (Izhikevich default).

### 3.4 Per-example protocol

```
t = 0         Start example
  stimulus: RATE_VECTOR_POISSON active with this example's rate vector
  teacher (training only): constant 400 pA injected into correct-class output neuron
t = 200 ms    Stimulus off, teacher off
t = 200..300  Gap / readout tail — 100 ms with no input
t = 300       Next example
```

Readout window = 100..200 ms into the stimulus period (skip the first 100 ms to let the network equilibrate, record the settled response). Count spikes in each output neuron during that window.

### 3.5 Decoder + loss

- **Prediction:** `argmax` of per-output-neuron spike counts in the readout window. Ties broken by lowest index (deterministic).
- **0/1 loss:** `1 - accuracy` averaged over a dataset.
- **Margin:** `mean(correct_neuron_spikes − max(other_neuron_spikes))` — a graded signal that can improve even when argmax still flips. Used to see learning curves earlier than accuracy alone.
- **Soft loss (tracked, not trained on):** cross-entropy on a softmax over spike counts. Just for reporting — no gradient path.

### 3.6 Training signal

Supervised teacher current — **not** the existing SUPERVISED_TARGET mode (which averages a scalar error across all groups and goes through `current_reward_signal`, wrong granularity). Instead:

During each training example, a 400 pA DC current is added to the correct class's output neuron for the 200 ms stimulus window. This is large enough to drive that neuron above threshold ≥1 spike at its own rate.

STDP then does the work:
- Input neurons that are firing heavily (class-characteristic pattern) pair with the teacher-driven output firing → LTP on input→correct-output synapses.
- Same input neurons don't pair with non-correct output neurons (silenced by lateral inhibition + no teacher drive) → modest LTD or no change.

This is the classic Diehl-style supervised-STDP teacher trick, adapted down to a 4-class toy. No new plasticity rule required.

### 3.7 Test examples

At evaluation time, teacher current is disabled. Stimulus plays as normal. Prediction is pure argmax of output spike counts.

Evaluation happens every epoch (after all 200 training examples): run all 50 test examples with teacher off, record accuracy, margin, and per-class confusion counts.

### 3.8 Execution wrapper

`research/runners/g1_runner.py`:
```
1. Load dataset from disk.
2. Build SimulationBridge with the G1 network (see §3.3).
3. For each epoch in {0..9}:
     For each training example (shuffled):
         Set encoder rate_vector to the example.
         Enable teacher current on correct class.
         Step sim for 300 ms (200 stim + 100 gap).
         Record spike counts in the 100..200 ms window.
     Disable teacher.
     For each test example:
         Set encoder rate_vector.
         Step sim for 300 ms.
         Record prediction.
     Log: train accuracy, test accuracy, margin, mean weight, weight variance.
4. Write results JSON to research/findings/g1-<seed>-<timestamp>.json.
5. Write a summary markdown with the learning curves.
```

Headless. No GUI. No OpenGL. Pure `SimulationBridge` usage.

---

## 4. Success criteria

### G1 PASS (GO to G2)
- Test accuracy > 55% (chance = 25%) at final epoch, averaged across seeds {42, 43, 44}.
- Each individual seed ≥ 45% test accuracy — no catastrophic seed.
- Train accuracy > test accuracy (sanity — model is actually learning the training set, not just generalizing by accident).
- Learning curve monotone-ish (test accuracy at epoch 9 ≥ test accuracy at epoch 0, ideally with intermediate epochs showing a visible bend).

### G1 NO-GO (stop, diagnose, pivot before touching G2)
- Test accuracy ≤ chance at all epochs across any seed.
- No weight statistics change (STDP not firing — infrastructure bug, not a design failure).
- Teacher current drives wrong class to spike (wiring bug).

Findings doc records actual numbers vs these thresholds and a verdict.

### G1 PARTIAL (document, decide)
- Accuracy in [35%, 55%] — above chance but below target. Likely parameter tuning helps. Document and attempt one round of targeted tuning (teacher magnitude, STDP A+ / A−, readout window) before declaring NO-GO.

---

## 5. Seeds, determinism, reproducibility

- Dataset seed: `0xD47A5E7` (fixed, separate from training seed, never changed once locked)
- Training seeds: `42, 43, 44`
- Each seed drives:
  - Initial synaptic weight draw
  - Example shuffle order per epoch
  - Izhikevich parameter heterogeneity
  - OU noise RNG
- All three existing RNG sources (CuPy, NumPy, Python) seeded via `SimulationBridge`'s seeding path.
- The runner writes the `actual_seed_used` to the output JSON for audit.

---

## 6. Testing strategy

New tests land alongside implementation. TDD throughout.

1. `tests/test_tiny_patterns_dataset.py`
   - Build with fixed seed → stable X_train, y_train hashes.
   - Class means well-separated (pairwise min distance in rate space > a threshold).
   - Rate values clipped to [1, 40] Hz.
   - Save/load round-trip preserves arrays exactly.

2. `tests/test_rate_vector_poisson_stimulus.py`
   - Run 1000 ms with a known rate_vector. Measure empirical per-neuron rate. Match expected Poisson rate within 15% (finite-sample tolerance).
   - Zero-rate entries produce zero spikes.
   - Disabling channel produces zero current.

3. `tests/test_g1_decoder.py`
   - Given synthetic spike counts, decoder returns correct argmax.
   - Tie-breaking is deterministic.
   - Margin is computed correctly.

4. `tests/test_g1_network_wiring.py`
   - Build the 68-neuron G1 network. Assert 256 input→output synapses, 12 lateral inhibitory, no others.
   - Assert all synapse weights in the expected init range.
   - STDP enabled on input→output sub-population only.

5. `tests/test_g1_runner_smoke.py`
   - Run 1 epoch with 10 training + 5 test examples.
   - Assert results JSON has expected schema.
   - Assert no crash with teacher on then off.
   - (Does not assert convergence — that's for the findings doc.)

6. Existing benchmarks and tests remain green. `pytest tests/` green is a commit gate.

---

## 7. Out of scope for G1

- Hidden layers, multi-layer STDP propagation — G2 if the single-layer pipeline works and we need headroom.
- Structural plasticity during training — G2 or later.
- R-STDP on this task — the teacher-current trick is simpler and we'll see if it's enough. R-STDP is a G2 lever.
- Lateral inhibition between input neurons, recurrent hidden dynamics — out of scope.
- Real-world data (MNIST, audio, text) — G4 after the pipeline proves out on the toy.
- Sensorimotor loop (brain output modulating next input) — G5.
- Multi-GPU, distributed training — never.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Single-layer STDP isn't enough for 4-class separability | Noise sigma and class separation are knobs; tune before declaring NO-GO. Dataset-builder exposes them. |
| Teacher current causes runaway firing on correct neuron, saturates STDP | Cap W at 1.5, rely on homeostatic threshold adaptation, report weight histograms at every epoch so we can see saturation. |
| Lateral inhibition too strong → no output ever fires; too weak → all outputs fire | Tune output→output weight magnitude as part of implementation. Acceptable-range check in the smoke test. |
| Our new `RATE_VECTOR_POISSON` pattern breaks existing experiments | New enum value, new branch — additive change. Existing experiments untouched. Run full benchmark suite pre-commit. |
| 3 seeds is too few — results not statistically robust | Acceptable for G1 (proof-of-principle). If results are marginal, G2 expands to 5+ seeds. |

---

## 9. Deliverables

- `research/datasets/tiny_patterns.py` + dataset `.npz` on disk
- `sim/enums.py`: new `StimulusPatternType.RATE_VECTOR_POISSON`
- `sim/config.py`: new fields in `StimulusPattern` (`rate_vector_hz`)
- `experiment/stimulus.py`: new pattern branch in `_compute_pattern`, initializer for the new pattern's Poisson timers
- `research/runners/g1_runner.py`: headless training + evaluation loop
- `research/findings/2026-04-20-g1-<seed-aggregate>.md`: results doc
- Tests 1–5 above
- CHANGELOG `[Unreleased]` entry
- Green `pytest tests/` and `run_benchmarks.py` after integration

---

## 10. Commit plan

1. Design doc (this file)
2. `tiny_patterns` dataset module + tests + built `.npz`
3. `RATE_VECTOR_POISSON` stimulus pattern + tests
4. G1 network wiring helper + tests
5. Decoder helpers + tests
6. `g1_runner.py` + smoke test
7. Full training run (3 seeds × 10 epochs)
8. Findings doc + CHANGELOG entry

Each step its own commit. Tests pass before each commit.
