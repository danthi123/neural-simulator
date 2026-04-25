# Science Roadmap

Living document tracking scientific improvements to the neural simulator.
Updated as features are implemented and validated.

**Last updated:** 2026-04-25

> **Recent arc (2026-04-20 → 2026-04-25):** the project pivoted from a flat
> "validate biology + optimise" agenda to an active research arc on
> reward-driven learning in spiking circuits. The original Pillars (analysis,
> bio benchmarks, performance) are largely DONE and are kept below as the
> credibility floor. The current frontier is summarised in
> [Pillar 4: Reward-Driven Learning Architecture](#pillar-4-reward-driven-learning-architecture)
> below, which subsumes Sessions D–I (silent-motor trap arc) and Phase A/B
> (preset audit + BG cascade). The 2026-04-25 acid test (74% improvement
> over G9 baseline) closes Phase B; next ceiling is the 22% BG-active rate.

---

## Pillar 1: Automated Analysis Pipeline

Goal: Turn the simulator from "runs experiments" into "produces publication-quality results."

### 1.1 Parameter Sweep Framework
- **Status:** DONE (2026-04-06)
- **Priority:** HIGH (multiplies value of every experiment)
- **What:** `run_parameter_sweep.py` accepts sweep configs (JSON or CLI `--sweep`) — varies one or more parameters across ranges, runs experiments in sequence, collects results into summary JSON + CSV.
- **Validated with:** STDP learning rate sweep (a_plus=0.004/0.012/0.024) showing monotonic dose-response (delta +6.4/+8.2/+12.1 Hz, all p<0.001).
- **Usage:** `python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.004,0.012,0.024"`

### 1.2 Spectral Analysis Integration
- **Status:** DONE (2026-04-06)
- **Priority:** HIGH
- **What:** `ReadoutEngine.compute_band_power()` extracts power in standard frequency bands. Band power is logged automatically at every phase transition via `ExperimentEngine._log_band_power()`.
- **Bands:** delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-80 Hz), high-gamma (80-150 Hz).
- **Validated with:** Frequency response experiment — PSD correctly captures stimulus frequency in input group and network's natural theta-band resonance.
- **Log event:** `{"event": "band_power", "group": "...", "delta": ..., "theta": ..., ...}`

### 1.3 Population Synchrony Metrics
- **Status:** DONE (2026-04-06)
- **Priority:** MEDIUM
- **What:** Fano factor of population spike count computed per readout window. Logged in every readout entry as `synchrony` dict.
- **Metric:** Fano factor = Var(N)/E[N] where N = spike count per step. Values: ~1.0 = Poisson (asynchronous), >1 = synchronous bursting, <1 = regular.
- **Validated with:** Stimulus-response experiment — baseline Fano ~0.9 (near-Poisson, correct for cortical networks without external synchronization), slight increase during stimulus.
- **Future extension:** Add Kuramoto order parameter for oscillatory networks, CV of ISI for individual neuron regularity.

### 1.4 Statistical Comparison Framework
- **Status:** DONE (2026-04-06)
- **Priority:** HIGH
- **What:** Automatic statistical tests in parameter sweep output for all experiment types.
- **Tests implemented:**
  - Welch's t-test (pre vs post, baseline vs stimulus)
  - Cohen's d effect size
  - Significance flags (|t| > 2.0)
- **Future extension:** 95% confidence intervals, Bonferroni correction for frequency sweep
- **Output:** p-values, effect sizes, and significance flags in summary JSON.

### 1.5 Batch Experiment Runner
- **Status:** NOT STARTED
- **Priority:** MEDIUM (depends on 1.1)
- **What:** Run multiple experiment types in sequence with a single config file. Generate a combined report.
- **Use case:** "Run all 4 experiments on the same network configuration and produce a comprehensive characterization."

---

## Pillar 2: Biological Benchmark Validation

Goal: Prove the simulator reproduces known neuroscience results, establishing credibility for novel predictions.

### 2.1 STDP Timing Curve (Bi & Poo 1998)
- **Status:** DONE (2026-04-06) - ALL CHECKS PASSED
- **Priority:** HIGH (gold standard validation)
- **Validation results:**
  - Kernel matches theory to machine precision (max error: 2.98e-08)
  - All dt>0 produce potentiation, all dt<0 produce depression
  - Magnitude decays exponentially with |dt| (tau=20ms)
  - Net LTP bias confirmed (A+=0.012 > A-=0.01, net=+0.1377)
  - Full simulation verification: 2-neuron forced-spike tests match kernel exactly at dt=-20,-5,+5,+20ms
- **Script:** `run_benchmarks.py --benchmark stdp-timing`

### 2.2 E/I Balance and Spontaneous Firing Rates
- **Status:** DONE (2026-04-06) - ALL CHECKS PASSED
- **Priority:** HIGH
- **Validation results (CORTEX_L23_RS_FS, 10K neurons, 10s):**
  - E/I ratio: 4.0 (80%/20% - correct for cortical L2/3)
  - Excitatory rate: 1.78 Hz (in 0.5-15 Hz range - PASS)
  - Inhibitory rate: 3.25 Hz (fires faster than excitatory - PASS)
  - CV of ISI: 0.86 excitatory, 0.85 inhibitory (in 0.3-2.0 range - PASS, irregular Poisson-like)
  - 100% neuron participation (PASS)
- **Note:** Inhibitory rate (3.25 Hz) is lower than typical in-vivo FS rates (10-50 Hz). This is expected for Izhikevich model without tonic drive — the OU noise sigma=100 pA provides the background that drives both populations, but doesn't differentiate E/I input levels as strongly as in vivo. The HH model with realistic channel conductances would show a larger E/I rate difference.
- **Script:** `run_benchmarks.py --benchmark ei-balance`

### 2.3 Short-Term Plasticity Fidelity (Tsodyks-Markram)
- **Status:** DONE (2026-04-06) - ALL CHECKS PASSED
- **Priority:** MEDIUM
- **Validation results:**
  - E->E (U=0.5, tau_d=200, tau_f=20): Depressing PPR=0.71 at 5ms ISI (PASS)
  - I->E (U=0.25, tau_d=100, tau_f=50): Facilitating PPR=1.28 at 5ms ISI (PASS)
  - All types recover toward PPR=1.0 at long ISI (1000ms) (PASS)
  - Fused kernel matches analytical solution (PPR error < 0.02) (PASS)
- **Script:** `run_benchmarks.py --benchmark stp-paired-pulse`

### 2.4 Gamma Oscillation Emergence from E/I Balance
- **Status:** DONE (2026-04-06) - ALL CHECKS PASSED
- **Priority:** MEDIUM
- **Bug found & fixed:** Spatial connectivity generator used additive trait bias with top-k selection, causing 100% same-type segregation (zero cross-type connections). Fixed by switching to probabilistic sampling weighted by distance * trait factor. This bug affected ALL spatial connectivity generation, not just gamma networks.
- **Validation results (CORTEX_GAMMA_FS_NETWORK, 5000 neurons, 5s):**
  - Cross-type connectivity: E->I=100K, I->E=93K (PING-capable)
  - Peak frequency: 27-45 Hz (gamma band) (PASS)
  - Beta+gamma fraction: 41-47% of power (PASS)
  - Inhibitory fires faster than excitatory (PASS)
- **Script:** `run_benchmarks.py --benchmark gamma-oscillations`

### 2.5 Homeostatic Firing Rate Regulation
- **Status:** DONE (2026-04-06) - ALL CHECKS PASSED
- **Priority:** LOW
- **Validation results (5000 neurons, 53s total):**
  - Baseline: 5.31 Hz spontaneous firing
  - Perturbation (+200 pA): 50.55 Hz (10x increase, PASS)
  - Recovery: returns to 4.99 Hz within 1s of current removal
  - Late recovery (30s): 4.99 Hz (6% of baseline, well within 50% tolerance, PASS)
  - Threshold adaptation rapidly compensates for perturbation
- **Script:** `run_benchmarks.py --benchmark homeostasis`

---

## Pillar 3: Performance Optimization

Goal: Enable larger networks (50K-100K neurons) to run interactively.

### 3.1 Batched Sparse Matrix Multiplication
- **Status:** DONE (2026-04-06)
- **Priority:** HIGH
- **What:** Stack exc/inh firing vectors into (n,2) matrix, perform single `A.T @ B` instead of two separate matmul calls. Reuses CSR index traversal.
- **Benchmark:** 10K neurons, 2000 steps: 6.33 ms/step (vs ~6.9 ms/step before in similar conditions).

### 3.2 GPU Memory Management
- **Status:** DONE (2026-04-06)
- **Priority:** MEDIUM
- **What:** Removed `free_all_blocks()` from mid-simulation hot paths (synapse growth, playback cache clear). These caused 50-200ms GPU sync stalls. CuPy's memory pool reuses freed blocks naturally. `free_all_blocks()` retained only in cleanup/shutdown paths.
- **Changed:** Lines 4544 (synapse growth) and 7635 (playback cache) — removed sync-inducing frees.

### 3.3 Render Frame Skipping
- **Status:** DONE (2026-04-06)
- **Priority:** MEDIUM
- **What:** Added `VBO_UPDATE_SKIP` (default: 2) to OpenGL config. VBOs now update every 2nd render frame, halving GPU->CPU sync overhead during rendering. Configurable via `opengl_viz_config["VBO_UPDATE_SKIP"]` and `GPUConfig.render_vbo_update_skip`.

### 3.4 LZ4 Recording Compression
- **Status:** ALREADY IMPLEMENTED (verified 2026-04-06)
- **Priority:** LOW
- **What:** LZ4 compression was already fully implemented in the recording system. `GPUConfig.recording_compression` defaults to `"lz4"`. Falls back to gzip if `hdf5plugin` not installed. Verified `hdf5plugin.LZ4()` works on this system.

---

## Completed Work (This Session)

### Experiment System Bug Fixes (2026-04-06)
- [x] Structural plasticity frozen during experiments (was adding 657K random synapses)
- [x] Hebbian weight decay frozen during experiments (was destroying 40% of network weights)
- [x] CS-to-US connections filtered to excitatory-only (inhibitory 2.1x propagation opposed learning)
- [x] `active_channels=[]` now means "no channels" not "all channels"
- [x] `sim_bridge_ref.cfg` -> `sim_bridge_ref.core_config` (reward signal was never delivered)
- [x] Eligibility trace sliced to `[:actual_nnz]` (shape mismatch crash)
- [x] RL target window and reward/punishment tuned for bootstrappability
- [x] `ExperimentConfig.override_propagation_strength` for experiment-level SNR boost
- [x] Inter-group initial weight lowered to 0.1 (10x STDP headroom)
- [x] Connection probability increased to 95% (excitatory-only pool)

### Experiment Validation Results (2026-04-06)
| Experiment | Status | Key Metric |
|-----------|--------|-----------|
| Basic Stimulus-Response | PASS | Input +15 Hz, Output +1.3 Hz (t=4.06) |
| Associative Conditioning | PASS | CS-ON: +9.1 Hz (t=11.36), weights 0.10->0.999 |
| Frequency Response | PASS | Bandpass, peak 8.1 Hz, 1.96x peak/trough ratio |
| Reinforcement Learning | PASS | Success rate 20% -> 40% over 100 trials |

### Headless Experiment Runner (2026-04-06)
- [x] `run_experiment_headless.py` — runs all 4 experiments without GUI
- [x] Per-experiment analysis with statistical summaries
- [x] CS-on/CS-off period filtering for associative conditioning

### Analysis Pipeline (2026-04-06)
- [x] `run_parameter_sweep.py` — grid/zip parameter sweeps with JSON+CSV output
- [x] `ReadoutEngine.compute_band_power()` — 6-band spectral analysis at phase transitions
- [x] Population synchrony (Fano factor) logged in every readout entry
- [x] Welch's t-test + Cohen's d in sweep analysis output

### Biological Benchmarks (2026-04-06)
- [x] `run_benchmarks.py` — 4 benchmark suite (STDP timing, E/I balance, STP PPR, gamma oscillations)
- [x] **Critical bug fixed:** Spatial connectivity generator used additive trait bias with top-k selection, causing 100% same-type segregation. Fixed with probabilistic sampling. This bug affected ALL networks using the GPU spatial generator.
- [x] Batched sparse matmul for conductance propagation (exc+inh in single operation)

### Benchmark Results Summary (2026-04-06)
| Benchmark | Status | Key Result |
|-----------|--------|-----------|
| STDP Timing (Bi & Poo 1998) | PASS | Kernel matches theory to 3e-8, full sim verified |
| E/I Balance | PASS | 80/20, exc 1.8 Hz, inh 3.2 Hz, CV(ISI) 0.86 |
| STP Paired-Pulse (Markram 1998) | PASS | E->E depressing PPR=0.71, I->E facilitating PPR=1.28 |
| Gamma Oscillations (PING) | PASS | Peak 27-45 Hz, beta+gamma 41-47% of power |

---

## Pillar 4: Reward-Driven Learning Architecture

Goal: build a trainable artificial-life core that matches real-world biology
and learns by reward modulation. This pillar emerged organically from the G1
through G11 research-gate progression and is the current frontier.

### 4.1 Module split + research-runner framework
- **Status:** DONE (2026-04-20 → 2026-04-24)
- **What:** monolithic `neural-simulator.py` (~12K lines) extracted into
  `sim/`, `viz/`, `ui/`, `experiment/` packages. `research/runners/` houses
  16 headless gate runners. Negative findings documented as first-class
  artifacts in `research/findings/`.

### 4.2 G1–G6 gates (2026-04-20 → 2026-04-21)
- **G1: encoder-decoder roundtrip** — **GO** (v3, 71.3% test acc, 3 seeds)
- **G2: STDP local learning** — NO-GO (no epoch-over-epoch improvement)
- **G3: persistence/checkpointing** — GO
- **G5: sensorimotor signed perceptron** — GO (v3 with LR decay, 3/3 seeds)
- **G6: 2D gridworld** — PARTIAL (agent learns but Q1→Q4 gate metric
  misfires because convergence is too fast)
- **G7: gate metric redesign** — proposed three replacements (fraction-of-
  steps-within-dist-1-of-goal, moving-goal robustness, random-start variance)

### 4.3 Silent-motor trap (Sessions D–I, 2026-04-22 → 2026-04-25)
- **Symptom:** on G9 moving-goal RL, motor neurons silent in phase 1
  cannot acquire STDP eligibility, so reward never reaches them. Agent
  stuck on phase-1 winners.
- **Variants tried:** V1 motor exploration noise (best of the lot, but
  still trapped), V2 first-spike + rate boost, V3 positive-only reward,
  V4 action attribution, V5 proportional sampling (effectively random),
  V6 weight reset on goal change, V7 ε-greedy. **All NO-GO.**
- **Diagnosis:** trap is structural — a shared 200-neuron reservoir with
  argmax readout has dominant-motor bias from random initial weights that
  no runner-side hack can fix.
- **Resolution:** Phase B BG cascade (see §4.5).
- **Findings:** `2026-04-24-session-d-part-a.md`, `2026-04-24-session-d-part-c.md`, `2026-04-24-session-c.md`, `2026-04-25-session-g-motor-exploration.md` (later session-G work), `2026-04-25-pfc-bistability-negative.md`.

### 4.4 Brain-region framework + neuromodulator subsystem (Sessions E.1–E.2, 2026-04-24)
- **Status:** Both DONE and merged. Composable, opt-in, default-OFF for
  backward compatibility.
- **E.1 — Neuromodulator subsystem** (`sim/neuromodulators.py`): declarative
  DA / NE / 5-HT with concentration dynamics and receptor effects
  (`synaptic_gain`, `plasticity_rate`, `excitability_drive`). Replaces
  ad-hoc `current_reward_signal`. Framework GO; NE-params on silent-motor
  task NO-GO (the trap is upstream of NE modulation).
- **E.2 — Brain-region framework** (`sim/regions.py`): declarative
  `BrainRegion` + `RegionPathway` for multi-region simulations on a single
  bridge. PFC + Motor + Striatum + Thalamus etc. each own a contiguous
  index slice. Cross-region pathways respect per-pathway plasticity flags.
  Composes with E.1 — regions auto-register as neuromodulator groups.
- **Findings:** `2026-04-24-session-e1-neuromodulator-subsystem.md`.

### 4.5 Phase A: biology-preset audit (2026-04-25)
- **Status:** DONE.
- **What:** comprehensive validation of all neuron-model presets at 37°C.
- **Major bug found:** HH model didn't fire APs at 37°C — uniform Q10=3
  over-compressed gating dynamics. Fixed with per-gate Q10 (`hh_q10_m=3.0`,
  `hh_q10_h=hh_q10_n=1.5`).
- **Other fixes:** Izhikevich `default_neuron_type` was always ignored
  (trait-split was always-on; now opt-in for `num_traits>1`); AdEx presets
  all behaved identically (bridge wasn't loading them); GPE/STN g_NaP was
  5–10× too high; added 8 new IZH2007 BG/thalamus/HC/DA presets and 4 new
  HH BG presets (D1, D2, GPi, TAN); built full AdEx preset library (7
  phenotypes).
- **Result:** 30 working biological presets across HH + Izh + AdEx.
- **Findings:** `2026-04-25-hh-temperature-bug.md`, `2026-04-25-hh-preset-audit.md`,
  `2026-04-25-hh-presets-after-q10-fix.md`, `2026-04-25-izh-preset-audit.md`.

### 4.6 Phase B: BG-style action selection (2026-04-25, GO)
- **Status:** **GO**. 3-seed acid test on moving-goal scenario:
  phase 1 finalQ 1.76 avg vs G9 baseline 6.74 (74% improvement).
- **Architecture:** `research/runners/g11_bg_runner.py` builds a 30-region
  cascade — per-action `cortex_X → str_D1_X / str_D2_X → gpi_X → thal_X →
  motor_X` with disinhibition gating, plus shared STN and dopamine.
  Selection emerges from independent disinhibition gates, not a shared
  argmax — silent-motor trap structurally cannot occur.
- **Two non-obvious bugs found and fixed during the acid test:**
  1. `n_cortex=400` over-drove D1 to ~220 Hz (saturated, breaking GPi
     inhibition). Static probe used 100 and worked. Probes must match
     deployment scale.
  2. `cortex→D1 weight_mean=25` against default `stdp_w_max=2` collapsed
     weights from 25→2 in milliseconds (soft-bound STDP rule).
- **Result:** 22–24% of trials show real BG-driven motor selection (rest
  random fallback); BG-driven trials show strong correct-direction bias.
  Agent stays at Manhattan distance ~1.7 from goal vs random walk's ~5.5.
- **Findings:** `2026-04-25-phase-b-acid-test-real-win.md` (final),
  `2026-04-25-phase-b-cascade-stability-fix.md` (n_cortex bug),
  `2026-04-25-phase-b-honest-correction.md` (the trail of an overstated
  intermediate result), `2026-04-25-phase-b-bg-acid-test.md` (initial
  overstated finding kept for trail).

### 4.7 Open questions / next ceiling (2026-04-25 → ?)

The Phase B win produced 22% BG-active trials. The next ceiling is to
push that toward 50%+ via:

1. **Per-action dopamine targeting**: currently DA is broadcast; targeting
   DA to the active D1 pool (per Schultz / Wickens biology) would sharpen
   credit assignment and break ties faster.
2. **Lateral inhibition between motor pools**: the previous motor→motor
   pathway was incorrectly excitatory. FS interneuron sub-pools would
   create proper winner-take-all behaviour.
3. **Position encoding**: cortex drive is currently a heuristic
   "drive cortex_X if direction X is goal-relative." A proper sensory
   cortex with position-tuned (Gaussian) inputs and plastic
   sensory→cortex weights would let position→action mapping be learned.
4. **Harder tasks**: 8×8 grid + single goal change is the simplest moving
   goal. Try multiple goal changes, longer episodes, harder grids,
   non-grid environments.
5. **Reward signal sharpening**: current reward is +1 / 0 / -1 per step.
   Distance-shaped or temporal-difference reward could improve credit
   assignment on the 22% BG-active trials.

### 4.8 Cross-cutting: doc / repo hygiene
- Module-split docs (CLAUDE.md, CONTRIBUTING.md) were stale (single-file
  era references) — refreshed 2026-04-25.
- README architecture diagram added (Mermaid) — 2026-04-25.
- Findings index added at [`research/findings/INDEX.md`](../research/findings/INDEX.md) — 2026-04-25.
