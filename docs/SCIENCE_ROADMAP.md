# Science Roadmap

Living document tracking scientific improvements to the neural simulator.
Updated as features are implemented and validated.

**Last updated:** 2026-04-28

> **Recent arc (2026-04-20 → 2026-04-28):** the project pivoted from a flat
> "validate biology + optimise" agenda to an active research arc on
> reward-driven learning in spiking circuits. The original Pillars (analysis,
> bio benchmarks, performance) are largely DONE and kept below as the
> credibility floor. The current frontier — [Pillar 4: Reward-Driven
> Learning Architecture](#pillar-4-reward-driven-learning-architecture) —
> has now reached a major milestone:
>
> **As of 2026-04-28: 4 of 5 perception/reward cheats closed, agent navigates
> from biologically-grounded sensory information, biology-grounded BEATS
> cheats-allowed (4.08 vs 4.41, p=0.00045, 30.6% over baseline, 6/6 seeds).**
>
> Arc summary:
> - **2026-04-25** Phase B BG cascade closes silent-motor trap (74% over G9 baseline)
> - **2026-04-26** Plastic-input-layer ceiling hit (7 NEGATIVE attempts in one day)
> - **2026-04-27 (early)** Phase C breakthrough: per-pathway plasticity gating + real curriculum (6/6 seeds, p=0.02)
> - **2026-04-27 (mid)** PFC working memory composing (4.41, p=0.018)
> - **2026-04-27 (night)** Item 1 perception arc complete: 3 of 5 cheats closed (4.56, p=0.00819)
> - **2026-04-27/28 (overnight)** + sensed reward → **4 of 5 cheats closed** (4.08, p=0.00045)
>
> See [`research/findings/2026-04-27-NEW-BEST-4cheats-closed.md`](../research/findings/2026-04-27-NEW-BEST-4cheats-closed.md)
> for the milestone, [`research/findings/2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md`](../research/findings/2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md)
> for the perception-only step, and [`research/findings/INDEX.md`](../research/findings/INDEX.md)
> for the full session-by-session arc.

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

### 4.7 Phase B refinement experiments (2026-04-26)

After Phase B's structural win (sum=5.24 baseline finalQ), an autonomous
overnight session iterated on three "what's next" candidates plus
several derived gates. Result summary:

| Variant | P0 finalQ | P1 finalQ | Sum | Status |
|---|---:|---:|---:|---|
| Baseline (Phase B as-is) | 3.48 | 1.76 | 5.24 | reference |
| Motor WTA (lateral inhibition) | 2.40 | 2.46 | 4.86 | PARTIAL — exploitation+, readaptation− |
| Per-action DA (hard) | 2.04 | 2.61 | 4.65 | PARTIAL — same trade-off as WTA |
| Adaptive per-action DA (sym tau~10) | 1.85 | 2.14 | 3.99 | GO — first clean win on sum |
| Adaptive DA (sym tau~3) | 2.19 | 2.13 | 4.33 | NEUTRAL — too noisy |
| **Adaptive DA (asym, slow+/fast-)** | **1.61** | **1.92** | **3.53** | **GO — current best** |
| WTA + adaptive DA (sym) | 2.23 | 2.18 | 4.41 | NEGATIVE — composes badly |
| WTA + asymmetric adaptive DA | 2.05 | 2.24 | 4.29 | NEGATIVE — same |
| DA-gated WTA + asymmetric DA | 2.12 | 2.42 | 4.54 | NEGATIVE — gating doesn't rescue |
| Learned perception (sensory→cortex) | 5.58 | 5.27 | 10.85 | NEGATIVE — cold-start fail |

**Key insights from this iteration:**

1. **Sharpening creates an exploitation/exploration trade-off.** WTA and
   per-action DA both improve phase 0 acquisition while hurting phase 1
   readaptation. Two independent mechanisms producing the same pattern
   confirms it's structural, not tuning.

2. **Adaptive sharpening solves the trade-off.** Reward-EMA-gated DA
   targeting commits when winning, broadcasts when losing. Asymmetric
   ramp (slow positive, fast negative — biologically: phasic DA dip
   faster than ramp) gives the best result.

3. **WTA is structurally redundant once credit is well-targeted.** WTA
   addresses motor selection ambiguity, but adaptive DA already produces
   decisive policy via selective reinforcement. Adding WTA on top is
   double-bookkeeping that hurts more than it helps. Even DA-gated WTA
   doesn't recover.

4. **Learned perception cold-starts catastrophically.** Replacing
   heuristic cortex drive with plastic sensory→cortex requires informed
   initialization or curriculum learning. Random init in 1800 trials
   produces no signal differentiation.

**Recommended Phase B configuration:** `python -m research.runners.g11_bg_runner
--moving-goal --adaptive-da --adaptive-da-ema-decay-negative 0.7 [...]`

### 4.8 Phase C: plastic-input-layer arc + curriculum (2026-04-26 → 2026-04-27)

After Phase B's success on heuristic-driven cortex, the next ceiling
was making **plastic input layers** (sensory, hippocampus) work
alongside the cascade. 7 NEGATIVE attempts on 2026-04-26 (cold-start
perception, informed-init, hippo replacement/additive, cortex WTA,
WTA+adaDA, drive-gated curriculum) all failed to match baseline.

**Resolution (2026-04-27):** built per-pathway plasticity gating
infrastructure + real curriculum learning + removed cortex WTA. 6/6
seeds beat baseline.

| Variant | 6-seed avg sum | beats baseline | p-value |
|---|---:|---|---:|
| Baseline (heuristic only) | 5.88 | reference | — |
| **Hippo + curriculum (full freeze)** | **4.72** | **6/6** | **0.02** |
| Sensory + hippo + curriculum (full freeze) | 4.63 | 5/6 | 0.05 |
| Sensory + hippo + curriculum (partial freeze 0.2) | 4.79 | 5/6 | 0.10 |

**Key infrastructure additions:**

1. **Per-pathway plasticity gating** — `RegionPathway.plasticity_gate`
   field; `cp_plasticity_gain` array gates STDP, eligibility, Hebbian,
   synaptic scaling; bridge methods `set_plasticity_gate(name, value)`.
2. **NM-driven plasticity gates** — `target_type="plasticity_gate"`
   with `scope="gate:<name>"` lets neuromodulator concentrations drive
   gate values directly (DA-gated corticostriatal LTP, ACh-gated
   cortical attention, developmental NM ramps).
3. **Real curriculum** — phase 1 corticostriatal plastic + input layers
   frozen; phase 2 cortex frozen (or partial) + input layers thawed.
4. **Heuristic-decay infrastructure** — for testing whether learned
   weights can navigate without heuristic teacher.
5. **Sleep-replay infrastructure** — random replay neutral; trajectory
   replay slightly worse (biased by stale entries); both worse than
   no-sleep on 2100-step task. Future work needs better replay content.

**Multi-goal correction (2026-04-27 night):** the 3-seed multi-goal
result (7.83) didn't replicate at 6 seeds (8.32, exactly tying baseline).
Curriculum doesn't help fast-change tasks because cortex frozen in
phase 2 can't track changing reward landscape. Lesson: 3-seed
indicators are unreliable; always validate with 6+ seeds.

**Recommended config (2-goal slow-change):**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --curriculum-phase2-cortex-gain 0.2 \
    --seed N --n-steps 1800
```

For multi-goal/fast-change tasks: skip curriculum; baseline broadcast
DA handles fast-change better.

**Findings written 2026-04-27:**
- `2026-04-27-plastic-input-layer-RESOLVED.md` (the breakthrough)
- `2026-04-27-perception-additive.md` (sensory layer additive)
- `2026-04-27-task-adaptive-curriculum.md` (partial freeze + corrections)
- `2026-04-27-sleep-replay-infrastructure.md` (sleep replay neutral)
- `2026-04-27-overnight-summary.md` (consolidated session summary)

### 4.9 PFC working memory (Item 3, 2026-04-27)

**Status:** GO. Adding a recurrent prefrontal region on top of the
hippocampus + sensory + curriculum stack gave a clean 6-seed win.

| Variant | 6-seed avg sum | beats baseline | p-value |
|---|---:|---|---:|
| Baseline | 5.88 | reference | — |
| Hippo + sensory + curriculum | 4.72 | 6/6 | 0.02 |
| **+ PFC working memory** | **4.41** | **6/6** | **0.018** |

PFC region is built declaratively via `BrainRegion` with internal
recurrent connectivity (sparse glutamatergic loops). It receives
goal-context input and projects forward through the cortex pools to
the BG cascade.

**Stage 2 delayed-response test (3-seed):** PFC's drop during goal
silence is 17% smaller than no-PFC's drop (d=0.73, p=0.51) — preliminary
indication of real persistent activity. 6-seed validation pending.

Findings:
- `2026-04-27-pfc-working-memory.md`
- `2026-04-27-pfc-stage2-delayed-response.md`

### 4.10 Item 1: Perception arc (2026-04-27 night)

**Status:** **COMPLETE.** All three coordinate cheats closed — agent
navigates from PERCEIVED beacon and landmark information with no direct
(gx, gy) or (x, y) access anywhere.

| Variant | 6-seed avg sum | beats baseline | p-value | Cheats closed |
|---|---:|---|---:|---|
| Baseline (all cheats) | 5.88 | reference | — | 0/5 |
| Best WITH cheats | 4.41 | 6/6 | 0.018 | 0/5 |
| Stage 1 (beacon only) | 5.36 | 5/6 | 0.342 | 1/5 |
| Stage 1+3 (beacon + reflex) | 4.77 | 6/6 | 0.00188 | 2/5 |
| **Stage 1+2+3 (full perception)** | **4.56** | **6/6** | **0.00819** | **3/5** |

**Architecture:**

```
Goal at (gx,gy) emits beacon  →  8 directional beacon sensors (cosine tuning)
   →  plastic beacon → goal_cells (curriculum-gated)
   →  cue-following reflex → cortex_X drive (replaces heuristic)

Landmark at fixed (lx,ly) emits cue  →  8 directional landmark sensors
   →  plastic landmark → place_cells (curriculum-gated, self-organizes)
```

The agent's only source of "knowing where the goal is" is the beacon;
its only source of "knowing where it is" is the landmark perception.
These match what real animals use.

**Recipe:**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception --enable-dlpfc-wm \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --enable-landmark-sensor --landmarks-replace-place \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```

Findings:
- `2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md` (the milestone)
- `2026-04-27-stage3-full-perception-BREAKTHROUGH.md`
- `2026-04-27-stage1-beacon-perception.md`
- `docs/plans/2026-04-27-perception-arc-plan.md`

### 4.11 NEW BEST: 4 of 5 cheats closed (2026-04-27/28 overnight)

**Status:** **GO — STATISTICALLY SIGNIFICANT.** Adds **sensed reward**
(beacon-intensity gradient instead of ground-truth distance) on top of
the perception arc. The result is the new flagship.

| Variant | 6-seed avg | beats baseline | p-value | Cheats closed |
|---|---:|---|---:|---|
| Baseline | 5.88 | reference | — | 0/5 |
| Best WITH cheats | 4.41 | 6/6 | 0.018 | 0/5 |
| Stage 1+2+3 (perception arc) | 4.56 | 6/6 | 0.00819 | 3/5 |
| **★ + sensed reward** | **4.08** | **6/6** | **0.00045** | **4/5** |

**The biology-grounded version (4.08) BEATS the cheats-allowed version
(4.41).** Closing perception/reward cheats actually *helps* — likely
because the new mechanisms add richer state information (gradient
direction, sensor patterns) than coordinate access.

**Sensed reward implementation:**
```python
intensity_before = beacon_max_intensity / (1.0 + beacon_falloff * d_before)
intensity_after  = beacon_max_intensity / (1.0 + beacon_falloff * d_after)
reward = sign(intensity_after - intensity_before)  # ±1 / 0
```

**Cheat #5 — ON HOLD pending biology buildout** (reframed 2026-04-28 afternoon):

| Variant | n-seed avg | P0 | P1 | Verdict |
|---|---:|---:|---:|---|
| v1 (curriculum-staged cross-projections) | 10.87 (3-seed) | — | — | NEGATIVE — gate doesn't freeze synaptic transmission |
| v2 (zero-init cross-projections) | 7.89 (3-seed) | 2.49 | 5.40 | NEGATIVE — thaw-time STDP corrupts converged policy |
| **v3 (`--bg-lateral-inhibition`)** | **4.26 ± 0.50 (6-seed)** | 2.35 | 1.91 | **GO — no regression; permanent default** |
| v3.1 (`... --bg-cross-projections` + adult thaw, single-goal) | 8.92 ± 2.44 (6-seed) | 2.58 | 6.35 | NO-GO — phase-2 breaks |
| **v4 (`... --developmental-pretraining`, single-goal)** | 11.34 ± 1.85 (3-seed Tier 2) | 4.88 | 6.46 | **NO-GO — worse than v3.1** |
| **Option 1 (`--enable-structural-pruning`, multi-goal)** | 22.46 ± 4.84 (n=2; seed 42 hung) | 4.0 | 7.5 | 6.7/4.3 (P2/P3) | **NO-GO — 3.2× worse than v3** |
| **Option 2 (`--cross-projection-density 0.25`, multi-goal)** | 8.76 ± 2.54 (n=3) | 1.83 | 2.05 | 3.36/1.53 | **HIGH-VARIANCE PARTIAL** — seed 44 hit 5.88 (beats baseline) |
| **Cluster B.1 + patch-matrix (`--enable-d1-d2-asymmetry`, multi-goal)** | **7.62 ± 1.23 (n=3)** | 1.81 | 2.42 | **1.92** | 1.46 | **PARTIAL SIGNAL** — variance halved, P2 catastrophe eliminated. First evidence cluster strategy works. |
| Cluster B.1 alone (no cross) | 7.00 ± 0.52 (n=3) | 2.21 | 1.68 | 1.63 | 1.48 | non-regression confirmed (B.1 doesn't hurt baseline) |
| **Cluster B.1+B.2 + patch-matrix (`--enable-striatal-fsis`)** | **8.44 ± 0.62 (n=3)** | 3.72 | 1.53 | 1.59 | 1.60 | **MIXED** — variance halved again (1.23 → 0.62), P1+P2+P3 (4.72) beats v3 baseline (4.89), but Phase 0 broken (3.72 vs 1.83) because FSIs broadcast pre-commitment. Phase-0 architectural issue. Proceeding to B.3. |
| Cluster B.1+B.2 alone (no cross, retuned weight=2.0) | 9.50 ± 0.85 (n=3) | 4.15 | 1.60 | 2.01 | 1.74 | regression: B.2 hurts baseline cascade. Original weight=8.0 was 19.78 ± 2.28 (catastrophic). |

- **v3 GO:** MSN cross-pool lateral inhibition (24 GABAergic pathways,
  `plastic=False`) is biology-grounded, harmless to flagship performance,
  **shipped as a permanent default in the recommended flagship config**.
- **v3.1 + v4 NO-GO under single-goal eval:** various cross-projection
  training regimes (adult thaw, developmental pretraining) all fail.
- **Multi-goal methodological correction (2026-04-28 afternoon):** all
  prior cheat-5 work used a single goal change at step 300 + 1500 stable
  steps — a "static adult" test, not the dynamic switching scenario
  cross-projections would actually be useful for. Switched to
  `--goal-schedule multi` (4 phases × 450 steps, 3 transitions).
- **Option 1 NO-GO under multi-goal:** structural pruning didn't reshape
  topology meaningfully; results catastrophic.
- **Option 2 partial signal under multi-goal:** sparse static topology
  shows high variance (σ=2.54). Phase 2 (1,6→1,1) has σ=2.09 across
  seeds — "topology luck" pattern. Seed 44 actually beat baseline at
  5.88, suggesting cross-projections aren't fundamentally broken when
  the right cross-pairs are present, just under-constrained without
  surrounding biology to consistently select them.
- **Reframe: cheat #5 is ON HOLD pending biology buildout, not closed by
  design.** Cross-projections need a complete striatal microcircuit, a
  closed BG loop, and a properly-structured DA system to behaviorally
  pay off. Building these out is a multi-month research program organized
  cluster-by-cluster:
    - Cluster B: striatal microcircuit (D1/D2 asymmetry + FSIs + TANs)
    - Cluster A: closed BG loop (thalamo-cortical feedback + hyperdirect)
    - Cluster C: DA system completeness (tonic + compartmentalized)
    - Cluster D: sequence-aware learning (hippocampus / PFC → striatum + replay)
    - Cluster E: connectivity refinement (already most explored)
- Findings: `2026-04-28-cheat5-post-v4-reframe.md`, `2026-04-28-cheat5-v4-results.md`,
  `2026-04-28-cheat5-v3-results.md`, `2026-04-28-cheat5-v2-NEGATIVE.md`,
  `2026-04-28-cheat5-v1-NEGATIVE.md`.

**Recipe (current flagship — includes v3 lateral inhibition):**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception --enable-dlpfc-wm \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --enable-landmark-sensor --landmarks-replace-place \
    --sensed-reward \
    --enable-msn-lateral-inhibition \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```

Findings:
- `2026-04-27-NEW-BEST-4cheats-closed.md` (the original 4.08 milestone)
- `2026-04-28-cheat5-v3-results.md` (v3 lateral inhibition GO; v3.1 cross NO-GO)
- `2026-04-28-cheat5-v4-results.md` (v4 developmental pretraining NO-GO; cheat #5 closed by design)

### 4.12 Open future directions

The cheats inventory is now down to one structural item plus a list of
larger architectural extensions:

1. **Cheat #5 v4 — developmental pre-training** — v1 (curriculum-staged),
   v2 (zero-init), v3.1 (with lateral inhibition) all NEGATIVE on adult
   learning. The next attempt frames cross-projection refinement as a
   **developmental phenomenon**: pre-train cross-projections across
   multiple goal positions during a critical period, freeze at the end,
   then evaluate. See [`docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md`](plans/2026-04-28-cheat5-v3-lateral-inhibition.md)
   Task 4 for the plan.
2. **Sleep-replay with proper trajectory content** — random and stale
   trajectory replay don't help. Recency-weighted, current-goal-only
   replay might. Infrastructure (NREM trajectory + REM random) added
   2026-04-27; content quality is the bottleneck.
3. **Multi-modal sensory integration** — visual + proprioceptive layers
   composing via separate plasticity gates.
4. **Cerebellum** — timing, error correction, fine motor.
5. **Larger task domains** — 16×16 grid recipe re-tuning (architecture
   scales but recipe is 8×8-tuned), more goal positions, multi-step
   plans.
6. **Continuous time / continuous actions** — major architecture change.
7. **Distance-shaped reward** — current ±1 binary is sparse; gradient-
   magnitude scaling could improve learning quality (note: sensed-reward
   already uses gradient direction; magnitude scaling would compose).

### 4.13 Cross-cutting: doc / repo hygiene
- Module-split docs (CLAUDE.md, CONTRIBUTING.md) were stale (single-file
  era references) — refreshed 2026-04-25.
- README architecture diagram added (Mermaid) — 2026-04-25.
- Findings index added at [`research/findings/INDEX.md`](../research/findings/INDEX.md) — 2026-04-25.
- QUICKSTART.md added 2026-04-27 — 60-second getting-started for new users.
- Documentation overhaul aligning with PyTorch/Django/Rust Book patterns — 2026-04-28.
- GPU throughput investigation (2026-04-28): concurrency knee at 4-6;
  CUDA MPS Linux-only (RTX 3090/Windows host has no MPS path); motor-counting
  code fix REVERTED (no measurable improvement). Webapp now defaults
  `--progress-print-interval=20` for non-interactive launches. Finding:
  `research/findings/2026-04-28-throughput-investigation.md`.
- Webapp polish (2026-04-28): live mode toggle, top-bar layout, font
  consistency via `--font-sans`/`--font-mono` CSS vars, collapsible HUDs,
  runs page no-flicker, goal-change dots on live chart, no-cache static,
  Windows DETACHED_PROCESS subprocess detach.
