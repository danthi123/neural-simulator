# Changelog

All notable changes to the GPU-Accelerated Neural Network Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This is a research codebase; entries are organised chronologically rather than by release tag. The freshest dated section is the working tip.

## [Unreleased] — 2026-05-03/04 — Permuted-label control debunks 28.5% W→A + autonomous-arc tooling

### Critical correction
- **Permuted-label control test — NEGATIVE.** Across 25 prior text I/O
  eval files (baseline / v2+SWR / H4 / curriculum / dpop / BigLang /
  BigMotor / NoLTD / NoT1 / xcouple / multidec / 200ep), 0/25 had the
  TRUE labeled mapping ranked best of 24 permutations. Best permutations
  consistently score 30-37% (8pp above chance), but the structure is
  randomly oriented per-seed, NOT aligned with task labels. The
  previously-documented 28.5% W→A "validated" result is structure above
  chance, not aligned word→action learning. The W→A binomial p=0.027
  is technically correct but doesn't measure aligned learning — only
  the presence of *some* above-chance structure. Real learning requires
  aligned ratio ≥ 4/6 across seeds. Finding:
  `research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`.
- **Cascade-as-cause hypothesis FALSIFIED.** Minimal-isolation test
  (`text_minimal_isolation.py`, 2026-05-04): minimal architecture
  (`language_input → motor_X` with NO cascade) gives mean 16.7% (BELOW
  chance) at 3 seeds. Cascade was a weak DAMPENER on seed-dependent
  random structure, not its source. Finding:
  `research/findings/2026-05-04-minimal-isolation-INVERSION.md`.

### Added — autonomous-arc tooling
- **`research/runners/permuted_label_check.py`** — definitive
  learning-vs-noise tool: ranks the true labeled mapping against all 24
  symbol permutations. Use as the gate for any text I/O claim.
- **`research/runners/eval_sanity_check.py`** — eval methodology
  validation via hand-built perfect weights. Tests whether the eval
  methodology itself works before debating architectural changes.
- **`research/runners/morning_briefing.py`** — summarizes overnight
  background runs into a single status report.
- **`research/runners/text_minimal_isolation.py`** — minimal architecture
  + biology helpers (`apply_topographic_bias`, `enable_motor_fs`,
  `freeze_stdp`) for testing biology fixes in isolation.
- **`research/runners/unaligned_pattern_analysis.py`** — cross-condition
  structural-bias analyzer; surfaces seed-dependent +3pp motor_E
  cascade bias.
- **`sim/progress.py`** — universal `[PROGRESS] {json}` event format
  consumed by experiment runner and webapp.
- **`research/experiment_runner.py`** — YAML-driven sweep orchestrator
  with built-in configs (biology, minimum_biology, sanity_check,
  b2_sparse_codes, b4_long_training).
- **`research/result_aggregator.py`** — cross-condition aggregation +
  verdict line.
- **Pre-staged decision chain**
  (`research/findings/raw/g11_bg/wait_biology_then_decide.ps1`) —
  auto-launches outcome-A vs outcome-B follow-ups based on biology
  sweep alignment ratio.
- **7-8× speedup stack shipped.** dt=1.0 + parallel-3 GPU sharing +
  `cfg.fast_spike_reset` (cp.where masked-update). Brings 6-seed
  batches from ~6h down to ~45-55 min. Finding:
  `research/findings/2026-05-04-perf-speedup-stack.md`.
- **Autonomous-runs skill** — guides multi-hour autonomous overnight
  arcs with quick-win prioritization, internal trade-off debate, and
  continuation through eval cycles.

### In-flight (2026-05-04)
- Biology-grounded sweep testing topographic prior (Pulvermüller
  2001-2003), PV-FS lateral inhibition between motor pools (Vogels 2011
  / Hofer 2011), and combined. Anti-cheat control runs first with STDP
  frozen — if alignment occurs without learning, prior is too strong.
  Tier-2 fallbacks (`b2_sparse_codes.yaml`, `b4_long_training.yaml`)
  pre-staged.

## [Unreleased] — 2026-04-29 — Catalog remediation pass + Clusters A/C/D scaffolding

### Added — Clusters A (closed BG loop) + C (tonic DA) + D (hippocampus trisynaptic loop)

After the catalog-driven remediation pass (R items below), three opt-in clusters
were scaffolded as the cheat-5 closure attempts continue:

- **Cluster A — Closed BG loop (`2d8be00`).** New `--enable-cluster-a-closed-loop`
  flag adds (a) cortex_X → stn hyperdirect (Nambu 2002, sparse 0.10, weight 3.0)
  and (b) thal_X → cortex_X feedback (action-specific only, density 0.50, weight
  5.0). Both static. Provides the post-synaptic activity / "teaching signal"
  that's been flagged as missing for cross-projection learning. Plan:
  `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md`. 4 tests.
  Cheat-5 multi-goal eval n=3 in progress.
- **Cluster C v1 — Tonic DA (`01fddf4`).** New `--enable-tonic-da` flag registers
  a `dopamine` neuromodulator (tonic baseline 0.5, decay_tau 200ms, plasticity_rate
  sensitivity +1.0, from_reward production rule). Bridge reward-modulation block
  switched to use (DA conc - DA baseline) as plasticity signal when DA registered;
  legacy path kept when off. Unlocks B.3 ACh window-gating which was a no-op
  without tonic DA-driven plasticity. Plan: `docs/plans/2026-04-29-cluster-c-tonic-da-design.md`.
  3 tests + smoke pass.
- **Cluster D v1 — Hippocampus trisynaptic loop (`3204c3e`).** New
  `--enable-cluster-d-hippocampus` flag adds 5 regions (ec, dg, dg_fs, ca3, ca1)
  with the canonical EC→DG→CA3→CA1 + EC→CA1 + CA3→CA3 wiring. DG sparsity via
  feedforward inhibition (dg_fs). CA3 plastic recurrent autoassociator
  (internal_density=0.30, plastic_internal=True). Mossy-fiber, perforant-path,
  Schaffer-collateral pathways all wired. CA1 → place_cells provides additional
  drive into existing perception arc when --hippocampus is also on. SWR generator
  (v2) and engram tagging (v3) deferred. Plan:
  `docs/plans/2026-04-29-cluster-d-hippocampus-design.md`. 6 tests + smoke pass
  (50 regions, 1454 neurons, 82,270 synapses).

### Eval status

Three chained background evals (each 6 runs at 1800 steps multi-goal, seeds 42/43/44):
1. Cluster A: baseline (no-A) vs +A (running)
2. Combo: A+C+B.3 vs C-only (queued)
3. Cluster D: D-only vs A+D (queued)

Total 18 data points across 6 conditions. Findings docs to follow.



### Added — Catalog-driven remediation pass (Kandel 6e + supplemental texts)

The textbook catalog session surfaced ~13 sim-level corrections. Plan: `docs/plans/2026-04-29-catalog-remediation-pass.md`.

- **R1.1 per-region E_inh override (`82b3d0d`).** PBR-160 ch 6/11: striatal MSNs use ~−60 mV (depolarizing-but-shunting GABA_A near rest); SNc DA neurons use ~−55 mV (lacks KCC2 chloride exporter). Global default −75 mV is wrong for these regions. Added `BrainRegion.syn_reversal_potential_i_override` field; bridge allocates per-neuron `cp_syn_reversal_potential_i_per_neuron` array; fused conductance kernel accepts scalar OR per-neuron array. Runner overrides on str_D1_*, str_D2_*, dopamine. 8 new tests.
- **R1.2 FSI cross-action wiring (`a1765b0`).** TK-2017 / Tepper-2018: paired recordings show MSN→MSN collaterals are weak (<0.5 mV IPSPs, ~14-25% conn prob); FSI→MSN feedforward is the dominant cross-pool WTA substrate. Rewired build_bg_brain_regions FSI loop: FS_X → MSN_Y for X≠Y only (24 paths, was 32 broadcast). Probe and tests updated.
- **R3.5 sparse + decorrelated cortex→MSN (`1521a9b`).** Bolam-2000 / Kincaid 1998: density 1.0 was anatomically dense. Same-action density 1.0 → 0.20; cross-action density 1.0 → 0.10. Synapse-count smoke drop: 39172 → 31113.
- **R3.10 SNr→SNc disinhibition (`dfa9d15`).** PBR-160 ch 11 Tepper & Lee: SNr collaterals onto SNc; the major in-vivo DA-burst driver. Adds 4 gpi_X → dopamine pathways (density 0.3, weight 2.0).
- **R3.7 GPe PV+/PV− split (`b359bb1`).** Mallet 2008 / Kita 2007: GPe is heterogeneous. New gpe_arky_X (4-neuron arkypallidal subpool); D2 drives both subpools; arky → all str_FS_Y when FSIs enabled (Mallet 2012 stop-signal; 16 broadcast pathways). 3 new tests.
- **R3.11 striosome (patch) / matrix split (`0e041e3`).** PBR-160 ch 9 Deniau: striosomes project to BOTH SNc and SNr; patch/matrix aligns with SNc/SNr at output level. New str_patch_X regions (8 D1-MSN-class neurons each, E_inh −60 mV via R1.1); cortex_X → patch (limbic placeholder); patch → dopamine (canonical SNc); patch → gpi (R3.11 SNr arm). 3 new tests.
- **R3.8 GPi/SNr pacemaker channels (`35f1908`).** PBR-160 ch 9 Deniau: SNr 40-80 Hz tonic pacemaker rests on NaP + SK (firing precision; we use M-current as AHP analogue) + Ih (slow Ca spikes). Tuned HH_GPI_OUTPUT preset: g_NaP_max 0.12 → 0.4; g_h_max 0.05 → 0.15; g_M_max 0.4 → 1.0. Affects HH-mode users; runner default unchanged.
- **R2.3 striatal interneuron taxonomy doc + R3.12 CA3 SWR framing (`8461a03`).** R2.3 (TK-2017 / Tepper-2018): clarified in CLAUDE.md that --enable-striatal-fsis models PV-FSI specifically, one of 8 distinct striatal GABAergic interneuron classes (non-isomorphic to cortex). R3.12 (Bz Cycle 12 / Leinekugel 2002): forward-looking design note — future SWR / replay must place generator inside CA3 intrinsic dynamics with NREM as passive gate, not a sleep-stage scheduler.
- **R3.6 D1/D2 neuropeptide arms (`bdb6452`).** PBR-160 ch 16 McGinty: D1 → dynorphin (KOR plasticity-rate brake) + substance P (NK-1 ACh boost); D2 → enkephalin (DOR plasticity-rate boost). Added new ProductionRule type "from_region_firing" (reads firing-fraction EMA across source_regions) + 3 default neuropeptide configs + `--enable-bg-neuropeptides` CLI flag. 39 neuromod tests pass.
- **R2.4 asymmetric aversive reward magnitude (`23b38fc`).** Schultz98 / Schultz16 / Fiorillo 2013: phasic DA aversive "activations" reflect physical-impact artifacts; underlying valence response is a depression below tonic, smaller magnitude than appetitive activations. Added `cfg.reward_aversive_scale: float = 0.5` (default 0.5 reflects observed ~50% magnitude); negative reward_prediction_error scaled by this factor before applying. Tunable per-experiment. 7 determinism tests pass.
- **R3.9 MSN KIR2/Kv2 (`befc1d0`, design-doc deferral).** PBR-160 ch 6 Wilson: biological MSN bistability rests on KIR2 + Kv-1.2/Kv-2.1 dual currents producing 6× input-resistance peak at -60 mV. Existing Izh `b=-20` approximates KIR2 but doesn't capture the IR-peak feature. Single largest deferral in pass — requires new GPU kernel work (~1-2 days). Catalog ref + design sketch + integration plan documented in `2026-04-29-catalog-remediation-pass.md`.

Final region/synapse counts (flagship smoke + Cluster B): 42 regions (was 30), 758 neurons (was 710), ~32K synapses (was ~41K — sparser cortex per Bolam). All 340 tests pass post-remediation.

## [Unreleased] — 2026-04-28 — Cheat #5 ON HOLD pending biology buildout (reframed) + throughput investigation + webapp polish

### Added
- **Cheat #5 v3 (`--bg-lateral-inhibition`) — GO, permanent default.** MSN cross-pool lateral inhibition: 24 GABAergic pathways between striatal action pools (D1↔D1' and D2↔D2' for X≠Y), `plastic=False`. The missing winner-take-all biology of the BG cascade. 6-seed sum 4.26 ± 0.50 (no regression vs flagship 4.08); P1 (1.91) actually beats P0 (2.35) — readaptation improved. Added to recommended flagship config in CLAUDE.md, README.md, QUICKSTART.md, SCIENCE_ROADMAP.md. Finding: `research/findings/2026-04-28-cheat5-v3-results.md`.
- **Cheat #5 v3.1 (cross-projections layered on v3 lateral inhibition) — NO-GO.** 6-seed sum 8.92 ± 2.44; P1=6.35 (2.5× P0). Phase-2 readaptation breaks even with proper lateral inhibition.
- **Cheat #5 v4 (`--developmental-pretraining`) — NO-GO under single-goal eval.** 3-seed Tier 2 sum 11.34 ± 1.85, P0 4.88, P1 6.46. Originally framed as "closed by design"; reframed later same day after the multi-goal eval correction + option 1 NO-GO + patch-matrix HIGH-VARIANCE PARTIAL signal. Finding: `research/findings/2026-04-28-cheat5-v4-results.md`.
- **Cheat #5 multi-goal eval correction (afternoon).** All prior cheat-5 NO-GO calls used single-goal scenario (one transition at step 300 + 1500 stable steps) — a "static adult" test. Multi-goal (`--goal-schedule multi`, 4 phases × 450 steps, 3 transitions) is the proper test for cross-action coordination. Re-baselined under multi-goal: v3 baseline 7.08 ± 0.12 (n=3).
- **Cheat #5 option 1 (`--enable-structural-pruning`) under multi-goal — NO-GO.** Catastrophic 22.46 ± 4.84 (n=2; seed 42 hung). Structural pruning didn't reshape topology meaningfully.
- **Cheat #5 option 2 (`--cross-projection-density 0.25`) under multi-goal — HIGH-VARIANCE PARTIAL.** 3-seed mean 8.76 ± 2.54, seed 44 actually beat baseline at 5.88. Phase 2 (1,6→1,1) shows topology-luck signal — std 2.09 across seeds vs 0.22-0.46 on others. Sparse cross-projections aren't fundamentally broken, just under-constrained without surrounding biology to consistently select useful pairs.
- **Cheat #5 reframe: ON HOLD pending biology buildout, not closed by design.** Cross-projections need a complete striatal microcircuit (D1/D2 asymmetry, FSIs, TANs), a closed BG loop (thalamo-cortical feedback, hyperdirect pathway), and a properly-structured DA system to behaviorally pay off. Building these out is a multi-month research program organized cluster-by-cluster. Finding: `research/findings/2026-04-28-cheat5-post-v4-reframe.md`. Strategy: `docs/plans/2026-04-28-cheat5-real-options-survey.md`.
- **Reference textbook directory + gitignore.** New `references/textbooks/` directory for source material (Kandel 6e PNS at `references/textbooks/kandel-pns-6e/`). PDFs gitignored (large binaries; only extracted catalogs are committed). Parallel session will build a comprehensive feature catalog from the textbook to drive cluster prioritization.
- **Cluster B.1 (`--enable-d1-d2-asymmetry`) — PARTIAL SIGNAL.** First piece of empirical evidence supporting the cluster-buildout strategy. D1/D2 plasticity asymmetry implemented (per-synapse `cp_d1_d2_sign` array, +1 default, -1 for D2-targeting; multiplicative on the reward-modulated weight update at sim/bridge.py:4309). Biology probe PASS (D1↑/D2↓ under +reward; inverted under -reward; magnitudes match closed-form expectation). Cheat-5 multi-goal: patch-matrix + B.1 = 7.62 ± 1.23 (n=3) vs patch-matrix alone 8.76 ± 2.54. Variance halved, Phase 2 catastrophe eliminated (P2 mean 3.36 → 1.92, std 2.09 → 0.77). Still 7% above v3 baseline 7.08; cheat-5 not fully closed by B.1 alone. Continuing to Cluster B.2 (striatal FSIs) + B.3 (cholinergic interneurons / TANs). 6 unit tests + 1 runner kwarg test + biology probe + 2 cheat-5 batches (n=3 each) shipped. Finding: `research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`.
- **`--enable-d1-d2-asymmetry` flag, `cp_d1_d2_sign` GPU array, `enable_d1_d2_asymmetry` cfg field.** New plumbing for Cluster B.1.
- **`research/probes/` directory + first probe (D1/D2 asymmetry).** Standalone validation scripts for biological correctness, distinct from pytest-based behavior tests. The probe is reproducible (`python -m research.probes.d1_d2_asymmetry_probe`) and writes JSON output for downstream tooling.
- **Cluster B.2 (`--enable-striatal-fsis`) — MIXED.** Adds 4 `str_FS_{N,E,S,W}` regions (5 PV-positive FSIs each, IZH2007_FS_CORTICAL_INTERNEURON preset, exc_fraction=0.0) + 4 cortex→FS excitatory pathways (weight 30) + 32 FS→MSN broadcast inhibitory pathways (weight 2.0 retuned from 8.0 after over-suppression diagnosis). Plastic=False on all FS pathways (static gating). Biology probe PASS (FSI broadcast inhibition observed: str_D1_N peak rate 36.4 → 23.6 Hz with FSIs engaged at 16 Hz). Cheat-5 multi-goal: patch-matrix+B.1+B.2 = 8.44 ± 0.62 (n=3) — variance keeps halving (1.23 → 0.62) and P1+P2+P3 (4.72) beats v3 baseline (4.89), but Phase 0 broken (3.72 vs 1.83) because FSIs broadcast inhibition before agent commits to winner. Phase-0 issue is architectural — real FSIs have tonic baseline + burst dynamics + high-pass filtering our model lacks. Proceeding to B.3 per unit-cluster strategy. Finding: `research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md`. Probe: `research/probes/striatal_fsi_probe.py`.
- **`--enable-striatal-fsis` flag, `enable_striatal_fsis` kwarg, `cortex_to_str_fs_weight=30`, `str_fs_to_msn_weight=2.0`, `n_striatal_fs_per_action=5` defaults.** New plumbing for Cluster B.2.
- **Cluster B.3 (`--enable-tans`) — NULL on cheat-5, infrastructure shipped (2026-04-28 evening).** Cholinergic TANs implementation: `pause_on_reward` neuromodulator production rule (drives ACh concentration DOWN by `sensitivity*(|reward|-threshold)`); `plasticity_window_gate` target type (`clip(1-conc/baseline, 0, 1)`, multiplicative aggregation); `_default_acetylcholine_config()` helper (baseline=1.0, decay_tau_ms=500, sensitivity=-2.0); `--enable-tans` CLI flag; biology probe at `research/probes/tan_ach_probe.py` (PASS). 47 unit tests pass (15 in `tests/test_tans.py` including step-order regression, 39 in `tests/test_neuromodulators.py`, 6 in `tests/test_d1_d2_asymmetry.py`, 38 in `tests/test_g11_bg_runner_flags.py`). Cheat-5 multi-goal eval at n=3: TAN-on vs TAN-off statistically neutral (B.1+B.2 alone 18.02 vs +TANs 18.59; patch-matrix variants 15.18 vs 14.83). Why no-op: gate fires inside reward-modulation block which is skipped when reward=0 (between rewards); at reward steps `pause_on_reward` drops ACh → gate ≈ 1 (no suppression). Real TAN function requires tonic DA-driven plasticity for ACh to gate; our model has only phasic DA. Real win: bridge step-order bug fix — `manager.step()` now runs BEFORE reward modulation, correcting one-step lag for fast-dynamics modulators (regression test `test_single_pulse_reward_fires_plasticity_within_step` catches the bug pattern). Finding: `research/findings/2026-04-28-cluster-b3-tans-results.md`.
- **Bridge step-order fix (`59dc1fc`).** `self.neuromodulator_manager.step(self)` and `compute_plasticity_gate_values()` propagation moved from after the reward-modulation block to before it. Pre-fix: gate was read from previous step's NM concentration → one-step lag. Post-fix: this step's reward signal drives this step's NM concentration changes (e.g., `pause_on_reward` → ACh pause → gate opens) AT THE SAME STEP. Required for fast-dynamics gates; harmless for slow-dynamics modulators (DA tonic, NE) which read at synaptic-conductance/excitability sections (still one-step-lagged). New regression test catches the pre-fix bug pattern; existing test_bridge_blocks_reward_weight_updates_when_ach_at_baseline updated to use a no-production-rule ACh config so the gate mechanic is tested in isolation.
- **Methodology finding: multi-goal benchmark regressed at seed 42** (2026-04-28 evening). v3 baseline 7.08 → 12.05 in current code; B.1+B.2 9.50 → 22.03; patch-matrix+B.1+B.2 8.44 → 18.87. Phase 3 (after 3 transitions) shows the dominant regression. Bisect at pre-B.3 commit `714bc29` reproduces 21.22 for B.1+B.2 — predates B.3 changes. Future cluster work should use fresh current-code baselines, not historical numbers.
- **`--developmental-pretraining`, `--pretraining-n-goals`, `--pretraining-steps-per-goal` flags + `_run_pretraining_phase` helper.** v4 implementation (8 commits, TDD, 30/30 pretraining/regression tests pass). Kept opt-in for future experiments (e.g., pretraining other pathways with structural plasticity).
- **GPU throughput investigation** — concurrency sweep, MPS daemon ruled out (Linux-only on RTX 3090/Windows host), motor-counting code fix REVERTED (no measurable improvement, n=1 showed -15%). Concurrency knee at 4-6 (4× hits 76% of 10× aggregate at 1.7× per-run speed). Finding: `research/findings/2026-04-28-throughput-investigation.md`.
- **Webapp polish (UX pass)** — live mode toggle, top-bar layout, font consistency via `--font-sans`/`--font-mono` CSS vars, collapsible HUDs, runs page no-flicker, goal-change dots on live chart, no-cache static asset serving, Windows DETACHED_PROCESS subprocess detach so closing dashboard doesn't kill running launcher subprocesses.

### Changed
- **Webapp default `--progress-print-interval`** changed from `1` (always-on for live-viz) to `20` for non-interactive presets. `interactive_*` presets keep `1` for live attach. Reduces stdout pressure during background batches.
- **Recommended flagship config now includes `--bg-lateral-inhibition`** by default. Backward compatible — flag is opt-in, off by default at the runner level, but the recommended/documented recipe ships with it on.

### Negative results (kept opt-in)
- **Cheat #5 v1 (curriculum-staged cross-projections)** — 3-seed mean 10.87. Plasticity gate freezes weight updates but not synaptic transmission; non-zero cross-projection weights disrupted BG disinhibition from step 0. Finding: `research/findings/2026-04-28-cheat5-v1-NEGATIVE.md`.
- **Cheat #5 v2 (zero-init cross-projections)** — 3-seed mean 7.89. Zero-init fixed the structural-damage failure mode (P0=2.49, intact), but exposed a learning-dynamics failure mode (P1=5.40): thaw-time STDP corrupts the converged policy. Diagnosis pointed at missing MSN lateral inhibition → led to v3. Finding: `research/findings/2026-04-28-cheat5-v2-NEGATIVE.md`.

## [Unreleased] — 2026-04-27/28 — NEW BEST: 4 of 5 cheats closed + Phase C + Item 1

### Added
- **🎉🎉🎉🎉 NEW BEST CONFIGURATION (2026-04-27/28 overnight) — 4 of 5 cheats closed, biology-grounded BEATS cheats-allowed**
  - **Sensed reward** (cheat #4 closed): reward computed from beacon-intensity gradient instead of ground-truth Manhattan distance. Real animals don't have access to ground-truth distances — they sense whether a cue is getting stronger or weaker. `--sensed-reward` flag.
  - **Result: sum 4.08, 6/6 seeds beat baseline 5.88, p=0.00045, 30.6% improvement.** Biology-grounded version (4.08) is *better* than cheats-allowed (4.41) — closing perception/reward cheats actually *helps* learning quality.
  - **Cheat #5 (BG cross-projections) tested — NEGATIVE.** Learnable cortex_X → str_D1_Y all-to-all broke phase-1 readaptation (3-seed avg 8.40, much worse). Phase-0 cortex_N/E activations reinforce cross-projections to all D1 pools, locking in motor bias the agent can't unlearn. Kept opt-in (`--bg-cross-projections`) for future experiments.
  - **Final cheats inventory:** 4 of 5 perception/reward cheats now closed. Only structural BG connectivity remains, plus minor items (discrete N/E/S/W actions, discrete time steps).
  - Finding: `research/findings/2026-04-27-NEW-BEST-4cheats-closed.md`
  - Recipe: `g11_bg_runner.py --moving-goal --hippocampus --learned-perception --pfc --beacon-perception --beacon-replaces-goal --cue-reflex --cue-reflex-replaces-heuristic --landmarks --landmarks-replace-place --sensed-reward --adaptive-da --adaptive-da-ema-decay-negative 0.7 --curriculum --curriculum-warmup-steps 600 --seed N --n-steps 1800`

- **🎉🎉🎉 Item 1 PERCEPTION ARC COMPLETE (2026-04-27 night)** — agent navigates from PERCEIVED sensory information; ALL major coordinate cheats closed
  - **Stage 1: Goal-beacon perception** — replaces direct (gx, gy) goal cell access with 8 directional sensors detecting beacon strength × cosine alignment. Plastic beacon → goal_cells pathway (curriculum-gated). 6-seed: 5/6 beat baseline (5.36 vs 5.88, p=0.34).
  - **Stage 3: Cue-following reflex** — replaces the heuristic with non-plastic reflex computing cortex drive from direction-normalized beacon sensor pattern. Models innate phototaxis-like wiring. Combined with Stage 1: 6/6 seeds beat baseline (4.77 vs 5.88, **p=0.00188**, 18.9% improvement).
  - **Stage 2: Landmark-based place cell self-organization** — fixed-position landmark (default at grid center) with 8 directional sensors + plastic landmark → place_cells pathway. Replaces direct (x, y) place cell access. Combined with Stage 1+3: **6/6 seeds beat baseline (4.56 vs 5.88, p=0.00819, 22.4% improvement)**.
  - **Final state**: agent has NO direct (gx, gy) AND NO direct (x, y) AND NO heuristic. Only 3% behind cheats-allowed best (4.41) — closing all coordinate cheats costs almost nothing.
  - Findings: `research/findings/2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md`, `2026-04-27-stage3-full-perception-BREAKTHROUGH.md`, `2026-04-27-stage1-beacon-perception.md`, `2026-04-27-perception-cheats-investigation.md`
  - Plan: `docs/plans/2026-04-27-perception-arc-plan.md` (executed in single session)

- **PFC working memory region (Item 3, 2026-04-27)** — recurrent prefrontal cortex for working memory dynamics. 60 neurons, internal_density=0.2, plastic recurrent. Pathways: `goal_cells → PFC → cortex_{N,E,S,W}`. 6-seed: 5/6 beat baseline (4.41 vs 5.88, p=0.018, 25% improvement).

- **Per-pathway plasticity gating (Phase C, 2026-04-27)** — biologically-grounded staged plasticity
  - `RegionPathway.plasticity_gate: str | None` field tags pathways
  - `cp_plasticity_gain` per-synapse array gates STDP/eligibility/Hebbian/synaptic-scaling
  - Bridge methods: `set_plasticity_gate(name, value)`, `get_plasticity_gate_value()`, `list_plasticity_gates()`
  - **NM-driven gates**: `target_type="plasticity_gate"` with `scope="gate:<name>"` lets NM concentrations drive gates
  - 8 unit tests for gating semantics; 1 test for NM-driven gates
  - Closed the 7-NEGATIVE plastic-input-layer arc that ran 2026-04-26

- **Real curriculum learning** — phase 1 corticostriatal plastic + input layers frozen; phase 2 cortex frozen + input layers plastic. Configurable warmup steps, smooth ramping, partial-freeze gain. (Gate renamed from `cortex_to_d1` to `corticostriatal` 2026-04-29; old name aliased with deprecation warning for one release cycle.)

- **Sleep-replay infrastructure** — NREM trajectory replay (logged successful (place, goal) tuples) + REM random replay alternation. Mechanism works; current task structure doesn't reward consolidation.

- **Spatial scaling** — `--grid-size`, `--n-hippocampus-per-layer` for arbitrary grid sizes. Architecture scales to 16×16; recipe needs re-tuning for larger grids.

- **g11_bg_runner CLI growth** — many opt-in flags: `--curriculum`, `--curriculum-warmup-steps`, `--curriculum-ramp-steps`, `--curriculum-phase2-cortex-gain`, `--pfc`, `--n-pfc`, `--beacon-perception`, `--beacon-replaces-goal`, `--cue-reflex`, `--cue-reflex-replaces-heuristic`, `--landmarks`, `--landmarks-replace-place`, `--sleep-replay-after-step`, `--sleep-nrem-rem-alternate`, `--goal-silence-after-step` (PFC delayed-response test), `--heuristic-decay-after-step` (heuristic-off validation)

- **TROUBLESHOOTING doc** (`research/runners/TROUBLESHOOTING.md`) — gotchas accumulated across sessions

### Changed
- **Recommended flagship config (current best, biology-grounded BEATS cheats-allowed)**:
  - Flagship: `--hippocampus --learned-perception --pfc --beacon-perception --beacon-replaces-goal --cue-reflex --cue-reflex-replaces-heuristic --landmarks --landmarks-replace-place --sensed-reward --adaptive-da --adaptive-da-ema-decay-negative 0.7 --curriculum --curriculum-warmup-steps 600` (**4.08, p=0.00045, 30.6% over baseline**)
  - Cheats-allowed (older): same minus beacon/reflex/landmarks/sensed-reward flags (4.41, p=0.018)
- QUICKSTART.md added — 60-second getting-started for new users
- CLAUDE.md, SCIENCE_ROADMAP.md, INDEX.md, README.md all reflect the new state

## [Unreleased] — 2026-04-25 — Phase A presets + Phase B BG action selection

### Added
- **Phase B: BG-style action selection cascade** — silent-motor trap resolved
  - `research/runners/g11_bg_runner.py` builds 30-region cascade: cortex → str_D1/str_D2 → GPi/GPe → STN → thalamus → motor with disinhibition gating
  - 3-seed acid test: phase 1 finalQ 1.76 avg vs G9 baseline 6.74 (74% improvement, agent stays at Manhattan distance ~1.7 from goal vs random walk's ~5.5)
  - Per-action populations replace shared reservoir + argmax — eliminates the dominant-motor bias that defeated 7 prior runner-side variants (V1–V7)
  - Findings: `research/findings/2026-04-25-phase-b-acid-test-real-win.md`, `2026-04-25-phase-b-cascade-stability-fix.md`, `2026-04-25-phase-b-honest-correction.md`
- **Phase A: comprehensive preset audit + retuning** (HH + Izh + AdEx — 30 working biological presets)
  - 4 new HH BG cell types: `HH_STRIATAL_MSN_D1`, `HH_STRIATAL_MSN_D2`, `HH_STRIATAL_TAN`, `HH_GPI_OUTPUT`
  - 8 new IZH2007 brain-region presets: `IZH2007_STRIATAL_MSN`, `IZH2007_STRIATAL_MSN_D1/D2`, `IZH2007_STRIATAL_TAN`, `IZH2007_GPE_PACEMAKER`, `IZH2007_GPI_OUTPUT`, `IZH2007_STN_BURST`, `IZH2007_THALAMIC_RELAY`, `IZH2007_THALAMIC_RETICULAR`, `IZH2007_HIPPO_PYRAMIDAL`, `IZH2007_DOPAMINE`
  - Full AdEx preset library (`DefaultAdExParamsManager`): RS, FS, IB, CH, LTS, MSN, DOPAMINE — all 7 fire at 37°C with biological rates
  - Per-region neuron type override on `BrainRegion`: `izh_neuron_type`, `hh_neuron_type`, `adex_neuron_type` (independent of global default)
  - Findings: `2026-04-25-hh-preset-audit.md`, `2026-04-25-izh-preset-audit.md`, `2026-04-25-hh-presets-after-q10-fix.md`
- **Per-gate Q10 temperature scaling** for Hodgkin–Huxley (`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`)
  - Replaces uniform Q10=3 that over-compressed gating dynamics at 37°C
  - HH model now produces action potentials at body temperature
  - Finding: `2026-04-25-hh-temperature-bug.md`

### Fixed
- **STDP soft-bound w_max collapse** — when synapse `weight_mean > stdp_w_max`, every "LTP" event is strongly negative (`Δw = A_plus * (w_max - w) * exp(...)`), collapsing weights to w_max within milliseconds. Set `cfg.stdp_w_max` above design weights (e.g. cortex→D1 `weight_mean=25` needs `stdp_w_max=30`). Documented in CLAUDE.md and Phase B findings; runners now set it explicitly.
- **n_cortex saturation in BG cascade** — over-driving D1 above ~150 Hz puts MSNs into refractory dominance and breaks D1→GPi inhibition. Probes must use the same `n_cortex` value as deployment. (`research/runners/g11_bg_runner.py` now uses `n_cortex=100` matching the static probe.)
- **Izhikevich preset wasn't applied** — bridge always trait-split via `traits % num_variants`; now opt-in only when `cfg.num_traits > 1`. `cfg.default_neuron_type_izh` now respected when single-type is intended.
- **AdEx presets all behaved identically** — bridge wasn't loading preset params into `cfg.adex_*` fields. Now overlays preset onto config before initialization.
- **GPE/STN didn't fire** — `g_NaP=0.8` was 5–10× too high for these cell types. Retuned in `HH_GPE_PACEMAKER` and `HH_STN_BURST` (commit `9f4c3f7`).

## [Unreleased] — 2026-04-24 — Brain-region framework + neuromodulator subsystem + Route C performance

### Added
- **Brain-region framework** (Session E.2, opt-in)
  - `sim/regions.py` — `BrainRegion`, `RegionPathway`, `RegionManager` dataclasses
  - Declarative multi-region simulations (PFC + Motor + Hippocampus + Striatum on one bridge)
  - Each region owns a contiguous neuron-index slice with its own internal connectivity
  - Cross-region pathways declared with density, weight_mean, plasticity flag, and optional neuromodulator gates
  - `cfg.enable_brain_region_framework=True` opt-in; default OFF for backward compatibility
  - Bridge integration: regions allocated before neuron arrays (auto-sets `num_neurons`); wiring fed through `inject_explicit_wiring()` replacing legacy motif/WS/spatial paths
  - Composes with neuromodulator subsystem — regions auto-register as NM groups
  - Plan: `docs/plans/2026-04-24-brain-region-framework.md`; tests: `tests/test_regions.py`
- **Neuromodulator subsystem** (Session E.1, opt-in)
  - `sim/neuromodulators.py` — `NeuromodulatorConfig`, `ModulatorTarget`, `ProductionRule`, `NeuromodulatorManager`
  - Declarative concentration dynamics for DA / NE / 5-HT / etc.
  - Built-in target types: `synaptic_gain`, `plasticity_rate`, `excitability_drive`
  - Built-in production rules: `manual`, `from_reward`, `from_error_persistence`
  - Replaces ad-hoc `current_reward_signal` and shelved `cp_synaptic_gain_modulator`
  - `cfg.enable_neuromodulator_subsystem=True` opt-in; default OFF
  - Plan: `docs/plans/2026-04-24-neuromodulator-subsystem.md`; tests: `tests/test_neuromodulators.py`
  - Finding: `research/findings/2026-04-24-session-e1-neuromodulator-subsystem.md` (framework GO, NE params NO-GO on silent-motor)
- **Route C: 101× synapse-update throughput** at 1.2× wall time (bigger networks performance)
  - Finding: `research/findings/2026-04-24-route-b-profile.md`
- **Module split** — extracted `sim/`, `viz/`, `ui/`, `experiment/` packages from monolithic `neural-simulator.py`
  - `sim/__init__.py` exposes public API (`SimulationBridge`, configs, `NeuronModel`, `NeuronType`)
  - `neural-simulator.py` reduced from ~12K lines to ~2.2K (now just GUI host)

## [Unreleased] — 2026-04-20/21 — Research-gate runner framework (G1–G6)

### Added
- **Research-gate runner framework** (`research/runners/`)
  - 16 headless runners (g1..g11) each invocable via `python -m research.runners.gN_runner`
  - Each writes raw data to `research/findings/raw/gN/` and a markdown finding to `research/findings/`
  - Negative results documented as findings, not failures
- **G1: encoder-decoder roundtrip** — GO (v3, 71.3% test acc, 3 seeds, threshold 55%)
  - `research/datasets/tiny_patterns.py` — K=4 Poisson-rate synthetic dataset
  - `RATE_VECTOR_POISSON` stimulus pattern type
  - `SimulationBridge.inject_explicit_wiring()` — injectable explicit CSR connectivity
  - Three runner variants explored; v3 (264-neuron reservoir + external LogReg) passes
  - Finding: `2026-04-20-g1.md`
- **G2: STDP local learning** — NO-GO (no epoch-over-epoch improvement on target task) — `2026-04-20-g2.md`
- **G3: persistence/checkpointing** — GO — `2026-04-20-g3.md`
- **G5: sensorimotor signed perceptron** — GO (v3 with LR decay, 3/3 seeds pass) — `2026-04-21-g5v3.md`, `2026-04-21-signed-eligibility-branch.md`
- **G6: 2D gridworld** — PARTIAL (gate metric needs redesign — agent converges too fast) — `2026-04-21-g6.md`, `2026-04-21-g7.md` (proposed metric replacements)

## [Unreleased] — Earlier (2026-04-06 baseline)

### Added
- **G1 pipeline GO** - First end-to-end dataset → encoder → sim → decoder → loss round-trip
  - `research/datasets/tiny_patterns.py` + canonical `.npz` - K=4 Poisson-rate synthetic dataset
  - `RATE_VECTOR_POISSON` stimulus pattern - per-neuron Poisson rate encoding
  - `SimulationBridge.inject_explicit_wiring()` - injectable explicit CSR connectivity for research networks
  - Three runners explored: v1 teacher-forced STDP (NO-GO), v2 external perceptron (NO-GO), v3 reservoir + external LogReg (**GO** - mean 71.3% test acc across 3 seeds, threshold 55%)
  - v1/v2 post-mortem: sim's default `propagation_strength=0.05` is calibrated for ~1000 converging synapses per neuron; the 68-neuron tiny architecture needs non-default params. v3 uses a 264-neuron reservoir in the sim's calibrated regime.
  - Full findings in `research/findings/2026-04-20-g1.md`

- **Profile System** - Biologically accurate brain region presets and UI integration
  - 9 brain region profile JSONs with realistic neuron models and connectivity: Cortex L2/3, Cortex L4, Hippocampus CA1, Hippocampus CA3, Thalamus TC-TRN, Basal Ganglia Striatum, Basal Ganglia STN-GPe, Cerebellar Cortex, Spinal Cord
  - Quick Demo profile for rapid testing
  - Full profile dropdown menu in UI that auto-populates from `simulation_profiles/*.json` files
  - Refresh button to reload profiles from disk without restarting

- **Plasticity Parameters in UI** - STDP, reward modulation, and structural plasticity
  - 28 new fields added to SimulationConfiguration for complete plasticity roundtrip (save/load)
  - Support for STDP timing windows, reward modulation learning rates, and structural synapse thresholds
  - Full persistence in simulation profiles and checkpoint files

- **Per-Connection-Type STP** - Biologically realistic short-term plasticity heterogeneity
  - New `enable_per_type_stp` parameter and per-type arrays `stp_U_per_type`, `stp_tau_d_per_type`, `stp_tau_f_per_type`
  - Each is indexed by connection type [E->E, E->I, I->E, I->I]
  - Different brain regions now use experimentally validated STP profiles per connection type
  - UI table exposes all 12 parameters (4 connection types × 3 STP variables)

- **Activity-Dependent Structural Synaptogenesis** - Cline & Haas 2008 model
  - New `struct_plast_activity_bias` parameter (0.0-1.0, default 0.5)
  - Biases new synapse formation toward co-active neuron pairs using activity EMA
  - 0 = random synapse formation; 1 = fully activity-driven (Hebbian structuring)

- **COO Cache Invalidation** - Fixed stale data handling in GPU memory
  - Cache invalidation in `clear_simulation_state_and_gpu_memory()` prevents stale sparse matrix data across reinitializations

### Fixed
- **STP/Connection Shape Mismatch at Scale** - CSR matrix deduplication bug
  - Fixed shape mismatch occurring at 100K+ neurons caused by CSR matrix addition deduplicating overlapping (pre,post) pairs
  - Now uses `cp_connections.nnz` as authoritative size instead of stale shape values
  - Structural plasticity synapse count now properly synced after CSR addition

- **Synaptic Scaling Crash** - Stale COO cache surviving reinitialization
  - COO cache no longer persists across simulation reinitializations, preventing crashes when synaptic scaling is active

- **Unicode Handling on Windows** - JSON I/O encoding issue
  - UnicodeDecodeError on Windows (cp1252) when loading profile JSONs with Unicode characters (em dashes, etc.)
  - All JSON I/O now uses UTF-8 encoding explicitly

- **Em Dash Rendering** - DearPyGui font limitation
  - Em dashes rendered as question marks in DearPyGUI default font
  - Replaced with regular hyphens in all UI text and profile names

### Changed
- **Hodgkin-Huxley Numerical Stability** - Automatic time step adjustment
  - dt automatically reduces to 0.05ms when switching to HH model for improved numerical stability
  - dt automatically restores to 0.5ms when switching away from HH model
  - Prevents instabilities in voltage-gated kinetics at larger time steps

- **Homeostatic Plasticity Timescale** - Biologically realistic adaptation
  - EMA alpha reduced from 0.01 (tau ~100ms) to 0.0002 (tau ~5s at dt=1ms)
  - Threshold adapt rate reduced from 0.015 to 0.0005
  - Homeostatic mechanisms now operate on seconds-to-minutes timescale, matching experimental observations

- **Inhibitory Reversal Potential** - Corrected Nernst equilibrium
  - E_inh changed from -70mV to -75mV (matches Cl- Nernst potential at 37°C)
  - Inhibitory propagation strength scaled by 0.7 to compensate for increased driving force
  - Improves accuracy of GABAergic synaptic transmission

- **.gitignore** - Profile tracking and auto-tuning separation
  - Now tracks `simulation_profiles/*.json` to include biologically accurate presets in repository
  - Excludes `auto_tuned_overrides.json` to prevent auto-tuned parameters overwriting checked-in profiles

- **Profile Files** - Superseded files removed
  - Removed 7 old profile files replaced by new standardized brain region profiles

- **System Logs Panel** - Comprehensive log viewing and management
  - Real-time display of all console output within the GUI
  - Auto-scroll functionality using DearPyGUI's `tracked` parameter
  - Search functionality with previous/next navigation through matches
  - Export logs to timestamped text files
  - Clear logs functionality
  - Thread-safe `LogCapture` class for zero-overhead console mirroring
  
- **Performance Test Controls**
  - Stop button for halting running benchmarks and auto-tuning mid-execution
  - Proper state tracking to preserve existing result files
  - Informative logging showing which test type was stopped
  - Located above "Reload Auto-Tuned Overrides" button in GUI

### Changed
- **VRAM Utilization for Initialization** - Increased chunking from 40% to 70% of free VRAM
  - ~2x faster initialization for networks with 50K+ neurons
  - Example: With 18GB free VRAM, now uses 12.6GB instead of 7.2GB for chunking
  - Maintains 30% safety margin for stability
  
- **GUI Layout Improvements**
  - Auto-tuning button now stretches to fill available width (width=-80)
  - Better space utilization when window is resized wider
  - "Quick" checkbox properly positioned at right edge

### Fixed
- Auto-scroll in System Logs now works correctly using DPG best practices
  - Replaced manual scroll manipulation with `tracked=True` and `track_offset=1.0`
  - Dynamic height adjustment based on text size for proper scrolling
  - Toggle auto-scroll on/off via checkbox callback
  
- Performance test stop functionality prevents corrupted result files
  - Benchmark and auto-tuning only save results at completion
  - Stopping mid-run preserves any previously existing result files

### Technical Details
- Log capture uses thread-safe deque with 5000-line rolling buffer
- System logs display widget uses `child_window` with `input_text` for proper scrolling
- Auto-scroll implementation follows official DearPyGUI documentation patterns
- Stop flags (`performance_test_stop_flag`, `performance_test_running_type`) properly managed in try/finally blocks

## [Previous Versions]

See git commit history for details on earlier changes.
