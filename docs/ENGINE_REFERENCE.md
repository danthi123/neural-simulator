# Engine reference — architecture, config traps, backends

**Split out of CLAUDE.md 2026-07-31.** It was 247 lines of a 494-line file loaded in FULL every session (~12K tokens), and most of it is needed only when touching the specific subsystem it describes. Worse, its load-bearing parts are now ENFORCED rather than remembered — the plasticity bound trap by `tools.lab.bound_check` (it raises), and the seed trap by `tests/test_determinism.py::TestSubstrateActuallySeeded`. Prose that duplicates a check costs context and adds nothing.

RAG-indexed: `.venv-rag/bin/python tools/rag/rag_search.py "<question>" --corpus doc`.

## Architecture


### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Config gotchas + the plasticity BOUND TRAPS (`sim/config.py`)

  - Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation
  - HH numerical stability: dt auto-adjusts to 0.05ms when HH model selected
  - **Per-gate Q10**: `hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5` (fixed 2026-04-25 — uniform Q10=3 over-compressed dynamics at 37°C; see Phase A below)
  - **STDP bounds gotcha**: `stdp_w_max=2.0` default. The STDP rule is **soft-bound** (`Δw_LTP = A_plus * (w_max - w) * exp(...)`) so when `weight_mean > stdp_w_max`, every "LTP" event is strongly negative and weights collapse to w_max within ms. Set `cfg.stdp_w_max` above your design weights (e.g. cortex→D1 in Phase B uses `weight_mean=25` → set `stdp_w_max=30`).
    **⚠️ THIS TRAP IS PER-RULE AND HAS NOW HIT FOUR RULES — STDP (`stdp_w_max`), BDSP (`bdsp_w_max`, below), BTSP (`btsp_w_max` — saturation silently crushed a rank-1 write to a flat null, 2026-07-25) and HEBBIAN (`hebbian_max_weight` **defaults to 1.0**, far below typical design weights: at a 3.015 pathway every "potentiation" was strongly negative and collapsed the TRAINED and UNTRAINED pathways identically, reading as "the rule doesn't help here").
    **⛔ FIFTH INSTANCE, 2026-07-31 — and the pre-flight below existed the whole time, as PROSE, so it was skipped again.** gap#5's *tuned* operating point ran `w_max=150` against an initial weight `W0=250`: the clamp sat BELOW the weights, dragged every one DOWN, and **97% of the measured weight change was the clamp — identical in the `lr=0` control** (the lr lever moved 3%). The tuning then walked DEEPER in (`w_max` 110→150→220, 150 picked as the "interior optimum"), because what the metric rewarded was clamp depth. Compounding it, `circ_resultant` RECTIFIES, so the `lr=0` arm scored `circ_dW` **exactly 0.000000** at every seed and was quoted as a clean control — while its own mean `|dW|` was **21.94**. An exact zero meant every increment was NEGATIVE, not absent.
    **⇒ THE PRE-FLIGHT NOW EXECUTES — use it instead of remembering this:** `from tools.lab import bound_check, sign_budget`. `bound_check("btsp_w_max", cfg.btsp_w_max, W0)` RAISES when a bound sits at/below its weights; `sign_budget(label, dW)` reports what fraction of `|dW|` a rectifying metric is about to discard. Both are wired into the gap#5 runner and tested in both directions.
    **STANDING PRE-FLIGHT for ANY plasticity rule: compare its bound against the ACTUAL weight (`_mean_gate_weight(bridge, gate)` vs `cfg.<rule>_max_weight`), and verify the trained pathway moves DIFFERENTLY from an untrained control.** A bound below the weights does not merely fail to learn — it destroys weights uniformly, which reads as a substrate limitation.
  - **BDSP clamp-at-lr=0 gotcha** (2026-07-24, commit 6a9a44c3): `fused_bdsp_update` applies `cp.clip(w, bdsp_w_min=-5, bdsp_w_max=5)` **unconditionally — even at `lr=0`** (a frozen/control arm), so any weight outside ±5 is silently flattened to the bound (it collapsed a gap#5 encode store to `bdsp_w_max=5` and plausibly caps gap#4's ±5-bounded FF weights on a 9-way task). Set `bdsp_w_max` above your design weights, and don't assume `lr=0` means "no weight change" for BDSP. A `sim/` clamp-fix (gate the clip by lr / plasticity gain, mirroring the STDP masked-clip) is filed.

**Note on dt Auto-Adjustment**: When switching to Hodgkin–Huxley model, dt is automatically
reduced to 0.05ms for numerical stability of voltage-gated kinetics. When switching to Izhikevich
or AdEx, dt restores to 0.5ms. This occurs in `apply_simulation_configuration_core()`.


### UI-Config Roundtrip
Two critical functions must be kept in sync for profile save/load to work correctly:
- `_update_sim_config_from_ui()`: Extracts all parameter values from UI widgets and builds `CoreSimConfig`, `VisualizationConfig`, `RuntimeState`, and `GPUConfig` dataclasses
- `_populate_ui_from_config_dict()`: Takes a configuration dictionary and updates all UI widgets to reflect those values

These are inverse operations: any parameter exposed in the UI must have a corresponding getter and setter to ensure bidirectional sync between UI state and simulation configuration.

**Built-in target types:**
- `synaptic_gain` — multiplies effective synaptic strength (scope=all only)
- `plasticity_rate` — multiplies reward_learning_rate (scope=all)
- `excitability_drive` — adds pA to membrane drive (scope=all, trait:N, group:NAME)
**Group registration:**
Runners that want `scope="group:NAME"` targets must call
`bridge.neuromodulator_manager.set_group_indices({name: indices})`
after the engine groups are known. G9 runner does this automatically
for the standard input/hidden/motor groups.
- Bridge allocates `region_manager` BEFORE neuron arrays (so num_neurons
  is set from `region_manager.total_neurons()`).
- Wiring is generated by `build_wiring_plan()` and fed through
**Purpose:** Defeats the silent-motor trap (motor neurons that never fire in
phase 1 cannot acquire STDP eligibility, so reward-mediated weight updates
never reach them; agent stays glued to phase-1 winners even when reward
flips sign).
**Two non-obvious bugs that almost killed the architecture** (both fixed 2026-04-25):
1. `n_cortex=400` over-drove D1 to ~220 Hz (saturated, unphysiological), GPi couldn't silence past STN excitation. **Fix:** use `n_cortex=100` (25 cortex/action). The static probe used 100; the moving-goal runner shipped with 400, so the probe "passed" but the deployment failed. Lesson: probes must call the same builder with the same args as deployment.
2. `cortex→D1` weight_mean=25 against default `stdp_w_max=2` collapsed weights from 25→2 in milliseconds via soft-bound STDP. **Fix:** set `cfg.stdp_w_max = 30.0` in the runner.
> **GOTCHA — plasticity gate vs synaptic transmission (2026-04-28):**
> `cp_plasticity_rate_gain` and `set_plasticity_gate(...)` freeze weight UPDATES
> only — STDP, eligibility, Hebbian, synaptic scaling. They do NOT freeze
> synaptic CURRENT (`g_syn × (V - E)`). A frozen pathway with non-zero
> `weight_mean` still injects current and affects forward dynamics. To
> staged-introduce a new pathway without disrupting the system before
> the thaw step, initialize it with `weight_mean=0.0` (then let STDP grow
> it from zero after thaw) — OR add a runtime weight scale per gate
> (small bridge change, not yet implemented). The cheat-5 v1 NEGATIVE
> result (2026-04-28) was caused by missing this distinction; v2 fixes
> it via zero-init.
>
> **UPDATE (2026-06-03): the complement now EXISTS — `transmission_gate`.**
> `RegionPathway(transmission_gate="name")` + `bridge.set_transmission_gate(name, value)`
> scale a pathway's effective synaptic **CURRENT** in [0,1] at runtime
> (the `cp_transmission_gain` per-synapse multiplier in `_run_one_simulation_step`,
> mirroring `cp_plasticity_rate_gain` but on current, not weight updates).
> Pre-wire a route with a fixed weight, hold it CLOSED (gate=0, no current,
> no STDP cold-start), OPEN it on command → **thalamocortical dynamical
> gating**: binding = which gate is open, not which weight grew
> (Logiaco-Abbott-Escola 2021). Validated in spikes
> (`tests/test_transmission_gate.py`): closed → target silent; open → target
> fires; re-binding reroutes the same source with **zero weight change**,
> where grown weights could not. Default `None` = always-on (additive, zero
> overhead unused). See `2026-06-03-deep-research-surpassing-the-blockers-synthesis.md`.
**Usage:**
```bash
# Default (CuPy if available, else NumPy)
python -m research.runners.chat_repl --mode tier1 --seed 42

# Force NumPy backend (Mac M-series, GPU-less Linux, CI)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 --seed 42

# Force CuPy explicitly (or fail if unavailable)
SIM_BACKEND=cupy python -m research.runners.chat_repl --mode tier1 --seed 42
```
**Pattern for new code:** instead of `import cupy as cp`, use:

```python
from sim.backend import get_backend, fuse, synchronize, to_host
xp, backend_name = get_backend()

@fuse()
def my_kernel(a, b):
    return a + b  # works on both cupy + numpy backends
```
**Backend selection** (in priority order):
1. Explicit `get_backend("cupy")` or `get_backend("numpy")` (test code)
2. `SIM_BACKEND` env var (`cupy` / `numpy` / `auto`)
3. Cached backend from a prior call (sticky)
4. Auto-detect: CuPy if installed AND `cp.cuda.runtime.getDeviceCount() > 0`,
   else NumPy
and track per-pathway activity each simulation step. Inference still
uses the monolithic `cp_connections`; the store is observational +
foundation for Phase 4 auto-tiering. Per-pathway shards can be
**Known limitation — composer is a principled idealization, not a functional cortex (2026-06-06):** the
FHRR/VSA composer is a *principled idealization* (Eliasmith Spaun / Semantic Pointer Architecture — a
serious hypothesis that cortex binds VSA-like), NOT a functional reproduction of cortex. Its binding is
a clean, exactly-invertible ALGEBRA that DEMANDS decorrelated full-precision codes (the whole whitening
requirement is downstream of this); a real cortex has LEARNED, lossy, redundant read-outs that learn to
read whatever messy code arrives. The binding OPERATIONS are already on-substrate spiking (FHRR
resonate-and-fire + complex synapses); the residual idealization is the exact-inverse algebra + the
clean-code demand. The spike-native robustness ladder (a phase-encoded handoff, b temporal integration,
c population redundancy + attractor cleanup) makes the scaffold spike-FAITHFUL; the genuine-cortical
conversion (d: learned read-outs replacing the fixed algebra) is **BENCHED** below the planned work
(cheat/shortcut removal → single-brain consolidation → capability addition + scaling). NOT labelled a
"cheat," but stay cognizant it is not functionally identical to the cortex it stands in for. Trade-off:
the algebra buys the no-confab moat + compositional reliability ~free; a learned cortex does not.
See `research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md`.
production D=2048 multi-seed; a `plastic=False` population still drifts under global Hebbian, so the
composer's fixed bind population is frozen by a per-synapse plasticity gate, `cp_plasticity_rate_gain=0`).
**The two standalone numpy phasor simulators are REFERENCE-only, NOT the production substrate:**
`research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` (and the unified agents that import
them — `nested_composition_agent` / `spiking_unified_agent` / `unified_agent_*`) carry a NUMPY-REFERENCE
header and are retained only as the FHRR validation ceiling. Do not treat them as "the brain analogue."
- **De-risk 5b (RF vs Izhikevich) — KILL confirmed → the minimal protected edit.** RF stores its complex
  phasor in the same `v`/`u` arrays Izhikevich uses; one Izhikevich step destroys a phasor (|z| 1.0 → 16.3).
  But the composer is stateless-per-op (re-kicks each op) and stores memory in complex synapses, so the
  minimal edit is to **slice the RF ops** (not a core-step-loop dual-dispatch): `rf_kick(..., neuron_mask=)`
  + `_rf_advance_one` mask all `v`/`u` writes to the RF slice. **Default `None` = byte-identical** (18/18
  (a) uses a HYBRID `run_moving_goal_episode` integration (4 additive no-op-default params + an index-based
  `finalize_conv_for_nav_gate` hook that runs AFTER the V1/SC post-init `set_pathway_weights(add_missing=True)`
  CSR rebuild — which re-sorts the data + stales gate-index maps + the Hebbian decay would erode the fixed
  perception weights; the hook handles all three by masking by index, not gate name). The **nav-on-merged
    TEST ORACLE + the numpy-CPU path** (`--composer rf`). NOT flipped (deliberate, safe): the library constructor
    defaults (`BrainConversationalAgent`/`MultiTurnAgent` `composer_kind="rf"`) + the CPU transcript demo — flipping
    those would force GPU on every default agent and break numpy-CPU portability. The bind stays the exact-inverse FHRR
- **UPDATE (2026-06-15) — the GENERALIZING learned cortex is achievable WITHOUT the (B) dendritic rewrite,
  and is REALIZED on the spiking substrate, learned from the conversation stream.** The fork's (B) framing
  ("decorrelate the correlated codes → needs the dendritic rewrite") was superseded by the CYCLE-88 reframe:
  the off-diagonal decorrelation was a **red herring**. A generalizing cortex needs **feedforward LOCAL
  normalization** (PPMI = log + per-hub + per-concept mean-subtraction + threshold, all local ops), NOT
  cross-neuron decorrelation (which would *destroy* generalization). PPMI codes reach host (+0.518) AND
  multi-attribute **bundling** (a fact = a superposition of bindings) is **not learnable from scratch** on the
  point-neuron substrate — additive has no inverse (0.193), a learned *linear* inverse cannot be a reciprocal
  (0.056, breaks even single-attribute), while a **fixed ±1 self-inverse bind bundles 0.989** on the same
  harness (positive control). ⇒ the conversational bind = **learned representations** (codes + single-attribute
  binding, both substrate-validated) flowing through a **fixed, biology-grounded coincidence/multiplicative
  binding primitive** (= the production composer binding the learned codes; binding-by-coincidence /
  dendritic-multiplication is a STRUCTURAL neural primitive — not a host shortcut, and not learnable from
  scratch on point neurons). Finding:
  `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`.
`SIM_BACKEND=cupy` (GPU) is required for the merged-bridge runs (numpy is a tiny-smoke / CI path only).
### 🧠🔗 Cross-region "one brain" FUNCTIONAL interaction + step-3 COMPOSE-PERCEIVED-CONTENT de-risked (2026-06-16)

**Roadmap step 2 merged nav + conversation onto one bridge but they were CO-LOCATED, not interacting** (owner
challenge [[project_one_brain_substrate_vs_functional]]). The cross-region SYNAPTIC interaction (the real "one
- **correlation boundary mapped** (`_step3_correlated_percept_boundary.py`): the compose algebra TOLERATES code
  correlation up to code-sim **≈0.98** (the role-binding decorrelates the cross-terms). CAVEAT recorded: this is
  compose-ROBUSTNESS to correlation, **NOT** generalization-across-similar-concepts (the separate dendritic/PPMI
  job; "decorrelation is a red herring", CYCLE 88). "Algebra tolerates correlation" ≠ "correlation buys
  generalization."
**🧠⚡ The merged "one brain" nav action-decision is now FULLY-SPIKING by DEFAULT (2026-06-19, roadmap #4 default-on).**
Per the owner's brain-based-purity directive, `run_moving_goal_episode`'s LIBRARY defaults are flipped to the
validated spiking config — `readout_source="spiking_wta"`, `sel_recurrent_weight=0.3`, `n_sel_per_action=n_commit_per_action=40`,
`urgency_max_pA=180.0` — so the action EMERGES from the spiking competition (Wang-2002 accumulator + Lo-Wang
commit-burst threshold-crossing), the host Python argmax RETIRED. Validated 6-seed grid-32/1800 at **1.16× host
(within the 25% deploy bar), 100% commit-burst** (zero argmax fallback) — down from the CYCLE-216 ~1.7× boundary via
two levers (Usher-McClelland accumulator LEAK + finite-size-noise N-scaling; the ~16% residual = the irreducible
commit-timing/finite-size floor, the honest BRAIN-BASED-ONLY deliverable). **The CLI `--readout-source` default stays
`"motor"`** so every documented standalone benchmark reproduces unchanged; `motor`/`thal` = the opt-in host-argmax
ORACLE (the tuned levers are inert under them). NO `sim/` edit (runner-only default flip); the spiking read-out is

Graceful error handling: missing tag names + empty tags silently
skipped. Caller manages awake/sleep gate transitions.
```bash
python -m research.runners.validate_trisynaptic_loop \
    --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \
    --direct-ca3-drive \
    --out research/findings/raw/g11_bg/trisynaptic_seed42.json
```
Methodology note: EC-driven test (drive lang_input, propagate
through trisynaptic chain) FAILED at all parameter combinations.
DIRECT-CA3 test (drive partial of stored CA3 ensemble directly) is
the cleaner Marr autoassociator test and PASSES at train=400 +
ca3_recurrent_weight=5.0.

> _Archived: **Realigned plan** (was CLAUDE.md L1499-1523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Concept-pool v1->v17 architecture + engram-composition saga** (was CLAUDE.md L1524-2523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **160/320-concept G.20 sparse-distributed ensemble + 320 flat-distinct composition** (was CLAUDE.md L2524-2611) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

> _Archived: **Path 3 Phase 3.2** (was CLAUDE.md L2613-2704) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

**Default workflow (continuous mode):**
```bash
# Loads lineage 'main' if it exists, skips ~6-20 min training.
# Saves back on exit; previous state goes to history/.
python -m research.runners.chat_repl --mode synonym
```

**Science mode (multi-seed reproducibility):**
```bash
# Always trains from random init; does NOT touch lineage.
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42
```
**Compatibility:**
- Lineage stores `mode` + arch in metadata. Loading a `tier1` lineage
  with `--mode synonym` triggers a "fallback to fresh training"
  warning — no shape-mismatch crash.
- `save_checkpoint` doesn't preserve firing thresholds / STP /
  eligibility per the CLAUDE.md gotcha above. Self-recovers in ~10ms
  of free running. Fine for inference (REPL chat); documented.

> _Archived: **Recommended configuration** (was CLAUDE.md L2768-2943) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Text I/O infrastructure** (was CLAUDE.md L2944-3252) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
**🎯 LATEST BREAKTHROUGH 2026-05-05: G v2.5 + K v2 SCALES to 32×32 at 2.57 ± 0.11 (n=6) — 13.3% BETTER than the 16×16 baseline.**
```bash
# G v2.5 + K v2 — biology-grounded, perception only, scales to 32×32:
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed N --n-steps 1800
```

**Scaling result (2026-05-05 step 3).** ⚠️ **RE-CORRECTED 2026-07-16 (the FIRST correction, written the same day, was
itself WRONG — it declared "all figures are `sum_finalQ`" one line above a figure that is a MEAN, thereby CERTIFYING
the very conflation it was written to kill. It fixed the label without re-checking the number.)** The runner prints
BOTH metrics on one line (`g11_bg_runner.py:8158-8161`): `sum_finalQ` = the SUM over the 4 goal phases of each phase's
final-quarter mean Manhattan distance; `mean_distance_overall` = the mean over all steps. **They differ ~3× at 16×16
and the two headline rows below were quoted from DIFFERENT metrics.** Recomputed from the raw artifacts:
[`2026-07-16-anchor-claim-audit-...`](research/findings/2026-07-16-anchor-claim-audit-10-defects-in-the-record-incl-my-own-correction.md). ⚠️ **CORRECTED 2026-07-16 — the "closes 4 of 5 cheats
(heuristic, (gx,gy), (x,y), beacon)" claim was FALSE and is WITHDRAWN.** This config leaves
`--heuristic-strength` at its **default 1.0** → 800 pA into `cortex_N/E/S/W` derived from **direct
`gy > y` / `gx > x` goal reads**. The flag that actually closes the heuristic is
`--cue-reflex-replaces-heuristic` (`g11_bg_runner.py:7042-7045`), and it is **absent from this run's own
recorded command** (`raw/g11_bg/k_v2_stress_16x16_seed100.cmd.json`). The claim was copied from the
2026-04-27 flagship, which DOES carry that flag (so the "NO heuristic" line further down, for THAT
config, is correct). **The 2.97/2.57 numbers stand as measured — with the heuristic ON;** the visual
pathway's independent contribution is unquantified. Finding:
[`2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md`](research/findings/2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md). See
> _Archived: **Superseded/earlier nav flagships part 1** (was CLAUDE.md L3306-3405) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Superseded/earlier nav flagships part 2** (was CLAUDE.md L3410-3691) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
Findings docs in `research/findings/` document each session's outcome; **negative results are real findings** and stored alongside positives. A new runner should be added whenever a new architectural variant is being tested.

