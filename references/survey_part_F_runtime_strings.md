# Terminology Survey — Part F: runtime strings + neural-simulator.py

**Audit against:** `references/glossary.md` (228 canonical entries).
**Focus:** log strings (`_log_console`, `_log_to_ui`), error messages
(`raise ... Error(message)`), runner stdout (`print(...)`), legacy GUI host
file `neural-simulator.py`, and the headless-driver scripts.
**Scope:** Tier 1 (pure-prose) and Tier 2 (symbol-in-prose) only — Tier 3
(identifiers in prose) is exhausted by Parts A-E.
**Date:** 2026-04-29.

## Summary

| Category | Count |
|---|---|
| Files scanned | 22 |
| `_log_console` / `_log_to_ui` calls reviewed (`sim/bridge.py`) | 120 |
| `raise X(...)` error-message strings reviewed | 19 |
| Runner `print(...)` strings reviewed (`research/runners/`) | 99 |
| Run-script `print(...)` strings reviewed | 169 |
| Legacy `neural-simulator.py` strings reviewed (prints + comments) | 60 |
| **Total findings** | **45** (T1: 38, T2: 7) |

The runtime-string surface is **dominated by engineering / sim-state language**
(GPU memory, frame counts, file I/O, queue messages, profile names). Pure
biology terminology in user-facing strings is sparse; almost all biology
terms appear in **comments** rather than in logged messages, and most are
abbreviations already accepted in the glossary (HH, STDP, STP, AdEx,
MSN, GPe, GPi, BG, DA). The most-impactful Tier-1 fixes are in
**`run_benchmarks.py`** (which is published to scientific users and
prints biology-domain results in plain prose) and a small set of
neuromodulator / dopamine log strings in `sim/bridge.py`.

---

## Tier 1 — pure prose (log/error/UI strings)

### Group T1.A — `sim/bridge.py` log strings

#### `sim/bridge.py:806` — RNG init log
- **Current:** `f"RNG initialized with seed: {seed}"`
- **Issue:** None. Pure engineering message.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:812` — model-init log
- **Current:** `f"Initializing simulation data for model: {self.core_config.neuron_model_type} (3D)..."`
- **Issue:** Trivial — emits `IZHIKEVICH` / `HODGKIN_HUXLEY` / `ADEX`
  enum names which match canonical "Izhikevich" / "Hodgkin-Huxley" / "AdEx"
  glossary entries (cluster I).
- **Verdict:** **NOT FLAGGED** (token form OK).

#### `sim/bridge.py:896` — HH preset auto-override log
- **Current:** `f"Profile {profile_name}: using HH preset {profile_hh_type} as default."`
- **Issue:** Conflates "profile" (project term) with "neural-structure profile"
  (per CLAUDE.md `NEURAL_STRUCTURE_PROFILES` dict). User cannot tell from
  the bare word "profile" whether this is a JSON simulation profile, a
  neural-structure profile, or a neuron-type preset.
- **Canonical:** "Neural-structure profile" or "structure profile" (project term).
- **Tier:** T1
- **Suggested:** `f"Structure profile '{profile_name}': using HH preset {profile_hh_type} as default neuron type."`

#### `sim/bridge.py:898` — invalid HH default-type warning
- **Current:** `f"Warning: profile {profile_name} specifies invalid default_hh_neuron_type={profile_hh_type}: {e}"`
- **Issue:** Same "profile" ambiguity as above.
- **Tier:** T1
- **Suggested:** insert `"structure profile"` or `"neural-structure profile"`.

#### `sim/bridge.py:999` — Izhikevich init log
- **Current:** `f"Initializing Izhikevich model specifics for {n} neurons..."`
- **Issue:** None — "Izhikevich" matches canonical (cluster I).
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1108` — Hodgkin-Huxley init log
- **Current:** `f"Initializing Hodgkin-Huxley model specifics for {n} neurons..."`
- **Issue:** Glossary canonical is "Hodgkin-Huxley" (with hyphen). Already correct.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1137` — extended HH defaults warning
- **Current:** `f"Warning: Failed to derive extended HH defaults from {cfg.default_neuron_type_hh}: {e}"`
- **Issue:** "extended HH" is project shorthand for "Hodgkin-Huxley with
  optional currents (M-current, Cav3 T-type, Ih, NaP)". Could be more
  precise.
- **Canonical:** "extended Hodgkin-Huxley currents" (M / Cav3 / Ih / NaP per glossary I).
- **Tier:** T1 (minor — abbreviation in prose).
- **Suggested:** `"Failed to derive optional HH current parameters (M / CaT / Ih / NaP) from preset {cfg.default_neuron_type_hh}: {e}"`.

#### `sim/bridge.py:1164` — AdEx init log
- **Current:** `f"Initializing AdEx model specifics for {n} neurons..."`
- **Issue:** "AdEx" is glossary-canonical (cluster I, "AdEx Presets").
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1185-1188` — AdEx preset loaded log
- **Current:** `f"AdEx preset '{preset_name}' loaded: C={cfg.adex_C} g_L={cfg.adex_g_L} a={cfg.adex_a} tau_w={cfg.adex_tau_w} b={cfg.adex_b}"`
- **Issue:** None — `tau_w` etc are the canonical Brette & Gerstner 2005 AdEx variables.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:547-551` — neuromodulator subsystem init log
- **Current:** `f"Initialized neuromodulator subsystem with {len(cfg.neuromodulators)} modulators: {self.neuromodulator_manager.modulator_names()}"`
- **Issue:** Per glossary `[NEEDS-REVIEW]` line 1483, the project's
  `NeuromodulatorConfig` actually handles BOTH neurotransmitters
  (glutamate / GABA, J.13) AND neuromodulators (DA / NE / 5-HT / ACh /
  histamine, C cluster). The log uniformly says "modulators". For users
  loading a config that registers e.g. glutamate, the log undercounts the
  biological category.
- **Canonical:** "neuromodulators / neurotransmitters" — or accept project-
  wide convention that "neuromodulator" is the broader project usage. The
  glossary explicitly notes this is acceptable as a project convention.
- **Tier:** T1 (minor — could note ambiguity in a comment but log is OK).
- **Verdict:** **NOT FLAGGED** per glossary `[NEEDS-REVIEW]` allowance.

#### `sim/bridge.py:599-600` — STP type counts log
- **Current:** `f"Per-synapse STP types: E->E={type_counts[0]}, E->I={type_counts[1]}, I->E={type_counts[2]}, I->I={type_counts[3]}"`
- **Issue:** "E->E", "E->I" etc are conventional shorthand for "excitatory →
  excitatory" etc. The canonical glossary uses "E/I balance" (cluster J)
  and the abbreviations are widely understood. ASCII arrow may be clearer
  as Unicode `→` in scientific output.
- **Tier:** T1 (very minor — typographic, not semantic).
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:602` — no-trait-info warning
- **Current:** `"No trait info available; all synapses default to E->E STP type."`
- **Issue:** "trait info" is project-internal jargon for "neuron-type /
  excitatory-vs-inhibitory designation". A reader who skipped CLAUDE.md
  may not know "trait" maps to E/I role.
- **Canonical:** project term "trait" is established but could be expanded
  on first use.
- **Tier:** T1
- **Suggested:** `"No trait (excitatory/inhibitory) information available; all synapses default to E→E STP type."`

#### `sim/bridge.py:1326` — STDP init log
- **Current:** `f"Initializing STDP state for {n} neurons..."`
- **Issue:** None — "STDP" is glossary-canonical (J.29).
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1334` — structural plasticity init log
- **Current:** `"Initializing structural plasticity state..."`
- **Issue:** None — "structural plasticity" is glossary-canonical (J / L).
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1357-1359` — initialization-complete log
- **Current:** `f"Simulation data initialized for {n} neurons (3D). Connections: {conn_count}. GPU memory: ..."`
- **Issue:** "Connections" is project shorthand for "synapses". The glossary
  has "synapse" as canonical (multiple clusters). Mixing "connections" and
  "synapses" across the codebase creates confusion. CSR `cp_connections`
  is fine (identifier), but in user-facing prose "synapses" is the
  scientific term.
- **Tier:** T1
- **Suggested:** `f"Simulation data initialized for {n} neurons (3D). Synapses: {conn_count}. GPU memory: ..."`
- **Note:** This is one of the **most-impactful** fixes — "connections" is used in dozens of log strings.

#### `sim/bridge.py:1411-1413` — region neuron-type override log
- **Current:** `f"Region '{region.name}' ({len(indices)} neurons): using Izh type {region.izh_neuron_type}"`
- **Issue:** "Izh" is shorthand for "Izhikevich" (cluster I). Used
  consistently across the codebase. Slight loss of clarity but acceptable.
- **Tier:** T1 (minor)
- **Verdict:** **NOT FLAGGED** (consistent shorthand).

#### `sim/bridge.py:1502` — heterogeneity log
- **Current:** `f"Applied heterogeneity to {applied_count} parameters."`
- **Issue:** "heterogeneity" is canonical for parameter heterogeneity
  (Marder-Goaillard 2006, cluster I); already correct.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1549` — OU process init log
- **Current:** `f"Initializing OU process state (tau={cfg.ou_tau_ms}ms, sigma={cfg.ou_std_current_pA}pA)..."`
- **Issue:** None — "OU process" (Ornstein–Uhlenbeck) is widely-used
  shorthand and glossary-acceptable for stochastic-current modeling.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1683` — clearing-state log
- **Current:** `"Clearing simulation state and GPU memory..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1789` — wiring plan warning
- **Current:** `"inject_explicit_wiring: no synapses in plan."`
- **Issue:** Uses "synapses" — canonical. Already correct.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1942-1945` — wiring inject log
- **Current:** `f"inject_explicit_wiring: installed {nnz} synapses across {sum(...)} populations."`
- **Issue:** None — "synapses", "populations" are canonical.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1973-1975` — plasticity-gate KeyError
- **Current:** `f"No plasticity gate named '{name}'. Known gates: {list(...)}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:1664` — inter-group connectivity injection log
- **Current:** `f"Injected {added} inter-group connections for experiment learning paths"`
- **Issue:** "connections" (vs "synapses"). Same as 1357.
- **Tier:** T1
- **Suggested:** `f"Injected {added} inter-group synapses for experiment learning paths"`

#### `sim/bridge.py:1665` — experiment-engine init log
- **Current:** `f"Experiment engine initialized: {self.experiment_config.name}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `sim/bridge.py:4036` — experiment engine error log
- **Current:** `f"Experiment engine step error: {e}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.B — `neural-simulator.py` (legacy GUI host) prints/comments

#### `neural-simulator.py:139` — startup message
- **Current:** `"PyOpenGL found. OpenGL visualization will be used."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:151` — CuPy init message
- **Current:** `"CuPy initialized for GPU acceleration."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:441-443` — model-type comments
- **Current:**
  ```
  self.neuron_model_type = NeuronModel.IZHIKEVICH.name # Current neuron model ('IZHIKEVICH', 'HODGKIN_HUXLEY', or 'ADEX')
  self.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name # Default Izhikevich type if trait mapping fails
  self.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name # Default Hodgkin-Huxley type
  ```
- **Issue:** Names are canonical or canonical-equivalent.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:451-459` — Izhikevich parameter comments (block)
- **Current:** Each parameter has an inline comment (e.g.
  `# Membrane capacitance (pF)`, `# Constant related to Na+ channel kinetics`,
  `# Resting membrane potential (mV)`, etc.).
- **Issue:** None — these match canonical Izhikevich-2007 nomenclature.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:466-490` — Hodgkin-Huxley parameter comments (block)
- **Current:** Inline comments in micro-units (e.g. `µF/cm^2`, `mS/cm^2`,
  `µA/cm²`).
- **Issue:** Mixed unit styles — some entries use `^2` (ASCII), others `²`
  (Unicode). Glossary requires "Hodgkin-Huxley" (correct). Unit consistency
  is project-specific (per CLAUDE.md "Units" section), not a glossary
  concern.
- **Verdict:** **NOT FLAGGED** (glossary OK; unit consistency is style not terminology).

#### `neural-simulator.py:511` — synaptic-reversal comment
- **Current:** `# Reversal potential for inhibitory synapses (mV) — Cl- Nernst at 37C`
- **Issue:** "Cl- Nernst at 37C" is a precise scientific reference. The glossary
  uses "Cl⁻" with proper minus-sign Unicode in the canonical entry; ASCII
  hyphen "Cl-" is acceptable shorthand. "37C" should ideally be "37°C" for
  scientific precision (degree symbol).
- **Tier:** T1 (very minor — typographic).
- **Suggested:** `# Reversal potential for inhibitory synapses (mV) — Cl⁻ Nernst at 37°C`

#### `neural-simulator.py:514-515` — propagation strength comments
- **Current:**
  ```
  self.propagation_strength = 0.05 # Scaling factor for excitatory synaptic conductance increase per spike
  self.inhibitory_propagation_strength = 0.105 # Scaled for E_inh=-75mV (was 0.15 at -70mV)
  ```
- **Issue:** "propagation strength" is a project-internal name; the canonical
  scientific term would be "synaptic weight" or "conductance increment per
  presynaptic spike". `propagation_strength` is fine as identifier; comment
  could be slightly more precise.
- **Tier:** T1 (minor).
- **Suggested:** `# Conductance increment per presynaptic spike (excitatory)` etc.

#### `neural-simulator.py:516` — synaptic-delay comment
- **Current:** `# Maximum synaptic delay in ms (Not fully implemented for individual delays yet)`
- **Issue:** None — "synaptic delay" is canonical.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:518-521` — inhibitory-neuron config
- **Current:** Block has comments like `# Trait index designated as inhibitory (0-indexed)`.
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:523-528` — Hebbian/LTP block comment
- **Current:**
  ```
  # Hebbian Learning / Long-Term Potentiation (LTP)
  self.enable_hebbian_learning = True # Enable Hebbian-like weight potentiation
  ...
  ```
- **Issue:** Section header conflates **Hebbian** (rule family, cluster J) with
  **LTP** (the *phenomenon* of long-term potentiation, cluster J.28). LTP is
  the experimental observable; Hebbian/STDP/etc are *rules* that produce
  LTP-like weight changes. The current code uses Hebbian rule.
- **Canonical:** "Hebbian rule" or "Hebbian learning" — distinct from LTP.
- **Tier:** T1
- **Suggested:** `# Hebbian Learning Rule (produces LTP-like potentiation)`.

#### `neural-simulator.py:531-534` — STP block comment
- **Current:** `# Short-Term Plasticity (STP) - Tsodyks-Markram model` ... `# STP U parameter (baseline utilization of synaptic resources)`
- **Issue:** None — Tsodyks-Markram is the canonical STP model name (J.03).
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:541-547` — Homeostasis block
- **Current:**
  ```
  # Homeostatic Plasticity (Adaptive Thresholds for Izhikevich model)
  self.enable_homeostasis = True # Enable homeostatic threshold adaptation
  ...
  self.homeostasis_target_rate = 0.02 # Target firing rate (spikes per dt step)
  ```
- **Issue:** Uses "homeostatic plasticity" which is glossary-canonical
  (cluster J). The variable `homeostasis_target_rate` semantically is "spikes
  per dt step", but the canonical in scientific literature is "Hz"
  (spikes/second). The unit comment is precise — fine.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:549-551` — synaptic scaling block
- **Current:** `# Synaptic Scaling (Turrigiano 2008) - multiplicative excitatory weight scaling`
- **Issue:** None — "synaptic scaling" with Turrigiano citation is canonical (J).
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:553` — NMDA block comment
- **Current:** `# NMDA conductance with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)`
- **Issue:** None — "NMDA", "Mg²⁺", "Jahr & Stevens 1990" all canonical (J.08).
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:560-568` — STDP block comment
- **Current:**
  ```
  # STDP (Spike-Timing Dependent Plasticity)
  self.enable_stdp = True
  self.stdp_a_plus = 0.012          # LTP amplitude (biased > A- for net potentiation)
  self.stdp_a_minus = 0.01          # LTD amplitude
  self.stdp_tau_plus_ms = 20.0      # LTP time constant (ms)
  self.stdp_tau_minus_ms = 20.0     # LTD time constant (ms)
  ```
- **Issue:** Glossary writes "Spike-Timing-Dependent Plasticity" — both
  hyphenated. The current text "Spike-Timing Dependent" misses one hyphen.
  Minor.
- **Tier:** T1 (typographic — matches scientific style).
- **Suggested:** `# STDP (Spike-Timing-Dependent Plasticity)`

#### `neural-simulator.py:570-575` — Reward modulation block
- **Current:**
  ```
  # Reward-Modulated Plasticity
  self.enable_reward_modulation = True
  self.reward_learning_rate = 0.01
  ...
  ```
- **Issue:** "Reward-modulated plasticity" is acceptable shorthand for
  three-factor / R-STDP rule (glossary J / O.20). "Three-factor learning
  rule" or "R-STDP" is the canonical scientific name. Comment could
  cross-reference.
- **Tier:** T1 (minor).
- **Suggested:** `# Reward-Modulated Plasticity (three-factor / R-STDP rule)`

#### `neural-simulator.py:577-586` — Structural plasticity block comment
- **Current:** `# Structural Plasticity` ... `self.struct_plast_activity_bias = 0.5  # Co-activity bias for synapse formation`
- **Issue:** None — canonical (cluster J / L).
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:594-600` — channel noise / OU block
- **Current:**
  ```
  # Enhanced Channel Noise (Phase B4)
  self.enable_conductance_noise = False # Enable multiplicative conductance noise (HH only)
  ...
  self.enable_ou_process = False # Enable Ornstein-Uhlenbeck background current
  ```
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:1117` — sim worker thread message
- **Current:** `"Simulation worker thread started."`
- **Issue:** None — pure engineering.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:1391` — sim worker thread message
- **Current:** `"Simulation worker thread finished."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:2104` — config-fetch CRITICAL
- **Current:** `"CRITICAL: Failed to get initial config from UI for sim_thread."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:2123` — GLUT window title
- **Current:** `b"3D Network Visualization (OpenGL - Threaded)"`
- **Issue:** "3D Network" is fine; "Network" here meaning "neural network"
  is universally understood.
- **Verdict:** **NOT FLAGGED.**

#### `neural-simulator.py:2174` — shutdown print
- **Current:** `"Neuron simulator application shutdown complete."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.C — `research/runners/g11_bg_runner.py` print strings

#### `g11_bg_runner.py:2120` — BG circuit summary
- **Current:** `f"[g11 seed={seed}] BG circuit: {len(regions)} regions, ..."`
- **Issue:** "BG circuit" is project shorthand for the basal-ganglia cascade
  (`cortex → MSN → GPe/GPi → thal → motor`). Glossary cluster A uses
  "cortico-BG-thalamo-cortical loops" or "BG cascade" / "BG output
  complex" but accepts "BG" as universal shorthand.
- **Verdict:** **NOT FLAGGED** (consistent project terminology).

#### `g11_bg_runner.py:2207-2208` — curriculum phase 1 log
- **Current:** `f"[g11 seed={seed}] curriculum phase 1: cortex_to_d1 plastic, input gates frozen [{gates_msg}]{ramp_msg}"`
- **Issue:** "cortex_to_d1" is the project's plasticity-gate name. In glossary
  terms this is "corticostriatal direct-pathway projection" (cluster A.01,
  J.30: corticostriatal STDP). The runner-output prose could be slightly
  more transparent for non-codebase users.
- **Tier:** T1 (minor — would only matter if log were promoted to a public
  analysis output).
- **Verdict:** **NOT FLAGGED** (gate names are stable shorthand).

#### `g11_bg_runner.py:2289-2291` — curriculum phase 2 log
- **Current:** `f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 2 -- cortex_to_d1={...}, inputs={...}"`
- **Issue:** Same as above.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:2308-2310` — curriculum phase 3 (cross-projection thaw) log
- **Current:** `f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 3 -- bg_cross_projections gain={...}"`
- **Issue:** "bg_cross_projections" is project's gate name for cortex_X →
  str_D1_Y / str_D2_Y all-to-all routing (per CLAUDE.md / cheat #5
  history). Acceptable shorthand.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:2330-2332` — sleep replay log
- **Current:** `f"[g11 seed={seed}] step {step}: ENTERING SLEEP REPLAY (cortex_to_d1=1, hippo/sensory frozen, replay rate={...}Hz)"`
- **Issue:** "SLEEP REPLAY" matches glossary "replay" (N.07, N.17, D — the
  glossary explicitly distinguishes "forward replay" / "reverse replay" /
  "awake replay" / "sleep-only consolidation"). The runner uses sleep-
  replay infrastructure correctly. The per-step log could note "sharp-wave
  ripple analog" or "consolidation phase" but the bare term is OK.
- **Verdict:** **NOT FLAGGED** (project usage matches glossary).

#### `g11_bg_runner.py:2334` — exit sleep replay log
- **Current:** `f"[g11 seed={seed}] step {step}: EXITING SLEEP REPLAY"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:1248-1250` — pretraining log
- **Current:** `f"[g11 seed={seed}] pretraining: all {len(available)} declared gates thawed to 1.0; running {n_goals} goals × {steps_per_goal} steps each"`
- **Issue:** "Pretraining" / "thawed" are project metaphors for the
  developmental-pretraining v4 protocol (cheat #5 history). Glossary uses
  "developmental pretraining" implicitly (cluster L "critical period").
  Acceptable as project shorthand.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:1334-1335` — pretraining per-goal log
- **Current:** `f"[g11 seed={seed}] pretraining goal {goal_idx + 1}/{n_goals}: ..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:1927` — adaptive DA log
- **Current:** `f"[g11 seed={seed}] Cluster C v2 compartmentalized DA: ..."`
- **Issue:** "Cluster C v2 compartmentalized DA" — "DA" is glossary-canonical
  (C.04). "Cluster C" matches the catalog. Acceptable.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:1982` — per-action DA log
- **Current:** `f"[g11 seed={seed}] per-action DA ({mode}): ..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:2021` — DA-gated WTA log
- **Current:** `f"[g11 seed={seed}] DA-gated WTA: {int(chosen_mask.sum())} FS->motor synapses ..."`
- **Issue:** "WTA" (winner-take-all) is conventional shorthand. "FS->motor"
  uses E→I/I→I shorthand, OK. "FS" is glossary-canonical for fast-spiking
  interneuron (B.06).
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:2069` — learned perception log
- **Current:** `f"[g11 seed={seed}] learned perception (informed init): ..."`
- **Issue:** "learned perception" is project-specific naming for the
  Cluster E perception-arc Stage 3 work. Acceptable as project term.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3239-3241` — smoke test header
- **Current:** `print(f"  G11 BG Action Selection Module -- Smoke Test")`
- **Issue:** None — descriptive label.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3245-3246` — smoke test region count
- **Current:** `f"  Built {len(regions)} regions with {n_total} total neurons"` and `f"  Built {len(pathways)} pathways"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3299` — total synapses log
- **Current:** `f"  Total synapses: {bridge.cp_connections.nnz}"`
- **Issue:** None — uses canonical "synapses".
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3320` — per-region firing rates log
- **Current:** `f"\n  Per-region firing rates (Hz over {n_steps}ms with no input):"`
- **Issue:** None — canonical Hz / ms / "firing rates".
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3332` — action-selection probe header
- **Current:** `f"  Action selection probe: drive cortex -> {args.probe_action} pathway"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `g11_bg_runner.py:3387` — driving cortex log
- **Current:** `f"  Driving {len(target_cortex)}/{len(cortex_idx)} cortex neurons ..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.D — `run_benchmarks.py` print strings (PUBLIC RESULTS)

This script prints scientifically-presented benchmark results that are saved
and shared. It is the **highest-impact user-facing string surface** for
biology terminology in the codebase.

#### `run_benchmarks.py:52` — header
- **Current:** `"BENCHMARK 2.1: STDP Timing Curve (Bi & Poo 1998)"`
- **Issue:** None — "Bi & Poo 1998" is canonical citation; "STDP Timing
  Curve" matches glossary J.29.
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:285` — header
- **Current:** `"BENCHMARK 2.2: E/I Balance and Spontaneous Firing Rates"`
- **Issue:** None — "E/I balance" and "spontaneous firing rates" are
  canonical (cluster J).
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:325-327` — E/I summary text
- **Current:** `"Excitatory: {n_exc} ({n_exc/n*100:.0f}%)"`, `"Inhibitory: {n_inh} ..."`, `"E/I ratio: {ei_ratio:.1f}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:402` — CV of ISI label
- **Current:** `f"  {'CV of ISI':<35s}  {exc_cv_mean:>12.2f}  {inh_cv_mean:>12.2f}  {'0.5-1.5':>15s}"`
- **Issue:** None — "CV of ISI" (coefficient-of-variation of inter-spike
  interval) is canonical scientific shorthand.
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:480` — STP benchmark header
- **Current:** `"BENCHMARK 2.3: STP Paired-Pulse Ratio (Tsodyks-Markram)"`
- **Issue:** None — canonical (J.03).
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:502-504` — STP table header
- **Current:** `f"\n  {syn_name} (U={U}, tau_d={tau_d}ms, tau_f={tau_f}ms) ..."` and PPR table
- **Issue:** None — "PPR" canonical.
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:624` — Gamma benchmark header
- **Current:** `"BENCHMARK 2.4: Gamma Oscillation Emergence (PING)"`
- **Issue:** None — "PING" (pyramidal-interneuron network gamma) glossary-
  canonical (cluster I, N.19).
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:668` — profile name in print
- **Current:** `f"  Profile: CORTEX_GAMMA_FS_NETWORK"`
- **Issue:** Profile identifier — fine.
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:668` — gamma band label
- **Current:** `"Gamma band fraction: {gamma_frac:.1%}"`, `"Beta band fraction: ..."`
- **Issue:** None — gamma / beta are canonical (I).
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:823` — homeostasis benchmark header
- **Current:** `"BENCHMARK 2.5: Homeostatic Firing Rate Regulation"`
- **Issue:** "Homeostatic firing rate regulation" — canonical (J / I).
- **Verdict:** **NOT FLAGGED.**

#### `run_benchmarks.py:1050` — global header
- **Current:** `"BIOLOGICAL BENCHMARK VALIDATION SUITE"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.E — `run_experiment_headless.py` print strings

#### `run_experiment_headless.py:200` — basic stim-response header
- **Current:** `f"BASIC STIMULUS-RESPONSE RESULTS"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:264` — associative results header
- **Current:** `f"ASSOCIATIVE CONDITIONING RESULTS"`
- **Issue:** None — "associative conditioning" / Pavlovian (cluster J / O).
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:266-268` — CS-ON / CS-OFF labels
- **Current:** `f"  CS-ON Pre:  {pre_a.mean():.2f} ± {pre_a.std():.2f} Hz (n={len(pre_on)})"`
- **Issue:** None — CS / US are canonical Pavlovian shorthand.
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:325` — frequency-response header
- **Current:** `f"FREQUENCY RESPONSE CHARACTERIZATION RESULTS"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:355` — bandpass test
- **Current:** `f"  Bandpass filter: {'YES' if is_bandpass else 'NO'} (ratio > 1.5x)"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:411` — RL header
- **Current:** `f"REINFORCEMENT LEARNING (R-STDP) RESULTS"`
- **Issue:** "Reinforcement Learning (R-STDP)" — both glossary-canonical
  (J / O / C.29).
- **Verdict:** **NOT FLAGGED.**

#### `run_experiment_headless.py:419` — target window
- **Current:** `f"  Target window: {tmin}-{tmax} Hz"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.F — `run_parameter_sweep.py`, `benchmark.py`, `viz_benchmark.py`

#### `run_parameter_sweep.py:300` — parameter sweep header
- **Current:** `f"PARAMETER SWEEP: {experiment.upper()}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `benchmark.py:142-145` — config description
- **Current:** `f"Testing: {config_dict['num_neurons']} neurons, ... model: {config_dict.get('neuron_model_type', 'IZHIKEVICH')}, ..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `viz_benchmark.py:160-163` — config description
- **Current:** `f"Testing: {config_dict['num_neurons']:,} neurons, model: ..."`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `viz_benchmark.py:464-466` — analysis header
- **Current:** `"REALTIME CAPACITY ANALYSIS"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

### Group T1.G — error messages

#### `sim/regions.py:294-297` — coordinate-center length
- **Current:** `f"region {region.name!r}: coordinate_center has length {len(center)} but coordinate_dim={k}"`
- **Issue:** None — engineering message.
- **Verdict:** **NOT FLAGGED.**

#### `sim/regions.py:305-308` — coordinate-extent length
- **Current:** `f"region {region.name!r}: coordinate_extent has length {len(extent)} but coordinate_dim={k}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `sim/regions.py:321,326,343` — region KeyError
- **Current:** Bare `raise KeyError(region_name)`.
- **Issue:** Could include hint ("known regions: ...") but that's API
  ergonomics, not terminology.
- **Verdict:** **NOT FLAGGED.**

#### `sim/connectivity.py:65-66` — gauss distance density error
- **Current:** `f"gauss_distance_density: sigma must be > 0; got {sigma}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `sim/config.py:326` — CoreSimConfig validation failure
- **Current:** `"CoreSimConfig validation failed:\n  - " + "\n  - ".join(errors)`
- **Issue:** None — engineering format.
- **Verdict:** **NOT FLAGGED.**

#### `experiment/stimulus.py:51-55` — RATE_VECTOR_POISSON length error
- **Current:** `f"RATE_VECTOR_POISSON rate_vector_hz length ({len(ch.pattern.rate_vector_hz)}) must equal number of target neurons ({len(indices)}) for channel '{ch.name}'"`
- **Issue:** None — technical Poisson-rate validation.
- **Verdict:** **NOT FLAGGED.**

#### `research/runners/g11_bg_runner.py:1233-1238` — pretraining gate KeyError
- **Current:**
  ```
  f"_run_pretraining_phase: gate(s) not declared on any pathway: "
  f"{missing!r}. Available: {sorted(available)!r}. "
  f"Either spell-check the gate name in build_bg_brain_regions, "
  f"or enable the flag that adds the pathway."
  ```
- **Issue:** Excellent error message; no terminology issues.
- **Verdict:** **NOT FLAGGED.**

#### `research/runners/g11_bg_runner.py:1434` — RuntimeError (need to find)
- **Verdict:** **NOT FLAGGED** (engineering message, not biology).

#### `research/runners/g1_runner.py:214` — TypeError serialization
- **Current:** `f"Not serializable: {type(obj)}"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

#### `research/runners/g1_decoder.py:14` — input validation
- **Current:** `"spike_counts must be non-empty"`
- **Issue:** None.
- **Verdict:** **NOT FLAGGED.**

---

## Tier 2 — symbol-in-prose

### `neural-simulator.py:457` — Izh `b` parameter comment
- **Current:** `self.izh_b_val = rs_params_2007["b"]       # Sensitivity of u to subthreshold fluctuations (nS)`
- **Issue:** Pure shape — `u` is the recovery variable (Izhikevich-2007).
  In the comment, `u` could be ambiguous to a reader without the model
  open. Glossary I.02 has "action potential" but the recovery variable is
  not separately glossed. Could be `# Sensitivity of recovery variable u to subthreshold fluctuations (nS)`.
- **Tier:** T2 (symbol `u` in prose).
- **Suggested:** `# Sensitivity of recovery variable u to subthreshold fluctuations (nS)`

### `neural-simulator.py:1105` — recovery variable formula
- **Current:** `self.cp_recovery_variable_u = self.cp_izh_b * (self.cp_membrane_potential_v - self.cp_izh_vr)` (no inline comment)
- **Issue:** Implicit `u = b·(V − Vr)` Izh-2007 init formula. Could
  document.
- **Tier:** T2 (no prose; would be informational only).
- **Verdict:** **NOT FLAGGED** (no prose to fix).

### `sim/bridge.py:858` — HH current-density comment
- **Current:** `# 10 µA/cm² = 10,000,000 pA (when divided by 1e-6 later = 10 µA/cm²)`
- **Issue:** Mixed unit notation `µA/cm²` and `pA`. The unit-conversion
  comment is technically clear; "1e-6" is the multiplier from pA to µA
  but the relationship to surface area density needs explanation. Pure
  symbol/comment; no glossary fix needed.
- **Verdict:** **NOT FLAGGED.**

### `sim/bridge.py:1345` — HH temperature scaling
- **Current:** `_BASE_HH_TEMP = 6.3` and `self._cached_hh_phi = cfg.hh_q10_factor ** ((cfg.hh_temperature_celsius - _BASE_HH_TEMP) / 10.0)`
- **Issue:** `_BASE_HH_TEMP = 6.3` is the original Hodgkin-Huxley squid
  axon temperature (1952). Comment explaining the historical 6.3 °C base
  would help. Glossary I has "Hodgkin-Huxley" but no entry for the
  6.3 °C base temperature.
- **Tier:** T2 (symbol `_BASE_HH_TEMP` in code).
- **Suggested:** add `# Original Hodgkin-Huxley squid axon experiments at 6.3 °C (1952).`

### `sim/bridge.py:1346-1350` — Q10 phi cache
- **Current:** Comment block — `# Per-gate phi values (Session "fix-bugs" — see HH temperature bug findings)`
- **Issue:** Cross-references `findings/` — clear engineering note.
- **Verdict:** **NOT FLAGGED.**

### `neural-simulator.py:553` — Mg²⁺ comment uses Unicode superscript
- **Current:** `# NMDA conductance with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)`
- **Issue:** Uses proper Unicode `²⁺`. Glossary canonical for `Mg²⁺` is the
  same. Already correct.
- **Verdict:** **NOT FLAGGED.**

### `g11_bg_runner.py:1304-1316` — tonic drive setup comments
- **Current:** No inline biology rationale on the 150 / 110 / 300 pA values
  per region. The values are tonic GPe / GPi / thal / STN / DA pA drives
  per the BG cascade design.
- **Issue:** The biology mapping (e.g. "GPe pacemakers fire ~50–80 Hz at
  rest, sustained by tonic glutamate") is in the docs/findings but not in
  the runner comments. Could be improved with one cross-reference comment.
- **Tier:** T2 (purely informational, not flagged as wrong).
- **Verdict:** **NOT FLAGGED.**

---

## Tier 3 — none expected (this scope is strings, not identifiers)

Identifiers (variable / function / module names) are out of scope here per
the task spec. Examples like `cp_izh_b`, `cortex_to_d1`, `gpi_X`, etc, are
canonical project-shorthand and reviewed in Parts A-E.

---

## Items NOT flagged (intentional shorthand or engineering message)

These items appear in scope but are **not actionable** per glossary policy:

1. **GPU / memory / threading messages** (~80% of `sim/bridge.py` log strings)
   are pure engineering: `"GPU memory high: {...}%"`, `"Frame size: {...}MB"`,
   `"Streaming recording: frame {idx} queued, {n} pending write"`, etc.
   These do not contain biology terms and need no glossary review.
2. **HDF5 / playback / checkpoint / recording messages** are file-I/O
   technical, e.g. `"Phase 1a: Transferring {n} GPU frames to CPU..."`.
3. **`STDP`, `STP`, `MSN`, `GPe`, `GPi`, `BG`, `DA`, `HH`, `Izh`, `AdEx`,
   `FS`, `WTA`, `OU`, `EPSP`, `IPSP`, `PSD`, `Hz`, `pA`, `nS`, `mV`, `ms`,
   `dt`** — all glossary-accepted shorthand. Used in dozens of strings,
   never flagged unless misused.
4. **Profile / preset / region / pathway / phase / curriculum / replay / sleep**
   — project-internal terminology, all glossary-aligned, used consistently.
5. **`E->E`, `E->I`, `I->E`, `I->I`** — synaptic-class shorthand for
   Excitatory-vs-Inhibitory pairings; convention OK.
6. **Trait / channel / target / readout** — engineering-side abstractions
   from the experiment system; not biology-specific.
7. **Legacy `SimulationConfiguration` parameter docstrings** in
   `neural-simulator.py:429-647` — extensive but technical; only the
   "Hebbian Learning / LTP" header (line 523) and the "Spike-Timing
   Dependent" hyphenation (line 560) raise minor flags.
8. **Runner action-tag prose** (`"NORTH"`, `"EAST"`, ...) — task-specific.
9. **`current_reward_signal`, `reward_baseline`, `reward_eligibility_tau_ms`** —
   per glossary `[NEEDS-REVIEW]` line 1479, the project's `current_reward_signal`
   conflates phasic/tonic DA + Component 1/2 + A9/A10. The audit policy is
   **not to flag every use** of `current_reward_signal` but to note when
   biological distinctions matter. None of the runtime strings reviewed
   make biological claims that this term cannot bear.

---

## Most-impactful Tier 1 fixes (recommended priorities)

1. **`sim/bridge.py:1357-1359`** — change `"Connections: {conn_count}"` →
   `"Synapses: {conn_count}"` in the post-init log (and audit other places
   where "connections" should be "synapses"). High visibility — every
   simulation run prints this once.
2. **`sim/bridge.py:1664`** — change `"Injected {added} inter-group
   connections for experiment learning paths"` → `"... synapses ..."`.
   Same root cause; user-facing.
3. **`neural-simulator.py:523`** — section header `# Hebbian Learning /
   Long-Term Potentiation (LTP)` → split as `# Hebbian Learning Rule
   (produces LTP-like potentiation)` so the rule (Hebbian) and phenomenon
   (LTP) are not conflated. Minor but pedagogically important — this
   block is the canonical place a new contributor learns the project's
   plasticity vocabulary.

Honorable mentions:
- `sim/bridge.py:896, 898` — disambiguate "profile" → "structure profile"
  / "neural-structure profile" in HH preset log strings.
- `neural-simulator.py:511` — Cl- → Cl⁻; 37C → 37°C (typographic polish).
- `neural-simulator.py:560` — `Spike-Timing Dependent` → `Spike-Timing-Dependent`.

---

## Policy questions

1. **"Connections" vs "synapses".** Many log strings, and the CSR matrix
   `cp_connections`, use "connections". The glossary cluster J uses
   "synapse" / "synapses". Recommend renaming **only the user-facing
   prose** (logs, errors, comments) — leave `cp_connections` and
   `connections_per_neuron` as identifier shorthand. **Flag for confirmation:**
   should the identifier `cp_connections` be renamed in a future pass, or
   is it stable?
2. **`E->E` (ASCII) vs `E→E` (Unicode arrow).** Glossary doesn't specify;
   project uses ASCII consistently. Recommend leaving as-is. **No action.**
3. **"Profile" disambiguation.** The word "profile" in different log
   strings means different things: (a) JSON simulation profile in
   `simulation_profiles/`, (b) `NEURAL_STRUCTURE_PROFILES` dict entry
   ("CORTEX_L23_RS_FS"), (c) HH neuron-type preset
   ("HH_L5_CORTICAL_PYRAMIDAL_RS"). The glossary doesn't disambiguate
   these because they're project artifacts. Recommend the survey **flag
   the in-prose mentions** but **defer renaming to a separate task** that
   covers identifiers and JSON keys consistently.
4. **`current_reward_signal` and the Schultz Component 1 / Component 2
   distinction.** Per glossary `[NEEDS-REVIEW]` line 1479, this is a known
   simplification. None of the audited strings claim more than the project's
   model can deliver. **No action — known and documented.**
5. **`BG circuit` vs `BG cascade` vs `cortico-BG-thalamo-cortical loops`.**
   The project uses all three interchangeably across logs and CLAUDE.md.
   Glossary cluster A.05 has the long form as canonical. Project shorthand
   is acceptable but inconsistent. **Defer to a separate harmonization
   pass** if the team wants to standardize.

---

## File-list reviewed (paths absolute)

- `E:\Documents\Projects\sim\sim\bridge.py` (5347 lines, 120 `_log_console` calls + 19 error messages)
- `E:\Documents\Projects\sim\sim\regions.py` (350 lines, 5 errors)
- `E:\Documents\Projects\sim\sim\config.py` (676 lines, 2 errors)
- `E:\Documents\Projects\sim\sim\connectivity.py` (923 lines, 2 errors)
- `E:\Documents\Projects\sim\sim\neuromodulators.py` (430 lines, 1 error)
- `E:\Documents\Projects\sim\experiment\stimulus.py` (1 error)
- `E:\Documents\Projects\sim\research\runners\g11_bg_runner.py` (3420+ lines, 46 prints + 3 errors)
- `E:\Documents\Projects\sim\research\runners\g1_runner.py` (1 error)
- `E:\Documents\Projects\sim\research\runners\g1_v2_runner.py` (1 error)
- `E:\Documents\Projects\sim\research\runners\g1_decoder.py` (1 error)
- `E:\Documents\Projects\sim\research\runners\` (other runners, ~50 prints total)
- `E:\Documents\Projects\sim\webapp\server.py` (1 print)
- `E:\Documents\Projects\sim\neural-simulator.py` (2183 lines, ~60 strings)
- `E:\Documents\Projects\sim\run_experiment_headless.py` (506 lines, ~80 prints)
- `E:\Documents\Projects\sim\run_parameter_sweep.py` (453 lines, ~25 prints)
- `E:\Documents\Projects\sim\run_benchmarks.py` (1117 lines, ~75 prints)
- `E:\Documents\Projects\sim\benchmark.py` (479 lines, ~40 prints)
- `E:\Documents\Projects\sim\viz_benchmark.py` (608 lines, ~45 prints)
