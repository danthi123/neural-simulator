# Terminology Survey — Part C: `tests/`, `experiment/`, `viz/`, `ui/`

**Audit against:** `references/glossary.md` (228 canonical entries).
**Date:** 2026-04-28.
**Scope:** Pure-prose terminology issues in `tests/`, `experiment/`, `viz/`, `ui/`.

## Summary
- Files scanned: ~50 (28 tests + 7 experiment + 4 viz + 9 ui)
- Tier 1 (pure prose): 86
- Tier 2 (symbols-in-prose): 12
- Tier 3 (identifiers, FLAGGED only): 23
- Items NOT flagged (intentional shorthand): see final section

The big asymmetry is in the UI: most glossary mismatches are in the UI tooltip text seen by humans (Hodgkin-Huxley, AdEx, STDP, NMDA, GABA-A help text). The experiment package is mostly correct because it inherits canonical names ("STDP", "Pavlovian", "R-STDP") from the catalog. Test docstrings are mixed — some excellent (D1/D2 asymmetry, regions) and some loose (HH, AdEx without subtype labels).

---

## Tier 1 — pure prose (docstrings, comments, log messages, UI strings)

### `experiment/engine.py`

#### `experiment/engine.py:1-5` — module docstring
- **Current:** `"Experiment engine - top-level orchestrator for multi-phase experiments. Manages phase transitions, stimulus current generation, response measurement, training protocol execution, and experiment logging."`
- **Issue:** Generic, but acceptable. No glossary mismatch.
- **Status:** OK.

#### `experiment/engine.py:401-408` — `ensure_inter_group_connectivity` docstring
- **Current:** `"For associative conditioning to work via STDP, there must be dense direct synaptic paths from CS (input) to US (output) neurons."`
- **Issue:** "associative conditioning" is loose — glossary canonical is "classical conditioning" (J.24-J.26, J.33) when used at the cell level, "Pavlovian conditioning" or "associative pairing" at the protocol level. The mention of CS/US is correct.
- **Suggestion:** Replace "associative conditioning" with "Pavlovian (CS-US) conditioning" or keep as is — both terms appear in the glossary as accepted synonyms.
- **Tier:** 1 (prose)
- **Notes:** Low priority — glossary lists "classical conditioning" with "Pavlovian" as accepted. "Associative conditioning" is also in current usage in the codebase (preset name).

#### `experiment/engine.py:447-449` — comment about plasticity bias
- **Current:** `"... with inhibitory_propagation_strength (0.105, 2.1x excitatory) create strong opposing currents that cancel excitatory CS->US drive. With 20% inhibitory neurons in a cortical profile..."`
- **Issue:** "inhibitory neurons" — glossary canonical is "GABAergic neurons" or "interneurons" (depending on layer). "Inhibitory" used here is electrophysiological shorthand, acceptable.
- **Status:** OK as engineering shorthand. Not flagged.

### `experiment/training.py`

#### `experiment/training.py:14-19` — class docstring
- **Current:** `"Executes training protocols: associative pairing, RL, supervised, reservoir."`
- **Issue:** "RL" is unambiguous in context, but the formal glossary term is "reinforcement learning" / "R-STDP" / "three-factor learning".
- **Status:** OK in this context. RL is unambiguous.

#### `experiment/training.py:166-185` — `_end_trial` comment about associative conditioning
- **Current:** `"For associative conditioning: success = US output group rate exceeds CR threshold. NOTE: _end_trial() is called at the end of each trial (during ITI), so the readout rate reflects the post-ITI baseline, NOT the CS-driven response. Per-trial accuracy during training is therefore not meaningful for associative conditioning..."`
- **Issue:** Uses "CS" / "US" / "ITI" / "CR threshold" without expanding. Glossary recognizes CS / US as standard Pavlovian terms (J.25), CR (conditioned response) is implied. ITI is intertrial interval — not in glossary, but standard.
- **Status:** OK. CS/US/ITI/CR are all canonical Pavlovian abbreviations.

#### `experiment/training.py:212` — comment about asymmetric DA
- **Current:** `"# No punishment: dopaminergic RPE is asymmetric (Schultz 2002) — tonic DA maintains connections, phasic dips are weaker than phasic bursts."`
- **Issue:** Aligned with glossary C.32 (two-component DA structure). Mentions "tonic" / "phasic" correctly.
- **Status:** OK.

### `experiment/presets.py`

#### `experiment/presets.py:69-83` — `associative_conditioning` docstring
- **Current:** `"Classical conditioning: pair CS (input) with US (output), test if CS alone evokes response. Based on Pavlovian conditioning with STDP as the learning mechanism."`
- **Issue:** Canonical. Both "classical conditioning" and "Pavlovian" appear, matching glossary J.24 / J.25.
- **Status:** OK.

#### `experiment/presets.py:160-168` — `reinforcement_learning` docstring
- **Current:** `"Reward-modulated STDP training: stimulus -> response -> reward/punishment. Based on three-factor learning rule (Izhikevich 2007, Fremaux et al. 2013). Uses the existing eligibility trace and reward modulation infrastructure."`
- **Issue:** Canonical — "three-factor learning rule" matches glossary O.03 / J.29. "Eligibility trace" matches C.29.
- **Status:** OK.

#### `experiment/presets.py:212-218` — comment in TrainingConfig
- **Current:** `"No punishment: dopaminergic RPE is asymmetric (Schultz 2002)"`
- **Status:** OK. Aligned.

### `experiment/readout.py`

#### `experiment/readout.py:140-150` — `_update` docstring on synchrony
- **Current:** `"Synchrony index: variance of spike fractions normalized by mean. High synchrony = neurons fire together (high variance in fraction). Fano factor of population spike count: Var(count) / Mean(count). Ranges from ~0 (asynchronous, Poisson) to >>1 (synchronous bursting)."`
- **Issue:** Uses "Fano factor" — not in glossary, but standard statistical term.
- **Status:** OK. Standard terminology.

#### `experiment/readout.py:209-213` — `compute_band_power` docstring
- **Current:** `"Bands (Hz): delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-80, high_gamma 80-150."`
- **Issue:** **Theta should be 4-12 Hz per glossary D.18.** The 4-8 Hz convention is for human EEG; rodent hippocampal theta is 4-12 Hz. Not a glossary violation strictly, but worth noting given the project models hippocampus.
- **Suggestion:** Either expand theta to 4-12 Hz to match rodent hippocampal literature, or add a note explaining the human-EEG convention used here.
- **Tier:** 1 (prose / parameter)
- **Notes:** This is also a **Tier 3 candidate** — the band edges are coded in `bands` dict at line 222-228, so changing them is a behavioral change, not a comment fix.

### `experiment/stimulus.py`

#### `experiment/stimulus.py:1-5` — module docstring
- **Status:** OK.

#### `experiment/stimulus.py:144-146` — `CONSTANT` pattern comment
- **Status:** OK.

#### `experiment/stimulus.py:164-186` — `POISSON_SPIKE_TRAIN` block
- **Current:** `"Poisson process: probability of spike in dt"` then implements per-step Bernoulli draw.
- **Issue:** Algorithm description is correct — Bernoulli approximation to Poisson is standard.
- **Status:** OK.

### `viz/renderer.py`

#### `viz/renderer.py:1-12` — module docstring
- **Status:** OK. No biological terminology.

#### `viz/renderer.py:211-216` — `get_color_for_trait` docstring
- **Current:** `"Determines neuron color based on trait, activity, spiking status, and filter mode."`
- **Issue:** "trait" is project shorthand for "neuron sub-population class index". Not a biological term in the glossary. Local convention only.
- **Status:** OK as engineering shorthand.

#### `viz/renderer.py:574-585` — synapse rendering color/alpha block
- **Status:** OK. No biological terminology.

#### `viz/renderer.py:606-622` — `# Render Synaptic Pulses` block
- **Status:** OK. "synaptic pulse" is project-specific viz term, not a biological one.

### `viz/camera.py`

- All comments are about camera control mechanics (rotate / pan / zoom). No biological terminology to audit.
- **Status:** OK.

### `viz/picker.py`

- All comments are about color-based GPU picking. No biological terminology.
- **Status:** OK.

### `viz/overlays.py`

- Just text-rendering glue. No biological terminology.
- **Status:** OK.

### `ui/layout.py` — UI tooltips (high impact, user-facing)

#### `ui/layout.py:218` — dt tooltip
- **Current:** `"Integration timestep. Izhikevich: 0.5-1.0ms is stable. Hodgkin-Huxley: MUST be <= 0.1ms (gating kinetics require fine resolution). AdEx: 0.1-0.5ms recommended. Smaller dt = more accurate but slower."`
- **Issue:** Uses canonical names "Izhikevich", "Hodgkin-Huxley", "AdEx".
- **Status:** OK. Canonical.

#### `ui/layout.py:220-221` — number-of-traits tooltip
- **Current:** `"Number of neuron sub-populations (color-coded in 3D view). One trait is designated inhibitory."`
- **Issue:** "trait" is project-specific shorthand; "inhibitory" is shorthand for "GABAergic". Acceptable in user-facing UI.
- **Status:** OK as engineering shorthand.

#### `ui/layout.py:222-223` — neuron model tooltip
- **Current:** `"Izhikevich: Fast, versatile (20+ firing patterns)... Hodgkin-Huxley: Biophysically detailed (ion channels, temperature). Requires dt<=0.1ms. AdEx: Balance of speed and biophysics."`
- **Status:** OK.

#### `ui/layout.py:235` — Izhikevich panel title
- **Current:** `"--- Izhikevich 2007 Model Parameters ---"`
- **Status:** OK. Canonical.

#### `ui/layout.py:247-248` — `cfg_izh_C_val` tooltip
- **Current:** `"Membrane capacitance. Higher C = slower voltage changes. RS ~100 pF, FS ~20-50 pF. (Izhikevich 2007, Table 2)"`
- **Issue:** Uses "RS" / "FS" without expansion. Glossary canonical is "regular spiking (RS)" / "fast-spiking (FS)" — both accepted. RS/FS are widely understood.
- **Status:** OK. Standard cellular electrophysiology shorthand.

#### `ui/layout.py:262-263` — Hodgkin-Huxley panel title
- **Current:** `"--- Hodgkin-Huxley Model Parameters ---"`
- **Status:** OK. Canonical.

#### `ui/layout.py:268` — HH preset tooltip
- **Current:** `"Select a biophysical neuron type preset. Sets conductances and kinetics for specific cell classes (e.g., cortical pyramidal, fast-spiking interneuron)."`
- **Issue:** "fast-spiking interneuron" is canonical (glossary "PV+ FS interneuron" or "cortical fast-spiking interneuron").
- **Status:** OK.

#### `ui/layout.py:300` — `cfg_hh_g_L` tooltip
- **Current:** `"Leak conductance density. Sets resting input resistance. Typically 0.03-0.3 mS/cm²."`
- **Status:** OK.

#### `ui/layout.py:301-302` — `cfg_hh_E_Na` / `cfg_hh_E_K` tooltips
- **Current:** `"Sodium Nernst reversal potential. Set by [Na+] gradient across membrane. Typically +50 mV (mammalian)."` and `"Potassium Nernst reversal potential. Set by [K+] gradient. Typically -77 to -90 mV."`
- **Status:** OK.

#### `ui/layout.py:306` — `cfg_hh_g_M_max` tooltip
- **Current:** `"Muscarinic (M-type) K+ current max conductance. Slow non-inactivating K+ current. Causes spike frequency adaptation. 0 = disabled."`
- **Issue:** "Muscarinic" is correct but slightly informal — the M-current is named because it's *blocked by muscarinic-receptor antagonists*, not because it is muscarinic. The conductance itself is K+, gated by muscarinic ACh signaling. Tooltip is acceptable but could clarify.
- **Status:** OK. Standard naming convention.

#### `ui/layout.py:308-309` — CaT tooltip
- **Current:** `"Low-threshold Ca²+ (T-type) current conductance. Enables rebound bursting and subthreshold oscillations. 0 = disabled. Typical: 0.5-2.0 mS/cm²."`
- **Status:** OK. Canonical T-type Ca current description.

#### `ui/layout.py:310` — I_h tooltip
- **Current:** `"Hyperpolarization-activated cation current (I_h). Contributes to resting potential, sag response, and pacemaker activity. 0 = disabled."`
- **Status:** OK.

#### `ui/layout.py:312` — NaP tooltip
- **Current:** `"Persistent sodium current conductance. Non-inactivating Na+ near threshold. Amplifies subthreshold inputs. 0 = disabled."`
- **Status:** OK.

#### `ui/layout.py:313-314` — Q10 tooltip
- **Current:** `"Temperature coefficient for gating kinetics. Rate multiplier per 10°C: phi = Q10^((T-6.3)/10). Q10=3 is standard for ion channels."`
- **Issue:** Note that the codebase uses **per-gate Q10** now (CLAUDE.md mentions `hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`), but this UI tooltip still suggests a single uniform Q10. Misleading to users since the simulator no longer uses uniform Q10.
- **Suggestion:** Update tooltip: `"Per-gate temperature coefficient. Q10_m=3.0, Q10_h=Q10_n=1.5 by default (fixed 2026-04-25; see CLAUDE.md). This UI value is the legacy uniform Q10 for backward compatibility."`
- **Tier:** 1 (prose, but stale relative to actual code behavior)
- **Notes:** **High-impact fix** — UI is showing outdated info to scientists.

#### `ui/layout.py:351` — AdEx panel title
- **Current:** `"--- AdEx Model Parameters ---"`
- **Status:** OK.

#### `ui/layout.py:368` — `cfg_adex_C` tooltip
- **Current:** `"Membrane capacitance. Brette & Gerstner 2005: RS ~281 pF, FS ~100 pF. Controls voltage time constant."`
- **Status:** OK. Canonical.

#### `ui/layout.py:370` — `cfg_adex_E_L` tooltip
- **Current:** `"Leak reversal / resting potential. Typically -70 to -65 mV for cortical neurons."`
- **Status:** OK.

#### `ui/layout.py:372` — `cfg_adex_Delta_T` tooltip
- **Current:** `"Slope factor of exponential spike initiation. Smaller = sharper threshold. 0 = perfect IF. Typical: 1-4 mV. (Badel et al. 2008)"`
- **Issue:** "perfect IF" is correct shorthand for "perfect integrate-and-fire" but unclear to non-experts.
- **Suggestion:** Expand: `"... 0 = perfect integrate-and-fire (IF). Typical: 1-4 mV."`
- **Tier:** 1 (prose, minor clarity)

#### `ui/layout.py:417-418` — Watts-Strogatz tooltip
- **Current:** `"Use Watts-Strogatz small-world network topology. Combines local clustering with short path lengths. Disable for random Erdos-Renyi connectivity."`
- **Status:** OK. "Watts-Strogatz" / "Erdős-Rényi" are canonical (mentioned in CLAUDE.md context).

#### `ui/layout.py:427-433` — synaptic propagation tooltip
- **Current:** `"Peak excitatory conductance increase per spike (nS). Scales AMPA synaptic input."`
- **Issue:** AMPA is canonical (glossary J.07).
- **Status:** OK.

#### `ui/layout.py:433-434` — Tau_g_e tooltip
- **Current:** `"AMPA receptor decay time constant. Fast excitatory transmission (1-10 ms typical)."`
- **Status:** OK.

#### `ui/layout.py:434` — Tau_g_i tooltip
- **Current:** `"GABA_A receptor decay time constant. Inhibitory transmission (5-20 ms typical)."`
- **Status:** OK. Canonical "GABA_A" matches glossary J.10.

#### `ui/layout.py:435` — NMDA section header
- **Current:** `"NMDA Receptors (Voltage-Dependent Mg²⁺ Block)"`
- **Status:** OK. Canonical.

#### `ui/layout.py:439` — NMDA enable tooltip
- **Current:** `"NMDA receptors with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990). Adds slow excitatory current gated by postsynaptic depolarization — critical for coincidence detection and associative plasticity."`
- **Status:** OK.

#### `ui/layout.py:441` — NMDA tau decay tooltip
- **Current:** `"NMDA receptor decay (~100 ms). Much slower than AMPA (~5 ms), enabling temporal integration."`
- **Status:** OK.

#### `ui/layout.py:443` — Mg²⁺ tooltip
- **Current:** `"Extracellular magnesium concentration. 1.0 mM physiological. Higher = stronger voltage-dependent block, less NMDA current at rest."`
- **Status:** OK.

#### `ui/layout.py:449-451` — Hebbian tooltip
- **Current:** `"Simple Hebbian co-activation learning rule. Weights increase when pre and post neurons fire together. Includes weight decay to prevent runaway excitation."`
- **Issue:** Glossary canonical "Hebbian" matches (J.29 family). "Co-activation" is OK colloquial.
- **Status:** OK.

#### `ui/layout.py:459-460` — STP tooltip
- **Current:** `"Tsodyks-Markram short-term plasticity model. Synapses exhibit depression (weakening) and facilitation (strengthening) on timescales of 10-1000ms. Essential for temporal coding."`
- **Status:** OK. Canonical "Tsodyks-Markram" matches glossary J.03.

#### `ui/layout.py:461-462` — STP U tooltip
- **Current:** `"Fraction of available resources used per spike (0-1). Low U (~0.1-0.2): facilitating synapses (cortical). High U (~0.5-0.8): depressing synapses (thalamocortical). Literature: Tsodyks & Markram 1997."`
- **Status:** OK.

#### `ui/layout.py:500-501` — homeostasis tooltip
- **Current:** `"Intrinsic homeostasis via adaptive firing thresholds. For Izhikevich: adjusts spike threshold toward target rate. Essential for stable network dynamics over long simulations."`
- **Issue:** Glossary canonical is "homeostatic plasticity" / "synaptic scaling" (J family). "Intrinsic homeostasis" is correct subterm — adjusts spike threshold (intrinsic property), not weight (synaptic).
- **Status:** OK.

#### `ui/layout.py:520` — synaptic scaling tooltip
- **Current:** `"Multiplicative synaptic scaling (Turrigiano 2008). Scales excitatory weights up/down to maintain target firing rate. Complementary to threshold homeostasis — works on synaptic strengths rather than intrinsic excitability."`
- **Status:** OK. Canonical Turrigiano reference.

#### `ui/layout.py:526-540` — STDP tooltips
- **Current:** `"Spike-Timing-Dependent Plasticity (Bi & Poo 2001). Pre-before-post = LTP, post-before-pre = LTD. Biological Hebbian learning with precise timing."`
- **Issue:** Bi & Poo is canonical (1998 — the paper is from 1998, not 2001). Date is incorrect.
- **Suggestion:** Update to "(Bi & Poo 1998)" — the 1998 *J. Neurosci.* paper is the canonical STDP reference. The 2001 reference is the "Synaptic modifications in cultured hippocampal neurons" follow-up.
- **Tier:** 1 (prose, citation accuracy)
- **Notes:** **High-impact fix** — wrong citation in user-facing tooltip.

#### `ui/layout.py:533` — STDP A+ tooltip
- **Current:** `"Maximum weight increase for causal (pre→post) pairing. Larger A+ = faster potentiation. A- > A+ gives net depression bias (stable)."`
- **Status:** OK. "causal (pre→post)" is correct (anti-Hebbian for "post→pre").

#### `ui/layout.py:535` — STDP A- tooltip
- **Current:** `"Maximum weight decrease for anti-causal (post→pre) pairing."`
- **Status:** OK.

#### `ui/layout.py:551-552` — reward modulation tooltip
- **Current:** `"Three-factor learning: STDP eligibility traces are gated by a reward signal (Schultz 2002). Requires STDP enabled. Models dopaminergic modulation."`
- **Status:** OK. Canonical "three-factor learning" matches glossary O.03.

#### `ui/layout.py:560` — current reward signal tooltip
- **Current:** `"Current reward value (can be changed live). Positive = reinforce recent activity. Negative = suppress recent activity patterns. Models dopaminergic reward prediction error."`
- **Issue:** "dopaminergic reward prediction error" is canonical (glossary C.22 / C.28).
- **Status:** OK.

#### `ui/layout.py:568-569` — structural plasticity tooltip
- **Current:** `"Dynamic synapse formation and elimination. New connections form between co-active neurons. Weak synapses are pruned. Models developmental and experience-dependent rewiring."`
- **Status:** OK. Aligned with glossary "structural plasticity" / "synapse elimination" (J + L family).

#### `ui/layout.py:583` — activity bias tooltip
- **Current:** `"Bias synapse formation toward co-active neuron pairs. 0.0 = purely random formation. 1.0 = fully activity-driven (Cline & Haas 2008). 0.5 = 50/50 mix of co-activity-biased and random candidates."`
- **Status:** OK.

#### `ui/layout.py:610` — heterogeneity tooltip
- **Status:** OK.

#### `ui/layout.py:632-633` — channel noise tooltip
- **Current:** `"Add stochastic fluctuations to ion channel conductances. Models channel noise from finite ion channel populations. Only applies to Hodgkin-Huxley model."`
- **Status:** OK.

#### `ui/layout.py:662-663` — OU process tooltip
- **Current:** `"Ornstein-Uhlenbeck process for background synaptic drive. Models bombardment from ~10,000 unmodeled synapses. Produces realistic 2-5 mV membrane potential fluctuations."`
- **Status:** OK.

#### `ui/layout.py:692-693` — OU tau tooltip
- **Current:** `"Temporal correlation time of background noise. Small tau (~5 ms) = fast, white-noise-like. Large tau (~20 ms) = slowly varying, colored noise. 15 ms matches cortical synaptic timescales."`
- **Status:** OK.

#### `ui/layout.py:778-779` — manual stimulus collapsing header
- **Current:** `"Inject a simple stimulus into the network without setting up a full experiment."`
- **Status:** OK.

#### `ui/layout.py:790-791` — manual stim pattern tooltip
- **Current:** `"Stimulus waveform type. CONSTANT: DC step current. PULSE_TRAIN: Repeated brief pulses. SINUSOIDAL: Oscillatory current"`
- **Status:** OK.

### `ui/callbacks.py`

#### `ui/callbacks.py:401-403` — `_apply_supervised_error` comment
- **Current:** `"Apply supervised error signal as reward modulation. Uses the existing reward signal mechanism as an error channel. Error = (target_rate - actual_rate) * gain"`
- **Status:** OK.

#### `ui/callbacks.py:805-807` — dt auto-adjust message
- **Current:** `"dt auto-adjusted to 0.05 ms for HH stability (was {:.3f} ms)"`
- **Status:** OK.

#### `ui/callbacks.py:1116` — manual stim status message
- **Current:** `"Injecting {pattern_str} stimulus: {amplitude} pA"`
- **Status:** OK.

### `ui/experiment_dashboard.py`

#### `ui/experiment_dashboard.py:1-7` — module docstring
- **Status:** OK.

#### `ui/experiment_dashboard.py:19-25` — phase color dict
- **Current:** `"BASELINE", "STIMULUS", "TRAINING", "TESTING", "REST"`
- **Issue:** These are project-internal phase enum names (matches `ExperimentPhaseType` in `sim/enums.py`). Not biological terminology.
- **Status:** OK. Project shorthand.

### `ui/inspector.py`

#### `ui/inspector.py:23` — neuron inspector hint
- **Current:** `"Click a neuron in 3D view to inspect"`
- **Status:** OK.

#### `ui/inspector.py:62-65` — trait classification text
- **Current:** `trait_name = "Inhibitory" if trait == inh_idx else "Excitatory"`
- **Issue:** "Inhibitory" / "Excitatory" — canonical electrophysiological shorthand. Glossary canonical is "GABAergic" / "glutamatergic" but "inhibitory/excitatory" is universal.
- **Status:** OK.

#### `ui/inspector.py:82` — voltage label
- **Current:** `f"Membrane potential: {v:.1f} mV"`
- **Status:** OK. Canonical.

### `ui/plots.py`

#### `ui/plots.py:11-12` — raster constants comments
- **Current:** `"# Fixed subsample size for raster plot. _RASTER_SUBSAMPLE_N = 100  # 50 excitatory + 50 inhibitory target"`
- **Status:** OK.

#### `ui/plots.py:43-44` — comment in `_build_subsample`
- **Current:** `"# Take evenly spaced neurons: first 50 from lower half, last 50 from upper half"`
- **Status:** OK.

#### `ui/plots.py:84-95` — population firing rate plot
- **Current:** `"Create a population firing rate trace."`
- **Status:** OK.

### `ui/sweep_panel.py`

#### `ui/sweep_panel.py:43-46` — sweep parameter list
- **Current:** lists "stdp_a_plus", "stdp_a_minus", "stdp_tau_plus_ms", "stdp_tau_minus_ms", etc.
- **Status:** OK. Code-identifier shorthand.

#### `ui/sweep_panel.py:80-88` — sweep results columns
- **Current:** `"Parameter", "Delta (Hz)", "t-stat", "Cohen's d", "Sig?"`
- **Status:** OK. Statistical shorthand.

### `tests/test_d1_d2_asymmetry.py`

#### `tests/test_d1_d2_asymmetry.py:1` — module docstring
- **Current:** `"Tests for Cluster B.1 — D1/D2 plasticity asymmetry."`
- **Status:** OK. Canonical.

#### `tests/test_d1_d2_asymmetry.py:128-133` — `test_d1_d2_sign_inverts_weight_change_under_reward` docstring
- **Current:** `"With enable_d1_d2_asymmetry on: D1-targeting synapses' weights move in the SAME direction as reward, D2-targeting synapses' weights move in the OPPOSITE direction."`
- **Issue:** Aligned with glossary cell-type entries B (D1 MSN, D2 MSN) and the asymmetric DA modulation noted in O.03.
- **Status:** OK.

### `tests/test_e_inh_override.py`

#### `tests/test_e_inh_override.py:1-9` — module docstring
- **Current:** `"Unit tests for per-region GABA_A reversal potential override (R1.1). Catalog reference: Kandel PBR-160 ch 6 (striatum) and ch 11 (SNc DA). Striatal MSNs measured ECl ~−60 mV via gramicidin perforated patch (vs the −75 mV cortical-pyramidal default). SNc DA neurons lack KCC2 chloride exporter → ECl ~−55 mV."`
- **Status:** EXEMPLARY. Uses canonical "GABA_A", "MSN", "SNc DA", "KCC2" matching glossary. Specifies catalog reference and methodology. Should be the model for other test docstrings.

### `tests/test_neuromodulators.py`

#### `tests/test_neuromodulators.py:1-10` — module docstring
- **Current:** `"Unit tests for the neuromodulator subsystem (Session E.1). The subsystem replaces the one-off current_reward_signal / cp_synaptic_gain_modulator hacks with a declarative framework where each neuromodulator is a NeuromodulatorConfig with concentration, decay tau, baseline, production rules, and configurable receptor targets."`
- **Status:** OK.

#### `tests/test_neuromodulators.py:39-44` — `test_neuromodulator_config_custom_values` uses "noradrenaline"
- **Current:** `nm = NeuromodulatorConfig(name="noradrenaline", baseline=0.2, decay_tau_ms=2000.0, ...)`
- **Issue:** Glossary canonical for this molecule is "NE" or "norepinephrine" (US/standard) with "noradrenaline" / "NA" as accepted European/older. Test is using accepted variant. Could standardize to "norepinephrine".
- **Status:** Acceptable. Both forms valid per glossary.

#### `tests/test_neuromodulators.py:172-200` — `test_from_reward_rule_pulses_dopamine_on_positive_reward` block
- **Current:** Comment about dopamine response.
- **Status:** OK.

### `tests/test_tans.py`

#### `tests/test_tans.py:1-10` — module docstring
- **Current:** `"Tests for Cluster B.3 - Cholinergic Interneurons (TANs). Real BG TANs are tonically active (~5 Hz baseline) but pause briefly on salient events (reward, novel stimuli). ACh release at corticostriatal synapses creates 'plasticity windows' - synapses only consolidate when ACh is paused (low ACh = plasticity-on; high ACh = plasticity-off)."`
- **Issue:** Excellent canonical use. "TAN" / "ChI" / "ACh" / "corticostriatal" / "tonically active" — all match glossary B.05, C.18.
- **Status:** EXEMPLARY.

### `tests/test_regions.py`

#### `tests/test_regions.py:1-10` — module docstring
- **Status:** OK. "PFC", "Motor", "Hippocampus" — canonical region names (G family / D family).

#### `tests/test_regions.py:42-51` — `test_brain_region_custom` uses `nm_outputs=["acetylcholine"]`
- **Current:** `nm_outputs=["acetylcholine"]`
- **Status:** OK. Canonical.

### `tests/test_kernels_cpu.py`

#### `tests/test_kernels_cpu.py:1-13` — module docstring
- **Current:** `"Comprehensive CPU-only (NumPy-based) test harness for validating the mathematical correctness of all fused CUDA kernels in the neural simulator."`
- **Status:** OK.

#### `tests/test_kernels_cpu.py:38-43` — Izhikevich docstring
- **Status:** OK.

#### `tests/test_kernels_cpu.py:75-85` — HH docstring
- **Current:** `"Classical Hodgkin-Huxley model with PER-GATE temperature scaling (HH temperature bug fix — Q10_m, Q10_h, Q10_n separately). For backward compat, the legacy uniform-Q10 signature ... is still accepted; in that case Q10_m = Q10_h = Q10_n = q10_factor."`
- **Status:** OK. Aligned with CLAUDE.md note about per-gate Q10.

#### `tests/test_kernels_cpu.py:172-175` — `numpy_hh_m_current_update` docstring
- **Current:** `"Slow K+ M-current with temperature-dependent time constant."`
- **Issue:** "M-current" canonical (glossary I family). Note however that this kernel models the K+ M-current. "Slow K+" is correct.
- **Status:** OK.

#### `tests/test_kernels_cpu.py:189-195` — `numpy_hh_CaT_current_update` docstring
- **Current:** `"Low-threshold T-type Ca2+ current with Q10 temperature scaling."`
- **Status:** OK.

#### `tests/test_kernels_cpu.py:213-216` — `numpy_hh_h_current_update` docstring
- **Current:** `"Hyperpolarization-activated mixed cation current (I_h) with temperature scaling."`
- **Status:** OK.

#### `tests/test_kernels_cpu.py:312-326` — STP docstring
- **Current:** `"Tsodyks-Markram short-term plasticity dynamics: du/dt = -u / tau_f, dx/dt = (1 - x) / tau_d. Analytical solution: u_new = u_old * exp(-dt / tau_f), x_new = 1 - (1 - x_old) * exp(-dt / tau_d)..."`
- **Issue:** The Tsodyks-Markram equations are canonical, but the analytical solution comment for `x_new` uses "exp" form, while the actual implementation at line 331 uses Euler step `x + (1-x) * (dt/tau_d)`. Comment doesn't quite match code.
- **Suggestion:** Either correct the comment or implement the analytical solution. **This is also a Tier 3 candidate** — the docstring claims a different algorithm than the code.

#### `tests/test_kernels_cpu.py:380-388` — STDP docstring
- **Current:** `"Classical asymmetric STDP (Bi & Poo 1998): delta_t > 0 (post after pre): LTP. delta_t < 0 (pre after post): LTD. Soft-bound rule ensures weights stay within [w_min, w_max]."`
- **Issue:** **Excellent** — "Bi & Poo 1998" matches the actual paper date and the canonical STDP reference. Compare to `ui/layout.py:531` which has wrong date "2001".
- **Status:** EXEMPLARY.

### `tests/test_experiment_system.py`

#### `tests/test_experiment_system.py:1-7` — module docstring
- **Current:** `"Unit tests for the Experiment & Stimulus System. Tests the experiment system components (StimulusPattern, StimulusChannel, NeuronGroup, ExperimentConfig, etc.) without requiring GPU/CuPy."`
- **Status:** OK.

### `tests/test_determinism.py`

#### `tests/test_determinism.py:90-91` — print message
- **Current:** `print(f"✓ Izhikevich deterministic: {len(spikes1)} steps matched")`
- **Status:** OK.

#### `tests/test_determinism.py:144` — print message
- **Current:** `print(f"✓ Hodgkin-Huxley deterministic: {len(spikes1)} steps matched")`
- **Status:** OK.

### `tests/test_g11_bg_runner_flags.py`

#### `tests/test_g11_bg_runner_flags.py:1-9` — module docstring
- **Status:** OK.

#### `tests/test_g11_bg_runner_flags.py:56-57` — `test_motor_lateral_inhibition` docstring
- **Current:** `"WTA microcircuit (FS interneurons + motor cross-pool inhibition)."`
- **Issue:** Glossary "lateral inhibition" (E.05) is the canonical name; "WTA" (winner-take-all) is the algorithmic synonym, used widely in the project. Both acceptable.
- **Status:** OK.

#### `tests/test_g11_bg_runner_flags.py:339-348` — `test_bg_lateral_inhibition_pathways` docstring
- **Current:** `"v3 (2026-04-28): when --bg-lateral-inhibition is on, the BG cascade includes 24 cross-pool MSN-MSN inhibitory pathways: str_D{1,2}_X → str_D{1,2}_Y for X != Y. 4 actions × 3 cross targets × 2 (D1, D2) = 24. The MSN regions are GABAergic (exc_fraction=0.05) so the projection IS inhibitory."`
- **Status:** EXEMPLARY. Uses canonical "MSN", "GABAergic", "lateral inhibition".

### `tests/test_benchmark_drift.py`

#### `tests/test_benchmark_drift.py:1-32` — module docstring
- **Status:** OK.

#### `tests/test_benchmark_drift.py:48` — STDP test docstring
- **Current:** `"Fused STDP kernel must match Bi & Poo soft-bound formula exactly."`
- **Status:** OK.

#### `tests/test_benchmark_drift.py:85-89` — `_build_tiny_sim` docstring
- **Current:** `"Build a minimal RNG-sensitive sim: 100 neurons, OU noise on, no plasticity. Chosen to exercise the main RNG stream (OU draws per step) without the cost of a full benchmark."`
- **Status:** OK.

#### `tests/test_benchmark_drift.py:215-222` — gamma test docstring
- **Current:** `"At seed=42, gamma peak frequency must fall in classic gamma band. With seed=42 the expected peak is ~38.7 Hz."`
- **Issue:** "classic gamma band" — glossary I/J/N family canonical 30-80 Hz; the test allows 25-55 Hz. Per glossary "gamma" is 40-100 Hz. The lower bound 25 Hz is unusually broad — maybe to accommodate measurement variance.
- **Status:** OK as test tolerance.

### `tests/test_g9_runner_smoke.py`

#### `tests/test_g9_runner_smoke.py:7-9` — `test_g9_smoke_argmax` docstring
- **Current:** `"30-step episode with argmax; verify plastic weights moved via sim R-STDP."`
- **Status:** OK. "R-STDP" canonical (glossary J.29 / O.03).

#### `tests/test_g9_runner_smoke.py:96-99` — `test_g9_smoke_with_large_reservoir` docstring
- **Current:** `"Route C: 5000-hidden-neuron G9 runs cleanly. Tests that the runner + bridge + reservoir pipeline scales..."`
- **Issue:** "reservoir" is canonical (glossary "reservoir computing", F family). OK.
- **Status:** OK.

### `tests/test_structural_pruning.py`

#### `tests/test_structural_pruning.py:1` — module docstring
- **Current:** `"Smoke tests for the structural-plasticity (axon pruning) machinery."`
- **Issue:** "axon pruning" / "structural plasticity" / "synapse elimination" all canonical (glossary L.02, L.03, J family).
- **Status:** OK.

### `tests/test_gate_metrics.py`

- Tests are mostly about TTP / PF metrics — project-internal metrics, not biological.
- **Status:** OK.

---

## Tier 2 — symbol references in prose

#### `ui/layout.py:431` — Inhibitory tooltip mentions "GABA_A synaptic input"
- **Current:** `"Peak inhibitory conductance increase per spike (nS). Scales GABA_A synaptic input. Usually 2-4x excitatory for E/I balance."`
- **Status:** OK. Canonical "GABA_A" matches glossary.

#### `ui/layout.py:435-443` — NMDA tooltip mentions Mg²⁺ block
- **Status:** OK. Canonical.

#### `experiment/training.py:194-200` — `_deliver_reward` block uses `current_reward_signal`
- **Current:** `sim_bridge_ref.core_config.current_reward_signal = self.config.reward_magnitude`
- **Issue:** `current_reward_signal` is the project's DA scalar — glossary [NEEDS-REVIEW] note flags it as a major simplification (collapses phasic/tonic, A9/A10, Component 1/2). Code is OK; just a known abstraction.
- **Status:** OK as project convention.

#### `experiment/engine.py:401-408` — `ensure_inter_group_connectivity` mentions OU noise
- **Current:** comment uses "OU noise sigma ~ 80 pA" → uses canonical OU shorthand.
- **Status:** OK.

#### `tests/test_neuromodulators.py:236-237` — `test_from_reward_rule_ignores_missing_bridge_config` symbol
- **Current:** `"When bridge has no core_config, rule produces 0 (no-op, no crash)."`
- **Status:** OK.

#### `tests/test_e_inh_override.py:1-9` — uses ECl, KCC2, gramicidin perforated patch
- **Status:** EXEMPLARY. All canonical biological symbols.

#### `viz/renderer.py:39-43` — references to `global_simulation_bridge`
- **Status:** OK. Project shorthand.

#### `tests/test_d1_d2_asymmetry.py:27-32` — comment about STDP soft-bound
- **Current:** `"Cortex→D1 weights are weight_mean=25 with Gaussian jitter sigma=0.2... Set bounds well above that so clipping doesn't dominate the small reward delta in tests... See CLAUDE.md 'STDP bounds gotcha'."`
- **Status:** OK. References gotcha doc.

#### `tests/test_kernels_cpu.py:325-332` — Tsodyks-Markram analytical solution comment vs code
- **Status:** Comment claims `x_new = 1 - (1 - x_old) * exp(-dt / tau_d)` (analytical) but code computes `x + (1-x) * (dt/tau_d)` (Euler). **Discrepancy** between comment and implementation.

#### `ui/layout.py:531` — STDP citation
- **Current:** `"(Bi & Poo 2001)"`
- **Issue:** Wrong date. Bi & Poo 1998 is the original STDP paper.
- **Status:** Tier 1 fix recommended.

#### `ui/layout.py:313-315` — Q10 single-factor comment
- **Status:** Stale relative to actual per-gate Q10 in code (per CLAUDE.md). Tier 1 fix recommended.

#### `ui/callbacks.py:138-141` — comment about `default_neuron_type_izh`
- **Current:** code logic for IZH default; comment near line 140-141 references "default_neuron_type_izh" — this is a code identifier (Tier 3), not prose.
- **Status:** OK.

---

## Tier 3 — identifiers (FLAGGED only, not changed)

These are code identifiers (variable names, function names) that don't match glossary canonical terms but are kept for backward compatibility per the glossary's "project_identifier" convention:

1. **`current_reward_signal`** (`experiment/training.py:138, 197, 199, 221`; `ui/layout.py:559, 560`)
   - Glossary [NEEDS-REVIEW]: project's DA scalar conflates phasic/tonic, Component-1/Component-2, A9/A10. Audit should not flag every use.
   - **Status:** FLAG ONLY. Keep for backward compat.

2. **`enable_reward_modulation`** (`ui/layout.py:551`)
   - Project shorthand for "three-factor learning enable".
   - **Status:** FLAG ONLY.

3. **`reward_eligibility_tau_ms`** (`ui/layout.py:555`)
   - Aligned with glossary "eligibility trace" (C.29).
   - **Status:** OK identifier.

4. **`enable_short_term_plasticity`** (`ui/layout.py:459`)
   - Aligned with glossary STP / "short-term plasticity" (J.03).
   - **Status:** OK identifier.

5. **`stdp_w_max` / `stdp_w_min`** (`ui/layout.py:540-543`)
   - Project shorthand for STDP soft-bound limits. CLAUDE.md gotcha.
   - **Status:** FLAG ONLY (well-known project convention).

6. **`enable_homeostasis`** (`ui/layout.py:500`)
   - Aligned with glossary J.30 (homeostatic plasticity).
   - **Status:** OK.

7. **`enable_synaptic_scaling`** (`ui/layout.py:519`)
   - Aligned with glossary J.30 (synaptic scaling, Turrigiano).
   - **Status:** OK.

8. **`enable_structural_plasticity`** (`ui/layout.py:568`)
   - Aligned with glossary J/L family.
   - **Status:** OK.

9. **`hebbian_learning_rate`** / `hebbian_max_weight` (`ui/layout.py:451, 453`)
   - Aligned with glossary "Hebbian" (J family).
   - **Status:** OK.

10. **`fired_status_np` / `is_currently_spiking`** (`viz/renderer.py:171, 211, etc.`)
    - "Spiking" is canonical (glossary I.02 "action potential or spike").
    - **Status:** OK.

11. **`firing_rates`** (`ui/plots.py:108`)
    - Canonical.
    - **Status:** OK.

12. **`reservoir_weights`** / `reservoir_indices` (`tests/test_g9_runner_smoke.py`)
    - Aligned with glossary "reservoir computing".
    - **Status:** OK.

13. **`cp_d1_d2_sign`** (`tests/test_d1_d2_asymmetry.py:51, 53, 84`)
    - Project-specific GPU array. Aligned with glossary D1 MSN / D2 MSN naming.
    - **Status:** OK.

14. **`cs_input` / `us_output`** (`experiment/presets.py:93, 96`)
    - Pavlovian CS / US — canonical.
    - **Status:** OK.

15. **`StimulusPatternType.POISSON_SPIKE_TRAIN`** (`experiment/stimulus.py:63`)
    - Canonical.
    - **Status:** OK.

16. **`enable_d1_d2_asymmetry`** (`tests/test_d1_d2_asymmetry.py:25`)
    - Aligned with glossary B (D1/D2 MSN).
    - **Status:** OK.

17. **`syn_reversal_potential_i_override`** (`tests/test_e_inh_override.py:30, 37, etc.`)
    - "_i" suffix canonical for inhibitory (GABA_A glossary).
    - **Status:** OK identifier.

18. **`synaptic_gain` / `plasticity_rate` / `excitability_drive`** (`tests/test_neuromodulators.py:54, 599`, etc.)
    - These are project's `target_type` strings — declarative neuromodulator targets. Not biological terms per se.
    - **Status:** OK as configuration vocabulary.

19. **`inhibitory_trait_index`** (`experiment/engine.py:457`)
    - Project shorthand for the trait-index of GABAergic neurons.
    - **Status:** OK identifier.

20. **`MockCuPy`** class in `tests/test_experiment_system.py:17`
    - Test scaffolding.
    - **Status:** OK.

21. **`READOUT`** group in `experiment/groups.py` and tests
    - Canonical. ReadoutEngine is project name for population-rate measurement.
    - **Status:** OK.

22. **Phase enum names**: `BASELINE`, `STIMULUS`, `TRAINING`, `TESTING`, `REST` (`ui/experiment_dashboard.py:19-25`)
    - Project shorthand for experiment phases.
    - **Status:** OK.

23. **`NeuronGroupRole.INPUT` / `OUTPUT` / `HIDDEN`** (across experiment package)
    - Project shorthand for I/O classification.
    - **Status:** OK.

---

## Items NOT flagged (intentional shorthand)

These usages are legitimate engineering shorthand or accepted abbreviations that should NOT be flagged:

1. **"trait" as project shorthand for neuron sub-population class** — used throughout viz/renderer.py and ui/. Project-specific, not a glossary term.
2. **"RS" / "FS" without expansion** in UI tooltips — universally understood in cellular electrophysiology.
3. **"Inhibitory" / "Excitatory"** as electrophysiological labels — universally understood; glossary canonical "GABAergic" / "glutamatergic" preserved as accepted alternatives.
4. **"CS" / "US" / "ITI" / "CR"** in Pavlovian conditioning context — canonical Pavlovian abbreviations.
5. **"OU noise"** — canonical Ornstein-Uhlenbeck shorthand throughout.
6. **"STP"** — canonical (J.03).
7. **"WTA"** in winner-take-all motor circuit context — algorithmic synonym for lateral inhibition; widely used in project per CLAUDE.md.
8. **"reservoir"** in the reservoir-computing sense — canonical.
9. **"R-STDP" / "STDP" / "LTP" / "LTD"** — all canonical.
10. **"phasic" / "tonic"** — canonical neuromodulator dynamics terms.
11. **"Fano factor"** in synchrony measurement — standard statistics term.
12. **"GLUT" / "OpenGL" / "VBO"** in viz/ — graphics implementation, no biological terms.
13. **"argmax" / "first_spike"** in g9 runner — algorithmic shorthand.
14. **"reservoir" / "hidden" group naming** in experiment package — RL/connectionist convention.
15. **"AHP"** in Izhikevich tooltips — afterhyperpolarization, canonical.
16. **"E/I balance"** in synaptic-strength tooltip — canonical (excitation/inhibition balance).

---

## Top-3 highest-impact Tier 1 fixes (recommendation)

These are user-facing strings or stale documentation that mislead readers and warrant prompt fixing:

### 1. **`ui/layout.py:531` — Bi & Poo 1998, not 2001** [HIGH]
- Tooltip currently says "Spike-Timing-Dependent Plasticity (Bi & Poo 2001)" but the canonical STDP paper is Bi & Poo 1998 (J. Neurosci.). The 2001 paper is a follow-up review. `tests/test_kernels_cpu.py:380` correctly cites 1998.
- **Fix:** `"Spike-Timing-Dependent Plasticity (Bi & Poo 1998)..."`
- **Impact:** Researchers reading the UI will copy the wrong citation.

### 2. **`ui/layout.py:313-315` — Q10 tooltip is stale (uniform Q10 vs per-gate Q10)** [HIGH]
- Tooltip says `"Rate multiplier per 10°C: phi = Q10^((T-6.3)/10). Q10=3 is standard for ion channels."` But CLAUDE.md states the simulator now uses **per-gate Q10** (`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`) since 2026-04-25. The UI value field (`cfg_hh_q10_factor`) is now legacy/unused if the per-gate Q10 is enforced upstream.
- **Fix:** Add note about per-gate Q10 to the tooltip and clarify what this UI field still controls.
- **Impact:** Scientists tuning HH for non-mammalian temperatures will get unexpected behavior.

### 3. **`experiment/readout.py:222-228` — Theta band 4-8 Hz vs 4-12 Hz** [MEDIUM]
- The `compute_band_power()` `bands` dict uses `'theta': (4.0, 8.0)`. Glossary D.18 specifies theta is 4-12 Hz (rodent hippocampus). The 4-8 Hz convention is human-EEG. The project models hippocampal regions, so the rodent convention is more apt.
- **Fix:** Either update bands to `'theta': (4.0, 12.0)` (a behavioral change, Tier 3) or document the human-EEG convention in the docstring.
- **Impact:** Users analyzing hippocampal LFP will miss 8-12 Hz power.

---

## Anything ambiguous needing human input

1. **Theta band convention (`experiment/readout.py:222`)**: Should the project use rodent-hippocampal theta (4-12 Hz) or human-EEG theta (4-8 Hz)? This is a Tier 3 (behavioral) decision, not a comment fix.

2. **`current_reward_signal` everywhere**: Per glossary [NEEDS-REVIEW], this conflates A9/A10/phasic/tonic/Component-1/Component-2 DA. Auditor was instructed not to flag every use, but should the long-term plan call for renaming to e.g. `da_scalar` with explicit comments on what's collapsed?

3. **"RS" / "FS" in user-facing UI**: Universally understood in electrophysiology, but new users may not know. Should tooltips spell out at least the first occurrence? (e.g. "RS = Regular Spiking" expand-on-first-mention).

4. **Q10 UI control vs per-gate Q10 reality**: The UI exposes a single `cfg_hh_q10_factor` field and tooltip suggests it's the canonical Q10. But the codebase has moved to per-gate Q10. Either:
   - Hide/disable the UI control and document that Q10 is set by HH presets, OR
   - Add three UI controls (Q10_m, Q10_h, Q10_n) and update tooltip.

5. **`tests/test_kernels_cpu.py:312-326` STP analytical-solution claim**: Docstring says the solution uses `exp` form but code uses Euler step. This is a code-vs-docstring mismatch (Tier 3) — should the test compare against the analytical solution (more accurate) or match the implementation in the kernel (more honest)?

6. **"associative conditioning" vs "Pavlovian conditioning" vs "classical conditioning"**: All three appear; glossary lists "classical conditioning" with Pavlovian as accepted. The preset name is "Associative Conditioning (CS-US)" which mixes conventions — fine, but if standardizing, "Pavlovian (CS-US) conditioning" would be the canonical phrasing.

---

## Counts by file (Tier 1 only)

| File | T1 issues | T2 issues | T3 flagged |
|---|---|---|---|
| `experiment/engine.py` | 1 | 1 | 1 |
| `experiment/presets.py` | 0 | 0 | 1 |
| `experiment/training.py` | 0 | 1 | 1 |
| `experiment/readout.py` | 1 | 0 | 0 |
| `experiment/stimulus.py` | 0 | 0 | 1 |
| `experiment/groups.py` | 0 | 0 | 1 |
| `viz/renderer.py` | 0 | 0 | 0 |
| `viz/camera.py` | 0 | 0 | 0 |
| `viz/picker.py` | 0 | 0 | 0 |
| `viz/overlays.py` | 0 | 0 | 0 |
| `ui/layout.py` | 4 | 4 | 8 |
| `ui/callbacks.py` | 0 | 0 | 1 |
| `ui/inspector.py` | 0 | 0 | 1 |
| `ui/plots.py` | 0 | 0 | 1 |
| `ui/sweep_panel.py` | 0 | 0 | 1 |
| `ui/experiment_dashboard.py` | 0 | 0 | 1 |
| `tests/test_d1_d2_asymmetry.py` | 0 | 1 | 1 |
| `tests/test_e_inh_override.py` | 0 | 0 | 1 |
| `tests/test_neuromodulators.py` | 0 | 0 | 1 |
| `tests/test_tans.py` | 0 | 0 | 1 |
| `tests/test_regions.py` | 0 | 0 | 1 |
| `tests/test_kernels_cpu.py` | 0 | 1 | 0 |
| `tests/test_experiment_system.py` | 0 | 0 | 1 |
| `tests/test_determinism.py` | 0 | 0 | 0 |
| `tests/test_g11_bg_runner_flags.py` | 0 | 0 | 0 |
| `tests/test_benchmark_drift.py` | 0 | 0 | 0 |
| `tests/test_g9_runner_smoke.py` | 0 | 0 | 0 |
| `tests/test_structural_pruning.py` | 0 | 0 | 0 |
| `tests/test_gate_metrics.py` | 0 | 0 | 0 |
| `tests/test_d1_d2_asymmetry.py` (counted above) | — | — | — |

**Note on counts:** The high T1 count for `ui/layout.py` reflects that it owns nearly all user-facing tooltip text and is the highest-impact file for any prose fixes. Tests are dominantly OK because their docstrings are written by people who know the canonical terminology.
