# Biology Buildout Roadmap

This document organizes the ~375 mechanism entries in [`feature-catalog.md`](feature-catalog.md) into a prioritized implementation roadmap for the next 6–18 months. It is a *strategy* document — the catalog is the encyclopedia; this is which pages to act on, in what order, and why.

**Last updated:** 2026-04-29 (post second enrichment pass — added Tier 0 quick wins surfaced by the supplemental texts).

**Sources:** Kandel et al., *Principles of Neural Science*, 6th edition (2021), plus 12 specialty texts (full inventory in [`textbooks/README.md`](textbooks/README.md)) — Marr 1969, Albus 1971, Hesslow 2013, Hesslow & Yeo 2002 chapter (cerebellum); Bolam 2000, Tepper & Koos 2017, Tepper 2018, Tepper/Abercrombie/Bolam 2007 PBR vol 160 (basal ganglia); O'Keefe & Nadel 1978 (hippocampus); Buzsáki 2006 (rhythms); Sutton & Barto 2018, Schultz 1998 + Hollerman & Schultz 1998 + Schultz 2016 NRN + Schultz 2016 *J. Neural Transm.* (RL + reward).

---

## Reading guide

- **The catalog** ([feature-catalog.md](feature-catalog.md)) lists every biological mechanism, with a short biological description and a per-mechanism sim-status assessment. **Read it when** you want to know what we have / don't have for a specific mechanism.
- **This roadmap** organizes those mechanisms into clusters with implementation order, dependencies, and validation criteria. **Read it when** you want to know what to build next.

The roadmap distinguishes:

- **"Modelable now"** — mechanisms whose substrate already exists in the simulator and need only a `BrainRegion`/`RegionPathway` config or a new `NeuromodulatorConfig` + minor wiring. **Days to weeks of work.**
- **"Modelable with focused infrastructure"** — mechanisms that need a specific new component (e.g., per-action eligibility tag for compartmentalized DA; engram-tagging API; topographic maps). **Weeks to months.**
- **"Modelable with major architecture change"** — mechanisms requiring a fundamentally new computational substrate (e.g., compartmental neurons for active dendrites; transcriptional state for late-LTP; muscle output). **Months to years; warrants a separate research arc decision.**
- **"Out of scope at this abstraction level"** — mechanisms that are below or alongside the simulator's level (molecular machinery: SNAREs, individual channels of the same family, cell-adhesion molecules, embryonic patterning).

---

## Prioritization principles

The roadmap orders mechanisms by a weighted combination of:

1. **Relevance to the current research arc** (closing perception/reward cheats; multi-modal navigation; long-horizon memory) — **highest weight**.
2. **Leverage** (how many other open questions does this unlock? — e.g., topographic maps unlock columnar processing, retinotopy, somatotopy in one stroke).
3. **Cost** (effort to implement, vs. reusing existing infrastructure).
4. **Validation tractability** (does the textbook give a clear behavioral signature we can replicate?).
5. **Substrate alignment** (does the mechanism map cleanly onto existing project primitives — `BrainRegion`, `NeuromodulatorConfig`, plasticity gates — or does it require new substrate?).

---

## Tier 0 — One-config-edit quick wins (week 0–2)

Surfaced by the second enrichment pass (Schultz 2016, Hesslow & Yeo 2002, Tepper/Bolam 2007 PBR vol 160). Each is hours-to-days of work, no new infrastructure. **Do these before Tier 1 starts** — they fix latent biological errors that other tiers compound.

### T0.A — Fix `cfg.E_inh` per region (MSN GABA reversal)

**What:** Replace the global `cfg.E_inh = -75 mV` with a per-region override:
- MSN (D1, D2 in striatum): `E_inh = -60 mV` (depolarizing at rest, shunting near threshold)
- SNc DA neurons: `E_inh = -55 mV` (lack KCC2; near firing threshold)
- Cortical pyramidals + most others: keep `-75 mV`

**Why:** Wilson ch 6 of PBR-160 (B.14, augmented B.02) nails E_GABA at MSN = −60 mV; SNc DA neurons lack KCC2 entirely (B.15). Project's global value is correct for cortical pyramidals but **fundamentally wrong** for striatum and DA neurons. Affects STDP windows, the sign of GABA action on DA bursts, and the realism of MSN dendritic input integration.

**Substrate:**
- Add `BrainRegion.E_inh_override: float | None` field.
- Plumb through `sim/bridge.py` synaptic conductance update — currently uses scalar `cfg.E_inh`; needs per-neuron lookup keyed by region.
- One-time migration: set override on the BG region declarations in `g11_bg_runner.build_bg_brain_regions()`.

**Validation:**
- Re-run flagship 6-seed flagship config; compare to baseline 4.08. Acceptable if equal-or-better; document any regression.
- Re-run gamma-oscillation benchmark (cortex-only) — should be unchanged.
- Verify `MSN ← inhibitory input` produces shunting-with-mild-depolarization rather than hyperpolarization.

**Catalog entries:** B.14 MSN GABA-A reversal, B.15 SNc DA E_Cl.

**Estimated effort:** 1 day (substrate + tests); 1 day for flagship re-validation.

### T0.B — Add `gpe_X → gpi_X` perisomatic inhibition pathway

**What:** Wire the missing GPe → GPi/SNr inhibitory projection that the textbook cascade has but the simulator's flagship currently lacks. Per-action. Weight ~3× the str_d1→gpi value (anatomical potency: perisomatic vs distal-dendritic). Conduction delay set to be ~9 ms shorter than striatonigral.

**Why:** Nambu ch 8 of PBR-160 (A.14, B.16): striatal terminals on GPi/SNr go to distal dendrites (~70%); GPe terminals cluster perisomatically (~15% but high somatic-veto potency). Conduction velocities: striatonigral 1.4 m/s vs pallidonigral 4 m/s. **GPe input arrives 9 ms before striatal D1 input AND has 3× more somatic veto power per synapse.** The flagship cascade currently has no such projection — adding it would replicate the canonical in-vivo three-phase GPi response (early STN excitation → mid striatal inhibition → late indirect-pathway excitation) that Alexander & DeLong's textbook model leaves implicit.

**Substrate:**
- Add `RegionPathway` entries for `gpe_X → gpi_X` (each action). All inhibitory, GABAergic.
- If T0.A is in: use the GPi-region E_inh override.
- Calibrate weight against existing str_d1→gpi by sweep.

**Validation:**
- Inject a brief transient cortical pulse; verify GPi response shows the three-phase signature on a peristimulus time histogram.
- Re-run flagship 6-seed config; should not regress.

**Catalog entries:** A.14 perisomatic GPe→GPi, A.02 indirect pathway, B.16 conduction velocity asymmetry.

**Estimated effort:** 2–3 days (config + sweep + validation).

### T0.C — Compose `--surprise-lr-boost` + `--adaptive-da` instead of treating as alternatives

**What:** Change the recommended-config table in CLAUDE.md to compose both flags by default, rather than presenting them as alternatives. Run a 6-seed validation to confirm the composition is no worse than the better alternative on its own (and possibly better).

**Why:** Schultz 2016 NRN (C.32) frames phasic DA as **two distinct components**:
- **Component 1** (60–90 ms latency, salience-blind detection of any unexpected event)
- **Component 2** (150–300 ms latency, utility-RPE)

The project's flags map onto these directly:
- `--surprise-lr-boost` is functionally **Component-1 analog** (valence-blind LR scalar on |RPE|, applied to ANY surprise event — explains its task-robustness across slow/fast change)
- `--adaptive-da --adaptive-da-ema-decay-negative 0.7` is functionally **Component-2 analog** (slow positive ramp, fast negative dip — biology-grounded by Schultz's omission-dip vs acquisition-burst data)

Currently the recommended-config table presents these as alternatives ("use one, not both"). **Biology says they should compose**: Component 1 sets the LR magnitude on any salient deviation; Component 2 sets the directional value-update target. Combo flag was tested previously and didn't compose well — but that test was before adaptive-DA's asymmetric variant was tuned, so worth retesting.

**Substrate:** No code change. Just a 6-seed run with both flags on.

**Validation:**
- 6-seed flagship + both flags on. Compare against current flagship 4.08.
- If composed result ≤ 4.08: ship as new flagship.
- If composed result > 4.08: document as NEGATIVE (still surfaces a `[discrepancy]` between Schultz-biology and project-empirics worth flagging; current alternative-treatment is empirically defensible even if biologically less faithful).

**Catalog entries:** C.32 two-component DA, C.04 DA primary, C.22 RPE.

**Estimated effort:** 0.5 day setup + 4–8h compute for 6-seed run.

---

## Tier 1 — Build now (next 1–3 months)

These mechanisms either have *no infrastructure cost* (use existing primitives) OR are already in active development.

### T1.A — Hippocampal trisynaptic loop (Cluster D core)

**What:** Three new `BrainRegion`s — DG, CA3, CA1 — wired as `EC → DG → CA3 → CA1 → output` (perforant path) plus `EC → CA1` (direct cortical) and `CA3 → CA3` recurrent (autoassociator).

**Why now:**
- Cluster D is the project's biggest "partial" — current "place cells" emerge from landmark perception alone; no DG sparsification, no CA3 pattern completion, no CA1 readout.
- Perception arc is complete; logical next chapter is sequence-and-memory (Cluster D).
- Uses existing primitives only: `BrainRegion`, `RegionPathway`, plasticity gates. No new GPU code.

**Substrate:**
- DG: high sparsity (~3% active), strong feedforward inhibition → orthogonalizes inputs (pattern separation).
- CA3: dense recurrent collaterals → autoassociator (pattern completion from partial cues).
- CA1: integrates direct EC input + CA3 output → outputs to subiculum / EC.

**Validation:**
- Pattern separation: present 2 highly similar place inputs, verify DG outputs are decorrelated.
- Pattern completion: train CA3 on a cue+context pair, present partial cue, verify full output reactivates.
- Place-field stability across trials in CA1 readout.

**Catalog entries primarily addressed:** D.05+ (additions) — place cells, grid cells (downstream), pattern separation, pattern completion, mossy-fiber LTP, perforant-path L-type Ca²⁺ LTP.

**Estimated effort:** 1–2 weeks for a working circuit; 4–6 weeks to full validation.

### T1.B — Sharp-wave ripple-driven sequential replay (Cluster D + N)

**What:** Augment existing sleep-replay infrastructure to generate time-compressed (10–20×) place-cell sequences during NREM windows, phase-locked to a slow-oscillation surrogate and nested by spindle envelopes.

**Why now:**
- Project explicitly identified "replay content quality is the bottleneck" (sleep-replay infra exists but content is random/stale).
- Composes T1.A (need place cells to replay).
- All ingredients exist: NREM/REM stage alternation already implemented; `StimulusManager` can inject high-frequency pulses; NM `excitability_drive` can gate CA3.

**Substrate:**
- Record CA1 spike-time sequences during waking trajectories.
- During NREM: replay reverses or compresses these sequences via excitatory drive on CA3.
- STDP performs the consolidation transfer to downstream cortical / striatal pathways automatically.

**Validation:**
- Verify ripple events show 10–20× temporal compression of waking sequences.
- Compare downstream-region weight changes during sleep vs no-sleep on a memory task.
- Replicate "blocking SWRs impairs spatial learning" (Girardeau et al. 2009).

**Catalog entries:** N.04 (SWRs), D.16 (replay), J.07 (LTP-mediated consolidation transfer).

**Estimated effort:** 2–3 weeks (composes onto existing infra).

### T1.C — Engram-tagging API (Cluster D + J)

**What:** ~50 LOC bridge addition: `bridge.tag_active_ensemble(name, threshold_hz, window_ms)` (snapshots which neurons fired above threshold during a window) + `bridge.stimulate_tag(name, drive_pA)` (drives only tagged neurons) + persistence across simulation steps.

**Why now:**
- Cheapest unlock for causal recall experiments (Tonegawa-style false-memory, optogenetic memory tagging).
- Validates pattern completion mechanism (T1.A) directly.
- Aligns with project's "biology-grounded" ethos — provides a mechanism for converting *correlation* (place-cell activity) into *causation* (driving the same cells produces the same behavior).

**Validation:**
- Train on "context A → reward"; tag the active ensemble.
- Place agent in context B, drive tagged neurons → verify reward-conditioned behavior emerges.
- Reproduces Liu et al. 2012 inception-of-fear-memory paradigm.

**Catalog entries:** D.13 (engram cells).

**Estimated effort:** 2–3 days for bridge code + tests; 1–2 weeks for first experiment.

### T1.D — Parkinson's, schizophrenia, epilepsy disease smoke tests (Cluster P)

**What:** Add 3 disease "modes" to `g11_bg_runner` as runtime flags. Each silences or perturbs an existing component and records the resulting behavioral signature.

**Why now:**
- Zero new infrastructure — these are *already modelable* on the current BG cascade + benchmark suite.
- Validates the model-disease alignment that the BG cascade implicitly claims to capture.
- Each is publishable as a standalone neuroscience-validation result.

**Three disease tests:**

| Disease | Recipe | Predicted phenotype | Catalog ref |
|---|---|---|---|
| Parkinson's | `nm_mgr.set_concentration("DA", 0)` + freeze production | Indirect-pathway dominance → motor-pool firing collapses → moving-goal sum increases (worse navigation), partial L-DOPA recovery via manual DA injection | P.23 |
| Schizophrenia (DA) | `neuromodulator.DA.baseline = 2.0 × baseline` | Spurious action selection on null cues (false-percept analog) | P.13 |
| Schizophrenia (NMDA hypofunction) | scale `fused_nmda_update_and_current()` to 30% on FS interneurons | Gamma-power decrease on existing `--benchmark gamma-oscillations` | P.14 |
| Focal epilepsy | one region: `inh_weight_mean × 0.3, exc_weight_mean × 1.5` | Synchronized population bursts (PDS-like) propagating through pathways | P.07 |

**Validation:** Each disease's phenotype should match the textbook clinical signature with no parameter tuning beyond the prescribed change.

**Estimated effort:** 1–2 weeks total for all four (each is a CLI flag + analysis).

### T1.E — Hypothalamic homeostatic drives (Cluster O + C)

**What:** Two new neuromodulators in the existing framework: "hunger" (slow accumulation; depletes on reward; aversive at high concentration) and "thirst" (similar). Reward weights modulated by drive concentrations (incentive salience).

**Why now:**
- Closes "agent has no internal state" gap.
- Uses only existing NM framework — no new infrastructure.
- Connects to longstanding RL critique: tonic motivation is missing from R-STDP.
- Behavioral richness gain is large (agent now seeks food when hungry, water when thirsty, balances both).

**Substrate:**
- `NeuromodulatorConfig("hunger", baseline=0.0, decay_tau_ms=∞, production_rules=[manual + decay-on-reward])`.
- Add reward-modulation target type: hunger-weighted reward = base_reward × hunger.
- Test scenario: 2 distinguishable food / water sources; agent learns to alternate based on internal drives.

**Validation:**
- Berridge "wanting vs liking" — hunger up-regulates approach to food cues even without consumption.
- Drive-reduction reinforcement: aversive high-drive state, eating reduces drive, that reduction is rewarding.

**Catalog entries:** O.01 hunger / thirst, O.04 incentive motivation, O.05 drive reduction.

**Estimated effort:** 2 weeks.

---

## Tier 2 — Build with focused infrastructure (3–9 months)

These need a specific new component but compose with existing architecture.

### T2.A — Cerebellum microcircuit (Cluster F)

**What:** New runner (`f01_cerebellum_runner.py`) implementing the Marr-Albus circuit: mossy fibers (MF) → granule cells (GC) → parallel fibers (PF) → Purkinje cells (PC) ← climbing fibers (CF, from inferior olive) → deep cerebellar nuclei (DCN). PF→PC plasticity is *anti-Hebbian* — coincident PF + CF input *depresses* the PF→PC synapse (long-term depression, LTD) — implementing supervised error-correction.

**Required new infrastructure:**
- CF-gated PF→PC LTD kernel: `Δw_pf = -η_lr × pf_active × cf_active`. **New fused kernel.**
- Inferior olive as special pacemaker region (~1–10 Hz tonic, switching to bursts on errors). Existing region framework + `IZH2007_INFERIOR_OLIVE` preset.
- Deep cerebellar nuclei as output region.
- **Nucleo-olivary feedback** (F.18): DCN → IO inhibition. Without this, training does not extinguish on its own. **Often omitted in simulator implementations of Marr-Albus** — flag it as a required component of the implementation, not an optional refinement.
- **Three distinct mossy-fiber input streams** (F.03): vestibular/reticular, cortico-pontine (efference copy), spinocerebellar (proprioception). Declare as separate `mossy_*` source regions, not a monolithic pool.
- **Basket/stellate-b plasticity** (F.16): without bidirectional weight rules, all PCs converge to silent over long training. Albus 1971 §IV.D–E argues PF→basket synapses must use the same CF-gated rule as PF→PC.
- **Intrinsic PC timer** (F.17, optional in v1): mGluR1-coupled slow Ca²⁺ + KCa cascade. Hesslow's 2013 evidence suggests adaptive CR timing requires this; first version can ship without and see if the timing benchmark passes on LTD alone.

**Why mid-priority:**
- Cluster F is fully missing (presets exist but no circuit).
- Unlocks: eyeblink conditioning (canonical), VOR adaptation, smooth-pursuit, forward-model experiments — all distinct experiments with established benchmarks.
- Uses existing region framework + one new kernel; not architecturally invasive.

**Validation suite (sharpened by Hesslow & Yeo 2002 chapter):**

1. **Eyeblink acquisition**: pair tone (CS, MF input) with airpuff (US, CF input) → after N trials, PC activity is suppressed in response to tone, releasing DCN to drive blink. Acquisition curves match published rabbit data (Thompson 1986).
2. **CR/UR double-dissociation gate** (F.06, F.08 — sharpest validation criterion): AIP lesion drops both CR and UR slightly; cortical (HVI) lesion drops CR but **raises UR amplitude** (because cortex inhibits AIP). **Single-pool cerebellar models fail this; only PC→DCN inhibition + DCN→motor excitation passes.** Make this a hard gate — implementations that don't pass don't ship.
3. **Reversible-inactivation triple-test** (F.20): use existing `set_plasticity_gate` infrastructure to (a) block AIP plasticity during acquisition → no learning after unblock, (b) block AIP during extinction → no extinction either, (c) block efferents (BC) → learning proceeds normally. Three orthogonal tests pin the learning site to the cerebellar somata.
4. **VOR adaptation**: reverse-prism goggles for N trials → PC LTD adapts the VOR gain.
5. **Trace conditioning + hippocampus** (F.22, NEW Cluster F↔D bridge): cerebellum bridges CS-US gaps up to ~500 ms alone; longer traces require hippocampus. After T1.A is in, test that adding hippocampal input lets CR acquire on traces > 500 ms.
6. **Adaptive CR latency** (F.24): a single brief MF pulse should evoke a normally-timed (~200 ms) PC pause. If the Marr-Albus + LTD-only implementation can't reproduce this, F.17 (intrinsic PC timer) becomes mandatory.

**Six hippocampus-dependent paradigms** (F.23) are downstream validation targets once T1.A + T2.A are both shipped: trace conditioning, discrimination reversal, latent inhibition, conditional discrimination, sensory preconditioning, blocking. Each is a separate experiment.

**Catalog entries:** F.01 Marr-Albus, F.02 (codon), F.03 MF stream split, F.04 climbing-fiber error signal, F.05 PF→PC LTD with sign discrepancy, F.06 DCN, F.08 eyeblink protocol, F.16 basket/stellate plasticity, F.17 intrinsic PC timer, F.18 nucleo-olivary feedback, F.20 reversible-inactivation methodology, F.22 trace conditioning, F.23 six HC paradigms, F.24 adaptive CR latency.

**Estimated effort:** 5–7 weeks (was 4–6; the validation suite is broader and the nucleo-olivary loop is a hard requirement).

### T2.B — Topographic maps in `BrainRegion` (Cluster E + I)

**What:** Add a `coordinate` field to `BrainRegion` (1D or 2D position per neuron) + a distance-dependent connection-probability term in connectivity generators. Allows columnar / retinotopic / tonotopic / somatotopic organization.

**Why now-ish:**
- Single highest-leverage gap flagged by Part IV agent — every cortical perception result in Kandel rests on adjacent-neurons-encode-adjacent-stimulus.
- Composes with existing patch-matrix work in `connectivity.py`.
- Unlocks: orientation-selective simple-cell-like receptive fields, ocular-dominance columns, retinotopic V1, somatotopic S1.

**Required infrastructure:**
- `BrainRegion.coordinate: Optional[np.ndarray]` (n_neurons, k_dim).
- Connection probability function: `p(i,j) ∝ exp(-||c_i - c_j||² / 2σ²) × p_base`.
- Existing `sim/connectivity.py` Watts-Strogatz / spatial generators already have spatial primitives — extend rather than rebuild.

**Validation:**
- Connect a 256-neuron sensory layer with 2D coordinates to a cortex layer; verify receptive-field tuning emerges from STDP on input patterns.
- Compare to Linsker 1986 / Miller 1989 ocular-dominance column emergence.

**Catalog entries:** E.02 cortical columns, I.13 cable equation (already not-applicable but coordinates partially fix this), L.04 critical-period (now applies to topographic maps).

**Estimated effort:** 1–2 weeks for substrate, 2–4 weeks for validation experiments.

### T2.C — Mechanoreceptor / labeled-line front-end (Cluster K + E)

**What:** Replace the abstract beacon/landmark with a real transducer front-end: 4 mechanoreceptor classes (Pacinian, Meissner, Merkel, Ruffini) implemented as AdEx neurons with adaptation-rate variation, projecting via labeled lines to S1 (somatotopic, T2.B-dependent).

**Why mid-priority:**
- Most tractable transduction system to add (mechanoreceptors map cleanly onto AdEx adaptation).
- Replaces "beacon/landmark" with real sensory channel where adaptation class, RF size, frequency tuning matter.
- Opens texture / vibration tasks beyond pure spatial navigation.

**Required infrastructure:**
- AdEx preset specialization for each mechanoreceptor class (different `tau_w`, `b`).
- Labeled-line projection wiring (T2.B helps).
- New stimulus modality: 2D contact pattern → spike trains from mechanoreceptor pool.

**Validation:**
- Replicate Pacinian high-frequency vibration tuning (200–300 Hz peak).
- Two-point discrimination at S1 readout.
- Texture discrimination via temporal patterns.

**Catalog entries:** K.07–K.08 mechanoreceptors, E.05 receptive fields, E.06 labeled lines.

**Estimated effort:** 3–4 weeks.

### T2.D — Compartmentalized dopamine (Cluster C — cheat-5 option 3)

**What:** Per-action DA pulses. Replace scalar `current_reward_signal` with a 4-vector DA[N], DA[E], DA[S], DA[W]. Synapses tagged with target action; only matching DA drives plasticity.

**Why mid-priority:**
- One of the three "real cheat-5 closure" options surveyed (Option 3).
- Independent of Option 1 (structural pruning, currently in active development) — could compose.
- Real biology: per-DA-axon specificity is documented (Schultz, Berke).

**Required infrastructure:**
- Per-synapse `synapse_action_tag[i] ∈ {0,1,2,3}`.
- 4-vector reward signal in eligibility trace.
- Modified `fused_eligibility_trace_decay` to use action-tag indexing.

**Validation:**
- 4-action moving-goal task: agent learns each action's reward landscape independently.
- Should outperform broadcast-DA on tasks with action-specific reward (e.g., right-arm reach is rewarded differently than left).

**Catalog entries:** C.04 (DA, current implementation), referenced from cheat-5 survey.

**Estimated effort:** 2–3 weeks for substrate; 4–8 weeks for validation against current adaptive-DA flagship.

### T2.E — Amygdala valence module (Cluster O)

**What:** Small (~50–100 neuron) LA → BLA → CeA structure handling fear/aversive valence in parallel with the reward (positive-valence) DA system. Inputs from "pain" / "negative-reward" channel; outputs gate behavior toward freezing / avoidance.

**Why mid-priority:**
- Cluster O is partial — current "negative reward" is just a sign-flipped scalar; biology routes it through a *separate substrate* (amygdala) that interacts with reward asymmetrically.
- Closes Pavlovian fear conditioning at the *circuit* level (current preset captures behavior but not biology).
- vmPFC-mediated extinction is testable.

**Required infrastructure:**
- New region declarations (LA, BLA, CeA).
- Aversive sensor channel (e.g., simulated "pain" injection).
- Fear-conditioning protocol in `experiment/presets.py`.

**Validation:**
- Pavlovian fear conditioning: paired CS (tone) + US (pain) → CS alone evokes freezing. Match LeDoux acquisition curves.
- Extinction (CS alone, repeated) suppresses freezing without erasing LA→CeA weights — verified by reinstatement after a single re-pairing.

**Catalog entries:** O.07 amygdala fear, O.10 extinction, O.06 limbic system.

**Estimated effort:** 3–4 weeks.

---

## Tier 3 — Major architecture changes (9–18 months, decide explicitly)

These require fundamental new substrate. Pursue only after Tier 1+2 confirm the existing architecture continues to deliver.

### T3.A — Compartmental neurons (Cluster G + I)

**What:** Multi-compartment cell models — at minimum a 2-compartment "soma + apical dendrite" version of the L5 pyramidal cell, supporting active-dendrite computation, NMDA spikes, and Larkum's BAC firing.

**Why eventually:**
- Largest single abstraction in the simulator (currently single-compartment everywhere).
- Required to replicate experiments where apical-basal coincidence is the substrate (perceptual inference, conscious-access models).
- Required to model: Martinotti dendrite-targeting inhibition (B.01), Kv4 transient K⁺ delay-to-first-spike (I.55), non-uniform AIS (I.01).

**Cost:**
- ~10× compute per neuron (rough estimate).
- New GPU kernel architecture (compartment-coupled membrane equations).
- Does not compose cleanly with existing kernels — requires substantial rewrite of `sim/kernels.py`.

**Decision criterion:** pursue when a target experiment requires it AND we've exhausted single-compartment alternatives.

### T3.B — Late-LTP / transcriptional state (Cluster J)

**What:** Per-synapse tier of weight that resists later LTD; updates slowly with cAMP/PKA/CREB-like protein-synthesis kinetics (hours). Enables early-vs-late LTP distinction (cycloheximide-blockable).

**Why eventually:**
- One of the more important missing mechanisms for long-horizon memory experiments.
- Composes naturally with existing structural-plasticity infrastructure (`cp_synapse_alive` could be paired with `cp_synapse_consolidated`).
- Required for: reconsolidation, prion-like CPEB tagging, late phase of LTP.

**Cost:**
- New per-synapse state array.
- New plasticity rule (slow consolidation kinetics).
- Validation requires experiments simulating "hours" — slow but tractable.

**Decision criterion:** pursue when first experiment shows current decay-resistant baseline weights are insufficient for a multi-day memory task.

### T3.C — Muscle output / Hill-type model (Cluster H + M)

**What:** 1D Hill-type muscle model fed by motor-neuron spike trains; produces simulated force; couples back to environment.

**Why eventually:**
- Cluster H and M are fully missing.
- Required to model: motor unit recruitment (Henneman size principle), twitch summation, fatigue, eccentric vs concentric loading.
- Needed for any "real" motor learning beyond abstract action-selection.

**Cost:**
- New module (`sim/muscle.py` or similar).
- Environment integration (motor force → world dynamics).
- Sensory feedback loop (muscle spindles, GTOs) becomes meaningful — interacts with K (sensory transduction).

**Decision criterion:** pursue if the project pivots from "abstract action selection" to "embodied motor control."

### T3.D — Glia + neurovascular coupling (Cluster Q)

**What:** Astrocyte syncytium with K⁺ buffering, glutamate clearance, Ca²⁺ waves; microglia-mediated synapse pruning (composes with structural plasticity).

**Why eventually:**
- Currently entirely missing; cluster Q proposed mid-Section IV.
- Microglia-mediated pruning is a *biological mechanism* for what we already do *functionally* (structural plasticity).
- K⁺ buffering would naturally cap runaway-firing (the "n_cortex=400 → D1 saturation" bug).

**Cost:**
- Substantial new mechanism; not covered by existing primitives.
- Requires extracellular-state representation (a bridge-level [K⁺]_ext, [glutamate]_ext array).

**Decision criterion:** pursue when modeling sustained high-rate firing reveals biological-realism gaps that K⁺-buffering would fix.

---

## Out of scope at current abstraction level

Mechanisms below the simulator's level of abstraction. **Will not implement** unless the abstraction itself is rethought:

- Embryonic patterning (Hox genes, BMP/Wnt/SHH/RA gradients) — Ch 45.
- Axon guidance (netrins, semaphorins, slits/Robo) — Ch 47. Connectivity is config-time in our framework.
- Cell-adhesion molecules (cadherins, neurexin/neuroligin) — Ch 48. PSD scaffolding (J.12) is `not-applicable` for the same reason.
- Individual ion-channel proteins below the family level (Nav1.1 vs 1.6 etc.) — Ch 8. Family-level effects captured in HH/Izh/AdEx parameters.
- Vesicle-fusion molecular machinery (SNAREs, synaptotagmin) — Ch 15. STP captures macroscopic dynamics.
- Sex differentiation — Ch 51.

---

## Cross-cluster validation targets

A few "biology benchmarks" that span multiple clusters and would validate the model's overall biological grounding.

| Target | Clusters | Why it matters |
|---|---|---|
| **CR/UR double-dissociation** | F (cerebellum) | Sharp gate (Hesslow & Yeo 2002 §pp 108-109, 114-116): cortical lesion drops CR but RAISES UR; AIP lesion drops both. Single-pool cerebellar models fail. Required for T2.A acceptance. |
| **Three-phase GPi response** | A (BG cascade) | Adding T0.B GPe→GPi pathway should produce the canonical early-STN / mid-striatal / late-indirect signature in GPi PSTH after a transient cortical pulse. Validates the pathway timing. |
| **Eyeblink conditioning** | F (cerebellum) + J (PF→PC LTD) | Canonical cerebellar-learning benchmark; clean acquisition + extinction curves available |
| **Ocular dominance plasticity** | E (cortex) + L (critical period) + Q (astrocyte K⁺?) | Hubel-Wiesel deprivation paradigm; tests both topographic maps and critical-period machinery |
| **Spatial memory + replay** | D (hippocampus) + N (sleep) + J (LTP consolidation) | Tests T1.A + T1.B integration; replicates Girardeau ripple-disruption result |
| **Inception of false memory** | D (engram) + J (LTP) + O (reward) | Tonegawa optogenetic paradigm; tests T1.C engram-tagging API |
| **Parkinson's BG dysfunction** | A (BG) + C (DA) + P (disease) | Already on roadmap (T1.D); validates BG cascade against canonical clinical model |
| **Pavlovian fear conditioning + extinction** | O (amygdala) + J (LTP) + G (vmPFC) | Tests T2.E amygdala module |
| **Trace conditioning bridges F + D** | F (cerebellum) + D (hippocampus) | After T1.A + T2.A both ship, CR should acquire on CS-US gaps > 500 ms only when hippocampus is connected. F.22 + F.23. |
| **Cue-shift transfer** | C (DA) + A (BG) | Schultz 1998: DA bursts gradually transfer from reward time to cue time over learning. Currently NOT reproduced (project has only the burst-on-unexpected-reward sign). T2.D (compartmentalized DA) + adding a critic population (C.30) needed; or PPN (C.33). |

---

## Implementation order (recommended)

A practical 12-month sequencing that respects dependencies:

| Week / month | Work | Tier |
|---|---|---|
| Week 0–1 | T0.A E_inh per region + T0.B GPe→GPi pathway | T0 |
| Week 1–2 | T0.C compose surprise-LR + adaptive-DA (validation only) | T0 |
| Month 1 | T1.A hippocampal trisynaptic loop | T1 |
| Month 2 | T1.B SWR-driven replay | T1 |
| Month 2 | T1.C engram-tagging API (parallel; small) | T1 |
| Month 3 | T1.D disease smoke tests (parallel; small) | T1 |
| Month 3 | T1.E homeostatic drives (parallel) | T1 |
| Month 4–5 | T2.A cerebellum microcircuit (with sharpened validation suite) | T2 |
| Month 6 | T2.B topographic maps | T2 |
| Month 7 | T2.D compartmentalized DA (cheat-5 option 3) | T2 |
| Month 8–9 | T2.C mechanoreceptor front-end | T2 |
| Month 10 | T2.E amygdala valence | T2 |
| Month 11 | Trace-conditioning F↔D bridge experiment + cue-shift validation | — |
| Month 12 | Integration + validation passes | — |

After month 12, the simulator should have:

- All 17 clusters at *partial* or better.
- 6+ canonical neuroscience benchmarks reproduced (eyeblink, ocular dominance, spatial memory, fear conditioning, Parkinson's, schizophrenia).
- Clear evidence base for whether to pursue Tier 3 (compartmental neurons, late-LTP, muscle, glia).

---

## Open meta-questions for the project owner

These are decisions the catalog can't make for you:

1. **How tightly to track the textbook?** Some entries are "could be added" but don't serve current research goals. We could reasonably skip Q (glia), M (NMJ), most of K (transduction beyond mechanoreceptors) without losing core capability. *Roadmap above assumes "track tightly enough to claim biological grounding for the perception/learning arc, skip the rest until there's a specific need."*
2. **When to commit to compartmental neurons (T3.A)?** Single-compartment is the largest single abstraction and constrains many sub-clusters. Currently OK; will increasingly bite.
3. **Whether to model muscle (T3.C)?** This is the gateway to "embodied" simulation. Big architectural decision.
4. **How much disease modeling to pursue (Cluster P)?** Each disease is publishable as standalone validation. Could become a research arc of its own.

The catalog and this roadmap put you in a position to answer all four with concrete data on cost and dependency.
