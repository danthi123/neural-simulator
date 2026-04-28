# Biology Buildout Roadmap

This document organizes the ~323 mechanism entries in [`feature-catalog.md`](feature-catalog.md) into a prioritized implementation roadmap for the next 6–18 months. It is a *strategy* document — the catalog is the encyclopedia; this is which pages to act on, in what order, and why.

**Last updated:** 2026-04-28 (initial draft post-merge of Section IV + parallel subagent passes).

**Source:** Kandel et al., *Principles of Neural Science*, 6th edition (2021). Approximately 1,500 pages of textbook surveyed.

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

**Why mid-priority:**
- Cluster F is fully missing (presets exist but no circuit).
- Unlocks: eyeblink conditioning (canonical), VOR adaptation, smooth-pursuit, forward-model experiments — all distinct experiments with established benchmarks.
- Uses existing region framework + one new kernel; not architecturally invasive.

**Validation:**
- **Eyeblink conditioning**: pair tone (CS, MF input) with airpuff (US, CF input) → after N trials, PC activity is suppressed in response to tone, releasing DCN to drive blink. Acquisition curves match published rabbit data (Thompson 1986).
- **VOR adaptation**: reverse-prism goggles for N trials → PC LTD adapts the VOR gain.

**Catalog entries:** F.01 Marr-Albus, F.02 PF→PC LTD, F.03 climbing-fiber error signal, F.04 deep nuclei output, F.05 eyeblink validation.

**Estimated effort:** 4–6 weeks.

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
| **Eyeblink conditioning** | F (cerebellum) + J (PF→PC LTD) | Canonical cerebellar-learning benchmark; clean acquisition + extinction curves available |
| **Ocular dominance plasticity** | E (cortex) + L (critical period) + Q (astrocyte K⁺?) | Hubel-Wiesel deprivation paradigm; tests both topographic maps and critical-period machinery |
| **Spatial memory + replay** | D (hippocampus) + N (sleep) + J (LTP consolidation) | Tests T1.A + T1.B integration; replicates Girardeau ripple-disruption result |
| **Inception of false memory** | D (engram) + J (LTP) + O (reward) | Tonegawa optogenetic paradigm; tests T1.C engram-tagging API |
| **Parkinson's BG dysfunction** | A (BG) + C (DA) + P (disease) | Already on roadmap (T1.D); validates BG cascade against canonical clinical model |
| **Pavlovian fear conditioning + extinction** | O (amygdala) + J (LTP) + G (vmPFC) | Tests T2.E amygdala module |

---

## Implementation order (recommended)

A practical 12-month sequencing that respects dependencies:

| Month | Work | Tier |
|---|---|---|
| 1 | T1.A hippocampal trisynaptic loop | T1 |
| 2 | T1.B SWR-driven replay | T1 |
| 2 | T1.C engram-tagging API (parallel; small) | T1 |
| 3 | T1.D disease smoke tests (parallel; small) | T1 |
| 3 | T1.E homeostatic drives (parallel) | T1 |
| 4–5 | T2.A cerebellum microcircuit | T2 |
| 6 | T2.B topographic maps | T2 |
| 7 | T2.D compartmentalized DA (cheat-5 option 3, in parallel with structural pruning option 1 if active) | T2 |
| 8–9 | T2.C mechanoreceptor front-end | T2 |
| 10 | T2.E amygdala valence | T2 |
| 11–12 | Integration + validation passes | — |

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
