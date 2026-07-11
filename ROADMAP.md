# ROADMAP — a brain you can talk to

> **Status: source of truth.** This is the canonical, at-a-glance record of *what the project has accomplished, what it is working on now, and what is left* on the path to the goal. It is updated as a standing part of the workflow (see the `neural-simulator` skill): when an arc lands a result, surpasses a boundary, or replaces a scaffold, the relevant stage below is updated in the same cycle. If a claim here conflicts with a document in `research/findings/`, the finding wins and this file is corrected.
>
> **Last synced:** 2026-07-10 (via a 5-agent deep-research pass over the feature catalog, Kandel 6e, and the findings).

---

## 1. The goal

Build **artificial life** — a real brain-analogue as a persistent, learning, growing lifeform — whose **conversational capability approaches that of a large language model** (open-domain, fluid, grounded, context-carrying), realized as **one fully-spiking brain** that is emergent, biology-grounded, and free of any permanent external model. Everything here serves that goal; capabilities are instrumental, the deliverable is a living, communicating brain whose language ability is genuinely its own.

## 2. The non-negotiable constraints (what "the honest way" means)

These bind every stage. A capability that violates them is a **scaffold** to be replaced, not a milestone.

1. **Brain-based only.** Everything between sensation and action is neurons, synapses, and their communication. Host (non-neural) code is legitimate *only* for the **environment** (world state; rendering the sensory image) and the **body** (enacting motor output). A biologically-correct Python formula (a reward, an argmax, a prediction error) is still a shortcut.
2. **One brain, one substrate, one process.** Faculties are disjoint/interacting slices on a shared `SimulationBridge` with cross-region *synaptic* interaction — not co-located demos, not bolted-on modules.
3. **Emergent / self-organized.** Structure is *discovered from experience*, not hand-designed feature-by-feature (that is whack-a-mole, not biology). Spiking-at-runtime with host-*designed* weights is still a residual shortcut.
4. **Fully-spiking end state.** The path (scaffold-then-clean vs biological-from-start) is an efficiency call; the end is fully spiking on the one brain. Every shortcut is tracked and burned down.
5. **No permanent external model.** A transformer/LLM may be a *temporary* fluency scaffold, but the end state simulates the circuitry. *"If Broca drives articulation, we simulate Broca."*
6. **Honest negatives are deliverables.** A boundary maps what the substrate can and cannot do on its own — and launches the search for the next mechanism. It is never a place to stop.

## 3. How to read this roadmap (status legend)

| Badge | Meaning |
|---|---|
| ✅ **EMERGENT** | Done, and the structure was *learned from experience* on the spiking substrate. |
| 🟩 **DONE** | Done and validated on-substrate, but with a hand-designed component (a fixed algebra, a wired structure) that is biologically defensible but not itself learned. |
| 🟨 **PARTIAL** | Works in a reduced form / at reduced scale / with a characterized gap. |
| 🟧 **BOUNDARY** | A characterized limit — validated as *not yet reachable* by the approaches tried; the next mechanism is named. |
| 🧩 **SCAFFOLD** | A temporary host stand-in (an external model, a template) that must be replaced by simulated circuitry. |
| ⬜ **OPEN** | Not yet built. |

**Validation bar.** A result is a **GO** only after multi-seed confirmation (dev seeds 42/43/44 → blind 100/101/102) with anti-cheats (lesion / permuted / memorization-floor / oracle-ceiling / scramble) and, for anything entering the record as a surpass, an independent **adversarial-verify** pass. "6-seed GO" is the standard.

---

## 4. The substrate (the brain engine) — 🟩 DONE

*The platform every faculty runs on: the neuron + synapse + region layer. Biologically this is catalog clusters I (channels/intrinsic dynamics), J (synapses/plasticity), and the region framework. It is legitimately "hardware" — the world/body-and-engine layer — not a brain computation, but it is production-grade and default-on-spiking.*

| Component | Biology reproduced | Evidence |
|---|---|---|
| GPU spiking engine (`SimulationBridge`) — per-step pipeline (STP → conductance → noise → dynamics → plasticity → viz → record), CSR-sparse to 100K+ neurons, HDF5 checkpoint, deterministic seeding | The whole brain as one stepped dynamical system; conductance synapses (E_inh=−75 mV, 0.7× propagation), OU noise, homeostasis | `sim/bridge.py` (~8.4k lines); `tests/test_determinism.py` |
| Neuron models — Izhikevich-2007 (~30 presets), Hodgkin-Huxley (per-gate Q10, ~22 presets), AdEx (7 phenotypes), Resonate-and-Fire (phasor, for VSA binding) | L5 pyramidal, cortical FS, PFC, CA1/CA3 burst, thalamic relay/TRN, striatal MSN/GPe/GPi/STN/SNc, Purkinje/granule, spinal motor (catalog I); RF = Frady-Sommer 2019 | `sim/enums.py`, `sim/kernels.py`; RF: `2026-06-05-fhrr-production-switch-DONE.md` |
| Brain-region framework — declarative regions + pathways, transmission-gate (runtime synaptic-current gating), plasticity-gate (per-pathway freeze), graded inhibition | Interacting regions; thalamocortical dynamical gating (Logiaco-Abbott-Escola 2021); critical-period curricula | `sim/regions.py`; `tests/test_transmission_gate.py` |
| Plasticity family — STDP (soft-bound), Tsodyks-Markram STP, Hebbian, 3-factor reward-eligibility, homeostasis, structural plasticity, and the BDSP microcircuit deep-credit kernel | Bi-Poo STDP, STP, BCM/Hebbian, Schultz 3-factor DA, Turrigiano homeostasis; BDSP = Payeur/Urbanczik-Senn (catalog J) | `sim/kernels.py`; `run_benchmarks.py` (stdp-timing/ei-balance/stp) |
| Neuromodulator subsystem — DA/NE/5-HT/ACh concentration dynamics + receptor effects; a *shared* spiking-DA limbic core across nav + conversation | Tonic/phasic DA and generic neuromodulation (catalog C) | `sim/neuromodulators.py` |
| Continuous-learning lineage + tiered storage + persistence | Memory across sessions without catastrophic forgetting | `sim/lineage.py`, `sim/synapse_storage.py` |
| Two-compartment dendritic substrate (apical + basal, burst multiplexing) — 🟨 | Active dendrites, BAC firing, burst-dependent plasticity (catalog G.02, J) | `enable_bdsp` / `enable_bdsp_microcircuit` (built; deep-credit-on-spikes is a live boundary — see §7) |

*Open at the substrate level:* 19 GPU call-sites are still CuPy-only (Phase-2 backend guards); a few `tests/test_regions.py` cupy-path failures are tracked. STDP is *measured-wrong* for symmetric co-occurrence (Δt≈0 → zero update, 656k events / 0 change) — rate-Hebbian is the matched rule there.

---

## 5. The developmental path

*Ordered as a developing brain builds them. Each stage: the biological function reproduced (region/pathway + citation), status, what's done, what's open, and the next step.*

### 5.1 Perception — *seeing the world* · 🟨 PARTIAL
- **Goal:** turn the world's rendered sensory input into decorrelated, similarity-structured neural codes — the legitimate sensor front end.
- **Biology:** labeled-line transduction into thalamic relays (Müller's law; catalog E.01/E.07/K; Kandel Ch 17); center-surround / difference-of-Gaussians edge decorrelation (E.02/E.05; Ch 22); V1 oriented Gabor simple → phase-invariant complex cells (Hubel-Wiesel; E.08–E.10; Ch 22); ventral V1-V2-V4-IT + ATL convergence (E.12; Ch 24); dorsal V1-MT-MST-PPC where/how (Ch 25).
- **Done:** a real Gabor/V1 simple-cell bank + retina render (`sim/visual_cortex.py`), giving genuine visual similarity structure (within-cat cos 0.86 vs between 0.08; RSA pixel-provenance r=0.99); population/topographic coding validated as the rate-code-wall lift (47% → 100–108% of host); dorsal-vs-ventral routing used as a diagnostic principle (the nav cold-start was root-caused as a wrong-pathway).
- **Open:** no V2/V4/IT ventral hierarchy or explicit complex-cell stage; no separate dorsal where-stream module; no auditory/olfactory/somatosensory front ends; no constructive/predictive percept stage (needs L5 apical tuft = the dendritic frontier); the Gabor bank is fixed, not learned.
- **Next:** build higher perceptual hierarchy only where a downstream capability demands it (a learned V2-IT stage feeding concept formation).

### 5.2 Attention / orienting — *where to look* · 🟩 DONE (orienting) / 🟨 PARTIAL (attention)
- **Goal:** decide where to attend/orient and commit the movement; gate salience.
- **Biology:** superior-colliculus topographic orienting map, released by SNr disinhibition (H.25/A.07/E.22; Ch 35; McHaffie 2005); pontine PPRF pulse-step burst (Ch 35); FEF/PPC→V4/MT multiplicative attentional gain (E.15; Ch 25); locus-coeruleus NE arousal (C.05/C.14).
- **Done:** a spiking retinotopic SC orienting reflex, default-on at 1.16× host (6-seed SC/host 0.883); NE-like surprise-LR-boost meta-modulation.
- **Open:** SC is orienting-only (sustained-approach reward was SNR-limited); no FEF/PPC spatial-attention gain field; no pontine burst generator; gamma binding-by-synchrony not deployed for feature grouping.
- **Next:** a spiking attentional gain controller only when multi-object scenes demand it.

### 5.3 Action selection — *choosing what to do* · 🟩 DONE
- **Goal:** select one action from competing channels via disinhibition, accumulate evidence to a bound, and commit — fully on spikes.
- **Biology:** BG direct/indirect/hyperdirect pathways (A.01–A.03; Ch 38 Surmeier; Nambu 2002); WTA at tonic GPi/SNr + MSN lateral inhibition (A.04/B.04); reentrant Alexander-DeLong loops (A.05/A.06); LIP/PFC bounded evidence accumulation → commit burst (Wang 2002 NMDA attractor; Lo-Wang 2006).
- **Done:** a per-action BG disinhibition cascade (30 regions / 32 pathways) resolving the silent-motor trap (`g11_bg_runner`); D1/D2 asymmetry + MSN lateral WTA + PV-FSI; a **fully-spiking Wang-2002 accumulator + Lo-Wang commit-burst as the DEFAULT decision (host argmax retired), 6-seed grid-32 at ~1.16× host, 100% commit-burst**.
- **Open:** cross-projection "cheat #5" on hold (under-constrained); no hyperdirect global-stop declared; only the motor loop (no associative/limbic reentrant stripes); 7 of 8 striatal interneuron classes unmodeled; ~16% over-host residual = the irreducible commit-timing/finite-size floor (the honest brain-based deliverable).
- **Next:** add associative/limbic reentrant loops when non-motor selection (which fact to say, which topic) needs the same disinhibition computation.

### 5.4 Reward / value — *what was worth doing* · 🟩 DONE (RPE + 3-factor + drive) / 🟨 PARTIAL (critic)
- **Goal:** compute a reward-prediction-error signal, learn state values, let drive/value modulate learning and choice — on spikes, not by host formula.
- **Biology:** VTA/SNc phasic-DA RPE δ=r+γV(s′)−V(s) (O.02/C.22; Schultz 1998/2016); three-factor corticostriatal plasticity, opposite-sign D1/D2 (O.03); actor-critic striosome-V(s)/matrix-policy (O.18; Houk-Adams-Barto); vmPFC/OFC value → drift rate (O.19); AgRP/POMC drive (Ch 41); amygdala CS-US valence (Ch 42).
- **Done:** a spiking SNc RPE (neuralized); three-factor eligibility→reward→STDP; the value-critic validated *behaviorally load-bearing* (value-lesion collapses the high-value pick 0.90→0.49; 6-seed); **one shared spiking dopamine drives BOTH nav and conversation** (3 DA→composer routes incl. DA-gated recall vigor).
- **Open:** the explicit V(s) striosome/critic population is missing (the highest-leverage cue-shift fix); some reward/value is still biologically-shaped-but-host-computed (tracked); no amygdala/aversive substrate; the cue-shift RPE signature (DA transfer to a predictive cue) is open.
- **Next:** build the explicit spiking striosome V(s) critic (completes actor-critic + the cue-shift signature); convert remaining host-computed reward to a spiking perceived-reward circuit (de-risked).

### 5.5 Memory (encoding / consolidation) — *holding on to experience* · 🟩 DONE (mechanism) / 🟧 one BOUNDARY
- **Goal:** encode episodes with separation + completion, tag reactivatable engrams, consolidate to cortex without catastrophic forgetting — via replay, not backprop.
- **Biology:** trisynaptic loop EC-DG-CA3-CA1 (D.03; Ch 54); DG pattern separation (D.12); CA3 recurrent autoassociator completion (Marr 1971; D.05/D.13); Tonegawa engram tagging (D.14; Liu 2012); theta phase-precession sequence compression (D.18/D.24); SWR replay for credit assignment + planning (D.19/N.16; Foster-Wilson 2006); NREM slow→spindle→ripple nesting.
- **Done:** trisynaptic loop — DG separation cos 0.800→0.218 (58 pp), CA3 completion cos 0.748 from a 50% partial cue (Marr); Tonegawa engram-tagging production API (9 bridge methods), multi-tag cue retrieval 90% FULL; SWR/CLS consolidation with **no catastrophic forgetting** (hippo-OFF retention 92–94%, strict-silence anti-cheat identical 3/3); **replay REPLACES backprop-through-time** for the discourse register (retrodictive replay recovers the held slot to 109% of the BPTT value); positional (word,position) binding substrate.
- **Open (🟧 boundary):** deep compositional-engram consolidation strands as *hippocampal-only* (missing CA1→concept-pool wire + concept-pool weak dynamics); CA3 completion is a *point* attractor — sequence/trajectory completion is the real target; no theta pacemaker / phase precession; no slow-osc/spindle/ripple generators.
- **Next:** attack the compositional-consolidation boundary via the sequence-attractor CA3 + theta-compression path (research-gated); build a theta pacemaker to lift replay content quality — both feed the dendritic deep-credit lever.

### 5.6 Concept formation — *carving the world into categories* · ✅ EMERGENT
- **Goal:** discover categories and amodal concept hubs from experience, and learn high-order sequence structure — unsupervised, emergent, spiking.
- **Biology:** Marr-Albus cerebellar-granule sparse-expansion codon (F.12/F.13); ATL hub-and-spoke + Pulvermüller distributed word ensembles / semantic folding (G.10/G.20; Patterson-Lambon Ralph 2007; Garagnani-Pulvermüller 2018); Eichenbaum relational memory-space (D.02/D.21); HTM temporal memory: two-compartment dAP neurons + three-term permanence rule (Bouhadjar-Diesmann 2022).
- **Done:** **categories discovered unsupervised** at increasing internality — co-occurrence (EMERGE-30), varied overlapping contexts (32), a competitive self-organizing HTM spatial-pooler column block (33), perception-grounded via real Gabor/V1 (34) — fully spiking end-to-end (only sim/ edit: one additive kernel); the competitive pooler surpasses the fixed codon on overlapping categories (0.98 vs 0.56), Földiák trace closes the on-substrate generalization boundary; a **learned-from-conversation stream cortex** (online rate-Hebbian co-occurrence, corr(M,C) +0.686…+0.885, 6-seed, generalizing PPMI codes); the **HTM dAP sequence cortex** learns high-order next-symbol prediction with an intrinsic no-confab moat; a two-compartment dendritic dAP **pattern-completion surpass** (held-out 0.571 vs point-neuron 0.007, 6-seed on-substrate).
- **Open:** the on-bridge log-domain PPMI normalization is still a host-side read-out scaffold (§6); 320-scale stream-scaling needs a corpus-grounded taxonomy (validated at 64); emergent attractor *formation* (not hand-installed) for the dAP completion is a separate open problem; recurrent-microcircuit *sequence* learning hit a boundary.
- **Next:** build the on-bridge normalization circuit (retire the host scaffold); scale the stream cortex to 320; pursue emergent attractor formation (feeds the dendritic lever).

### 5.7 Language comprehension — *understanding what is said* · ✅ EMERGENT / 🟧 recursion boundary
- **Goal:** map word-forms → thematic roles and meaning, including non-canonical structure and non-local dependencies — learned, on spikes, no hand branch per sentence shape.
- **Biology:** Hickok-Poeppel dual-stream ventral semantics + Wernicke word selection (G.11/G.13; Ch 55); fronto-striatal reservoir form→role over the discovered closed-class configuration (Hinaut-Dominey 2013; G.12); Broca parse of reversible/relative-clause syntax (G.10/G.12; Ch 55).
- **Done:** a voice-invariant word-position→role **Hebbian parser** (active + passive assign the same agent), synaptic comprehension; a **fronto-striatal reservoir that LEARNS the form→role map** (retiring the hand labeler; adversarially hardened after a 5-skeptic workflow refuted a trivially-local first pass); **non-local relative-clause resolution ON SPIKES** across ~33 tokens (1.000 vs windows-at-chance; `OnBridgeLSM` recurrent region, region-silence lesion collapses); neural question-type routing replaces the host keyword router (held-out 1.000).
- **Open (🟧):** center-embedding recursion degrades past the reservoir's fading-memory depth (d*=2; theta-gamma stack-match → d*=3 then bounded); cross-language case/multicue parsing is a deliberate opt-in carve-out (Bates-MacWhinney; Phase 2/3); no auditory front end (comprehension via the learned parser, not sound→meaning).
- **Next:** build the *spiking* theta-gamma WM-buffer on the substrate (the recursion lever); co-reside the reservoir region on the shared nav/conv bridge.

### 5.8 Semantic reasoning — *inference beyond what was told* · ✅ EMERGENT / ⬜ symbol accumulation
- **Goal:** infer beyond told facts — inheritance, cancellation, transitivity — and accumulate evidence over symbols with learned reliability.
- **Biology:** hippocampal relational network / cognitive-map traversal (D.02/D.21; Dusek-Eichenbaum; O'Keefe-Nadel); Collins-Quillian hierarchical semantic memory; LIP logLR symbol accumulation, same drift-diffusion math as perceptual/value decisions (G.16/G.18; Ch 56).
- **Done:** **inheritance, multi-level taxonomy inheritance, per-dimension cancellation, and transitive inference all EMERGE** from overlapping/shared codes × the next-state predictor — no inference engine — over both host-designed and *discovered* categories; a 23-agent adversarial audit found + remediated a systematic held-out/control/framing defect class and every GO survived its corrected test.
- **Open:** no symbol-with-learned-reliability accumulator wired as a reasoning primitive (LIP logLR); fully-spiking fact-tag recall for a novel-perceived object is an honest 🟧 boundary (the hybrid where host routes which concept spiked works at 0.92); free open-world inference beyond learned/overlapping facts is a field wall (managed by domain-constraint + abstention).
- **Next:** wire a spiking logLR evidence-accumulator (reusing the action-selection drift-diffusion machinery); pursue fully-spiking fact-tag recall via the dendritic binder.

### 5.9 Language production — *speaking* · ✅ EMERGENT (bounded inventory) / 🟧 open prose deferred
- **Goal:** map meaning → correctly-ordered speech (function words + inflection + slot order + slot inventory) with every word spelled on spikes, structure self-organized from corpus.
- **Biology:** Broca grammatical encoding + articulation + closed-class "furniture" (G.12; Ch 55); pre-SMA/SMA sequences + Grossberg/Bullock-Rhodes competitive queuing (G.07/H.19; Ch 34); concept-pool→language_output articulation (G.08); Yang-Getz closed-class signature; Dominey-Hinaut construction router; Tomasello/Goldberg construction grammar.
- **Done:** **spiking Broca serial-order production** (frame-conditioned competitive queuing on real spikes, order 0.993 vs permuted 0.269, render-exact 1.00 6-seed, position-independence proven); **the ENTIRE grammatical structure self-organizes from corpus statistics** — function-word inventory, slot order, slot inventory — with the host FRAMES dict fully removed as input; **100% spiking A→W** (every content *and* function word decoded from `cp_firing_states[language_output]`); the **whole flagship turn co-executes in ONE cupy process** (single additive `SimulationBridge.xp` — the only sim/ edit in the EMERGE-56..71 arc); 7 corpus-mined constructions on spikes incl. transitive-motion PP, attributive/predicative adjective, core SVO, and ditransitive.
- **Open (🟧):** renders only the *bounded corpus-attested frame inventory*, NOT open prose (R4 = the ~4-orders-too-small wall, honestly deferred); A→W vocab caps ~16/bridge (needs more A→W bridges to scale, linear); real-corpus function-word precision is modest; the first emergent *open-ended* generation is token-level (see §5.10 / §7).
- **Next:** scale A→W bridges for a larger vocab, then drive the emergent-generation RUNG ladder to replace the transformer.

### 5.10 Discourse / conversation — *tracking who-what across turns* · ✅ EMERGENT (spiking core) / 🧩 fluid chat leans on the transformer
- **Goal:** track referents/events across a multi-turn conversation, resolve anaphora, abstain when unknown, hold a fluid grounded chat that grows through conversation.
- **Biology:** dlPFC persistent-activity WM for referents, D1-DA-stabilized (G.06/G.08; Ch 52); Grosz-Sidner attentional stack via PBWM input/output gating (O'Reilly-Frank 2006) + transmission-gate thalamocortical gating on a bidirectional route; Frankland-Greene lmSTC data-registers (two slow-NMDA attractor slots); Bogacz-Brown familiarity / perirhinal novelty.
- **Done:** the **D3 two-gate event register** — spiking discourse memory that tracks *who-now* vs *who-before* (push copies the running event to a held slot; pop reads it back), resumption 0.778 vs 0.139 gate-shut, RETURN-specific gate; **replay replaces BPTT** for the register (transition learned by Urbanczik-Senn clean-error feedback alignment, no weight transport, 97% next-emission); multi-turn **anaphora + centering** on a persistent spiking WM loop; a learned **no-confab moat** (Bogacz-Brown familiarity gate matches host abstention at V=320, zero breaches, gate-first); **the talkable-brain flagship console** (discover → reason → speak on spikes → describe → abstain → teach-live → remember → learn real-corpus facts).
- **Open:** multi-*referent* disambiguation is a characterized 🟧 negative (needs WTA biased-competition); fluent single-pass *synthesis* over multiple facts confabulates on the 21M (it lists/groups, doesn't synthesize); the fully-on-bridge end-to-end learning of the D3 transition to *accuracy* is open; the who-was-before capability was just wired into the talkable console.
- **Next:** build the WTA biased-competition multi-referent disambiguator; run the D3 transition-to-accuracy study; drive the emergent-generation ladder to replace transformer fluency.

### 5.11 Working memory / sequence / recursion — *holding structure* · 🟩 DONE (WM + reservoir) / 🟧 recursion
- **Goal:** hold an ordered set of items across a delay, process sequences with graded fading memory, match nested structure with a bounded stack.
- **Biology:** Lisman-Idiart theta-gamma 7±2 multiplex buffer (N.15; Lisman-Idiart 1995); Wang NMDA persistent-attractor WM surviving dt=1.0 (G.06/G.08; Wang 2002); Hinaut-Dominey graded fading-memory reservoir; theta-gamma stack-match (mirror-pair coincidence = LIFO pop) for bounded recursion.
- **Done:** NMDA persistent-attractor WM latches validated (dlPFC unification; D3 slots hold across clauses); the on-bridge recurrent **LSM holds a distal cue ≥16 fillers on spikes** (6-seed, region-silence lesion collapses); the **theta-gamma stack-match surpasses the reservoir to recursion d*=3** (buffer-slot-scramble collapses — the ordered slots are the structure).
- **Open (🟧):** the *spiking* theta-gamma realization on the substrate is research-gated/unbuilt (d*=3 uses a hand-structured bounded buffer); depth past ~3 is the human-faithful bounded limit (a feature, not a bug); no theta pacemaker.
- **Next:** build the spiking theta-gamma buffer on the substrate (shared lever with comprehension recursion + replay quality), including a theta pacemaker.

### 5.12 Artificial life — *living, developing, remembering* · 🟩 DONE (pieces) / 🟨 PARTIAL (unified whole)
- **Goal:** a persistent brain that lives, perceives+remembers its own experience, develops over time, is driven by one limbic core across all halves, and can be talked to about its life — on one brain.
- **Biology:** CLS complementary learning + self-replay (McClelland 1995 / Buzsáki); shared limbic dopamine (Niv-2007 vigor; O.19) + AgRP hunger modulating a DA-gated familiarity threshold; engram persistence (D.14) across sessions; cross-region synaptic interaction.
- **Done:** a **develop-over-time loop** (vocab 6→24, facts 2→11, recall 1.00 daily, *zero* forgetting, moat 0-FA every day via real stream-cortex Hebbian; week-1 capstone with 8 per-day consoles); a **one-brain nav+conv merge with cross-region SYNAPTIC interaction** (language→action 6-seed GO, perception→memory 6-seed GO, navigate-to-*compose*-then-answer 6-seed GO); **one drive both halves** (shared spiking DA modulates conversation + a hungry brain tightens the moat, 6/6 GO); **persistence/lineage** (live-and-remember 6/6 GO: resume 2/2, cold-start empty, lived recall collapses if the grounded codes are corrupted).
- **Open:** the learned spatial policy is a validated rate-proxy Q stand-in (the deferred Tier-4 dendrite wall); persistence is JSON re-instate (not the raw `cp_connections` tensor); open-endedness is encounter-driven on a corridor; the compose path still uses the FHRR-algebra scaffold + shared grounded codes.
- **Next:** consolidate the merged composer into a single co-resident OneBrain; replace the rate-proxy policy with the dendritic-learned policy (Tier-4 lever); move persistence toward raw-tensor continuity + richer open worlds.

---

## 6. Scaffolds still in place (to be replaced by simulated circuitry) · 🧩

*The honest inventory of host stand-ins on the critical path — each with what it stands for and the replacement plan.*

1. **The ~21M TinyStories transformer (open-ended fluency).** Stands for: open-domain fluent English the brain cannot yet produce as circuitry. Use: fluency *only*, gate-first behind the moat (never invoked on abstain); the brain does all cognition/grounding/verification. The forbidden permanent external model. **Replacement:** the emergent reservoir-readout LM RUNG ladder — **RUNG-1 is 6-seed GO** (a fixed on-bridge reservoir + a one-step-local-delta read-out beats bigram/bag/4-gram/non-recurrent on held-out CE, no BPTT); RUNG 2 (theta-gamma WM conditioning) / 3 (compositional generalization) / 4 (open-vocab spiking spell-out) / 5 (multi-clause discourse) are mapped-but-unbuilt and gated on scale + the dendritic lever.
2. **The VSA/FHRR composer's exact-inverse binding algebra.** Stands for: a learned, lossy, redundant cortical binder. Use: role-filler bind/unbind/bundle/cleanup for who-what/yes-no/negation/clauses/multi-hop — the *operations* are already spiking (resonate-and-fire + complex synapses; cleanup and the familiarity gate are neuralized == host). But the exact-inverse algebra + the decorrelated-full-precision-code *demand* is not what a cortex does; multi-attribute bundling is provably not learnable from scratch on point neurons, so a fixed ±1 self-inverse coincidence primitive stands in. **Replacement:** a learned cortical binder = the deferred dendritic step-3 frontier (§7).
3. **Host-computed nav readouts / reward** (biologically-shaped but host-side). **Replacement:** the fully-spiking decision is already the library default (host argmax retired); convert remaining host-computed reward to the de-risked spiking perceived-reward circuit + build the explicit striosome V(s) critic.
4. **Surrogate-gradient BPTT-SNN** (development stand-in, toy scale, path-f-hybrid branch). **Replacement:** the biological dendritic deep-credit rule — exactly as replay already replaced BPTT for the D3 register.
5. **PPMI log-domain read-out normalization** (host-side double-centring). **Replacement:** an on-bridge per-concept feedforward-inhibition + per-hub spike-frequency-adaptation circuit (designed, CYCLE 93b).
6. **JSON persistence + rate-proxy spatial policy** (artificial-life continuity). **Replacement:** raw `cp_connections` continuity + the dendritic-learned policy, gated on the deep-credit lever.

---

## 7. The honest frontier (what's left, and the genuine walls) · 🟧 / ⬜

1. **On-bridge deep credit assignment for recursive/sequential composition — the top emergence lever.** The feedforward arc is complete *off-bridge* (clean-error feedback-alignment clears depth-2 held-out 0.964, transport-free) and a two-compartment dendritic dAP completion surpass is 6-seed GO on-substrate, but *end-to-end on-bridge learning-to-accuracy* did not train at cheap scale (2026-07-07). Root-caused 2026-07-10 (a `bdsp_w_max=5.0` clip silenced the forward pass; fixed → the net learns above chance). **Reachability: actively-being-worked, not accepted** — the genuine depth-required composition + a "real neurons don't burst enough" tension are the live boundary. This wall gates open prose, the learned composer binder, and the Tier-4 spatial policy. *(Note: this is one candidate — the dendritic route — for lifting the ceiling; it is unproven-on-spikes and not on the critical path for the reservoir-generation ladder, which needs no deep credit.)*
2. **Transformer-free open-ended fluent generation.** RUNG-1 is GO but token-level over a bounded V=24 IID template grammar with `<unk>` content, ceilinged at the reservoir's ~depth-3 fading memory. **Reachability:** the RUNG ladder (2–5) is mapped but unbuilt and depends on scale (and, for depth, the dendritic lever); fully-transformer-free open-domain fluency is a genuine field wall even for LLMs (they use scale + retrieval). Honest interim: minimize + spiking-forward + fluency-only-behind-the-moat.
3. **Recursion beyond ~depth-3 (center-embedding).** The reservoir is fading-memory (d*=2); the theta-gamma stack-match reaches d*=3 then bounds at buffer capacity. **Reachability:** d*=3 *is* the biologically-faithful bounded human limit — an accepted boundary (a feature); the only open build is the *spiking* theta-gamma realization (research-gated, reachable).
4. **Compositional / relational memory consolidation.** The deep-consolidation compositional engram strands hippocampal-only at the cortical noise floor (the CA1→concept-pool wire is necessary-not-sufficient — concept-pool weak dynamics also block it). **Reachability:** the sequence-attractor CA3 + theta-compression path is the research-gated next lever (reachable, non-trivial).
5. **Multi-referent pronoun disambiguation.** Which of several held referents a bare pronoun binds — two converging negatives (not recency, not salience). **Reachability:** the mechanism is *specified* (WTA biased-competition between referent attractors); a bounded build when multi-referent dialogue is prioritized.
6. **Fully-spiking generalization fact-tag recall.** For a novel-perceived object it is at chance with a moat breach (the runner refused to weaken the moat); the *hybrid* (host routes which concept spiked) works at 0.92. **Reachability:** the all-spiking version is reachable via the dendritic binder + a spiking WTA fact-tag selector + the Bogacz-Brown gate.
7. **Free open-world inference + open-domain non-fact conversation.** The field walls — reasoning/chat beyond learned + code-overlap-derived facts. **Reachability:** not solved by anyone unconstrained; managed here (as LLMs do) by domain-constraint + grounded-retrieval + abstention — the honest scope, not a near-term wall.

**Breadth walls (each a scoped build, gated on a downstream need, not a fundamental boundary):** V2/V4/IT ventral hierarchy + a dorsal where-stream; auditory/olfactory/somatosensory front ends; the explicit striosome V(s) critic; an amygdala/aversive substrate; a theta pacemaker + phase precession; NREM slow-osc/spindle/ripple generators; hyperdirect cortex→STN global stop; associative/limbic reentrant BG loops.

---

## 8. Honest end-state assessment

**What is already LLM-like and brain-based** (mostly emergent, spiking, one brain, no `sim/` edit across nearly the whole EMERGE/D3 arc): comprehension (voice-invariant parser + fronto-striatal reservoir role-labeling + non-local relative-clause resolution on spikes); concepts/semantics *discovered from experience* (co-occurrence + real perception → categories → multi-level taxonomy → inheritance/cancellation/transitive inference, all emergent); a *self-organized* production grammar (function words + slot order + slot inventory mined from corpus, spoken 100% on spikes); a two-gate spiking discourse register (who-now vs who-before, with replay replacing backprop-through-time); multi-turn anaphora; and a learned no-confab moat — all co-executing in ONE cupy process for a full flagship turn.

**The two honest gaps between this and an LLM:**
1. **Open-ended fluent generation** is currently the ~21M transformer (the forbidden scaffold); the emergent replacement path exists with its first rung (RUNG-1 GO) but is token-level over a bounded grammar. The distance to open prose is roughly *4 orders of scale*.
2. **Deep credit assignment for recursive/sequential composition** — the learned binder, deep sequence composition, and the reservoir's recursion ceiling all bottleneck on the same missing rule (on-bridge burst-multiplexed dendritic deep credit). The feedforward arc is done off-bridge; end-to-end on-bridge learning-to-accuracy for genuine language depth is the live boundary.

**Bottom line:** the substrate, navigation, memory mechanisms, comprehension, concept-formation, and self-organized production are **DONE and largely emergent**. What separates this from an LLM is **(a) transformer-free open-ended fluency** and **(b) a deep-credit rule (dendritic is one unproven candidate; scale is another) that would lift the reservoir/composer ceilings** — plus field walls (true open-world inference, open-domain non-fact chat) that even LLMs manage only by scale + retrieval, and which this project manages by domain-constraint + grounded-retrieval + abstention. **This is a real, bounded, multi-month-to-longer distance — not a demo away, and not blocked.**

---

## Appendix A — biological systems reference (feature catalog clusters)

The biology is catalogued in `sim-catalog/references/feature-catalog.md` (~323 mechanism entries across 17 clusters A–Q, mapped to Kandel 6e). The catalog's `sim-status` is a dated snapshot; §4–5 above are the current truth.

| Cluster | System | Roadmap stage(s) |
|---|---|---|
| A / B | Closed BG action-selection loop / striatal microcircuit & WTA | §5.3 |
| C / O | Dopamine & neuromodulation / emotion, reward, motivation | §5.4, §5.12, §4 |
| D | Hippocampus & sequence learning | §5.5, §5.8, §5.11 |
| E / K | Sensory perception & cortical encoding / transduction | §5.1, §5.2, §5.6 |
| F | Cerebellum & error-correction | (supporting — predictive timing; presets only) |
| G | Working memory / PFC / cortical integration / **language** | §5.7–§5.11 |
| H | Motor & spinal output | §5.3 (body interface) |
| I / J | Channels & intrinsic dynamics / synapses & plasticity | §4 |
| L | Development & critical periods | §5.12 |
| N | Sleep, arousal & replay | §5.5, §5.11, §5.12 |

## Appendix B — how this roadmap is maintained

Updated as a standing part of the workflow (`neural-simulator` skill): when an arc lands a result / surpasses a boundary / replaces a scaffold / opens a frontier, the relevant stage's status badge + done/open bullets + next-step + citation is updated in the **same cycle**. Periodically (or on request) a deeper sync runs a deep-research pass — reading the sources in depth (the catalog, Kandel, the findings) to re-verify the biology map, the frontier, and the end-state assessment. This file — not any single findings doc or the `CLAUDE.md` arc log — is the intended at-a-glance **source of truth** for monitoring progress toward the goal.
