# ROADMAP — a brain you can talk to

> **Status: source of truth.** This document is the canonical record of *what the project has accomplished, what it is working on now, and what is left* on the path to the goal. It is updated regularly as part of the standing workflow (see the `neural-simulator` skill). If a claim here conflicts with a finding in `research/findings/`, the finding wins and this file is corrected.
>
> **Last synced:** 2026-07-10.

---

## 1. The goal

Simulate a **real brain** as the core of an **artificial lifeform** that learns and grows — and whose primary demonstrated behaviour is **communication**: the owner can hold a genuine, fluid conversation with it. The north-star bar is **conversational capability approaching that of a large language model**, reached the honest way — *emergently, from experience, on a single spiking substrate* — not by bolting on an external model.

Everything in this roadmap serves that goal. Capabilities are instrumental; the deliverable is a living, communicating brain analogue whose language ability is genuinely its own.

## 2. The non-negotiable constraints (what "the honest way" means)

These define what counts as progress. A capability that violates them is a **scaffold** to be replaced, not a milestone.

1. **No shortcuts, cheats, or host scaffolding.** The only legitimate host (non-neural) code is the **world/body interface** — a simulated world, rendering the brain's senses, and enacting its motor output. *Everything between sensation and action is neurons, synapses, and their communication.* A capability realized by a Python formula (a reward, a reflex, an argmax, a template) is a documented shortcut to be converted to a spiking/synaptic mechanism.
2. **Emergent, not hand-designed.** Structure should be *developed from experience* (a learning substrate + a training stream), not installed by hand. The standing bar (2026-07-10): for any new conversational capability, ask *"does it emerge from a learning substrate, or am I hand-installing it?"* New hand-built mechanisms are allowed only as (a) temporary scaffolds on the ladder to their learned replacement, or (b) probes that map a substrate limit.
3. **One brain, one spiking substrate.** All faculties run as regions on a single `SimulationBridge`, interacting through synapses — not as a battery of separate models.
4. **Biology-grounded.** Each mechanism corresponds to a real brain region / pathway / function, cited to the source (Kandel 6e, the feature catalog, primary papers). `sim/` edits are legitimate when a faithful biological mechanism needs them (additive, default-off, byte-identical-when-off).
5. **No permanent external ML artifact as a faculty.** A transformer/LLM may be a *temporary* scaffold for fluency, but the end state *simulates the circuitry*. *"If Broca drives articulation, we simulate Broca."*
6. **Honest negatives are deliverables.** A boundary maps what the substrate can and cannot do on its own — and launches the search for the next mechanism. It is never a place to stop.

## 3. How to read this roadmap (status legend)

Each capability below carries a status:

| Badge | Meaning |
|---|---|
| ✅ **EMERGENT** | Done, and the structure was *learned from experience* on the spiking substrate. |
| 🟩 **DONE** | Done and validated on-substrate, but with a hand-designed component (a fixed algebra, a structural primitive) that is biologically defensible but not itself learned. |
| 🟨 **PARTIAL** | Works in a reduced form / at reduced scale / with a characterized gap. |
| 🟧 **BOUNDARY** | A characterized limit — validated as *not yet reachable* by the approaches tried; the next mechanism is named. |
| 🧩 **SCAFFOLD** | A temporary host stand-in (an external model, a template) that must be replaced by simulated circuitry. |
| ⬜ **OPEN** | Not yet built. |

Validation convention: a result is a **GO** only after multi-seed confirmation (dev seeds 42/43/44 → blind 100/101/102) with anti-cheats (lesion / permuted / memorization-floor / oracle-ceiling / scramble) and, for anything entering the record as a surpass, an **adversarial-verify** pass. "6-seed GO" is the standard bar.

---

## 4. The substrate (the brain engine)

*This is the platform every faculty runs on. Biologically it is the neuron + synapse + region layer — catalog clusters I (channels/intrinsic dynamics), J (synapses/plasticity), and the brain-region framework.*

| Component | Biology reproduced | Status | Evidence |
|---|---|---|---|
| GPU spiking engine (`SimulationBridge`) | Large-scale networks of conductance-based neurons | 🟩 DONE | `sim/bridge.py` (~8k lines); CuPy/NumPy backends |
| Neuron models — Izhikevich 2007, Hodgkin-Huxley (per-gate Q10), AdEx, Resonate-and-Fire | Membrane dynamics, spike generation, bursting; ~30 region-specific presets (catalog I) | 🟩 DONE | `sim/kernels.py`, `sim/enums.py` |
| Synapses & plasticity — STDP, STP (Tsodyks-Markram), Hebbian, homeostasis, NMDA/AMPA/GABA, eligibility traces, transmission gates | LTP/LTD, short-term facilitation/depression, receptor kinetics, three-factor learning (catalog J) | 🟩 DONE | `sim/kernels.py`, `sim/bridge.py` |
| Neuromodulator subsystem — DA / NE / 5-HT / ACh concentration dynamics + receptor effects | Neuromodulation (catalog C) | 🟨 PARTIAL | `sim/neuromodulators.py` (DA deployed; NE/5-HT/ACh framework-supported) |
| Brain-region framework — declared regions + directed pathways on one bridge | Cortical/subcortical parcellation + projections | 🟩 DONE | `sim/regions.py` |
| Continuous-learning lineage + persistence | Memory across sessions without catastrophic forgetting | 🟩 DONE | `sim/lineage.py`, `sim/synapse_storage.py` |
| Dendritic / two-compartment substrate (apical + basal, burst multiplexing) | Active dendrites, BAC firing, burst-dependent plasticity (catalog G.02, J) | 🟨 PARTIAL | `enable_bdsp` / `enable_bdsp_microcircuit` (built; deep-credit-on-spikes is a BOUNDARY — see §6) |

---

## 5. The developmental path (the capabilities)

*Ordered as a developing brain would build them — from sensation, through action, memory, and concepts, to comprehension, reasoning, production, and conversation. Each stage: the biological function reproduced (region/pathway + citation), the status, what's done, what's open, and the next step. **The detailed per-stage content below is being assembled from a deep-research pass (accomplishments inventory + biology map + frontier analysis) and will be filled in this section.***

> **[SECTION UNDER ASSEMBLY]** — the stages below are the confirmed structure and biology backbone (grounded from the feature catalog clusters A–Q and Kandel 6e); the detailed accomplishment/status/citation content per stage is being merged from the research synthesis and will replace these stubs.

### 5.1 Perception — *seeing the world*
Biology: retina → V1 (Gabor simple/complex cells) → dorsal "where" + ventral "what" streams (catalog E, K; Kandel Ch 21–25). Status + detail: _to fill._

### 5.2 Attention & orienting — *where to look*
Biology: superior colliculus orienting, dorsal-stream salience (catalog E; Kandel Ch 25, 29). Status + detail: _to fill._

### 5.3 Action selection — *choosing what to do*
Biology: the closed basal-ganglia loop — cortex → D1/D2 striatum → GPe/GPi → thalamus → motor disinhibition; a spiking accumulator + commit-burst decision (catalog A, B; Kandel Ch 38; Wang 2002, Lo-Wang 2006). Status + detail: _to fill._

### 5.4 Reward & value — *what was worth doing*
Biology: midbrain dopamine reward-prediction-error (SNc/VTA), striatal value, limbic drives (catalog C, O; Kandel Ch 43; Schultz). Status + detail: _to fill._

### 5.5 Memory — *holding on to experience*
Biology: the hippocampal trisynaptic loop EC → DG → CA3 → CA1 (pattern separation + completion), engram tagging, sharp-wave-ripple replay + systems consolidation (complementary learning systems) (catalog D, N; Kandel Ch 52–54; Marr 1971, Tonegawa, Buzsáki). Status + detail: _to fill._

### 5.6 Concept formation — *carving the world into categories*
Biology: cortical convergence / anterior-temporal-lobe hub-and-spoke; categories from co-occurrence statistics; perception-grounded concepts (catalog E, G; Patterson-Lambon-Ralph). Status + detail: _to fill._

### 5.7 Language comprehension — *understanding what is said*
Biology: the **dual-stream model** — ventral (superior/middle temporal → semantic interface, sound→meaning) + dorsal (posterior superior temporal → arcuate fasciculus → Broca, sensorimotor); **Wernicke's area** auditory→semantic; a fronto-striatal reservoir mapping word-order → thematic role (catalog G.10–G.13; Kandel Ch 55; Hickok-Poeppel; Hinaut-Dominey 2013). Status + detail: _to fill._

### 5.8 Semantic reasoning — *inference beyond what was told*
Biology: inheritance / transitivity / cancellation over shared and overlapping codes; relational binding (catalog D, G; Collins-Quillian; Dusek-Eichenbaum). Status + detail: _to fill._

### 5.9 Language production — *speaking*
Biology: **Broca's area** — speech production + grammatical processing + sensorimotor mapping; competitive-queuing serial order; self-organized grammar from distributional statistics (catalog G.12; Kandel Ch 55; Grossberg/Bullock-Rhodes; Yang-Getz). Status + detail: _to fill._

### 5.10 Discourse & conversation — *tracking who/what across turns*
Biology: attentional-stack discourse memory (push on a boundary, pop on a return), working-memory maintenance of referents, factored event registers (catalog G; Grosz-Sidner; Frankland-Greene; O'Reilly-Frank PBWM). Status + detail: _to fill._

### 5.11 Working memory, sequence & recursion — *holding structure*
Biology: prefrontal persistent activity; theta-gamma multiplexed slot buffer (Lisman-Idiart); reservoir fading memory (catalog G.06/G.08, N.15; Kandel Ch 55; Lisman-Idiart 1995). Status + detail: _to fill._

### 5.12 The generation frontier — *producing open, novel, grounded language*
Biology: a fixed recurrent cortex (reservoir) + a locally-trained read-out predicting the next token; predictive-coding output learning (Hinaut-Dominey; reservoir/ESN language modelling). Status: the **active frontier** — Rung 1 GO (an emergent, on-bridge, no-backprop next-token language model); Rungs 2–5 open. Detail: _to fill._

### 5.13 Artificial life — *living, developing, remembering*
Biology: homeostatic drives, develop-over-time from lived experience, one-brain integration of perception + action + conversation, persistence across "sleep" (catalog O; Kandel Ch 43, 51). Status + detail: _to fill._

---

## 6. Scaffolds still in place (to be replaced by simulated circuitry)

*Honest inventory of the host stand-ins on the critical path. Each names what it stands for and the replacement plan.* **[UNDER ASSEMBLY — merged from the frontier analysis.]**

- **A minimized (~21M-parameter) transformer generator** — the temporary source of *open* fluency inside the brain's gate→constrain→verify loop. Stands for: open-ended articulation. Replacement: the emergent reservoir-generation ladder (§5.12).
- **The VSA composer's exact-inverse binding algebra** — a principled idealization (Spaun/Semantic-Pointer-Architecture) for role-filler binding. Stands for: a learned cortical binder. Replacement: learned representations flowing through a fixed biological coincidence/multiplicative binding primitive (partially done) + the emergent path.
- _Further scaffolds to fill from the synthesis._

## 7. The honest frontier (what's left, and the genuine walls)

*What stands between the current state and LLM-matching conversation, rated for reachability at this project's scale.* **[UNDER ASSEMBLY — merged from the frontier analysis.]**

- **Open-ended fluent generation** beyond a bounded, corpus-attested construction inventory — the active frontier (§5.12). Reachable to *bounded-but-emergent* fluency without deep credit; the honest ceiling is the reservoir's fading memory (~depth-3).
- **Deep multi-layer credit assignment on spikes** — needed for deep hierarchical abstraction. Status: 🟧 BOUNDARY — burst-dependent (BDSP) / microcircuit / population-coded credit all fail to train on real spikes at cheap scale; the residual is credit-structure. An *unproven* candidate for a far-off ceiling, **not** a blocker for the generation path — deprioritized as a low-priority parallel probe.
- **Scale / data** — the documented "R4" gap (~orders of magnitude in neurons/corpus) between a controlled-grammar token LM and open-domain fluency. A lever, to be *measured* not assumed.
- _Further walls + the honest end-state assessment to fill from the synthesis._

## 8. Honest end-state assessment

*What "LLM-matching conversation" realistically means on this substrate, and the genuine remaining distance.* **[UNDER ASSEMBLY.]**

---

## Appendix A — biological systems reference (feature catalog clusters)

The project's biology is catalogued in `sim-catalog/references/feature-catalog.md` (~323 mechanism entries across 17 clusters, mapped to Kandel 6e). Cluster overview (sim-status per the catalog; the project has since advanced several beyond the catalog's snapshot — see §4–5):

| Cluster | System | Roadmap stage |
|---|---|---|
| A | Closed BG action-selection loop | §5.3 |
| B | Striatal microcircuit & WTA | §5.3 |
| C | Dopamine & neuromodulation | §5.4, §4 |
| D | Hippocampus & sequence learning | §5.5, §5.8 |
| E | Sensory perception & cortical encoding | §5.1, §5.2, §5.6 |
| F | Cerebellum & error-correction | (supporting — predictive timing) |
| G | Working memory / PFC / cortical integration / **language** | §5.7–§5.11 |
| H | Motor & spinal output | §5.3 (body interface) |
| I | Channels & intrinsic dynamics | §4 |
| J | Synapses & plasticity rules | §4 |
| K | Sensory transduction | §5.1 |
| L | Development & critical periods | §5.13 |
| N | Sleep, arousal & replay | §5.5, §5.13 |
| O | Emotion, reward, motivation | §5.4, §5.13 |

## Appendix B — how this roadmap is maintained

Updated as a standing part of the workflow (`neural-simulator` skill): when an arc lands a result, its stage status here is updated and the finding is cited; when a scaffold is replaced or a boundary surpassed, §6/§7 are revised. This file — not any single findings doc or the CLAUDE.md arc log — is the intended *at-a-glance source of truth* for external monitoring of progress toward the goal.
