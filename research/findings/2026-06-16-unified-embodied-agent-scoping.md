# Unified embodied agent — scoping the ONE-brain integration of nav + perceive + generalize + compose + converse

**Date:** 2026-06-16
**Type:** read-only deep-research + design-scoping doc (the standing "deep-research/design-FIRST at a new
major arc" move). **No code was edited; no heavy GPU was run.** This document scopes the build; it does not
build it.
**Author role:** read-only scoping subagent.
**Status of the inputs:** every cognitive piece below is already validated in isolation or in a staged merge on
a real spiking `SimulationBridge` (the citations are opened and verified in §1).

---

## Terms (defined once; no undefined acronyms after this)

- **`SimulationBridge` / "the bridge":** the project's GPU/CPU spiking neural engine instance (`sim/bridge.py`).
  "One brain" = one `SimulationBridge` with one step loop holding all the regions as disjoint neuron-index
  slices.
- **merged bridge:** the single bridge built by `research/runners/nav_conv_merged_bridge.py`
  (`build_merged_nav_conv_bridge`) that already co-locates the navigation basal-ganglia action cascade (the
  body), the conversational Hebbian parser, the dlPFC dialogue planner, and — optionally — the resonate-and-fire
  composer slice and a bare perception region.
- **basal-ganglia (BG) cascade:** the per-action `cortex → striatum (D1/D2) → GPe/GPi → thalamus → motor`
  disinhibition circuit that selects each navigation move by which `sel_X`/`motor_X` pool wins (a neural
  decision, built by `research/runners/g11_bg_runner.py:build_bg_brain_regions`).
- **`cortex_it`:** the navigation **perception** region (the ventral "what"-stream object-identity ensembles).
  "The agent sees object X" = X's sub-ensemble of `cortex_it` fires. The environment renders the percept by
  driving X's band (a legitimate sensory render).
- **parser (`BridgeParser` / the merged ports):** a Hebbian-learned region mapping (word-position × voice) →
  grammatical role; comprehension. Voice-invariant (active and passive frames give the same agent).
- **dlPFC planner:** the dorsolateral-prefrontal working-memory loop (`cortex_ctx ↔ dlpfc_wm`, NMDA attractor)
  that runs spreading-activation dialogue planning (`elaborate`).
- **RF composer (`RFPhasorComposer`):** the resonate-and-fire phasor composer
  (`research/runners/rf_phasor_composer.py`). It stores facts and answers who/what queries by a
  **Fourier Holographic Reduced Representation (FHRR)** — a vector-symbolic algebra where each concept/role is a
  phasor (a vector of phases in `[0,1)^D`), bind = complex product, bundle = sum, unbind = multiply by the
  conjugate, cleanup = nearest concept. **`concepts[word]`** is its phasor codebook; **`grounded_codes=`** is its
  documented injection point for externally-supplied codes.
- **no-confab moat:** the composer abstains (returns `None`) on an unstored query. Never weakened to make a
  number look better — a breach is a HARD STOP. The load-bearing honesty guarantee.
- **NMDA:** the slow excitatory synaptic conductance (N-methyl-D-aspartate receptor; ~100 ms decay) the project
  uses as a per-region mask to give a region temporal integration / working-memory bistability.
- **rate code vs phasor code:** the perception/navigation/generalization regions carry information in **firing
  rate** (spike counts per window); the composer carries it in **phase**. These are not directly commensurable
  — the central representation handoff problem (§2).
- **handoff:** a place where one region's representation must drive another region in a different code.
- **generalization (across similar concepts):** perceiving a NOVEL object whose features resemble a known
  category, and having its concept neurons fire for that category → recall a fact about the category. This is the
  capstone arc completed this session (hybrid, 0.92 three-seed).

---

## 1. Diagnosis — what is validated, where each piece lives, the exact integration gap

The project has, across this session and before, validated the cognitive pieces of an embodied agent, each on a
real spiking bridge. I opened each cited runner/finding and confirmed the claims below.

| # | Capability | Bridge / runner it lives on | Representation it speaks | Status (verified) |
|---|-----------|-----------------------------|--------------------------|-------------------|
| 1 | **Navigate** — BG cascade selects each move neurally | merged bridge; `nav_conv_merged_bridge.build_merged_nav_conv_bridge` (the body) + `g11_bg_runner.build_bg_brain_regions` (cascade) | `motor_X`/`sel_X` **rate** (the disinhibition winner) | merged-bridge nav score byte-identical to standalone (STEP 2a GREEN_INERT; 3/6 seeds byte-identical, mechanistically seed-independent) |
| 2 | **Converse** — comprehend (parser) + plan (dlPFC) + fact memory + who/what recall + the no-confab moat | merged bridge; `nav_conv_merged_bridge` (parser+dlPFC ports) + `rf_phasor_composer.RFPhasorComposer` (co-resident on the `rf` slice, STEP 2b) + `brain_conversational_agent` (the surface) | parser = role-ensemble **rate**; composer = **phasor** | full conversational matrix co-resident at production `D=128`, `test_nav_conv_step2b_coresident` 7/7 GPU incl. the `is None` moat |
| 3 | **Compose-perceived** — navigate → perceive an object → ground its live `cortex_it` rate into a phasor concept code → compose a NOVEL (held-out) fact → answer + abstain | **already on the merged bridge** (perception co-resident); `navigate_to_compose_then_answer.py` + `_step3_grounded_codes_production_composer_derisk.py` (the grounding map) | perception **rate** → fixed complex projection → composer **phasor** | **6-seed GO** (held-out compose 1.000 ≫ floor 0.444, moat 6/6, LESION collapses, ISO 0, byte-identity True; NO `sim/` edit) |
| 4 | **Generalize across similar concepts** — perceive a NOVEL object through the real Gabor/V1 vision hierarchy → its concept neurons SPIKE for the right category → recall a fact about that category | **separate de-risk bridges** (NOT yet on the merged bridge); `_genfrontier_capstone_vision_to_concept_derisk.py` (vision→spiking-concept, stage 1) + `_genfrontier_graded_propagation_derisk.py` (NMDA read-out) + `_genfrontier_capstone_verbalize_derisk.py` (the hybrid recall) | vision **rate** (Gabor/V1 top-K) → perception→concept Hebbian convergence → NMDA **concept assembly rate (spikes)** → keys the composer | capstone **achieved via the hybrid, 0.92 three-seed** (0.75/1.00/1.00); the fully-spiking recall is an honest boundary; all four mechanism de-risks GO |

**The exact integration gap (two sentences).** Pieces 1, 2, and 3 already run on the ONE merged
`SimulationBridge` (`navigate_to_compose_then_answer.py` proves nav + parser + dlPFC + composer + perception
co-resident, 6-seed GO); piece 4 (generalization-across-similar-concepts) is validated only on its OWN dedicated
de-risk bridges and has **never been placed on the merged bridge** — it requires the real Gabor/V1 vision front
end, a structured-perception region, and a perception→concept Hebbian-convergence region that the merged bridge
does not yet contain. The unified embodied agent is therefore **one increment of distance from existing**: add
the generalization stack to the agent that already navigates-perceives-composes-converses, and route the new
generalization output (a spiking concept-category) into the composer's recall — the **one handoff the capstone
already proved works** (the hybrid, 0.92).

This is a **consolidation of separately-validated capabilities**, not a new capability — the same honest framing
the compose-perceived milestone carried. The scientific novelty is "one brain does all of it at once with no
regression," not a new cognitive primitive.

**Biology grounding (catalog + literature).** The integration target is exactly what the unified-cognition
biology describes:
- **A.05 reentrant cortico-BG-thalamo-cortical loops** (catalog) — parallel channels routed by the basal
  ganglia. The project's nav cascade IS such a loop; using it (and the dlPFC) to *route which representation is
  active* is the biologically-correct integration substrate, not a bolt-on switchboard.
- **G.20 consciousness / global workspace** (catalog, Kandel 6e Ch 56; Dehaene) — "selective gating of specific
  representations to a global workspace." Catalog sim-status: *missing — no global-workspace gate.* The unified
  agent's representation handoffs are a concrete, minimal instance of cross-region gated broadcast (transmission
  gates + the BG/dlPFC routing), i.e. the project's first step toward this entry.
- **Embodied-semantics discrepancy note** (catalog §High-level coverage) — "Kandel-grade perception involves
  transduction → parallel channels → topographic maps → hierarchical RFs → **multisensory binding** → predictive
  inference." The generalization stack (Gabor/V1 → structured perception → cross-modal Hebbian convergence with
  the word cortex) is precisely the multisensory-binding / convergence-zone computation (Damasio convergence
  zones; Pulvermüller distributed word webs, catalog G.20-family) that this note flags as compressed-away today.
- **Precedent for one spiking model doing many tasks: Spaun** (Eliasmith et al. 2012, *Science*) — 2.3M spiking
  neurons, perception → cognition → action, **basal-ganglia-routed**, **no rewiring across 6 tasks**. The
  project's merged bridge is a smaller, learned-where-Spaun-is-designed analogue of exactly this; Spaun is the
  existence proof that BG-routed selective recruitment of cortical components on one spiking substrate is
  tractable. (See Sources.)

---

## 2. The representation-handoff map — the crux, decisively assessed

The pieces speak different neural codes. The unified agent needs the full set of handoffs to interoperate on one
bridge. Below, each handoff is mapped: what code → what code, whether it is already validated (with the
de-risk), and the cleanest faithful mechanism using the project's **standing routing patterns** (transmission
gates `RegionPathway(transmission_gate=)` + `bridge.set_transmission_gate`; the masked RF ops `rf_kick(neuron_mask=)`;
plasticity-gate isolation `cp_plasticity_rate_gain=0`; the per-region NMDA mask `BrainRegion(enable_nmda=True)`;
population reads of `cp_firing_states`; the fixed complex grounding projection M).

| # | Handoff (from → to) | code → code | Validated? | Cleanest faithful mechanism | Verdict |
|---|---------------------|-------------|-----------|-----------------------------|---------|
| H1 | **environment → perception** (render object identity into `cortex_it`) | world → **rate** | YES (the (B) probe + compose-perceived; a legitimate sensory render) | environment drives `cortex_it`'s orthogonal band; for the generalization path, render a SHAPE → real Gabor/V1 (`sim.visual_cortex.build_v1_simple_weights`) → top-K V1-complex drive | **VALIDATED — reuse verbatim** |
| H2 | **nav cascade → body** (which move) | **rate** → action | YES (STEP 2a, merged nav score byte-identical) | read the `sel_X`/`motor_X` winner; step the agent (host = the body, legitimate) | **VALIDATED — reuse verbatim** |
| H3 | **perception rate → composer phasor** (ground a perceived object as a bindable filler) | **rate → phasor** | YES (compose-perceived 6-seed GO) | the fixed complex projection M: `composer.concepts[o] = angle(M @ live_cortex_it_rate)` (`_step3_grounded_codes_production_composer_derisk.grounded_phases`) | **VALIDATED — reuse verbatim** |
| H4 | **parser role rate → composer** (comprehended SVO → stored fact) | **rate → phasor (indirect)** | YES (the merged agent `hear`) | `parser.parse(...)` returns role→word strings; `composer.store(agent, action, patient)` keys the phasor codebook by those strings (a string handoff, not a code handoff — clean) | **VALIDATED — reuse verbatim** |
| H5 | **perception rate → generalizing concept assembly (spikes)** (a perceived NOVEL object fires concept neurons for the right category) | **rate → rate (spikes), via learned convergence + NMDA** | YES (4 de-risks GO: convergence 0.92; vision→concept stage-1 0.75; graded-prop NMDA read-out spikes 146/cue; Option-B structured perception RSA-to-pixels 0.99) | a structured-perception region (Gabor/V1 top-K) + a `concept` region with `enable_nmda=True`; a **rate-Hebbian** perception→concept pathway learns the convergence (STDP is the WRONG rule — symmetric co-occurrence, CYCLE-95 finding); the NMDA concept assembly integrates the sparse drive to spikes | **VALIDATED on de-risk bridges; NEW on the merged bridge** (the integration work) |
| H6 | **generalizing concept spikes → composer recall** (verbalize the generalization: recall the matched category's fact) | **concept rate (spikes) → composer query** | YES — **this is the one handoff the capstone already proved** (the hybrid, 0.92 three-seed) | read which concept-category SPIKED (population read of `cp_firing_states`), then key the validated `RFPhasorComposer` recall by that category's concept code + its intact moat (a brain-to-brain route; host only routes WHICH concept spiked, as the merged bridge already routes elsewhere) | **VALIDATED (hybrid) — the fully-spiking version is a bounded honest boundary, NOT required for the unified agent** |
| H7 | **composer ↔ dlPFC** (fact memory feeds dialogue planning) | phasor (kb) → dlPFC rate | YES (the merged `elaborate` builds the association graph from `composer.kb`) | `_assoc_graph()` over `composer.kb` → the dlPFC spreading-activation Control | **VALIDATED — reuse verbatim** |
| H8 | **co-residence isolation** (all the above on ONE bridge, no cross-corruption) | — | YES (5a plasticity isolation; 5b masked RF ops; STEP 2b co-resident 7/7) | `cp_plasticity_rate_gain=0` freezes conv/perception weights under the live nav reward-STDP+dopamine stressor; the masked `rf_kick(neuron_mask=rf_mask)` keeps RF phasor state off the Izhikevich `v/u`; `stdp_w_max`/`hebbian_max_weight` raised above frozen weights to defeat the two ungated clips | **VALIDATED — reuse the established discipline** |

**The decisive verdict on the crux.** **Seven of the eight handoffs are already validated**, and the eighth (H5,
perception→generalizing-concept) is validated on its own de-risk bridges and is "new" only in the sense of
**not-yet-placed-on-the-merged-bridge** — it is region-addition work, not a new mechanism. Critically, the one
**cross-code handoff that the unified agent newly requires end-to-end (H6: a spiking concept-category keying the
phasor composer's recall) is exactly the handoff the capstone-verbalize stage already demonstrated at 0.92**.
There is **no un-de-risked cross-code wall** between the generalization output and the conversational pipeline.
The representation-handoff feasibility verdict is therefore **FEASIBLE by staged reuse** — the codes interoperate
through three already-proven bridges: the fixed complex projection M (rate→phasor for grounded objects, H3), the
string handoff (parser→composer, H4), and the population-read + composer-key (concept-spikes→recall, H6).

The one residual subtlety to respect (not a wall): H6's **fully-spiking** form (a downstream fact-tag region with
winner-take-all + a spiking familiarity gate) is an honest BOUNDARY (`capstone-verbalize` option a: cat-acc ≈
chance + moat breach). The unified agent must use the **hybrid** H6 (read the clean concept spikes → key the
validated composer recall), which keeps the answer and the abstention on the validated mechanism and preserves
the moat. The fully-spiking H6 is a bounded follow-on, explicitly out of the unified-agent's load-bearing scope.

---

## 3. Ranked, STAGED build plan — cheapest-first integration increment, then to the full unified agent

The cheapest-first principle: the compose-perceived agent (`navigate_to_compose_then_answer.py`) is **already on
the merged bridge with perception co-resident**. The minimal increment is to **add the generalization stack to
that agent** and route its output into the composer recall (H6 hybrid). Everything else is already integrated.

### Stage 0 — CPU/numpy integration smoke (the cheapest-first de-risk; see §5 for the GATE)
Before any GPU build, run a single-seed CPU smoke that wires the generalization read (H5+H6 hybrid) onto a
small merged-style bridge and confirms the route closes end-to-end (a novel perceived shape → spiking concept
category → composer recall of a category fact → moat abstains on a no-category shape). Numeric gate in §5.

### Stage 1 — generalization stack co-resident on the merged bridge (the load-bearing increment)
**Adds:** to `build_merged_nav_conv_bridge`, three additive default-off regions (appended LAST, after `rf` and
`cortex_it`, so all existing index bases stay byte-identical — the exact pattern `co_resident_perception`/`rf`
already use): (i) a `structured_perception` region sized to the V1-complex feature dimension
(`N_ORIENTATIONS·V1_POSITIONS_PER_DIM² = 2048`); (ii) a `concept` region (`F·n_per`, `enable_nmda=True`); (iii)
a fixed block-diagonal read-out is NOT needed on the bridge — the unified agent reads the concept assembly's OWN
clean spikes (the `capstone-vision-to-concept` 0.75 / graded-prop 0.92 path), then routes via H6 hybrid. The
perception→concept pathway is plastic (rate-Hebbian) and is **trained in a dedicated setup pass then frozen**
(plasticity-gate isolation), exactly as the parser is trained-then-frozen in `build_merged_nav_conv_bridge`
step 5. **Gate:** Option-B → A closes on the merged bridge for a held-out novel shape (concept-spike cat-acc >
chance, every seed; flat-distinct baseline ≈ chance; derangement collapses; moat intact) AND the existing
co-residence battery (nav byte-identical, conversational 7/7) still passes. **Anti-cheats:** §4 (1)–(5), plus the
Gabor-structure-preservation assert (within-category active-set overlap > between) and the flat-distinct +
derangement controls the de-risks already carry.

### Stage 2 — the live unified episode (the end-to-end demonstration)
**Adds:** the agent NAVIGATES, and on arrival at an object's cell EITHER (a) grounds a flat-distinct object into
the composer as a bindable filler (the existing compose-perceived path, H3) OR (b) — for a recognizable-category
object rendered as a SHAPE through Gabor/V1 — drives the generalizing concept assembly (H5) and, on a who/what
query about that category, recalls via the H6 hybrid; abstains on a truly-novel no-category object. **Gate:** in
ONE episode on ONE bridge, the agent (i) navigates (score not regressed, byte-identical to the compose-perceived
nav), (ii) composes a held-out perceived-object fact (compose-perceived parity), (iii) generalizes a novel
similar perceived object to its category and recalls a category fact (≥ the hybrid 0.92 single-seed bar), (iv)
answers the conversational who/what matrix, and (v) the no-confab moat abstains on every unstored query / no-
category object — with NO regression on any of (i)–(v). **Anti-cheats:** the union of every piece's anti-cheats
(§4), asserted simultaneously in the one episode.

### Stage 3 — multi-seed validation (the standing 6-seed discipline)
Run Stage 2 across 6 seeds (42/43/44/100/101/102). **Gate:** all five sub-capabilities pass on all 6 seeds, the
moat never breaches on any seed, and the byte-identity / co-residence asserts hold on all 6. (Honest watch:
per-seed nav-encounter variance from seed-randomized layout — a seed grounding < 2 objects is a body-trajectory
scaffold, not a failure; report per-seed. Vision noise — the stage-1 vision→concept was 0.75 with one seed at
0.50, 2× chance — so report per-seed cat-acc and use the population-code levers if a seed dips, never loosen the
moat.)

**Ranking rationale.** Stage 1 is ranked first because it is the only genuinely-new integration (region
addition + a trained-then-frozen convergence pathway on the merged bridge); Stages 2–3 are then assembly +
validation of already-proven routes. The cheapest-first ordering deliberately front-loads the single risk
(does the Gabor/V1 structured perception + the NMDA concept convergence survive co-residence with the full
nav+conv stack and the byte-identity discipline) into a small CPU smoke (Stage 0) and a single-seed GPU gate
(Stage 1) before the expensive multi-seed live episode.

---

## 4. Anti-cheats — the integration must PRESERVE every validated capability, with NO regression

The standing discipline: the unified agent is a *consolidation*; **it must not regress any constituent
capability, and it must not weaken the no-confab moat.** The integration is held to the union of every piece's
anti-cheats, asserted together:

1. **Nav-not-regressed (byte-identity).** The navigation score on the unified bridge is byte-identical to the
   compose-perceived nav (the established STEP 2a `GREEN_INERT` standard: standalone-vs-merged score
   byte-identical, per-phase). Tool: the `nav_gate2a_aggregate` discipline. The new regions are appended LAST
   with `internal_density=0` and NO `cp_connections` out-edges into navigation (the `rf`/`cortex_it` pattern), so
   the nav index bases and dynamics are provably unchanged.
2. **Conversational matrix preserved.** `test_nav_conv_step2b_coresident` (7/7) + the `BrainConversationalAgent`
   suite pass VERBATIM on the unified bridge, **including the three `is None` no-confab assertions**
   (`what_does`/`elaborate`/`describe`). The moat is asserted, never relaxed.
3. **Compose-perceived parity.** The `navigate_to_compose_then_answer` battery (held-out compose ≫ floor; the
   cut-after-encode LESION collapses the compose; ISO-perception → 0 grounded; provenance: the grounded code is
   the live-rate projection) passes on the unified bridge unchanged.
4. **Generalization controls.** For the generalization path: the FLAT-distinct perception baseline is ≈ chance
   (visual structure is load-bearing), the category-DERANGEMENT control collapses (the transfer is the LEARNED
   vision-category↔concept-category correspondence, not geometry coincidence), the Gabor-structure-preservation
   assert holds (within-category active-set overlap > between), and the generalization moat survives (a
   visually-novel no-category shape does NOT drive confident category spikes / does NOT confabulate).
5. **Co-residence isolation under the live stressor.** The conversational + perception + concept weights stay
   byte-frozen (`cp_plasticity_rate_gain=0`, with `stdp_w_max`/`hebbian_max_weight` ≥ the max frozen weight to
   defeat the two ungated weight clips — the Hebbian clip at `sim/bridge.py:6509` and the STDP/Hebbian clip at
   `:6814`, both clip `cp_connections.data` regardless of the plasticity gate) across a live navigation burst (reward-STDP +
   the global dopamine `scope="all"` + Hebbian). The RF composer's complex weights (`cp_rf_w_re/im`) are
   array-disjoint from `cp_connections` and so immune; the masked `rf_kick(neuron_mask=)` keeps the phasor state
   off the Izhikevich `v/u`.
6. **The moat is sacred.** Any moat breach anywhere (composer abstention, the generalization familiarity check)
   is a HARD STOP — the agent must NOT confabulate to manufacture a GO. (This is why H6 uses the validated-
   composer hybrid, not the fully-spiking fact-tag recall that breached the moat in `capstone-verbalize`.)

**`sim/` edit flag.** **No `sim/` edit is expected.** Every piece reached its GO by reuse-by-import, and the
merged-bridge co-residence the unified agent rides (the masked `rf_kick`) is the **already-landed,
default-off-byte-identical STEP-2b/5b edit** — the unified agent merely *uses* it. The generalization stack is
built from existing `sim/` primitives (the brain-region framework, the per-region NMDA mask, `enable_nmda`,
rate-Hebbian plasticity, `sim.visual_cortex` Gabor/V1, the engram API). If any stage discovers a needed `sim/`
change, it must be raised for owner byte-level diff review FIRST (the standing rule); the strong prior is that
none is needed.

---

## 5. The cheapest-first de-risk — the smallest test that the first increment works, with a numeric GATE

**Goal:** confirm, as cheaply as possible, that the **generalization stack composes with the conversational
pipeline through the H6 hybrid** before committing the merged-bridge region-addition + GPU build.

**The test (Stage 0, CPU/numpy, single seed, minutes):**
- Reuse-by-import the validated convergence + read pieces (`_genfrontier_capstone_vision_to_concept_derisk`'s
  vision→spiking-concept on a small `F`/`n_per`, OR — if Gabor/V1 is too heavy for numpy in minutes — the
  synthetic structured-perception variant `_genfrontier_graded_propagation_derisk` which is the same convergence
  with a controlled structured input), plus a small `RFPhasorComposer` (`D` small, CPU).
- Wire the H6 hybrid: for a HELD-OUT novel-category cue, read which concept-category SPIKES (population read of
  `cp_firing_states`), key the composer's recall by that category's concept code, and answer a who/what query
  about a fact stored for that category. Run the no-confab control: a no-category cue must NOT recall (abstain).

**The numeric GATE (single seed; promote to GPU + multi-seed only if met):**
- **Generalization read:** held-out concept-category spike accuracy **> chance (1/n_cat) with a positive
  same-vs-other margin** (the de-risks land 0.75–0.92; the gate is simply "> chance + positive margin" at one
  seed to keep it cheap).
- **H6 recall:** the category fact recalled via the hybrid for the held-out cue is **correct ≥ chance + a clear
  margin** (the de-risk hybrid is 0.75–1.00 per seed; gate ≥ 0.50 single-seed as the cheap bar).
- **Moat (HARD):** the no-category cue **abstains (recall returns `None` / familiarity below the held-out band)**
  — **zero moat breaches**. A breach FAILS the de-risk outright (do not proceed; do not loosen the gate).
- **Co-residence sanity (cheap):** building the small structured-perception + concept regions alongside a parser
  stub leaves the parser's read byte-stable (the 5a discipline, in miniature).

If the gate is met, promote to Stage 1 (the merged-bridge region addition + the single-seed GPU gate). If the
generalization read or H6 misses, localize on the de-risk knobs the runners already expose (`top_k`,
`n_concept_per`, `nmda_ratio`, `read_weight`, `perc_scale`, `epochs`) — these are bounded refinements, not walls.
If the moat breaches, STOP.

---

## 6. Honest feasibility + scope

**Is the full unified agent tractable by staged reuse? YES.** The integration is one genuinely-new increment
(the generalization stack on the merged bridge, Stage 1) followed by assembly + validation of already-proven
routes. Three independent facts make this a low-risk consolidation rather than an open research problem:
1. **Pieces 1–3 are already on the ONE merged bridge** (`navigate_to_compose_then_answer.py`, 6-seed GO) — the
   hard co-residence/isolation work (nav byte-identity, the masked RF ops, plasticity-gate freezing of conv
   weights under the live nav stressor) is DONE and reusable.
2. **Every handoff is validated** (§2): 7/8 outright, and the 8th (H5) is validated on its own bridges and is
   region-addition work, not a new mechanism. The **one new end-to-end cross-code handoff (H6) is the exact
   handoff the capstone already proved (0.92)**.
3. **No `sim/` edit is expected** — the whole arc is reuse-by-import on primitives that already exist.

**Where a handoff could still bite (the single biggest risk).** H5 on the merged bridge: the generalization
mechanism was de-risked on **dedicated bridges sized for it** (perception 1600–2048, concept `F·100`, NMDA on,
specific drive/Hebbian-rate knobs). Folding it onto the merged bridge means it must co-exist with the full
nav+conv dynamics and survive the **byte-identity discipline** (the new regions must not perturb nav, and the
trained-then-frozen convergence pathway must not be eroded by the ungated Hebbian decay during nav episodes —
exactly the foot-gun `finalize_conv_for_nav_gate` already handles by masking plasticity by index). The risk is
not a representational wall (the info is present — cat-acc 0.92); it is a **tuning/co-residence risk** (does the
NMDA concept convergence still spike cleanly when it shares a bridge and a global dopamine `scope="all"` with the
nav cascade, and does the byte-identity hold). This is precisely what Stage 0 (cheap CPU smoke) and the Stage 1
single-seed GPU gate are designed to retire before the expensive multi-seed run.

**In scope (this arc): integration.** Wiring the validated pieces into one bridge with no regression; the H6
hybrid recall; the multi-seed live episode. The agent navigates, perceives, generalizes across similar objects,
composes facts, and converses — on one `SimulationBridge`.

**Out of scope (separate axes, deliberately):**
- **Production-scale vocabulary** — the unified agent integrates at the validated probe scale (the merged bridge
  defaults to the 16-word probe vocab; generalization at a handful of categories). The 320-/2,048-concept
  multi-bridge production cortex (#17/#19) is a **separate scaling axis**, not part of the integration.
- **The fully-spiking H6** — the all-spiking fact-tag recall + spiking familiarity gate (the brain-based ideal)
  is an honest BOUNDARY (`capstone-verbalize` option a); the unified agent uses the hybrid (clean concept spikes
  → validated composer recall). Completing the fully-spiking version (fact-tag lateral inhibition + the
  Bogacz-Brown familiarity gate) is a bounded follow-on, not load-bearing.
- **A learned cortical bind / the dendritic substrate** — the composer's exact-inverse FHRR algebra remains the
  principled idealization (the 2026-06-16 capability map settled that multi-attribute bundling is not learnable
  from scratch on point neurons; the fixed self-inverse primitive is load-bearing biology, not a shortcut). The
  unified agent rides the validated fixed composer; the dendritic rewrite (for a *learned* generalizing cortex
  beyond the PPMI/convergence path) is a separate, deferred owner call.
- **The semantically-STRUCTURED-everywhere cortex** — the generalization path generalizes across **similar
  perceived objects** (vision-derived category structure); it does not yet make the composer's *abstract* concept
  codes (verbs) carry semantic similarity. That is the deferred PPMI/dendritic frontier, not this integration.

**Biology grounding (why this is the right next arc).** Integrating perception, memory, language, and action on
one BG-routed spiking substrate is the unified-embodied-cognition target the catalog flags as missing (G.20
global workspace; the multisensory-binding discrepancy note) and that Spaun established as tractable (one spiking
model, BG-routed, perception→cognition→action, no rewiring across tasks). The unified agent is the project's
concrete, learned-where-possible step toward that target — and, per the owner's standing "one brain doing
everything" theme and the load-bearing artificial-life-with-a-brain-analogue goal, the highest-leverage next arc.

---

## Sources

- Eliasmith, Stewart, Choo, Bekolay, DeWolf, Tang, Rasmussen (2012). *Spaun: A Perception-Cognition-Action Model
  Using Spiking Neurons* / *A Large-Scale Model of the Functioning Brain*, Science 338. (2.3M spiking neurons,
  basal-ganglia-routed, 6 tasks, no rewiring.) [compneuro.uwaterloo.ca/files/2012-Spaun.pdf](https://compneuro.uwaterloo.ca/files/2012-Spaun.pdf)
- Eliasmith, Gosmann et al. (2016). *BioSpaun: A large-scale behaving brain model with complex neurons.*
  [arxiv.org/pdf/1602.05220](https://arxiv.org/pdf/1602.05220)
- *Neural Brain: A Neuroscience-inspired Framework for Embodied Agents* (2025) — perception-cognition-action
  integration on spiking substrates for embodied agents. [arxiv.org/html/2505.07634v1](https://arxiv.org/html/2505.07634v1)
- Kandel 6e Ch 56 (consciousness / global workspace; Dehaene) — via catalog G.20.
- Project catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` — A.05 (reentrant
  cortico-BG-thalamo-cortical loops), G.20 (consciousness / global workspace), the embodied-semantics
  discrepancy note (§High-level coverage summary).
- Project findings/runners verified inline: `navigate_to_compose_then_answer.py` +
  `2026-06-16-navigate-to-compose-then-answer.md`; `nav_conv_merged_bridge.py`;
  `_genfrontier_capstone_vision_to_concept_derisk.py`, `_genfrontier_graded_propagation_derisk.py`,
  `_genfrontier_capstone_verbalize_derisk.py` + `2026-06-16-generalization-*.md`;
  `_step3_grounded_codes_production_composer_derisk.py`; `rf_phasor_composer.py`.
