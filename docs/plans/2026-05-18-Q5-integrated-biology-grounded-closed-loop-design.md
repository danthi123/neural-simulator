# Q5 — Integrated biology-grounded CLOSED-LOOP architecture: compositional capability hypothesized EMERGENT from composing the project's already-validated subsystems the way real biology integrates them (NOT another isolated mechanism) (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered gate
> -> honest propagation EVERY outcome -> ITERATE FOLLOWING REAL
> BIOLOGY (no hand-back, no "owner needed", no declaring-unfit). The
> no-confab moat byte-identical throughout. Owner-directed (option 3),
> autonomous.

## 0. The owner critique this design is founded on (do NOT re-litigate)

Every prior arm was tested as an ISOLATED standalone mechanism and
each found "insufficient alone". But real biology achieves
compositional/conversational cognition ONLY through MANY complex
systems operating as ONE integrated CLOSED LOOP. "Insufficient in
isolation" is therefore the EXPECTED signature if the capability is
emergent from the integration -- NOT a verdict that the local
implementation is unfit. The prior isolated-arm negatives are
**evidence FOR the integration hypothesis, not against the program.**
This design stops testing isolated parts, asks WHY biology does this
(grounded below), and reproduces the biological INTEGRATION, iterating
by improving biological fidelity when a result is negative.

## 1. Genuine biological investigation -- WHY biology achieves this (grounded, cited)

Compositional/conversational cognition in the brain is, by
well-established systems-neuroscience consensus, an EMERGENT property
of a closed multi-region loop -- no single region does it:

- **Complementary Learning Systems (CLS) + dynamic hippocampal
  re-engagement.** Episodic/relational structure is rapidly bound in
  hippocampus; systems consolidation slowly extracts a generalizable
  *schema* in neocortex; remote recall RE-ENGAGES hippocampus
  (bidirectional, not one-way) [Wang & Morris, hippocampal-neocortical
  interactions in consolidation/reconsolidation; McClelland 1995 CLS;
  2024 bidirectional-CLS sequential-consolidation model]. The 2024
  framing **"hippocampo-neocortical interaction as compressive
  retrieval-augmented generation"** is decisive for our goal:
  problem-solving = a *generalizable neocortical prior COMBINED with
  relevant specifics loaded from hippocampus/ongoing experience*. That
  is biologically a grounded-generation loop -- exactly the
  "grounded + fluent compose" target, and it is a LOOP, not a part.
- **PFC variable-binding working memory gated by basal ganglia via
  thalamocortical disinhibition.** PFC "stripes" implement
  variable-binding (key/query -> content slots); the basal ganglia
  provide *selective dynamic gating* of which stripe updates, via
  striatal disinhibition of dorsomedial thalamocortical loops
  (superficial->deep transfer); multiple stripes are independently
  gated, so several role-filler bindings are maintained simultaneously
  = compositional structure held in WM [Frank/O'Reilly PBWM,
  *Interactions between frontal cortex and basal ganglia in working
  memory*; *PFC and BG control access to working memory*, Nature
  Neuroscience; 2024 *adaptive chunking improves WM capacity in a
  PFC-BG circuit*]. Composition is NOT an engram or a credit signal --
  it is BG-gated updating of PFC variable-binding stripes.
- **Neuromodulators time the loop.** DA gates/credits the BG-WM update
  and learning; ACh sets cortical precision + the plasticity window;
  NE signals surprise / learning-rate. They COORDINATE *when* each
  stage operates (Kandel ch. on neuromodulation; catalog C.30/B.3).

**The honest synthesis (the WHY):** composition emerges from the loop
{hippocampus rapidly binds the relational episode} -> {BG-gated
thalamocortical updating loads/maintains that structure in PFC
variable-binding stripes} -> {neocortical CLS-consolidated schema
supplies the generalizable prior} -> {composed readout = schema prior
conditioned on the hippocampally-bound specifics, BG-sequenced} ->
{neuromodulators time every stage} -> {trustworthy gate suppresses
ungrounded output}. **Our sim has built and VALIDATED every node of
this loop individually, but NEVER composed them into the closed loop
and tested the whole.** That specific INTEGRATION is the gap.

## 2. Reframe: the unit of analysis is the INTEGRATED CLOSED LOOP

Hypothesis under test (pre-registered): a compositional capability
EMERGES from the integrated closed loop that provably does NOT exist in
any single node alone. Necessity+sufficiency is established the way
biology establishes it -- a **lesion/ablation study**: the full loop
shows the capability AND every single-node ablation abolishes it.

## 3. Validated subsystems REUSED byte-UNMODIFIED as the loop's nodes (DRY; protected)

Mapping each biological node to an ALREADY-VALIDATED in-repo subsystem
(reuse byte-UNMODIFIED; do NOT rebuild; do NOT modify protected/frozen
modules or the no-confab moat):

| Biological node | Validated in-repo subsystem (CLAUDE.md) |
|---|---|
| Neocortical concept/schema layer | v16 16-pool `build_biological_brain_regions` (88.75% multi-seed bidirectional binding) |
| Neocortical schema consolidation (CLS) | Phase 1.3 hippocampus consolidation (3/3 strict anti-cheat, no catastrophic forgetting) |
| Hippocampal rapid relational binding | trisynaptic loop P1 (D.12 separation / D.13 completion) + engram tagging D.14 (`start/commit/stimulate`, 87.5%/90%) |
| PFC variable-binding working memory | `dlpfc_wm` + Cluster-G per-region NMDA bistable attractor (validated 2.00 +/- 0.00) |
| BG-gated thalamocortical updating | the Phase-B BG cascade (validated GO, 74% improvement) -- biologically-correct repurpose: BG gating of WM access IS the same circuit as BG action-selection (PBWM) |
| Neuromodulatory timing | the neuromodulator subsystem (DA/ACh/NE, validated) |
| DA-gated learning INSIDE the loop | temporal-credit TD(lambda) (validated PASS n=70/71) |
| Trustworthy output gate | the no-confab abstention moat (`abstention_gate`, 7/7, byte-identical the whole arc) |

**The genuinely-NEW net-new work is ONLY the biology-faithful
CLOSED-LOOP WIRING** that composes these validated parts per Section 1
-- specifically the consensus-identified missing link: *BG-gated
thalamocortical updating of PFC variable-binding stripes, conditioned
on hippocampally-bound specifics + the CLS-consolidated neocortical
schema, neuromodulator-timed, no-confab-gated*. NOT any new isolated
mechanism.

## 4. Candidate integrated architectures (2-3) + recommendation

- **Q5-min (RECOMMENDED): the minimal closed loop that the biology
  says is the irreducible core of composition.** Compose, in
  `sim.bridge` via the REUSED `build_biological_brain_regions` +
  region/pathway framework: cortical concept pools (v16) <->
  hippocampal engram binding (D.14) <-> PFC variable-binding WM
  (`dlpfc_wm` NMDA bistable) with **BG-gated updating** of the PFC
  slots (Phase-B cascade repurposed) <-> CLS replay
  consolidating bound structure into the cortical schema, all
  **neuromodulator-timed** (DA gate/credit, ACh plasticity-window),
  output through the no-confab moat. Net-new = the wiring + the
  gating/timing controller ONLY. Honest ceiling: tests whether
  *relational composition* (bind (role,filler) tuples, maintain
  several, compose a novel combination) emerges from the loop where
  every single-node ablation abolishes it. Cheaply falsify-first
  de-riskable (abstract closed-loop simulation of the gating algebra
  before the heavy in-bridge build). RISK: the integration controller
  is genuine net-new and load-bearing -> dedicated adversarial review.
- **Q5+seq: Q5-min + BG-gated cognitive-move SEQUENCING** (multi-step
  compositional generation, not just one bind). Honest ceiling: a
  short grounded compositional *utterance*. Higher risk/scope; build
  ONLY if Q5-min passes (biology builds sequence on top of binding).
- **Q5-full: + Generator-F as the consolidated-schema fluent prior**
  (the biological "RAG": neocortical schema = fluent generator,
  hippocampus = grounded specifics, composed in-loop, no-confab-gated).
  Highest ceiling (grounded fluent composition) but largest scope;
  ONLY after Q5+seq, and explicitly the heaviest -- not first.

**Recommendation: Q5-min first**, with a mandatory cheap falsify-first
precursor. The honest staged ceiling: Q5-min PASS = the emergent
relational-composition core exists in the integrated loop and in NO
ablation; THEN Q5+seq; THEN Q5-full. Each stage is its own
pre-registered gate; biology builds them in that order, so we do too.

## 5. Pre-registered scale-confidence THREE-STATE gate (frozen, NEVER tuned) -- on the INTEGRATED capability

Net-new `research/runners/q5_core.py` (own frozen `_Q5_*`; mirrors the
adversarial-hardened `*_core` discipline EXACTLY; imports only
stdlib+typing; imports/mutates no existing core). The gate measures a
RELATIONAL-COMPOSITION accuracy (bind N (role,filler) pairs, maintain,
read out a queried/composed combination) on the integrated loop, with
the biologically-correct ABLATION controls:

- **V1 (instrument soundness):** the full integrated loop on a NO-GAP
  trivial bind reaches a justified high bar (the loop machinery can
  learn the bijection at all).
- **Science:** the full loop reaches the composition bar on the
  genuine compositional task (novel role-filler combination, multiple
  maintained bindings).
- **ABLATION controls (must ALL fail -- this is the emergence test):**
  loop minus hippocampal-binding; minus PFC-WM-maintenance; minus
  BG-gating; minus neuromodulator-timing; minus CLS-consolidation.
  Each single-node ablation MUST fall to/near chance. (Emergence =
  whole passes AND every node necessary. A config-crank cannot make
  ALL ablations fail while the whole passes unless the integration
  genuinely does the work -- this is the hard-to-fake instrument.)
- **SCALE LADDER (frozen):** composition load N in {2, 4, 8}
  simultaneously-maintained bindings; SCALE-CONFIDENT iff every rung:
  V1 sound + science met + every ablation fails, AND composition
  accuracy non-decreasing up to a frozen tol across N, AND holds at
  the largest rung. `q5_scale_confidence` mirrors the established pure
  fail-closed aggregator pattern.
- Outcome map (honest, never spun): SCALE-CONFIDENT-PASS /
  WORKS-SMALL-NO-SCALE-CONFIDENCE / FAIL / VOID -- all propagated.

## 6. Falsify-cheaply precursor (MANDATORY, runs FIRST)

Throwaway pure-numpy/CPU probe: an ABSTRACT closed-loop simulation of
the gating algebra ONLY (engram-bound key->slot; BG-gated PFC-slot
update/maintain; schema-prior compose; neuromod timing) -- NOT the
heavy spiking bridge. Pre-registered cheap THREE-STATE with the SAME
ablation controls: does relational composition emerge in the abstract
loop AND vanish under every single-node ablation, scale-confidently
across N{2,4,8}? GREEN -> heavy in-bridge build via writing-plans.
NEGATIVE/VOID -> **do NOT declare unfit; ITERATE FOLLOWING BIOLOGY**:
identify which integration point the abstract loop got biologically
wrong (per the Section-1 cited mechanisms -- e.g. wrong gating
discipline, wrong timing order, missing precision-weighting), fix the
biological fidelity, re-pre-register if a bar must change for
soundness (transparently, never toward an outcome), and re-run. Only
genuine exhaustion of biology-grounded integration refinements (each
propagated) is a terminus -- and even then the next step is the next
biology-identified integration gap, autonomously, NOT owner-deferral.

## 7. ITERATE-FOLLOWING-BIOLOGY discipline (the core behavioral change)

A negative result NEVER triggers "declare local implementation unfit
and pivot to another isolated part." It triggers: (1) name, with
citation, the specific biological integration mechanism the wiring is
not faithfully reproducing; (2) fix that biological fidelity in the
net-new wiring (reused validated nodes stay byte-UNMODIFIED); (3)
re-run the SAME pre-registered gate (bars frozen; a bar changes ONLY
if required for instrument SOUNDNESS, transparently logged, never
toward an outcome -- the established STRENGTHEN-only discipline); (4)
propagate honestly; (5) continue autonomously. This is bounded by
honest exhaustion of cited biological refinements, not by a hand-back.

## 8. Anti-cheat plan (non-negotiable, unchanged)

Pre-registered FIXED-bar THREE-STATE + ablation controls + scale
ladder in `q5_core` (own frozen `_Q5_*` NEVER tuned; no new GLOBAL
bar; does not import/mutate any existing core); falsify-cheaply FIRST;
DEDICATED ADVERSARIAL REVIEWER on the load-bearing integration
controller + `q5_core` BEFORE Phase B (probe: is composition genuinely
emergent [whole passes, every ablation fails] not a wiring artifact;
is each ablation faithful [identical to full minus exactly that node];
are the validated nodes reused byte-UNMODIFIED not copy-tweaked; can a
non-discriminating/V1-broken run be scored PASS; frozen bars
result-movable; any new autograd); controller trust-but-verify EVERY
diff with the FULL protected set byte-empty (all original protected +
every frozen `*_core` + constrained_decode_core/gate + q2r_core/gate +
engram_bootstrap_gate + abstention_gate+test 7/7 + grounded_decode +
generator_g_core + sim validated modules + text_minimal_isolation
`build_biological_brain_regions` byte-UNMODIFIED); mandatory smell-test
scrutinizing a nominal PASS HARDER than a FAIL (recompute from the
single recorded JSON; ablations genuinely fail; emergence genuine; NO
re-run/NO bar-tuning/NO overclaim); honest propagation EVERY outcome
(findings + capability_status pillar n=79.. + schema-green + push BOTH
remotes). MONITORING DISCIPLINE: decisive runs use Bash
`run_in_background` OR foreground; completion ACTIVELY confirmed
(poll JSON+process); NEVER a false "I will be notified". Honest
ceiling baked in + NEVER spun: an integrated PASS = a biology-grounded
multi-system closed loop shows an emergent compositional capability
scale-confidently where every ablation fails -- explicitly NOT claimed
GPT-class / open-ended-fluent unless the gate genuinely shows it; the
prior validated assets + honest boundaries are unaffected; the
isolated-arm negatives stand and are reinterpreted as predicted-by
this hypothesis, not refuted.

## 9. Build sequence (subagent-driven; anti-cheat) -- detailed by writing-plans

Cheap falsify-first abstract-loop probe (controller-run, foreground,
recorded) is the GATE. If GREEN: Task 0 grounding pin -> Task 1
`q5_core.py` (frozen `_Q5_*` THREE-STATE + ablation logic +
`q5_scale_confidence`, fully specified) -> Task 2 `q5_gate.py` (the
biology-faithful closed-loop wiring composing the REUSED validated
nodes via `build_biological_brain_regions` + region/pathway framework
+ engram API + neuromodulator subsystem + BG cascade + dlpfc_wm, ALL
byte-UNMODIFIED; the net-new integration controller; kill-safe;
`--tiny` smoke not propagated; ASCII) -> Task 3 dedicated adversarial
reviewer BEFORE Phase B -> Phase B no-harm -> Task 5 CONTROLLER-ONLY
decisive run (monitored to active completion) + mandatory smell-test +
honest propagation n=79 both remotes -> ITERATE-FOLLOWING-BIOLOGY
autonomously on any non-PASS (Section 7), then Q5+seq, then Q5-full,
each its own pre-registered gate. If cheap NEGATIVE/VOID:
iterate-following-biology (Section 6), autonomously, no hand-back.
