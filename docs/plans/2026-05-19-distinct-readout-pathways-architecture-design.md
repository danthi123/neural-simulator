---
type: plan
status: live
date: 2026-05-19
---

# Distinct readout pathways: order-preserving episodic recall vs order-invariant concept/working-memory recall (design only)

The biologically-correct fix the phase-factored attempt missed: the
episodic-sequence readout and the concept/working-memory readout are
served by genuinely SEPARATE pathways that never share one
order-constrained trace -- the order-preserving hippocampal
trisynaptic pattern-completion path for episodic order, the
order-invariant neocortical schema path for the concept/working-memory
content. This document confronts the pre-registered necessity
partition head-on and reaches a decisive, honest conclusion. No code
in this document; this is a design, not an implementation. No plan, no
build, no GPU run is authorized by this file.

Date: 2026-05-19. Status: design pass for the next program step that
was pre-registered by the twice-convergent program-level finding
(citation in Section 1). Reuse-only; net-new is a per-trial controller
plus distinct-pathway wiring; NO new learning, NO automatic
differentiation, NO edit to any frozen module.


## 0. One-paragraph orientation

The integrated-loop instrument was built and adversarially hardened;
episodic-sequence binding became perfect; one gap remained
(role-selective working-memory binding). Applying the project's
validated concept-binding mechanism to that gap recovered
working-memory selectivity but destroyed episodic order, because the
validated concept mechanism structurally needs a SHUFFLED encode order
while the theta-gamma episodic store structurally needs FIXED
order==index. The phase-factored design tried to dissolve that by
putting the shuffle in a separate offline replay phase -- but it
routed BOTH readouts ultimately through the SAME consolidated trace,
so the offline shuffle that concept selectivity needs DESTROYED the
episodic order the consolidated trace then had to carry (full
episodic = 0.0), while skipping consolidation PRESERVED the online
tag's order (no-replay episodic = 1.0). That inverts the frozen
necessity partition's duty for the remove-consolidation lesion and the
frozen verdict correctly returns "cannot conclude" (VOID-by-
construction). This document designs the architecture the
phase-factored attempt missed: stop routing both readouts through one
trace. In real biology these are two distinct structures with two
distinct readout pathways. The episodic-order readout is served by the
order-PRESERVING hippocampal trisynaptic pattern-completion pathway
(NOT the consolidated trace). The concept/working-memory readout is
served by the order-INVARIANT neocortical-schema pathway built by
interleaved offline consolidation. They never share one
order-constrained trace, so the order-monotone constraint and the
order-shuffled constraint live in physically separate pathways and
cannot contend. Section 4 then confronts the unavoidable consequence:
under a biology-faithful distinct-pathways architecture the
remove-consolidation lesion does NOT collapse episodic recall (because
episodic order is carried by the online trisynaptic store, not the
consolidation system) -- which is precisely the frozen partition's
`_HELPER_EP` duty for `no_cls_replay`. The design states the decisive,
honest conclusion (b) and specifies what replaces the falsified
necessity prediction.


## 1. Problem and biology grounding

### 1.1 The twice-convergent program-level finding (restated crisply, NOT re-litigated)

Sources, in load-bearing order:

- `research/findings/2026-05-19-phase-factored-VOID-by-construction-and-the-twice-convergent-necessity-partition-finding.md`
  (the latest; the load-bearing finding).
- `research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-between-validated-concept-binding-and-episodic-store.md` ⛔ [SUPERSEDED — the encode-order framing was generalized by `research/findings/2026-05-19-FIFTH-convergent-UNIFYING-TERMINAL-the-integrated-loop-necessity-instrument-is-biologically-unsatisfiable-by-the-CLS-division-of-labor.md`: the conflict IS the complementary-learning-systems division of labor (5 convergent routes), and phase-factoring RELOCATES rather than dissolves it; the phase-factored build then landed VOID/two-horns, `research/findings/2026-05-30-phase-factored-decisive-iteration2-engram-wm-SOUND-but-VOID-two-horns-characterized.md`]
  (the first program-level finding).
- `docs/plans/2026-05-19-phase-factored-consolidation-architecture-design.md`
  (the prior, now-VOID design).
- The FROZEN verdict `research/runners/integrated_loop_core.py`
  (unchanged since commit `2048750`; FROZEN, never to be edited).
- The faithful phase-factored runner
  `research/runners/integrated_loop_gate.py` at HEAD (commit
  `2582992`); now VOID-by-construction.

The established structural facts (these are NOT re-opened here; they
are the foundation this design rests on):

1. Single online pass. The project's validated concept-binding
   mechanism (embodied co-firing plus a topographic prior; the
   validated 16-pool multi-pool concept substrate) needs a SHUFFLED /
   interleaved encode order -- the shuffle is precisely what breaks
   the within-kind winner-take-all dominant attractor so multiple
   same-kind pools become individually selectable. The theta-gamma
   episodic store recovers the sequence as the argsort of per-item
   activity-peak times, so it structurally needs presentation-order ==
   binding-index. In one online encode pass these are directly
   contradictory. GPU-measured, not inferred.

2. Phase-factored (online theta-ordered encode + offline
   shuffled-replay consolidation, BOTH readouts ultimately from the
   consolidated trace). This RELOCATES the contradiction, it does not
   dissolve it: the offline shuffled replay required for concept
   selectivity DESTROYS the episodic order the consolidated trace then
   has to carry (GPU: full episodic = 0.0), while skipping
   consolidation preserves the online tag order (no-replay episodic =
   1.0). Therefore the frozen `_HELPER_EP` duty of `no_cls_replay`
   inverts: under the frozen partition removing consolidation MUST
   collapse the episodic readout, but with episodic routed off the
   consolidated trace, removing consolidation does the OPPOSITE. The
   frozen verdict, recomputed unchanged, returns VOID. This is
   VOID-by-construction.

3. The deeper reading the latest finding states explicitly:
   biology-faithful architectures realize a DIFFERENT causal necessity
   structure than the frozen partition assumes; the partition itself
   may be the falsified element. A pre-registered necessity prediction
   that no faithful architecture realizes is a falsified prediction --
   a legitimate scientific result, not a local-implementation
   deficiency to keep patching.

### 1.2 The hippocampal-neocortical division of labor (the catalog basis)

The conflict is diagnostic of a conflation. In the brain these two
operations are NOT one trace with one shared order constraint. They
are two distinct structures, with two distinct readout pathways and
two OPPOSITE order properties:

- Episodic-sequence recall is ORDER-PRESERVING and is served by the
  hippocampal trisynaptic loop. Entorhinal cortex layer II projects to
  dentate gyrus, which sparsifies overlapping inputs (PATTERN
  SEPARATION); dentate gyrus projects to CA3, whose recurrent
  collaterals form a Marr autoassociator that reconstructs the full
  bound pattern from a partial cue (PATTERN COMPLETION); CA3 projects
  to CA1 via the Schaffer collaterals. The project's catalog grounds
  this as entries D.03 (trisynaptic pathway, Kandel 6e Ch 54), D.12
  (pattern separation, Kandel 6e Ch 54 pp 1357-1360), and D.13
  (pattern completion, Kandel 6e Ch 54 pp 1342, 1360-1361; Marr 1971
  autoassociator). The project has a VALIDATED trisynaptic loop:
  `research/runners/validate_trisynaptic_loop.py` confirms D.12
  (separation: dentate cosine 0.218 from input cosine 0.800 --
  ~58 percentage-point orthogonalization) and D.13 (completion: CA3
  cosine 0.748, target > 0.7) at single-seed, with D.12 robust 3/3
  multi-seed. Critically for this design: a CA3 autoassociator
  retrieves the WHOLE bound pattern, including the theta-ordered
  serial structure written ONLINE at encode, from a partial cue --
  WITHOUT requiring any offline replay. Episodic order is a property
  of the ONLINE trisynaptic store. This is the established biology
  (the project's own validated D.12/D.13 asset), not an aspiration.

- Concept / schema is ORDER-INVARIANT and is served by neocortex. A
  concept's identity is stable regardless of the order in which the
  episodes that taught it occurred. The complementary-learning-systems
  account (McClelland 1995; O'Reilly; Buzsaki 2013) builds stable,
  selective neocortical representations by INTERLEAVED replay during
  offline (quiet/sleep) consolidation -- sharp-wave-ripple-gated
  replay reorders experience across many offline cycles. The project's
  VALIDATED Phase-1.3 hippocampus-to-cortex consolidation is exactly
  this subsystem (CLAUDE.md "Phase 1.3 CONSOLIDATION CONFIRMED";
  "Phase 1.3 + Tier 2.1 COMBINED CONFIRMED 3/3 GO + ANTI-CHEAT
  VALIDATED": hippo-OFF retention 94% single-seed; 3/3 strict
  anti-cheat multi-seed with 10x stronger hippo silencing plus zeroing
  ~194k CA1->cortex edges, identical retention to non-strict). The
  offline reordering is exactly the SHUFFLED encode order the
  validated concept mechanism structurally requires -- and it
  legitimately lives in the OFFLINE phase, not stolen from the online
  episodic encode.

The phase-factored attempt got the PHASES right (online encode /
offline replay) but the PATHWAYS wrong: it still routed the
episodic-ORDER readout through the consolidated trace, so the offline
shuffle that the concept pathway legitimately needs destroyed the
order the episodic readout was (incorrectly) asked to recover from the
SAME consolidated trace. The biologically-correct architecture is two
distinct readout pathways: episodic order from the order-preserving
online CA3->CA1 pattern-completion path; concept/working-memory from
the order-invariant offline-consolidated neocortical schema. They
share the ONLINE engram write (the episode is written once, in theta
order) but DIVERGE immediately after: the episodic readout never
touches the consolidation system; the concept readout never touches
the order-sensitive recall.

### 1.3 The project's own validated assets this design reuses (by name; protected)

Byte-unchanged, imported not copied, never edited by this design:

- Validated trisynaptic / engram episodic store. CLAUDE.md
  "Hippocampal trisynaptic loop (P1)" (D.12 separation / D.13
  completion validated) and the Tonegawa-style "Engram-tagging API"
  (`start_engram_recording`, `commit_engram_tag`, `stimulate_tag`,
  `clear_tag_drive`, `delete_engram_tag`; persists through save/load).
  `build_biological_brain_regions(enable_hippocampus_consolidation=
  True)` builds the trisynaptic store (ec/dg/ca3/ca1) plus the CA3
  recurrent autoassociator (`ca3_swr_burst` gate) and the
  CA1->concept/motor consolidation pathways. This is the
  order-PRESERVING online store that already scores episodic = 1.0 in
  the loop and is the episodic-readout pathway in this design.
- Validated Phase-1.3 complementary-learning-systems consolidation.
  `set_awake_gates` (encoding ON, consolidation OFF), `set_sleep_gates`
  (input drive zeroed, direct lang->motor frozen, `ca3_swr_burst` /
  `ca1_to_motor` / `ca1_to_lang_out` ON), `run_concept_replay_phase`
  (`randomize_order=True`; SWR-style bursts driving CA3 attractors so
  STDP at ca3->ca1->cortex consolidates the trace),
  `run_swr_replay_phase`, `freeze_all_gates` (the validated pre-eval
  freeze), and the hippo-OFF / `--strict-silence` anti-cheat contrast.
  This is the order-INVARIANT offline pathway that trains the
  neocortical concept layer in this design.
- Validated 16-pool concept substrate. CLAUDE.md "v14 5-SEED MULTI-SEED
  GO ... orthogonal codes + 16 pools" / "v16 5-seed MULTI-SEED GO"
  (88.75% multi-seed bidirectional role-selective concept-pool
  binding; weak concept dynamics 0.05/0.3/0.8, per-pool FS
  cross-inhibition, orthogonal codes, Pulvermuller topographic prior,
  reciprocal pool->language_output bias). This is the concept
  (role->filler) selectivity layer; the concept/working-memory
  readout pathway reads it.
- The no-confabulation abstention moat:
  `research/runners/abstention_gate.py` (`gate`,
  `DEFAULT_THRESHOLD = 650.0`; AUC 0.990 know/don't-know separation,
  18 lines, byte-unchanged). Every drilled binding must clear it or
  the readout abstains.
- Per-stripe homeostatic equalization: the project's validated
  homeostasis, as already composed in the integrated loop. Byte-
  unchanged.
- The verified-correct non-zero-initialization foundation: the
  documented zero-initialization gotcha fix on the slot-to-concept
  efferent, preserved exactly as in the loop's verified foundation.
  Byte-unchanged; not re-litigated.

The frozen acceptance gate is the existing
`research/runners/integrated_loop_core.py` (`integrated_loop_verdict`,
`_IL_*` bars, the `_SHARED` / `_HELPER_WM` / `_HELPER_EP` /
`_HELPER_BOTH` partition). It is read EXACTLY as written in Section 4
and is NOT touched by this design.


## 2. Candidate distinct-pathway factorings and a recommendation

All candidates share ONE invariant: the ONLINE theta-ordered episodic
encode + the engram tag write are byte-unchanged from the verified
foundation (presentation-order == binding-index). After that single
shared online write the two readouts use PHYSICALLY DISTINCT pathways
that never share one order-constrained trace:

- Episodic-order readout pathway: the order-PRESERVING hippocampal
  trisynaptic CA3->CA1 pattern-completion path. The committed engram
  tag is the natural partial CA3 retrieval cue (D.13). The CA3
  recurrent autoassociator reconstructs the WHOLE bound pattern --
  including the theta-ordered serial structure written ONLINE -- and
  the per-role activity-peak order is read from CA1. NO offline replay
  is on this path. This is byte-identical to the verified foundation's
  hippocampus recall (the e02f692 idiom), explicitly NOT moved
  post-consolidation.
- Concept/working-memory readout pathway: the order-INVARIANT
  neocortical schema. A separate offline shuffled-replay consolidation
  (the validated Phase-1.3 subsystem, `run_concept_replay_phase`,
  `randomize_order=True`) trains ONLY the 16-pool concept layer from
  the committed episode's engram tag. The working-memory query reads
  the consolidated concept pools under `freeze_all_gates`. NO
  order-sensitive recall is on this path.

Net-new in every candidate is ONLY (a) a per-trial controller that
sequences the existing validated calls and (b) the wiring that routes
the shared online engram tag into the two distinct pathways. NO new
learning rule. NO automatic differentiation.

### Candidate 1 (minimal; RECOMMENDED): split readout immediately after the shared online engram write

Per composition trial:

1. ONLINE ENCODE (UNCHANGED from the verified foundation).
   `set_awake_gates`. The shared theta-gamma clock gates the
   basal-ganglia-selected prefrontal slot and times the engram write;
   role+filler orthogonal codes + teacher co-fire drive the bound
   pools; the episode is written to the engram store in theta order
   (presentation-order == binding-index). ONE shared online write;
   nothing here changes.
2. EPISODIC-ORDER READOUT (order-PRESERVING; trisynaptic completion;
   NOT consolidation). With weights frozen for the readout window,
   `stimulate_tag` the committed engram tag as the partial CA3
   retrieval cue; the CA3 recurrent autoassociator (`ca3_swr_burst`
   gate active for the recall, the validated D.13 idiom) reconstructs
   the bound pattern; the per-role activity-peak order is read from
   CA1 (argsort of per-item peak times against the true online encode
   order). This pathway never touches `run_concept_replay_phase`. This
   is exactly the verified foundation's recall -- it already scores
   episodic = 1.0 -- explicitly retained, NOT moved post-consolidation.
3. CONCEPT/WORKING-MEMORY CONSOLIDATION + READOUT (order-INVARIANT
   neocortical schema). `set_sleep_gates`; drive the committed episode
   tag through the validated `run_concept_replay_phase`
   (`randomize_order=True`) so the `ca3_swr_burst` autoassociator +
   CA1->concept consolidation transfers the bound (role, filler)
   structure into the 16-pool concept layer in SHUFFLED order across
   many replay events; `freeze_all_gates`; query each role and read
   the consolidated concept pool, emit only if the 650 moat passes
   else abstain. This pathway never touches the order-sensitive
   episodic recall.

Steps 2 and 3 read DIFFERENT physical structures (CA1 trisynaptic
completion vs the offline-consolidated 16-pool concept layer) from the
SAME single online engram write. The order-preserving constraint lives
entirely in step 2; the order-shuffling constraint lives entirely in
step 3; they cannot contend because no single trace carries both.

- Reuse: online encode + engram write + the trisynaptic D.13 recall +
  homeostasis + non-zero-init = byte-unchanged from the verified
  foundation; Phase-1.3 consolidation + the 16-pool concept substrate
  + the no-confab moat = byte-unchanged validated modules.
- Net-new: the per-trial controller (online encode; then the
  trisynaptic-completion episodic readout; then the
  offline-consolidation concept readout) + the wiring that fans the
  one engram tag out to the two distinct pathways. No new learning
  rule; no autograd.
- Honest ceiling for Candidate 1: episodic stays perfect because the
  episodic pathway is the byte-unchanged verified-foundation
  trisynaptic recall (no offline shuffle on it). The concept pathway
  is trained by the SAME validated mechanism (88.75% multi-seed) in
  the SAME phase it was validated in (offline interleaved replay), so
  its selectivity SHOULD transfer. The genuine open science question
  -- reserved for the controller-only decisive run -- is whether the
  consolidated concept readout is role-selective ENOUGH at the closed-
  loop minimal slice to clear the frozen `_IL_V1_MIN` (0.90) and
  `_IL_SCI_MIN` (0.80) on working-memory. THE DECISIVE RISK is NOT
  capability; it is Section 4: a faithful split-pathway architecture
  makes the remove-consolidation lesion NOT collapse episodic, which
  is the frozen `_HELPER_EP` duty for `no_cls_replay`. That is not a
  tuning risk; it is the structural reason the conclusion is (b).
- Risks: (i) consolidation transfer quality at the minimal slice
  (mitigation: the validated `run_concept_replay_phase` intensities
  and the validated v16 recipe are reused, not re-tuned); (ii) the
  controller must not let the consolidation phase perturb the
  episodic-pathway RNG draw order or the encode/write path (handled in
  Section 5: identical RNG draws across all modes, lesioned system
  removed only); (iii) trial wall-clock grows by the replay phase
  (bounded by the existing replay-budget, controller-run, kill-safe).
- Cheapest falsify-first de-risk (process lesson honored): a
  single-seed, minimal-load (N=2) GPU smoke of the FULL mode that
  probes the working-memory AND episodic readouts JOINTLY -- NOT the
  trivial-soundness mode alone (the recorded process lesson: the prior
  de-risk checked only the soundness mode and missed that
  consolidation destroys episodic order in the actual science mode).
  The smoke measures, on the full science mode at N=2: (a) episodic
  via the trisynaptic-completion pathway is still ~1.0 with the
  offline concept-consolidation phase running, AND (b) the
  consolidated concept readout is role-selective above chance. If
  either fails, the Section 6 escalation fires immediately with that
  exact structural cause; no configuration crank.

### Candidate 2 (Candidate 1 + consolidation-gated prefrontal reinstatement of the concept readout only)

Candidate 1, plus: between consolidation and the working-memory
readout, the consolidated concept representation re-instates the
prefrontal working-memory slots through the existing basal-ganglia-
gated thalamus->prefrontal afferent (a biologically standard
cortico-thalamo-prefrontal reinstatement), so the working-memory
readout is the consolidated concept content HELD in the prefrontal
slots rather than read straight off the concept pools. The episodic
pathway (step 2) is byte-identical to Candidate 1 -- the prefrontal
reinstatement is on the CONCEPT pathway ONLY and never touches the
episodic recall.

- Adds robustness if the bare consolidated-pool readout is selective
  but weak through the frozen 650 moat (a prefrontal hold could
  sharpen it). Reuses the validated basal-ganglia-working-memory
  gating wiring already present; net-new is only the ordering
  (consolidate, then reinstate, then read), not a new mechanism.
- Honest ceiling: same capability ceiling as Candidate 1; strictly
  more wiring surface on the concept pathway and one more place a
  faithfulness slip could hide -- the SECOND choice, taken only if
  Candidate 1's joint falsify-first smoke shows the bare consolidated
  readout is real but sub-moat.
- Risk: more net-new wiring -> more adversarial-review surface
  (Section 5). Do not build before Candidate 1's smoke result is in.

### Candidate 3 (fullest: multi-cycle offline schedule on the concept pathway)

Candidate 1 with the concept pathway's offline phase being multiple
SWR replay cycles across an alternating awake/sleep schedule over many
trials (the full CLS schedule the validated Phase-1.3 + Tier 2.1 3/3
anti-cheat result actually used), instead of one replay phase per
trial. The episodic pathway (step 2) is unchanged.

- Highest fidelity to the validated CLS regime; most likely to
  maximize consolidated concept selectivity. Largest wall-clock and
  largest schedule surface; justified only if Candidates 1 and 2 both
  show real-but-insufficient consolidated selectivity at the minimal
  slice.
- Honest ceiling: same capability claim type; more compute, not a
  stronger claim.

### Recommendation

Build Candidate 1. It is the minimal architecture that gives the two
readouts physically distinct pathways with opposite order properties
sharing only the single online engram write, it is the most
reuse-heavy (net-new is only the per-trial controller + the
fan-out wiring), and it is the cheapest to falsify-first via the
single-seed N=2 FULL-mode joint smoke above. Candidates 2 and 3 are
pre-described in-architecture escalations (not new architectures),
taken ONLY on an honest, propagated Candidate-1 smoke signal, never as
a reflexive configuration crank. CRITICAL: the recommendation to build
is CONDITIONAL on Section 4. Section 4 establishes -- before any run
-- that a biology-faithful Candidate 1 cannot satisfy the frozen
`_HELPER_EP` duty of `no_cls_replay`, because biologically episodic
order does NOT depend on the consolidation system. Therefore the
honest primary deliverable of this design is NOT "build Candidate 1
and run the frozen gate"; it is conclusion (b) and the new
separately-frozen necessity specification in Section 4. Candidate 1's
construction is specified fully (so the new necessity module can be
built and exercised against a faithful architecture), but the existing
frozen verdict is NOT the acceptance instrument for it -- Section 4
explains why, decisively.


## 3. Reuse map (DRY; protected)

Byte-UNCHANGED (imported, not copied; not edited by any candidate):

| Subsystem | Module / interface | Pathway it serves |
|---|---|---|
| Online theta-ordered episodic encode + engram write | the shared theta-gamma controller + engram-tagging API (`start_engram_recording` / `commit_engram_tag` / `clear_tag_drive` / `delete_engram_tag`) as wired in the verified integrated-loop foundation | SHARED online write (single) |
| Order-PRESERVING episodic recall | `stimulate_tag` on the committed tag as the CA3 partial cue + the `ca3_swr_burst` recurrent autoassociator + CA1 per-role peak-time read (the verified-foundation hippocampus recall idiom; D.12/D.13; `validate_trisynaptic_loop.py` validated) | EPISODIC readout (NOT consolidation) |
| Hippocampal trisynaptic store (ec/dg/ca3/ca1) | `build_biological_brain_regions(enable_hippocampus_consolidation=True)` | online write + episodic recall |
| Phase-1.3 CLS consolidation | `set_awake_gates`, `set_sleep_gates`, `run_concept_replay_phase` (`randomize_order=True`) / `run_swr_replay_phase`, `freeze_all_gates`, the hippo-OFF / `--strict-silence` anti-cheat | CONCEPT/WM readout ONLY |
| 16-pool concept (role->filler) selectivity | the validated v16 recipe + topographic-prior helper already used by the integrated loop (weak dynamics, FS cross-inhibition, orthogonal codes, reciprocal bias) | CONCEPT/WM readout |
| Per-stripe homeostasis | the project's validated homeostasis, as already composed in the loop | both pathways |
| Non-zero-init foundation | the documented slot-to-concept efferent non-zero prior, exactly as in the verified foundation | both pathways |
| No-confabulation moat | `research/runners/abstention_gate.py` (`gate`, `DEFAULT_THRESHOLD = 650.0`) | CONCEPT/WM readout |
| Native temporal-credit / phasic-dopamine / ACh window | the bridge eligibility path + the reused neuromodulator subsystem as in the verified foundation -- relegated to credit/gating, NOT binding | both pathways |

NET-NEW (the ONLY new code any candidate introduces):

1. A per-trial controller: run the ONLINE awake-encode (unchanged);
   then the EPISODIC-ORDER readout via the trisynaptic-completion path
   (the verified-foundation hippocampus recall, NOT moved
   post-consolidation); then the CONCEPT/WM offline-replay
   consolidation + consolidated-pool readout. Pure sequencing of
   EXISTING validated calls; no learning logic of its own.
2. The fan-out wiring connecting the single online engram tag to (i)
   the trisynaptic episodic-recall cue and (ii) the offline
   `run_concept_replay_phase` consolidation input feeding the 16-pool
   concept layer. Wiring only.

There is NO new learning mechanism, NO new plasticity rule, NO
automatic differentiation, NO change to any frozen bar, NO change to
the no-confabulation moat, NO edit to any protected/validated module.
The frozen verdict and the moat are reused-or-superseded-by-a-new-
frozen-module (Section 4 conclusion (b)), NEVER edited.


## 4. HEAD-ON frozen-partition analysis: does distinct-pathways realize the EXISTING frozen partition, or not?

This is the load-bearing section. The frozen partition
(`integrated_loop_core.py`, read exactly as written, NOT edited) is:

- `_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")`
  -- each must collapse BOTH readouts (wm <= 0.40 AND ep <= 0.40).
- `_HELPER_WM = ("no_bg_gate",)` -- must collapse the working-memory
  readout (wm <= 0.40).
- `_HELPER_EP = ("no_sequencing", "no_cls_replay")` -- each must
  collapse the episodic readout (ep <= 0.40).
- `_HELPER_BOTH = ("no_neuromod_timing",)` -- must collapse BOTH.

Lesion-by-lesion, against a biology-faithful Candidate 1 (episodic =
order-preserving trisynaptic completion from the online tag; concept/WM
= order-invariant offline-consolidated neocortical schema; ONE shared
online engram write):

1. `no_binding` (SHARED; must collapse BOTH). Removing the
   combinatorial binding step removes the excitability bias that binds
   role+filler at encode. No bound assembly is written to the engram
   tag. The episodic trisynaptic cue has no bound pattern to complete
   (ep collapses) AND the offline consolidation has no bound (role,
   filler) structure to transfer to the concept layer (wm collapses).
   BOTH collapse. The shared online write is upstream of BOTH distinct
   pathways. SATISFIED.

2. `no_shared_clock` (SHARED; must collapse BOTH). Removing the one
   shared theta-gamma rhythm desynchronizes the slot gating and the
   engram-write timing. The online write is not theta-ordered and the
   role-filler co-fire is not coincident. The episodic pathway has no
   ordered bound pattern to complete (ep collapses) AND the concept
   pathway has no coherent bound structure to consolidate (wm
   collapses). BOTH collapse, because both pathways descend from the
   single clock-timed online write. SATISFIED.

3. `no_hippo_store` (SHARED; must collapse BOTH). Removing the fast
   hippocampal store removes the engram tag entirely. The episodic
   trisynaptic-completion pathway has no tag to cue (ep = 0.0 by
   construction) AND the offline consolidation has no tag to replay,
   so nothing reaches the concept layer (wm collapses). BOTH collapse.
   SATISFIED.

4. `no_bg_gate` (HELPER_WM; must collapse working-memory). Removing
   the basal-ganglia selective gate drives ALL channels: no single
   prefrontal slot is selected at encode. The role->filler binding has
   no clean slot, so the consolidated concept layer cannot resolve a
   role-selective filler (wm collapses). Episodic order, carried by
   the trisynaptic completion of whatever bound pattern WAS written,
   is not necessarily destroyed by the absence of slot SELECTION (the
   serial theta order can still be present even if the slot is
   diffuse), so ep need not collapse. The required direction (wm
   collapses) holds. SATISFIED.

5. `no_sequencing` (HELPER_EP; must collapse episodic). Removing
   sequencing makes the shared clock REPEAT the role-filler assembly
   every theta cycle instead of SHIFTING it across cycles. No serial
   order is written at the online encode. The trisynaptic completion
   then reconstructs an order-degenerate pattern -- the per-role
   activity-peak times carry no recoverable sequence (ep collapses to
   chance). The concept/WM pathway is order-invariant by construction,
   so wm need not collapse. The required direction (ep collapses)
   holds. SATISFIED.

6. `no_cls_replay` (HELPER_EP; must collapse episodic). THE DECISIVE
   LESION. Removing the offline shuffled-replay consolidation removes
   the CONCEPT pathway's training entirely (the concept readout
   collapses -- the concept layer is never consolidated). But under a
   biology-faithful distinct-pathways architecture the EPISODIC
   readout is served by the order-preserving online trisynaptic
   pattern-completion pathway, which DOES NOT DEPEND ON THE
   CONSOLIDATION SYSTEM AT ALL. With consolidation removed, the online
   engram tag is still written in theta order and the CA3
   autoassociator still completes it; the per-role peak order is still
   recovered. Therefore `no_cls_replay` leaves the episodic readout
   INTACT (ep stays high). The frozen partition places `no_cls_replay`
   in `_HELPER_EP` and REQUIRES it to collapse the EPISODIC readout
   (ep <= 0.40). A faithful distinct-pathways architecture produces
   the OPPOSITE: `no_cls_replay` collapses the CONCEPT/WM readout and
   PRESERVES episodic. The frozen verdict, recomputed unchanged from
   such numbers, returns VOID at the
   `non-discriminating: helper lesion 'no_cls_replay' did NOT collapse
   the episodic-recall readout` clause. NOT SATISFIED -- and it CANNOT
   be satisfied by any biology-faithful distinct-pathways wiring,
   because biologically episodic order is a property of the online
   trisynaptic store, not of the consolidation/replay system. This is
   exactly the inversion that VOIDed the phase-factored attempt --
   reached here from the OPPOSITE direction (the phase-factored attempt
   failed because consolidation DESTROYED episodic; the distinct-
   pathways architecture fails the SAME frozen duty because
   consolidation is IRRELEVANT to episodic). Two opposite faithful
   wirings, same frozen-partition refutation: this is the THIRD
   convergent signal.

7. `no_neuromod_timing` (HELPER_BOTH; must collapse BOTH). Removing
   the timed-plasticity neuromodulator window removes the ACh-gated
   "when the weight update is allowed" timing from the WHOLE loop
   consistently. With plasticity untimed, the online binding does not
   form correctly (ep collapses -- the trisynaptic pattern is not
   correctly written) AND the offline consolidation cannot
   selectively strengthen the bound structure (wm collapses). BOTH
   collapse. SATISFIED.

8. (Instrument-soundness, not a lesion, stated for completeness.) The
   frozen `_IL_V1_MIN = 0.90` requires the FULL loop to learn the
   no-gap trivial bind on BOTH readouts. Candidate 1's episodic
   pathway already does this (the verified foundation scores episodic
   = 1.0); the working-memory pathway's trivial-bind soundness is the
   genuine open question -- but it is moot, because lesion 6 already
   forces VOID before the science verdict is reached.

### Conclusion: (b). Distinct-pathways CANNOT realize the existing frozen partition; it is the THIRD convergent signal that the pre-registered necessity partition itself is the falsified element.

Seven of the eight lesions are satisfied by a faithful Candidate 1.
The eighth, `no_cls_replay`, CANNOT be satisfied -- not by tuning, not
by any in-architecture escalation (Candidates 2/3 do not change which
pathway carries episodic order), but for a STRUCTURAL,
biology-grounded reason: the frozen partition asserts that the
consolidation/replay system is NECESSARY for episodic-sequence recall
(`no_cls_replay` in `_HELPER_EP`). The catalog's documented
hippocampal-neocortical division of labor says the opposite:
episodic-sequence recall is order-preserving via the hippocampal
trisynaptic pattern-completion pathway (D.03/D.12/D.13; the project's
own VALIDATED trisynaptic loop), built ONLINE; the consolidation/
replay system builds the order-invariant NEOCORTICAL CONCEPT/SCHEMA
representation (CLS; the project's own VALIDATED Phase-1.3), and is
necessary for the CONCEPT/working-memory readout, NOT for episodic
order. A pre-registered necessity prediction that NO biology-faithful
architecture realizes is a FALSIFIED prediction. This is now the third
convergent signal:

- First (single online pass): the two validated subsystems' encode-
  order requirements are directly contradictory.
- Second (phase-factored, both readouts off the consolidated trace):
  the contradiction relocates; consolidation DESTROYS episodic order;
  `no_cls_replay` inverts; VOID-by-construction.
- Third (distinct pathways, the biologically-correct fix): episodic
  order is carried by the online trisynaptic store, so consolidation
  is IRRELEVANT to episodic; `no_cls_replay` still fails its frozen
  `_HELPER_EP` duty -- from the opposite direction. The partition's
  membership of `no_cls_replay` is the falsified element.

The honest path is therefore the PRE-COMMITTED TERMINAL step: a
catalog-grounded RE-DERIVATION of the necessity hypothesis, built as a
NEW, separately-frozen, separately-pre-registered necessity module.
The existing `integrated_loop_core.py` is NOT edited; its VOID stands
as the honest scientific record that the original necessity prediction
was wrong. Section 5 specifies the new module's pre-registered fixed
bars and partition with a-priori justification, mirroring the existing
module's discipline EXACTLY.


## 5. The NEW separately-frozen necessity module (conclusion (b) deliverable)

Because the conclusion is (b), the existing frozen acceptance gate is
NOT the instrument for the distinct-pathways architecture. A NEW,
separately-frozen, separately-pre-registered necessity module is
required. This section specifies it as a DESIGN (no code here); the
plan/build pass implements it as a new pure standard-library module
that owns its OWN frozen bars, imports no other verdict module, and
mirrors `integrated_loop_core.py`'s discipline EXACTLY:
instrument-validity FIRST, fail-closed, fixed bars pre-registered and
NEVER tuned to a result, "cannot conclude" (VOID) strictly distinct
from "fails" (FAIL), malformed/non-numeric/unorderable input -> VOID
(never an exception), no autograd, ASCII only.

The existing module's VOID is preserved verbatim as the honest record
that the original necessity prediction (consolidation necessary for
episodic) was falsified. The new module does NOT supersede the
existing one's RECORD; it supersedes only the necessity HYPOTHESIS,
re-derived from the catalog.

### 5.1 The catalog-grounded corrected necessity claims (the precise basis)

The correction is a single, decisive re-assignment of one lesion's
readout responsibility, grounded entry-by-entry in the project's
catalog and its own validated assets:

- Episodic-sequence recall is NECESSARILY dependent on the ONLINE
  hippocampal trisynaptic pattern-completion path: entorhinal ->
  dentate (pattern separation, catalog D.12, Kandel 6e Ch 54
  pp 1357-1360) -> CA3 recurrent autoassociator (pattern completion,
  catalog D.13, Kandel 6e Ch 54 pp 1342, 1360-1361; Marr 1971) ->
  CA1 (catalog D.03 trisynaptic pathway). Validated in the project by
  `validate_trisynaptic_loop.py` (D.12 3/3 multi-seed; D.13
  single-seed PASS). Episodic order is written ONLINE in theta order
  and recovered by CA3 completion WITHOUT offline replay.
- Concept/schema (the working-memory role->filler readout) is
  NECESSARILY dependent on the OFFLINE complementary-learning-systems
  consolidation path (sharp-wave-ripple-gated interleaved replay
  building the order-invariant NEOCORTICAL representation; McClelland
  1995; Buzsaki 2013; the project's VALIDATED Phase-1.3 consolidation,
  3/3 strict anti-cheat multi-seed).
- THE CORRECTED CLAIM: the consolidation/replay system
  (`no_cls_replay`) is necessary for the CONCEPT/WORKING-MEMORY
  readout, NOT for the episodic readout. In the existing frozen
  partition `no_cls_replay` is in `_HELPER_EP`; the catalog-grounded
  corrected partition places it in the working-memory helper set. This
  is the ONLY assignment that changes; every other lesion's
  responsibility is UNCHANGED and is reproduced below with its
  original a-priori justification intact (the corrected module is a
  minimal, catalog-justified correction of exactly the one falsified
  element, not a free re-design).

### 5.2 The new module's pre-registered FIXED partition (a-priori; never tuned)

The corrected lesion partition (each membership justified WITHOUT
reference to any observed run, by the catalog basis in 5.1):

- SHARED (must collapse BOTH readouts -- the non-separability
  signature; UNCHANGED from the existing module, same a-priori
  justification): `no_binding`, `no_shared_clock`, `no_hippo_store`.
  Each is upstream of the single shared online engram write on which
  BOTH distinct pathways depend; removing any collapses both. (Same
  reasoning as the existing module's `_SHARED`.)
- HELPER, working-memory (must collapse the working-memory readout):
  `no_bg_gate` (UNCHANGED: selective slot gating is necessary for the
  role-selective filler) AND `no_cls_replay` (CORRECTED, the single
  re-assignment: offline CLS consolidation builds the order-invariant
  neocortical concept/schema representation the working-memory readout
  reads -- catalog D.* CLS basis in 5.1; the project's validated
  Phase-1.3 is exactly this).
- HELPER, episodic (must collapse the episodic readout):
  `no_sequencing` (UNCHANGED: the SHIFT-across-theta is what writes
  the serial order the trisynaptic completion recovers; removing it
  makes the recovered order degenerate). `no_cls_replay` is REMOVED
  from this set (the falsified membership).
- HELPER, both (must collapse BOTH; UNCHANGED, same a-priori
  justification): `no_neuromod_timing` (timed plasticity gates the
  whole loop consistently; removing it corrupts both the online
  trisynaptic write and the offline consolidation).

### 5.3 The new module's pre-registered FIXED bars (a-priori; mirroring the existing module EXACTLY)

The new module reuses the existing module's FIXED bar VALUES verbatim,
with the existing module's verbatim a-priori justifications (the bar
VALUES were never the falsified element -- only the
`no_cls_replay` partition membership was). Restated here so the new
module is self-contained and owns its own frozen constants:

- LADDER = (2, 4, 8). Compositional load = number of role-filler
  bindings held and composed simultaneously; two is the smallest
  non-trivial composition; geometric doubling to a scale-confidence
  load. (Verbatim from the existing module's a-priori justification.)
- V1_MIN = 0.90. The full loop, on a no-gap trivial single bind, must
  nearly perfectly learn the bijection on BOTH readouts, or the
  instrument cannot measure composition (soundness, not science).
- SCI_MIN = 0.80. The full loop must clear a clear-majority bar on the
  genuine compositional task on BOTH readouts.
- LESION_MAX = 0.40. A lesioned readout has "collapsed" iff at/near
  chance; same at/near-chance ceiling the existing module uses.
- SCALE_TOL = 0.10. Max permitted DROP between ascending rungs (the
  stochastic multi-seed noise floor).
- MIN_SEEDS = 3. Below three seeds a multi-seed claim is not
  supportable.

The new module's verdict precedence is IDENTICAL to the existing
module's (instrument-validity FIRST -> soundness/discrimination defect
=> VOID; else science precedence PASS / WORKS-SMALL-NO-SCALE-
CONFIDENCE / FAIL), with the ONLY difference being the corrected
partition of 5.2. VOID remains strictly distinct from FAIL. The new
module is itself FROZEN on creation and NEVER tuned to a result; if a
faithful distinct-pathways architecture ALSO cannot satisfy this
corrected, catalog-grounded partition, that is itself an honest
program-level result surfaced without spin (the pre-committed bound,
Section 6) -- not a license to edit the new module.

### 5.4 The new module's pre-registered falsify-first (process lesson honored)

The new module's acceptance is gated by the SAME joint falsify-first
discipline: the cheap de-risk MUST probe the FULL science mode's
working-memory AND episodic readouts JOINTLY at N=2 single-seed (NOT
the trivial-soundness mode alone -- the recorded process lesson from
the phase-factored de-risk that checked only the soundness mode). The
pre-registered early trigger: if the full-mode joint smoke shows
either (a) episodic via the trisynaptic-completion pathway is not
~1.0 with the offline concept-consolidation running, or (b) the
consolidated concept readout is not role-selective above chance, the
Section 6 escalation fires immediately with that exact structural
cause; no configuration crank.


## 6. Pre-committed bound (restated; stated now, before any run)

If a faithful distinct-pathways build (Candidate 1, with Candidates
2/3 as the pre-described in-architecture escalations) evaluated against
the NEW separately-frozen catalog-grounded necessity module ALSO
reaches "cannot conclude" or fails the corrected partition, that is a
deeper program-level result and is surfaced honestly with its precise,
GPU-measured structural cause -- not a configuration iteration, not
spin, not a hand-back, not a declare-globally-unfit. The next step is
then the next catalog-identified integration factorization, pursued
autonomously with the SAME adversarial and anti-cheat discipline and
the SAME (new-module) frozen acceptance. The joint falsify-first smoke
in Section 5.4 is the explicit early trigger. This bound is stated in
advance so the next outcome cannot be rationalized after the fact.

The existing `integrated_loop_core.py` is NOT the acceptance
instrument for the distinct-pathways architecture and is NOT edited;
its VOID is the honest record that the original necessity prediction
was falsified. The no-confabulation moat
(`abstention_gate.DEFAULT_THRESHOLD = 650.0`) is byte-unchanged; every
drilled working-memory binding must clear it or the readout abstains.


## 7. Honest ceiling (stated, never overstated)

A distinct-pathways PASS against the NEW catalog-grounded frozen
necessity module would mean exactly this: a biology-grounded loop with
two physically distinct readout pathways (an order-preserving online
hippocampal trisynaptic pattern-completion path for episodic order;
an order-invariant offline-consolidated neocortical schema path for
concept/working-memory) shows emergent compositional memory that holds
and scales across the frozen (2, 4, 8) load ladder, where every
single-system lesion abolishes the capability it is responsible for
under the CORRECTED partition, and the three shared systems collapse
both readouts together. It is explicitly NOT fluent or open-ended
language, NOT a large language model, NOT conversation solved. No
claim here asserts any decisive test has passed; this document only
designs the next properly-scoped program step AND states, decisively
and before any run, that the honest deliverable is conclusion (b): the
existing pre-registered necessity partition is the falsified element,
and a new separately-frozen, catalog-grounded necessity module is
required, with the existing module's VOID preserved as the honest
scientific record. Nothing here claims compositional memory in the
full model; it claims a precise, thrice-convergent, catalog-grounded
characterization of why the validated-subsystem integration is hard at
this slice and the disciplined, pre-committed path forward.
