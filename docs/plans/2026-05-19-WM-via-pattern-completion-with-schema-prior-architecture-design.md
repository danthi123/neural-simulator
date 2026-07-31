---
type: plan
status: live
date: 2026-05-19
---

# Design: working-memory readout of the current binding via the
# specificity-preserving hippocampal pattern-completion pathway, with
# the consolidated neocortical schema as a generalizable prior only

Status: DESIGN ONLY. No code, no implementation, no GPU run is part of
this document. It specifies one next architectural factorization and
analyzes it head-on against the unchanged corrected frozen necessity
instrument. It changes no code, no frozen module, no moat, no other
plan or finding.

Scope honesty (stated up front, never to be softened anywhere below):
this design does NOT claim conversational capability, compositional
memory, fluent language, or a large language model. The single
load-bearing, scale-confident scientific result of this line remains
the thrice-convergent falsification and catalog-grounded correction of
the original pre-registered necessity prediction. A pass of any
candidate built from this design against the corrected frozen module is
"consistent-with the corrected biology" ONLY -- never a validated
success, never spun.


## 1. Problem and biology grounding

### 1.1 The finding that necessitates this design (restated crisply; not re-derived)

The distinct-readout-pathways candidate (runner mode `--distinct-
pathways` in `research/runners/integrated_loop_gate.py` at commit
`b4a8106`; design `docs/plans/2026-05-19-distinct-readout-pathways-
architecture-design.md`) was built and exercised faithfully on GPU
(CuPy, RTX 3090, full science mode, minimal load, joint working-memory
and episodic readouts). The recorded, honestly-propagated result
(`research/findings/2026-05-19-distinct-pathways-honest-negative-
episodic-contradiction-DISSOLVED-WM-blocked-by-consolidation-
genericization.md`):

- Episodic-order readout `ep` = 1.0. The encode-order contradiction
  that dominated iterations 1-4 and the phase-factored attempt is
  STRUCTURALLY DISSOLVED. The order-preserving online hippocampal
  trisynaptic CA3->CA1 pattern-completion path reconstructs the bound,
  theta-ordered pattern, and inserting the separate offline
  consolidation does NOT degrade it (the two pathways are genuinely
  physically distinct; the online episodic readout is taken before and
  independently of consolidation). Because the pattern-completion path
  reconstructs the *ordered bound* pattern, it provably RETAINS the
  role-to-filler binding specificity -- that retention is exactly why
  `ep` = 1.0.

- Working-memory readout `wm` = 0.0. The working-memory query was read
  from the order-INVARIANT consolidated neocortical schema. Systems
  consolidation, by design (interleaved shuffled replay), abstracts
  away episode-specific role-to-filler binding; it transfers a generic
  "most-consolidated filler" attractor, so queries collapse onto one
  dominant filler regardless of the queried role -> `wm` = 0.0. The
  consolidated schema still clears the trustworthy-grounding moat and
  episodic order is perfect throughout; the failure is specifically
  that consolidation discards the binding specificity role-selective
  working memory requires.

This is the classic complementary-learning-systems (CLS) specificity-
versus-generalization trade-off, observed cleanly in isolation. It is
forward progress, not the same wall: one structural blocker
(encode-order contradiction) is genuinely solved, and the remaining
blocker is precisely localized to a recognized biological trade-off
rather than mysterious.

### 1.2 The biology, grounded in the project's OWN validated assets and the reference catalog

The correct CLS division of labor (cited by name; these are the
project's validated subsystems, not hypotheticals):

- Recent, episode-SPECIFIC bindings are served by the hippocampal
  trisynaptic loop via pattern separation and pattern completion. The
  project's validated asset is the trisynaptic loop work
  (`research/runners/validate_trisynaptic_loop.py`; CLAUDE.md
  "Hippocampal trisynaptic loop (P1)"): catalog D.12 pattern
  separation (Kandel 6e Ch 54 pp 1357-1360) -- DG orthogonalizes;
  catalog D.13 pattern completion (Kandel 6e Ch 54 pp 1342,
  1360-1361; Marr 1971 autoassociator) -- the CA3 recurrent
  autoassociator (`ca3_swr_burst`) reconstructs the full bound pattern
  from a partial cue. Catalog D.03 is the EC->DG->CA3->CA1 trisynaptic
  pathway itself. The runner already drives this online via the
  committed engram tag as the CA3 retrieval cue (the byte-unchanged
  `commit_engram_tag` / `stimulate_tag` engram API).

- Remote, GENERIC structure is supplied by the neocortical schema,
  built OFFLINE by CLS consolidation (McClelland 1995; Buzsaki 2013;
  the project's validated Phase-1.3 consolidation,
  `research/runners/consolidation_trainer.run_concept_replay_phase`,
  3/3 strict-anti-cheat multi-seed; CLAUDE.md "Phase 1.3 hippocampus
  consolidation"). Interleaved shuffled replay deliberately
  generalizes -- it abstracts away the episode-specific binding by
  construction.

The distinct-pathways finding GPU-demonstrated this exact division:
the online trisynaptic pattern-completion path PRESERVES binding
specificity (hence `ep` = 1.0); the offline-consolidated schema
GENERALIZES it away (hence `wm` = 0.0 when WM is read from the schema).
The biology says role-selective working memory for a SPECIFIC, recent,
just-bound item is a hippocampal-pattern-completion readout, with
neocortex contributing only a generalizable prior. The prior attempts
read WM from the wrong store:

- iters 1-4: WM read from the concept pools directly (single online
  pass; the concept mechanism needs shuffled order; episodic store
  needs fixed order -- direct contradiction).
- phase-factored: WM read from the consolidated trace, which also
  carried the episodic constraint -> the two constraints contended in
  one trace -> VOID-by-construction.
- distinct-pathways: WM read from the order-invariant consolidated
  schema -> generic attractor -> `wm` = 0.0.

NO prior attempt routed the role-selective WM readout through the
specificity-preserving pattern-completion path. That is the genuinely
new factorization this design specifies.


## 2. The factorization to design

Core idea (one sentence): route the role-selective working-memory
readout of the CURRENT (just-bound) item through the SAME
specificity-preserving online hippocampal trisynaptic pattern-
completion pathway that already delivers `ep` = 1.0, with the
Phase-1.3-consolidated order-invariant neocortical schema serving ONLY
as a generalizable soft PRIOR/bias on that readout (it shapes /
regularizes the competition; it is never the binding source).

Why this dissolves the consolidation-genericization blocker while
keeping `ep` = 1.0: the binding-specific content of the WM answer now
comes from the store that the GPU evidence proved RETAINS role->filler
specificity (the online CA3->CA1 autoassociator). The consolidated
schema is demoted from "the WM source" to "a weak prior", so its
deliberate genericization no longer determines the answer -- it can at
most bias a competition whose signal is already specific. The episodic
readout is completely untouched (taken at the identical position,
before and physically independent of consolidation), so `ep` = 1.0 is
structurally preserved by construction.

Shared definitions (defined once, used throughout):
- "WM-pattern-completion cue": the committed online engram tag for the
  current episode (the same tag the episodic readout uses as its CA3
  retrieval cue). It is the partial pattern; the CA3 recurrent
  autoassociator (`ca3_swr_burst`) completes it.
- "Role-conditioned completion": pattern completion is run with the
  queried role's orthogonal code co-presented on `language_input`
  (exactly the runner's existing role-cue drive `_code(q_ridx,...)`),
  so the autoassociator is biased toward the CA3 sub-pattern bound to
  THAT role, and the bound filler concept pool is read out (the
  existing filler-pool population vote + the byte-unchanged
  abstention moat at `DEFAULT_THRESHOLD` = 650).
- "Schema prior": a small, binding-agnostic additive bias derived from
  the consolidated neocortical concept layer, applied to the filler-
  pool competition during the WM readout. It carries generic
  "this filler is a plausible filler at all" mass, NOT
  "this role binds this filler".

### 2.1 Candidate A -- pure pattern-completion WM readout; schema as a weak ADDITIVE prior (RECOMMENDED)

- WM readout: at query, present the queried role code on
  `language_input`, drive the committed engram tag as the CA3 cue
  (`stimulate_tag` -- the byte-unchanged engram API), run the CA3
  recurrent autoassociator (`ca3_swr_burst`) to completion exactly as
  the episodic-order readout already does, and read the bound filler by
  the existing filler-pool population vote through the same gate. This
  is the byte-unchanged online-recall idiom, just role-conditioned and
  scored on the FILLER pools (the episodic readout scores ROLE peak
  order; the WM readout scores the FILLER identity -- different
  projections of the SAME completed pattern, no new mechanism).
- Schema prior: the offline `run_concept_replay_phase` consolidation
  still runs (so the `no_cls_replay` lesion still has something to
  remove -- see Section 4), producing the consolidated concept layer.
  Its contribution to WM is a SMALL additive bias on the filler-pool
  score vector: a fixed-magnitude term proportional to each filler
  pool's consolidated baseline drive, scaled so it is strictly weaker
  than the pattern-completion-driven signal (a regularizer that can
  break a near-tie but cannot overturn a specific completion).
- Honest ceiling/risks:
  - Risk 1 (the central one): if the schema prior is weak enough to
    not overturn pattern completion, then removing it (the
    `no_cls_replay` lesion) may NOT collapse `wm` -- pattern
    completion alone may still answer correctly. This is the head-on
    v2-partition question; analyzed precisely in Section 4. Under this
    candidate it is the single biggest risk and is a potential
    honest-negative-by-construction, NOT a reason to retune.
  - Risk 2: role-conditioned completion may not be sufficiently
    role-selective at minimal load if the engram tag over-completes
    (catalog D.13 "too much completion -> confused episodes"). The
    validated trisynaptic asset already characterizes this regime; the
    risk is real but is the documented completion-vs-separation
    tension, not a new failure mode.
  - Ceiling: at most "consistent-with the corrected biology"; never a
    scale-confident validated result.

### 2.2 Candidate B -- schema-GATED / biased pattern completion

- WM readout: as Candidate A, but the schema does not add to the final
  score; instead the consolidated concept layer provides a soft
  multiplicative GATE on the CA3 completion input (it biases WHICH CA3
  sub-pattern the autoassociator settles into by gently pre-activating
  the consolidated-plausible filler region before completion). The
  binding selectivity is still produced by the role-conditioned
  pattern completion; the schema only shapes the basin.
- Honest ceiling/risks:
  - Risk: a multiplicative schema gate is "more entangled" with the
    completion dynamics, so removing it (the `no_cls_replay` lesion)
    is MORE likely to perturb `wm` than the additive prior of
    Candidate A -- which is favorable for satisfying the v2 partition,
    but at the cost of the schema being less cleanly "a prior only"
    and more "part of the binding readout". This risks blurring the
    very CLS factorization the design is built on, and could re-import
    a softer form of the genericization failure if the gate is too
    strong.
  - Ceiling: same as Candidate A; entanglement makes the honest
    interpretation harder, not easier.

### 2.3 Candidate C (fullest) -- dual-readout with explicit recent/remote arbitration

- WM readout: run BOTH a pattern-completion readout (recent-specific,
  Candidate A's mechanism) and a consolidated-schema readout
  (remote-generic, the current distinct-pathways mechanism), and
  combine them by a fixed, pre-registered, NON-learned arbitration:
  the pattern-completion readout dominates when its top filler-pool
  score clears the moat (recent, specific, confidently bound), and the
  schema readout only contributes when the pattern-completion signal is
  below the moat (a generalization fallback for items no longer
  episodically retrievable). No new learning rule; the arbitration is a
  pure deterministic function of the two already-computed score
  vectors and the existing `DEFAULT_THRESHOLD`.
- Honest ceiling/risks:
  - Risk: the most moving parts; the arbitration constant is a new
    pre-registered structural choice (must be justified a-priori,
    never tuned to a result, exactly like the runner's existing
    a-priori sizing discipline). Greatest risk of accidentally
    constructing a configuration that passes the v2 partition for the
    wrong reason (arbitration-tuned, not biology-driven).
  - Benefit: it is the closest to the literal CLS statement
    (recent-specific via hippocampus; remote-generic via neocortex,
    with a principled hand-off), so a clean PASS here would be the
    most biologically interpretable -- but the honesty ceiling still
    binds it to "consistent-with".

### 2.4 Recommendation and cheapest falsify-first de-risk

Recommend Candidate A. It is the minimal faithful realization of the
exact biology the GPU evidence motivates (recent-specific WM via the
proven specificity-preserving pattern-completion path; neocortex a
weak prior), it adds the least net-new wiring, it does not entangle
the schema with the binding readout (preserving the clean CLS
factorization), and it makes the head-on v2-partition question
maximally crisp rather than hiding it behind an arbitration constant.

Cheapest falsify-first de-risk (MUST follow the recorded process
lesson -- probe the FULL science mode, not the soundness mode alone;
the phase-factored false-green was caused by checking only the
soundness mode): run the FULL science-mode candidate at the SINGLE
minimal ladder load (the smallest rung only) and read `wm` AND `ep`
JOINTLY in one minimal run. Required joint outcome to proceed: `ep`
still ~1.0 (the dissolution is preserved -- non-negotiable) AND `wm`
materially above its distinct-pathways 0.0 floor (the pattern-
completion WM readout produces role-selective signal at all). If `ep`
regresses below ~1.0, the candidate broke the one thing that was
working and is rejected immediately. If `wm` does not lift off 0.0,
that is an honest negative at minimal load -- propagated, no
configuration-cranking, proceed to the next catalog factorization.
This is one minimal full-mode run, the cheapest possible test that
the new WM-readout routing does the intended thing without disturbing
the dissolved episodic readout.


## 3. Reuse map and net-new surface

Byte-UNCHANGED (imported and used exactly as-is; NOT edited by this
design or any candidate built from it):
- Hippocampal trisynaptic loop + pattern completion + the engram API:
  `build_biological_brain_regions` (ec/dg/dg_pv_basket/ca3/ca1, the
  `ca3_swr_burst` CA3 recurrent autoassociator), `start_engram_
  recording` / `commit_engram_tag` / `stimulate_tag` / `clear_tag_
  drive`. Validated asset: `research/runners/validate_trisynaptic_
  loop.py` (catalog D.03/D.12/D.13).
- Phase-1.3 consolidation: `research/runners/consolidation_trainer.
  run_concept_replay_phase`, `set_sleep_gates`, `set_awake_gates`,
  `freeze_all_gates` (the validated freeze-then-evaluate idiom).
- 16-pool concept substrate + the v16 selectivity recipe (weak
  dynamics 0.05/0.3/0.8, FS-per-pool cross-inhibition, orthogonal
  codes, the Pulvermuller topographic prior) +
  homeostasis (the validated `fused_homeostasis_update` per-neuron
  equalization) + non-zero readout-pathway init (the Barlow-1972
  baseline-weight gotcha): `research/runners/text_minimal_
  isolation.py`, used exactly as the runner already uses it.
- No-confabulation moat: `research/runners/abstention_gate.py`
  (`gate`, `DEFAULT_THRESHOLD` = 650). Byte-unchanged; every drilled
  WM binding must clear it or the readout abstains.
- Corrected frozen necessity instrument:
  `research/runners/integrated_loop_core_v2.py` (`integrated_loop_
  verdict_v2`, commit `36a7975`). Byte-unchanged; the sole acceptance
  scorer. The ORIGINAL `integrated_loop_core.py` (`2048750`) is never
  imported and never edited; its "cannot conclude" stands as the
  honest record.
- The already-working distinct online/offline pathways in
  `integrated_loop_gate.py` at `b4a8106`: the per-trial controller,
  the engram-tag fan-out, the genuinely-distinct online trisynaptic
  pattern-completion episodic-order pathway (GPU-proven `ep` = 1.0),
  and the separate offline Phase-1.3 consolidation. All reused as-is.

Net-new surface (the ONLY new code a future implementation would add;
specified here for the plan, not implemented in this document):
1. The WM-readout ROUTING change: take the `wm` score from a
   role-conditioned run of the EXISTING online pattern-completion
   readout (the same `stimulate_tag` + `ca3_swr_burst` completion the
   `_episodic_order_readout` closure already performs), scored on the
   FILLER pools through the byte-unchanged moat -- instead of from the
   post-consolidation order-invariant schema. This is a re-routing of
   already-validated calls plus reading a different projection of the
   already-completed pattern.
2. The schema-as-prior bias wiring: a small, binding-agnostic
   additive bias on the filler-pool WM score derived from the
   consolidated concept layer (Candidate A). Pure arithmetic on
   already-computed quantities.

Confirmed: NO new learning rule. NO automatic differentiation. The
only learning anywhere remains the reused native 3-factor
eligibility/temporal-credit STDP rule. No frozen module, no moat, no
validated subsystem is modified. The net-new surface is strictly the
WM-readout routing + the schema-prior bias arithmetic.


## 4. Head-on confrontation with the v2 frozen partition

The corrected frozen module `integrated_loop_core_v2.py` partitions
the 8 lesions (verbatim from the module, never edited):
- SHARED (must collapse BOTH `wm` and `ep`): `no_binding`,
  `no_shared_clock`, `no_hippo_store`.
- HELPER_WM (must collapse `wm`): `no_bg_gate`, `no_cls_replay`.
- HELPER_EP (must collapse `ep`): `no_sequencing`.
- HELPER_BOTH (must collapse both): `no_neuromod_timing`.

(NOTE: `integrated_loop_gate.py` at `b4a8106` still has the runner-side
`_HELPER_*` tuples with `no_cls_replay` in the EP set; the SOLE
acceptance scorer is `integrated_loop_verdict_v2`, whose
`_ILV2_HELPER_WM = ("no_bg_gate", "no_cls_replay")` is the corrected
partition. The analysis below is against the v2 module's partition,
which is what decides PASS/FAIL/VOID.)

Per-lesion mechanism under Candidate A (does each lesion still collapse
exactly the readout the corrected v2 partition requires?):

- `no_binding` (SHARED -> both): without the excitability bias the
  bound assembly is never written, so the online engram has no bound
  pattern. Pattern completion has nothing to reconstruct -> WM
  readout collapses; the episodic readout (same completion path) also
  has no bound ordered pattern -> `ep` collapses. Collapses BOTH.
  Satisfied.
- `no_shared_clock` (SHARED -> both): two desynchronized clocks
  desync WM-slot gating vs the hippocampal write, so the bound
  theta-ordered assembly is not coherently written. Pattern
  completion reconstructs no coherent role->filler binding -> WM
  collapses; the ordered episodic pattern is also not written -> `ep`
  collapses. Collapses BOTH. Satisfied.
- `no_hippo_store` (SHARED -> both): no engram tag is committed
  (the runner already returns 0.0 for the episodic readout under this
  lesion, and skips consolidation). Under Candidate A the WM readout
  IS the pattern-completion path, which has no tag/cue -> WM = 0.0 by
  the same construction; `ep` = 0.0. Collapses BOTH. Satisfied --
  and note this is now even cleaner than under distinct-pathways,
  because WM no longer depends on the schema at all.
- `no_neuromod_timing` (HELPER_BOTH -> both): removes the ACh-gated
  timed plasticity window, so the native STDP/eligibility writes onto
  the bound assembly are not gated to the correct timing -> the
  online bound pattern is degraded -> pattern completion yields no
  role-selective filler (WM collapses) AND the ordered episodic
  pattern is degraded (`ep` collapses). Collapses BOTH. Satisfied.
- `no_sequencing` (HELPER_EP -> `ep` only): makes the shared clock
  REPEAT instead of SHIFT, so no serial ORDER is written; the role->
  filler bindings themselves are still written (each binding is still
  encoded, just not order-tagged). The episodic-ORDER readout
  collapses (degenerate recovered order). The WM readout asks "which
  filler is bound to this role", which does not require serial order
  -> WM is NOT required to collapse. Collapses `ep` only, leaves `wm`
  intact. Satisfied (this is the desired asymmetry).
- `no_bg_gate` (HELPER_WM -> `wm` only): all `bg_cortex` channels
  driven -> all `thal_<chan>` partially disinhibit -> all
  `thal_<chan> -> dlpfc_verb` pathways inject -> no single prefrontal
  slot is cleanly held during encode, so the role->slot->filler
  binding is not cleanly written into the assembly the engram
  captures. Pattern completion then cannot recover a role-selective
  filler -> WM collapses. The episodic-ORDER readout depends on the
  SHIFTED theta assembly (the clock), not on clean single-slot BG
  gating, so `ep` is not required to collapse. Collapses `wm` only.
  Satisfied.
- `no_cls_replay` (HELPER_WM -> `wm` only) -- THE decisive case.
  Under the corrected v2 partition, removing the consolidation/replay
  MUST collapse `wm`. Under Candidate A the binding-specific WM signal
  comes from the online pattern-completion path, and the consolidated
  schema is only a WEAK additive prior. Precise analysis: if the
  schema prior is weak enough that it cannot overturn a correct
  pattern completion (the explicit Candidate A design intent), then
  removing it (the `no_cls_replay` lesion) will in general NOT collapse
  `wm` -- pattern completion alone still answers correctly. The v2
  instrument requires `wm <= 0.40` under `no_cls_replay`; a
  pattern-completion WM readout that is correct without the schema
  yields `wm` well ABOVE 0.40 under this lesion -> the v2 instrument
  reports "non-discriminating: helper lesion 'no_cls_replay' did NOT
  collapse the working-memory readout" -> VOID.

  This is an honest negative-by-CONSTRUCTION for Candidate A
  (and, by the same reasoning, for Candidate B unless its schema gate
  is made strong enough to be load-bearing -- which re-imports the
  genericization failure -- and for Candidate C unless the
  arbitration is tuned so the schema is necessary, which is
  goalpost-tuning). The biological reason is exact and is itself a
  finding: under correct CLS, a RECENT, just-bound item is retrievable
  by hippocampal pattern completion ALONE; the neocortical schema is
  NOT necessary for the recent-specific WM readout (it becomes
  necessary only for REMOTE items after the hippocampal trace has
  decayed/been overwritten). The v2 partition's
  `no_cls_replay -> WM-helper` membership is biologically correct for
  REMOTE working memory but the instrument's WM probe is a RECENT
  just-bound query (no long delay, no hippocampal overwrite), for
  which consolidation is by CLS theory NOT necessary. The two clean
  ways to make `no_cls_replay` collapse this RECENT `wm` are both
  forbidden: (a) make the schema load-bearing for the binding readout
  -- that is exactly the genericization failure the design exists to
  escape and also blurs the CLS factorization; (b) move a partition
  membership -- forbidden absolutely (no further partition edits,
  ever, on this line; the one biologically-cited correction was the
  single permitted move).

### 4.1 Decision: (a) realizes the corrected partition, or (b) honest negative-by-construction

DECISION: (b) honest negative-by-construction.

A faithful Candidate A (and B and C, for the precise reasons in
Section 4) cannot realize the corrected v2 partition, because the
corrected partition requires the consolidation/replay lesion to
collapse the working-memory readout, while the biologically-correct
recent-specific WM readout this design routes through hippocampal
pattern completion is -- by CLS theory and by the GPU evidence that
the pattern-completion path retains binding specificity -- retrievable
WITHOUT the neocortical schema for a RECENT just-bound item. The v2
instrument's WM probe is a recent just-bound query; under correct CLS
the schema is not necessary for it, so removing consolidation does NOT
collapse this `wm`, so the v2 discrimination check VOIDs.

This is decisive, not optimistic. The precise cause: the corrected v2
partition's `no_cls_replay -> WM-helper` membership is biologically
correct for REMOTE working memory, but the integrated-loop instrument
probes RECENT working memory, for which CLS theory says hippocampal
pattern completion suffices and neocortical consolidation is not
necessary. This is an honest negative-by-construction, surfaced with
its precise structural cause; it is NOT a reason to edit the frozen v2
partition (forbidden), NOT a configuration iteration, NOT spin, NOT a
hand-back.

Forward-looking note (for the controller, NOT a partition edit and NOT
part of this candidate's acceptance): this negative-by-construction is
itself a fourth convergent, biology-grounded signal about the
corrected partition -- specifically that the integrated-loop
instrument probes RECENT (not remote) working memory, and the
recent-WM / remote-WM distinction is the next catalog-identified
factorization axis. Pursuing it is autonomous, with NO partition edit,
under the same frozen v2 acceptance and the same honesty ceiling.


## 5. Acceptance gate, honesty ceiling, and pre-committed bound

Acceptance instrument: UNCHANGED. The sole scorer is
`research.runners.integrated_loop_core_v2.integrated_loop_verdict_v2`
(commit `36a7975`), with every numeric bar verbatim
(`_ILV2_LADDER=(2,4,8)`, `_ILV2_V1_MIN=0.90`, `_ILV2_SCI_MIN=0.80`,
`_ILV2_LESION_MAX=0.40`, `_ILV2_SCALE_TOL=0.10`, `_ILV2_MIN_SEEDS=3`)
and the corrected partition exactly as frozen. The original
`integrated_loop_core.py` is never the acceptance instrument and is
never edited; its "cannot conclude" (VOID) stands permanently as the
honest record that the original pre-registered prediction was
falsified.

Honesty ceiling (binding; restated, never softened): the load-bearing,
scale-confident scientific result of this arc is the THRICE-CONVERGENT
FALSIFICATION and catalog-grounded correction of the original
pre-registered necessity prediction. A PASS of any candidate built
from this design against the corrected v2 module is explicitly NOT
claimable as the scale-confident validated deliverable; at most it is
"consistent-with the corrected (biologically-revised) necessity
structure", always reported with this exact limitation, never spun. A
VOID or FAIL is a strong, unambiguous, informative negative and is
propagated as such, triggering the next catalog-identified
factorization with NO further partition edits.

New pre-committed bound (stated in advance, before any build, so no
outcome can be rationalized): a faithful build of the recommended
candidate (Candidate A, with B/C as the pre-described in-architecture
escalations) evaluated against the unchanged corrected frozen module
that reaches "cannot conclude" (VOID) or fails the corrected partition
is an honest negative. It is surfaced honestly with its precise,
GPU-measured (or, as here, precisely-reasoned) structural cause -- not
a configuration iteration, not spin, not a hand-back, not a
declare-globally-unfit. The next step is then the NEXT catalog-
identified integration factorization (per Section 4.1, the
recent-WM/remote-WM distinction is the indicated axis), pursued
autonomously with the SAME adversarial and anti-cheat discipline and
the SAME (corrected-module) frozen acceptance. NO further partition
edits, ever, on this line: exactly one biologically-cited correction
was permitted; a second would itself be goalpost-moving and is
forbidden.


## 6. Honest ceiling (final restatement)

This design does not claim conversational capability, compositional
memory, fluent language, or a large language model. All
previously-validated capabilities (trustworthy grounded memory,
no-confabulation abstention, simple generation, no catastrophic
forgetting) remain intact and unaffected; the no-confabulation moat is
byte-unchanged. The genuine, durable results of this line remain: (1)
the thrice-convergent falsification and catalog-grounded correction of
the original necessity prediction (the single scale-confident result),
and (2) the structural dissolution of the encode-order contradiction
via genuinely distinct pathways, with the remaining blocker precisely
localized -- and now, by this design's head-on analysis, precisely
characterized as the recent-WM vs remote-WM distinction within the
corrected CLS partition. No claim of fluent language or a large
language model is made anywhere.


## Files / evidence

- Necessitating finding:
  `research/findings/2026-05-19-distinct-pathways-honest-negative-
  episodic-contradiction-DISSOLVED-WM-blocked-by-consolidation-
  genericization.md`.
- Thrice-convergent falsification + honesty ceiling:
  `research/findings/2026-05-19-THIRD-convergent-signal-original-
  necessity-hypothesis-falsified-catalog-grounded-correction.md`,
  `research/findings/2026-05-19-CONTROLLER-precommitted-honesty-
  ceiling-on-the-corrected-module-PASS.md`.
- Corrected frozen necessity module (unchanged, sole scorer):
  `research/runners/integrated_loop_core_v2.py` (`36a7975`).
- Original frozen verdict (unchanged; "cannot conclude" preserved):
  `research/runners/integrated_loop_core.py` (`2048750`).
- Distinct-pathways runner (reused; the working online/offline
  pathways): `research/runners/integrated_loop_gate.py` (`b4a8106`).
- Reused validated assets: `research/runners/validate_trisynaptic_
  loop.py`, `research/runners/consolidation_trainer.py`,
  `research/runners/text_minimal_isolation.py`,
  `research/runners/abstention_gate.py`.
- Prior design in this line:
  `docs/plans/2026-05-19-distinct-readout-pathways-architecture-
  design.md`.
