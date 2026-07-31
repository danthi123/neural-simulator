---
type: plan
status: live
date: 2026-05-19
---

# Phase-factored consolidation architecture (design only)

Online theta-ordered hippocampal episodic encode + a separate offline
shuffled-replay neocortical consolidation, so the two validated
subsystems each operate in the phase whose encode-order requirement it
satisfies. SAME pre-registered frozen gate. No code in this document;
this is a design, not an implementation.

Date: 2026-05-19. Status: design pass for the next program step that
was pre-registered by the program-level finding (citation below). No
plan, no code, no run is authorized by this file.


## 0. One-paragraph orientation

The integrated-loop full-model instrument was built, adversarially
hardened, and iterated four times. Episodic-sequence binding is now
perfect; the basal-ganglia causal wiring, per-stripe homeostasis, and
the documented non-zero-initialization fix are all in place and
preserved in the honest work-in-progress commit `e02f692`. The single
remaining instrument-soundness gap is role-selective working-memory
binding. Iteration 4 applied the project's own validated
concept-binding mechanism to that gap; each faithful form recovered
working-memory selectivity but destroyed the now-perfect episodic
binding (episodic score fell from 1.0 to 0.0). The cause was measured
on the GPU, not inferred: the validated concept-binding mechanism
structurally requires a shuffled encode order, and the theta-gamma
episodic store structurally requires presentation-order to equal the
binding index. Forced into one online encode pass at the minimal
two-binding slice, these two validated subsystems impose contradictory
encode-order requirements; no configuration reconciles them. This
document designs the biology-faithful resolution: stop forcing both
into one pass.


## 1. Problem and biology grounding

### 1.1 The program-level finding (restated crisply)

Source: `research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-between-validated-concept-binding-and-episodic-store.md`
and its predecessors `...iter3-deeper-architecture...md` and
`...iter2-homeostasis...md`.

- The integrated loop is mechanically sound on every measured
  dimension except one. Episodic-sequence recall is perfect
  (episodic score = 1.0). The remaining gap is role-selective
  working-memory binding only.
- Iteration 3 established that a global scalar three-factor
  (dopamine / temporal-credit) signal does not produce role-selective
  working-memory binding even though the same machinery yields perfect
  episodic binding. This converged with the project's own months-old
  independent verdict: global scalar feedback fails to produce
  selective word-to-action binding at biological scale; the
  architecture is sufficient, the credit-assignment rule is the
  bottleneck. The project's own resolution to that verdict is the
  validated embodied co-firing plus topographic-prior binding
  mechanism (the validated multi-pool concept substrate).
- Iteration 4 applied exactly that validated concept-binding mechanism
  to the working-memory gap. Two faithful applications each recovered
  working-memory selectivity but drove the episodic score from 1.0 to
  0.0.
- Structural cause, GPU-measured: the validated concept-binding
  mechanism requires a SHUFFLED encode order -- that shuffle is
  precisely what breaks the within-kind winner-take-all dominant
  attractor so multiple same-kind pools become individually
  selectable. The theta-gamma episodic store recovers the sequence as
  the argsort of per-item activity-peak times, so it structurally
  requires presentation-order == binding-index; a shuffled encode
  order destroys episodic recall by construction. At the minimal
  two-binding slice these are contradictory in a single online encode
  pass. Verified by direct measurement; the pre-registered bound was
  hit and honored; there is no iteration 5 and no further
  configuration crank.

### 1.2 Why the conflict is diagnostic, not terminal

In the brain these two operations are not one encode pass with one
shared order constraint. They are two biologically distinct phases:

- ONLINE hippocampal episodic encoding binds items-in-sequence in a
  single theta-ordered pass; order is carried by theta-gamma phase
  (the theta-gamma multiplexing of a working-memory buffer; the
  encode rule that makes ordered recall possible). The episodic store
  in this project already implements this and already works perfectly
  in the loop.
- OFFLINE neocortical concept selectivity is built by INTERLEAVED
  REPLAY during consolidation. Sharp-wave-ripple-gated replay during
  quiet/sleep states reorders experience across many offline cycles
  and slowly trains stable, selective neocortical representations.
  This is the complementary-learning-systems account
  (McClelland/O'Reilly; the catalog's consolidation entries; and --
  decisively for this project -- the project's OWN validated
  Phase 1.3 hippocampus-to-cortex consolidation).

The encode-order contradiction iteration 4 measured is exactly the
signature expected when an integration has conflated these two
biologically distinct phases into one pass. The shuffle the concept
mechanism needs is the OFFLINE replay reordering; the
presentation-order==index the episodic store needs is the ONLINE
theta-ordered encode. They only conflict because both were forced into
one online pass. Separating them along the biological phase boundary
dissolves the contradiction by construction -- it is not a tuning
trick and not a new learning rule.

### 1.3 The project's own validated assets this design rests on

Cited from CLAUDE.md by name (these are protected, validated,
multi-seed assets -- this design REUSES them, it does not re-derive
them):

- Validated Phase 1.3 hippocampus-to-cortex consolidation:
  CLAUDE.md "Phase 1.3 CONSOLIDATION CONFIRMED" and "Phase 1.3 +
  Tier 2.1 COMBINED CONFIRMED 3/3 GO + ANTI-CHEAT VALIDATED".
  Empirically: hippo-OFF retention 94% single-seed; 3/3 strict
  anti-cheat multi-seed (`--strict-silence`, 10x stronger hippo
  silencing + zeroing ~194k ca1->cortex edges, identical retention to
  non-strict). This is the validated McClelland-1995 / Buzsaki-2013
  complementary-learning-systems consolidation subsystem -- "THE
  mechanism that makes continual learning possible without
  catastrophic forgetting at scale", empirically confirmed with SWR
  sleep replay. Concrete interfaces (all already byte-present in
  `research/runners/text_minimal_isolation.py` and
  `research/runners/consolidation_trainer.py`):
  - `build_biological_brain_regions(enable_hippocampus_consolidation=True)`
    -- builds the trisynaptic store (ec/dg/ca3/ca1) plus the
    `ca1 -> motor` / `ca1 -> language_output` consolidation pathways
    and the `ca3_swr_burst` recurrent autoassociator gate.
  - `set_awake_gates(bridge)` -- encoding ON, consolidation OFF
    (`ca3_swr_burst`/`ca1_to_motor`/`ca1_to_lang_out` = 0).
  - `set_sleep_gates(bridge)` -- input drive zeroed, direct
    lang->motor frozen, `ca3_swr_burst`/`ca1_to_motor`/
    `ca1_to_lang_out` = 1 (the consolidation phase).
  - `run_swr_replay_phase(...)` / `run_concept_replay_phase(...)` --
    SWR-style bursts (~150 Hz, 100 ms windows) that drive CA3
    attractors (random sparse, or selective to engram-tagged
    ensembles) so STDP at ca3->ca1->cortex consolidates the trace.
    `run_concept_replay_phase` already shuffles replay order
    (`randomize_order=True`) -- this IS the offline-shuffle the
    concept mechanism structurally requires.
  - `freeze_all_gates(bridge)` -- the validated pre-eval freeze.
  - The validated hippo-OFF / `--strict-silence` evaluation contrast
    (the anti-cheat that proves the trace is genuinely in cortex, not
    leaking from an imperfectly silenced hippocampus).
- Validated 16-pool concept substrate: CLAUDE.md
  "v14 5-SEED MULTI-SEED GO ... orthogonal codes + 16 pools" /
  "v16 5-seed MULTI-SEED GO". 88.75% multi-seed bidirectional
  role-selective concept-pool binding via weak concept dynamics
  (0.05/0.3/0.8) + per-pool FS cross-inhibition + orthogonal codes +
  the Pulvermuller topographic prior + reciprocal pool->language_output
  bias. This is the concept (role->filler) selectivity layer; it is
  already specified as the integrated loop's concept layer.
- Validated trisynaptic / engram episodic store: CLAUDE.md
  "Hippocampal trisynaptic loop (P1)" (pattern separation /
  completion validated) and the Tonegawa-style "Engram-tagging API"
  (`start_engram_recording` / `commit_engram_tag` / `stimulate_tag` /
  `clear_tag_drive` / `delete_engram_tag`) -- the fast relational
  episode store; persists through save/load. This is the online
  theta-ordered store that already scores episodic = 1.0 in the loop.
- The no-confabulation abstention moat: `research/runners/abstention_gate.py`
  (`gate`, `DEFAULT_THRESHOLD = 650.0`; AUC 0.990 know/don't-know
  separation). Byte-unchanged. Every drilled binding must clear it or
  the readout abstains.
- The verified-correct non-zero-init foundation: the documented
  zero-initialization fix on the slot-to-concept efferent, preserved
  exactly in honest-wip commit `e02f692` (the foundation this design
  builds ON; not re-litigated).
- Per-stripe homeostatic equalization: the project's validated
  homeostasis, already composed into the loop and working
  (iteration 2). Byte-unchanged.

The frozen acceptance gate is the existing
`research/runners/integrated_loop_core.py`
(`integrated_loop_verdict`, `_IL_*` bars). It is NOT touched by this
design (Section 4).


## 2. Architecture: candidate factorings and a recommendation

All three candidates share ONE invariant: the online episodic encode
keeps presentation-order == binding-index EXACTLY as the now-perfect
episodic store needs (the shared theta-gamma controller and the engram
write are unchanged); the concept (role->filler) selectivity is moved
OUT of that online pass and into a separate OFFLINE shuffled-replay
consolidation phase that uses the validated Phase 1.3 subsystem.
Net-new code in every candidate is ONLY (a) a phase controller that
sequences online-encode then offline-consolidation and (b) the wiring
that composes the two. NO new learning mechanism. NO automatic
differentiation. The working-memory readout is served from the
CONSOLIDATED neocortical concept representation; the episodic-sequence
readout remains served from the online theta-ordered hippocampal
store. This is what dissolves the encode-order contradiction: the
order-monotone constraint and the order-shuffled constraint now live
in different phases and never contend for one pass.

### Candidate A (minimal, recommended): two-phase online-encode then offline-replay consolidation

Per composition trial:

1. ONLINE ENCODE phase (UNCHANGED from `e02f692`). `set_awake_gates`.
   The shared theta-gamma clock gates the BG-selected prefrontal slot
   and times the engram write; role+filler orthogonal codes + teacher
   co-fire drive the bound pools; the episode is written to the
   engram store in theta order (presentation-order == binding-index).
   The episodic-sequence readout is taken here exactly as today
   (argsort of per-item activity-peak times against the true encode
   order) -- this is why episodic stays perfect.
2. OFFLINE CONSOLIDATION phase (NET-NEW WIRING, validated subsystem).
   `set_sleep_gates`. Drive the committed episode's engram-tagged CA3
   ensemble with SWR-style bursts via the validated
   `run_concept_replay_phase` with `randomize_order=True`. The
   validated `ca3_swr_burst` autoassociator + `ca1 -> concept`
   consolidation path replays the bound (role, filler) structure into
   the 16-pool concept layer in SHUFFLED order across many replay
   events -- exactly the encode-order the validated concept-binding
   mechanism structurally requires, now legitimately supplied by the
   offline phase instead of stolen from the online pass.
3. WM READOUT from the CONSOLIDATED concept layer. After the
   consolidation phase, `freeze_all_gates`, then query each role and
   read the consolidated filler pool. The role-selective filler now
   lives in the offline-trained neocortical concept representation, so
   the validated v16 selectivity recipe (already specified as the
   concept layer) operates in the phase it needs, not against the
   episodic store.

- Reuse: online encode + engram store + episodic readout +
  homeostasis + non-zero-init foundation = byte-unchanged from
  `e02f692`; Phase 1.3 consolidation (`set_awake_gates`,
  `set_sleep_gates`, `run_concept_replay_phase`, `freeze_all_gates`,
  the hippo-OFF/strict-silence anti-cheat) + the 16-pool concept
  substrate + the no-confab moat = byte-unchanged validated modules.
- Net-new: the per-trial phase controller (call awake-encode, then
  the offline replay-consolidation phase, then the consolidated-layer
  readout) and the composition wiring that routes the engram-tagged
  episode into `run_concept_replay_phase`. No new learning rule.
- Honest ceiling for A: the concept (role->filler) selectivity is now
  trained by the SAME validated mechanism (88.75% multi-seed) in the
  SAME phase it was validated in (offline interleaved replay), so its
  selectivity should transfer; episodic stays perfect because the
  online pass is untouched. Ceiling risk: whether the consolidated
  concept layer is role-selective ENOUGH, at the closed-loop minimal
  slice, to clear the frozen `_IL_V1_MIN` (0.90) AND `_IL_SCI_MIN`
  (0.80) on the working-memory readout WHILE every shared-system
  lesion still collapses BOTH readouts. That is the genuine science
  question and is reserved for the controller-only decisive run.
- Risks: (i) consolidation transfer quality at this slice (mitigation:
  the validated `run_concept_replay_phase` SWR intensities and the
  validated v16 recipe are reused, not re-tuned); (ii) the new
  offline-consolidation phase must itself be a lesionable system
  (handled in Section 5 -- the existing `no_cls_replay` helper lesion
  ALREADY targets exactly this path, so the lesion structure is
  already in place); (iii) trial wall-clock grows by the replay phase
  (bounded by the existing `replay_steps`-style budget, controller-run
  and kill-safe).
- Cheapest falsify-first de-risk: a single-seed, minimal-load (N=2),
  GPU smoke of just steps 1->2->3 measuring (a) episodic still == 1.0
  after the offline phase is inserted and (b) the consolidated WM
  readout is role-selective above chance. If episodic does NOT stay
  at 1.0 once the offline phase is inserted, the factoring assumption
  is wrong and the program-level escalation in Section 6 fires
  immediately -- no config crank.

### Candidate B (A + consolidation-gated prefrontal maintenance)

Candidate A, plus: between consolidation and readout, the consolidated
concept representation re-instates the prefrontal working-memory slots
through the existing BG-gated `thal_<chan> -> dlpfc_verb` afferent (a
biologically standard cortico-thalamo-prefrontal reinstatement), so
the working-memory readout is the consolidated concept content held in
the prefrontal slots rather than read straight off the concept pools.

- Adds robustness if the bare consolidated-pool readout is selective
  but weak through the frozen 650 gate (the prefrontal hold could
  sharpen it). Reuses the SAME validated BG-WM-gating wiring already
  in `e02f692`; net-new is only the ordering (consolidate, then
  reinstate, then read), not a new mechanism.
- Honest ceiling: same capability ceiling as A; strictly more wiring
  surface and one more place a faithfulness slip could hide -- so it
  is the SECOND choice, taken only if A's falsify-first smoke shows
  the bare consolidated readout is real but sub-gate.
- Risk: more net-new wiring => more adversarial-review surface
  (Section 5). Do not build B before A's smoke result is in.

### Candidate C (fullest: multi-cycle offline schedule)

Candidate A with the offline phase being multiple SWR replay cycles
across an alternating awake/sleep schedule over many trials (the full
CLS schedule the validated Phase 1.3 + Tier 2.1 3/3 anti-cheat result
actually used), instead of one replay phase per trial.

- Highest fidelity to the validated CLS regime and the catalog
  consolidation biology; most likely to maximize consolidated
  selectivity. But the largest wall-clock and the largest schedule
  surface; only justified if A and B both show real-but-insufficient
  consolidated selectivity at the minimal slice.
- Honest ceiling: same capability claim; more compute, not a stronger
  claim type.

### Recommendation

Build Candidate A. It is the minimal factoring that places each
validated subsystem in the phase whose encode-order requirement it
satisfies, it is the most reuse-heavy (net-new is only the phase
controller + composition wiring), and it is the cheapest to
falsify-first (the single-seed N=2 GPU smoke above either preserves
episodic == 1.0 with above-chance consolidated WM selectivity, or it
fails fast and triggers the Section 6 escalation with a precise
structural cause). B and C are pre-described escalations within the
SAME phase-factored architecture (not new architectures) and are taken
ONLY on an honest, propagated A-smoke signal -- never as a reflexive
config crank.


## 3. Reuse map (DRY; protected)

Byte-UNCHANGED (imported, not copied; not edited by any candidate):

| Subsystem | Module / interface | Phase it serves |
|---|---|---|
| Online theta-ordered episodic encode + episodic readout | the shared theta-gamma controller + engram-tagging API (`start_engram_recording`/`commit_engram_tag`/`stimulate_tag`/`clear_tag_drive`/`delete_engram_tag`) as wired in `integrated_loop_gate.py` at `e02f692` | ONLINE |
| Hippocampal relational store (trisynaptic ec/dg/ca3/ca1) | `build_biological_brain_regions(enable_hippocampus_consolidation=True)` | ONLINE encode + OFFLINE replay |
| Phase 1.3 CLS consolidation | `set_awake_gates`, `set_sleep_gates`, `run_concept_replay_phase` (`randomize_order=True`) / `run_swr_replay_phase`, `freeze_all_gates`, the hippo-OFF / `--strict-silence` anti-cheat contrast | OFFLINE |
| 16-pool concept (role->filler) selectivity | the validated v16 recipe + topographic-prior helper already used by `integrated_loop_gate.py` (weak dynamics, FS cross-inhibition, orthogonal codes, reciprocal bias) | OFFLINE-trained, read at WM readout |
| Per-stripe homeostasis | the project's validated homeostasis, as composed in `e02f692` | both |
| Non-zero-init foundation | the documented slot-to-concept efferent non-zero prior, exactly as in `e02f692` | both |
| No-confabulation moat | `research/runners/abstention_gate.py` (`gate`, `DEFAULT_THRESHOLD = 650.0`) | WM readout |
| Frozen acceptance verdict | `research/runners/integrated_loop_core.py` (`integrated_loop_verdict`, `_IL_*`) | scoring |
| Native temporal-credit / phasic-DA / ACh window | bridge `cp_eligibility_trace` path + the reused NM subsystem (`_da_modulator_from_delta`, `_ach_window_modulator`) as in `e02f692` -- relegated to credit/gating, NOT binding | both |

NET-NEW (the ONLY new code any candidate introduces):

1. A phase controller: per composition trial, run the ONLINE
   awake-encode phase (presentation-order == binding-index,
   unchanged), then the OFFLINE replay-consolidation phase
   (`set_sleep_gates` -> `run_concept_replay_phase` on the committed
   episode's engram tag, shuffled), then `freeze_all_gates` and read
   the consolidated concept layer for the WM readout. Pure sequencing
   of EXISTING validated calls. No learning logic of its own.
2. The composition wiring connecting the online engram tag to the
   offline replay input and the consolidated concept layer to the WM
   readout. Wiring only.

There is NO new learning mechanism, NO new plasticity rule, NO
automatic differentiation, NO change to any frozen bar, NO change to
the no-confabulation moat, NO edit to any protected/validated module.


## 4. The SAME pre-registered FROZEN gate (no new bar invented)

Acceptance is UNCHANGED. The existing frozen module
`research/runners/integrated_loop_core.py` (`integrated_loop_verdict`,
`_IL_LADDER = (2,4,8)`, `_IL_V1_MIN = 0.90`, `_IL_SCI_MIN = 0.80`,
`_IL_LESION_MAX = 0.40`, `_IL_SCALE_TOL = 0.10`, `_IL_MIN_SEEDS = 3`,
the `_SHARED` / `_HELPER_WM` / `_HELPER_EP` / `_HELPER_BOTH`
partition) decides PASS / FAIL / VOID exactly as it does today.

Stated explicitly:

- No frozen bar is changed, added, removed, or re-justified.
- The `_IL_*` values, the `(2,4,8)` ladder, and the lesion partition
  are byte-unchanged.
- The no-confabulation moat (`abstention_gate.DEFAULT_THRESHOLD =
  650.0`) is byte-unchanged; every drilled binding must clear it or
  the readout abstains, exactly as today.
- The instrument-validity-first, fail-closed, VOID-distinct-from-FAIL
  discipline is unchanged: the full loop must clear `_IL_V1_MIN` on
  BOTH readouts at the no-gap trivial bind; every shared-system lesion
  must collapse BOTH readouts at/under `_IL_LESION_MAX`; each helper
  lesion must collapse the readout it owns; full+lesions on the
  identical novel probe with identical RNG draws; sub-bar or
  non-discriminating => VOID, not a quiet pass.
- Real runs are GPU/CuPy (the documented `--deterministic`
  CUBLAS_WORKSPACE_CONFIG practice; multi-seed >= `_IL_MIN_SEEDS`;
  verdict recomputed from the single recorded JSON by the frozen
  module). NumPy is for the tiny non-propagated smoke only. NO
  automatic differentiation anywhere.
- The factoring changes WHICH phase produces the WM-readout content
  (offline-consolidated concept layer instead of a single online
  pass); it does NOT change what is measured or the bar it is
  measured against. The science question (does the consolidated WM
  readout clear the unchanged bars while the lesion contrasts still
  discriminate) is reserved for the controller-only decisive
  multi-seed run.


## 5. Anti-cheat and the lesion study under the new factoring

The decisive emergent-from-integration signature is unchanged: the
three SHARED systems (the combinatorial binding step, the one shared
theta-gamma rhythm, the fast hippocampal store) must collapse BOTH
readouts when lesioned; each HELPER collapses the readout it owns.
That logic is owned by the frozen `integrated_loop_core` and is not
touched.

The new offline-consolidation phase is itself a lesionable system, and
crucially the lesion ALREADY EXISTS: `no_cls_replay` is a frozen
`_HELPER_EP` lesion in `integrated_loop_core` and is already
implemented in `integrated_loop_gate.py` (it skips the
replay/consolidation path). Under the phase-factored architecture:

- `no_cls_replay` now also removes the offline shuffled-replay that
  trains the consolidated concept layer. If the WM readout is genuinely
  served from the consolidated representation, removing the
  consolidation phase must collapse the WM readout it is responsible
  for. (Whether `no_cls_replay`'s frozen partition membership still
  classifies it correctly under the new factoring is a question for
  the dedicated adversarial review below; the frozen module is NOT
  edited to accommodate the architecture -- if the architecture makes
  a frozen lesion non-discriminating, that is a VOID the frozen
  verdict must surface, not a bar to soften.)
- The three SHARED lesions keep their meaning: removing the binding
  step, the shared clock, or the hippocampal store still collapses
  BOTH the online episodic readout and the (now consolidation-fed) WM
  readout, because the consolidated WM content is DOWNSTREAM of the
  online engram write -- no online episode, no replay source, no
  consolidated concept, no WM. The shared-systems-collapse-both
  signature must still hold and is checked by the unchanged frozen
  partition.

Discipline (unchanged from the program's standing practice):

- A dedicated adversarial review of the net-new phase controller +
  composition wiring BEFORE any no-harm/decisive phase, specifically
  probing: that the online pass is byte-identical to `e02f692` (so
  episodic cannot silently degrade), that the offline replay genuinely
  shuffles (so it is the real CLS mechanism, not a relabeled online
  pass), that no query-time teaching or answer key leaks through the
  consolidation phase, and that every mode (full, v1, all 8 lesions)
  consumes IDENTICAL RNG draws in IDENTICAL order with only the
  lesioned system removed (the no-strawman faithfulness rule).
- Controller trust-but-verify: the decisive multi-seed run is
  controller-only and kill-safe; the verdict is recomputed from the
  single recorded JSON by the frozen module.
- Honest propagation of every outcome (PASS, FAIL, VOID, or a deeper
  program-level structural result) to both remotes, no spin.


## 6. Pre-registered bound (stated now, before any run)

If a faithful phase-factored build (Candidate A, with B/C as the
pre-described in-architecture escalations) ALSO cannot achieve
`v1 wm AND ep >= 0.90` with the frozen lesion contrasts
discriminating, that is a deeper program-level result and is surfaced
honestly with its precise, GPU-measured structural cause -- not a
configuration iteration, not spin, not a hand-back, not a
declare-globally-unfit. The next step is then the next
catalog-identified integration factorization (e.g. a deeper separation
of relational binding from schema abstraction along the catalog's
hippocampal-neocortical interaction entries), pursued autonomously
with the SAME adversarial and anti-cheat discipline and the SAME
frozen acceptance. The falsify-first smoke in Section 2 is the
explicit early trigger: if inserting the offline phase does not keep
episodic == 1.0, the escalation fires immediately with that exact
structural cause, with no config crank. This bound is stated in
advance so the next outcome cannot be rationalized after the fact.


## 7. Honest ceiling (stated, never overstated)

A phase-factored PASS means exactly this: a biology-grounded
multi-phase loop (an online theta-ordered hippocampal episodic encode
plus a separate offline shuffled-replay neocortical consolidation)
shows emergent compositional memory that holds and scales across the
frozen `(2,4,8)` load ladder, where every single-system lesion
abolishes the capability it is responsible for and every shared-system
lesion collapses both readouts together. It is explicitly NOT fluent
or open-ended language, NOT a large language model, NOT conversation
solved. No claim here asserts the decisive test has passed; this
document only designs the next properly-scoped program step.
