# Remote/consolidated-memory-regime necessity test -- architecture design (design-only)

Status: DESIGN ONLY. No code, no frozen-module edit, no moat edit, no
GPU run. This document specifies the architecture for the next
catalog-identified factorization the fourth convergent finding points
straight at: probe the necessity structure in the REMOTE/consolidated
memory regime, reusing the project's already-validated Phase-1.3
strict-silence / hippocampus-OFF protocol byte-unchanged. It is written
to be exercised against the UNCHANGED corrected frozen necessity module
`research/runners/integrated_loop_core_v2.py` (commit `36a7975`); that
module's bars and partition are NOT touched here and MUST NOT be touched.

Terms used once, then reused literally:

- Recent memory: a binding queried within the same trial it was written,
  after only a brief maintenance gap. Complementary-learning-systems
  (CLS) theory says recent memory is served by the hippocampus and is
  consolidation-INDEPENDENT.
- Remote (consolidated) memory: a binding queried only AFTER an offline
  systems-consolidation phase has transferred it into a neocortical
  store, and queried while the hippocampus is silenced so only the
  neocortical store can answer. CLS theory says remote memory is
  neocortical and consolidation-DEPENDENT.
- WM readout (`wm`): the role-selective working-memory query -- present
  a queried role, population-vote the filler concept pools, emit only if
  the no-confabulation gate passes, else abstain.
- EP readout (`ep`): the episodic-sequence-order recall -- recover the
  order in which the bound pairs were written and score it against the
  true encode order.
- Strict-silence / hippocampus-OFF: the project's validated Phase-1.3
  protocol that forces a strong negative current onto every hippocampal
  region every step so only consolidated neocortex can answer. The exact
  validated mechanism is `evaluate_with_hippo_off` in
  `research/runners/consolidation_eval.py`
  (`HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3","ca1"]`; a
  monkey-patched `_run_one_simulation_step` that re-applies
  `silence_current_pA` -- `-200 pA` standard, `-2000 pA` in the
  3/3-strict-anti-cheat configuration -- to those region indices before
  every step).
- Corrected frozen v2 module: `integrated_loop_core_v2.py` (`36a7975`).
  Its only substantive change vs the original frozen module is one
  biologically-cited partition move: the consolidation/replay lesion
  `no_cls_replay` is in the working-memory helper set
  (`_ILV2_HELPER_WM = ("no_bg_gate","no_cls_replay")`), not the episodic
  helper set. All numeric bars are byte-identical to the original.

ASCII only. Plain professional language. No internal codenames as
load-bearing terms.

---

## 1. Problem and biology grounding

### 1.1 The fourth convergent finding, restated crisply

Four faithful, independent routes converged on one structural
characterization of the integrated-loop necessity instrument:

1. Single online pass: the validated concept-binding mechanism needs a
   shuffled encode order; the validated theta-gamma episodic store needs
   a fixed order equal to the binding index. Directly contradictory.
2. Phase-factored relocation of (1): the shuffled offline replay
   required for concept selectivity destroys episodic order; skipping
   consolidation preserves it. The original partition's
   "remove-consolidation -> collapse-episodic" duty inverts.
3. Distinct readout pathways: the encode-order contradiction is
   genuinely DISSOLVED (online trisynaptic pattern-completion gives
   episodic order = 1.0, undamaged by inserting a separate offline
   consolidation pathway), but the working-memory readout reads 0.0
   because the order-invariant consolidated schema abstracts away the
   episode-specific role-to-filler binding.
4. The structural realization (the load-bearing fourth finding): the
   corrected frozen v2 module places `no_cls_replay` in the
   working-memory helper set -- removing the consolidation system must
   collapse the WM readout. That membership is biologically CORRECT for
   REMOTE working memory (CLS: remote memory is neocortical and
   consolidation-dependent). But the integrated-loop instrument, across
   the entire arc, probes RECENT working memory: it binds role-filler
   pairs and queries them within the same trial after a brief
   maintenance gap, never after systems consolidation has had to take
   over. CLS theory is unambiguous that recent memory is
   hippocampus-served and consolidation-INDEPENDENT. The arc has already
   GPU-proven that the hippocampal trisynaptic pattern-completion
   pathway retains the binding specificity (that is exactly why it
   delivers perfect episodic order). Therefore, in ANY biology-faithful
   architecture that serves recent bindings via that pathway, removing
   the consolidation system CANNOT collapse the recent WM readout --
   because, biologically, recent recall does not need consolidation. The
   only two ways to force that collapse are both forbidden: make the
   consolidated schema load-bearing for recent recall (re-imports the
   genericization blocker, itself zeroing WM), or move the partition
   (forbidden -- no further partition edits). Hence the corrected
   necessity instrument, posed on RECENT memory, cannot be satisfied by
   faithful biology, by construction.

The durable, scale-confident scientific result of this entire line is
therefore complete and precise and is NOT re-litigated here: a
biology-faithful integrated loop, tested for necessity on RECENT memory,
cannot make the consolidation system necessary -- not because the local
implementation is unfit, but because CLS biology says consolidation is
simply not necessary for recent recall. This identifies exactly the
regime in which the necessity question IS well-posed.

### 1.2 Remote vs recent: the catalog division of labor and the project's OWN validated proof

The well-posed regime is the REMOTE/consolidated regime. Its biology is
the standard CLS hippocampal-neocortical division of labor (McClelland
1995; Buzsaki 2013; reference-catalog D.03 trisynaptic pathway, D.12
pattern separation, D.13 pattern completion, D.14 engram cells, D.19
sharp-wave ripples) and -- decisively for this design -- the project's
OWN validated result:

- Validated Phase-1.3 strict-silence / hippocampus-OFF consolidation
  (cited by name; this is the EXACT protocol being reused
  byte-unchanged). Per CLAUDE.md: "Phase 1.3 + Tier 2.1 COMBINED
  CONFIRMED 3/3 GO + ANTI-CHEAT VALIDATED". The validated anti-cheat run
  (`--strict-silence`: 10x stronger hippo silencing at `-2000 pA` plus
  zeroing the ca1->cortex edges) produced retention IDENTICAL to
  non-strict across all three seeds, confirming Hypothesis B: the cortex
  TRULY retains the bound pattern after consolidation; sleep replay
  genuinely transfers the binding into cortical internal recurrence so
  the cortex does not need the hippocampus at all post-consolidation.
  Findings:
  `research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md`,
  `research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`,
  `research/findings/2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md`.

The single load-bearing consequence for this design: in the remote
regime, with the hippocampus silenced by the validated strict-silence
mechanism, the ONLY thing that can answer a query is what offline
consolidation actually transferred into neocortex. Therefore removing
the consolidation/replay system (`no_cls_replay`) genuinely collapses
remote recall (nothing was consolidated -> nothing to recall once the
hippocampus is silenced), while intact consolidation supports it. The
corrected v2 module's "`no_cls_replay` -> WM-collapse" duty is BOTH
biologically correct AND satisfiable in this regime -- the precise
property that was unsatisfiable on the recent-memory probe.

The recent-regime episodic-order contradiction (signals 1-3) does NOT
arise here, because order-specific recent recall is not what is probed
in the consolidated regime. The probed quantity is consolidated
semantic/role recall plus, optionally, the order that consolidation
itself transferred. This is a substantively different instrument regime
(WHAT memory is probed), not a partition edit, not a configuration
crank, and it reuses a validated project protocol.

---

## 2. Concrete remote-regime instrument designs

All three designs share an identical per-trial spine and differ only in
which readout(s) are taken in the consolidated regime. The spine is:

  online theta-ordered ENCODE + engram WRITE (byte-unchanged, the
  existing `b4a8106` online path)
    -> offline consolidation phase (validated Phase-1.3
       `run_concept_replay_phase` under `set_sleep_gates`,
       byte-unchanged)
    -> STRICT-SILENCE the hippocampus (validated `evaluate_with_hippo_off`
       mechanism, byte-unchanged: monkey-patched step re-applies the
       strong negative current to `HIPPO_REGIONS` every step; the
       `--strict-silence -2000 pA` configuration, the exact validated
       3/3 anti-cheat strength)
    -> under the validated `freeze_all_gates` pre-eval freeze, query the
       CONSOLIDATED NEOCORTICAL store for the regime's readout(s).

Across full + every lesion + v1 the per-trial RNG draws are IDENTICAL
(the existing faithfulness discipline: a dedicated deterministic local
RNG seeded from the episode id for the replay phase; `_make_pairs` in
`_run_mode` remains the sole consumer of the per-trial `rng`). A lesion
that deterministically skips a phase produces a deterministic skip, not
an extra/missing draw -- exactly how the existing `b4a8106` path already
skips its replay for `no_cls_replay` / `no_hippo_store`.

### Design A (minimal) -- consolidated role-selective WM only

Take ONLY the `wm` readout, in the remote regime, from the consolidated
neocortical concept layer with the hippocampus strict-silenced. `ep` is
NOT scored as a science readout in this design (set to a deterministic
construction value that satisfies the instrument's discrimination
checks; see Section 4 for why this is sound for the `ep`-side helper
lesion only if its duty is met by construction -- which it is not in
Design A alone, so Design A standing alone CANNOT satisfy the v2
instrument, see the honest ceiling below).

- Honest ceiling / risk: the corrected v2 module REQUIRES both readouts
  to be scored and requires the EP-side helper (`no_sequencing`) and the
  shared lesions to collapse the EP readout too. A WM-only instrument
  cannot exercise `_ILV2_HELPER_EP` or the EP half of the shared
  lesions, so the v2 module's instrument-validity gate (which iterates
  ALL seven lesions and checks the EP side of shared/both lesions) is
  not satisfiable by Design A alone. Design A is therefore informative
  as a falsify-first PROBE of the single hardest mechanism (does
  consolidated WM survive strict-silence at the minimal load, and does
  `no_cls_replay` genuinely collapse it there) but is NOT a complete
  acceptance instrument. Cheapest, but not sufficient on its own.

### Design B (recommended) -- consolidated WM + consolidated episodic-order, both in the remote regime

Take BOTH readouts in the remote regime from the consolidated
neocortical store with the hippocampus strict-silenced:

- `wm`: consolidated role-selective working memory. Present the queried
  role on `language_input` only (no query-time teacher/external current
  into noun_pool/dlpfc -- exactly the existing query discipline);
  population-vote the consolidated filler concept pools; emit through
  the byte-unchanged no-confabulation gate (`DEFAULT_THRESHOLD = 650.0`)
  or abstain. The novel-recombination probe applies to full + every
  lesion exactly as today (the science is not made easier); v1 keeps the
  trivial drilled-binding soundness query.
- `ep`: consolidated episodic-sequence ORDER. The per-binding theta
  SHIFT order is written ONLY into the hippocampal episode at the
  byte-unchanged online encode. The 16-pool concept layer acquires that
  ordered sequence ONLY when the offline `run_concept_replay_phase`
  drives the ca3_swr_burst autoassociator -> ca1 -> concept
  consolidation. With the hippocampus strict-silenced, the recovered
  per-role peak order is read from the CONSOLIDATED ca1->concept trace
  (the engram tag is used only as the natural CA3 retrieval cue; the
  ORDER is carried by the consolidated cortical pathway -- exactly the
  validated strict-silence anti-cheat mechanism that the project already
  proved transfers ordered W->A binding into cortical recurrence).

This is the biologically-correct consolidated-regime analogue of the
distinct-pathways factoring: in the remote regime BOTH readouts are
legitimately served by the consolidated neocortical store (because the
hippocampus is silenced), so there is no order-shuffled-vs-order-monotone
single-trace contention of the recent-regime kind -- the order that
exists post-consolidation is precisely the order consolidation
transferred, and a query for it reads that transferred order; it does
not require an order-PRESERVING online pathway because the online
pathway is silenced and is not the thing under test.

- Honest ceiling / risk: the central scientific risk (Section 4.8) is
  whether the project's validated strict-silence consolidation, which
  was validated for an ORDER-AGNOSTIC W->A semantic binding, also
  transfers ORDER cleanly enough that a consolidated `ep` readout clears
  the science bar at the minimal load. The validated Phase-1.3 result is
  about semantic/role retention surviving hippo-OFF; it did not
  separately validate that SERIAL ORDER survives consolidation. This is
  the single biggest honest risk and is called out for the
  writing-plans step. If consolidated order does NOT survive, Design B
  is an honest negative-by-construction with a precise, catalog-located
  cause (Section 4.9).

### Design C (fullest) -- B plus a consolidated free-generation episodic readback

Design B plus a third readback in the remote regime: after strict-silence,
drive a partial role cue and let the consolidated neocortical store
free-run, scoring both the recovered binding AND the recovered order
from the same consolidated free-running trace (a stricter,
pattern-completion-style probe entirely within neocortex).

- Honest ceiling / risk: strictly harder than B with no additional
  partition coverage; a consolidated free-run is more sensitive to the
  same order-transfer risk as B and adds a second failure surface
  (free-run drift in the 16-pool concept layer without hippocampal
  pattern completion). Not recommended as the primary instrument;
  retained as a pre-described in-architecture escalation only if B
  PASSES and a stricter confirmation is wanted -- it never makes B's
  acceptance easier.

### Recommendation and cheapest falsify-first de-risk

Recommend Design B. It is the minimal instrument that (a) makes the
corrected v2 `no_cls_replay` -> WM duty satisfiable AND biologically
correct, (b) scores BOTH readouts so the v2 module's full
instrument-validity gate over all seven lesions is exercisable, and
(c) reuses the validated strict-silence protocol byte-unchanged.

Cheapest falsify-first de-risk (MANDATORY shape; honors the recorded
process lesson that a de-risk must probe the FULL science-mode readout
set JOINTLY at the single minimal load, never the soundness mode alone):
at the SINGLE minimal load (N = 2, the smallest pre-registered rung) and
the minimum seed count (`_ILV2_MIN_SEEDS = 3`), run the full
remote-regime spine ONCE in full mode and ONCE in the `no_cls_replay`
mode, and read BOTH the consolidated `wm` AND the consolidated `ep`
readouts JOINTLY in the same run. The falsify-first PASSES (proceed)
only if, jointly: (i) full-mode consolidated `wm` >= the science bar AND
full-mode consolidated `ep` >= the science bar at N = 2, AND (ii)
`no_cls_replay` collapses BOTH consolidated readouts (<= the lesion
ceiling) at N = 2. Any other joint outcome is an immediate honest
negative with the precise measured cause; do NOT escalate the ladder, do
NOT crank configuration. This single joint minimal-load probe falsifies
the dominant risk (consolidated-order transfer + consolidated-WM
survival under strict-silence + the `no_cls_replay` collapse) before any
expensive full-ladder multi-seed run.

---

## 3. Reuse map and net-new

### Byte-unchanged (imported, not modified)

- Validated Phase-1.3 consolidation: `run_concept_replay_phase`
  (`research/runners/consolidation_trainer.py`) under the validated
  `set_sleep_gates` / `set_awake_gates` / `freeze_all_gates`
  (`research/runners/text_minimal_isolation.py`).
- Validated strict-silence / hippocampus-OFF mechanism: the
  `evaluate_with_hippo_off` silencing idiom from
  `research/runners/consolidation_eval.py`
  (`HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3","ca1"]`;
  monkey-patched step re-applying the strong negative current every
  step; the `-2000 pA` strict / ca1->cortex-zeroing configuration that
  was the validated 3/3 strict anti-cheat). Reused as the regime
  mechanism, semantics preserved exactly.
- Trisynaptic / engram store + the consolidation pathway:
  `build_biological_brain_regions` (ec/dg/ca3/ca1 + ca1->concept
  consolidation) and the engram-tagging API.
- 16-pool concept substrate + homeostasis + non-zero-init:
  `build_biological_brain_regions` / `text_minimal_isolation.py`
  (validated v14/v16 weak-dynamics regime, orthogonal codes, per-pool FS
  lateral inhibition).
- No-confabulation moat: `research/runners/abstention_gate.py`
  (`gate`, `DEFAULT_THRESHOLD = 650.0`), byte-unchanged. Every drilled
  WM binding must clear it or the readout abstains.
- Corrected frozen v2 acceptance module:
  `research/runners/integrated_loop_core_v2.py` (`36a7975`),
  byte-unchanged -- the sole verdict authority.
- Original frozen core: `research/runners/integrated_loop_core.py`
  (`2048750`), byte-unchanged and NEVER imported by the runner; its
  prior "cannot conclude" (VOID) stands permanently as the honest record
  that the original pre-registered necessity prediction was falsified.
- The `b4a8106` online/offline pathways in
  `research/runners/integrated_loop_gate.py`: the genuinely-distinct
  online trisynaptic pattern-completion episodic write/path and the
  separate offline Phase-1.3 consolidation, the per-trial controller,
  the engram fan-out, the native eligibility/temporal-credit reward path
  -- all reused; the online ENCODE + engram WRITE stay byte-identical.

### Net-new (the ONLY new code)

- A remote-regime per-trial controller: the phase sequencing that, after
  the byte-unchanged online ENCODE + WRITE and the byte-unchanged
  offline consolidation, ENGAGES the validated strict-silence mechanism
  and THEN takes BOTH readouts from the consolidated neocortical store
  under the validated `freeze_all_gates` pre-eval freeze.
- The hippo-silence sequencing/wiring: invoking the validated
  `evaluate_with_hippo_off` silencing idiom at the correct point in the
  per-trial sequence (after consolidation, around the consolidated
  readouts, restored at trial end) with the validated strict
  configuration, and the deterministic per-lesion skip wiring
  (`no_cls_replay` / `no_hippo_store` skip consolidation exactly as the
  existing path already does, with identical RNG effect).

Explicitly NO new learning rule. NO automatic differentiation. NO new
verdict/partition module. NO bar change. NO moat change. Every learning
update remains the reused validated native eligibility/temporal-credit
rule.

---

## 4. Head-on confrontation of the corrected v2 frozen partition, lesion by lesion, in the REMOTE regime

The verdict is decided ENTIRELY by the corrected frozen v2 module
(`36a7975`). Its frozen partition (UNCHANGED, restated for the analysis,
not edited):

- `_ILV2_SHARED = ("no_binding","no_shared_clock","no_hippo_store")` --
  each MUST collapse BOTH `wm` AND `ep` (<= 0.40) at every load.
- `_ILV2_HELPER_WM = ("no_bg_gate","no_cls_replay")` -- each MUST
  collapse `wm` (<= 0.40) at every load.
- `_ILV2_HELPER_EP = ("no_sequencing",)` -- MUST collapse `ep`
  (<= 0.40) at every load.
- `_ILV2_HELPER_BOTH = ("no_neuromod_timing",)` -- MUST collapse BOTH.

NOTE on the runner's local comment partition: the `b4a8106` runner
contains a LOCAL comment partition (`_HELPER_EP = ("no_sequencing",
"no_cls_replay")`) that pre-dates the regime change. That local comment
is NOT the acceptance authority and is NOT edited by this design (the
runner is reused as-is for its online/offline pathways). The acceptance
authority is the v2 module's partition, where `no_cls_replay` is a
WM-helper. In the remote regime the runner's `no_cls_replay` behavior
(skip consolidation) produces a consolidated-WM collapse, which is
exactly what the v2 WM-helper duty requires -- so the regime change
resolves the apparent local/authority mismatch in the v2 module's
favor WITHOUT any partition edit. This is stated explicitly so the
adversarial review cannot mistake the unchanged local comment for a
goalpost-move.

Per-lesion mechanism in the REMOTE regime, with IDENTICAL per-trial RNG:

1. `no_cls_replay` (v2: WM-helper -> MUST collapse `wm`). THE
   load-bearing case. In the remote regime the query is answered ONLY by
   the consolidated neocortical store while the hippocampus is
   strict-silenced. `no_cls_replay` deterministically SKIPS the offline
   `run_concept_replay_phase` -> NOTHING is consolidated into the
   16-pool concept layer for this episode -> with the hippocampus
   silenced there is no surviving trace anywhere -> the consolidated WM
   query has no above-threshold filler -> the no-confab gate abstains ->
   `wm` collapses to chance/zero (<= 0.40). This is the SAME validated
   strict-silence logic the project already proved: post-consolidation
   recall is carried by cortex; remove the consolidation that builds it
   and silence the hippocampus, and there is nothing to recall.
   CONCLUSION: genuinely collapses the consolidated WM readout ->
   satisfies the corrected v2 WM-helper duty, biologically correctly.
   This is exactly the duty that was UNSATISFIABLE on the recent-memory
   probe and is now satisfiable purely because the regime is remote.

2. `no_hippo_store` (v2: SHARED -> MUST collapse BOTH). No engram tag is
   ever committed -> the online write has no relational store -> there
   is no tag for `run_concept_replay_phase` to replay -> nothing
   consolidates -> with the hippocampus also silenced, both the
   consolidated `wm` and the consolidated `ep` have no trace ->
   `wm` = 0 and `ep` = 0 by construction. Collapses BOTH -> satisfies
   the SHARED duty.

3. `no_binding` (v2: SHARED -> MUST collapse BOTH). Without the binding
   step's excitability bias no bound (role, filler) assembly is written
   online -> the engram captures no bound pattern -> consolidation
   transfers no role-selective structure and no per-role order ->
   consolidated `wm` has no role-selective filler (gate abstains) and
   consolidated `ep` has no recoverable order -> BOTH collapse ->
   satisfies the SHARED duty.

4. `no_shared_clock` (v2: SHARED -> MUST collapse BOTH). Without the one
   shared theta-gamma rhythm the WM-slot timing and the hippocampal
   write are not co-timed: the bound assembly is not coherently written,
   so the online engram is not a clean bound pattern AND no consistent
   per-binding SHIFT order exists -> consolidation has nothing coherent
   to transfer -> consolidated `wm` and consolidated `ep` both collapse
   -> satisfies the SHARED duty. (Per the runner, the lesion
   instantiates two unsynchronized clocks; the consolidated readouts
   inherit the incoherence.)

5. `no_neuromod_timing` (v2: HELPER_BOTH -> MUST collapse BOTH). The
   clock-gated ACh plasticity window is removed consistently across the
   whole loop, so the online binding write is untimed (no clean bound
   assembly) AND the consolidation-relevant plasticity is untimed. The
   online engram is not a clean bound, ordered pattern, so the
   subsequent consolidation transfers neither a role-selective binding
   nor a recoverable order -> consolidated `wm` and consolidated `ep`
   both collapse -> satisfies the HELPER_BOTH duty.

6. `no_sequencing` (v2: HELPER_EP -> MUST collapse `ep`). With
   sequencing off the online theta clock REPEATS instead of SHIFTING ->
   no per-binding order is written at encode -> there is no ordered
   sequence for `run_concept_replay_phase` to consolidate -> the
   consolidated ca1->concept trace yields a degenerate (non-recoverable)
   per-role peak order under strict-silence -> consolidated `ep`
   collapses (<= 0.40). The consolidated `wm` (a role-selective query,
   order-agnostic) need NOT collapse here -- correct, because
   `no_sequencing` is an EP-only helper in v2. CONCLUSION: collapses the
   consolidated EP readout -> satisfies the HELPER_EP duty.

7. `no_bg_gate` (v2: WM-helper -> MUST collapse `wm`). The basal-ganglia
   selective gate is repurposed to gate WHICH prefrontal WM slot updates
   vs holds. With it removed ALL channels are driven during encode -> no
   single slot holds a clean (role -> filler) binding -> the online
   engram captures an unselective mixture -> consolidation transfers a
   non-role-selective blur -> the consolidated WM query has no
   above-threshold role-specific filler -> the no-confab gate abstains
   -> consolidated `wm` collapses (<= 0.40). EP is order-driven and not
   gated by the BG slot selector, so `no_bg_gate` need not collapse
   consolidated `ep` -- correct, because `no_bg_gate` is a WM-only
   helper in v2. CONCLUSION: collapses the consolidated WM readout ->
   satisfies the WM-helper duty.

### 4.8 Decisive conclusion: (a) realizes the corrected v2 partition in the remote regime

By the per-lesion mechanisms above, the remote regime realizes the
corrected v2 partition for ALL SEVEN lesions: the three SHARED lesions
(`no_binding`, `no_shared_clock`, `no_hippo_store`) and the
HELPER_BOTH lesion (`no_neuromod_timing`) each collapse BOTH
consolidated readouts; the two WM-helpers (`no_bg_gate`,
`no_cls_replay`) each collapse the consolidated WM readout; the EP-helper
(`no_sequencing`) collapses the consolidated EP readout. Crucially,
`no_cls_replay` GENUINELY collapses the consolidated WM readout under
strict-silence (nothing consolidated -> nothing to recall once the
hippocampus is silenced), so the corrected v2 WM-helper duty -- the duty
that was UNSATISFIABLE-by-construction on the recent-memory probe -- is
satisfiable AND biologically correct in this regime. This is a
mechanistic argument from the validated strict-silence biology, NOT a
configuration claim and NOT derived from "what makes a candidate pass".
The conclusion is (a): the remote regime realizes the corrected v2
partition, decisively, at the mechanism level, for every lesion.

### 4.9 The honest negative-by-construction fallback (precise cause, NO partition edit)

The single mechanism on which conclusion (a) is empirically contingent
(and which the falsify-first exists to test FIRST) is consolidated-order
transfer for the `ep` readout. The project's validated Phase-1.3 result
proved consolidation transfers ORDER-AGNOSTIC semantic/role binding into
cortex under strict-silence; it did not separately validate that SERIAL
ORDER survives consolidation. If the joint minimal-load falsify-first
shows full-mode consolidated `ep` does NOT clear the science bar at
N = 2 (while consolidated `wm` does, and `no_cls_replay` collapses `wm`),
the precise cause is: CLS systems consolidation, by design, builds an
order-INVARIANT neocortical schema (McClelland 1995; Buzsaki 2013), so a
purely consolidated episodic-ORDER readout is structurally
under-served -- the same specificity-versus-generalization trade-off the
distinct-pathways finding localized, now relocated to serial order in
the consolidated regime. That is an honest negative-by-construction,
propagated WITHOUT spin, with this exact cause stated. The response is
the NEXT catalog-identified factorization (for example: a consolidated
schema serving as a generalizable PRIOR while a complementary, still
biology-faithful structure carries consolidated order -- specified only
if and when reached) -- pursued autonomously, NO hand-back, NO
configuration-cranking, and explicitly NO partition edit. The single
biologically-cited correction already made (the v2 module) was the only
permitted partition move; a second is forbidden as goalpost-moving.

The analysis is decisive, not optimistic: conclusion (a) holds at the
mechanism level for all seven lesions GIVEN consolidated-order transfer;
that one empirical contingency is isolated, named, and made the cheapest
falsify-first; its failure has a pre-stated precise cause and a
pre-committed non-partition-editing response.

---

## 5. Acceptance gate, honesty ceiling, pre-committed bound

### 5.1 The SAME corrected frozen v2 acceptance gate, unchanged

Acceptance is decided solely by `integrated_loop_verdict_v2` in
`research/runners/integrated_loop_core_v2.py` (`36a7975`),
byte-unchanged: instrument-validity FIRST (fail-closed; malformed /
non-numeric / unorderable input -> VOID, never an exception); the
pre-registered ladder `_ILV2_LADDER = (2,4,8)`; `_ILV2_MIN_SEEDS = 3`;
`_ILV2_V1_MIN = 0.90`; `_ILV2_SCI_MIN = 0.80`; `_ILV2_LESION_MAX = 0.40`;
`_ILV2_SCALE_TOL = 0.10`; the frozen partition of Section 4; "cannot
conclude" (VOID) strictly distinct from "fails" (FAIL). No bar is
softened; the partition is not edited; the original frozen core is not
imported and not edited (its VOID stands as the honest record).

### 5.2 Honesty ceiling (restated, binding, stated BEFORE any build/run)

1. The load-bearing, scale-confident scientific result of this entire
   line is the THRICE-CONVERGENT FALSIFICATION of the original
   pre-registered necessity prediction, EXTENDED by the fourth,
   regime-level structural characterization (the necessity question is
   well-posed ONLY in the remote/consolidated regime). This result is
   independent of any future build.
2. A clean, scale-confident, validated PASS was only ever obtainable
   against the ORIGINAL frozen instrument, which is now known
   unobtainable from this architectural line because the original
   prediction it encodes was falsified. This is the honest terminus of
   the "validated-PASS against the original pre-registration" goal for
   this line and is not a deficiency to be patched around.
3. Therefore a PASS of this remote-regime instrument against the
   corrected v2 module is explicitly NOT claimable as the
   scale-confident validated deliverable. At most it is
   "consistent-with the corrected (biologically-revised) necessity
   structure in the remote/consolidated regime" -- a weak positive whose
   strength rests on the single biologically-cited partition correction
   AND on the regime choice; it MUST always be reported with this exact
   limitation, never spun, never as a validated success.
4. A VOID or FAIL against the corrected v2 module IS a strong,
   unambiguous, informative negative (a faithful architecture cannot
   realize even the biologically-corrected necessity structure in the
   regime where it is well-posed) and is propagated as such.
5. No further partition edit is permitted, ever, on this line. The one
   biologically-cited correction was the single pre-committed move; a
   second would be unambiguous goalpost-moving and is forbidden.
6. Conversational / compositional / fluent-language / large-language-
   model capability is NOT claimed and is NOT in scope. All
   previously-validated capabilities (trustworthy grounded memory,
   no-confabulation abstention, simple generation, no catastrophic
   forgetting) are intact and unaffected; the no-confabulation gate, the
   original frozen verdict, and the corrected frozen v2 module are all
   byte-unchanged.

### 5.3 Pre-committed bound (restated; stated in advance so no outcome can be rationalized)

A faithful remote-regime build (Design B; Design C as the pre-described
in-architecture escalation) evaluated against the UNCHANGED corrected
frozen v2 module that reaches "cannot conclude" (VOID) or "fails" the
corrected partition is an honest negative. It is surfaced honestly with
its precise, GPU-measured structural cause -- not a configuration
iteration, not spin, not a hand-back, not a declare-globally-unfit. The
next step is then the NEXT catalog-identified integration factorization,
pursued autonomously with the SAME adversarial and anti-cheat discipline
and the SAME (v2-module) frozen acceptance, with NO further partition
edits. This bound is fixed in advance.

---

## 6. Honest ceiling (closing, never overstated)

This design does NOT claim conversational capability, compositional
memory, fluent language, or a large language model, and does NOT claim a
validated PASS. The genuine, durable result of this line remains the
four-times-convergent structural characterization of why and where the
integrated-loop necessity test is and is not well-posed (recent memory =
consolidation not necessary by CLS theory; the question is well-posed
only in the remote/consolidated regime). This document specifies the
single, biology-faithful instrument in which the corrected v2
consolidation-necessity duty is BOTH biologically correct AND
satisfiable -- reusing the project's own validated Phase-1.3
strict-silence protocol byte-unchanged -- so that a future faithful
build can be honestly tested against the unchanged corrected frozen
acceptance gate, with the honesty ceiling and the pre-committed
non-partition-editing bound fixed in advance.

---

## Files / evidence

- The load-bearing fourth finding (this design's basis):
  `research/findings/2026-05-19-FOURTH-convergent-structural-finding-necessity-instrument-probes-recent-memory-where-consolidation-is-not-needed.md`.
- Distinct-pathways honest negative (encode-order dissolved;
  consolidation-genericization localized):
  `research/findings/2026-05-19-distinct-pathways-honest-negative-episodic-contradiction-DISSOLVED-WM-blocked-by-consolidation-genericization.md`.
- Thrice-convergent falsification + honesty ceiling:
  `research/findings/2026-05-19-THIRD-convergent-signal-original-necessity-hypothesis-falsified-catalog-grounded-correction.md`,
  `research/findings/2026-05-19-CONTROLLER-precommitted-honesty-ceiling-on-the-corrected-module-PASS.md`.
- Validated Phase-1.3 strict-silence / hippocampus-OFF protocol (reused
  byte-unchanged):
  `research/runners/consolidation_trainer.py` (`run_concept_replay_phase`),
  `research/runners/consolidation_eval.py` (`evaluate_with_hippo_off`,
  `HIPPO_REGIONS`, the `-2000 pA` strict configuration);
  `research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`,
  `research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md`,
  `research/findings/2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md`.
- Corrected frozen necessity module (UNCHANGED; sole verdict authority):
  `research/runners/integrated_loop_core_v2.py` (`36a7975`).
- Original frozen verdict (UNCHANGED; never imported; its "cannot
  conclude" preserved as the honest record):
  `research/runners/integrated_loop_core.py` (`2048750`).
- Online/offline pathways reused (online ENCODE + engram WRITE
  byte-identical; the genuinely-distinct online trisynaptic episodic
  path + the separate offline Phase-1.3 consolidation + per-trial
  controller + engram fan-out):
  `research/runners/integrated_loop_gate.py` (`b4a8106`).
- 16-pool substrate + homeostasis + non-zero-init:
  `research/runners/text_minimal_isolation.py`.
- No-confabulation moat (byte-unchanged):
  `research/runners/abstention_gate.py`.
