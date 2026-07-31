---
type: plan
status: live
date: 2026-05-19
---

# Corrected-necessity module + distinct-readout-pathways implementation plan

> **For Claude:** REQUIRED SUB-SKILL: use superpowers:executing-plans to
> implement this plan task-by-task. Standing autonomy applies: one fresh
> subagent per task; strict failing-test -> minimal-implementation ->
> run -> commit; controller verifies every commit leaves the protected
> set byte-unchanged; honest propagation of every outcome; iterate
> following the reference catalog on any non-success; no hand-back. This
> plan is plan-only; it authorizes no GPU run and no code by itself.

**Goal.** Build, in the correct order, (1) a NEW, separately
pre-registered and separately frozen, catalog-grounded necessity
instrument that corrects exactly one falsified element of the original
pre-registered necessity hypothesis, (2) a dedicated adversarial review
of that new instrument whose primary mandate is goalpost-move
detection, and only if that review clears, (3) a biology-faithful
distinct-readout-pathways candidate exercised against the new
instrument with a falsify-first that probes the full science mode
jointly.

**Why this plan exists (the evidence chain, not re-litigated here).**
Three faithful, GPU-verified, honestly-propagated negatives converged
on one conclusion: the original pre-registered necessity partition's
assignment of the consolidation/replay lesion (`no_cls_replay`) to the
episodic-helper role is a falsified prediction. The decisive
conclusion is conclusion (b) of the approved design
`docs/plans/2026-05-19-distinct-readout-pathways-architecture-design.md`
(commit `72e359a`): a biology-faithful architecture provably cannot
satisfy the original frozen `no_cls_replay` -> episodic-helper duty,
because biologically episodic-sequence order is a property of the
ONLINE hippocampal trisynaptic store, not of the offline
consolidation/replay system. The decisive finding and the mandatory
anti-goalpost-move safeguards are
`research/findings/2026-05-19-THIRD-convergent-signal-original-necessity-hypothesis-falsified-catalog-grounded-correction.md`.
Signals 1 and 2 are
`research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-between-validated-concept-binding-and-episodic-store.md`
and
`research/findings/2026-05-19-phase-factored-VOID-by-construction-and-the-twice-convergent-necessity-partition-finding.md`.
These are NOT re-opened by this plan; they are its foundation.

**The original frozen verdict is permanent and is NEVER edited.**
`research/runners/integrated_loop_core.py` (commit `2048750`) and its
16-case test `tests/test_integrated_loop_core.py` are FROZEN. Their
prior "cannot conclude" (VOID) stands permanently as the honest
scientific record that the original pre-registered prediction was
falsified. This plan creates a NEW module beside it; it does not
supersede the original's RECORD, only the necessity HYPOTHESIS, and
only by the single biologically-cited correction below.

**Architecture being planned (from the approved design, not
re-derived).** The episodic-order readout is served by the
order-PRESERVING hippocampal trisynaptic CA3->CA1 pattern-completion
path (the byte-unchanged verified-foundation hippocampus recall;
reference-catalog entries D.03 trisynaptic pathway, D.12 pattern
separation, D.13 pattern completion; validated by
`research/runners/validate_trisynaptic_loop.py`). The
concept/working-memory readout is served by the order-INVARIANT
neocortical schema built by the validated Phase-1.3 offline
shuffled-replay complementary-learning-systems consolidation
(`research/runners/consolidation_trainer.py`; McClelland 1995;
Buzsaki 2013). The two readouts are genuinely DISTINCT physical
pathways that share only the single online engram write then diverge.
Net-new is ONLY (a) a per-trial controller that sequences existing
validated calls and (b) the fan-out wiring routing the one online
engram tag into the two distinct pathways. No new learning rule. No
automatic differentiation.

## Terms (defined once; used consistently)

- **Original frozen verdict / original module.**
  `research/runners/integrated_loop_core.py` at commit `2048750`.
  FROZEN. Never edited. Its VOID is the honest record.
- **New module / corrected-necessity module.** The NEW file
  `research/runners/integrated_loop_core_v2.py` created by Task 1.
  Separately pre-registered; frozen on creation; never tuned to a
  result.
- **Episodic readout (`ep`).** The recovered presentation order of the
  bound role-filler pairs, read from the order-PRESERVING online
  hippocampal trisynaptic CA3->CA1 pattern-completion path. Accuracy
  against the true online encode order.
- **Working-memory / concept readout (`wm`).** The role-selective
  filler, read from the order-INVARIANT neocortical concept/schema
  representation built by offline complementary-learning-systems
  consolidation. Emitted only if the byte-unchanged no-confabulation
  gate passes; otherwise the loop abstains.
- **Lesion.** The full loop minus EXACTLY one system, consuming the
  SAME random draws in the SAME order as `full`, with only the
  lesioned system's effect removed.
- **VOID ("cannot conclude").** The instrument cannot draw a science
  conclusion (soundness or discrimination defect, or malformed input).
  Strictly distinct from FAIL.
- **FAIL.** The instrument is sound and discriminating, but the loop
  does not perform the capability at the required bar.
- **Goalpost-move.** Any partition or bar change motivated by what
  makes a candidate architecture pass, rather than by independently
  cited biology. Forbidden.
- **The moat.** `research/runners/abstention_gate.py` (`gate`,
  `DEFAULT_THRESHOLD = 650.0`). Byte-unchanged throughout.

## The single biologically-cited correction (the ONLY partition change)

Original frozen partition (read exactly as written in the frozen
module; NOT edited):

- `_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")`
- `_HELPER_WM = ("no_bg_gate",)`
- `_HELPER_EP = ("no_sequencing", "no_cls_replay")`
- `_HELPER_BOTH = ("no_neuromod_timing",)`

Corrected partition in the NEW module (EXACTLY one membership moves:
`no_cls_replay` leaves the episodic-helper set and joins the
working-memory-helper set; every other membership and every numeric
bar is byte-identical to the original, with the original a-priori
justifications copied verbatim):

- `_ILV2_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")`
  -- UNCHANGED. Each is upstream of the single shared online engram
  write on which BOTH distinct pathways depend; removing any collapses
  both. Same a-priori justification as the original module's `_SHARED`.
- `_ILV2_HELPER_WM = ("no_bg_gate", "no_cls_replay")` -- `no_bg_gate`
  UNCHANGED (selective slot gating is necessary for the role-selective
  filler). `no_cls_replay` CORRECTED into this set: offline
  complementary-learning-systems consolidation builds the
  order-invariant NEOCORTICAL concept/schema representation the
  working-memory readout reads (reference-catalog CLS basis: McClelland
  1995; Buzsaki 2013; the project's validated Phase-1.3 consolidation,
  3/3 strict anti-cheat multi-seed). This is the single
  biologically-cited re-assignment.
- `_ILV2_HELPER_EP = ("no_sequencing",)` -- `no_sequencing` UNCHANGED
  (the SHIFT-across-theta is what writes the serial order the
  trisynaptic completion recovers; removing it makes the recovered
  order degenerate). `no_cls_replay` is REMOVED from this set: this
  membership was the single falsified element (episodic-sequence order
  is a property of the ONLINE trisynaptic store -- catalog D.03/D.12/
  D.13; the project's validated trisynaptic loop -- not of the
  consolidation/replay system).
- `_ILV2_HELPER_BOTH = ("no_neuromod_timing",)` -- UNCHANGED. Timed
  plasticity gates the whole loop consistently; removing it corrupts
  both the online trisynaptic write and the offline consolidation.
  Same a-priori justification as the original module's `_HELPER_BOTH`.

Every numeric bar in the NEW module is verbatim-equal to the original:
`_ILV2_LADDER = (2, 4, 8)`, `_ILV2_V1_MIN = 0.90`,
`_ILV2_SCI_MIN = 0.80`, `_ILV2_LESION_MAX = 0.40`,
`_ILV2_SCALE_TOL = 0.10`, `_ILV2_MIN_SEEDS = 3`, with the original
module's verbatim a-priori justification docstrings copied. The bar
values were NEVER the falsified element -- only the `no_cls_replay`
partition membership was.

## Reuse-by-import only (the protected set -- byte-unchanged)

The candidate runner imports and composes these; it does NOT modify,
copy-edit, or re-implement any of them. The controller verifies (per
task, and across the whole branch) that every path below is byte-empty
in the commit-scoped `git diff` AND across `git diff <branch-base>..HEAD`:

- `research/runners/integrated_loop_core.py` +
  `tests/test_integrated_loop_core.py` (the ORIGINAL frozen verdict;
  byte-unchanged forever; its VOID preserved).
- `research/runners/abstention_gate.py` +
  `tests/test_abstention_gate.py` (the moat; MUST stay green and
  byte-identical the entire build; `DEFAULT_THRESHOLD = 650.0`).
- every existing frozen verdict module: `research/runners/
  compose_bridge_core.py`, `compose_bind_core.py`, `td_critic_core.py`,
  `dendritic_fair_core.py`, `constrained_decode_core.py`,
  `q2r_core.py`, and every other `*_core.py`.
- every existing gate that pairs with the above:
  `research/runners/constrained_decode_gate.py`, `q2r_gate.py`,
  `compose_bridge_gate.py`, `engram_bootstrap_gate.py`.
- `research/runners/text_minimal_isolation.py`
  (`build_biological_brain_regions` REUSED UNMODIFIED -- the
  trisynaptic store + the 16-pool concept substrate + non-zero-init).
- `research/runners/consolidation_trainer.py` (the validated
  Phase-1.3 complementary-learning-systems consolidation:
  `set_awake_gates`, `set_sleep_gates`, `run_concept_replay_phase`
  with `randomize_order=True`, `run_swr_replay_phase`,
  `freeze_all_gates`; REUSED UNMODIFIED).
- `research/runners/validate_trisynaptic_loop.py` (the validated
  D.12/D.13 trisynaptic asset; REUSED UNMODIFIED as the documented
  episodic-recall idiom reference).
- `research/runners/g11_bg_runner.py` (`build_bg_brain_regions`
  REUSED UNMODIFIED).
- the validated simulator modules: `sim/bridge.py` (the engram-tagging
  API: `start_engram_recording`, `commit_engram_tag`, `stimulate_tag`,
  `clear_tag_drive`, `delete_engram_tag`), `sim/kernels.py`,
  `sim/neuromodulators.py`, `sim/train_checkpoint.py`,
  `sim/backend.py`, `sim/regions.py`, `sim/text_embeddings.py`,
  `sim/td_value_critic.py`, `sim/compose_temporal_bind.py`,
  `sim/dendritic_plasticity.py`.
- `research/runners/grounded_decode.py`, `sim/grounded_decode.py`,
  `research/runners/generator_g_core.py`.

The only files this plan's build creates or modifies:
`research/runners/integrated_loop_core_v2.py`,
`tests/test_integrated_loop_core_v2.py`, and an EXTENSION of the
existing `research/runners/integrated_loop_gate.py` (the faithful
phase-factored runner; this build extends it with the distinct-pathways
controller mode, it does not rewrite the protected modules it imports)
plus its existing test file `tests/test_integrated_loop_gate.py`. Plus
the Task-7 propagation artifacts (a findings doc, a capability-status
pillar edit, a git commit). The original `integrated_loop_core.py` is
NOT in this list -- it is never edited.

## No new automatic differentiation / training anywhere

Every learning update in this build is a REUSED validated local rule
(the native eligibility-trace reward path with the validated
temporal-credit values; the validated spike-timing plasticity; the
validated Phase-1.3 consolidation replay). No `torch`, no
`.backward()`, no autograd, no gradient-descent objective is
introduced in any shipped path. Tasks 6 (adversarial review of the
runner) and 7 (no-harm) both explicitly assert this. The NEW verdict
module imports only the standard library and typing.

## Pre-committed bound (stated verbatim, NOW, before any run)

A faithful distinct-readout-pathways build (Candidate 1 of the
approved design, with the design's Candidates 2/3 as the pre-described
in-architecture escalations) evaluated against the NEW separately-
frozen catalog-grounded necessity module that ALSO reaches "cannot
conclude" (VOID) or fails the corrected partition is an honest
negative. It is surfaced honestly with its precise, GPU-measured
structural cause -- not a configuration iteration, not spin, not a
hand-back, not a declare-globally-unfit. The next step is then the
next catalog-identified integration factorization, pursued
autonomously with the SAME adversarial and anti-cheat discipline and
the SAME (new-module) frozen acceptance. **No further partition edits:
exactly one biologically-cited correction is permitted in the NEW
module; a second partition change would itself be goalpost-moving and
is forbidden.** This bound is stated in advance so the next outcome
cannot be rationalized after the fact. The original
`integrated_loop_core.py` is not the acceptance instrument for the
distinct-pathways architecture and is not edited; its VOID is the
honest record. The moat is byte-unchanged; every drilled
working-memory binding must clear it or the readout abstains.

---

## Task 1 (FIRST, load-bearing): the NEW separately-pre-registered + frozen catalog-grounded necessity module

**Files:**
- Create: `research/runners/integrated_loop_core_v2.py`
- Create test: `tests/test_integrated_loop_core_v2.py`

**Context.** This is a pure, deterministic, fail-closed verdict
module. It mirrors the ORIGINAL frozen module's discipline EXACTLY:
pure standard library + typing only; no `torch`, no autograd;
instrument-validity checked FIRST; fail-closed; fixed bars
pre-registered HERE and NEVER tuned to a result; "cannot conclude"
(VOID) strictly distinct from "fails" (FAIL); malformed / non-numeric
/ unorderable input -> VOID, never an exception; ASCII only; owns its
OWN frozen bars; imports no other verdict module (it does NOT import
`integrated_loop_core`). It is a NEW file -- the original
`integrated_loop_core.py` is NEVER edited; its VOID stands permanently
as the honest record that the original pre-registered prediction was
falsified.

The ONLY difference from the original module is the single
biologically-cited partition correction in the section "The single
biologically-cited correction" above: `no_cls_replay` moves from the
episodic-helper set to the working-memory-helper set. Every numeric
bar is verbatim-equal to the original, with the original a-priori
justification docstrings copied. The new module's verdict precedence
is IDENTICAL to the original's (instrument-validity FIRST ->
soundness/discrimination defect => VOID; else science precedence
SCALE-CONFIDENT-PASS / WORKS-SMALL-NO-SCALE-CONFIDENCE / FAIL), with
the ONLY difference being the corrected partition.

**Module specification (for verbatim transcription by the implementer):**

The implementer transcribes the ORIGINAL frozen
`research/runners/integrated_loop_core.py` body EXACTLY, with these
mechanical, fully-specified edits and NO others:

1. **Symbol rename only (no logic change):** every `_IL_*` constant
   name becomes `_ILV2_*` (`_ILV2_LADDER`, `_ILV2_V1_MIN`,
   `_ILV2_SCI_MIN`, `_ILV2_LESION_MAX`, `_ILV2_SCALE_TOL`,
   `_ILV2_MIN_SEEDS`); the partition tuples become `_ILV2_SHARED`,
   `_ILV2_HELPER_WM`, `_ILV2_HELPER_EP`, `_ILV2_HELPER_BOTH`,
   `_ILV2_ALL_LESIONS`; the public entry point becomes
   `integrated_loop_verdict_v2`. The renamed module is self-contained
   and owns its own frozen constants. The bar VALUES are byte-identical
   to the original.
2. **The single partition correction (the only semantic change):**
   - `_ILV2_SHARED = ("no_binding", "no_shared_clock",
     "no_hippo_store")` -- identical tuple to the original `_SHARED`.
   - `_ILV2_HELPER_WM = ("no_bg_gate", "no_cls_replay")` -- the
     original `_HELPER_WM` was `("no_bg_gate",)`; `no_cls_replay` is
     added here (the single biologically-cited re-assignment).
   - `_ILV2_HELPER_EP = ("no_sequencing",)` -- the original
     `_HELPER_EP` was `("no_sequencing", "no_cls_replay")`;
     `no_cls_replay` is removed here (the falsified membership).
   - `_ILV2_HELPER_BOTH = ("no_neuromod_timing",)` -- identical tuple
     to the original `_HELPER_BOTH`.
   - `_ILV2_ALL_LESIONS = _ILV2_SHARED + _ILV2_HELPER_WM +
     _ILV2_HELPER_EP + _ILV2_HELPER_BOTH` -- the SAME 8 lesion names
     as the original `_ALL_LESIONS`, only repartitioned (the set of
     lesion names is unchanged; the set membership of exactly one name
     changed).
3. **Bar values + their a-priori justification docstrings:** copy the
   ORIGINAL module's module-level docstring "A-priori justification of
   every frozen value" block verbatim (the bar values are unchanged so
   the original justifications still hold), then ADD a clearly
   delimited new docstring section titled "Catalog-grounded
   correction (the single change vs the original frozen module)" that
   states (i) the biological basis for the single change: episodic-
   sequence order is a property of the ONLINE hippocampal trisynaptic
   pattern-completion path (reference-catalog D.03 trisynaptic
   pathway; D.12 pattern separation, Kandel 6e Ch 54 pp 1357-1360;
   D.13 pattern completion, Kandel 6e Ch 54 pp 1342, 1360-1361, Marr
   1971; the project's validated `validate_trisynaptic_loop.py`),
   while the order-invariant neocortical concept/schema the
   working-memory readout reads is built by the offline
   complementary-learning-systems consolidation system (McClelland
   1995; Buzsaki 2013; the project's validated Phase-1.3
   consolidation, 3/3 strict anti-cheat multi-seed) -- therefore the
   consolidation/replay lesion `no_cls_replay` is necessary for the
   working-memory readout, NOT for the episodic readout; (ii) an
   explicit statement that this module SUPERSEDES the original's
   necessity HYPOTHESIS (not its RECORD) because the original's
   `no_cls_replay`-in-episodic-helper membership was a FALSIFIED
   pre-registered prediction (three convergent faithful negatives;
   the original module is byte-unchanged and its VOID is preserved as
   the honest record); (iii) an explicit statement that this single
   partition change was pre-committed in writing -- in pushed commits
   and the cited findings -- BEFORE the outcome that motivated it, and
   that it is derived from cited biology and is implied independently
   by all THREE convergent signals, NOT by what makes any candidate
   architecture pass; (iv) the pre-committed bound restated verbatim
   from this plan's "Pre-committed bound" section, including "no
   further partition edits".
4. **No other change.** `_num`, `_pair`, the verdict function body,
   the precedence, the VOID/FAIL distinction, the malformed-input
   handling, the ladder/seed guards, the science-bar/monotone/top-ok
   logic are transcribed BYTE-FOR-BYTE from the original (only the
   `_IL_*`/`_ILV2_*` symbol names and the four partition tuples
   differ). The instrument-validity-FIRST ordering is preserved
   exactly: the per-lesion discrimination check still iterates
   `_ILV2_ALL_LESIONS` and applies "SHARED or HELPER_BOTH -> both must
   collapse; HELPER_WM -> wm must collapse; else (HELPER_EP) -> ep
   must collapse", which now correctly requires `no_cls_replay` to
   collapse the WORKING-MEMORY readout (it is in `_ILV2_HELPER_WM`)
   and no longer requires it to collapse the episodic readout.

**Step 1: Write the failing test.**

Create `tests/test_integrated_loop_core_v2.py` mirroring the
original's 16-case matrix (`tests/test_integrated_loop_core.py`),
adapted to `integrated_loop_verdict_v2` and the `_ILV2_*` symbols,
with these REQUIRED additions so the matrix is >= 12 cases and pins
the discipline AND the goalpost-move guards:

- `test_01_scale_confident_pass` -- a fully-good rung ladder ->
  GATE PASS / SCALE-CONFIDENT-PASS. The rung helper must place
  `no_cls_replay` as a WORKING-MEMORY collapse (wm <= 0.40, ep high),
  reflecting the corrected partition.
- `test_02_works_small_trend_break` -- WORKS-SMALL-NO-SCALE-CONFIDENCE.
- `test_03_works_small_top_below_floor` -- WORKS-SMALL-NO-SCALE-
  CONFIDENCE.
- `test_04_shared_lesion_does_not_collapse_wm_is_void` -- VOID,
  `instrument_valid is False`.
- `test_05_shared_lesion_collapses_wm_but_not_ep_is_void` -- VOID
  (the non-separability signature for SHARED is unchanged).
- `test_06_helper_wm_no_bg_gate_does_not_collapse_wm_is_void` --
  `no_bg_gate` not collapsing wm -> VOID.
- `test_07_helper_wm_no_cls_replay_does_not_collapse_wm_is_void` --
  THE CORRECTED-MEMBERSHIP CASE: `no_cls_replay` with wm high (not
  collapsed) -> VOID. (Under the original module this same numeric
  record was scored against the episodic readout; this test pins the
  corrected behavior.)
- `test_08_helper_ep_no_sequencing_does_not_collapse_ep_is_void` --
  `no_sequencing` not collapsing ep -> VOID.
- `test_09_no_cls_replay_collapsing_only_ep_not_wm_is_void` --
  EXPLICIT GUARD: a record where `no_cls_replay` collapses ONLY the
  episodic readout (ep <= 0.40) and leaves working-memory intact
  (wm high) -> VOID (because in the corrected partition
  `no_cls_replay` is a WORKING-MEMORY helper and MUST collapse wm; an
  ep-only collapse is now non-discriminating). This is the case that
  would have been a (spurious) "satisfied" under the original
  partition and is the precise behavioral fingerprint of the single
  correction.
- `test_10_helper_both_collapses_only_one_is_void` --
  `no_neuromod_timing` collapsing only one readout -> VOID.
- `test_11_v1_unmet_is_void_not_fail` -- soundness unmet -> VOID,
  `instrument_valid is False`.
- `test_12_sound_discriminating_but_science_below_bar_is_fail` ->
  GATE FAIL, `instrument_valid is True`, classification FAIL.
- `test_13_ladder_mismatch_is_void`.
- `test_14_non_numeric_and_nan_is_void_not_raise`.
- `test_15_too_few_seeds_is_void`.
- `test_16_malformed_top_level_is_void_not_raise`
  (`None`, `[]`, `"garbage"`, `[{"no": "N"}]`).
- `test_17_threshold_tamper_pins_VERBATIM_EQUAL_TO_ORIGINAL` -- imports
  BOTH the original `_IL_*` constants from
  `research.runners.integrated_loop_core` AND the new `_ILV2_*`
  constants, and asserts each new bar is byte-equal to its original
  counterpart: `_ILV2_LADDER == _IL_LADDER == (2, 4, 8)`,
  `_ILV2_V1_MIN == _IL_V1_MIN == 0.90`,
  `_ILV2_SCI_MIN == _IL_SCI_MIN == 0.80`,
  `_ILV2_LESION_MAX == _IL_LESION_MAX == 0.40`,
  `_ILV2_SCALE_TOL == _IL_SCALE_TOL == 0.10`,
  `_ILV2_MIN_SEEDS == _IL_MIN_SEEDS == 3`. This test FAILS if any bar
  drifts from the original.
- `test_18_partition_has_exactly_one_documented_change_vs_original` --
  imports BOTH partitions and asserts: (a) `_ILV2_SHARED == _SHARED`;
  (b) `_ILV2_HELPER_BOTH == _HELPER_BOTH`; (c) the SET of all lesion
  names is identical: `set(_ILV2_ALL_LESIONS) == set(_IL... )` where
  the original all-lesions set is reconstructed from
  `_SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH`; (d) EXACTLY the
  name `"no_cls_replay"` changed helper set: it is in
  `_ILV2_HELPER_WM` and NOT in `_ILV2_HELPER_EP`, while in the
  original it is in `_HELPER_EP` and NOT in `_HELPER_WM`; (e) the
  symmetric difference of `(set(_HELPER_WM) ^ set(_ILV2_HELPER_WM))`
  and of `(set(_HELPER_EP) ^ set(_ILV2_HELPER_EP))` each equals
  exactly `{"no_cls_replay"}`; (f) no other lesion name changed any
  membership. This test FAILS if the partition has more than the
  single documented change.

Run: `pytest tests/test_integrated_loop_core_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named
'research.runners.integrated_loop_core_v2'`.

**Step 2: Write the minimal implementation.**

Transcribe the original frozen module body with EXACTLY the mechanical
edits 1-4 specified above and no others.

Run: `pytest tests/test_integrated_loop_core_v2.py -v`
Expected: all cases pass (>= 18 listed).

**Step 3: Commit.**

```
git add research/runners/integrated_loop_core_v2.py tests/test_integrated_loop_core_v2.py
git commit -m "feat: corrected-necessity verdict module (NEW, frozen; single biologically-cited no_cls_replay re-assignment; original frozen module byte-unchanged + its VOID preserved)"
```

**Controller verification.** The commit-scoped `git diff` touches only
the two new files. `research/runners/integrated_loop_core.py` and
`tests/test_integrated_loop_core.py` are byte-empty in the diff (the
original frozen module is NOT edited). The whole protected set is
byte-empty. `integrated_loop_core_v2.py` imports only `math` and
`typing`; it does not import `integrated_loop_core` or any `*_core`
or any `sim`/runner module. `pytest tests/test_abstention_gate.py -q`
is still green and `abstention_gate.py` is byte-unchanged.
`pytest tests/test_integrated_loop_core.py -q` is still 16/16 (the
original is unaffected).

---

## Task 2: DEDICATED ADVERSARIAL REVIEW of the NEW module BEFORE any architecture work (primary mandate = goalpost-move detection)

**Files:** none modified by the reviewer subagent. Strengthen-only
fixes (if any, and only to restore byte-equality with the original or
to restore discipline) are applied by a follow-up implementer subagent
and re-reviewed; the single documented partition change and every bar
stay exactly as Task 1 specified.

**Context.** Dispatch a fresh subagent as a dedicated adversarial
reviewer whose EXPLICIT, PRIMARY charter is goalpost-move detection.
The charter question, stated verbatim to the reviewer:

> "Is `research/runners/integrated_loop_core_v2.py` a LEGITIMATE
> catalog-derived correction of a falsified pre-registered prediction,
> or a rationalized repartition engineered to make the next candidate
> (the distinct-readout-pathways architecture) pass?"

**The reviewer MUST verify, with file:line evidence, ALL of the
following; ANY failure makes the verdict "goalpost-move" and BLOCKS
all downstream tasks:**

1. **Every numeric bar is byte-equal to the original, with the
   original a-priori justifications.** `_ILV2_LADDER/V1_MIN/SCI_MIN/
   LESION_MAX/SCALE_TOL/MIN_SEEDS` each equal the corresponding
   `_IL_*` in the byte-unchanged original module; the original
   a-priori justification docstring block is copied verbatim.
2. **EXACTLY ONE partition change.** `_ILV2_SHARED == _SHARED`;
   `_ILV2_HELPER_BOTH == _HELPER_BOTH`; the full set of 8 lesion names
   is unchanged; exactly the name `no_cls_replay` moved, from
   `_HELPER_EP` to `_HELPER_WM`; no other membership changed.
   (Confirm `tests/test_integrated_loop_core_v2.py::
   test_18_partition_has_exactly_one_documented_change_vs_original`
   actually enforces this and passes.)
3. **The change is independently implied by the cited biology, NOT by
   what makes the candidate pass.** The reviewer independently traces
   the cited reference-catalog entries: episodic-sequence order is a
   property of the ONLINE trisynaptic pattern-completion path
   (D.03/D.12/D.13; the validated `validate_trisynaptic_loop.py`);
   the order-invariant neocortical concept/schema is built by the
   offline complementary-learning-systems consolidation (the validated
   Phase-1.3 `consolidation_trainer.py`). The reviewer confirms the
   same single correction is implied INDEPENDENTLY by all THREE
   convergent signals (signals 1, 2, and 3 as documented in the cited
   findings) and is NOT derivable from "this is the membership that
   would let the distinct-pathways runner score a PASS".
4. **The original module is byte-unchanged and its VOID is
   preserved.** `git diff <branch-base>..HEAD --
   research/runners/integrated_loop_core.py
   tests/test_integrated_loop_core.py` is empty;
   `pytest tests/test_integrated_loop_core.py -q` is 16/16.
5. **The new module's discipline is identical to the original's.**
   Instrument-validity FIRST; VOID strictly distinct from FAIL;
   fail-closed; malformed/non-numeric/unorderable -> VOID-not-raise;
   pure standard library + typing; no `torch`/autograd; imports no
   other verdict module; ASCII only; bars not derived from any
   observed number.
6. **The pre-commitment is genuine.** The docstring states the single
   change was pre-committed in writing before the outcome; the
   reviewer confirms the cited findings and pushed commits
   (`72e359a` design; the THIRD-convergent-signal finding; signals 1
   and 2) specify the condition and the response in advance, so the
   outcome could not be rationalized after the fact.

**Step:** Dispatch the reviewer subagent with the six probes above as
its explicit charter. It returns a written report with a single
top-line verdict: `LEGITIMATE-CORRECTION` or `GOALPOST-MOVE`.

- If `GOALPOST-MOVE`: **Task 2 BLOCKS all downstream tasks.** The
  corrected module is rejected; the honest outcome is propagated
  (write `research/findings/2026-05-19-corrected-necessity-module-
  rejected-as-goalpost-move.md` in plain professional language) and
  the original VOID stands as a DEEPER terminal scientific finding
  (the program could not even produce a defensible corrected
  necessity instrument without goalpost-moving; that is itself the
  honest result). No architecture is built. Stop here; bring this to
  the controller.
- If `LEGITIMATE-CORRECTION`: the controller applies any strengthen-
  only fixes (only to restore byte-equality with the original or
  restore discipline; the single documented partition change and the
  bars are not otherwise touched), re-dispatches the reviewer until
  the report has no open holes, commits any strengthen-only fix with
  a clear message (controller verifies the protected set + the
  original module stay byte-empty in the diff), then proceeds to
  Task 3.

**This gate is non-negotiable and precedes any architecture build.**
Do NOT proceed to Task 3 until the adversarial review's top-line
verdict is `LEGITIMATE-CORRECTION` with no open issues.

---

## Task 3: Grounding pin test for the distinct-pathways runner mode

**Files:**
- Extend: `tests/test_integrated_loop_gate.py` (add ONLY the
  distinct-pathways grounding pin in this task; later tasks add the
  rest).

**Context.** This is a deliberately-failing pin that turns green only
after Task 4 lands. It IS the Task-4 gate: it asserts the existing
`research/runners/integrated_loop_gate.py` exposes a
`--distinct-pathways` mode whose kill-safe `--tiny-synth` smoke runs
end-to-end on the NumPy CPU backend and writes a verdict JSON marked
TINY (never a real PASS/FAIL/VOID at toy scale), and that the runner
scores via the NEW core module `integrated_loop_verdict_v2`.

**Step 1: Write the failing test.**

```python
def test_distinct_pathways_tiny_smoke_produces_tiny_verdict(tmp_path):
    """Grounding pin: the runner's distinct-readout-pathways mode runs
    a fast --tiny-synth smoke end-to-end on the CPU backend and writes
    a verdict JSON marked TINY (never propagated at toy scale)."""
    import json, subprocess, sys
    from pathlib import Path
    out = tmp_path / "tiny_dp.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--distinct-pathways", "--tiny-synth",
         "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)
    assert out.exists()
    v = json.loads(out.read_text())
    assert "GATE" in v
    assert "TINY" in json.dumps(v)
```

Run: `pytest tests/test_integrated_loop_gate.py
-k distinct_pathways_tiny_smoke -v`
Expected: FAIL (the `--distinct-pathways` mode does not exist yet).
This is intentional and correct at Task 3.

**Step 2: Commit the red pin.**

```
git add tests/test_integrated_loop_gate.py
git commit -m "test: grounding pin for the distinct-readout-pathways runner mode (red until the mode lands)"
```

**Controller verification.** The commit-scoped `git diff` touches only
`tests/test_integrated_loop_gate.py`. The whole protected set + the
original `integrated_loop_core.py` are byte-empty in the diff. Do NOT
mark Task 3 "green" -- it is intentionally red and is the Task-4
acceptance gate.

---

## Task 4: The distinct-readout-pathways candidate runner (genuine net-new wiring -- NOT transcription)

**Files:**
- Extend: `research/runners/integrated_loop_gate.py` (add the
  `--distinct-pathways` controller mode; do NOT rewrite or copy-edit
  any imported protected module).
- Extend test: `tests/test_integrated_loop_gate.py` (the Task-3 pin
  must now go green; add a small structural test).

**Context.** This builds Candidate 1 of the approved design: two
physically distinct readout pathways that share only the single online
engram write then diverge. It mirrors the ORIGINAL plan's Task-2
discipline (`docs/plans/2026-05-18-integrated-loop-full-model-
implementation.md` Task 2) and the proven kill-safe CLI/checkpoint/
verdict scaffold of `research/runners/compose_bridge_gate.py` (backend
pin, builder via reused interfaces, per-mode episode loop, per-seed
kill-safe checkpoint, `KeyboardInterrupt` -> resumable, `--tiny-synth`
smoke whose verdict is never propagated, ASCII prints, verdict from
the frozen core -- here the NEW core). This is genuine net-new wiring,
not transcribe-a-reference.

**Reused interfaces (import byte-unchanged; do NOT modify):**

- `from research.runners.text_minimal_isolation import
  build_biological_brain_regions` -- build with
  `enable_hippocampus_consolidation=True` (the trisynaptic store
  `ec/dg/dg_pv_basket/ca3/ca1` + the `ca1 -> concept` / `ca1 ->
  language_output` consolidation pathways + the `ca3_swr_burst`
  recurrent autoassociator), `enable_dlpfc_verb=True` (the prefrontal
  working-memory slots), the validated 16-pool concept substrate, and
  `enable_nmda=True` (exactly as `compose_bridge_gate` does).
  Non-zero-init foundation preserved exactly as in the verified
  foundation.
- `from research.runners.consolidation_trainer import` the validated
  Phase-1.3 calls used UNMODIFIED: `set_awake_gates`,
  `set_sleep_gates`, `run_concept_replay_phase` (with
  `randomize_order=True`), `freeze_all_gates` (and
  `run_swr_replay_phase` if the existing runner already uses it). This
  is the order-INVARIANT concept/working-memory pathway's offline
  training. Construction/sequencing only; the module is byte-unchanged.
- The engram-tagging API on `sim/bridge.py` (byte-unchanged):
  `bridge.start_engram_recording(name)`,
  `bridge.commit_engram_tag(name, top_k=..., region_filter=[...])`,
  `bridge.stimulate_tag(name, drive_pA=...)`,
  `bridge.clear_tag_drive(name)` -- the single online relational
  binding write and the order-PRESERVING trisynaptic recall cue.
- `from research.runners.g11_bg_runner import build_bg_brain_regions`
  -- the validated basal-ganglia cascade; its disinhibition gate
  repurposed to the prefrontal/associative channel (which prefrontal
  working-memory slot updates vs holds), exactly as the original plan
  Task 2 specifies. Imported, not modified.
- `from sim.neuromodulators import NeuromodulatorConfig,
  ProductionRule, ModulatorTarget` -- the dopamine-from-reward
  modulator and the acetylcholine-style clock-gated plasticity-window
  modulator, constructed exactly as `compose_bridge_gate` does.
  Construction only.
- `from sim.kernels import fused_eligibility_trace_decay` + the bridge
  native `cp_eligibility_trace` reward path with the validated
  temporal-credit values (`_GAMMA = 0.95`, `_LAMBDA = 0.9`, as
  `compose_bridge_gate`). No new learning rule.
- `from sim.train_checkpoint import save_checkpoint, load_checkpoint,
  resume_epoch` -- kill-safe per-seed checkpoint/resume.
- `from research.runners.abstention_gate import gate,
  DEFAULT_THRESHOLD` -- the moat at the working-memory output, byte-
  unchanged. Every drilled working-memory binding must clear it or the
  loop abstains.
- `from research.runners.integrated_loop_core_v2 import
  integrated_loop_verdict_v2` -- the NEW frozen acceptance instrument
  from Task 1. (The runner does NOT import the original
  `integrated_loop_core`.)
- `from sim.text_embeddings import orthogonal_drive_pattern` -- the
  proven orthogonal role/filler code idiom.

**The net-new pieces (the ONLY new code; everything else reused by
import, byte-unchanged):**

1. **A per-trial controller** that sequences existing validated calls:
   online awake-encode (UNCHANGED from the verified foundation: one
   shared theta-gamma clock gates the basal-ganglia-selected
   prefrontal slot and times the engram write; role+filler orthogonal
   codes + teacher co-fire; the episode is written ONCE to the engram
   store in theta order, presentation-order == binding-index); then
   the EPISODIC-ORDER readout via the order-PRESERVING trisynaptic
   CA3->CA1 pattern-completion path (the verified-foundation
   hippocampus recall idiom: with weights frozen for the readout
   window, `stimulate_tag` the committed tag as the partial CA3 cue,
   the `ca3_swr_burst` recurrent autoassociator reconstructs the bound
   pattern, the per-role activity-peak order is read from CA1 and
   scored against the true online encode order -- this pathway NEVER
   touches `run_concept_replay_phase`); then the CONCEPT/
   WORKING-MEMORY pathway (`set_sleep_gates`; the committed episode
   tag is driven through the validated `run_concept_replay_phase`
   with `randomize_order=True` so the autoassociator + `ca1 ->
   concept` consolidation transfers the bound (role, filler) structure
   into the 16-pool concept layer in SHUFFLED order; `freeze_all_gates`;
   query each role, read the consolidated concept pool, emit only if
   the byte-unchanged moat passes else abstain -- this pathway NEVER
   touches the order-sensitive episodic recall). The controller
   introduces no learning logic of its own; no autograd.
2. **The fan-out wiring** connecting the single online engram tag to
   (i) the trisynaptic episodic-recall cue and (ii) the offline
   `run_concept_replay_phase` consolidation input feeding the 16-pool
   concept layer. Wiring only.

**The modes (each = the full distinct-pathways loop minus EXACTLY one
system; identical RNG draws; everything else byte-identical -- the
`compose_bridge_gate` faithfulness discipline). Each lesion is defined
so it collapses its CORRECTED-partition-required readout:**

- `full` -- the complete distinct-pathways loop.
- `v1` -- the full loop on a NO-GAP trivial single drilled bind
  (instrument soundness; the trivial drilled bijection on BOTH
  readouts; mirrors `compose_bridge_gate`'s `gap_zero`). The Science
  task (novel role-filler recombination not drilled) is used in `full`
  + every lesion, exactly as the original plan Task 2's
  pre-registration-conformance log fixed it.
- `no_binding` (SHARED) -- suppress the combinatorial binding step
  (drive role and filler but form no combined relational assembly).
  No bound assembly is written to the engram tag; the episodic
  trisynaptic cue has no bound pattern to complete AND the offline
  consolidation has no bound structure to transfer. Must collapse
  BOTH.
- `no_shared_clock` (SHARED) -- replace the ONE shared clock with TWO
  independent clocks (prefrontal and hippocampal timing
  desynchronized); everything else identical. The online write is not
  theta-ordered and role-filler co-fire is not coincident; both
  pathways descend from the single clock-timed online write. Must
  collapse BOTH.
- `no_hippo_store` (SHARED) -- skip `start/commit/stimulate` engram
  tagging (no fast relational store). The episodic pathway has no tag
  to cue AND the offline consolidation has no tag to replay. Must
  collapse BOTH.
- `no_bg_gate` (HELPER_WM) -- remove basal-ganglia selective gating
  (all prefrontal slots always open / never selectively gated). No
  clean slot, so the consolidated concept layer cannot resolve a
  role-selective filler. Must collapse the WORKING-MEMORY readout.
- `no_cls_replay` (HELPER_WM -- the CORRECTED membership) -- skip the
  offline replay/consolidation phase (no transfer into the concept/
  schema layer). The concept readout collapses (the concept layer is
  never consolidated); the episodic readout, served by the
  order-preserving ONLINE trisynaptic completion which does NOT depend
  on the consolidation system, stays intact. Must collapse the
  WORKING-MEMORY readout (and is required by the corrected partition
  to do so; it is NOT required to collapse episodic). This lesion is
  the precise behavioral fingerprint of the single correction.
- `no_sequencing` (HELPER_EP) -- the shared clock REPEATS the
  assembly instead of SHIFTING it across theta cycles (no episodic
  order written). The trisynaptic completion reconstructs an
  order-degenerate pattern; the per-role peak times carry no
  recoverable sequence. Must collapse the EPISODIC readout.
- `no_neuromod_timing` (HELPER_BOTH) -- remove the clock-gated
  plasticity-window modulator (plasticity always on, untimed). The
  online binding does not form correctly AND the offline consolidation
  cannot selectively strengthen the bound structure. Must collapse
  BOTH.

Each lesion is the full distinct-pathways loop minus exactly that one
system, consuming the SAME random draws in the SAME order as `full`
(only the lesioned system's effect removed) -- a strawman crippled
elsewhere is a Task-6 reject.

**Scale ladder + scaffold (mirrors the original plan Task 2 exactly):**

- Full run: load `N` over the frozen ladder `(2, 4, 8)`; `--seeds`
  default `[42, 43, 44, 45, 46]`; require `>= 3` seeds or print
  `NOT-RUNNABLE` and return 2.
- `--distinct-pathways --tiny-synth`: shrink the ladder to its first
  rung, shrink pools/steps/epochs/replay-budget so the smoke completes
  fast on the NumPy CPU backend; the tiny verdict is marked TINY and
  NEVER propagated (this is what makes the Task-3 pin go green).
- `os.environ.setdefault("SIM_BACKEND", "numpy")` BEFORE any sim
  import is reserved for `--tiny-synth` only; the real/decisive path
  runs on the GPU/CuPy backend (the heavier in-bridge spiking +
  consolidation replay path).
- Kill-safe: per-seed checkpoint via `save_checkpoint`; on
  `KeyboardInterrupt` flush the partial and return 130 with a
  "resumable" message; on resume, skip completed seeds. The
  consolidation replay phase is bounded by the existing replay-budget,
  controller-run, kill-safe.
- The decisive output: assemble per-rung `{"N", "n_seeds",
  "v1":{wm,ep}, "full":{wm,ep}, "lesions":{...:{wm,ep}}}` (aggregated
  mean over seeds), call `integrated_loop_verdict_v2(rungs)` from the
  NEW frozen core, write the JSON, print `GATE=... <honest-ceiling
  banner>`. Honest-ceiling banner: emergent compositional memory in a
  biology-grounded two-distinct-pathway loop ONLY -- NOT fluent
  open-ended language, NOT a large language model, NOT conversation
  solved.
- ASCII only. No `torch`, no autograd anywhere.

**Step 1: Make the Task-3 pin executable, add a structural test
(failing).**

Add to `tests/test_integrated_loop_gate.py`:

```python
def test_distinct_pathways_reuses_parts_byte_unchanged_and_new_core():
    """The distinct-pathways mode composes the validated parts by
    import, adds no autograd, scores via the NEW core (not the
    original), and never imports the original frozen core."""
    from pathlib import Path
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "run_concept_replay_phase" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "integrated_loop_verdict_v2" in src
    assert "from research.runners.integrated_loop_core import" not in src
    assert "import integrated_loop_core\n" not in src
```

Run: `pytest tests/test_integrated_loop_gate.py -v` -> the new tests
FAIL (mode missing).

**Step 2: Implement the `--distinct-pathways` controller mode** in
`research/runners/integrated_loop_gate.py` per the behavioral spec.
Reuse-by-import only; modify none of the protected set; do not import
the original `integrated_loop_core`; score via
`integrated_loop_verdict_v2`.

**Step 3: Run the smoke + structural test.**

Run: `pytest tests/test_integrated_loop_gate.py -v`
Expected: the distinct-pathways pin + the structural test PASS. Run
the tiny smoke directly once and read the JSON to confirm a verdict
object with a `GATE` field and the TINY marker.

**Step 4: Commit.**

```
git add research/runners/integrated_loop_gate.py tests/test_integrated_loop_gate.py
git commit -m "feat: distinct-readout-pathways controller mode (order-preserving trisynaptic episodic recall + order-invariant consolidated concept readout; scored by the NEW frozen core)"
```

**Controller verification (trust-but-verify).** The commit-scoped
`git diff` touches only `research/runners/integrated_loop_gate.py` and
`tests/test_integrated_loop_gate.py`. Every protected path AND
`research/runners/integrated_loop_core.py` /
`tests/test_integrated_loop_core.py` are byte-empty in the diff AND
across `git diff <branch-base>..HEAD`. The runner contains no
`import torch` / `.backward(` / autograd, does not import the original
frozen core, and scores via `integrated_loop_verdict_v2`.
`build_biological_brain_regions`, `build_bg_brain_regions`,
`run_concept_replay_phase`, the engram API are imported, not
redefined. `pytest tests/test_abstention_gate.py -q` is still green;
`pytest tests/test_integrated_loop_core.py -q` is still 16/16. The
`--tiny-synth` verdict is marked TINY.

---

## Task 5: Pre-registered FALSIFY-FIRST -- probe the FULL science mode's working-memory AND episodic readouts JOINTLY at minimal load

**Files:** none modified. This is a controller-run cheap de-risk that
produces a short evidence note appended to the eventual findings doc
(Task 7), not a code change.

**Context.** The recorded process lesson (from the phase-factored
de-risk that checked only the soundness mode and reported a
false-green) is honored here: the cheap de-risk MUST probe the FULL
science mode's working-memory AND episodic readouts JOINTLY at N=2
single-seed -- NOT the trivial-soundness (`v1`) mode alone.

**Step 1: Run the joint full-mode smoke (single seed, N=2, GPU).**

Run the `--distinct-pathways` runner restricted to the FULL science
mode at the smallest rung (N=2), single seed 42, on the GPU path
(not `--tiny-synth`), measuring on the FULL mode at N=2:

- (a) the episodic readout via the order-preserving trisynaptic
  CA3->CA1 pattern-completion pathway is still approximately 1.0 WITH
  the offline concept-consolidation phase running, AND
- (b) the consolidated concept/working-memory readout is
  role-selective above chance.

**Step 2: Pre-registered early trigger.** If EITHER (a) episodic via
the trisynaptic-completion pathway is not approximately 1.0 with the
offline concept-consolidation running, OR (b) the consolidated concept
readout is not role-selective above chance, the Task-7 escalation
fires immediately with that EXACT structural cause -- no configuration
crank. The escalation order is the approved design's pre-described
in-architecture escalations only (Candidate 2 = consolidation-gated
prefrontal reinstatement of the concept readout only, if the bare
consolidated readout is real but sub-moat; Candidate 3 = the
multi-cycle offline CLS schedule, only if Candidates 1 and 2 both show
real-but-insufficient consolidated selectivity), each its own
pre-registered re-entry, never a reflexive crank, the new
pre-committed bound in force.

**Step 3:** Record the joint-smoke numbers as a health/de-risk
evidence note (NOT a propagated result). Do not state any decisive
result from this step.

Do NOT proceed to Task 6 until the joint falsify-first has run and its
outcome (proceed, or escalate per the pre-described in-architecture
escalation) is recorded.

---

## Task 6: Dedicated adversarial review of the distinct-pathways runner (BEFORE the no-harm phase)

**Files:** none modified by the reviewer subagent. Strengthen-only
fixes (if any) are applied by a follow-up implementer subagent and
re-reviewed; frozen bars and the single documented partition change
stay byte-unchanged.

**Context.** Dispatch a fresh subagent as a dedicated adversarial
reviewer of the `--distinct-pathways` mode in
`research/runners/integrated_loop_gate.py` (and that it scores via
`research/runners/integrated_loop_core_v2.py`). Its job is to find
holes, not to bless. It produces a written report; the controller
decides on strengthen-only fixes.

**The reviewer must specifically probe and answer, with file:line
evidence:**

1. **Are the two readout pathways genuinely physically distinct,
   sharing only the single online engram write?** The episodic readout
   is the byte-unchanged trisynaptic CA3->CA1 pattern-completion path
   (NOT moved post-consolidation; never touches
   `run_concept_replay_phase`); the concept/working-memory readout is
   the offline-consolidated 16-pool concept layer (never touches the
   order-sensitive episodic recall). No single trace carries both
   order constraints.
2. **Is each lesion faithful?** Each lesion is the full
   distinct-pathways loop minus EXACTLY one system, consuming the SAME
   random draws in the SAME order as `full`. Specifically:
   `no_shared_clock` is truly "one shared clock -> two independent
   clocks" with nothing else changed; `no_binding` is not secretly
   also crippling a readout; `no_cls_replay` removes ONLY the offline
   consolidation (and so collapses the concept/working-memory readout
   while leaving the order-preserving online episodic recall intact --
   exactly the corrected-partition fingerprint); the helper lesions
   are not strawmen crippled elsewhere.
3. **Is there any hard-feed?** The working-memory answer is read from
   the consolidated concept pools and gated by the byte-unchanged
   moat; the episodic order is read from CA1 peak times against the
   true encode order. Neither readout is hand-fed the answer; the
   role-filler selectivity is LEARNED by the reused validated
   mechanism, not wired in.
4. **Are the validated subsystems genuinely reused unchanged, not
   copy-edited?** `build_biological_brain_regions`,
   `build_bg_brain_regions`, the engram API, the Phase-1.3
   consolidation calls, the neuromodulator subsystem, the native
   eligibility/temporal-credit path, the checkpoint module, the moat
   -- all imported byte-unchanged; the original `integrated_loop_core`
   is NOT imported; the NEW `integrated_loop_core_v2` is the scoring
   instrument.
5. **Can a broken or unsound run be scored a success?** Trace the
   NEW-core verdict precedence: a `v1`-unsound run, a
   non-discriminating run (a lesion that didn't collapse its
   corrected-partition-required readout), a malformed/NaN record --
   each must be VOID, never PASS/FAIL. "Cannot conclude" stays
   strictly distinct from "fails".
6. **Is any new automatic differentiation/training added?** Must be
   none. Grep the runner and its import graph for `torch`,
   `backward`, autograd, any gradient objective.
7. **Is the shared theta-gamma controller genuinely ONE shared rhythm
   driving BOTH prefrontal maintenance and the online hippocampal
   episodic write** (so `no_shared_clock` is a real, decisive lesion),
   not secretly two?

**Step:** Dispatch the reviewer subagent with the seven probes above
as the explicit charter. It returns a report. The controller applies
strengthen-only fixes via a follow-up implementer subagent (frozen
bars + the single documented partition change byte-unchanged), then
re-dispatches the reviewer until the report has no open holes. Commit
any strengthen-only fixes with a clear message; controller verifies
the protected set + the original frozen core stay byte-empty in the
diff.

Do NOT proceed to Task 7 until the adversarial review has no open
issues.

---

## Task 7: No-harm phase + CONTROLLER-ONLY decisive run + honest propagation

This task is performed by the controller directly, never delegated to
a subagent. It is the decisive arbiter.

**Step 1: No-harm -- the protected set is byte-unchanged across the
whole branch.** Run `git diff --stat <branch-base>..HEAD` and confirm
the ONLY changed paths are: `research/runners/integrated_loop_core_v2.py`,
`tests/test_integrated_loop_core_v2.py`,
`research/runners/integrated_loop_gate.py`,
`tests/test_integrated_loop_gate.py` (plus, only after Step 5, the
findings doc + capability-status pillar). Explicitly confirm byte-empty
for every protected path, INCLUDING the original frozen verdict:

```
git diff --stat <branch-base>..HEAD -- research/runners/integrated_loop_core.py tests/test_integrated_loop_core.py research/runners/abstention_gate.py tests/test_abstention_gate.py research/runners/text_minimal_isolation.py research/runners/consolidation_trainer.py research/runners/validate_trisynaptic_loop.py research/runners/g11_bg_runner.py research/runners/compose_bridge_core.py research/runners/q2r_core.py research/runners/q2r_gate.py research/runners/constrained_decode_core.py research/runners/constrained_decode_gate.py research/runners/compose_bridge_gate.py research/runners/engram_bootstrap_gate.py sim/bridge.py sim/kernels.py sim/neuromodulators.py sim/train_checkpoint.py sim/backend.py sim/regions.py sim/text_embeddings.py sim/td_value_critic.py sim/compose_temporal_bind.py sim/dendritic_plasticity.py sim/grounded_decode.py research/runners/grounded_decode.py research/runners/generator_g_core.py
```

Expected: empty output.

**Step 2: The moat still passes.** Run
`pytest tests/test_abstention_gate.py -q`. Expected: green.
`abstention_gate.py` is byte-unchanged (`DEFAULT_THRESHOLD = 650.0`).

**Step 3: Both verdict modules + the runner suites still green.** Run
`pytest tests/test_integrated_loop_core.py
tests/test_integrated_loop_core_v2.py
tests/test_integrated_loop_gate.py -q`. Expected: 16/16 original core
(byte-unchanged, unaffected) + the full new-core matrix + the runner
pins/structural tests all pass.

**Step 4: No shipped path imports autograd.** Grep
`research/runners/integrated_loop_core_v2.py` and the
`--distinct-pathways` code path in
`research/runners/integrated_loop_gate.py` for `torch`, `.backward(`,
`autograd`, `grad(` -- expected: no matches in any shipped code path
(a docstring mention is acceptable; an actual import/call is a hard
stop -> back to Task 4).

Do NOT proceed to Step 5 until Steps 1-4 pass. If Step 1 or 2 fails,
the build harmed the protected set -- stop, revert the offending
change, and redo the task that caused it.

**Step 5: Grounding run first (toy numbers NOT reported).** Run the
`--distinct-pathways --tiny-synth` smoke once more on the exact
machine that will do the decisive run; confirm return code 0 and a
TINY-marked verdict. Health check only; not reported as a result.

**Step 6: The decisive multi-seed GPU run.** Fixed pre-registered
configuration: `--distinct-pathways`, the frozen ladder `(2, 4, 8)`,
seeds `42 43 44 45 46`, full (non-tiny) scale, GPU/CuPy path,
kill-safe and monitored to ACTUAL completion. Use a mechanism that
genuinely notifies on completion OR run in the foreground -- NEVER a
detached process with a false "I will be notified" claim. Completion
is actively confirmed (poll the output JSON existence + process state)
before any result is stated. If interrupted, resume the same command
(kill-safe).

**Step 7: Mandatory anti-cheat check -- scrutinize a nominal success
HARDER than a failure.** Recompute the verdict from the single
recorded JSON via `integrated_loop_verdict_v2` WITHOUT re-running and
WITHOUT changing any threshold or the partition. Confirm, by hand from
the recorded numbers:

- The full loop genuinely clears the science bar on BOTH readouts at
  every rung.
- Each single-system lesion genuinely collapses the readout it is
  responsible for UNDER THE CORRECTED PARTITION (in particular:
  `no_cls_replay` collapses the WORKING-MEMORY readout and is NOT
  required to collapse episodic; `no_sequencing` collapses episodic;
  `no_bg_gate` collapses working-memory).
- The three shared-system lesions (`no_binding`, `no_shared_clock`,
  `no_hippo_store`) collapse BOTH readouts together at every rung.
- Instrument soundness (`v1`) is met at every rung.
- The composition accuracy is non-decreasing up to tolerance across
  the ascending ladder and holds at the largest load (for a
  SCALE-CONFIDENT-PASS) -- or, honestly, where it does not.
- The classification returned by `integrated_loop_verdict_v2`
  recomputed from the JSON matches what the recorded numbers imply. No
  re-run, no bar tuning, no partition edit, no overclaim. A nominal
  PASS gets MORE scrutiny than a FAIL.

**Step 8: Honest propagation of EVERY outcome (plain professional
language).**

- Write `research/findings/2026-05-19-distinct-readout-pathways-full-model-<outcome>.md`
  in plain professional language (computational-neuroscientist
  briefing an informed colleague; no codenames as load-bearing terms;
  every technical term defined once). State exactly what the run
  showed, the recomputed `integrated_loop_verdict_v2` verdict, the
  honest ceiling, and what is and is not claimed. State explicitly
  that the original `integrated_loop_core.py` was never edited and its
  VOID stands as the honest record that the original pre-registered
  prediction was falsified. A FAIL or VOID is an honest, valuable
  finding -- write it as such, not as a setback to spin.
- Update the capability-status panel: flip the existing integration
  pillar to the decisive outcome (VALIDATED if SCALE-CONFIDENT-PASS;
  otherwise the honest classification), in plain language. Do not
  append a new pillar number; update the existing one in place.
- Run the capability-status schema test green:
  `pytest tests/test_webapp_server.py -k capability_status -q`.
- Append the Step 1-4 no-harm evidence (protected set + original
  frozen core byte-empty; moat green; original 16/16) to the findings
  doc.
- Commit and push to BOTH remotes (origin and gitea). (The plan-only
  commit for THIS planning document is separate and is NOT pushed by
  the planning agent; the controller pushes.)

**Step 9: Continue autonomously per the pre-committed bound (no
hand-back).**

- **Clean SCALE-CONFIDENT-PASS:** proceed to the design's next staged
  integration step (multi-step sequential composition, then the
  fluent-prior variant), each its own pre-registered gate, each built
  by returning to writing-plans then subagent-driven-development. Do
  not stop; do not declare victory beyond the honest ceiling.
- **Honest non-success (FAIL / WORKS-SMALL-NO-SCALE-CONFIDENCE /
  VOID against the NEW module):** the pre-committed bound governs. Do
  NOT declare the approach unfit and do NOT hand back. Surface the
  precise GPU-measured structural cause honestly. **No further
  partition edits** -- exactly one biologically-cited correction was
  permitted; a second is forbidden as itself goalpost-moving. The next
  step is the next catalog-identified integration factorization,
  pursued autonomously with the SAME adversarial and anti-cheat
  discipline and the SAME (new-module) frozen acceptance. The joint
  falsify-first (Task 5) is the explicit early trigger; the design's
  Candidates 2/3 are the only pre-described in-architecture
  escalations.

Bring Task 7 back to the controller. Tasks 1, 3, 4, 6 are
subagent-driven; Task 2 is a subagent review gate that BLOCKS on
goalpost-move; Tasks 5 and 7 are controller-only.

---

## Remember

- ORDER IS LOAD-BEARING: NEW frozen module (Task 1) -> goalpost-move
  adversarial review gate (Task 2, BLOCKS all downstream tasks if
  rationalized) -> only then the grounding pin (Task 3) and the
  distinct-pathways candidate (Task 4) -> joint falsify-first (Task 5)
  -> runner adversarial review (Task 6) -> controller-only no-harm +
  decisive run + propagation (Task 7).
- The original `research/runners/integrated_loop_core.py` (commit
  `2048750`) and `tests/test_integrated_loop_core.py` are FROZEN and
  NEVER edited; their VOID is the permanent honest record.
- One fresh subagent per task; strict failing-test ->
  minimal-implementation -> run -> commit.
- Controller trust-but-verify EVERY commit: the protected set AND the
  original frozen core are byte-empty in the commit-scoped diff AND
  across the whole branch.
- The moat (`abstention_gate.py`, `DEFAULT_THRESHOLD = 650.0`) stays
  byte-identical and green throughout.
- `integrated_loop_core_v2.py` is the original module transcribed
  verbatim with EXACTLY the symbol rename + the single documented
  partition change; every numeric bar verbatim-equal to the original
  with the original a-priori justifications; its >= 18-case adversarial
  matrix (including the verbatim-bar pin and the exactly-one-partition-
  change pin) must pass; its bars and partition are never tuned to a
  result.
- The distinct-pathways mode is genuine net-new wiring (the per-trial
  controller + the engram-tag fan-out only); everything else reused by
  import, byte-unchanged; no new learning rule; no autograd anywhere;
  GPU/CuPy for the real/decisive path, NumPy only for `--tiny-synth`.
- The falsify-first probes the FULL science mode's working-memory AND
  episodic readouts JOINTLY at minimal load (the recorded process
  lesson).
- Pre-committed bound in force: an honest VOID/FAIL against the NEW
  module is propagated without spin -> next catalog-identified
  factorization, autonomous, no hand-back, no config-crank, and NO
  FURTHER PARTITION EDITS.
- Plain professional language in every artifact and commit message;
  ASCII; no internal codenames as load-bearing terms; honest ceiling
  stated and never overstated.

## Honest ceiling (state this; never overstate it)

A distinct-pathways SCALE-CONFIDENT-PASS against the NEW
catalog-grounded frozen necessity module would mean exactly this: a
biology-grounded loop with two physically distinct readout pathways
(an order-preserving online hippocampal trisynaptic pattern-completion
path for episodic order; an order-invariant offline-consolidated
neocortical schema path for concept/working-memory) shows emergent
compositional memory that holds and scales across the frozen (2, 4, 8)
load ladder, where every single-system lesion abolishes the capability
it is responsible for UNDER THE CORRECTED PARTITION, and the three
shared systems collapse both readouts together. It is explicitly NOT
fluent or open-ended language, NOT a large language model, NOT
conversation solved. This plan authorizes no GPU run and no code by
itself; it specifies the disciplined, pre-committed order in which the
new frozen necessity instrument is pre-registered and adversarially
cleared for goalpost-moving BEFORE any architecture is built or
exercised against it. The original module's VOID is preserved as the
honest scientific record that the original pre-registered necessity
prediction was falsified.
