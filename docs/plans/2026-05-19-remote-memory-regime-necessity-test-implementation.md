---
type: plan
status: live
date: 2026-05-19
---

# Remote/consolidated-memory-regime necessity test -- implementation plan (Design B, falsify-first-GATED)

Status: PLAN ONLY. No code in this document. No GPU run. This plan
implements Design B of the AUTHORITATIVE architecture design at
`docs/plans/2026-05-19-remote-memory-regime-necessity-test-architecture-design.md`
(commit `aa90dac`, conclusion (a): the corrected v2 partition is
realized for all seven lesions in the remote/consolidated regime). The
design is NOT re-litigated here; this plan turns its Design B into
bite-sized test-driven steps with a single hard gate up front.

The verdict authority is the UNCHANGED corrected frozen module
`research/runners/integrated_loop_core_v2.py` (commit `36a7975`). No
bar, no partition, no moat is edited. The original frozen verdict
`research/runners/integrated_loop_core.py` (commit `2048750`) is NEVER
imported and NEVER edited; its prior "cannot conclude" (VOID) stands as
the honest record that the original pre-registered necessity prediction
was falsified.

ASCII only. Plain professional language. Terms are defined once below
and then reused literally. No internal codenames are load-bearing.

---

## Terms (defined once, then reused literally)

- Recent memory: a binding queried within the same trial it was
  written, after only a brief maintenance gap. Complementary-learning-
  systems (CLS) theory holds recent memory is hippocampal and
  consolidation-INDEPENDENT.
- Remote (consolidated) memory: a binding queried only AFTER an offline
  systems-consolidation phase has transferred it into a neocortical
  store, queried while the hippocampus is silenced so only the
  neocortical store can answer. CLS theory holds remote memory is
  neocortical and consolidation-DEPENDENT.
- WM readout (`wm`): the role-selective working-memory query. Present a
  queried role on the language input only; population-vote the filler
  concept pools; emit only if the no-confabulation gate passes, else
  abstain.
- EP readout (`ep`): the episodic-sequence-ORDER recall. Recover the
  order in which the bound pairs were written and score it against the
  true encode order.
- Strict-silence / hippocampus-OFF: the project's VALIDATED Phase-1.3
  protocol that forces a strong negative current onto every hippocampal
  region every step so only consolidated neocortex can answer. The exact
  validated mechanism is `evaluate_with_hippo_off` in
  `research/runners/consolidation_eval.py`
  (`HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3","ca1"]`; a
  monkey-patched `_run_one_simulation_step` that re-applies
  `silence_current_pA` to those region indices before every step; the
  validated 3/3 strict anti-cheat strength is `-2000 pA`).
- Corrected frozen v2 module: `integrated_loop_core_v2.py` (`36a7975`).
  Its only substantive change vs the original frozen module is one
  biologically-cited partition move: the consolidation/replay lesion
  `no_cls_replay` is in the working-memory helper set
  (`_ILV2_HELPER_WM = ("no_bg_gate","no_cls_replay")`), not the
  episodic helper set. Every numeric bar is byte-identical to the
  original.
- Remote-regime per-trial controller: the NET-NEW per-trial phase
  sequencing -- after the byte-unchanged online theta-ordered ENCODE +
  engram WRITE and the byte-unchanged offline Phase-1.3 consolidation,
  it ENGAGES the validated strict-silence mechanism and THEN takes BOTH
  consolidated readouts under the validated `freeze_all_gates` pre-eval
  freeze. This plus the hippo-silence sequencing/wiring is the ONLY
  net-new code.

---

## The central gating question (state it before anything else)

THE LOAD-BEARING UNCERTAINTY. The project's validated Phase-1.3
strict-silence consolidation was validated ONLY for ORDER-AGNOSTIC
semantic W->A retention (CLAUDE.md "Phase 1.3 + Tier 2.1 ... ANTI-CHEAT
VALIDATED"; the strict `-2000 pA` run reproduced retention identically,
confirming the cortex truly retains the order-agnostic bound pattern
post-consolidation). The remote-regime instrument additionally needs
the CONSOLIDATED store to answer an episodic SERIAL-ORDER query (`ep`).

Complementary-learning-systems theory strongly predicts systems
consolidation builds an ORDER-INVARIANT neocortical schema BY DESIGN
(McClelland 1995; Buzsaki 2013). Therefore the consolidated `ep`
readout very likely will NOT clear the 0.80 science bar. If so, Design
B is negative-by-construction and that outcome IS the FIFTH convergent
and unifying terminal structural finding: serial-episodic-ORDER recall
and consolidated-semantic-GENERALIZATION are mutually exclusive in BOTH
memory regimes by the CLS division of labor (recent regime: the
order-shuffled-vs-order-monotone single-trace contention localized by
the distinct-pathways finding; remote regime: the same trade-off
relocated to serial order in the consolidated schema).

This is a STRONG structural prediction but it is empirically
degree-decidable cheaply. It MUST be settled by a cheap falsify-first
BEFORE any expensive build -- not assumed, not skipped. Confirming it
cheaply is rigorous, not wasteful; skipping it would be over-confident.

---

## Plan Task 0 -- grounding pin (red until the remote-regime runner path lands)

Goal: a failing test that pins the remote-regime runner path into
existence, mirroring the established grounding-pin discipline in
`tests/test_integrated_loop_gate.py` (the `--phase-factored` /
`--distinct-pathways` pins). The runner is reused as-is; the new mode
is selected by a new argv flag `--remote-regime` read BEFORE argparse
exactly like the existing `_PHASE_FACTORED` / `_DISTINCT_PATHWAYS`
idiom (no extra rng draw, no signature churn).

Bite-sized TDD steps:

- Task 0.1 (RED). Add `tests/test_integrated_loop_remote_regime.py` with
  a single grounding pin
  `test_remote_regime_tiny_smoke_produces_tiny_verdict`: it runs
  `python -m research.runners.integrated_loop_gate --remote-regime
  --tiny-synth --seeds 42 43 44 --out <tmp>` (timeout 1200s), asserts
  return code 0, the JSON exists, has a `GATE` field, and the verdict
  JSON string contains `TINY` (the toy verdict is NEVER propagated).
  This is RED until the `--remote-regime` mode lands (Task 2). Run it,
  observe the precise failure (unknown flag / no behavior), record it.
- Task 0.2 (structural RED -> GREEN when Task 2 lands). Add
  `test_remote_regime_reuses_validated_parts_byte_unchanged` mirroring
  the existing `test_runner_reuses_validated_phase_factored_parts`: the
  runner source must contain `build_biological_brain_regions`,
  `build_bg_brain_regions`, `run_concept_replay_phase`,
  `set_awake_gates`, `set_sleep_gates`, `freeze_all_gates`,
  `start_engram_recording` or `commit_engram_tag`,
  `from research.runners.abstention_gate import`,
  `integrated_loop_verdict_v2`; and MUST NOT contain
  `from research.runners.integrated_loop_core import`,
  `import integrated_loop_core\n`, `import torch`, or `.backward(`. Add
  remote-regime-specific markers: `_REMOTE_REGIME`, `--remote-regime`,
  `evaluate_with_hippo_off` (the validated strict-silence idiom by
  name) or `HIPPO_REGIONS`, and `-2000` (the validated strict
  configuration strength as a literal). These pin the reuse map.

Definition of done: both pins exist and FAIL for the right reason (the
mode does not yet exist). No production code in Task 0.

---

## Plan Task 1 -- the GATING cheap FALSIFY-FIRST (the load-bearing first milestone)

Everything downstream is GATED on this task. This task implements ONLY
the minimal remote-regime wiring needed to run two modes -- `full` and
the `no_cls_replay` lesion -- at the SINGLE minimal load `N = 2` (the
smallest pre-registered rung `_ILV2_LADDER[0]`) with the minimum seed
count `_ILV2_MIN_SEEDS = 3` (seeds 42, 43, 44), on the GPU/CuPy real
path, JOINTLY measuring BOTH consolidated `wm` AND consolidated `ep` in
the same run. It honors the recorded process lesson: the falsify-first
probes the FULL science mode's readouts JOINTLY at the minimal load,
NEVER a soundness mode alone.

This is deliberately the minimum code that can answer the gating
question. The full 7-lesion runner (Tasks 2-6) is built ONLY if this
gate is GREEN.

### What Task 1 builds (net-new only; everything else reused byte-unchanged)

The remote-regime per-trial spine, exactly as the design Section 2
specifies, added behind the new `_REMOTE_REGIME` argv flag so the
existing default / `--phase-factored` / `--distinct-pathways` paths
stay byte-identical:

1. byte-unchanged online theta-ordered ENCODE + engram WRITE (the
   existing `b4a8106` online path; unchanged).
2. byte-unchanged offline consolidation: `set_sleep_gates(bridge)` then
   `run_concept_replay_phase(bridge, tag_names=[tag], ...,
   randomize_order=True, rng=np.random.default_rng(1000 +
   episode_id))` exactly as the existing `--distinct-pathways` /
   `--phase-factored` branch already calls it (same arguments, same
   dedicated deterministic local rng seeded from the episode id, same
   single-tag content no-op shuffle). Deterministically SKIPPED for
   `no_cls_replay` and `no_hippo_store` exactly as the existing path
   already skips it (a deterministic skip, not an extra/missing draw;
   `_make_pairs` in `_run_mode` remains the SOLE per-trial `rng`
   consumer for every mode).
3. NET-NEW: ENGAGE the validated strict-silence mechanism. Invoke the
   `evaluate_with_hippo_off` silencing IDIOM byte-unchanged in
   semantics: gather `HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3",
   "ca1"]` indices via the bridge region manager, monkey-patch
   `bridge._run_one_simulation_step` so it re-applies
   `silence_current_pA = -2000.0` (the validated 3/3 strict anti-cheat
   strength) to those indices before EVERY step, restore the original
   step and zero the silencing current at the end of the readout
   window. This wraps ONLY the consolidated-readout window; it is the
   project's already-validated mechanism reused, not a new mechanism.
4. byte-unchanged `freeze_all_gates(bridge)` pre-eval freeze (already
   applied at the end of the offline consolidation in the existing
   branch); BOTH consolidated readouts are taken inside the
   strict-silence + frozen window.
5. consolidated `wm`: the existing byte-unchanged WM readout (role code
   on the language input ONLY, population-vote filler pools, emit
   through the byte-unchanged `gate` / `DEFAULT_THRESHOLD = 650.0` or
   abstain; the novel-recombination probe applies to `full`; `v1`
   keeps the trivial drilled-binding soundness query). It now reads the
   CONSOLIDATED neocortical concept layer because the hippocampus is
   strict-silenced.
6. consolidated `ep`: the existing `_episodic_order_readout` closure,
   taken HERE (post-consolidation, inside the strict-silence + frozen
   window) -- the consolidated-trace EP source the runner already
   implements for `--phase-factored` (the `b4a8106` runner already
   contains the post-consolidation consolidated-trace EP readout; the
   `--distinct-pathways` mode took it pre-consolidation, which the
   remote regime does NOT do). Net-new = only that this readout is
   inside the strict-silence window; the readout closure itself is
   reused unchanged.

Net-new in Task 1 is strictly: the `_REMOTE_REGIME` flag plumb-through,
the strict-silence sequencing/wiring around the consolidated-readout
window (semantics of `evaluate_with_hippo_off` reused exactly), and the
per-lesion deterministic skip wiring for `no_cls_replay` /
`no_hippo_store` (identical RNG effect to the existing skip). NO new
learning rule. NO autograd. NO new verdict/partition/moat module. NO
bar change.

### Bite-sized TDD steps for Task 1

- Task 1.1 (RED). Extend `tests/test_integrated_loop_remote_regime.py`
  with `test_remote_regime_strict_silence_wired_around_readout`: assert
  the runner source, in the remote-regime branch, contains the
  strict-silence markers (`HIPPO_REGIONS` or
  `evaluate_with_hippo_off`, the `-2000` literal, a monkey-patched step
  restore-in-`finally`), `run_concept_replay_phase(`,
  `set_sleep_gates(`, `freeze_all_gates(`, and that the consolidated
  EP readout (`_episodic_order_readout`) is invoked in the
  post-consolidation position for this mode. RED until 1.3.
- Task 1.2 (RED). Add `test_remote_regime_skips_consolidation_for_two
  _lesions_with_identical_rng`: a CPU `--tiny-synth --remote-regime`
  invocation that records, for `full` and `no_cls_replay` at `N = 2`,
  the per-trial `_make_pairs` draw (the runner already exposes the
  per-mode pairs in its diagnostic JSON / can be asserted via a
  deterministic-seed equality check) and asserts the `_make_pairs`
  bijection is IDENTICAL across the two modes at the same seed (the
  faithfulness discipline: a lesion that skips a phase produces a
  deterministic skip, not an extra/missing draw). RED until 1.3.
- Task 1.3 (GREEN). Implement the minimal remote-regime spine behind
  `_REMOTE_REGIME` so Tasks 0.1, 0.2, 1.1, 1.2 and the CPU
  `--tiny-synth` smoke all pass. Scope-limit the tiny path to `full`
  and `no_cls_replay` only is NOT required (the runner already iterates
  all modes for `--tiny-synth`; the remote-regime branch only changes
  the per-trial spine, identically for every mode). The tiny verdict
  stays marked TINY and is NEVER propagated.
- Task 1.4 (the GATE measurement; CONTROLLER-dispatched GPU run, NOT in
  this plan's automated test). Specify exactly: run
  `python -m research.runners.integrated_loop_gate --remote-regime
  --seeds 42 43 44 --only-modes full no_cls_replay --only-load 2
  --out <gate.json>` on GPU/CuPy (the runner's existing `--seeds`;
  `--only-modes` / `--only-load` are the minimal-subset selectors --
  if the runner lacks them, Task 1.3 adds them as pure run-subset
  filters that change NO rng draw and NO scored quantity, only WHICH
  (mode, load) cells are executed; this is a run-scope filter, not a
  science change, and is asserted by a test that the full-run JSON for
  `full`/`no_cls_replay` at `N=2` is byte-identical with and without
  the filters). Read BOTH consolidated `wm` AND consolidated `ep`
  jointly from the same run.

### The GATE criterion (state both branches explicitly)

Use the v2 bars verbatim: `_ILV2_SCI_MIN = 0.80`,
`_ILV2_LESION_MAX = 0.40`.

- GATE GREEN (proceed to Tasks 2-6) iff JOINTLY, at `N = 2`, 3 seeds,
  GPU:
  (i) `full` consolidated `wm` >= 0.80 AND `full` consolidated `ep`
      >= 0.80, AND
  (ii) `no_cls_replay` collapses BOTH consolidated readouts:
       `wm <= 0.40` AND `ep <= 0.40` (consolidation removed -> nothing
       transferred -> hippocampus strict-silenced -> nothing anywhere
       to recall).
- GATE RED-by-construction-CONFIRMED (STOP; do NOT build further) iff
  `full` consolidated `ep` does NOT clear 0.80 (the strongly-predicted
  CLS outcome: the consolidated store is order-invariant by design),
  whether or not `full` `wm` clears and whether or not `no_cls_replay`
  collapses `wm`. This IS the FIFTH convergent and unifying terminal
  structural finding (serial-episodic-order and consolidated-semantic-
  generalization mutually exclusive in BOTH memory regimes by the CLS
  division of labor). The controller propagates it honestly with the
  precise measured cause, NO expensive build, NO config-crank, NO
  partition edit, NO hand-back; the next step is the next
  catalog-identified factorization. The plan pre-states this branch so
  the outcome cannot be rationalized after the fact.
- ANY OTHER joint outcome (e.g. `full` `wm` < 0.80 while `ep` >= 0.80;
  or `no_cls_replay` fails to collapse a readout it must) is an
  immediate honest negative with the precise measured cause, recorded
  with the exact failing quantity; do NOT escalate the ladder, do NOT
  crank configuration. Classify it against the v2 module's own
  precedence (VOID for a discrimination/soundness defect; FAIL
  otherwise) and propagate honestly.

Confirming the CLS prediction cheaply here is rigorous; skipping this
joint minimal-load probe and building the full ladder first would be
over-confident. This is the load-bearing first milestone.

---

## Plan Tasks 2-6 -- ONLY if the Task-1 gate is GREEN

These tasks are STRICTLY GATED on Task 1 GREEN. If Task 1 is RED-by-
construction-confirmed or any other negative, Tasks 2-6 are NOT
executed; the controller propagates the Task-1 finding and moves to the
next catalog factorization.

### Plan Task 2 -- the full remote-regime runner (all seven lesions)

Extend the `_REMOTE_REGIME` mode to the full instrument: `v1`, `full`,
and ALL seven lesions, across the full pre-registered ladder
`_ILV2_LADDER = (2,4,8)` with `_ILV2_MIN_SEEDS = 3`. Each lesion must
collapse exactly its corrected-v2-required readout in the REMOTE
regime, per the design's head-on per-lesion analysis (design Section 4,
referenced, NOT re-derived here):

- `no_cls_replay` (v2 WM-helper): deterministic skip of offline
  consolidation -> nothing consolidated -> hippocampus strict-silenced
  -> consolidated `wm` collapses (gate abstains). The load-bearing
  case; satisfiable purely because the regime is remote.
- `no_hippo_store` (v2 SHARED): no engram tag -> nothing to replay ->
  nothing consolidated -> with the hippocampus also silenced, BOTH
  consolidated readouts collapse.
- `no_binding` (v2 SHARED): no bound assembly written online -> engram
  captures no bound pattern -> consolidation transfers no role-
  selective structure and no per-role order -> BOTH collapse.
- `no_shared_clock` (v2 SHARED): two unsynchronized clocks -> incoherent
  online write -> consolidation has nothing coherent to transfer ->
  BOTH collapse.
- `no_neuromod_timing` (v2 HELPER_BOTH): untimed plasticity across the
  whole loop -> no clean bound, ordered online pattern -> consolidation
  transfers neither binding nor order -> BOTH collapse.
- `no_sequencing` (v2 HELPER_EP): online theta REPEATS instead of
  SHIFTING -> no order written at encode -> nothing ordered to
  consolidate -> consolidated `ep` collapses; consolidated `wm`
  (order-agnostic) need not collapse.
- `no_bg_gate` (v2 WM-helper): all channels driven during encode -> no
  single slot holds a clean (role -> filler) binding -> consolidation
  transfers a non-role-selective blur -> consolidated `wm` collapses;
  consolidated `ep` (order-driven) need not collapse.

Faithfulness invariants (pinned by tests, mirroring the existing
discipline): identical per-trial RNG across `v1`/`full`/every lesion
(`_make_pairs` the sole per-trial `rng` consumer; deterministic skips
only); the online ENCODE + engram WRITE stay byte-identical to
`b4a8106`; the strict-silence is the byte-unchanged validated mechanism;
net-new = controller + hippo-silence wiring only; scored by v2.

NOTE on the runner's local comment partition: the `b4a8106` runner
contains a LOCAL comment partition (`_HELPER_EP = ("no_sequencing",
"no_cls_replay")`) that pre-dates the regime change. That local comment
is NOT the acceptance authority and is NOT edited. The acceptance
authority is the v2 module's partition
(`_ILV2_HELPER_WM = ("no_bg_gate","no_cls_replay")`). In the remote
regime the runner's `no_cls_replay` behavior (skip consolidation)
produces a consolidated-WM collapse, exactly what the v2 WM-helper duty
requires -- so the regime change resolves the apparent local/authority
mismatch in the v2 module's favor WITHOUT any partition edit. This is
stated explicitly so the Task 3 review cannot mistake the unchanged
local comment for a goalpost-move.

Bite-sized TDD steps:
- Task 2.1 (RED). Add a structural pin: the remote-regime branch
  iterates `_ALL_LESIONS` (all seven) + `v1` + `full` across
  `_IL_LADDER = (2,4,8)`; scored solely via `integrated_loop_verdict_v2`.
- Task 2.2 (RED). Add a per-lesion deterministic-skip / RNG-identity
  pin extending Task 1.2 to all seven lesions at `N = 2` on the CPU
  `--tiny-synth` path (the `_make_pairs` bijection identical across all
  modes at a fixed seed).
- Task 2.3 (GREEN). Implement the full seven-lesion remote-regime
  spine; CPU `--tiny-synth` smoke passes end-to-end and writes a
  TINY-marked verdict (never propagated).

### Plan Task 3 -- dedicated adversarial review of the runner

A separate review pass (its own commit by the controller; this plan
only specifies it) that confirms, against the source and the design:
- every lesion is faithful in the REMOTE regime (each collapses exactly
  its corrected-v2-required consolidated readout; no strawman);
- the strict-silence is the byte-unchanged VALIDATED mechanism
  (`evaluate_with_hippo_off` semantics, `HIPPO_REGIONS`, `-2000 pA`,
  monkey-patched step restored in `finally`), not a re-implementation;
- the consolidated readouts are GENUINE spiking measurements (the
  reused WM population-vote + `_episodic_order_readout` closure), NOT
  hard-feeds / Python-injected answers;
- the reused parts are byte-unchanged (online encode/write,
  consolidation interface, moat, v2);
- NO autograd anywhere;
- scored by v2, NOT the original (the original is never imported);
- the unchanged local `_HELPER_EP` comment is NOT a goalpost-move (per
  the explicit NOTE above).
The review is documented as a findings entry by the controller; a
non-clear review STOPS the line (honest negative, propagated).

### Plan Task 4 -- no-harm checks (byte-empty diffs on the frozen surfaces)

Add/extend tests asserting the frozen surfaces are byte-unchanged by
this work:
- original frozen verdict `integrated_loop_core.py` (`2048750`) diff
  byte-empty against this branch's base; its existing test suite still
  16/16;
- corrected frozen v2 `integrated_loop_core_v2.py` (`36a7975`) diff
  byte-empty; its existing test suite still 18/18;
- no-confabulation moat `abstention_gate.py` diff byte-empty; its
  existing suite still 7/7;
- the validated Phase-1.3 / strict-silence sources
  (`consolidation_trainer.py` `run_concept_replay_phase`,
  `consolidation_eval.py` `evaluate_with_hippo_off` / `HIPPO_REGIONS`,
  `text_minimal_isolation.py` `set_sleep_gates` / `set_awake_gates` /
  `freeze_all_gates`) diff byte-empty;
- the runner's online ENCODE + engram WRITE region byte-identical to
  `b4a8106` (the controller's git-range check; this plan pins the
  structural markers exist and that the new branch only adds behavior
  AFTER the maintain phase, exactly like the existing `_PHASE_FACTORED`
  / `_DISTINCT_PATHWAYS` discipline).
Definition of done: all no-harm assertions GREEN; full pre-existing
suites for the frozen modules still pass at their pinned counts.

### Plan Task 5 -- CONTROLLER-ONLY decisive multi-seed run

CONTROLLER dispatches the decisive GPU/CuPy run (NOT an automated test
in this plan): the full remote-regime instrument across the full ladder
`(2,4,8)`, all seven lesions + `v1` + `full`, at >= `_ILV2_MIN_SEEDS`
seeds, scored solely by the UNCHANGED `integrated_loop_verdict_v2`
recomputed from the single recorded JSON. Mandatory anti-cheat
smell-test BEFORE accepting any classification:
- the lesion ceiling holds for every lesion at every load (no lesion
  silently above `_ILV2_LESION_MAX` on a readout it must collapse);
- `full` does not pass via a degenerate constant readout (inspect the
  recorded per-mode readouts; a suspiciously perfect or flat profile is
  a VOID smell, not a PASS);
- the strict-silence actually silenced (recorded
  `n_hippo_neurons_silenced` > 0 and non-trivial);
- the per-mode `_make_pairs` faithfulness identity held.
Honest propagation BOTH ways: a PASS is reported ONLY with the binding
honesty ceiling below (consistent-with the corrected biology in the
remote regime; NEVER a validated success); a VOID/FAIL is reported as a
strong informative negative with its precise GPU-measured structural
cause.

### Plan Task 6 -- honest propagation + next factorization (pre-committed)

Whatever Task 5 yields, propagate per the pre-committed bound below
(verbatim). If VOID/FAIL: an honest negative, surfaced with its
precise measured cause, then the next catalog-identified integration
factorization, autonomously, with the SAME adversarial + anti-cheat
discipline and the SAME (v2-module) frozen acceptance, NO further
partition edits, NO hand-back, NO config-crank. The original frozen
verdict + v2 + moat are NEVER edited.

---

## State verbatim in the plan

### Honesty ceiling (binding; stated BEFORE any build/run)

1. The load-bearing, scale-confident scientific result of this entire
   line is the THRICE-CONVERGENT FALSIFICATION of the original
   pre-registered necessity prediction, EXTENDED by the fourth
   (regime-level) structural characterization (the necessity question
   is well-posed ONLY in the remote/consolidated regime), and possibly
   extended by the fifth (the Task-1 CLS-confirmed terminal finding).
   This result is independent of any future build.
2. A clean, scale-confident, validated PASS was only ever obtainable
   against the ORIGINAL frozen instrument, now known unobtainable from
   this architectural line because the original prediction it encodes
   was falsified. This is the honest terminus of the "validated-PASS
   against the original pre-registration" goal for this line and is NOT
   a deficiency to be patched around.
3. Therefore any pass of this remote-regime instrument against the
   corrected v2 module is explicitly "consistent-with the corrected
   (biologically-revised) necessity structure in the
   remote/consolidated regime" ONLY -- a weak positive whose strength
   rests on the single biologically-cited partition correction AND on
   the regime choice; it MUST always be reported with this exact
   limitation, NEVER spun, NEVER as a validated success, NEVER as
   scale-confident-validated.
4. A VOID or FAIL against the corrected v2 module (or the Task-1 gate
   confirming the CLS prediction) IS a strong, unambiguous, informative
   negative/terminal finding and is propagated as such.
5. No further partition edit is permitted, ever, on this line. The one
   biologically-cited correction (the v2 module) was the single
   pre-committed move; a second would be unambiguous goalpost-moving
   and is forbidden.
6. Conversational / compositional / fluent-language / large-language-
   model capability is NOT claimed and is NOT in scope. All
   previously-validated capabilities (trustworthy grounded memory,
   no-confabulation abstention, simple generation, no catastrophic
   forgetting) are intact and unaffected; the no-confabulation gate,
   the original frozen verdict, and the corrected frozen v2 module are
   all byte-unchanged.

### Pre-committed bound (stated in advance so no outcome can be rationalized)

A faithful remote-regime build (Design B; Design C as the
pre-described in-architecture escalation only if Design B PASSES and a
stricter confirmation is wanted -- it never makes Design B's acceptance
easier) evaluated against the UNCHANGED corrected frozen v2 module that
reaches "cannot conclude" (VOID) or "fails" the corrected partition --
OR the Task-1 cheap gate confirming the strong CLS order-invariant
prediction -- is an honest negative/terminal finding. It is surfaced
honestly with its precise, GPU-measured structural cause -- not a
configuration iteration, not spin, not a hand-back, not a
declare-globally-unfit. The next step is then the NEXT catalog-
identified integration factorization, pursued autonomously with the
SAME adversarial and anti-cheat discipline and the SAME (v2-module)
frozen acceptance, with NO further partition edits. The original
frozen verdict + v2 + moat are NEVER edited. This bound is fixed in
advance.

### The recorded process lesson

The falsify-first probes the FULL science mode's readouts JOINTLY (here
BOTH consolidated `wm` AND consolidated `ep`), at the single minimal
load (`N = 2`), with the minimum seed count (3) -- NOT a soundness mode
alone, NOT one readout in isolation, NOT a larger load first.

---

## Reuse map and net-new (summary)

Byte-unchanged (imported, not modified):
- Validated Phase-1.3 consolidation `run_concept_replay_phase`
  (`consolidation_trainer.py`) under `set_sleep_gates` /
  `set_awake_gates` / `freeze_all_gates`
  (`text_minimal_isolation.py`).
- Validated strict-silence / hippocampus-OFF mechanism:
  `evaluate_with_hippo_off` idiom from `consolidation_eval.py`
  (`HIPPO_REGIONS = ["ec","dg","dg_pv_basket","ca3","ca1"]`;
  monkey-patched step re-applying the strong negative current every
  step; the `-2000 pA` strict / 3/3 anti-cheat configuration). Semantics
  preserved exactly.
- Trisynaptic / engram store + ca1->concept consolidation pathway +
  16-pool concept substrate + homeostasis + non-zero-init:
  `build_biological_brain_regions` / `text_minimal_isolation.py`.
- No-confabulation moat `abstention_gate.py` (`gate`,
  `DEFAULT_THRESHOLD = 650.0`).
- Corrected frozen v2 acceptance module
  `integrated_loop_core_v2.py` (`36a7975`) -- the sole verdict
  authority.
- Original frozen core `integrated_loop_core.py` (`2048750`) -- NEVER
  imported, NEVER edited; its VOID is the honest record.
- The `b4a8106` online/offline pathways in `integrated_loop_gate.py`:
  the online theta-ordered ENCODE + engram WRITE (byte-identical), the
  separate offline Phase-1.3 consolidation, the per-trial controller,
  the engram fan-out, the native eligibility/temporal-credit reward
  path, the WM population-vote readout, the `_episodic_order_readout`
  closure.

Net-new (the ONLY new code):
- The `_REMOTE_REGIME` argv flag plumb-through (read before argparse,
  same idiom as `_PHASE_FACTORED` / `_DISTINCT_PATHWAYS`; no extra rng
  draw, no signature churn).
- The remote-regime per-trial controller: after the byte-unchanged
  online ENCODE + WRITE and the byte-unchanged offline consolidation,
  ENGAGE the validated strict-silence mechanism, then take BOTH
  consolidated readouts under the validated `freeze_all_gates` freeze.
- The hippo-silence sequencing/wiring: invoking the validated
  `evaluate_with_hippo_off` silencing idiom at the correct point in the
  per-trial sequence (around the consolidated readouts, restored at
  trial end, `-2000 pA`), plus the deterministic per-lesion skip wiring
  for `no_cls_replay` / `no_hippo_store` (identical RNG effect to the
  existing skip).
- Pure run-scope selectors `--only-modes` / `--only-load` if absent
  (Task 1.4) -- they change NO rng draw and NO scored quantity; only
  WHICH (mode, load) cells run; asserted byte-identical to the
  unfiltered run for the selected cells.

Explicitly NO new learning rule. NO automatic differentiation. NO new
verdict/partition/moat module. NO bar change.

---

## Hard constraints (restated for the executor)

PLAN ONLY. The implementer of this plan creates production code in
later tasks; THIS document changes NO code. No GPU run is performed by
this document. The original/v2 frozen modules, the moat, other plans,
and findings are NOT modified by this document. Exactly one file is
created: this plan. The single commit contains ONLY this file.

---

## Files / evidence

- Authoritative design (Design B; not re-litigated):
  `docs/plans/2026-05-19-remote-memory-regime-necessity-test-architecture-design.md`
  (`aa90dac`).
- Corrected frozen necessity module (UNCHANGED; sole verdict
  authority): `research/runners/integrated_loop_core_v2.py`
  (`36a7975`).
- Original frozen verdict (UNCHANGED; never imported; its "cannot
  conclude" preserved as the honest record):
  `research/runners/integrated_loop_core.py` (`2048750`).
- Runner reused/extended (online ENCODE + engram WRITE byte-identical):
  `research/runners/integrated_loop_gate.py` (`b4a8106`).
- Validated Phase-1.3 strict-silence / hippocampus-OFF protocol (reused
  byte-unchanged): `research/runners/consolidation_trainer.py`
  (`run_concept_replay_phase`), `research/runners/consolidation_eval.py`
  (`evaluate_with_hippo_off`, `HIPPO_REGIONS`, the `-2000 pA` strict
  configuration); `research/runners/text_minimal_isolation.py`
  (`set_sleep_gates` / `set_awake_gates` / `freeze_all_gates`).
- No-confabulation moat (byte-unchanged):
  `research/runners/abstention_gate.py`.
- Convergent-finding evidence chain (read, not re-litigated):
  `research/findings/2026-05-19-FOURTH-convergent-structural-finding-necessity-instrument-probes-recent-memory-where-consolidation-is-not-needed.md`,
  `research/findings/2026-05-19-distinct-pathways-honest-negative-episodic-contradiction-DISSOLVED-WM-blocked-by-consolidation-genericization.md`,
  `research/findings/2026-05-19-THIRD-convergent-signal-original-necessity-hypothesis-falsified-catalog-grounded-correction.md`,
  `research/findings/2026-05-19-CONTROLLER-precommitted-honesty-ceiling-on-the-corrected-module-PASS.md`.
- Validated Phase-1.3 findings:
  `research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`,
  `research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md`,
  `research/findings/2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md`.
- Established grounding-pin discipline mirrored by Task 0:
  `tests/test_integrated_loop_gate.py`.
