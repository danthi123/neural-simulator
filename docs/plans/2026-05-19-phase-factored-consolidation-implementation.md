# Phase-factored consolidation: TDD implementation plan

> **For Claude:** REQUIRED SUB-SKILL: use superpowers:executing-plans to
> implement this plan task-by-task. Standing autonomy applies: one fresh
> subagent per task; strict failing-test -> minimal-implementation ->
> run -> commit; the controller verifies every commit leaves the
> protected set byte-unchanged; honest propagation of every outcome;
> iterate following the reference biology on any non-success; no
> hand-back. Task 5 is controller-only and is conditioned on the gating
> section below resolving to outcome (a).

**Goal.** Build the phase-factored consolidation form of the
integrated-loop full-model test: keep the online theta-ordered
hippocampal episodic encode and episodic readout byte-unchanged from
honest-work-in-progress commit `e02f692`; add a separate offline
sleep-gated consolidation that drives the committed engram-tagged
ensemble through the project's validated Phase-1.3 replay path in
shuffled order to train the sixteen-pool concept layer; serve the
working-memory readout from the consolidated neocortical concept layer;
keep the episodic-sequence readout served from the online store. The
only net-new code is a per-trial phase controller (sequencing existing
validated calls) plus the composition wiring. No new learning rule, no
automatic differentiation.

**Source design (authoritative).**
`docs/plans/2026-05-19-phase-factored-consolidation-architecture-design.md`
(commit `65f6c52`), Candidate A (recommended). Its evidence chain:
`research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-between-validated-concept-binding-and-episodic-store.md`;
the original implementation plan
`docs/plans/2026-05-18-integrated-loop-full-model-implementation.md`
(the Tasks 0-5 structure and anti-cheat discipline this plan mirrors);
the frozen verdict `research/runners/integrated_loop_core.py`
(unchanged since commit `2048750`); the honest-work-in-progress runner
`research/runners/integrated_loop_gate.py` at `e02f692` (online
theta-ordered episodic encode + engram store + per-stripe homeostasis +
documented non-zero initialization; episodic binding perfect, episodic
score 1.0).

**Tech stack.** Python; the project spiking simulator
(`sim/bridge.py`); the validated region/pathway builder
(`build_biological_brain_regions`); the validated basal-ganglia cascade
(`build_bg_brain_regions`); the bridge engram-tagging interface; the
neuromodulator subsystem; the native eligibility-trace temporal-credit
path; the validated Phase-1.3 consolidation interfaces
(`set_awake_gates`, `set_sleep_gates`, `run_concept_replay_phase`,
`freeze_all_gates`); the checkpoint module; the no-confabulation
abstention gate. Verdict module: standard library + typing only,
reused byte-unchanged.

**Honest ceiling (state it; never overstate it).** A phase-factored
PASS means exactly this: a biology-grounded multi-phase loop (an online
theta-ordered hippocampal episodic encode plus a separate offline
shuffled-replay neocortical consolidation) shows emergent compositional
memory that holds across the frozen load ladder, where every
single-system lesion abolishes the capability it is responsible for and
every shared-system lesion collapses both readouts together. It is
explicitly NOT fluent open-ended language, NOT a large language model,
NOT conversation solved. Prior validated results and documented
boundaries are unaffected. The earlier isolated-mechanism negatives
stand and are reinterpreted, not refuted.

---

## Frozen-lesion-partition interaction (load-bearing; resolved before any build)

This section is the load-bearing decision. The build does NOT proceed
to Task 5 until this section resolves to outcome (a). Be rigorous and
honest, not optimistic.

### The constraint that may not be edited

The frozen verdict `research/runners/integrated_loop_core.py`
(`integrated_loop_verdict`, unchanged since commit `2048750`)
pre-registers a FIXED lesion partition (file lines 62-65):

- `_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")` --
  each must collapse BOTH the working-memory readout AND the episodic
  readout (file lines 165-173: both `wm <= 0.40` and `ep <= 0.40`).
- `_HELPER_WM = ("no_bg_gate",)` -- must collapse the working-memory
  readout (file lines 174-181: only `wm <= 0.40` is checked; `ep` is
  unconstrained).
- `_HELPER_EP = ("no_sequencing", "no_cls_replay")` -- must collapse
  the episodic readout (file lines 182-189: only `ep <= 0.40` is
  checked; `wm` is unconstrained).
- `_HELPER_BOTH = ("no_neuromod_timing",)` -- must collapse BOTH
  (file lines 165-173, same branch as `_SHARED`).

Editing this partition (or any frozen bar) to make a new architecture
pass is the cardinal anti-cheat violation. The partition, the bars,
and the no-confabulation moat are reused byte-unchanged. If a faithful
Candidate-A wiring makes any frozen lesion non-discriminating, the
honest outcome is VOID surfaced by the unchanged verdict, never a
softened partition.

### The exact load-bearing tension

In the `e02f692` runner the per-trial order is: reset -> ENCODE (online
theta-ordered, engram recording active) -> commit engram tag ->
MAINTAIN -> WORKING-MEMORY READOUT (file lines 1262-1456: query the
role code on `language_input`, read the filler concept pools through
the basal-ganglia-gated `dlpfc_verb -> noun_pool_F` efferent) ->
EPISODIC READOUT (file lines 1466-1502: `stimulate_tag` the committed
online engram tag, recover order from per-item activity-peak times) ->
LEARN -> REPLAY/CONSOLIDATION (file lines 1526-1593: drive the
committed tag under sleep gates; `no_cls_replay` skips this).

Decisive structural fact: in `e02f692` the replay/consolidation phase
runs AFTER both readouts and feeds NOTHING that either readout measures.
The `no_cls_replay` lesion there skips a phase that is a no-op for
scoring. (That is precisely an instrument-soundness gap the design's
program-level finding flags; it is not a property to preserve.)

Under Candidate A the working-memory readout is moved to AFTER the
offline consolidation and reads the consolidated neocortical concept
layer. Then a naive `no_cls_replay` (remove the offline consolidation)
collapses the working-memory readout and NOT the episodic readout --
which INVERTS its frozen `_HELPER_EP` responsibility (it would behave
like a `_HELPER_WM` lesion). The frozen verdict would correctly surface
that as a non-discriminating VOID (file lines 182-189: `no_cls_replay`
in `_HELPER_EP` is checked ONLY on `ep`; if `ep` does not collapse, the
run is VOID). A naive Candidate-A wiring is therefore VOID-by-
construction against the frozen verdict.

### The honest resolution (outcome (a)), with the precise mechanism

The resolution is NOT to edit the partition. It is to wire the
architecture so each frozen lesion's pre-registered responsibility
genuinely still holds, justified mechanistically by the biology. The
key is the project's OWN validated Phase-1.3 result: after
sharp-wave-ripple-gated replay consolidation, the trace is genuinely in
neocortex; the strict-silence anti-cheat (10x stronger hippocampal
silencing + zeroing the ca1->cortex edges) gave IDENTICAL retention to
non-strict at 3/3 multi-seed, confirming the consolidated cortical
representation is what carries the recalled content
(CLAUDE.md "Phase 1.3 + Tier 2.1 ... ANTI-CHEAT VALIDATED";
McClelland 1995 / Buzsaki 2013 complementary learning systems).

Faithful Candidate-A wiring (the per-trial phase controller in
Task 2), and why each frozen lesion still discharges its frozen duty:

1. **`no_binding` (`_SHARED`, must collapse BOTH).** Ablation: suppress
   the slot-agnostic prefrontal excitability bias so the
   basal-ganglia-selected slot never reaches threshold and the
   role+filler combinatorial assembly never forms (byte-identical to
   `e02f692` file lines 999-1007). Mechanism: with no relational
   assembly there is (i) no bound (role, filler) structure for the
   online engram write -> the committed tag is empty/degenerate -> the
   online episodic readout collapses; AND (ii) no bound structure for
   the offline replay to consolidate -> the consolidated concept layer
   has no role-selective filler -> the working-memory readout collapses.
   Both collapse. Frozen `_SHARED` duty PRESERVED.

2. **`no_shared_clock` (`_SHARED`, must collapse BOTH).** Ablation:
   replace the ONE shared theta-gamma instance with TWO independent
   instances (prefrontal-gating clock vs hippocampal-write clock,
   phase-desynchronized), nothing else changed (byte-identical to
   `e02f692` file lines 934-943). Mechanism: desynchronized timing
   destroys the theta-ordered episodic write -> the online episodic
   readout collapses; AND it destroys the binding-to-slot phase
   alignment at encode, so the engram tag does not capture a clean
   per-binding ensemble -> the offline replay has no coherent tagged
   ensemble to consolidate -> the consolidated working-memory readout
   collapses. Both collapse. Frozen `_SHARED` duty PRESERVED.

3. **`no_hippo_store` (`_SHARED`, must collapse BOTH).** Ablation: skip
   `start_engram_recording` / `commit_engram_tag` / `stimulate_tag`
   (no fast relational store; byte-identical to `e02f692` file lines
   956-957, 1241-1247, 1472-1473). Mechanism: no engram tag means the
   online episodic readout collapses by construction (file line 1473
   already sets `ep_acc = 0.0`); AND the offline consolidation has NO
   tag to replay through `run_concept_replay_phase` (its `tag_names`
   input is empty) -> nothing is consolidated into the concept layer ->
   the consolidated working-memory readout collapses. The consolidated
   working-memory content is strictly DOWNSTREAM of the online engram
   write: no online episode -> no replay source -> no consolidated
   concept -> no working memory. Both collapse. Frozen `_SHARED` duty
   PRESERVED.

4. **`no_bg_gate` (`_HELPER_WM`, must collapse WM; EP unconstrained).**
   Ablation: drive ALL basal-ganglia channels so all thalamic outputs
   partially disinhibit and no single prefrontal slot is cleanly held
   (byte-identical to `e02f692` file lines 1016-1022). Mechanism: with
   no selective slot gating the encode writes a smeared, non-slot-
   selective assembly; the engram tag still records SOME ensemble and
   the online theta order is still intact, so the episodic-sequence
   readout (order of per-item peaks) is NOT required to collapse
   (`_HELPER_WM` does not check `ep`). But the offline replay then
   consolidates a non-role-selective blob -> the consolidated
   working-memory readout cannot return the correct role-selective
   filler -> the working-memory readout collapses. Frozen `_HELPER_WM`
   duty (collapse WM) PRESERVED. (The frozen verdict does not require
   `no_bg_gate` to preserve EP; whatever EP does is acceptable.)

5. **`no_sequencing` (`_HELPER_EP`, must collapse EP; WM
   unconstrained).** Ablation: the shared clock REPEATS the assembly
   instead of SHIFTING it across theta cycles, so no recoverable
   episodic order is written (byte-identical to `e02f692` file lines
   316-322, 935-942). Mechanism: the online episodic readout recovers
   order from the argsort of per-item activity-peak times; with a
   repeated (un-shifted) assembly every binding peaks at the same phase
   -> the recovered order is degenerate -> the online episodic readout
   collapses. The (role, filler) pairing content is still present in
   the tag, so the offline consolidation can still build a role-
   selective concept layer; `_HELPER_EP` does not check `wm`, so the
   working-memory readout is unconstrained here. Frozen `_HELPER_EP`
   duty (collapse EP) PRESERVED.

6. **`no_cls_replay` (`_HELPER_EP`, must collapse EP; WM
   unconstrained) -- THE load-bearing lesion.** Ablation: skip the
   offline shuffled-replay consolidation phase entirely. Resolution
   mechanism (this is the crux, and it is biologically grounded, not a
   relabel): under Candidate A the episodic-sequence readout is served
   from the CONSOLIDATED trace, exactly as the project's own validated
   Phase-1.3 strict-silence anti-cheat established -- after replay
   consolidation, the recalled sequence is carried by the cortical
   representation, and silencing the hippocampal store leaves retention
   unchanged BECAUSE the trace consolidated. Concretely, the episodic
   readout in Candidate A is taken AFTER the offline consolidation and
   under the validated `freeze_all_gates` pre-eval freeze (the Phase-1.3
   evaluation idiom), reading the order from the consolidated pathway,
   not from a hippocampus-only `stimulate_tag`. Therefore removing the
   offline consolidation (`no_cls_replay`) means there is no
   consolidated trace to recall the sequence from -> the episodic
   readout collapses. Frozen `_HELPER_EP` duty (collapse EP) PRESERVED.
   Note the frozen verdict checks `no_cls_replay` ONLY on `ep` (file
   lines 182-189); `wm` is unconstrained, so it is acceptable that
   `no_cls_replay` ALSO collapses the consolidated working-memory
   readout. Both collapsing satisfies `_HELPER_EP` (only `ep` is
   gated). The inversion risk (collapses WM but NOT EP) is removed
   precisely because the episodic readout is now consolidation-
   dependent. This is a biology-faithful design choice (systems
   consolidation of episodic sequences; Buzsaki 2013; the project's
   own Phase-1.3 + Tier-2.1 strict anti-cheat), NOT a tuning trick and
   NOT an edit to the frozen verdict.

   Honest dependency this introduces (stated now so it cannot be
   rationalized later): Candidate A as designed says "episodic readout
   stays from the online store, byte-unchanged from `e02f692`."
   Preserving `no_cls_replay`'s frozen `_HELPER_EP` duty REQUIRES the
   episodic readout to be consolidation-dependent. These two statements
   are reconciled by reading the episodic ORDER from the consolidated
   trace after the offline phase under the validated Phase-1.3
   freeze-then-evaluate idiom (the same anti-cheat-validated mechanism),
   while the ONLINE theta-ordered ENCODE + the engram WRITE remain
   byte-unchanged from `e02f692` (presentation-order == binding-index at
   encode is exactly what makes the consolidated order recoverable).
   "Online store byte-unchanged" is preserved for the encode/write path
   (the part whose order constraint conflicts with the concept
   mechanism); the READOUT is taken post-consolidation. This is the
   single substantive design refinement this gating analysis imposes on
   Candidate A, it is biologically grounded, and it is what makes the
   factoring NOT VOID-by-construction. Task 2 implements exactly this;
   Task 3 adversarially verifies the encode/write path is byte-identical
   to `e02f692` and only the readout timing/source changed.

7. **`no_neuromod_timing` (`_HELPER_BOTH`, must collapse BOTH).**
   Ablation: remove the shared-clock-gated acetylcholine plasticity
   window so plasticity is always on and untimed, applied consistently
   across the whole loop (byte-identical to `e02f692` file lines
   1027-1031, 1225, 1514). Mechanism: untimed plasticity corrupts the
   theta-paced relational write at encode -> the online episodic order
   write is degraded -> the consolidated episodic readout collapses;
   AND untimed plasticity during the offline replay corrupts the
   ca3->ca1->concept consolidation (no acetylcholine window gating the
   replay update) -> the consolidated concept layer is not role-
   selective -> the working-memory readout collapses. Both collapse.
   Frozen `_HELPER_BOTH` duty PRESERVED.

8. **`v1` and `full` (not lesions; the soundness and science modes).**
   `v1` = the full phase-factored loop on a no-gap trivial single bind;
   the working-memory query is the trivial drilled binding (query a
   drilled role, expect its own bound filler) per the original plan's
   corrected `wm` readout bullet and design Section 5; must clear
   `_IL_V1_MIN` (0.90) on BOTH readouts. `full` and every lesion keep
   the novel-recombination science probe (the last query uses a role
   bound to a different filler than drilled) unchanged and exactly as
   hard; must clear `_IL_SCI_MIN` (0.80) on BOTH readouts.

### Gating conclusion

Every one of the eight frozen lesions discharges its pre-registered
`integrated_loop_core` responsibility under a faithful Candidate-A
wiring, PROVIDED the single biology-grounded design refinement in
lesion 6 is applied: the episodic-sequence ORDER is read from the
consolidated trace after the offline phase under the validated
Phase-1.3 freeze-then-evaluate idiom, while the online theta-ordered
ENCODE and the engram WRITE remain byte-unchanged from `e02f692`. With
that refinement the analysis resolves to **outcome (a): the build
proceeds.** No frozen bar, partition, or moat is edited. Task 5 is
conditioned on this conclusion remaining (a) after the Task 3
adversarial review independently re-derives it; if the Task 3 review
finds that the lesion-6 refinement cannot be implemented without
either (i) editing the frozen verdict/partition or (ii) a non-faithful
strawman lesion, then the gating section is re-classified as **outcome
(b): VOID-by-construction against the frozen verdict**, Task 5 is NOT
run, and the controller surfaces this as a program-level finding (the
next catalog-identified factorization -- a deeper separation of
relational binding from schema abstraction along the catalog's
hippocampal-neocortical interaction entries), autonomously, with no
hand-back.

### Pre-registered bound (stated now, before any run)

If a faithful phase-factored build (Candidate A, with the design's
pre-described Candidate B / Candidate C as the only permitted
in-architecture escalations) cannot achieve `v1 wm AND ep >= 0.90`
with the frozen lesion contrasts discriminating, that is a deeper
program-level result. It is surfaced honestly with its precise,
GPU-measured structural cause -- not a configuration iteration, not
spin, not a hand-back, not a declare-globally-unfit, not a
config-crank, and never an edit to a frozen bar/partition/moat. The
next step is then the next catalog-identified integration factorization
(a deeper separation of relational binding from schema abstraction
along the catalog's hippocampal-neocortical interaction entries),
pursued autonomously with the SAME adversarial and anti-cheat
discipline and the SAME frozen acceptance. The design's falsify-first
smoke is the explicit early trigger: if inserting the offline phase
does not keep the episodic score at 1.0, the escalation fires
immediately with that exact structural cause, with no config crank.
This bound is stated in advance so the next outcome cannot be
rationalized after the fact.

---

## Reuse-by-import only (the protected set -- byte-unchanged)

The phase-factored runner imports and composes these; it does NOT
modify, copy-edit, or re-implement any of them. The controller verifies
(per task, and across the whole branch) that every path below is
byte-empty in the commit-scoped `git diff`:

- `research/runners/integrated_loop_core.py` (the FROZEN verdict;
  unchanged since `2048750`; reused byte-unchanged; its 16-case test
  matrix in `tests/test_integrated_loop_core.py` stays green and
  byte-identical).
- `research/runners/abstention_gate.py` +
  `tests/test_abstention_gate.py` (the no-confabulation moat,
  `DEFAULT_THRESHOLD = 650.0`; MUST stay 7/7 green and byte-identical
  the entire build).
- `research/runners/text_minimal_isolation.py`
  (`build_biological_brain_regions`, `set_awake_gates`,
  `set_sleep_gates`, `freeze_all_gates` REUSED UNMODIFIED).
- `research/runners/consolidation_trainer.py`
  (`run_concept_replay_phase`, `run_swr_replay_phase` REUSED
  UNMODIFIED).
- `research/runners/g11_bg_runner.py` (`build_bg_brain_regions`
  REUSED UNMODIFIED).
- every other existing frozen verdict module and its paired gate
  (`compose_bridge_core.py`, `compose_bind_core.py`, `td_critic_core.py`,
  `q2r_core.py`, every other `*_core.py`; `compose_bridge_gate.py`,
  `constrained_decode_gate.py`, `q2r_gate.py`,
  `engram_bootstrap_gate.py`).
- the validated simulator modules: `sim/bridge.py`, `sim/kernels.py`,
  `sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/backend.py`,
  `sim/regions.py`, `sim/text_embeddings.py`.

The only file this build modifies is the existing honest-work-in-
progress runner `research/runners/integrated_loop_gate.py` (the
net-new phase controller + composition wiring go HERE, evolving the
`e02f692` foundation; the online encode/write path stays byte-identical
to `e02f692`). The only file this build adds, if a tiny pure helper is
genuinely unavoidable, is documented under Task 1 (prefer reuse; expect
none). Plus the propagation artifacts in Task 5 (a findings doc, a
capability-status pillar edit, a git commit). Tests are extended in
`tests/test_integrated_loop_gate.py`.

## No new automatic differentiation / training anywhere

Every learning update in this build is a REUSED validated local rule
(the native eligibility-trace temporal-credit path with the validated
constants; the validated spike-timing plasticity inside the reused
Phase-1.3 replay path). No `torch`, no `.backward()`, no autograd, no
gradient-descent objective is introduced in any shipped path. Task 3
(adversarial review) and Task 4 (no-harm) both explicitly assert this.

## Naming

Plain descriptive names; no internal codenames as load-bearing terms.
"Phase-factored consolidation" = the online theta-ordered episodic
encode plus the separate offline shuffled-replay neocortical
consolidation. "Online store" = the engram-tagged hippocampal episode
written during the awake encode. "Consolidated concept layer" = the
sixteen-pool neocortical concept representation trained by the offline
replay. "Frozen verdict" = `research/runners/integrated_loop_core.py`.
"No-confabulation moat" = `research/runners/abstention_gate.py`. Each
term is defined once here and used consistently.

---

### Task 0: Grounding pin test (red until the phase-factored path lands)

**Files:**
- Modify: `tests/test_integrated_loop_gate.py` (add ONE new
  phase-factored grounding-pin test; do NOT alter the existing Task-0
  pin from the original plan, which already passes for `e02f692`).

**Context.** This new pin turns green only after Task 2 lands the
phase-factored path. It IS the Task-2 gate: it asserts the runner
exposes a phase-factored mode flag, its `--tiny-synth` smoke runs
end-to-end on the CPU backend, and the smoke verdict is the
explicitly-not-propagated TINY marker. Committing it red now and seeing
it go green after Task 2 is the proof Task 2 actually wired the
two-phase loop.

**Step 1: Write the failing test.** Add to
`tests/test_integrated_loop_gate.py`:

```python
def test_phase_factored_tiny_smoke_produces_tiny_verdict(tmp_path):
    """Grounding pin (phase-factored): the runner accepts the
    phase-factored flag, runs a fast --tiny-synth smoke end-to-end on
    the CPU backend, and writes a TINY-marked verdict JSON (never a
    real PASS/FAIL/VOID at toy scale)."""
    out = tmp_path / "tiny_pf.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--tiny-synth", "--phase-factored",
         "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "runner failed: %s\n%s" % (proc.stdout, proc.stderr))
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "tiny-synth verdict must be marked TINY / NOT propagated")
```

**Step 2: Run it to verify it fails.**
`pytest tests/test_integrated_loop_gate.py::test_phase_factored_tiny_smoke_produces_tiny_verdict -v`
Expected: FAIL (the `--phase-factored` flag does not exist yet, so the
runner errors with a non-zero return code). Intentional and correct at
Task 0.

**Step 3: Commit the red pin.**

```bash
git add tests/test_integrated_loop_gate.py
git commit -m "test: phase-factored grounding pin (red until the two-phase path lands)"
```

**Controller verification.** The commit-scoped `git diff` touches only
`tests/test_integrated_loop_gate.py`. The protected set is byte-empty
in the diff. Do NOT mark Task 0 "green" -- it is intentionally red and
is the Task-2 acceptance gate.

---

### Task 1: Confirm/spec the reused frozen verdict + reused validated subsystems

**Files:**
- Modify: `tests/test_integrated_loop_gate.py` (add ONE structural
  reuse-assertion test; no new pure module expected).

**Context.** The frozen verdict `integrated_loop_core.py` is reused
BYTE-UNCHANGED (unchanged since `2048750`; its own 16-case adversarial
matrix in `tests/test_integrated_loop_core.py` already passes and stays
byte-identical). No net-new pure verdict module is created. The
validated subsystems are reused by import, byte-unchanged. This task
pins, in the runner's test file, that the phase-factored runner reuses
exactly the named validated interfaces (no copy-edit, no autograd) and
imports the frozen verdict.

If, and only if, a tiny pure helper is genuinely unavoidable during
Task 2 (none is expected -- the phase controller is pure sequencing of
existing calls), it gets its OWN file with its OWN fixed-bar discipline
mirroring `integrated_loop_core.py` (standard library + typing only;
fixed constants justified a-priori and never tuned to a result; its own
adversarial test matrix; imports no other verdict module). Prefer
reuse; do not introduce a helper to avoid honest sequencing code.

**Step 1: Write the failing test.** Add to
`tests/test_integrated_loop_gate.py`:

```python
def test_runner_reuses_validated_phase_factored_parts():
    """The phase-factored runner composes the validated parts by
    import; it must not declare its own copies, must add no autograd,
    and must reuse the frozen verdict + the Phase-1.3 consolidation
    interface + the no-confab moat byte-unchanged."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "import torch" not in src and ".backward(" not in src
    assert "build_biological_brain_regions" in src
    assert "build_bg_brain_regions" in src
    assert "run_concept_replay_phase" in src
    assert "set_awake_gates" in src and "set_sleep_gates" in src
    assert "freeze_all_gates" in src
    assert "start_engram_recording" in src or "commit_engram_tag" in src
    assert "from research.runners.abstention_gate import" in src
    assert "from research.runners.integrated_loop_core import" in src
```

**Step 2: Run it to verify it fails.**
`pytest tests/test_integrated_loop_gate.py::test_runner_reuses_validated_phase_factored_parts -v`
Expected: FAIL (the runner at `e02f692` does not yet import
`run_concept_replay_phase` / `set_awake_gates` / `set_sleep_gates` /
`freeze_all_gates`; those are added in Task 2). Intentional at Task 1.

**Step 3: Commit the red structural pin.**

```bash
git add tests/test_integrated_loop_gate.py
git commit -m "test: pin phase-factored runner reuses the Phase-1.3 consolidation interface + frozen verdict byte-unchanged"
```

**Controller verification.** Commit-scoped diff touches only
`tests/test_integrated_loop_gate.py`. Protected set byte-empty.
`pytest tests/test_integrated_loop_core.py -q` is still 16/16
(byte-unchanged). `pytest tests/test_abstention_gate.py -q` is still
7/7.

---

### Task 2: The net-new per-trial phase controller + composition wiring

**Files:**
- Modify: `research/runners/integrated_loop_gate.py` (add the net-new
  phase-factored path on the `e02f692` foundation; the online
  encode/write path stays byte-identical to `e02f692`).
- Modify: `tests/test_integrated_loop_gate.py` (Task-0 + Task-1 pins
  must now go green; add a small structural test).

**Context.** This is the decisive net-new integration. It evolves the
`e02f692` runner by adding a phase-factored per-trial controller plus
the composition wiring. The net-new is ONLY: (a) a per-trial phase
controller that sequences the existing validated calls
(online-encode -> offline-consolidation -> consolidated readout), and
(b) the wiring that routes the committed online engram tag into
`run_concept_replay_phase` and routes the consolidated concept layer
into the working-memory readout, plus the lesion-6 design refinement
(read the episodic ORDER from the consolidated trace under the
validated freeze-then-evaluate idiom). Everything else is reused by
import, byte-unchanged. NO new learning rule. NO automatic
differentiation. GPU/CuPy is the real path; NumPy only for
`--tiny-synth`. Kill-safe. ASCII only.

**Reused interfaces (import byte-unchanged; do NOT modify):**

- `from research.runners.text_minimal_isolation import
  build_biological_brain_regions, set_awake_gates, set_sleep_gates,
  freeze_all_gates` -- the substrate built with
  `enable_hippocampus_consolidation=True` (gives `ec/dg/ca3/ca1` + the
  `ca1 -> concept` consolidation pathways + the `ca3_swr_burst`
  recurrent autoassociator gate) and `enable_dlpfc_verb=True` (the
  prefrontal NMDA-bistable working-memory slots), exactly as `e02f692`
  builds it. `set_awake_gates` = encoding ON, consolidation OFF;
  `set_sleep_gates` = input drive zeroed, direct lang->motor frozen,
  `ca3_swr_burst`/`ca1_to_motor`/`ca1_to_lang_out` = 1;
  `freeze_all_gates` = the validated pre-eval freeze. These are the
  Phase-1.3 awake/sleep idiom, reused unmodified.
- `from research.runners.consolidation_trainer import
  run_concept_replay_phase` -- the validated selective SWR replay over
  engram-tagged ensembles, `randomize_order=True` (shuffled offline
  replay; signature: `bridge, tag_names, n_replays_per_tag=...,
  burst_duration_ms=..., inter_burst_ms=..., drive_pA=...,
  randomize_order=True, rng=...`). This IS the offline-shuffle the
  concept mechanism structurally requires; reused unmodified.
- `from research.runners.g11_bg_runner import build_bg_brain_regions` --
  the validated basal-ganglia disinhibition cascade, its selective gate
  repurposed to the prefrontal/associative channel exactly as
  `e02f692` already wires it (the net-new `thal_<chan> -> dlpfc_verb`
  afferent is added in this runner's pathway list; the builder is
  byte-unchanged).
- The bridge engram interface (`start_engram_recording`,
  `commit_engram_tag(..., region_filter=["ec","dg","ca3","ca1"])`,
  `stimulate_tag`, `clear_tag_drive`, `delete_engram_tag`) -- the
  online theta-ordered store, reused exactly as `e02f692`.
- `from sim.neuromodulators import NeuromodulatorConfig,
  ProductionRule, ModulatorTarget` -- the phasic-dopamine-from-delta
  and acetylcholine plasticity-window modulators, constructed exactly
  as `e02f692` (`_da_modulator_from_delta`, `_ach_window_modulator`);
  construction only, subsystem byte-unchanged. Relegated to
  credit/gating, NOT binding.
- `from sim.train_checkpoint import save_checkpoint, load_checkpoint,
  resume_epoch` -- kill-safe per-seed checkpoint/resume, exactly the
  `e02f692` pattern.
- `from research.runners.abstention_gate import gate,
  DEFAULT_THRESHOLD` -- the no-confabulation moat at the working-memory
  output; the consolidated working-memory readout is emitted ONLY when
  the gate passes, else the loop abstains. Byte-unchanged.
- `from sim.text_embeddings import orthogonal_drive_pattern` -- the
  proven orthogonal role/filler code idiom, exactly as `e02f692`.
- `from research.runners.integrated_loop_core import
  integrated_loop_verdict` -- the FROZEN verdict, reused byte-unchanged.

**The net-new pieces (the only new code):**

1. **A per-trial phase controller** (a `--phase-factored` branch in
   `_episode`; pure sequencing, no learning logic of its own, no
   autograd). Per composition trial at load N, in order:
   - **ONLINE ENCODE phase -- BYTE-IDENTICAL to `e02f692`.**
     `set_awake_gates(bridge)`. Reuse the EXACT `e02f692` encode block
     (file lines ~953-1247): the one shared theta-gamma clock
     (`SharedThetaGamma`, unchanged) gates the basal-ganglia-selected
     prefrontal slot and times the engram write; role+filler orthogonal
     codes + teacher co-fire drive the bound pools;
     `start_engram_recording` is active; the assembly SHIFTS across
     theta cycles so presentation-order == binding-index;
     `commit_engram_tag(tag, region_filter=["ec","dg","ca3","ca1"])`
     finalizes the relational episode. This block is NOT rewritten --
     it is the same code path; the `--phase-factored` flag only changes
     what happens AFTER it. (Task 3 verifies the encode/write bytes are
     identical to `e02f692`.)
   - **MAINTAIN phase -- unchanged from `e02f692`** (the NMDA-bistable
     prefrontal hold; reward held at zero; clock refreshes within
     theta).
   - **OFFLINE CONSOLIDATION phase -- NET-NEW WIRING, validated
     subsystem.** `set_sleep_gates(bridge)`. Call
     `run_concept_replay_phase(bridge, tag_names=[<committed tag>],
     n_replays_per_tag=<bounded>, randomize_order=True,
     rng=<the per-trial numpy rng, NOT a fresh draw that perturbs
     cross-mode RNG faithfulness>)`. The validated `ca3_swr_burst`
     autoassociator + `ca1 -> concept` consolidation path replays the
     bound (role, filler) structure into the sixteen-pool concept layer
     in SHUFFLED order across many replay events -- exactly the
     encode-order the validated concept mechanism structurally
     requires, now legitimately supplied by the offline phase. For the
     `no_cls_replay` lesion this entire phase is skipped (the existing
     `e02f692` `no_cls_replay` branch already does the skip; the
     phase-factored branch keeps that skip but now the skip is
     load-bearing per gating lesion 6).
   - **WORKING-MEMORY READOUT from the CONSOLIDATED concept layer.**
     `freeze_all_gates(bridge)` (the validated Phase-1.3 pre-eval
     freeze). Then, for the queried role per mode (v1: the trivial
     drilled binding; full + every lesion: including the novel
     recombination, exactly as `e02f692`/the original plan's corrected
     `wm` bullet/design Section 5), drive the role code on
     `language_input` ONLY (no query-time teacher/external current into
     the concept pools or the prefrontal slots) and population-vote the
     consolidated filler concept pools. Emit the answer ONLY if the
     no-confabulation gate passes, else abstain. A wrong emission and
     an abstention on a groundable query both score 0; a correct gated
     emission scores 1. Accuracy over the N queries = `wm`. The
     selectivity here is LEARNED by the offline consolidation replay --
     it must be demonstrably absent before consolidation and present
     only after; it is NOT pre-wired into connectivity and NOT fed at
     query (the anti-hard-feed control; Task 3 verifies).
   - **EPISODIC-SEQUENCE READOUT from the consolidated trace (lesion-6
     design refinement).** After the offline phase, under the same
     `freeze_all_gates` freeze, recover the bound-pair ORDER from the
     consolidated pathway (the project's validated Phase-1.3
     freeze-then-evaluate idiom: the consolidated cortical
     representation carries the recalled content, as the strict-silence
     anti-cheat established). Score the recovered order against the
     true encode order. Accuracy = `ep`. The ONLINE theta-ordered
     ENCODE + the engram WRITE stay byte-unchanged from `e02f692`
     (presentation-order == binding-index at encode is exactly what
     makes the consolidated order recoverable); only the READOUT is
     taken post-consolidation. For `no_hippo_store` this is `0.0` by
     construction (no tag, nothing to consolidate, nothing to recall);
     for `no_sequencing` the recovered order is degenerate (collapses);
     for `no_cls_replay` there is no consolidated trace (collapses) --
     each exactly the gating-section mechanism.
   - **LEARN phase -- unchanged from `e02f692`** (the delayed reward
     drives the native eligibility-trace temporal-credit path with the
     validated constants `_GAMMA=0.95`, `_LAMBDA=0.9`; the
     clock-gated acetylcholine modulator times the update; this is
     credit/gating, NOT the binding rule).

2. **The composition wiring** connecting the online committed engram
   tag to `run_concept_replay_phase`'s `tag_names`, and the
   consolidated concept layer to the working-memory + episodic
   readouts. Wiring only. No new mechanism.

**Mode faithfulness (the `compose_bridge_gate`/`e02f692`
discipline).** `full`, `v1` (full with the no-gap trivial bind), and
each of the 8 lesions = the full phase-factored loop minus EXACTLY one
system, consuming the SAME random draws in the SAME order (only the
lesioned system's effect removed). The per-trial `numpy` rng is the
single `_make_pairs` consumer exactly as `e02f692`; `_episode` itself
draws no rng; passing that same rng into `run_concept_replay_phase`
must NOT change the cross-mode draw order (the replay rng usage is
identical across every mode that runs the offline phase, and
`no_cls_replay` skips it identically to how `e02f692` already skips
its replay -- a deterministic skip, not an extra draw). Each lesion is
exactly the gating-section ablation; a strawman crippled elsewhere is
a Task-3 reject.

**Scale ladder + scaffold (unchanged from `e02f692`).** Frozen ladder
`(2, 4, 8)`; `--seeds` default `[42, 43, 44, 45, 46]`; require `>= 3`
seeds or print `NOT-RUNNABLE` and return 2. `--tiny-synth` shrinks the
ladder to its first rung and shrinks pools/steps/epochs/replays so the
smoke completes fast on the NumPy CPU backend; the tiny verdict is
marked TINY and NEVER propagated (this is what makes the Task-0
phase-factored pin go green). `SIM_BACKEND` is set exactly as
`e02f692` (numpy for `--tiny-synth`; auto + `CUBLAS_WORKSPACE_CONFIG`
otherwise). Kill-safe per-seed checkpoint/resume exactly as `e02f692`.
The decisive output assembles per-rung
`{"N", "n_seeds", "v1":{wm,ep}, "full":{wm,ep}, "lesions":{...:{wm,ep}}}`
(mean over seeds), calls `integrated_loop_verdict(rungs)` from the
frozen core, writes the JSON, prints `GATE=... <honest-ceiling
banner>`. ASCII only. No `torch`, no autograd anywhere.

**Step 1: Make the Task-0 + Task-1 pins executable; add a structural
test (failing).** Add to `tests/test_integrated_loop_gate.py`:

```python
def test_phase_factored_runs_offline_after_online_before_readout():
    """Structural: the phase-factored path calls set_sleep_gates +
    run_concept_replay_phase AFTER the online encode/commit and BEFORE
    the consolidated readout, and freeze_all_gates before the readout
    (the validated Phase-1.3 freeze-then-evaluate idiom). The online
    encode/write path is byte-unchanged from e02f692 (verified
    separately by the controller's git-range check; this test pins the
    ordering markers exist)."""
    src = Path("research/runners/integrated_loop_gate.py").read_text()
    assert "--phase-factored" in src or "phase_factored" in src
    assert "run_concept_replay_phase(" in src
    assert "set_sleep_gates(" in src and "freeze_all_gates(" in src
```

Run: `pytest tests/test_integrated_loop_gate.py -v` -> the Task-0
phase-factored pin, the Task-1 reuse pin, and this structural test all
FAIL (the `--phase-factored` path does not exist yet).

**Step 2: Implement the phase-factored path** in
`research/runners/integrated_loop_gate.py` per the behavioral spec
above. Add the `--phase-factored` flag; in `_episode`, branch into the
phase controller (online encode block byte-identical to `e02f692`;
then `set_sleep_gates` -> `run_concept_replay_phase` ->
`freeze_all_gates` -> consolidated wm readout -> consolidated ep
readout); add the composition wiring. Reuse-by-import only; modify none
of the protected set. NO autograd.

**Step 3: Run the smoke + structural tests.**
`pytest tests/test_integrated_loop_gate.py -v`
Expected: the Task-0 phase-factored pin, the Task-1 reuse pin, and the
Step-1 structural test all PASS (the runner exposes `--phase-factored`,
the `--tiny-synth --phase-factored` smoke produces a TINY-marked
verdict end-to-end). Run the tiny phase-factored smoke directly once
and read the JSON to confirm a verdict object with a `GATE` field and
the TINY marker. The existing (non-phase-factored) Task-0 pin from the
original plan must STILL pass (the `e02f692` path is untouched when
`--phase-factored` is absent).

**Step 4: Commit.**

```bash
git add research/runners/integrated_loop_gate.py tests/test_integrated_loop_gate.py
git commit -m "feat: phase-factored per-trial controller -- online theta-ordered encode + separate offline shuffled-replay consolidation; WM from consolidated cortex, EP order from consolidated trace"
```

**Controller verification (trust-but-verify).** The commit-scoped
`git diff` touches only `research/runners/integrated_loop_gate.py` and
`tests/test_integrated_loop_gate.py`. Every protected path is byte-empty
in the diff AND across `git diff bd27292..HEAD`. The runner contains no
`import torch` / `.backward(` / autograd. `build_biological_brain_regions`,
`build_bg_brain_regions`, `run_concept_replay_phase`, `set_awake_gates`,
`set_sleep_gates`, `freeze_all_gates` are imported, not redefined.
`pytest tests/test_integrated_loop_core.py -q` is still 16/16,
byte-identical. `pytest tests/test_abstention_gate.py -q` is still 7/7.
The `--tiny-synth --phase-factored` verdict is marked TINY. The
controller diffs the online encode/write block against `e02f692` and
confirms it is byte-identical (only post-MAINTAIN code changed under
the `--phase-factored` branch).

---

### Task 3: Dedicated adversarial review (BEFORE the no-harm phase)

**Files:** none modified by the reviewer subagent. Strengthen-only
fixes (if any) are applied by a follow-up implementer subagent and
re-reviewed; the frozen verdict, partition, bars, and moat stay
byte-unchanged.

**Context.** Dispatch a fresh subagent as a dedicated adversarial
reviewer of `research/runners/integrated_loop_gate.py` (the
phase-factored path) against
`research/runners/integrated_loop_core.py` (frozen). Its job is to
find holes, not to bless. It produces a written report; the controller
decides on strengthen-only fixes. Do NOT proceed to Task 4 until the
review has no open issues.

**The reviewer MUST independently scrutinize and answer, with
file:line evidence:**

1. **The frozen-lesion-partition interaction (re-derive the gating
   section independently).** For EACH of the 8 frozen lesions, does it
   genuinely collapse exactly the readout(s) its frozen
   `integrated_loop_core` membership requires under the phase-factored
   wiring -- and is none of them INVERTED? Specifically and most
   importantly: does `no_cls_replay` (skip offline consolidation)
   collapse the EPISODIC readout (its frozen `_HELPER_EP` duty), given
   that the episodic order is read from the consolidated trace -- or
   does the episodic readout in the shipped code still read from a
   hippocampus-only `stimulate_tag` (which would NOT collapse under
   `no_cls_replay`, inverting it to a `_HELPER_WM`-like lesion and
   making the run VOID-by-construction)? If the latter, this is a hard
   reject and the gating section is re-classified outcome (b).
2. **No query-time hard-feed.** The consolidated working-memory
   selectivity must be LEARNED by the offline replay -- demonstrably
   absent before consolidation and present only after -- not pre-wired
   into connectivity and not fed at query (no teacher/external current
   into the concept pools or prefrontal slots during the readout). The
   episodic order must be recovered, not supplied.
3. **Lesion faithfulness.** Each lesion = the full phase-factored loop
   minus EXACTLY one system, consuming IDENTICAL per-trial RNG draws in
   IDENTICAL order (only the lesioned system removed). Specifically:
   passing the per-trial rng into `run_concept_replay_phase` must not
   shift the cross-mode draw order; `no_cls_replay`'s skip of the
   offline phase must be a deterministic skip identical in RNG effect
   to how `e02f692` already skips its replay (no extra/!missing draw);
   `no_shared_clock` is truly one-clock -> two-clocks with nothing else
   changed; the helper lesions are not strawmen crippled elsewhere.
4. **Validated subsystems reused byte-unchanged, not copy-edited.**
   `build_biological_brain_regions`, `build_bg_brain_regions`,
   `run_concept_replay_phase`, `set_awake_gates`, `set_sleep_gates`,
   `freeze_all_gates`, the engram interface, the neuromodulator
   subsystem, the native eligibility/temporal-credit path, the
   checkpoint module, the no-confabulation moat, and the frozen
   verdict -- all imported byte-unchanged (confirm via
   `git diff bd27292..HEAD` over the protected set being empty).
5. **The online encode/write path is byte-identical to `e02f692`.**
   The `--phase-factored` flag changes ONLY what happens after
   MAINTAIN; the theta-ordered encode + engram write (the part whose
   order constraint conflicts with the concept mechanism) is the same
   code path as `e02f692` (so the episodic encode cannot silently
   degrade and the program-level encode-order conflict is genuinely
   dissolved by phase separation, not by a relabeled online pass).
6. **No new automatic differentiation/training.** Grep the runner and
   its import graph for `torch`, `backward`, autograd, any gradient
   objective -- must be none; every learning update is the reused
   validated local rule.
7. **The frozen bars/partition/moat are unmovable by results.** Confirm
   `integrated_loop_core.py` is byte-unchanged since `2048750`, its
   16-case matrix is byte-unchanged and green, the abstention moat is
   byte-unchanged and 7/7, and no `_IL_*` value or the
   `_SHARED/_HELPER_*` partition is touched.
8. **A broken/unsound run cannot be scored a success.** Trace the
   verdict precedence on the recorded JSON shape: a v1-unsound run, a
   non-discriminating run (a frozen lesion that did not collapse its
   required readout -- especially an inverted `no_cls_replay`), a
   malformed/NaN record -- each must be VOID, never PASS/FAIL.

**Step.** Dispatch the reviewer subagent with the eight probes above as
its explicit charter. It returns a report. The controller applies
strengthen-only fixes via a follow-up implementer subagent (frozen
verdict/partition/bars/moat byte-unchanged; the online encode/write
path byte-unchanged from `e02f692`), then re-dispatches the reviewer
until the report has no open holes. If the reviewer concludes the
lesion-6 refinement cannot be implemented faithfully without editing
the frozen verdict/partition or a strawman lesion, the gating section
is re-classified **outcome (b)**: Task 5 is NOT run; the controller
surfaces a program-level VOID-by-construction finding and the next
catalog-identified factorization, autonomously, no hand-back. Commit
any strengthen-only fixes with a clear message; the controller verifies
the protected set stays byte-empty in the diff. Do NOT proceed to Task
4 until the review has no open issues AND the gating section remains
outcome (a).

---

### Task 4: No-harm phase

**Files:** none created/modified. Verification only; produces a short
evidence note appended to the eventual findings doc (Task 5).

**Step 1: Prove the protected set is byte-unchanged across the whole
branch.** Run `git diff --stat bd27292..HEAD` and confirm the ONLY
changed paths are `research/runners/integrated_loop_gate.py` and
`tests/test_integrated_loop_gate.py` (plus, only after Task 5, the
findings doc + the capability-status pillar). Explicitly confirm
byte-empty for every protected path:

```bash
git diff --stat bd27292..HEAD -- research/runners/integrated_loop_core.py tests/test_integrated_loop_core.py research/runners/abstention_gate.py tests/test_abstention_gate.py research/runners/text_minimal_isolation.py research/runners/consolidation_trainer.py research/runners/g11_bg_runner.py research/runners/compose_bridge_core.py research/runners/compose_bridge_gate.py research/runners/q2r_core.py research/runners/q2r_gate.py research/runners/constrained_decode_gate.py research/runners/engram_bootstrap_gate.py sim/bridge.py sim/kernels.py sim/neuromodulators.py sim/train_checkpoint.py sim/backend.py sim/regions.py sim/text_embeddings.py
```

Expected: empty output (no protected file changed).

**Step 2: The no-confabulation moat still passes 7/7.**
`pytest tests/test_abstention_gate.py -q` -> 7 passed.

**Step 3: The frozen verdict matrix still passes byte-unchanged.**
`pytest tests/test_integrated_loop_core.py -q` -> 16 passed; confirm
`git diff bd27292..HEAD -- research/runners/integrated_loop_core.py
tests/test_integrated_loop_core.py` is empty.

**Step 4: The runner test suite is green.**
`pytest tests/test_integrated_loop_gate.py -q` -> all passed (the
original Task-0 pin, the phase-factored Task-0 pin, the Task-1 reuse
pin, the Task-2 structural test).

**Step 5: Assert no shipped path imports autograd.** Grep
`research/runners/integrated_loop_gate.py` for `torch`, `.backward(`,
`autograd`, `grad(` -- no matches in any shipped code path (a docstring
mention of "no autograd" is acceptable; an actual import/call is a hard
stop -> back to Task 2).

Do NOT proceed to Task 5 until all five steps pass. If Step 1, 2, or 3
fails, the build harmed the protected set -- stop, revert the offending
change, redo the task that caused it.

---

### Task 5: CONTROLLER-ONLY decisive multi-seed GPU run

This task is performed by the controller directly, never delegated to a
subagent. **It is conditioned on the gating section resolving to
outcome (a) and remaining (a) after the Task 3 adversarial review.** If
the gating section is outcome (b) (VOID-by-construction), Task 5 is NOT
run; the controller surfaces the program-level finding (the next
catalog-identified factorization) and continues autonomously, no
hand-back.

**Step 1: Falsify-first smoke + grounding run (numbers NOT reported).**
First run the design's cheapest falsify-first de-risk: a single-seed,
minimal-load (N=2) GPU smoke of just the online-encode -> offline
-> consolidated-readout sequence, measuring (a) the episodic score
stays at 1.0 after the offline phase is inserted and (b) the
consolidated working-memory readout is role-selective above chance. If
the episodic score does NOT stay at 1.0 once the offline phase is
inserted, the factoring assumption is wrong: fire the pre-registered
bound immediately with that exact GPU-measured structural cause -- no
config crank, no Candidate B/C reflex; surface the program-level result
and the next catalog factorization. If the smoke is healthy, also run
the `--tiny-synth --phase-factored` smoke once on the decisive machine
and confirm return code 0 + a TINY-marked verdict. These numbers are a
health check ONLY and are explicitly not reported as a result.

**Step 2: The decisive multi-seed run at increasing compositional
load.** Fixed pre-registered configuration: the frozen ladder
`(2, 4, 8)`, seeds `42 43 44 45 46`, full (non-tiny) scale,
`--phase-factored`. Kill-safe and monitored to ACTUAL completion (a
genuine completion notification or foreground run -- never a detached
process with a false "I will be notified" claim; completion actively
confirmed by polling the output JSON existence + the process state
before any result is stated).

```bash
python -m research.runners.integrated_loop_gate --phase-factored --seeds 42 43 44 45 46 --ckpt research/findings/raw/integrated_loop_pf_ckpt --out research/findings/raw/integrated_loop_pf_decisive.json
```

If interrupted, resume the same command (kill-safe). Do not state any
result until the JSON is written and the process has genuinely exited
0.

**Step 3: Mandatory anti-cheat smell-test -- scrutinize a nominal PASS
HARDER than a FAIL.** Recompute the verdict from the single recorded
JSON WITHOUT re-running and WITHOUT changing any threshold. Confirm by
hand from the recorded numbers:

- The full phase-factored loop genuinely clears `_IL_SCI_MIN` on BOTH
  readouts at every rung.
- Each single-system lesion genuinely collapses the readout its frozen
  `integrated_loop_core` membership requires -- and crucially
  `no_cls_replay` collapses `ep` (its frozen `_HELPER_EP` duty), NOT
  inverted to a WM-only collapse.
- The three `_SHARED` lesions (`no_binding`, `no_shared_clock`,
  `no_hippo_store`) and `no_neuromod_timing` (`_HELPER_BOTH`) collapse
  BOTH readouts together at every rung (the decisive
  emergent-from-integration signature).
- Every drilled binding clears the byte-unchanged no-confabulation gate
  (DEFAULT_THRESHOLD = 650.0); the full+lesions novel probe is
  byte-identical (same RNG draws); instrument soundness (`v1` wm AND
  ep >= 0.90) is met at every rung.
- Composition is non-decreasing up to tolerance across the ascending
  load ladder and holds at the largest load (for a SCALE-CONFIDENT-
  PASS) -- or, honestly, where it does not.
- The classification returned by `integrated_loop_verdict` recomputed
  from the JSON matches what the recorded numbers imply. No re-run, no
  bar tuning, no overclaim. A nominal PASS gets MORE scrutiny than a
  FAIL, not less. If anything is off, the honest classification stands
  (VOID/FAIL/WORKS-SMALL) and is propagated as such.

**Step 4: Honest propagation of EVERY outcome (plain professional
language).**

- Write
  `research/findings/2026-05-19-phase-factored-consolidation-<outcome>.md`
  in plain professional language (computational neuroscientist briefing
  an informed colleague; no codenames as load-bearing terms; every
  technical term defined once). State exactly what the run showed, the
  recomputed verdict, the honest ceiling, and what is and is not
  claimed. A FAIL or VOID is an honest, valuable finding -- write it as
  such, not as a setback to spin. Append the Task-4 no-harm evidence
  (protected set byte-empty bd27292..HEAD; no-confab moat 7/7;
  `integrated_loop_core` byte-unchanged since `2048750`; no autograd in
  shipped paths).
- Update the capability-status panel: flip the existing integration
  pillar to the decisive outcome (VALIDATED if SCALE-CONFIDENT-PASS;
  otherwise the honest classification) in plain language. Do not append
  a new pillar number; update the existing one in place.
- Run the capability-status schema test green:
  `pytest tests/test_webapp_server.py -k capability_status -q`.
- Commit and push to BOTH remotes (origin and gitea).

**Step 5: Continue autonomously per the reference biology (no
hand-back).**

- **Clean SCALE-CONFIDENT-PASS:** proceed to the design's next staged
  integration step -- the design's pre-described Candidate B
  (consolidation-gated prefrontal maintenance) only if a smoke shows
  the bare consolidated readout is real but sub-gate, then Candidate C
  (multi-cycle offline schedule) only if B is real-but-insufficient --
  each its own pre-registered gate, each built by returning to
  writing-plans then subagent-driven-development. Do not stop; do not
  declare victory beyond the honest ceiling.
- **Honest non-success (FAIL / WORKS-SMALL-NO-SCALE-CONFIDENCE /
  VOID):** do NOT declare the approach unfit and do NOT hand back. If
  the design's Candidate B / Candidate C in-architecture escalations
  are genuinely warranted by an honest, propagated signal (never a
  reflexive config crank), take them with the SAME frozen acceptance
  and the SAME adversarial discipline. Otherwise the pre-registered
  bound fires: surface the precise GPU-measured structural cause as a
  program-level result; the next step is the next catalog-identified
  integration factorization (a deeper separation of relational binding
  from schema abstraction along the catalog's hippocampal-neocortical
  interaction entries), pursued autonomously with the SAME frozen
  acceptance. Bounded only by honest exhaustion of cited biological
  refinements -- and even then the next step is the next
  catalog-identified gap, autonomously, no hand-back, no
  declare-globally-unfit, no config-cranking a frozen bar/partition/
  moat.

Bring Task 5 back to the controller. Tasks 0-3 are subagent-driven;
Task 4 is controller-verified; Task 5 is controller-only and
conditioned on gating = outcome (a).

---

## Remember

- One fresh subagent per task; strict failing-test -> minimal-
  implementation -> run -> commit.
- The gating section is load-bearing: the build does NOT proceed to
  Task 5 unless it resolves to outcome (a) and the Task-3 adversarial
  review independently confirms it. Outcome (b) means a program-level
  VOID-by-construction finding, autonomously, no hand-back.
- Controller trust-but-verify EVERY commit: the protected set is
  byte-empty in the commit-scoped diff AND across `bd27292..HEAD`. The
  frozen verdict (`integrated_loop_core.py`, since `2048750`), its
  16-case matrix, the `_SHARED/_HELPER_*` partition, every `_IL_*` bar,
  and the no-confabulation moat are byte-unchanged.
- Net-new = ONLY the per-trial phase controller (pure sequencing of
  existing validated calls) + the composition wiring + the lesion-6
  consolidated-episodic-readout design refinement. No new learning
  rule, no automatic differentiation, no new plasticity, no new sim
  module.
- The online theta-ordered ENCODE + the engram WRITE stay byte-
  identical to `e02f692`; only the post-MAINTAIN phase
  (offline-consolidation -> consolidated readouts) is net-new.
- Acceptance is the SAME pre-registered frozen gate. The pre-registered
  bound (a faithful build that cannot reach `v1 wm AND ep >= 0.90` with
  the lesion contrasts discriminating -> honest program-level result +
  next catalog factorization, never a config crank) is stated in
  advance.
- Plain professional language in every artifact and commit message;
  honest propagation of every outcome to both remotes; iterate
  following the reference biology on any non-success; no hand-back.
- The honest ceiling is stated and never overstated.
