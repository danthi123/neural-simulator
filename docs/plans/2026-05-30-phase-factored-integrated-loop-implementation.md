# Phase-factored integrated closed-loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Task 6 is CONTROLLER-ONLY (not a subagent task). There is a HARD GATE after Task 1: the spiking-build tasks (2+) run ONLY if the Task 1 cheap probe shows the factorization resolves the conflict.

**Goal:** Test whether splitting compositional encoding into two phases — a fast online phase that records episode order, and a slow offline phase that builds concept selectivity by replaying episodes in shuffled order — resolves the encode-order conflict that stalled the single-pass integrated loop, first cheaply (numpy), then, only if that clears, in the real spiking network.

**Architecture:** A cheap falsification probe (Task 1) models the two phases plus the two readouts (a concept query and an order query) and the one structural risk — the online order-index points at concept representations that the offline phase then moves, so the index may stop resolving unless the offline phase also updates it. If the probe shows the split resolves the conflict (and a single-pass control reproduces the conflict, so the pass is not trivial), Tasks 2+ build the two-phase controller in the spiking bridge, reusing four already-validated subsystems unchanged, scored by the parked loop's already-reviewed frozen verdict module.

**Tech Stack:** Python, numpy (cheap probe, CPU-only); the project's CuPy spiking bridge + `build_biological_brain_regions` + engram-tagging API + Phase-1.3 consolidation + v16 concept-binding + abstention gate + the theta-gamma timing controller (spiking build); pytest.

**Plain-language glossary (defined once):**
- *Concept selectivity* — a stable set of neurons that responds to one concept (e.g. "apple") and not others. Built by the validated co-firing + topographic-prior mechanism, which needs **shuffled** presentation across examples to avoid one concept dominating.
- *Episode order* — the sequence in which concepts were presented (apple then river). Recovered from the **order** of activation across gamma sub-cycles within one theta period (the timing rhythm).
- *Online phase* — encoding that happens once, in real time, in presentation order (the hippocampal index).
- *Offline phase* — replay-driven consolidation that happens afterwards, in shuffled order, building cortical selectivity (the validated Phase-1.3 SWR replay).
- *The conflict* — selectivity needs shuffle, order needs no-shuffle; one online pass cannot do both. (Established: `research/findings/2026-05-19-integrated-loop-PROGRAM-LEVEL-encode-order-conflict-...md`.)
- *Residual coupling* — even with the phases split, the online index points at concept representations that the offline phase moves to build selectivity; if the index is content-addressed and is not updated, it may stop resolving. The probe tests whether the offline phase updating the index (biologically: consolidation strengthening the hippocampus->cortex pathway) dissolves this coupling.

---

## Task 0: Grounding pin (red until Task 2; intentional)

**Files:**
- Create: `tests/test_phase_factored_loop_grounding.py`

A pin test that fixes the contracts up front so the implementation cannot drift. It checks: (a) the cheap probe module exists and exposes `run_probe()` + `probe_verdict()`; (b) the cheap probe's frozen bar constant is exactly 0.90 and is a module-level constant (not a function argument); (c) the spiking controller module exists and reuses the four validated subsystems by import (string-grep for the import lines), and imports the parked frozen verdict `integrated_loop_core.integrated_loop_verdict`; (d) the parked verdict module `research/runners/integrated_loop_core.py` is byte-identical to its committed form (pin its frozen bars: `_IL_V1_MIN==0.90`, `_IL_SCI_MIN==0.80`, `_IL_LESION_MAX==0.40`, `_IL_LADDER==(2,4,8)`).

**Steps:**
1. Write the test with `pytest.skip(...)` guards: parts (a)-(c) skip if their target file does not exist yet (so Task 0 is green-with-skips now and turns fully green as Tasks 1-2 land); part (d) runs immediately (the parked module already exists).
2. Run: `pytest tests/test_phase_factored_loop_grounding.py -v`. Expected: part (d) PASS, parts (a)-(c) SKIP.
3. Commit: `git add tests/test_phase_factored_loop_grounding.py && git commit -m "Task 0: phase-factored loop grounding pin (parts a-c skip until Tasks 1-2)"`.

---

## Task 1: Cheap-first falsification probe (LOAD-BEARING GATE; fully specified)

**Files:**
- Create: `research/findings/raw/phase_factored_cheap_probe.py`
- Test: `tests/test_phase_factored_cheap_probe.py`

This is the de-risking gate. It must (i) reproduce the conflict with a single-pass CONTROL (so a two-phase pass is a genuine falsification), (ii) test the residual coupling, (iii) emit a three-state verdict against a frozen 0.90 bar with instrument-validity checked first. Standard library + numpy only; NO import of any protected/verdict module.

### The model (transcribe exactly)

`n_concepts=16`, vector dim `D=64`, episode length `N=2`, `n_trials=400`, `n_distractors_per_trial=4`, seeds `(42,43,44)`.

Concept representations: `rep[i]` = a fixed unit-norm random vector in R^D (seeded). Two readouts, each scored as accuracy in [0,1] over trials:
- **wm (concept query):** present an in-buffer concept and a set of out-of-buffer distractors; the selectivity classifier must rank the in-buffer concept above all distractors. Selectivity quality is governed by a scalar `sep in [0,1]` = how orthogonalized the reps are (0 = raw random reps, overlapping; 1 = fully pattern-separated). Model wm accuracy as a smooth increasing function of `sep`, calibrated so raw reps (`sep≈0.15`, random-overlap regime) give wm≈0.55 (just above chance for the ranking) and well-separated reps (`sep≥0.6`) give wm≥0.95.
- **ep (order query):** recover which concept was at position 0 vs 1, from the order-index. Model ep accuracy from `idx_fidelity in [0,1]` = how well the index still resolves the two positions to the correct concepts. ep≈0.5 at idx_fidelity 0 (chance for 2 positions), ep≥0.95 at idx_fidelity≥0.9.

**Single-pass control — the conflict (one knob `phi in [0,1]` = shuffle degree).**
A single online pass has ONE presentation order. `phi=0` (strictly ordered): the index is perfect (`idx_fidelity=1 -> ep high`) but selectivity suffers winner-take-most under correlated ordered presentation (`sep` low -> wm≈chance). `phi=1` (fully shuffled): selectivity builds well (`sep` high -> wm high) but the order-index is destroyed (`idx_fidelity≈0 -> ep≈chance`). Model `sep(phi)` increasing and `idx_fidelity(phi)=1-phi` decreasing, monotone opposed. Compute `single_pass_best = max over phi in {0,0.1,...,1.0} of min(wm(sep(phi)), ep(idx_fidelity(phi)))`. The conflict is real iff `single_pass_best < 0.90`.

**Two-phase treatment — three index variants (the residual-coupling test).**
Phase 1 builds the order-index from ORDERED presentation (`idx_fidelity_raw=1`). Phase 2 builds selectivity from SHUFFLED replay (`sep=0.7 -> wm high`) and, in doing so, MOVES the reps (rep -> rep', moved by an amount tied to `sep`). The index must still resolve after the move. Three variants:
- `two_phase_pointer`: the index stored a concept-IDENTITY pointer (sparse id), independent of the rep vector. The move does not affect it: `idx_fidelity=1 -> ep high`. (Biological analogue: a hippocampal sparse index code.)
- `two_phase_content_noupdate`: the index stored the original rep VECTOR (content) and is NOT updated by Phase 2. After rep->rep', matching the stored old vector to the moved reps degrades: `idx_fidelity = overlap(rep, rep')`, which falls as `sep` rises -> ep drops. (Tests whether the coupling SURVIVES the split.)
- `two_phase_content_update`: the index stored content but Phase 2 (consolidation) UPDATES it to track rep->rep' (biological analogue: consolidation strengthening the hippocampus->cortex pathway). `idx_fidelity` restored to ≈1 -> ep high.

For each variant compute `min(wm, ep)`.

### `run_probe(seed) -> dict`
Returns `{"single_pass_best": float, "two_phase_pointer": float, "two_phase_content_noupdate": float, "two_phase_content_update": float, "wm_at_sep07": float, "ep_pointer": float}` — all finite floats in [0,1], deterministic given seed.

### `probe_verdict(per_seed: list[dict]) -> dict` (frozen, three-state)
Module-level frozen constant `_PROBE_BAR = 0.90`; `_PROBE_MIN_SEEDS = 3`. Instrument-validity FIRST, fail-closed, never raises:
- malformed / non-list / < 3 seeds / non-finite field -> `{"verdict":"CANNOT_CONCLUDE", ...}` (VOID-not-crash).
- **Instrument-validity gate:** if multi-seed-mean `single_pass_best >= _PROBE_BAR` -> `CANNOT_CONCLUDE` ("control did NOT reproduce the conflict; the cheap model does not capture the encode-order tradeoff, so it cannot test the resolution"). This makes a two-phase PASS a genuine falsification.
- Else (conflict reproduced): let `tp = multi-seed-mean two_phase_content_update` (the biologically-faithful variant — content index updated by consolidation). 
  - `tp >= _PROBE_BAR` -> `{"verdict":"RESOLVES", ...}`: the split, WITH consolidation updating the index, resolves what one pass cannot. (Also report whether `two_phase_content_noupdate` falls below bar — the expected demonstration that the coupling is real and is dissolved specifically by the index-update.)
  - `0.80 <= tp < 0.90` -> `{"verdict":"BOUNDARY", ...}`: partial resolution.
  - `tp < 0.80` -> `{"verdict":"DOES_NOT_RESOLVE", ...}`: the factorization is insufficient even with index-update; a deeper blocker — a high-value NEGATIVE to propagate.

### Adversarial test matrix (>= 12 cases) — `tests/test_phase_factored_cheap_probe.py`
1. `run_probe(42)` returns all six finite floats in [0,1].
2. determinism: `run_probe(42) == run_probe(42)`.
3. single_pass_best < 0.90 for each seed (the conflict is reproduced).
4. two_phase_content_update >= 0.90 for each seed (resolves with index-update).
5. two_phase_content_noupdate < two_phase_content_update for each seed (the coupling is real and the index-update is what dissolves it).
6. two_phase_pointer >= 0.90 (a sparse-id index also resolves).
7. `probe_verdict` on 3 good seeds -> "RESOLVES".
8. instrument-validity: a hand-built per_seed with single_pass_best=0.95 -> "CANNOT_CONCLUDE".
9. malformed (None / not-a-list / empty) -> "CANNOT_CONCLUDE", no raise.
10. < 3 seeds -> "CANNOT_CONCLUDE".
11. non-finite field (NaN/inf/string/bool) -> "CANNOT_CONCLUDE".
12. hand-built tp=0.85 -> "BOUNDARY"; tp=0.70 -> "DOES_NOT_RESOLVE"; tp=0.93 -> "RESOLVES" (bar-edge pins).
13. `_PROBE_BAR == 0.90` and is module-level (frozen-bar pin).

**Steps:** write the test matrix (fails: module missing) -> run to confirm fail -> implement the probe + verdict to satisfy the matrix -> run to PASS -> commit.

### HARD GATE (controller, after Task 1 lands + its test passes)
Run `run_probe` over seeds (42,43,44), feed to `probe_verdict`, and record the result to `research/findings/raw/phase_factored_cheap_probe_result.json`:
- verdict **RESOLVES** -> proceed to Task 2.
- verdict **BOUNDARY / DOES_NOT_RESOLVE / CANNOT_CONCLUDE** -> STOP the spiking build. Write a findings doc (`research/findings/2026-05-30-phase-factored-cheap-probe-<verdict>.md`) propagating the honest result + its precise meaning (the two-phase factorization is necessary but not sufficient, or the model could not test it), commit + push both remotes, and surface to the controller. Do NOT build Tasks 2+ on a falsified premise.

---

## Task 2: Two-phase controller + order-preserving index readout (spiking; CONDITIONAL on Task 1 = RESOLVES)

**Files:**
- Create: `research/runners/phase_factored_loop_gate.py`
- Reference (reuse by import, byte-unchanged): the engram-tag API (`bridge.start_engram_recording/commit_engram_tag/stimulate_tag`), Phase-1.3 consolidation (`research/runners/consolidation_trainer.py`), v16 concept-binding (`research/runners/concept_pool_demo.py` train_word_to_pool + topographic prior), the abstention gate (`research/runners/abstention_gate.py`), the theta-gamma timing controller (from the parked loop: `research/runners/integrated_loop_gate.py`), `build_biological_brain_regions` (`research/runners/text_minimal_isolation.py`), the spiking bridge + checkpoint module.

The genuinely net-new piece is the **two-phase controller**: it runs Phase 1 (online, theta-ordered engram-tag bind of the length-N sequence; gamma sub-cycle k binds item k) THEN Phase 2 (offline, shuffled SWR replay via the consolidation trainer, building concept selectivity), then evaluates the two readouts and the lesion variants. The order-preserving index readout (Readout 2) recovers sequence from the gamma-slot order of the index replay. Mirror `integrated_loop_gate.py`'s kill-safe scaffold; reuse its theta-gamma controller; do NOT modify it.

**Behavioural spec (tests in `tests/test_phase_factored_loop_gate.py`, --tiny-synth scale, CPU/numpy bridge where possible):**
- The controller exposes `run_rung(N, seed) -> dict` returning the rung dict shape the frozen verdict consumes: `{"N":N, "n_seeds":1, "v1":{...}, "full":{...}, "lesions":{<all 7 names>:{...}}}`.
- Phase ordering pin: Phase 2 must run AFTER Phase 1 (assert the controller calls online-bind before offline-consolidate; a swapped order is a bug).
- Lesion fidelity pin: each lesion variant is identical to the full run minus exactly one subsystem, same RNG draws (assert seed/draw parity between `full` and each `lesion` except the ablated call).
- `no_shared_clock` lesion must drive BOTH readouts toward chance (the non-separability signature) — pin that the lesion actually disables the shared theta-gamma controller, not a no-op.
- No autograd: assert no `torch`/`.backward()` import in the shipped path.
- Reuse pin: assert the four validated subsystems + theta-gamma controller are imported, not reimplemented.

**Steps:** TDD each pin (write failing test -> minimal wire-up -> pass -> commit). Keep each subsystem call a thin reuse. Commit per pin.

---

## Task 3: Frozen verdict reuse (inherit the parked reviewed module)

**Files:**
- Test: `tests/test_phase_factored_verdict_reuse.py`

No new verdict logic. Task 2's `run_rung` emits the rung shape that `integrated_loop_core.integrated_loop_verdict` already scores. This task pins that the spiking gate feeds the inherited frozen verdict unchanged and that the bars are not shadowed/overridden.

**Steps:**
1. Test: import `integrated_loop_verdict`; assert frozen bars unchanged; feed a synthetic 3-rung PASS-shaped input -> "PASS"; feed a shared-lesion-not-collapsing input -> "VOID"; feed below-bar-at-N=2 -> "FAIL"/"FAIL". Assert `phase_factored_loop_gate` imports `integrated_loop_verdict` and does not define its own bars.
2. Run -> PASS. 3. Commit.

---

## Task 4: Dedicated adversarial review of the load-bearing glue (BEFORE the decisive run)

Dispatch an adversarial reviewer (subagent) on: the Task 1 cheap probe + its verdict, and the Task 2 two-phase controller + the inherited verdict reuse. Scrutiny: is the cheap probe's conflict genuinely reproduced (not a rigged single-pass)? Is the residual-coupling test honest (does `content_noupdate` really fall and `content_update` really restore, or are they hardcoded)? Is each spiking lesion faithful (full-minus-exactly-one, same draws)? Can a broken/unsound run score a PASS (verdict VOID-gates it)? Are the four subsystems genuinely reused byte-unchanged? Any autograd? Are the frozen bars movable by results? Reviewer returns CLEAR or BLOCK with strengthen-only fixes. Apply fixes; re-review if BLOCK. Commit the verdict file.

---

## Task 5: No-harm phase (protected set byte-unchanged)

**Files:**
- Test: `tests/test_phase_factored_no_harm.py`

Prove the protected set is byte-empty in `git diff` vs the pre-arc commit: `research/runners/abstention_gate.py` + its 7/7 test, `research/runners/integrated_loop_core.py`, `research/runners/consolidation_trainer.py`, `research/runners/concept_pool_demo.py`, `research/runners/text_minimal_isolation.py` (build_biological_brain_regions), `sim/bridge.py`, `sim/kernels.py`, the engram-tag API, the theta-gamma controller. Assert `pytest tests/test_abstention_gate.py` is still 7/7. Assert no shipped path imports `torch.autograd`/`.backward()`.

**Steps:** write the diff + 7/7 + no-autograd assertions -> run -> PASS -> commit.

---

## Task 6: CONTROLLER-ONLY decisive multi-seed run (NOT a subagent task — bring back to the controller)

Not a subagent task. The controller:
1. A small grounding run first (tiny-synth; toy numbers NOT propagated as a result) to confirm the pipeline runs end-to-end on all rungs + all 7 lesions.
2. The decisive kill-safe multi-seed run: seeds (42,43,44), ladder N=(2,4,8), full + all 7 lesion variants per rung, recorded to one JSON. KILL-SAFE per (N,seed) cache (this in-bridge two-phase run is heavy). Monitored to actual completion (foreground or genuine background-completion notification — never a detached process with a false "will be notified").
3. Feed the single recorded JSON to `integrated_loop_verdict`. Then the MANDATORY smell-test (scrutinize a nominal PASS HARDER than a FAIL): recompute every load-bearing number from the recorded JSON; confirm instrument soundness (v1 both readouts >= 0.90); confirm each lesion genuinely collapses its responsibility; confirm the three shared-system lesions collapse BOTH readouts together; no re-run, no bar change.
4. Honest propagation of EVERY outcome (PASS / WORKS-SMALL / FAIL / VOID): findings doc + capability_status pillar entry + schema-test green + commit + push both remotes.
5. Continue per the design's staged sequence: a clean PASS -> the next catalog integration increment (multi-step sequential composition, then the fluent-prior variant), each its own pre-registered test; an honest non-success -> the next biology-identified integration-fidelity fix, iterate following the catalog, no hand-back, no declaring-unfit.

---

## Discipline (applies to every task)

Reuse-by-import only; no new autograd/training anywhere (only the reused validated learning rules); no protected/frozen/moat module modification; the cheap probe owns its OWN frozen bar and imports no verdict module; the spiking build reuses the parked frozen verdict byte-unchanged; frozen bars never tuned by results; cheap-first before spiking (HARD GATE after Task 1); adversarial review before the decisive run; honest propagation of every outcome (including a cheap NEGATIVE) to both git remotes (origin + gitea); plain ASCII; no hand-back.
