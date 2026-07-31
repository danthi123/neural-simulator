---
type: plan
status: live
date: 2026-05-20
---

# Per-regime metacognitive monitor stage — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans, task by
> task. Owner standing instruction pre-selects same-session subagent-driven
> execution (one fresh subagent per task; failing-test → minimal-impl → run
> → commit; controller trust-but-verify every diff). Task 5 (no-harm) and
> Task 6 (decisive run) are CONTROLLER-ONLY -- not subagent tasks. Mirrors
> the proven Stage-1 / SPEAR / Pirazzini arc structure (each of which
> caught real load-bearing defects via the dedicated adversarial review
> and closed them via the net-new-runner-only fix loop).

**Goal:** Build a per-regime metacognitive-monitor architecture (Miyamoto
2017 doubly-dissociable parallel metamemory streams) on the project's
validated substrate: the existing direct-retrieval no-confabulation moat
stays byte-unchanged; a new compositional-regime gate sits alongside with
its own pre-registered calibrated threshold; a runner-level routing layer
selects the appropriate monitor per query type; test whether per-regime
threshold separation lifts compositional retrieval above the
compositional-regime threshold while direct retrieval remains gated by the
original 650 AND the trustworthy property holds under composition.

**Architecture:** Reuse byte-unchanged: the validated v14/v16 + hippocampus
+ dlpfc substrate; the existing `abstention_gate.py` (`DEFAULT_THRESHOLD =
650.0`; 7/7 tests); every previously-validated subsystem. Net-new code is
ONLY: (a) a new frozen capability-verdict module mirroring the existing
frozen-verdict discipline; (b) a new `abstention_gate_compositional.py` +
its 7-case test matrix + a pre-registered separate calibration step;
(c) a per-regime-monitor runner that builds the substrate, runs queries
through the validated readout path, and routes each answer through the
appropriate gate per query type. No automatic differentiation anywhere.
Design: `docs/plans/2026-05-20-per-regime-metacognitive-monitor-architecture-design.md`.

**Tech Stack:** Python; CuPy on RTX 3090 for decisive runs (NumPy only for
the smoke); the verdict module + the new abstention gate module import
standard library + typing only; reuse-by-import for all subsystems;
ASCII-only output; kill-safe via the reused checkpoint module.

**Protected set (MUST be byte-unchanged across `git diff` for every task
commit; controller verifies):** `research/runners/abstention_gate.py`
(+ `tests/test_abstention_gate.py`, MUST stay 7/7); every frozen `*_core.py`
incl. `compose_retrieval_core.py` + `spear_conversational_core.py` +
`pirazzini_three_layer_core.py` + `integrated_loop_core{,_v2}.py`;
`research/runners/text_minimal_isolation.py`;
`research/runners/consolidation_trainer.py`;
`research/runners/validate_trisynaptic_loop.py`;
`research/runners/compose_concept_chat.py`;
`research/runners/compose_concept_engram.py`;
`research/runners/compose_retrieval_runner.py`;
`research/runners/spear_conversational_runner.py`;
`research/runners/pirazzini_three_layer_runner.py`; `sim/bridge.py`;
`sim/regions.py`; `sim/neuromodulators.py`; `sim/train_checkpoint.py`;
`sim/backend.py`; `sim/kernels.py`.

---

## Task 0: Grounding pin (red until Tasks 2 + 3)

**Files:** Create `tests/test_per_regime_monitor_pin.py`.

```python
"""Grounding pin; intentionally RED until Tasks 2 + 3 land.

This IS the Tasks 1-3 completion gate (see
docs/plans/2026-05-20-per-regime-metacognitive-monitor-implementation.md).
"""
import importlib


def test_per_regime_core_importable():
    m = importlib.import_module("research.runners.per_regime_monitor_core")
    assert hasattr(m, "per_regime_monitor_verdict")


def test_compositional_gate_importable():
    m = importlib.import_module("research.runners.abstention_gate_compositional")
    assert hasattr(m, "gate")
    assert hasattr(m, "COMPOSITIONAL_THRESHOLD")


def test_per_regime_runner_importable():
    m = importlib.import_module("research.runners.per_regime_monitor_runner")
    assert hasattr(m, "run_per_regime_monitor")
```

Run `pytest tests/test_per_regime_monitor_pin.py -q` → FAIL (3 modules
missing; intentional). Commit (`test: grounding pin for per-regime
metacognitive-monitor stage (red until Tasks 1-3)`). Controller verifies
protected set byte-empty.

---

## Task 1: The frozen capability-verdict module (LOAD-BEARING; transcribe exactly)

Mirrors the Stage-1 / SPEAR / Pirazzini frozen-verdict discipline (all of
which the dedicated adversarial review CLEARed): fixed thresholds set now
and NEVER tuned; instrument-validity FIRST; malformed → safe VOID, never
crash; VOID strictly distinct from FAIL; standard library + typing only;
does NOT import or change any existing verdict module or the moats.

**Files:** Create `research/runners/per_regime_monitor_core.py`; Test
`tests/test_per_regime_monitor_core.py`.

**Frozen constants (verbatim; NEVER tuned):**
- `_PR_FULL_MIN = 0.80` (full per-regime-monitor compositional accuracy bar)
- `_PR_UNIFORM_CTRL_MAX = 0.10` (single-threshold-applied-uniformly
  variant control: reduces to the triple-convergent ceiling; must collapse
  ≤ 0.10 so the capability is attributable to the per-regime separation)
- `_PR_DIRECT_RETAIN_MIN = 0.80` (direct-retrieval accuracy under the
  per-regime architecture must NOT degrade vs the validated v16
  88.75%-multi-seed baseline; setting at 0.80 as a conservative floor)
- `_PR_ABSTAIN_CORRECT_MIN = 0.90` (trustworthy property must hold:
  fraction of ungroundable queries on which the system abstains)
- `_PR_SCALE_TOL = 0.10`, `_PR_LADDER = (2, 3, 5)`, `_PR_MIN_SEEDS = 3`

**Rung required keys:** `N, n_seeds, full_acc, uniform_ctrl_acc,
direct_retain_acc, abstain_correct`.

**`per_regime_monitor_verdict(rungs) -> Dict[str, Any]` contract:**
1. Instrument-validity FIRST → VOID (never raise): rungs non-empty list;
   each rung is a dict with all required keys; N int (not bool) in
   `_PR_LADDER`; n_seeds int (not bool) ≥ `_PR_MIN_SEEDS`; all four
   accuracy fields are `isinstance(int, float)` and `not isinstance(bool)`
   and `math.isfinite` and in `[0.0, 1.0]`; the rung N multiset has no
   duplicates AND, sorted, equals `list(_PR_LADDER[:len(rungs)])` (a
   prefix). Extra caller keys (e.g. "verdict") IGNORED — recompute from
   raw only.
2. `ok(r) := r["full_acc"] >= _PR_FULL_MIN and r["uniform_ctrl_acc"] <=
   _PR_UNIFORM_CTRL_MAX and r["direct_retain_acc"] >= _PR_DIRECT_RETAIN_MIN
   and r["abstain_correct"] >= _PR_ABSTAIN_CORRECT_MIN`.
3. r0 = smallest-N rung. Not `ok(r0)` → `{"gate":"FAIL",...}`. Else if any
   larger rung not `ok` OR `r["full_acc"] < r0["full_acc"] - _PR_SCALE_TOL`
   → `{"gate":"WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",...}`. Else
   `{"gate":"PASS",...}`. Every return dict has `gate`, `reason`,
   `frozen_bars` (echoing the seven constants).

**Adversarial test matrix (≥17 cases; written FIRST, must fail before
impl):** frozen-constant pins (each `_PR_*` value asserted verbatim);
clean PASS across the ladder; uniform_ctrl not collapsing (> max) → FAIL;
direct_retain below floor → FAIL (no degradation of direct retrieval
allowed); abstain below bar → FAIL; full below bar → FAIL; small-load-only
→ WORKS-AT-SMALL-LOAD; n_seeds < 3 → VOID; ladder mismatch (e.g. N=4) →
VOID; non-finite → VOID; missing key → VOID; empty/non-list/None → VOID;
bool not numeric → VOID; duplicate N → VOID; precomputed "verdict" field
ignored → FAIL on bad raw; degenerate always-abstain (full=0,
uniform_ctrl=0, direct_retain=0, abstain=1) → FAIL; degenerate
always-answer (abstain=0) → FAIL; VOID and FAIL are distinct strings;
both carry `reason` + `frozen_bars`.

**Step 1: Write the failing test** — full ≥17-case matrix as above.
**Step 2: Run-to-fail.** **Step 3: Implement minimally** per the contract.
**Step 4: Run-to-pass.** **Step 5: Commit** (`feat: frozen fixed-bar
three-state verdict module for per-regime metacognitive-monitor stage`).
Controller verifies protected set byte-empty + constants verbatim.

---

## Task 2: New compositional-regime abstention gate + calibration

**Files:** Create `research/runners/abstention_gate_compositional.py`;
Test `tests/test_abstention_gate_compositional.py`.

**The new gate mirrors `abstention_gate.py` discipline EXACTLY** —
standard library + typing only; same signatures (`gate(ranked, threshold)`
returning Optional[(concept, rate, tag)] or None; `abstain(top_confidence,
threshold)` returning bool); a `COMPOSITIONAL_THRESHOLD` constant
calibrated separately from the direct-retrieval 650; the calibration value
is set in a pre-registered separate step BEFORE the decisive run AND
frozen after the calibration step.

**The calibration step (pre-registered, not result-driven):** in a
separate runner-local block of the runner (Task 3), the calibration is
measured on a representative held-out compositional ground-truth signal
that the validated substrate produces (encoded compositional ~X vs
control ~Y, mirroring the direct-retrieval calibration that produced
650). The calibration MUST be done on a separate held-out set from the
decisive evaluation set; once measured, the value is set as the
`COMPOSITIONAL_THRESHOLD` constant in the source file and committed AS
THE FROZEN VALUE. Retroactive recalibration after the decisive run is
itself goalpost-moving and is forbidden.

For Task 2, the gate module is created with a placeholder threshold
constant (e.g. `COMPOSITIONAL_THRESHOLD = 0.0` with a comment that the
calibration is the runner's job in Task 3); the gate logic is
byte-identical in shape to the existing `abstention_gate.py` (read it
to ground the implementation).

**Adversarial test matrix (mirrors `tests/test_abstention_gate.py` — read
it for the 7-case shape):** structural validity; gate(ranked, threshold)
returns top tuple when rate > threshold else None; abstain(c, threshold)
returns True iff c ≤ threshold; threshold-default-pin (asserts the
COMPOSITIONAL_THRESHOLD constant); none/empty/non-list inputs handled
gracefully without crash; etc. ≥7 cases mirroring the existing moat tests.

**Step 1: Write the failing test.** **Step 2: Run-to-fail.** **Step 3:
Implement minimally** (stdlib-only; mirror existing moat's structure).
**Step 4: Run-to-pass + confirm `tests/test_abstention_gate.py` is still
7/7 byte-identical.** **Step 5: Commit** (`feat: new compositional-regime
abstention gate module (mirrors existing moat discipline; placeholder
threshold to be calibrated in Task 3)`).

---

## Task 3: The net-new per-regime-monitor runner + calibration

**Files:** Create `research/runners/per_regime_monitor_runner.py`; Test
`tests/test_per_regime_monitor_runner.py`.

**Behavioral spec (the only genuinely net-new integration logic):**
- Build the validated substrate exactly as the Stage-1 / SPEAR / Pirazzini
  cleared runners do (`build_biological_brain_regions(...,
  enable_hippocampus_consolidation=True, enable_dlpfc_verb=True)` +
  `enable_nmda=True`; do NOT override num_traits).
- **Calibration block (the runner's own internal pre-registered step):**
  before any decisive evaluation, run a held-out calibration on a
  representative compositional ground-truth signal — the runner builds a
  separate set of compositional queries with known correct answers
  (different facts from those used for the decisive evaluation), measures
  the raw firing-rate confidence at `lang_output` for the correct word
  when the compositional answer is groundable vs when it is not, and
  computes a calibrated threshold separating the two. Write the resulting
  threshold to a calibration JSON output AND assert that the
  `COMPOSITIONAL_THRESHOLD` constant in
  `abstention_gate_compositional.py` matches it (if it does not, the
  runner records a CALIBRATION-MISMATCH outcome and exits gracefully;
  retroactive threshold changes are forbidden).
- **Per-query-type routing:** the runner accepts a per-query
  `query_type` (`"direct"` or `"compositional"`) and routes the answer
  through the appropriate gate:
  - direct → `abstention_gate.gate(ranked, 650.0)` (existing moat,
    byte-unchanged)
  - compositional → `abstention_gate_compositional.gate(ranked,
    COMPOSITIONAL_THRESHOLD)` (new gate)
- **Three measurement arms per (seed, N):**
  - `full` = per-regime architecture (each query routed to its
    regime-appropriate gate); read out `full_acc` (fraction of all
    queries answered correctly).
  - `uniform_ctrl` = the same architecture except BOTH gates set to 650
    (the existing direct-retrieval threshold applied uniformly to
    everything); the decisive built-in control — must collapse to the
    triple-convergent ceiling (`uniform_ctrl_acc ≤ 0.10`).
  - `direct_retain` = read out only the direct queries' accuracy under
    the per-regime architecture (direct retrieval must NOT degrade vs the
    validated baseline; `direct_retain_acc ≥ 0.80`).
- `abstain_correct` = fraction of ungroundable queries on which the
  appropriate-regime gate abstained.
- Emit per (seed, N): `N, n_seeds, full_acc, uniform_ctrl_acc,
  direct_retain_acc, abstain_correct`. Aggregate across seeds to one rung
  dict per N. Call `per_regime_monitor_verdict(rungs)`. Output JSON.
  `--tiny-synth` shrinks the smoke; CuPy for real, NumPy only for
  tiny-synth. ASCII. NO torch / NO autograd.

**Anti-cheat (carry forward lessons from Stage-1 / SPEAR / Pirazzini):**
- OPAQUE tag names (`f"ep_{i}"`); no `.split("_")` on tag names.
- Moat input is raw firing-rate confidence (the validated `_ranked_from_pattern`
  formula) for BOTH gates.
- `uniform_ctrl` differs from `full` ONLY in the threshold-routing
  decision; same seed, same draws, same encoding.
- `direct_retain` is read out from the same run as `full` (no separate
  draws).
- COMPOSITIONAL_THRESHOLD calibration is the runner's first step;
  decisive evaluation uses the resulting frozen threshold; retroactive
  recalibration forbidden.

**Step 1: Write the failing test** — `--tiny-synth` runs end-to-end and
produces a well-formed rung list `per_regime_monitor_verdict` accepts;
no torch/autograd; opaque tags; calibrated moat-input; `uniform_ctrl`
threads the single difference (both thresholds = 650); `direct_retain`
read from the same run; pin asserts the calibrated COMPOSITIONAL_THRESHOLD
constant matches whatever the calibration block produces in a separate
tiny-synth calibration smoke (so the constant becomes the frozen value).
**Step 2: Run-to-fail.** **Step 3: Implement minimally.** **Step 4:
Run-to-pass + pin (Task 0) now green; existing moat 7/7; both new gate
modules' tests still green.** **Step 5: Commit** (`feat: net-new
per-regime metacognitive-monitor runner (calibration + per-query-type
routing; uniform_ctrl built-in control; reuse-only; no autograd)`).
Controller verifies protected set byte-empty.

---

## Task 4: Dedicated adversarial review (BEFORE no-harm)

Mirror the proven Stage-1 / SPEAR / Pirazzini reviews (each of which
caught real defects). Specific high-risk items for this stage:

- Is the COMPOSITIONAL_THRESHOLD calibration genuinely separate from the
  decisive evaluation, on a different held-out set, and frozen *before*
  the decisive run? Or is it tuned to PASS? Reviewer must reproduce the
  calibration on the held-out set themselves and verify the resulting
  value matches the committed constant.
- Is `uniform_ctrl` genuinely "full minus only the threshold-routing
  decision" (same seed, same draws, same encoding) and does it
  empirically collapse to ≈ the triple-convergent ceiling on a small
  reproducible probe?
- Can a degenerate / over-permissive new gate (e.g. COMPOSITIONAL_THRESHOLD
  = 0) score PASS by accepting everything as a "compositional answer"?
  Construct the exploit; assert FAIL via the runner + verdict end-to-end.
- Is the existing `abstention_gate.py` byte-unchanged (7/7 green)? Run
  the moat tests; diff the file.
- Are the new gate's tests genuinely mirroring the existing moat's
  discipline (no laxer tests that paper over a weak gate)? Reviewer
  reads `tests/test_abstention_gate.py` and compares.
- Frozen bars `_PR_*` immovable; no autograd; subsystems byte-unchanged.

STRENGTHEN-only fixes to non-protected files only; commit `review:`;
re-review until CLEAR. Controller verifies protected set byte-empty.

---

## Task 5: No-harm phase (controller-only)

Verify protected set + the existing `abstention_gate.py` + its 7/7 test
byte-unchanged from the pre-Task-0 base to HEAD; full new + prior suites
green (per-regime + Pirazzini + SPEAR + Stage-1 + moat); no
torch/autograd on shipped paths; the new gate's calibrated threshold
constant matches the calibration JSON output (no drift). Commit the
no-harm evidence; push both remotes.

---

## Task 6: CONTROLLER-ONLY decisive run

Controller, same turn, never stopping on a promise:

1. Grounding tiny-synth run (toy numbers explicitly NOT propagated).
2. Decisive kill-safe multi-seed run at the frozen ladder (2, 3, 5),
   seeds 42 43 44, CuPy on RTX 3090, DURABLE capture, monitored to
   ACTUAL completion via a genuine completion waiter (never a detached
   process with a false "will be notified"). The calibration is run
   FIRST on a held-out set; the resulting threshold is verified to match
   the committed constant; only then is the decisive evaluation executed.
3. Mandatory smell-test scrutinising a nominal PASS HARDER than a FAIL:
   recompute the verdict from the single recorded output (no re-run, no
   bar change); confirm `full` genuinely clears the bars AND
   `uniform_ctrl` genuinely collapses (per-regime separation is the
   differentiator) AND `direct_retain` is preserved AND abstention holds.
4. Honest propagation of EVERY outcome (findings doc + capability pillar
   + state file + commit + push BOTH remotes).
5. Autonomous next step per outcome.

**Honest ceiling (never overstated):** a clean success = a biology-
grounded per-regime metacognitive-monitor architecture shows
compositional retrieval correctly routed to a regime-appropriate
threshold above which compositional queries are answered, while direct
retrieval stays gated at 650 and the trustworthy property holds.
Explicitly NOT fluent open-ended language, NOT an LLM, NOT a
threshold-relaxation that defeats the trustworthy property. The
orienting goal is artificial life with a proper brain analogue; biology-
translatable insights are the deliverable.
