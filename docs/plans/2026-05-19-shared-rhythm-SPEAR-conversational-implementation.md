---
type: plan
status: live
date: 2026-05-19
---

# Shared theta-gamma SPEAR + generative-replay conversational stage — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans, task by
> task. Owner standing instruction pre-selects same-session subagent-driven
> execution (one fresh subagent per task; failing-test → minimal-impl → run
> → commit; controller trust-but-verify every diff). Task 5 is
> CONTROLLER-ONLY, not a subagent task. Mirrors the proven Stage-1 arc
> structure (which caught a real false-PASS via the dedicated adversarial
> review).

**Goal:** Build a single shared theta-gamma rhythm that time-multiplexes an
encode phase and a retrieve/pattern-complete phase (Separate Phases of
Encoding And Retrieval, acetylcholine-gated) across the validated
subsystems, with a prefrontal compositional frame and a generative replay
loop, and test — against a new pre-registered fixed-bar three-state verdict
whose decisive built-in control is the Stage-1 static-composition negative —
whether the rhythm-multiplexed composition yields the capability the static
composition provably could not.

**Architecture:** Reuse byte-unchanged: the validated concept substrate +
hippocampal theta-gamma store + trisynaptic pattern-completion, the
neuromodulator subsystem (acetylcholine phase gate via
`plasticity_window_gate` + manual `set_concentration`), the
replay-consolidation phase functions + awake/sleep gates, the dlpfc PFC
working-memory frame + NMDA bistability, the no-confabulation moat, bridge
stepping. The only net-new code is a small theta-gamma rhythm/phase
controller + wiring (a timing controller, NOT a new learning mechanism; no
automatic differentiation anywhere). Design:
`docs/plans/2026-05-19-shared-rhythm-SPEAR-conversational-architecture-design.md`.

**Tech Stack:** Python; CuPy on RTX 3090 for decisive runs (NumPy only for
the smoke); the verdict module imports standard library + typing only;
reuse-by-import for all subsystems; ASCII-only output; kill-safe via the
reused checkpoint module.

**Protected set (MUST be byte-unchanged across `git diff` for every task
commit; controller verifies):** `research/runners/abstention_gate.py` +
`tests/test_abstention_gate.py` (no-confabulation moat, MUST stay 7/7);
every frozen `*_core.py` incl. `research/runners/compose_retrieval_core.py`
and `research/runners/integrated_loop_core.py` /
`integrated_loop_core_v2.py`; `research/runners/text_minimal_isolation.py`;
`research/runners/consolidation_trainer.py`;
`research/runners/validate_trisynaptic_loop.py`;
`research/runners/compose_concept_chat.py`; `sim/bridge.py`;
`sim/regions.py`; `sim/neuromodulators.py`; `sim/train_checkpoint.py`;
`sim/backend.py`; `sim/kernels.py`.

---

## Task 0: Grounding pin (red until Task 2)

**Files:** Create `tests/test_spear_conversational_pin.py`.

```python
"""Grounding pin; intentionally RED until Task 2 lands the runner."""
import importlib

def test_spear_runner_importable():
    m = importlib.import_module("research.runners.spear_conversational_runner")
    assert hasattr(m, "run_spear_conversational")

def test_spear_core_importable():
    m = importlib.import_module("research.runners.spear_conversational_core")
    assert hasattr(m, "spear_conversational_verdict")
```

Run `pytest tests/test_spear_conversational_pin.py -q` → FAIL (modules
absent; intentional, this is the Task-1/Task-2 gate). Commit
(`test: grounding pin for shared-rhythm SPEAR conversational stage (red
until Task 2)`). Controller verifies protected set byte-empty.

---

## Task 1: The frozen capability-verdict module (LOAD-BEARING; transcribe exactly)

Mirrors the Stage-1 frozen-verdict discipline (which the adversarial review
confirmed sound): fixed thresholds set now and NEVER tuned;
instrument-validity first; malformed → VOID, never crash; VOID strictly
distinct from FAIL; standard library + typing only; does NOT import or
change any existing verdict module or the moat.

**Files:** Create `research/runners/spear_conversational_core.py`; Test
`tests/test_spear_conversational_core.py`.

**Frozen constants (verbatim; NEVER tuned):**
`_SP_FULL_MIN = 0.80`, `_SP_STATIC_CTRL_MAX = 0.40`,
`_SP_ABSTAIN_MIN = 0.90`, `_SP_SCALE_TOL = 0.10`,
`_SP_LADDER = (2, 4, 8)`, `_SP_MIN_SEEDS = 3`.

**Rung required keys:** `N, n_seeds, full_acc, rhythm_removed_acc,
abstain_correct_rhythm_removed`. (`rhythm_removed_acc` = accuracy when the
shared rhythm is removed, which reduces the system to the Stage-1 static
composition — the decisive built-in control: Stage-1 static empirically
scored 0.00, so a faithful rhythm-removed arm must collapse ≤
`_SP_STATIC_CTRL_MAX`; the capability must be attributable to the rhythm.)

**`spear_conversational_verdict(rungs) -> Dict[str, Any]` contract:**
1. Instrument-validity FIRST → VOID (never raise): `rungs` non-empty list;
   each a dict with all required keys; `N` int (not bool) in `_SP_LADDER`;
   `n_seeds` int (not bool) ≥ `_SP_MIN_SEEDS`; the three accuracy fields
   finite (`isinstance (int,float)` not bool, `math.isfinite`) in [0,1];
   rung N set has no duplicates and equals the first `len(rungs)` entries
   of `_SP_LADDER` (a prefix). Any extra key (e.g. caller `verdict`)
   ignored — recompute from raw only.
2. `ok(r) := r.full_acc >= _SP_FULL_MIN and r.rhythm_removed_acc <=
   _SP_STATIC_CTRL_MAX and r.abstain_correct_rhythm_removed >=
   _SP_ABSTAIN_MIN`.
3. `r0` = smallest-N rung. Not `ok(r0)` → `{"gate":"FAIL",...}`. Else if
   any larger rung not `ok` OR `full_acc < r0.full_acc - _SP_SCALE_TOL` →
   `{"gate":"WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",...}`. Else
   `{"gate":"PASS",...}`. Every return dict has `gate`, `reason`,
   `frozen_bars` (echoing the six constants). `gate` ∈ exactly
   {`VOID`,`FAIL`,`WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE`,`PASS`}.

**Adversarial test matrix (≥12 cases, written FIRST, must fail before
impl):** frozen-constant pins (each `_SP_*` verbatim); clean PASS across
ladder; rhythm_removed not collapsing (> max) → FAIL; abstain below bar →
FAIL; small-load-only → WORKS-AT-SMALL-LOAD; n_seeds<3 → VOID; ladder
mismatch → VOID; non-finite → VOID; missing key → VOID; empty/non-list/None
→ VOID; bool not numeric → VOID; duplicate N → VOID; precomputed `verdict`
ignored → FAIL on bad raw; degenerate always-abstain (full 0) → FAIL;
degenerate always-answer (abstain low) → FAIL; VOID != FAIL distinct
strings + both carry frozen_bars+reason. Run → all green. Commit
(`feat: frozen fixed-bar three-state verdict for shared-rhythm SPEAR
conversational stage`). Controller verifies protected set byte-empty +
constants verbatim.

---

## Task 2: The net-new shared-rhythm controller + runner

**Files:** Create `research/runners/spear_conversational_runner.py`; Test
`tests/test_spear_conversational_runner.py`.

**Behavioral spec (genuine net-new wiring; reuse everything else
byte-unchanged; read the real reused interfaces, do not guess):**
- Build the validated concept substrate + hippocampus
  (`enable_hippocampus_consolidation=True`) + the dlpfc PFC frame
  (`enable_dlpfc_verb=True`, global `enable_nmda=True`) via the reused
  builder + the same construction the Stage-1 re-review cleared.
- **Net-new shared theta-gamma rhythm/phase controller:** a theta-phase
  clock (~125 ms sim-time period) with a nested gamma sub-cycle index;
  on the **encode** phase set acetylcholine HIGH via the reused
  neuromodulator `set_concentration` with a reused
  `plasticity_window_gate` target wired to the plastic pathways
  (plasticity on, afferent drive, retrieval suppressed); on the
  **retrieve** phase set acetylcholine LOW (plasticity off,
  CA3-recurrent/pattern-completion). The controller drives the reused
  `bridge.step_simulation(...)` in its own loop (NO edit to the step).
- Reuse the dlpfc region + reused NMDA bistability to hold/advance the
  ordered sequence slot across gamma sub-cycles; reuse the
  replay-consolidation phase functions + awake/sleep gate helpers for the
  slower encode↔consolidate transition + the generative replay loop;
  gate each emitted item through the reused `gate(ranked, 650.0)` moat
  (raw firing-rate confidence, the calibrated quantity — same lesson as
  the Stage-1 fix).
- **Decisive built-in control arm `rhythm_removed`:** identical to the
  full run with the shared-rhythm controller DISABLED (no phase
  multiplexing — reduces to the Stage-1 static composition), same seed
  and same draws as `full`. Emits per (seed,N): `full_acc`,
  `rhythm_removed_acc`, `abstain_correct_rhythm_removed`; aggregates to
  the rung dicts the verdict module consumes; `--tiny-synth` shrinks
  scale for a fast smoke (toy numbers NOT a result; make Task 0 green);
  kill-safe via reused checkpoint; CuPy real path / NumPy only for
  `--tiny-synth`; ASCII; NO torch/autograd anywhere.
- `run_spear_conversational(seeds, loads=_SP_LADDER, tiny_synth=False,
  out_path=None, ckpt=None)` + argparse `main()` (`--seeds`, `--loads`,
  `--tiny-synth`, `--out`, `--ckpt`).

**TDD:** tests FIRST — `--tiny-synth` runs end-to-end and produces a
well-formed rungs list `spear_conversational_verdict` accepts (one of the
four states, never raises, not VOID for a structural reason); `full` and
`rhythm_removed` for a (seed,N) cell consume the SAME seed/draws and differ
ONLY by the shared-rhythm controller being enabled vs disabled; no
torch/autograd on shipped paths; the answer is decoded from the validated
neural readout, not a string, and the moat is fed the calibrated
firing-rate quantity. Run-fail → implement minimally against the reused
interfaces → run-pass (pin now green; core 19+; moat 7/7) → commit
(`feat: net-new shared-rhythm SPEAR controller + runner (reuse-only; no
autograd)`). Controller verifies protected set byte-empty.

---

## Task 3: Dedicated adversarial review (BEFORE no-harm)

Fresh adversarial reviewer (mirror the Stage-1 review that found a real
false-PASS). Primary mandate: is the capability genuinely emergent from the
shared-rhythm temporal multiplexing, or a wiring/timing artifact / single
reused-subsystem leakage? Is `rhythm_removed` a FAITHFUL "full minus only
the shared rhythm" (genuinely reduces to the Stage-1 static composition,
same draws)? Can a degenerate/empty/single-subsystem solver score PASS via
the runner+frozen-verdict end-to-end (re-run the Stage-1-class exploits
adapted here)? Are the `_SP_*` bars movable by results? Any
autograd/torch? Are the subsystems genuine byte-unchanged identity-imports
not copy-edited (esp. the neuromodulator ACh gate, the hippocampal/
trisynaptic/replay/dlpfc reuse)? STRENGTHEN-only fixes to non-protected
files only; frozen bars byte-unchanged; commit `review:` prefix; re-review
loop until CLEAR. Controller verifies protected set byte-empty.

---

## Task 4: No-harm phase

Prove the full protected set is byte-unchanged from the pre-Task-0 base to
HEAD (empty diff for every protected path); `tests/test_abstention_gate.py`
still 7/7; the full SPEAR + Stage-1 + integrated-loop suites green; assert
no shipped path imports `torch.autograd` / `.backward`. Commit the no-harm
evidence; controller trust-but-verify; push both remotes.

---

## Task 5: CONTROLLER-ONLY decisive run (NOT a subagent task)

Controller, same turn, never stopping on a promise: (1) grounding-first
tiny-synth run (toy numbers explicitly NOT propagated); (2) decisive
kill-safe multi-seed run at the frozen ladder (2,4,8), seeds 42 43 44, CuPy
on RTX 3090, DURABLE capture to `research/findings/raw/`, monitored to
ACTUAL completion via a mechanism that genuinely notifies on process exit
(never a detached process with a false "will be notified"; completion
actively confirmed before any result is stated); (3) mandatory smell-test
scrutinising a nominal PASS HARDER than a FAIL — recompute the verdict from
the single recorded output (no re-run, no bar change); confirm the full
system genuinely succeeds, the `rhythm_removed` control genuinely collapses
to the Stage-1 static level (the capability is attributable to the rhythm),
abstention genuinely holds; (4) honest propagation of EVERY outcome in
plain language: findings doc + `webapp/capability_status.json` pillar
(status PREDICTED until a clean scrutinised PASS; schema test green) +
`AUTONOMOUS_STATE.md` + commit + push BOTH remotes; (5) then autonomously:
a clean scrutinised PASS → the next pre-registered staged step (design
Architecture B then C); an honest FAIL/VOID/WORKS-AT-SMALL-LOAD → follow
the biology to the next integration-fidelity refinement and iterate — NOT
declare unfit, NOT hand back, NOT config-crank, NO bar change.

**Honest ceiling (never overstated):** a clean scrutinised success = a
biology-grounded shared-theta-gamma-rhythm composition shows grounded
compositional/sequential capability the static two-store composition
provably does not (the Stage-1 negative is the built-in control), holding
or improving with load, abstaining rather than confabulating — explicitly
NOT fluent open-ended language, NOT an LLM, NOT the retracted
transitive-inference claim, unless a later pre-registered stage genuinely
shows it; all prior validated results and honest boundaries unaffected.
