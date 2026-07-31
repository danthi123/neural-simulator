---
type: plan
status: live
date: 2026-05-19
---

# Regime-correct compositional retrieval — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Owner standing instruction pre-selects
> same-session subagent-driven execution (one fresh subagent per task,
> failing-test → minimal-impl → run → commit, controller trust-but-verify
> each diff). Task 5 is CONTROLLER-ONLY — not a subagent task.

**Goal:** Build a constructive two-path composition that answers grounded
compositional queries by reading recent-specific content from the hippocampal
pathway and order-invariant semantic content from the consolidated neocortical
pathway — each in its biologically-correct regime — and abstains rather than
confabulates under ablation; scored by a new pre-registered fixed-bar
three-state verdict module.

**Architecture:** Reuse the project's already-validated subsystems
byte-unchanged (16-pool concept binding, trisynaptic/engram hippocampal path,
replay-consolidation, multi-tag retrieval, the no-confabulation abstention
moat). The only net-new code is a composition/routing controller (Architecture
A from the design) plus a new frozen capability-verdict module. No automatic
differentiation anywhere. Design:
`docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`.

**Tech Stack:** Python; CuPy on RTX 3090 for decisive runs (NumPy only for the
smoke); the verdict module imports standard library + typing only; reuse-by-
import for all subsystems; ASCII-only output; kill-safe via the reused
checkpoint module.

**Protected set (MUST be byte-unchanged across `git diff` for every task
commit; controller verifies):** `research/runners/abstention_gate.py` +
`tests/test_abstention_gate.py` (the no-confabulation moat, MUST stay 7/7
green); `research/runners/integrated_loop_core.py` +
`research/runners/integrated_loop_core_v2.py` + every other frozen `*_core.py`;
`research/runners/text_minimal_isolation.py` (build_biological_brain_regions /
run_minimal_isolation REUSED UNMODIFIED); `research/runners/consolidation_trainer.py`;
`research/runners/consolidation_eval.py`; `research/runners/compose_concept_chat.py`;
`sim/bridge.py`; `sim/regions.py`; `sim/neuromodulators.py`;
`sim/train_checkpoint.py`; `sim/backend.py`; `sim/kernels.py`.

---

## Task 0: Grounding pin (red until Task 2)

**Files:**
- Create: `tests/test_compose_retrieval_pin.py`

**Step 1: Write the failing pin test**

```python
import importlib
import pytest

def test_compose_retrieval_runner_importable():
    mod = importlib.import_module("research.runners.compose_retrieval_runner")
    assert hasattr(mod, "run_compose_retrieval")

def test_compose_retrieval_core_importable():
    mod = importlib.import_module("research.runners.compose_retrieval_core")
    assert hasattr(mod, "compose_retrieval_verdict")
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_compose_retrieval_pin.py -q`
Expected: FAIL (ModuleNotFoundError) — intentional; this IS the Task-1/Task-2
completion gate.

**Step 3: Commit (red pin)**

```bash
git add tests/test_compose_retrieval_pin.py
git commit -m "test: grounding pin for regime-correct compositional retrieval (red until Task 2)"
```

Controller verifies the protected set is byte-empty in this commit's diff.

---

## Task 1: The frozen capability-verdict module (LOAD-BEARING; transcribe exactly)

Mirrors the discipline of `integrated_loop_core.py` exactly: fixed numeric
thresholds set in advance and NEVER moved; instrument-validity checked first;
malformed input → safe "cannot conclude" (VOID), never a crash; VOID strictly
distinct from FAIL; imports only standard library + typing; does NOT import or
change any existing verdict module or the moat.

**Files:**
- Create: `research/runners/compose_retrieval_core.py`
- Test: `tests/test_compose_retrieval_core.py`

**Step 1: Write the failing adversarial test matrix (>=12 cases)**

The matrix MUST include: (1) clean PASS across the full ladder; (2) recent-only
ablation fails to collapse → FAIL; (3) remote-only ablation fails to collapse →
FAIL; (4) abstention-under-recent-only below bar → FAIL; (5) abstention-under-
remote-only below bar → FAIL; (6) works at N=2 but full_acc decreases beyond
scale tolerance at N=4/8 → WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE; (7)
< MIN_SEEDS on any rung → VOID; (8) ladder mismatch (rung N not in frozen
ladder) → VOID; (9) non-finite/NaN/inf in any field → VOID; (10) missing
required key → VOID; (11) empty rungs / not-a-list → VOID; (12) frozen-constant
pins (each `_CR_*` value asserted verbatim); (13) a caller-supplied
pre-computed "verdict" field is IGNORED — recompute from raw numbers (no
fabricated PASS); (14) VOID is returned as the string "VOID" and is `!=`
"FAIL"; (15) a degenerate always-abstain run (full_acc 0.0, abstain 1.0) →
FAIL not PASS; (16) a degenerate always-answer run (abstain 0.0) → FAIL not
PASS.

```python
from research.runners.compose_retrieval_core import (
    compose_retrieval_verdict,
    _CR_FULL_MIN, _CR_ABLATION_MAX, _CR_ABSTAIN_MIN,
    _CR_SCALE_TOL, _CR_LADDER, _CR_MIN_SEEDS,
)

def _rung(N, full=0.88, recent_only=0.20, remote_only=0.18,
          ab_recent=0.97, ab_remote=0.96, n_seeds=3):
    return {"N": N, "n_seeds": n_seeds, "full_acc": full,
            "recent_only_acc": recent_only, "remote_only_acc": remote_only,
            "abstain_correct_recent_only": ab_recent,
            "abstain_correct_remote_only": ab_remote}

def test_frozen_constant_pins():
    assert _CR_FULL_MIN == 0.80
    assert _CR_ABLATION_MAX == 0.40
    assert _CR_ABSTAIN_MIN == 0.90
    assert _CR_SCALE_TOL == 0.10
    assert _CR_LADDER == (2, 4, 8)
    assert _CR_MIN_SEEDS == 3

def test_clean_pass():
    rungs = [_rung(2), _rung(4, full=0.86), _rung(8, full=0.84)]
    assert compose_retrieval_verdict(rungs)["gate"] == "PASS"

def test_recent_only_not_collapsing_is_fail():
    rungs = [_rung(2, recent_only=0.75)]
    v = compose_retrieval_verdict(rungs)
    assert v["gate"] == "FAIL" and v["gate"] != "VOID"

def test_abstain_below_bar_is_fail():
    rungs = [_rung(2, ab_remote=0.50)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"

def test_small_load_only_is_works_small():
    rungs = [_rung(2, full=0.88), _rung(4, full=0.60), _rung(8, full=0.45)]
    assert compose_retrieval_verdict(rungs)["gate"] == "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"

def test_below_min_seeds_is_void():
    rungs = [_rung(2, n_seeds=2)]
    v = compose_retrieval_verdict(rungs)
    assert v["gate"] == "VOID" and v["gate"] != "FAIL"

def test_ladder_mismatch_is_void():
    assert compose_retrieval_verdict([_rung(3)])["gate"] == "VOID"

def test_nonfinite_is_void():
    assert compose_retrieval_verdict([_rung(2, full=float("nan"))])["gate"] == "VOID"

def test_missing_key_is_void():
    bad = {"N": 2, "n_seeds": 3, "full_acc": 0.9}
    assert compose_retrieval_verdict([bad])["gate"] == "VOID"

def test_empty_and_nonlist_is_void():
    assert compose_retrieval_verdict([])["gate"] == "VOID"
    assert compose_retrieval_verdict("nope")["gate"] == "VOID"

def test_precomputed_verdict_is_ignored():
    r = _rung(2); r["verdict"] = "PASS"; r["full_acc"] = 0.10
    assert compose_retrieval_verdict([r])["gate"] == "FAIL"

def test_degenerate_always_abstain_is_fail():
    rungs = [_rung(2, full=0.0, recent_only=0.0, remote_only=0.0,
                   ab_recent=1.0, ab_remote=1.0)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"

def test_degenerate_always_answer_is_fail():
    rungs = [_rung(2, ab_recent=0.0, ab_remote=0.0)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_compose_retrieval_core.py -q`
Expected: FAIL (ModuleNotFoundError).

**Step 3: Write the minimal module (exact contract)**

`research/runners/compose_retrieval_core.py` — stdlib + typing only:

- Frozen constants (NEVER tuned): `_CR_FULL_MIN = 0.80`,
  `_CR_ABLATION_MAX = 0.40`, `_CR_ABSTAIN_MIN = 0.90`,
  `_CR_SCALE_TOL = 0.10`, `_CR_LADDER = (2, 4, 8)`, `_CR_MIN_SEEDS = 3`.
- `compose_retrieval_verdict(rungs) -> Dict[str, Any]`:
  1. Instrument-validity FIRST: `rungs` is a non-empty list; each rung is a
     dict containing exactly the required numeric keys
     (`N, n_seeds, full_acc, recent_only_acc, remote_only_acc,
     abstain_correct_recent_only, abstain_correct_remote_only`); every value
     `isinstance` int/float (not bool) and `math.isfinite`; every `N` in
     `_CR_LADDER`; the set of rung Ns is exactly the prefix of the ladder
     present with no duplicates; every `n_seeds >= _CR_MIN_SEEDS`. ANY
     violation → return `{"gate": "VOID", "reason": "...",
     "frozen_bars": {...}}`. Never raise. Any extra keys (e.g. a caller
     "verdict") are ignored — the verdict recomputes from raw numbers only.
  2. Per-rung capability predicate `ok(r)` :=
     `r.full_acc >= _CR_FULL_MIN and r.recent_only_acc <= _CR_ABLATION_MAX
      and r.remote_only_acc <= _CR_ABLATION_MAX
      and r.abstain_correct_recent_only >= _CR_ABSTAIN_MIN
      and r.abstain_correct_remote_only >= _CR_ABSTAIN_MIN`.
  3. Let `r0` be the smallest-N rung. If NOT `ok(r0)` → `{"gate": "FAIL", ...}`.
  4. If `ok(r0)` but any larger rung either fails `ok(r)` OR has
     `full_acc < r0.full_acc - _CR_SCALE_TOL` →
     `{"gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE", ...}`.
  5. Else (all rungs `ok` and non-decreasing within tolerance) →
     `{"gate": "PASS", ...}`.
  Always include `"frozen_bars"` echoing the six constants and a
  `"reason"` string. `"VOID"` and `"FAIL"` are distinct strings;
  "cannot conclude" (VOID) is never reported as success.

**Step 4: Run to verify all pass**

Run: `pytest tests/test_compose_retrieval_core.py -q`
Expected: PASS (>=12 cases).

**Step 5: Commit**

```bash
git add research/runners/compose_retrieval_core.py tests/test_compose_retrieval_core.py
git commit -m "feat: frozen fixed-bar three-state verdict module for regime-correct compositional retrieval"
```

Controller verifies protected set byte-empty in the diff; confirms
`_CR_*` constants match this spec verbatim.

---

## Task 2: The net-new composition/routing runner (Architecture A)

**Files:**
- Create: `research/runners/compose_retrieval_runner.py`
- Test: `tests/test_compose_retrieval_runner.py`

**Behavioral spec (genuine net-new wiring; reuse everything else byte-
unchanged):**
- Build the substrate + hippocampus via the REUSED
  `build_biological_brain_regions(... enable_hippocampus_consolidation=True
  ...)` and bridge via the REUSED `run_minimal_isolation` entry path
  (text_minimal_isolation.py). Concept pools = the validated v16 recipe.
- **Recent-specific encode (hippocampal regime):** for each recent fact,
  use the REUSED engram API (`start_engram_recording` →
  drive the pair → `commit_engram_tag`) — the validated stim-recall path.
- **Remote-semantic build (consolidated regime):** the REUSED
  `run_concept_replay_phase` / `run_swr_replay_phase` /
  `run_consolidation_training` to build the order-invariant schema over the
  base vocabulary.
- **Net-new composition/routing controller (the ONLY new wiring):** given a
  compositional query, (i) retrieve the recent-specific part via engram
  stim-recall (hippocampal regime), (ii) read the general/semantic part via
  the REUSED multi-tag/consolidated readout, (iii) compose them
  (retrieval-augmented: the hippocampal retrieval conditions the consolidated
  readout) into a single ranked answer, (iv) pass the top candidate through
  the REUSED `gate(ranked, threshold=650.0)` no-confabulation moat —
  answer if it clears, else "I don't know".
- **Ablations (faithful, same random draws as full):**
  - recent-only := skip the consolidation/replay build (no remote schema);
    hippocampus on.
  - remote-only := the REUSED hippo-OFF protocol
    (`evaluate_with_hippo_off` / strict-silence over `HIPPO_REGIONS`),
    consolidated schema on.
  Each ablation is identical to the full run minus exactly that one regime,
  with identical seeds/draws.
- Emits, per load N in the frozen ladder and per seed: `full_acc`,
  `recent_only_acc`, `remote_only_acc`, `abstain_correct_recent_only`,
  `abstain_correct_remote_only`; aggregates to the rung dict the verdict
  module consumes. Kill-safe/resumable via the reused checkpoint module.
  `--tiny-synth` shrinks pools/episodes for the smoke (its toy numbers are
  explicitly NOT a result and make Task 0 green). CuPy for the real path,
  NumPy only for `--tiny-synth`. ASCII only. NO autograd anywhere.

**Step 1: Write failing tests** — `--tiny-synth` smoke runs end-to-end and
produces a well-formed rung list that `compose_retrieval_verdict` accepts
(returns one of the four states, never raises); the runner imports no
`torch.autograd`; ablations consume the same seed as full.

**Step 2: Run to verify fail.** **Step 3: Implement minimally** against the
REUSED interfaces (Task-1 inventory line numbers). **Step 4: Run to verify
pass** (including `tests/test_compose_retrieval_pin.py` now green). **Step 5:
Commit** (`feat: net-new regime-correct composition runner (reuse-only;
no autograd)`); controller verifies protected set byte-empty in the diff.

---

## Task 3: Dedicated adversarial review (BEFORE no-harm)

Dispatch a fresh adversarial-reviewer subagent (mirror the reviews that found
real holes earlier this project). Primary mandate:
- Is the compositional capability genuinely emergent from composing the two
  regimes, or a wiring artifact / single-path leakage?
- Are both ablations faithful (identical to full minus exactly one regime,
  same draws)? Is remote-only genuinely the validated hippo-OFF protocol
  byte-unchanged? Is recent-only genuinely just consolidation removed?
- Can a broken/degenerate run be scored PASS? Are the `_CR_*` bars movable by
  results? Is the abstention moat genuinely the byte-unchanged 7/7 gate?
- Any automatic differentiation/training added (must be none — all reused
  validated rules)? Are the validated subsystems genuinely reused unchanged,
  not copy-edited?
STRENGTHEN-only fixes; frozen bars byte-unchanged. Commit any fixes; re-review
until clear. Controller verifies protected set byte-empty.

---

## Task 4: No-harm phase

Prove the full protected set is byte-unchanged across `git diff` from the
pre-Task-0 base to HEAD (MUST be empty for every protected path) and
`tests/test_abstention_gate.py` is still 7/7 green; the full compose-retrieval
test suite green; assert no shipped path imports `torch.autograd` / `.backward`.
Commit the no-harm evidence. Controller trust-but-verify.

---

## Task 5: CONTROLLER-ONLY decisive run (NOT a subagent task)

Controller (not a subagent) performs, in the same turn, never stopping on a
promise:
1. Grounding-first tiny run (`--tiny-synth`) — toy numbers explicitly NOT
   propagated; confirms the pipeline end-to-end + verdict module wiring.
2. Decisive kill-safe multi-seed run at the frozen load ladder (2, 4, 8),
   seeds 42 43 44 (>= `_CR_MIN_SEEDS`), CuPy on RTX 3090, DURABLE output
   capture to `research/findings/raw/`, monitored to actual completion in the
   foreground or via a mechanism that genuinely notifies on completion —
   never a detached process with a false "will be notified"; completion
   actively confirmed before any result is stated.
3. Mandatory smell-test scrutinising a nominal PASS HARDER than a FAIL:
   recompute the verdict from the single recorded output (no re-run, no bar
   change); confirm the full system genuinely succeeds, each ablation
   genuinely collapses its regime part, abstention genuinely holds under both
   ablations, results recomputed from the one recording.
4. Honest propagation of EVERY outcome in plain language: findings doc +
   `webapp/capability_status.json` pillar (status stays PREDICTED until a
   clean scrutinised PASS; schema test green) +
   `research/findings/AUTONOMOUS_STATE.md` updated + commit + push BOTH
   remotes (origin & gitea).
5. Then, per the standing instruction, continue autonomously: a clean
   scrutinised PASS → proceed to the next pre-registered staged step (the
   design's Architecture B: schema-accelerated assimilation), its own
   pre-registered fixed-bar test; an honest non-success (FAIL/VOID/
   WORKS-AT-SMALL-LOAD) → follow the biology to the next integration-fidelity
   refinement and iterate — NOT declare unfit, NOT hand back, NOT
   config-crank, NO bar change. Bring nothing back to the owner as a
   hand-back; surface only as an eyes-open report at the decision point.

**Honest ceiling (never overstated):** a clean success = a biology-grounded
two-system composition answers grounded compositional queries by reading
recent-specific and remote-semantic content each in its correct regime,
holding/improving as load scales, and abstaining rather than confabulating
under ablation — explicitly NOT fluent open-ended language, NOT an LLM, NOT
the retracted transitive-inference claim; prior validated results and honest
boundaries unaffected.
