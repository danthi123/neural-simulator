# Theta-gamma mode-unification implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or
> superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build the theta-gamma mode-unification architecture (5th
arc; cue-suppression-during-retrieve mechanism per the localisation
finding) on the cached unified substrate; produce per-rung capability
metrics for the frozen verdict; controller-only decisive run with
mandatory smell-test + honest propagation.

**Architecture:** Net-new runner + Task-1 frozen capability-verdict
module mirroring the prior 4 arcs (Stage-1 / SPEAR / Pirazzini /
Unified per-regime monitor). The only genuine net-new piece is the
shared-theta-rhythm controller + cue-suppression wiring; everything
else is reuse-by-import.

**Tech Stack:** CuPy/NumPy via the project's existing pluggable backend;
stdlib + typing only for the Task-1 frozen module; reuses
`build_biological_brain_regions`, `encode_concept_pair`,
`_compositional_query_ranked`, `per_regime_monitor_core` (REFERENCE
ONLY -- the new module is structurally analogous), the neuromodulator
subsystem byte-unchanged, and the 4 calibrated abstention moats
byte-unchanged.

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_theta_gamma_mode_unification_grounding.py`

**Step 1: Write the failing pin test**

```python
"""Grounding pin for theta-gamma mode-unification arc.
RED until Task 1 + Task 2 land. Verifies that the new frozen verdict
module + runner module are importable + the runner exposes a main()
entry point + the frozen-bars constants are pinned.
"""
def test_frozen_verdict_module_importable():
    from research.runners.theta_gamma_mode_unification_core import (
        theta_gamma_mode_unification_verdict,
        REQUIRED_KEYS,
        _TG_FULL_MIN, _TG_UNIFORM_CTRL_MAX, _TG_DIRECT_RETAIN_MIN,
        _TG_ABSTAIN_CORRECT_MIN, _TG_SCALE_TOL, _TG_LADDER, _TG_MIN_SEEDS,
    )
    assert _TG_FULL_MIN == 0.80
    assert _TG_UNIFORM_CTRL_MAX == 0.10
    assert _TG_DIRECT_RETAIN_MIN == 0.80
    assert _TG_ABSTAIN_CORRECT_MIN == 0.90
    assert _TG_SCALE_TOL == 0.10
    assert _TG_LADDER == (2, 3, 5)
    assert _TG_MIN_SEEDS == 3

def test_runner_main_importable():
    from research.runners.theta_gamma_mode_unification_runner import main
    assert callable(main)
```

**Step 2: Run + verify RED**: `pytest tests/test_theta_gamma_mode_unification_grounding.py` -> fail (modules don't exist).

**Step 3: Commit** (grounding pin; RED expected until Task 2).

---

### Task 1: Frozen capability-verdict module

**Files:**
- Create: `research/runners/theta_gamma_mode_unification_core.py`
- Create: `tests/test_theta_gamma_mode_unification_core.py`

**Module contents (transcribe from `per_regime_monitor_core.py` byte-for-byte EXCEPT rename _PR_* -> _TG_* and rename function):**

```python
"""Theta-gamma mode-unification stage: pre-registered fixed-bar verdict instrument."""
from __future__ import annotations
import math
from typing import Any, Dict

_TG_FULL_MIN = 0.80
_TG_UNIFORM_CTRL_MAX = 0.10
_TG_DIRECT_RETAIN_MIN = 0.80
_TG_ABSTAIN_CORRECT_MIN = 0.90
_TG_SCALE_TOL = 0.10
_TG_LADDER = (2, 3, 5)
_TG_MIN_SEEDS = 3

REQUIRED_KEYS = ("N", "n_seeds", "full_acc", "uniform_ctrl_acc",
                 "direct_retain_acc", "abstain_correct")
_ACC_KEYS = ("full_acc", "uniform_ctrl_acc", "direct_retain_acc", "abstain_correct")

# ... (transcribe _finite_number, _frozen_bars, _void, _fail, _works_small, _pass,
# theta_gamma_mode_unification_verdict from per_regime_monitor_core.py with the
# bars-rename only; same logic; stdlib + typing; ASCII; malformed -> VOID never raise;
# instrument-validity FIRST; VOID strictly distinct from FAIL).
```

**Tests (17+ adversarial cases; transcribe from `tests/test_per_regime_monitor_core.py` byte-for-byte except rename verdict function + bars):**
- Smallest-N rung at frozen bars -> PASS
- Smallest-N rung below any bar -> FAIL
- Larger-N rung below bars -> WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE
- full_acc drop > _TG_SCALE_TOL -> WORKS-AT-SMALL-LOAD
- Malformed rung (missing key, non-list, empty) -> VOID
- N not in ladder -> VOID
- Duplicate N -> VOID
- acc out of [0,1] -> VOID
- n_seeds below min -> VOID
- Non-finite / non-numeric acc -> VOID
- (Plus the prior arcs' adversarial cases byte-shape verbatim)

**Step 1**: Write failing tests (RED).
**Step 2**: Verify RED.
**Step 3**: Transcribe implementation; verify GREEN.
**Step 4**: Commit.

---

### Task 2: Net-new theta-gamma runner

**Files:**
- Create: `research/runners/theta_gamma_mode_unification_runner.py`

**Specification** (genuine net-new integration; ~700-900 lines mirroring `unified_per_regime_monitor_runner.py` structure):

- Imports: `from research.runners.theta_gamma_mode_unification_core import theta_gamma_mode_unification_verdict, _TG_*`; reuse-by-import from `unified_per_regime_monitor_runner` for `_build_bridge_with_phase1_recipe`, `_phase1_recipe`, `_phase1_cache_path`, `_freeze_phase1_gates`, `_all_pool_regions`, `_all_words_word_to_idx`, `_direct_pool_target`, `_direct_query_ranked`, `_compositional_query_ranked`, `_unified_compositional_pairs`, `_encode_facts`; abstention moats (650 + 5.6887 + 0.197712 + 0.284167) byte-unchanged.

- Shared-theta-rhythm controller (the genuine net-new piece): a function `_apply_theta_phase(bridge, phase, theta_step, n_lang_input, cue_drive_pattern, ach_modulator_name)` that writes per-phase modulation:
  - Phase ENCODE (theta-trough; steps 0..ENCODE_STEPS): bridge.cp_external_input_current[lang_input slice] = cue_drive_pattern * drive_pA; modulator(ach) = HIGH
  - Phase GAP (steps ENCODE_STEPS..ENCODE_STEPS+GAP_STEPS): bridge.cp_external_input_current[lang_input slice] = 0; modulator(ach) ramps down
  - Phase RETRIEVE (theta-peak; steps ENCODE_STEPS+GAP_STEPS..CYCLE_STEPS): bridge.cp_external_input_current[lang_input slice] = 0 (CUE SUPPRESSED); modulator(ach) = LOW; CA3 recurrence pathway gates open (gate name `ca3_recurrent` if exists, else no-op); the `_compositional_query_ranked` readout measures lang_output during this window ONLY
  - The function MUST be called at the runner's per-step loop, not at step_idx=0 (the Pirazzini doubly-inert defect)
  - The cue suppression MUST persist across `encode_concept_pair` clearing `cp_external_input_current` (the Pirazzini defect-2): the runner schedules the write AFTER any sub-helper's clear

- Per-cell evaluation arm (`_run_eval_arm(seed, N, tiny_synth, cache_dir)`):
  - Load substrate + Phase-1 checkpoint + freeze plasticity gates
  - Encode N pairs via `_encode_facts` reused byte-unchanged
  - Direct queries: ranked = `_direct_query_ranked(...)`; full and uniform_ctrl arms BOTH route through `gate_direct_unified(ranked, DIRECT_UNIFIED_THRESHOLD)` (same as prior unified arc)
  - Compositional queries: per query, run THE THETA-GAMMA CYCLE: encode/gap/retrieve. ranked is measured ONLY during the retrieve phase (with cue suppressed). FULL arm uses `gate_compositional_unified(ranked, COMPOSITIONAL_UNIFIED_THRESHOLD)`. UNIFORM_CTRL arm runs the same encode/gap/retrieve cycle EXCEPT cue suppression is OFF (cue stays ON during retrieve); ranked is still measured during the same retrieve window; gate is the same.
  - Ungroundable queries: same theta cycle but with non-encoded cue word; abstain_correct counts when gate emits None
  - Per-cell raw metrics: n_direct_correct_full, n_direct_correct_uniform, n_comp_correct_full, n_comp_correct_uniform, n_abstain_ok, n_ungroundable, n_direct, n_compositional

- Adversarial structural-effect pin (MANDATORY; mirrors Pirazzini fix d462bf0 lesson): a small probe at the runner's main() entry runs a 50-step cycle with theta-gamma ON vs OFF on a fresh substrate; assert that the resulting `bridge.cp_membrane_potential_v` differs by >1mV bridge-state divergence; if byte-identical, the runner raises a clear error message (the mechanism is structurally inert; fix and re-run BEFORE the decisive run). This catches the defect class that took the Pirazzini arc multiple iterations to localise.

- Kill-safe via `sim.train_checkpoint` byte-unchanged (mirrors prior arcs).
- CLI: `--seeds 42 43 44 --loads 2 3 5 --phase1-cache-dir <dir> --ckpt <ckpt> --out <json>`.
- Output JSON: `{mode: "evaluation", rungs: [{N, n_seeds, full_acc, ...}], verdict: {gate: ..., reason: ..., frozen_bars: ...}, raw_cells: [...], ...}` matching the per_regime_monitor_runner output shape exactly so the smell-test recompute script reuses byte-unchanged.

**Step 1**: Write failing integration test that calls main() with --tiny-synth and asserts the output JSON has the expected shape + the verdict gate is one of {PASS, FAIL, VOID, WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE}.
**Step 2**: Verify RED.
**Step 3**: Implement the runner with the cue-suppression theta-cycle wiring + the structural-effect probe; verify the integration test GREEN.
**Step 4**: Re-run the Task 0 grounding pin; verify GREEN.
**Step 5**: Commit.

---

### Task 3: Dedicated adversarial review (EIGHTH consecutive review)

Dispatch a subagent (NOT a controller-side review). Prompt mirrors prior 4 arcs' adversarial review prompts:

Specific exploit-class probes the reviewer MUST run:
1. **Structural-effect probe** (the Pirazzini lesson): theta-gamma ON vs OFF on a 50-step constant-input cycle MUST produce non-byte-identical bridge state via the runner's ACTUAL code path (not a synthetic per-step loop bypass). Run the probe and verify the bridge-state divergence is real (>1mV typical).
2. **False-PASS vector**: construct a degenerate substrate / cue-suppression-only-pretending mechanism (e.g., cue suppressed but ACh still high; or cue suppressed but readout window misaligned with the suppress window) and verify the verdict is FAIL or VOID, not a false PASS.
3. **Byte-unchanged audit**: every protected file + the 4 calibrated moats + `per_regime_monitor_core.py` (the reference verdict module) + the existing `unified_per_regime_monitor_runner.py` (REUSED via import only) byte-unchanged vs `e8a99a2`.
4. **No autograd / no torch**: zero matches for `import torch`, `import torch.autograd`, `backward(`, `requires_grad` anywhere in the new arc.
5. **No protected-module modification**: the only files touched are the 3 net-new files; everything else byte-unchanged.
6. **Sub_seed isolation**: the runner's deterministic noise (theta-step indexing, ACh ramp) must be reproducible across runs with the same seed.
7. **Frozen-bar immutability**: the `_TG_*` constants are set ONCE in `theta_gamma_mode_unification_core.py` and NEVER referenced for re-tuning in the runner; the runner's verdict path uses ONLY the imported function.
8. **Pirazzini doubly-inert prevention**: verify `_apply_theta_phase` is called at every per-step in the runner (not step_idx=0 hardcoded); verify the cue-suppression write SURVIVES any sub-helper's `cp_external_input_current[:] = 0` clear (must be re-applied per step or scheduled via a persistent mechanism).

Report classification: BLOCK (load-bearing defect) | CLEAR (no load-bearing defect) | CLEAR-WITH-NOTES (cosmetic items only). Fix any BLOCK in a net-new-runner-only follow-up commit; re-review until CLEAR.

---

### Task 4: No-harm verification

```bash
python -m pytest tests/test_abstention_gate.py tests/test_abstention_gate_compositional.py tests/test_abstention_gate_compositional_unified.py tests/test_abstention_gate_direct_unified.py tests/test_theta_gamma_mode_unification_core.py tests/test_theta_gamma_mode_unification_grounding.py tests/test_unified_per_regime_monitor_runner.py -q
```

Expected: all green. No-confab moat 7/7 byte-identical.

```bash
git diff --stat e8a99a2..HEAD -- research/runners/abstention_gate.py tests/test_abstention_gate.py sim/td_value_critic.py sim/compose_temporal_bind.py sim/kernels.py sim/bridge.py sim/neuromodulators.py sim/train_checkpoint.py sim/backend.py sim/dendritic_plasticity.py research/runners/text_minimal_isolation.py
```

Expected: empty (protected set byte-empty).

---

### Task 5: Controller-only decisive run + smell-test + honest propagation

**This is NOT a subagent task. Controller runs it directly.**

```bash
python -m research.runners.theta_gamma_mode_unification_runner \
    --seeds 42 43 44 \
    --phase1-cache-dir research/findings/raw/unified_per_regime/phase1 \
    --ckpt research/findings/raw/theta_gamma_decisive.ckpt \
    --out research/findings/raw/theta_gamma_DECISIVE_fullscale.json
```

Launch in background with kill-safe `--ckpt`; arm a genuine completion waiter (until `! ps -p $PID`).

**Mandatory smell-test recompute** (after decisive completes):

```bash
python research/findings/raw/unified_DECISIVE_smell_test.py \
    research/findings/raw/theta_gamma_DECISIVE_fullscale.json
```

(Update the smell-test script to also accept the theta-gamma verdict module; or write a thin twin.)

The smell-test recompute MUST match the runner-reported verdict exactly. Per-rung internal consistency + ladder + scale-tolerance checks all green. ANY discrepancy = instrument-validity issue requiring localisation, NOT a verdict to publish.

**Honest propagation EVERY outcome both remotes**:
- PASS: findings doc + capability_status.json update + AUTONOMOUS_STATE update + push both remotes; this would be the FIRST architecture in the 5-arc series to clear the frozen bars at biological scale -- a SUBSTANTIVE biology-translatable positive finding; the next conversational stage queues automatically.
- FAIL: findings doc + AUTONOMOUS_STATE update + push both remotes; the 5-architecture convergent ceiling extends; the next staged step is deeper-mechanism design (e.g., generative replay + PFC compositional frame as a phase-multiplexed addition) OR honest closure of the design line as a terminal finding.
- VOID: findings doc + AUTONOMOUS_STATE update + push both remotes; investigate instrument-validity issue (e.g., n_seeds, ladder, NaN, out-of-bounds); fix; re-run.
- WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE: findings doc + propagation; the next staged step is targeted at improving the scale-tolerance OR honest framing.

Update AUTONOMOUS_STATE with the next exact action regardless of outcome. Never end on a future-tense promise. The autonomous next-action tool call is always in the same turn.

---

## Discipline pins (mirrors prior 4 arcs)

- NO bar change anywhere; the `_TG_*` constants are FROZEN before any results.
- NO protected file modification; protected set byte-empty diff vs `e8a99a2` holds.
- NO autograd / no torch / no LLM call.
- NO declare-unfit; NO hand-back; NO config-crank.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE decisive run.
- Honest propagation EVERY outcome both remotes (`origin` + `gitea`).
- The autonomous next-action tool call is always in the same turn.
- The 4 substrate-and-protocol-specific calibrated moats stay byte-stable.
- The no-confabulation moat (`abstention_gate.py` + `tests/test_abstention_gate.py`) stays byte-identical and 7/7 green.

## Execution handoff

After this plan ships, the standing autonomy directive pre-selects same-session
subagent-driven execution. Transition directly to subagent-driven-development:
dispatch a subagent for Task 0 (grounding pin), then Task 1 (frozen verdict),
then Task 2 (runner). Each task = exactly the files specified; controller
verifies protected-set byte-empty after each commit; no-confab moat 7/7 after
each commit. The eighth consecutive dedicated adversarial review is Task 3.
Tasks 4 + 5 are controller-only. After Task 5: honest propagation + autonomous
next staged step per outcome.
