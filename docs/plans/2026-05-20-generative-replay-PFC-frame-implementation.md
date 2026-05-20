# Generative replay + PFC-held compositional frame implementation plan (6th arc)

> **For Claude:** Use superpowers:subagent-driven-development to implement
> this plan task-by-task. Standing user directive: same-session
> subagent-driven execution; no execution-choice prompt; transition
> directly.

**Goal:** Build the generative-replay + PFC-frame architecture on the
cached unified substrate; produce per-rung capability metrics for the
frozen verdict; controller-only decisive run + mandatory smell-test +
honest propagation.

**Architecture:** Net-new runner + Task-1 frozen capability-verdict
module mirroring the prior 5 arcs. The genuine net-new piece is the
generative-replay phase between encode and eval + the PFC-frame priming
during eval queries; everything else is reuse-by-import.

**Tech Stack:** CuPy/NumPy via the project's pluggable backend; stdlib
+ typing only for the Task-1 frozen module; reuses
`build_biological_brain_regions(..., enable_dlpfc_verb=True, enable_pfc_nmda=True)`,
`encode_concept_pair`, `run_concept_replay_phase`, the 4 calibrated
abstention moats byte-unchanged.

---

### Task 0: Grounding pin

**Files:**
- Create: `tests/test_generative_replay_pfc_frame_grounding.py`

```python
def test_frozen_verdict_module_importable():
    from research.runners.generative_replay_pfc_frame_core import (
        generative_replay_pfc_frame_verdict, REQUIRED_KEYS,
        _GR_FULL_MIN, _GR_UNIFORM_CTRL_MAX, _GR_DIRECT_RETAIN_MIN,
        _GR_ABSTAIN_CORRECT_MIN, _GR_SCALE_TOL, _GR_LADDER, _GR_MIN_SEEDS,
    )
    assert _GR_FULL_MIN == 0.80
    assert _GR_UNIFORM_CTRL_MAX == 0.10
    assert _GR_DIRECT_RETAIN_MIN == 0.80
    assert _GR_ABSTAIN_CORRECT_MIN == 0.90
    assert _GR_SCALE_TOL == 0.10
    assert _GR_LADDER == (2, 3, 5)
    assert _GR_MIN_SEEDS == 3

def test_runner_main_importable():
    from research.runners.generative_replay_pfc_frame_runner import main
    assert callable(main)
```

RED until Tasks 1 + 2. Commit as "Task 0 grounding pin".

---

### Task 1: Frozen capability-verdict module

**Files:**
- Create: `research/runners/generative_replay_pfc_frame_core.py`
- Create: `tests/test_generative_replay_pfc_frame_core.py`

Transcribe `research/runners/per_regime_monitor_core.py` byte-for-byte with:
- Constant rename: `_PR_*` -> `_GR_*` (all 7 frozen bars; VALUES UNCHANGED)
- Function rename: `per_regime_monitor_verdict` -> `generative_replay_pfc_frame_verdict`
- Module docstring updated to reference the 6th arc + design doc + 5-architecture ceiling motivation
- Everything else byte-identical (stdlib + typing only; instrument-validity-FIRST; VOID strictly distinct from FAIL; malformed -> VOID never crash)

Tests: transcribe from the verdict module's test pattern (~18 tests) with renames. All must PASS.

Verify: protected set byte-empty diff vs `e8a99a2`; no-confab moat 7/7;
Task 0 `test_frozen_verdict_module_importable` now GREEN (runner-import
still RED until Task 2).

---

### Task 2: Net-new runner

**Files:**
- Create: `research/runners/generative_replay_pfc_frame_runner.py`

Mirror `unified_per_regime_monitor_runner.py` structure (~700-900
lines) with these changes:

1. **Imports**: add `from research.runners.consolidation_trainer import
   run_concept_replay_phase`. Frozen verdict from
   `generative_replay_pfc_frame_core`.

2. **Eval arm structure**:
   - Encode N pairs via reused `_encode_facts`
   - **NEW (full arm only)**: Run
     `run_concept_replay_phase(bridge_full, tag_names=tags,
     n_replays_per_tag=20)` BEFORE eval queries
   - Eval each query on BOTH bridges (full + uniform_ctrl); cue PRESENT
     in both arms during retrieve (encoding-specificity respected)
   - **NEW (full arm only)**: BEFORE each compositional query on
     bridge_full, brief PFC-frame drive on dlpfc_verb (e.g.,
     `bridge.cp_external_input_current[dlpfc_slice] += 200 pA` for ~10
     steps; relies on NMDA bistability to hold the frame)
   - The uniform_ctrl arm: skip replay + skip PFC-frame priming

3. **RNG isolation** (per the eighth review's lesson): seed
   `cp.random` (or backend xp.random) deterministically before each
   query so both arms see identical noise; the SOLE difference is the
   augmenting mechanisms.

4. **Structural-effect probe** (MANDATORY; mirrors theta-gamma `e6b17da`
   lesson):
   - bridge_with_replay vs bridge_no_replay (post-replay membrane
     potentials must differ by > 1 mV)
   - bridge_with_replay vs bridge_with_replay (control; same RNG; both
     replay; must show ~0 mV)
   - bridge_no_replay vs bridge_no_replay (control; same RNG; both
     skip replay; must show ~0 mV)
   - Raise RuntimeError if any control fails.

5. **CLI**: `--seeds`, `--loads` (default `_GR_LADDER` = (2, 3, 5)),
   `--tiny-synth`, `--phase1-cache-dir`, `--ckpt`, `--out`.

6. **Output JSON**: same shape as prior arcs (per-rung + per-cell +
   verdict + frozen_bars) so the smell-test recompute script reuses
   byte-unchanged.

Tests:
- `test_runner_main_importable` (covered by grounding pin)
- `test_structural_effect_probe_validates_replay_mechanism` (probe
  with controls)
- `test_tiny_synth_smoke_outputs_expected_json_shape`
- `test_full_vs_uniform_arms_differ_at_least_on_some_query`

Verify: all new tests PASS at tiny-synth; 18-test verdict + 2-test pin
+ no-confab moat 7/7 all PASS; protected set byte-empty.

---

### Task 3: Tenth consecutive dedicated adversarial review

Dispatch a subagent. Specific exploit-class probes:

1. **Replay-effect probe verification.** Verify the structural-effect
   probe genuinely isolates the replay mechanism (controls match;
   flag-differing > 1 mV).

2. **PFC-frame-effect probe.** Run a separate probe: bridge with
   PFC-frame priming vs without (same RNG; same replay state). Verify
   bridge state differs (>1 mV) when PFC-frame priming is on.

3. **Encoding-specificity preserved.** The cue MUST be present during
   retrieve in BOTH arms (the theta-gamma lesson). Verify by reading
   the eval-arm code; the only difference between arms should be the
   AUGMENTING mechanisms (replay + PFC-frame), not the cue presence.

4. **False-PASS vector probes.**
   - Could a replay-only-pretending mechanism (e.g., function called
     but no actual STDP update) score PASS? Verify the replay phase
     actually modifies bridge state (post-replay synaptic weights differ
     from pre-replay).
   - Could a PFC-frame-only-pretending mechanism (write but no NMDA
     response) score PASS? Verify the dlpfc_verb region is firing
     during the eval queries.

5. **Subsystem byte-unchanged audit.** `run_concept_replay_phase` from
   `consolidation_trainer.py` byte-unchanged. `dlpfc_verb` region
   construction byte-unchanged. NMDA subsystem byte-unchanged.

6. **No autograd / no torch / no protected modification.** `git diff
   e8a99a2..HEAD --stat` on the 11 protected files; all empty. zero
   `import torch` / `autograd` / `backward(`.

7. **Frozen-bar immutability.** `_GR_*` constants defined ONCE in core
   module; never tuned in runner.

8. **Tests cover the load-bearing routing.** Read new tests; verify
   they exercise replay + PFC-frame mechanisms (not just imports).

Classification: CLEAR / CLEAR-WITH-NOTE / BLOCK. Fix any BLOCK via
net-new-runner-only follow-up; re-review until CLEAR.

---

### Task 4: No-harm verification

```bash
python -m pytest tests/test_abstention_gate.py tests/test_abstention_gate_compositional.py tests/test_abstention_gate_compositional_unified.py tests/test_abstention_gate_direct_unified.py tests/test_theta_gamma_mode_unification_core.py tests/test_theta_gamma_mode_unification_grounding.py tests/test_theta_gamma_mode_unification_runner.py tests/test_generative_replay_pfc_frame_core.py tests/test_generative_replay_pfc_frame_grounding.py tests/test_generative_replay_pfc_frame_runner.py tests/test_unified_per_regime_monitor_runner.py -q
```

Expected: all green. No-confab moat 7/7 byte-identical. Protected set
byte-empty.

---

### Task 5: Controller-only decisive run + smell-test + honest propagation

**Controller runs directly:**

```bash
python -m research.runners.generative_replay_pfc_frame_runner \
    --seeds 42 43 44 \
    --phase1-cache-dir research/findings/raw/unified_per_regime/phase1 \
    --ckpt research/findings/raw/generative_replay_decisive.ckpt \
    --out research/findings/raw/generative_replay_DECISIVE_fullscale.json
```

Launch in background; genuine completion waiter.

**Mandatory smell-test recompute**:

```bash
python research/findings/raw/unified_DECISIVE_smell_test.py \
    research/findings/raw/generative_replay_DECISIVE_fullscale.json
```

The smell-test recompute MUST match the runner-reported verdict exactly.

**Honest propagation EVERY outcome**:
- PASS: would be the FIRST architecture in the 6-arc series to clear
  the frozen bars; substantive biology-translatable positive finding
- FAIL: 6-architecture convergent ceiling extends; terminal finding
  for the gating-based composition design line at this substrate
- VOID: instrument-validity issue; investigate; fix; re-run
- WORKS-AT-SMALL-LOAD: scale-tolerance finding; documented as such

Update AUTONOMOUS_STATE with the next exact action regardless. Never
end on a future-tense promise.

---

## Discipline pins (mirrors prior 5 arcs)

- NO bar change anywhere; `_GR_*` constants FROZEN before any results.
- NO protected file modification; protected set byte-empty diff vs
  `e8a99a2` holds.
- NO autograd / no torch / no LLM call.
- NO declare-unfit; NO hand-back; NO config-crank.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE decisive.
- Honest propagation EVERY outcome both remotes.
- The autonomous next-action tool call is always in the same turn.
- 4 calibrated moats + no-confab moat byte-stable.

## Execution handoff

Standing autonomy: same-session subagent-driven. Transition directly to
subagent-driven-development: dispatch Task 0 grounding-pin subagent;
then Task 1 frozen verdict; then Task 2 runner; then Task 3 adversarial
review subagent; Tasks 4 + 5 are controller-only. No execution-choice
prompt.
