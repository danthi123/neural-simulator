# 7th arc implementation plan (targeted cue-suppression + amplified tag stim + persistent PFC-frame)

**Goal:** Build the 7th architecture incorporating the four
empirically-targeted mechanisms; controller-only decisive run with
mandatory smell-test + honest propagation; continue the cross-arc
trajectory analysis.

**Architecture:** Net-new runner + Task-1 frozen verdict module
mirroring the prior 6 arcs. Mirrors the 6th arc (`659c2d8`) structure
closely; adds 4 targeted mechanism changes.

**Tech Stack:** CuPy/NumPy via the project's pluggable backend; stdlib
+ typing only for Task-1; reuses all prior arc subsystems
byte-unchanged.

---

### Task 0: Grounding pin

**Files:** Create `tests/test_targeted_cue_suppression_replay_grounding.py`

```python
def test_frozen_verdict_module_importable():
    from research.runners.targeted_cue_suppression_replay_core import (
        targeted_cue_suppression_replay_verdict, REQUIRED_KEYS,
        _TC_FULL_MIN, _TC_UNIFORM_CTRL_MAX, _TC_DIRECT_RETAIN_MIN,
        _TC_ABSTAIN_CORRECT_MIN, _TC_SCALE_TOL, _TC_LADDER, _TC_MIN_SEEDS,
    )
    assert _TC_FULL_MIN == 0.80
    assert _TC_UNIFORM_CTRL_MAX == 0.10
    assert _TC_DIRECT_RETAIN_MIN == 0.80
    assert _TC_ABSTAIN_CORRECT_MIN == 0.90
    assert _TC_SCALE_TOL == 0.10
    assert _TC_LADDER == (2, 3, 5)
    assert _TC_MIN_SEEDS == 3

def test_runner_main_importable():
    from research.runners.targeted_cue_suppression_replay_runner import main
    assert callable(main)
```

RED until Tasks 1+2.

---

### Task 1: Frozen capability-verdict module

**Files:**
- Create: `research/runners/targeted_cue_suppression_replay_core.py`
- Create: `tests/test_targeted_cue_suppression_replay_core.py`

Transcribe `per_regime_monitor_core.py` byte-for-byte with:
- Constant rename: `_PR_*` -> `_TC_*` (VALUES UNCHANGED)
- Function rename: `per_regime_monitor_verdict` -> `targeted_cue_suppression_replay_verdict`
- Docstring updated to reference 7th arc + cross-arc trajectory motivation
- 18+ adversarial test cases (transcribe from existing arc test files)

---

### Task 2: Net-new runner

**Files:** Create `research/runners/targeted_cue_suppression_replay_runner.py`

Mirror `generative_replay_pfc_frame_runner.py` (commit `13f73e8`) with these targeted changes:

1. **Imports**: add nothing new beyond what 6th arc imports. Frozen verdict from `targeted_cue_suppression_replay_core`.

2. **New constants** (replacing 6th arc defaults):
   - `REPLAY_CYCLES_PER_TAG = 50` (was 20)
   - `RETRIEVE_TAG_AMP_FACTOR = 3.0` (the amplified-tag-stim factor; baseline 1500 -> 4500 pA effective)
   - `PFC_FRAME_STIM_STEPS = 50` (was 10)

3. **Cue-suppression-during-replay helper** (NEW; the load-bearing genuine net-new mechanism):

```python
def _run_replay_with_cue_suppressed(bridge, tag_names, n_replays_per_tag):
    """Run the validated SWR replay phase with the cortico-hippocampal
    cue input pathway gated DOWN. Reuses run_concept_replay_phase
    byte-unchanged; just suppresses the lang_to_ec transmission via
    bridge.set_plasticity_gate('lang_to_ec', 0.0) before invocation
    (no plasticity during replay anyway, so this is consistent with the
    consolidation regime; net effect is no cue contamination in the
    consolidation signal).

    NOTE: plasticity_gate suppresses UPDATE rate, not transmission per
    the documented CLAUDE.md gotcha. For genuine TRANSMISSION suppression
    during replay, we additionally write 0.0 to the bridge's
    cp_external_input_current[lang_input slice] during the replay window.
    Both mechanisms together ensure the cue doesn't contaminate the
    replay-induced consolidation.
    """
    # Save current cue input state
    n_lang_input = ...
    saved_lang_input = bridge.cp_external_input_current[:n_lang_input].copy()
    # Zero cue input during replay
    bridge.cp_external_input_current[:n_lang_input] = 0.0
    # Save lang_to_ec gate state
    try:
        saved_lang_to_ec_gate = bridge.get_plasticity_gate('lang_to_ec')
    except Exception:
        saved_lang_to_ec_gate = None
    bridge.set_plasticity_gate('lang_to_ec', 0.0)
    try:
        # Run reused replay subsystem (byte-unchanged)
        run_concept_replay_phase(bridge, tag_names=tag_names,
                                  n_replays_per_tag=n_replays_per_tag)
    finally:
        # Restore
        bridge.cp_external_input_current[:n_lang_input] = saved_lang_input
        if saved_lang_to_ec_gate is not None:
            bridge.set_plasticity_gate('lang_to_ec', saved_lang_to_ec_gate)
```

4. **Amplified-tag-stim wrapper** (NEW):

```python
def _compositional_query_amplified(bridge, cue_noun, tag_name, dims,
                                     recall_steps, tag_amp_factor):
    """Wrapper around _compositional_query_ranked that amplifies the
    engram tag stim by tag_amp_factor. The cue stays at baseline drive_pA.

    Since _compositional_query_ranked doesn't accept tag amp as a param,
    we replicate its logic locally with the amplified tag drive.
    """
    # ... (transcribe the function body with tag_drive_pA *= tag_amp_factor)
```

5. **Persistent PFC-frame** (UPDATED from 6th arc):
   - Change `PFC_FRAME_STIM_STEPS` from 10 to 50
   - Per-step write inside the stim loop (Pirazzini FIX B pattern; same as 6th arc)

6. **Eval arm structure**:
   - Build two parallel bridges (full + uniform_ctrl) with same Phase-1 cache
   - Encode pairs (identical RNG on both via `_seed_query_rng`)
   - **FULL arm**: `_run_replay_with_cue_suppressed(bridge_full, tags, REPLAY_CYCLES_PER_TAG=50)`
   - **UNIFORM_CTRL arm**: `run_concept_replay_phase(bridge_uniform, tags, n_replays_per_tag=20)` (no cue suppression; baseline replay count)
   - **Direct queries**: same gate on both arms (identical outcomes)
   - **Compositional queries**: 
     - FULL: `_prime_pfc_frame(bridge_full, ..., n_steps=50)` then `_compositional_query_amplified(bridge_full, ..., tag_amp_factor=3.0)`
     - UNIFORM_CTRL: skip PFC-frame; use baseline tag amp
   - Cache-scale validation in probes (per `13f73e8`)

7. **Structural-effect probes** (MANDATORY):
   - Cue-suppression-during-replay probe: bridge_with_cue_supp_replay vs bridge_without; same RNG; > 1 mV divergence; controls 0.00 mV
   - Amplified-tag-stim probe: similar
   - Persistent-PFC-frame probe: similar (50-step vs 10-step)
   - All probe controls assert RNG isolation works

8. **CLI**: same as 6th arc.

9. **Tests in `tests/test_targeted_cue_suppression_replay_runner.py`**:
   - `test_runner_module_exposes_entry_point`
   - `test_structural_effect_probes_validate_all_three_mechanisms`
   - `test_tiny_synth_smoke_outputs_expected_json_shape`
   - `test_no_autograd_on_shipped_path`
   - `test_full_vs_uniform_arms_differ_at_least_on_some_query`
   - `test_cache_scale_mismatch_raises` (re-use pattern from `13f73e8`)

---

### Task 3: 12th consecutive dedicated adversarial review

Dispatch subagent. Specific exploit-class probes:
1. Cue-suppression-during-replay effect probe (verify mechanism active; controls clean)
2. Amplified-tag-stim effect probe (same)
3. Persistent-PFC-frame effect probe (same; verify 50-step window genuinely extends NMDA dynamics)
4. False-PASS vectors (no-op cue-suppression; no-op amp; no-op extended PFC-frame)
5. Subsystem byte-unchanged audit
6. No autograd / no torch / no protected modification
7. Frozen-bar immutability
8. Test coverage adequacy
9. **NEW for 7th arc**: cache-scale validation re-applied (per `13f73e8` lesson); verify all three probes have the same validator

---

### Task 4: No-harm verification

```bash
python -m pytest tests/test_abstention_gate.py tests/test_abstention_gate_compositional.py tests/test_abstention_gate_compositional_unified.py tests/test_abstention_gate_direct_unified.py tests/test_theta_gamma_mode_unification_core.py tests/test_theta_gamma_mode_unification_grounding.py tests/test_theta_gamma_mode_unification_runner.py tests/test_generative_replay_pfc_frame_core.py tests/test_generative_replay_pfc_frame_grounding.py tests/test_generative_replay_pfc_frame_runner.py tests/test_targeted_cue_suppression_replay_core.py tests/test_targeted_cue_suppression_replay_grounding.py tests/test_targeted_cue_suppression_replay_runner.py tests/test_unified_per_regime_monitor_runner.py -q
```

Expected: all green. No-confab moat 7/7. Protected set byte-empty.

---

### Task 5: Controller-only decisive run + smell-test + honest propagation

**Controller runs directly:**

```bash
python -m research.runners.targeted_cue_suppression_replay_runner \
    --seeds 42 43 44 \
    --phase1-cache-dir research/findings/raw/unified_per_regime/phase1 \
    --ckpt research/findings/raw/targeted_replay_decisive.ckpt \
    --out research/findings/raw/targeted_replay_DECISIVE_fullscale.json
```

Background; genuine completion waiter.

**Mandatory smell-test recompute** (reused script):

```bash
python research/findings/raw/unified_DECISIVE_smell_test.py \
    research/findings/raw/targeted_replay_DECISIVE_fullscale.json
```

**Honest propagation EVERY outcome**:
- PASS: FIRST architecture to clear bars; biology-grounded compositional retrieval validated
- Partial improvement (gap closure ~35%+): continues the trajectory; queue 8th arc
- Plateau (no further gap closure): substrate-level asymptote; pivot to deeper refinement OR honest closure
- VOID: instrument-validity issue; investigate

Update AUTONOMOUS_STATE with the next exact action regardless. Never end on a future-tense promise.

---

## Discipline pins (mirrors prior 6 arcs)

- NO bar change; `_TC_*` constants frozen.
- NO protected file modification; protected set byte-empty diff vs `e8a99a2`.
- NO autograd / no torch / no LLM.
- NO declare-unfit; NO hand-back.
- Mandatory dedicated adversarial review BEFORE no-harm BEFORE decisive.
- Honest propagation EVERY outcome both remotes.
- Same-turn discipline.
- 4 calibrated moats + no-confab moat byte-stable.

## Execution handoff

Standing autonomy: same-session subagent-driven. Dispatch Task 0 first.
