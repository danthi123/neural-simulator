# Generative replay — TDD implementation plan

> Companion to `docs/plans/2026-05-23-generative-replay-design.md` (commit 97f21c5).
> Substrate decision empirically resolved by pillars n=96/n=97 (and pending
> n=98 confirming dlpfc_wm-extension PASS): the `build_biological_brain_regions`
> substrate with hippocampus + Phase 1.3 SWR consolidation + dlpfc_wm region
> validated to support parallel-matching biologized mode-unification at the
> 16-concept tier with multi-seed margin. This plan defines the TDD steps to
> build the genuinely-new generative-replay loop controller on top of that
> substrate.

## Status

**CONDITIONAL on dlpfc_wm-extension probe (in-flight; harness task `b7lyujjei`) PASS.**
If DLPFC PASS, this plan executes directly. If DLPFC NEGATIVE, the plan needs
substrate revision (try smaller dlpfc_wm region; or use g11_bg_runner's pattern
verbatim with all the cortex_X regions; or reconsider OPTION 1 substrate-merge).

## Substrate prerequisites (all VALIDATED on the same substrate)

- v14/v16 16-pool concept architecture with W→A multi-seed binding (88.75%)
- Hippocampus EC/DG/CA3/CA1 trisynaptic loop (D.12 separation + D.13 completion)
- Engram tagging (D.14)
- Phase 1.3 SWR consolidation (3/3 strict anti-cheat multi-seed)
- dlpfc_wm NMDA bistable PFC working memory region (pillar n=98 pending)
- Parallel-matching biologized mode-unification (pillars n=93/n=94/n=96/n=97)
- Resonate-and-fire FHRR bind/unbind/bundle (FHRR-biologization arc)
- Common-mode-removed grounded symbols (pillar n=84 + extensions)
- Spiking-phasor attractor + familiarity gate (clean-up; FHRR arc)
- Abstention gate (no-confab moat 7/7; preserved throughout)

## Genuinely-new code (this plan)

The generative-replay loop CONTROLLER. ~300-500 lines net-new (controller +
runner + tests). Structure:

1. `encode_pfc_frame(items, positions)`: produces the FHRR composite C from
   the initial K-tuple of bound concepts; injects C into dlpfc_wm via the
   lang_input → dlpfc_wm pathway (gated open during injection; closed during
   replay).
2. `trigger_swr_replay(bridge, n_steps)`: opens the ca3_swr_burst gate;
   runs the bridge for n_steps; closes the gate. This is the validated Phase
   1.3 mechanism reused byte-unchanged.
3. `capture_post_replay_cortical_activity(bridge, ...)`: per-neuron firing
   across the 16-pool union over a stim window (same primitive as OPTION 3
   capture).
4. `decode_continuation(activity, grounded_vocab, position)`: applies the
   parallel-matching decoder at the next gamma-slot position to identify the
   replayed continuation. Reuses cross-bridge probe's batched primitive.
5. `update_pfc_frame(C, decoded_item, position)`: extends C by binding the
   decoded item at the next gamma-slot position; re-injects updated C into
   dlpfc_wm.
6. `run_generative_loop(initial_C, n_iterations)`: orchestrates 1-5 for N
   iterations; returns the trajectory of decoded continuations.

## Pre-registered test

**Partial-sequence completion via replay** — the (c) design's pre-registered test:

1. **Train substrate** on K stored sequences of ordered (item, slot) bindings.
   Use the validated dlpfc-extension substrate (post-pillar n=98); engram-tag
   each sequence; run Phase 1.3 consolidation to embed in cortex.
2. **Initialise PFC frame** with a partial cue (first 2 of 3 slots filled).
3. **Run the generative-replay loop** for N iterations.
4. **Measure** whether the loop completes the partial cue to the correct
   stored sequence (e.g., slot 3 = C when (A,B,C) is stored).

PASS iff multi-seed-mean completion accuracy ≥ 0.80 at every K in a fixed
K-ladder (e.g., {4, 8, 16} sequences in the schema). NEGATIVE if the replay-
decoded continuation does NOT match the stored sequence's next item.

## Task 0 — grounding pin

**Files**:
- Create: `tests/test_generative_replay_grounding.py`

**Tests (intentionally RED until later tasks land)**:
- `test_loop_controller_module_exists`: `import research.runners.generative_replay_loop` succeeds
- `test_encode_pfc_frame_returns_fhrr_composite`: helper produces a ResonateFireFHRR-compatible spike pattern
- `test_swr_trigger_opens_then_closes_gate`: `ca3_swr_burst` gate transitions correctly
- `test_decoder_uses_parallel_matching`: import + structural check
- `test_no_oracle_leak_in_loop_controller`: trace via inspect.getsource; true items must not appear in decoder argument

**Run**: `pytest tests/test_generative_replay_grounding.py -q` → expect 5 FAILS until Task 2 lands.

**Commit**: "Task 0: generative-replay grounding pin (5 tests RED until Task 2)".

## Task 1 — sequence vocabulary helper

**Files**:
- Create: `research/findings/raw/generative_replay_sequence_vocab.py`

**Helper**: `generate_k_stored_sequences(seed, k, n_words, slot_count)` —
deterministic per-seed; returns list of K sequences, each an ordered tuple of
slot_count words drawn without replacement from the V-word vocabulary.

**Tests** (in `tests/test_generative_replay_sequence_vocab.py`):
- Determinism across runs given seed
- No within-sequence repeats
- Inter-sequence diversity (no two sequences identical)
- Vocab consistency (uses the v16 16-word vocab from concept_pool_demo)

**Run**: 4/4 PASS.

**Commit**: "Task 1: sequence vocabulary helper for generative replay (4/4 unit tests)".

## Task 2 — loop controller runner

**Files**:
- Create: `research/runners/generative_replay_loop.py` (the loop controller)
- Create: `research/findings/raw/generative_replay_decisive.py` (the decisive runner)

**Loop controller**: implements `encode_pfc_frame`, `trigger_swr_replay`,
`capture_post_replay_cortical_activity`, `decode_continuation`,
`update_pfc_frame`, `run_generative_loop` per the design above. Reuses-by-
import all FHRR / mode-unification / hippocampus primitives byte-unchanged.

**Decisive runner**: builds substrate (cached from dlpfc-extension); trains;
runs Phase 1.3 consolidation on K stored sequences; per trial initialises PFC
frame with partial cue and runs the generative-replay loop; measures
completion accuracy; multi-seed. Kill-safe per-seed cache.

**Smoke mode**: tiny K (e.g., 4 sequences) + few trials; verifies the loop
end-to-end works mechanically. Smoke numbers NOT propagated.

**Run after Task 2 commit**: re-run Task 0's `tests/test_generative_replay_grounding.py`
→ expect 5/5 PASS.

**Commit**: "Task 2: generative-replay loop controller + decisive runner
(grounding pin now green; smoke validates loop assembles mechanically)".

## Task 3 — soundness tests

**Files**:
- Create: `tests/test_generative_replay_soundness.py`

**Tests**:
1. `test_pfc_frame_is_genuinely_held`: encode a frame; advance bridge N steps;
   verify dlpfc_wm activity reflects the frame (NMDA bistability working) and
   the frame is not just re-injected from outside.
2. `test_swr_replay_is_genuine_not_seeded`: confirm SWR-window mechanism
   (validated Phase 1.3 reused byte-unchanged) is called; no hand-supplied
   seed pattern.
3. `test_decoder_unchanged_from_parallel_matching`: the loop's
   `decode_continuation` uses exactly the (b)/(e)/n=96/n=97 parallel-matching
   primitive byte-unchanged (verified via import + diff).
4. `test_no_oracle_leak_in_loop_runtime`: the true stored sequence is used
   ONLY for post-hoc scoring; the loop runtime never reads it.
5. `test_consolidated_schema_is_substrate_trained_content`: the cortical
   schema the SWR replays against is post-Phase-1.3-consolidation, not a
   hand-supplied lookup.
6. `test_reuse_by_import_only`: loop controller imports unchanged primitives;
   protected set zero diff after Task 2 commit.
7. `test_no_autograd`: grep for autograd/torch.backward/loss.backward.
8. `test_no_confab_moat_still_green`: pytest tests/test_abstention_gate.py.
9. `test_frozen_bar_unchanged`: BAR = 0.80 unchanged.

**Run**: 9/9 PASS.

**Commit**: "Task 3: generative-replay soundness tests (9/9 pass; reuse-by-
import + no oracle leak + no autograd + moat 7/7 + bar immovable verified)".

## Task 4 — dedicated adversarial review BEFORE decisive run

Dispatch a fresh-agent reviewer (matching the standing project discipline)
with full tool access. Provide:
- The loop controller (`research/runners/generative_replay_loop.py`)
- The decisive runner (`research/findings/raw/generative_replay_decisive.py`)
- The soundness tests (`tests/test_generative_replay_soundness.py`)
- Reference: the design doc + this plan + AUTONOMOUS_STATE.

Reviewer must RUN (minimum) 15+ exploit-class checks including:
1. Frozen bar immovable since pillar n=97/n=98 commit
2. Loop controller doesn't violate reuse-by-import (protected set zero diff)
3. PFC frame is genuinely held by dlpfc_wm bistability (probe-based check)
4. SWR replay is the validated Phase 1.3 mechanism (not a substitute)
5. decode_continuation byte-equivalent to parallel-matching primitive
6. No oracle leak in loop runtime
7. The consolidated schema is the substrate's trained content, not a lookup
8. Independent reproduction of one trial byte-exact
9. No autograd
10. No-confab moat 7/7
11. Smoke result reproduces from cache (deterministic per seed)
12. Loop iteration count + per-iteration state evolution looks biologically
    plausible (PFC frame size grows per iteration; no degenerate collapse)
13. The decoder argmaxes over the full 16-concept vocabulary (not restricted)
14. Per-seed kill-safe cache mechanism works
15. Reviewer must judge: would a broken-loop run score a false PASS? Trace.

Required verdict: CLEAR (safe to launch decisive run) or BLOCK (specific
defect + corrective action).

**Commit**: "Task 4: generative-replay adversarial review = [CLEAR | BLOCK];
[if BLOCK: defect details + fix in Task 4b; re-review]".

## Task 5 — CONTROLLER-ONLY decisive multi-seed GPU run

Per standing project discipline: Task 5 is NOT a subagent task. Controller-only.

**Pre-run**: confirm Task 4 verdict CLEAR; capability_status moat 7/7 green;
protected set zero diff.

**Run**:
```bash
python research/findings/raw/generative_replay_decisive.py
```

**Configuration** (pre-registered; never tuned):
- Seeds: 42, 43, 44 (matches (b)/(e)/n=96/n=97)
- K-ladder: 4, 8, 16 sequences in schema
- Slot count: 3 (initial cue = 2 slots filled; loop completes slot 3)
- 200 trials per (K, seed)
- Frozen 0.80 bar
- Substrate: cached from dlpfc-extension (~hours of training avoided)
- Kill-safe per-seed cache for the loop trajectories

**Estimated wall-clock**: substrate cached → only training the sequence
engrams + running the loop trials. Per seed: ~1 hr engram training + ~1-2 hr
loop trials = ~2-3 hr/seed; × 3 seeds = ~6-9 hr total GPU.

**Mandatory anti-cheat smell-test** (scrutinise a nominal PASS harder than a
FAIL):
- Recompute per-K completion accuracy from the raw trial log
- Verify per-seed determinism (re-run one trial from seed → byte-exact)
- Verify the loop genuinely iterates (frame size grows per iteration; not a
  fixed-point degeneracy)
- Verify dlpfc_wm activity sustained across replay (NMDA bistability working)
- Verify SWR replay actually drives cortex (post-replay activity differs from
  pre-replay)
- Verify completion accuracy depends on the cue (random-cue baseline << bar)
- Verify completion accuracy depends on the stored schema (untrained-schema
  baseline << bar)

**Post-run propagation** (per standing discipline):
- Findings doc
- Dedicated fresh-agent adversarial review of the decisive result
- If CLEAR: VALIDATED pillar n=99 (or whatever the next n is)
- AUTONOMOUS_STATE update
- Commit + push both remotes
- Wiki-sync session summary

## Pre-registered outcomes

- **PASS** (multi-seed-mean ≥ 0.80 at every K): the biology-grounded generative-
  replay loop produces partial-sequence completion at multi-seed margin on the
  validated substrate. This is the conversational substrate the owner's
  2026-05-19 reframe described: PFC frame + SWR replay against consolidated
  schema + mode-unification decoder + loop integration. Honest scope: this is
  NOT LLM-fluent prose; it IS the biology-grounded compositional loop that
  generates context-appropriate continuations from consolidated schematic
  content. Subject to oracle-adjacency caveat (carrying from mode-unification
  thread).
- **NEGATIVE** (any K below 0.80): an honest finding about which integration
  property fails to scale. Each component subsystem is individually validated;
  if the LOOP fails, the failure mode precisely localises which integration
  property is missing. Biology-translatable either way.

## Standing constraints (preserved throughout)

- Pre-registered frozen 0.80 bar; never tuned.
- Reuse-by-import only; no protected/validated/frozen/moat module modified.
- No autograd anywhere.
- The decoder must NOT pass true item labels (no oracle leak).
- No-confab moat 7/7 throughout.
- Plain ASCII output.
- Both remotes (origin + gitea) propagated at every commit.
- Each task's commit follows red-green discipline: write failing test → write
  minimum code to pass → run → commit.

## Wall-clock estimate (total)

- Task 0-3: ~3-5 hours dispatch + write + test (subagent-driven per the
  superpowers:subagent-driven-development pattern)
- Task 4: ~30 min adversarial review (foreground)
- Task 5: ~6-9 hour controller-only decisive run + smell-test + adversarial
  review + propagation

**Total**: 10-15 hours, mostly bounded GPU compute; the subagent-driven
build of Tasks 0-3 is the work-intensive portion.

## Files (final inventory)

- This plan: `docs/plans/2026-05-24-generative-replay-implementation.md`
- Design doc (parent): `docs/plans/2026-05-23-generative-replay-design.md`
- Tests:
  - `tests/test_generative_replay_grounding.py` (Task 0)
  - `tests/test_generative_replay_sequence_vocab.py` (Task 1)
  - `tests/test_generative_replay_soundness.py` (Task 3)
- Net-new code:
  - `research/findings/raw/generative_replay_sequence_vocab.py` (Task 1)
  - `research/runners/generative_replay_loop.py` (Task 2; the controller)
  - `research/findings/raw/generative_replay_decisive.py` (Task 2; the runner)
- Reused unchanged (every primitive, by import):
  - All FHRR / mode-unification / hippocampus / capture / grounding / decoder
    primitives from the existing validated stack
  - concept_pool_demo's v16 production recipe
  - text_minimal_isolation's build_biological_brain_regions
- Post-run artifacts:
  - `research/findings/raw/generative_replay_decisive_full.json`
  - `research/findings/raw/generative_replay_decisive_full.log`
  - Findings doc: `research/findings/2026-05-XX-generative-replay-multi-seed-VERDICT.md`
