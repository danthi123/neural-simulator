---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 v4 — Developmental Pretraining (Design)

**Status**: design approved 2026-04-28. Implementation plan to follow via `superpowers:writing-plans`.

**Supersedes**: the high-level sketch at [`2026-04-28-cheat5-v4-developmental.md`](2026-04-28-cheat5-v4-developmental.md) — that file remains as historical context but this is the validated design.

## Goal

Close cheat #5 (BG cross-projections) by acknowledging that cross-projection refinement is a **developmental phenomenon, not adult learning**. Pre-train cross-projections under varied tasks during a "critical period" (all plasticity gates open), then **freeze** them at the end of pretraining and run the standard moving-goal eval as the "adult" phase.

v3.1 NO-GO (sum 8.92, P1=6.35) demonstrated that adult STDP+reward on a converged BG cascade can't shape useful cross-projection structure from random init even with all the local biology pieces in place. v4 tests whether prior **experience-dependent shaping** during a developmental window changes that.

## Architecture (option A — most biorealistic)

Three-phase execution, with a one-way gate transition at the pretraining→eval boundary:

```
[ Pretraining phase ]                 [ Eval phase ]
  No curriculum (all gates @ 1.0)       Standard flagship curriculum
  - cortex_to_d1                        - Phase 1 (warmup 600):
  - sensory_to_cortex                     cortex_to_d1=1.0, sensory=0.0
  - hippo_to_cortex                       bg_cross_projections=0.0  ← frozen
  - beacon_to_goal                      - Phase 2 (mature):
  - landmark_to_place                     cortex_to_d1=0.0, sensory=1.0
  - pfc_pathways                          bg_cross_projections=0.0  ← stays frozen
  - bg_cross_projections (PLASTIC)
  10 random goals × 3000 steps          1800-step moving-goal eval
         |                                       |
         v                                       v
   "critical period closes"            "adult task evaluation"
   (no manual freeze needed —
    eval-phase curriculum init
    forces bg_cross to 0.0
    naturally)
```

**Why this maps to neurodevelopment**: during a critical period, multiple circuits are highly plastic simultaneously. Sensory→cortex, cortex→cortex, cortex→BG, and the cross-action BG connectivity all refine together under varied experience. After the critical period closes, plasticity drops sharply and the circuit operates in adult mode with much more limited tuning. Option A models this clean transition.

## Implementation strategy: pragmatic insertion

The existing main eval loop is ~580 lines. Instead of a clean refactor (high regression risk for the validated flagship), we insert a **separate, simpler pretraining loop** before the curriculum init block. Some logic is duplicated between pretraining and eval, but it's contained and easy to audit. If v4 turns GO, we can refactor as a follow-up.

### Components

**New CLI flags**:
```
--developmental-pretraining          # bool, default off
--pretraining-n-goals N              # default 10
--pretraining-steps-per-goal M       # default 3000
```

**New kwargs on `run_moving_goal_episode`**:
- `enable_developmental_pretraining: bool = False`
- `pretraining_n_goals: int = 10`
- `pretraining_steps_per_goal: int = 3000`

**New top-level helper** (in [`research/runners/g11_bg_runner.py`](../../research/runners/g11_bg_runner.py)): `_run_pretraining_phase(bridge, cfg, regions, n_goals, steps_per_goal, grid_size, start_pos, seed, verbose=True, …)`. Responsibilities:
1. Validate expected gates exist via `bridge.list_plasticity_gates()`. Raise `KeyError` with the bad name AND the actual list if mismatched (catches typos early).
2. Force ALL gates to 1.0 via `bridge.set_plasticity_gate(name, 1.0)`.
3. Loop `n_goals × steps_per_goal` simulation trials. Per goal: pick a random `(gx, gy)` Manhattan ≥ 3 from start AND ≠ previous goal. Run trials with that goal teleported each trial.
4. NaN-check cross-projection weights at end; raise `RuntimeError` if any NaN.
5. Return summary dict: `{n_trials, n_goal_changes, cross_weights_mean, cross_weights_std}`.

**Insertion point**: after bridge creation + region setup but BEFORE the curriculum init block at [g11_bg_runner.py:1206](../../research/runners/g11_bg_runner.py#L1206). The existing curriculum init then naturally forces phase-1 gates (incl. `bg_cross_projections=0.0`) at eval start — no manual freeze needed at the boundary.

### State at the pretraining→eval boundary

| State | What happens |
|---|---|
| `bridge.cp_synapse_weights` | persists — pretrained values carry into eval (this is the whole point) |
| `bridge.cp_eligibility_trace` | persists, decays naturally — no reset |
| `bridge.cp_plasticity_gain` | fully reset by curriculum init at eval start |
| reward EMA / adaptive-DA state | persists across boundary (biologically faithful — adults don't reset prediction at task switch) |

### Logging

- Per pretraining-goal start: `[g11 seed=42] pretraining goal 3/10: (4,2)`
- End of pretraining: `[g11 seed=42] pretraining complete: 30000 steps, cross weights mean=0.41 std=0.18 → freezing`

## Validation: tiered approach

| Tier | n_seeds | pretraining steps | wall-clock | Purpose |
|---|---|---|---|---|
| 1 | 1 | 1 goal × 1000 = 1K | ~25 min | wiring check (gates flip, weights develop, freeze persists) |
| 2 | 3 | 5 goals × 1000 = 5K | ~4h batch (4-concurrent) | signal check vs flagship |
| 3 | 6 | 10 goals × 3000 = 30K | ~14h batch (4-concurrent) | full validation |

Tier 2 only runs if tier 1 passes (wiring is right). Tier 3 only runs if tier 2 shows ≥ partial signal (per decision matrix below).

## Decision matrix (tier 3, 6-seed)

| Eval-phase mean sum | Verdict | Action |
|---|---|---|
| ≤ 4.1 | **GO** — cheat #5 closed via development | Propagate, document, optional pretrained-weight persistence |
| 4.1–4.5 | **GO MARGINAL** | Document closure-without-improvement |
| 4.5–6.0 | **PARTIAL** | Try longer pretraining, more goals, or structural plasticity |
| > 6.0 | **NO-GO v4** | Cross-projections are off-axis. Close cheat #5 by acknowledging same-action-only as biologically equivalent in our reduced model |

Per-phase floor: P0 ≤ 2.5 AND P1 ≤ 2.5 required for GO (matches the v3.1 decision matrix in `2026-04-28-cheat5-v3-lateral-inhibition.md`).

## Testing

**Unit tests** (`tests/test_g11_bg_runner_flags.py`):
1. `test_pretraining_raises_on_missing_gate` — KeyError mentions the bad name AND the actual gate list.
2. `test_pretraining_thaws_all_gates_at_start` — every expected gate reads 1.0 after first call.
3. `test_pretraining_goal_sampling_respects_manhattan_3` — 100 sampled goals all Manhattan ≥ 3 from start.
4. `test_pretraining_goal_no_consecutive_repeats` — `goals[i] != goals[i-1]` for all i ≥ 1.
5. `test_pretraining_returns_summary` — returned dict contains `n_trials`, `n_goal_changes`, `cross_weights_mean`, `cross_weights_std`.

**Integration test** (same file):
6. `test_run_moving_goal_with_pretraining_smoke` — end-to-end with `--developmental-pretraining --pretraining-n-goals 1 --pretraining-steps-per-goal 50 --n-steps 100`. Asserts run completes; cross-weights changed during pretraining (≥ 1% of synapses moved); cross-weights frozen during eval (identical within float tolerance); output JSON has expected keys.

## Error handling

- **Missing gate** (typo): `KeyError` from `set_plasticity_gate` propagates with available gate list. Caught by test #1.
- **NaN weights**: hard fail with `RuntimeError("pretraining produced NaN cross-projection weights — likely STDP instability")`. No silent corruption.
- **Conflicting flags**: `--developmental-pretraining` + `--bg-cross-thaw-step >= 0` raises `ValueError` (v3.1+v4 mix is meaningless).
- **Pretraining without cross-projections**: log warning, proceed. Lateral-inhibition-only pretraining is harmless but pointless.
- **Tiny step counts**: warn if `pretraining_steps_per_goal < 50`.

## Out of scope (deferred)

- **Pretrained weights persistence** (save/load HDF5): originally Task 4 of the v4 plan sketch. Defer to a follow-up if v4 turns GO. For now every researcher reruns pretraining each time. This avoids HDF5 schema work that may be wasted if v4 is NO-GO.
- **Staged critical periods within pretraining** (sensory matures before motor before association — option D in brainstorming): more biorealistic but significantly more complex. Pursue only if option A is PARTIAL and we want to refine.
- **Pretraining-time early-stop on weight magnitude plateau**: simpler to use fixed step counts at this stage. Add only if we observe waste in tier 2/3 runs.

## Biological grounding

- **Critical periods**: visual cortex ocular dominance, sensory cortex maturation, etc. After the period closes, plasticity drops via PV interneuron maturation. We model this as the cross-projection gate flipping to 0 at end of pretraining.
- **Experience-dependent pruning**: real BG connectivity in adults is shaped by months of varied motor experience during development. Our 30K pretraining steps stand in for this.
- **Adult learning ≠ developmental learning**: adults fine-tune within existing connectivity but typically don't rewire BG cross-projections. Our flagship eval (1800 steps with frozen cross) reflects this.

If v4 succeeds, the project demonstrates the simulator can support **two distinct learning regimes** — developmental (high plasticity, varied experience) and adult (lower plasticity, task-specific) — which is a significant capability beyond the immediate cheat #5 closure.

## Done criteria

- [ ] Unit tests 1-5 + integration test 6 all pass
- [ ] Tier 1 wiring smoke shows weight development during pretraining + freeze during eval
- [ ] Tier 2 reduced smoke produces a 3-seed result we can score against the decision matrix
- [ ] Tier 3 (only if tier 2 promising) gives 6-seed validation
- [ ] Findings doc, CLAUDE.md update, INDEX update, memory update propagated
- [ ] If GO: optional follow-up plan for pretrained-weight persistence

## If v4 also fails

Last-resort plan from the original v4 sketch: acknowledge cheat #5 is closed *by design* — same-action-only is the biological winner-take-all in our reduced model. Document explicitly that real BG is anatomically dense + functionally same-action-dominant; our model achieves the equivalent functional outcome with a simpler substrate. This is a principled choice given the simulator's level of abstraction, not a punt.
