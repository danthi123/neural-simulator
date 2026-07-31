---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 v4 — Multi-Task Developmental Pre-Training

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **Conditional:** only execute if v3.1 (cross-projections on top of MSN lateral inhibition) is NEGATIVE.

**Goal:** Close cheat #5 by acknowledging that BG cross-connectivity is shaped *developmentally* — through experience-dependent pruning during critical periods — rather than by adult task learning. Implement a developmental pre-training phase where the agent experiences many random goal positions, lets STDP+reward shape cross-projections during this phase, then freezes them for the standard task evaluation.

**Premise:** v3.1 (lateral inhibition + cross-projections) succeeded would mean cross-projections can be learned within a single 1800-step task. v3.1 failing would mean even with lateral inhibition, the 2-goal task doesn't expose the agent to enough varied (cortex_X firing → reward at action Y) correlations to shape cross-projections cleanly. Real animals have months of varied experience before adult task evaluation.

**User authorization:** "Needing a multi-task developmental phase isn't an inherent flaw, it aligns with the ultimate goal of this project." (2026-04-28)

---

## Task 1: Add a `--developmental-pretraining` mode to the runner

**Files:**
- Modify: `research/runners/g11_bg_runner.py`
- Test: `tests/test_g11_bg_runner_flags.py`

The mode should:
1. Run the agent through N random goal positions for M steps each (e.g., 10 goals × 3000 steps = 30,000 pretraining steps).
2. During pretraining: lateral inhibition ON, cross-projections plastic (`bg_cross_projections` gate at full 1.0), all other plasticity at usual settings.
3. After pretraining ends: freeze cross-projections (`bg_cross_projections` gate to 0.0), then run the standard 1800-step moving-goal evaluation.
4. The pretrained weights persist into the eval phase (no re-init).

**CLI:**
```bash
--developmental-pretraining           # enable
--pretraining-n-goals 10              # number of varied goals
--pretraining-steps-per-goal 3000     # steps per goal
--pretraining-eval-after              # run normal eval after (default true)
```

**Goal sampling during pretraining:** uniform random over the grid, excluding cells too close to the start (Manhattan distance ≥ 3). Re-sample at each pretraining-goal boundary.

**Heuristic + perception during pretraining:** ALL existing flags (perception arc, sensed reward, etc.) are active so the agent has cues to follow. Pretraining is about EXPERIENCE diversity, not isolating cross-projection learning.

---

## Task 2: 3-seed pretraining smoke

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-lateral-inhibition \
        --bg-cross-projections --cross-projection-weight 0.0 \
        --developmental-pretraining \
        --pretraining-n-goals 10 \
        --pretraining-steps-per-goal 3000 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_v4dev.json
done
```

Wall-clock: 30,000 pretraining + 1800 eval steps = ~30× longer than usual. Estimate: ~7 hours/seed × 3 = ~21 hours total. **Run overnight.**

(If user wants faster validation: `--pretraining-n-goals 5 --pretraining-steps-per-goal 1000` → 5,000 pretraining + 1,800 eval ≈ 4× longer, ~1 hour/seed × 3 = ~3 hours total.)

---

## Task 3: Decision matrix

| Eval-phase mean sum | Verdict | Action |
|---|---|---|
| ≤ 4.1 | **GO** — cheat #5 closed via development | 6-seed validation, propagate, close |
| 4.1–4.5 | **GO MARGINAL** | 6-seed; document closure-without-improvement |
| 4.5–5.5 | **PARTIAL** — pretraining doesn't quite suffice | Try longer pretraining (15,000+ steps/goal), more goals (20+), or add structural plasticity for cross-projections |
| > 5.5 | **NO-GO v4** | Cross-projections are fundamentally off-axis. Close cheat #5 by acknowledging same-action-only as the biological winner-take-all in our reduced model. |

---

## Task 4: Persist pretrained weights for re-use

If v4 is GO, we don't want every researcher to redo 30K pretraining steps. Add:
- `--save-pretrained-weights <path>` after pretraining completes
- `--load-pretrained-weights <path>` to skip pretraining and use saved weights

Use HDF5 format consistent with the existing checkpoint system. Most likely just save:
- `cp_synapse_weights` (the full weight array)
- Region/pathway metadata for sanity-check alignment

This lets the flagship recipe stay short (one extra `--load-pretrained-weights` flag) while keeping the developmental approach honest.

---

## Notes on biological grounding

This v4 approach explicitly models:
- **Critical periods** in development (sensory cortex maturation, ocular dominance plasticity, etc.). After the critical period closes, plasticity drops. We model this as the gate flipping to 0.
- **Experience-dependent pruning.** Real BG connectivity in adults is the result of months of varied experience shaping which corticostriatal connections are functional. Our 30,000 pretraining steps stand in for this.
- **Adult learning ≠ developmental learning.** Adults can fine-tune within existing connectivity, but they don't typically rewire BG cross-projections. Our flagship eval (1800 steps with frozen cross) reflects this.

If v4 succeeds, the project demonstrates that the simulator can support **two distinct learning regimes** — developmental (high plasticity, varied experience) and adult (lower plasticity, task-specific) — which is a significant capability beyond the immediate cheat #5 closure.

---

## Done criteria

- [ ] Task 1: `--developmental-pretraining` mode wired + tests
- [ ] Task 2: 3-seed pretraining smoke (overnight)
- [ ] Task 3 decision: GO / MARGINAL / NO-GO
- [ ] Task 4 (if GO): pretrained weights persistence + flagship recipe update
- [ ] Finding doc + INDEX update + CHANGELOG + SCIENCE_ROADMAP

---

## If v4 also fails

Last-resort plan: acknowledge cheat #5 is closed *by design* — same-action-only is the biological winner-take-all in our reduced model, with cross-projection development happening implicitly via the architecture. Document explicitly:
- Real BG anatomically dense, functionally same-action-dominant
- Our model: same-action-only structurally, equivalent functional behavior
- Closure rationale: identical functional outcome, simpler substrate

This isn't a punt — it's a principled choice given the simulator's level of abstraction. v3 lateral inhibition + same-action structure ≈ functional equivalent of real BG's anatomically-dense + winner-take-all.
