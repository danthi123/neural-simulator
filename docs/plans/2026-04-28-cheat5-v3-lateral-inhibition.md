# Cheat #5 v3 — Add MSN Lateral Inhibition (Prerequisite for Cross-Projections)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the missing winner-take-all biology to the BG cascade — MSN lateral inhibition between action pools — as a *prerequisite* for re-attempting cheat #5 closure. v3 itself is independently valuable: more biology, no risk to flagship.

**Background (from v2 NEGATIVE):** v2 fixed the structural-damage failure mode (zero-init cross-projections) but exposed a learning-dynamics failure mode: STDP+reward learns spurious cross-projection patterns from a converged BG cascade because the cascade has no lateral inhibition to suppress cross-talk. Real BG handles this via:
- MSN-MSN GABAergic collaterals (within and between action pools)
- Striatal FS interneurons (strong feed-forward inhibition)
- Center-surround organization in pallidum

**v3 scope:** Add **MSN cross-pool lateral inhibition** (the simplest and most impactful piece). FS interneurons and pallidal center-surround are deferred to v3.5 if v3+v3.1 are insufficient.

**Tech stack:** Pure config addition — new pathways in `build_bg_brain_regions`. Opt-in via flag `--bg-lateral-inhibition` (default OFF for safety).

---

## Task 1: Add MSN cross-pool inhibition pathways (TDD)

**Files:**
- Test: `tests/test_g11_bg_runner_flags.py`
- Modify: `research/runners/g11_bg_runner.py:build_bg_brain_regions`

**Step 1: Write failing test**

```python
def test_bg_lateral_inhibition_pathways():
    """When --bg-lateral-inhibition is on, the BG cascade includes
    cross-pool inhibitory pathways: str_D1_X → str_D1_Y for X != Y, and
    same for D2. 4 actions × 4 → 12 cross-pool pairs × 2 (D1, D2) = 24
    new pathways. Their `plastic` is False (static lateral inhibition).
    The MSN regions are GABAergic (exc_fraction=0.05) so the projection
    IS inhibitory."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, no_inhib = build_bg_brain_regions(enable_bg_lateral_inhibition=False)
    regions, with_inhib = build_bg_brain_regions(enable_bg_lateral_inhibition=True)

    def msn_lateral_count(pathways):
        n = 0
        for p in pathways:
            from_d1 = p.from_region.startswith("str_D1_")
            to_d1 = p.to_region.startswith("str_D1_")
            from_d2 = p.from_region.startswith("str_D2_")
            to_d2 = p.to_region.startswith("str_D2_")
            same_action = p.from_region.split("_")[-1] == p.to_region.split("_")[-1]
            if (from_d1 and to_d1 or from_d2 and to_d2) and not same_action:
                n += 1
        return n

    assert msn_lateral_count(no_inhib) == 0
    assert msn_lateral_count(with_inhib) == 24, (
        "4 cortex actions × 3 cross targets × 2 (D1/D2) = 24"
    )

    msn_laterals = [p for p in with_inhib
                    if p.from_region.startswith("str_D")
                    and p.to_region.startswith("str_D")
                    and p.from_region.split("_")[-1] != p.to_region.split("_")[-1]]
    for p in msn_laterals:
        assert not p.plastic, "lateral inhibition should be static"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_g11_bg_runner_flags.py::test_bg_lateral_inhibition_pathways -v
```

Expected: FAIL because `enable_bg_lateral_inhibition` doesn't exist yet.

**Step 3: Add the kwarg + pathways**

In `build_bg_brain_regions` signature, add:
```python
# v3 (2026-04-28): MSN cross-pool lateral inhibition. Real BG has
# GABAergic collaterals between MSNs (both within pool and between pools)
# and FS interneurons mediating strong feed-forward inhibition. Without
# these, cross-projections (cheat #5) inject noise into the cascade
# during STDP learning. v3 adds cross-pool MSN→MSN inhibitory projections
# as the minimal piece. FS interneurons + pallidal center-surround are
# v3.5 if needed. Static (plastic=False).
enable_bg_lateral_inhibition: bool = False,
lateral_inhibition_density: float = 0.3,
lateral_inhibition_weight: float = 2.0,
```

After the existing same-action cortex→striatum loop (around line 498), add:
```python
if enable_bg_lateral_inhibition:
    for x in ACTION_NAMES:
        for y in ACTION_NAMES:
            if x == y:
                continue
            # MSNs are GABAergic (exc_fraction=0.05), so this projection IS
            # inhibitory. Cross-pool: str_D1_X → str_D1_Y for X != Y.
            for d_type in ("D1", "D2"):
                pathways.append(RegionPathway(
                    from_region=f"str_{d_type}_{x}",
                    to_region=f"str_{d_type}_{y}",
                    density=lateral_inhibition_density,
                    weight_mean=lateral_inhibition_weight,
                    weight_jitter=0.2,
                    plastic=False,
                ))
```

**Step 4: Verify test passes**

```bash
python -m pytest tests/test_g11_bg_runner_flags.py::test_bg_lateral_inhibition_pathways -v
```

Expected: PASS.

**Step 5: Wire CLI flag**

In argparse:
```python
ap.add_argument("--bg-lateral-inhibition", action="store_true",
                help="v3 (2026-04-28): add MSN cross-pool inhibition (24 GABAergic pathways) for BG winner-take-all. Improves biological accuracy regardless; required prerequisite for cheat #5 closure (cross-projections).")
ap.add_argument("--lateral-inhibition-density", type=float, default=0.3)
ap.add_argument("--lateral-inhibition-weight", type=float, default=2.0)
```

In the `run_moving_goal_episode` call:
```python
enable_bg_lateral_inhibition=args.bg_lateral_inhibition,
lateral_inhibition_density=args.lateral_inhibition_density,
lateral_inhibition_weight=args.lateral_inhibition_weight,
```

In `run_moving_goal_episode` signature: same kwargs, pass to `build_bg_brain_regions`.

**Step 6: Smoke test the runner with the new flag**

```bash
python -m research.runners.g11_bg_runner --moving-goal --bg-lateral-inhibition \
    --seed 42 --n-steps 50 --out /tmp/smoke.json
```

Should run without crashing.

**Step 7: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(cheat5-v3): MSN cross-pool lateral inhibition pathways"
```

---

## Task 2: Validate v3 doesn't regress flagship

The headline check: with `--bg-lateral-inhibition` added to the flagship config, the agent should still produce ≤ 4.5 sum on a 3-seed smoke. If lateral inhibition makes things WORSE, we have a different problem.

**Files:** none (validation only)

**Step 1: Run 3 seeds with flagship + lateral inhibition (no cross-projections yet)**

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-lateral-inhibition \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_v3lateral.json
done
```

Wall-clock: ~14 min/seed × 3 = ~42 min.

**Step 2: Decision**

- mean ≤ 4.5 → v3 lateral inhibition itself is healthy. **GO** to Task 3 (v3.1 cross-projections).
- mean 4.5–5.5 → marginal. Lateral inhibition may be too strong. Try `--lateral-inhibition-weight 1.0` (half strength).
- mean > 5.5 → broken. Lateral inhibition is killing the cascade somehow. Reduce `--lateral-inhibition-density 0.1` AND `--lateral-inhibition-weight 1.0`.

**Step 3: Commit data + write a brief finding**

```bash
git add research/findings/raw/g11_bg/g11_seed{42,43,44}_v3lateral.json
git commit -m "validation(v3-lateral): 3-seed flagship with MSN lateral inhibition"
```

If GO: short finding `2026-04-28-v3-lateral-inhibition-NO-REGRESSION.md`.
If marginal/broken: spin tuning task.

---

## Task 3: v3.1 — re-attempt cross-projections on top of lateral inhibition

Once v3 is healthy (no regression), the actual cheat #5 attempt:

**Files:** none (uses existing flags)

**Step 1: Run 3-seed smoke with full stack**

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
        --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_v3.1cross.json
done
```

**Step 2: Decision matrix**

| Mean sum | P0 | P1 | Verdict | Next |
|---|---|---|---|---|
| ≤ 4.1 | ≤ 2.5 | ≤ 2.5 | **GO** | 6-seed validation, propagate, cheat #5 closed |
| 4.1–4.5 | OK | OK | **GO MARGINAL** | 6-seed; document closure-without-improvement |
| 4.5–6.0 | OK | high | **PARTIAL** | Try slower phase-3 gain (0.2 instead of 0.5), longer phase-2 (warmup_steps=900) |
| > 6.0 OR P0 high | — | — | **NO-GO v3.1** | Move to v4 (multi-task developmental phase) |

---

## Task 4: v4 — multi-task developmental phase (only if v3.1 NEGATIVE)

If v3.1 still fails after v3 lateral inhibition is healthy, the deeper interpretation is correct: **cross-projection refinement is a developmental phenomenon, not adult learning.** Real BG cross-connectivity is shaped by experience-dependent pruning during developmental critical periods, with the agent exposed to many tasks.

User has explicitly authorized this scope: "Needing a multi-task developmental phase isn't an inherent flaw, it aligns with the ultimate goal of this project."

**Approach:**
1. **Pre-training phase** (`--developmental-pretraining`): before the moving-goal task, run the agent through ~10–20 random goal positions for ~3000 steps each. Cross-projections are PLASTIC during this phase only. Lateral inhibition is ON.
2. **At the end of pre-training, freeze cross-projections.** They've been shaped by varied task experience — analogous to closed critical period.
3. **Run the standard moving-goal evaluation** with cross-projections frozen at their developed state.

This isn't about boosting the moving-goal score per se — it's about whether developmentally-shaped cross-projections are *neutral or beneficial* (i.e., the agent does at least as well as without them, with biology-grounded connectivity now in place).

Plan deferred until v3.1 actually fails — premature to detail now.

---

## Done criteria (v3 + v3.1)

- [ ] v3 Task 1: pathways + tests pass
- [ ] v3 Task 2: 3-seed no-regression mean ≤ 4.5
- [ ] v3.1 Task 3: 6-seed validation if smoke passes
- [ ] If v3.1 GO: cheat #5 closed → propagate to all flagship docs
- [ ] If v3.1 NO-GO: pivot to Task 4 (v4 developmental phase)
- [ ] All findings written; INDEX updated; CHANGELOG updated

---

## Why this matters

Beyond cheat #5, **MSN lateral inhibition is missing biology** that we should have anyway. The cascade currently relies on the indirect path (D2→GPe→STN→GPi excitation) for cross-action suppression, which is a *coarse* mechanism. Real lateral inhibition is *fast, local, and selective*. Adding it:
- Sharpens action selection (less mushy motor-pool firing)
- Improves robustness to noise
- Enables future biology that depends on it (e.g., pattern separation in striatum, episodic memory consolidation)

Even if cheat #5 didn't exist, v3 would be on the roadmap.
