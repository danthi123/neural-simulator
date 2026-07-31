---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 Closure — Curriculum-Staged BG Cross-Projections

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the last remaining perception/control cheat — hand-designed BG connectivity (cortex_X → str_D1_X same-action only) — by adding *learnable* cross-projections that don't break phase-1 readaptation.

**Architecture:** Tag cross-projections with their own plasticity gate (separate from same-action `cortex_to_d1`) so the curriculum can stage them differently. Phase 1: same-action plastic, cross-projections frozen (no bias accumulation). Phase 2: same-action frozen, cross-projections **delayed** until a third phase begins after the first goal change — by then the system has experienced both N and W as winners and STDP+reward can shape cross-projections in both directions instead of locking in phase-0 bias.

**Tech Stack:** existing `RegionPathway.plasticity_gate` infrastructure + `bridge.set_plasticity_gate()` + curriculum logic in `g11_bg_runner.py`. No new bridge code needed — only runner changes.

**Why this should work:** the prior NEGATIVE result (3-seed avg 8.40) was caused by `enable_bg_cross_projections=True` tagging cross-projections with the same `cortex_to_d1` gate as same-action, so cross-projections were plastic during phase 1 and accumulated N/E motor bias. By the time phase 2 froze them, the bias was locked in. Separating the gate eliminates the failure mode entirely.

---

## Task 1: Separate the cross-projection plasticity gate

**Files:**
- Modify: `research/runners/g11_bg_runner.py:478-498` (the cortex→striatum loop)

**Step 1: Write the failing test**

```python
# tests/test_g11_bg_runner_flags.py — add new test
def test_bg_cross_projections_use_separate_gate():
    """Cross-projection cortex→D1 pathways should be tagged with a distinct
    plasticity gate from same-action pathways, so the curriculum can stage
    them independently."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(enable_bg_cross_projections=True)
    cortex_to_d1_paths = [p for p in pathways if p.from_region.startswith("cortex_")
                          and p.to_region.startswith("str_D1_")]
    assert len(cortex_to_d1_paths) == 16, "4 cortex pools × 4 D1 pools = 16 paths"
    same_action = [p for p in cortex_to_d1_paths
                   if p.from_region.split("_")[1] == p.to_region.split("_")[2]]
    cross = [p for p in cortex_to_d1_paths
             if p.from_region.split("_")[1] != p.to_region.split("_")[2]]
    assert len(same_action) == 4 and len(cross) == 12
    assert all(p.plasticity_gate == "cortex_to_d1" for p in same_action)
    assert all(p.plasticity_gate == "bg_cross_projections" for p in cross)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_g11_bg_runner_flags.py::test_bg_cross_projections_use_separate_gate -v`
Expected: FAIL with both gates currently being `cortex_to_d1`

**Step 3: Modify the builder**

In `research/runners/g11_bg_runner.py:478-498`, change cross-projection branch to use `plasticity_gate="bg_cross_projections"` instead of inheriting `"cortex_to_d1"`:

```python
elif enable_bg_cross_projections:
    # Cheat #5 closure (2026-04-28): cross-projections need their own gate.
    # Same-action stays on "cortex_to_d1" so curriculum freezes/thaws them
    # together with the original schedule. Cross-projections go on
    # "bg_cross_projections" so the curriculum can stage them later.
    density = 1.0
    weight = cross_projection_weight
    cross_gate = "bg_cross_projections"
else:
    continue
pathways.append(RegionPathway(
    from_region=f"cortex_{cortex_action}",
    to_region=f"str_D1_{str_action}",
    density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
    plasticity_gate=("cortex_to_d1" if same else cross_gate),
))
# (same for str_D2_X)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_g11_bg_runner_flags.py::test_bg_cross_projections_use_separate_gate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_g11_bg_runner_flags.py research/runners/g11_bg_runner.py
git commit -m "feat(cheat5): separate plasticity gate for BG cross-projections"
```

---

## Task 2: Wire the new gate into the curriculum

**Files:**
- Modify: `research/runners/g11_bg_runner.py:1126-1231` (curriculum logic)

**Step 1: Understand the schedule**

The curriculum currently has two phases:
- Phase 1 (steps 0..warmup, default 600): cortex_to_d1 plastic, input gates frozen
- Phase 2 (steps warmup..end): cortex_to_d1 frozen (or partial), input gates plastic

Add a third phase for cross-projections:
- Phase 1: cross-projections frozen (gain=0.0)
- Phase 2: cross-projections frozen (gain=0.0) — wait through input-layer learning
- Phase 3 (post-goal-change, e.g. steps 1200+): cross-projections plastic (gain=1.0 or 0.5)

Goal change happens at step 900 in the default `--moving-goal` schedule. Steps 1200..1800 = 600 steps of cross-projection learning AFTER the agent has seen both N-goal and W-goal regimes. STDP+reward can shape cross-projections symmetrically rather than locking in phase-0 winners.

**Step 2: Add CLI flag for the third-phase boundary**

```python
ap.add_argument("--bg-cross-thaw-step", type=int, default=1200,
    help="Step at which bg_cross_projections gate thaws to its phase-3 value. "
         "Default 1200 = ~300 steps after first goal change at step 900, "
         "letting the agent experience both regimes before cross-projections "
         "can learn cross-action routing.")
ap.add_argument("--bg-cross-phase3-gain", type=float, default=0.5,
    help="Plasticity gain for bg_cross_projections in phase 3. 1.0 = full "
         "plastic, 0.5 = half-rate (slower than same-action), 0.0 = stay frozen.")
```

**Step 3: Wire the gate**

In the curriculum init block (around line 1128), add:

```python
has_bg_cross_gate = enable_curriculum and "bg_cross_projections" in available_gates
if enable_curriculum and has_bg_cross_gate:
    bridge.set_plasticity_gate("bg_cross_projections", 0.0)  # frozen in phase 1+2
```

In the per-step gate update (around line 1195-1208), add a branch for the third phase:

```python
if has_bg_cross_gate and step == args.bg_cross_thaw_step:
    bridge.set_plasticity_gate("bg_cross_projections", float(args.bg_cross_phase3_gain))
    if verbose:
        print(f"[g11 seed={seed}] step {step}: PHASE 3 — bg_cross_projections "
              f"gain={args.bg_cross_phase3_gain:.2f}", flush=True)
```

**Step 4: Add a test for the curriculum schedule**

```python
def test_curriculum_phase3_bg_cross_thaw():
    """bg_cross_projections gate should be 0.0 before --bg-cross-thaw-step and
    --bg-cross-phase3-gain after."""
    # Use a tiny n_steps + small thaw step to make the test fast
    # ... call run_g11 with enable_bg_cross_projections=True, enable_curriculum=True,
    #     bg_cross_thaw_step=100, n_steps=200, and capture gate values via
    #     bridge.get_plasticity_gate_value(...) at known step boundaries
```

**Step 5: Run all tests**

Run: `pytest tests/test_g11_bg_runner_flags.py -v`
Expected: PASS (existing tests + 2 new ones)

**Step 6: Commit**

```bash
git add research/runners/g11_bg_runner.py tests/test_g11_bg_runner_flags.py
git commit -m "feat(cheat5): curriculum phase-3 thaw for bg_cross_projections"
```

---

## Task 3: 3-seed smoke test before full validation

**Files:**
- Run only — no edits

**Step 1: Run 3 seeds with the new flagship + cheat5 flags**

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-cross-projections --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_cheat5.json
done
```

Expected wall-clock: ~16 min/seed × 3 seeds ≈ 48 min total.

**Step 2: Compute summary**

```bash
python research/runners/aggregate_seeds.py \
    research/findings/raw/g11_bg/g11_seed42_cheat5.json \
    research/findings/raw/g11_bg/g11_seed43_cheat5.json \
    research/findings/raw/g11_bg/g11_seed44_cheat5.json
```

**Decision criterion:** if 3-seed avg sum ≤ 5.0 (similar to or better than current flagship 4.08), proceed to Task 4 (6-seed validation). If avg sum > 6.0 (worse than flagship), the third-phase schedule needs more tuning — try `--bg-cross-phase3-gain 0.25` (slower learning) or `--bg-cross-thaw-step 1500` (later thaw, longer settle time).

**Decision criterion #2 (failure recovery):** if 3-seed average is between 5.0 and 6.0, also try the inhibitory-cross-projection variant (Task 5) before deciding GO/NO-GO.

---

## Task 4: 6-seed validation (only if Task 3 passes)

**Files:**
- Run only

**Step 1: Run remaining 3 seeds**

```bash
for SEED in 100 101 102; do
    python -m research.runners.g11_bg_runner --moving-goal \
        # ... same flags as Task 3 ...
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_cheat5.json
done
```

**Step 2: Compute t-test against baseline 5.88**

The script `aggregate_seeds.py` already does this. Acceptance criteria:
- ≥5 of 6 seeds beat baseline (sum < 5.88)
- t-test p < 0.05 against baseline
- Mean sum ≤ 4.5 (i.e., we don't *lose* performance vs the 4.08 flagship that closes 4 cheats)

**Step 3: Write finding**

```
research/findings/2026-04-28-cheat5-curriculum-staged-RESOLVED.md  (if GO)
research/findings/2026-04-28-cheat5-curriculum-staged-NEGATIVE.md  (if NO-GO)
```

Include per-seed table, t-test, vs-baseline comparison, vs-flagship (4.08) comparison, recipe.

---

## Task 5: Fallback experiment — inhibitory cross-projections (only if Task 3 fails)

**Hypothesis:** the original NEGATIVE was caused by *uniform excitatory* cross-projections. In real BG, the indirect pathway (cortex → D2 → GPe → STN → GPi) cancels the bias from the direct pathway. Adding *inhibitory* cross-projections via D2 might balance.

**Files:**
- Modify: `research/runners/g11_bg_runner.py` (add `--bg-cross-d2-only` flag)

Make cross-projections go ONLY to str_D2_Y (indirect, inhibitory net effect on action Y) rather than both D1 and D2. When cortex_N fires, it weakly inhibits actions E/S/W via the indirect pathway — symmetric, doesn't bias one action over others.

Skip the detailed task breakdown unless Task 3 actually requires this fallback.

---

## Done criteria

- [ ] Task 1 commit lands; new test passes
- [ ] Task 2 commit lands; curriculum-schedule test passes
- [ ] Task 3 3-seed smoke meets criterion
- [ ] Task 4 6-seed validation passes (or finding NEGATIVE if not)
- [ ] CHANGELOG.md, CLAUDE.md, README.md, SCIENCE_ROADMAP.md updated if GO
- [ ] research/findings/INDEX.md row added at top

## Followups

If GO: cheat #5 is closed and we move to scope extensions (multi-modal, cerebellum, larger task domains).

If NO-GO at Task 4 even with curriculum staging: cheat #5 may simply require a *different* learning rule (not STDP+reward) for cross-projections. Possibilities for a future plan: contrastive Hebbian, target propagation, or local error signals. Document as "structural cheat #5 closure deferred — needs new learning rule, not just timing."
