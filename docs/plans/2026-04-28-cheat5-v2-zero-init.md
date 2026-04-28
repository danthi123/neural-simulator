# Cheat #5 v2 — Zero-Initial-Weight BG Cross-Projections

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close cheat #5 (hand-designed BG connectivity) by adding learnable cross-projections that don't break the BG cascade *structurally* during phase 0.

**Why v1 failed:** The plasticity gate (`cp_plasticity_gain`) freezes weight UPDATES but not synaptic CURRENT. Cross-projection synapses initialized at `weight_mean=5.0` disrupt disinhibition gating from step 0. v1 phase-0 finalQ was 5.77–6.00 (vs the perception-arc benchmark 1.92).

**v2 approach:** Initialize cross-projections at `weight_mean=0.0` so they have no functional effect during phases 1+2. At phase-3 thaw, the plasticity gate becomes 0.5 and STDP+reward grows the weights from zero. Slow but biologically grounded — real synapses also start at zero efficacy.

**Risk:** STDP from a zero baseline may be too slow to produce a measurable effect by step 1800. Mitigation built in: if v2's 3-seed smoke is mid-pack (sum 5.0–6.0), spin v2.5 (small initial random weight + small `bg_cross_phase3_gain`).

**Tech Stack:** Existing `RegionPathway.plasticity_gate`, `cross_projection_weight` knob. No bridge changes — just reuse the runner's `--cross-projection-weight 0.0` argument.

---

## Task 1: Run a 3-seed smoke test with `--cross-projection-weight 0.0`

**Files:** none (no code changes — uses existing flag)

**Step 1: Launch 3 seeds**

```bash
for SEED in 42 43 44; do
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-cross-projections --cross-projection-weight 0.0 \
        --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed $SEED --n-steps 1800 \
        --out research/findings/raw/g11_bg/g11_seed${SEED}_cheat5v2.json
done
```

Expected wall-clock: ~14 min/seed × 3 = ~42 min total.

**Step 2: Compute 3-seed mean (sum P0+P1 finalQ)**

**Decision criterion:**
- mean ≤ 4.5 → success direction. Spin Task 2 (full 6-seed validation).
- mean 4.5–6.0 → mid-pack. Phase 0 not damaged (good) but cross-projections may be too slow to grow. Try v2.5 (`--cross-projection-weight 0.1`).
- mean > 6.0 → still failing. Investigate why phase 0 isn't matching the flagship's 1.75 — maybe the synapses still inject noise even at weight=0 (unlikely but possible).

**Step 3: Commit smoke results regardless of outcome**

```bash
git add research/findings/raw/g11_bg/g11_seed{42,43,44}_cheat5v2.json
git commit -m "validation(cheat5-v2): 3-seed smoke with cross_projection_weight=0.0"
```

---

## Task 2: 6-seed validation (only if Task 1 mean ≤ 4.5)

```bash
for SEED in 100 101 102; do
    # ... same recipe ...
    --seed $SEED \
    --out research/findings/raw/g11_bg/g11_seed${SEED}_cheat5v2.json
done
```

**Acceptance criteria for GO:**
- ≥5 of 6 seeds beat baseline (sum < 5.88)
- 6-seed t-test p < 0.05 against baseline 5.88
- Mean sum ≤ 4.5 (don't lose performance vs the 4.08 flagship)
- **Phase 0 finalQ avg < 2.5** (structural integrity check — flagship benchmark is ~1.75–2.05)

The phase-0 check is critical. If phase 0 is fine but later phases are mid-pack, that's "cross-projections didn't help much" (acceptable closure), not "cross-projections broke things" (failure).

---

## Task 3: Decision matrix

After 6-seed validation:

| Mean sum | P0 avg | Verdict | Action |
|---|---|---|---|
| ≤ 4.1 | ≤ 2.5 | **GO** — better than flagship | Update flagship to include `--bg-cross-projections --cross-projection-weight 0.0`. Closes cheat #5. |
| 4.1–4.5 | ≤ 2.5 | **MARGINAL GO** — closes cheat without performance gain | Document as closure-without-improvement. Keep optional. |
| 4.5–5.5 | ≤ 2.5 | **PARTIAL** — phase 0 fine, learning incomplete | Spin v2.5 with `--cross-projection-weight 0.1` (small but nonzero) and `--bg-cross-phase3-gain 0.8` (stronger plasticity). |
| > 5.5 OR P0 > 2.5 | — | **NO-GO v2** | Move to v3 (alternate approach: inhibitory-only cross via D2, or delayed structural plasticity). |

---

## Task 4: Write finding + propagate

If GO (Marginal or full):
1. `research/findings/2026-04-28-cheat5-v2-RESOLVED.md`
2. Update INDEX.md (top row)
3. Update `CHANGELOG.md`, `README.md` flagship recipe, `CLAUDE.md` recommended config, `SCIENCE_ROADMAP.md` §4.11
4. Update `QUICKSTART.md` flagship recipe to include the new flags

If NO-GO:
1. `research/findings/2026-04-28-cheat5-v2-NEGATIVE.md`
2. Document v3 plan: alternative approaches

---

## Done criteria

- [ ] Task 1 3-seed smoke executed
- [ ] Decision per Task 3 matrix
- [ ] Task 2 6-seed validation (if conditional)
- [ ] Finding written + INDEX updated
- [ ] If GO: propagate to all flagship-mention docs

---

## Notes on the plasticity-gate / weight-init insight

This v2 plan exists because v1 hit a real architectural insight:

> Plasticity gates freeze *learning*, not *synaptic transmission*. The bridge's `cp_plasticity_gain` array gates STDP/eligibility/Hebbian/synaptic-scaling. It does NOT gate the forward `g_syn × (V - E)` current contribution.

For most existing pathways this distinction doesn't matter — they're either always-on or always-off. But for *staged-introduction* of new pathways, the distinction matters a lot. v2's `weight_mean=0.0` sidesteps the issue by making forward transmission zero by construction.

If v2 also fails, the next architectural step would be a **runtime weight scale** (e.g. `cp_weight_scale_per_gate`) that the bridge multiplies into effective synaptic current per gate. That's a small bridge change but more invasive than v2. We're trying v2 first because it's literally a CLI flag.
