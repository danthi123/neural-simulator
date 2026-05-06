# 2026-05-05 — Comprehensive strategic state

**Date:** 2026-05-05 ~20:40 EDT
**Status:** End-of-day comprehensive state. ALL major experiments
landed. User decisions outlined.

---

## What landed today (chronological)

### Morning (06:30 — 12:00 EDT)

**Closing the W→A learning investigation arc (18 days):**

- 3-factor sign-only DA at biological canon: 1/6 aligned at
  `tf_with_topo_fs` (noise floor, seed 101 only)
- 3-factor magnitude-graded DA (Schultz 1998): 0/6 aligned at same
  config — magnitude info doesn't rescue
- B3 supervised gradient under same arch: 3/3 PERFECT alignment
- **Verdict:** global scalar feedback fails at biological scale.
  Architecture is sufficient; rule is the bottleneck.
- **Doc:** `2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`

**Perf wave 2 measured cleanly:**

- Phase 1 GPU-port: 5% faster (vs speculative 2×)
- Phase 2 FP16 cumulative: 7% faster (vs speculative 1.5×)
- Both opt-ins kept; perf-roadmap updated with measured numbers
- Bench contamination doc + GPU-idle gate fix
- **Doc:** `2026-05-05-perf-wave2-VERDICT.md`

### Afternoon (12:00 — 18:00 EDT)

**4-step post-verdict plan (user-requested validation before pivot):**

1. **Step 1 — validate verdict.** Gradient passes permuted-label
   control (3/3 NESW aligned, +0.0pp excess at full biology canon).
   Clean dose-response: vanilla 1/3 → topo 1/3 → topo+FS 3/3.
   **Doc:** `2026-05-05-gradient-passes-permuted-label-VALIDATED.md`

2. **Step 2 — test rescue alternatives.** Three negative results:
   - LR sweep (5x = 5e-3, 10x = 1e-2): 0/3 each, +8-9pp excess
   - Orthogonal cues (verified 0 input overlap): 0/6, +6.3pp excess
   - Magnitude-graded DA: 0/6 (pre-verdict, +7.8pp excess)
   - Conclusion: rule fails for fundamental reasons; not under-tuned
   **Docs:** `2026-05-05-3factor-LR-sweep-NOT-LR-limited.md`,
   `2026-05-05-step2-COMPLETE-rule-not-rescuable.md`

3. **Step 3 — scale what works.** 32×32 navigation 6-seed
   validation: 2.57 ± 0.11 (BEATS 16×16 baseline 2.97 ± 0.12 by
   13.3%). All 6 seeds beat baseline. Tighter variance.
   **Doc:** `2026-05-05-step3-32x32-scaling-success.md`

4. **Final synthesis** — 4 steps complete; user decision pending.
   **Doc:** `2026-05-05-FINAL-autonomous-arc-synthesis.md`

### Evening (18:00 — 20:30 EDT)

**Text I/O bidirectionally validated:**

- I→W permuted-label across 56 historical eval JSONs: 2/56 aligned
  (chance is 56/24 ≈ 2.3 expected — exactly chance)
- Same architectural noise pattern as W→A (excess +7-9pp)
- Both directions of text I/O fail; rule is bidirectional bottleneck
- **Doc:** `2026-05-05-I-to-W-also-fails-permuted-label.md`

**64×64 scaling test:**

- Required two infrastructure fixes: --visual-image-size flag
  (was hardcoded 32, gave pixels_per_cell=0 at grid_size=64) and
  bumping n-hippocampus-per-layer to 256
- 6-seed 64×64: 8.80 ± 0.54 (graceful degradation, not catastrophic)
- Hi-res variant (image=128 + 1024 place cells): 8.34 — no improvement
- Resolution NOT the bottleneck; phase budget is (each 450-step
  phase has only ~7.5 steps margin to traverse 60+ Manhattan
  distance + wander)
- **Doc:** `2026-05-05-step3-64x64-graceful-degradation.md`

## Complete scaling map (all measured today)

| Grid | Config | n | Mean Manhattan | std | Notes |
|---|---|---|---|---|---|
| 16×16 | Cluster K v2 | 3 | 2.97 | ± 0.12 | Earlier baseline |
| **32×32** | **Same architecture** | **6** | **2.57** | **± 0.11** | **PEAK — operational sweet spot** |
| 64×64 | Same architecture | 6 | 8.80 | ± 0.54 | Graceful degradation |
| 64×64 | Hi-res (image=128, place=1024) | 1 | 8.34 | n/a | Within 1σ of baseline |

**Random walk baselines** (estimated): 16×16 ~7, 32×32 ~14, 64×64 ~21.
At every grid size, learning beats random walk; at 64×64 by 2.4×; at
32×32 by 5.4×.

## Strategic state — three open directions

### Direction A — push other axes at 32×32 (recommended)

The 32×32 result is the project's strongest empirical claim. Pushing
OTHER axes (not grid size) on the confirmed baseline:

1. **Dynamic obstacles** — add walls/movable obstacles to 32×32. Tests
   visual cortex robustness in cluttered scenes. ~1 week implementation.
2. **Real visual input** — replace synthetic Gabor with MNIST 28×28
   or Tiny ImageNet. Tests whether the architecture handles natural
   image statistics. ~1 week.
3. **Multi-step plans** — agent must visit subgoals in sequence
   before reward. Engages dlpfc_wm working memory more deeply.
   ~2 weeks.
4. **Sequence learning via SWR replay** — Cluster D v2 infrastructure
   exists; could test sequence consolidation during quiet periods.
   ~1-2 weeks.

**Cost:** Each axis is incremental; minimum viable test fits in 1
week. **Risk:** Low. **Reward:** Strong systems-neuroscience
research line.

### Direction B — dendritic learning for text I/O

Apical-basal multi-compartment Izhikevich neurons + per-region top-
down feedback per Bono & Clopath 2017. Address W→A AND I→W
simultaneously.

**Cost:** 1.5-2 months focused engineering. Major architectural
change (multi-compartment kernels). **Risk:** Moderate; even after
implementation, accuracy might cap at gradient's 35% ceiling.
**Reward:** Real per-region credit assignment with biology-plausible
rules. Bono-Clopath 2017 has 200+ citations.

### Direction C — pivot away from text I/O entirely

Text I/O failure is bidirectional and rule-fundamental. Real animal
language learning is grounded (spatial/temporal context, embodied).
Our flashcard task may simply be the wrong benchmark.

If the project's higher-level goal is "biology-grounded RL agent,"
text I/O might never be the right test. The 32×32 navigation result
already demonstrates working biology-grounded cross-region credit
assignment.

**Cost:** None — just stops investing in text I/O.
**Risk:** Lose the language-understanding research direction.
**Reward:** Concentrate effort where the architecture excels.

## Recommendation for tomorrow

**Direction A**, starting with **dynamic obstacles at 32×32**.

Rationale:
1. **The 32×32 result is fresh and strongest.** Building on it is
   high momentum.
2. **Dynamic obstacles is the most biological-realistic next step.**
   Real animals navigate cluttered environments; static empty
   grids are toy.
3. **Lower risk than dendritic.** Each axis test is ~1 week and
   reveals graceful failure modes.
4. **Doesn't preclude dendritic.** Direction B remains valid; we
   come back to it later if W→A specifically becomes a priority
   OR if a future Direction A axis surfaces a need for per-region
   credit.

The autonomous arc has now landed all known immediate-cost
experiments. The next move requires a real implementation effort
(dynamic obstacles, real visual, etc.) which deserves user input on
priority.

## What's currently running

`scale_32x32_LONGER_seed42` — single-seed test of 32×32 with
n-steps=3600 (2× normal). Tests whether the 32×32 mean of 2.57 is
saturation point or has further room. Result will land ~21:00 EDT.

If 32×32 longer drops to ~1.5-2.0: architecture has more capacity at
32×32 than measured. If it stays at ~2.5: 2.57 is the saturation
point. Either way is informative.

## Today's full doc index

```
research/findings/2026-05-05-*.md (10 docs)
├── W-to-A-VERDICT-global-scalar-feedback-fails.md         ← morning verdict
├── bench-phase1-contamination.md                           ← perf bench note
├── bio-three-factor-graded-da-results.md                   ← graded DA data
├── bio-three-factor-results.md                             ← classical 3-factor data
├── perf-wave2-VERDICT.md                                   ← perf wave 2
├── gradient-passes-permuted-label-VALIDATED.md             ← step 1
├── 3factor-LR-sweep-NOT-LR-limited.md                      ← step 2a
├── step2-COMPLETE-rule-not-rescuable.md                    ← step 2 verdict
├── step3-32x32-scaling-success.md                          ← step 3 (32×32)
├── step3-64x64-graceful-degradation.md                     ← step 3 (64×64)
├── I-to-W-also-fails-permuted-label.md                     ← evening I→W
├── FINAL-autonomous-arc-synthesis.md                       ← evening synthesis
└── COMPREHENSIVE-strategic-state.md                        ← THIS

docs/plans/2026-05-05-*.md (3 docs)
├── dendritic-learning-design.md
├── step2-orthogonal-cues-design.md
└── step3-scaling-design.md

experiments/ (4 today)
├── bio_three_factor_high_lr.yaml
├── bio_three_factor_orthogonal.yaml
├── scale_32x32_validation.yaml
└── scale_64x64_validation.yaml
```

## Total commits today: ~22

```
6b4b77c research: 64×64 hi-res test — resolution not the bottleneck
252a8f8 research: 64×64 scaling — graceful degradation, not catastrophic
dbb7cae research: I→W validation + 64×64 fixes
b87f77c research: 32×32 6-seed validation — 2.57 ± 0.11
529794e research: final synthesis doc — autonomous arc complete
3f6f5a3 research: step 3 — 32×32 nav scales (2.70 vs 16×16's 2.97)
b6b1af2 research: step 2 complete — 3-factor rule not rescuable
9066687 research: step 2a verdict — 3-factor NOT learning-rate-limited
20c0cbd ops: post-orthogonal step3 orchestrator
0bba210 research: orthogonal-cues encoding for 3-factor
86fde11 research: step 2 launch — high-LR sweep + design docs
80133c1 research: gradient passes permuted-label control
d1839f5 ops: morning_briefing detects W→A verdict landed
5e9b9f3 docs: update perf-roadmap with measured Phase 1+2
7f0a89a research: perf wave 2 verdict — 7% cumulative speedup
6e208b5 docs: mark CLAUDE.md "STATISTICALLY SIGNIFICANT W→A" superseded
baf331a ops + docs: GPU-idle gate in post_chain_bench
4481e69 research: 18-day W→A verdict
6a26629 research: chain extension + dendritic-learning design doc
221b51b research: graded-DA probe + headline finding
4c338ba tests: regression tests for bio_three_factor CLI flag wiring
3b7ce28 ops + docs: outcome-conditional 3-factor decision orchestrator
6eafea1 perf: --fp16-synapse-state CLI flag + GPU drift validation
1b1aa87 ops: cloud H100 deployment + benchmark harness
ee42605 perf: FP16 synapse state opt-in + parallel=6 + perf roadmap
a3187e4 perf: GPU-port three-factor eligibility trace (Phase 1)
```

All on github + gitea. Wiki-synced 2x today.

## TL;DR for tomorrow morning

1. **Read** the FINAL synthesis doc + this comprehensive state doc.
   Together they give the full picture.
2. **Decide** Direction A (continue scaling on 32×32 with new axes)
   vs Direction B (dendritic learning for text I/O) vs Direction C
   (pivot away from text I/O).
3. **If A**: launch dynamic obstacles design + smoke test (~1 week
   to ship).
4. **If B**: greenlight Week 1+ dendritic learning kernel work.
5. **If C**: focus on extending the 32×32 line; text I/O becomes a
   negative result paper.

The 32×32 navigation result (2.57 ± 0.11, n=6, 13.3% better than
16×16) is the strongest empirical claim. Build on that.
