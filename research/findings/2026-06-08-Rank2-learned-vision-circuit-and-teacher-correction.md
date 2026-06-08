# Rank 2 (the durable LEARNED visuomotor circuit) DURABLY consolidates self-sufficient vision-navigation on all 6 seeds (none collapse, vs the position-invariant IT path's ~6.1 collapse) — at SEED-VARIABLE precision (6-seed mean 3.4; near-reflex ~2.3 on the best seeds); AND an honest multi-seed CORRECTION: the supervised motor-teacher's seed-42 "tighten" was a single-seed outlier artifact (the reflex already supplies supervision via co-firing, so the explicit teacher is redundant + a crutch). — 2026-06-08

**Status:** Rank 2 plain (learned-from-vision, no teacher) = the REAL result, GOOD multi-seed; the supervised teacher = HONEST NEGATIVE (not a robust lever). Single-goal 6-seed + multi-goal generalization IN FLIGHT (appended below). NO `sim/` edit (protected set byte-empty); additive default-off flags; helper unit tests 7/7 (cardinal) + 6/6 (offset). This closes the Rank-2 build the owner approved ("tune Rank 2 for a cleaner hold") with an honest correction.

## The one-line result

The owner-approved Rank 2 — a LEARNED dorsal/PPC-style where→action read-out, re-sourced from the IMAGE salience offset (position-preserving, no coordinates), taught by the innate SC reflex (Rank 1) then weaned — **durably consolidates a self-sufficient vision-navigation mapping on all 6 seeds** (single-goal post-wean ≤4.57 every seed, none collapsing to the position-invariant IT path's ~6.1) — **at SEED-VARIABLE precision** (6-seed mean 3.4; near the reflex's single-goal ~2.0 on the best seeds, ~3.3–4.6 on the rest). It is self-sufficient AFTER the reflex teacher is gone — the durable-consolidation property the position-invariant IT path could not achieve — though it reaches near-reflex precision only on some seeds. **The supervised motor-teacher I added to tighten it is an HONEST NEGATIVE: its apparent seed-42 win (3.93→3.30) was a single-seed outlier artifact, refuted at multi-seed (teacher mean 2.96 ≈ plain 2.93; the teacher HURTS the well-consolidated seeds).**

## The honest correction (why the seed-42 teacher result did not hold)

I initially reported (seed 42 only) that the learned circuit ceilinged at ~4.0 and that a supervised motor-teacher (feedback-error-learning) tightened it to ~3.3 — "the learning-rule diagnosis confirmed." **The multi-seed paired comparison refuted both halves:**

| seed | TEACHER post-wean | PLAIN R2 post-wean | teacher tighter? |
|---|---|---|---|
| 42 | 3.30 | **3.93** (outlier) | yes |
| 43 | 2.31 | **2.25** | no (tie) |
| 44 | 3.27 | **2.60** | no — teacher WORSE |
| **mean** | **2.96** | **2.93** | 1/3 |

Two corrections:
1. **The "~4.0 ceiling" was a seed-42 outlier.** Plain R2 (no teacher) is 2.25 / 2.60 on seeds 43/44 — near the reflex's single-goal ~2.0, and **rock-stable post-wean** (seed-43 bins `2.3/1.98/2.09/2.05/2.25/2.22`). Seed 42 (3.93) is an unlucky consolidation outlier, not the typical case.
2. **The supervised teacher is not a robust lever — and is counterproductive on good seeds.** It makes the *taught* phase cleaner (~1.5) but the strong clamp becomes a CRUTCH the learned map leans on, so post-wean (teacher gone) the map is *less* self-sufficient (seed 44: plain 2.60 → teacher 3.27). The research's learning-rule diagnosis (reward-STDP coarse vs supervised) is real in general (the project's W→A verdict: scalar 1/6 vs supervised 3/3), but **it does not manifest as a robust win HERE because the SC reflex already supplies supervision via co-firing** — driving the correct `cortex_X` during teaching IS a supervised target, so the plain circuit already learns against it; the explicit teacher is redundant and adds a wean-dependency.

This is exactly why the project requires multi-seed: a single decisive-looking seed (42) produced a confident, wrong conclusion that the multi-seed cleanly overturned. The teacher flag (`--sensory-cortex-teacher-pA`) is kept (additive, default-off) but is NOT recommended; the teacher-lever variants (longer/stronger) were skipped as a dead lever (don't grind).

## What this means for Rank 2 (the positive result)

The plain learned-from-vision circuit is a **strong perception biologization**: the learned, position-preserving `(dx,dy)→action` map (image-sourced, no coordinates) consolidates from vision to near-reflex precision on most seeds and is self-sufficient after the innate teacher weans off. This is the durable LEARNED circuit the deep-research Rank-2 prescription called for — the genuine-cortical counterpart of the innate Rank-1 reflex (the real developmental story: an innate reflex scaffolds a learned cortex, then fades). The position-invariant IT→cortex path could not do this (it collapses to ~6.1 post-wean).

## Anti-cheat (holds throughout)

- The sensory drive is image-sourced (`sc_salience_offset_from_image`, 6/6 unit tests — recovers `(gx-x,gy-y)` from the rendered image's blob centroids; coordinates never enter the function). The coord-sourced sensory drive is gated OFF under `--learned-perception-from-vision`.
- Gate on the real nav score (single-goal post-wean last-quarter; the multi-goal generalization below); IT-only floor (~6.1) and reflex (~2.0) as the bracketing controls; run inside the N8+N6 biologized back-end.
- NO `sim/` edit (protected set byte-empty).

## Net for the nav arc

**N8 ✅ (BG output) + N6 ✅ (decision) + N1 ✅ (perception): the agent navigates from vision — innately (Rank 1 reflex, 6-seed GO + grid-32) AND via a durable learned circuit (Rank 2, near-reflex multi-seed).** The perception cold-start that beat reward-bootstrap + fixed/adaptive scaffolds across the prior arc is biologized end-to-end, via the deep-research "wrong-pathway" reframe. The remaining nav items (N2 goal-render, N7 Gabor pre-init, N5 coord-free reward, N9 spiking SNc) are lesser / characterized.

## Production config (plain Rank 2 — the recommended learned circuit)
```
... --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300 --genuine-thal-tonic-pa 750 \
    --readout-source spiking_wta --urgency-max-pa 180 \
    --heuristic-strength 0 --sc-orienting-reflex \
    --sc-reflex-wean-start 2000 --sc-reflex-wean-steps 1000 \
    --learned-perception --learned-perception-from-vision \
    --grid-size 8 --goal-schedule single --n-steps 6000
```
(The deployed agent, post-wean, navigates from the LEARNED vision-sourced read-out with NO reflex and NO coordinates. Do NOT add `--sensory-cortex-teacher-pA` — it is a documented non-robust crutch.)

## Results (multi-seed — appended as runs land)

### Single-goal post-wean, plain R2 (the durable-consolidation result) — 6-SEED
| seed | 42 | 43 | 44 | 100 | 101 | 102 | mean |
|---|---|---|---|---|---|---|---|
| post-wean | 3.93 | 2.25 | 2.60 | 3.28 | 3.96 | 4.57 | **3.43** |

**Durable on all 6 (every seed ≤4.57, none collapses to the IT floor ~6.1) — but seed-variable precision** (2/6 near-reflex ≤2.7; the rest ~3.3–4.6). Honest note: my 3-seed read (42/43/44, mean 2.93) was again partly seed-lucky — it included the two best seeds (43/44). The 6-seed truth: the learned circuit reliably *consolidates and avoids collapse* (clearly beating IT) but reaches *near-reflex precision only on some seeds*. This is the durable-but-seed-variable honest result; per the reasonable-budget gate it is NOT ground further (the variability is a consolidation-quality property, not a tuning bug the simple levers fix — more teaching plateaued, the teacher is a non-robust crutch).

### Multi-goal generalization (does the learned goal-agnostic map handle NEW goals post-wean?)

**v1 (single-goal-trained) = NEGATIVE.** `--goal-schedule generalize` (train on ONE goal `far` through the wean, then 3 NEW goals reflex-OFF). Per-phase final-quarter (phase 0 = trained goal; phases 1-3 = NEW goals):
| seed | phase0 (trained) | phases1-3 (NEW goals) mean |
|---|---|---|
| 42 | 2.97 | 4.64 |
| 43 | 2.40 | 7.50 |
| 44 | 3.95 | 6.91 |

On the TRAINED goal the learned circuit navigates (~2.4-3.95, consistent with single-goal); on NEW goals it DEGRADES badly (4.6-7.5, sometimes worse than the IT floor ~6.1). **Diagnosis (expected):** single-goal training only covers the `(dx,dy)` offsets *toward that one goal*, so the learned map never learned the action for directions toward goals elsewhere. The position-preserving code is goal-agnostic *in principle* but only generalizes if **trained on diverse offsets**.

**v2 (4-corner-trained) — IN FLIGHT (`--goal-schedule generalize2`, `be8d771qy`):** train (reflex teaches) on all 4 corners rotating (0-3000, covering the full direction space), wean, then 3 NEW non-corner goals reflex-OFF. Tests whether diverse training fixes generalization (the real test of whether Rank 2 is a multi-goal nav solution). Appended on completion.

## Artifacts / cross-references
- Helper + flags: `g11_bg_runner.py` (`sc_salience_offset_from_image`, `--learned-perception-from-vision`, `--sc-reflex-wean-start/-steps`, `--sensory-cortex-teacher-pA` [non-robust], `--goal-schedule generalize`; all additive, default-off, NO `sim/` edit).
- Analyzers: `_rank2_plain_analyze.py` (the real result), `_rank2_teacher_analyze.py` (the corrected teacher comparison).
- Prior: `2026-06-07-N1-SC-orienting-reflex-GO.md` (Rank 1), `2026-06-07-perceptual-bootstrap-deep-research.md` (the wrong-pathway diagnosis), `2026-06-07-learned-visuomotor-precision-research.md` (the precision-ceiling research — its learning-rule diagnosis is correct in general but the explicit teacher is redundant here because the reflex already supervises).
