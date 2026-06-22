# Tier-2 #6 Route-B content-matched — 6-seed CORRECTION: the DA encoding-gain effect is REAL but MODEST (diff-in-diff +0.208, 5/6 positive); the single-seed GO was a clean-assignment artifact (2026-06-22)

**Supersedes** the single-seed `2026-06-22-tier2-routeB-content-matched-GO.md` (retracted — seed 42 drew a clean HI/LO
assignment so its raw +0.375 was almost entirely DA; the 6-seed picture is more nuanced + more honest).
`research/runners/_tier2_routeB_content_matched.py`, GPU. NO `sim/` edit. On `main`.

## The 6-seed result — the runner's strict per-seed verdict + the proper difference-in-differences
| seed | main effect (meanHI−meanLO) | lesion effect (content-only) | **DA effect (diff-in-diff = main − lesion)** | runner verdict |
|---|---|---|---|---|
| 42  | +0.375 | +0.000 | **+0.375** | GO |
| 43  | +0.375 | +0.125 | **+0.250** | NEGATIVE\* |
| 44  | −0.125 | −0.250 | **+0.125** | LATENT\* |
| 100 | +0.375 | +0.125 | **+0.250** | NEGATIVE\* |
| 101 | +0.500 | +0.625 | **−0.125** | NEGATIVE\* |
| 102 | +0.250 | −0.125 | **+0.375** | NEGATIVE\* |
| **mean** | **+0.292** | **+0.083** | **+0.208 (5/6 positive)** | 1 GO / 4 NEG / 1 LATENT |

\* The runner's per-seed verdict requires the LESION effect to be ~0 (i.e. the random HI/LO assignment must balance the
per-fact content-robustness). At N=16 it does NOT (the lesion ranges +0.000 to +0.625), so the strict gate fails on
5/6 seeds — NOT because the DA effect is absent, but because the small-N assignment leaves a per-seed content imbalance
comparable in size to the effect.

## The honest read
- The **proper isolation** of the DA effect is the **difference-in-differences**: (main effect) − (lesion effect) =
  the recall benefit attributable to the DA-driven encoding gain ALONE, with the per-seed content imbalance
  subtracted. That is **+0.208 mean, positive on 5/6 seeds** (only seed 101 inverts).
- ⇒ the DA encoding-gain effect is **REAL but MODEST** (~+0.2 recall, content-controlled), NOT the clean +0.375 the
  single-seed GO suggested. Seed 42 happened to draw a balanced assignment (lesion +0.000), so its raw +0.375 was
  almost all DA — a best-case seed, not the typical one. Honest self-correction of an overstated single-seed GO
  (the project's permuted-label discipline: a single clean seed is not a multi-seed result).
- The **mechanism is unaffected + solid:** the shared spiking dopamine reaches + modulates the composer's encoding
  (the deployment smoke established this; g 1.08→1.69 from the real SNc). The **no-confab moat held at 0 false-accepts
  on ALL 6 seeds** — the salience lever never trades against the moat.

## Verdict — MODEST (real, content-controlled, moat-safe; NOT a clean GO)
Tier-2 #6 (the owner's "one self" closure) is **mechanistically real and behaviorally positive-but-modest**: a shared
spiking dopamine that biases the composer's encoding strength produces a ~+0.2 content-controlled recall benefit
(5/6 seeds), moat-safe. An honest, characterized result.

## Next
1. **Larger-N re-run (N=48) — DONE, CONFIRMS the effect (2026-06-22).** 6-seed at N=48: the per-seed strict verdict
   is now **4/6 GO** (vs N=16's 1/6 — the larger fact set cleaned the per-seed content confound substantially), and
   the **diff-in-diff (main − lesion) is +0.306 mean, positive on ALL 6/6 seeds** (vs N=16's +0.208, 5/6). Per-seed
   main/lesion/diff-in-diff: 42 [+0.250/−0.042/**+0.292** GO], 43 [+0.333/−0.042/**+0.375** GO], 44
   [+0.250/−0.125/**+0.375** NEG\*], 100 [+0.333/+0.083/**+0.250** GO], 101 [+0.167/+0.000/**+0.167** GO], 102
   [+0.167/−0.208/**+0.375** NEG\*]. Moat 0-FA every seed. (\*The two NEGATIVEs are residual per-seed content
   imbalance — non-zero lesion — but their diff-in-diff stays +0.375.) ⇒ the DA encoding-gain effect is CONFIRMED
   real-but-modest (~+0.3 content-controlled recall), slightly stronger + more consistent at N=48. **Tier-2 #6
   reward→memory: closed as real-but-modest, multi-seed-confirmed.**
2. The richer DA→memory routes (DA-gated **reconsolidation**, salience-gated **recall**) likely give a LARGER
   behavioral effect than encoding-strength-as-noise-robustness (they act where the read model is more sensitive than
   a uniform per-fact magnitude scalar) — the natural next one-self levers, now that encoding-strength is confirmed
   real-but-modest.
