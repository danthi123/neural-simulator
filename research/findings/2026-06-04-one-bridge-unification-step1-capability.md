# One-bridge unification step 1 — capability gate: robust core PRESERVED, two-attribute regresses at the marginal D — 2026-06-04

**One line:** Task 6 (the no-regression gate) at production scale, multi-seed (42/43/44), comparing the UNIFIED
one-bridge agent to the SEPARATE two-bridge baseline on the real `denoise64` codes: the **robust core (flat fact /
one-attribute / negation / comprehension) is preserved** on the merged bridge; **two-attribute — the K=5
capacity-edge category, already a documented seed-variable boundary — regresses ~1 trial mean** at the marginal
`proj_dim=800`. A re-run at the stage-1.5 production dimension `D=2048` (where two-attribute has dimensional
headroom) is in flight to test whether the regression is a marginal-D artifact.

## Result (proj_dim=800, separate vs unified, 6 trials/category)

| seed | flat | one-attribute | two-attribute | negation | parser (merged bridge) |
|---|---|---|---|---|---|
| 42 | 6/6 → 6/6 | 6/6 → 6/6 | 6/6 → **3/6 (−3)** | 12/12 → 11/12 | active dog/go/north, passive agent=dog ✓ |
| 43 | 6/6 → 6/6 | 5/6 → 5/6 | 4/6 → **2/6 (−2)** | 12/12 → 12/12 | ✓ |
| 44 | 6/6 → 6/6 | 4/6 → 4/6 | 3/6 → 5/6 (+2) | 10/12 → 11/12 | ✓ |

- **Robust core preserved.** Flat fact, one-attribute, and negation are within ±1 trial of the separate baseline on
  every seed; the parser comprehends correctly (voice-invariant: active "dog go north" and the passive frame both
  assign agent=dog) on the merged bridge. So the load-bearing conversational capabilities — comprehend, store,
  recall (who/what), abstain, negate — run on ONE bridge with no regression.
- **Two-attribute regresses** at seeds 42 (−3) and 43 (−2), improves at 44 (+2). Mean: separate (6+4+3)/3 = 4.33 vs
  unified (3+2+5)/3 = 3.33 — a ~1-trial mean drop. Note two-attribute is **already seed-variable in the separate
  baseline** (6 / 4 / 3) — it is the documented K=5 capacity-edge boundary, the noisiest category.

## Why two-attribute specifically — and the mitigation

Two-attribute is the highest-load category (five bindings → the noisiest decoded estimate → the smallest margin),
already at its capacity edge at `D=800`. The shared bridge adds perturbation to that marginal category: shared OU
background noise, and — the likely dominant cause — the composer's neurons receive **different per-neuron
heterogeneity/OU draws** on the merged bridge (they sit at index offset 126 on a 6526-neuron bridge, so the
seeded random draws differ from their standalone 6400-neuron bridge). The robust categories have margin to absorb
this; two-attribute, at the edge, tips at 2/3 seeds.

The principled mitigation is **dimension**: stage 1.5 already decided the production agent runs at **D=2048**, where
two-attribute resolves even under high code correlation (the dimensional-cost finding). At D=2048 two-attribute has
headroom that should absorb the shared-bridge perturbation. **Re-run at D=2048 is in flight**
(`research/findings/raw/_unified_capability_D2048.log`); this finding will be updated with the outcome:
- If two-attribute is preserved at D=2048 → step 1 has **no regression at the production dimension** (the D=800 drop
  was a marginal-regime artifact) → step 1 of B is DONE.
- If it still regresses at D=2048 → the cost is structural (the composer-neuron heterogeneity shift), and the
  mitigation is per-region seeding (align the composer slice's neuron parameters to its standalone draws) — a real
  but localized fix, surfaced for the decision.

## Honest framing

Step 1's load-bearing claim — *the parser + composer run on ONE interacting bridge with the core conversational
capabilities preserved* — is VALIDATED at production scale, multi-seed. The single regression is on the documented
capacity-edge boundary category at the marginal dimension, with a principled dimensional mitigation in flight. The
regression is recorded, not hidden; the no-regression test was left honest (it fails on two-attribute at D=800) and
the heavy multi-seed comparison is the on-demand probe `_unified_bridge_capability_probe.py`.

## Files
- `research/findings/raw/_unified_bridge_capability_probe.py` (the multi-seed unified-vs-separate comparison)
- `research/findings/raw/_unified_capability_probe_run.log` (D=800 result) + `_unified_capability_D2048.log` (D=2048, in flight)
- `tests/test_unified_brain_bridge.py::test_unified_capability_no_regression` (heavy on-demand; honest)
