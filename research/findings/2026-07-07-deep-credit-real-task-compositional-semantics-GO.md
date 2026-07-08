# Deep-credit real-task de-risk, part 2 (compositional semantics) — the FIRST real-task traction off the toy: feedback-alignment/deep-credit TRAINS a deep net to a genuine depth-required compositional-GENERALIZATION task (6-seed held-out 0.691, margin +0.41 over the well-trained 1-layer floor, strong on 5/6 + above-chance on 6/6, no leakage, depth load-bearing), where FC-vision failed. Honest scope (adversarial-verify `wcrugptwx` = SURVIVES_WITH_SCOPE_FIXES): at the numpy RATE reference the headline arm is BYTE-IDENTICAL to plain feedback-alignment (the microcircuit/burst machinery is inert here); the credit is functionally load-bearing but NOT oracle-aligned; it is depth-2 XOR-over-shared-pool generalization (not a member-abstraction inheritance hop); the features are structured-synthetic EMERGE-style, not raw corpus.

**Date:** 2026-07-07
**Runner:** `research/runners/_semantic_inheritance_deep_credit_derisk.py` (reuse-by-import; NO `sim/` edit). Adversarial-verify: workflow `wcrugptwx` (2 SURVIVES + 2 CONCERN → SURVIVES_WITH_SCOPE_FIXES; every scope-fix folded in).
**Verdict:** a genuine real-task composition SIGNAL (the first traction off the toy) with the framing corrected to exactly what it demonstrates. NOT a distinctively-microcircuit or oracle-aligned-deep-credit GO.

## Why this task (from part 1)
FC-vision (`2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md`) was the wrong instrument (depth-required ⟂ rule-learnable). This task makes depth required by SYSTEMATIC GENERALIZATION: a 2-level XOR-pair taxonomy where the class PROPERTY is a NONLINEAR (XOR) function of the superordinate pool; held-out members appear with their is-a features but their property is NEVER a training target → a 1-hidden-layer net memorizes train members but cannot compose the nonlinear property for held-out members; a 2-hidden-layer net can. (EMERGE-style co-occurrence feature overlap; the depth-required analogue of EMERGE-26, as a SUPERVISED deep-credit problem.)

## The result (6-seed: dev 42/43/44 + blind 100/101/102; hidden=96, ep=250, lr=0.3, deep-layers=2) — the conservative single primary arm (test_fixed)
| seed | held-out (rule) | 1-layer floor | oracle | margin (rule − floor) | above chance |
|---|---|---|---|---|---|
| 42 dev | 0.870 | 0.407 | 1.000 | +0.463 | +0.703 |
| 43 dev | 0.889 | 0.296 | 0.981 | +0.593 | +0.722 |
| 44 dev | 0.796 | 0.278 | 0.926 | +0.519 | +0.629 |
| 100 blind | 0.407 | 0.333 | 0.981 | +0.074 (weak) | +0.240 |
| 101 blind | 0.630 | 0.148 | 0.870 | +0.481 | +0.463 |
| 102 blind | 0.556 | 0.241 | 1.000 | +0.315 | +0.389 |
| **6-seed** | **0.691 ± 0.176** | 0.284 | 0.960 | **+0.407 ± 0.171 (>0.20 on 5/6)** | **above chance on 6/6** |

- **A composition SIGNAL on 6/6 seeds** (every seed above chance 0.167, all controls hold), STRONG on 5/6 (margin +0.32–0.59 over the floor), WEAK-but-above-chance on blind seed 100 (+0.074 over floor, +0.240 over chance). The rule sits between the shallow floor (0.28) and the oracle (0.96).

## What the adversarial-verify independently REPRODUCED + confirmed (the valid basis for the signal)
- **Genuine composition, NO leakage** (Lens 1, reproduced byte-exact): held-out properties are never a training target (0 row-leakage all seeds); the memorization control (untaught superordinate → reserved novel class never in training) is **0.000 on all 6 seeds** (oracle AND every arm — structurally un-inferable); the property is NOT linearly decodable (honest linear probe 0.185 ≈ chance); the member-id block cannot leak (id-only 0.093, below chance); permuted → chance.
- **The floor is FAIR + depth is LOAD-BEARING** (Lens 2): the 1-layer floor is same-init/same-budget, trains to train=1.0, and PLATEAUS at ~0.44 under 8× epochs / higher lr / 2–8× width — it genuinely cannot compose. A **deepest-layer FREEZE ablation collapses the net to the floor (0.870→0.167)** → the deep layer IS load-bearing for the accuracy.

## The three framing corrections (the scope-fixes — the honest what-this-is-NOT)
1. **At the numpy RATE reference the headline "microcircuit" arm is BYTE-IDENTICAL to plain feedback-alignment** (test_fixed == plain_fa, all 6 seeds — same accuracy AND same per-layer alignment). The interneuron/burst machinery is inert at rate (consistent with D1/rung-2). So this is **feedback-alignment trained deep credit**, not a distinctively-microcircuit result; the microcircuit/burst distinction is only claimed to matter on the spiking substrate (the deferred on-bridge run).
2. **The credit is functionally load-bearing but NOT oracle-aligned.** The deepest-layer credit-alignment-to-oracle is ~0 every seed (−0.03 to +0.13); the runner's own SIGNAL gate (which requires alignment > 0.15) returns **False on both committed JSONs**. The GO is by ACCURACY (valid — composition is genuinely required + the freeze-ablation shows the deep layer is used), NOT by oracle-aligned deep credit. The correct claim: **the rule trains the deep net to a functional non-oracle-aligned solution that generalizes** — not "assigns oracle-correct deep credit."
3. **It is depth-2 XOR-over-shared-pool generalization, NOT member→superordinate abstraction inheritance.** A depth-2 MLP on the POOL features ALONE (member-id dropped) recovers held-out at 1.000 → the composition is XOR-pair-decode → class-combine over the SHARED superordinate pool; member-id is a memorization affordance, not a tested abstraction hop. "Inheritance across distinct members" over-narrates; the honest description is depth-2 nonlinear-composition generalization.
- **Realism (Lens 4):** structured-synthetic XOR-pair taxonomy over 9 EMERGE-style features (6 XOR pool dims + 3 random member-id dims), NOT raw TinyStories PPMI corpus codes (the cache is absent). "Goal-relevant compositional structure," not raw language.

## What this establishes (honestly)
The deep-credit APPROACH (feedback-alignment at the rate level) DOES gain real-task traction where FC-vision failed: it trains a deep net to a genuine depth-required compositional-generalization task (5/6 strong, 6/6 above chance, no leakage, depth load-bearing) on goal-relevant structured-semantic features. This is the first real-task signal off the XOR toy — depth-required compositional generalization is the setting where the approach works (not FC-vision). BUT it does NOT yet demonstrate (a) the distinctive microcircuit/burst rule (inert at rate — the spiking on-bridge run is where it could matter), (b) oracle-aligned deep credit (the rule finds a different functional solution), or (c) raw-corpus semantics. Next: the raw-PPMI-code swap; the on-bridge spiking run (where the microcircuit distinction could appear); a member→super-abstraction control if "inheritance across members" is to be load-bearing; probing seed 100. NO expensive training.

## Files
`research/runners/_semantic_inheritance_deep_credit_derisk.py`; `research/findings/raw/_semantic_inherit_{dev,blind}.json`. Part 1: `2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md`; scoping: `2026-07-07-deep-credit-real-task-scoping.md`; verify: `wcrugptwx`.
