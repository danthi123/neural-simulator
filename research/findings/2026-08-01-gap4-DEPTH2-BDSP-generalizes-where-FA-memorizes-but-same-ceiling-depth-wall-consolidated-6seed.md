---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/depth2_bdsp/depth2_bdsp_6seed_aggregate.json
---

# gap#4 crux — DEPTH-2: the coincidence-gated BDSP GENERALIZES through depth where feedback-alignment MEMORIZES, but both cap at the same held-out ceiling — the depth-scaling wall is CONSOLIDATED (6-seed)

<!--derived-->
**One-line verdict:** the first test of gap#4 credit assignment **at depth** — every prior rule this session was
single-hidden-layer against a depth-2 oracle, so "deep credit" (credit *through* multiple layers, the literal gap#4
question) was never actually tested. The depth regime has a known wall (`2026-07-01-emerge1-...BOUNDARY`: a depth-2
local rule via feedback alignment MEMORIZES but doesn't generalize). This runs the SAME depth-2 task (pair-XORs →
threshold-over-XORs, held-out generalization) with this session's genuinely-different rule — the **coincidence-gated
+ sigmoid-baseline BDSP** — at each of two hidden layers. Result (6-seed): **BDSP GENERALIZES through depth** (train
0.660 ≈ held-out 0.636, gen-gap **0.024**, 6/6) where **feedback-alignment MEMORIZES** (FA held-out 0.633 but train
~0.96, gap **0.33**). But they reach the **same held-out ceiling** (BDSP 0.636 ≈ FA 0.633, oracle 0.974), and the
capacity lever is closed (more lr/epochs/hidden destabilize the sigmoid MLP — the oracle itself collapses 0.97→0.45).
So the depth-scaling wall is **consolidated across FA AND BDSP**; BDSP's advance is *qualitative* (it generalizes
cleanly instead of overfitting), not a higher ceiling. No `sim/` edit (subclass of `DendriticMLP`).

Artifact: `research/findings/raw/gap4/depth2_bdsp/depth2_bdsp_6seed_aggregate.json` (backend numpy/CPU). Runner:
`research/runners/_gap4_depth2_bdsp_credit_derisk.py`.

## Result — depth-2 XOR-threshold task, 6 seeds

<!--derived-->
| arm | held-out | train | gen-gap | reads |
|---|---|---|---|---|
| oracle (backprop) | 0.974 | ~1.0 | — | task IS deep-learnable with weight transport |
| **deep FA** (feedback alignment) | 0.633 | ~0.96 | **0.33** | **MEMORIZES** (the emerge1 wall reproduced) |
| **deep BDSP** (coincidence-gated) | 0.636 | 0.660 | **0.024** | **GENERALIZES** (no memorization), 6/6 |
| single-layer | ~0.21 | — | — | provably fails (below chance) |
| apical-lesion (B=0) | ~0.47 | — | — | no-credit floor |
| wrong-sign BDSP | ~0.61 | — | — | anti-learn control |

chance ≈ 0.52. Both FA and BDSP recover the level-1 XOR latents *partially* (linear probe ≈ 0.64, vs chance ~0.5)
but neither captures the full level-2 (threshold-over-XORs) structure to approach the oracle.

## What this settles for the crux

<!--derived-->
Two things, both important. **(1) The deep-credit-through-depth wall is real and rule-family-robust.** A depth-2
local, transport-free rule caps at held-out ~0.63 on this task whether the credit is feedback-alignment (graded DFA)
or the coincidence-gated BDSP (event-gated bounded credit); the oracle reaches 0.97. This consolidates the emerge1
boundary across a second, genuinely-different rule — the wall is not an artifact of FA specifically. **(2) But the
FAILURE MODE differs, and BDSP's is better.** FA reaches the ceiling by MEMORIZING (train ~0.96, held-out 0.63 — it
overfits the non-generalizable part); BDSP reaches it by GENERALIZING (train ≈ held-out ~0.65 — it captures only the
generalizable structure and does not overfit). The binary coincidence gate + bounded sigmoid-baseline credit act as
a strong implicit regularizer through depth. That is a qualitatively more biologically-faithful and more desirable
learning profile (a local rule that generalizes-or-declines rather than memorizes), even though it does not move the
held-out ceiling on this task.

## Next
The residual is precise: a local transport-free rule extracts the *level-1* structure (probe ~0.64) but not the
*level-2* composition, capping held-out at ~0.63 regardless of rule family. The wall is the **depth-2 composition**,
not the memorization (BDSP already fixed that). The next levers are the ones that could capture level-2 without
weight transport: (a) a stronger inter-layer credit signal than a single fixed-random feedback (e.g. the burst-
multiplexed two-compartment channel where the burst carries a genuinely top-down target, not just a projected
error); (b) an unsupervised depth-building objective (learn the level-1 latents first, then compose) rather than
end-to-end error; (c) the on-bridge SPIKING port of the BDSP-generalizes result (this is the rate stand-in — the
generalization profile must be confirmed on real spikes). A consolidated boundary with a qualitative advance banked
and the composition residual named — the capability is open, the depth wall precisely located.
