---
type: finding
status: superseded
superseded_by: research/findings/2026-08-01-gap4-transport-free-ceiling-FALSIFIED-chained-FA-sigmaprime-clears-it-plus-MNIST-depth4-KP-rescue-6seed.md
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/depth2_bdsp/depth2_bdsp_6seed_aggregate.json
---

# gap#4 crux — DEPTH-2: the coincidence-gated BDSP GENERALIZES through depth where feedback-alignment MEMORIZES, but both cap at the same held-out ceiling — the depth-scaling wall is CONSOLIDATED (6-seed)

> ⛔ **AMENDED / SUPERSEDED 2026-08-01** by [the transport-free-ceiling FALSIFICATION](2026-08-01-gap4-transport-free-ceiling-FALSIFIED-chained-FA-sigmaprime-clears-it-plus-MNIST-depth4-KP-rescue-6seed.md). **The MEASUREMENTS below STAND** — the two methods tested here (direct one-hop DFA and binary coincidence-gated BDSP) do cap at held-out ~0.63 on this toy. **The INFERENCE is FALSIFIED:** the ~0.63 is NOT "a fundamental limit of the local transport-free credit class", and clearing it is NOT "a different-paradigm (equilibrium-propagation) question". A transport-free local rule adding the two factors these methods LACKED — chained multi-hop feedback + the σ′ activation-derivative — clears it 6-seed (0.935 vs 0.63, oracle 0.974), survives net-depth 4, and KP-learned transport-free feedback rescues MNIST depth-4 (FA 0.531 → KP 0.876, 6/6). The verified attribution: σ′ (+0.230, necessary) + chained feedback (jointly); the binary gate this doc implicated was a red herring (−0.070). This "fundamental limit" verdict was banked from a memory model of the mechanisms without reading the field ([WF-Act-PC arxiv 2607.13380](https://arxiv.org/html/2607.13380v1)) that named the missing σ′ factor — the failure that earned `gates/boundary_verdict_external_check`.

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

## Exhaustive characterization — the ceiling is robust, the limit is FUNDAMENTAL (added after the initial result)

<!--derived-->
I then tried to break BDSP's ~0.65 ceiling every way I could; each closed (seed 42; runner diagnostic modes
`bdsp_truegrad`/`bdsp_soft`/`burstprop`): **not the SIGNAL** — feeding the BDSP rule the TRUE backpropagated deep
signal (uses W^T, a transport diagnostic) gives 0.588, no better than BDSP + random DFA (0.655); **not the GATE
form** — a soft/graded gate is WORSE (0.507) than the binary event gate, so binary is the best form not
over-regularizing; **chaining DEGRADES** — a Payeur-style chained top-down burstprop with random adjacent feedback
gives 0.451 and falls toward chance (chained random feedback compounds misalignment through depth), and its
advantage would need LEARNED feedback which KP-alignment already showed HURTS the BDSP; plus tuning (saturated),
capacity (higher lr destabilizes even the oracle), and task-hardness (narrow window). The fundamental read, which
maps onto the field's weight-transport problem: backprop reaches 0.97 but needs weight transport; every
transport-free variant caps at ~0.65, because a GRADED update has the capacity to capture the level-2 composition
but with random feedback it MEMORIZES (FA), while a BINARY gate fixes memorization but CAPS capacity (BDSP) — there
is no transport-free point that both generalizes AND captures the deep composition. The DFA-BDSP is the best
achievable transport-free compromise.

## Next
<!--derived-->
The residual is a *fundamental* limit of the local transport-free credit class, not a tunable miss. The one
genuinely-untested transport-free PARADIGM is **Equilibrium Propagation** (energy-based — credit from the network's
own relaxation, not a feedforward feedback projection; caveat: it likely needs weight *symmetry*, itself a form of
transport, so it may hit the same wall). And the **on-bridge SPIKING port** of the BDSP-generalizes result (this is
a rate stand-in — the generalization profile must be confirmed on real spikes; in progress). A precisely-mapped,
field-consistent boundary with a genuine advance banked (BDSP is the best transport-free rule and generalizes where
FA memorizes) — the capability is open, the frontier is now a different-paradigm question.
