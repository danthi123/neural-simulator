# EMERGE-1 — BOUNDARY (the honest prior): the emergence wall is the LOCAL RULE's depth-scaling, not (only) the point neuron

**2026-07-01 (autonomous; the pivotal cheap-first gate from `2026-07-01-dendritic-cortex-for-emergence-scoping.md`,
run to answer the owner's dendrite-for-emergence question with EVIDENCE before any months-scale build).** Reuse-by-
import (`sim.dendritic_mlp.DendriticMLP`, the already-built DEEP feedback-alignment learner); **NO `sim/` edit**; CPU.

## The question
Does a DEEP (≥2 hidden-layer) dendritic net, learning by a **biologically-plausible LOCAL rule** (feedback alignment —
fixed-random top-down feedback, no weight transport — + the committed Urbanczik-Senn hidden rule), **DEVELOP**
generalizable hierarchical structure that a point-neuron / single-layer network provably can't? This is the one regime
every prior dendrite NEGATIVE skipped (they were single-layer). Task: a depth-2 Boolean function — threshold over the
5 pair-XORs of 10 bits (XOR needs one hidden layer; threshold-over-XORs needs a second) — with **held-out
generalization** (unseen bit-patterns) + a **linear probe** on the frozen hidden reps for the level-1 XOR latents
(did the intermediate features emerge, though never a training target?).

## Result — BOUNDARY, robust across two training budgets (3 seeds each)
| arm | held-out (300 ep) | held-out (900 ep / lr .7) | reads |
|---|---|---|---|
| **oracle backprop** (fenced ceiling / task-sanity) | **0.94** | **0.95** | the task IS cleanly deep-learnable **with weight-transport credit assignment** |
| **deep dendritic FA** (the TEST, local rule) | 0.61 (train .63–.97) | **0.58 (train → 1.00)** | **memorizes, does not generalize** — more epochs = more memorization, NOT more generalization |
| single-layer dendrite (prior-NEGATIVE regime) | 0.25 | 0.18 | genuinely fails (below chance), as expected |
| apical-lesion (B=0 → frozen hidden = point-neuron/no-credit floor) | 0.47 (probe .51≈chance) | 0.50 (probe .52) | the floor; **no hidden features emerge without top-down error** |
| wrong-sign (anti-learn) | 0.59 | 0.66 | muddy (not cleanly at chance) — the local update is weak/noisy either way |
| chance | 0.53 | 0.54 | |

- **Confirmed a real wall, not undertraining:** giving the local rule 3× the epochs + higher lr drove **train → 1.000**
  while held-out **stayed ~0.58** — it memorized harder, it did not generalize. Backprop generalizes on the identical
  task/splits (0.95). So the failure is specifically the **local credit-assignment rule's inability to develop
  *generalizable* depth-2 structure**, matching the field's documented evidence (feedback alignment / predictive coding
  credit assignment degrades with depth; test-gap widens; doesn't scale to hard problems).
- **The nuance (a characterized PARTIAL, not a flat zero):** the deep FA arm *did* develop SOME structure — the level-1
  XOR latents partially emerged in its hidden rep (probe **0.65–0.78** vs the frozen-hidden floor **0.51 = chance**),
  and it beat the single-layer regime (0.61 vs 0.25). The deep dendritic regime helps; it just doesn't cross into clean
  generalization. That is exactly the scoping doc's forecast ("characterized partial at toy scale, not a from-
  experience competence").

## What this means (the honest, build-saving read)
1. **The emergence wall is NOT (only) the point neuron — it's the DEPTH-SCALING of biologically-plausible LOCAL
   learning.** Even with the deep two-compartment / apical-feedback machine, the local rule can't credit-assign through
   two hidden layers well enough for the structure to *generalize*. Building more true-to-life *substrate* (the
   months-scale `NeuronModel.TWO_COMPARTMENT` rewrite) would **not** clear this wall — we'd hit the same rule-depth
   ceiling. **Do NOT start that build on the emergence premise.** (This is the point of the gate; it just saved a
   months-scale build.)
2. **The tension, now empirical:** *backprop* credit-assigns through depth and generalizes (oracle 0.95) — but backprop
   is not biologically local (weight transport). The biologically-*plausible* local rules do not (at our scale). So
   "emergence the way a brain develops it" (local rules) is a confirmed hard wall here; "emergence the way an LLM gets
   it" (backprop) works but isn't biology. **That is the crux of why we hand-design**, made concrete: it is not merely
   "point neuron vs dendrite," it is "local plausible rule vs backprop through depth."
3. **This is the honest negative the owner set as the deliverable** (`project_actual_goal_artificial_life_brain_analogue`
   — capabilities instrumental, honest negatives under strict biology ARE the deliverable). It maps the wall precisely.

## The precisely-narrowed open question (a CHEAP follow-on, NOT the substrate rewrite)
EMERGE-1 tested **vanilla feedback alignment** (the built `DendriticMLP`). The field has *stronger* biologically-
plausible local rules that target exactly this depth-scaling wall: **burst-multiplexed plasticity** (Payeur-Guerguiev-
Zenke-Richards-Naud 2021), **predictive-coding with depth-scaling parameterization** (Depth-µP), and **local
target-propagation** variants. So the honest open question narrows to: *is there a biologically-plausible LOCAL rule
that credit-assigns through depth at our scale?* — the field's own open problem. Testing a stronger local rule is a
**cheap follow-on** (extend/replace the rule in `DendriticMLP`, CPU, hours, NO `sim/` edit), and it is the *right* next
probe **before** any substrate build — because if no local rule clears the depth wall, the substrate rewrite is moot,
and if one does, it localizes the build to that rule.

## Verdict
- Owner's intuition (dendrite = top substrate lever for emergence): **directionally right but confirmed NOT SUFFICIENT
  at our scale** — the binding constraint is the local rule's depth-scaling, empirically, not the neuron model.
- **Build-saving:** the months-scale two-compartment `sim/` rewrite is NOT warranted on the emergence premise (it
  wouldn't clear the confirmed wall). 
- **Next cheap probe (if we pursue emergence further): EMERGE-1b** — swap a stronger biologically-plausible local rule
  (burst-multiplexing / PC-Depth-µP) into the deep learner and re-run this exact harness; GO only if it *generalizes*
  through depth where vanilla FA memorized. Otherwise the honest posture is the earlier **re-scope**: backprop-trained
  components (the small generator) do the deep-emergent-structure parts, the spiking brain does its comparative
  advantage (grounding, continual memory, the no-confab moat, embodiment).

**Artifacts:** `research/runners/_emerge1_deep_dendritic_representation_derisk.py`; result
`research/findings/raw/_emerge1_deep_dendritic_representation.json`; scoping
`2026-07-01-dendritic-cortex-for-emergence-scoping.md`.
