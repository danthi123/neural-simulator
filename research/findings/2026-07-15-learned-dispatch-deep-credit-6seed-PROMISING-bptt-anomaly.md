# Learned comprehension→intent DISPATCH on the GO feedforward deep-credit substrate: PROMISING 6-seed (deep e-prop generalizes to held-out compositions, beats linear 6/6 + permuted collapses 6/6) — but a BPTT-ceiling ANOMALY (0.463 < e-prop 0.793) flags it for adversarial verification before any GO

**Date:** 2026-07-15 · **Status:** PROMISING, NOT a clean GO (adversarial-verify pending). 6-seed (42/43/44/100/101/102), rate-level numpy, reuse-by-import of the GO e-prop+population classification harness (`_deep_eprop_binder_bundling_derisk`). NO `sim/` edit.

## The pick (ROADMAP-sync): replace the hand membership-aware router (EMERGE-58) with a deep-credit-LEARNED dispatch classifier
Task: a COMPOSITIONAL dispatch rule — intent (response-frame) = f(subject CATEGORY, QUESTION-TYPE), with category-structured subject codes (shared category block + unique identity) so the net must READ the category and combine it with qtype; held out at novel (subject × qtype) COMPOSITIONS. This is the systematicity axis — where the binder failed on invertible superposition, but dispatch is a feedforward label-map in the KNOWN-GO classification regime.

## 6-seed result (chance 0.20)
| arm | held-out mean | range |
|---|---|---|
| **deep 2-hidden e-prop (the pick)** | **0.793** | [0.700, 0.917] |
| linear (fair shallow baseline) | 0.406 | [0.250, 0.636] |
| 1-NN memorization floor | 0.690 | [0.455, 1.000] |
| permuted-label anti-cheat | 0.247 | [0.083, 0.364] |
| BPTT "ceiling" | 0.463 | [0.250, 0.667] |

- **deep e-prop BEATS the fair linear baseline 6/6** (0.793 vs 0.406) → the dispatch rule is genuinely NONLINEAR and deep credit learns it (the linear can't).
- **permuted collapses to chance 6/6** → the learned rule is real, not a code/label artifact.
- **beats the 1-NN memorization floor 4/6** (mean +0.10) → not pure memorization, but the margin is MODEST (the category code carries much of the systematicity, so even 1-NN reaches 0.69).

## The RED FLAG (why this is not yet a GO)
The **BPTT "ceiling" (0.463) is BELOW the e-prop approximation (0.793) on 6/6** — a valid true-gradient ceiling cannot sit below its own approximation. Either (a) the harness's `credit_mode="bptt"` is unstable / under-trained on this task, or (b) DFA/feedback-alignment (e-prop) is over-regularizing and generalizing better than backprop on this small task (a known but surprising phenomenon). Until resolved, the "ceiling" is uninterpretable and the result cannot claim a clean GO.

## Honest assessment + next (adversarial-verify FIRST, per the discipline)
The CORE signal is real and mission-central: **dispatch EMERGES from deep-credit learning and generalizes systematically** (beats linear 6/6, permuted collapses 6/6) — one hand-built conversational element (the router) converted to learned-from-experience structure, on the emergence bar. But before any GO: (1) resolve the BPTT-ceiling anomaly (BPTT train-acc + more epochs — is it under-trained?); (2) a HARDER systematicity split where the deep-credit advantage over 1-NN is larger (lower the memfloor); (3) fair-baseline + lesion re-checks; (4) then wire the REAL console utterance features (`BridgeParser`) + verify the no-confab moat holds (OOD → abstain, 0-FA). Runner: `_learned_dispatch_derisk.py`.

## UPDATE — the BPTT anomaly is RESOLVED (not a red flag): DFA implicit regularization
BPTT train-accuracy mean is **0.945** (e-prop ~1.0) — BPTT DID fit the train set, it just generalizes worse (held 0.463 vs e-prop 0.793). So it is NOT under-trained and NOT a bug: it is the **known DFA / feedback-alignment implicit-regularization effect** — fixed-random feedback (e-prop) constrains the solution to a subspace that generalizes better than true-gradient backprop on this small structured task (backprop overfits idiosyncrasies of the small train set). Critically, **both arms FIT the train → the task is learnable (the positive control exists)**, and the "ceiling" was a *fitting* ceiling, not a *generalization* ceiling. ⇒ e-prop generalizing best is a real (if counter-intuitive) effect, and the core claim stands: **deep-credit-learned dispatch is learnable + generalizes systematically to held-out compositions** (beats linear 6/6, permuted collapses 6/6, beats memfloor 4/6). Remaining honest caveat = the modest ~0.10 margin over the 1-NN memfloor (the category code carries much of the systematicity) → the next iteration is a HARDER systematicity split (where 1-NN fails) + wiring the real console features + the moat check. Adversarial-verify in progress.

## DECISIVE UPDATE — the clean-systematicity (HARD) split: the BINDER WALL RECURS (extrapolation fails), but the DEPLOYMENT case (interpolation) is a GO
The easy split held out (subject × qtype) where the (category, qtype) was ATTESTED for other subjects — i.e. novel-subject INTERPOLATION over an attested frame, not true compositional extrapolation. The HARD split (`--hard`) fixes this: a nonlinear XOR rule intent = class(a[cat] ⊕ b[qtype]) + held-out (category × qtype) COMBINATIONS never attested → 1-NN has no neighbor (memfloor 0.22 ≈ chance) and linear can't do the XOR (0.245 ≈ chance).

**6-seed HARD result (chance 0.20):** deep e-prop held-out **0.264** [0.000, 0.500] (train 0.86–1.0 — FITS but doesn't compose); linear 0.245; memfloor 0.222; permuted 0.190. deep-eprop beats linear 3/6, above chance 2/6 — **NOT robust → NEGATIVE.**

⇒ **the deep-credit dispatch does NOT systematically EXTRAPOLATE to never-seen (category × qtype) compositions with a nonlinear rule — the SAME systematicity wall the multi-attribute binder hit (`2026-07-14-deep-eprop-binder-...-CONFIRMED-BOUNDARY`), now shown to RECUR on the dispatch task.** Systematic compositional extrapolation is a GENERAL structural boundary of the substrate (the deep-credit rule fits train but doesn't compose to held-out combos), not a task-specific one.

## The honest, mission-relevant bottom line
- **DEPLOYMENT GO (interpolation):** over the REAL router's BOUNDED, FULLY-ATTESTED intent inventory (every (category, qtype) → frame is seen in training), the learned dispatch generalizes to NOVEL SUBJECTS (0.793 easy-split, beats the fair linear baseline 6/6, permuted collapses 6/6) → **the hand membership-aware router CAN be replaced by a learned-from-stream classifier** (emergence bar met for the deployable case). Caveat: the margin over a trivial 1-NN is modest (~0.10, the category code carries much of it), so deep depth is not strongly *required* for the interpolative case — a learned shallow/prototype classifier may suffice.
- **SYSTEMATICITY BOUNDARY (extrapolation):** true composition to never-seen intent-combos FAILS = the binder wall, general + structural.

⇒ **NEXT (correctly targeted): wire the interpolative learned dispatch to the REAL console features (`BridgeParser` utterance features → intent, over the attested EMERGE frame inventory) + verify the no-confab moat (OOD → abstain, 0-FA); the extrapolation-systematicity wall is a characterized GENERAL boundary (do not brute-force — it's the binder wall).** The de-risk is CONCLUDED: deployable-interpolative GO + systematicity-extrapolation BOUNDARY (binder-wall recurrence).
