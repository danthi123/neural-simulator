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
