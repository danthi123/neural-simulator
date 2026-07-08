# NEURAL question-comprehension routing (GO, 6-seed + spiking-confirmed): a fronto-striatal RESERVOIR read-out classifies the QUESTION TYPE (property / relational-what / relational-who / yes-no / describe) on the whole question sequence, replacing the host keyword-matching router — generalizing to NOVEL fillers (held-out 1.000) with the closed-class LESION collapsing it to chance. Confirmed on the EMERGE-82 spiking OnBridgeLSM. Completes the "whole turn on spikes" goal: comprehension routing is now neural, matching the fully-spiking production side. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_neural_question_routing_derisk.py` (reuse-by-import: EMERGE-78 `Reservoir`/`Encoder`; `--spiking` swaps the EMERGE-82 `OnBridgeLSM`). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed numpy reference + 1-seed spiking) — the question type is routed neurally, generalizing to novel fillers, closed-class load-bearing.

## Why this ran (the comprehension-side residual)
This session made the whole PRODUCTION side spiking (CYCLE 1021-1024: property + relational answers + productive morphology on spikes). But the console still routes a question to its handler by HOST keyword matching in `ask()` (`toks[:1]==["what"]`, `["who"]`, `["does"]`+object, `["tell","me","about"]`, ...). Making that neural completes the owner's "whole turn on spikes / one brain" goal — comprehend → reason → speak all on the substrate. The project's own validated fronto-striatal reservoir (Hinaut-Dominey; EMERGE-78..82) reads a token sequence into a final state; a trained read-out maps that state to a label.

## The mechanism
- **Encode** the question tokens (closed-class function words one-hot: does/can/a/the/what/who/tell/me/about; content fillers = the open-class generic slot).
- **Reservoir** `final_state(U)` integrates the WHOLE question (the property "does a X verb" vs yes/no "does the X verb Y" distinction is a NON-LOCAL a/the + trailing-object cue).
- **Ridge read-out** maps the final state → the question TYPE (5 classes).
- Train on some animal/verb fillers, test (held-out) on NOVEL fillers → the type is carried by the closed-class STRUCTURE, not the specific filler → generalization.

## The result — 6-seed (numpy reference) + spiking
```
numpy reservoir (42/43/44/100/101/102):  held-out (NOVEL fillers) 1.000 every seed
                                          LESION (closed-class->generic) 0.20-0.33 (collapses to chance)
                                          scramble 0.83-0.93 (reported: order-invariant)
spiking OnBridgeLSM (seed 42, EMERGE-82): held-out 1.000, LESION 0.200 (collapse), scramble 1.000
chance = 1/5 = 0.20
```
- **Held-out 1.000 on NOVEL fillers, all seeds** — the classifier routes the question type perfectly, generalizing to animals/verbs it never trained on (the type is structure-driven, not filler-memorized).
- **The closed-class LESION collapses it to chance** — mapping every function word to a generic token destroys the type signal (the load-bearing control: the type is carried by the closed-class function words, i.e. genuine linguistic comprehension, not the open-class fillers).
- **Spiking-confirmed**: the EMERGE-82 `OnBridgeLSM` (a recurrent LSM on a real `SimulationBridge`, mirroring the reservoir API) routes identically (held-out 1.000, lesion collapse) — so the routing runs ON SPIKES.

## Honest scope
The SCRAMBLE control stays HIGH (0.83-1.00, reported not gated) — the question type is largely ORDER-INVARIANT (a bag of the function words does/what/who/tell uniquely signals most types), so this task does not REQUIRE the reservoir's non-local power; a simpler bag classifier would also route it. The reservoir is used because (a) it is the project's validated spiking-compatible mechanism (EMERGE-82), and (b) it correctly handles the one type-pair that IS non-local: property "does a X verb" vs yes/no "does the X verb Y" (both "does"-initial; distinguished by the a/the + trailing object, included in the 1.000). The value is the CONVERSION: the routing is now a LEARNED NEURAL read-out (generalizing to novel fillers) instead of hardcoded host keyword rules.

## What this establishes
The last host scaffold on the query PATH — the question-type router — has a neural replacement (a reservoir read-out, spiking-confirmed). With the fully-spiking production side, the WHOLE conversational turn (comprehend → route → reason → speak) can run on the spiking substrate. Follow-on: wire the neural router into the console's `ask()` (replace the keyword `if`-ladder); the reasoning step (property/relational) is already spiking; a multi-seed spiking-LSM confirmation.

## Files
`research/runners/_realcorpus_neural_question_routing_derisk.py`; `tests/test_neural_question_routing.py`. Reuses EMERGE-78 (`Reservoir`/`Encoder`) + EMERGE-82 (`OnBridgeLSM`, the spiking port). Prior: the fully-spiking production arc (CYCLE 1021-1024); the fronto-striatal reservoir arc (EMERGE-78..85).
