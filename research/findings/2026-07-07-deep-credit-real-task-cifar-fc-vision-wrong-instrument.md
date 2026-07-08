# Deep-credit real-task de-risk, part 1 (Gabor/raw-CIFAR): FC-vision is the WRONG instrument — on real images, "depth is required" and "the bio-plausible rule can learn it" are ANTI-CORRELATED. Where depth genuinely helps (k=10 raw, oracle depth-gap +0.061) the rule is at CHANCE (0.100); where the rule learns (k≤4, 0.5–0.63) depth is NOT required (1-layer ≈ deep). ⇒ change instruments to a goal-relevant COMPOSITIONAL-SEMANTIC task, not more FC-vision.

**Date:** 2026-07-07
**Runner:** `research/runners/_gabor_cifar_deep_credit_derisk.py` (`--input-mode {v1,raw,rawrgb}`; Stage-0 depth-genuineness oracle gate + Stage-1 deep-credit arms + per-layer credit-alignment + the Trap-B wrong-sign-fails-alignment control). NO `sim/` edit. Real CIFAR-10 images. This is a first-class instrument-and-scale finding (the owner directive: pursue the solution off the toy; a toy limitation is not a boundary).
**Verdict:** FC-vision is not a valid deep-credit instrument. NOT a wall in the mechanism — a mismatch between the task class and the mechanism's reachable scale. Redirect to the goal-relevant compositional-semantic task.

## The evidence (1-seed diagnostic sweep, real CIFAR, lr=0.2, ep=200–300, H=128–256)
| task | oracle: linear / 1-layer / deep-best | depth-gap (deep − shallow) | deep-credit rule (test-fixed) held-out |
|---|---|---|---|
| V1 k=3 | 0.454 / 0.460 / 0.565 | +0.105 (but deep-best 0.565, weak) | 0.333 = chance (at lr=0.5; lr was the earlier issue) |
| V1 k=2 cat/dog | 0.650 / 0.555 / 0.629 | **−0.021 (LINEAR best)** | 0.536 (learns; depth not needed) |
| V1 k=2 plane/auto | 0.710 / 0.619 / 0.674 | **−0.036 (LINEAR best)** | 0.626 (learns; depth not needed) |
| raw k=2 cat/dog | 0.536 / 0.583 / 0.598 | +0.014 (tiny) | 0.500 ≈ chance/1-layer |
| raw k=2 plane/auto | 0.698 / 0.752 / 0.757 | +0.005 (tiny) | 0.588 ≈ 1-layer |
| raw k=4 animals | 0.288 / 0.370 / 0.383 | +0.013 (tiny) | 0.318 ≈ 1-layer |
| **raw k=10** | 0.162 / 0.224 / 0.285 | **+0.061 (REAL depth-gap)** | **0.100 = CHANCE (rule CANNOT learn)** |

## The two load-bearing findings
1. **On real FC-vision, depth-requirement and rule-learnability are ANTI-CORRELATED.** The only config with a real oracle depth-gap (k=10 raw, +0.061 — matching the scoping's "1-hidden 59%→2-hidden 52% err") is exactly where the bio-plausible deep-credit rule collapses to CHANCE (0.100; even backprop only reaches 0.285 — FC-CIFAR-10 is hard). Every config the rule CAN learn (k≤4) has a ~0 depth-gap (a 1-hidden layer captures the FC-decodable signal). So the deep-credit rule cannot demonstrate a depth-BENEFIT on real FC-vision: the tasks that need depth are beyond its reachable scale; the tasks in-scale don't need depth.
2. **Real image depth-benefit is CONVOLUTIONAL, not FC-MLP.** A fixed V1/Gabor front end makes CIFAR linearly decodable (depth-gap ≤ 0, linear best) — the same reason MNIST-FC is shallow. Raw-pixel FC shows only a tiny depth-gap until k=10, where the rule fails. FC depth is simply not where CIFAR's structure lives.

## Why this is a REDIRECT, not a boundary (the reframe)
This is the multi-order SCALE wall the research gate + scoping pre-flagged ("the burst family is the least-scaled bio-plausible method; a depth-3-on-a-toy GO is 3–4 orders below scale"), made concrete on a real task. It does NOT wall the mechanism (validated: D1 ports to spikes, D2 KP transport-free). It says **FC-vision is the wrong task class**:
- Real depth-benefit is ARCHITECTURAL (convolution for vision, recurrence for language) — the FC deep-credit rule's value is INSIDE a deep architecture, not standalone FC.
- The GOAL is language, and the goal-relevant depth is COMPOSITIONAL: hierarchical semantics (multi-level is-a / relational inference) is genuinely depth-requiring (a shallow net cannot compose multi-level relations — the EMERGE inheritance/transitivity arc showed this needs 2-hop structure) AND small (fewer concepts than CIFAR-10 pixels → within the rule's reachable scale). That is the scoping's #2 task, now promoted.

## Next (driving): the compositional-semantic deep-credit de-risk
Does the validated deep-credit rule LEARN a real hierarchical-semantic task where DEPTH = multi-level composition (from the EMERGE corpus co-occurrence features), where a 1-hidden-layer net provably underfits the composition? Same Stage-0 depth-genuineness gate + per-layer-alignment metric + Trap-B control, reusing the EMERGE stream/semantic infra. If depth is required AND the rule learns it with an alignment signal → the real-task GO the FC-vision instrument couldn't give. If not → the honest scale wall stands and the frontier is architecture/scale.

## Files
`research/runners/_gabor_cifar_deep_credit_derisk.py` (`--input-mode`); `research/findings/raw/_gabor_cifar_{smoke42,catdog42,planeauto42,raw_*}.json`. Scoping: `2026-07-07-deep-credit-real-task-scoping.md`.
