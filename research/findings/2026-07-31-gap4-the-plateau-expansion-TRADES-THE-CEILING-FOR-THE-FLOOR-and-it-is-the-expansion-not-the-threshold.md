---
type: finding
status: live
date: 2026-07-31
mechanism: dendritic-plateau-coincidence-burst
runner: research/runners/_gap4_deep_credit_on_expanded_forward_derisk.py
artifacts:
  - research/findings/raw/gap4/AGG_deep_credit_on_expanded_6seed.json
  - research/findings/raw/gap4/deep_credit_on_expanded_SMOKE.json
  - research/findings/raw/gap4/credit_on_expanded_6seed.json
---

# gap#4 — the plateau expansion buys a linear floor by destroying the depth structure, and it is the EXPANSION, not the binarization

**Verdict: 6 seeds — 5 UNDEFINED, 1 NO-GO, 0 GO.** The kill criterion fires wherever a verdict is available.

## What was asked

gap#4 has two halves that were solved and tested in isolation. The FORWARD half was banked 2026-07-25 as
*"forward representability SURPASSED on-bridge"* — the coincidence dendritic-plateau expander lifts held-out
linear decodability to 0.611 at 6 seeds <!--derived--> (quoted from the 2026-07-25 surpass finding, which carries it in its own artifact; NOT re-measured here)
and its own title ends *"so the CPU-rate-GO credit has features to shape"*. The CREDIT half is a powered
NO-GO on the sparse point-neuron forward. **`PlateauExpander` was imported by exactly one file — its own
probe.** The two halves had never met.

This runs the actual combination: same task, same arms, same depth, same seeds, same budget, changing only
the forward representation. Everything is imported unmodified from `_semantic_inheritance_deep_credit_derisk`
— the task, the Stage-0 depth-genuineness oracle, and the Stage-1 arms with their full anti-cheat set.

## The guard that decided the answer

Stage 0 is re-run **on each forward**. This is the load-bearing design choice, and it exists because a
one-layer learner had already produced a clean-looking GO on this exact comparison
(`credit_on_expanded_6seed.json`: raw 0.277767 → expanded 0.586433, 6/6 seeds, shuffle 0.2901, lesion and
no-credit both at the majority floor, memorization leak 0.0000). That result is real and it is a
**forward-representability** result — its learner is a one-layer softmax. Read as evidence about credit, it
would have banked "representability was the blocker". It is not.

## Result (6 seeds; `research/findings/raw/gap4/AGG_deep_credit_on_expanded_6seed.json`, aggregated from the six per-seed artifacts in `research/findings/raw/gap4/deepcredit_par/`)

Stage 0 — is the task still DEPTH-REQUIRED on this forward?

| forward | linear | 1-hidden | 2-hidden | depth-separating |
|---|---|---|---|---|
| raw (9 features) | 0.1481 | 0.2438 | **0.9599** | **6 / 6** |
| plateau codon, thresholded (200) | 0.6327 | 0.6883 | 0.6914 | **0 / 6** |
| plateau codon, graded (200) | 0.6265 | 0.6358 | 0.6914 | **1 / 6** |

Stage 1 — the deep-credit arms:

| forward | microcircuit | KP-learned | plain-FA | 1-hidden floor | permuted | apical lesion |
|---|---|---|---|---|---|---|
| raw | 0.6944 | 0.6574 | 0.6944 | 0.2840 | 0.1451 | 0.1173 |
| thresholded | 0.6173 | 0.5617 | 0.6173 | 0.6389 | 0.1481 | 0.2315 |
| graded | 0.6265 | 0.6173 | 0.6265 | 0.6296 | 0.1173 | 0.2870 |
| expander lesion | 0.1204 | 0.1296 | 0.1204 | — | — | — |

Majority floor 0.1667. Codon reproducibility 1.0000. The lesion arm at 0.1204 confirms the expansion carries
real information, so this is not "a wider input layer".

## The reading

**The expansion trades the ceiling for the floor.** It raises linear decodability 0.1481 → 0.6327, a 4.3×
lift that replicates the banked 07-25 surpass. In the same move it collapses the 2-hidden ceiling
0.9599 → 0.6914 (best over depths 0.7006), and every depth then lands at ~0.63–0.70. The 07-25 result measured the floor rising.
**Nobody measured the ceiling.**

Consequently the deep arms cannot be read there at all: on both expanded forwards they sit at or BELOW their
own 1-hidden floor (0.6173 vs 0.6389; 0.6265 vs 0.6296). Depth stops paying. A deep net succeeding on a task
that has gone shallow is not a deep-credit result, which is why 5 seeds return UNDEFINED rather than a
negative, and the one seed whose graded forward stayed depth-separating returned an honest NO-GO.

**It is the expansion, not the binarization — and that kills the cheap fix, which was mine.** I added the
graded arm expecting the threshold to be the culprit: `codon = (v_apical > FLOOR)` replaces a graded
biological voltage with a constant, which is exactly the "what did we swap for a constant?" reframe. It is
not the answer. Removing the threshold changes essentially nothing — linear 0.6265 vs 0.6327, 2-hidden 0.6914
vs 0.6914 (identical), depth-separating 1/6 vs 0/6. The information the depth requirement lives in is destroyed by the
random-projection-and-pool structure itself, not by reading it as bits.

## Scope, stated rather than implied

- The learner here is the **numpy** microcircuit/BDSP stack, not the on-bridge spiking substrate. That the
  deep arms reach 0.6944 on the raw forward against a 0.2840 1-hidden floor is the ALREADY-KNOWN smooth-rate
  result, not a new claim about spikes, and it does not touch the banked on-spikes NO-GO.
- This says nothing about whether the expander is useful for its original purpose. As a device for making a
  representation linearly decodable it does what it claims, at 6 seeds. The claim being corrected is the
  inference that it therefore gives deep credit "features to shape".

## Consequence

The pre-registered KILL CRITERION fires: **representability was not what blocked credit.** More forward work
on this expander is not the lever. The next lever is the graded-state escape — lever (b) of the 2026-07-24
root-cause finding — and the `dendritic-plateau-coincidence-burst` mechanism entry needs its `current_status`
scoped to linear decodability rather than left implying a credit unlock.

The general lesson is the session's own: a surpass measured with one instrument is a claim about that
instrument. The 07-25 result raised a floor and was read as raising a capability; the ceiling was never
measured until an experiment was built that had to measure both.
