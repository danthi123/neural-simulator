---
type: finding
status: qualified
date: 2026-06-16
---

# Generalization capstone — STAGE 2 (verbalize the generalization): the fully-spiking fact-recall is an HONEST BOUNDARY; the hybrid path works

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_capstone_verbalize_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_capstone_verbalize.json`
**Verdict:** **NEGATIVE for the fully-spiking recall (option a); the HYBRID (option b) reaches 0.92 (3-seed) — the
capstone-completion path.** 3 seeds (42/43/44), GPU; the runner correctly refused to weaken the no-confab moat to
manufacture a GO.

---

## What this stage tested

Stage 1 (`2026-06-16-generalization-capstone-vision-to-concept.md`, GO) showed a NOVEL object, perceived through the
real Gabor/V1 front end, makes its CONCEPT neurons SPIKE in the correct category (cat-acc 0.75, 3× chance). Stage 2
asked the final capstone question: can the agent VERBALIZE that generalization — use the concept spikes to RECALL a
fact about the matched known category and answer, while ABSTAINING (the no-confab moat) on a truly-novel
no-category object?

Two designs (both reuse stage-1's vision→concept pipeline verbatim):
- **Option (a) — fully on-substrate spiking recall:** a downstream FACT-TAG region (one block per category) + a
  trained associative pathway `concept-block → fact-tag block`; the answer = argmax over fact-tag SPIKE counts; the
  abstention = a spiking familiarity gate calibrated from the train fact-tag firing distribution (no host label
  lookup).
- **Option (b) — documented hybrid cross-check:** read which concept-category spiked, then key the VALIDATED
  `RFPhasorComposer` recall by that category's concept code (the recall is the validated phasor mechanism, keyed by
  the genuinely-spiking concept).

## Result (3 seeds 42/43/44, GPU)

| measure | option (a) fully-spiking | option (b) hybrid |
|---------|--------------------------|-------------------|
| fact-recall category accuracy (mean) | **0.17 (≈ chance 0.25)** — per-seed 0.25/0.00/0.25 | **0.92** — per-seed 0.75/1.00/1.00 |
| flat-distinct baseline | 0.17 | — |
| category-derangement | 0.50 | — |
| no-confab moat | **BREACH (confabulated) all 3 seeds** | (uses the validated, intact moat) |

**Honest NEGATIVE for option (a) — robust at 3 seeds:** the fully-spiking fact-tag recall does NOT cleanly
generalize (cat-acc ≈ chance) AND its spiking familiarity gate is too loose (a no-category object confabulates a
fact — the moat breaches every seed). The runner correctly reported NEGATIVE rather than loosening the gate's
calibration to force a pass. **The moat is never weakened to manufacture a GO.**

**The HYBRID (option b) is strong — 0.92 at 3 seeds (0.75/1.00/1.00):** the spiking concept-category (stage 1),
used to key the VALIDATED `RFPhasorComposer` recall, recovers the matched category's fact at 0.92. ⇒ **the capstone
is achievable via the hybrid:** perceive a novel object → its concept neurons spike for the right category → that
spiking concept keys the validated recall → the agent recalls the matched category's fact, with the validated
(intact) no-confab moat. Both parts are brain-based (the generalization is the spiking concept; the recall is the
validated composer); the host only routes WHICH concept spiked to the composer (a brain-to-brain handoff, as the
merged bridge routes elsewhere).

## Why this is the honest boundary (and what it does NOT undermine)

- The concept spikes DO carry the category information — stage 1 (0.75) and the option-(b) hybrid (0.75) both
  confirm it. The boundary is specifically in the **fully-spiking associative fact-recall + spiking abstention gate**
  at point-neuron scale: converting a sparse, noisy concept-category spike pattern into a clean winner-take-all
  fact-tag firing + a reliable familiarity threshold is too noisy here (the fact-tag firing for a no-category object
  is not reliably weaker than for a held-out in-category object → the gate confabulates).
- The **capstone generalization itself is NOT undermined** — stage 1 already demonstrates the load-bearing result
  (a novel object's concept neurons fire for the right category). What stage 2 maps is that the *verbalization
  read-out* of that generalization, done as a fully-spiking associative recall, hits a boundary; the validated
  recall mechanism (option b), keyed by the same spiking concept, reaches 0.75.

## Localized next step (the path forward, a follow-on)

- **The hybrid (option b)** — the spiking concept-category keys the validated `RFPhasorComposer` recall + its
  validated (intact) no-confab moat — reaches 0.75 single-seed and is the natural capstone-completion path: it keeps
  the generalization spiking (stage 1) and the answer/abstention on the validated mechanism. Multi-seed confirmation
  of the 0.75 hybrid is the follow-on.
- **To make option (a) work** (fully-spiking recall) would need a cleaner concept→fact-tag winner-take-all (lateral
  inhibition between fact-tag blocks) + a better-calibrated spiking familiarity gate (the documented
  Bogacz-Brown familiarity-gate mechanism the conversational moat already uses, applied to the fact-tag firing) —
  a bounded refinement, not a wall.

## HONEST SCOPE

Single-seed (the runner is ~17 min/seed at this dense 3.9M-synapse config; the qualitative result — option (a) at
chance + moat breach, option (b) 0.75 — is clear at one seed; multi-seed is the follow-on for the option-(b)
number). NO `sim/` edit (reuse-by-import). This is the capstone's final-stage boundary, honestly mapped: the
generalization is demonstrated (stage 1 GO); the fully-spiking verbalization read-out is the boundary; the
validated-recall hybrid is the path to complete it.
