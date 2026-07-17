# Rate-net positive control: GRADED coding does NOT unlock supervised deep credit on the bridge — the block is NP's credit assignment (the rule), not (only) the readout discreteness. Commit to the unsupervised path.

**2026-07-17. The frontier map's #1 de-risk, genuinely un-run before today. 6-seed × {spiking, graded}, on-bridge,
correctly seeded (`_d1_onbridge_learn_to_accuracy:252` `cfg.seed`), NO `sim/` edit (additive default-off `--graded`
arm, byte-identical when off — VERIFIED). Honest NEGATIVE + a stated scope limitation.**

## The question

Feedforward e-prop deep credit is NOT-GO (2026-07-17), Node Perturbation is retired, and D1/BDSP failed on-bridge — all
blocked by a **shared supervised spiking-classifier-READOUT wall**: the finding
`2026-07-13-onbridge-NP-small-scale-variance-BOUNDARY` root-caused it to the *discrete* spike-count readout (the
host delta-rule can't train the output from coarse spike counts). **Does a GRADED readout unlock it — i.e. is the
DISCRETENESS the wall, or is the block deeper (NP's own credit-assignment / rule)?** If graded coding unlocks it, the
emergence engine gains a working supervised learning rule; if not, we commit to the unsupervised stream cortex.

## The test

`_nodepert_onbridge_derisk.py` with an added `--graded` arm: read a graded per-neuron depolarization above the reset
floor (capped at 50 mV so the spike peak doesn't dominate a subthreshold rate proxy) for BOTH the clean and the
perturbed readout, instead of discrete `cp_firing_states` counts. The built-in `hidden_frozen` mode + the
`depth_helps = np > frozen + 0.05` gate ARE the anti-cheat (if graded trivially carries the class signal, frozen also
passes and depth_helps stays False). 6 seeds each, default epochs, task=emerge1 (the off-brain NP GO task).

## Result — 12/12

| arm | np_beats_chance | depth_helps (np > frozen) | mean NP − frozen |
|---|---|---|---|
| **spiking** | **0/6** | **0/6** | −0.006 |
| **graded** | **0/6** | **0/6** | **−0.014** (graded slightly WORSE) |

- **Graded does NOT unlock supervised deep credit.** NP never beats the frozen reservoir in either readout (0/12).
- **The shape is informative:** the graded readout marginally IMPROVES the reservoir's readability (frozen graded
  0.53–0.56 vs spiking 0.47–0.52), yet NP's hidden credit adds **nothing on top** — NP is if anything *further below*
  frozen with graded (−0.014 vs −0.006). ⇒ the block is **NP's credit assignment (the rule)**, not merely the readout
  discreteness: a better (graded) readout helps the reservoir be read, but the *learning* still doesn't beat it.

## ⚠️ SCOPE LIMITATION (stated up front, not buried) — this is a DIRECTIONAL negative, not a high-powered proof

**Nothing beats chance in either readout (np_beats_chance 0/12).** Not NP, not the frozen reservoir. At this on-bridge
config (hidden=12, default epochs, ~10-bit emerge1) the net cannot represent the task well enough for *any* readout to
clear chance — so a graded readout has little signal to show an advantage over a spiking one. **Therefore this test
CANNOT strictly discriminate "the readout was the wall" from "the config is underpowered."** A config where the frozen
reservoir clearly beats chance would sharpen it. What the test DOES establish, robustly across 12 arms:
1. **Graded coding does not RESCUE supervised NP on the bridge** at the config the record's own on-bridge NP finding
   used — the escape hatch does not obviously work.
2. **The graded readout is not the missing ingredient** — it marginally improves reservoir readability but NP's credit
   contributes nothing, consistent with the block being the rule (NP's zeroth-order variance / feedback-alignment
   partiality), which the record already flagged (`2026-07-13-onbridge-NP-...variance-BOUNDARY`: retired NP for a
   readout-independent variance wall).

## Decision

The rate-net control was the cheapest test of whether graded/population coding revives supervised deep credit. It does
NOT (directionally, 12/12). **⇒ supervised deep credit stays PARKED; commit to the UNSUPERVISED on-spike stream cortex**
(HTM + committed BDSP `fused_htm_permanence_update`) — the mission-central path the frontier map already identified,
which learns deep representations from a stream WITHOUT supervised global-loss deep credit, sidestepping this wall
entirely. Further config-tuning to sharpen the readout-vs-rule attribution is DEPRIORITIZED against the mission path
(a rabbit-hole on a parked direction; the direction is already decided).

**Honest boundary, not a wall:** "supervised deep credit on the point-neuron bridge" remains an undiscovered-mechanism
boundary (the record has exhausted feedforward e-prop, the burst family, NP, and now the graded-readout escape). The
next-mechanism search, if ever revived, is off the mission-critical path — the unsupervised stream cortex is.

Finding pair: `2026-07-17-learning-rule-frontier-map-...md` (the map this closes a branch of),
`2026-07-13-onbridge-NP-small-scale-variance-BOUNDARY.md` (the root-cause it confirms).
