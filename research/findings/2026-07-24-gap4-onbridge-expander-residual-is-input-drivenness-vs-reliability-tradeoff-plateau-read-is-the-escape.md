# gap#4 on-bridge spiking expander: the residual is a precise INPUT-DRIVENNESS ↔ RELIABILITY tradeoff on the noisy substrate — the named escape is EMERGE-35's coincidence dendritic-plateau read (reliable + input-driven at once) (2026-07-24)

## Context
The numpy expansion GO (`2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-numpy-GO...`) surpassed the
gap#4 forward-representability boundary at the mechanism level (a fixed nonlinear expansion lifts held-out LINEAR
decodability 0.284 → 0.772). It named the on-bridge requirement: the expanded spiking columns must be INPUT-DRIVEN (lever-g
failed because they were tonic-pinned) AND RELIABLE (lever-h overfit because the raw code is noisy). This tests that on-bridge.

## Test (`scratchpad/gap4_onbridge_expander.py`, Gap4OnBridgeNet hidden=200 = EXPANSION, 3 seeds, held-out probe)
Sweep the operating point from the tonic-pinned default toward input-driven (low tonic + high input gain), read the pooled
200-column event code held-out, + the research-gate fingerprint (input_cv, same-input reproducibility, active fraction).

| operating point | ho-LIN | input_cv | reproducibility | active_frac |
|---|---|---|---|---|
| default (tonic-pinned) | 0.309 | 0.228 | 0.505 | 0.997 |
| input-driven-A | 0.333 | 0.701 | 0.089 | 0.534 |
| input-driven-B | 0.333 | 0.966 | 0.068 | 0.289 |
| input-driven-C (strong w) | 0.333 | 0.545 | 0.145 | 0.744 |

## The decisive finding: an INPUT-DRIVENNESS ↔ RELIABILITY tradeoff
input_cv (how much columns vary ACROSS inputs) and reproducibility (how much the SAME input reproduces its code) are
**anti-correlated**: pushing the operating point input-driven (input_cv 0.23 → 0.97) COLLAPSES reproducibility
(0.505 → 0.068), and **every** point gives degenerate held-out (~0.33, ≈ the majority-class floor). The sensitive
operating point that makes columns respond to input ALSO makes them respond to noise (OU/Poisson/carried-over membrane
state) — the input signal is swamped by trial variability, so the code is not a reliable function of the input.
- The **numpy random-ReLU expander has NO such tradeoff** (deterministic → input-driven AND reliable simultaneously → 0.772).
  The tradeoff is a property of the NOISY SPIKING substrate, not the expansion idea.
- **Honest caveat:** the reproducibility metric re-runs the same inputs without a full bridge-state reset, so it conflates
  trial noise with OU/membrane carryover; the load-bearing result (input_cv↑ ⟹ reproducibility↓, all points degenerate
  held-out) holds regardless, but the absolute reproducibility floor is an upper bound on the noise.

## Verdict (per THE LAW — the residual is named + the escape is named; NOT a wall)
- **The on-bridge residual is a spiking-code RELIABILITY problem at an input-driven operating point** — precisely, not a
  vague "point neurons can't." Expansion (hidden=200) and input-drivenness (input_cv 0.97) are both ACHIEVABLE; what is
  missing is a code that is reliable WHILE input-driven.
- **NAMED ESCAPE (validated elsewhere on-substrate): EMERGE-35's coincidence dendritic-PLATEAU read.** EMERGE-35 reaches
  held-out inheritance 1.00 on-bridge precisely because it does NOT read the noisy event rate — it reads `cp_v_apical >
  FLOOR`, a reliable dendritic-plateau threshold-crossing that is input-driven by construction (coincidence_weighted_drive
  fires a column's plateau reliably when its sampled input features coincide). The bistable-ish plateau DECOUPLES
  input-drivenness from rate noise — exactly the tradeoff this finding isolates. `enable_two_compartment_dap`,
  `coincidence_weighted_drive`, `coincidence_plateau_strength` are the committed knobs.
- **NEXT ACTION:** build the on-bridge expander with EMERGE-35-style coincidence-plateau columns (reliable + input-driven),
  adapted for the continuous 7-dim input (binarize by per-input top-k, or a graded-coincidence variant), read the plateau
  codon, run the held-out probe. GO iff held-out LINEAR rises off 0.34 toward the numpy ceiling on ≥5/6 seeds WITH high
  reproducibility (>~0.8). HONEST BOUND: the numpy CODON on this 7-dim continuous input capped at 0.617 (< random-ReLU
  0.772) — so a reliable coincidence-plateau expander is expected to surpass the boundary (0.34) but may need a
  graded/continuous reliable expander to reach the full 0.772; reliability, not expansion, is the frontier.

## Provenance
`scratchpad/gap4_onbridge_expander.py`; reuses Gap4OnBridgeNet + `_gap4_representability_probe`. Builds on the numpy
expansion GO + the research gate `wf_1f9812d7-0eb`. EMERGE-35 escape: `2026-07-02-emerge35-spiking-pooler-GO.md` +
`research/runners/_emerge35_spiking_pooler_derisk.py`. NO `sim/` edit.
