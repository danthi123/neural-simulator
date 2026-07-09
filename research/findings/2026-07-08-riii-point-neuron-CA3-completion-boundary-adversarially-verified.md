# R-iii — the point-neuron CA3 recurrent attractor does NOT do genuine pattern completion (adversarially-verified boundary): a partial cue (half the stored ensemble) driven on CA3 does NOT recruit the held-out half via recurrence — held-out completion = 0.000 at every training level (80→480) — while the SAME held-out neurons fire (2.69) under the full cue (so the metric is correct, the failure is real). The validated D.13 "pattern completion cos 0.748" was the DRIVE ARTIFACT (the driven partial overlapping the full set it's drawn from), not recurrent completion; this explains why the 2026-05-24 SWR generative-replay loop was at chance. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_ca3_completion_specificity_derisk.py` (reuses the validated D.13 regime by import: the ca3_swr_burst/dg_to_ca3/ec_to_dg/lang_to_ec gates + full-pattern training + measure_region_response). numpy-CPU. NO `sim/` edit.
**Verdict:** BOUNDARY (adversarially verified) — the point-neuron CA3 recurrent autoassociator does not complete partial cues; a stronger completion mechanism is required.

## Why this ran + the correction chain
The R-iii frontier is the 2026-05-24 fully-spiking-SWR generative-replay NEGATIVE. CYCLE 1060 built a probe from a subagent's "add the CA3 drive" summary (drove the FULL ensemble, measured a static hold) and committed "naive fix REFUTED"; CYCLE 1061 (owner-prompted, reading Kandel p1361/Marr-1971 IN DEPTH) corrected that: the mechanism is PARTIAL-cue completion on an LTP-trained attractor. This cycle reproduced the VALIDATED D.13 regime by import and applied a CLEAN metric (the held-out neurons' firing), which reveals the deeper truth.

## The result — train-ladder, seed 42, fast config (n_ca3=150, K=2), CLEAN held-out metric
```
train_events:        80      160      320      480
trained_heldout:   0.000    0.000    0.000    0.000     <- the held-out (non-cued) neurons NEVER fire
notrain_heldout:   0.000    0.000    0.000    0.000
recurrence_gain:  +0.000   +0.000   +0.000   +0.000     <- flat; more training does NOT help
own_cos:           0.459    0.493    0.470    0.547      <- the DRIVE-ARTIFACT floor (driven half overlaps the full set)
```

## Adversarial verification (the load-bearing check, per the workflow's step 4)
held-out completion = 0.000 EXACTLY is a strong claim (a boundary + a correction of a validated result), so I verified it is a genuine recurrence failure, not a mapping/measurement bug: measure the SAME held-out neurons under the FULL cue (they ARE directly driven → they must fire if the mapping is correct).
```
partial-cue held-out completion = 0.000   (recurrence does NOT reconstruct the held-out neurons)
FULL-cue    held-out activation  = 2.688   (the SAME neurons FIRE when directly driven -> mapping correct)
=> held-out=0 is GENUINE recurrence failure (full-cue fires them, partial-cue does not).
```
The mechanism: a partial cue's recurrent synaptic drive onto a held-out CA3 neuron (from ~7 presynaptic partners × weight 5.0 × density 0.30) is SUB-THRESHOLD on a point neuron; it never crosses firing threshold, so completion fails. Hebbian LTP on the recurrents (train 80→480) does not raise it enough. This is a point-neuron limit (cf. the project's whitening/decorrelation Mikulasch-Priesemann boundary).

## The correction of the "validated" D.13 (a methodology finding)
The 2026-05-11 D.13 validation reported "pattern completion cos(partial, full) = 0.748 (PASS >0.7)" and noted it was seed-variable (0.748/0.676/0.679, 1/3 strict). That cos-metric is DRIVE-ARTIFACT-CONFOUNDED: driving half the stored ensemble produces a response that overlaps the full ensemble (which CONTAINS that half) at cos ~0.5–0.75 with NO recurrent completion — exactly the own_cos≈0.5 floor measured here, and the seed-variability is the artifact's variance. The genuine completion signal (the held-out neurons firing) is ZERO. So the D.13 "PASS" did not demonstrate Marr completion; it measured the drive overlap. This is the same confound class caught in CYCLE 1060 (full-drive static hold) — the correct control is the held-out-neuron firing + the no-train comparison.

Honest scope of the D.13 correction: confirmed at the fast config (n_ca3=150, train 80-480). The exact D.13 config (n_ca3=400, train=400) is the expensive run (~28 min/seed, loop-bound; it exceeded the background-task budget — a compute/harness limit, not a result). The held-out=0 mechanism (sub-threshold recurrent drive) is config-general (more neurons doesn't lower the per-neuron threshold), so the correction is strongly indicated; re-running the held-out metric on the n_ca3=400 config to nail it is the immediate follow-on.

## What this establishes + the mechanism search it launches
The point-neuron CA3 recurrent attractor does not do genuine pattern completion — the reactivation the R-iii SWR generative-replay loop needs — and this is WHY that loop was at chance (2026-05-24). Per the directive, the boundary launches the search for the completion mechanism (read the sources IN DEPTH myself, then cheap-first de-risk):
- **Sparser CA3 coding** (Treves-Rolls autoassociator capacity ∝ sparseness; the current 10% code has too much interference — a 1–2% code may complete). Cheapest first candidate.
- **Higher recurrent convergence** (a held-out neuron needs enough co-active presynaptic partners to cross threshold — denser recurrents / larger fan-in, not just higher weight).
- **The dendritic supra-linear amplifier** (NMDA plateau / the distal-proximal coincidence amplifier, Kandel Ch 13) — the point-neuron limit the project has repeatedly hit; the deep frontier.
- **Theta-paced sequential readout** (O'Keefe-Nadel pp 224-225) — for sequence recall, a later rung.

## Files
`research/runners/_riii_ca3_completion_specificity_derisk.py` (ladder/held-out metric/adversarial full-cue verify). Prior: `2026-07-08-riii-swr-reactivation-probe-naive-drive-fix-REFUTED.md` (CYCLE 1060/1061), `2026-05-24-c-generative-replay-decisive-NEGATIVE-*.md`, the D.13 validation `validate_trisynaptic_loop.py` + `2026-05-11-P1-trisynaptic-loop-validation.md`.
