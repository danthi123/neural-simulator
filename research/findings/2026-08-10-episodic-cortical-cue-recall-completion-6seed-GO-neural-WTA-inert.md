---
type: finding
status: contributing
date: 2026-08-10
mechanism: episodic-cortical-cue-recall
lane: EPISODIC
---

# Episodic neural cortical cue-recall GENERALIZES (6-seed GO): completion-driven, specific, neural — recall 0.65 vs 0.25 chance; the neural WTA competition is INERT (the heteroassociative afferent selects directly)

**The completion sub-wall closes at 6-seed generalization** (was SMOKE seed-42 only, `_completion_loadbearing_SMOKE_s42.json`).
Cue ONE attribute of a stored episode → the CLOSED bistable+specific CA3 attractor completes the assembly → the
CA3→cortex heteroassociative pathway reactivates the bound held-out attributes → a NEURAL cortical readout (which
cells fire), no host argmax. Runner `research/runners/_riii_ca3_cortical_episodic_wta_derisk.py`, cupy/RTX3090,
`cfg.seed` byte-identical substrate (build-twice threshold hash IDENTICAL), NO `sim/` edit.

## Result — 6-seed GO with the full teeth panel (mean over seeds 42/43/44/100/101/102)

<!--derived-->

- **end-to-end neural cortical recall = 0.646 vs chance 0.25** (2.6× chance; readout IGNITES, max cortical rate
  > 0) — GO.
- **completion LOAD-BEARING:** full 0.646 vs recurrent-zero 0.25 (+0.396) — zeroing the ca3→ca3 recurrents
  collapses the held-out readout to chance, so recall is genuine attractor COMPLETION, not feedforward re-drive.
- **permuted-cue SPECIFIC:** full 0.646 vs permuted-cue 0.083 (+0.563) — a wrong cue does not complete.
- **real-vs-sham lesion has TEETH (non-tautological):** real (ablate CA3→cortex heteroassoc) drops recall to 0.25;
  sham (equal-size unrelated ablation) preserves it (~full) — the lesion hits the readout PATHWAY, not the metric.
- **untrained-CA3 fails:** full 0.646 vs untrained 0.25 (engram load-bearing).
- OU off + plasticity frozen at recall (kills the self-sustaining + noise confounds). All conditions GO.

## The neural WTA competition is INERT (confirms the 2026-08-10 reframe)

<!--derived-->

full-sep 0.0086 vs wta-off-sep 0.0088 → the lateral-inhibition WTA changes the readout separation by −0.0002, i.e.
it is NOT load-bearing: opening the WTA gate leaves the result unchanged. The recall works because the
heteroassociative afferent selects the winner DIRECTLY (which is exactly the reframe
`2026-08-10-neural-WTA-separable-assemblies-weight-controllable-*`: a separable-assembly WTA has no common-mode; the
afferent decides, the competition is superfluous). So the "neural WTA sub-wall" is not a blocker for recall — it is
an unused circuit at this operating point. **The operating-point retune CONFIRMS this: `wta_ie_w` 18→4 (weaker
latch) leaves recall unchanged (0.625 @ wie=4, 3-seed, vs 0.646 @ wie=18; WTA still inert full-sep≈wta-off-sep), so
the WTA latch strength is irrelevant** — the recall residual is NOT the WTA operating point.

## Honest residuals

<!--derived-->

- **Recall is 0.65, not ceiling.** CA3 completion magnitude is 0.18 (moderate) — the upstream attractor drive into
  the heteroassociative pathway is weak (the gap#5 completion is only ~0.2-0.33), so the cortical readout inherits a
  soft ceiling. The residual 0.35 is readout specificity, not a broken mechanism.
- **The CA3 assembly is PRE-ASSIGNED** (a fixed sparse mask inherited from the closed completion) — this finding
  tests the READOUT given a working attractor; it does NOT claim emergent DG/mossy episode SELECTION (a separate
  downstream piece).
- 3-way attribute recall (WHO/WHAT/WHEN) on disjoint cortical pools; the co-resident (source-monitor) case is out of
  scope (dendritic, mapped).

Artifacts: `research/findings/raw/cortical_episodic_wta/_completion_loadbearing_6seed.json` (verdict GO, all
preconditions/controls pass); `research/findings/raw/cortical_episodic_wta/_retune_wie4_3seed.json` (the WTA-latch
retune, recall 0.625 unchanged). SIM_BACKEND=cupy.
