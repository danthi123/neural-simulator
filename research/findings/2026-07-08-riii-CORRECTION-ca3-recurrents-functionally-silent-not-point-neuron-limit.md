# R-iii CORRECTION (retracts the CYCLE-1062 diagnosis): the CA3 completion failure is FUNCTIONALLY-SILENT recurrents (a transmission/scaling issue), NOT a point-neuron linear-summation limit. Root-cause debugging (weight-invariance → Vm-invariance → direct transmission test) shows the ca3→ca3 recurrent synapses are wired (weight 5→120 in cp_connections) but deliver ~0.2 mV to postsynaptic neurons (~1000× too weak). So I never tested a working attractor; the dendritic-mechanism conclusion was PREMATURE. The D.13-was-drive-artifact finding STANDS and is now EXPLAINED (silent recurrents). NO `sim/` edit yet.

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_ca3_completion_specificity_derisk.py` + inline diagnostics. numpy-CPU. NO `sim/` edit.
**Verdict:** CORRECTION — CYCLE 1062's "point-neuron completion boundary" diagnosis is retracted; the cause is silent recurrents, to be fixed and re-tested before any boundary claim.

## What CYCLE 1062 got right vs wrong
- **RIGHT (stands, strengthened):** genuine held-out pattern completion = 0 on this substrate, and the "validated D.13 cos 0.748 PASS" was the DRIVE ARTIFACT (own_cos≈0.5 floor from the driven partial overlapping the full set). This is now EXPLAINED: the recurrents are silent, so D.13 could only ever measure the drive overlap.
- **WRONG (retracted):** attributing the no-completion to a "point-neuron linear-summation limit" and pointing to the dendritic NMDA-plateau mechanism. That diagnosis was premature — the recurrents never transmitted, so I never tested whether the point-neuron CA3 CAN complete with working recurrents.

## The root-cause debugging chain (the discipline that caught it)
1. Density sweep (0.3/0.6/0.9): held-out completion byte-identical (0.000). Suspicious. Verified the param works: ca3→ca3 synapses 6654→20278 (3×). So 3× more synapses = zero effect.
2. Weight sweep (5/15/40, then genuine 20/60/120 with the cap auto-raised): byte-identical again — own_cos=0.493, n_stored=15, held-out=0 to 3 decimals across a 24× weight range. Impossible if the recurrents transmit anything.
3. Verified the weight IS applied: mean |ca3→ca3 w| = 4.997 (w=5) vs 119.928 (w=120); no cp_transmission_gain array.
4. **Vm check (decisive):** held-out neurons' membrane potential during partial-cue recall = **-65.39 mV (rest), IDENTICAL at weight 5 and 120**. A 24× stronger recurrent weight delivers ZERO depolarization.
5. Gate check: ca3_swr_burst open vs closed during recall → -65.39 vs -65.98 mV (no difference) — not the gate.
6. **Direct transmission test:** drove 8 ca3→ca3 presynaptic neurons (18 spikes) → their 30 postsynaptic targets reached only **-64.80 mV (~0.2 mV depolarization)**. The recurrent synapses deliver ~1000× too little current to matter, while feedforward-driven CA3 fires fine from direct 200pA current.

⇒ the ca3→ca3 recurrent pathway (a RegionPathway from CA3 to itself; the CA3 region has internal_density=0.0 by design, so this pathway IS the recurrence) is functionally silent — its synaptic current is negligible at any weight/density tested.

## What this means (honest, corrected)
- CA3 pattern completion has apparently NEVER worked in this project — every "D.13 pattern completion" result was the drive artifact (consistent with its seed-variability 0.748/0.676/0.679).
- The R-iii SWR generative-replay loop was at chance (2026-05-24) because the CA3 autoassociator it depends on has no functional recurrence — a MECHANISM/IMPLEMENTATION gap, not a proven point-neuron limit.
- Whether point-neurons CAN do completion (vs needing the dendritic plateau) is UNDECIDED until the recurrent transmission is fixed and re-tested.

## The next concrete step (the real R-iii enabler)
Root-cause + fix the ca3→ca3 recurrent CURRENT delivery (sim/-internals): why does a weight-120 recurrent synapse deliver ~0.2 mV when a direct 200pA current fires the neuron? Candidates to investigate (read the sim/ code): the effective-synaptic-strength scaling for within-region recurrent RegionPathways; the conductance-decay time constant vs the recurrent input rate; the CSR matvec orientation for self-pathways; whether the recurrent synapses are included in the per-step conductance update at all. Then re-run the (now-clean, adversarially-instrumented) held-out completion probe with WORKING recurrents — and only THEN decide whether the point-neuron substrate completes or needs the dendritic plateau (Kandel Ch 13, read in depth this session).

## Files
`research/runners/_riii_ca3_completion_specificity_derisk.py` (density/weight/Vm/transmission diagnostics). Corrects: `2026-07-08-riii-point-neuron-CA3-completion-boundary-adversarially-verified.md` (CYCLE 1062). The D.13 drive-artifact point is retained; the point-neuron-limit diagnosis is retracted.
