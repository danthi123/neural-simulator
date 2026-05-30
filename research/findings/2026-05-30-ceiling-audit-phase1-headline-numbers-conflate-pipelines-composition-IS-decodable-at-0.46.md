# Conversational-ceiling audit, Phase 1 (measurement-consistency, no GPU): the headline "~0.46 / 0.00 representation ceiling" CONFLATES two non-comparable readout pipelines. Composition IS decodably represented at ~0.46 trustworthy gated-emission (6th-arc cosine pipeline) -- the "composition not a structured decodable object" framing that motivates phase-coded VSA is OVERSTATED. The SPEAR-gate-scale-mismatch-bug hypothesis is FALSIFIED. The real open question is the 0.46->0.80 gap (representational vs readout-limited), which Phase 2 settles.

**Date:** 2026-05-30
**Status:** Phase 1 of the owner-chosen "audit the ceiling" direction (cheap, read-only measurement-consistency pass). INTERIM finding -- Phase 2 (latent-composition decode probe) is the decisive representation-vs-readout test and is pending. Honest scope: this REFRAMES the ceiling and corrects a framing conflation; it does NOT by itself dissolve the ceiling (which is real at ~0.46 trustworthy compositional emission).

## Why this audit

The next-arc survey found an 8+ arc conversational sequence converging on a "~0.46 ceiling" (6th-arc local optimum) / "0.00 above moat" (SPEAR), framed as a REPRESENTATION limit (composed readout needs a structured decodable object; dynamics-gating can't make one) prescribing phase-coded vector-symbolic composition (Orchard spiking-phasor FHRR) as the next big arc. The owner chose to AUDIT that ceiling first (bug-discovery-first on chance results; scrutinize the negative before building on it) rather than commit to the big arc on an unverified premise. Pre-registered (before reading the code): ARTIFACT if (a) the headline metrics conflate raw-ranking vs gated-emission, or (b) Phase 2's held-out decoder hits >= 2x chance on composed states where the spiking readout is at chance; CEILING-CONFIRMED if metrics consistent AND no decoder beats chance; three-state.

## Phase 1 findings (measurement-consistency, code-read only)

1. **SPEAR full_acc = 0.00 is gated on RAW FIRING RATE at the 650 moat.** `spear_conversational_runner.py:515-528`: the readout SUMS per-concept raw firing-rate confidences across the consolidated + hippocampal regimes and gates the sum at `_MOAT = DEFAULT_THRESHOLD = 650.0` (abstention_gate.py). The design requires the composed SUM to exceed 650 ("neither regime alone clears the moat; only the composed sum does"). 0.00 = the composed firing-rate sum rarely reaches 650.

2. **The SPEAR-gate-scale-mismatch-bug hypothesis is FALSIFIED.** I checked whether SPEAR ranks cosine-scale values (0-1) and gates them at 650 (which would make the full arm always abstain -> trivial 0.00). It does NOT: the SPEAR readout is explicitly raw-firing-rate scale, correctly matched to the 650 firing-rate moat. SPEAR's 0.00 is a legitimate firing-rate-pipeline measurement, not a units bug. (Scrutinized the artifact hypothesis as hard as the ceiling claim; it did not hold here.)

3. **But the headline numbers conflate two non-comparable pipelines.** The 6th/8th-arc `full_acc` (0.458 / 0.315) is a lang_output COSINE readout gated at `COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977` (a cosine value), per `pool_readout_8th_arc_runner.py:694-719` + `abstention_gate_compositional_unified.py:75`. That is a COMPLETELY DIFFERENT readout quantity and threshold scale from SPEAR's firing-rate-sum @ 650. So "8 arcs + SPEAR converge on a ~0.46 / 0.00 ceiling" puts two different measurements on one axis. They are not directly comparable.

4. **Composition IS decodably represented at ~0.46 trustworthy gated-emission.** The 6th-arc 0.458 is gated-emission accuracy through a CALIBRATED trustworthy cosine gate (margins 0.064-0.118 at the calibrated threshold) -- not raw ranking, not chance. So composition reaches ~46% correct TRUSTWORTHY emission in the cosine pipeline. The framing that motivates phase-coded VSA -- "composition is not a structured decodable object; the rhythm must be made to carry it before it is decodable at all" -- is OVERSTATED. Composition is already decodable at ~0.46.

## What this does and does NOT establish

- DOES: corrects a real measurement-framing conflation (different readout pipelines/thresholds on one "ceiling" axis); reframes the ceiling from "0.00 / not represented" to "~0.46 trustworthy compositional emission, capped below 0.80"; falsifies the SPEAR units-bug hypothesis.
- Honest caveat against my own pre-registration: the LITERAL pre-registered artifact (a) was "raw-ranking vs gated-emission." That specific thing is NOT what I found -- BOTH pipelines are gated. What I found is the related-but-distinct "two different gated pipelines on different quantities/thresholds" conflation. Under strict pre-registration I do NOT claim artifact-(a); I report the conflation + ceiling-reframe as the honest Phase 1 result.
- Does NOT dissolve the ceiling: ~0.46 trustworthy compositional emission is a REAL cap (cannot reach the 0.80 bar). The open question is whether the 0.46->0.80 gap is representational (composition only carries ~46% recoverable signal) or a readout limit (a richer decoder recovers more from the same composed state).

## Phase 2 (decisive, pending owner nod): latent-composition decode probe

On an existing validated bridge, generate composed-query states and test whether a HELD-OUT linear / nearest-neighbour decoder recovers the composed answer ABOVE the 0.46 cap (and above chance) where the current spiking readout is capped. Pre-registered: READOUT-LIMIT (artifact (b)) if a held-out decoder reaches >= 2x the spiking readout's accuracy on the same composed states; REPRESENTATIONAL CEILING CONFIRMED if no decoder beats the spiking readout's ~0.46 by a margin. This is the clean representation-vs-readout discriminator and directly informs whether the big phase-coded VSA arc is warranted (representational) or whether a cheaper readout/cleanup fix unlocks the existing mechanisms (readout-limited).

## Discipline

No protected/frozen/moat module read or modified (read-only). No bars moved. Pre-registration honored, including honestly NOT claiming the literal artifact-(a) when the specific condition was not met. Reuse-by-import only for any Phase 2 probe. The phase-coded VSA arc is NOT started -- the audit gates it.
