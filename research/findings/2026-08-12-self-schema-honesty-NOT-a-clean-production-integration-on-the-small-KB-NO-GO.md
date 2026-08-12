---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — self-schema honesty (confidence-graded hedging) does NOT drop into the production chat on the current KB
lane: integration-first (a faculty NO-GO, honestly banked)
verdict: NO-GO (for now). Wiring the SelfSchemaHonestyMonitor (the brain hedging assert->hedge->abstain from its OWN familiarity read) into the production gate is not a clean, lesion-verifiable integration on the tiny-demo KB, for two measured reasons: (1) NOT CALIBRATED — on the rf composer (D=64) the trace confidence of CORRECT recalls is 0.32–0.47, all BELOW the default assert threshold (0.55) and around/below the hedge threshold (0.38), so the default operating point would hedge or abstain on EVERY correct answer (an over-hedge regression). (2) NOT LESION-VERIFIABLE HERE — the tiny-demo KB has only clean facts, so a recall either returns the correct fact (confident enough) or the moat abstains (None); there are NO genuine uncertain-recall cases for a hedge to help, so even a recalibrated monitor would never fire meaningfully -> disabling it would change no answer -> it cannot earn lesion-load-bearing (level-3) credit; wiring it would be a cosmetic flip, exactly what CLASS PI forbids. Confidence-graded honesty needs a LARGER/fuzzier knowledge base (where recalls have a real confidence spread) + a substrate-recalibrated operating point.
artifacts:
  - research/runners/self_schema_honesty.py
verification: direct measurement — trace_confidence on the tiny-demo recalls (dog chase 0.316, cat eat 0.466, brain use 0.448, taught wolf hunt 0.464) vs SelfSchemaHonestyConfig (assert 0.55, hedge 0.38).
---

# self-schema honesty is NOT a clean production integration on the current small KB — NO-GO (honest, measured)

## Why this was attempted + why it is a NO-GO

The wiring backlog names self-awareness / honesty-boundary faculties as high-value ("the brain expresses graded
confidence"). The `SelfSchemaHonestyMonitor` (research/runners/self_schema_honesty.py) is a de-risked GO: it reads a
confidence scalar from the composer trace, drives a metacognitive confidence population, and can DOWNGRADE an answer
assert->hedge->abstain (never a hard-moat miss into an answer). I attempted to wire it into `ChatBrain.gate()` (the
production entry point, per the entry-point lesson). Two measured facts make it a NO-GO on the current KB:

<!--derived-->
1. **Not calibrated for this substrate.** `trace_confidence` on the tiny-demo's CORRECT recalls: dog→chase 0.316,
   cat→eat 0.466, brain→use 0.448, taught wolf→hunt 0.464 — all BELOW the default `confidence_assert` (0.55), and
   dog→chase is below `confidence_hedge` (0.38). At the default operating point the monitor would HEDGE or ABSTAIN on
   every correct answer. (The confidence scale is substrate-specific; the de-risk operating point does not transfer.)
2. **Not lesion-verifiable here.** The tiny-demo KB is 5 clean facts; a recall either returns the correct fact or the
   moat abstains (None). There are no "recalled-but-uncertain" cases for a hedge to act on. So even a recalibrated
   monitor would never change an answer -> disabling it changes nothing -> it cannot earn lesion-load-bearing (level-3)
   credit under CLASS PI. Wiring a never-firing monitor is a cosmetic flip — the exact thing the lesion discipline
   forbids.

## The honest conclusion (what it would take)

Confidence-graded honesty is a real faculty, but it needs (a) a LARGER/fuzzier knowledge base where recalls have a
genuine confidence spread (near-misses, partial matches), and (b) a confidence operating point RE-CALIBRATED to that
substrate's confidence distribution (not the de-risk's 0.55/0.38). Until then it is dormant-by-construction and cannot be
lesion-verified. Banked so it is not re-attempted as a "quick flip." The higher-value integrations that CAN be
lesion-verified now (a genuinely-spiking store, scaffold retirement, deeper LEARN) come first.
