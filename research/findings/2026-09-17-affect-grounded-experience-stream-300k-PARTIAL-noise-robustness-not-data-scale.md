---
type: finding
status: verified
date: 2026-09-17
mechanism: emergent Hebbian convergence of an affect code over a GROUNDED-experience story stream, scaled to 300k
  stories (5x the prior 60k) — testing whether the affect noise-robustness residual is a DATA-SCALE limit
integration_faculty: affect (grounded-experience stream)
lane: emotion / affect grounding
seeds: [42, 43, 44, 100, 101, 102]
verdict: PARTIAL/BOUNDARY (GO=False on the strict bar). More grounding data does NOT clear the residual. At clean/full
  grounding the emergent Hebbian code IS separable (near-perfect vs the near-zero text baseline) and grounding is
  load-bearing (lesion + shuffle both collapse it), and it holds out on clean data. But the pre-registered strict bar
  fails on the REALISTIC-noise arm (falls below the bar), and likewise the three-factor US-gated realistic arm. So the
  affect residual is a NOISE-ROBUSTNESS limit, NOT a data-scale limit — 5x more grounding data left the realistic-noise
  arm essentially unchanged. This CONFIRMS the owner's steer (2026-09-06): affect is multimodal — surpass the text boundary
  by GROUNDING (interoception/prosody/face), not by text-data-scale. The (rho,sigma) frontier + relaxed-FP sensitivity
  in the artifact map what coverage/noise/FP-tolerance WOULD clear it. Wires nothing (the affect_production_organ
  _STRONG_MARGIN gate is unchanged).
runner: research/runners/_affect_grounded_experience_stream_hebbian_derisk.py (--max-stories 300000)
artifacts:
  - research/findings/raw/_affect_grounded_experience_stream_hebbian_maxstories300k_6seed.json
external: NO-EXTERNAL-NEEDED — a data-scale control on an already-characterized affect residual; the honest negative
  (data-scale doesn't fix noise-robustness) is the deliverable, aligning with the multimodal-grounding direction.
builds_on:
  - feedback (owner 2026-09-06): affect is multimodal — ground it, don't text-cleverness it
---

# Affect grounded-experience stream at 300k stories — PARTIAL: the residual is noise-robustness, not data scale

The affect-grounding lane's residual is a realistic-noise robustness shortfall. This tests the simplest hypothesis —
more grounding data — by scaling the emergent Hebbian convergence from 60k to 300k stories.

## Result (from research/findings/raw/_affect_grounded_experience_stream_hebbian_maxstories300k_6seed.json)
<!--derived-->
- CLEAN/full grounding: the emergent code is separable, worst-case ~1.000 vs text ~0.059; grounding load-bearing
  (lesion ~0.069, shuffle ~0.059); held-out clean ~1.000; text-only transfer ~0.010-0.023 (grounding, not text, carries it).
- REALISTIC-noise arm: ~0.049 worst (below the pre-registered bar); three-factor US-gated realistic ~0.049. FAILED:
  G1_lift_clears_bar. GO=False.

## What it means

5x more grounding data did NOT move the realistic-noise arm — so the affect residual is a NOISE-ROBUSTNESS limit, not a
data-scale limit. This is an honest negative that CONFIRMS the direction: the text/data boundary on affect is expected
(owner 2026-09-06 — affect is multimodal), and the surpass is GROUNDING (interoception + prosody/audio + face/social
vision), not more text-derived grounding data. The artifact's (rho,sigma) frontier + relaxed-FP sensitivity bound what
coverage/noise-model/FP-tolerance would clear the bar — the input for a realistic-noise or multimodal grounding arc.
A wall defers a METHOD (text-data-scale), not the capability (grounded affect).
