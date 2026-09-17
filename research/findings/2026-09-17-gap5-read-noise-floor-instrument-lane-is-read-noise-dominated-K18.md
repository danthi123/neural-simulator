---
type: finding
status: verified
date: 2026-09-17
mechanism: a read-fidelity INSTRUMENT for the gap#5 learn-through-use lane — repeat N weak-cue graded reads of the SAME
  frozen post-consolidation weights with a read-trial seed independent of the substrate-build seed, isolating the
  population-read noise floor at the weak-cue recall operating point (the instrument is part of the emulation)
integration_faculty: learn-through-use (gap#5 memory / Ecker AdEx CA3)
lane: memory (learn-through-use, gap#5)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (instrument-VALID 6/6 — graded 6/6, weights frozen 6/6, sigma_read>0 6/6 anti-cheat). The DIAGNOSTIC NUMBER
  the two weight-side NO-GOs (forward-band, reverse-edge) were missing: the single-read weak-cue noise (sigma_read_bar,
  depth_frac units) is LARGER than the across-seed spread of the operating-point mean (sigma_substrate) -> read_SNR
  below one -> this lane's seed-to-seed spread is READ-NOISE-DOMINATED. The minimum-detectable-gain from a SINGLE read
  trial is several-fold the lane's own gain bar; resolving that bar would need roughly a dozen-plus independent read
  trials per arm. So the "no gain" verdicts on the weight levers were partly UNDER-SAMPLED instrument reads, NOT proof
  the mechanism does nothing — the next LTU test must AVERAGE ~18 reads/arm (or coarsen the bar), before any further
  weight lever. This does not itself confirm/refute a mechanism; it re-scopes how the lane must MEASURE.
runner: research/runners/_gap5_read_noise_floor_ltu_derisk.py
artifacts:
  - research/findings/raw/gap5_ecker_adex/read_noise_floor_6seed.json
external: NO-EXTERNAL-NEEDED — a diagnostic instrument on the substrate's own spikes (no host shortcut to cognition);
  the "instrument is part of the emulation" discipline (CLAUDE.md).
builds_on:
  - research/findings/2026-09-16-gap5-forward-band-homeostatic-scaling-ltu-NO-GO-redirect-read-noise-floor.md
---

# gap#5 read-noise-floor instrument — the learn-through-use lane is READ-NOISE-DOMINATED (needs ~18 reads/arm)

The gap#5 learn-through-use residual (weak-cue forward recall not improving after consolidation) produced two
weight-side NO-GOs (forward-band homeostatic scaling; reverse-edge heterosynaptic depression), each concluding with
"characterize the READ-side noise floor." This builds that instrument and returns the number.

## The number (from research/findings/raw/gap5_ecker_adex/read_noise_floor_6seed.json)
<!--derived-->
- Instrument-VALID 6/6: graded read confirmed per seed, weights frozen across every read trial, and sigma_read>0 on
  6/6 (distinct read-trial seeds actually vary the read — the anti-cheat against a zero-noise degenerate).
- sigma_read_bar ~0.0745 (single weak-cue read-trial noise, depth_frac units, at the post-consolidation operating
  point) vs sigma_substrate ~0.0446 (across-6-seed spread of that operating point's mean read) => read_SNR ~0.60.
- Minimum-detectable-gain from one read trial ~0.21; K_reads_needed ~18 to resolve the lane's 0.05 depth_frac gain bar
  (this run used 12/arm).

## What it means

The lane's seed-to-seed spread is READ-NOISE-DOMINATED (read_SNR < 1): a SINGLE weak-cue read cannot resolve a 0.05
depth-gain because the single-read noise (~0.21 MDG) is 4x the bar. The forward-band + reverse-edge "no gain" verdicts
were measured with single (or few) reads, so they are partly under-sampled instrument reads — NOT proof the weight
mechanisms do nothing. The corrected protocol for the LTU lane: AVERAGE ~18 reads/arm (or coarsen the bar) before
re-adjudicating any weight lever. This is a measurement re-scope (the instrument is part of the emulation), not a new
weight mechanism — and it explains months of ambiguous single-read LTU results.
