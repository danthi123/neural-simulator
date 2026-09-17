---
type: finding
status: verified
date: 2026-09-16
mechanism: Turrigiano-style multiplicative synaptic scaling of the CA3 FORWARD band toward an absolute (per-seed-relative)
  magnitude target, applied once between encode and consolidation, on top of the established BTSP + forward-conduction-delay
  directional write (unmodified) — the "forward-band magnitude" candidate for the learn-through-use recall residual
integration_faculty: learn-through-use (gap#5 memory / Ecker AdEx CA3 store)
lane: memory (learn-through-use, gap#5)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO 1/6 (mult=2.0). Absolute-magnitude forward-band homeostatic scaling raises the adjusted forward band
  (n_fwd_raised 6/6), keeps the write directional (6/6) and keeps read headroom (6/6) — but it does NOT lift weak-cue
  recall: depth_frac DROPS rather than rises and use-dependence clears the gain bar on only 1/6 seeds (lesion-null
  6/6). So forward-band ABSOLUTE MAGNITUDE is NOT the learn-through-use residual either. This CONVERGES
  with the reverse-edge heterosynaptic-depression NO-GO (commit 170cee350), whose own redirect was "characterize the
  READ-SIDE NOISE FLOOR, not more weight-side levers." Both weight-side levers (reverse-edge suppression; forward-band
  magnitude) are now exhausted -> the next mechanism is the read-side noise-floor INSTRUMENT (a diagnostic, per
  "the instrument is part of the emulation"), designed 2026-09-16 and queued to build.
runner: research/runners/_gap5_forward_band_homeostatic_scaling_ltu_derisk.py
artifacts:
  - research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_mult2p0_6seed.json
external: Turrigiano 2008 (synaptic scaling homeostasis) grounds the mechanism; NO new external needed — this is a
  decisive 6-seed adjudication of a named build-ahead candidate, and its own redirect names the next instrument.
builds_on:
  - research/findings/2026-08-27-reverse-edge-heterosynaptic-depression-learn-through-use-NOGO.md
---

# gap#5 forward-band homeostatic scaling — learn-through-use recall NO-GO (redirect to the read-side noise floor)

The learn-through-use (gap#5) lane's residual — weak-cue forward recall not improving AFTER consolidation vs BEFORE —
had two named weight-side candidates left. The reverse-edge heterosynaptic-depression lever was a 6-seed NO-GO
(commit 170cee350) that re-diagnosed the blocker as the read side. This run adjudicates the OTHER weight-side
candidate: Turrigiano absolute-magnitude scaling of the forward band.

## Result (values from the artifact)

(from `research/findings/raw/gap5_ecker_adex/forward_band_homeostatic_scaling_ltu_mult2p0_6seed.json`; numbers below rounded from that artifact)
<!--derived-->


- The scaling WORKS mechanically: `adj_fwd_before` 319.4 -> `adj_fwd_after` 342.8 (n_fwd_raised 6/6), the write stays
  directional (n_directional 6/6), and read headroom is intact (`weak_depth_frac_before` 0.586, n_headroom 6/6).
- But it does NOT lift recall: `weak_depth_frac` 0.586 -> 0.500 (it DROPS), `weak_tau` 1.000 -> 0.798; use-dependence
  clears the gain bar on only n_use_dependent 1/6 (lesion-null 6/6). `GO: false`, n_go 1, status NO-GO.

## Interpretation — both weight-side levers exhausted; the instrument is next

Absolute forward-band magnitude is not the residual. With the reverse-edge NO-GO, both weight-side levers are now
exhausted. The convergent redirect (this runner's own verdict + the reverse-edge finding's) is to build a read-side
noise-floor INSTRUMENT: repeat N weak-cue graded reads of the SAME frozen post-consolidation weights (a read-trial
seed independent of the substrate-build seed) to separate genuine per-seed substrate variance from instrument read
noise at the weak-cue operating point — giving the next mechanism a number to design against, and classifying whether
the 4/6 "insensitive" seeds are read-noise-dominated. Designed 2026-09-16 (`_gap5_read_noise_floor_ltu_derisk.py`,
queued). A wall defers a METHOD, not the capability.
