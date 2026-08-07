---
type: finding
status: negative
date: 2026-08-07
lane: laneC
mechanism: source-monitor-heterosynaptic-encoding
runner: research/runners/_laneC_source_monitor_hetero_encoding.py
artifacts:
  - research/findings/raw/laneC_source_monitor_hetero_encoding/smoke_650_651_lam1.0.json
  - research/findings/raw/laneC_source_monitor_hetero_encoding/smoke_650_651_lam1.0.json.prov.json
  - research/findings/raw/laneC_source_monitor_hetero_encoding/smoke_650_651_overlap0.6_lam1.0.json
  - research/findings/raw/laneC_source_monitor_hetero_encoding/smoke_650_651_overlap0.6_lam1.0.json.prov.json
---

# Source-monitor hetero-encoding NO-GO (smoke): the shared core fires in EVERY recall, so pedestal depression redistributes rather than removes the rival burden

**Smoke, not a verdict on the capability.** Two calibration seeds (650, 651), numpy, deterministic. This is the 5th lever
against the source-monitor encoding wall and the first to act on the ENCODING fan-out (the prior 4 acted at recall / on
activity). Mechanism: thresholded heterosynaptic depression at encoding ("protect the peak, depress the pedestal"), keyed
to each episode cell's CUMULATIVE per-source co-activation eligibility, applied via the committed CA3-GO kernel
`sim.kernels.fused_htm_winner_inactive_depression` (imported BY REFERENCE, no `sim/` edit) to the `episode -> source`
CSR weights. Pre-scoped in `research/findings/raw/_source_monitor_heterosynaptic_encoding_scoping.md`.

## Result: NO-GO. min M never clears the floor F=0.15, never beats the lesion arm, and is often WORSE than baseline

| seed | overlap | core | commitment (seen/heard/self) | min M (treat) | min L | min M (base, lam=0) | clears 0.15 | beats L |
|------|---------|------|------------------------------|---------------|-------|---------------------|-------------|---------|
| 650  | 0.20    | 2    | 0 / 1 / 1                    | **+0.0050**   | +0.0367 | -0.0092           | no          | no      |
| 651  | 0.20    | 2    | 1 / 1 / 0                    | -0.1567       | -0.0425 | -0.1233           | no          | no      |
| 650  | 0.40    | 5    | 2 / 3 / 0                    | -0.0333       | -0.0167 | -0.1292           | no          | no      |
| 651  | 0.40    | 5    | 2 / 3 / 0                    | -0.1650       | -0.0800 | -0.1292           | no          | no      |
| 650  | 0.60    | 7    | spans 3 (H=0.87)             | -0.0908       | -0.0758 | -0.1225           | no          | no      |
| 651  | 0.60    | 7    | 2 of 3 (H=0.54)              | -0.1258       | -0.1575 | -0.1117           | no          | no      |

Best min M anywhere is **+0.005** (seed 650, overlap 0.2) — ~30x below the 0.15 floor, and essentially the base best the
scoping doc already recorded. On seed 651 the lever makes the weakest margin WORSE than the symmetric-Hebbian baseline.

## The symmetry-breaking risk the scoping doc pre-flagged MATERIALIZED — and a deeper structural reason underneath it

- **(a) null control PASSES:** `lam_hetero=0` is byte-identical to the symmetric-Hebbian overlap NO-GO on every row
  (asserted against the original `_laneC_source_monitor_overlap_sweep.evaluate_overlap`). The lever, not the instrument,
  is under test.
- **(b) instrument guard PASSES:** the zero-learned-weight control stays `strict=False` on every row (no
  stepping-history artifact).
- **(c) commitment does NOT span three sources** (except at overlap 0.6 seed 650, H=0.87). At overlaps 0.2/0.4 one
  source — consistently `self_generated`, the WEAKEST-co-firing source, not the last-encoded — receives ZERO core-cell
  commitments (e.g. commit = seen:2 / heard:3 / self:0 at overlap 0.4). The break is biased, not uniform.
- **(d) reliability FAILS on every row:** the depression cuts the DEPRIVED source's OWN recall rate (e.g. seed 651,
  overlap 0.2: `self_generated` own rate 0.171 -> 0.140) because a core cell that commits to seen/heard has its
  `self_generated` synapse cratered, so during `self_generated`'s own recall that cell no longer contributes.

**The structural reason the encoding fan-out lever cannot work here (measured, not just the tuning failing):** the shared
core fires in EVERY recall, because the same core cells are present in all three pure-source episode patterns. Committing
a core cell to its peak source therefore makes that cell a RIVAL during the other two sources' recalls (its protected
peak weight drives a rival), while cratering its pedestal removes its (real, needed) contribution to those sources.
Post-hoc heterosynaptic depression on a fully-shared fan-out thus REDISTRIBUTES the rival burden across sources rather
than removing it — net margin does not rise even when commitment spans all three (overlap 0.6 seed 650: commitment
H=0.87, yet min M = -0.091). Sharpening the fan-out cannot separate sources whose separation must live in WHICH cells
fire, not in which synapses survive.

## Verdict + next rung (no-defer: a verdict on the METHOD, not the capability)

Post-hoc thresholded heterosynaptic depression on the shared episode->source fan-out is a NO-GO for the source-monitor
weakest-margin criterion. The pre-scoped FALLBACK is the correct next method and is indicated by this exact result: a
**conjunctive source-tag** — let the physical source afferent WEAKLY modulate the overlap layer DURING ENCODING so that
different shared-cell SUBSETS fire preferentially per source (Komorowski-Manns-Eichenbaum 2009 item-in-context
conjunctive cells). That moves the separation into WHICH cells fire (the thing this lever proved must change), not which
synapses survive, so a source's rivals stop firing during its recall instead of being merely down-weighted. Not built
here (scope: build `lam_hetero` first, per the scoping doc). Runner + instrument retained; the byte-identical null
control makes the fallback a clean A/B against this arm.

Full-validation commands (calib 650/651 + dev 652/653/654 + held-out 655/656/657) are recorded for the parent to run
orphan-proof; this smoke does not run them.
