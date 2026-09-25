---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: v3 DA-gated synaptic tagging-and-capture where the DA acts THROUGH synapses (webapp/da_tag_capture.py SynapticTagCaptureLedger, BRAIN_DA_TAG_CAPTURE default OFF) under the same natural surprising-vs-expected drive, read as 24 h recall at the production composer D=128
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_encoding_natural_drive_v3_pilot/pilot_seed7.json
  - research/findings/raw/_da_encoding_natural_drive_v3/offcheck.json
  - research/findings/raw/_da_encoding_natural_drive_D128/aggregate.json
---

# PRE-REGISTRATION v3 — the DA effect acts through synapses; each arm in a fresh process; a band that can fail (2026-09-23)

Committed on its own, BEFORE any v3 gate-seed run. It amends the v2 prereg
(`2026-09-23-da-encoding-natural-drive-v2-production-D128-PREREGISTRATION.md`) after an adversarial review found
three defects in v2. The code it governs was committed first, in `492231df3`, with every constant below fixed.

## The three review defects and what v3 does about each

1. **A host rule decided the 24 h outcome.** v2's `TagCaptureLedger` compared the DA scalar to 0.62 and scaled every
   uncaptured block by exp(-24/1.5). Once the DA trace separated at 0.62, G3-G5 followed arithmetically.
   **v3:** `SynapticTagCaptureLedger`. The brain's SNc DA level is broadcast onto the spiking D1 population (the
   `write_gain` IZH2007 CA1-pyramidal pool whose rate already sets the production write gain). Its measured
   firing-rate excess drives a cell-wide PRP pool. Each synapse's tag is set by its own early-LTP amplitude (so the
   DA write gain enters through the tag). Each synapse's bistable late-phase variable decides that synapse. There is
   no DA-vs-number compare in the capture path.
2. **The band could not fail.** The E-LTP decay points could not fail at 24 h, and baseline ratio 2.0 was UNDEFINED
   on every seed and silently dropped. **v3:** the capture band perturbs only constants that move the capture
   boundary. Each point carries a kernel-only fail-ability precondition, and an UNDEFINED point FAILS G8.
3. **Arms depended on run order.** **v3:** every arm runs in its own fresh subprocess; arm state is reset at arm
   start; the gain the store actually used is logged; one arm is re-run in a second process (G0). Pinned by
   `tests/test_da_encoding_arm_isolation.py`.

## Mechanism (all constants fixed in `492231df3`, before any v3 brain run)

Per turn (30 s world clock) the D1 activation is a = clip((r(DA) - r(0.5)) / (r(1.24) - r(0.5)), 0, 1), with r the
`write_gain` pool's measured rate (production reader, seed 42, OU noise on) and the two anchors the pool's own
tonic and arousal rates. Then

- PRP pool: dp/dt = -p/tau_p + a/tau_p, tau_p = 1.0 h (Moncada & Viola 2007: novelty up to 1 h before still rescues).
- Tag on synapse k: h_k = |inc_k| at the write, decaying with tau_tag = 1.5 h (Frey & Morris 1997: tag < 3 h).
- Late phase: tau_z dz_k/dt = -z_k (z_k - 1/2)(z_k - 1) + gamma p h_k, tau_z = 0.5 h (declared a priori).
- Early phase e = exp(-(t - t_w)/1.5 h); weight w_k = b_k + inc_k (e + z_k (1 - e)), baseline b_k as in v2 (beta = 1).

**gamma is calibrated with no brain data.** It is the smallest gamma at which a unit-tag synapse driven for 5 min (the
novelty exposure of Moncada & Viola 2007 and Wang, Redondo & Morris 2010) at the D1 activation the pool produces at
the Go boundary DA 0.62 ends with z > 1/2 at 12 h. It is recomputed in every arm process from the pool's calibration
reads and must come out identical in every arm (G0).

## Arms (each a fresh subprocess at the same seed)

`intact`; `lesion_da_encoding` (`BRAIN_DA_ENCODING_LESION=1`: write gain pinned to 1 AND the D1 pool receives tonic
DA); `lesion_capture` (`BRAIN_DA_CAPTURE_LESION=1`: D1 pool receives tonic DA and the D1->PRP coupling is 0; write
gain intact); `companion_off` (production today); `lesion_novelty` (exploratory). Band arms: `intact` and
`lesion_da_encoding` at each band point, both conditions. Replicate: salient/intact primary, second process.

## Gates (per seed). UNDEFINED unless every precondition holds

Preconditions:
- **G0** the replicate process reproduces the write gains used, every D1 read, the synapse state and the 24 h replies
  exactly; the calibrated gamma is identical in every arm process.
- **G1** every salient fact turn DA >= 0.72 and every neutral turn DA < 0.62 (unchanged; a stimulus check only — the
  mechanism no longer reads 0.62 as a threshold). **G1b** DA trace identical when rebuilt.
- **G2** 4/4 immediate recall (+1 min) in every primary arm.
- **G8-defined** every capture-band point is DEFINED (4/4 immediate recall in both of its arms, both conditions).
- Reach: the DA->encoding lesion changes the used write gain and the PRP pool maximum; the capture lesion changes
  the PRP pool maximum.

GO requires all of:
- **G3** salient 24 h: intact >= 3/4 and DA->encoding lesion <= 1/4.
- **G4** neutral 24 h: intact <= 1/4.
- **G5** capture lesion (salient) <= 1/4.
- **G6** companion off: 4/4 in both conditions.
- **G8** capture band: at least 4 informative points, and every informative point holds (salient intact >= 3/4,
  salient lesion <= 1/4, neutral intact <= 1/4).

<!--derived-->
Capture band: gamma x0.5, x2; tau_p 0.5 h, 2 h; tau_tag 0.75 h, 3 h; tau_z 0.25 h, 1 h (gamma held at the primary
calibrated value, so each point MOVES the boundary). A point is **informative** iff its own critical 5-min D1
activation (kernel only, no brain data) differs from the primary's by >= 10 % and lies inside (0, 1). Kernel values
at a_go = 0.224 (for reference; the runner recomputes them from each seed's calibration): primary 0.224; gamma x0.5
0.448, x2 0.112; tau_p 0.5 h 0.169, 2 h 0.334; tau_tag 0.75 h 0.307, 3 h 0.184; tau_z 0.25 h 0.129, 1 h 0.412.

Readability band (reported, NOT in the GO): baseline ratio 0.67 and 2.0. The baseline ratio cannot move the capture
boundary, so it cannot test the DA claim; and 2.0 is already known from v2 to fail immediate recall.

**Aggregate GO** requires all six seeds GO, no seed UNDEFINED, and a one-sided sign-flip permutation p <= 0.05 on the
per-seed (salient intact - lesion) 24 h differences, with the SEED as the unit (the four facts of one conversation
share one PRP pool and one DA trace, so they are not exchangeable). The smallest reachable p is 1/64.

## What a GO would and would not license

It would license: "the brain's spiking DA, read by a spiking D1 population, drove a pre-registered synaptic
tag-and-capture rule on the store synapses so that surprising facts were kept at 24 h and plain ones were not; the
reply changed under a lesion of the DA edge". It would NOT license "the brain decides by itself": the per-synapse
equations are host-integrated (the same category as every plasticity rule in the engine), and the rate-to-activation
normalization is host arithmetic on a measured spike count. Not wired; default OFF.

If G4 fails (plain facts captured), the reading is that the natural neutral drive carries enough D1 activation to
capture under these constants; the next method is a competitive/homeostatic companion on the PRP pool (e.g. PRP
competition between tagged synapses, Fonseca et al. 2004) rather than retuning gamma.

## AMENDMENT LOG — what I had seen when writing this

<!--derived-->
- All v1 and v2 artifacts and findings, including per-seed DA extremes (salient fact DA minimum 0.792-0.897; neutral
  maximum 0.196-0.573 across the six gate seeds), every v2 arm outcome, and the reviewer's notes.
- The production D1 reader's rates (seed 42 reader, one read each, during a timing probe): r(0.5) = 42.8 Hz,
  r(0.62) = 44.8 Hz, r(1.24) = 52.0 Hz, so a_go is about 0.22. This is the instrument, not gate data.
- The kernel's band shifts listed above (computed from the equations only).
- The v3 pilot on seed 7 (not a gate seed), run at `492231df3` after the constants were committed:
  verdict GO; all preconditions held (G0 replicate identical in a second process, gamma identical in all 43 arm
  processes). a_go = 0.187 from the three-read calibration (gamma = 32.77). Salient: D1 activation 0.50-1.00 on the <!--derived-->
  fact and following turns, PRP maximum 0.068, all four blocks z = 1.0, 4/4 at 24 h; DA->encoding lesion PRP maximum <!--derived-->
  0.0018, 0/4; capture lesion PRP 0, 0/4. Neutral (DA maximum 0.554): D1 activation 0-0.125, PRP maximum 0.0023, <!--derived-->
  z about 1e-11, 0/4. Companion off 4/4 both. All 8 capture-band points informative and holding; both beta points
  DEFINED and holding on this seed; the exploratory novelty-lesion arm kept neutral facts (4/4). Runtime 238 s with
  3 workers; about 0.4 GB per arm process.
- Nothing was changed after the pilot.

## Byte-identical OFF, asserted in data

`research/runners/_da_tag_capture_offcheck.py` builds the production DA-encoding store path with
`BRAIN_DA_TAG_CAPTURE` unset on a `git archive` of the pinned pre-change SHA `5e1f0ec79` (origin/main at the start of
this fix round) and on the branch tree, and compares a sha256 over every store synapse and the recall replies. A
sensitivity control (the same hash with the companion armed) must differ. Artifact:
`research/findings/raw/_da_encoding_natural_drive_v3/offcheck.json`.

## Command (one per seed)

`bash tools/memcap.sh 6 -- .venv/bin/python -u -m research.runners._da_encoding_natural_drive_synaptic --seed <s> --workers 3 --out research/findings/raw/_da_encoding_natural_drive_v3/seed<s>.json`
then `--aggregate research/findings/raw/_da_encoding_natural_drive_v3`.
