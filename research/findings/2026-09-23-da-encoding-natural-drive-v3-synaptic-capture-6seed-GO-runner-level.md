---
type: finding
status: qualified
lane: load-bearing
date: 2026-09-23
mechanism: v3 DA-gated synaptic tagging-and-capture where the DA acts through synapses (webapp/da_tag_capture.py SynapticTagCaptureLedger, BRAIN_DA_TAG_CAPTURE default OFF) under a natural surprising-vs-expected conversational drive, 24 h recall, production composer D=128
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO at runner level, 6 of 6 seeds and aggregate (pre-registered v3 gates, prereg 014298f62); NOT wired into /api/brain-chat; companion default OFF
artifacts:
  - research/findings/raw/_da_encoding_natural_drive_v3/aggregate.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed42.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed43.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed44.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed100.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed101.json
  - research/findings/raw/_da_encoding_natural_drive_v3/seed102.json
  - research/findings/raw/_da_encoding_natural_drive_v3/margins_exploratory.json
  - research/findings/raw/_da_encoding_natural_drive_v3/offcheck.json
---

# DA-gated encoding v3: the brain's spiking DA, read by a spiking D1 population, drives a synaptic tag-and-capture rule that keeps surprising facts at 24 h — 6/6 GO at runner level (2026-09-23)

Pre-registration: `2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-PREREGISTRATION.md` (commit `014298f62`,
before any gate-seed run; code and constants in `492231df3`). It answers the three defects the adversarial review
found in v2 (`2026-09-23-da-encoding-natural-drive-v2-D128-6seed-GO-runner-level.md`, now retitled). Terms follow
`docs/TERMS.md`. Artifacts: `research/findings/raw/_da_encoding_natural_drive_v3/seed<s>.json`,
`research/findings/raw/_da_encoding_natural_drive_v3/aggregate.json`,
`research/findings/raw/_da_encoding_natural_drive_v3/margins_exploratory.json`,
`research/findings/raw/_da_encoding_natural_drive_v3/offcheck.json`.

## What changed from v2

1. **No host threshold decides capture.** The brain's SNc DA level is broadcast onto the spiking D1 (`write_gain`)
   population. Its measured rate excess drives a cell-wide PRP pool. Each store synapse carries a tag set by its own
   early-LTP amplitude and a bistable late-phase variable; each synapse's own variable decides that synapse. The
   Go-boundary number 0.62 appears only in the a-priori calibration of one coupling constant (a 5-min exposure at the
   Go-boundary D1 level just captures), never as a compare on the trace.
2. **The band can fail.** Eight capture-band points (gamma, tau_p, tau_tag, tau_z at x0.5 and x2) each move the
   critical D1 activation by 18-100 % inside (0, 1) (kernel-only precondition), and an UNDEFINED point fails G8.
3. **Arms are order-independent.** Each arm ran in its own subprocess with its state reset; a replicate arm in a
   second process reproduced gains, every D1 read, the synapse state and the replies exactly on every seed (G0).

## Result

| seed | salient DA min | neutral DA max | salient intact 24 h | DA->encoding lesion | neutral intact | capture lesion | companion off (sal / neu) | salient PRP max (intact / lesion) |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.897 | 0.549 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.070 / 0.0047 |  <!--derived-->
| 43 | 0.798 | 0.411 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.060 / 0.0066 |  <!--derived-->
| 44 | 0.897 | 0.413 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.068 / 0.0038 |  <!--derived-->
| 100 | 0.897 | 0.335 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.070 / 0.0030 |  <!--derived-->
| 101 | 0.897 | 0.573 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.066 / 0.0030 |  <!--derived-->
| 102 | 0.792 | 0.196 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 | 0.056 / 0.0036 |  <!--derived-->

- **All six seeds GO; the aggregate is GO.** No seed UNDEFINED. With the seed as the unit (the four facts of one
  conversation share one PRP pool and are not exchangeable), the one-sided sign-flip p is 1/64 for intact vs
  DA->encoding lesion and 1/64 for salient vs neutral.
- **The reply changes under the lesion.** A fact told as surprising news is answered the next day; with the DA edge
  lesioned the same question gets an abstention. The lesion was verified to hold at measurement: the PRP pool never
  exceeded 0.0066 in the lesion arm (0 in the capture-lesion arm) against 0.056-0.070 intact.  <!--derived-->
- **All 8 capture-band points were informative and held on every seed.** Zero confabulations at the primary point
  and across the capture band. Immediate recall was 4/4 in every primary arm.
- **Production default reproduces the old null.** With the companion off every fact is recalled at 24 h.
- **Byte-identical off.** The production store path with `BRAIN_DA_TAG_CAPTURE` unset hashes identically on a
  `git archive` of the pinned pre-change SHA `5e1f0ec79` and on the branch; the same hash with the companion armed
  differs (`offcheck.json`).

## How close to the boundary (EXPLORATORY, not pre-registered)

Replaying the kernel on each seed's recorded D1 drive and tags (`margins_exploratory.json`; the replay reproduces the
run's capture pattern on every seed): salient capture would be lost only if gamma fell below x0.09-0.12 of its
calibrated value. Neutral facts would be captured at gamma x3.24 on seed 42 and x39.8 on seed 101, and never on the
other four seeds, whose neutral DA stayed at or below tonic so the D1 pool got no drive at all. So the x0.5/x2 band
was informative in the kernel sense but, on this drive, only seed 42's neutral conversation comes within a factor of
~3 of flipping.

## What this licenses, and what it does not

- **Licensed:** the brain's spiking DA, read by a spiking D1 population, drove a pre-registered synaptic
  tag-and-capture rule on the store synapses; surprising facts were kept at 24 h and plain ones were not; the reply
  changed under a lesion of the DA edge; no host compare on the DA trace decided it.
- **Not licensed — host shortcuts still in the path (declared):** the per-synapse tag / PRP / late-phase equations are
  host-integrated (the same category as every plasticity rule in the engine, but not neurons); their constants are
  pre-registered, not learned; the rate-to-activation normalization is host arithmetic on a measured spike count; the
  runner is the parse boundary that calls `comp.store`; the world clock is environment. The late-phase variable is a
  synaptic state, not a replay path, so this is not "consolidation" in the `docs/TERMS.md` sense.
- **Not wired.** Nothing in `webapp/server.py` builds the ledger; the battery has no next-day turn. `da-gated-encoding`
  stays hollow on the load-bearing battery until that rung lands.

## Honest residuals

- **Wide margins mean the DA contrast still does most of the work.** The neutral conversations mostly held DA at or
  below tonic, so the D1 pool got no drive on four seeds; the synaptic dynamics were only tested near their boundary
  on seed 42 (x3.2). A harder neutral stimulus (moderately engaging plain facts) is the next test of the dynamics.
- **Readability band (reported, not gated, as pre-registered):** baseline ratio 0.67 held on 6/6 seeds. Baseline
  ratio 2.0 was UNDEFINED on 6/6 seeds (unit-gain writes misread immediately), with 1-2 immediate confabulations on
  seeds 43, 101 and 102. The no-confabulation property does not survive a weaker-LTP assumption.
- **The write gain still carries information into the tag**, so the DA->encoding lesion removes two routes at once
  (tag size and PRP). The capture-only lesion isolates the PRP route (0/4 on every seed); a gain-only lesion was not run.
- **Calibration sits on seed 42's D1 reader for every run seed.** That is the production reader (`_leaf_gain` uses
  seed 42 regardless of the brain seed), so a_go = 0.187 and gamma = 32.77 are identical across all runs.  <!--derived-->
- **Provenance marks the runs git_dirty** (the shared provenance ledger and untracked artifacts); the code commit is
  `d40de0817`, which differs from the prereg-governed `492231df3` only in docs. For about a minute after seeds
  101/102 launched (14:24:53) the working-tree runner carried extra exploratory functions not on the run path, so
  their parent processes may have imported that version; it was restored at 14:25:38, before any of their arm
  subprocesses started (after the ~2 min trace phase), and the functions were committed afterwards (`1859e0188`).
- **Exploratory novelty-lesion arm:** neutral facts were kept on seeds 42 and 101 (4/4) and lost elsewhere; salient
  facts were kept on every seed. It does not locate where the drive originates.
- **No adversarial verify-go pass yet** (this lane had no subagent tool). The controller should run `verify-go`
  before this moves any board status.

## Next rungs (ordered)

1. Wire the ledger into the live chat store behind `BRAIN_DA_TAG_CAPTURE` with a world-clock advance on the idle
   tick, and add a store / next-day / recall probe pair to the load-bearing battery.
2. A harder neutral drive (plain but engaging facts) and a gain-only lesion, to test the dynamics near the boundary.
3. The other late-phase routes (repetition-triggered PRP, sleep replay) so ordinary facts that matter are kept, and
   PRP competition between tagged synapses (Fonseca et al. 2004).
4. Move the per-synapse late-phase state into the spiking substrate's own synapse kernel.

## Corrections (2026-09-25 claim-check audit)

- Seed 101 "salient PRP max (intact)": 0.067 -> 0.066 (`gates.salient_p_max_intact` = 0.066491, from  <!--derived-->
  `research/findings/raw/_da_encoding_natural_drive_v3/seed101.json`). Does not change any verdict or the
  "0.056-0.070 intact" range quoted elsewhere in this document (0.066 is still inside that range, and no gate  <!--derived-->
  in `aggregate.json` reads this field).
