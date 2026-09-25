---
type: finding
status: live
date: 2026-09-25
lane: E · Language (learned referent lexicon, ready-to-switch-on)
mechanism: frame-junction referent lexicon (BRAIN_LEARNED_REFERENT_JUNCTION, default OFF), AMENDMENT 2 --
  short-term depression on FR->FJ, a runner-side homeostatic settle for the learned_edge lesion, a
  sentence-boundary pause token, token-level ground truth, per-word margins, a drive-matched OR control, a
  silent-NON gate (G4). research/runners/lexicon_frame_junction.py, dev check at seeds 7 AND 42 under
  research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md (AMENDMENT 2)
seeds: [7, 42]
verdict: DEV CHECK at seeds 7 AND 42 (neither an evaluation seed), NOT READY for the six-seed evaluation.
  Default-off holds (byte-identical to the pinned hashes at seed 7). Mechanism A (short-term depression) and
  mechanism B (the R4 homeostatic settle) are CONFIRMED FIXES: AND-population violations 102/10,000 (round 1)
  -> 0/10,000 (seed 7) / 1/10,000 (seed 42); R4 recovered-both-rate 0.33333 (round 1, FAIL) -> 0.0 (both seeds,
  PASS). But G2 (0 mismatches) still fails at both seeds on ONE word ('might', a modal), G3's direction is
  INCONSISTENT across the two dev seeds (moves as expected at 42, moves BACKWARD at 7), the new G4 (silent-NON
  <=0.30) fails at both seeds and reads WORSE than v2 at both, and G1's R3 (population recall >=0.50) fails at
  seed 7 (0.25) while exactly clearing the bar at seed 42 (0.50). No six-seed run was staged; nothing was flipped.
runner: research/runners/_lexicon_closed_class_junction_dev.py
artifacts:
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/dev_s7_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/junction_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/route_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/and_population_trained_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/off_frame_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/dev_s42_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/junction_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/route_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/and_population_trained_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/off_frame_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s7/route_s7.json
seed-waiver: a declared dev-seed check whose headline is NOT READY; the six-seed evaluation is pre-registered
  separately and was deliberately not run.
builds_on:
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-dev-s7-not-ready.md
---

# Frame-junction referent lexicon, AMENDMENT 2 dev check (seeds 7 + 42): mechanisms A and B confirmed fixed, G2/G3/G4/R3 still not ready

Artifacts: `research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/dev_s7_result.json` (seed 7 summary),
`research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/dev_s42_result.json` (seed 42 summary), and each
seed's own `junction_s{seed}.json` (D3 parse arms), `route_s{seed}.json` (D4 route R1-R4) and
`and_population_trained_s{seed}.json` (D2, all 10,000 junctions) under the same directories.

## Result (dev seeds only; not evidence for the six-seed gates)

<!--derived-->
| gate | bar | seed 7 | seed 42 | reads |
|---|---|---|---|---|
| default OFF | pinned hashes (7) / variant==frame (42) | identical | frame, unimported | holds both |
| D1 curriculum (no teacher, report-only) | -- | 69/75 correct, 6 abstain, 0 wrong | 71/75 correct, 4 abstain, 0 wrong | holds |
| D2 AND smoke (256, trained) | lone 0, pair fires | holds | holds | holds both |
| D2 AND population (10,000, trained) | 0 violations | 0 | 1 (lone_left=1, lone_right=1) | near-perfect both; PASS at 7 |
| G2 parse match (intact) | 0 mismatches | 5 (offending: east, might) | 3 (offending: might) | FAILS both -- 'might' at both |
| G3 coincidence lesion moves parse | lesioned > intact | 5 -> 1 (WRONG direction) | 3 -> 6 (right direction) | inconsistent across seeds |
| G3 coincidence_matched (honest-negative control) | n/a | 5 -> 5 (unchanged) | 3 -> 3 (unchanged) | confirms AND barely perturbed |
| G4 silent-NON fraction (new) | <=0.30, vs v2 same seed | 0.5965 (v2: 0.10526) | 0.4035 (v2: 0.17544) | FAILS both, worse than v2 both |
| G1 route R1 / R2 | owl on / off | pass / pass | pass / pass | holds both |
| G1 route R3 | recovered-both >= 0.50 | 0.25 | 0.50 | FAILS at 7, clears at 42 |
| G1 route R4 | learned-edge lesion <= 0.20, lever moves | 0.0, moved | 0.0, moved | **PASSES both** (was 0.33333 FAIL) |
| attribution: fraction of v2 mismatches removed | -- | 50.0% (v2 had 10) | 91.4% (v2 had 35) | large reduction both |

## Mechanism A (short-term depression): CONFIRMED FIX

The AND-population integrity check on the TRAINED circuit (all 10,000 junctions, not a sample) reads 0 violations
at seed 7 and 1 at seed 42 (`and_population_trained_s{7,42}.json`) -- against round 1's 102/10,000 (pre-STP) and
the pre-STP grid's best-ever 47/10,000. The single seed-42 violation is one junction firing to BOTH a lone left
AND a lone right afferent (not the 'day'-column pattern round 1 found); not investigated further at dev scale.
The re-calibrated constants (W_J 300->2950, I_TONIC_J -650->-762.5, DRIVE_MATCH_S 27.53->165.1) hold up on the
TRAINED circuit, not just the untrained calibration `_Env()` they were chosen on.

## Mechanism B (R4 homeostatic settle): CONFIRMED FIX

R4's recovered-both-rate is 0.0 at BOTH seeds (round 1: 0.33333, FAIL against the <=0.20 bar) and the lesion lever
records as MOVED at both (`lever` seed 7: 0.25 -> 0.0; seed 42: 0.5 -> 0.0). The uniform-jittered start weights,
once settled by one epoch of runner-side Turrigiano-style scaling, no longer decide held-out words at random --
if anything the settled circuit now UNDER-recovers relative to intact (0.0 vs the intact rate), which is the
correct direction for "an untrained circuit should abstain, not guess."

## G2 (0 mismatches): fails at both seeds on the SAME word

'might' (a modal, MD under the token-level tagger) is admitted as a noun referent at both seeds, with a REAL
margin, not a boundary artifact: seed 7 rate_cn=0.00253 vs rate_cx=0.000188 (margin 0.00234, `near_boundary` --
within 1.5x DEAD_MARGIN); seed 42 rate_cn=0.00872 vs rate_cx=0.00475 (margin 0.00397, NOT near-boundary -- a
clearer admission). This is a genuine, cross-seed-consistent mechanism residual the token-level ground-truth fix
newly exposed (round 1's less accurate instrument never flagged 'might', since modals fell to UNKNOWN under the
old two-map lookup). Seed 7 additionally admits 'east', the tagger's own declared residual (mistags it RB in
"the sun rises in the east"), not a new mechanism failure.

## G3 (coincidence lesion moves the parse): direction is INCONSISTENT across the two dev seeds

At seed 42 the lesion moves as pre-registered (3 -> 6 mismatches, matching AMENDMENT 1's rule). At seed 7 it
moves BACKWARD (5 -> 1): the OR lesion's ~380x drive increase over intact (measured during `OR_MATCH_FACTOR`
calibration) does not cleanly "add more admissions" -- it floods the whole population, and which words end up
mismatching shifts unpredictably rather than monotonically increasing. The `coincidence_matched` honest-negative
control (unchanged at both seeds, 5->5 and 3->3) is consistent with what AMENDMENT 2 already declared: the ONLY
drive-matched factor (1.02) barely perturbs the AND, so it cannot by itself validate or refute "does removing the
conjunction specifically drive the effect" -- and seed 7's result shows the RAW 2x lesion is not a clean enough
instrument for that question either. G3 as currently specified (a single fixed 2x OR factor, PASS iff
`lesioned > intact` on 5 of 6 EVALUATION seeds) has a real chance of failing on directional grounds alone, not
just magnitude.

## G4 (new, silent-NON fraction <=0.30): fails at both seeds, and is WORSE than v2 at both

Junction-intact silent-NON fraction is 0.5965 at seed 7 (34/57 heard NON words silent in both pools) and 0.4035 at
seed 42 (23/57), against v2's OWN reading at the SAME seeds (0.10526 and 0.17544). Both seeds worsen over v2 by a
similar multiple (~5.7x at seed 7, ~2.3x at seed 42) despite the mechanism otherwise removing 50-91% of v2's
closed-class admissions -- the AND's extra selectivity (a junction needs BOTH neighbours heard together, and few
NON-class words share a heard (-1,+1) frame with anything) trades precision for a large increase in "no signal at
all" on NON words specifically, not just a shift toward correct abstention on borderline ones.

## G1 route R3 (population recall): fails at seed 7, clears the bar exactly at seed 42

Recovered-both-rate over 12 held-out noun pairs: 0.25 at seed 7 (3/12), 0.50 at seed 42 (6/12, exactly at the
bar). The hand-baseline (flag-off) reads 0.0 at both seeds (these pairs are held out of the hand list by
design, so this is not a regression against production). The AND's extra selectivity plausibly explains this
too: a held-out noun needs a heard (-1,+1) frame with two of the 100 most-frequent context words (one of which
is now PAUSE_TOKEN, displacing 'make' -- see AMENDMENT 2's addendum), a stricter requirement than v2's
single-offset OR-like evidence.

## Honest residuals carried from AMENDMENT 2, unchanged by this run

The token-level tagger mistags 'east'; PAUSE_TOKEN displaces 'make' from the top-100 context words (declared,
not re-measured here since it's a fixed corpus-level fact, seed-independent); `admitted_margins`' `gt_class`
field uses the TYPE-level classifier for its diagnostic label even though the actual G2 adjudication uses the
more accurate per-turn TOKEN-level one (a minor reporting inconsistency, not a gate-correctness bug -- 'might'
and 'east' show `gt_class: UNKNOWN` in the margins table above despite being correctly caught as NON by the real
per-turn adjudication; noted here so a reader of the raw JSON is not confused, not fixed in this lane).

## What would be needed next (not built here)

- **G2 'might' (and modals generally):** the frame-junction AND is not sufficient on its own to exclude modals;
  check whether MD-tagged words share enough heard frames with the noun curriculum to need a dedicated cue
  (frequency-based, per Hochmann et al. 2010) rather than relying on the conjunction alone.
- **G3 instrument:** a single fixed OR factor cannot cleanly separate "conjunction removed" from "drive
  changed" (declared in AMENDMENT 2) AND does not even move monotonically across seeds (found here). A next
  amendment needs either a per-seed drive-matched factor (not just seed 7's) or a structurally different
  lesion (e.g. zeroing one afferent's synapse only, leaving the other, rather than boosting both).
- **G4 / R3 (the precision-recall trade):** the AND's extra selectivity that suppresses NON-word admissions
  also suppresses genuine noun recall and leaves more NON words with no signal at all. The companion process to
  look for (per the wall reframe) is what real word-learning runs to keep partial/weak frame evidence USABLE
  rather than silent -- e.g. a partial-match pathway for a heard neighbour on ONE side only, gated by confidence,
  not the strict two-input AND alone.

Each is a new method and needs its own AMENDMENT, committed before any run it governs. The six-seed evaluation
should not be staged until the seed-7 AND seed-42 dev checks pass every pre-registered bar.

## Honesty

Functional read-outs only; two dev seeds, neither an evaluation seed; nothing here is an evaluation result.
