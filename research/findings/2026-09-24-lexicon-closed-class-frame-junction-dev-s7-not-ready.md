---
type: finding
status: live
date: 2026-09-24
lane: E · Language (learned referent lexicon, ready-to-switch-on)
mechanism: frame-junction referent lexicon (BRAIN_LEARNED_REFERENT_JUNCTION, default OFF),
  research/runners/lexicon_frame_junction.py, dev check at seed 7 under
  research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md (AMENDMENT 1)
seeds: [7]
verdict: DEV CHECK at seed 7 (not an evaluation seed), NOT READY for the six-seed evaluation. Default-off holds
  (byte-identical to the pinned hashes). The junction lexicon removes 5 of the 7 seed-7 parse mismatches, but three
  pre-registered checks would fail - the parse gate ('most' twice), the route lesion gate (R4 0.333 > 0.20) <!--derived--> and
  the AND integrity smoke (102 of 10,000 junctions). No six-seed run was staged; nothing was flipped.
runner: research/runners/_lexicon_closed_class_junction_dev.py
artifacts:
  - research/findings/raw/_lexicon_closed_class/dev_s7/dev_s7_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s7/junction_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7/route_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7/off_frame_s7.json
  - research/findings/raw/_lexicon_closed_class/and_population_s7.json
  - research/findings/raw/_lexicon_closed_class/diag_frame_s7_gt3.json
seed-waiver: a declared dev-seed check whose headline is NOT READY; the six-seed evaluation is pre-registered
  separately and was deliberately not run.
builds_on:
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md
---

# Frame-junction referent lexicon, seed-7 dev check: most closed-class admissions are gone, but not ready

## Result (seed 7 only; not evidence for the six-seed gates)

Artifacts: research/findings/raw/_lexicon_closed_class/dev_s7/dev_s7_result.json (summary),
research/findings/raw/_lexicon_closed_class/dev_s7/junction_s7.json (parse arms),
research/findings/raw/_lexicon_closed_class/dev_s7/route_s7.json (route R1-R4),
research/findings/raw/_lexicon_closed_class/dev_s7/off_frame_s7.json (default-off parse),
research/findings/raw/_lexicon_closed_class/and_population_s7.json (the AND on all junctions). Runner:
`research/runners/_lexicon_closed_class_junction_dev.py` at `e5f949ae7` (clean tree).

<!--derived-->
| check | pre-registered bar | seed 7 | reads |
|---|---|---|---|
| default OFF, asserted in data | parse and decision hashes equal the pinned pre-change ones | identical; junction module not imported | holds |
| G2 parse match (junction, intact) | 0 mismatching turns | 2 ('most' in datc_t5 and datci_t5) | would fail |
| G3 coincidence lesion moves the parse | lesioned > intact | 2 -> 6 (adds 'who' on dr_c, dr2_c, wmb_ask, wmb1_ask) | moved |
| G1 route R1 / R2 | 'owl' routed on / not off | pass / pass | holds |
| G1 route R3 | recovered-both >= 0.50 | 0.667 | holds |
| G1 route R4 | learned-edge lesion <= 0.20 | 0.333 (hand path 0.0) | would fail |
| AND integrity smoke (256 sampled junctions) | lone afferent 0 spikes, pair fires | 3 junctions fire to a lone right afferent | fails |

Against the single-offset lexicon at the same seed (`diag_frame_s7_gt3.json`, 7 mismatching turns), the junction
lexicon removes 5. `attributable_to` puts 71.4% of v2's mismatch count on the change of lexicon. 'when',
'wonderful' and 'crazy' are no longer admitted; 'who', 'what', 'before' and 'today' abstain (v2 at seed 42 admitted
the last four). 25 hand-table-missing nouns are still recovered (v2: 26). 'circus' (heard 3 times) is lost. 'bite'
is newly admitted in "what does the wolf bite", where it is a verb: its word form is a noun by dominant POS, the
declared type-level residual. 'east' is admitted and stays UNKNOWN. On `tom_fb`, 'anne' is still displaced by the
5-referent cap (marble, basket, sally, leaves, room), as with v2. Curriculum words without the teacher: 67 of 75
correct, 8 abstain, 0 wrong. Build + training took 1099 s on numpy; peak RSS was 587 MB.

## Why each failing check fails (what the data say)

<!--derived-->
**'most'.** (Host tally of its 32 sampled occurrences at seed 7, `FrameEnvironment.occurrences('most', 32, 7)`,
not saved as an artifact.) 24 of the 32 are "the most" + a word outside the 100 frame words ("the most
beautiful"), so no junction fires. The remaining junction evidence comes from sentence-final uses. "... the most."
followed by "Tom ...", "They ...", "... was": the environment strips punctuation, so these read as the noun frames
(the, tom), (the, they), (the, was). CN rate 0.0051 against CX 0.0 clears MIN_RATE 0.002 (owl: 0.027). The frame is
right that "the most." is a nominal use. The residual is a sentence boundary the heard input does not carry, plus
a decision rule that counts six noun-framed occurrences out of 32 as a referent.

**R4 lesion 0.333.** The learned-edge lesion restores the uniform, 10%-jittered start weights (at the drive-matched
scale). One junction fires per occurrence, so a word's drive comes from a few junctions. The jitter does not
average out; it decides some words for CN at random instead of leaving them to abstain. v2's lesion read 0.0.

**AND smoke.** `and_population` (new, all 10,000 junctions at once: junctions have no lateral or feedback input) finds
102 violations at the frozen W_J 300, I_TONIC_J -650. 100 of them are one whole column: the junctions whose right
input is context word 12 ('day'). That afferent fires at 0.233 spikes/step against a median of 0.147 (neuron
heterogeneity), so it drives its junctions alone. The other 2 are silent pairs. No point of the committed grid (W_J 250-500 x I_TONIC_J
-550 to -1000) has zero violations; the best is 47. The 64-junction calibration sample could not see an
afferent-level column. That lapse is logged in research/FAILURE_LOG.md.

## What would be needed next (not built here)

- **AND robust to afferent rate**: short-term depression at the frame -> junction synapses, so a junction responds to
  input onset and coincidence, not to how fast one afferent fires (depressing synapses make the response nearly
  rate-independent at high rates; Abbott et al. 1997, Science 275:220 <!--derived-->). The engine has per-synapse STP flags
  (`stp_disabled` in explicit wiring). Re-select the operating point with `and_population` over all junctions.
- **'most'**: render the sentence boundary the listener hears (a prosodic pause afferent, a legitimate environment
  cue) instead of stripping punctuation. Separately, a decision that weighs how CONSISTENTLY a word's occurrences
  carry noun frames, not the summed drive. The first is a change to the environment; the second needs biology
  first (evidence accumulation), per the wall rule.
- **R4**: the untrained sparse circuit should abstain, not guess. The companion process to look for is what the
  real system runs to keep an untrained category circuit silent. Candidates: homeostatic scaling of each pool's
  total input, or a divisive confidence read.

Each is a new method and needs its own AMENDMENT, committed before any run it governs. The six-seed evaluation
should not be staged until the seed-7 dev check passes every pre-registered bar.

## Honesty

Functional read-outs only; a single dev seed; nothing here is an evaluation result.
