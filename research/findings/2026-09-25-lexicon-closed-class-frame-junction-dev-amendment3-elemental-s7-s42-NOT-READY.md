---
type: finding
status: live
date: 2026-09-25
lane: E · Language (learned referent lexicon, ready-to-switch-on)
mechanism: frame-junction referent lexicon with an ELEMENTAL partial-match edge beside the junction edge
  (BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL, default OFF; AMENDMENT 3 of
  research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md), dev check at seeds 7 AND 42
  with the redesigned G3' (two lesions that each remove drive)
seeds: [7, 42]
verdict: DEV CHECK at seeds 7 AND 42 (neither an evaluation seed), NOT READY for the six-seed evaluation. G4
  (silent-NON <= 0.30) and G1 R1-R4 now PASS at both seeds (R3 was 0.25 at seed 7 in round 2). G2 FAILS at both
  seeds, worse than round 2. G3' FAILS at both seeds: removing the elemental edge raises the silent-NON fraction as
  predicted (G3'b), but removing the conjunction leaves the parse unchanged (G3'a). The drive instrument shows why:
  on the queried battery words the elemental edge, meant to be the weaker vote, carries most of the afferent drive.
  Default-off holds, asserted in data. No six-seed run was staged; nothing was flipped.
runner: research/runners/_lexicon_closed_class_junction_dev.py
artifacts:
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/dev_s7_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/junction_elemental_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/route_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/and_population_trained_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/off_frame_s7.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/dev_s42_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/junction_elemental_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/route_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/and_population_trained_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/off_frame_s42.json
  - research/findings/raw/_lexicon_closed_class/a3_offidentity_s7.json
  - research/findings/raw/_lexicon_closed_class/frame_composition_s7_s42.json
  - research/findings/raw/_lexicon_closed_class/dev_s7_amendment2/dev_s7_result.json
  - research/findings/raw/_lexicon_closed_class/dev_s42_amendment2/dev_s42_result.json
seed-waiver: a declared dev-seed check whose headline is NOT READY; the six-seed evaluation is pre-registered
  separately (AMENDMENT 3's commands) and was deliberately not run.
builds_on:
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md
  - research/findings/2026-09-25-lexicon-closed-class-frame-junction-dev-amendment2-s7-s42-NOT-READY.md
---

# Frame-junction lexicon, AMENDMENT 3 dev check (seeds 7 + 42): the elemental edge restores recall, then takes over the decision

Branch `research/lexicon-closed-class-a3`. AMENDMENT 3 (commit `f055235f1`) was committed on its own before the
mechanism was built (`9a121952d`) and before any run it governs. Biology:
`research/biology/elemental-partial-match-beside-conjunction.md`.

Artifacts: `research/findings/raw/_lexicon_closed_class/dev_s7_amendment3/dev_s7_result.json` and
`research/findings/raw/_lexicon_closed_class/dev_s42_amendment3/dev_s42_result.json` (per-seed summaries), each
seed's `junction_elemental_s{seed}.json` (the three parse arms, per-word decisions and drives), `route_s{seed}.json`
(R1-R4), `and_population_trained_s{seed}.json` and `off_frame_s{seed}.json` (D0) in the same directories; the
flag-off identity smoke `research/findings/raw/_lexicon_closed_class/a3_offidentity_s7.json`.

## What was built

Beside the junction edge (every Amendment 2 constant unchanged), the two afferent blocks each junction reads,
FR(-1, a) and FR(+1, b), also project straight to the category pools CN0/CX0 through an ELEMENTAL edge (200 x 40
synapses, no STP). It is learned by the same Oja rule, jointly with the junction edge, at v2's own frame->category
constants, unscaled. The junction edge keeps its drive-matching boost; the elemental edge gets none. That was the
amendment's argument that the elemental vote would be the weaker one, with no new constant. G3 was replaced by G3':
`conjunctive` zeroes every FR->FJ weight, and `elemental` zeroes the elemental edge. Each removes an edge, and the
afferent drive into the pools is recorded per word and per edge. The VOID guard checks the arm-mean total drive
(summed over edges, averaged over queried words) against the intact arm, as AMENDMENT 3 specifies, so a lesion
cannot pass by raising that total. It does not check each word and edge separately: with no non-negativity floor on
the weights, one word's edge drive could rise toward zero under a lesion without tripping VOID (review 2026-09-25;
the measured drives here fall well clear of that boundary).

## Result (dev seeds only; not evidence for the six-seed gates)

<!--derived-->
| gate | bar | seed 7 | seed 42 | reads |
|---|---|---|---|---|
| default OFF, v2 (D0) | pinned hashes (7) / variant frame (42) | identical to pinned | frame, junction unimported; hashes equal round 2's | holds both |
| default OFF, junction (smoke 1) | pre-change == post-change, flag unset | identical (build, train, decide, settle) | -- | holds |
| D1 curriculum (no teacher, report only) | -- | 69/75, 1 abstain, 5 wrong (big, small, little, red, again) | 68/75, 3 abstain, 4 wrong (big, little, red, again) | round 2 had 0 wrong at both |
| D2 AND population (10,000, trained) | 0 violations | 0 | 1 | as round 2 |
| D2 `conjunctive` lesion | every junction silent to its pair | 10,000 / 10,000 silent | 10,000 / 10,000 silent | holds both |
| **G2** parse match (intact) | 0 mismatches | **24** (v2: 10; round 2: 5) | **9** (v2: 35; round 2: 3) | **FAILS both**, worse than round 2 |
| **G3'a** `conjunctive` raises mismatches | lesioned > intact, drive <= intact | 24 -> 24 (drive 19.23 -> 15.59) | 9 -> 9 (drive 29.13 -> 19.72) | **FAILS both** |
| **G3'b** `elemental` raises silent-NON | lesioned > intact, drive <= intact | 0.263 -> 0.614 (drive 19.23 -> 3.64) | 0.175 -> 0.4386 (drive 29.13 -> 9.41) | passes both |
| **G3'** (a AND b) | -- | FAIL | FAIL | **FAILS both**, not VOID (drive fell under both lesions) |
| **G4** silent-NON fraction | <= 0.30, v2 same seed | **0.263** (v2: 0.105; round 2: 0.596) | **0.175** (v2: 0.175; round 2: 0.4035) | **PASSES both** |
| G1 route R1 / R2 | owl on / off | pass / pass | pass / pass | holds both |
| G1 route R3 | recovered-both >= 0.50 | **0.75** (round 2: 0.25) | **0.833** (round 2: 0.50) | **PASSES both** |
| G1 route R4 | learned-edge lesion <= 0.20, lever moves | 0.0, moved | 0.0, moved | passes both (settle now scales both edges) |
| new noun forms recovered (report only) | -- | 27 (round 2: 19) | 25 (round 2: 11) | recall up both |

Drive numbers are arm means of the total afferent drive into both pools over the queried words (weight x
spikes/step, per pool neuron), `mean_afferent_drive` in each seed's `junction_elemental_s{seed}.json`.

## What the elemental edge fixed

- **G4.** Silent-NON fell from 0.596 to 0.263 (seed 7) and from 0.404 to 0.175 (seed 42); seed 42 now equals <!--derived-->
  v2 at the same seed. Removing the elemental edge again (G3'b) brings it back to 0.614 and 0.439, close to round 2. <!--derived-->
  So the added decisions come from the elemental edge, and the lesion that shows it removes drive rather than
  adding it. But G4 counts silence,
  not correctness. Of the NON words silent without the elemental edge and decided with it (20 at seed 7, 15 at
  seed 42), the intact circuit calls 8 and 4 of them nouns (seed 7: again, before, happened, happens, here, most,
  rises, screaming; seed 42: amazing, crazy, most, wonderful). Part of the G4 gain is wrong decisions, and those
  are the G2 failures below.
- **R3.** Held-out noun pairs recovered: 0.75 at seed 7 (round 2: 0.25) and 0.833 at seed 42 (round 2: 0.50). <!--derived-->
  At seed 42 the route runner rebuilt its own lexicon through the real `get_lexicon()` with both flags set
  (`route_lexicon_variant: junction_elemental`). R4 still reads 0.0 at both seeds, with the lever moved: the
  learned-edge lesion now resets both edges, and the untrained circuit abstains.
- **'might'** (round 2's G2 failure at both seeds) is no longer admitted at either seed. At seed 42 it abstains in
  the intact arm and is admitted again when the elemental edge is removed, so the elemental edge's non-noun evidence
  is what removed it (`queried['might']` in `junction_elemental_s42.json`). AMENDMENT 3 predicted this.

## What it broke: the "weaker" vote carries most of the drive

AMENDMENT 3 argued that the elemental vote would be the weaker component, because only the junction edge is
boosted by the drive-matching factor. Measured on the queried battery words, the elemental edge carries 81%
(seed 7: 15.59 of 19.23) and 68% (seed 42: 19.72 of 29.13) of the afferent drive into the pools. <!--derived-->
The junction edge carries 3.64 and 9.41. The boost S = 165.1 was measured on untrained curriculum presentations.
On battery words, after training, the junctions carry far less, while every occurrence with one or two frequent
neighbours drives the elemental edge. The weight balance the amendment took as fixed by construction was not.

The consequence is v2-like admissions:
- **'most'** (predicted in advance): admitted at both seeds, decided entirely by the elemental edge. Its elemental
  CN drive is 36.38 (seed 7) and 51.25 (seed 42) against CX 3.56 and 5.84; its junction drive is 0 at both seeds.
- **Attributive adjectives** 'crazy', 'wonderful' (both seeds) and 'amazing' (seed 42) are admitted. So are
  'again' and 'leaves' at both seeds, and 'before', 'happened', 'happens', 'here', 'rises', 'screaming' at seed 7.
  'leaves' is the verb in `tom_fb` ("Sally leaves the room"), so the false-belief turn now mismatches at both seeds.
  For every admitted NON word at both seeds, the elemental edge's CN drive alone exceeds the junction edge's total
  drive into both pools. The junction edge leans CX or is near zero for all of them except 'again' at seed 42
  (CN 2.29 vs CX 1.77) and 'east' at seed 7.
- **Curriculum words.** Without the teacher, the circuit now calls some of its own taught non-nouns nouns: the
  adjectives big, little, red and the adverb again at both seeds, plus small at seed 7 (round 2: none wrong).
- 'east' is still admitted at both seeds; the token tagger mistags it (declared residual, not a mechanism result).

## G3': the instrument separates the pathways, and says the conjunction is not doing the exclusion here

Both lesions lowered the drive at both seeds (the VOID condition did not fire), so neither reading can come from
added drive. The round-2 G3 could not say that. The readings:
- Removing the ELEMENTAL edge (junction edge alone, jointly trained): mismatches fall to 2 (seed 7, 'east' only)
  and 3 (seed 42, 'might' only), and silent-NON rises. The junction edge, trained beside the elemental edge, is still
  about as clean as round 2's circuit on G2 and just as silent.
- Removing the CONJUNCTION (elemental edge alone): the turn-level parse is unchanged at both seeds (24 and 9
  mismatches, the same turn labels). At the word level, seed 7 adds one admission ('amazing') and seed 42 none.

So in the combined circuit the elemental edge decides every admitted word, and the conjunction's non-noun evidence
is outvoted. G3'a fails for a reason the instrument makes visible, not because it is noisy.

## Stated in advance vs observed

- 'most' admitted by the elemental edge (26 of its 32 occurrences are left-only, 24 of those with 'the'):
  predicted, observed at both seeds. The per-synapse split within the elemental edge was not measured.
- Attributive adjectives at risk: predicted, observed ('crazy', 'wonderful' both seeds, 'amazing' seed 42).
- G4 might stay above 0.30 if the elemental vote were too weak: the opposite happened; it is strong enough to
  dominate.
- 'might' pushed toward CX: predicted, observed.

## What would be needed next (not built here; each needs its own amendment)

The wall question again: what does the real neuron run that this build replaced with a constant? Here it is the
split of the synaptic budget between the two inputs. Each edge has its own Oja normalisation (the junction edge's
scaled by S squared), so each is normalised to its own full scale independently. A real cell's synaptic resources
are shared, and heterosynaptic competition moves weight toward the inputs that predict its activity. Candidates:
1. **Cell-wide normalisation across both edges** (one Oja budget per postsynaptic neuron over junction AND elemental
   synapses): the edge that predicts the teacher better on familiar frames would win weight from the other. This is
   the most direct answer to the measured imbalance. It needs the pre-factor scaling of the two edges put on one
   footing first.
2. **Marr's gated elemental vote** (AMENDMENT 3's recorded next rung): a Golgi-like regulator that silences the
   single-input code whenever a complete frame is heard. It would stop the elemental edge outvoting the junction on
   complete frames. It would not fix 'most', whose occurrences are 26 of 32 left-only
   (frame_composition_s7_s42.json).
3. **Measure the drive balance on the trained circuit, not the untrained curriculum**, before any constant is
   frozen. S = 165.1 described the untrained curriculum. This round shows that the balance on the words that matter
   is a different quantity; the new drive instrument measures it directly.

The six-seed evaluation should not be staged until the seed-7 and seed-42 dev checks pass every bar.

## Six-seed evaluation commands (NOT run; as pre-registered in AMENDMENT 3)

    for S in 42 43 44 100 101 102; do
      bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy BRAIN_LEARNED_REFERENT_JUNCTION=1 \
        BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL=1 .venv/bin/python -u -m \
        research.runners._lexicon_closed_class_parse_diag --seed $S --corpus data/corpus/tinystories.txt \
        --lesions none,elemental,conjunctive \
        --json research/findings/raw/_lexicon_closed_class/eval_a3/junction_elemental_s$S.json
      bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy BRAIN_LEARNED_REFERENT_JUNCTION=1 \
        BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL=1 .venv/bin/python -u -m \
        research.runners._d6_learned_referent_env_flag_derisk --seed $S --corpus data/corpus/tinystories.txt \
        --json research/findings/raw/_lexicon_closed_class/eval_a3_route/s$S.json
    done
    .venv/bin/python -m research.runners._d6_learned_referent_env_flag_derisk \
        --score research/findings/raw/_lexicon_closed_class/eval_a3_route
    .venv/bin/python -m research.runners._lexicon_closed_class_parse_diag \
        --score research/findings/raw/_lexicon_closed_class/eval_a3 \
        --route research/findings/raw/_lexicon_closed_class/eval_a3_route

Dev reproduction (both seeds; two at once at most):

    bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy .venv/bin/python -u -m \
        research.runners._lexicon_closed_class_junction_dev --elemental --seed 7 \
        --corpus data/corpus/tinystories.txt --out research/findings/raw/_lexicon_closed_class/dev_s7_amendment3
    (the same with --seed 42 and --out .../dev_s42_amendment3)

Cost at these seeds: build + train 5,282 s (seed 7) and 5,257 s (seed 42); the whole dev check 10,420 s (seed 7)
and 15,855 s (seed 42, which trains a second lexicon for the route); peak RSS about 600 MB each.

Provenance: both runs loaded their code at commit `ea37170e6` (the mechanism and instruments). A later commit
(`668cc8b0a`) changed only the dev script's file-writing helper while seed 42 was mid-run; a running process does
not reload it. Their `git_dirty: true` reflects untracked scratch files only. The dev script writes into
an `--out` directory, which the provenance door does not sidecar, so the sidecars were backfilled from each run's
own runs.jsonl start record (`_lexicon_closed_class_a3_backfill_prov.py`, each marked `backfilled`). The script
now declares each file it writes (research/FAILURE_LOG.md, 2026-09-25 row).

## Honest residuals

- Everything round 2 declared still holds (host-designed junction wiring, a somatic AND on a point neuron,
  teacher-driven curriculum, host read-out, noun-hood not referent-hood, type- and token-level tagger residuals).
- The elemental and junction pathways are two populations in a point-neuron circuit, not two integration modes of
  one dendrite.
- The drive instrument is an afferent-drive estimate (presynaptic rate x installed weight, pool mean); it is not
  the pools' membrane current and ignores the inhibitory competition. It was used only for the G3' drive condition
  and the pathway split above.
- G4's bar (0.30) and G2's zero-mismatch bar are unchanged. G4 measures silence only; read it with G2 (see "What
  the elemental edge fixed").

## Honesty

Functional read-outs only. Two dev seeds, neither an evaluation seed; nothing here is an evaluation result.
