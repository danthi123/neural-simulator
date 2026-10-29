---
type: finding
status: qualified
date: 2026-10-27
lane: substrate (integrator validity, mechanism c); follow-up of research/findings/2026-10-27-stable-preset-centred-dev7.md (the region-preset switch did not reach the 110 association-graph cells because they are typed by trait)
mechanism: SIM_TRAIT_CENTRED_HET=1 (sim/bridge.py TRAIT_CENTRED_HET_ENV) forces cfg.trait_centred_parameter_heterogeneity=True onto every SimulationBridge; default unset = unchanged
flip: no - default OFF; it removes the mechanism-c cells it was built for, changes the word order of the replies, fails the association graph's own parity gate on all 8 dev seeds (5 of 8 pass without it) and lowers the Broca slot bridge's own de-risk from 7 of 8 to 4 of 8 seeds (3 pass-to-fail, 0 fail-to-pass: not separable at n = 8)
seeds: [7, 1, 2, 3, 4, 5, 6, 8]
runner: research/runners/membrane_census.py
artifacts:
  - research/findings/raw/_stable_mechc/compare_s7_trait.json
  - research/findings/raw/_stable_mechc/compare_s7_preset_trait.json
  - research/findings/raw/_stable_mechc/compare_s7_preset_vs_preset_trait.json
  - research/findings/raw/_stable_mechc/compare_s7_trait_only_broca.json
  - research/findings/raw/_stable_mechc/compare_s7_trait_only_novelty.json
  - research/findings/raw/_stable_mechc/census/stable_trait/s7/conv__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_trait/s7/infer__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_trait/s7/ask__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_trait/s7/emo__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_trait/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/conv__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/infer__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/ask__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/emo__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/census/trait_only_broca/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/census/trait_only_novelty/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/lag_gate/off.json
  - research/findings/raw/_stable_mechc/lag_gate/trait.json
  - research/findings/raw/_stable_mechc/lag_gate/stable.json
  - research/findings/raw/_stable_mechc/lag_gate/stable_trait.json
  - research/findings/raw/_stable_mechc/lag_gate/g3_off.json
  - research/findings/raw/_stable_mechc/lag_gate/g3_trait.json
  - research/findings/raw/_stable_mechc/lag_gate/g3_stable.json
  - research/findings/raw/_stable_mechc/lag_gate/g3_stable_trait.json
  - research/findings/raw/_stable_mechc/census/stable/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_unset_check/s7/inner__intact_a.json
---

seed-waiver: a labelled dev-seed diagnostic (dev seed 7 for the five-group census, dev seeds 1 to 8 for the association graph's and the Broca slot bridge's own gates); no gate seed was run, no default was flipped and no capability claim is made.

# Centring trait-typed cells on their own preset removes the cells below -110 mV and breaks the replies' word order and the association graph's parity gate (dev seed 7 and dev seeds 1 to 8, 2026-10-27)

## Answer in one paragraph

The variant built here (`SIM_TRAIT_CENTRED_HET=1`, config flag `trait_centred_parameter_heterogeneity`, default OFF, the unset
path equal by hash to the engine before the edit) centres the heterogeneity draw of Izhikevich cells typed by trait on their
own trait's preset, which is the method the preset finding named for the 110 association-graph cells that stay below -110 mV
on the stable integrator. On the five Stage-2 groups (dev seed 7, stable integrator) it does that: the 110 `LearnedAssocGraph`
offenders fall to 4 (every FS and thalamic-reticular cell stops leaving the range), the brain's u-held cells go from 112 to 6,
and no cell of the organs it moved newly leaves the range (one cell-count exception sits outside them: the `BridgeParser`
phasor slice gains 4 in infer, 718 -> 722, path untraced; Result 1). It is not a flip candidate, and it answers why the region
flag left these cells alone. It moves three organs' cells at once, and each has its own cost. What each organ does alone was
measured on the inner group only, with the variant on for that organ's bridges alone: the two Broca frame-slot bridges alone
change 7 of the 10 answer texts and turn the `fluent_hypothesis` flag true -> false in 4 turns, and change nothing else (11 of
2274 leaves; the replies lose the article's place and the inflection: "the lynx the chases vole", "perhaps lynx chase vole"),
with no turn changing abstention, recall or derivation, so the answer rate cannot see it; the novelty-habituation bank alone
changes no answer text and moves the DA drive (194 of 2274 leaves: `da_level` by up to 5.0 times, the DA `mode` in 4 turns).
With the variant on everywhere, over all five groups, 32 of 63 turns have a different answer text (25 change word order or
inflection), `da_level` moves by up to 5.8 times and the DA `mode` in 19 turns. The association graph's parity gate (G1), run
on dev seeds 1 to 8 against the stable integrator without the variant, goes from 5 of 8 seeds to 0 of 8 (5 pass-to-fail flips,
0 fail-to-pass; the oracle edges recovered, summed over seeds, fall 192 -> 85 of 192, and are lower on every seed), and the
Broca bridge's de-risk (G3) from 7 of 8 to 4 of 8 (3 pass-to-fail, 0 fail-to-pass; the D cue-recall gate G2 goes 1 of 8 -> 0
of 8, a single flip on a failing baseline): at n = 8 the pass counts cannot separate the arms on G2 or G3. The three organs'
effects add exactly (the 205 changed leaves of the inner group are the disjoint union of the Broca-only 11 and the
novelty-only 194). These organs were built and validated on the RS-centred hybrid, so removing the hybrid is a change of their
dynamics, not a repair. The 4 RS-preset offenders that remain are a separate residual.

## What was built (default OFF)

- Switch and flag. `CoreSimConfig.trait_centred_parameter_heterogeneity` (default False) and `SIM_TRAIT_CENTRED_HET=1`
  (`sim/bridge.py` `TRAIT_CENTRED_HET_ENV`, same style as `SIM_PRESET_CENTRED_HET`: unset, "" or "0" touch nothing, "1" forces
  the flag on at construction and at every `_initialize_simulation_data`, any other value raises). Read-back on the bridge:
  `trait_centred_het_requested`, `_applied`, `_organ_value` and `_readback` (whether the last draw was trait-centred, how many
  neurons are trait-typed with no region preset, how many of those sit off their distribution's centre, per parameter).
- What it does. Draws the same samples in the same order (RNG consumption unchanged: the global RNG state after a build is
  identical with the variant on and off). For an Izhikevich neuron typed by trait (`cfg.num_traits > 1`, so the init gave it one
  of the defined IZH2007 presets) that no region preset covers, the sample becomes a ratio against its distribution's centre
  (the engine's own `_recentre_het_draw_host`, which the region flag uses) times the preset value of the neuron's own trait,
  read from the arrays the init has just filled. A trait whose preset already sits on the centre (the RS trait) keeps today's
  sample bit for bit. It applies to the global draw, the per-region mask path and the region-scoped redraw. Izhikevich only; a
  pool with `num_traits == 1` (every neuron the default type, the separate case C2.2 of the census) and HH pools are not reached.
- Why the original kept these neurons on the global centre, and how this variant answers it.
  `docs/drafts/preset-centred-het/CENSUS.md` (C2.1) records the choice: the region flag keeps trait-typed neurons on the global
  centre "on purpose: the requirement is that neurons with no REGION preset stay identical with the flag on and off", and the
  ruling that followed (preset-het-1020b) makes every flip an organ-by-organ decision behind its own six-seed gate, because a
  flip changes that organ's dynamics. The variant keeps that requirement true instead of reversing it: it is a separate flag
  and switch on a disjoint neuron set (a neuron in a region with an `izh_neuron_type` is never touched by it; a trait-typed neuron
  is never touched by the region flag), so with the region flag alone every trait-typed neuron is identical to flag off, which
  is pinned by a test, and with both on the result is the region flag's on region neurons and the variant's on trait neurons. It
  is default OFF with the unset path byte-identical, shown by hash against a golden captured on the unedited bridge before the
  edit, and at brain level: the inner group re-run at 72bc75e7c with both new switches unset reproduces the earlier stable census
  (`census/stable/s7`, taken before the edit) turn for turn: the 10 per-turn response hashes, the all-response hash and the
  out-of-range and spike totals of all 26 organ rows are equal
  (`research/findings/raw/_stable_mechc/census/stable_unset_check/s7/inner__intact_a.json`). It does not hide the blast radius: the census read-back lists, per organ, how many trait-typed cells it moved, and the
  gates of the organs it moved were run (Results 2 and 3), which is the evidence the ruling asks for before any flip.
- Tests. `tests/test_trait_centred_heterogeneity.py` (47 passed): the four unset spellings reproduce the pre-edit golden
  (`tests/_trait_centred_het_golden.json`, 7 cases: the LearnedAssocGraph bridge at two sizes, a plain pool, a typed region
  beside an untyped one, the region-scoped path, the global-OFF mask path and a single-type pool; every array, the thresholds and
  the RNG state hash equal); on equals the engine's own recentring rule applied to the off bridge's draw, neuron for neuron;
  RS-trait neurons are bit-identical; region neurons are untouched by the variant and trait neurons by the region flag; single-type
  and HH pools are not reached; re-initialisation re-asserts it. Mutating the engine (collecting the centres but not merging them,
  or dropping the region-coverage exclusion) fails 7 and 3 of them. `tests/test_membrane_census.py` 26 passed with the trait
  read-back, `tests/test_stable_preset_compare.py` 6, `tests/test_stable_mechc_lag_gate.py` 4,
  `tests/test_stable_mechc_trait_attrib.py` 3.
- Related tests, base against branch (re-run for this note on numpy CPU, one process; base = 3c902abb1, the commit before the
  `sim/` edit; branch = the tip of `research/stable-mechc`). The failing set is identical on both. The 17 CPU-runnable files
  of the 18 that the names select (list in Sources) give 283 passed, 7 skipped, 0 failed on the branch and 226 passed, 7
  skipped, 0 failed on the base; the 57 extra passes are the new test files (47 + 4 + 3) and the tests added to
  `test_membrane_census.py` (26 against 25) and `test_stable_preset_compare.py` (6 against 4). The 18th file,
  `tests/test_determinism.py`, needs a CUDA device when it is collected and was not run. A wider sweep of the 38 test files
  that mention heterogeneity, the census or the stable integrator fails the same 28 tests and the same one collection error on
  base and branch (450 against 393 passed); it is order-dependent (its 18 `test_membrane_census.py` failures pass in the
  17-file run) and its failures were not analysed. A review run over 18 related test files reported 289 passed and 5
  pre-existing failures, the failing set identical on base and branch; that figure is quoted from the review, its file list
  was not recorded, and it was not reproduced here.
- Census and comparison (observe-only). `research/runners/membrane_census.py` records the trait read-back per bridge and per
  organ; `research/runners/stable_preset_compare.py` adds the per-type split of the u-held cells, the trait fields and a leaf diff
  of the two shard records (every number, string and flag of the stage-2 worker output that differs, with the largest relative
  change per read-out). `research/runners/stable_mechc_lag_gate.py` runs the association graph's own gates and the Broca
  slot bridge's de-risk; `research/runners/stable_mechc_trait_attrib.py` runs a census group with the variant on for one
  organ's bridges only.

## Result 1: the five Stage-2 groups, STABLE vs STABLE + TRAIT (census, dev seed 7)

`stable` = `SIM_STABLE_INTEGRATION=1`; `stable_trait` = both that and `SIM_TRAIT_CENTRED_HET=1`; the five Stage-2 groups of the
live `brain_chat` handler at production defaults under the gate's frozen environment, numpy CPU, intact_a, dev seed 7, each arm in
a fresh subprocess under `tools/memcap.sh 10`. The `stable` arm is the census of the preset-centred finding (same records,
`research/findings/raw/_stable_mechc/census/stable/s7/`). HEAD moved from 19fda7fc4 to 65d60f996 while the arms ran
(research files only; `sim/` is unchanged since bd13ab71c), which is why the records' `git_sha` differ between groups.

| group | out-of-range neurons stable / +trait | u-held b < 0 cells stable / +trait | trait-typed cells centred (off their centre) | turns with a changed response hash | turns with a changed answer text | turns with a changed abstention, recalled fact or derived flag | answered turns stable / +trait | gate scorer `all` |
|---|---|---|---|---|---|---|---|---|
| conv | 803 / 690 | 112 / 6 | 5688 (4497) | 11 of 11 | 9 | 0 | 6 of 11 / 6 of 11 | false / false |
| infer | 830 / 728 | 112 / 6 | 5688 (4497) | 10 of 10 | 8 | 0 | 8 of 10 / 8 of 10 | false / false |
| ask | 794 / 688 | 112 / 6 | 5688 (4497) | 18 of 19 | 6 | 0 | 4 of 19 / 4 of 19 | false / false |
| emo | 110 / 4 | 110 / 4 | 5320 (4197) | 8 of 13 | 2 | 0 | 5 of 13 / 5 of 13 | true / true |
| inner | 787 / 678 | 112 / 6 | 5688 (4497) | 10 of 10 | 7 | 0 | 8 of 10 / 8 of 10 | false / false |

Every turn that answered under stable answered under +trait and every abstention stayed an abstention, in all five groups.
Wall time is within 0.87 to 1.10 of stable per group.

What moved, by organ (cell counts and offender counts are identical in every group that steps the organ; the spike and
membrane figures of the last two rows are from the inner group):

| organ | trait-typed cells (off their centre) | out-of-range cells stable -> +trait | u-held b < 0 cells stable -> +trait | spikes stable -> +trait | lowest membrane stable -> +trait |
|---|---|---|---|---|---|
| `LearnedAssocGraph` (all five groups) | 1800 (1407) | 110 -> 4 | 110 -> 4 | 125094 -> 103901 | -166.53 -> -122.4 mV |
| `SpikingNoveltyHabituationOrgan` bank (all five) | 3520 (2790) | 0 -> 0 | 0 -> 0 | 1393 -> 2773 (inner) | -80.0 -> -85.72 mV (inner) |
| Broca frame-slot bridges, `slot_pool_rates` (conv, infer, ask, inner) | 368 (300) | 0 -> 0 | 0 -> 0 | 17108 -> 20427 (inner) | -105.8 -> -90.24 mV |
| thought-swap workspace (`_gnw_active_overwrite_derisk`, `num_traits = 1`) | not reached | 2 -> 2 | 2 -> 2 | - | - |

- The cells the variant was built for are gone, but not all of them. Of the 110 `LearnedAssocGraph` offenders, 58 thalamic-reticular
  and 30 FS cells no longer leave the range and the RS-preset ones (22 to 24 in the stable arm) stay as 4 in every group; with the
  2 workspace cells that is 6 u-held cells (4 in emo, where the workspace contributes none). The totals fall by the association graph's 106 and by the drift of the resonate-and-fire
  phasor slice (`BridgeParser`: 691 -> 684 conv, 718 -> 722 infer, 682 -> 682 ask, 675 -> 672 inner; as in the preset finding, not
  membranes, with no heterogeneity draw in that bridge). The brain census reproduces the build-only probe of the preset finding
  (92 to 115 down to 3 to 9 of 1800 on four dev seeds) with 110 down to 4. The lowest remaining membrane is still -122.4 mV: those
  cells are RS cells, which the rule cannot move (RS is the global centre), the separate residual named there.
- No cell of the organs the variant moved newly left the range: the novelty bank and the Broca slot bridges, whose cells it
  moved, were clean under stable and stay clean (0 -> 0 in every group that steps them). There is one exception to "nothing
  new left the range", outside those organs: the `BridgeParser` phasor slice gains 4 in infer (718 -> 722). It falls or stays
  in the other groups (691 -> 684 conv, 682 -> 682 ask, 675 -> 672 inner, 0 -> 0 emo), and its spike total moves with the
  switch in every group that steps it: 1180350 -> 1032261 in conv (-12.5 %), 1050867 -> 835704 in infer (-20.5 %), 1356102 ->
  1289778 in ask (-4.9 %), 1265461 -> 982747 in inner (-22.3 %), unchanged in emo. The path from the switch to this slice is
  untraced: the bridge holds no trait-typed cell (0 of its 54320 neurons) and no heterogeneity draw, so the change reaches it
  from what the switch changed elsewhere. In the inner group the one-organ arms split it without explaining it (Broca slot
  bridges alone: spikes 1265461 -> 985024, -22.2 %, out-of-range count 675 -> 675; novelty bank alone: spikes 1265461 ->
  1262041, -0.3 %, out-of-range count 675 -> 672); infer, where the +4 occurs, has no one-organ arm. In the preset + trait arm
  (Result 4) the same slice gains 1 cell in conv (691 -> 692).
- The variant moved the replies. 57 of the 63 turns have a different response hash and 32 of the 63 a different answer text;
  none has a different abstention, recalled fact or derived flag, and the answered count is the same in every group, which is
  exactly what `tools/answer_rate.py` `classify` / `rate` measures, so it reports no change while 25 of the 32 texts are
  worse. Of the 32 changed texts, 25 change word order or inflection (conv 6, infer 8, ask 4, inner 7), 5 change only the
  appended curiosity lead phrase (conv 3, emo 2) and 2 add a curiosity question that stable did not ask (ask, `g_newt` and
  `u_toad_1`). The teach-turn echoes lose the article's place and the guesses lose articles and inflection:

| group, turn | stable | +trait |
|---|---|---|
| conv, t_fox_chase | the fox chases the hare the dog chases the cat the cat eats the fish | the fox the chases hare the dog the chases cat the cat the eats fish |
| conv, t_deer_sleep | the deer sleeps the meadow | the deer the sleeps meadow |
| infer, t_wren_chase | the wren chases the moth | the wren the chases moth |
| inner, p_chase_all | perhaps the lynx chases the vole | perhaps lynx chase vole |
| conv, p_deer_sleep | I don't know about that. | I don't know about that. - worth going further here. |
| emo, s_anger_1 | ...what can you tell me about someone? - worth going further here. | ...what can you tell me about someone? - there's plenty more to dig into here! |

  (The last two rows differ only in the curiosity lead phrase appended to the reply.)
- The leaf diff of the two shard records (`compare_s7_trait.json`, key `leaf_diff_total`): 1125 of the 12251 leaves differ
  (1040 floats, 69 strings, 14 flags, 2 ints), 83 of them not numbers: the answer text (32), the DA drive's `mode` (19: rest -> neutral 8, neutral -> focus 5,
  rest -> focus 2, focus -> arousal 2, focus -> neutral 1, neutral -> rest 1), its `lead` phrase (10) and `reason` (8), the
  `fluent_hypothesis` flag (10 turns, every one true -> false) and the curiosity flag (4, in two turns). By comparison the preset switch, run on the same five groups, changed 235 leaves and none was a string or a
  flag. The numbers behind the DA drive moved by large factors (`da_level` up by up to 5.8 times, from 0.046 to 0.266 in ask; the DA <!--derived-->
  write gain `g` by up to 52 %; the resonate-and-fire mean magnitude and the Hebbian held read-out by up to 20 %). <!--derived-->

## Result 2: which organ changes which part of the reply (one organ at a time, inner group)

`research/runners/stable_mechc_trait_attrib.py` runs the `inner` group on the stable integrator with the variant on for the
bridges built under one organ only (the construction call chain contains `build_slot_bridge`, or
`spiking_novelty_habituation_organ`) and off for every other bridge; the bridges' read-back shows which were centred (Broca: the
two slot bridges, 368 cells of which 300 off their centre; novelty: the bank, 3520 cells of which 2790 off).

| arm (inner, dev seed 7) | leaves changed of 2274 | answer texts changed | `fluent_hypothesis` true -> false | DA-drive read-outs | out-of-range neurons | u-held cells |
|---|---|---|---|---|---|---|
| stable (reference) | - | - | - | - | 787 | 112 |
| variant on for the Broca slot bridges only | 11 (7 strings, 4 flags) | 7 | 4 | unchanged | 787 | 112 |
| variant on for the novelty bank only | 194 (188 floats, 2 ints, 4 strings) | 0 | 0 | changed (`da_level` up to 5.0 times, `mode` in 4 turns) | 784 | 112 |
| variant on everywhere (`stable_trait`) | 205 | 7 | 4 | changed | 678 | 6 |

- The full variant's 205 changed leaves are exactly the union of the two single-organ sets: no leaf is in both, the values are
  equal wherever a leaf is in one, and none is left over. So the lost article, the lost inflection and the `fluent_hypothesis`
  flag come from the Broca frame-slot bridges alone, the DA drive numbers and mode from the novelty bank alone, and the
  association graph contributes no reply leaf: it changes its own membranes and spikes. This attribution is the inner group
  only. The 3 cells by which the resonate-and-fire phasor slice drifts in this group (787 to 784) already appear in the
  novelty-only arm.
- The other four groups have no one-organ arm, so what holds there is consistent with this split and not shown by it. The 10
  turns where the DA drive's `lead` phrase changes are all turns where its `mode` changes (conv 5, emo 2, infer 2, ask 1), and
  the 2 questions that `ask` newly asks (`g_newt`, `u_toad_1`; Result 1) go with a higher `da_level` (0.08 -> 0.15 and 0.05 ->
  0.24) and a lower effective curiosity threshold (61.0 -> 46.3 and 69.6 -> 36.4), read from the `curiosity_da` block of
  `research/findings/raw/_stable_mechc/census/stable/s7/ask__intact_a.json` and
  `research/findings/raw/_stable_mechc/census/stable_trait/s7/ask__intact_a.json`. That is the DA drive, which the inner
  attribution puts on the novelty bank; for `ask` this is not tested.
- The Broca bridges are the only organ whose edit shows in the words. The G3 de-risk below is the same bridge's own bar run
  without the brain around it: with the variant it passes on 4 of 8 dev seeds against 7 of 8 without (3 pass-to-fail, 0
  fail-to-pass; n = 8 cannot separate the arms on the pass count; under the stable integrator the mean held-out order falls from
  0.904 to 0.834 <!--derived-->
  across the 8 seeds).

## Result 3: the organs' own gates under the variant (dev seeds 1 to 8)

`research/runners/stable_mechc_lag_gate.py` runs, in process and with the spikes of every bridge step counted, three existing
checks of the two organs whose trait-typed cells the variant changes most visibly, for each of four arms (the default integrator
and `SIM_STABLE_INTEGRATION=1`, each with the variant unset and set):

- G1, the association graph's own parity harness (`learned_assoc_graph.main`, 10 concepts, 4 stored facts): the learned graph
  against the Python co-occurrence oracle. Validated 2026-06-21 at 24 of 24 oracle edges and 9 of 9 concepts whose top learned
  associate is a true associate; pass here = both, in full.
- G2, the D cue-recall resolution (`_D_sparse_heteroassoc.run`, 2 pairs): cue -> associate top-1 completion 2 of 2 after encode and
  after the SWR cycles. Pass here = both 2 of 2.
- G3, the Broca frame-slot bridge's own per-seed de-risk (`_emerge59_spiking_broca_frame_slots_derisk._derisk_one`), added because
  the census found this trait-typed bridge among the three the variant changes and the replies lost their function-word order:
  held-out order and exact-slot rendering at least 0.90, at least 0.20 above the permuted and no-learning controls, function words
  load-bearing, 0 producer calls on abstains. Pass here = every bar of the de-risk's own `go` rule, per seed.

| arm | G1 passes (oracle edges recovered, of 192) | G2 passes (seeds with 2 of 2 after the SWR cycles) | G3 passes (mean held-out order / exact) | G1 + G2 spikes |
|---|---|---|---|---|
| default integrator, variant off | 0 of 8 (182) | 1 of 8 | 7 of 8 (0.991 / 0.983) | 4792478 | <!--derived-->
| default integrator, variant on | 0 of 8 (129) | 0 of 8 | 4 of 8 (0.884 / 0.781) | 9073269 | <!--derived-->
| stable integrator, variant off | 5 of 8 (192) | 1 of 8 | 7 of 8 (0.904 / 0.892) | 1542002 | <!--derived-->
| stable integrator, variant on | 0 of 8 (85) | 0 of 8 | 4 of 8 (0.834 / 0.774) | 1540391 | <!--derived-->

- The comparison that matters for the census is stable off against stable on, paired by seed, with the flips behind each pass
  count: G1 5 of 8 -> 0 of 8 (5 pass-to-fail, seeds 1, 3, 5, 7, 8; 0 fail-to-pass); G3 7 of 8 -> 4 of 8 (3 pass-to-fail, seeds
  1, 5, 7; 0 fail-to-pass); G2 1 of 8 -> 0 of 8 (1 pass-to-fail, seed 3; 0 fail-to-pass). G2 is not counted as a gate the
  variant broke: its stable baseline already fails on 7 of 8 seeds and one flip is no evidence either way. At n = 8 the pass
  counts cannot separate the arms on G2 (1 flip) or on G3 (3 flips against 0, exact two-sided sign test p = 0.25). G1 (5
  against 0, p = 0.06) is the gate whose pass count alone comes closest, and its continuous read-out is unambiguous: the
  oracle edges recovered are lower with the variant on all 8 seeds (24 per seed without it, 6 to 16 with it). The G3 read-outs
  point the same way (held-out order lower on all 8 seeds, exact-slot rendering lower on 5 and equal on 3), which shows a
  direction and is not a gate result. On the default integrator the same pairing gives G1 0 of 8 -> 0 of 8, G2 1 of 8 -> 0 of
  8 (1 flip) and G3 7 of 8 -> 4 of 8 (4 pass-to-fail, seeds 1, 2, 3, 8; 1 fail-to-pass, seed 5).
- With the variant the learned graph is sparser, not wrong where it exists: under stable + variant every concept that has a learned
  associate has a true top associate (the top-associate count equals the number of concepts with an associate on all 8 seeds,
  for example 6 of 6, 9 of 9, 5 of 5), but only 85 of the 192 oracle edges clear the 0.3 weight threshold.
- The baseline is not the June figure: with the default integrator on numpy the gates already fail on the dev seeds (G1 0 of 8,
  3 to 6 of 9 top associates per seed, G2 1 of 8), the stable integrator recovers all 192 edges and passes G1 on 5 seeds, and G3
  already fails on seed 2 under stable (held-out order 0.295, where it is 0.989 with the default integrator; not caused by the variant). So these are <!--derived-->
  measured baselines on this backend, and they are the ones the variant is compared with.
- With the default integrator and the variant the pool fires 1.9 times as often (G1 + G2 spikes 9073269 against 4792478); with the
  stable integrator the variant leaves the total spike count unchanged (1540391 against 1542002) while cutting the edges recovered.

## Result 4: STABLE + PRESET + TRAIT, and the trait variant on top of the preset switch

`stable_preset_trait` = `SIM_STABLE_INTEGRATION=1`, `SIM_PRESET_CENTRED_HET=1` and `SIM_TRAIT_CENTRED_HET=1` (same five groups,
same conditions as Result 1).

| group | out-of-range neurons stable / +preset +trait | u-held b < 0 cells | cells with b < 0 in the brain stable / +preset +trait | turns with a changed response hash | turns with a changed answer text | turns with a changed abstention, recalled fact or derived flag | answered turns stable / +preset +trait |
|---|---|---|---|---|---|---|---|
| conv | 803 / 698 | 112 / 6 | 118539 / 108089 | 11 of 11 | 9 | 0 | 6 of 11 / 6 of 11 |
| infer | 830 / 723 | 112 / 6 | 118235 / 107785 | 10 of 10 | 8 | 0 | 8 of 10 / 8 of 10 |
| ask | 794 / 687 | 112 / 6 | 117283 / 109233 | 18 of 19 | 6 | 0 | 4 of 19 / 4 of 19 |
| emo | 110 / 4 | 110 / 4 | 104117 / 101993 | 8 of 13 | 2 | 0 | 5 of 13 / 5 of 13 |
| inner | 787 / 677 | 112 / 6 | 116051 / 108361 | 10 of 10 | 7 | 0 | 8 of 10 / 8 of 10 |

- The two switches act on disjoint cells and the results add: the u-held cells fall exactly as with the trait variant alone (112
  to 6), the region switch lowers the count of cells with b < 0 further (conv 118539 to 108089, against 116329 with the trait
  variant alone) because it also centres the region-typed cells of the other organs, and the replies carry the trait variant's
  changes: the same 57 of 63 response hashes, the same 32 of 63 answer texts, no abstention, recall or derivation changed.
- Against the preset-only arm (`compare_s7_preset_vs_preset_trait.json`), which changed no answer in the preset finding, adding
  the trait variant changes 32 of 63 answer texts and 1110 leaves, 83 of them strings or flags. The text changes are therefore the
  trait variant's, not the region switch's. The leaf total of the pair (1128 against stable) is not the sum of the two switches'
  separate effects, since the cells they change feed the same downstream organs.
- The only out-of-range count that rises in this arm is the same `BridgeParser` phasor slice, by 1 in conv (691 -> 692); it
  falls in infer (718 -> 717), ask (682 -> 681) and inner (675 -> 671). Its spike total is down 12.7 %, 20.6 %, 5.0 % and 22.5
  % in conv, infer, ask and inner. The +4 of the stable + trait arm in infer does not recur here (Result 1: path untraced).

## What this does not show

- One seed and one arm for the census. Dev seed 7, intact_a, five scripted conversations; the brain-level numbers (112 to 6,
  32 of 63) are this seed's. The attribution is the inner group only, on the stable integrator. Organs these five groups do not
  step (the BG selector, the d3 FS-WTA score bridge, the rf-phasor composer's bridges, the lazily built organs) were not
  exercised, so the variant's reach in other trait-typed bridges is known only where the census read-back listed it: the
  association graph, the novelty bank and the Broca slot bridges, plus bridges that have no trait-typed cell outside a region.
- The gates are dev seeds 1 to 8, numpy CPU, in process. Their baselines are weak (parity passes on 0 of 8 seeds at the
  default integrator, cue-recall on 1 of 8), so "5 of 8 to 0 of 8" is a statement about the stable integrator, where parity
  passes on 5 seeds. With 8 seeds the pass counts of G2 (one flip) and G3 (three flips against none) cannot separate the arms;
  only G1's edges recovered, which fall on every seed, and the direction of G3's continuous read-outs say more. The pass rules
  are my transcription of each organ's validated figures (24 of 24 and 9 of 9; 2 of 2; the de-risk's own bars); G3 is an
  attribution check I added, not a gate registered for this variant. The association graph's agent-level test
  (`test_learned_assoc_graph_agent`) is GPU-only and was not run.
- It does not show that trait-centred cells are the wrong model. The traits are assigned at random per neuron
  (`cfg.num_traits` of them, by `randint`), so a pool mixes striatal projection, thalamic relay, thalamic reticular, FS and RS
  cells by chance; centring makes each cell a coherent instance of its type but does not make the mixture biologically
  motivated (what a CA3-like associative pool or a Broca slot pool should be made of is a separate question). It shows that
  these three organs behave as built on the RS-centred hybrid and not on the mixture of pure types.
- Whether re-tuning the organs, or typing them explicitly (a region preset, or `num_traits = 1`), would pass the gates again
  was not tried. This is a verdict on the method as a whole-brain switch.
- The 4 u-held cells that remain in the association graph (6 with the thought-swap workspace's 2) are RS-preset cells. Their own
  parameters are not changed by the variant (the RS trait is on the centre; shown bit for bit), yet the offenders fell from 22 to
  24 down to 4, because their inputs changed with their neighbours' parameters. The lowest remaining membrane is still -122.4 mV.
  They are the separate residual of the preset finding and need their own answer.
- The answer rate is blind to this change: it classifies abstentions, and no turn changed between answering and abstaining. A
  word-order or inflection regression needs a text check. The registered gate scorer's `all` verdict did not move in any group
  (conv, infer, ask and inner false -> false; emo true -> true), but its sub-bars did in two groups, so it is not true that
  none of its read-outs moved. In ask, `ask_rate_gap` goes from 0.667 to 1.0 <!--derived--> (2 of 3 gap topics asked to 3 of
  3) and `lp_unlearnable` from false to true, with two more questions asked: `g_newt` and `u_toad_1` (asked and curious, false
  -> true; the two added texts of Result 1). In conv, `frames_fox`, the frames the scorer parses out of the reply to
  `p_fox_chase`, goes from one frame (fox, chases, hare) to none, while `fox_chase` stays false. The two ask changes move
  clauses of that group's pass rule toward passing (`ask_rate_gap` feeds `discriminates`, `lp_unlearnable` is one of the
  clauses `all` needs; `all` stays false because `lp_learnable` and `builtin_dog_cat` are false in both arms); the conv one
  removes a parseable frame from a reply that did not pass `fox_chase` before either. In infer, emo and inner no sub-bar moved
  (infer's yes/no reply text changed, its flags did not). The same sub-bar changes appear in the preset + trait arm and in the
  preset-only against preset + trait pair, none in the preset-only arm (`compare_s7.json`) or in the two one-organ arms (inner
  only). Source: `gate_score` of each group in `compare_s7_trait.json`.
- HEAD moved during the census (research files added; `sim/` unchanged since the variant's commit). The test mutations that show
  the new tests can fail (two engine edits, 7 and 3 failures) were run by hand and are not committed.

Next methods, not built here: (1) give the Broca slot bridge and the novelty bank their intended homogeneous type explicitly and
re-run G3 and the novelty organ's own gates, since these pools were designed as one population and received a trait mixture from
the default `num_traits`; (2) attack the association graph's remaining offenders at their cause, a pool that fires 125094 spikes in
2250 steps with strong recurrent weights, rather than at the cell type; (3) if the trait variant is kept as a research switch,
adopt it per organ behind that organ's gates (the ruling of the region flag), starting with an organ whose own gates it passes.

## Host, inputs and provenance

- Host and backend. The 27 provenance sidecars of the artifacts this finding cites (the 25 in the front matter plus
  `compare_s7_trait_only_broca.json` and `compare_s7_trait_only_novelty.json`) all record host `dant123-wk` (the workstation,
  20 CPUs), `sim_backend` numpy, `CUDA_VISIBLE_DEVICES=''` (no GPU was used) and the interpreter
  `/home/dant123/Projects/sim/.venv/bin/python`. 26 of them ran from the worktree `sim-rev/wt-mechc2`, the reference arm
  `research/findings/raw/_stable_mechc/census/stable/s7/inner__intact_a.json` from `sim-rev/wt-stable-mechc`; they started
  between 2026-10-27 12:33 and 18:24 (the census, one-organ and gate runs under `tools/memcap.sh`).
- Inputs. The 23 census runs (the reference and variant arms, the two one-organ arms and the unset-path check) read two inputs
  that are not in the repository. First, `data/`: in each worktree a symlink to the main checkout's `data/`
  (`/home/dant123/Projects/sim/data`), which the corpus-learned organs read; every comparison record carries `corpus_present`
  true for both arms of every group (a first attempt without the symlink was discarded, FAILURE_LOG 2026-10-27). Second, the
  persisted long-term-memory bundle `wikidata_100k`, loaded once per process from
  `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k` (78857 facts; a plain directory read by absolute path, not
  a symlink); the run logs of all 23 census runs name that path and fact count, and the size and modification time they record
  for its three files equal the files today. sha256 of those files, computed 2026-10-28: `manifest.json`
  783dd0f4f5ba65aee4b1e59186a67ec4452e90c73d379d6c70461ce65efd0583, `facts.json`
  92466eeeaccdb62bad396b1a6a580400abec6967eab89fbb78e8783a8d06b90c, `composites.npz`
  48faad473478300321a91d91496deb327be92e080a04f5410d335487ae3c9365. The run logs are not committed (they sit in
  `sim-rev/mechc2_scratch` and `sim-rev/stable_mechc_scratch`). The gates (G1 to G3) are in-process harnesses; their logs show
  no LTM load.
- Git state of the records. 24 of the 27 sidecars have `git_dirty` true, so the tree held uncommitted files when they were
  started; what those files were is not recorded. `git_sha` of the 27: 65d60f996 (13), 19fda7fc4 (10), 3b9037b6f (2),
  e49829cd2 (1), 72bc75e7c (1). `sim/` is the variant's at four of those five commits (19fda7fc4, 3b9037b6f,
  65d60f996 and 72bc75e7c; no commit after bd13ab71c touches `sim/`, `git diff bd13ab71c HEAD -- sim` is empty) and the pre-edit
  engine at e49829cd2; whether a dirty tree also held an uncommitted `sim/` edit is something the flag cannot show. The reference arms of conv, infer, ask
  and emo (`census/stable/s7`, not in the front-matter list) are the preset finding's records, taken before the edit at
  e13c7d968 or 0606d7fc4, also dirty-tree.
- Clean-tree anchors. Three of the 27 records come from a clean tree:
  `research/findings/raw/_stable_mechc/census/stable/s7/inner__intact_a.json` (e49829cd2, the engine before the edit),
  `research/findings/raw/_stable_mechc/census/stable_trait/s7/inner__intact_a.json` (19fda7fc4, the variant) and
  `research/findings/raw/_stable_mechc/census/stable_unset_check/s7/inner__intact_a.json` (72bc75e7c, both new switches
  unset). The dirty-tree runs agree with them where they overlap: the two one-organ arms (dirty, 65d60f996) change exactly the
  leaves of the clean variant record, the disjoint union of 11 and 194 leaves equal to its 205 with equal values; and the
  clean unset-path record reproduces the clean reference record turn for turn (10 of 10 per-turn records identical, 26 of 26
  organ rows equal in organ, out-of-range count, spikes and steps). The other four groups, the gates and the compare files
  have no clean twin.

## Sources

- Census records: `research/findings/raw/_stable_mechc/census/stable_trait/s7/<group>__intact_a.json` and
  `research/findings/raw/_stable_mechc/census/stable_preset_trait/s7/<group>__intact_a.json` (conv, infer, ask, emo, inner, each
  with its `.prov.json`); the reference arms `census/stable/s7` and `census/stable_preset/s7` are those of the preset finding;
  the one-organ runs `research/findings/raw/_stable_mechc/census/trait_only_broca/s7/inner__intact_a.json` and
  `research/findings/raw/_stable_mechc/census/trait_only_novelty/s7/inner__intact_a.json`.
- Comparisons (`python -m research.runners.stable_preset_compare`): `research/findings/raw/_stable_mechc/compare_s7_trait.json`
  (stable against stable_trait), `compare_s7_preset_trait.json` (stable against stable_preset_trait),
  `compare_s7_preset_vs_preset_trait.json`, `compare_s7_trait_only_broca.json`, `compare_s7_trait_only_novelty.json`; each holds
  the per-organ classes, the parameters as drawn, the reply differences, the answer-rate lens and the leaf diff.
- Gates (`python -m research.runners.stable_mechc_lag_gate`): `research/findings/raw/_stable_mechc/lag_gate/{off,trait,stable,
  stable_trait}.json` (G1 and G2) and `g3_{off,trait,stable,stable_trait}.json` (G3), dev seeds 1 to 8.
- Related-test files run for the base-against-branch comparison (17 on numpy CPU; `tests/test_determinism.py` is the 18th and
  was not run): `test_backend_neutral_izh_arithmetic`, `test_backend_neutral_izh_initialization`,
  `test_izh_membrane_substeps`, `test_membrane_bound_check`, `test_membrane_census`, `test_preset_centred_heterogeneity`,
  `test_preset_centred_het_switch`, `test_stable_integration_switch`, `test_stable_json_evidence`,
  `test_stable_mechc_lag_gate`, `test_stable_mechc_trait_attrib`, `test_stable_preset_compare`,
  `test_trait_centred_heterogeneity`, `test_v13_backend_neutral_izh_arithmetic_replay`,
  `test_v13_backend_neutral_izh_arithmetic_replay_v2`, `test_v13_backend_neutral_izh_initialization_diagnostic`,
  `test_v13_izh_arithmetic_localizer`.
- Engine and tests: `sim/bridge.py` (`TRAIT_CENTRED_HET_ENV`, `_collect_het_trait_centres`), `sim/config.py`,
  `tests/test_trait_centred_heterogeneity.py` and its pre-edit golden `tests/_trait_centred_het_golden.json`
  (`tests/_capture_trait_centred_het_golden.py`, captured at f58274479 before the edit).
- Design record this follows: `docs/drafts/preset-centred-het/CENSUS.md` (C2.1) and `research/findings/2026-10-27-stable-preset-centred-dev7.md`.

## Derived

- Out-of-range neurons removed in the stable arm: 803 - 690 = 113 in conv (106 association graph + 7 phasor drift), 830 - 728 = 102
  in infer (106 - 4), 794 - 688 = 106 in ask, 110 - 4 = 106 in emo, 787 - 678 = 109 in inner (106 + 3).
- Association graph spikes 125094 -> 103901: a fall of 21193, 16.9 percent. Gate sums: oracle edges recovered over 8 seeds, off
  182, variant 129, stable 192, stable + variant 85 (of 192 = 8 x 24). Mean G3 held-out order and exact-slot, off 0.991 and
  0.983, variant 0.884 and 0.781, stable 0.904 and 0.892, stable + variant 0.834 and 0.774. G1 + G2 spikes ratio 9073269 / 4792478
  = 1.89.
- Leaf counts: 1125 = 1040 + 69 + 14 + 2 (stable against stable_trait); non-numeric 83 = 32 answers + 19 DA mode + 10 lead + 8
  reason + 10 fluent_hypothesis + 4 curiosity flags; inner 205 = 11 (Broca only) + 194 (novelty only); turns 63 = 11 + 10 + 19 +
  13 + 10; answer texts changed 32 = 9 + 8 + 6 + 2 + 7; response hashes changed 57 = 11 + 10 + 18 + 8 + 10.
- Largest relative changes quoted as factors: `da_level` 4.77 more than before = 5.8 times, 4.04 more = 5.0 times; `g` 0.52 = 52 percent.
- Answer texts changed 32 = 25 word order or inflection (conv 6 + infer 8 + ask 4 + inner 7) + 5 curiosity lead phrase only
  (conv 3 + emo 2) + 2 added curiosity questions (ask); DA `lead` phrase changes 10 = 5 + 2 + 2 + 1 (conv, emo, infer, ask),
  each with a `mode` change. Broca-only inner: 7 answer texts + 4 flags = 11 leaves; novelty-only inner: 188 floats + 2 ints +
  4 strings = 194.
- `BridgeParser` spike totals, stable -> stable + trait: 1032261 / 1180350 = 0.875 (-12.5 %), 835704 / 1050867 = 0.795 (-20.5
  %), 1289778 / 1356102 = 0.951 (-4.9 %), 982747 / 1265461 = 0.777 (-22.3 %); inner one-organ arms 985024 / 1265461 (-22.2 %)
  and 1262041 / 1265461 (-0.3 %); preset + trait arm -12.7 %, -20.6 %, -5.0 %, -22.5 %.
- Gate flips, stable against stable + variant, paired by seed: G1 pass on seeds 1, 3, 5, 7, 8 without the variant and on none
  with it (5 and 0); G2 pass on seed 3 only without it, none with it (1 and 0); G3 pass on 7 seeds without it (all but 2) and
  on seeds 3, 4, 6, 8 with it, so seeds 1, 5, 7 flip (3 and 0). Exact two-sided sign tests on the discordant seeds: 3 against
  0 gives 2 x (1/2)^3 = 0.25, 5 against 0 gives 2 x (1/2)^5 = 0.06, 1 against 0 gives 1. Oracle edges per seed without the
  variant 24 on all 8 seeds, with it 12, 16, 8, 6, 14, 14, 6, 9 (sum 85). Default integrator, off against variant: G3 pass on
  seeds 1, 2, 3, 4, 6, 7, 8 against 4, 5, 6, 7 (flips 1, 2, 3, 8 to fail and 5 to pass); G2 pass on seed 4 only against none.
- Tests: 283 + 7 = 290 collected on the branch against 226 + 7 = 233 on the base, 57 more passes = 47 + 4 + 3 + (26 - 25) + (6
  - 4).
- Provenance: 27 sidecars = 24 `git_dirty` true + 3 false; `git_sha` 13 + 10 + 2 + 1 + 1 = 27; 26 from `wt-mechc2` + 1 from
  `wt-stable-mechc`.
