---
type: finding
status: qualified
date: 2026-10-27
lane: substrate (integrator validity, mechanism c); follow-up of research/findings/2026-10-27-stable-preset-centred-dev7.md (the region-preset switch did not reach the 110 association-graph cells because they are typed by trait)
mechanism: SIM_TRAIT_CENTRED_HET=1 (sim/bridge.py TRAIT_CENTRED_HET_ENV) forces cfg.trait_centred_parameter_heterogeneity=True onto every SimulationBridge; default unset = unchanged
flip: no - default OFF; it removes the mechanism-c cells it was built for and breaks the Broca slot bridge's own de-risk, the association graph's own gates and the word order of the replies
seeds: [7, 1, 2, 3, 4, 5, 6, 8]
runner: research/runners/membrane_census.py
artifacts:
  - research/findings/raw/_stable_mechc/compare_s7_trait.json
  - research/findings/raw/_stable_mechc/compare_s7_preset_trait.json
  - research/findings/raw/_stable_mechc/compare_s7_preset_vs_preset_trait.json
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

# Centring trait-typed cells on their own preset removes the cells below -110 mV and breaks the replies' word order and the association graph's gates (dev seed 7 and dev seeds 1 to 8, 2026-10-27)

## Answer in one paragraph

The variant built here (`SIM_TRAIT_CENTRED_HET=1`, config flag `trait_centred_parameter_heterogeneity`, default OFF, the unset
path equal by hash to the engine before the edit) centres the heterogeneity draw of Izhikevich cells typed by trait on their own
trait's preset, which is the method the preset finding named for the 110 association-graph cells that stay below -110 mV on the
stable integrator. On the five Stage-2 groups (dev seed 7, stable integrator) it does that: the 110 `LearnedAssocGraph` offenders
fall to 4 (every FS and thalamic-reticular cell stops leaving the range), the brain's u-held cells go from 112 to 6, and nothing new
leaves the range. It is not a flip candidate, and it answers why the region flag left these cells alone. It moves three organs'
cells at once, and each has its own cost: the two Broca frame-slot bridges alone make the replies lose the article's place and the
inflection ("the fox the chases hare", "perhaps lynx chase vole", 32 of 63 turns have a different answer text, 0 have a different
abstention, recall or derivation, so the answer rate cannot see it); the novelty-habituation bank alone changes the DA drive (its
level by up to 5.8 times, its mode in 19 turns); and the association graph's own gates, run on dev seeds 1 to 8, go from 5 of 8
to 0 of 8 (parity with the oracle) and from 7 of 8 to 4 of 8 (the Broca bridge's de-risk) against the stable integrator without it.
The three organs' effects add exactly (the 205 changed leaves of the inner group are the disjoint union of the Broca-only 11 and
the novelty-only 194). These organs were built and validated on the RS-centred hybrid, so removing the hybrid is a change of
their dynamics, not a repair. The 4 RS-preset offenders that remain are a separate residual.

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
- Nothing new left the range: the novelty bank and the Broca slot bridges, whose cells the variant moved, were clean under stable
  and stay clean.
- The variant moved the replies. 57 of the 63 turns have a different response hash and 32 of the 63 a different answer text; none
  has a different abstention, recalled fact or derived flag, and the answered count is the same in every group, which is exactly
  what `tools/answer_rate.py` `classify` / `rate` measures, so it reports no change while the text is worse. The teach-turn
  echoes lose the article's place and the guesses lose articles and inflection:

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
  flag come from the Broca frame-slot bridges alone, the DA drive numbers and mode from the novelty bank alone (and, in the
  other groups, the appended curiosity lead phrase follows the DA mode), and the association graph contributes no reply leaf:
  it changes its own membranes and spikes. The 3 cells by which the resonate-and-fire phasor slice drifts in this group
  (787 to 784) already appear in the novelty-only arm.
- The Broca bridges are the only organ whose edit shows in the words. The G3 de-risk below is the same bridge's own bar run
  without the brain around it, and it fails with the variant for the same bridge.

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

- The comparison that matters for the census is stable off against stable on: G1 5 of 8 -> 0 of 8, G2 1 of 8 -> 0 of 8, G3 7 of 8 ->
  4 of 8. The variant breaks the association graph's parity and the Broca bridge's de-risk on dev seeds that passed without it.
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

## What this does not show

- One seed and one arm for the census. Dev seed 7, intact_a, five scripted conversations; the brain-level numbers (112 to 6,
  32 of 63) are this seed's. The attribution is the inner group only, on the stable integrator. Organs these five groups do not
  step (the BG selector, the d3 FS-WTA score bridge, the rf-phasor composer's bridges, the lazily built organs) were not
  exercised, so the variant's reach in other trait-typed bridges is known only where the census read-back listed it: the
  association graph, the novelty bank and the Broca slot bridges, plus bridges that have no trait-typed cell outside a region.
- The gates are dev seeds 1 to 8, numpy CPU, in process. Their baselines are weak (parity passes on 0 of 8 seeds at the default
  integrator, cue-recall on 1 of 8), so "5 of 8 to 0 of 8" is a statement about the stable integrator, where parity passes on 5
  seeds. The pass rules are my transcription of each organ's validated figures (24 of 24 and 9 of 9; 2 of 2; the de-risk's own
  bars); G3 is an attribution check I added, not a gate registered for this variant. The association graph's agent-level test
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
- The answer rate is blind to this change: it classifies abstentions, and no turn changed between answering and abstaining.
  A word-order or inflection regression needs a text check; none of the registered scorers (`gate scorer all`) moved either.
- HEAD moved during the census (research files added; `sim/` unchanged since the variant's commit). The test mutations that show
  the new tests can fail (two engine edits, 7 and 3 failures) were run by hand and are not committed.

Next methods, not built here: (1) give the Broca slot bridge and the novelty bank their intended homogeneous type explicitly and
re-run G3 and the novelty organ's own gates, since these pools were designed as one population and received a trait mixture from
the default `num_traits`; (2) attack the association graph's remaining offenders at their cause, a pool that fires 125094 spikes in
2250 steps with strong recurrent weights, rather than at the cell type; (3) if the trait variant is kept as a research switch,
adopt it per organ behind that organ's gates (the ruling of the region flag), starting with an organ whose own gates it passes.

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
