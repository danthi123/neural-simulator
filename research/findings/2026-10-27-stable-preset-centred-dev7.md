---
type: finding
status: qualified
date: 2026-10-27
lane: substrate (integrator validity, mechanism c); follow-up of research/findings/2026-10-26-stable-integration-dev7.md and the 2026-10-20 preset-centred heterogeneity fix
mechanism: SIM_PRESET_CENTRED_HET=1 (sim/bridge.py PRESET_CENTRED_HET_ENV) forces cfg.preset_centred_parameter_heterogeneity=True onto every SimulationBridge; default unset = unchanged
flip: no - default OFF; one dev seed; the switch does not reach the cells it was built for
seeds: [7, 1, 2, 3]
runner: research/runners/membrane_census.py
artifacts:
  - research/findings/raw/_stable_mechc/compare_s7.json
  - research/findings/raw/_stable_mechc/leaf_diff_stable_vs_preset_s7.json
  - research/findings/raw/_stable_mechc/lag_probe_s7.json
  - research/findings/raw/_stable_mechc/lag_trait_probe.json
  - research/findings/raw/_stable_mechc/census/stable/s7/conv__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable/s7/infer__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable/s7/ask__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable/s7/emo__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable/s7/inner__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset/s7/conv__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset/s7/infer__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset/s7/ask__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset/s7/emo__intact_a.json
  - research/findings/raw/_stable_mechc/census/stable_preset/s7/inner__intact_a.json
  - research/findings/raw/_stable_integration/census/on/s7/inner__intact_a.json
---

seed-waiver: a labelled dev-seed diagnostic (dev seed 7 for the five-group census, dev seeds 7, 1, 2 and 3 for the build-only and trait-centring probes); no gate seed was run, no default was flipped and no capability claim is made.

# The preset-centred switch does not reach the cells that stay below -110 mV on the stable brain (dev seed 7, 2026-10-27)

## Answer in one paragraph

The 112 cells that still sit below -110 mV under `SIM_STABLE_INTEGRATION=1` (mechanism c: with b < 0 their own recovery
variable u holds v down) are not region-typed cells, so the existing preset-centred heterogeneity fix cannot touch them.
110 are `LearnedAssocGraph` cells: its bridge (`_D_sparse_heteroassoc.build`) has two regions and neither carries a neuron
type, so every cell is typed by trait and gets the RS-centred a/b/d/C draw that the flag deliberately leaves alone. The
other 2 are cells of the thought-swap workspace, built as plain RS (`num_traits = 1`), where the global centre and the
preset are the same value. I built the missing whole-brain switch (`SIM_PRESET_CENTRED_HET=1`, default OFF, same style as
`SIM_STABLE_INTEGRATION`) and ran the five Stage-2 groups on dev seed 7 on the stable integrator with and without it: the
switch ran in 10 production organs and moved about 4900 to 12800 neurons per group (summed over bridges), the u-held
b < 0 cells stayed exactly 112 (110 in emo) in every group, out-of-range neurons moved by at most 4 in either direction
(phasor slice only), and no turn changed its answer, abstention, recalled fact or derived flag in any group; every
response hash moved in most turns (all of them in conv, infer and inner), and every leaf that differs between the two arms is a number: the DA write-gain `g` moved by up to 12.1 %, the resonate-and-fire `mean_magnitude` and the Hebbian held read-out by up to 8.6 %, the idle-replay candidate scores by up to 5.3 % (its winner is unchanged), and the role-competition margins only at the sixth decimal. The cause of the 110 is a
trait-typed population, and a diagnostic that centres those cells on their own trait preset with the engine's own rule
(a probe, not an engine change) takes the cells below -110 mV on the stable integrator from 92 to 115 down to 3 to 9 of
1800 on four dev seeds: every FS and thalamic-reticular offender goes, the RS-preset ones (72 down to 23 over the four seeds) stay. So the next method is a
trait-centred variant of the flag; it reverses a choice the flag's author made on purpose, so it is reported here and
not built.

## What was built (default OFF)

- Switch, `sim/bridge.py` (`PRESET_CENTRED_HET_ENV = "SIM_PRESET_CENTRED_HET"`): unset, "" or "0" touch nothing; "1" sets
  `cfg.preset_centred_parameter_heterogeneity = True` on every `SimulationBridge`'s core config at construction and at every
  `_initialize_simulation_data` (the only place the draws run), whatever the organ configured; any other value raises.
  Read-back on the bridge: `preset_centred_het_requested`, `_applied`, `_organ_value` (what the organ had configured)
  and `_readback` (what the last heterogeneity draw saw: whether it ran centred, how many neurons belong to a region
  preset and how many of those presets sit more than 10 % from the centre of the distribution that jitters them).
- Tests, `tests/test_preset_centred_het_switch.py` (34 passed): the four off spellings are byte-identical to the
  flag's own pre-edit golden for every case it captured (global draw, untyped region, region-scoped redraw, per-region
  mask, HH overrides, affect marker); on forces the flag over an organ that set it False and equals building with the
  flag on and the env unset; re-initialisation re-asserts it; a bad value raises; an untyped region and a trait-typed
  population come out identical with the switch on (the limit this finding is about, pinned).
  `tests/test_membrane_census.py` 25 passed with the census read-back; `tests/test_stable_preset_compare.py` 4 passed.
- Census (observe-only): per bridge, the switch read-back and a per-region summary of the drawn a, b, d, C, k, vpeak, the
  number of cells with b < 0 and the preset each cell was built from (identified by its never-jittered k, vr, vt, vpeak,
  c_reset, because the viz type labels are written before the per-region override and go stale for a typed region).
- Comparison runner, `research/runners/stable_preset_compare.py`: classifies every out-of-range row (phasor, conductance
  hazard, high side, u-held b < 0 = mechanism c by the dev7 finding's own definition, other low), reads the parameters the
  bridges drew, the per-turn replies and `tools/answer_rate.py` classify/rate. It refuses a record whose workload had no
  corpus, a comprehension or turn error, or a nonzero rc (a first attempt in a worktree without the `data` link ran with
  the comprehension organ erroring and was thrown away; `research/FAILURE_LOG.md`). That attempt's wall time, quoted as
  782 s here earlier and as 781 s in the FAILURE_LOG line, comes from a discarded, uncommitted run whose outputs were
  deleted, so neither figure can be checked against a file in the repository.

## Step 1: the flag, where it applies, and whether it executes

`CoreSimConfig.preset_centred_parameter_heterogeneity` defaults to False (`sim/config.py`, pinned by
`test_flag_defaults_off`). In `SimulationBridge._initialize_simulation_data` the preset centres are collected only when the
heterogeneity block runs (`enable_parameter_heterogeneity`, or a per-region heterogeneity mask) and a region manager
exists, and `_region_parameter_presets` returns centres only for neurons in a region with an `izh_neuron_type` (HH: with
overrides). Reading that does not say which production organs pass through it, so the ON arm records it live. In the
five groups (26 to 46 bridges each) the switch was applied on every bridge and no organ had configured the flag itself:

| what the heterogeneity draw saw (ON arm) | production organs (stepper label) |
|---|---|
| draw ran with region presets, so the switch changed cells | `_affect_marker_wta_derisk` (FS), `_appraisal_interoceptive_ladder_derisk`, `_curiosity_seek_learn_onbridge_derisk` (striosome MSN-D1 and others), `_da_write_gain_spiking_derisk` (HIPPO write_gain), `_phaseB_multicue_competition_spiking_derisk` (FS), `_stageA_full_integration_derisk`, `content_selection_spiking` (HIPPO cortex_ctx, dlpfc_wm), `metacog_production_organ`, and in one group each `_affective_world_model_derisk` and `_spiking_expectation_rpe_derisk` |
| draw ran, no region has a preset, so the switch changed nothing | `LearnedAssocGraph`, `BridgeParser`, `AttributedBridgeParser`, the thought-swap workspace (`_gnw_active_overwrite_derisk`), the distributed-overwrite workspace, `_gnw_acc_bg_stop_trigger_derisk`, `CommonGroundLedgerOrgan`, `EmotionCategoryOrgan`, `CategoryInheritOrgan`, `OneBrainComposer`, `SpikingNoveltyHabituationOrgan`, `_self_schema_region_derisk`, the Broca slot pool, the two comprehension lexicons, `ReplayCircuit` |
| no heterogeneity draw ran at all | `_p1_2_workspace_deliberation_loop_derisk` (the ignition-bus workspace), `_graded_affect_attractor_derisk`, `_neuromod_spiking_da_mode_derisk`, the 47440-cell resonate-and-fire bridge of `BridgeParser`, and `_spiking_expectation_rpe_derisk` in four groups |

The static census of 2026-10-20 (`docs/drafts/preset-centred-het/CENSUS.md`) listed the same regions as reachable and
listed trait-typed populations as a variant the flag does not cover; this table is the dynamic confirmation for the
organs the five groups step (the BG selector and the other lazily built organs it lists were not stepped here).

## Result 1: out-of-range membranes and the mechanism-c cells, STABLE vs STABLE + PRESET

`stable` = `SIM_STABLE_INTEGRATION=1`; `stable_preset` = both switches 1; same code (this branch), the five Stage-2 groups
of the live `brain_chat` handler at production defaults under the gate's frozen environment, numpy CPU, intact_a, dev
seed 7, each arm in a fresh subprocess under `tools/memcap.sh 10`. The `stable` arm reproduces the earlier
`SIM_STABLE_INTEGRATION=1` census of the stable-integration finding exactly (every turn's response hash equal in all
five groups, same out-of-range counts), which is the brain-level check that the new code is inert when the new switch is unset.

| group | out-of-range neurons stable / +preset | u-held b < 0 cells stable / +preset | neurons the switch moved | cells with b < 0 in the brain stable / +preset | turns with a changed response hash | turns with a changed answer, abstained, recalled or derived | answered turns stable / +preset | gate scorer `all` |
|---|---|---|---|---|---|---|---|---|
| conv | 803 / 800 | 112 / 112 | 12846 | 118539 / 110299 | 11 of 11 | 0 | 6 of 11 / 6 of 11 | false / false |
| infer | 830 / 831 | 112 / 112 | 12798 | 118235 / 109995 | 10 of 10 | 0 | 8 of 10 / 8 of 10 | false / false |
| ask | 794 / 790 | 112 / 112 | 10878 | 117283 / 111443 | 18 of 19 | 0 | 4 of 19 / 4 of 19 | false / false |
| emo | 110 / 110 | 110 / 110 | 4878 | 104117 / 104037 | 8 of 13 | 0 | 5 of 13 / 5 of 13 | true / true |
| inner | 787 / 784 | 112 / 112 | 10086 | 116051 / 110571 | 10 of 10 | 0 | 8 of 10 / 8 of 10 | false / false |

(`neurons the switch moved` is the sum over bridges of the neurons whose region preset sits more than 10 % from the
centre of the distribution that jitters them. The scorer is the Stage-2 gate's own `score_<group>` on each shard, reported
as a lens on these two arms only. Answer-rate transitions per turn: every turn that answered under stable answered under
+preset and every abstention stayed an abstention, in all five groups.)

Per offender organ, identical in every group that steps it: `LearnedAssocGraph` 110 u-held b < 0 cells stable and 110
with the switch; the thought-swap workspace 2 and 2; the `BridgeParser` resonate-and-fire phasor slice 691, 718, 682, 675
stable and 688, 719, 678, 672 with the switch in conv, infer, ask and inner (a drift of 3, 1, 4 and 3 cells; not
membranes). Its cause is the switch: it is the only difference between the two arms, and the stable arm reproduces the
earlier census hash for hash. The coupling path that carries it is not identified. It is not a parameter or stream
change in that bridge: the 47440-cell resonate-and-fire bridge runs no heterogeneity draw, the other `BridgeParser`
bridge draws with no region preset and its drawn-parameter summary is the same in both arms (a, b, d, C means to four
decimals, `n_with_region_preset` 0), and the host RNG state after a build is identical with the switch on and off
(pinned in `tests/test_preset_centred_heterogeneity.py`). The upstream organs the switch does change (the table below)
feed values into the replies, which makes an input change the likely route, and I did not trace it. Nothing the switch
did created an out-of-range cell: every organ it changed was clean under stable and stays clean.

What the switch did change, in the cells of the organs it reached (b is the drawn mean per region, one group shown):

| organ, region | preset | b mean stable -> +preset | cells with b < 0 stable -> +preset |
|---|---|---|---|
| `content_selection_spiking`, cortex_ctx and dlpfc_wm (4080 cells each) | HIPPO_PYRAMIDAL | -2 -> 5 | 4080 -> 0 in each |
| `_da_write_gain_spiking_derisk`, write_gain (80) | HIPPO_PYRAMIDAL | -2 -> 5 | 80 -> 0 |
| `_curiosity_seek_learn_onbridge_derisk` and `metacog_production_organ`, striosome_value (60) | MSN_D1 | -2 -> -20 | 60 -> 60 |

The two arms' shard records have the same set of 12251 leaves (summed over the five groups; timings and process
statistics excluded); 235 of them differ (233 floats, 2 ints) and nothing else does: no answer text, abstention flag, recalled fact or derived flag changed, and the swap, stop, verification and
authorship read-outs are identical in all 63 turns. The 235 are, with the largest relative change of each
(`research/findings/raw/_stable_mechc/leaf_diff_stable_vs_preset_s7.json`, written by
`python -m research.runners.stable_preset_compare --leaf-diff-only` from the `workload.shard.responses` of the census
records; first found by an ad hoc script and now reproduced by the committed runner):

| read-out | leaves changed | largest relative change | where |
|---|---|---|---|
| DA encoding gain `da_encoding.g` | 56 | 12.1 % (1.0831 -> 1.2144) | ask, turn m_moose |
| resonate-and-fire `activity.rf.mean_magnitude` | 28 | 8.6 % (22964.7 -> 21000.5) | conv, turn t_deer_sleep |
| Hebbian held read-out `d6_hebbian.held.readout` | 14 | 8.6 % (434.54 -> 397.37) | conv, turn t_deer_sleep |
| idle-replay candidate score `s` / position `x` / applied `x` | 3 / 3 / 3 | 5.3 % / 2.0 % / 2.0 % | inner, turn p_private |
| idle-replay engram count / first gate step | 1 / 1 | 1.0 % (101 -> 100) / 1.3 % (76 -> 75) | inner, turn p_private |
| role-competition `activity.roles.margin` | 111 | 4.8e-5 relative, 1.65e-6 absolute | conv, turn t_deer_sleep |
| `metacog.mean_role_conf` | 15 | 3.9e-7 absolute | conv, infer, ask, inner |

The idle-replay winner (candidate 7) is the same in both arms. A 12 % change of the DA gain or an 8.6 % change of the
resonate-and-fire magnitude is not a "small" number: it is a real change in the quantity, one that none of the
answer, abstention, recall or derivation fields of these 63 turns happened to depend on. Wall time is unchanged (ratio
+preset over stable 0.993 to 1.014 per group).

## Result 2: the 110 cells are trait-typed, so no region-preset fix reaches them (build-only probe)

`research/runners/stable_mechc_lag_probe.py` builds `LearnedAssocGraph`'s bridge at its production size (1500 pool + 300 fs
cells, dev seed 7) with the switch unset and set (`lag_probe_s7.json`): none of the nine parameter arrays (a, b, d, C, k, vr,
vt, vpeak, c_reset) differs in a single one of the 1800 cells, and the ON build's own read-back says the draw ran
centred over 0 region-preset neurons. All 110 census offenders are among the cells of this build (their b matches the
census row for row) and are unchanged by the switch. They were built from the thalamic-reticular preset (58, b +10 in
the preset), the FS preset (30) and the RS preset (22); the draw replaced a/b/d/C of every one by RS-centred values (b about -2, C about 100)
and left the preset's k, vr, vt, vpeak, c_reset, so a thalamic-reticular cell runs with its low threshold and a cell body
sized like an RS pyramid, the hybrid the 2026-10-20 census described for typed regions.

## Result 3: what centring those cells on their own preset would do (diagnostic on dev seeds 7, 1, 2, 3)

`research/runners/stable_mechc_lag_trait_probe.py` builds the same `LearnedAssocGraph` on the stable integrator and stores
five facts (30 cycles each, 2250 steps, as the brain does; the concept list is a stand-in, so counts differ from the
brain's 110), rewriting a, b, d, C and the initial u of every cell before the first step. Same patterns and facts in every
arm of a seed; cells that went below -110 mV at any step, of 1800:

| arm | seed 7 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| as built (engine as is) | 92 | 99 | 115 | 97 |
| trait-centred (engine's own rule `_recentre_het_draw_host`, centre = the cell's trait preset) | 3 | 9 | 7 | 4 |
| trait preset exactly, no jitter | 6 | 14 | 10 | 6 |
| RS preset exactly, no jitter | 103 | 122 | 135 | 101 |

By trait preset, summed over the four seeds: FS 121 -> 0, thalamic-reticular 210 -> 0, striatal MSN 0 -> 0, thalamic relay
0 -> 0, RS 72 -> 23 (trait-centred). The lowest membrane goes from -153 to -165 mV down to -120 to -122. So the offenders
are the hybrids, not the heterogeneity itself (RS preset exactly leaves them), and giving each cell its own preset removes
the FS and thalamic-reticular ones. The RS-preset cells that remain are cells whose draw the rule cannot change (RS is the
global centre): mechanism (c) in plain RS cells is a separate residual, about 6 cells per seed here and 22 + 2 in the
brain census, and it needs its own answer (the 22 RS offenders of the brain census had u between 2680 and 4512 pA at their first excursion).
The total spike count of the graph also moves (for example seed 7: 117159 as built, 102450 trait-centred), so the learned association graph itself would change: any
flip needs the association-graph gates, not this census.

## What this does not show

- It does not show that trait-centring is the right design. The 2026-10-20 census kept trait-typed neurons on the global
  centre on purpose ("neurons with no REGION preset stay identical with the flag on and off"); a trait-centred variant
  reverses that, so it is an engine and design decision (default OFF, its own six-seed gates) and is not built here.
- Result 3 is a stand-in workload on four dev seeds with a post-build rewrite, not the brain's own run: it ranks the
  hypotheses and sizes the effect, it does not reproduce the 110 (92 to 115 here).
- The census is one dev seed and one arm (intact_a); organs not stepped in these five groups (the BG selector among them)
  were not exercised. The 112 are cells on this seed's numpy random stream.
- The unchanged replies are for this seed and these five scripted conversations: the switch moved about 10000 neurons per
  group and no user-visible reply field, which says these organs' region-preset cells do not decide these replies, not
  that they cannot.
- The answer rates are measured only on a `SIM_STABLE_INTEGRATION=1` brain, on dev seed 7: both arms ran the stable
  integrator, so nothing here says what the switch does to the answer rate of the production-default (non-stable) brain.
  They come from `tools/answer_rate.py` `classify` / `rate` applied directly to each turn's reply in the census records;
  the tool's own report (`FLIP-HURTS-ANSWERS`) reads arm files laid out as load-bearing shard cells (`prefixed` or
  `paired`), the census records are not in that layout, so that verdict was not produced and no load-bearing-layout
  answer-rate check is possible on these records.
- Numeric read-outs moved by up to 12 % (table in Result 1), so "no reply field changed" is a statement about these 63
  turns' answer, abstention, recall and derivation fields, not about the quantities behind them.

## Sources

- Census records: `research/findings/raw/_stable_mechc/census/stable/s7/<group>__intact_a.json` and
  `research/findings/raw/_stable_mechc/census/stable_preset/s7/<group>__intact_a.json` (conv, infer, ask, emo, inner, each with its
  `.prov.json`); the earlier stable census they reproduce: `research/findings/raw/_stable_integration/census/on/s7/<group>__intact_a.json`.
- Comparison: `research/findings/raw/_stable_mechc/compare_s7.json` (`python -m research.runners.stable_preset_compare`).
- Probes: `research/findings/raw/_stable_mechc/lag_probe_s7.json` (`research.runners.stable_mechc_lag_probe`) and
  `research/findings/raw/_stable_mechc/lag_trait_probe.json` (`research.runners.stable_mechc_lag_trait_probe`).

## Derived

- Cells in `LearnedAssocGraph` offenders by preset: 58 + 30 + 22 = 110 (`lag_probe_s7.json`).
- Trait-centred cells below -110 mV, summed over four seeds: 3 + 9 + 7 + 4 = 23 against 92 + 99 + 115 + 97 = 403 as built;
  the same 23 is the RS count in the by-type split.
