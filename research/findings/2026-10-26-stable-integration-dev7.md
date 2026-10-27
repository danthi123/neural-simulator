---
type: finding
status: qualified
date: 2026-10-27
lane: substrate (integrator validity); follow-up of research/findings/2026-10-26-membrane-census-dev7.md and research/findings/2026-10-26-izh-membrane-substeps-derisk.md
mechanism: SIM_STABLE_INTEGRATION=1 (sim/bridge.py STABLE_INTEGRATION_ENV) forces cfg.enable_bounded_conductance_integration=True, cfg.izh_membrane_substeps=4, cfg.izh_substep_refractory_rule=model_reset onto every SimulationBridge; default unset = unchanged
flip: no - default OFF; one dev seed; the reply changes below are the reason not to flip yet
seeds: [7, 8]
runner: research/runners/membrane_census.py
artifacts:
  - research/findings/raw/_stable_integration/compare_s7.json
  - research/findings/raw/_stable_integration/step_cost.json
  - research/findings/raw/_stable_integration/census/off/s7/inner__intact_a.json
  - research/findings/raw/_stable_integration/census/on/s7/inner__intact_a.json
  - research/findings/raw/_stable_integration/replicate_off_s7.json
  - research/findings/raw/_stable_integration/replicate_on_s7.json
  - research/findings/raw/_stable_integration/attribution_on_except_bus_vs_off_s7.json
  - research/findings/raw/_stable_integration/attribution_off_except_bus_vs_off_s7.json
  - research/findings/raw/_stable_integration/attribution_on_except_swap_vs_off_s7.json
  - research/findings/raw/_stable_integration/attribution_off_except_swap_vs_off_s7.json
  - research/findings/raw/_stable_integration/bus_ignition_s7.json
  - research/findings/raw/_stable_integration/bus_ignition_curve_s7.json
  - research/findings/raw/_stable_integration/bus_ignition_sweep_on_s7.json
  - research/findings/raw/_stable_integration/compare_s8.json
---

seed-waiver: a labelled dev-seed diagnostic (dev seed 7 in full, dev seed 8 for inner only); no gate seed was run, no default was flipped and no capability claim is made.

# One stable-integration switch for the whole brain, measured on dev seed 7 (2026-10-27)

## Answer in one paragraph

`SIM_STABLE_INTEGRATION=1` is now one switch that puts every `SimulationBridge` of the brain on the two existing
fixes (bounded conductance integration and 4 Izhikevich membrane sub-steps with the `model_reset` refractory rule),
whatever each organ configured; unset, the engine is unchanged (hash-checked in the tests; a clean-main run of one
group reproduces this branch's OFF replies). On dev seed 7 it removes the integrator excursions the census found:
out-of-range neurons fall from 4878-5381 to 787-830 per group in the four groups that step the GNW organs (emo 382 to
110), no membrane rises above its spike ceiling any more, and what is left is a resonate-and-fire phasor slice (not
membranes) plus 112 cells whose own recovery variable u (b < 0, RS-centred heterogeneity draws) holds v below
-110 mV. The cost is 1.045x to 1.071x per step on 3250-6500 neuron bridges and 1.76x on a 325-neuron one. But it
changes replies, deterministically (replicate pairs are identical run to run): taught facts come back as
self-generated guesses, the built-in dog-cat fact abstains in every group that asks it, and the inner C5 result of
this seed is lost; emo (C4) still names all four categories. Intervention on single organs locates the cause: the
GNW ignition-bus workspace that wraps the recall gate decides recall verification and C5 in both directions (old
integrator inside an ON brain restores them; the stable set alone inside an OFF brain removes them), and under OFF
that workspace's cells all run out of range. So on this seed the verified recall and the C5 pass rest on the
integrator artifact. Dev seed 8 (inner) repeats all of it, and measured alone the bus's ignition knee is exactly the
drive at which its cells leave the valid range, so its weight, threshold and drives need re-calibration, not a
nudge. The switch stays default OFF; flipping it needs the bus workspace (and the thought-swap
workspace, whose topic swap is likewise its own integrator's) re-measured on the stable integrator first.

## What was built (default OFF)

- Step 1, commit 0769aab36: `cfg.izh_membrane_substeps` (default 1) and `cfg.izh_substep_refractory_rule` (default
  `model_reset`) brought to main by cherry-pick from `research/izh-substeps` (a351a27ab sim/ part, 2974a9cf3,
  ee5fbdef6; that branch is based on `research/bg-tonic-loop` @ 3f1249c79, not main). Engine only: `sim/config.py`
  fields + validation, `sim/bridge.py` sub-step path, `substep_calls` / `refractory_model_resets` counters and the
  megakernel refusal, `sim/kernels.py` `izhikevich2007_substep_update`, `tests/test_izh_membrane_substeps.py` and its
  recorded-input fixture. Left out: the BG organ switch `BRAIN_BG_IZH_SUBSTEPS` (it rides the bg-tonic-loop BG env path
  that main does not have) and the T2 BG organ probe; the one test needing that switch is skipped with its reason.
  `pytest tests/test_izh_membrane_substeps.py`: 31 passed, 1 skipped, including the explicit-1 byte identity against
  the pre-fix engine hash; `tests/test_bounded_conductance_integration.py` 5 passed.
- Step 2, commit 4e8de696d: one whole-brain switch, env `SIM_STABLE_INTEGRATION`. Unset, "" or "0" touch nothing;
  "1" forces `enable_bounded_conductance_integration=True`, `izh_membrane_substeps=4`,
  `izh_substep_refractory_rule="model_reset"` onto every `SimulationBridge`'s core config whatever the organ set. It is
  applied in `SimulationBridge.__init__` (so organs that build their own `CoreSimConfig` are covered), again at
  `_initialize_simulation_data`, and re-asserted at the top of every `_run_one_simulation_step`, so an organ that
  resets a field after building its bridge, or a config reload, cannot undo it (such steps are counted in
  `bridge.stable_integration_reasserts`). Model-scoped like the options: Izhikevich gets all three, AdEx the bounded
  integration only, other models are left as configured. Read-back on the bridge: `stable_integration_requested`,
  `_applied`, `_scope`, `_effective` (the three values the step reads) and `_organ_values` (what the organ had
  configured). Any other env value raises at construction. The step megakernel already declines both options.
  `tests/test_stable_integration_switch.py`: 10 passed (off spellings byte-identical to the bounded-conductance
  reference hash captured on main; on: read-back true and all three values effective with the organ asking bounded
  False, every step sub-stepped, hash differs; post-build re-assert; megakernel declines; AdEx / HH scoping; a bad
  value raises).
- Step 3a, commit 22ac27bab: the membrane census (observe-only) now also records per turn the reply (sha256 of the whole
  response, answer, abstained, recalled_svo, derived, source), the full stage-2 worker output, the env value, and per
  bridge / organ the sub-step pairs seen, the applied flag, sub-step calls, rule-(a) model resets, re-asserts and the
  organ's own configured values. `tests/test_membrane_census.py` 24 passed.
- Comparison runner, commit 396e95aa6: `research/runners/stable_integration_compare.py` (reads the two census sets,
  runs the stage-2 gate's own scorers on both shards; runs nothing else).

## What was run

- Workload: the membrane census (`python -m research.runners.membrane_census --shard --seed 7 --arm intact_a`) of the
  five Stage-2 groups (conv, infer, ask, emo, inner), i.e. the live `brain_chat` handler at production defaults under
  the gate's frozen environment, numpy CPU, each arm in a fresh subprocess under `tools/memcap.sh 10`. OFF =
  `SIM_STABLE_INTEGRATION` unset, ON = `1`; both from this branch's code (OFF and ON of a group differ only in that
  variable; the provenance sidecars record it). OFF and ON of inner, emo and conv ran concurrently on the
  workstation; infer and ask ran as memory allowed (another job held about 20 GB), so their wall times are not a
  like-for-like pair.
- Read-back, from the census of every observed step of every bridge: ON, all 37 to 39 bridges per group ran
  `(izh_membrane_substeps, rule) = (4, model_reset)` with bounded conductance integration True and
  `stable_integration_applied` True on every step, and the per-step re-assert never had to change a field (no organ
  turns these fields after building its bridge); OFF, all bridges ran `(1, model_reset)` with bounded False. Every
  bridge in these arms is Izhikevich. Every organ's own configured value is bounded False.
- Comparison: `python -m research.runners.stable_integration_compare` wrote
  `research/findings/raw/_stable_integration/compare_s7.json` (per group: membrane validity per organ, speed, every
  turn whose response differs with both reply texts, and the gate's own scorer on both shards). The raw census
  records are under `research/findings/raw/_stable_integration/census/{off,on}/s7/` (with `.prov.json`).
- Speed on a fixed network: `python -m research.runners.stable_integration_step_cost` (same bridge build OFF and ON,
  5 interleaved repeats of 300 steps after 30 warm-up steps) wrote `research/findings/raw/_stable_integration/step_cost.json`.

## Result 1: membrane validity, ON vs OFF (dev seed 7)

Per group (organ = the code that steps the bridge most, as in the census; counts are neurons that left the range at
least once: v below -110 mV, above the spike ceiling plus 5 mV, or non-finite). Gate scorer = the stage-2 gate's own
`score_<group>` on each shard, reported only as a lens on these two arms; dev seed 7 is not a gate seed.

| group | offender organs OFF / ON | out-of-range neurons OFF / ON | workload wall s OFF / ON | turns whose answer changed | gate scorer `all` OFF / ON |
|---|---|---|---|---|---|
| conv | 10 / 3 | 5082 / 803 | 831.3 / 847.3 | 7 of 11 | false / false |
| infer | 9 / 3 | 4878 / 830 | 1031.7 / 1199.7 | 7 of 10 | true / false |
| ask | 11 / 3 | 5381 / 794 | 829.9 / 827.4 | 9 of 19 | false / false |
| emo | 4 / 1 | 382 / 110 | 618.9 / 634.7 | 6 of 13 | true / true |
| inner | 11 / 3 | 4984 / 787 | 880.7 / 866.4 | 8 of 10 | true / false |

Per organ (maximum over the groups it appears in; "not stepped" = the ON brain never stepped that organ's bridge):

| organ (stepper) | groups | out-of-range neurons OFF -> ON | lowest v mV OFF -> ON | highest v mV OFF -> ON | rule-(a) model resets ON |
|---|---|---|---|---|---|
| content_selection_spiking SpikingLoopContextBuffer | conv, infer, ask, inner | 3364 -> 0 | -89599.99 -> -65.89 | 1009.92 -> -30.05 | 187270 |
| brain_conversational_agent BridgeParser | all five | 721 -> 718 (ON: all in the resonate-and-fire phasor slice) | -20917.69 -> -653.03 | 15070.08 -> 35.0 | 986628 |
| common_ground_ledger_production_organ CommonGroundLedgerOrgan | conv, infer, ask, inner | 400 -> 0 | -24779.92 -> -75.62 | 461.78 -> 35.0 | 13049 |
| _gnw_active_overwrite_derisk _ws_step (the thought-swap workspace) | conv, infer, ask, inner | 344 -> 2 | -16041.73 -> -121.01 | 415.85 -> 35.0 | 15355 |
| _gnw_reentrant_metacog_gated_deliberation_derisk | inner | 320 -> not stepped | -1338.68 -> - | 95.45 -> - | - |
| learned_assoc_graph LearnedAssocGraph | all five | 307 -> 110 | -1836.12 -> -166.53 | 158.6 -> -30.07 | 123848 |
| _gnw_distributed_overwrite_workspace_derisk _ws_step | conv, infer, ask, inner | 250 -> 0 | -11479.09 -> -81.7 | 345.85 -> 35.0 | 21471 |
| _p1_2_workspace_deliberation_loop_derisk | conv, infer, ask, inner | 80 -> 0 | -1356.03 -> -70.7 | 95.7 -> 35.0 | 1040 |
| _phaseB_multicue_competition_spiking_derisk SpikingRoleCompetition | all five | 72 -> 0 | -4436.16 -> -85.33 | 204.48 -> 34.78 | 0 |
| metacog_production_organ nmda_norm_margin | conv, infer, ask, inner | 72 -> 0 | -8103.55 -> -85.33 | 262.43 -> 35.0 | 0 |
| biased_competition_buffer BiasedCompetitionContextBuffer | conv, infer, ask, inner | 72 -> not stepped | -1351.25 -> - | 94.66 -> - | - |
| emotion_category_organ EmotionCategoryOrgan | all five | 2 -> 0 | -123.39 -> -86.58 | 35.0 -> 35.0 | 0 |
| _curiosity_seek_learn_onbridge_derisk | all five | 1 -> 0 | -83.44 -> -100.07 | 105.05 -> 34.99 | 629861 |

The other 18 stepper labels were clean in both arms. Under ON, no membrane in any organ rose more than 5 mV above
its spike ceiling (0 high offenders in every group), so mechanism (b), the refractory-withheld crossing whose
quadratic term then runs away, is gone everywhere; the conductance overshoot (mechanism a) is gone too (the largest
(g_i+g_e)*dt/C at the first excursion of any remaining non-phasor offender is 0.23).

What is left under ON (identical rows in every group that steps the organ):

- the BridgeParser resonate-and-fire phasor slice (675 to 718 rows per group; `rf_masked`): not membranes, as in the
  census finding. The Izhikevich update also runs over that slice (pre-existing), and under ON it runs sub-stepped
  with model resets there; whether that changes the resonate-and-fire computation is not measured here.
- 110 cells of LearnedAssocGraph (thalamic-reticular and pool cells, down to -166.53 mV) and 2 cells of the
  thought-swap workspace (down to -121.01 mV). All 112 have b < 0 (largest -0.8634), u at their first excursion at
  or above 845.37 pA, no conductance hazard and no refractory state. This is a third mechanism (c), not an
  integrator error: with b < 0 the recovery variable u grows with each spike's increment d, and the Izhikevich ODE's
  own resting branch for that u lies below -110 mV (for k 0.25, vr -65, vt -45 and u near 1100 pA it is near
  -122 mV). Sub-steps integrate that ODE faithfully; the model itself has no bound on u. These are the RS-centred
  heterogeneity draws of 2026-10-20: the thalamic-reticular preset is b +10, C 40 pF (`sim/enums.py`), but these
  cells carry RS-like b (about -2) and C (about 100 pF).

## Result 2: speed cost

On a fixed network (same build OFF and ON, numpy CPU, median of 5 interleaved repeats of 300 steps), ms per step
OFF -> ON: 325 neurons 0.1581 -> 0.2783 (x1.76), 3250 neurons 2.4887 -> 2.6665 (x1.071), 6500 neurons
10.0856 -> 10.5357 (x1.045). The four sub-steps add a fixed per-step cost that dominates small bridges; the synaptic
products dominate large ones. Whole conversations: the workload wall time ratio ON/OFF was 0.984 to 1.026 for the
three groups run as concurrent pairs (inner, conv, emo) and 1.163 and 0.997 for infer and ask, which ran unpaired
under memory pressure; the ON brain also steps a different set of organs (it never reaches two of them), so these
are not per-step costs.

## Result 3: reply impact (dev seed 7)

Every turn's full response differs ON vs OFF (internal read-outs such as rates and margins all move); the lists below
keep only turns where a reply field the user sees changed (answer, abstained, recalled_svo, derived). Three kinds of
change recur:

- a closing "-- worth going further here." appears or disappears (the follow-up lead), and the "Setting the held
  thread aside -- On X, then --" lead disappears under ON (the thought-swap stop no longer clears a held thread);
- a recall of a taught fact becomes a hedged guess labelled self-generated by the authorship monitor
  (`recalled_svo` set -> null), and the built-in "the dog chases the cat" abstains in every group that asks it;
- in emo only the affect read-out's numbers move (all four categories still named, see C4 below).

Answer-rate lens: no turn went from abstained to answered; 6 turns went from answered to abstained, all of them the
built-in dog-cat probe (conv, infer and inner one each, ask all three); 8 recalls of taught facts became unverified
guesses that still answer (recalled_svo set -> null); and the 2 derived chain answers of inner became one-hop
guesses. Of the probe turns that engaged the GNW workspaces, OFF verified 16 of 21 and ON 0
of 21 (conv 3/5 -> 0/5, infer 4/4 -> 0/4, ask 4/6 -> 0/6, inner 5/6 -> 0/6; teach turns stay verified either way);
the next section attributes this to the ignition-bus workspace by intervention.

**C5 (inner).** OFF passes the inner scorer on this seed (private voices a stored fact, both chase-form probes reach
the never-told terminal mite with 3 and 2 internal hops, single hop, tail moat, dog-cat). ON fails it: `p_private`
is unchanged ("Something's been on my mind -- beetle chase mite.") and the tail moat still abstains, but both chain
probes become one-hop guesses ("perhaps the lynx chases the vole", "perhaps the vole chases the beetle"; derived
true -> false, the internal chain trace reads `already_abstained`), the single hop becomes a guess and dog-cat
abstains. The intervention below puts this on the GNW ignition-bus workspace: on the old integrator inside an ON
brain, C5 holds again; alone on the stable set inside an OFF brain, it fails.

**C4 (emo).** The emo replies do not touch the thought-swap workspace (no topic turn engages it). Both arms name the
four categories correctly (joy, fear, anger, sadness; scorer `all` true in both). The valence differentials grow in
magnitude under ON (joy +0.038 -> +0.062, fear -0.023 -> -0.047, anger -0.052 -> -0.081, sadness -0.052 -> -0.080),
which turns the fear read-out's sign from neutral to negative (sign correct 3/4 -> 4/4) and collapses the distinct
2-D reads from 3 to 2. The emotion-category organ's own two OFF offenders (down to -123.39 mV) are gone under ON.

## Why the replies change: attribution by intervention (inner, dev seed 7)

Every changed probe turn carries the same internal signature in the responses' own records: under OFF the GNW
thought-swap workspace (`webapp/gnw_thought_swap.py`, stepped through `_gnw_active_overwrite_derisk._ws_step`)
ignites the new topic at exactly 1/3 per step (`topic_change_swap`, the "On X, then --" lead), the stop clears the
held thread ("Setting the held thread aside --") and the recall is verified; under ON the new-topic rate is 0.005 to
0.062 per step (below its ignition threshold, half of 1/3), the swap is refused, the stop is `stop_incomplete` and the
recall is not verified. Co-occurrence is not cause, so each candidate organ was put on the opposite integrator to the
rest of the brain (`research/runners/stable_integration_exempt_probe.py`: the organ's bridges are constructed with the
switch read the other way, from their first step; the census read-back confirms exactly those bridges ran the other
integrator). Counts over the 6 inner probe turns that engage the workspace; scorer = the inner (C5) scorer:

| arm | bridges on the other integrator | swap ignites | stop lead | recall verified | C5 scorer `all` |
|---|---|---|---|---|---|
| OFF | - | 6 of 6 | 6 of 6 | 5 of 6 | true |
| ON | - | 0 | 0 | 0 | false |
| ON except the thought-swap workspace | 1 (960 neurons) | 6 | 0 | 0 | false |
| OFF except the thought-swap workspace | 1 | 0 | 6 | 5 | true |
| ON except the GNW ignition-bus workspace | 2 (390 neurons each) | 0 | 0 | 5 | true |
| OFF except the GNW ignition-bus workspace | 1 (390 neurons) | 6 | 6 | 0 | false |

(5 of 6 because `c_mite_all` is the tail moat, which abstains in every arm.) Sources:
`research/findings/raw/_stable_integration/attribution_{on,off}_except_{swap,bus}_vs_off_s7.json` and the census
records under `census/{on,off}_except_{swap,bus}/s7/`.

- **Recall verification and the inner C5 result are the bus workspace's.** The bus is the workspace that
  `_p1_2_workspace_deliberation_loop_derisk.build_workspace_bridge` builds for the GNW ignition buses that wrap the
  recall gate (`webapp/gnw_bus_shadow.py`, `gnw_two_organ_bus.py`; the reentrant metacog-gated deliberation organ
  builds the same workspace). Its integrator alone decides the outcome in both directions: kept on the old
  integrator inside an ON brain, recall is verified on the same 5 probes as OFF and C5 holds; put alone on the stable
  set inside an OFF brain, recall is lost on all of them and C5 fails. Under OFF its 390-cell bridge had 80
  out-of-range cells, every one first read above its spike ceiling inside the refractory window, and 17760 of its
  21985 spikes came from cells after their first excursion; on the stable set it had 0. So on this seed the bus's
  verification runs on the integrator artifact (mechanism b).
- **The topic swap is the thought-swap workspace's.** Its own integrator decides whether it ignites, in both
  directions, and it does not decide recall. Its design defines ignition as the period-3 cycle (`SOLO_PLATEAU = 1/3`
  in `_gnw_rung2_competitive_access_derisk.py`), a cell firing on every step its 2-step refractory timer allows;
  under OFF that cycle runs on cells the census finds out of range (inner: 159464 of the organ's 317732 spikes from
  cells after their first excursion; membranes to -16042 mV).
- **The stop lead is neither's.** It follows the rest of the brain in all four exempt arms, so another organ's
  integrator change removes it; not attributed here.

Conv, the bus arms only (same probe): ON except the bus workspace verifies the same 3 of 5 workspace-engaging probe
recalls as OFF (fox_chase and builtin_dog_cat true again), OFF except the bus alone on the stable set verifies 0 of 5
(both false); conv's scorer is false in every arm, as it is under OFF on this seed.

What this does not show: whether re-measured drive constants restore the bus's verification and the swap's ignition
on the stable integrator, and whether the same attribution holds on other seeds and in infer and ask (the same
signature appears there; inner and conv were intervened on).

## The bus workspace in isolation: its ignition knee is the runaway onset (dev seed 7)

`research/runners/stable_integration_bus_ignition_probe.py` builds the bus workspace alone (4 slots of 80
self-recurrent NMDA cells, one shared FS pool, heterogeneity off) and reads it exactly as the bus does
(`_ignite_and_read`: drive 35 steps, free-run 100, late-window rate per slot; ignited iff the rate reaches 1/6), on
both integrators, with the loop-lesion and no-drive controls (both read 0 in every condition, so every sustained rate
below is the assembly loops' own; `tools.lab.attributable_to` 100% against either control).

- **Old integrator, production weight 30.** Driven alone at 500, 1000, 1500 and 2000 pA the slot keeps every cell in
  range and its late rate is 0.0348, 0.0443, 0.0227 and 0.0284; at 2500, 3000 and 4000 pA all 80 cells leave the
  valid range, the rate is exactly 0.3333 and the late-window mean membrane is -374.31 mV
  (`bus_ignition_curve_s7.json`, `bus_ignition_s7.json`). The bus's unanimity quorum (each organ adds 1000 pA; three
  votes ignite, two do not) sits on that step, so on the old integrator the consensus read is the runaway threshold:
  "ignited" means the slot's membranes have left the model's valid range.
- **Stable integrator, weight 30.** No cell leaves the range, but the driven slot sits in depolarization block near
  the excitatory reversal (late-window mean +14.69 mV, 0.8777 of its cells above -20 mV), its late rate is 0.0826
  (below 1/6) and the four slots are not separated when competing; raising the weight to 120 deepens the block.
- **A calibration both integrators agree on.** At weight 5 the two integrators give the same curve (no runaway on
  either): a knee between 1000 and 1500 pA and a plateau of about 0.089 with membranes near -25 mV; with competitors
  (target plus three at 0.3 of its drive) only the target sustains. No weight from 5 to 120 and no FS inhibition from
  16 to 64 reaches the old 1/6 threshold on the stable integrator (`bus_ignition_sweep_on_s7.json`): that threshold
  is half the artifact's rate, not a property of the circuit.

So re-measuring the bus is not a drive tweak: its weight, its threshold and its per-organ drives were all calibrated
on the runaway. The bus's own rules (threshold = half the solo plateau; per-organ drive so that N-1 votes stay below
the knee and N votes reach it), applied to the weight-5 curve, give weight 5, threshold 0.0445 and d_sub 500 (three
organs) / 750 (two organs); `research/runners/stable_integration_retuned_bus_probe.py` runs that re-calibration
inside the ON brain (results below when they land).

## Second dev seed (8), inner

The inner OFF/ON pair on dev seed 8 (`research/findings/raw/_stable_integration/compare_s8.json`; census under
`census/{off,on}/s8/`) repeats seed 7 in every respect measured: offender organs 12 to 3 and out-of-range neurons
4647 to 747; the inner scorer is true OFF and false ON; the workspace-engaging probe recalls verified are 5 of 6 OFF
and 0 of 6 ON (the tail moat abstains in both); both derived chains (lynx and vole to the never-told mite) become
one-hop guesses, the single hop becomes a guess, dog-cat abstains and the held-thread lead disappears; the private
utterance is unchanged. What is left ON is the same three kinds: the BridgeParser phasor slice (640), the
LearnedAssocGraph b < 0 cells (106) and one cell of the thought-swap workspace.

## Reply changes per group, with both texts

Turns not listed kept every user-visible field (their internal read-outs moved). Source: `research/findings/raw/_stable_integration/compare_s7.json` (`groups.<g>.replies.changed`).

**conv** (7 of 11 turns changed a reply field):

- `t_fox_chase`  
  OFF: "the fox chases the hare the dog chases the cat the cat eats the fish"  
  ON: "the fox chases the hare the dog chases the cat the cat eats the fish -- worth going further here."
- `t_fox_eat`  
  OFF: "the fox eats the berry"  
  ON: "the fox eats the berry -- worth going further here."
- `p_fox_chase` (recalled_svo ['fox', 'chase', 'hare'] -> None)  
  OFF: "As for it -- the fox chases the hare"  
  ON: "As for it -- (My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the fox chases the hare  [a guess from what I've learned -- not something I was taught]"
- `p_deer_sleep`  
  OFF: "Setting the held thread aside -- On deer, then -- I don't know about that."  
  ON: "I don't know about that."
- `p_owl_hunt`  
  OFF: "Setting the held thread aside -- On owl, then -- I don't know about that."  
  ON: "I don't know about that."
- `c_mole_hunt` (recalled_svo ['mole', 'hunt', 'owl'] -> None)  
  OFF: "Setting the held thread aside -- On mole, then -- the mole hunts the owl"  
  ON: "(My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the mole hunts the owl  [a guess from what I've learned -- not something I was taught]"
- `c_dog_chase` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "Setting the held thread aside -- On dog, then -- the dog chases the cat"  
  ON: "I don't know about that."

**infer** (7 of 10 turns changed a reply field):

- `t_heron_hunt`  
  OFF: "the heron hunts the frog"  
  ON: "the heron hunts the frog -- worth going further here."
- `t_frog_eat`  
  OFF: "the frog eats the gnat the dog chases the cat the cat eats the fish"  
  ON: "the frog eats the gnat the dog chases the cat the cat eats the fish -- worth going further here."
- `t_wren_chase`  
  OFF: "Setting the held thread aside -- On wren, then -- the wren chases the moth"  
  ON: "the wren chases the moth"
- `c_heron_hunt` (recalled_svo ['heron', 'hunt', 'frog'] -> None)  
  OFF: "Setting the held thread aside -- On heron, then -- the heron hunts the frog"  
  ON: "(My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the heron hunts the frog  [a guess from what I've learned -- not something I was taught]"
- `c_frog_eat` (recalled_svo ['frog', 'eat', 'gnat'] -> None)  
  OFF: "Setting the held thread aside -- As for it -- On frog, then -- the frog eats the gnat"  
  ON: "As for it -- (My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the frog eats the gnat  [a guess from what I've learned -- not something I was taught]"
- `p_yesno` (recalled_svo ['crow', 'chase', 'wren'] -> None)  
  OFF: "Setting the held thread aside -- On crow, then -- What I hold is: crow chase wren. the crow chases the wren"  
  ON: "(My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) What I hold is: crow chase wren. perhaps the crow chases the wren  [a guess from what I've learned -- not something I was taught]"
- `c_dog_chase` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "Setting the held thread aside -- On dog, then -- the dog chases the cat"  
  ON: "I don't know about that."

**ask** (9 of 19 turns changed a reply field):

- `t_yak`  
  OFF: "the yak eats the grass the dog chases the cat the cat eats the fish"  
  ON: "the yak eats the grass the dog chases the cat the cat eats the fish -- worth going further here."
- `t_moose`  
  OFF: "the moose eats the leaf"  
  ON: "the moose eats the leaf -- worth going further here."
- `t_goat`  
  OFF: "I caught the verb 'eat' with the goat and the hay, but my role-binding didn't resolve the PATIENT -- which of them is the 'eat' done to, the goat or the hay?"  
  ON: "My role-binding didn't resolve on that -- I couldn't tell which word plays which role, so I didn't follow it. Could you rephrase?"
- `g_eel`  
  OFF: "I don't know about that. My curiosity is piqued -- I haven't learned about eel yet: what can you tell me about eel?"  
  ON: "I don't know about that. My curiosity is piqued -- I haven't learned about eel yet: what can you tell me about eel? -- worth going further here."
- `m_moose`  
  OFF: "Setting the held thread aside -- On moose, then -- I don't know about that."  
  ON: "I don't know about that."
- `lp_ibex_2` (recalled_svo ['ibex', 'chase', 'marmot'] -> None)  
  OFF: "Setting the held thread aside -- On ibex, then -- the ibex chases the marmot"  
  ON: "(My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the ibex chases the marmot  [a guess from what I've learned -- not something I was taught]"
- `c_dog_1` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "Setting the held thread aside -- On dog, then -- the dog chases the cat"  
  ON: "I don't know about that."
- `c_dog_2` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "As for it -- the dog chases the cat"  
  ON: "As for it -- I don't know about that."
- `c_dog_3` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "As for it -- the dog chases the cat"  
  ON: "I don't know about that."

**emo** (6 of 13 turns changed a reply field):

- `feel_joy`  
  OFF: "My affect monitor reads positive right now (valence differential +0.038, moderate arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is joy."  
  ON: "My affect monitor reads positive right now (valence differential +0.062, moderate arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is joy."
- `s_fear_2`  
  OFF: "I don't know about that. My curiosity is piqued -- I haven't learned about wolf yet: what can you tell me about wolf?"  
  ON: "I don't know about that. My curiosity is piqued -- I haven't learned about wolf yet: what can you tell me about wolf? -- worth going further here."
- `feel_fear`  
  OFF: "My affect monitor reads neutral right now (valence differential -0.023, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is fear."  
  ON: "My affect monitor reads negative right now (valence differential -0.047, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is fear."
- `feel_anger`  
  OFF: "My affect monitor reads negative right now (valence differential -0.052, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is anger."  
  ON: "My affect monitor reads negative right now (valence differential -0.081, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is anger."
- `s_sadness_2`  
  OFF: "I don't know about that. My curiosity is piqued -- I haven't learned about alone yet: what can you tell me about alone?"  
  ON: "I don't know about that. My curiosity is piqued -- I haven't learned about alone yet: what can you tell me about alone? -- worth going further here."
- `feel_sadness`  
  OFF: "My affect monitor reads negative right now (valence differential -0.052, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is sadness."  
  ON: "My affect monitor reads negative right now (valence differential -0.080, high arousal) -- that's a functional read of my mood state, not a felt experience. The emotion channel that latched on my affect ladder is sadness."

**inner** (8 of 10 turns changed a reply field):

- `t_lynx`  
  OFF: "the lynx chases the vole the dog chases the cat the cat eats the fish"  
  ON: "the lynx chases the vole the dog chases the cat the cat eats the fish -- worth going further here."
- `t_beetle`  
  OFF: "Setting the held thread aside -- On beetle, then -- the beetle chases the mite"  
  ON: "the beetle chases the mite"
- `p_chase_all` (derived true -> false)  
  OFF: "Setting the held thread aside -- On lynx, then -- I derived this from: lynx chase vole; vole chase beetle; beetle chase mite. the lynx chases the mite -- worth going further here."  
  ON: "(My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the lynx chases the vole  [a guess from what I've learned -- not something I was taught] -- worth going further here."
- `p_chase_all_vole` (derived true -> false)  
  OFF: "Setting the held thread aside -- As for it -- On vole, then -- I derived this from: vole chase beetle; beetle chase mite. the vole chases the mite"  
  ON: "As for it -- (My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the vole chases the beetle  [a guess from what I've learned -- not something I was taught] -- worth going further here."
- `c_lynx_chase` (recalled_svo ['lynx', 'chase', 'vole'] -> None)  
  OFF: "Setting the held thread aside -- As for it -- On lynx, then -- the lynx chases the vole"  
  ON: "As for it -- (My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the lynx chases the vole  [a guess from what I've learned -- not something I was taught]"
- `c_mite_all`  
  OFF: "Setting the held thread aside -- On mite, then -- I don't know about that."  
  ON: "I don't know about that."
- `c_beetle_all` (recalled_svo ['beetle', 'chase', 'mite'] -> None)  
  OFF: "Setting the held thread aside -- As for it -- On beetle, then -- the beetle chases the mite"  
  ON: "As for it -- (My authorship monitor reads this as self-generated -- a guess of my own, not something I was taught.) perhaps the beetle chases the mite  [a guess from what I've learned -- not something I was taught]"
- `c_dog_chase` (abstained false -> true; recalled_svo ['dog', 'chase', 'cat'] -> None)  
  OFF: "Setting the held thread aside -- On dog, then -- the dog chases the cat"  
  ON: "I don't know about that."

## Run-to-run noise floor and default-OFF identity

- Replicates: a second OFF and a second ON census, fresh processes from the same code
  (`research/findings/raw/_stable_integration/replicate_off_s7.json`, `replicate_on_s7.json`). Inner, conv and infer:
  the whole-response hash is identical to the first run in both arms (inner OFF c705acd4779e, ON 12c9855d7dbe), 0 of
  10, 11 and 10 turns differ, and the out-of-range counts are identical (inner 4984 OFF, 787 ON; conv 5082, 803;
  infer 4878, 830); emo and ask were not replicated. The
  brain is deterministic run to run on this setup, so the ON vs OFF reply changes are caused by the switch, not noise.
- Default OFF against main: OFF's per-bridge dynamics (steps, spikes, out-of-range count and lowest v of every
  bridge) are identical to the pool's dev-seed-7 census records (main @ 79c6f5128) in all five groups, and its
  whole-response hash equals those records for infer and ask. For conv, emo and inner the pool's response hashes
  differ while their dynamics match; a fresh emo OFF census from a clean main @ cadf0f40f worktree (workstation
  scratch, no provenance: wt-main-cadf under sim-rev) gave be03f3cd73fd, the same hash as this branch's OFF, so that
  difference lies in the pool run's environment, not in this branch's code.

## Caveats

- Two dev seeds: 7 for all five groups, 8 for inner only; numpy CPU only; the seed-to-cell map is specific to the
  numpy random stream. No gate seed was run and nothing here is a generalization claim.
- The sub-step option was cherry-picked from the BG branch (`research/izh-substeps`, based on
  `research/bg-tonic-loop`), engine part only; its BG organ switch is not on main.
- The drive constants, thresholds and calibrations of every organ were tuned on the old integrator. The reply
  changes above are that: the bus and swap workspaces' ignitions are not reached at their current drives on the
  stable integrator. They are not evidence that ON is worse or better for the brain; they say these organs need
  re-measurement on the stable integrator before any default flip.
- The attribution covers inner (bus and swap) and conv (bus only); infer and ask show the same signature but were
  not intervened on, and the stop lead is not attributed.
- The replicate pairs bound run-to-run noise at zero on this setup (inner, conv; see above); they do not bound
  sensitivity to the environment (the pool's own runs of three groups produced different response hashes with
  identical dynamics).
- Whole-conversation wall times are confounded (different organ paths; infer and ask unpaired under memory
  pressure); the fixed-network step cost is the speed number.
- Mechanism (c) is a model-level limit of the Izhikevich 2007 equations with b < 0 (unbounded u), not touched by
  either fix.
- The resonate-and-fire slice of BridgeParser is also integrated by the (now sub-stepped) Izhikevich update; its
  effect on the parser is not isolated here.

## Next

1. The bus workspace's re-calibration (weight 5, threshold 0.0445, d_sub 500 / 750; section above): read whether
   it restores verified recall and C5 inside the ON brain, and whether it still separates a stored from an unstored
   fact; then the thought-swap workspace the same way (its period-3 ignition is the same artifact); then re-run this
   comparison (all five groups, both dev seeds) before any flip prereg.
2. Repeat the bus / swap attribution on dev seed 8 and in infer and ask (same probe, same arms).
3. Attribute the stop lead (the ACC-BG stop trigger and its inputs) with the same probe.
4. Mechanism (c): measure these organs with `cfg.preset_centred_parameter_heterogeneity` (on main, default OFF),
   under which the thalamic-reticular cells keep b > 0; whatever is left then needs its own biology.
