# Shortcut burndown inventory — every genuine host-computation shortcut in the SHIPPED pipelines (2026-06-20)

**Type:** READ-ONLY audit (no code, no experiments). Single deliverable = this doc. Stayed on `main`.
**Goal:** a complete burndown list of every NON-SPIKING host computation that does a COGNITIVE job (perception/salience,
normalization, selection, binding, reward, value, comparison, routing) between sensation and action in the project's
**SHIPPED** pipelines — so the owner can close them before resuming capability expansion. Exhaustive over the shipped
conversational loop + the merged nav cascade; NOT every research runner.

**The bar (applied strictly).** A genuine shortcut = a host op doing cognition, EVEN IF the host calc is biologically
correct. Host code is LEGITIMATE only for (1) the ENVIRONMENT (world state + rendering the sensory input the neural
front-end then receives) and (2) the BODY (acting on motor output — reading WHICH motor pool / answer-channel a SPIKING
competition selected, to move the agent / emit a token). A host `argmax / max / mean / normalization / comparison` doing
a COGNITIVE op (selecting the answer, deciding the action, computing reward/value/salience, normalizing a code) = a
shortcut. A body-read of a spiking result to act = legitimate. (Framing: CLAUDE.md "Standing standard: BRAIN-BASED ONLY";
`feedback_brain_based_only_standard`.)

**Method.** Read in full the shipped conversational files (`one_brain_composer.py`, `rf_phasor_composer.py`,
`brain_conversational_agent.py`, `multi_turn_agent.py`, `_phaseC_task2_wholeturn_loop.py`,
`consolidated_320_conversation_demo.py`), the merged-nav builder + gate runner + composer
(`nav_conv_merged_bridge.py`, `_nav_gate_merged_run.py`), and grepped/sub-agent-audited `g11_bg_runner.py` (the nav
cognition) for the host cognitive ops. Cross-checked the documented-shortcut notes + the boundary ledger
(`2026-06-20-boundary-ledger-dendritic-audit.md`, commit `79a9fb1d`). Every cited file:line verified.

---

## TOP-LINE COUNT (the honest call)

**12 genuine shortcuts remain in the shipped pipelines.** They split:

| closure bucket | count | meaning |
|---|---|---|
| **CLEAN-CONVERSION** (spiking replacement already validated/de-risked; just adopt) | **5** | a drop-in spiking mechanism exists and matched the host op multi-seed — the work is to flip the default / wire it in |
| **OPEN-RESEARCH** (no validated spiking replacement yet; point-neuron engineering) | **4** | the nav reward/value/perception loop — organs exist in isolation, the closed loop does not sustain |
| **DEEP-FRONTIER (NEGATIVE-so-far)** | **1** | the composer's exact-inverse FHRR algebra + given codes — the learned-cortex replacement tested NEGATIVE |
| **ALREADY-SPIKING-DEFAULT** (listed for completeness; NOT counted as remaining) | **2** | the nav action decision (library default) + the parser comprehension — already neural |

So: of the 12 remaining, **5 are cheap clean conversions, 4 are genuine open research (all in the nav reward/value/place
loop), 1 is the deep frontier.** The conversational who/what core is ~1 clean-conversion + 1 deep-frontier away from
fully brain-based; the navigation cascade is where most of the host cognition still lives.

**The single most-surprising shortcut** (one the project's own notes do NOT foreground as "still host"): **#1 — the
production OneBrainComposer's cleanup SELECTION is a host `np.argmax` over the matched-filter membrane** (`one_brain_composer.py:308/427`).
The matched-FILTER is on-substrate (complex-synapse matvec), but the WINNER-PICK is a numpy argmax over the readout
voltages — the spiking-WTA cleanup (`RFPhasorComposer._spiking_cleanup`, validated == numpy at D=2048) is **opt-in only
on the `rf` composer and is NOT wired into `OneBrainComposer`**, which is now the production default
(`consolidated_320_conversation_demo.py --composer default="onebrain"`). So the flagship one-brain conversation selects
every recalled word with a host argmax. CLAUDE.md presents the NEF/spiking cleanup as "CLEARED," but that clearance
lives on the legacy `rf` path, not on the shipped one-brain default.

---

## THE BURNDOWN TABLE

| # | shortcut (file:func/line) | cognitive op | why it's a shortcut (not a legit body/sensory boundary) | bucket | the reusable mechanism / gate it needs | priority |
|---|---|---|---|---|---|---|
| **1** | `one_brain_composer.py:308,310,427,429,489` `_read_block` / `_decode_batched_mem` / `_decode_clause`: `self.words[int(np.argmax(scores[ri]))]` over the cleanup membrane | **cleanup SELECTION** (pick the nearest concept = the recalled word) | The matched filter is on-substrate (complex matvec → `cp_membrane_potential_v`), but the host argmax PICKS the winner. A spiking WTA over the scores would make the pick neural; the host argmax is the cognitive selection, not a body-read of a spiking result. **On by default in the production `onebrain` agent.** | **CLEAN-CONVERSION** | `RFPhasorComposer._spiking_cleanup` (Stewart-Tang-Eliasmith NEF: input-norm matched filter on the complex synapse + Izhikevich WTA, winner = argmax-over-FIRING) — validated == numpy 27/27 @ D=2048 (`2026-06-05-composer-cleanup-NEF-GO.md`). Wire it into OneBrain's `_read_block`/`_decode_batched_mem` (it exists only on `rf` via `enable_spiking_cleanup`). | **P1** |
| **2** | `rf_phasor_composer.py:296-297` `_cleanup`: `words[int(np.argmax(sims))]` over `np.mean(np.cos(...))` | **cleanup SELECTION** (the `rf` composer / test-oracle path; also the DA-gate margin in `nav_conv_merged_bridge.py:1351-1353,1365`) | Same op as #1, on the `rf` composer (the default when `composer_kind="rf"`, and the numpy oracle). `enable_spiking_cleanup=False` by default → the cosine + argmax IS the selection. | **CLEAN-CONVERSION** | Same `_spiking_cleanup` (already in the file, gated by `enable_spiking_cleanup`). The fix is to default it ON (or accept `rf` as the explicit numpy oracle + ship `onebrain`+spiking-cleanup). Same lever as #1. | **P1** (with #1) |
| **3** | `one_brain_composer.py:443-446,517,522-525` `_scan` / `query_agent` / `ask_yes_no`: `for got in self._read_blocks(): if all(got.get(role)==want ...): return ...` | **cue-MATCH COMPARISON + first-match routing** (decide which stored block answers, or abstain) | The store/unbind/cleanup are on-substrate, but the host `for/if/return` compares the decoded cue words and routes the answer / abstains. This is the no-confab moat's DECISION, done in Python. | **CLEAN-CONVERSION** | The Phase-B spiking sequencer (`_phaseB_onebrain_sequencer_derisk.build_sequencer_bridge`/`run_sequencer`: spiking BG/thalamocortical gated-match cascade + BG production rule → ans0/ans1/abstain). Validated ==host, moat 0 false-accepts, lesion fails-safe, permuted-rule inverts (`2026-06-19-onebrain-sequencer-derisk.md`); composed into the K=2 whole-turn loop (`_phaseC_task2_wholeturn_loop.py`). Gate: scale the sequencer past K=2 and wire into OneBrain's query path. | **P2** |
| **4** | `_phaseC_task2_wholeturn_loop.py:112-115,146-147` `block_role_scores` → `run_sequencer`'s `scores_to_drive` (the "S5" host read) | **result→sequencer DATA hand-off** (read the cleanup score to host to drive the sequencer's decoded word-lines) | This is the ONE residual host DATA read inside the otherwise-on-substrate whole-turn loop. NOTE: the on-bridge projection (option a) was tested and WALLS (the graded-magnitude-through-a-binary-spike limit, Task-1 verdict `2026-06-19-phaseC-task1-S5-seam-derisk.md`); option (b) host-read is the current stand-in. | **CLEAN-CONVERSION** (de-risked, build pending) | Divisive GAIN on the DIAGONAL (Carandini-Heeger; the `input_divisive_norm` primitive already in `sim/bridge.py`; NEF input-norm FS pool already at D=2048). The rectified non-negative score has no common mode → the diagonal/gain half, not off-diagonal whitening. Deep-research GO `2026-06-19-S5-on-bridge-normalization-deep-research.md` (commits `94ca9fb8`/`1270397b`); the falsification de-risk (peak-sweep + moat) is the next step, **not yet run**. | **P2** (with #3) |
| **5** | `_phaseB_onbridge_stream_conversation_derisk.py:120` (offline cortex-code GENERATION): `double_center(L)` (log + per-hub + per-concept mean-subtraction) | **read-out NORMALIZATION** (PPMI double-centring that makes the learned cortex codes generalize) | Host `double_center` is the DEFAULT (`--readout-norm default="host"`); it produces the cached `.npy` codes the production agent loads. The normalization is a cognitive gain-control op the cortex should do with neurons. (Offline preprocessing, not in the runtime conversational path — a learning-pipeline shortcut.) | **CLEAN-CONVERSION** | The `neural` path EXISTS and is de-risked: per-hub spike-frequency adaptation + per-concept feedforward inhibition (`_phaseB_biologize_readout_norm_derisk.neural_norm`, `--readout-norm neural`) — 96% of host (`2026-06-16-biologization-sweep-conversational-pipeline.md`; ledger #4/#5). Gate: produce the 320-scale `neural` codes for seeds 43/44 (currently only seed 42 cached) and default the demo to `--readout neural`. | **P3** |
| **6** | `g11_bg_runner.py:2937-2944` (nav step): `if gx > x: cp_external_input_current[cortex_E]=HEURISTIC_DRIVE_PA` (N/E/S/W Manhattan compare); gated by `heuristic_strength` (default **1.0**) | **goal-pointing ORIENTING / salience** (decide which cardinal points toward the goal and bias the cortex toward it) | The host compares agent (x,y) to goal (gx,gy) and injects a goal-direction drive into the action cortex — this is the orienting DECISION done by a host distance comparison. **ON by default** (`heuristic_strength=1.0`); the merged gate runner only zeros it under `--spiking-sc`. The single most load-bearing nav cognitive shortcut. | **OPEN-RESEARCH** (organ exists; closed loop NO-GO) | Spiking superior colliculus (`enable_spiking_sc`: sc_retina→sc_map Mexican-hat WTA→sc_rostral; orienting from sc_map→cortex_X pooling). N1 validated standalone (`sc_n5_rpe_probe`); but the **deployed closed loop is NO-GO** (`2026-06-19-nav-spiking-sc-deploy-NO-GO.md`, ~58× worse, actor goes silent — ledger #13). Needs the deep-research gate as a unit. | **P4** |
| **7** | `g11_bg_runner.py:7197,7260-7271` (nav step): `current_reward_signal = delivered_reward`; `reward_us` driven by `reward_us_drive_pa * max(0,reward)` (the `else` branch) | **REWARD computation** (compute the scalar reward `r` from the host distance change) | The reward `r` is computed by a host formula (`dist_after < dist_before → +1` etc., lines 7056-7061) and written into the bridge as a scalar — the brain does not compute it. **Host-default ON**; the synaptic SC-proximity reward (`enable_spiking_sc_approach`, which zeros the host write at :7271) is opt-in (default False). | **OPEN-RESEARCH** (organ exists; closed loop NO-GO) | `sc_rostral → reward_us` synaptic approach-reward (N5, QUALIFIED GO in isolation `2026-06-18-merged-neural-reward-QUALIFIED-GO.md`); but it is part of the same NO-GO closed loop as #6 (ledger #12/#13). | **P4** (with #6) |
| **8** | `g11_bg_runner.py:7135-7145,7284` (nav step): `reward_ema = _decay*reward_ema + (1-_decay)*reward`; `_V_scaffold = max(0, reward_ema_pre)` fed to the SNc current | **VALUE / RPE** (compute the value baseline V the dopamine RPE subtracts) | Value is a host EMA scalar; even with `spiking_snc=True` the SNc current uses the host `_V_scaffold`. **Host-default ON** (`spiking_snc=False`, `enable_neural_critic=False`). | **OPEN-RESEARCH** | The spiking striosome MSN-D1 critic (`enable_neural_critic`, GABA_B/GIRK → SNc) learns V on-substrate, but the merged value-train δ is graded-but-WEAK (~1.3× vs 4-19× ceiling, capped by the position-blind up-state floor — `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`, ledger #14). Open boundary. | **P4** (with #6/#7) |
| **9** | `g11_bg_runner.py:6563-6565,6586-6588` (nav step): `place_dsq=(pref_x-x)^2+...; place_drive=...np.exp(-place_dsq/...)`; same for `goal_drive` over (gx,gy) | **PERCEPTION → state code** (turn the agent's grid position + goal coords into a host Gaussian place/goal code) | Host computes a Gaussian over the TRUE (x,y)/(gx,gy) coords and injects it as the "place"/"goal" drive — this is a hand-coded place-cell perception, reading the coords the brain should sense. **Host-default ON** when `enable_hippocampus=True` and `neural_place_selforg=False` (the default). | **OPEN-RESEARCH** (dendritic flavor) | Self-organized spiking place code (`neural_place_selforg`: place_sensors → place threshold-WTA + place_fs PING → plastic place→striosome). But the self-org read regime is NOT location-selective (a few cells fire everywhere) and over-clamps the SNc (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`, ledger #15 — "dendritic flavor"). NOT on the conversational path. | **P5** |
| **10** | `g11_bg_runner.py:6968-6988`, `_nav_gate_merged_run.py:37` (the SHIPPED gate runner CLI): `--readout-source default="motor"` → `action_idx = max(range(N_ACTIONS), key=lambda i: counts[i])` (host argmax over motor spike counts) | **ACTION DECISION** (pick the move from motor-pool spike counts) | In the DEPLOYED gate runner the CLI default is `"motor"` → a host argmax over motor_X spike counts decides the action (the cognition is host). NB this is a SPLIT (see #11): the LIBRARY `run_moving_goal_episode` default is `"spiking_wta"`, but the shipped gate/benchmark runner passes the CLI default `"motor"`. | **OPEN-RESEARCH→CLEAN** (the spiking path is validated; the host-argmax default is the documented-benchmark oracle) | The `spiking_wta` commit-burst readout (Wang-2002 accumulator + Lo-Wang commit-burst threshold-crossing) is validated default-on in the library at 1.16× host (`2026-06-19-spiking-decision-default-on-GO.md`, ledger #7). Gate: flip the gate/demo CLI default to `spiking_wta` + `--urgency-max-pa 180` (kept `"motor"` only to reproduce historical benchmarks). | **P4** |
| **11** | `consolidated_320_conversation_demo.py:74-78` `grounded_phases`: `np.angle(proj @ code_vec)` (fixed complex projection cortex-code → composer phases) | **representational TRANSFORM** (map a learned cortex rate-code into the composer's phasor code) | A host matmul + `np.angle` projects the perception/cortex code into the binding space. Documented as "a fixed cortico-cortical fan-in, not learned per fact" — i.e. a fixed projection, the kind biology realizes as a fixed synaptic fan-in. **Borderline:** load-bearing-but-arguably-legit (a fixed projection ≈ a fixed weight matrix). Flagged, not strongly counted. | **CLEAN-CONVERSION** (or EXCLUSION) | A fixed complex synapse fan-in on the bridge (the step-3 grounding pattern already routes a live `cortex_it` rate vector through a fixed complex projection — `_step3_grounded_codes_production_composer_derisk.py`). If realized as a fixed bridge synapse it's a legit sensory/representational projection; as a host matmul it's a host transform. Low priority. | **P3** |

**Counting note.** Items #1+#2 are the same op on two composer substrates (one fix). #3+#4 are the two halves of the
same whole-turn loop conversion. #6+#7+#8+#9+#10 are the nav reward/value/perception/decision loop (the bulk of the open
research). #11 is borderline (a fixed projection). If you collapse the dup pairs, the distinct *mechanisms* to close are:
**4 conversational** (cleanup-WTA · scan-sequencer · S5-diagonal-gain · PPMI-neural-norm) + **5 nav** (orienting-SC ·
reward-SC · value-critic · place-selforg · decision-spiking_wta) + **1 deep frontier** (#12 below) + the borderline #11.

| # | shortcut | cognitive op | why a shortcut | bucket | mechanism / gate | priority |
|---|---|---|---|---|---|---|
| **12** | `rf_phasor_composer.py:156-185` `_bind`/`_bundle`/`_encode` + `one_brain_composer.py:229-249` `_compose_phases`: the exact-inverse FHRR bind/bundle/unbind ALGEBRA + the GIVEN/curated concept codes (`grounded_codes` / random per-seed) | **role-filler BINDING** (the compositional algebra the whole who/what memory rests on) | The bind OPERATIONS are on-substrate spiking (resonate-and-fire + complex synapses); the residual idealization is the **exact-inverse algebra** (a clean, exactly-invertible binding a real cortex would have to LEARN, lossy + redundant) + the **clean-code demand** (codes given/curated, not learned by the binder). The "composer-as-idealization" note. | **DEEP-FRONTIER (NEGATIVE-so-far)** | A LEARNED spiking-cortical binder replacing the exact-inverse algebra. Tested cheap-first: dendritic multiplicative binding MEMORIZES but does NOT generalize (held-out 0.168 < the fixed FHRR 0.261 — `2026-06-19-dendritic-binding-toy-derisk.md`), and apical-basal credit-assignment NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`). ⇒ both dendrite jobs ruled out; the path is more codes/capacity or the F=3 resonator (which DEGRADES on correlated learned codes, `2026-06-19-resonator-on-learned-codes-derisk.md`, ledger #20). Owner-deferred (months-scale; the artificial-life goal). | **P6 (defer)** |

---

## ALREADY-SPIKING-DEFAULT (listed for auditability; NOT counted as remaining shortcuts)

| component | file:line | why it is NOT a shortcut |
|---|---|---|
| **Nav action decision (LIBRARY default)** | `g11_bg_runner.py:3734` `readout_source="spiking_wta"`; :6968-6982 commit-burst | The LIBRARY `run_moving_goal_episode` default makes the action EMERGE from the spiking accumulate-then-commit competition; the host `max()` at :6981 is a tie-break-of-last-resort body-read of which commit pool bursted (validated 100% commit-burst, zero argmax fallback — `2026-06-19-spiking-decision-default-on-GO.md`). **CAVEAT:** the shipped *gate/benchmark* runner overrides this to host-argmax `"motor"` via the CLI default — that override is shortcut #10. The library mechanism itself is neural. |
| **Comprehension parser** | `brain_conversational_agent.py:123-143` `BridgeParser.role_of`; `one_brain_composer.py:159` | The (word-position × voice)→role assignment is a Hebbian-learned spiking read-out on the bridge; the host `max(rates, key=...)` at :137 is a body-read of which role ensemble FIRED (legit — reads a spiking result to route the word). The MultiCue/Case/Frame parsers (opt-in) are likewise spiking WTA competitions. |

---

## EXCLUSIONS — legitimate body/sensory/bookkeeping host ops found and deliberately NOT counted

These were inspected and ruled legitimate (so the list is auditable):

- **Parser/composer firing read-outs as body/route reads:** `brain_conversational_agent.py:135-137` (`max(rates,...)`),
  `one_brain_composer.py` `to_host(cp_membrane_potential_v)` reads — these read a SPIKING result to route a word /
  observe a winner = legitimate body-reads, not host computation of the cognitive quantity. (Where the *winner-pick
  itself* is a host argmax over a non-spiking membrane — #1/#2 — it IS counted.)
- **`run_sequencer`'s body-read of the won BG channel** (`_phaseC_task2_wholeturn_loop.py:151-156`): reading which BG
  channel the spiking WTA selected to pick the answer block = legitimate body-read (the DECISION is the spiking
  sequencer; only the S5 score→drive hand-off, #4, is the host op).
- **The reconsolidation prediction-error gate** (`rf_phasor_composer.py:379-404`, `one_brain_composer.py:570-595`):
  `1 - mean(cos(...))` + the auto-calibrated labilization midpoint. This is opt-in (not in the default who/what turn)
  AND is a phase-cosine self-comparison; it is a *follow-on* capability, flagged here but below the shipped core. If
  prioritized it would need a neural prediction-error/mismatch detector — noting it rather than counting it in the
  shipped-core total.
- **The multi-turn anaphora WM read** (`multi_turn_agent.py:97-104` `held_referent`): `np.mean` over WM attractor rates
  + a specificity threshold. The WM hold + the biased-competition WTA (`biased_competition_buffer`) ARE spiking
  (validated); the `held_referent` `np.mean`/threshold is a body-read of the WM attractor that won = legitimate
  (analogous to a readout of which attractor dominates). The biased-competition resolution itself is a spiking WTA.
- **Metrics / reporting / gates:** `consolidated_320_conversation_demo.py:119-120` (`np.mean`/`np.max` over grounded
  similarities), `:130-160` (recall/abstain counting), all `_phaseC_task2_wholeturn_loop.py` GO-gate aggregation — these
  are experiment scoring, outside the sensation→action path = legitimate.
- **CSR/plasticity-gain bookkeeping** (`nav_conv_merged_bridge.py:375-443,1496-1497,1640-1662` `to_host(indptr/indices)`,
  `_train_merged_convergence`): host index arithmetic to build/freeze synapse masks = wiring/bookkeeping, not cognition.
- **`np.argmax(scores)` zero-peak fallbacks** (`rf_phasor_composer.py:274,289`): only reached when the spiking WTA
  emitted zero spikes (a degenerate safety fallback); the primary path is the spiking firing-argmax. Noted as a fallback,
  not the primary selection.
- **The grounding projection #11** is listed in the table as borderline (it MAY be a legit fixed sensory/representational
  projection rather than a cognitive normalization); included in the table at low priority so the owner can rule.

---

## RECOMMENDED CLOSURE ORDER (cheap clean-conversions first → gate the open-research → deep frontier last)

1. **#1 + #2 — composer cleanup WTA (P1, cheapest, biggest brain-based win).** Wire `_spiking_cleanup` (already in
   `rf_phasor_composer.py`, validated == numpy @ D=2048) into `OneBrainComposer._read_block`/`_decode_batched_mem`, and
   default it on. This converts the production who/what cleanup SELECTION from host argmax to a spiking WTA — the single
   most-impactful + lowest-risk conversion (the mechanism is shipped, just not on the one-brain path). Re-run the 320 GO
   gate to confirm == host + 0 moat breaches.
2. **#3 + #4 — scan→spiking-sequencer + S5 diagonal-gain (P2).** First run the S5 falsification de-risk (peak-sweep +
   moat) for the diagonal-gain normalizer (`input_divisive_norm` already in `sim/`), then scale the Phase-B sequencer
   past K=2 and wire it into OneBrain's query path, removing the host `_scan` for/if/return. This converts the cue-match
   COMPARISON + routing to spiking. (The K=2 whole-turn loop `_phaseC_task2_wholeturn_loop.py` is the proof of concept.)
3. **#5 — PPMI neural read-out norm (P3).** Produce the 320-scale `--readout-norm neural` codes for seeds 43/44 and
   default the production demo to them. (Offline learning-pipeline conversion; the neural circuit is de-risked at 96% of
   host.) Plus the borderline #11 projection ruling.
4. **#10 — flip the nav gate/demo CLI default to `spiking_wta` + urgency 180 (P4, cheap).** The spiking decision is the
   validated library default; the gate runner's `"motor"` CLI default is only the historical-benchmark oracle. Flipping
   the deployed default (keeping `"motor"` as an explicit `--readout-source motor` oracle) retires the host-argmax
   decision in the shipped gate.
5. **#6 + #7 + #8 + #9 — the nav reward/value/perception loop (P4-P5, OPEN-RESEARCH).** This is the big one and needs the
   **deep-research-first gate as a unit** (the standing directive): the SC-orient + neural-reward + critic + SNc closed
   loop is NO-GO (~58× worse, actor goes silent — ledger #13). The organs each work in isolation; the loop does not
   sustain. Do NOT flip these defaults until the loop is closed — an honest NEGATIVE here is the deliverable (it maps
   what the point-neuron substrate can/can't do for sustained reward-driven control). #9 (place) has a dendritic flavor
   but the immediate blocker is a readout-regime issue.
6. **#12 — the composer's exact-inverse FHRR algebra (P6, DEFER).** The deep frontier. Both dendrite jobs tested
   cheap-first and ruled out; the path is more codes/capacity or a different representation (F=3 resonator), owner-
   deferred to the artificial-life goal. Closing this will most likely be an honest-negative-rich research arc, not a
   clean conversion — the trade-off (the exact algebra buys the no-confab moat + compositional reliability ~free) is
   explicit.

---

## HONESTY FLAGS

- **#10 is load-bearing-but-legit-by-library-default, host-by-deployed-default.** The LIBRARY decision is spiking
  (counted as ALREADY-SPIKING-DEFAULT); the shipped gate/benchmark CLI overrides it to host-argmax. Both are true; the
  shortcut is the *deployed* override, and it is a cheap flip.
- **#6-#9 "closing" will likely be honest NEGATIVES, not conversions.** The nav reward/value/place loop is the foremost
  OPEN boundary; the deployed closed loop is documented NO-GO. Per BRAIN-BASED-ONLY, the neural version underperforming
  the host shortcut IS the scientific deliverable. Do not expect a clean win here; expect a map of the substrate limit.
- **#11 (the grounding projection) may not be a shortcut at all** — a fixed complex fan-in is the kind of thing biology
  realizes as a fixed synapse. Flagged for the owner to rule; low stakes either way.
- **#5 is an OFFLINE learning-pipeline op, not a runtime conversational shortcut.** The production agent loads pre-computed
  codes; `double_center` runs once during cortex code generation. Counted because it is a cognitive normalization the
  brain should do, and the neural circuit is validated — but it does not sit in the live who/what turn.
- **The composer bind/store/unbind OPERATIONS are already spiking** (resonate-and-fire + complex synapses). The DEEP
  frontier (#12) is narrowly the exact-inverse *algebra* + the *given codes*, NOT the spiking realization of the ops —
  the project's "composer-as-idealization" framing is accurate and is preserved here.

---

## Sources (file:line verified, cross-refs read)

- Shipped conversational: `research/runners/one_brain_composer.py` (lines 229-249, 271-313, 417-446, 489, 504-553,
  570-631), `rf_phasor_composer.py` (62-185, 242-301), `brain_conversational_agent.py` (123-143, 337-475),
  `multi_turn_agent.py` (94-178), `_phaseC_task2_wholeturn_loop.py` (84-167), `consolidated_320_conversation_demo.py`
  (68-79, 105-201).
- Shipped merged nav: `research/runners/nav_conv_merged_bridge.py` (447-718 builder, 1300-1419
  `MergedNavConvAgent`/`MergedRFComposer`), `_nav_gate_merged_run.py` (29-150 the deployed gate CLI),
  `g11_bg_runner.py` (2937-2944 orienting, 3262/3277/3344/3540/3542/3613/3734 signature defaults, 6450-6633 perception,
  6968-6988 readout, 7056-7061/7135-7197/7260-7289 reward/value).
- Documented-shortcut cross-refs: `research/findings/2026-06-20-boundary-ledger-dendritic-audit.md` (commit `79a9fb1d`),
  `2026-06-19-dendritic-binding-toy-derisk.md`, `2026-06-19-S5-on-bridge-normalization-deep-research.md`,
  `2026-06-19-spiking-decision-default-on-GO.md`, `2026-06-05-composer-cleanup-NEF-GO.md`,
  `2026-06-16-biologization-sweep-conversational-pipeline.md`, `2026-06-19-nav-spiking-sc-deploy-NO-GO.md`,
  `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`, `2026-06-19-place-code-sparsify-default-BOUNDARY.md`,
  CLAUDE.md (the documented capability arc + "Standing standard: BRAIN-BASED ONLY").

_Read-only audit deliverable. No code, no experiments. Every cited file:line verified against the source; defaults
quoted verbatim from the function signatures. The split between LIBRARY defaults and deployed-CLI defaults (nav decision)
is called out explicitly because it is the crux of what actually ships._
