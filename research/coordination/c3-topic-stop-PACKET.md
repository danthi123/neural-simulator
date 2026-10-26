# PACKET c3-topic-stop (Claude, 2026-10-26): the per-topic stop in the production ask path, and a topic-locked DA

host-decision: scaffold
decides: whether the curiosity follow-up question is asked, through the crave-gain formula clip(0.2, 3.0, 1 + 1.5 (da_eff - 0.5)) applied to the topic-stop ASK-pool want (P method B), the rate-to-da_eff normalization and the two linear afferent encodings; and, for L, which topic slice a topic takes when all 11 slices are in use (least recently presented)
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: a per-pathway dopamine synaptic-gain target on the topic -> ASK synapses (a sim/neuromodulators.py scope extension), so the SNc dopamine scales the topic afferents on the substrate and no host formula scales the want
cap: 2 builds

Board: Stage-2 C3 for the 11-21 retake (RULING stage2-fixes-1026, C3 item). Branch `research/c3-topic-stop` (worktree
`sim-rev/wt-c3int`, cut from main 3065c5614). Default OFF everywhere; no default is flipped. No `sim/` edit. Gate seeds
42/43/44/100/101/102 were not run; the smoke used dev seed 7, the design rule seed 9. Seed 9 is the DESIGN seed
(method 2's W_TOPIC_ASK, and this packet's G_PHASIC_PA and neutral SNc rate, were set there), so it is not a counted
dev seed anywhere below (section 7).

## 1. The defect (research/findings/2026-10-26-stage2-fail-diagnoses.md)

- Layer 1: every ask is `want_hz * clip(0.2, 3.0, 1 + 1.5 (DA - 0.5)) >= threshold` in host code
  (`webapp/da_curiosity_drives_chat.py`, applied in `webapp/server.py::_curiosity_followup`). DA is a session-global
  host EMA that the habituating carrier word drives to its floor (about 0.046) within 4-6 probes, so nothing is asked.
- Layer 2: no per-topic stop in the handler.
- With the gain pinned (`BRAIN_CURIOSITY_DA_LESION=1`), dev 7 reversed: every novel topic asked, no taught moat asked,
  lp_ibex_1 asked and lp_ibex_2 not, toad asked on all 4 probes; gate da_lesion arm 6/6: gap 1.0, learnable True,
  unlearnable-stops False. The per-topic stop is the only C3 failure left once the gain is gone.

## 2. Prior art read before building

- `bash tools/before_you_build.sh "curiosity per-topic stop in production handler"`: the Gate-B curiosity wiring
  (2026-08-12), the GNW stop wire-in (2026-08-26), the rank-10 graded novelty (2026-09-05). None holds a per-topic stop.
- Method 2 (`research/runners/_curiosity_lp_trace_derisk.py --no-ou`, s/research/c3-method2-noou 2277ae294, dev read
  `research/coordination/c3-method2-devread-1018.md`): per-topic depressing slices onto an ASK pool, OU off; every bar
  8/8 on dev seeds 1-8, provisional; `asks_before_stop` 1-2. This is the mechanism L reuses (cherry-picked as cf70ed78d).
- Method 3 (`_curiosity_lp_slope_gate_derisk.py`, research/c3-method3-slopegate e559d4ec9): a host fast/slow LP slope
  plus a host omission-veto accumulator on the same bridge. More host state than method 2 and no dev read yet; not used.
- `docs/drafts/curiosity-lp/PLAN.md`: method 1's bars B1-B9 and the integration note (section 8) this packet follows.
- The spiking omission circuit (finding 2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed:
  reward_us -> rmtg -| omit -> veto, plastic cue -> veto). It is the existing spiking veto that separates a topic that
  yields reward from one that yields none. It is the rung beyond L (section 3, label), not built here.

## 3. L: `BRAIN_CURIOSITY_TOPIC_STOP` (lesion `BRAIN_CURIOSITY_TOPIC_STOP_LESION`)

Code: `research/runners/curiosity_topic_stop_organ.py::TopicStopOrgan` (subclass of method 2's `CuriosityLPOrgan`),
`webapp/curiosity_topic_stop_chat.py`, two flag-gated blocks in `webapp/server.py`.

- **Mechanism (reused, frozen).** On an abstain with topic T, T's slice of `topic_aff` (25 RS) gets one burst at
  `novelty * WORD_DRIVE` (novelty = the production graded `topic_novelty` read already computed that turn). The slice
  projects onto one 80-RS ASK pool through Tsodyks-Markram synapses (U 0.35, tau_f 10 ms, tau_d 12 s). The want is the
  ASK rate over the 28-step window; `want >= threshold` (the organ's calibrated midpoint) decides. The neuron state is
  washed before and after each read; the slice synapses are NOT restored: that is the trace. W_TOPIC_ASK 80 and every
  other constant are method 2's.
- **Where the state lives.** On THIS conversation's `Context` (`webapp/one_brain.py`), attribute `curiosity_topic_stop`:
  the organ (450 neurons, built lazily on the first abstain, about 0.6 s) and the turn counter. One organ per
  conversation, so the slice depression persists across turns and never crosses conversations; `ContextRegistry.remove`
  (reset or eviction) drops it. The production shared-pool read restores a clean state per read, so this trace cannot
  live there; this follows the per-conversation organ pattern (`_SESSION_MULTIREF`, `_SESSION_PMEM`, `_SESSION_SELFINIT`).
- **Turn clock.** `brain_reply` counts each turn while L is on; at a read the organ runs the owed 400 ms silent idles
  (one per turn, method 2's battery protocol). Compressed clock (method 2's declaration): ~2 s between turns against a
  ~1 min recovery.
- **Topic -> slice.** First-seen allocation of the token `extract_topic` already returns (a fixed input code, no word
  list). Capacity: 11 slices per conversation; a 12th topic takes the least recently presented slice, reset to rest.
  Declared host bookkeeping; never reached in the Stage-2 battery (8 topics).
- **Background noise.** OFF for the whole organ, as validated by method 2 (the topic-afferent-only variant was never
  run). DECLARED PER-MECHANISM REALISM TRADE: the ASK pool loses its background synaptic noise too. Reason: the OU
  process at 100 pA std fired 167 topic-afferent spikes per inter-turn gap and depleted every slice by elapsed time
  (method 1 NO-GO), which made the trace a clock. The decision margin given up is large (~35 Hz want against a ~18 Hz
  threshold on a fresh slice). The afferent-only variant is a battery arm request below (section 7, informational).
- **Label.** COUNT-BASED, BY TOPIC HISTORY, NOT LEARNING PROGRESS. The drive to ask falls with the number of times a
  topic was presented; a topic being learned is silenced the same way. Method 2's DA-gated dishabituation on a teach
  turn is NOT integrated (it is reply-silent in this battery and is a host write; its rung is a sim/ DA -> STP-recovery
  target). The learning-progress-sensitive rung is the spiking omission circuit above.
- **Composition with the existing gain.** With L on and P off, the default-ON host gain still applies, now to the
  organ's own want and threshold (`crave_decision(chat, ts.want_hz, ts.threshold)`). The shared-pool read still runs,
  is reported (`want_hz_shared`) and still feeds every other consumer of the shared salience afferent.
- **Lesions.** lesion_L: tau_d 200 ms (method 2's lesion_habituation), read back as `held.stp_tau_d_ms`.
  `BRAIN_CURIOSITY_LESION=1` also cuts the organ's topic -> ASK weights (method 2's lesion_drive), read back as
  `held.topic_ask_weight_sum` = 0.

## 4. P: `BRAIN_CURIOSITY_PHASIC_DA` (lesion `BRAIN_CURIOSITY_PHASIC_DA_LESION`; requires L)

Registered before any data (commit d25e25002): the tonic term K_TONIC_PA = 440 pA per DA unit, intercept 0 (session
DA 0.5 -> 220 pA, the SNc's neutral drive in method 2); the phasic term G_PHASIC_PA * salience(T), salience(T) =
`salience_of`'s normalization applied to the shared-pool `judge()` want (no extra read); and the design rule in
`design()` on seed 9.

- **Method A, the bus form (tried first, rejected by its own registered rule).** A dopamine modulator on the engine bus
  (`from_region_firing_signed` on `snc`, concentration +-1 around baseline, `excitability_drive` on group:ask, S_DA pA
  per unit). Rule: S_DA = the smallest grid value for which (a) at the floor tonic a fresh novel slice falls below
  threshold and (b) DA alone at the ceiling never fires the ASK pool. Amendment D1 (before any handler run): run 1 was
  void (the rule sensitivity missed the engine's dt/1000 factor, so the concentration never left baseline) and the
  neutral SNc rate is now read in the P window; the grid was extended to 200 pA. Result
  (`research/findings/raw/_c3_topic_stop/design_s9.json`, produced at d25e25002 with an uncommitted tree: its sidecar
  records git_dirty true, and the working-tree edits were committed afterwards in 86d8b461c): no value meets both halves. 60 pA leaves the floor want at
  21.4 Hz (threshold 17.9); 80 pA brings it to 14.7 Hz, but the drive alone then fires the ASK pool at 15.6 Hz with no
  topic input; 100 pA and above fire it past threshold. An additive excitability drive is not a gain. Verdict on the
  METHOD; its rung is the `replaced-by` above (a per-pathway synaptic-gain target needs a sim/ edit).
- **Method B, built (the handler path).** On an abstain with topic T the organ reads its own SNc (the method-2
  `reward_us -> snc` populations) with the tonic and phasic afferents on: 60 warm-up steps, then the rate over the next
  28 steps. `da_eff = 0.5 * snc_rate / snc_neutral_rate` (neutral = the same window at session DA 0.5, no phasic).
  `da_eff` enters the EXISTING gain formula (`da_crave_gain`, unchanged), applied to L's want; `want * gain >= threshold`
  decides, and the handler's own `crave_decision` is skipped. G_PHASIC_PA = 600 pA from the rule's phasic half (at the
  floor tonic, salience 1 brings the SNc to 47.6 Hz against its 40.5 Hz neutral rate on seed 9).
- **What P changes, honestly.** The DA entering the gain is a topic-locked spiking read in which T's own salience is an
  afferent, instead of the session-global EMA alone. The gain formula, its clip, the rate normalization and the two
  linear encodings remain host arithmetic (the header above). The tonic afferent still carries the host EMA
  (`webapp/da_mode_drives_chat.py`).
- **Lesion P.** The phasic afferent pinned to 0, the tonic intact; read back as `phasic_da.reward_us_spikes` = 0.
- **`BRAIN_CURIOSITY_DA_LESION` is not applicable when P is on.** The P branch skips `crave_decision`
  (`webapp/server.py`, the `pass  # P:` branch), which is the only reader of that pin, so no `curiosity_da` trace is
  attached and the gate's informational `da_lesion` arm is UNDEFINED under P (its `lesion_held` cannot be true). Under L
  alone it applies and is used (S1-S3 and the `L only, gain pinned` battery arm). The test of P that does not rely on
  it is lesion_P, plus the `p_const_salience` arm (section 7), which separates topic-locking from the phasic drive.
- **Organ-level sanity read (dev seed 7, synthetic DA sequence, after the method-B build, before these predictions and
  before any P handler run):** at the floor tonic a novel topic reads da_eff 0.63, gain 1.19; under lesion_P da_eff 0,
  gain 0.25 (the clip), want_eff 8.9 Hz < 17.6; toad's 2nd presentation reads 15.2 Hz x 1.19 = 18.1 Hz (asked), the 3rd
  0 Hz.

## 5. Engagement fields (per curiosity probe, under `curiosity.curiosity_topic_stop`)

`want_hz` (the topic-stop ASK read), `want_decision_hz` (after P's gain), `threshold`, `want_hz_shared`,
`threshold_shared`; `tonic_da` {da_level, i_tonic_pa}; `phasic_da` {salience, i_phasic_pa, snc_hz, snc_neutral_hz,
da_eff, da_crave_gain, reward_us_spikes, lesioned}; `slice_pre` / `slice_post` {weight_sum, x_mean, efficacy_sum};
`stop` {presentations, x_pre, x_post, stopped}; `held` {stp_tau_d_ms, ou_enabled, topic_ask_weight_sum}; `idles_run`;
`turns_seen`. With the `p_const_salience` switch on, `phasic_da.salience` is the constant used, `salience_in` the topic's
own and `const_salience` true (keys present only then). With L on and P off, `curiosity.curiosity_da` (the host gain trace) is kept, so the gate's
`da_lesion_held` reads the pinned gain.

## 6. Registered predictions for the dev-seed-7 smoke

Ordering disclosure: S1 (the first smoke arm) was launched at 14:21:47, the predictions were committed at 14:24:21
(f27f1efb4), and the first smoke record was written at 14:35. S1 was therefore already running when the predictions were
committed, and no handler record existed at the commit.

Each arm is one `stage2_exit_gate --shard --seed 7 --group ask` run at 86d8b461c, flags in the environment.
Scores: `score_ask` (`discriminates`, `lp_learnable`, `lp_unlearnable`, the gap and moat rates).
- **S1 L, gain pinned, intact_a.** Asked: g_eel, g_crab, g_newt, lp_ibex_1, u_toad_1. Not asked: m_yak, m_moose,
  m_goat, lp_ibex_2, u_toad_4. u_toad_2 not asked (organ dev 7: 13.4 Hz). All C3 clauses true.
- **S2 L, gain pinned, reversed.** The same asked pattern by label; all C3 clauses true.
- **S3 flag-off, gain pinned, intact_a (control).** Gap 1.0, learnable true, toad asked on all 4 probes, so
  lp_unlearnable false (the gate's da_lesion arm).
- **S4 L+P, intact_a.** All C3 clauses true; u_toad_2 may be asked (organ read 18.1 vs 17.6), u_toad_3/4 not.
- **S5 L+P, reversed.** All C3 clauses true; the toad block runs while the session DA is still high (about 0.55), so
  u_toad_2 is likelier to be asked than in S4; u_toad_4 not.
- **S6 L+P, lesion_P, intact_a.** Reproduces today's failure: g_eel asked; lp_ibex_1 and u_toad_1 not asked, so
  lp_learnable and lp_unlearnable false; gap rate at most 2/3 (g_newt not asked).
- **Answer rate.** `abstained` and `recalled_svo` identical to the flag-off run of the same order on every turn
  (S1, S4, S6 vs S3; S2, S5 vs the diagnosis run `diag_1026/runs/s7_ask__reversed_dalesion.json`, same code base).
- **Read-backs.** stp_tau_d 12000 ms and ou_enabled false on every L probe; reward_us_spikes > 0 on every novel P probe
  and = 0 on every S6 probe.

## 7. Dev-seed battery for the pool (`stage2_exit_gate --shard ... --group ask`)

Counted dev seeds: 1-8 and 10-12 (11 seeds). Dev seed 9 is the design seed (header): it may be run, and is reported
separately, but it is never counted in any bar below, so the bars are written over 11 seeds ("11/11").

Candidate configuration C = `BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_PHASIC_DA=1`. Arms per seed (one fresh
process each):

| arm | runner arm | extra env | counted |
|---|---|---|---|
| C intact_a | intact_a | C | yes |
| C intact_b (null) | intact_b | C | yes |
| C reversed | reversed | C | yes |
| C permuted | permuted | C | yes |
| lesion_L | intact_a, reversed | C + `BRAIN_CURIOSITY_TOPIC_STOP_LESION=1` | yes |
| lesion_P | intact_a, reversed | C + `BRAIN_CURIOSITY_PHASIC_DA_LESION=1` | yes |
| lesion_curiosity | lesion_curiosity | C | yes |
| flag-off | intact_a, reversed | none (production defaults) | baseline |
| flag-off permuted | permuted | none (production defaults) | baseline (D7's comparator for C permuted) |
| L only, gain pinned | intact_a, reversed | `BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_DA_LESION=1` | informational |
| p_const_salience | intact_a, reversed | C + `BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE=1` | informational |
| null_count | computed, no run | none | informational |
| afferent-only noise | needs a build (section 3) | none | informational, later |

`p_const_salience` keeps the tonic SNc transfer and replaces the topic's salience by a constant, the running mean of the
saliences this conversation has presented (a default-OFF switch in `curiosity_topic_stop_organ.py`, unit-tested; built
after the smoke, so S1-S6 did not use it). It tests topic-locking apart from lesion_P: lesion_P removes the phasic drive,
this arm keeps the drive and removes only its dependence on the topic. It is reported against C and lesion_P, with no bar
and no registered prediction. `BRAIN_CURIOSITY_DA_LESION` cannot serve here (section 4: not applicable under P).

`null_count` (computed from C intact_a's trace): ask iff the shared-pool read is curious and the topic has been
presented fewer than k times before (k = 1, 2). It reports how much of C's asked vector a host count alone reproduces.
L is count-based by design, so high agreement is expected and is reported, not hidden. The gain-pinned flag-off
baseline in both orders is already in the pool at the Stage-2 pin (research/findings/raw/_stage2_diag/c3_gainpinned/).

Bars (dev read; every bar must hold for the method to go to a gate-seed preregistration):
- **D1** C intact_a and C reversed: every C3 clause of `score_ask` holds, 11/11 counted seeds each.
- **D2** C intact_b: the per-label asked vector equals C intact_a, 11/11.
- **D3** C permuted: `score_ask(permuted=True)` discriminates (the asks follow the exposure), 11/11.
- **D4** lesion_L: u_toad_4 asked, and different from C on u_toad_4, on 11/11 in both orders; attributable fraction
  of the extra toad asks against the intact_b null >= 0.5 (`tools.lab.attributable_to`).
- **D5** lesion_P intact_a: lp_learnable false and lp_unlearnable false on >= 9/11 (the phasic afferent is what keeps
  novel topics asked at the DA floor).
- **D6** lesion_curiosity: 0 asks on every turn, 11/11.
- **D7** answer rate: `abstained` and `recalled_svo` identical to the same-order flag-off arm on every turn, 11/11, for
  every counted arm that has such a comparator: C intact_a, C intact_b, lesion_L and lesion_P intact_a against flag-off
  intact_a; C reversed, lesion_L and lesion_P reversed against flag-off reversed; C permuted against flag-off permuted.
  lesion_curiosity has no same-order flag-off arm and is outside D7 (D6 covers its asks).
- **D8** read-backs: every lesion holds on every probe of its arm (stp_tau_d 200 ms; reward_us_spikes 0; weight sum 0).
Reported, not barred: asks_before_stop per seed; null_count agreement; the flag-off failure in both orders.

## 8. Smoke result (dev seed 7)

Runs: one fresh process per arm, numpy CPU, `SIM_NO_PROVENANCE=1`, branch code at 86d8b461c; records on the
workstation under `sim-rev/c3int_scratch/<config>/s7/ask__<arm>.json` (diagnostic, not committed), scored by
`sim-rev/c3int_scratch/analyze.py` with the runner's own `score_ask`. The runner's `--aggregate` does not accept a
partial dev directory (it reports the seed UNDEFINED, "intact arm missing", until every arm of every group is present),
so the clauses below are `score_ask` on each shard. m_goat is UNDEFINED in every arm: the default-ON D4 comprehension
gate refused `the goat eats the hay` (margin 0.0167 against 0.2, the known C2 defect), so goat stays a novel topic and
the scorer drops it from the moat rate.

Asked by label (g_eel m_yak g_crab m_moose g_newt m_goat lp_ibex_1 lp_ibex_2 | u_toad_1..4) and the C3 clauses:

| arm | asked | toads | gap | moat | discr. | learnable | unlearnable stops | abstained/recalled_svo vs flag-off |
|---|---|---|---|---|---|---|---|---|
| S1 L, gain pinned, intact_a | 1 0 1 0 1 1u 1 0 | 1 0 0 0 | 1.0 | 0.0 | T | T | T | 0 of 19 turns differ (S3) |
| S2 L, gain pinned, reversed | 1 0 1 0 1 1u 1 0 | 1 0 0 0 | 1.0 | 0.0 | T | T | T | 0 of 19 (S3r) |
| S3 flag-off, gain pinned, intact_a | 1 0 1 0 1 1u 1 0 | 1 1 1 1 | 1.0 | 0.0 | T | T | F | reference |
| S3r flag-off, gain pinned, reversed | 1 0 1 0 1 1u 1 0 | 1 1 1 1 | 1.0 | 0.0 | T | T | F | reference |
| S4 L+P, intact_a | 1 0 1 0 1 1u 1 0 | 1 0 0 0 | 1.0 | 0.0 | T | T | T | 0 of 19 (S3) |
| S5 L+P, reversed | 1 0 1 0 1 1u 1 0 | 1 1 0 0 | 1.0 | 0.0 | T | T | T | 0 of 19 (S3r) |
| S6 L+P, lesion_P, intact_a | 1 0 0 0 0 0u 0 0 | 0 0 0 0 | 0.33 | 0.0 | F | F | F | 0 of 19 (S3) |

(1u / 0u: m_goat, undefined.) `all_gaps_abstain` and `builtin_dog_cat` are true in every arm.

Engagement reads:
- **L.** The topic-stop want on a fresh slice is 35.3-35.7 Hz against its threshold 17.63 Hz (the shared pool reads
  38.2-38.9 Hz against 19.10 Hz on the same probes). Toad, intact_a order: x_pre 1.000, 0.673, 0.475, 0.354; want
  35.3, 12.9, 0, 0 Hz. Reversed (S2): x_pre 1.000, 0.663, 0.464, 0.346; want 35.7, 15.6, 0, 0 Hz (16.07 Hz is S5's u_toad_2, below). A familiar moat drives
  0 afferent spikes and its slice stays at x = 1. Read back on every L probe: stp_tau_d 12000 ms, ou_enabled false.
  On S1/S2 every `curiosity_da` trace is lesioned with gain 1.0 (`da_lesion_held`: held).
- **P (S4).** Session DA 0.554 / 0.262 / 0.055 / 0.046 with salience 1.0 gives SNc 71.4 / 56.0 / 44.0 / 46.4 Hz,
  da_eff 0.97 / 0.76 / 0.60 / 0.63, gain 1.70 / 1.39 / 1.15 / 1.19. Moats (salience 0) read da_eff 0.42 at DA 0.407 and
  0 at DA 0.061 (gain 0.88, 0.25). reward_us fires 337 spikes on every novel probe and 0 on the moats. u_toad_2:
  12.9 Hz x 1.19 = 15.5 Hz, not asked. No `curiosity_da` trace is attached (the host `crave_decision` is skipped).
- **S5.** The toad block runs first, at session DA 0.554 then 0.246: gains 1.70 and 1.36. u_toad_2: 16.1 Hz x 1.36 =
  21.9 Hz, asked; u_toad_3 and u_toad_4 read 0 Hz. asks_before_stop 2 (S1, S2, S4: 1).
- **S6.** reward_us fires 0 spikes on every probe (the lesion held). g_crab (DA 0.262): gain 0.44, 15.8 Hz, not asked.
  From g_newt on: da_eff 0, gain 0.25 (the clip), 8.8-8.9 Hz, nothing asked.
- **null_count (informational, from S1 and S4).** k = 1 agrees with the intact asked vector on 11 of 11 abstain probes;
  k = 2 on 10 of 11. On this seed L's decision equals a one-ask count rule, as its label says.

Predictions (section 6) against the results:
- S1 held except one listed element: m_goat was predicted unasked as a taught moat, but its teach was refused, so it is
  a novel topic and was asked; the scorer marks it undefined. The prediction should have conditioned on acquisition.
- S2, S3, S4, S5 (u_toad_2 asked, as predicted likelier) and S6 (g_eel asked; lp_ibex_1 and u_toad_1 not; gap 1/3)
  held. The answer-rate prediction held: 0 differing turns in 5 comparisons (95 turns). The read-backs held.
- The reversed flag-off control named in section 6 (the diagnosis run) was replaced by S3r: that run recorded
  `corpus_present` false and its comprehension organ raised FileNotFoundError on 4 turns
  (`sim-rev/wt-s2pin/data/corpus/tinystories.txt`), which is why it acquired t_goat. It is not a valid control. Its toad
  claim (asked 4 of 4 with the gain pinned) is unaffected: S3r reproduces it with the corpus present.

Flag-off byte identity:
- **Whole record, branch vs main.** S3 (branch, flags unset, gain pinned, intact_a) against the same arm run from a
  detached main worktree at 3065c5614 (`sim-rev/wt-c3int-main`): every record field except wall-clock and peak RSS is
  equal, and the 19 responses serialize to the same 81,508 bytes.
- **One handler turn, opt-in test.** `C3TS_HANDLER_TEST=1 pytest tests/test_curiosity_topic_stop.py -k handler_turn`:
  1 passed (142 s). With every flag unset an abstain turn carries no `curiosity_topic_stop` key, no Context attribute
  is created, and `decide` / `note_turn` (patched to raise) are never reached.

What the smoke does not show: anything beyond dev seed 7 (one seed; the battery in section 7 is the read), the gain
left at its production default with L alone, or any gate seed.

## 9. Residuals carried forward

- Two ASK pools: the shared-pool ASK slice (from_novelty drive, read and reported) and the conversation's topic-stop
  ASK pool (topic slices, decides when L is on). The integration rung is the slices on the shared pool as afferents
  of its ASK slice (PLAN.md section 8), which needs a per-conversation STP swap around the shared read.
- P method B keeps the host gain formula (header). P method A's rung is a per-pathway synaptic-gain target (sim/).
- The tonic afferent still carries the session-global host EMA (`da_mode_drives_chat.py`).
- L is count-based; the learning-progress-sensitive rung is the spiking omission veto (section 2).
- One-brain residual: the per-topic habituation state (the topic slices' synaptic depression) lives on the
  per-conversation `Context` today. In `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md`'s terms this is
  "B target, C today": one brain would hold one habituation substrate across interlocutors. A topic asked about in one
  conversation is still fresh in another.
