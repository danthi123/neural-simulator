# curiosity-lp: a per-topic, synaptic learning-progress brake on asking (Stage-2 C3 gap), section-5 de-risk

*Claude, 2026-10-13, branch `claude-build/curiosity-lp-derisk` (cut from s/main bef736ce7). Dev seeds only (battery
1-8, design smoke 9). Default-off by construction: a runner-only module, never imported by `webapp/` or by the
production organ; `research/runners/curiosity_production_organ.py` is not edited; no `sim/` edit. Bars, arms, the
design-calibration rule and the verdict rule below were written, and committed, BEFORE any bridge was built.*

## 1. The criterion and the gap

Stage-2 C3 (`s/research/stage2-exit-gate:research/runners/stage2_exit_gate.py`, group `ask`): the follow-up question
fires on novel-topic gaps at >= 2x the taught-topic moat rate (and >= 2/3 itself); a learnable gap is asked, answered,
then recalled and not asked; an unlearnable gap is asked at first and NOT asked by its 4th probe.

Dev seed 7 (`research/findings/raw/_stage2_exit/dev/s7/ask__intact_a.json`): g_eel and g_crab are asked; every later
probe (g_newt, m_goat, lp_ibex_1, u_toad_1..4) is not, although each novel topic still drives the spiking ASK pool to
38-39 Hz against a 19.1 Hz threshold. The brake is the GLOBAL self-produced DA level read by
`webapp/da_curiosity_drives_chat.py` (`BRAIN_CURIOSITY_DA`, default ON): on the abstain turns DA reads 0.55 (g_eel),
0.41 (m_yak), 0.26 (g_crab), 0.06 (m_moose), then 0.05; the crave gain 1.08, 0.86, 0.64, 0.34, 0.33; from g_newt on
no topic can cross. Nothing in the curiosity path
knows WHICH topic went unanswered, so a fresh topic is stopped as hard as one asked three times. C3's `lp_learnable`
and `lp_unlearnable` clauses then fail for the wrong reason (u_toad_1 is never asked at all).

The named gap: a per-topic learning-progress (LP) signal, so that asking stops for the topic that yields nothing and
continues for one that has not been tried.

## 2. What the record already had (read before building)

`bash tools/before_you_build.sh "curiosity learning progress per-topic"` and
`rag_search.py "curiosity learning progress ask rate DA threshold" 5 --corpus all` (both run 2026-10-13). Read:

- `2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-...` and `2026-08-07-laneB-curiosity-
  learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md`: LP as a phasic-minus-tonic EMA slope is a CPU proxy
  (host numpy EMA); its on-bridge substrate-memory promotion was 1/6 (seed-fragile). Not reused as the trace.
- `2026-09-03-on-bridge-spiking-LP-max-curiosity-selection-layer-6seed-GO.md`: a neural max over options driven by a
  SYNTHETIC LP vector; the LP estimate itself stayed host. Different question (selection among options, not whether
  to ask about the one topic the user raised).
- `2026-08-21-da-gated-curiosity-threshold-wired-GO.md` and `webapp/da_curiosity_drives_chat.py`: the global DA crave
  gain (the brake in section 1).
- `research/biology/spiking-habituation-novelty.md`, `research/runners/spiking_novelty_habituation_organ.py` (6/6 GO,
  production since 2026-09-16 for message novelty): per-word Tsodyks-Markram depression (`cp_stp_x`) as a recoverable
  habituation read, U = 0.35, tau_f = 10 ms, an 8-step burst + 20-step read per presentation. REUSED here (the STP
  regime, INPOP, the presentation protocol), but on a different synapse: the novelty -> ASK drive, not the novelty read.
- `research/runners/curiosity_production_organ.py`: REUSED by subclassing (the calibration rule, `TopicNoveltyGate`,
  `extract_topic`, `followup_question`, the OU wash-out); the ASK pool keeps the organ's settings (80 RS,
  homeostasis-free, per-neuron OU seed, name-keyed per-region heterogeneity).
- `research/runners/_curiosity_seek_learn_onbridge_derisk.py`: REUSED for the reward_us -> snc DA read (sizes, weight,
  SNC_TONIC_PA, US_GAIN_PA, SNC_REWARD_SCALE, W_WARMUP/W_MEASURE) and the bridge helpers.

## 3. Biology relied on

- Habituation is homosynaptic depression in the activated pathway, stimulus specific, lasting many minutes (Kandel
  PNS 6e ch. 53, Fig. 53-2, `sim-catalog/.../kandel-pns-6e/full-book.txt` ~73660-73730; Castellucci & Kandel 1974 PNAS
  71:5004; Thompson & Spencer 1966 Psychol Rev 73:16; Rankin et al. 2009 Neurobiol Learn Mem 92:135).
- Dishabituation: a modulatory transmitter acting on the depressed terminal restores release (Hochner, Klein, Schacher
  & Kandel 1986 PNAS 83:8794; Kandel ch. 53 ~73782). Here DA plays the modulator, gated by the answer's LP.
- Depressing neocortical synapses (Tsodyks & Markram 1997 PNAS 94:719; Markram, Wang & Tsodyks 1998 PNAS 95:5323).
- Learning progress as the intrinsic reward that should steer information seeking (Oudeyer, Kaplan & Hafner 2007 IEEE
  TEC 11:265; Gottlieb, Oudeyer, Lopes & Baranes 2013 TICS 17:585; Kidd & Hayden 2015 Neuron 88:449). Infants withdraw
  attention from sequences too complex to learn (Kidd, Piantadosi & Aslin 2012 PLoS ONE 7:e36399).
- Midbrain DA neurons signal the arrival of information (Bromberg-Martin & Hikosaka 2009 Neuron 63:119); curiosity
  states recruit the DA circuit and enhance hippocampal encoding (Gruber, Gelman & Ranganath 2014 Neuron 84:486).

## 4. The mechanism (`research/runners/_curiosity_lp_trace_derisk.py`)

One `SimulationBridge`: `topic_aff` (12 slices x 25 RS) -> `ask` (80 RS) through depression-dominant STP synapses;
`reward_us` (40 RS) -> `snc` (30 RS), STP disabled on that pathway so the DA read never habituates.

- **Presentation.** On an abstain, the topic's slice gets an 8-step burst of `novelty * WORD_DRIVE` (novelty = the
  production `TopicNoveltyGate` read over the battery world's vocabulary, seed 42 like production), then 20 silent
  steps; the ASK-pool rate over the 28 steps is the want; `curious = want >= threshold` (the organ's rule). The
  neuron state is washed before and after (the organ's drift-free protocol); the synapses' `cp_stp_x`/`cp_stp_u`
  are NOT restored: that state is the trace. Slice 0 is the calibration reference, never a topic.
- **Trace.** Each presentation depletes only that slice's synapses (U = 0.35). Recovery tau_d = 12 000 ms of
  substrate time against 400 ms of silent substrate time between turns (3.3 % of the deficit per turn): a compressed
  clock for a ~2 s inter-turn gap against a ~1 min habituation recovery (Kandel: minutes).
- **Answer.** A teach turn is the store write. For each word of the taught fact that has a topic slice, LP = its
  novelty before minus after the write; `reward_us` gets LP x 2400 pA on top of the SNc's 220 pA tonic drive; the
  DA read is `clip((snc_hz - snc_neutral_hz) / 180, 0, 1)` and the slice's synapses are restored
  `x <- x + DA (1 - x)`. **Declared host step:** the engine has no neuromodulator target on the STP resource
  variable, so this restore is written by the module, gated by the spiking SNc read (the same routing class as the
  critic's `current_reward_signal`). Realization rung: an additive, default-off `sim/` DA -> STP-recovery target.
- **Declared host boundaries (unchanged from the organ):** the topic word (`extract_topic`), the graded novelty scalar
  (`TopicNoveltyGate`), the word -> slice table (first-seen allocation; an input code, not a count), the question
  frame. The organ battery's store/recall is the world scaffold and is not under test.

### 4.1 Parameters fixed by reuse (not chosen here)

STP_U 0.35, STP_TAU_F 10 ms, INPOP 25, DRIVE_STEPS 8, READ_STEPS 20, WORD_DRIVE 650 pA (habituation organ); ASK 80 RS
and its build flags (curiosity organ); reward_us 40, snc 30, weight 10 at density 0.6, SNC_TONIC_PA 220, US_GAIN_PA
2400, SNC_REWARD_SCALE 180, W_WARMUP 18, W_MEASURE 24 (DR-1 bridge); N_READ_REPS 4 for calibration. Chosen here,
before data: TAU_HAB_MS 12 000, INTER_TURN_MS 400, N_SLOTS 12, topic -> ASK density 1.0 and jitter 0.2, lesion tau_d
200 ms (the engine's own default E->E STP recovery).

### 4.2 The one calibrated value: the topic -> ASK weight (design smoke, seed 9, rule fixed now)

`--design` builds the bridge on DESIGN_SEED 9 for each W in (10, 15, 20, 30, 45, 60, 80) and reads the fresh
calibration want at NOVEL_SIGNAL 0.95. W_TOPIC_ASK := the grid value whose want_novel is closest to 40 Hz (the
organ's on-spec FAITHFUL/CALMER operating point, ~42 Hz; s7 pooled 38.2 Hz); ties go to the smaller W. Nothing about
asking order, stopping or the battery turns is read in that smoke. The chosen W is written into the module and
committed with the design artifact BEFORE the battery runs. No other parameter is changed after that commit.

**Design result (seed 9, `research/findings/raw/_curiosity_lp/design_s9.json`, run at 3449ba18a; recorded before
the battery).** W_TOPIC_ASK = 80 (want_novel 34.9 Hz, familiar 1.3 Hz, threshold 18.1 Hz; w = 60 gives 23.7 Hz).
One presentation fires all 25 afferents once and takes the slice's x from 1.000 to 0.650. Two further reads, written
down here because they bear on the bars: (a) the 400 ms silent gap is NOT silent for the afferents: the OU background
fires 167 topic-afferent spikes per gap (all slices), so x kept falling over the gap (0.650 -> 0.564) instead of
recovering; (b) a second presentation after one gap reads 13.4 Hz, below threshold. Prediction from (a), stated
before the battery: every slice, presented or not, loses transmission with elapsed turns, so B1/B1b (first asks)
are at risk from turn count alone, which would make the trace a clock rather than an ask count. Per the no-tuning
rule NOTHING is changed in response: the battery runs as frozen, and a failure is reported as this method's
verdict (an OU-driven afferent background depletes a long-tau_d synapse), with the next method named, not fitted.

## 5. Arms (each a fresh build per seed)

| arm | manipulation | counted |
|---|---|---|
| intact | as built | yes |
| intact_b | same-seed rebuild (the null) | yes |
| lesion_habituation | tau_d 200 ms: depression recovers between turns (the trace's memory removed) | yes |
| lesion_drive | topic -> ASK weights zeroed (the analog of `BRAIN_CURIOSITY_LESION`) | yes |
| perm_order | a per-seed shuffle of the 16 probe/teach turns (ibex triple and toad 1..4 keep their order) | yes |
| perm_exposure | Stage-2's own permuted exposure: eel/crab/newt taught instead of yak/moose/goat | yes |
| shared_slice | every topic on ONE slice (topic specificity removed) | informational |
| lesion_reset | the DA-gated restore off | informational |

The manipulation is re-read at the end of every run (`manipulation_at_end`: the live stp_tau_d and the summed topic ->
ASK weight), per docs/TERMS.md "lesion".

## 6. Bars (pre-registered; GO only if every bar holds on all 8 battery seeds)

- **B1** intact asks on g_eel (the first novel probe): 8/8.
- **B1b** intact asks on the first probe of every novel topic (g_eel, g_crab, g_newt, lp_ibex_1, u_toad_1): 8/8.
- **B2** intact does NOT ask on u_toad_4 (the 4th unanswered probe of one topic): 8/8.
- **B3** intact does not re-ask the answered gap: lp_ibex_2 recalls "marmot" and is not asked, 8/8 (0/8 re-asked).
- **B4** taught-topic moats m_yak, m_moose, m_goat: 0 asks (0 of 24).
- **B5** lesion_habituation asks on u_toad_4 on 8/8, differs from intact there on every seed, and the attributable
  fraction of the lesion's extra toad asks against the intact_b null is >= 0.5 (`tools.lab.attributable_to`).
- **B6** perm_order: B1b, B2, B3 and B4 hold, 8/8.
- **B7** perm_exposure: asks on >= 2 of m_yak, m_moose, m_goat (now untaught) and on none of g_eel, g_crab, g_newt
  (now taught); B2 and B3 hold; 8/8.
- **B8** intact_b's per-label asked vector equals intact's, 8/8.
- **B9** lesion_drive: 0 asks on every turn, 8/8.

Informational predictions (stated now, not counted): shared_slice leaves at least one later first-gap probe unasked
on >= 6/8 (a slice shared by all topics is a global brake again); lesion_reset is reply-identical to intact (with
word-level novelty an answered topic reads familiar and is never presented again) while its ibex slice x after
t_ibex stays below intact's; the number of toad probes asked before the first unasked one is reported, not barred.

Not claimed by a GO: anything about the integrated chat (the organ battery has no DA crave gain, no chat parse/store,
no co-resident organs); "load-bearing" in the docs/TERMS.md reply-level sense (that needs the handler arms); six-seed
generalization (these are dev seeds); an LP estimate richer than "the answer reduced this topic's novelty".

## 7. Commands (CPU, numpy; each under the brain lock, mem_ok 9 and memcap 9)

```bash
.venv/bin/python -m research.runners._curiosity_lp_trace_derisk --selftest
flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock bash -c 'bash tools/mem_ok.sh 9 && bash tools/memcap.sh 9 -- \
  env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -u -m \
  research.runners._curiosity_lp_trace_derisk --design --out research/findings/raw/_curiosity_lp/design_s9.json'
flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock bash -c 'bash tools/mem_ok.sh 9 && bash tools/memcap.sh 9 -- \
  env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -u -m \
  research.runners._curiosity_lp_trace_derisk --battery --seeds 1,2,3,4,5,6,7,8 \
  --out research/findings/raw/_curiosity_lp/battery_dev1-8.json'
.venv/bin/python -m pytest -q tests/test_curiosity_lp_trace_derisk.py
```

## 8. Residuals and the integration path (named now)

- The DA-gated restore is a host write gated by a spiking read (section 4); an additive `sim/` target is the rung.
- With word-level topic novelty the restore is reply-silent in this battery: relation-level novelty (topic x
  relation) is what would make a re-opened topic depend on it.
- Integration (not this packet): the slices would sit on the shared cortical pool as afferents of the organ's own ASK
  slice, and the global DA crave gain would stop being the only brake. Whether the DA gain should keep scaling the
  want on top of the per-topic trace is a separate question for the handler battery.

## Result: dev battery, seeds 1-8 (2026-10-13; recorded by Claude after the builder was cut off)

Verdict **NO-GO**, at the frozen weight 80 and tau_hab 12000 ms. File: `research/findings/raw/_curiosity_lp/battery_dev1-8.json`.

Bars that failed:
- B1 (asks on the first novel probe): 1/8.
- B1b (asks on every first gap): 0/8.
- B6 (perm_order): 0/8.
- B7 (perm_exposure): 1/8.

Bars that held:
- B2 (stops by the 4th probe): 8/8.
- B3 (an answered gap is not re-asked): 8/8.
- B4 (moats silent): 8/8.
- B5 (habituation lesion asks on the 4th probe; attributable fraction 1.0): 8/8.
- B8 (null): 8/8.
- B9 (drive lesion): 8/8.

Informational: `asks_before_stop` is 0 on every seed. The intact brain never asked at all. B2, B3 and B4 hold only because nothing was ever asked; they are not evidence of a brake.

This is the failure predicted in the pre-battery note. Background noise fires the topic slice during the silent inter-turn gaps, so the synapses are already depressed before the first probe. The brake runs on a clock, not on unanswered asks.

Next method (no-defer; a new design, registered before data): topic input neurons with background noise OFF, as the habituation organ already runs. Keep every bar and arm.

What this verdict covers: the METHOD "noise-driven slice depression", not the capability.
