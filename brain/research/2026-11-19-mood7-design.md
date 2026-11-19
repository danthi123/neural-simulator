# Capability: Affect
# Record: brain/research/2026-11-03-mood-state-design.md
# Mood iteration 7 design: appraisal-selected lived register episode (2026-11-19)

Produced by workflow wf_d0120a3c-4a5 (3 designs: event-grounded 4/10, state-carry 4/10, test-first 3/10 after sonnet
skeptics; opus synthesis, below unedited). Reopens the 11-19 mood timebox on its own condition (the curiosity route
worked: results/2026-11-19-inquiry-{test,weekly}-1.5b.md). Stage 0 (reads only) runs first and can stop the arc.

**Winner: a merge, built on design 1 (event-grounded).** Design 1 keeps the signal inside the cortex. The brain's own pair-axis appraisal of the heard event picks which lived episode is reinstated, which is the action that worked for curiosity. Design 3 is rejected. Its mood is a host-supplied body value, so the mood test only measures a value the host supplied, and it never fires in open conversation. Design 2 is rejected as the main design, because its stored own reply to bad news carries sympathy words, which the owner's marker-word rule bans.

The merge takes four things from the other two:
- From design 3: two episodes matched in length and form, and a CONFLICT-style check that the reply follows the appraisal, not the event's words.
- From design 2: a fixed carry rule with a built-in dose prediction (stronger events carry longer), and no free decay constant.
- From the critiques: the missing controls. These are a shuffled-axis lesion of the appraisal, a host-coin arm, a neutral-register episode arm, an oracle arm, and surprise-matched neutral events.
- From the critiques: an open-conversation read.

Record note: the 11-19 timebox in `brain/experiments/results/2026-11-19-arousal6-1.5b.md` killed further mood action levers at 1.5B. Its own reopen condition was that the curiosity route works, and that is now met (6/6 PASS NARROWED). The reopen has to be written down through before-build, not assumed.

---

**(A) Record brief**

1. **What we knew.** Pushing a fixed direction or gain into matrix memory moves content, not intent, at 1.5B:
   - mood4: the push moved a clause, not the reply.
   - mood5 Stage A: neutral events read |v| 0.24-0.30, so the gate failed.
   - arousal6: carry was 0.17 in OFF, ON and CONSTANT alike, and echo rose by 0.08.
   - What the appraisal does read well: the sign of negative events (6/6 seeds, 3/3) and the cortex's own surprise (AUROC 1.00).
   - What it reads badly: positive events (2/6 seeds right; 0.625-0.75 on the dev bank), and neutral events sit at |v| 0.24-0.30.
   - What works is curiosity: a gate reinstates a lived own-reply episode kept outside ordinary memory (DiD +0.253, 6/6 seeds, lesson-only equal to OFF on 864/864 rows). Its known costs are that the episode's form is copied, its words can leak through sleep, and the gate fires on some real words.
2. **What is new.** Mood becomes the choice of which lived register episode the reply is spoken from. The cortex's appraisal of each heard turn drives it, held by the existing leaky integrator with TAU 4 (already a declared scaffold) and switched off by a dead zone. Carry is repeated, transient reinstatement while the integrator stays above its bar. Nothing is ever added to the ongoing state.
3. **Prerequisites met:**
   - transient reinstatement of an episode kept outside the list (`live_inquiry`, `_live_inquiry_at_birth`, `hippocampus.reinstate`);
   - a side-effect-free appraisal (`Affect.project_pure`, pairs read-out, band 8-15, with a sign-shuffled axis available for a lesion);
   - the instruments in `arousal_gain.py`;
   - a weekly guard (`eval/weekly.py`).
4. **Prerequisites missing:**
   - a learned appraisal: the pair axis is a text-side scaffold, not multimodal grounding;
   - emotional tagging and consolidation of tagged episodes in sleep;
   - a learned mapping from mood to register (the two lessons are teacher-written).
   - So a PASS can only be "MET at scaffold level", like milestone (c).
5. **Expected outcome:**
   - P(Stage 0 appraisal gate passes) is about 0.45. The neutral |v| fault and sad-topic neutral events are the likely killers.
   - P(pilot GO given Stage 0) is about 0.55.
   - P(full six-seed PASS given the pilot) is about 0.50.
   - Overall P(milestone (b) recorded MET, negative pole, by 11-30) is about 0.12-0.15.
   - P(narrowed partial: an effect shown but the appraisal-lesion or neutral-episode control fails, so not "mood") is about 0.25.
   - The most likely single outcome is a NOT MET verdict on the pair-axis appraisal at Stage 0, for about 2 node-hours.
6. **The expected cheat, and its guard.** A PASS that is only "reinstating a short closed reply makes closed replies" is the expected cheat. It is blocked by three things: episodes matched in length and question-back, the NEUTRAL-EP control, and the SHUFFLED-AXIS and HOST-COIN controls. Without them a PASS is not reportable.

---

**(B) Build plan, smallest first**

0. **Record and pre-registration.**
   - Run `bash tools/before_you_build.sh "mood lived register episode selected by appraisal"`; run `tools/w_follow.sh` first if the corpus is STALE.
   - Add a reopen line under the 11-19 timebox in `brain/experiments/results/2026-11-19-arousal6-1.5b.md`.
   - Write section 20 of `brain/research/2026-11-03-mood-state-design.md` as the pre-registration (section C below), committed before any read.

1. **Stage 0 appraisal probe.** No Brain change and no generation, so this is the cheapest kill. New `brain/experiments/mood_episode.py appraise`.
   - Grow `Affect(enabled=False, readout="pairs")` per seed with `draw_pairs(rng)`.
   - For every bank text, record:
     - v from `project_pure`;
     - v under the sign-shuffled axis (`Affect.shuffled`);
     - surprise from `top2_mean` / `brain.novelty`, normalised with the arousal6 `calib_1.5b.json`;
     - whether `hippocampus.recall` would fire on the fillers and probe after T1 is stored. A recall blocks the mood branch, because ordinary recall comes first in `hear()`;
     - whether the inquiry gate would fire (`_novelty` against the `INQUIRY_BY_SIZE` bar).
   - Simulate the m trajectory and the fire pattern offline.
   - Banks are frozen and committed before the run (see C). Workstation CPU, seeds 42/43/44, about 15 minutes.

2. **`brain/affect.py`.** Add `mood_step(v, dz)`. It returns `v_gated = v if |v| > dz else 0`, then calls the existing `update(v_gated)` with `TAU_MOOD=4` (fixed, not fitted). Add a `MOOD_EP_BY_SIZE = {}` constant, filled only after Stage 0. `modulate`/`readout` are unchanged.

3. **`brain/brain.py`, shared lesson helper.**
   - Factor `live_inquiry` into `_live_lesson(telling, probe, guided) -> Episode`. Ep delta is the own turn read from `self.base`; `source="self"`; it is popped from `hippocampus.episodes`.
   - Factor `_live_inquiry_at_birth` into `_lived_at_birth(fn)`. It restores Python, numpy and torch RNG, arousal, `monitor_log`, `_last_novelty`, state, `_recent_replies` and the meta turn count.
   - `live_inquiry` must stay byte-identical (the existing inquiry tests).

4. **`brain/brain.py`, the switch.**
   - Add `Brain(mood_episode: str|None=None, mood_pin: bool=False)`. `"auto"` uses the `MOOD_EP_BY_SIZE` row for this size; no row means off. Not saved in the home.
   - `_grow_mood()` builds a private `self._mood_affect`. The existing `self.affect` is left untouched and `enabled` stays False, so the old push can never be applied.
   - `live_affect_episode(pole, event, probe, guided)` fills `self._affect_eps["low"|"high"|"neutral"]`. The third, a neutral-register episode, exists only as a control.
   - Save and load them as `affect_episodes.pt`, next to `INQUIRY_FILE`.

5. **`brain/brain.py hear()`.**
   - When the switch is on: `v = project_pure(cortex, base, text)` and `mood_step(v, dz)`. The update runs BEFORE the gate.
   - Add a new `elif`, placed after the inquiry branch and before the self_belief branch:
     - `pole = _mood_gate()`: low if m <= -B, high if m >= +B and the positive pole is enabled.
     - If `pole` is set and `not mood_pin`: `active = hippocampus.reinstate(cortex, state, _affect_eps[pole])`.
     - Set `info["mood"] = {v, m, fired, pole, blocked_by}`, where `blocked_by` is recall or inquiry, logged for the displacement count.
   - Harness hooks, each consumed once per turn:
     - `mood_fire_override` (a list): HOST-COIN replays fire patterns.
     - `mood_pole_override` (`"swap"|"neutral"|"oracle"`).
     - `mood_axis="shuffled"`: the appraisal lesion.
     - `mood_current=True`: m = v_gated with no integrator, i.e. TAU 1.
   - With the switch at None, no new code runs.

6. **`brain/tests/test_mood_episode.py`.** All must pass, together with the 146 existing tests:
   - switch None is byte-identical to the current `hear()`;
   - PIN equals OFF over a 6-turn conversation;
   - an unfired turn with the lessons lived equals a brain without them;
   - nothing is written into `self.state`;
   - the episodes are absent from `hippocampus.episodes`, recall, familiarity, key centring, `_night_episodes`, the near-miss guard texts and the memory listing;
   - save/load round trip;
   - a recalled told fact blocks the mood gate;
   - `project_pure` touches no Affect state;
   - lesson texts are disjoint from the mood_probe lexicon, the primary_echo words, the pair words, every bank and a sympathy/apology list ("sorry", "that's tough", "I'm here", ...);
   - the two lesson replies are within 2 words of each other in length and agree on question-back.

7. **Stage 1 wording** (`mood_episode.py wording`): pool CPU, dev seeds 1144/1146/1147.

8. **Pilot** (`mood_episode.py pilot`): seed 42, DEV bank.

9. **Six-seed test** (`mood_episode.py test`): 6 seeds on pool nodes via `tools/gpu_queue.sh` and `tools/memcap.sh`, 2 seeds per node. Then:
   - `export` blind reader packets, and `score`/`table` (reuse `boot_ci`, `paired`, `cohen_kappa`, provenance and the where_ran sidecar from `arousal_gain.py`);
   - a results file `brain/experiments/results/2026-11-2x-mood7-1.5b.md`;
   - land-result with 3 fresh sonnet skeptics who recompute from raw labels.

10. **Weekly guard and open-conversation read.** Run `eval/weekly.py --model 1.5b --shared-sleep` with mood on and off, 6 seeds, and read the whole transcripts. Only then update `SCAFFOLDS.md`, `PREREQUISITES.md` (mood), `HANDOFF.md` and the board. The default stays OFF until this passes.

New `SCAFFOLDS.md` rows (each with its learned target):
- the three lived register lessons (teacher-written reply form);
- dz and B (dev-fit, per size);
- the TAU 4 integrator as the carry (the existing row, now used for gating);
- the reply priority: recall, then inquiry, then mood.

Recorded with them: the pair axis remains the only grounding, which is text-side.

Estimated cost: about 2 node-hours (Stage 0) + about 3 (Stage 1) + about 1 (pilot) + about 24 (test) + about 6 (weekly), so about 36 node-hours in all.

Suggested dates:
- Stage 0: 11-20
- build and tests: 11-20/21
- Stage 1 and pilot: 11-22
- test: 11-23 to 11-25
- readers and skeptics: 11-26
- weekly guard: 11-27/28

There is no slack for a second method before 11-30.

---

**(C) Pre-registration draft (section 20)**

**Banks.** Committed before any read; written by a different agent pass than the lessons; common words only, no coined words.

- **Classes, per frame:**
  - NEG: a real-life bad happening.
  - NEU: a mundane happening.
  - SNEU: neutral but unexpected or surprising; its surprise must be matched to NEG's.
  - HARD: neutral text on a sad topic ("I read about the plague"), negation ("not bad at all"), mild everyday annoyance.
  - POS: secondary.
- **Conversation:** T1 the event (a telling) / T2 and T3 the existing FILLERS / T4 "How's your day going?".
- **DEV-FIT:** 40 NEG, 40 NEU, 20 SNEU, 20 HARD, 20 POS, as T1 texts only.
- **DEV-VAL:** a disjoint 20/20/10/10/10.
- **DEV-CONV:** 12 frames x NEG/NEU/SNEU.
- **TEST:** 16 frames x NEG/NEU/SNEU/POS = 64 conversations.
- **EVERYDAY:** 24 ordinary tellings with real names, foods and places, plus 12 real bad-news tellings.
- **Lessons:** the lesson events and 3 wording pairs per pole are disjoint from all banks (unit test).

**Stage 0** (seeds 42/43/44; reads only).
- **Fit on DEV-FIT:**
  - dz = the larger of the 95th percentile of |v| over NEU and the 90th percentile over SNEU+HARD.
  - TAU = 4, fixed.
  - B = 0.15·dz. This predicts that an event at v = dz fires on T1-T2, and v >= 1.42·dz fires through T4. That is the dose prediction.
- **Validate on DEV-VAL. All must hold on 3/3 seeds:**
  - S0a: NEG T1 fire >= 0.70.
  - S0b: NEU + SNEU fire <= 5% of all turns.
  - S0c: HARD fire <= 15%.
  - S0d: AUROC of v between NEG and SNEU >= 0.80, and the AUROC of surprise alone between NEG and SNEU <= 0.65. The second checks the bank's surprise matching, so the gate cannot ride on surprise.
  - S0e: the mood branch is blocked by recall or inquiry on <= 20% of the T2-T4 turns of DEV-CONV.
- **Fail:** STOP. The verdict is on the pair-axis appraisal at 1.5B, not on reinstatement. Freeze the numbers, write the result, and the milestone is NOT MET. No fallback onto surprise, because that is arousal6 again under another label.

**Stage 1 wording** (dev seeds 1144/1146/1147, DEV-CONV, forced reinstatement).
- Low and high candidate replies are matched in length (within ±2 words) and question-back. They differ only in register: closed and flat versus open and bright. No emotion words and no sympathy words.
- Pick rule, frozen in advance: the highest blind register separation between low and high, subject to:
  - verbatim copy <= 10%;
  - lesson 3-gram overlap in replies <= 5%;
  - NLL fluency ratio <= 1.15.
- The neutral-register control episode is the same length, from the same pick set.
- The pick is written into section 20 before the pilot.

**Pilot** (seed 42, DEV-CONV, arms OFF / PIN / ON / ORACLE / SHUFFLED-AXIS; 1 sample).
- The pilot transcripts are read whole by the agent before any scoring.
- GO requires all of:
  - neg-vs-neu forced-choice accuracy DiD (ON − OFF) at T2+T4 >= +0.15;
  - OFF accuracy <= 0.65, so there is headroom;
  - PIN == OFF on 100% of rows;
  - ON differs from OFF on >= 50% of fired turns.
- ORACLE (fires on the class label) is diagnostic only:
  - ORACLE >> ON means the appraisal is the bottleneck;
  - ORACLE ≈ OFF means the episode does not work.
- Otherwise NO-GO with no test spend.

**Test arms** (seeds 42/43/44/100/101/102; turn_seed shared across arms; 40 tokens; 1.5B CPU fp32).
- ON: the method. All 64 conversations, 2 samples.
- OFF: switch None. All 64 conversations, 2 samples.
- PIN: switch on, lessons lived, gate pinned closed. Seed 42 only, as a byte-identity check.
- SHUFFLED-AXIS: the appraisal lesion. v comes from the sign-shuffled, norm-matched axis; everything else is the same.
- HOST-COIN: fire pattern copied from a random conversation of any class, so the marginal fire rate is matched.
- SWAP: NEG conversations get the high episode.
- NEUTRAL-EP: the neutral-register episode at ON's fire turns.
- CURRENT: TAU 1, with no integrator.
- SHUFFLED-AXIS, HOST-COIN, SWAP, NEUTRAL-EP and CURRENT run on NEG and NEU conversations only.
- FORCED: low episode on every turn. Exploratory, seed 42 only.

**Reader rubric** (haiku and sonnet; blind to arm and class; shuffled).
- Readers see only the probe and the reply. They never see the event.
- Primary is forced choice: a pair is the same frame, turn, sample and arm, with one reply from the NEG conversation and one from the NEU conversation. The question: "Which speaker comes across as more subdued or withdrawn, judged by how they engage (openness, energy, willingness to continue), not by any feeling they name?" The score is accuracy against truth, and chance is 0.5.
- Per-reply labels: (a) mentions the earlier event's content; (b) sympathy or apology; (c) echoes the user; (d) invents content; (e) coherent yes/no.
- Cohen's kappa on primary agreement over a 40-item overlap must be >= 0.6, else INCONCLUSIVE.
- Mechanical co-measures, no reader: word count, the share of replies of 6 words or fewer, question-back rate, sympathy-token tally, mood_probe lexicon hits.

**Bars.**
- Unit and statistics:
  - Primary turns are T2+T4.
  - The unit is the frame, with a paired bootstrap over 16 frames pooled across seeds.
  - The per-seed sign is reported as a sampling-replication check only, never as replication.
- C0, appraisal on TEST T1 texts:
  - NEG fire >= 0.75;
  - NEU + SNEU fire <= 5% of all turns;
  - the S0d AUROCs, reported.
- C1, the claim: forced-choice DiD (ON − OFF) >= +0.15, with CI > 0 and positive on >= 5/6 seeds. It must also hold:
  - (i) on the subset of replies with no event mention and no sympathy label;
  - (ii) in a logistic fit with the word-count difference as a covariate, where the ON coefficient has CI > 0.
- C2, the cortex appraisal is causal:
  - ON − SHUFFLED-AXIS >= +0.10, CI > 0;
  - ON − HOST-COIN >= +0.10, CI > 0.
- C3, the register is in the selected episode and not in reinstatement as such:
  - ON − NEUTRAL-EP >= +0.10, CI > 0;
  - SWAP accuracy <= OFF accuracy (the direction flips).
- C4, selectivity:
  - on NEU and SNEU conversations, the share of rows where ON differs from OFF is <= 5% at every turn;
  - EVERYDAY false fires <= 2/24, and real bad-news fires are reported.
- C5, lesion changes the REPLY:
  - PIN == OFF on 100% of rows (inquiry precedent: 864/864);
  - ON differs from OFF on >= 50% of fired turns, and on <= 2% of unfired turns.
- C6, harms (ON − OFF):
  - sympathy/apology labels <= +0.03;
  - lexicon hits <= +0.02;
  - echo <= +0.05;
  - invents <= +0.05;
  - coherence >= −0.05;
  - verbatim lesson reply <= 10% of fired turns;
  - lesson 3-grams <= 5%;
  - distinct replies on fired turns >= 0.5 of OFF;
  - NLL ratio <= 1.15;
  - no "User:" leaks;
  - the recall/inquiry displacement count per arm is reported.
- Carry naming: report ON − CURRENT at T3+T4. It is mechanically expected to be positive, so it is not a pass criterion. Report dose too (fires at T4 on NEG events with v >= 1.42·dz versus below that).
  - "mood state carries" is used only if ON − CURRENT >= +0.05 AND the dose ordering holds.
  - Otherwise the name is "appraisal-gated register episode".
- C7, secondary positive pole: expansive DiD on POS >= +0.15 with CI > 0. If it fails, the claim is named "low mood only".
- C8, weekly guard:
  - keyed answers with mood on >= off − 2;
  - no lesson string or lesson 3-gram in any brain reply, including after the night;
  - the agent's whole-transcript read finds no new defect class (for example, going terse on ordinary questions).

**Decisions.**
- C0-C6 and C8 pass: milestone (b) is recorded MET at scaffold level. The narrow name is: "negative-valence appraisal selects a lived register episode over the next turns; text-side pair axis; teacher lessons; 16 frames". The switch stays OFF until land-result and C8.
- C1 passes but C2 fails: an episode effect without a shown appraisal. The milestone is NOT MET, and the verdict is on the appraisal read.
- C1 passes but C3 fails: generic priming from reinstatement. The milestone is NOT MET, and the result is named "episode register priming".
- C1 fails with C0 passing: a verdict on this method (lived register reinstatement at gain 0.5). The next method is per-token gain on surprising tokens, judged against PREREQUISITES before calling any wall.
- Kappa < 0.6: INCONCLUSIVE. The mechanical measures are reported, but no claim is made.
- The bars are not relaxed if the 11-30 date is missed.

---

**(D) Open questions only evidence can settle**

1. Is the pair-axis negative read above a dead zone clean against surprise-matched and sad-topic neutral events (S0d, S0c)? No run has recorded the magnitudes of negative events.
2. With ordinary recall first in `hear()`, how often does T4 "How's your day going?" recall the stored T1 telling and pre-empt the mood gate (S0e)?
3. Can a register difference survive matching length and question-back, or is "withdrawn" at 1.5B only shortness? If C1 holds only without the length covariate, the answer is shortness.
4. Does the cortex's own spontaneous reply to the follow-up, lived after the lesson event, carry enough register to replace the teacher-guided reply? That would be less host authorship. Try it as a 4th wording candidate in Stage 1, report it, and do not pick it unless it wins on the frozen rule.
5. Does the integrator's dose prediction (stronger events carry longer) appear in the replies, or does any carry vanish at reinstatement gain 0.5?
6. Does the gate hurt real conversation, for example by going terse after genuinely bad news when the user wants help? Only the whole-transcript weekly read can say.
7. Is the positive pole usable at all, given the earlier 2/6 and 0.625-0.75 sign accuracy?

Files referenced:
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/brain.py`: `hear` L445, `live_inquiry` L627, `_live_inquiry_at_birth` L658, `_night_episodes` L763, `INQUIRY_FILE` L48.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/affect.py`: `project_pure` L320, `update` L338, `TAU_MOOD` L171, `shuffled` L301.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/arousal_gain.py`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-19-arousal6-1.5b.md`: the timebox note that needs the reopen line.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-03-mood-state-design.md`: target for section 20.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/SCAFFOLDS.md`