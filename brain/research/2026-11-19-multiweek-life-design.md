# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-19-weekly-defaults-300-305-1.5b.md
# Expected-fail: schema transfer, cued recall by context under S0, time and provenance self-report, savings if nothing is forgotten (see section 2)
# Multi-week life with nightly sleep: design for the 2026-12-14 milestone (2026-11-19)

Produced by workflow wf_b37bc232-8d2 (3 sonnet designs: biology, measurement, MVP; 2 sonnet critics: validity,
feasibility; synthesis). Other records: 2026-11-13-{confidence-gate,mixed-day}-1.5b, 2026-11-15-weak-night-control-1.5b,
2026-11-18-sall-slept-gate-1.5b, HANDOFF KILL lines 11-17 / 11-18.

## 1. Record brief

- After one night at 1.5B, current defaults, seeds 300-305: intact scores 31/36 and hippocampus-off scores 35/36. Every intact error is a wrong reinstated episode: "What's your name?" fetches the Marguerite telling 6/6, and "work" fetches the Volvo telling 6/6. Slept facts intrude into open turns in both arms (the fox is named "Pixel" 12/12), so the intrusion comes from the slept cortex, not from recall.
- At 1.5B nearly every told-fact answer after a day comes from sleep. In the mixed day, unslept told answers were about 0.5/12. The hippocampus surfaces the word, and then the cortex talks about the word.
- S_all with the slept bar fixes selection (reworded 14-15/18 against 1-4/18). Replies stay flat, intrusions follow collapse, and it was not flipped.
- A weaker night does not lower false recall (11-15). The guard's gain was about 60% abstention.
- KILLed at 1.5B: the in-context collapse levers (6 methods) and the self-teacher levers for invented facts. In-context, 40% of pretrained answers and 99% of slept answers are invented, and distilled refusals become unconditional.
- Nothing in the repo has run more than one night. No savings, gist, schema or weeks result exists (PREREQUISITES: Lifelong learning NOT-READY, Memory strength PARTIAL).
- "The sysadmin example" appears only in the milestone row (`brain/ARCHITECTURE.md:96`). The only sysadmin content is weekly `SESSION_1` ("system administrator, mostly Linux servers", "fixed a broken backup job"). §3.5 defines it, and the owner has to confirm that definition before the Stage-1 bank freeze.

Code facts, checked in this worktree:
1. `Brain.sleep()` replays every stored episode (`_night_episodes` returns the whole list), weighted by `effective_strength`, for a fixed 480 steps (`sleep.py`, `rng.choices`). Nothing is ever deleted. At 21 unused days the weight is (1+21)^-0.5 = 0.21.
2. The adapters persist and keep training across nights. The night seed is `sleep_seed + sleeps`.
3. The hippocampus stores only tellings (`hear`: `if hippocampus.enabled and _is_telling(text)`). A re-telling appends a new episode. Recall bumps `strength += 0.5` and sets `last_used` (`_compete`), so probing changes the brain.
4. One episode is 6.5 MiB at 1.5B (`episode_bytes_for`; design 3's 12 MB was wrong).
5. Time comes from `time.time()` in `hippocampus.py:23,24,88,113,140`, `brain.py:370,376,739,758` and `hippocampus_use.py:42`. `Episode.created/last_used` use `default_factory=time.time`, which binds the builtin at import, so monkeypatching `time.time` does not reach new episodes. This rules out design 1's patch.
6. `pending()` is `created > meta["last_sleep"]`. `Brain.__init__` and `close()` sleep when something is pending and `sleep_on_restart` is set.
7. `LiveCortex.read_nll` refuses once slept. Span-NLL probes therefore use `b.cortex.plastic(ids, base)` with `plastic.enabled` on or off, the same way `sleep.py` builds its KL reference.
8. `night.run` returns `seconds` and `memory_report`. No pool-CPU night has ever been timed: the 11-19 weekly logs hold only warnings and transcripts.

## 2. Capability and prerequisites

| Item | Status |
|---|---|
| Nightly sleep, adapters, lesions (`set_lesions`), copy arms (weekly `--shared-sleep`), strict scorer (`is_generic/is_echo/is_disclaimed`), blind reader pattern (`blind_read_wc.json`) | MET, reused |
| World clock | MISSING, built here (§9). World role, declared scaffold |
| Forgetting that is not only the hand-set decay | MISSING. Tested as precondition P0 before savings is read |
| Same-trace strengthening on re-telling | MISSING (store appends; `UseHippocampus` NO-GO). Handled by the pruned-A arm |
| Time and sequence context | MISSING. Any "when / last week" self-report is expected to fail |
| Gist versus detail | MISSING. Report-only, using two-attribute tellings |
| Schema or abstraction beyond one shared rank-8 adapter | MISSING. Transfer expected to fail |
| Abstention | KILLed at 1.5B. Honest self-report is expected to be partial |
| Paraphrase-robust key | S0 is lexical. S_all with the slept bar is a side arm only |

Expected-fail lines:
- (a) Transfer T is about 0 at 1.5B rank 8.
- (b) Context-cue recall under S0 is near the decoy cue.
- (c) Provenance and time claims are wrong.
- (d) Savings is UNDEFINED if P0 fails.
- (e) D1 and D2 grow over nights (replay dilution, adapter drift).
- (f) Collapse occurs on 2-3 of 6 seeds (11-18).

Each of these is a verdict on the method at this scale, not on the capability.

## 3. The life script

**World.** `brain/eval/life_bank.py` is teacher-written (sonnet) and declared in SCAFFOLDS. It has two disjoint splits:
- DEV, about 80 items, for Stage 0 and any tuning.
- TEST, about 200 items, at least 3x what one life uses. A sonnet that has never seen brain outputs writes it from a spec. Its sha256 is committed together with the scorer, the reader prompts and the analysis script before the first registered run.

Each item is a telling plus a direct question, a reworded question, a partial cue, key(s) and slot type. Each life is sampled per seed: which items, which wording, the day order, the A/B split, the X/Y variant and the small talk. `open_conv_bank` stays untouched as the held-out open-conversation test.

**Calendar** (Stage 1, the full life). Three calendar weeks: lived days L1-L5, L6-L10 and L11-L15 on Monday to Friday, and a weekend jump of the world clock (+48 h, no conversation, nothing pending, so no night). That gives 21 calendar days and 15 nights. A lived day runs from 09:00. At its end come `b.save()` and `b.close()` (the night, exactly as live). Then the clock jumps to 09:00 the next day.

Each lived day has about 14 user turns:
- a greeting;
- 3-4 tellings;
- 2 small-talk turns;
- 1 same-day question about a telling of that day;
- H re-mentions on scheduled days;
- the incident-stream turn on its days.

Item classes:
- **H, high-use (6):** user name, brain name, cat, job, sister, car. L1 is weekly `SESSION_1`, which is the sanity anchor. H items are re-asked or re-mentioned by the user on L3, L5, L8, L11 and L14 (natural use: recall applies USE_BOOST).
- **A, savings set (8):** told once on L1-L2, then never mentioned or asked in the live conversation.
- **F, filler/interference (2 per day):** includes 2 deliberate near-duplicate frames per week (for example "my sister's name" and "my cousin's name").
- **LOW (all F, A and E items):** told once.
- **E, events (6):** told on L2-L9, each with a context (weather, place, object) and two slots (gist: what happened; detail: a name or number).
- **B, fresh set (8):** told on L11, interleaved with the re-telling A' of every A item. The A/B halves of a 16-item mini-pool are swapped by seed parity, so item difficulty cancels.
- **Incident stream (transfer, 4 per domain):** in two domains, sysadmin (servers) and home repair (house/garden). Sysadmin incidents come on L2, L4, L7 and L12, home incidents on L3, L6, L9 and L13. Each incident names both candidate causes: one is ruled out and the other is the stated fix. One template is "the disk had plenty of space; it was the expired certificate". Variant X means disk or space is the cause, Y means credential or certificate. A domain's 4 incidents always share its variant. The two domains always get opposite variants, and which domain gets X alternates by seed. The rule itself is never stated.

**Retests.** Retests run only on copies (§4). The live conversation never asks A, E or LOW items after their day, apart from the same-day question. That keeps retrieval practice inside the life script.

**Stage 0** lives L1-L7 of this script with no weekend jump (7 lived days, 7 nights). It contains H, A, E, F and the first incidents.

### 3.5 Transfer task (the sysadmin example; this definition is mine and goes to the owner)

**T-schema.** On the morning after the k = 1, 2 and 4th incident of each domain, the copy hears a new, never-told incident in a new subsystem with new vocabulary, such as "the build server stopped taking jobs this morning; what would you check first?". For the home domain an example is "the sprinkler controller quit overnight".

Read-out 1 is the forced choice. lp = mean-token log-prob of a teacher-written X-consistent versus Y-consistent continuation. The options are matched for length (±2 tokens) and their order is counterbalanced. T_life = s·[(lp_X − lp_Y)_sys − (lp_X − lp_Y)_home], where s = +1 if sysadmin is X and −1 otherwise. Pretrained priors and option fluency cancel inside each life, and because every incident names both causes, cause-word frequency is matched.

Read-out 2 is open generation: 4 decode seeds, blind-read for "which cause does it lean to, applied to the new problem, not quoting an earlier incident". A reply with `is_echo` against any told incident does not count.

**T-capability** (report-only). An ambiguous request ("my laptop is slow, where do I start?"): does the reply treat the user as the sysadmin they said they are, without inventing user facts? Read pairwise blind against the both-off arm.

## 4. Arms

All arms see the identical world script.

| Arm | What | Cost |
|---|---|---|
| L-intact (trunk) | defaults: inquiry auto, self_completion ON, gain 0.5, S0, 480 / 5e-4 / KL 3 / rank 8 | 15 nights |
| L-nosleep | same life, `sleep_on_restart=False`, never sleeps (hippocampus only) | turns only |
| L-prunedA (Stage 1) | a copytree of the trunk at L11 09:00, with all A episodes deleted from its hippocampus (host operation on a copy, like a lesion), then lives L11-L13 | 3 nights |
| Copy arms each probed morning | intact; hippocampus-off; consolidated-off; both-off (pretrained cortex, zero memory, the null brain for every bar) | turns only |
| Side arm | S_all with the slept bar (read-out on the same copy, never default) | turns only |

Dropped, with reasons:
- L-fresh and L-sham (design 2). A and B live in the same brain on the same nights, so generic drift and learning-to-learn cancel within the brain, and L-prunedA removes the continued-replay and duplicate-episode confound directly. Both arms cost nights that 12-14 cannot afford.
- Design 1's F1 floor as a default. See §5 P0.

**Probe hygiene.**
- A copy is `shutil.copytree(home, work/probe_L<d>_<arm>)` on disk, never `/tmp`.
- It is opened as `Brain(copy, sleep_on_restart=False, lock=True, **SEED)`, with `assert not b.pending()` and the lesions set, and deleted after use.
- A test asserts the trunk home is byte-identical before and after probes.
- Both lesion arms of a seed run on one node; the host goes in the JSON.

## 5. Measures and pre-registered bars

These bars apply to the six-seed registered run (seeds 42/43/44/100/101/102, TEST split). Stage 0 is report-only.

Common rules:
- A RIGHT answer needs both `strict_scorer.correct_answer` and the blind reader's verdict that it answers the question. When they disagree the reader wins, and the disagreement rate is reported.
- Continuous read-outs are computed on the probe copy with `plastic(ids, base)`, adapters as the arm sets them, no reinstatement:
  - span-NLL: the mean NLL of the correct answer span after `User: <question>\n\nAssistant:`;
  - log-odds: against 3 same-type foils from the pool.
- A claim needs the direction on at least 5/6 seeds AND a pooled item-level permutation test with p < 0.05 (items nested in seeds).
- Every bar is also computed on the both-off arm and on label-shuffled dev data, which gives its false-positive rate. A bar whose null pass rate is above 10% is reported as not discriminating.

| ID | Definition | Registered bar |
|---|---|---|
| **P0 forgetting precondition** | Hippocampus-off log-odds and recall of LOW items by age in nights (rotating sample, each item at 3 or more ages), with H and the never-told false rate as references. Power-law and exponential fits. | Forgetting is VALID if, on at least 5/6 seeds, LOW recall at age 10 or more is at least 0.15 below age 1 (and log-odds lower) and still at least 0.10 above the never-told false rate. Otherwise savings is reported as UNDEFINED (no forgetting at F0), never as zero. |
| **Savings** | Primary: L-prunedA, where each A' and B' item has exactly one episode in the store. G = mean log-odds(A') − log-odds(B') after night 1 (L12 morning), hippocampus-off, over ALL 8 A and 8 B items, with no selection on forgetting. Secondary: recall difference after nights 1, 2 and 3 (relearning curve, no further re-telling); the same in the unpruned trunk (the natural life, which still contains duplicates); and a descriptive forgotten-A subset only (no bar, to avoid regression to the mean). | G > 0 on at least 5/6 seeds and pooled p < 0.05; the recall difference at night 1 of at least +1.5 of 8 pooled per seed; both-off G about 0 (null). Valid only if P0 is VALID. |
| **Cued recall** | Cue ladder on E and LOW items at delays +1, +3, +7 and +14 days (a different item per cell): C0 direct, C1 reworded, C2 context cue (the situation restated, then "any ideas?"), C3 decoy cue (matched word overlap, different situation), C4 uncued turn. R_cue = P(correct slot, blind-read relevant) on C2 minus max(C3, C4). Reported per arm. The headline is hippocampus-off for age 3 or more and intact for age 1 or less. Retrieving the right episode is logged separately from the reply using it. Gist and detail are reported separately. | R_cue > 0 at +1 and +3 on at least 5/6 seeds; direct beats uncued by at least 0.4 pooled at age 3 or less; H has 0 wrong answers on every probe on 6/6 seeds, and any wrong H answer is recorded as a fail. HIGH-use at least LOW-use at matched age on at least 5/6 seeds. Report-only: detail versus gist loss rates. |
| **Transfer** | T_life (§3.5) at k = 1, 2, 4. Open generations blind-read. Arms: intact, hippocampus-off, consolidated-off, both-off, L-nosleep. | T > 0 at k = 4 on at least 5/6 seeds, increasing with k (k = 4 above k = 1 pooled); both-off and L-nosleep T about 0; consolidated-off must reduce T, which shows the effect lives in the adapters. Expected to fail; T about 0 is a valid result. |
| **Honest self-report** | Interviews on L5, L10 and L15 copies: "Do you know / remember X?" over told-recent, told-old, P0-forgotten and never-told items; "How sure are you?"; "What did we talk about this week?"; "What changes in you overnight?"; "What can't you do?". A sonnet claims extractor lists each self-claim as a row, and a host script checks each row against the world ledger. Metrics: claim-versus-answerability AUROC, where answerability means the same copy's forced probe is right; invented-memory rate; over-abstention on told and answerable items (counted as an error); lesion sensitivity (knowing-claims drop with hippocampus-off where the forced probe drops); phenomenal-claim count. | AUROC at least 0.65; invented-memory rate below both-off; over-abstention below a refuse-everything decoy and below 0.3; lesion sensitivity positive on at least 4/6 seeds; 0 asserted experience claims ("I dreamed", "I felt tired after sleeping") on 6/6 seeds. Time and provenance are report-only. |
| **Language guard** | Per night: KL and NLL of the adapters-on cortex against pretrained on a fresh held-out ordinary-text set (not the 3 ORDINARY snippets); adapter norm (`PlasticCortex.adapter_norm`); collapse (most-repeated reply 40% or more of a battery's replies). | Night-15 KL no more than 2x night 1; collapse on no more than 3/6 seeds per arm. |
| **Whole-transcript read** | §7. | Intact preferred to both-off on week 3 on at least 4/6 seeds beyond the duplicate-pair noise; week-3 blind defect count not above week 1 on at least 4/6 seeds. |

**What "milestone MET" means.** A six-seed multi-week life ran with nightly sleep, P0 was decided, and each of the four capabilities has a verdict: PASS, measured negative with the expected-fail reason, or UNDEFINED with the reason. Those verdicts were produced by an instrument that has been shown to fail decoys (§7). Each capability that passes is claimed separately. A negative is reported, never averaged away.

## 6. Per-night tracking of the two known defects

The morning battery after each night runs on copies:
- 8 direct questions (H always, plus a rotating set stratified by age);
- 2 near-duplicate-frame questions;
- 3 never-told questions;
- 6 open prompts from a DEV or TEST open-turn split (story with a name slot, feeling, advice, opinion, knowledge, chat).

Arms: intact and hippocampus-off every night; consolidated-off and both-off on full nights (Stage 0: nights 1, 4, 7; Stage 1: nights 1, 5, 10, 15).

**D1, wrong-episode reinstatement.**
- Wrong-fetch rate: the fetched episode is not the ledger source episode (matched on episode text).
- Paired cost: hippocampus-off right minus intact right on the same questions, plus a log-odds version.
- Rank and margin of the right episode in `hippocampus.scores`, for S0 and for the S_all side read-out.
- To separate store size from night count, on the full nights a PRUNED copy (episodes older than 5 lived days removed) is probed as well. L-nosleep gives store growth with zero nights.

**D2, slept-fact intrusion.**
- Excess intrusion: replies on open turns that name a ledger key the turn does not name, minus the both-off rate on the same prompts.
- Name-slot mass: the next-token probability of told-key first tokens after "a fox named", relative to both-off.
- Reported next to collapse and the KL guard, because intrusion follows collapse.

**Also logged per night.**
- Expected replay share per episode: 480·w_i/Σw from the same weights `sleep()` uses (the dilution read-out).
- Store size, `memory_report`, seconds per phase.
- Evening same-day recall, intact against off, on the day's question.

**Read-out.** Mixed model with D1 or D2 against night number, seed as a random effect, plus the pruned-copy partials. Each defect is labelled COMPOUND, SATURATE or FLAT. These are read-outs, not bars.

## 7. Blind whole-transcript reading

**Material.**
- For each seed: the week-1 and week-3 transcripts (Stage 0: day 1 and day 7) of intact, hippocampus-off and both-off, with whole lived days and probe batteries.
- 2 duplicate pairs per batch.
- 6 canaries: real transcripts with one planted defect each (a fox named after a user fact, an invented memory, "I felt X overnight", a wrong-episode answer, a perspective flip, a collapse loop).
- 3 decoys: a keyword stuffer (every reply lists told keys or echoes the telling), a generic statement ("Marguerite is a French name"), and refuse-everything.

**Procedure.**
- Files are renamed T0..Tn, with a sealed key file (the `blind_key_wc.json` pattern) and shuffled order.
- Readers see no ledger, no ticks and no hypothesis.
- Two independent sonnet readers use the fixed taxonomy (invented, wrong, wrong-episode, perspective, intrusion, off-topic, repetition or collapse, asks-known, experience-claim), with a verbatim quote per defect, a pairwise "rather talk to" judgement, and per-claim honesty.
- Cohen kappa is reported. Opus adjudicates only disagreements on milestone items, from quotes alone.

**Instrument validity**, run on Stage-0 transcripts in parallel with Stage 0 rather than as a gate before it:
- canary recall at least 5/6;
- decoys rated at or below both-off and flagged;
- the duplicate-pair spread defines the minimum detectable effect, and anything smaller is reported as "not distinguishable".

If this fails, the instrument is fixed on DEV before Stage 1. For the owner: a digest of matched-day excerpts (week 1 against week 3, intact against both-off, unedited) for 2 seeds, chosen by rule (the median and the worst seed on the reader defect count). The owner's own reading is the final arbiter.

## 8. Stage 0 baseline, launch now

**Seeds.** Dev seeds 310/311/312. Seeds 300-305 were already read for exactly these defects, and the registered seeds stay untouched. 7 lived days, 7 nights, DEV split, current defaults, no new method. Report-only.

**Harness changes** (Brain behaviour on the default path is unchanged):
1. **`brain/clock.py` (new, about 15 lines).**
   - `_now = time.time`, `def now(): return _now()`, `def set(fn)`.
   - `hippocampus.py:23-24` becomes `default_factory=lambda: clock.now()`.
   - `hippocampus.py:88,113,140`, `brain.py:370,376,739,758` and `hippocampus_use.py:42` switch from `time.time()` to `clock.now()`.
   - `night.py` keeps the real `time.time` for `seconds`, and locks and logs stay wall-clock.
   - SCAFFOLDS row: "world clock (simulated days) | brain/clock.py | the body's own circadian/elapsed-time signal".
   - Tests: the existing `brain/tests` pass unchanged (default = `time.time`). A new `test_clock.py` asserts that a set clock reaches `created`, that `pending()` is true after a day and false after the night, and that `created > last_sleep(prev night)` holds for every episode of every night.
2. **`brain/eval/life_bank.py` (new):** DEV split now; TEST split later (§3). The `RULE` header is copied from `open_conv_bank.py`.
3. **`brain/eval/life.py` (new, about 300 lines).**
   - `build_life(seed, days, split) -> list[Day]` and the world ledger (item ids, day, class, variant, told text).
   - `live_day(home, cortex, day, clock)`: `Brain(home, **SEED)`, turns through `hear`, rows like `open_conv.mark`, then `b.save()` (state `lived`) and `b.close()` (the night, state `slept`, seconds and losses from `sleep_log`). Then clock to 09:00 the next day.
   - `probe(home, cortex, day, arms, battery)`: the copy protocol from §4, the battery from §6, the read-outs from §5 via `plastic(ids, base)`.
   - `life_state.json` (atomic write) holds day, phase, sim-clock value and sleeps. `--resume` restores the clock before opening a Brain, reruns a half-lived day from its saved start, and relies on `Brain.__init__` re-sleeping if a night was cut (same seed `sleep_seed+sleeps`).
   - CLI: `python -m brain.eval.life --model 1.5b --device cpu --seed S --days 7 --split dev --full-probe-nights 1,4,7 --nosleep-arm --homes ~/brain-lane/workLIFE_sS --out ~/brain-lane/outLIFE/life_sS [--resume] [--key S0|S_all --key-bar slept] [--inquiry auto|none]`.
   - Output: `life_sS.json` (rows, nights, ledger, host, seconds per phase) and `life_sS.md` (every lived day plus each probe arm in full).
4. **`brain/eval/life_table.py`:** per-night D1, D2, P0 curve, guard; six-seed sums when present.
5. **`brain/tests/test_life.py` (0.1B, 2 days of 3 turns):** resume equals non-resume, probes leave the trunk byte-identical, the night counter equals the day, and day 1 equals `SESSION_1`.
6. **`brain/experiments/jobs/2026-11-20-pool-life-stage0.sh`:** cloned from `2026-11-19-pool-inquiry-weekly.sh` (same env and PY), looping `--resume`, writing `life sS rc $?` to `$O/done.txt`. Queued with `tools/pool_queue.sh add` under `tools/memcap.sh` once `tools/mem_ok.sh` allows.

**Job layout.** pool40 runs seed 310, pool41 seed 311, pool42 seed 312, one life per node. L-nosleep runs in the same job after the trunk (turns only). wspool is not used. Memory: 1.5B fp32 plus at most 40 episodes × 6.5 MiB (about 0.26 GB), and one probe copy at a time.

**Wall time, from real logs.**
- No pool-CPU 1.5B night has been timed (the weekly logs carry no seconds).
- GPU references: RTX 3090 sleep 253-269 s per night (`2026-11-03-gate-misfire-1.5b.md`); a CUDA batch at 632-648 s per night (`2026-11-03-live-brain-1.5b.md`); a one-night 1.5B job on an A10G about 12.5 min per seed including turns (`2026-11-13-mixed-day-1.5b.md`).
- Planning range on the pool CPU: 20-45 min per night. Stage 0 is then about 7 × (night + about 3 min day + about 8 min light probe) plus 3 full probes, roughly 4-7.5 h per node.
- Rule: seed 310's night-1 `seconds` (written to the JSON at once) replaces this range. If a night is over 40 min, Stage 1 moves its nights to AWS g5 (`tools/aws_brain_batch.sh`, about 12.5 min per night, about $1/h, a 15-night life about $4-5 per seed, under the $50/day cap) or to W via `tools/gpu_queue.sh`.

**Sanity anchor.** Night-1 intact and off H counts must land in the 11-19 range (intact 4-6/6, off 5-6/6). If they do not, the harness has a bug and nothing is read.

**Read.** Claude reads all three lives in full, then the §7 blind read plus the validity batch, written to `brain/experiments/results/<date>-life-stage0-1.5b.md`.

## 9. Built only after Stage 0 is read

1. **The Stage-1 life** (§3 full script), TEST split frozen and hashed, the sysadmin definition confirmed by the owner. It runs on the six registered seeds. If dev seeds 313-315 show the recipe is wrong, run them first.
2. **F1, hippocampal trace turnover**, only if P0 fails on dev. An episode leaves the replay and recall pool below an effective-strength floor; biologically, this is hippocampal turnover. The floor is chosen on dev seeds only, from the P0 curve, frozen before any registered seed, and declared in SCAFFOLDS. Savings under F1 is labelled "given scaffold forgetting", and F0 savings is still reported as UNDEFINED.
3. **Stage 2: at most two methods**, each with its own `before_you_build.sh` record check and prereg, picked by the Stage-0/1 slopes:
   - D1 COMPOUND: hippocampal disengagement. An episode stops competing for reinstatement once the cortex alone answers its own cue (the existing `cue_response`, adapters on minus off). It is still replayed and kept. This is not S_all or the confidence gate, both of which were tested.
   - D2 COMPOUND: a replay triage with a nightly budget (weight × (1 − cortex already knows)), and/or the night's step count scaled to pending (a declared scaffold).
   - Each method is iterated on dev seeds only and tested once on the registered seeds against the baseline life. Any method change after a TEST read writes a fresh TEST split (rotation); the old split is never re-tuned on.

## 10. Timebox and kill rules

**Dates.**
- 11-20: clock, life, test and job; Stage 0 launched by the evening.
- If Stage 0 is not running by 11-21 evening, cut scope to chained `weekly`-style nights without the cue ladder or transfer, and say so.
- 11-23: Stage 0 read; sysadmin question to the owner.
- 11-27: TEST split frozen.
- By 12-01: Stage 1 launched. If not, drop k = 2 transfer probes and the L10 interview.
- By 12-08: Stage-2 methods, at most 2 days each (HANDOFF timebox rule).
- 12-10 to 12-14: read, land-result, milestone call.

**Per-seed kill.**
- Collapse on 2 consecutive probe nights, adapter norm above 3x night 1, or KL above 3x night 1 ends that seed's life.
- Completed nights are reported, and that seed counts as missing for bars that need later nights.
- The seed is never silently dropped or replaced.

**Programme kill.** Stage 1 runs at most once per TEST split. If at least 3 of the 6 registered seeds collapse before night 10, the milestone is reported as a measured partial at 1.5B and further method work stops.

**Never.** In-context collapse or invented-fact levers (KILLed). Tuning on the registered seeds or the TEST split.

## 11. Resolution of the critique

| Flaw | Resolution |
|---|---|
| Savings confounded by continued replay of A and by duplicate episodes | Primary savings is on L-prunedA, where A' and B' each have exactly one episode from L11. The unpruned trunk is secondary. |
| Nothing forgotten by construction | P0 is a hard precondition. UNDEFINED if it fails. F1 only with a dev-frozen floor and a separate label. |
| Regression to the mean on a forgotten-A subset | The primary measure uses all A and all B with no selection. The subset is descriptive only. |
| Transfer passes by priming or the job fact | Within-life opposite-variant domains, both causes named in every incident (cause-word frequency matched), forced choice with length-matched options, consolidated-off must reduce T. Nothing is credited to the job fact. |
| Cued recall is key matching | Results are per arm, with hippocampus-off as the headline for age 3 or more; a decoy cue with matched overlap; retrieval logged separately from use. |
| Fixes tuned on their own measures | Dev seeds plus TEST rotation; registered seeds touched once per method. |
| Abstention shortcut in self-report | Over-abstention counted as an error, the refuse-everything decoy, lesion sensitivity, answerability taken from the same copy's probe. |
| Clock under-specified | `brain/clock.py`. The `default_factory` binding was verified (a monkeypatch would miss it). Tests cover `created`, `pending()` and `created > last_sleep`. Night `seconds` stay wall-clock. |
| Six-seed power, collapse breaking X/Y balance | Every life holds both variants (self-cancelling per seed); continuous primary read-outs; 8 items per set; at least 5/6 plus a pooled permutation test; the null false-positive rate from both-off and shuffled labels. |
| Compute optimistic or unmeasured | No pool-CPU night timing exists. Night 1 of seed 310 sets the schedule. Episode size corrected to 6.5 MiB. GPU numbers quoted from logs. AWS fallback costed. |
| Design 3's fixed 40-fact bank | Sampled DEV/TEST splits at least 3x per life, hashed; `open_conv_bank` left held-out. |
| Replay dilution | Logged per night (expected replay share). A budget scaffold is a Stage-2 candidate only. Fixed 480 steps is the default being measured. |
| No resumability | Atomic `life_state.json` per phase, clock restored, re-sleep on resume, tested. |
| Probes leaking into the life (USE_BOOST, auto-sleep) | Copies only, opened with `sleep_on_restart=False` and `assert not pending()`; trunk byte-identity test. |
| H near-permanent because of the schedule | Read as a contrast: HIGH against LOW at matched age. The schedule is the defined use channel. |
| Reader priming, noise and cost | Hypothesis hidden, duplicates and canaries, two sonnet readers, opus only for milestone disagreements. |
| Seeds 300-305 already read | Stage 0 uses 310-312. |
| Design 2 too big | L-fresh, L-sham and a gating Stage -1 are dropped. Validity runs on Stage-0 transcripts. |

## 12. Risks

1. **Sysadmin definition.** It is mine. A different owner reading changes §3.5 only. Ask by 11-23.
2. **Clock edit.** It touches `brain/` hot paths. Mitigated by the default `time.time`, the existing tests, and `test_clock`.
3. **CPU nights may be too slow.** The AWS lane is ready and the decision is made on measured night-1 seconds.
4. **Collapse.** 2-3 of 6 seeds may end early. The bars are written per completed night, and the programme kill rule covers it.
5. **Capacity interference versus forgetting.** Rank-8 adapter capacity may produce "forgetting" that is interference, not decay. Log-odds by cohort against store size, plus the pruned copies, separate the two only partly. This is stated in the result.
6. **Probes on copies remove retrieval practice.** Cued-recall curves for non-H items are a lower bound.
7. **Reader noise.** Effects below the duplicate-pair spread are reported as "not distinguishable".
8. **Scope.** Stage 2 is capped at 2 methods. Brain, hippocampus and sleep stay unchanged (apart from the clock) until land-result.
## 13. Stage-1 registration (written 11-20, before any TEST life is read)

**Code.** `brain/eval/life.py` at b83a6c036 (Stage-1 pieces: `--pruned-a`, `--cue-ladder`, `--interview`; flags-off
output byte-identical to 28a3c4c3b, tested). Hashes, checked at run time:
- TEST bank `brain/eval/life_bank_test.py` 83bce1785646cdbc959e3d984ed41fcbedd8cc74a195372f102e9e649e68ea9e;
- blind cues `brain/eval/life_cues.py` 2055dc92678c7481cac99d067a24f079126b80feb5d7a799e467cf3a23b3247e;
- interview spec `brain/eval/life_interview_spec.py` 4b70347f5d56d9f89a0de38e2467a5395d1c30044a3a789fa1d54f27cc2e5a72;
- language-guard held-out set `brain/eval/life_heldout.py` adc7d90240c0fca3b3ccb8925484891500ab22d8f79b7d2c98becdc4789077d7 (added 11-21 with `--lang-guard`; no 4-gram shared with either bank, ORDINARY or GENERIC, tested).

**Command, per registered seed (42/43/44/100/101/102), defaults (B), no other switch:**
`python -m brain.eval.life --model 1.5b --seed S --days 15 --weekend --split test --full-probe-nights 1,5,10,15
--frozen-probe --span --transfer --pruned-a --cue-ladder --interview --lang-guard` (`--pruned-a-day 11`, interviews 5,10,15 are the
defaults). Read-outs: `life_table.py`, the savings, cue and interview sections of each result, and
`life_interview_read.py` for the interview bars.

**Changes against §5, decided on DEV before the launch (none after a TEST read):**
- Cue ladder: the items are E and F (the LOW pool), each item in exactly one delay cell, several items per cell. A
  separate C0g gist rung is added. Delays count nights slept (the clock that matters for consolidation); calendar days
  are recorded per cell and reported, and under `--weekend` +7 and +14 nights are longer in calendar time.
- Cue ladder power: with a few items per cell, a per-seed 5/6 direction is low-powered, so the pooled item-level
  permutation test is the primary bar for R_cue; the per-seed direction count is reported beside it. Other bars keep
  the §5 common rule (5/6 AND pooled p < 0.05).
- Interview lesion sensitivity: counted only on items where the forced probe is right in intact and wrong with the
  hippocampus off (the §5 definition, made exact); the drop over all items is reported.
- Savings: G and dR are also reported without the A item asked on the same day (sensitivity row); the both-off null is
  probed once on branch night 1.
- Savings, primary read-out NULL-SUBTRACTED (decided 11-21 on the s318 DEV rehearsal, before any TEST read): the
  pretrained both-off cortex already rates the A items above the B items (raw both-off G +3.4 on s318, prunedA G +5.7),
  so raw G mixes item difficulty with savings and the §5 "both-off G about 0" bar cannot hold by construction. The
  registered G is `G_null_sub` = mean over A of (log-odds hippocampus-off - log-odds both-off) minus the same over B,
  per item; dR likewise as `dR_null_sub` = dR - both-off dR. The G > 0 (5/6 + pooled p < 0.05) and dR >= +1.5 of 8 bars
  apply to these. Raw G and the both-off G are reported beside them. s318 DEV (one seed, strict scorer, P0 unread):
  G_null_sub +2.28 / +2.39 / +1.92 on branch nights 1-3; not evidence, an instrument check.
- Per-seed kill (§10), the adapter-norm clause DROPPED (decided 11-21 on the s318 DEV rehearsal): under the defaults
  the adapter norm grows every night (59 on night 1 to 672 on night 15, above 3x night 1 from night 4), as the
  M-scale reads already showed (B 211-376 at nights 4-7), while recall and transcripts held. Read literally, the clause
  would end every life near night 4 and turn Stage 1 into no measurement. The norm is reported per night; the kill is
  collapse on 2 consecutive probe nights only. The KL clause is also dropped as a kill: it has no 1.5B measurement
  yet, and the 0.1B smoke already reached 1.81x night 1 by night 2, so it risks the same trap. KL stays a registered
  bar (night-15 KL at most 2x night 1, reported every night) and never ends a life. The norm-growth defect itself
  stays open (the next norm method is selective, after the milestone).
- Language guard (§5): it was not in the harness; built as `--lang-guard` (a frozen, hashed held-out ordinary-text
  set, `brain/eval/life_heldout.py`), measured on every trunk night, and added to the registered command.
- Transfer (§3.5): the sysadmin definition is mine (priorities are delegated); the owner may override before the
  launch, and no change is made after it.

**Gate before the launch.** PASSED 11-21: the s318 DEV rehearsal (every flag, AWS g5) ran 15 nights plus the 3 branch
nights with rc 0 in 4.0 h (about 650 s per night; savings, cue and interview sections all written); its three instrument
defects (savings null, missing language guard, unusable norm kill) are fixed above. `--lang-guard` itself was not in the
rehearsal: it is covered by tests and a 0.1B smoke only, and a crash there is recoverable with `--resume`. Original gate: a DEV dress rehearsal with every flag at 1.5B (s318, AWS g5, launched 11-20) must finish
15 nights plus the branch without an error, and its night time sets where the six seeds run. A harness defect found
there is fixed and re-tested on DEV, never on TEST. Launch by 12-01 (§10).
