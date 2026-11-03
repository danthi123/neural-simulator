# Text-replay sleep: KEEP or KILL on 2026-11-04 (decision memo)

Capability: Sleep consolidation. Written 2026-11-03 for the HANDOFF TIMEBOX decision. Every number below was read
from an artifact under `brain/experiments/results/` (means recomputed from the JSONs for the cortex-size and 0.1B
rows; other rows copied from the committed writeups, which say so). Nothing here is a new run.

## Plain-language summary

"Text-replay sleep" means: while the brain sleeps, it re-reads the stored text of what it was told and trains small
adapters in the cortex on it, so the facts live in the cortex itself and not only in the hippocampus.

- It works as a memory writer. With the hippocampus switched off, the sleeping brain answers told facts and a
  brain given scrambled facts answers none (0 of 12 on every 1.5B seed). Switching the consolidated adapters off
  removes the answers (5.5 of 12 down to 0.17 in the live brain). Drift on ordinary text stays small.
- It improves a lot with cortex size: 0.1B about 3.8 of 12, 0.4B 4.0, 1.5B 8.83 of 12, and 1.5B beats 0.4B on 6 of 6
  seeds. The 1.5B mean is about 80% of its in-context ceiling (11 of 12).
- It has one standing defect: it makes the brain answer never-told questions with learned names (false recall about
  2 of 4 at every size, not reduced by size, KL guard, or more steps). The no-sleep brain already confabulates 2 of 4
  through the hippocampus at 0.1B, so sleep is not the only source, but it adds about one more (3.0 vs 2.0).
- At 0.1B it adds almost nothing a hippocampus-on brain does not already have (+0.17 of 12 recall). Whether that is
  true at 1.5B is untested: there is no live-brain, hippocampus-on versus sleep comparison at 1.5B yet.
- Its add-ons failed at 0.1B: varied replay NOT SHOWN, retrieval-practice replay (M3) FALSIFIED, high-use facts
  NO-GO. These are mostly scale-bound (0.1B answers only 6 of 12 even with the fact in front of it).

**Recommendation: KEEP, with a one-week fuse.** The mechanism is real, six-seed, lesion-proven and strongly
scale-responsive; killing it now would be judging a method at the scale where its negatives are expected
(`brain/PREREQUISITES.md`). But what remains unproven is exactly what the product needs: that sleep earns something
the hippocampus does not, and does not buy it with confabulation. One test at 1.5B decides it (criterion below).

## Evidence table

Cells are recall of 12 told facts, hippocampus OFF / paraphrase of 12 / false recall of 4 never-told questions
(unless stated). Seeds 42/43/44/100/101/102 unless stated. Shuffled = same training on deranged answers (control).

| # | Experiment (artifact) | Setting | Sleep arm | Control | Verdict as written |
|---|---|---|---|---|---|
| 1 | 11-02 six seeds (`2026-11-02-sleep-0.1b-six-seeds.md`) | 0.1B, 240 st, lr 5e-4, kl 1 | 3.8 / 2.2 / 2.2, drift KL 0.016-0.038 | shuffled 0.3 / 1.3 / 2.2; no-sleep 0/0/0 | recall holds 6/6; paraphrase not shown; false recall a defect |
| 2 | Overnight, 0.4B (`2026-11-03-sleep-overnight-0.4b-kl3-st480.md`) | 0.4B at 0.1B settings | 2.3 / 1.5 / 0.5 (seed 43: 0) | shuffled 0.3 / 0.5 / 0.7 | under-trained, not a size verdict |
| 3 | Overnight, 0.1B kl 3 | 240 st, kl 3 | 3.2 / 2.5 / 1.5 | shuffled 0.3 / 0.7 / 1.8 | drift lower; false recall not below shuffled |
| 4 | Overnight, 0.1B 480 st | 480 st, kl 1 | 4.5 / 3.0 / 3.0 (seed 42: 8) | shuffled 0.5 / 0.8 / 2.8 | more recall AND more false recall |
| 5 | Cortex size, 0.4B tuned (`2026-11-03-cortex-size/six_0.4b_s*.json`, recomputed) | lr 1e-3, 480, kl 3 | 4.0 / 3.0 / 2.0 (1-6), gKL max 0.020 | shuffled 0.33 / 0.33 / 0.83 | working operating point |
| 6 | Cortex size, 1.5B (`b15lr5e-4_s*.json`, recomputed) | lr 5e-4, 480, kl 3 | **8.83** / 5.17 / 2.17 (3, 7, 11, 8, 12, 12), gKL max 0.040 | shuffled 0.00 / 0.17 / 2.00 | 1.5B > 0.4B on 6/6 seeds; cortex size decided 1.5B |
| 7 | Sleep in the live brain, 3 facts (`2026-11-03-sleep-in-brain/`) | 0.1B, restart = sleep | hippocampus off 8/18 told (strict, corrected from 15/18) | consolidation also off 0/18 | wiring works; lead |
| 8 | Live brain, 12 facts (`2026-11-03-live-brain-12facts.md`, pre-registered) | 0.1B | hippo off 5.50 / 3.33 / 3.67; intact+sleep 6.50 / 5.50 / 3.00 | consolidation off 0.17; shuffled 0.00; intact NO sleep 6.33 / 3.83 / 2.00 | P1 pass; P2 FAIL (+0.17 vs +1 needed); P3 FAIL (false 3.0 vs 2.0) |
| 9 | Varied replay (`2026-11-03-varied-replay.md`, pre-registered analysis) | 0.1B, 3 wordings per fact | 4.2 / 3.2 / 1.7 vs true 3.8 / 2.2 / 2.2 | shuffled 0.3 / 1.3 / 2.2 | NOT SHOWN: clean paraphrase beats true on 3/6 seeds, mean +0.5 (needed 5/6, +1.0) |
| 10 | Confab M3 retrieval-practice sleep (`2026-11-03-confab-m3-retrieval-sleep/aggregate.txt`) | 0.1B CPU, reachable facts | recall 4.0 vs true-replay 4.5; false 2.67 vs 3.33; gain per seed 0,-3,+2,-1,0,-1 | shuffled-retrieval recall 0.17 | FALSIFIED at 0.1B. The 0.4B run is in progress on the W GPU (s42, s43 files exist; not read, not complete) |
| 11 | High-use facts (`night-summary.md` rows 4-5) | 0.1B | k=10 facts 4/4 on 0/6 seeds, then 1/6 seeds on reachable facts | equal-replay control | NO-GO at 0.1B; "use helps" weak lead (2.33 vs 1.33 of 4) |

Caveats that apply across rows: the strict scorer in rows 7-8 under-counts first-person correct answers and
over-counts tautologies (not re-scored; the corrected `strict_scorer.correct_answer` exists); row 6 is fp16 on a
3090 and rows 1-4 are CPU fp32 (one-seed check at 0.1B: 4 vs 3 recall); row 6 used one lr only (5e-4) while 0.4B
had a grid on one dev seed, so tuning effort is not matched; 1.5B seed spread is 3 to 12.

What is NOT in the table because it was never run: any 1.5B live-brain test (hippocampus on, no sleep, vs sleep);
1.5B varied replay or M3; 1.5B high-use facts; 0.4B live-brain 12-fact (stopped unfinished, 2 h 45 min on loaded CPU).

## What KEEP and KILL each imply

| Milestone (`ARCHITECTURE.md` §4) | KEEP text replay | KILL text replay |
|---|---|---|
| 11-09 weekly test (intact / hippocampus-off / sleep arms) | Unchanged; first 1.5B transcript needs the GPU (about 25 min, plan step 11). | The sleep arm has no mechanism to test; the weekly test degenerates to hippocampus-only and loses its lesion arm. |
| 11-16 sleep on every restart; lesion visibly changes replies; cortex size chosen | Met by construction: restart = sleep is built, consolidation lesion 5.5 to 0.17 (row 8), size 1.5B (row 6). Ship with SLEEP_DEFAULTS 1.5B lr 5e-4 / 480 / kl 3. | Restart = sleep must be replaced; the "lesion visibly changes replies" item loses its best evidence (mood lesion, byte-identical 6/6, remains). Size decision becomes moot. |
| 11-16 high-use facts never wrong | Still at risk: NO-GO twice at 0.1B, untested at 1.5B. Run the k=10 test at 1.5B before calling it. | No path left; the milestone would be dropped or re-scoped. |
| 11-30 reworded recall | Carried by S_all selection (hippocampal side), not by sleep; sleep paraphrase 5.17 at 1.5B is a bonus. | Unaffected (S_all is independent). |
| 11-30 asks about what is new | Sleep's false recall is the thing the abstention work fights; KEEP keeps that defect open until the gate works. | Removing the consolidated names removes most false recall but also the cortex memory. |
| 12-14 multi-week life, savings, cued recall, transfer | Needs a consolidating process; this is the only one that works. | Nothing to measure savings against. |
| 12-31 local-rule sleep, scale to 1.5B | Text replay is the declared baseline the local rule must beat. | No baseline to replace. |

## Alternatives (SCAFFOLDS.md "Replace with" column)

Text replay is itself the scaffold rows "Sleep replays the stored episode text" and "gradient descent on low-rank
adapters". The replacements are the real alternatives, none ready:
1. Replay generated from the reinstated cortical state, no stored text (needs the reinstatement adapter, record brief A1-A4).
2. Local plasticity rule (Hebbian/BTSP-style) on the same synapses (milestone 12-31, not started).
3. Variants of the replay content: varied wordings (NOT SHOWN at 0.1B; 1.5B untested), retrieval practice M3
   (FALSIFIED at 0.1B; 0.4B pending), a replay floor for rarely used facts.
4. Hippocampus-only (no sleep): the 0.1B control already recalls 6.33 of 12 live, so this is the real fallback,
   but it keeps paraphrase low (3.83) and has no cortical memory to lesion.

None of these is a substitute available this week; KILL would mean choosing option 4 by default.

## Recommendation and the single flip criterion

KEEP text-replay sleep as the consolidation mechanism through 11-16, labelled a declared scaffold.

Flip criterion (one, pre-registered here before it is run): in the live Brain at 1.5B with the SLEEP_DEFAULTS
(restart = sleep), the 12-fact test, six seeds, corrected scorer, shared sleep for both arms, compare
(a) hippocampus off after sleep with (b) hippocampus ON, no sleep. KILL if mean (a) < mean (b) - 1 recall of 12
(sleep-alone does not even approach what the hippocampus already gives) OR shuffled-sleep recall is not 0 on all
six seeds. Otherwise KEEP and move to false-recall and high-use work at 1.5B. Rationale: the offline probe already
shows (a) = 8.83 of 12 at 1.5B; the single thing that decides worth is whether the live path reproduces it and
sleep stands next to the hippocampus, which 0.1B showed it did not exceed. The test costs about 25 min per seed
on the 3090 with the LLM yielded (plan step 11), so it fits the 11-09 week.

Dependencies: the GPU is busy with the 0.4B M3 job; the 1.5B live-brain test is a Claude/AWS-lane launch. Not
launched here.
