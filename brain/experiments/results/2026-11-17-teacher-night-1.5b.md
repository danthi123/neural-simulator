# Result: teacher night (near-miss guard, teacher reads the day's tellings) at 1.5B: NARROWED at best, no flip (skeptic)

Pre-registration: `brain/research/2026-11-17-teacher-night-context-1.5b.md` (commit 178cd330f, before any 1.5B run).
Code: `Brain(teacher_context=True)`, `near_miss=4`, `near_miss_frame="F"`. Files in `2026-11-17-teacher-night-1.5b/`.

## Part A: registered mixed day, arm s (AWS g6.xlarge L4, batch 20261117-0935)
**Only 5 of 6 seeds.** `run.log` shows "start nmtc_s102" and no end line, and its log stops after the model load.
The instance has stopped and pull fails with InsufficientInstanceCapacity. Whether s102 finished is unverified. The
missing seed cannot change D2 or D4 (below). The Part A JSONs record only `device: cuda`. The L4 host comes from the
batch launch record, not from the files. `table.txt` prints VOID because the M4 table needs 6 seeds; the D1-D4 numbers are computed from the same
re-scored files (`brain.experiments.rescore`) against banked nmoff (batch 20261114-2031).

| seed | confab (false+invented) off/tc | told off/tc | slept told off/tc | self tc | told-row abstentions off/tc | never-row abstentions off/tc |
|---|---|---|---|---|---|---|
| 42 | 12/2 | 5/3 | 2/2 | 3 | 0/6 | 0/10 |
| 43 | 12/10 | 7/6 | 6/5 | 3 | 0/0 | 0/2 |
| 44 | 12/11 | 9/9 | 4/5 | 3 | 0/1 | 0/1 |
| 100 | 12/8 | 7/4 | 6/4 | 3 | 0/2 | 0/4 |
| 101 | 12/10 | 8/6 | 4/5 | 3 | 0/5 | 0/2 |
| mean | 12.00/8.20 | 7.20/5.60 | 4.40/4.20 | 3.00 | 0.00/2.80 | 0.00/3.80 |

- D1 (confab at most 8.00, lower on at least 5/6 seeds): lower on 5/5, but the mean is 8.20. FAILS on 5 seeds. It
  holds at 6 seeds only if s102 scores 7 or less.
- D2 (told at least 6.67, slept told at least 4.17): told 5.60. FAILS. No s102 value can reach 6.67.
- D3 (self 3.00): holds.
- D4 (told-row abstentions at most off + 0.5): 2.80. FAILS, also beyond reach for s102.
- **Pre-registered verdict: NARROWED at best, no flip.** D1 is judged on six-seed means, so it is undecided until s102
  is in: NO-GO if s102 confab is 8 or more, otherwise NARROWED. D2 and D4 fail with any s102 value.
- Seed 42 carries most of the drop (12 to 2) by declining broadly: 10 never-told and 6 told rows. Abstention is
  only weakly selective: 19 of 60 never-told rows against 14 of the told rows.

## Part B: open conversation (pool CPU, seeds 1100-1105), against the banked 11-15 default transcripts
Hosts: pool40 (node-mxl1462drl) for 1100/1101, pool41 (node-dl4g243) for 1102/1103, pool42 (node-dl5d243) for
1104/1105. These are the same hosts as the baseline. Intact arm; inventions re-scored with `open_conv.invented`.

| seed | never-told invented base/tn | recall_ok base/tn | most-repeated day-2 reply base/tn |
|---|---|---|---|
| 1100 | 3/3 | 1/3 | 11 ("You've been told.") / 2 |
| 1101 | 3/0 | 0/0 | 13 / **14 ("I don't know.")** |
| 1102 | 3/0 | 0/0 | 13 / **11 ("I'm not sure.")** |
| 1103 | 3/2 | 0/3 | 14 / 2 |
| 1104 | 3/3 | 3/2 | 2 / 3 |
| 1105 | 3/2 | 6/5 | 2 / 1 |

O1 is lower on 4/6 seeds and O2 is not lower (10 to 13). Read whole, the transcripts reverse that:
- **Two of the four "wins" (1101, 1102) swap the baseline's repetition collapse for a refusal collapse.**
  - 1101 refuses 14 of 16 day-2 turns. The refusals include "Why is the sky blue?", a story request, small talk, and
    every told fact ("What is my rabbit called?" -> "I don't know." while the rabbit's episode is reinstated). Its
    hippocampus-off arm does the same.
  - 1102 refuses 11 of 16. Its hippocampus-off arm instead repeats "I've been trying to go to bed earlier lately." x14.
- Only 2 inventions out of 18 turn into real, selective declines (1103 "What am I allergic to?", 1105
  "Where was my grandmother born?"). 1103 also declines a told fact ("Where is my gran from?" -> "I'm not sure.").
- On 1100 and 1103 the teacher night breaks the baseline's repetition collapse ("You've been told." x11, "I finally
  finished reading a long novel." x14), and recall rises (1 to 3, 0 to 3). Those seeds still invent ("My horse is
  called Clementine.").
- O3 is up on 2 seeds (1101, 1104), by 1 each. That fails the registered bar, though the margin is noise-sized.

## Why (diagnosis from the logged guard targets)
- The teacher's replies are 6-9 short refusals out of 18-19 targets per night ("I don't know.", "I'm not sure."). The
  rest are still inventions, for example "What is my name?" -> "Your name is Pixel.", read from the tellings.
- Distilled into the context-free student, these refusals do not keep the teacher's condition (is the attribute in
  the tellings?). They become an unconditional short reply that, on some seeds, wins every turn.
- The selectivity lives in the context, and the student is trained without it, on a few dozen cue/reply pairs with
  no told-topic contrast.
- The method as specified does not transfer the in-context judgement. The ceiling stays about 40% invention in
  context, against 99% after the night.

## Lever count on invented facts at 1.5B: 4 failed
The near-miss guard, the confidence gate, M1, and this teacher night. The capability stays open; Metacognition stays
PARTIAL. Any next lever must give the student the told-topic contrast, for example paired targets per topic: a told
attribute answered and an untold attribute declined. A broader target set is also needed so that a refusal cannot
become the default reply.

## Caveats
- Part A has 5 seeds (s102 is on the stopped instance; retry `tools/aws_brain_batch.sh pull`).
- One night per day, greedy decoding.
- The baseline O1 counts collapsed repeats as inventions (`invented` marks any confident reply).
- The open-conversation never-told set is 3 questions per seed.
- Cue hygiene leak in Part A: the kept cue "Who is my best friend?" is an eval told question on s100 and s101. Told still
  fell, so it does not rescue the method.
- **Skeptic** (fresh sonnet, numbers recomputed from the JSONs): NARROWED. The no-flip conclusion survives. Its
  corrections are applied above: the conditional label, s102 unverified, the collapse wording, and O3 noise.
  FLIP-HURTS-ANSWERS: told-row abstentions rise on 4/5 seeds.
