# Result: rehearsal night and mouth habituation on the open-conversation eval (1.5B). No switch flips

Prereg: `brain/research/2026-11-16-rehearsal-night-1.5b.md` (R1-R4 plus addenda 1-3, all written before the AWS
results). Runner: `brain/eval/open_conv.py`; scorer: `brain/experiments/rehearsal_night_score.py`. AWS g6.xlarge
(cuda), fresh seeds 1106-1111, all 24 runs rc 0 (two batches, `run.log` 12/12 each). Pool CPU fp32 seeds 1112-1114
(report-only, addendum 2): pool40 1112, pool41 1113, pool42 1114, every run rc 0. Transcripts and rows are in
`2026-11-16-rehearsal-night-1.5b/` and `2026-11-16-rehearsal-night-pool-replication/`. The blind reading, key
included, is in `2026-11-16-rehearsal-night-1.5b/r4_blind_reading.json`.

Arms:
- default: the night as it is now;
- rehearse: `sleep_rehearse=True`;
- habituation: `habituation=3`;
- habituation+rehearse: both.

## Counted measures (AWS seeds 1106-1111, intact arm, day 2)

| arm | collapsed seeds (reply x>3) | recall /36 | hippocampus-off recall | false | intrusions | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|
| default | 1/6 (1107, x7) | 25 | 26 | 6 | 8 | | | |
| rehearse | 1/6 | 10 | 6 | 6 | 5 | no | no | yes |
| habituation | 0/6 | 27 | 26 | 4 | 5 | yes | yes | yes |
| habituation+rehearse | 0/6 | 9 | 6 | 5 | 1 | yes | no | yes |

## R4: blind reading

One fresh sonnet reader per seed saw all four conversations under shuffled letters, with no score tags. It scored
each day 2 from 1 to 10 and counted right and wrong memory answers. "No worse" means a score >= the default's on that
seed. The R4 bar is >= 5/6.

Two departures from the prereg's wording, both made before unblinding:
- the prereg said "transcript pairs", but one reader saw all four arms at once (addendum 1);
- the operational rule "score >= default's" stands in for "preferred or tied".

| arm | no worse than default | summed score (default 33) | right / wrong memory answers (default 27 / 19) |
|---|---|---|---|
| rehearse | 1/6 | 18 | 10 / 35 |
| habituation | 4/6 | 31 | 27 / 24 |
| habituation+rehearse | 1/6 | 17 | 9 / 36 |

## Verdicts

- **rehearse (whole-text guard): NO-GO as a method.** R2 and R4 fail by wide margins.
  - R1 "no" is only its "fewer collapsed seeds than the default" clause: 1/6 against 1/6, with 5/6 seeds free of
    collapse.
  - It also lowers the answer rate. Rehearse abstains ("not sure", "don't know") on 10/54 memory questions, against
    0/54 for the default.
  - Mechanism, untested until reply-only reports (addendum 3): each guard text is `User: <heard turn>\n\nAssistant: <reply>`, and the KL covers every
    position. So the guard holds the fact sentence itself to the pretrained cortex while replay learns it.
  - The readers saw what the counts show. Facts become generic or invented ("Your niece lives in a big city.").
  - Recall in the hippocampus-off arm falls with it, so this undoes consolidation, not only speech.
- **habituation: not flipped.** R1-R3 hold, but R4 fails.
  - The reader scored it equal to the default on 4/6 seeds and one point lower on 2/6 (1106 and 1107).
  - The reader never ranked it above the default.
  - Its abstentions are 0/54, the same as the default.
  - R1 is weak evidence here. The default collapsed on only 1/6 fresh seeds, and habituation can pass R1's
    exact-repeat count without removing the loop. On pool seed 1112 the habituation arm fell into the night's stored
    line anyway, in paraphrase: "I made pancakes for breakfast this morning." / "...and eggs for breakfast this
    weekend." / "...eggs, and bacon for breakfast this week." The same line, varied, filled 13 of 16 turns, and the
    reader scored it 1/10.
  - The same happens on AWS seed 1107: "You're Sam, my friend." / "You're Sam, I'm Quill." / "I'm Quill, my
    friend, and I was born in Tallinn." alternate across about 10 of 16 turns.
  - Reuse of stored small-talk lines in day-2 replies, counted by `brain/experiments/stored_line_reuse.py` (two or
    more content words shared with one day-1 small-talk turn, none in the current turn). The pool's 13 includes 10
    from seed 1112. The count depends on the method: a fresh reviewer's variant counted AWS 2 vs 2. So the table is
    supporting evidence, and the claim rests on the 1112 and 1107 transcripts:

    | arm | AWS | pool |
    |---|---|---|
    | default | 2 | 3 |
    | habituation | 6 | 13 |

    The block hides the loop rather than removing its cause, which is in the night.
- **habituation+rehearse: NO-GO** (R2 and R4 fail; the rehearse half carries the loss).

## What else the transcripts show

- The day-2 collapse is seed-dependent. The default collapsed on 4/6 baseline seeds (1100-1105), 1/6 fresh AWS seeds
  and 0/3 pool seeds. Its stored-line reuse is 45/96 at baseline against 2/96 on 1106-1111.
- When it does not collapse, the default's biggest visible defect is memory accuracy, not loops.
  - The readers counted 27 right and 19 wrong memory answers over six seeds.
  - The wrong ones are mostly invented facts for questions never told ("Your parrot is called Sam", "Your neighbour
    is Tom").
  - Others are a right fact moved to the wrong slot ("Your tortoise is called Clementine").
  - The hippocampus usually returns a wrong episode on day 2. The night-trained cortex gets many answers right
    anyway, and that is real consolidation.
- Seed 1107 is poor in every arm.
  - The default loops on "You're Sam, my friend." (x7).
  - "I'm not sure" (x9) is the habituation+rehearse arm. The reader ranked that arm first on 1107 because it invents
    nothing.

## Next

Reply-only rehearsal (`sleep_rehearse="reply"`, built, default OFF, addendum 3) is running report-only on the pool
seeds. The open defects, by size in these transcripts:
1. invented or misplaced facts after the night;
2. the stored-line collapse on some seeds;
3. needless recall (18 per six conversations in every arm).

The collapse criterion must count paraphrased loops (content-word reuse), not only exact repeats, before any further
mouth-side fix is judged.
