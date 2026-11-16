# Result: reply-only rehearsal night, 1.5B, AWS seeds 1115-1120 — NO-GO (R4 fails), no switch flips

Prereg `brain/research/2026-11-16-rehearsal-night-1.5b.md`, addenda 4 (bars) and 5 (echo counts as collapse).
Arms: `brain/eval/open_conv.py` default (ocd) vs `--rehearse-reply` (ocy, `Brain(sleep_rehearse="reply")`). Host: AWS
g5.xlarge, batch 20261116-1041 (`run.log`). Rows and transcripts in `2026-11-16-reply-rehearse-1.5b/`.

Scored with `brain/experiments/rehearsal_night_score.py <dir> 1115 1116 1117 1118 1119 1120` (intact day 2):

| arm | collapsed | recall | false | intrusions | needless recall | echoes |
|---|---|---|---|---|---|---|
| default | 2/6 (1117 one line x13, 1120 echoes 5) | 20/36 | 7 | 11 | 15 | 5 |
| reply-only | 1/6 (1118 echoes 6) | 23/36 | 4 | 6 | 14 | 6 |

- R1 (collapsed <= 1/6 and <= default): holds. R2, R3: hold (scorer printout `R1 True R2 True R3 True`).
- Stored-line reuse (`stored_line_reuse.py`): default 5 of 96 day-2 replies, reply-only 2.
- Answer-rate bar (abstentions on memory questions <= default + 2): default 0, reply-only 1. Holds.
- **R4 (paired blind reading, reply-only >= default on >= 5/6): FAILS, 4/6.** One sonnet reader per seed, two
  conversations under shuffled letters, key stored apart (`r4_blind_reading.json`). Scores default / reply-only:
  1115 6/5, 1116 3/6, 1117 1/4, 1118 6/1, 1119 4/5, 1120 1/4 (mean 3.5 vs 4.2). Reply-only is lower on 1115 (it
  invents more: "You're studying Estonian", told Basque) and on 1118, where it echoes the user on every turn
  ("What's the name of your hedgehog?") and answers nothing.

The echo scorer counts exact matches only. Pronoun-normalised echoes (`echo_onset_probe._norm`) per seed: default
[0, 0, 0, 0, 0, 9], reply-only [0, 0, 0, 15, 0, 0]. R1 is unchanged: 1118 and 1120 are counted collapsed either way.

Report-only pool replication (`2026-11-16-reply-rehearse-pool-replication/README.md`, seeds 1121-1125) disagrees with
AWS in direction on recall: reply-only 7/30 vs default 16/30, driven by 1124 and 1125 (0 recall each), where it
abstains on every question ("I'm not sure, I don't know..."): 25 abstaining day-2 replies vs 0 for the default, a
FLIP-HURTS-ANSWERS pattern that alone would block a flip. Its mechanism there (the guard holds the pretrained
abstention on the day-1 question) is untested.

**Verdict on the method.** NO-GO on R4 (4/6 < 5/6). The mean blind score favours reply-only (4.2 vs 3.5) and two of
its four wins are against a collapsed default (1117, 1120); the loss that decides it is one collapse of its own (1118
echo). R4 is one reader per seed. No switch flips (`sleep_rehearse` stays False).

**Caveats (skeptic, PARTLY REFUTED on wording, the NO-GO stands).** Engagement of the reply-only guard is recorded
only by the transcript header (`sleep_rehearse=reply`) and by changed behaviour; no guard size or KL value is logged,
so lens (a) is UNDEFINED in the data. Hosts: `tools/where_ran.py` reports unknown for all rows files; the transcript
headers record ip-172-31-83-254 (AWS) for all 12 AWS runs and node-dl4g243 / node-dl5d243 for the pool. The abstention
bar misses non-answers: on 1118 reply-only answers no memory question (it echoes them) yet abstains on none.

**The biggest defect the conversations show now: in-context collapse, in either arm.** 3 of 12 AWS day-2
conversations collapse (default 1117 one line x13, default 1120 echo, reply-only 1118 echo); the arm that collapses
changes from seed to seed. The echo-onset probe (`brain/experiments/echo_onset_probe.py`, pool40, output
`2026-11-16-reply-rehearse-pool-replication/echo_diag/echo_onset_probe.json`) runs the pretrained cortex alone (the
hippocampus and the night off) over the day-2 scripts of seeds 1121 and 1118, 6 sampling seeds each: 11 of 12
conversations never echo; 1 (script 1121) echoes from turn 1 on 15/16 turns. So the pretrained cortex alone can start
an echo by a sampled onset that the conversation state then repeats. Twelve samples bound the rate only loosely, and
the probe does not show the night plays no part in WHICH arm collapses on a given AWS seed. Next method, preregistered
in `brain/research/2026-11-16-reply-monitor-1.5b.md`: a reply monitor that withholds a draft copying the turn it
answers or an earlier own reply before it enters the conversation state.
